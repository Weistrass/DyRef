import os, torch
from accelerate import Accelerator

class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        save_peft_format=False,
        lora_base_model=None,
        peft_adapter_name="default",
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.save_peft_format = save_peft_format
        self.lora_base_model = lora_base_model
        self.peft_adapter_name = peft_adapter_name
        self.num_steps = 0

    def _get_lora_model(self, module):
        if self.lora_base_model is None:
            return None
        if hasattr(module, "pipe") and hasattr(module.pipe, self.lora_base_model):
            return getattr(module.pipe, self.lora_base_model)
        if hasattr(module, self.lora_base_model):
            return getattr(module, self.lora_base_model)
        return None

    def _save_peft_adapter(self, accelerator: Accelerator, model: torch.nn.Module, folder_name: str):
        # Gather full state dict for ZeRO-2/3 - ALL processes must participate
        full_state = accelerator.get_state_dict(model)
        accelerator.wait_for_everyone()

        # Only main process does the actual saving
        if not accelerator.is_main_process:
            return

        module = accelerator.unwrap_model(model)
        lora_model = self._get_lora_model(module)
        if lora_model is None:
            raise ValueError("LoRA base model not found; please set --lora_base_model correctly.")

        save_dir = os.path.join(self.output_path, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # Extract LoRA weights from gathered state dict
        try:
            from peft import get_peft_model_state_dict
        except Exception:
            from peft.utils import get_peft_model_state_dict

        try:
            state_dict = get_peft_model_state_dict(
                lora_model,
                state_dict=full_state,
                adapter_name=self.peft_adapter_name
            )
        except TypeError:
            state_dict = get_peft_model_state_dict(
                lora_model,
                state_dict=full_state
            )

        if hasattr(lora_model, "peft_config"):
            peft_config = lora_model.peft_config
            if isinstance(peft_config, dict):
                config = peft_config.get(self.peft_adapter_name, list(peft_config.values())[0])
            else:
                config = peft_config
            config.save_pretrained(save_dir)

        from safetensors.torch import save_file
        save_file(state_dict, os.path.join(save_dir, "adapter_model.safetensors"))

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            if self.save_peft_format:
                self._save_peft_adapter(accelerator, model, f"step-{self.num_steps}")
            else:
                self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        # PEFT save needs all processes to participate in get_state_dict for ZeRO-2/3
        if self.save_peft_format:
            self._save_peft_adapter(accelerator, model, f"epoch-{epoch_id}")
            return
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            if self.save_peft_format:
                self._save_peft_adapter(accelerator, model, f"step-{self.num_steps}")
            else:
                self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)