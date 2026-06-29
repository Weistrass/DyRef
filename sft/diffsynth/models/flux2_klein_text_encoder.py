from transformers import Qwen3Config, Qwen3ForCausalLM


class Flux2KleinTextEncoder(Qwen3ForCausalLM):
    """
    Qwen3-8B based text encoder for FLUX.2 [klein] models.
    
    FLUX.2 [klein] uses Qwen3 8B as its text encoder, extracting hidden states
    from multiple intermediate layers and stacking them for conditioning.
    """
    def __init__(self):
        config = Qwen3Config(**{
            "architectures": [
                "Qwen3ForCausalLM"
            ],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 12288,
            "max_position_embeddings": 40960,
            "max_window_layers": 36,
            "model_type": "qwen3",
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000.0,
            "sliding_window": None,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151936
        })
        super().__init__(config)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs
        )
