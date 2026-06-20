# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen3-VL vLLM 服务端

使用 FastAPI + vLLM 部署 Qwen3-VL 模型，提供 HTTP API 供 reward model 调用。
这样可以将模型部署与训练进程解耦，避免多进程环境下的 NCCL 冲突。

启动方式:
    python app_qwen3vl.py --model-path /path/to/Qwen3-VL --port 8100

或使用 uvicorn:
    uvicorn app_qwen3vl:app --host 0.0.0.0 --port 8100
"""

import io
import base64
import logging
import argparse
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .prompts import SYSTEM_PROMPT_BG, USER_PROMPT_BG, parse_vlm_response

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for API
# ============================================================================

class InferenceRequest(BaseModel):
    """单个推理请求"""
    generated_image_b64: str  # Base64 编码的生成图像
    fg_object_list: List[str]  # 前景对象列表
    bg_reference_image_b64: str  # Base64 编码的背景参考图像


class BatchInferenceRequest(BaseModel):
    """批量推理请求"""
    requests: List[InferenceRequest]


class InferenceResponse(BaseModel):
    """推理响应"""
    reward: float
    detailed_result: Dict[str, Any]


class BatchInferenceResponse(BaseModel):
    """批量推理响应"""
    responses: List[InferenceResponse]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool


# ============================================================================
# Global State
# ============================================================================

class ModelState:
    """全局模型状态"""
    llm = None
    sampling_params = None
    model_path: str = ""
    max_new_tokens: int = 1024


model_state = ModelState()


# ============================================================================
# Helper Functions
# ============================================================================

def decode_base64_image(b64_string: str):
    """将 Base64 字符串解码为 PIL Image"""
    from PIL import Image
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def build_vllm_prompt(
    generated_image,
    fg_object_list: List[str],
    bg_reference_image
) -> Dict[str, Any]:
    """
    构建 vLLM 推理所需的 prompt 和多模态数据
    """
    user_prompt_filled = USER_PROMPT_BG.format(
        fg_object_list=fg_object_list
    )

    # 收集所有图像（按顺序）
    images = []

    # Qwen3-VL 的图像占位符格式
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"

    # 构建 prompt 文本
    prompt_parts = []

    # System prompt
    prompt_parts.append(f"<|im_start|>system\n{SYSTEM_PROMPT_BG}<|im_end|>\n")

    # User prompt
    prompt_parts.append("<|im_start|>user\n")
    prompt_parts.append(user_prompt_filled)

    # Generated image
    prompt_parts.append(f"\n[Generated Image]:\n{image_placeholder}")
    images.append(generated_image)

    prompt_parts.append(f"\n[Background Reference Image]:\n{image_placeholder}")
    images.append(bg_reference_image)

    prompt_parts.append("<|im_end|>\n")
    prompt_parts.append("<|im_start|>assistant\n")

    prompt_text = "".join(prompt_parts)

    return {
        "prompt": prompt_text,
        "multi_modal_data": {"image": images},
    }


def compute_reward_from_result(result: dict) -> float:
    """从 VLM 解析结果计算奖励分数"""
    if "final_bg_consistency" in result:
        reward = result.get("final_bg_consistency", 0.0)
    else:
        # 兼容旧格式
        reward = result.get("score", 0.0)
    return float(reward)


def process_single_request(request: InferenceRequest) -> InferenceResponse:
    """处理单个推理请求"""
    try:
        # 解码图像
        generated_image = decode_base64_image(request.generated_image_b64)
        bg_reference_image = decode_base64_image(request.bg_reference_image_b64)

        # 构建 vLLM 输入（BG prompt）
        vllm_input = build_vllm_prompt(
            generated_image=generated_image,
            fg_object_list=request.fg_object_list,
            bg_reference_image=bg_reference_image,
        )

        # 执行推理
        outputs = model_state.llm.generate(
            [vllm_input],
            sampling_params=model_state.sampling_params,
        )

        # 解析输出
        response_text = outputs[0].outputs[0].text.strip()
        result = parse_vlm_response(response_text)
        reward = compute_reward_from_result(result)

        return InferenceResponse(reward=reward, detailed_result=result)

    except Exception as e:  # pylint: disable=broad-exception-caught
        # 服务端最外层兜底，向调用方返回结构化错误而非 500，故必须捕获所有异常。
        logger.exception("Inference failed: %s", e)
        return InferenceResponse(
            reward=0.0,
            detailed_result={"error": str(e)}
        )


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时不自动加载模型，等待 /load_model 调用或命令行参数
    yield
    # 关闭时清理
    if model_state.llm is not None:
        del model_state.llm
        model_state.llm = None


app = FastAPI(
    title="Qwen3-VL Reward Service",
    description="vLLM-based Qwen3-VL inference service for image quality evaluation",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_state.llm is not None
    )


@app.post("/load_model")
async def load_model(
    model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
    max_new_tokens: int = 1024,
    max_model_len: int = 32768,
    max_images_per_prompt: int = 10,
    tensor_parallel_size: int = 8,
    gpu_memory_utilization: float = 0.9,
):
    """动态加载模型"""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="vLLM is not installed. Install it with: pip install vllm"
        )

    # 如果已加载，先清理
    if model_state.llm is not None:
        del model_state.llm
        model_state.llm = None

    try:
        model_state.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": max_images_per_prompt},
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="auto",
        )

        model_state.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )

        model_state.model_path = model_path
        model_state.max_new_tokens = max_new_tokens

        return {"status": "success", "model_path": model_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """单个推理请求"""
    if model_state.llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /load_model first.")

    return process_single_request(request)


@app.post("/batch_inference", response_model=BatchInferenceResponse)
async def batch_inference(batch_request: BatchInferenceRequest):
    """
    批量推理请求 - 使用 vLLM 的批量推理能力
    """
    if model_state.llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /load_model first.")

    try:
        # 准备所有 vLLM 输入
        vllm_inputs = []
        valid_indices = []
        errors = {}

        for i, req in enumerate(batch_request.requests):
            try:
                generated_image = decode_base64_image(req.generated_image_b64)
                bg_reference_image = decode_base64_image(req.bg_reference_image_b64)

                vllm_input = build_vllm_prompt(
                    generated_image=generated_image,
                    fg_object_list=req.fg_object_list,
                    bg_reference_image=bg_reference_image,
                )

                vllm_inputs.append(vllm_input)
                valid_indices.append(i)

            except Exception as e:  # pylint: disable=broad-exception-caught
                # 单条请求出错时不应阻塞整个批量推理；将错误信息记录后由调用方逐条处理。
                logger.warning("Batch item %d failed during prompt build: %s", i, e)
                errors[i] = str(e)

        # 批量推理
        responses = [None] * len(batch_request.requests)

        # 填充错误响应
        for idx, error_msg in errors.items():
            responses[idx] = InferenceResponse(
                reward=0.0,
                detailed_result={"error": error_msg}
            )

        # 执行批量推理
        if vllm_inputs:
            outputs = model_state.llm.generate(
                vllm_inputs,
                sampling_params=model_state.sampling_params,
            )

            # 解析输出
            for idx, output in zip(valid_indices, outputs):
                response_text = output.outputs[0].text.strip()
                result = parse_vlm_response(response_text)
                reward = compute_reward_from_result(result)

                responses[idx] = InferenceResponse(
                    reward=reward,
                    detailed_result=result
                )

        return BatchInferenceResponse(responses=responses)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """命令行入口"""
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过，忽略错误
        pass
    parser = argparse.ArgumentParser(description="Qwen3-VL vLLM Service")
    parser.add_argument("--model-path", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Path to Qwen3-VL model")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind")
    parser.add_argument("--port", type=int, default=8100,
                        help="Port to bind")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Maximum tokens to generate")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Maximum model context length")
    parser.add_argument("--max-images-per-prompt", type=int, default=10,
                        help="Maximum images per prompt")
    parser.add_argument("--tensor-parallel-size", type=int, default=8,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.15,
                        help="GPU memory utilization ratio")

    args = parser.parse_args()

    # 预加载模型
    print(f"Loading Qwen3-VL model from {args.model_path}...")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM is required. Install it with: pip install vllm")

    model_state.llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
    )

    model_state.sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    model_state.model_path = args.model_path
    model_state.max_new_tokens = args.max_new_tokens

    print(f"Model loaded successfully!")
    print(f"Starting server on {args.host}:{args.port}")

    # 启动服务器
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
