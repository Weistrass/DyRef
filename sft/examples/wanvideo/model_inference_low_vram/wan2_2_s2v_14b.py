# This script can generate a single video clip.
# If you need generate long videos, please refer to `Wan2.2-S2V-14B_multi_clips.py`.
import librosa
import torch
from modelscope import dataset_snapshot_download
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video_with_audio

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),  # pylint: disable=line-too-long
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors", **vram_config),  # pylint: disable=line-too-long
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),  # pylint: disable=line-too-long
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    audio_processor_config=ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/"),  # pylint: disable=line-too-long
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_video_dataset",
    local_dir="./data/example_video_dataset",
    allow_file_pattern=f"wans2v/*"
)

NUM_FRAMES = 81  # 4n+1
HEIGHT = 448
WIDTH = 832

PROMPT = "a person is singing"
NEGATIVE_PROMPT = "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"  # pylint: disable=line-too-long
input_image = Image.open("data/example_video_dataset/wans2v/pose.png").convert("RGB").resize((WIDTH, HEIGHT))
# s2v audio input, recommend 16kHz sampling rate
AUDIO_PATH = 'data/example_video_dataset/wans2v/sing.MP3'
input_audio, sample_rate = librosa.load(AUDIO_PATH, sr=16000)

# Speech-to-video
video = pipe(
    prompt=PROMPT,
    input_image=input_image,
    negative_prompt=NEGATIVE_PROMPT,
    seed=0,
    num_frames=NUM_FRAMES,
    height=HEIGHT,
    width=WIDTH,
    audio_sample_rate=sample_rate,
    input_audio=input_audio,
    num_inference_steps=40,
)
save_video_with_audio(video[1:], "video_1_Wan2.2-S2V-14B.mp4", AUDIO_PATH, fps=16, quality=5)

# s2v will use the first (num_frames) frames as reference. height and
# width must be the same as input_image. And fps should be 16, the same as
# output video fps.
POSE_VIDEO_PATH = 'data/example_video_dataset/wans2v/pose.mp4'
pose_video = VideoData(POSE_VIDEO_PATH, height=HEIGHT, width=WIDTH)

# Speech-to-video with pose
video = pipe(
    prompt=PROMPT,
    input_image=input_image,
    negative_prompt=NEGATIVE_PROMPT,
    seed=0,
    num_frames=NUM_FRAMES,
    height=HEIGHT,
    width=WIDTH,
    audio_sample_rate=sample_rate,
    input_audio=input_audio,
    s2v_pose_video=pose_video,
    num_inference_steps=40,
)
save_video_with_audio(video[1:], "video_2_Wan2.2-S2V-14B.mp4", AUDIO_PATH, fps=16, quality=5)
