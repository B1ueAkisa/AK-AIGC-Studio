"""
AK-AIGC-Studio: 视频生成模块 (Video Generation - I2V & T2V)
支持 Wan 2.2 I2V (Wan-AI/Wan2.2-I2V-A14B-Diffusers)、CogVideoX 1.5-5B-I2V，
自动接入 colab/tracker.py 记录模型、显卡与算力点数消耗。
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from colab.tracker import track_generation


VIDEO_MODEL_REGISTRY = {
    "wan2.2-i2v": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "cogvideox1.5-i2v": "THUDM/CogVideoX1.5-5B-I2V",
}


def generate_video(
    image_path: str,
    prompt: str,
    model_type: str = "wan2.2-i2v",
    output_path: str = "/content/outputs/generated_video.mp4",
    width: int = 832,
    height: int = 480,
    num_frames: int = 49,
    fps: int = 16,
    steps: int = 30,
    guidance_scale: float = 5.0,
    seed: Optional[int] = 42,
    cache_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> str:
    """Generate an image-to-video clip on GPU using diffusers with automatic tracking."""
    import torch
    from diffusers.utils import export_to_video, load_image

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 GPU！请在 Colab 中切换为 A100/L4 GPU 运行时。")

    repo_id = VIDEO_MODEL_REGISTRY.get(model_type, model_type)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
    init_img = load_image(image_path).resize((width, height))

    print(f"🎬 [Video] 模型: {repo_id} | 分辨率: {width}x{height} | 帧数: {num_frames} ({fps}fps) | 步数: {steps}")

    with track_generation(
        modality="video",
        model=repo_id,
        outputs=[output_path],
        inputs=[image_path],
        prompt=prompt,
        seed=seed,
        extra={"width": width, "height": height, "num_frames": num_frames, "fps": fps, "steps": steps},
        log_dir=Path(log_dir) if log_dir else None,
    ):
        load_kwargs = {"torch_dtype": dtype}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        if "wan" in repo_id.lower():
            from diffusers import WanImageToVideoPipeline

            pipe = WanImageToVideoPipeline.from_pretrained(repo_id, **load_kwargs)
            pipe.enable_model_cpu_offload()
            frames = pipe(
                image=init_img,
                prompt=prompt,
                negative_prompt="blurry, distorted face, extra limbs, static image, low quality, watermark",
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
        elif "cogvideox" in repo_id.lower():
            from diffusers import CogVideoXImageToVideoPipeline

            pipe = CogVideoXImageToVideoPipeline.from_pretrained(repo_id, **load_kwargs)
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            frames = pipe(
                image=init_img,
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
        else:
            raise ValueError(f"Unsupported video model: {repo_id}")

        export_to_video(frames, output_path, fps=fps)
        print(f"✅ 视频已生成并保存到: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AK-AIGC-Studio Video Generator")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default="wan2.2-i2v")
    parser.add_argument("--output", type=str, default="/content/outputs/test_video.mp4")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    generate_video(
        image_path=args.image,
        prompt=args.prompt,
        model_type=args.model,
        output_path=args.output,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        cache_dir=args.cache_dir,
        log_dir=args.log_dir,
    )
