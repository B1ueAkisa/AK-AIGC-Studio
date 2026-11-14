"""
AK-AIGC-Studio: 图像生成模块 (Image Generation, Img2Img & Multi-Reference Fusion)
支持 Qwen-Image-2.1 (多图角色融合)、FLUX.1-schnell、SDXL-Turbo、SDXL，
自动接入 colab/tracker.py 记录模型、显卡与算力点数消耗。
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from colab.tracker import track_generation


DEFAULT_CACHE_DIR = "/content/drive/MyDrive/AIGC_Workspace/models/image"

MODEL_REGISTRY = {
    "qwen-image-2.1": "Qwen/Qwen-Image-2.1",
    "qwen-image-2.1-turbo": "Qwen/Qwen-Image-2.1-Turbo",
    "omnigen2": "OmniGen2/OmniGen2",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}


def generate_image(
    prompt: str,
    model_type: str = "qwen-image-2.1",
    output_path: str = "/content/outputs/generated_image.png",
    init_image: Optional[str] = None,
    ref_images: Optional[List[str]] = None,
    strength: float = 0.65,
    width: int = 1280,
    height: int = 720,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> str:
    """Generate or edit an image on GPU using diffusers with automatic tracking."""
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 GPU！请在 Colab 中切换为 L4 或 A100 GPU 运行时。")

    repo_id = MODEL_REGISTRY.get(model_type, model_type)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    is_qwen = "qwen-image" in repo_id.lower()
    is_omnigen = "omnigen2" in repo_id.lower()
    is_flux = "flux" in repo_id.lower()
    is_turbo = "turbo" in repo_id.lower()

    if steps is None:
        if is_qwen and is_turbo:
            steps = 8
        elif is_qwen:
            steps = 30
        elif is_omnigen:
            steps = 40
        elif is_flux or is_turbo:
            steps = 4
        else:
            steps = 25

    if guidance_scale is None:
        guidance_scale = 0.0 if (is_flux or is_turbo) else 7.5

    generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
    input_list = list(ref_images or [])
    if init_image and init_image not in input_list:
        input_list.insert(0, init_image)

    print(f"🖼️ [Image] 模型: {repo_id} | 分辨率: {width}x{height} | 步数: {steps} | 参考图: {len(input_list)}")

    with track_generation(
        modality="image",
        model=repo_id,
        outputs=[output_path],
        inputs=input_list,
        prompt=prompt,
        seed=seed,
        extra={"width": width, "height": height, "steps": steps},
        log_dir=Path(log_dir) if log_dir else None,
    ):
        load_kwargs = {"torch_dtype": dtype}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        if is_omnigen:
            import subprocess as _sp

            try:
                from diffusers import OmniGen2Pipeline
            except ImportError:
                omni_repo = "/content/OmniGen2"
                if not os.path.exists(omni_repo):
                    print("📦 正在拉取 OmniGen2 官方仓库源码...")
                    _sp.check_call(["git", "clone", "--depth", "1", "https://github.com/VectorSpaceLab/OmniGen2.git", omni_repo])
                    _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "einops", "omegaconf"])
                if omni_repo not in sys.path:
                    sys.path.insert(0, omni_repo)
                from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

            pipe = OmniGen2Pipeline.from_pretrained(repo_id, trust_remote_code=True, **load_kwargs)
            if not hasattr(pipe.transformer, "enable_taylorseer"):
                pipe.transformer.enable_taylorseer = False
            if not hasattr(pipe.transformer, "enable_teacache"):
                pipe.transformer.enable_teacache = False
            pipe.enable_model_cpu_offload()
            call_kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 2.6,
                "generator": generator,
            }
            if input_list:
                call_kwargs["input_images"] = [Image.open(p).convert("RGB") for p in input_list]
            result = pipe(**call_kwargs).images[0]

        elif is_qwen:
            from diffusers import QwenImage21Pipeline

            pipe = QwenImage21Pipeline.from_pretrained(repo_id, **load_kwargs)
            pipe.enable_model_cpu_offload()
            call_kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "generator": generator,
            }
            if input_list:
                pil_refs = [Image.open(p).convert("RGB") for p in input_list]
                call_kwargs["image"] = pil_refs if len(pil_refs) > 1 else pil_refs[0]
            result = pipe(**call_kwargs).images[0]

        elif init_image:
            from diffusers import AutoPipelineForImage2Image

            pipe = AutoPipelineForImage2Image.from_pretrained(repo_id, **load_kwargs)
            pipe.enable_model_cpu_offload()
            src_img = Image.open(init_image).convert("RGB").resize((width, height))
            result = pipe(
                prompt=prompt,
                image=src_img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        else:
            from diffusers import AutoPipelineForText2Image

            pipe = AutoPipelineForText2Image.from_pretrained(repo_id, **load_kwargs)
            pipe.enable_model_cpu_offload()
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        result.save(output_path)
        print(f"✅ 图像已生成并保存到: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AK-AIGC-Studio Image Generator")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default="qwen-image-2.1")
    parser.add_argument("--output", type=str, default="/content/outputs/test_image.png")
    parser.add_argument("--init-image", type=str, default=None)
    parser.add_argument("--ref-images", nargs="*", default=None, help="Multi-reference image paths")
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    generate_image(
        prompt=args.prompt,
        model_type=args.model,
        output_path=args.output,
        init_image=args.init_image,
        ref_images=args.ref_images,
        strength=args.strength,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        cache_dir=args.cache_dir,
        log_dir=args.log_dir,
    )
