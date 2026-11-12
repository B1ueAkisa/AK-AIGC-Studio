#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 AK-AIGC-Studio | CosyVoice 2 云端语音工坊 WebUI
支持高保真 Zero-Shot 声音克隆、跨语种克隆以及方言/多情感指令跟随
"""

import os
import sys
import time
import argparse
import traceback
from datetime import datetime

# 自动注入 CosyVoice 与 Matcha-TTS 模块路径
COSYVOICE_PATHS = [
    "/content/CosyVoice",
    "/content/CosyVoice/third_party/Matcha-TTS",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../CosyVoice")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../CosyVoice/third_party/Matcha-TTS")),
]
for p in COSYVOICE_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch
import torchaudio

# 全局模型实例与配置
GLOBAL_MODEL = None
GLOBAL_MODEL_DIR = None

DEFAULT_DRIVE_MODEL = "/content/drive/MyDrive/AIGC_Workspace/models/audio/CosyVoice2-0.5B"
DEFAULT_DRIVE_OUTPUTS = "/content/drive/MyDrive/AIGC_Workspace/outputs"
DEFAULT_LOCAL_OUTPUTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
DEFAULT_SAMPLE_PROMPT = "/content/CosyVoice/asset/zero_shot_prompt.wav"
DEFAULT_SAMPLE_TEXT = "希望你以后能够做的比我还好呦。"


def resolve_model_dir(custom_path=None):
    """按优先级寻找 CosyVoice 2 模型路径"""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    candidates = [
        DEFAULT_DRIVE_MODEL,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/audio/CosyVoice2-0.5B")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/CosyVoice2-0.5B")),
        "/content/CosyVoice2-0.5B",
    ]
    for c in candidates:
        if os.path.exists(c) and (
            os.path.exists(os.path.join(c, "cosyvoice2.yaml")) or
            os.path.exists(os.path.join(c, "llm.pt"))
        ):
            return c
            
    # 如果 Google Drive 已挂载但尚未下载，直接返回预定目录以便自动下载
    if os.path.exists("/content/drive/MyDrive"):
        return DEFAULT_DRIVE_MODEL
    return candidates[1]


def resolve_output_dir(custom_path=None):
    """自动选择输出路径，优先 Google Drive 持久化"""
    if custom_path:
        os.makedirs(custom_path, exist_ok=True)
        return custom_path
    if os.path.exists("/content/drive/MyDrive/AIGC_Workspace"):
        os.makedirs(DEFAULT_DRIVE_OUTPUTS, exist_ok=True)
        return DEFAULT_DRIVE_OUTPUTS
    os.makedirs(DEFAULT_LOCAL_OUTPUTS, exist_ok=True)
    return DEFAULT_LOCAL_OUTPUTS


def load_cosyvoice_model(model_dir=None):
    """加载 CosyVoice 2 模型（支持智能下载与 GPU 加速）"""
    global GLOBAL_MODEL, GLOBAL_MODEL_DIR
    if GLOBAL_MODEL is not None and GLOBAL_MODEL_DIR == model_dir:
        return GLOBAL_MODEL

    target_dir = resolve_model_dir(model_dir)
    print(f"📦 [WebUI] 目标模型路径: {target_dir}")

    required_weights = ["llm.pt", "flow.pt", "hift.pt", "cosyvoice2.yaml"]
    is_ready = os.path.exists(target_dir) and all(
        os.path.exists(os.path.join(target_dir, f)) for f in required_weights
    )

    if not is_ready:
        print(f"⬇️ 未检测到完整模型，正在从 ModelScope 拉取 CosyVoice2-0.5B 到: {target_dir}...")
        os.makedirs(target_dir, exist_ok=True)
        try:
            from modelscope import snapshot_download
            snapshot_download("iic/CosyVoice2-0.5B", local_dir=target_dir)
            print("✅ 权重下载完成！")
        except Exception as e:
            raise RuntimeError(f"从 ModelScope 下载模型失败: {e}\n请检查网络连接或依赖 `modelscope` 是否安装。")

    print("🚀 正在载入 CosyVoice2 模型并推入推理设备...")
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
    except ImportError as e:
        raise ImportError(
            f"未找到 cosyvoice 模块 ({e})。\n"
            "请确认已克隆 CosyVoice 仓库并设置 sys.path：\n"
            "git clone --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice.git /content/CosyVoice"
        )

    device_fp16 = torch.cuda.is_available()
    GLOBAL_MODEL = CosyVoice2(target_dir, load_jit=False, load_trt=False, fp16=device_fp16)
    GLOBAL_MODEL_DIR = target_dir
    print(f"✅ CosyVoice 2 模型加载完成！FP16={device_fp16}, 采样率: {GLOBAL_MODEL.sample_rate}Hz")
    return GLOBAL_MODEL


def synthesize(
    mode,
    tts_text,
    prompt_audio,
    prompt_text,
    instruct_text,
    speed,
    output_dir=None
):
    """WebUI 推理调度核心函数"""
    import gradio as gr

    if not tts_text or not tts_text.strip():
        raise gr.Error("❌ 请输入待合成的文本内容！")

    # 校验参考音频
    effective_prompt_audio = prompt_audio
    effective_prompt_text = prompt_text

    if not effective_prompt_audio:
        # 尝试使用官方演示 sample
        if os.path.exists(DEFAULT_SAMPLE_PROMPT):
            effective_prompt_audio = DEFAULT_SAMPLE_PROMPT
            if not effective_prompt_text:
                effective_prompt_text = DEFAULT_SAMPLE_TEXT
            gr.Info("ℹ️ 未上传自定义音频，已自动载入系统内置演示样本。")
        else:
            raise gr.Error("❌ 请上传或录制一段参考音频 (3~10秒为佳)！")

    model = load_cosyvoice_model()
    save_dir = resolve_output_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "zero_shot" if "零样本" in mode else ("instruct" if "指令" in mode else "cross_lingual")
    output_wav = os.path.join(save_dir, f"cosyvoice_{timestamp}_{mode_tag}.wav")

    print(f"\n🎙️ [WebUI] 开始合成语音:")
    print(f"  - 模式: {mode}")
    print(f"  - 目标文本: {tts_text}")
    print(f"  - 参考音频: {effective_prompt_audio}")
    print(f"  - 语速: {speed}x")

    speech_chunks = []
    start_time = time.time()

    try:
        if "零样本" in mode:
            if not effective_prompt_text or not effective_prompt_text.strip():
                raise gr.Error("❌ 零样本声音克隆模式下，必须提供参考音频对应的文字 (Prompt Text)！")
            generator = model.inference_zero_shot(
                tts_text, effective_prompt_text, effective_prompt_audio, stream=False, speed=speed
            )
        elif "指令" in mode:
            if not instruct_text or not instruct_text.strip():
                raise gr.Error("❌ 方言与情感指令模式下，请输入指令内容 (例如: 用四川话说这句话<|endofprompt|>)！")
            
            # 格式化指令末尾标记
            cleaned_instruct = instruct_text.strip()
            if "<|endofprompt|>" not in cleaned_instruct:
                cleaned_instruct += "<|endofprompt|>"

            if hasattr(model, "inference_instruct2"):
                generator = model.inference_instruct2(
                    tts_text, cleaned_instruct, effective_prompt_audio, stream=False, speed=speed
                )
            elif hasattr(model, "inference_instruct"):
                generator = model.inference_instruct(
                    tts_text, cleaned_instruct, effective_prompt_audio, stream=False, speed=speed
                )
            else:
                raise RuntimeError("当前 CosyVoice 版本不支持指令跟随推理接口。")
        else:
            # 跨语种克隆模式
            generator = model.inference_cross_lingual(
                tts_text, effective_prompt_audio, stream=False, speed=speed
            )

        for chunk in generator:
            if "tts_speech" in chunk and chunk["tts_speech"] is not None:
                speech_chunks.append(chunk["tts_speech"])

        if not speech_chunks:
            raise RuntimeError("模型未返回有效音频张量，请检查输入音频格式与文本。")

        full_speech = (
            torch.concat(speech_chunks, dim=1)
            if len(speech_chunks) > 1
            else speech_chunks[0]
        )
        torchaudio.save(output_wav, full_speech, model.sample_rate)

        elapsed = time.time() - start_time
        info_log = (
            f"✅ **语音生成成功！**\n\n"
            f"- ⏱️ **耗时**: `{elapsed:.2f} 秒`\n"
            f"- 📁 **持久化路径**: `{output_wav}`\n"
            f"- 🎯 **合成模式**: `{mode}`\n"
            f"- ⚡ **采样率**: `{model.sample_rate} Hz`"
        )
        return output_wav, info_log

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"❌ [WebUI Error]: {err_msg}")
        raise gr.Error(f"合成失败: {str(e)}")


def build_app():
    """构建 Gradio 界面应用"""
    import gradio as gr

    custom_css = """
    #title-header {
        text-align: center;
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        color: #f1f1f1;
        padding: 24px 20px;
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    #title-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    #title-header p {
        font-size: 0.95rem;
        opacity: 0.85;
        margin: 0;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        margin: 2px 4px;
        border-radius: 6px;
        font-size: 0.82rem;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .preset-btn {
        margin: 2px !important;
    }
    """

    with gr.Blocks(title="🎨 AK-AIGC-Studio | CosyVoice 2 语音工坊", css=custom_css, theme=gr.themes.Soft()) as demo:
        # 页面头部
        gpu_info = "🎮 GPU: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU 模式")
        with gr.HTML():
            html_content = f"""
            <div id="title-header">
                <h1>🎨 AK-AIGC-Studio | CosyVoice 2 语音工坊</h1>
                <p>高自然度多语言零样本声音克隆 · 跨语种克隆 · 方言与情感指令控制 · 云端永久同步</p>
                <div style="margin-top: 10px;">
                    <span class="status-badge">⚡ CosyVoice2-0.5B</span>
                    <span class="status-badge">{gpu_info}</span>
                    <span class="status-badge">☁️ Google Drive 自动持久化</span>
                </div>
            </div>
            """
            return_val = html_content
        gr.HTML(value=return_val)

        with gr.Row():
            # 左侧：配置与输入面板
            with gr.Column(scale=5):
                gr.Markdown("### ⚙️ 语音生成配置")

                mode_selector = gr.Radio(
                    choices=[
                        "🎯 零样本声音克隆 (Zero-Shot)",
                        "🎭 方言与情感指令控制 (Instruct)",
                        "🌍 跨语种声音克隆 (Cross-Lingual)"
                    ],
                    value="🎯 零样本声音克隆 (Zero-Shot)",
                    label="选择推理模式",
                    info="零样本：需参考音频+原文字；指令控制：支持方言/情绪指令；跨语种：纯音色迁移"
                )

                tts_input = gr.Textbox(
                    label="待合成文本 (TTS Text)",
                    lines=4,
                    placeholder="请输入需要转换为语音的文字内容...\n例如：欢迎来到 AK-AIGC-Studio，这里是属于你的全能多模态生成工坊！",
                    value="欢迎来到 AK-AIGC-Studio！这里是你的专属多模态生成工坊，由主理人 B1ueAkisa 与总副手共同打造。"
                )

                prompt_audio_input = gr.Audio(
                    label="声音克隆参考音频 (支持上传或在线录音，3~10秒为佳)",
                    sources=["upload", "microphone"],
                    type="filepath",
                    value=DEFAULT_SAMPLE_PROMPT if os.path.exists(DEFAULT_SAMPLE_PROMPT) else None
                )

                prompt_text_input = gr.Textbox(
                    label="参考音频原文本 (Prompt Text)",
                    lines=2,
                    placeholder="输入参考音频中实际说出的文字（零样本模式必填，指令/跨语种模式可留空）...",
                    value=DEFAULT_SAMPLE_TEXT if os.path.exists(DEFAULT_SAMPLE_PROMPT) else ""
                )

                with gr.Group():
                    instruct_input = gr.Textbox(
                        label="方言 / 情感指令 (仅指令模式生效)",
                        placeholder="例如: 用四川话说这句话<|endofprompt|>",
                        value="用四川话说这句话<|endofprompt|>"
                    )
                    gr.Markdown("⚡ **快捷指令填入:**")
                    with gr.Row():
                        btn_sichuan = gr.Button("🌶️ 四川话", size="sm", elem_classes="preset-btn")
                        btn_cantonese = gr.Button("🥟 粤语", size="sm", elem_classes="preset-btn")
                        btn_happy = gr.Button("😄 开心情感", size="sm", elem_classes="preset-btn")
                        btn_serious = gr.Button("😠 严肃质问", size="sm", elem_classes="preset-btn")
                        btn_whisper = gr.Button("🤫 温柔耳语", size="sm", elem_classes="preset-btn")

                speed_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="语速调节 (Speed Factor)"
                )

                submit_btn = gr.Button("🚀 立即开始合成语音", variant="primary", size="lg")

            # 右侧：输出与结果试听
            with gr.Column(scale=4):
                gr.Markdown("### 🎧 生成成果与状态")
                output_audio = gr.Audio(
                    label="合成语音结果播放",
                    type="filepath",
                    interactive=False
                )
                output_info = gr.Markdown(
                    value="💡 **提示**: 点击左侧「立即开始合成语音」后，结果将在此播放，并自动归档至 Google Drive `AIGC_Workspace/outputs/`。"
                )

                # 底部使用指引卡片
                with gr.Accordion("📖 功能说明与玩法技巧", open=True):
                    gr.Markdown(
                        """
                        - **零样本声音克隆**: 上传一段任意说话人的清晰音频，并在「参考音频原文本」中准确输入该音频的说话内容，即可实现 1:1 极高保真音色复刻。
                        - **方言与情感指令控制**: 在指令中加上方言名称（如“用四川话说这句话”、“用粤语说这句话”）或情感（如“非常开心地说”、“带着悲伤的语气”），模型将自动切换发音习惯与语调。
                        - **自动云端持久化**: 产出的所有音频均自动写入 `/content/drive/MyDrive/AIGC_Workspace/outputs/`，防止 Colab 断开丢失！
                        """
                    )

        # 预设按钮事件绑定
        btn_sichuan.click(lambda: "用四川话说这句话<|endofprompt|>", outputs=[instruct_input])
        btn_cantonese.click(lambda: "用粤语说这句话<|endofprompt|>", outputs=[instruct_input])
        btn_happy.click(lambda: "非常开心地说这句话<|endofprompt|>", outputs=[instruct_input])
        btn_serious.click(lambda: "带着严肃质问的语气说这句话<|endofprompt|>", outputs=[instruct_input])
        btn_whisper.click(lambda: "用温柔轻声的耳语说这句话<|endofprompt|>", outputs=[instruct_input])

        # 模式切换动态调整组件可见度提示
        def on_mode_change(mode):
            if "零样本" in mode:
                return gr.update(interactive=True), gr.update(interactive=False)
            elif "指令" in mode:
                return gr.update(interactive=False), gr.update(interactive=True)
            else:
                return gr.update(interactive=False), gr.update(interactive=False)

        mode_selector.change(
            on_mode_change,
            inputs=[mode_selector],
            outputs=[prompt_text_input, instruct_input]
        )

        # 提交生成
        submit_btn.click(
            fn=synthesize,
            inputs=[
                mode_selector,
                tts_input,
                prompt_audio_input,
                prompt_text_input,
                instruct_input,
                speed_slider
            ],
            outputs=[output_audio, output_info]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="🎨 AK-AIGC-Studio: CosyVoice 2 云端 WebUI")
    parser.add_argument("--share", action="store_true", help="是否创建 Gradio 公网临时共享链接")
    parser.add_argument("--port", type=int, default=7860, help="Web 服务监听端口 (默认: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web 服务监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--model_dir", type=str, default=None, help="自定义 CosyVoice2-0.5B 模型权重目录")
    parser.add_argument("--outputs_dir", type=str, default=None, help="自定义生成音频保存目录")
    args = parser.parse_args()

    print("=" * 65)
    print("🎨 AK-AIGC-Studio | CosyVoice 2 WebUI 启动中...")
    print(f"  - 监听地址: http://{args.host}:{args.port}")
    print(f"  - 公网共享 (share): {args.share}")
    print("=" * 65)

    # 预加载模型，防止初次请求过长
    try:
        load_cosyvoice_model(args.model_dir)
    except Exception as e:
        print(f"⚠️ [Startup Warning] 启动期预载模型跳过 (将在首次请求时载入): {e}")

    demo = build_app()
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
