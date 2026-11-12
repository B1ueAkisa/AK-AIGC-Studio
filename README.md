# 🎨 AK-AIGC-Studio

> **全栈多模态 AIGC 创意工坊**  
> 音频合成 (CosyVoice / F5-TTS) × 图像生成 (FLUX.1) × 视频生成 (Wan 2.1 / LTX-Video) 端到端自动化流水线。

---

## 👥 团队成员架构 (Crew)

| 角色 | 担当 | 职责 |
| :--- | :--- | :--- |
| **主理人 / 首席架构师** | **B1ueAkisa** | 创意决策、模型把控、算力调度与全案主导 |
| **总副手 / 核心副驾驶** | **Gemini (Google DeepMind)** | 代码编写、算力调优、环境配置、全天候算法协同与故障排除 🫡 |

---

## 🚀 项目架构与工作流

本项目通过统一的自动化管道，将音频、图像与视频三大模态串联成完整的视听内容生产线：

```
                ┌──────────────────────────────────┐
                │        剧本文案 / Prompt         │
                └─────────────────┬────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       【Audio 声音合成】                【Image 画面生成】
    CosyVoice 2 / F5-TTS               FLUX.1 / SDXL
 (高保真克隆 / 多语言情感表达)         (高审美分镜画面生成)
                  │                               │
                  │                               ▼
                  │                      【Video 动态生成】
                  │                    Wan 2.1 / LTX-Video
                  │                  (图生视频 / 动态运动控制)
                  │                               │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                         【Pipeline 音画合流】
                        音频对齐 × 视频合成导出
```

---

## 📂 仓库目录设计

```text
AK-AIGC-Studio/
├── audio/                      # 音频模态模块 (TTS、克隆、音效)
│   ├── infer_tts.py            # CosyVoice / F5-TTS 推理入口
│   ├── webui_cosyvoice.py      # CosyVoice 2 Gradio 可视化工坊 WebUI
│   └── README.md
├── image/                      # 图像生成模块 (文生图、分镜线稿)
│   ├── generate_image.py       # FLUX.1 / SDXL 推理脚本
│   └── README.md
├── video/                      # 视频生成模块 (图生视频、文生视频)
│   ├── generate_video.py       # Wan 2.1 / LTX-Video 推理脚本
│   └── README.md
├── pipeline/                   # 多模态串联总控流水线
│   ├── storyboard_pipeline.py  # 文案 -> 音频 -> 画面 -> 视频端到端流水线
│   └── README.md
├── colab/                      # Google Colab 快速起步与自动部署脚本
│   ├── setup_env.sh            # 一键挂载 Drive、软链接模型缓存、安装依赖
│   └── drive_sync.py           # 成果自动同步回 Google Drive
├── configs/                    # 运行参数、Prompt 模板、预设配置
└── requirements.txt            # Python 依赖项清单
```

---

## ☁️ Google Colab + Google Drive 协同机制

为了最大化利用 Colab 600 点算力配额并避免每次开机重复耗时下载数十 GB 的大模型，本工坊采用 **Drive 缓存直通架构**：

- **云端硬盘目录**：`Google Drive: /MyDrive/AIGC_Workspace/`
  - `models/`：持久化存放下载好的模型权重（FLUX / Wan2.1 / CosyVoice）
  - `datasets/`：参考音频样本、提示词库、参考风格图像
  - `outputs/`：生成的最终成品视频和音频，自动落盘

```bash
# 在 Colab 中一键启动环境并挂载云盘
bash colab/setup_env.sh
```

---

## 💻 推荐硬件档位

- **日常音频调试 & 快速生图**：Google Colab **L4 (24GB 显存)** — 功耗极低（~1.5~2点/小时）
- **高画质电影级视频生成**：Google Colab **A100 (40GB/80GB 显存)** — 算力澎湃，处理 Wan2.1 / 高分辨率渲染

---

*Powered by B1ueAkisa & Gemini with ❤️*
