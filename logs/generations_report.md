# 📊 AK-AIGC-Studio 算力消耗与生成账本

> 自动生成自 `generation_log.jsonl` | 统计条数：**8** 次 | 累计耗时：**39.3 分钟** | 累计估算消耗：**7.577 点 (CU)**

## 📦 按模型统计

| 模型 | 运行次数 | 累计耗时 | 估算点数 (CU) |
| :--- | :---: | :---: | :---: |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 1 | 1048.5s | 3.786 |
| `Qwen/Qwen-Image-2.1` | 4 | 793.9s | 2.867 |
| `OmniGen2/OmniGen2` | 3 | 517.9s | 0.924 |

## 📝 详细生成流水

| 时间 (UTC+8) | 类型 | 模型 | 显卡 | 耗时 | 估算点数 | 产物 | 状态 |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- | :---: |
| 2026-11-14 07:52:53 | video | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | NVIDIA A100-SXM4-40GB | 1048.5s | 3.786 | `axe_at_library_wan22.mp4` | success |
| 2026-11-14 07:30:28 | image | `Qwen/Qwen-Image-2.1` | NVIDIA A100-SXM4-40GB | 83.1s | 0.300 | `qwen_kimi_square_dance.png` | success |
| 2026-11-14 07:29:04 | image | `Qwen/Qwen-Image-2.1` | NVIDIA A100-SXM4-40GB | 174.7s | 0.631 | `gpt_downgrade_village.png` | success |
| 2026-11-14 07:04:52 | image | `Qwen/Qwen-Image-2.1` | NVIDIA A100-SXM4-40GB | 176.1s | 0.636 | `axe_at_library_qwen21.png` | success |
| 2026-11-14 06:46:40 | image | `OmniGen2/OmniGen2` | NVIDIA A100-SXM4-40GB | 210.9s | 0.762 | `axe_at_library_omnigen2.png` | success |
| 2026-11-14 06:37:01 | image | `OmniGen2/OmniGen2` | NVIDIA L4 | 188.3s | 0.099 | `axe_at_library_omnigen2.png` | error: AttributeError |
| 2026-11-14 06:30:29 | image | `OmniGen2/OmniGen2` | NVIDIA L4 | 118.7s | 0.063 | `axe_at_library_omnigen2.png` | error: ValueError |
| 2026-11-14 13:52:44 | image | `Qwen/Qwen-Image-2.1` | NVIDIA A100-SXM4-40GB | 360.0s | 1.300 | `cat_eats_fish_qwen21.png, fish_rides_dragon_qwen21.png` | success |
