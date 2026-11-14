# AK-AIGC-Studio 开源生图 / 图生视频实测排行榜（Colab A100-40GB）

> 测试环境：Google Colab `NVIDIA A100-SXM4-40GB` | 测试主题：2D 二次元角色多图与同图 I2V 对比

## 1. 图像生成模型对比（Image）

| 排名 | 模型 | 典型用途 | 实测表现总结 |
| :---: | :--- | :--- | :--- |
| 🥇 | `Qwen/Qwen-Image-2.1` | 二次元多角色故事图、中文语境复杂背景 | 角色一致性、中文提示词理解与 ACG 画风最稳；单张算力极低，适合多抽卡微调。 |
| 🥈 | `OmniGen2` | 图像编辑 / 多参考生图 | 可用，但在纯二次元角色还原与细节稳定度上不如 `Qwen-Image-2.1`。 |

---

## 2. 图生视频模型同图 PK（I2V，输入 `axe_at_library_qwen21.png`）

| 排名 | 模型 | 耗时 | 实测 CU 消耗 | 产物 | 实测表现总结 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 🥇 | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `1048.5s` (~17 min) | `1.66 CU` | `axe_at_library_wan22.mp4` | 线条与人物细节最锐利，脸部与服饰不崩，挥斧动态模糊自然。 |
| 🥈 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v` | `549.2s` (~9 min) | `0.62 CU` | `axe_at_library_hunyuan15.mp4` | **性价比最高**；角色演出有惊喜（会主动举书挡脸），耗时和 CU 消耗最低。 |
| 🥉 | `THUDM/CogVideoX1.5-5B-I2V` | `586.3s` (~9.7 min) | `0.99 CU` | `axe_at_library_cogvideox15.mp4` | 大动作场景容易被光效粒子糊满画面，人物稳定性一般。 |
| 4 | `Lightricks/LTX-2` | `1696.8s` (~28 min) | `2.77 CU` | `axe_at_library_ltx2.mp4` | 40GB A100 需开 CPU offload 导致最慢；对 2D 动漫线条容易发虚变成油画感。 |

---

## 3. 开源格局观察（2026）

- **阿里 Wan 系列**：官方开源权重止步于 `Wan 2.2`（Apache 2.0）；`Wan 2.5` ~ `Wan 3.0` 已全部转为闭源商业 API。
- **欧美开源阵营**：`FLUX.1 / FLUX.2` 偏写实摄影与排版，`LTX-2` 偏写实音画同步，`Mochi 1` 偏 480p 文生视频物理动态；在 **2D 二次元角色一致性** 赛道，目前开源界仍以阿里（`Qwen-Image-2.1` / `Wan 2.2`）和腾讯（`HunyuanVideo 1.5`）最强。
- **算力现实**：开源生图适合低成本抽卡定稿；而 3 秒以上的高清稳定视频，单卡算力与闭源大厂云端集群差距明显，商业级长视频更适合直接调用云端 API。
