"""
AK-AIGC-Studio: Colab CLI 一键远程调度器 (Colab Remote Runner)

通过 google-colab-cli 自动完成：
1. 复用或新建指定 GPU 的 Colab 会话
2. 上传素材图与最新项目代码 (tracker.py / generate_image.py)
3. 按配置文件批量执行生成任务，自动记录 GPU/模型/耗时/估算算力点数
4. 下载成品到本地 outputs/ 并合并远端日志到本地 logs/generation_log.jsonl
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from colab.tracker import load_logs, log_generation, rebuild_report


def query_colab_usage(env: Dict[str, str]) -> Optional[float]:
    """Query real Colab Compute Unit balance via `colab usage`."""
    try:
        out = run_cli(["usage"], env=env, capture=True)
        m = re.search(r"Current balance:\s*([0-9.]+)\s*compute units", out)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def build_env(proxy: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    if proxy:
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
    return env


def run_cli(args: List[str], env: Dict[str, str], capture: bool = False, retries: int = 3) -> str:
    import time

    cmd = ["colab"] + args
    for attempt in range(1, retries + 1):
        try:
            if capture:
                res = subprocess.run(cmd, env=env, check=True, text=True, capture_output=True, encoding="utf-8")
                return res.stdout
            subprocess.check_call(cmd, env=env)
            return ""
        except subprocess.CalledProcessError:
            if attempt >= retries or args[0] not in {"upload", "download", "usage", "sessions", "exec"}:
                raise
            print(f"⚠️ colab {args[0]} 网络波动，3 秒后第 {attempt + 1} 次重试...")
            time.sleep(3)
    return ""


def run_exec_watchdog(sid: str, script_path: Path, timeout: int, env: Dict[str, str], retries: int = 3) -> None:
    """Run `colab exec` and kill/retry immediately if colab-cli hangs after Connection was lost."""
    import threading
    import time

    cmd = ["colab", "exec", "-s", sid, "--timeout", str(timeout), "-f", str(script_path)]
    for attempt in range(1, retries + 1):
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        lost_flag = False
        last_output_at = time.monotonic()

        def _reader() -> None:
            nonlocal lost_flag, last_output_at
            assert proc.stdout is not None
            for line in proc.stdout:
                last_output_at = time.monotonic()
                print(line, end="", flush=True)
                if "Connection was lost" in line:
                    lost_flag = True

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        while proc.poll() is None:
            time.sleep(1)
            if lost_flag and (time.monotonic() - last_output_at > 5):
                print("⚠️ 检测到 colab-cli 断连卡死，强制终止子进程准备重连...")
                proc.kill()
                break
            if time.monotonic() - last_output_at > timeout + 60:
                print("⚠️ 执行超时无响应，强制终止子进程...")
                proc.kill()
                break

        t.join(timeout=2)
        rc = proc.wait()
        if rc == 0 and not lost_flag:
            return
        if attempt < retries:
            print(f"🔄 3 秒后进行第 {attempt + 1} 次重连执行...")
            time.sleep(3)
        else:
            raise RuntimeError(f"colab exec 连续 {retries} 次失败 (exit={rc})")


def find_or_create_session(gpu: str, session_id: Optional[str], env: Dict[str, str]) -> str:
    if session_id:
        return session_id
    out = run_cli(["sessions"], env=env, capture=True)
    for line in out.splitlines():
        if f"Hardware: {gpu.upper()}" in line:
            head = line.split("|")[0].replace("[?]", "").strip()
            m_sid = re.search(r"\[([^\]]+)\]", head)
            sid = m_sid.group(1) if m_sid else head
            if sid:
                print(f"♻️ 复用在线 {gpu} 会话: {sid}")
                return sid
    print(f"🚀 正在申请新的 Colab {gpu} 运行时...")
    new_out = run_cli(["new", "--gpu", gpu], env=env, capture=True)
    m = re.search(r"Creating session '([^']+)'", new_out)
    if not m:
        raise RuntimeError(f"无法解析新会话 ID: {new_out}")
    sid = m.group(1)
    print(f"✅ 新建 {gpu} 会话就绪: {sid}")
    return sid


def merge_remote_logs(remote_jsonl: Path, local_log_dir: Path) -> None:
    if not remote_jsonl.exists():
        return
    local_log_dir.mkdir(parents=True, exist_ok=True)
    local_jsonl = local_log_dir / "generation_log.jsonl"
    existing = {(r.get("timestamp"), r.get("model"), tuple(r.get("outputs", []))) for r in load_logs(local_jsonl)}
    new_records = load_logs(remote_jsonl)
    added = 0
    with local_jsonl.open("a", encoding="utf-8") as f:
        for r in new_records:
            key = (r.get("timestamp"), r.get("model"), tuple(r.get("outputs", [])))
            if key not in existing:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                added += 1
    rebuild_report(local_log_dir)
    print(f"📊 已合并 {added} 条新生成日志到本地账本: {local_log_dir / 'generations_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AK-AIGC-Studio Colab Remote Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON task config")
    parser.add_argument("--session", type=str, default=None, help="Existing Colab session ID")
    parser.add_argument("--gpu", type=str, default=None, help="Override GPU type (A100, L4, T4)")
    parser.add_argument("--proxy", type=str, default="http://127.0.0.1:7897", help="HTTP/HTTPS proxy URL")
    parser.add_argument("--timeout", type=int, default=600, help="Kernel silence timeout in seconds")
    parser.add_argument("--stop", action="store_true", help="Stop Colab session after finishing")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    env = build_env(args.proxy)
    balance_before = query_colab_usage(env)
    if balance_before is not None:
        print(f"💰 运行前真实点数余额: {balance_before:.2f} CU")

    gpu = args.gpu or cfg.get("gpu", "A100")
    sid = find_or_create_session(gpu, args.session, env)

    # 确保远端父目录存在，避免 Jupyter Contents API 上传子目录时报 500
    subprocess.run(
        ["colab", "exec", "-s", sid],
        input="import os\nfor d in ['/content/datasets', '/content/outputs', '/content/logs', '/content/ak_studio/colab', '/content/ak_studio/image', '/content/ak_studio/video']:\n    os.makedirs(d, exist_ok=True)\n",
        env=env,
        check=True,
        text=True,
        encoding="utf-8",
    )

    assets_dir = Path(cfg["assets_dir"])
    if not assets_dir.is_absolute():
        assets_dir = (cfg_path.parent / assets_dir).resolve()
    remote_assets: Dict[str, str] = {}
    for key, filename in cfg.get("assets", {}).items():
        local_file = assets_dir / filename
        ext = local_file.suffix
        remote_file = f"/content/datasets/{key}{ext}"
        remote_assets[key] = remote_file
        print(f"📤 上传素材 [{key}]: {filename} -> {remote_file}")
        run_cli(["upload", "-s", sid, str(local_file), remote_file], env=env)

    # 上传核心代码到远端 /content/ak_studio/
    for rel in ["colab/tracker.py", "image/generate_image.py", "video/generate_video.py"]:
        local_src = PROJECT_ROOT / rel
        remote_dst = f"/content/ak_studio/{rel}"
        run_cli(["upload", "-s", sid, str(local_src), remote_dst], env=env)

    # 生成远端批量执行脚本
    remote_job_lines = [
        "import os, subprocess, sys",
        "os.makedirs('/content/outputs', exist_ok=True)",
        "os.makedirs('/content/logs', exist_ok=True)",
        "open('/content/ak_studio/colab/__init__.py', 'w').close()",
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'transformers', 'accelerate', 'pillow', 'imageio', 'imageio-ffmpeg', 'ftfy'])",
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/huggingface/diffusers'])",
        "sys.path.insert(0, '/content/ak_studio')",
        "from image.generate_image import generate_image",
        "from video.generate_video import generate_video",
    ]

    output_files = []
    for task in cfg.get("tasks", []):
        out_name = task["output"]
        output_files.append(out_name)
        modality = task.get("modality", cfg.get("modality", "image"))
        model = task.get("model", cfg.get("model", "qwen-image-2.1"))
        if modality == "video":
            src_key = task["image_key"]
            remote_job_lines.append(
                f"generate_video("
                f"image_path={remote_assets[src_key]!r}, "
                f"prompt={task['prompt']!r}, "
                f"model_type={model!r}, "
                f"output_path={f'/content/outputs/{out_name}'!r}, "
                f"width={task.get('width', 832)}, "
                f"height={task.get('height', 480)}, "
                f"num_frames={task.get('num_frames', 49)}, "
                f"fps={task.get('fps', 16)}, "
                f"steps={task.get('steps', 30)}, "
                f"seed={task.get('seed', 42)}, "
                f"log_dir='/content/logs')"
            )
        else:
            refs = [remote_assets[k] for k in task.get("ref_keys", [])]
            remote_job_lines.append(
                f"generate_image("
                f"prompt={task['prompt']!r}, "
                f"model_type={model!r}, "
                f"output_path={f'/content/outputs/{out_name}'!r}, "
                f"ref_images={refs!r}, "
                f"width={task.get('width', 1280)}, "
                f"height={task.get('height', 720)}, "
                f"steps={task.get('steps', 30)}, "
                f"seed={task.get('seed', 42)}, "
                f"log_dir='/content/logs')"
            )

    tmp_job = PROJECT_ROOT / "logs" / "_remote_job.py"
    tmp_job.parent.mkdir(parents=True, exist_ok=True)
    tmp_job.write_text("\n".join(remote_job_lines) + "\n", encoding="utf-8")

    try:
        try:
            print(f"⚡ 在 Colab ({sid}) 上执行批量任务...")
            run_exec_watchdog(sid, tmp_job, args.timeout, env)
        finally:
            if tmp_job.exists():
                tmp_job.unlink()

        if cfg_path.parent.parent.name == "projects":
            local_out_dir = cfg_path.parent / "outputs"
        else:
            local_out_dir = PROJECT_ROOT / "outputs"
        local_out_dir.mkdir(parents=True, exist_ok=True)
        for out_name in output_files:
            local_dst = local_out_dir / out_name
            try:
                run_cli(["download", "-s", sid, f"/content/outputs/{out_name}", str(local_dst)], env=env, retries=1)
                print(f"📥 已下载成品: {local_dst}")
            except subprocess.CalledProcessError:
                print(f"⚠️ 未找到远端产物 /content/outputs/{out_name}（可能生成阶段失败）")

        remote_log_tmp = PROJECT_ROOT / "logs" / "_remote_generation_log.jsonl"
        try:
            run_cli(["download", "-s", sid, "/content/logs/generation_log.jsonl", str(remote_log_tmp)], env=env)
            merge_remote_logs(remote_log_tmp, PROJECT_ROOT / "logs")
        finally:
            if remote_log_tmp.exists():
                remote_log_tmp.unlink()
    finally:
        if args.stop:
            print(f"🛑 正在释放 Colab 会话 {sid}...")
            run_cli(["stop", "-s", sid], env=env)

    balance_after = query_colab_usage(env)
    if balance_before is not None and balance_after is not None:
        real_cost = round(max(balance_before - balance_after, 0.0), 4)
        print(f"💰 运行后真实点数余额: {balance_after:.2f} CU | 本次实际扣减: {real_cost:.2f} CU")


if __name__ == "__main__":
    main()
