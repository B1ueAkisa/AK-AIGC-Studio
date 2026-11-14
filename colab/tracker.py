"""
AK-AIGC-Studio: 生成日志与算力点数追踪器 (Generation & Compute-Unit Tracker)

自动记录每次生成任务使用的模型、GPU 型号、耗时、输入/输出与估算 Colab 计算点数 (CU)，
同时输出机器可读的 JSONL 日志与人类可读的 Markdown 账本报表。
"""
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Google Colab 各档位硬件估算消耗速率 (Compute Units / hour，仅供参考)
GPU_CU_PER_HOUR = {
    "A100-SXM4-80GB": 15.0,
    "A100-SXM4-40GB": 13.0,
    "A100": 13.0,
    "V100": 5.5,
    "L4": 1.9,
    "T4": 1.1,
    "CPU": 0.07,
}


def detect_gpu_name() -> str:
    """Detect current CUDA GPU name or fallback to CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


def estimate_cu_rate(gpu_name: str) -> float:
    """Match GPU name against known Colab Compute Unit burn rates."""
    upper = (gpu_name or "CPU").upper()
    for key, rate in GPU_CU_PER_HOUR.items():
        if key.upper() in upper:
            return rate
    return 1.5 if "GPU" in upper else 0.07


def estimate_compute_units(duration_sec: float, gpu_name: str, cu_per_hour: Optional[float] = None) -> float:
    """Estimate consumed Colab Compute Units for a given duration."""
    rate = cu_per_hour if cu_per_hour is not None else estimate_cu_rate(gpu_name)
    return round((max(duration_sec, 0.0) / 3600.0) * rate, 4)


def get_default_log_dir() -> Path:
    """Resolve logs directory under project root."""
    return Path(__file__).resolve().parent.parent / "logs"


def load_logs(jsonl_path: Path) -> List[Dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def rebuild_report(log_dir: Optional[Path] = None) -> Path:
    """Rebuild human-readable Markdown summary report from generation_log.jsonl."""
    target_dir = Path(log_dir) if log_dir else get_default_log_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = target_dir / "generation_log.jsonl"
    report_path = target_dir / "generations_report.md"

    records = load_logs(jsonl_path)
    total_runs = len(records)
    total_sec = sum(r.get("duration_sec", 0.0) for r in records)
    total_cu = sum(r.get("estimated_cu", 0.0) for r in records)

    model_stats: Dict[str, Dict[str, float]] = {}
    for r in records:
        m = r.get("model", "unknown")
        if m not in model_stats:
            model_stats[m] = {"runs": 0, "sec": 0.0, "cu": 0.0}
        model_stats[m]["runs"] += 1
        model_stats[m]["sec"] += r.get("duration_sec", 0.0)
        model_stats[m]["cu"] += r.get("estimated_cu", 0.0)

    lines = [
        "# 📊 AK-AIGC-Studio 算力消耗与生成账本",
        "",
        f"> 自动生成自 `generation_log.jsonl` | 统计条数：**{total_runs}** 次 | 累计耗时：**{total_sec / 60:.1f} 分钟** | 累计估算消耗：**{total_cu:.3f} 点 (CU)**",
        "",
        "## 📦 按模型统计",
        "",
        "| 模型 | 运行次数 | 累计耗时 | 估算点数 (CU) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for model_name, st in sorted(model_stats.items(), key=lambda x: -x[1]["cu"]):
        lines.append(f"| `{model_name}` | {int(st['runs'])} | {st['sec']:.1f}s | {st['cu']:.3f} |")

    lines.extend([
        "",
        "## 📝 详细生成流水",
        "",
        "| 时间 (UTC+8) | 类型 | 模型 | 显卡 | 耗时 | 估算点数 | 产物 | 状态 |",
        "| :--- | :---: | :--- | :--- | :---: | :---: | :--- | :---: |",
    ])
    for r in reversed(records):
        ts = r.get("timestamp", "")[:19].replace("T", " ")
        outputs = ", ".join(Path(p).name for p in r.get("outputs", [])) or "-"
        lines.append(
            f"| {ts} | {r.get('modality', '-')} | `{r.get('model', '-')}` | {r.get('gpu', '-')} | "
            f"{r.get('duration_sec', 0):.1f}s | {r.get('estimated_cu', 0):.3f} | `{outputs}` | {r.get('status', 'ok')} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def log_generation(
    modality: str,
    model: str,
    duration_sec: float,
    gpu: Optional[str] = None,
    outputs: Optional[List[str]] = None,
    inputs: Optional[List[str]] = None,
    prompt: str = "",
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    status: str = "success",
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Append one generation record to JSONL and refresh the Markdown report."""
    target_dir = Path(log_dir) if log_dir else get_default_log_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = target_dir / "generation_log.jsonl"

    gpu_name = gpu or detect_gpu_name()
    cu = estimate_compute_units(duration_sec, gpu_name)
    now_local = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    record = {
        "timestamp": now_local,
        "modality": modality,
        "model": model,
        "gpu": gpu_name,
        "duration_sec": round(duration_sec, 2),
        "estimated_cu": cu,
        "status": status,
        "seed": seed,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "prompt": prompt,
        "extra": extra or {},
    }

    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    rebuild_report(target_dir)
    print(f"📊 [Tracker] 已记账 | 模型: {model} | 卡: {gpu_name} | 耗时: {duration_sec:.1f}s | 估算点数: {cu:.4f} CU")
    return record


@contextmanager
def track_generation(
    modality: str,
    model: str,
    outputs: Optional[List[str]] = None,
    inputs: Optional[List[str]] = None,
    prompt: str = "",
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    log_dir: Optional[Path] = None,
):
    """Context manager that times a generation block and logs it automatically."""
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception as exc:
        status = f"error: {type(exc).__name__}"
        raise
    finally:
        elapsed = time.perf_counter() - start
        log_generation(
            modality=modality,
            model=model,
            duration_sec=elapsed,
            outputs=outputs,
            inputs=inputs,
            prompt=prompt,
            seed=seed,
            extra=extra,
            status=status,
            log_dir=log_dir,
        )
