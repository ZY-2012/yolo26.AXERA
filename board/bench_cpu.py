#!/usr/bin/env python3
"""Board-side latency / CPU-usage benchmark for AX620E axmodels (axengine, no extra deps).

- 默认用模型输入 shape 的全 0 数据测速（与 Axera 官方 benchmark 口径一致，耗时与数据无关）
- 如需真实图片输入，额外传 --input xxx.npy（U8 NHWC 或模型声明的 dtype）
- CPU 占用双口径：
    system  : /proc/stat 基准窗口前后差值（全部核，0–100%）
    process : /proc/self/stat utime+stime / wall（单核=100%）
- 打印的 “Benchmark Results” 与参考截图的字段一致
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

try:
    import axengine
except ImportError:
    raise SystemExit(
        "无法导入 axengine：板端请先设置 LD_LIBRARY_PATH=/opt/lib（或安装好 axengine 运行库）"
    )

CLK_TCK = os.sysconf("SC_CLK_TCK")
NCPU = os.cpu_count() or 1


def read_proc_stat_sum() -> tuple[int, int]:
    """Return (busy_jiffies, total_jiffies) summed over all cpus."""
    busy = total = 0
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            parts = [int(x) for x in line.split()[1:]]
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
            total += sum(parts)
            busy += sum(parts) - idle
    return busy, total


def read_self_cpu_ticks() -> int:
    with open("/proc/self/stat") as f:
        fields = f.read().split()
    return int(fields[13]) + int(fields[14])


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def make_input(meta, path: str | None) -> np.ndarray:
    if path:
        return np.load(path)
    shape = tuple(int(d) for d in meta.shape)
    try:
        dtype = np.dtype(meta.dtype)  # pyaxengine NodeArg exposes numpy dtype
    except (TypeError, ValueError):
        dtype = np.dtype("float32")
    return np.zeros(shape, dtype=dtype)


def bench(model: str, input_path: str | None, warmup: int, repeat: int) -> dict:
    sess = axengine.InferenceSession(model)
    meta_in = sess.get_inputs()[0]
    name = meta_in.name
    x = make_input(meta_in, input_path)
    print(f"input: {name} shape={tuple(x.shape)} dtype={x.dtype}")

    for _ in range(warmup):
        sess.run(None, {name: x})

    sys_b0, sys_t0 = read_proc_stat_sum()
    proc0 = read_self_cpu_ticks()
    wall0 = time.perf_counter()

    lat = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        lat.append((time.perf_counter() - t0) * 1000.0)

    wall = time.perf_counter() - wall0
    sys_b1, sys_t1 = read_proc_stat_sum()
    proc1 = read_self_cpu_ticks()

    return {
        "latency_ms": lat,
        "wall_s": wall,
        "sys_cpu_pct": 100.0 * (sys_b1 - sys_b0) / max(1, sys_t1 - sys_t0),
        "proc_cpu_pct": 100.0 * (proc1 - proc0) / CLK_TCK / wall,
        "ncpu": NCPU,
    }


def get_chip() -> str:
    for p in ("/proc/ax_proc/chip_type", "/proc/ax_proc/version"):
        try:
            return Path(p).read_text().strip()
        except OSError:
            continue
    return "unknown"


def report(model: str, r: dict) -> dict:
    lat = r["latency_ms"]
    total = sum(lat)
    avg = total / len(lat)
    print("\nBenchmark Results")
    print(f"  Model          : {os.path.basename(model)}")
    print(f"  Repeat count   : {len(lat)}")
    print(f"  Total time     : {total:.2f} ms")
    print(f"  Avg latency    : {avg:.3f} ms ({avg * 1000:.1f} us)")
    print(f"  Min latency    : {min(lat):.3f} ms ({min(lat) * 1000:.1f} us)")
    print(f"  Max latency    : {max(lat):.3f} ms ({max(lat) * 1000:.1f} us)")
    print(f"  P50 latency    : {pct(lat, 50):.3f} ms ({pct(lat, 50) * 1000:.1f} us)")
    print(f"  P90 latency    : {pct(lat, 90):.3f} ms ({pct(lat, 90) * 1000:.1f} us)")
    print(f"  P99 latency    : {pct(lat, 99):.3f} ms ({pct(lat, 99) * 1000:.1f} us)")
    print(f"  CPU usage      : {r['sys_cpu_pct']:.1f} %  (system, {r['ncpu']} cores)")
    print(f"  Process CPU    : {r['proc_cpu_pct']:.1f} %  (this python process)")
    print(f"  Throughput     : {len(lat) / r['wall_s']:.1f} fps")
    return {
        "model": model,
        "repeat": len(lat),
        "total_ms": total,
        "avg_ms": avg,
        "min_ms": min(lat),
        "max_ms": max(lat),
        "p50_ms": pct(lat, 50),
        "p90_ms": pct(lat, 90),
        "p99_ms": pct(lat, 99),
        "std_ms": statistics.pstdev(lat) if len(lat) > 1 else 0.0,
        "sys_cpu_pct": r["sys_cpu_pct"],
        "proc_cpu_pct": r["proc_cpu_pct"],
        "throughput_fps": len(lat) / r["wall_s"],
        "ncpu": r["ncpu"],
        "chip": get_chip(),
        "python": platform.python_version(),
        "axengine": getattr(axengine, "__version__", "unknown"),
    }


def idle_baseline(seconds: float) -> dict:
    b0, t0 = read_proc_stat_sum()
    time.sleep(seconds)
    b1, t1 = read_proc_stat_sum()
    cpu = 100.0 * (b1 - b0) / max(1, t1 - t0)
    print(f"idle system CPU over {seconds:.1f}s: {cpu:.1f} %")
    return {"idle_sys_cpu_pct": cpu, "seconds": seconds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default="", help="可选：.npy 输入；默认用全 0 数据")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--json", default="")
    ap.add_argument("--idle-seconds", type=float, default=0.0)
    args = ap.parse_args()

    results = {"chip": get_chip(), "model": args.model, "input": args.input}
    if args.idle_seconds > 0:
        results["idle"] = idle_baseline(args.idle_seconds)
    r = bench(args.model, args.input or None, args.warmup, args.repeat)
    results["bench"] = report(args.model, r)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"json -> {args.json}")


if __name__ == "__main__":
    main()
