#!/usr/bin/env python3
"""板端汇总：读取 results/bench_*.json 与 results/ax_run_model.txt，打印对比表并写 comparison.json。

只依赖 Python 标准库，板端可直接运行。
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics
from pathlib import Path

DIR = Path(__file__).resolve().parent
RESULTS = DIR / "results"
ORDER = ["u8", "mix", "u16"]
NATIVE_RE = re.compile(r"min\s*=\s*([\d.]+)\s*ms\s+max\s*=\s*([\d.]+)\s*ms\s+avg\s*=\s*([\d.]+)")


def parse_native() -> dict[str, dict]:
    path = RESULTS / "ax_run_model.txt"
    native: dict[str, dict] = {}
    if not path.exists():
        return native
    current = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"---\s+(\w+):", line.strip())
        if m:
            current = m.group(1)
            continue
        m = NATIVE_RE.search(line)
        if m and current:
            native[current] = {"min": float(m.group(1)), "max": float(m.group(2)), "avg": float(m.group(3))}
    return native


def main() -> None:
    native = parse_native()
    models: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(str(RESULTS / "bench_*_r*.json"))):
        model = os.path.basename(f).split("_")[1]
        models.setdefault(model, []).append(json.load(open(f))["bench"])

    out: dict = {"models": {}}
    idle = RESULTS / "idle.json"
    if idle.exists():
        out["idle_sys_cpu_pct"] = json.load(open(idle))["idle"]["idle_sys_cpu_pct"]

    for model, rounds in models.items():
        avg = statistics.fmean(r["avg_ms"] for r in rounds)
        entry = {
            "avg_ms": avg,
            "min_ms": min(r["min_ms"] for r in rounds),
            "p50_ms": statistics.fmean(r["p50_ms"] for r in rounds),
            "p90_ms": statistics.fmean(r["p90_ms"] for r in rounds),
            "p99_ms": max(r["p99_ms"] for r in rounds),
            "proc_cpu_pct": statistics.fmean(r["proc_cpu_pct"] for r in rounds),
            "sys_cpu_pct": statistics.fmean(r["sys_cpu_pct"] for r in rounds),
            "throughput_fps": statistics.fmean(r["throughput_fps"] for r in rounds),
            "chip": rounds[0].get("chip", "unknown"),
            "rounds": rounds,
        }
        if model in native:
            entry["native_ax_run_model"] = native[model]
            entry["cpu_overhead_ms"] = avg - native[model]["avg"]
        out["models"][model] = entry
    out["board"] = next((v["chip"] for v in out["models"].values()), "unknown")
    (RESULTS / "comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    if "idle_sys_cpu_pct" in out:
        print(f"\nidle system CPU: {out['idle_sys_cpu_pct']:.1f} %")
    print(
        f"{'model':5s} {'avg_ms':>8s} {'min_ms':>8s} {'p50_ms':>8s} {'p90_ms':>8s} {'p99_ms':>8s} "
        f"{'procCPU%':>9s} {'sysCPU%':>8s} {'fps':>7s} {'native_ms':>10s} {'cpu_ms':>8s}"
    )
    for model in sorted(out["models"], key=lambda m: ORDER.index(m) if m in ORDER else 99):
        v = out["models"][model]
        native_avg = v.get("native_ax_run_model", {}).get("avg", float("nan"))
        overhead = v.get("cpu_overhead_ms", float("nan"))
        print(
            f"{model:5s} {v['avg_ms']:8.3f} {v['min_ms']:8.3f} {v['p50_ms']:8.3f} {v['p90_ms']:8.3f} "
            f"{v['p99_ms']:8.3f} {v['proc_cpu_pct']:9.1f} {v['sys_cpu_pct']:8.1f} {v['throughput_fps']:7.1f} "
            f"{native_avg:10.3f} {overhead:8.3f}"
        )
    print(f"\ncomparison.json -> {RESULTS / 'comparison.json'}")


if __name__ == "__main__":
    main()
