#!/usr/bin/env python3
"""Aggregate board benchmark rounds (and native ax_run_model output) into results/comparison.json."""

from __future__ import annotations

import glob
import json
import os
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
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
        name = os.path.basename(f)
        model = name.split("_")[1]
        models.setdefault(model, []).append(json.load(open(f))["bench"])

    out = {"models": {}}
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
            entry["host_overhead_ms"] = avg - native[model]["avg"]
        out["models"][model] = entry
    out["board"] = next((v["chip"] for v in out["models"].values()), "unknown")

    (RESULTS / "comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS / 'comparison.json'} ({len(out['models'])} models)")


if __name__ == "__main__":
    main()
