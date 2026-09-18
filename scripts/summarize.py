#!/usr/bin/env python3
"""Aggregate board benchmark rounds into a comparison table."""

from __future__ import annotations

import glob
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def main() -> None:
    models: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(str(RESULTS / "bench_*_r*.json"))):
        model = os.path.basename(f).split("_")[1]
        models.setdefault(model, []).append(json.load(open(f))["bench"])

    native = {}
    try:
        comp = json.load(open(RESULTS / "comparison.json"))
        native = {m: v["native_ax_run_model"]["avg"] for m, v in comp.get("models", {}).items()}
    except OSError:
        pass

    header = f"{'model':5s} {'avg_ms':>8s} {'min_ms':>8s} {'p50_ms':>8s} {'p90_ms':>8s} {'p99_ms':>8s} {'procCPU%':>9s} {'sysCPU%':>8s} {'fps':>7s}"
    print(header)
    for model, rounds in models.items():
        avg = statistics.fmean(r["avg_ms"] for r in rounds)
        p50 = statistics.fmean(r["p50_ms"] for r in rounds)
        p90 = statistics.fmean(r["p90_ms"] for r in rounds)
        proc = statistics.fmean(r["proc_cpu_pct"] for r in rounds)
        sys_ = statistics.fmean(r["sys_cpu_pct"] for r in rounds)
        fps = statistics.fmean(r["throughput_fps"] for r in rounds)
        print(
            f"{model:5s} {avg:8.3f} {min(r['min_ms'] for r in rounds):8.3f} {p50:8.3f} "
            f"{p90:8.3f} {max(r['p99_ms'] for r in rounds):8.3f} {proc:9.1f} {sys_:8.1f} {fps:7.1f}"
        )
        if model in native:
            print(f"      native ax_run_model avg={native[model]:.3f} ms, host overhead={avg - native[model]:.3f} ms")


if __name__ == "__main__":
    main()
