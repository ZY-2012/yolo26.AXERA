#!/usr/bin/env python3
"""Print the host+board comparison table from results/comparison.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

ORDER = ["u8", "mix", "u16"]


def main() -> None:
    comp = json.load(open(RESULTS / "comparison.json"))
    models = comp.get("models", {})
    if "idle_sys_cpu_pct" in comp:
        print(f"idle system CPU: {comp['idle_sys_cpu_pct']:.1f} %")
    header = (
        f"{'model':5s} {'avg_ms':>8s} {'min_ms':>8s} {'p50_ms':>8s} {'p90_ms':>8s} {'p99_ms':>8s} "
        f"{'procCPU%':>9s} {'sysCPU%':>8s} {'fps':>7s} {'native_ms':>10s} {'host_ms':>8s}"
    )
    print(header)
    for model in sorted(models, key=lambda m: ORDER.index(m) if m in ORDER else 99):
        v = models[model]
        native = v.get("native_ax_run_model", {}).get("avg", float("nan"))
        overhead = v.get("host_overhead_ms", float("nan"))
        print(
            f"{model:5s} {v['avg_ms']:8.3f} {v['min_ms']:8.3f} {v['p50_ms']:8.3f} {v['p90_ms']:8.3f} "
            f"{v['p99_ms']:8.3f} {v['proc_cpu_pct']:9.1f} {v['sys_cpu_pct']:8.1f} {v['throughput_fps']:7.1f} "
            f"{native:10.3f} {overhead:8.3f}"
        )


if __name__ == "__main__":
    main()
