#!/usr/bin/env python3
"""Generate the three Pulsar2 AX620E configs from the customer's exp_e3_mix.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
CUSTOMER = CONFIG_DIR / "customer_exp_e3_mix.json"


def base_config() -> dict:
    cfg = json.loads(CUSTOMER.read_text())
    cfg["input"] = "../onnx/yolo26n.onnx"
    cfg["target_hardware"] = "AX620E"
    cfg["quant"]["input_configs"][0]["calibration_dataset"] = "../dataset/images100.tar"
    return cfg


def write(name: str, cfg: dict) -> None:
    path = CONFIG_DIR / name
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    n_layers = sum(len(lc.get("layer_names", [])) for lc in cfg["quant"].get("layer_configs", []))
    print(f"{path.name}: target={cfg['target_hardware']} npu_mode={cfg['npu_mode']} layer_configs={n_layers}")


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    customer = base_config()
    (CONFIG_DIR / "customer_exp_e3_mix.json").write_text(
        CUSTOMER.read_text(), encoding="utf-8"
    )
    print("customer_exp_e3_mix.json: verbatim copy")

    u8 = base_config()
    u8["output_dir"] = "../build/u8"
    u8["output_name"] = "yolo26n_u8.axmodel"
    u8["quant"].pop("layer_configs", None)
    write("u8.json", u8)

    u16 = base_config()
    u16["output_dir"] = "../build/u16"
    u16["output_name"] = "yolo26n_u16.axmodel"
    u16["quant"]["layer_configs"] = [
        {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "U16"}
    ]
    write("u16.json", u16)

    mix = base_config()
    mix["output_dir"] = "../build/mix"
    mix["output_name"] = "yolo26n_e3_mix.axmodel"
    write("mix.json", mix)

    # diffs vs customer original
    for name in ("u8.json", "u16.json", "mix.json"):
        cfg = json.loads((CONFIG_DIR / name).read_text())
        diff = {k: (customer.get(k), cfg.get(k)) for k in set(customer) | set(cfg) if customer.get(k) != cfg.get(k)}
        print(f"--- {name} diffs ---")
        for k, (a, b) in diff.items():
            if k == "quant":
                print(f"  quant.layer_configs: {len(customer['quant'].get('layer_configs', []))} -> {len(cfg['quant'].get('layer_configs', []))}")
            else:
                print(f"  {k}: {a!r} -> {b!r}")


if __name__ == "__main__":
    main()
