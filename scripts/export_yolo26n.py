#!/usr/bin/env python3
"""Export public ultralytics yolo26n.pt to fixed-shape 640 ONNX for the AX620E repro."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "onnx"
WEIGHTS = ONNX_DIR / "yolo26n.pt"
RAW = ONNX_DIR / "yolo26n_raw.onnx"

os.chdir(ONNX_DIR)

from ultralytics import YOLO  # noqa: E402


def main() -> None:
    if not WEIGHTS.exists():
        print(f"downloading weights to {WEIGHTS} ...")
        YOLO("yolo26n.pt")  # auto-downloads into cwd
    if not WEIGHTS.exists():
        raise SystemExit(f"weights not found: {WEIGHTS}")

    model = YOLO(str(WEIGHTS))
    out = model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=True,
        dynamic=False,
        batch=1,
        device="cpu",
        half=False,
    )
    out = Path(out)
    if out.resolve() != RAW.resolve():
        RAW.unlink(missing_ok=True)
        out.rename(RAW)
    print(f"exported: {RAW}")


if __name__ == "__main__":
    main()
