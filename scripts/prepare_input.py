#!/usr/bin/env python3
"""Make a fixed U8 640x640x3 NHWC input tensor (npy) for the board benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def letterbox(image: np.ndarray, size: int = 640) -> np.ndarray:
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    nw, nh = round(w * ratio), round(h * ratio)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, right = (size - nw) // 2, size - nw - (size - nw) // 2
    top, bottom = (size - nh) // 2, size - nh - (size - nh) // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/data/shared/huyuan/YOLO/YOLO26.axera/bus.jpg")
    ap.add_argument("--out", default=str(ROOT / "board" / "input_640.npy"))
    args = ap.parse_args()

    img = cv2.imread(args.image)
    assert img is not None, args.image
    lb = letterbox(img, 640)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    tensor = rgb[None, ...].astype(np.uint8)  # (1,640,640,3) NHWC U8
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, tensor)
    cv2.imwrite(str(out.with_suffix(".jpg")), lb)
    print(f"saved {out} shape={tensor.shape} dtype={tensor.dtype} max={tensor.max()}")


if __name__ == "__main__":
    main()
