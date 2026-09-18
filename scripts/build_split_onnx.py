#!/usr/bin/env python3
"""Build the customer-shaped yolo26n ONNX: one2one (end2end) head split into 3 per-scale outputs.

Outputs: output_split_0/1/2 -> (1, 84, H, W) raw logits for P3/P4/P5, NCHW.
This matches exp_e3_mix.json layer_configs (one2one_* conv nodes) and output_processors
(dst_perm [0,2,3,1]).
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "onnx"
RAW = ONNX_DIR / "yolo26n_raw.onnx"
OUT = ONNX_DIR / "yolo26n.onnx"
CUSTOMER_CFG = ROOT / "configs" / "customer_exp_e3_mix.json"

sys.path.insert(0, os.environ.get("ULTRALYTICS_REPO", "/data/shared/huyuan/YOLO/ultralytics"))


def raw_one2one_forward(self, x):
    """Detect.forward replacement: raw one2one per-scale (B, 4+nc, H, W) outputs, no decode."""
    import torch

    outs = []
    for i in range(self.nl):
        box = self.one2one_cv2[i](x[i])  # (B, 4*reg_max, H, W); reg_max=1 -> (B, 4, H, W)
        cls = self.one2one_cv3[i](x[i])  # (B, nc, H, W)
        outs.append(torch.cat((box, cls), 1))
    return tuple(outs)


def main() -> None:
    import torch

    from ultralytics import YOLO

    model = YOLO(str(ONNX_DIR / "yolo26n.pt"))
    torch_model = model.model.eval().float()
    head = torch_model.model[-1]
    assert getattr(head, "one2one_cv2", None) is not None, "checkpoint has no one2one head"
    assert head.reg_max == 1, f"expected reg_max=1, got {head.reg_max}"

    head.forward = types.MethodType(raw_one2one_forward, head)

    im = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    with torch.no_grad():
        out = torch_model(im)
    print("torch outputs:", [tuple(o.shape) for o in out])

    torch.onnx.export(
        torch_model,
        im,
        str(OUT),
        opset_version=17,
        input_names=["images"],
        output_names=["output_split_0", "output_split_1", "output_split_2"],
        dynamic_axes=None,
        do_constant_folding=True,
    )
    print(f"exported: {OUT}")

    # verify node names hit every customer layer name
    import onnx

    m = onnx.load(str(OUT))
    names = {n.name for n in m.graph.node}
    cfg = json.loads(CUSTOMER_CFG.read_text())
    missing = [n for lc in cfg["quant"]["layer_configs"] for n in lc["layer_names"] if n not in names]
    print(f"customer layer_configs missing: {len(missing)}")
    for n in missing:
        print("  MISSING", n)

    print("onnx inputs :", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
    print("onnx outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])

    # numeric check torch vs onnxruntime
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.random((1, 3, 640, 640), dtype=np.float32)
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {"images": x})
    with torch.no_grad():
        torch_outs = torch_model(torch.from_numpy(x))
    for i, (a, b) in enumerate(zip(ort_outs, torch_outs)):
        a = a.reshape(-1).astype(np.float64)
        b = b.numpy().reshape(-1).astype(np.float64)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"output_split_{i}: shape={a.shape} cosine={cos:.8f} max_abs={np.abs(a-b).max():.3e}")

    RAW.unlink(missing_ok=True)
    print("removed intermediate:", RAW)


if __name__ == "__main__":
    main()
