#!/usr/bin/env python3
"""Build dataset/images100.tar (100 real images) for Pulsar2 calibration."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
ZIP = DATASET / "coco128.zip"
OUT = DATASET / "images100.tar"
N = 100
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def main() -> None:
    if not ZIP.exists():
        import urllib.request

        DATASET.mkdir(parents=True, exist_ok=True)
        print(f"downloading {COCO128_URL} -> {ZIP}")
        urllib.request.urlretrieve(COCO128_URL, ZIP)
    with zipfile.ZipFile(ZIP) as z:
        images = sorted(n for n in z.namelist() if "/images/" in n and n.lower().endswith((".jpg", ".jpeg", ".png")))
        images = images[:N]
        assert len(images) == N, f"only {len(images)} images in coco128"
        with tarfile.open(OUT, "w") as tar:
            for name in images:
                data = z.read(name)
                info = tarfile.TarInfo(name=Path(name).name)
                info.size = len(data)
                info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {N} images)")


if __name__ == "__main__":
    main()
