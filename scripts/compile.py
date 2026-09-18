#!/usr/bin/env python3
"""Compile the AX620E configs with the Pulsar2 7.0 docker image.

Streams the full docker stdout into logs/<name>.log in real time so an interrupted
run still leaves a usable log.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE = os.environ.get("PULSAR2_IMAGE", "docker-registry.aitsw.axera-tech.com/pulsar2:7.0")
HASP = Path(os.environ.get("MAGNETAR_HASP_SRC", ROOT / ".p2_home" / ".hasplm"))


def build(name: str) -> int:
    log = ROOT / "logs" / f"{Path(name).stem}.log"
    uid, gid = os.getuid(), os.getgid()
    wrapped = (
        "set +e; mkdir -p /root/.hasplm/installed/32434 && "
        "cp -f /root/*.v2c /root/.hasplm/installed/32434/ 2>/dev/null; "
        f'PATH=/usr/local/bin/.venv/bin:/opt/pulsar2:$PATH sh -c "cd /workspace/configs && pulsar2 build --config {name}"; '
        f"status=$?; chown -R {uid}:{gid} /workspace; exit $status"
    )
    cmd = [
        "docker", "run", "--rm", "--network", "host",
        "-v", f"{ROOT}:/workspace",
        "-v", "/var/hasplm:/var/hasplm",
        "-v", f"{HASP}:/root/.hasplm",
        "-e", "HASP_HOME=/root/.hasplm",
        IMAGE, "-lc", wrapped,
    ]
    print(f"=== compiling {name} -> {log} ===", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def main() -> None:
    names = sys.argv[1:] or ["u8.json", "u16.json", "mix.json"]
    for name in names:
        rc = build(name)
        print(f"=== {name}: exit {rc} ===", flush=True)
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
