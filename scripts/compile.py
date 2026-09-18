#!/usr/bin/env python3
"""Compile the AX620E configs with the Pulsar2 7.0 docker image.

License handling (no environment variable required):
  1. if MAGNETAR_HASP_SRC is set, mount that dir at /root/.hasplm;
  2. else if <repo>/.p2_home/.hasplm contains installed/32434/*.v2c, mount it;
  3. else use the v2c bundled in the docker image (wrapper copies /root/*.v2c
     into /root/.hasplm/installed/32434/ at container start).
The full docker stdout is streamed into logs/<name>.log in real time.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE = os.environ.get("PULSAR2_IMAGE", "docker-registry.aitsw.axera-tech.com/pulsar2:7.0")


def find_hasp() -> Path | None:
    env = os.environ.get("MAGNETAR_HASP_SRC")
    if env:
        return Path(env)
    cand = ROOT / ".p2_home" / ".hasplm"
    v2c_dir = cand / "installed" / "32434"
    if v2c_dir.is_dir() and any(v2c_dir.glob("*.v2c")):
        return cand
    return None


def build(name: str) -> int:
    log = ROOT / "logs" / f"{Path(name).stem}.log"
    uid, gid = os.getuid(), os.getgid()
    wrapped = (
        "set +e; mkdir -p /root/.hasplm/installed/32434 && "
        "cp -f /root/*.v2c /root/.hasplm/installed/32434/ 2>/dev/null; "
        f'PATH=/usr/local/bin/.venv/bin:/opt/pulsar2:$PATH sh -c "cd /workspace/configs && pulsar2 build --config {name}"; '
        f"status=$?; chown -R {uid}:{gid} /workspace; exit $status"
    )
    cmd = ["docker", "run", "--rm", "--network", "host", "-v", f"{ROOT}:/workspace"]
    if Path("/var/hasplm").exists():
        cmd += ["-v", "/var/hasplm:/var/hasplm"]
    hasp = find_hasp()
    if hasp:
        cmd += ["-v", f"{hasp}:/root/.hasplm"]
        print(f"license: {hasp}", flush=True)
    else:
        print("license: docker image built-in v2c", flush=True)
    cmd += ["-e", "HASP_HOME=/root/.hasplm", IMAGE, "-lc", wrapped]

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
