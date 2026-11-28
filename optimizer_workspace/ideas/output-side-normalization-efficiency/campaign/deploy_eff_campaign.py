# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy the self-contained efficiency workspace to all 4 cluster nodes."""

from __future__ import annotations

import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
PYTHON = "/opt/conda/envs/torch-base/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]


def deploy() -> None:
    files = []
    for sub in ("experiment", "campaign", "configs", "artifacts"):
        base = TASK / sub
        if base.exists():
            files += sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for top in ("PLAN.md", "README.md"):
        p = TASK / top
        if p.is_file():
            files.append(p)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(TASK)), recursive=False)
    payload = buf.getvalue()

    remote_script = (
        "import io, pathlib, sys, tarfile\n"
        f"root = pathlib.Path({str(TASK)!r})\n"
        "root.mkdir(parents=True, exist_ok=True)\n"
        "with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()), mode='r:gz') as tar:\n"
        "    members = tar.getmembers()\n"
        "    tar.extractall(root, members=members, filter='data')\n"
        "print(f'deployed {len(members)} files to {root}')\n"
    )

    local_ips = set(subprocess.check_output(["hostname", "-I"], text=True).split())
    results = []
    for node in NODES:
        if node in local_ips:
            results.append({"node": node, "output": f"local node ({len(files)} files verified)"})
            continue
        proc = subprocess.run(
            [*SSH, f"root@{node}", shlex.join([PYTHON, "-c", remote_script])],
            input=payload,
            capture_output=True,
            timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Deploy to {node} failed: {proc.stderr.decode()}")
        results.append({"node": node, "output": proc.stdout.decode().strip()})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    deploy()
