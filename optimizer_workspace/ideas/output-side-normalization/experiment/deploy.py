# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy only task-owned files; refuse to overwrite divergent remote files."""

import argparse
import hashlib
import io
import json
import shlex
import subprocess
import tarfile

from .launch import NODES, PYTHON, SSH
from .records import atomic_json, sha256, utc_now
from .reference import TASK, UPSTREAM


def deploy():
    directory = TASK / "artifacts" / "run-preparation"
    receipt_path = directory / "deployment.json"
    previous = json.loads(receipt_path.read_text())["files"] if receipt_path.exists() else {}
    files = sorted(p for p in (TASK / "experiment").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files += sorted(p for p in UPSTREAM.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    mousse = TASK / "artifacts/external-sources/Mousse-d00c1bf17790fbe56424ee5567cce80d8e75f4b2"
    files += [mousse / "dion/dion/mousse.py", mousse / "LICENSE", mousse / "dion/LICENSE"]
    files += [TASK / "ENDEAVOR.md"]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(TASK)), recursive=False)
    payload = buffer.getvalue()
    # Compare all existing files before extracting any file; retries preserve identical snapshots.
    remote_script = (
        "import hashlib,io,pathlib,sys,tarfile\n"
        f"root=pathlib.Path({str(TASK)!r})\n"
        f"previous={previous!r}\n"
        "archive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz')\n"
        "members=archive.getmembers()\n"
        "for member in members:\n"
        " path=root/member.name\n"
        " if path.exists():\n"
        "  expected=archive.extractfile(member).read()\n"
        "  actual=hashlib.sha256(path.read_bytes()).hexdigest()\n"
        "  if actual!=hashlib.sha256(expected).hexdigest() and actual!=previous.get(member.name):\n"
        "   raise RuntimeError('Refusing divergent existing file: '+str(path))\n"
        "root.mkdir(parents=True,exist_ok=True)\n"
        "archive.extractall(root,members=members,filter='data')\n"
        "print('verified '+str(len(members))+' task files')\n"
    )
    records = []
    for node in NODES[1:]:
        result = subprocess.run(
            [*SSH, f"root@{node}", shlex.join([PYTHON, "-c", remote_script])],
            input=payload,
            capture_output=True,
            timeout=120,
        )
        if result.returncode:
            raise RuntimeError(f"Deployment to {node} failed: {result.stderr.decode()}")
        records.append({"node": node, "result": result.stdout.decode().strip()})
    receipt = {
        "time": utc_now(),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "files": {str(path.relative_to(TASK)): sha256(path) for path in files},
        "nodes": records,
    }
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt["nodes"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    deploy()


if __name__ == "__main__":
    main()
