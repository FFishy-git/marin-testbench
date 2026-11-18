# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a CPU-checked isolated candidate to its gated wave-4 successor queue."""

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root: Path, candidate: Path) -> dict:
    preparation = root / "run-preparation"
    receipt_path = preparation / "wave5-candidate.json"
    archive_path = preparation / "source-wave5-fused-diagonal-20261116.tar"
    if receipt_path.exists() or archive_path.exists():
        raise FileExistsError("Candidate freeze already has evidence; inspect it instead of replacing it")
    cpu_path = preparation / "validation-wave5-cpu.json"
    cpu = json.loads(cpu_path.read_text())
    sources = {
        str(path.relative_to(candidate)): sha256(path) for path in sorted((candidate / "experiment").rglob("*.py"))
    }
    if sources != cpu["source_sha256"] or cpu["evidence"]["cpu_test_exit_code"] != 0:
        raise ValueError("Final candidate CPU receipt is stale or failed")
    original = json.loads((preparation / "validation-wave4.json").read_text())
    if original["source_sha256"] != {
        str(path.relative_to(TASK)): sha256(path) for path in sorted((TASK / "experiment").rglob("*.py"))
    }:
        raise ValueError("Active wave-4 experiment source changed")
    seed2 = json.loads((preparation / "seed2-companion-queue.json").read_text())
    if any(sha256(TASK / name) != expected for name, expected in seed2["helpers_sha256"].items()):
        raise ValueError("Bound seed-2 helpers changed")
    materialized = candidate / "artifacts/pinned-sources-materialized"
    if (candidate / "artifacts/external-sources").readlink() != Path("pinned-sources-materialized"):
        raise ValueError("Candidate pinned sources must resolve within the archived tree")
    shutil.copyfile(TASK / "ENDEAVOR.md", candidate / "ENDEAVOR.md")
    files = sorted(
        [
            *(candidate / "experiment").rglob("*.py"),
            *(candidate / "experiment").rglob("*.json"),
            *(path for path in materialized.rglob("*") if path.is_file() and "__pycache__" not in path.parts),
        ]
    )
    with tarfile.open(archive_path, "x") as archive:
        for path in [candidate / "ENDEAVOR.md", candidate / "experiment/README.md", *files]:
            archive.add(path, arcname=str(path.relative_to(candidate)), recursive=False)
        archive.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    receipt = {
        "time": datetime.now(UTC).isoformat(),
        "status": "frozen_pending_gpu_validation",
        "candidate": str(candidate.resolve()),
        "files_sha256": {str(path.relative_to(candidate)): sha256(path) for path in files},
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": sha256(cpu_path),
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "validator_sha256": sha256(TASK / "tracking/validate_wave5.py"),
        "queue_script_sha256": sha256(TASK / "tracking/queue_wave5_validation.sh"),
        "predecessor_pid": 2715673,
        "node": "28.216.17.9",
        "release_policy": "GPU validation only after predecessor completes and keeper restores; no full-budget launch",
    }
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    receipt = freeze(args.root, args.candidate)
    print(json.dumps({name: receipt[name] for name in ("status", "archive", "archive_sha256")}, indent=2))


if __name__ == "__main__":
    main()
