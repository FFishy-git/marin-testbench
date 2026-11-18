# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the CPU-verified numerical candidate and prepare its bounded validation observer."""

import argparse
import ast
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from launch_wave5 import PYTHON, sha256
from prepare_upstream_outer import replace_once
from watch_wave5 import write
from wave7_release import candidate_inventory

CHANGES = {
    "experiment/train.py",
    "experiment/owned_output_observer.py",
    "experiment/tests/test_owned_output_observer.py",
}
HELPERS = ("prepare_owned_output_observer.py", "owned_output_observer.py", "owned_output_observer_checks.py")


def prepare(receipt: Path, candidate: Path) -> dict:
    """Add only observation code; all tested numerical/config bytes remain identical."""
    record = json.loads(receipt.read_text())
    previous = Path(record["candidate"])
    if (
        record["status"] != "owned_output_cpu_verified_gpu_denied"
        or type(record["exit_code"]) is not int
        or record["exit_code"] != 0
        or record["cuda_visible_devices"] != ""
        or candidate_inventory(previous) != record["files_sha256"]
        or sha256(Path(record["archive"])) != record["archive_sha256"]
        or sha256(Path(record["log"])) != record["log_sha256"]
        or sha256(Path(record["preparation"])) != record["preparation_sha256"]
    ):
        raise ValueError("Owned-factor predecessor CPU evidence differs")
    if candidate.exists():
        raise FileExistsError("Owned-output observer needs a distinct unused candidate")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    helpers = Path(__file__).parent
    shutil.copy2(helpers / "owned_output_observer.py", candidate / "experiment/owned_output_observer.py")
    shutil.copy2(
        helpers / "owned_output_observer_checks.py", candidate / "experiment/tests/test_owned_output_observer.py"
    )
    path = candidate / "experiment/train.py"
    text = replace_once(
        path.read_text(),
        "from .outer_diagonal_checks import OuterDiagonalVerifier",
        "from .outer_diagonal_checks import OuterDiagonalVerifier\n"
        "from .owned_output_observer import OwnedOutputVerifier",
    )
    text = replace_once(
        text,
        "from .block_toner import BLOCK_ARMS, BlockToner, DiagonalMoments",
        "from .block_toner import BLOCK_ARMS, BlockToner, DiagonalMoments, OwnedOutputMoments",
    )
    text = replace_once(
        text,
        "        outer_verifier = UpstreamProductVerifier(fast) "
        "if fast.arm.upstream_product_kernels else OuterDiagonalVerifier(fast)",
        "        verifier = (\n"
        "            OwnedOutputVerifier if isinstance(fast.output_diagonal, OwnedOutputMoments) "
        "else UpstreamProductVerifier\n"
        "        )\n"
        "        outer_verifier = verifier(fast) if fast.arm.upstream_product_kernels else OuterDiagonalVerifier(fast)",
    )
    text = replace_once(
        text,
        "from .upstream_full_model_checks import UpstreamProductVerifier\nfrom .records import (",
        "from .records import (",
    )
    text = replace_once(
        text,
        "    utc_now,\n)\n",
        "    utc_now,\n)\nfrom .upstream_full_model_checks import UpstreamProductVerifier\n",
    )
    path.write_text(text)
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | record["files_sha256"] if files.get(name) != record["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError(f"Owned-output observer changes exceed observation scope: {sorted(changed ^ CHANGES)}")
    previous_tree = ast.parse((previous / "experiment/block_toner.py").read_text())
    current_tree = ast.parse((candidate / "experiment/block_toner.py").read_text())
    assert ast.dump(previous_tree) == ast.dump(current_tree)
    result = {
        "status": "owned_output_observer_prepared_gpu_denied",
        "candidate": str(candidate.resolve()),
        "predecessor_cpu_receipt": str(receipt.resolve()),
        "predecessor_cpu_receipt_sha256": sha256(receipt),
        "files_sha256": files,
        "changed_files": sorted(changed),
        "helpers_sha256": {name: sha256(helpers / name) for name in HELPERS},
        "policy": "Original CPU-verified numerics/configs preserved. Owned-factor observation only. No GPU release.",
    }
    write(candidate / "owned-output-observer-preparation.json", result)
    return result


def cpu(candidate: Path, output: Path) -> dict:
    """Archive the observer successor's full CPU result separately from numerical preparation."""
    path = candidate / "owned-output-observer-preparation.json"
    prepared = json.loads(path.read_text())
    files = candidate_inventory(candidate)
    helpers = Path(__file__).parent
    hashes = {name: sha256(helpers / name) for name in HELPERS}
    receipt = Path(prepared["predecessor_cpu_receipt"])
    if (
        files != prepared["files_sha256"]
        or prepared["candidate"] != str(candidate.resolve())
        or hashes != prepared["helpers_sha256"]
        or sha256(receipt) != prepared["predecessor_cpu_receipt_sha256"]
    ):
        raise ValueError("Owned-output observer preparation changed before CPU evidence")
    if output.exists() or output.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Owned-output observer CPU evidence needs a new external directory")
    output.mkdir(parents=True, exist_ok=False)
    command = [
        PYTHON,
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "-p",
        "no:cacheprovider",
        "-m",
        "not slow and not integration and not data_integration and not cluster "
        "and not requires_cluster and not docker and not manual",
        "experiment/tests",
        "-q",
    ]
    log = output / "candidate.log"
    with log.open("x") as stream:
        completed = subprocess.run(
            command,
            cwd=candidate,
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    if files != candidate_inventory(candidate) or hashes != {name: sha256(helpers / name) for name in HELPERS}:
        raise ValueError("Owned-output observer source changed during CPU tests")
    archive = output / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in files:
            bundle.add(candidate / name, arcname=f"candidate/{name}", recursive=False)
        bundle.add(
            candidate / "artifacts/external-sources", arcname="candidate/artifacts/external-sources", recursive=False
        )
        bundle.add(path, arcname="candidate/owned-output-observer-preparation.json", recursive=False)
        for name in HELPERS:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    result = {
        **prepared,
        "status": (
            "owned_output_observer_cpu_verified_gpu_denied"
            if completed.returncode == 0
            else "owned_output_observer_cpu_failed"
        ),
        "preparation": str(path),
        "preparation_sha256": sha256(path),
        "command": command,
        "cwd": str(candidate),
        "exit_code": completed.returncode,
        "log": str(log),
        "log_sha256": sha256(log),
        "cuda_visible_devices": "",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
    write(output / "receipt.json", result)
    if completed.returncode:
        raise RuntimeError(f"Owned-output observer CPU failure preserved at {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "cpu"), default="prepare")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare" and args.cpu_receipt is None:
        parser.error("Preparation requires --cpu-receipt")
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU evidence requires --out")
    result = prepare(args.cpu_receipt, args.candidate) if args.phase == "prepare" else cpu(args.candidate, args.out)
    print(json.dumps({"status": result["status"], "candidate": result["candidate"]}))


if __name__ == "__main__":
    main()
