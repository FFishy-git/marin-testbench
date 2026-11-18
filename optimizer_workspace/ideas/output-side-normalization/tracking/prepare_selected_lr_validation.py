# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare distinct validation-only ownership without changing the issued LR preparations."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from launch_wave5 import read, sha256
from prepare_selected_lr_full_budget import PREPARATION as PARENT_PREPARATION
from prepare_selected_lr_full_budget import require_preparation
from prepare_upstream_outer import replace_once
from selected_lr_validation import PREPARATION, plan
from upstream_production import cpu_command
from watch_wave5 import write
from wave7_release import candidate_inventory

STATUS = "selected_lr_validation_prepared_no_gpu_binding"
CPU_STATUS = "selected_lr_validation_cpu_verified_no_gpu_binding"
CHANGES = {
    "experiment/train.py",
    "experiment/selected_lr_owner.py",
    "experiment/selected_lr_validation.py",
    "experiment/tests/test_selected_lr_execution_owner.py",
}
HELPERS = (
    "prepare_selected_lr_validation.py",
    "selected_lr_execution_owner.py",
    "selected_lr_validation.py",
    "test_selected_lr_execution_owner.py",
)


def require_archive(path: Path, expected: dict, *, alias: bool) -> None:
    with tarfile.open(path) as bundle:
        members = bundle.getmembers()
        names = set(expected) | ({"artifacts/external-sources"} if alias else set())
        if len(members) != len(names) or {member.name for member in members} != names:
            raise ValueError("Selected-LR source archive inventory differs")
        if alias:
            link = bundle.getmember("artifacts/external-sources")
            if not link.issym() or link.linkname != "pinned-sources-materialized":
                raise ValueError("Selected-LR source archive has an unexpected alias")
        for name, digest in expected.items():
            member = bundle.getmember(name)
            if not member.isfile():
                raise ValueError("Selected-LR archived source contains an alias")
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("Selected-LR archived bytes differ from the executed source")


def require_cpu_source(receipt_path: Path) -> tuple[Path, dict]:
    receipt = read(receipt_path)
    previous = Path(receipt["candidate"])
    prepared = require_preparation(previous)
    helper = Path(__file__).with_name("prepare_selected_lr_full_budget.py")
    if (
        receipt["status"] != "selected_lr_full_budget_cpu_verified_gpu_denied"
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["command"] != cpu_command()
        or receipt["cwd"] != str(previous)
        or receipt["files_sha256"] != prepared["files_sha256"]
        or receipt["preparation_sha256"] != sha256(previous / PARENT_PREPARATION)
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
        or "error" in receipt
    ):
        raise ValueError("Selected-LR corrected CPU preparation is incomplete or changed")
    require_archive(
        Path(receipt["archive"]),
        prepared["files_sha256"]
        | {
            PARENT_PREPARATION: receipt["preparation_sha256"],
            "tracking/prepare_selected_lr_full_budget.py": sha256(helper),
        },
        alias=True,
    )
    return previous, prepared


def transformed_train(previous: Path) -> str:
    text = (previous / "experiment/train.py").read_text()
    text = replace_once(
        text,
        "from .selected_lr_owner import require_selected_lr_owner",
        "from .selected_lr_owner import require_selected_lr_training",
    )
    return replace_once(text, "    require_selected_lr_owner()\n", "    require_selected_lr_training(args)\n")


def prepare(receipt_path: Path, candidate: Path) -> dict:
    previous, parent = require_cpu_source(receipt_path)
    candidate = candidate.resolve()
    if candidate.exists() or candidate.is_relative_to(previous.resolve()):
        raise FileExistsError("Selected-LR validation needs a distinct unused candidate")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    helpers = Path(__file__).parent
    for source, destination in (
        ("selected_lr_execution_owner.py", "experiment/selected_lr_owner.py"),
        ("selected_lr_validation.py", "experiment/selected_lr_validation.py"),
        ("test_selected_lr_execution_owner.py", "experiment/tests/test_selected_lr_execution_owner.py"),
    ):
        shutil.copy2(helpers / source, candidate / destination)
    (candidate / "experiment/train.py").write_text(transformed_train(previous))
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | parent["files_sha256"] if files.get(name) != parent["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError("Selected-LR validation successor exceeds its ownership/test-only transformation")
    root = Path(parent["root"])
    result = {
        "status": STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_cpu_receipt": str(receipt_path.resolve()),
        "parent_cpu_receipt_sha256": sha256(receipt_path),
        "parent_preparation_sha256": sha256(previous / PARENT_PREPARATION),
        "files_sha256": files,
        "changed_files": sorted(CHANGES),
        "data_receipt_sha256": parent["data_receipt_sha256"],
        "helpers_sha256": {name: sha256(helpers / name) for name in HELPERS},
        "plan": plan(candidate, root, root / "validation/wave11_selected_lr_full_budget"),
        "policy": (
            "Preparation grants no GPU execution. Both issued parent identities remain unchanged. "
            "A separate archived, CPU-verified controller binding and completed Wave10 acceptance precede "
            "bounded actual-config validation. No queue, production authorization or retry."
        ),
    }
    write(candidate / PREPARATION, result)
    return result


def require_candidate(candidate: Path) -> dict:
    prepared = read(candidate / PREPARATION)
    previous, parent = require_cpu_source(Path(prepared["parent_cpu_receipt"]))
    helpers = Path(__file__).parent
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | parent["files_sha256"] if files.get(name) != parent["files_sha256"].get(name)
    }
    root = Path(prepared["root"])
    if (
        prepared["status"] != STATUS
        or prepared["candidate"] != str(candidate.resolve())
        or prepared["root"] != parent["root"]
        or prepared["parent_candidate"] != str(previous)
        or prepared["parent_cpu_receipt_sha256"] != sha256(Path(prepared["parent_cpu_receipt"]))
        or prepared["parent_preparation_sha256"] != sha256(previous / PARENT_PREPARATION)
        or prepared["files_sha256"] != files
        or changed != CHANGES
        or set(prepared["changed_files"]) != CHANGES
        or prepared["data_receipt_sha256"] != parent["data_receipt_sha256"]
        or prepared["helpers_sha256"] != {name: sha256(helpers / name) for name in HELPERS}
        or (candidate / "experiment/train.py").read_text() != transformed_train(previous)
        or prepared["plan"] != plan(candidate, root, root / "validation/wave11_selected_lr_full_budget")
        or any(
            (candidate / destination).read_bytes() != (helpers / source).read_bytes()
            for source, destination in (
                ("selected_lr_execution_owner.py", "experiment/selected_lr_owner.py"),
                ("selected_lr_validation.py", "experiment/selected_lr_validation.py"),
                ("test_selected_lr_execution_owner.py", "experiment/tests/test_selected_lr_execution_owner.py"),
            )
        )
    ):
        raise ValueError("Selected-LR validation source, parent provenance or bounded plan changed")
    return prepared


def cpu(candidate: Path, out: Path) -> dict:
    """Preserve success or failure and archive exactly the ownership successor tested on CPU."""
    prepared = require_candidate(candidate)
    if out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Selected-LR validation CPU evidence requires an unused external directory")
    out.mkdir(parents=True)
    command, log, archive = cpu_command(), out / "candidate.log", out / "source.tar"
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                command,
                cwd=candidate,
                env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "CPU timeout at 600 seconds"
    helper_hashes = {name: sha256(Path(__file__).parent / name) for name in HELPERS}
    preparation_hash = sha256(candidate / PREPARATION)
    expected = prepared["files_sha256"] | {PREPARATION: preparation_hash}
    expected |= {f"tracking/{name}": digest for name, digest in helper_hashes.items()}
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in prepared["files_sha256"]:
                bundle.add(candidate / name, arcname=name, recursive=False)
            bundle.add(candidate / PREPARATION, arcname=PREPARATION, recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
            for name in HELPERS:
                bundle.add(Path(__file__).parent / name, arcname=f"tracking/{name}", recursive=False)
        require_archive(archive, expected, alias=True)
        if require_candidate(candidate) != prepared or preparation_hash != sha256(candidate / PREPARATION):
            raise ValueError("Selected-LR validation source changed during CPU verification")
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": CPU_STATUS if code == 0 and error is None else "selected_lr_validation_cpu_failed",
        "candidate": str(candidate.resolve()),
        "preparation_sha256": preparation_hash,
        "files_sha256": prepared["files_sha256"],
        "helpers_sha256": helper_hashes,
        "command": command,
        "cwd": str(candidate.resolve()),
        "cuda_visible_devices": "",
        "exit_code": code,
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive) if archive.is_file() else None,
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CPU_STATUS:
        raise RuntimeError(f"Selected-LR validation CPU failure preserved at {out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "check", "cpu"), required=True)
    parser.add_argument("--parent-cpu-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare" and args.parent_cpu_receipt is None:
        parser.error("Preparation requires --parent-cpu-receipt")
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU verification requires --out")
    result = (
        prepare(args.parent_cpu_receipt, args.candidate)
        if args.phase == "prepare"
        else cpu(args.candidate, args.out) if args.phase == "cpu" else require_candidate(args.candidate)
    )
    print(json.dumps({"status": result["status"], "candidate": result["candidate"]}))


if __name__ == "__main__":
    main()
