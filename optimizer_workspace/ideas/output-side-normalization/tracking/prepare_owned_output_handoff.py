# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the issued validator and prepare only a distinct handoff ownership identity."""

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from launch_wave5 import PYTHON, read, sha256
from prepare_owned_output_validation import cpu_command
from watch_wave5 import write
from wave7_release import candidate_inventory

PREVIOUS = "run-preparation/owned-output-full-model-binding.json"
PREPARATION = "owned-output-handoff-preparation.json"
STATUS = "owned_output_handoff_prepared_gpu_denied"
CPU_STATUS = "owned_output_handoff_cpu_verified_gpu_denied"
HELPERS = (
    "prepare_owned_output_handoff.py",
    "test_prepare_owned_output_handoff.py",
    "owned_output_handoff.py",
    "test_owned_output_handoff.py",
)
CHANGES = {
    "experiment/execution_owner.py",
    "experiment/tests/test_owned_output_execution_owner.py",
    "experiment/cluster.json",
}
REPLACEMENTS = (
    ("run_owned_output_validation.py", "run_owned_output_handoff.py"),
    ("owned-output-full-model-binding.json", "owned-output-handoff-binding.json"),
    ("owned-output-full-model-state.json", "owned-output-handoff-state.json"),
    ("owned-output-full-model-validation-20261117", "owned-output-handoff-validation-20261117"),
    ("owned_output_full_model", "owned_output_handoff"),
    ("validation-owned-output-full-model-gpu", "validation-owned-output-handoff-gpu"),
)


def require_original(root: Path) -> dict:
    """Revalidate the issued identity without altering any prior helper or receipt."""
    frozen = read(root / PREVIOUS)
    helpers = Path(frozen["helpers"])
    if (
        frozen["root"] != str(root)
        or frozen["status"] != "frozen_pending_completed_ceiling_and_owned_gpu_validation"
        or frozen["helpers"] != str(root / "controllers/owned-output-full-model-validation-20261117/tracking")
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or frozen["files_sha256"] != candidate_inventory(Path(frozen["candidate"]))
        or sha256(Path(frozen["cpu_receipt"])) != frozen["cpu_receipt_sha256"]
        or sha256(Path(frozen["controller_cpu_receipt"])) != frozen["controller_cpu_receipt_sha256"]
    ):
        raise ValueError("Original owned-output validation binding changed")
    subprocess.run(
        [PYTHON, str(helpers / "run_owned_output_validation.py"), "--root", str(root), "--phase", "check"],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    return frozen


def transformed(text: str) -> str:
    for old, new in REPLACEMENTS:
        if old not in text:
            raise ValueError(f"Ownership transformation is missing its source identity: {old}")
        text = text.replace(old, new)
    return text


def expected_cluster(previous: Path) -> dict:
    cluster = read(previous / "experiment/cluster.json")
    cluster["wave_name"] = "owned-output-handoff-validation-unreleased"
    return cluster


def prepare(root: Path, candidate: Path) -> dict:
    root = root.resolve()
    frozen = require_original(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.resolve().is_relative_to(previous.resolve()):
        raise FileExistsError("Owned handoff requires a distinct unused candidate")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in CHANGES - {"experiment/cluster.json"}:
        (candidate / name).write_text(transformed((previous / name).read_text()))
    write(candidate / "experiment/cluster.json", expected_cluster(previous))
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | frozen["files_sha256"] if files.get(name) != frozen["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError("Owned handoff exceeds its ownership-only changed files")
    result = {
        "status": STATUS,
        "root": str(root),
        "candidate": str(candidate.resolve()),
        "original_binding_sha256": sha256(root / PREVIOUS),
        "original_candidate": str(previous),
        "files_sha256": files,
        "changed_files": sorted(CHANGES),
        "preparation_helper_sha256": sha256(Path(__file__)),
        "policy": (
            "Ownership identity only; numerical training/kernel/config/validation bytes unchanged. No GPU release."
        ),
    }
    write(candidate / PREPARATION, result)
    return result


def require_preparation(candidate: Path) -> dict:
    initial = read(candidate / PREPARATION)
    root = Path(initial["root"])
    frozen = require_original(root)
    previous = Path(frozen["candidate"])
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | frozen["files_sha256"] if files.get(name) != frozen["files_sha256"].get(name)
    }
    cluster = read(candidate / "experiment/cluster.json")
    if (
        initial["status"] != STATUS
        or initial["candidate"] != str(candidate.resolve())
        or initial["original_binding_sha256"] != sha256(root / PREVIOUS)
        or initial["original_candidate"] != str(previous)
        or initial["preparation_helper_sha256"] != sha256(Path(__file__))
        or initial["files_sha256"] != files
        or initial["changed_files"] != sorted(CHANGES)
        or changed != CHANGES
        or any(
            (candidate / name).read_text() != transformed((previous / name).read_text())
            for name in CHANGES
            if name.endswith(".py")
        )
        or {key: value for key, value in cluster.items() if key != "time"}
        != {key: value for key, value in expected_cluster(previous).items() if key != "time"}
    ):
        raise ValueError("Owned handoff preparation or ownership-only source changed")
    return initial


def require_cpu_archive(archive: Path, initial: dict, preparation_sha256: str) -> None:
    expected = {f"candidate/{name}": digest for name, digest in initial["files_sha256"].items()}
    expected.update(
        {
            f"candidate/{PREPARATION}": preparation_sha256,
            "tracking/prepare_owned_output_handoff.py": initial["preparation_helper_sha256"],
        }
    )
    alias = "candidate/artifacts/external-sources"
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) + 1 or {member.name for member in members} != set(expected) | {alias}:
            raise ValueError("Owned handoff CPU archive inventory differs")
        link = bundle.getmember(alias)
        if not link.issym() or link.linkname != "pinned-sources-materialized":
            raise ValueError("Owned handoff archived pinned alias differs")
        for name, digest in expected.items():
            member = bundle.getmember(name)
            if not member.isfile():
                raise ValueError("Owned handoff archived source is not a regular file")
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("Owned handoff archive bytes differ from tested source")


def cpu(candidate: Path, out: Path) -> dict:
    initial = require_preparation(candidate)
    if out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Owned handoff CPU outcome requires a distinct external directory")
    out.mkdir(parents=True, exist_ok=False)
    preparation = candidate / PREPARATION
    preparation_hash = sha256(preparation)
    error = None
    log = out / "candidate.log"
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(),
                cwd=candidate,
                env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code = 124
            error = "CPU command timeout; exit 124 is a sentinel"
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in initial["files_sha256"]:
            bundle.add(candidate / name, arcname=f"candidate/{name}", recursive=False)
        bundle.add(
            candidate / "artifacts/external-sources", arcname="candidate/artifacts/external-sources", recursive=False
        )
        bundle.add(preparation, arcname=f"candidate/{PREPARATION}", recursive=False)
        bundle.add(Path(__file__), arcname="tracking/prepare_owned_output_handoff.py", recursive=False)
    try:
        if require_preparation(candidate) != initial or sha256(preparation) != preparation_hash:
            raise ValueError("Owned handoff source changed during CPU testing or archival")
        require_cpu_archive(archive, initial, preparation_hash)
    except ValueError as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        **initial,
        "status": CPU_STATUS if code == 0 and error is None else "owned_output_handoff_cpu_failed",
        "preparation": str(preparation),
        "preparation_sha256": preparation_hash,
        "command": cpu_command(),
        "cwd": str(candidate),
        "cuda_visible_devices": "",
        "exit_code": code,
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if code or error:
        raise RuntimeError(f"Owned handoff CPU failure preserved at {out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "cpu"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU verification requires --out")
    result = prepare(args.root, args.candidate) if args.phase == "prepare" else cpu(args.candidate, args.out)
    print(result["status"])


if __name__ == "__main__":
    main()
