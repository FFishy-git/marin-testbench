# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a distinct Wave11 release owner while preserving issued candidates."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from launch_wave5 import read, sha256
from prepare_selected_lr_production import original
from prepare_selected_lr_validation import require_archive
from prepare_upstream_outer import replace_once
from selected_lr_release_owner import CHANGES, CPU_STATUS, PREPARATION, PREPARED_STATUS, transformed_train
from selected_lr_validation import PREPARATION as PARENT_PREPARATION
from selected_lr_validation import locations
from upstream_production import cpu_command
from watch_wave5 import write
from wave7_release import candidate_inventory

STATUS = PREPARED_STATUS
COPIES = {
    "selected_lr_release_owner.py": "experiment/selected_lr_release_owner.py",
    "test_selected_lr_release_owner.py": "experiment/tests/test_selected_lr_release_owner.py",
}
HELPERS = ("prepare_selected_lr_release.py", *COPIES)
POLICY = (
    "Ownership-only preparation is not a GPU release. Preserve original validation and denied "
    "production candidates. A separate frozen production controller must reconstruct all twelve "
    "original commands/thirty gates, all ranks, original process absence and exact keeper restoration "
    "before issuing generation-bound four-node ownership. No freeze, launch, allocation or retry here."
)


def helper_hashes() -> dict:
    return {name: sha256(Path(__file__).parent / name) for name in HELPERS}


def transformed_files(previous: Path) -> dict:
    cluster = read(previous / "experiment/cluster.json")
    cluster["wave_name"] = "wave11"
    contents = {destination: (Path(__file__).parent / source).read_bytes() for source, destination in COPIES.items()}
    test_path = COPIES["test_selected_lr_release_owner.py"]
    contents[test_path] = replace_once(
        contents[test_path].decode(),
        "import selected_lr_release_owner as owner",
        "from experiment import selected_lr_release_owner as owner",
    ).encode()
    return {
        **contents,
        "experiment/train.py": transformed_train(previous),
        "experiment/cluster.json": (json.dumps(cluster, indent=2) + "\n").encode(),
    }


def preparation_record(root: Path, candidate: Path, frozen: dict, files: dict) -> dict:
    return {
        "status": STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": frozen["candidate"],
        "validation_binding": str(locations(root)[0]),
        "validation_binding_sha256": sha256(locations(root)[0]),
        "parent_cpu_receipt": frozen["cpu_receipt"],
        "parent_cpu_receipt_sha256": frozen["cpu_receipt_sha256"],
        "parent_controller_cpu_receipt": frozen["controller_cpu_receipt"],
        "parent_controller_cpu_receipt_sha256": frozen["controller_cpu_receipt_sha256"],
        "parent_archive": frozen["archive"],
        "parent_archive_sha256": frozen["archive_sha256"],
        "parent_preparation_sha256": frozen["preparation_sha256"],
        "data_receipt_sha256": frozen["data_receipt_sha256"],
        "files_sha256": files,
        "changed_files": sorted(CHANGES),
        "helpers_sha256": helper_hashes(),
        "policy": POLICY,
    }


def prepare(root: Path, candidate: Path) -> dict:
    root, candidate = root.resolve(), candidate.resolve()
    frozen = original(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.is_relative_to(previous) or candidate.is_relative_to(Path(frozen["helpers"])):
        raise FileExistsError("Wave11 release preparation needs a distinct unused candidate")
    if any(name in frozen["files_sha256"] for name in COPIES.values()):
        raise ValueError("Original Wave11 source already contains a release successor")
    contents, hashes = transformed_files(previous), helper_hashes()
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, content in contents.items():
        (candidate / name).write_bytes(content)
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError("Wave11 release preparation exceeds its exact four-file ownership scope")
    prepared = preparation_record(root, candidate, frozen, files)
    if prepared["helpers_sha256"] != hashes:
        raise ValueError("Wave11 release helpers changed during preparation")
    write(candidate / PREPARATION, prepared)
    require_preparation(root, candidate)
    return prepared


def require_preparation(root: Path, candidate: Path) -> dict:
    root, candidate = root.resolve(), candidate.resolve()
    frozen = original(root)
    previous = Path(frozen["candidate"])
    prepared, files = read(candidate / PREPARATION), candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    expected = preparation_record(root, candidate, frozen, files)
    if (
        any(prepared[key] != value for key, value in expected.items())
        or any(key in prepared for key in ("error", "errors"))
        or changed != CHANGES
        or sha256(candidate / PARENT_PREPARATION) != frozen["preparation_sha256"]
        or any((candidate / name).read_bytes() != content for name, content in transformed_files(previous).items())
    ):
        raise ValueError("Wave11 release preparation or exact numerical/ownership scope differs")
    return prepared


def archive_inventory(prepared: dict, preparation_digest: str) -> dict:
    return (
        prepared["files_sha256"]
        | {PREPARATION: preparation_digest}
        | {f"tracking/{name}": digest for name, digest in prepared["helpers_sha256"].items()}
    )


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    root, candidate, out = root.resolve(), candidate.resolve(), out.resolve()
    prepared = require_preparation(root, candidate)
    if out.exists() or any(out.is_relative_to(path) for path in (candidate, Path(prepared["parent_candidate"]))):
        raise FileExistsError("Wave11 release CPU evidence needs an unused directory outside both candidates")
    digest = sha256(candidate / PREPARATION)
    out.mkdir(parents=True, exist_ok=False)
    command, log, archive = cpu_command(), out / "candidate.log", out / "source.tar"
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                command,
                cwd=candidate,
                env=os.environ
                | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(candidate)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "CPU timeout at 600 seconds"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in prepared["files_sha256"]:
                bundle.add(candidate / name, arcname=name, recursive=False)
            bundle.add(candidate / PREPARATION, arcname=PREPARATION, recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
            for name in HELPERS:
                bundle.add(Path(__file__).parent / name, arcname=f"tracking/{name}", recursive=False)
        require_archive(archive, archive_inventory(prepared, digest), alias=True)
        if require_preparation(root, candidate) != prepared or sha256(candidate / PREPARATION) != digest:
            raise ValueError("Wave11 release source changed during CPU verification")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": CPU_STATUS if type(code) is int and code == 0 and error is None else "selected_lr_release_cpu_failed",
        "root": str(root),
        "candidate": str(candidate),
        "preparation_sha256": digest,
        "files_sha256": prepared["files_sha256"],
        "helpers_sha256": prepared["helpers_sha256"],
        "validation_binding_sha256": prepared["validation_binding_sha256"],
        "command": command,
        "cwd": str(candidate),
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
        raise RuntimeError(f"Wave11 release CPU failure preserved at {out}")
    return result


def require_cpu(root: Path, receipt_path: Path) -> dict:
    receipt = read(receipt_path)
    candidate = Path(receipt["candidate"])
    prepared = require_preparation(root, candidate)
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root.resolve())
        or receipt["files_sha256"] != prepared["files_sha256"]
        or receipt["helpers_sha256"] != prepared["helpers_sha256"]
        or receipt["validation_binding_sha256"] != prepared["validation_binding_sha256"]
        or receipt["preparation_sha256"] != sha256(candidate / PREPARATION)
        or receipt["command"] != cpu_command()
        or receipt["cwd"] != str(candidate.resolve())
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or any(key in receipt for key in ("error", "errors"))
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
    ):
        raise ValueError("Wave11 release CPU command, source, archive or outcome differs")
    require_archive(Path(receipt["archive"]), archive_inventory(prepared, receipt["preparation_sha256"]), alias=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "check", "cpu"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU verification requires --out")
    if args.phase == "prepare":
        result = prepare(args.root, args.candidate)
    elif args.phase == "check":
        result = require_preparation(args.root, args.candidate)
    else:
        result = cpu(args.root, args.candidate, args.out)
    print(json.dumps({"status": result["status"]}, allow_nan=False))


if __name__ == "__main__":
    main()
