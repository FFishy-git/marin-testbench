# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare an ownership-only Wave11 successor without production authorization."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from launch_wave5 import read, sha256
from prepare_selected_lr_validation import require_archive
from prepare_upstream_outer import replace_once
from selected_lr_validation import PREPARATION as PARENT_PREPARATION
from selected_lr_validation import PYTHON, locations
from selected_lr_validation import STATUS as PARENT_STATUS
from upstream_production import cpu_command
from watch_wave5 import write
from wave7_release import candidate_inventory

WAVE = "wave11"
PREPARATION = "selected-lr-production-preparation.json"
STATUS = "selected_lr_production_prepared_gpu_denied"
CPU_STATUS = "selected_lr_production_cpu_verified_gpu_denied"
CHANGES = {
    "experiment/train.py",
    "experiment/selected_lr_production_owner.py",
    "experiment/cluster.json",
    "experiment/tests/test_selected_lr_production_owner.py",
}
COPIES = {
    "selected_lr_production_owner.py": "experiment/selected_lr_production_owner.py",
    "test_selected_lr_production_owner.py": "experiment/tests/test_selected_lr_production_owner.py",
}
HELPERS = (Path(__file__).name, *COPIES)
POLICY = (
    "Preparation and CPU evidence grant no GPU execution or launch trust. Preserve the original "
    "validation candidate, numerical/configuration bytes, owners and frozen evidence. A separate "
    "generation-bound production release must reconstruct the original twelve commands/thirty flags, "
    "all ranks and exact restoration before four-node worker ownership, cleanup, history and upload "
    "support can authorize production. No freeze, launch, queue or acceptance API exists here."
)


def original(root: Path) -> dict:
    """Use the original frozen check, never mutable validation helpers or GPU reconstruction."""
    root = root.resolve()
    binding_path, helpers, _, _, _ = locations(root)
    digest = sha256(binding_path)
    frozen = read(binding_path)
    helper = helpers / "run_selected_lr_validation.py"
    if (
        frozen["status"] != PARENT_STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or frozen["helpers_sha256"][helper.name] != sha256(helper)
    ):
        raise ValueError("Wave11 preparation requires the original frozen validation binding")
    checked = subprocess.run(
        [PYTHON, "-P", str(helper), "--root", str(root), "--phase", "check"],
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    previous = Path(frozen["candidate"])
    identities = [
        (Path(frozen["archive"]), frozen["archive_sha256"]),
        (Path(frozen["cpu_receipt"]), frozen["cpu_receipt_sha256"]),
        (Path(frozen["controller_cpu_receipt"]), frozen["controller_cpu_receipt_sha256"]),
        (previous / PARENT_PREPARATION, frozen["preparation_sha256"]),
        (root / "data/experiment_receipt.json", frozen["data_receipt_sha256"]),
        (root / "run-preparation/wave10-launch-binding.json", frozen["predecessor_release_sha256"]),
        *((helpers / name, expected) for name, expected in frozen["helpers_sha256"].items()),
    ]
    if (
        json.loads(checked.stdout)["status"] != frozen["status"]
        or sha256(binding_path) != digest
        or frozen["candidate"] != str(previous.resolve())
        or candidate_inventory(previous) != frozen["files_sha256"]
        or any(sha256(path) != expected for path, expected in identities)
    ):
        raise ValueError("Original Wave11 source or frozen CPU evidence changed during preparation")
    return frozen


def helper_hashes() -> dict:
    return {name: sha256(Path(__file__).parent / name) for name in HELPERS}


def transformed_files(previous: Path) -> dict[str, bytes]:
    """Limit trainer changes to its existing pre-config ownership import and call."""
    train = replace_once(
        (previous / "experiment/train.py").read_text(),
        "from .selected_lr_owner import require_selected_lr_training",
        "from .selected_lr_production_owner import require_selected_lr_production_owner",
    )
    train = replace_once(
        train,
        "    require_selected_lr_training(args)\n",
        "    require_selected_lr_production_owner(args)\n",
    )
    cluster = read(previous / "experiment/cluster.json")
    cluster["wave_name"] = WAVE
    contents = {destination: (Path(__file__).parent / source).read_bytes() for source, destination in COPIES.items()}
    test_path = COPIES["test_selected_lr_production_owner.py"]
    contents[test_path] = replace_once(
        contents[test_path].decode(),
        "from selected_lr_production_owner import require_selected_lr_production_owner",
        "from experiment.selected_lr_production_owner import require_selected_lr_production_owner",
    ).encode()
    return {
        "experiment/train.py": train.encode(),
        "experiment/cluster.json": (json.dumps(cluster, indent=2) + "\n").encode(),
        **contents,
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
    """Copy a distinct candidate, retaining all issued bytes except four exact ownership files."""
    root, candidate = root.resolve(), candidate.resolve()
    frozen = original(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.is_relative_to(previous) or candidate.is_relative_to(Path(frozen["helpers"])):
        raise FileExistsError("Wave11 production preparation requires a distinct unused candidate")
    if any(destination in frozen["files_sha256"] for destination in COPIES.values()):
        raise ValueError("Original validation candidate already contains a production successor owner")
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
        raise ValueError("Wave11 production preparation exceeds its exact four-file ownership scope")
    result = preparation_record(root, candidate, frozen, files)
    if result["helpers_sha256"] != hashes:
        raise ValueError("Wave11 preparation helpers changed during the ownership-only copy")
    write(candidate / PREPARATION, result)
    require_preparation(root, candidate)
    return result


def require_preparation(root: Path, candidate: Path) -> dict:
    """Rederive exact transformations, including allowed files, rather than trusting their digest."""
    root, candidate = root.resolve(), candidate.resolve()
    frozen = original(root)
    previous = Path(frozen["candidate"])
    record, files = read(candidate / PREPARATION), candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    expected = preparation_record(root, candidate, frozen, files)
    if (
        any(record[key] != value for key, value in expected.items())
        or changed != CHANGES
        or sha256(candidate / PARENT_PREPARATION) != frozen["preparation_sha256"]
        or any((candidate / name).read_bytes() != content for name, content in transformed_files(previous).items())
    ):
        raise ValueError("Wave11 production preparation, parent identities or exact ownership scope differs")
    return record


def archive_inventory(prepared: dict, preparation_digest: str) -> dict:
    return (
        prepared["files_sha256"]
        | {PREPARATION: preparation_digest}
        | {f"tracking/{name}": digest for name, digest in prepared["helpers_sha256"].items()}
    )


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    """Archive a hidden-CUDA CPU outcome; even success leaves production unconditionally denied."""
    root, candidate, out = root.resolve(), candidate.resolve(), out.resolve()
    prepared = require_preparation(root, candidate)
    if out.exists() or any(out.is_relative_to(path) for path in (candidate, Path(prepared["parent_candidate"]))):
        raise FileExistsError("Wave11 CPU evidence requires an unused directory outside both candidates")
    preparation_digest = sha256(candidate / PREPARATION)
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
        require_archive(archive, archive_inventory(prepared, preparation_digest), alias=True)
        if require_preparation(root, candidate) != prepared or sha256(candidate / PREPARATION) != preparation_digest:
            raise ValueError("Wave11 preparation changed during CPU verification")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": (
            CPU_STATUS if type(code) is int and code == 0 and error is None else "selected_lr_production_cpu_failed"
        ),
        "root": str(root),
        "candidate": str(candidate),
        "preparation_sha256": preparation_digest,
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
        raise RuntimeError(f"Wave11 production CPU failure preserved at {out}")
    return result


def require_cpu(root: Path, receipt_path: Path) -> dict:
    """Verify CPU command, source, log and archive identities without granting a release."""
    record = read(receipt_path)
    candidate = Path(record["candidate"])
    prepared = require_preparation(root, candidate)
    if (
        record["status"] != CPU_STATUS
        or record["root"] != str(root.resolve())
        or record["preparation_sha256"] != sha256(candidate / PREPARATION)
        or record["files_sha256"] != prepared["files_sha256"]
        or record["helpers_sha256"] != prepared["helpers_sha256"]
        or record["validation_binding_sha256"] != prepared["validation_binding_sha256"]
        or record["command"] != cpu_command()
        or record["cwd"] != str(candidate.resolve())
        or record["cuda_visible_devices"] != ""
        or type(record["exit_code"]) is not int
        or record["exit_code"] != 0
        or "error" in record
        or record["log_sha256"] != sha256(Path(record["log"]))
        or record["archive_sha256"] != sha256(Path(record["archive"]))
    ):
        raise ValueError("Wave11 CPU command, source, archive or outcome differs")
    require_archive(Path(record["archive"]), archive_inventory(prepared, record["preparation_sha256"]), alias=True)
    return record


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
