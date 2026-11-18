# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a distinct validation-only owner from an archived GPU-denied profiler."""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from launch_wave5 import PYTHON, read, sha256
from prepare_selected_lr_stage_profile import PREPARATION as PARENT_PREPARATION
from prepare_selected_lr_stage_profile import archive_inventory
from prepare_selected_lr_validation import require_archive
from selected_lr_stage_profile_contract import CPU_STATUS, PREPARATION, PREPARED_STATUS, locations
from selected_lr_stage_profile_validation import plan
from upstream_production import cpu_command
from watch_wave5 import write
from wave7_release import candidate_inventory

COPIES = {
    "selected_lr_stage_profile_execution_owner.py": "experiment/selected_lr_stage_profile_owner.py",
    "selected_lr_stage_profile_contract.py": "experiment/selected_lr_stage_profile_contract.py",
    "test_selected_lr_stage_profile_execution_owner.py": (
        "experiment/tests/test_selected_lr_stage_profile_execution_owner.py"
    ),
}
CHANGES = set(COPIES.values())
EXTERNAL_MODULES = sys.stdlib_module_names | {"pytest", "torch", "swanlab", "dotenv"}


def helper_names(directory: Path, seeds: tuple[str, ...]) -> tuple[str, ...]:
    """Bind flat local import dependencies; candidate-relative imports remain in candidate inventory."""
    pending, found = set(seeds), set()
    while pending:
        name = pending.pop()
        path = directory / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError(f"Stage-profile helper is absent or aliased: {name}")
        found.add(name)
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                modules = [node.module] if node.module else []
            else:
                continue
            for module in modules:
                head = module.split(".")[0]
                dependency = f"{head}.py"
                if (directory / dependency).is_file():
                    if module != head:
                        raise ValueError("Stage-profile flat helper imports a local package")
                    if dependency not in found:
                        pending.add(dependency)
                elif head not in EXTERNAL_MODULES and head != "experiment":
                    raise ValueError(f"Unbound stage-profile helper dependency: {name}: {module}")
    return tuple(sorted(found))


def parent_cpu(root: Path, receipt_path: Path) -> tuple[dict, dict]:
    """Use archived parent helpers in a fresh isolated process, then verify every returned byte."""
    receipt = read(receipt_path)
    candidate = Path(receipt["candidate"])
    prepared = read(candidate / PARENT_PREPARATION)
    hashes = receipt["helpers_sha256"]
    if hashes != prepared["helpers_sha256"]:
        raise ValueError("Stage-profile parent helper inventories differ")
    require_archive(Path(receipt["archive"]), archive_inventory(prepared, receipt["preparation_sha256"]), alias=True)
    with tempfile.TemporaryDirectory(prefix="stage-profile-parent-") as temporary:
        helpers = Path(temporary)
        with tarfile.open(receipt["archive"]) as bundle:
            for name in hashes:
                if Path(name).name != name:
                    raise ValueError("Stage-profile parent archive contains a nonlocal helper")
                with bundle.extractfile(f"tracking/{name}") as source:
                    (helpers / name).write_bytes(source.read())
        if any(sha256(helpers / name) != digest for name, digest in hashes.items()):
            raise ValueError("Stage-profile parent helper extraction differs from its verified archive")
        checked = subprocess.run(
            [
                PYTHON,
                "-P",
                "-c",
                "import json,sys;from pathlib import Path;from prepare_selected_lr_stage_profile import require_cpu;"
                "print(json.dumps(require_cpu(Path(sys.argv[1]),Path(sys.argv[2]))))",
                str(root),
                str(receipt_path),
            ],
            cwd=helpers,
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(helpers)},
            capture_output=True,
            text=True,
            check=True,
            timeout=1500,
        )
    if json.loads(checked.stdout) != receipt or sha256(candidate / PARENT_PREPARATION) != receipt["preparation_sha256"]:
        raise ValueError("Stage-profile parent CPU acceptance changed during reconstruction")
    if receipt["root"] != str(root):
        raise ValueError("Stage-profile parent CPU source is not independently accepted")
    return receipt, prepared


def contents() -> dict[str, bytes]:
    result = {destination: (Path(__file__).parent / source).read_bytes() for source, destination in COPIES.items()}
    return result


def prepare(root: Path, receipt_path: Path, candidate: Path) -> dict:
    root, receipt_path, candidate = root.resolve(), receipt_path.resolve(), candidate.resolve()
    receipt, prepared = parent_cpu(root, receipt_path)
    previous = Path(receipt["candidate"])
    if candidate.exists() or candidate.is_relative_to(previous):
        raise FileExistsError("Stage-profile validation requires an unused separate candidate")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, content in contents().items():
        (candidate / name).write_bytes(content)
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | receipt["files_sha256"].keys()
        if files.get(name) != receipt["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError("Stage-profile validation exceeded its owner/contract/test-only scope")
    helpers = helper_names(Path(__file__).parent, (Path(__file__).name, *COPIES))
    result = {
        "status": PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_cpu_receipt": str(receipt_path),
        "parent_cpu_receipt_sha256": sha256(receipt_path),
        "parent_preparation_sha256": sha256(previous / PARENT_PREPARATION),
        "parent_metadata_sha256": {path.name: sha256(path) for path in previous.glob("*.json")},
        "parent_release_sha256": prepared["parent_release_sha256"],
        "files_sha256": files,
        "changed_files": sorted(CHANGES),
        "data_receipt_sha256": prepared["data_receipt_sha256"],
        "helpers_sha256": {name: sha256(Path(__file__).parent / name) for name in helpers},
        "plan": plan(candidate, root, locations(root)[3]),
        "policy": "Validation-only preparation; no frozen controller, allocation transfer or GPU authorization.",
    }
    write(candidate / PREPARATION, result)
    require_preparation(root, candidate)
    return result


def require_preparation(root: Path, candidate: Path) -> dict:
    prepared = read(candidate / PREPARATION)
    receipt, parent = parent_cpu(root, Path(prepared["parent_cpu_receipt"]))
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | receipt["files_sha256"].keys()
        if files.get(name) != receipt["files_sha256"].get(name)
    }
    previous = Path(receipt["candidate"])
    names = helper_names(Path(__file__).parent, (Path(__file__).name, *COPIES))
    if (
        prepared["status"] != PREPARED_STATUS
        or prepared["root"] != str(root)
        or prepared["candidate"] != str(candidate.resolve())
        or prepared["parent_candidate"] != str(previous)
        or prepared["parent_cpu_receipt_sha256"] != sha256(Path(prepared["parent_cpu_receipt"]))
        or prepared["parent_preparation_sha256"] != sha256(previous / PARENT_PREPARATION)
        or prepared["parent_release_sha256"] != parent["parent_release_sha256"]
        or prepared["files_sha256"] != files
        or changed != CHANGES
        or prepared["changed_files"] != sorted(CHANGES)
        or prepared["data_receipt_sha256"] != parent["data_receipt_sha256"]
        or prepared["helpers_sha256"] != {name: sha256(Path(__file__).parent / name) for name in names}
        or prepared["parent_metadata_sha256"] != {path.name: sha256(path) for path in previous.glob("*.json")}
        or any(sha256(candidate / name) != digest for name, digest in prepared["parent_metadata_sha256"].items())
        or any((candidate / name).read_bytes() != content for name, content in contents().items())
        or prepared["plan"] != plan(candidate, root, locations(root)[3])
    ):
        raise ValueError("Stage-profile validation candidate or parent evidence changed")
    return prepared


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    prepared = require_preparation(root, candidate)
    if out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Stage-profile validation CPU output already exists or aliases its candidate")
    out.mkdir(parents=True)
    hashes = prepared["helpers_sha256"]
    helpers = out / "tracking"
    helpers.mkdir()
    for name in hashes:
        shutil.copy2(Path(__file__).parent / name, helpers / name)
    command, log, archive = cpu_command(), out / "candidate.log", out / "source.tar"
    digest = sha256(candidate / PREPARATION)
    error = None
    try:
        with log.open("x") as stream:
            code = subprocess.run(
                command,
                cwd=candidate,
                env=os.environ
                | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(candidate)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
    except subprocess.TimeoutExpired:
        code, error = 124, "Candidate CPU timeout"
    expected = (
        prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: digest}
        | {f"tracking/{name}": value for name, value in hashes.items()}
    )
    with tarfile.open(archive, "x") as bundle:
        for name in prepared["files_sha256"] | prepared["parent_metadata_sha256"] | {PREPARATION: digest}:
            bundle.add(candidate / name, arcname=name, recursive=False)
        for name in hashes:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    require_archive(archive, expected, alias=True)
    if require_preparation(root, candidate) != prepared:
        raise ValueError("Stage-profile validation candidate changed during CPU execution")
    result = {
        "status": CPU_STATUS if code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": str(candidate),
        "files_sha256": prepared["files_sha256"],
        "preparation_sha256": digest,
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "command": command,
        "cwd": str(candidate),
        "cuda_visible_devices": "",
        "exit_code": code,
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
    if error:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CPU_STATUS:
        raise RuntimeError(f"Stage-profile validation CPU failure preserved at {out}")
    return result


def require_cpu(root: Path, receipt_path: Path) -> dict:
    receipt = read(receipt_path)
    candidate = Path(receipt["candidate"])
    prepared = require_preparation(root, candidate)
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root)
        or receipt["files_sha256"] != prepared["files_sha256"]
        or receipt["helpers_sha256"] != prepared["helpers_sha256"]
        or receipt["preparation_sha256"] != sha256(candidate / PREPARATION)
        or receipt["command"] != cpu_command()
        or receipt["cwd"] != str(candidate)
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
        or "error" in receipt
        or any(sha256(Path(receipt["helpers"]) / name) != digest for name, digest in receipt["helpers_sha256"].items())
    ):
        raise ValueError("Stage-profile validation CPU command, archive or outcome differs")
    require_archive(
        Path(receipt["archive"]),
        prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: receipt["preparation_sha256"]}
        | {f"tracking/{name}": digest for name, digest in receipt["helpers_sha256"].items()},
        alias=True,
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "check", "cpu"), required=True)
    parser.add_argument("--parent-cpu-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare" and args.parent_cpu_receipt is None:
        parser.error("prepare requires --parent-cpu-receipt")
    if args.phase == "cpu" and args.out is None:
        parser.error("cpu requires --out")
    root, candidate = args.root.resolve(), args.candidate.resolve()
    result = (
        prepare(root, args.parent_cpu_receipt, candidate)
        if args.phase == "prepare"
        else cpu(root, candidate, args.out.resolve()) if args.phase == "cpu" else require_preparation(root, candidate)
    )
    print(json.dumps({"status": result["status"], "candidate": result["candidate"]}))


if __name__ == "__main__":
    main()
