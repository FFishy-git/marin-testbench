# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and CPU-archive a separate attribution owner, without GPU authorization."""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from fused_attribution_production_contract import (
    CPU_STATUS,
    FAMILIES,
    PREPARATION,
    PREPARED_STATUS,
    PYTHON,
    WAVES,
    config_name,
    require_config,
    wave_spec,
)
from fused_attribution_validation_contract import FROZEN_STATUS as VALIDATION_STATUS
from fused_attribution_validation_contract import PREPARATION as PARENT_PREPARATION
from fused_attribution_validation_contract import locations
from selected_lr_release_owner import candidate_files as candidate_inventory
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

STATUS = PREPARED_STATUS
COPIES = {
    "fused_attribution_production_owner.py": "experiment/fused_attribution_production_owner.py",
    "fused_attribution_production_contract.py": "experiment/fused_attribution_production_contract.py",
    "test_fused_attribution_production_owner.py": "experiment/tests/test_fused_attribution_production_owner.py",
}
CHANGES = {"experiment/train.py", *COPIES.values()}
CPU_TIMEOUT = 900
POLICY = (
    "Ownership-only CPU preparation grants no GPU authorization. Preserve the frozen validation "
    "candidate, all nine numerical/configuration identities and historical owners. Production "
    "requires a distinct release accepting all 27 commands/93 checks, original generations absent, "
    "exact keeper succession and generation-bound workers. No reconstruction, freeze or launch here."
)
TEST_IMPORT = (
    'package = ModuleType("_fused_attribution_production_cpu")\n'
    "package.__path__ = [str(Path(__file__).parent)]\n"
    "sys.modules[package.__name__] = package\n"
    'owner = importlib.import_module(f"{package.__name__}.fused_attribution_production_owner")'
)
EXTERNAL_MODULES = sys.stdlib_module_names | {"pytest"}


def cpu_command() -> list[str]:
    return [
        PYTHON,
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "-p",
        "no:cacheprovider",
        "-q",
        "-m",
        "not slow and not integration and not data_integration and not cluster "
        "and not requires_cluster and not docker and not manual",
        "experiment/tests",
    ]


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError("Attribution ownership insertion differs from its frozen reference")
    return text.replace(before, after, 1)


def helper_names(directory: Path, seeds: tuple[str, ...]) -> tuple[str, ...]:
    """Discover the complete flat import closure without importing tensor-dependent helpers."""
    pending, found = set(seeds), set()
    while pending:
        name = pending.pop()
        path = directory / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError(f"Attribution helper is missing or aliased: {name}")
        found.add(name)
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module else [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                head = module.split(".")[0]
                dependency = f"{head}.py"
                if (directory / dependency).is_file():
                    if module != head:
                        raise ValueError("Attribution flat helper imports an unexpected local package")
                    if dependency not in found:
                        pending.add(dependency)
                elif head not in EXTERNAL_MODULES and head != "experiment":
                    raise ValueError(f"Unbound attribution helper dependency: {name}: {module}")
    return tuple(sorted(found))


def require_archive(path: Path, expected: dict, *, alias: bool) -> None:
    """Verify exact regular members and the sole optional pinned-source alias."""
    with tarfile.open(path) as bundle:
        members = bundle.getmembers()
        names = set(expected) | ({"artifacts/external-sources"} if alias else set())
        if len(members) != len(names) or {member.name for member in members} != names:
            raise ValueError("Attribution archive inventory differs")
        if alias:
            member = bundle.getmember("artifacts/external-sources")
            if not member.issym() or member.linkname != "pinned-sources-materialized":
                raise ValueError("Attribution archive pinned-source alias differs")
        for name, digest in expected.items():
            member = bundle.getmember(name)
            if not member.isfile():
                raise ValueError("Attribution archive source is aliased")
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("Attribution archive bytes differ from their source identities")


def original(root: Path) -> dict:
    """Authenticate original bytes before running the frozen, CPU-only binding check."""
    root = root.resolve()
    binding_path, helpers, _, _, _ = locations(root)
    raw = binding_path.read_bytes()
    frozen = json.loads(raw)
    candidate = Path(frozen["candidate"])
    prepared = read(candidate / PARENT_PREPARATION)
    cpu = read(Path(frozen["cpu_receipt"]))
    controller = read(Path(frozen["controller_cpu_receipt"]))
    hashes = frozen["helpers_sha256"]
    helper = helpers / "fused_attribution_validation_controller.py"
    identities = {
        Path(frozen["archive"]): frozen["archive_sha256"],
        Path(frozen["cpu_receipt"]): frozen["cpu_receipt_sha256"],
        Path(frozen["controller_cpu_receipt"]): frozen["controller_cpu_receipt_sha256"],
        candidate / PARENT_PREPARATION: frozen["preparation_sha256"],
        root / "data/experiment_receipt.json": frozen["data_receipt_sha256"],
        **{helpers / name: digest for name, digest in hashes.items()},
    }
    if (
        frozen["status"] != VALIDATION_STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or candidate != candidate.resolve()
        or helpers.is_symlink()
        or helper.name not in hashes
        or any(Path(name).name != name for name in hashes)
        or {path.name for path in helpers.iterdir() if path.name != "__pycache__"} != set(hashes)
        or any(path.is_symlink() or not path.is_file() or sha256(path) != digest for path, digest in identities.items())
        or candidate_inventory(candidate) != frozen["files_sha256"]
        or any(sha256(candidate / name) != digest for name, digest in prepared["parent_metadata_sha256"].items())
        or cpu["status"] != "fused_attribution_validation_cpu_verified_gpu_denied"
        or controller["status"] != "fused_attribution_validation_controller_cpu_verified_no_gpu_binding"
        or any(
            receipt["root"] != str(root)
            or type(receipt["exit_code"]) is not int
            or receipt["exit_code"] != 0
            or receipt["cuda_visible_devices"] != ""
            or any(key in receipt for key in ("error", "errors"))
            or receipt["log_sha256"] != sha256(Path(receipt["log"]))
            or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
            for receipt in (cpu, controller)
        )
        or cpu["candidate"] != str(candidate)
        or cpu["files_sha256"] != frozen["files_sha256"]
        or cpu["preparation_sha256"] != frozen["preparation_sha256"]
        or cpu["helpers_sha256"] != prepared["helpers_sha256"]
        or controller["helpers_sha256"] != hashes
        or any(hashes.get(name) != digest for name, digest in cpu["helpers_sha256"].items())
    ):
        raise ValueError("Attribution production requires authentic original frozen source and CPU evidence")
    metadata = prepared["parent_metadata_sha256"] | {PARENT_PREPARATION: frozen["preparation_sha256"]}
    require_archive(
        Path(cpu["archive"]),
        frozen["files_sha256"]
        | metadata
        | {f"tracking/{name}": digest for name, digest in cpu["helpers_sha256"].items()},
        alias=True,
    )
    require_archive(
        Path(controller["archive"]),
        {f"tracking/{name}": digest for name, digest in hashes.items()} | controller["fixture_files_sha256"],
        alias=False,
    )
    require_archive(
        Path(frozen["archive"]),
        frozen["files_sha256"]
        | metadata
        | {
            "candidate-cpu-receipt.json": frozen["cpu_receipt_sha256"],
            "controller-cpu-receipt.json": frozen["controller_cpu_receipt_sha256"],
        }
        | {f"tracking/{name}": digest for name, digest in hashes.items()},
        alias=True,
    )
    checked = subprocess.run(
        [PYTHON, "-P", str(helper), "--root", str(root), "--phase", "check"],
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        check=True,
        timeout=1500,
    )
    if (
        json.loads(checked.stdout)["status"] != frozen["status"]
        or binding_path.read_bytes() != raw
        or candidate_inventory(candidate) != frozen["files_sha256"]
        or any(sha256(path) != digest for path, digest in identities.items())
    ):
        raise ValueError("Attribution frozen identity changed during its isolated binding check")
    return frozen


def helper_hashes() -> dict:
    directory = Path(__file__).parent
    names = helper_names(
        directory,
        (Path(__file__).name, *COPIES, "selected_lr_release_owner.py", "test_prepare_fused_attribution_production.py"),
    )
    return {name: sha256(directory / name) for name in names}


def transformed_files(previous: Path) -> dict[str, bytes]:
    """Replace only the existing early dispatch; retain the validation owner and trainer body."""
    train = replace_once(
        (previous / "experiment/train.py").read_text(),
        "from .fused_attribution_owner import require_fused_attribution_owner",
        "from .fused_attribution_production_owner import require_fused_attribution_production_owner",
    )
    train = replace_once(
        train, "    require_fused_attribution_owner(args)\n", "    require_fused_attribution_production_owner(args)\n"
    )
    payload = {destination: (Path(__file__).parent / source).read_bytes() for source, destination in COPIES.items()}
    test_path = COPIES["test_fused_attribution_production_owner.py"]
    test = replace_once(
        payload[test_path].decode(),
        TEST_IMPORT,
        'owner = importlib.import_module("experiment.fused_attribution_production_owner")',
    )
    payload[test_path] = replace_once(
        test, "from types import ModuleType, SimpleNamespace", "from types import SimpleNamespace"
    ).encode()
    return {"experiment/train.py": train.encode(), **payload}


def preparation_record(root: Path, candidate: Path, frozen: dict, files: dict) -> dict:
    previous = Path(frozen["candidate"])
    for wave in WAVES:
        for family in FAMILIES:
            name = config_name(family, wave_spec(wave)["lr_label"])
            if (candidate / name).read_bytes() != (previous / name).read_bytes():
                raise ValueError("Attribution production changed a validated full-budget configuration")
            require_config(read(candidate / name), wave, family)
    return {
        "status": PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "validation_binding": str(locations(root)[0]),
        "validation_binding_sha256": sha256(locations(root)[0]),
        "parent_cpu_receipt": frozen["cpu_receipt"],
        "parent_cpu_receipt_sha256": frozen["cpu_receipt_sha256"],
        "parent_controller_cpu_receipt": frozen["controller_cpu_receipt"],
        "parent_controller_cpu_receipt_sha256": frozen["controller_cpu_receipt_sha256"],
        "parent_archive": frozen["archive"],
        "parent_archive_sha256": frozen["archive_sha256"],
        "parent_preparation_sha256": frozen["preparation_sha256"],
        "parent_metadata_sha256": {path.name: sha256(path) for path in sorted(previous.glob("*.json"))},
        "data_receipt_sha256": frozen["data_receipt_sha256"],
        "files_sha256": files,
        "changed_files": sorted(CHANGES),
        "helpers_sha256": helper_hashes(),
        "waves": [wave_spec(wave) for wave in WAVES],
        "policy": POLICY,
    }


def prepare(root: Path, candidate: Path) -> dict:
    """Create an unused ownership-only successor; never accept or release GPUs."""
    root, candidate = root.resolve(), candidate.resolve()
    frozen = original(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or any(
        candidate.is_relative_to(path) or path.is_relative_to(candidate) for path in (previous, Path(frozen["helpers"]))
    ):
        raise FileExistsError("Attribution production needs an unused separate candidate")
    if any(name in frozen["files_sha256"] for name in COPIES.values()) or (previous / PREPARATION).exists():
        raise ValueError("Original attribution validation source already contains a production successor")
    payload, hashes = transformed_files(previous), helper_hashes()
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, content in payload.items():
        (candidate / name).parent.mkdir(parents=True, exist_ok=True)
        (candidate / name).write_bytes(content)
    result = preparation_record(root, candidate, frozen, candidate_inventory(candidate))
    if result["helpers_sha256"] != hashes:
        raise ValueError("Attribution production helper source changed during preparation")
    write(candidate / PREPARATION, result)
    return require_preparation(root, candidate)


def require_preparation(root: Path, candidate: Path) -> dict:
    """Rederive allowed bytes, inventories and parent metadata rather than trusting repaired hashes."""
    root, candidate = root.resolve(), candidate.resolve()
    frozen = original(root)
    previous = Path(frozen["candidate"])
    prepared, files = read(candidate / PREPARATION), candidate_inventory(candidate)
    expected = preparation_record(root, candidate, frozen, files)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    if (
        prepared.keys() != expected.keys() | {"time"}
        or any(prepared[key] != value for key, value in expected.items())
        or changed != CHANGES
        or any((candidate / name).read_bytes() != content for name, content in transformed_files(previous).items())
        or any(sha256(candidate / name) != digest for name, digest in expected["parent_metadata_sha256"].items())
    ):
        raise ValueError("Attribution production preparation or exact four-file ownership scope differs")
    return prepared


def archive_inventory(prepared: dict, digest: str) -> dict:
    return (
        prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: digest}
        | {f"tracking/{name}": value for name, value in prepared["helpers_sha256"].items()}
    )


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    """Preserve hidden-CUDA CPU execution and exact archives, including failure evidence."""
    root, candidate, out = root.resolve(), candidate.resolve(), out.resolve()
    prepared = require_preparation(root, candidate)
    directory = Path(__file__).parent.resolve()
    if out.exists() or any(
        out.is_relative_to(path) or path.is_relative_to(out)
        for path in (candidate, Path(prepared["parent_candidate"]), directory)
    ):
        raise FileExistsError("Attribution production CPU evidence needs an unused external directory")
    digest, hashes = sha256(candidate / PREPARATION), prepared["helpers_sha256"]
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, value in hashes.items():
        shutil.copy2(directory / name, helpers / name)
        if sha256(helpers / name) != value:
            raise ValueError("Attribution production helper changed before CPU execution")
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
                timeout=CPU_TIMEOUT,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, f"Attribution production CPU timeout at {CPU_TIMEOUT} seconds"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in prepared["files_sha256"] | prepared["parent_metadata_sha256"] | {PREPARATION: digest}:
                bundle.add(candidate / name, arcname=name, recursive=False)
            for name in hashes:
                bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
        require_archive(archive, archive_inventory(prepared, digest), alias=True)
        if require_preparation(root, candidate) != prepared or sha256(candidate / PREPARATION) != digest:
            raise ValueError("Attribution production source changed during CPU execution")
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": CPU_STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": str(candidate),
        "files_sha256": prepared["files_sha256"],
        "preparation_sha256": digest,
        "validation_binding_sha256": prepared["validation_binding_sha256"],
        "helpers": str(helpers),
        "helpers_sha256": hashes,
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
        raise RuntimeError(f"Attribution production CPU/archive failure preserved at {out}")
    try:
        return require_cpu(root, out / "receipt.json")
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as failure:
        result.update(status="needs_attention", error=str(failure))
        write(out / "receipt.json", result)
        raise RuntimeError(f"Attribution production final CPU verification failure preserved at {out}") from failure


def require_cpu(root: Path, receipt_path: Path) -> dict:
    """Verify source, command, complete helper closure and exact archived CPU evidence."""
    root, receipt_path = root.resolve(), receipt_path.resolve()
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw)
    candidate = Path(receipt["candidate"])
    prepared = require_preparation(root, candidate)
    helpers = receipt_path.parent / "tracking"
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root)
        or candidate != candidate.resolve()
        or receipt["files_sha256"] != prepared["files_sha256"]
        or receipt["helpers_sha256"] != prepared["helpers_sha256"]
        or receipt["preparation_sha256"] != sha256(candidate / PREPARATION)
        or receipt["validation_binding_sha256"] != prepared["validation_binding_sha256"]
        or receipt["command"] != cpu_command()
        or receipt["cwd"] != str(candidate)
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or any(key in receipt for key in ("error", "errors"))
        or receipt["helpers"] != str(helpers)
        or helpers.is_symlink()
        or {path.name for path in helpers.iterdir() if path.name != "__pycache__"} != set(prepared["helpers_sha256"])
        or any(
            (helpers / name).is_symlink() or sha256(helpers / name) != digest
            for name, digest in prepared["helpers_sha256"].items()
        )
        or receipt["log"] != str(receipt_path.parent / "candidate.log")
        or receipt["archive"] != str(receipt_path.parent / "source.tar")
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
    ):
        raise ValueError("Attribution production CPU source, command, archive or outcome differs")
    require_archive(Path(receipt["archive"]), archive_inventory(prepared, receipt["preparation_sha256"]), alias=True)
    if receipt_path.read_bytes() != raw:
        raise ValueError("Attribution production CPU receipt changed during verification")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "check", "cpu"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "cpu" and args.out is None:
        parser.error("cpu requires --out")
    if args.phase == "prepare":
        result = prepare(args.root, args.candidate)
    elif args.phase == "check":
        result = require_preparation(args.root, args.candidate)
    else:
        result = cpu(args.root, args.candidate, args.out)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
