# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare/check and CPU-verify a separate bounded-validation successor; no freeze or launch."""

import argparse
import copy
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from fused_attribution_validation import failed_profiler_evidence, plan
from fused_attribution_validation_contract import (
    EXPANSION_POLICY,
    PREPARATION,
    PREPARED_STATUS,
    PYTHON,
    locations,
    trial_specs,
)
from launch_wave5 import read, sha256
from prepare_fused_attribution import PREPARATION as PARENT_PREPARATION
from prepare_fused_attribution import archive_inventory as parent_archive_inventory
from prepare_selected_lr_stage_profile_validation import helper_names
from prepare_selected_lr_validation import require_archive
from prepare_upstream_outer import replace_once
from upstream_production import cpu_command
from watch_wave5 import write
from wave7_release import candidate_inventory

CPU_STATUS = "fused_attribution_validation_cpu_verified_gpu_denied"
COPIES = {
    "fused_attribution_validation_contract.py": "experiment/fused_attribution_validation_contract.py",
    "fused_attribution_validation_owner.py": "experiment/fused_attribution_owner.py",
    "fused_attribution_validation_checks.py": "experiment/fused_attribution_validation_checks.py",
    "test_fused_attribution_validation_owner.py": "experiment/tests/test_fused_attribution_validation_owner.py",
    "test_fused_attribution_validation_checks.py": "experiment/tests/test_fused_attribution_validation_checks.py",
}
CHANGES = {"experiment/train.py", *COPIES.values()}
TESTS = ("test_fused_attribution_validation.py", "test_prepare_fused_attribution_validation.py")


def parent_cpu(root: Path, receipt_path: Path) -> tuple[dict, dict]:
    """Reconstruct acceptance using the parent's own archived helper closure in isolation."""
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    candidate = Path(receipt["candidate"])
    prepared = read(candidate / PARENT_PREPARATION)
    hashes = prepared["helpers_sha256"]
    if receipt["helpers_sha256"] != hashes:
        raise ValueError("Attribution parent helper identities differ")
    require_archive(
        Path(receipt["archive"]), parent_archive_inventory(prepared, receipt["preparation_sha256"]), alias=True
    )
    with tempfile.TemporaryDirectory(prefix="attribution-validation-parent-") as temporary:
        helpers = Path(temporary)
        with tarfile.open(receipt["archive"]) as bundle:
            for name, digest in hashes.items():
                if Path(name).name != name:
                    raise ValueError("Attribution parent helper basename is not local")
                with bundle.extractfile(f"tracking/{name}") as stream:
                    raw = stream.read()
                (helpers / name).write_bytes(raw)
                if sha256(helpers / name) != digest:
                    raise ValueError("Attribution parent archive helper bytes differ")
        checked = subprocess.run(
            [
                PYTHON,
                "-P",
                "-c",
                "import json,sys;from pathlib import Path;from prepare_fused_attribution import require_cpu;"
                "print(json.dumps(require_cpu(Path(sys.argv[1]),Path(sys.argv[2])),allow_nan=False))",
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
    if (
        json.loads(checked.stdout) != receipt
        or receipt_path.read_bytes() != receipt_raw
        or sha256(candidate / PARENT_PREPARATION) != receipt["preparation_sha256"]
        or candidate_inventory(candidate) != receipt["files_sha256"]
        or receipt["root"] != str(root)
    ):
        raise ValueError("Attribution parent CPU identity changed during independent reconstruction")
    return receipt, prepared


def transformed_train(previous: Path) -> bytes:
    text = (previous / "experiment/train.py").read_text()
    replacements = (
        ("from pathlib import Path\n", "from contextlib import nullcontext\nfrom pathlib import Path\n"),
        (
            "from .outer_diagonal_checks import OuterDiagonalVerifier",
            "from .fused_attribution_validation_checks import FusedAttributionVerifier",
        ),
        ("    require_fused_attribution_owner()", "    require_fused_attribution_owner(args)"),
        (
            '    if args.verify_outer_diagonal_updates and config["experiment"]["arm"] != "outer_leftdiag_fused":\n'
            '        raise ValueError("Outer update verification covers only outer_leftdiag_fused")',
            '    if args.verify_outer_diagonal_updates and config["experiment"]["arm"] not in (\n'
            '        "outer_leftdiag_fused", "null_leftdiag_fused"\n'
            '    ):\n        raise ValueError("Attribution update verification requires an exact fused/null family")',
        ),
        (
            "        outer_verifier = OuterDiagonalVerifier(fast)",
            "        outer_verifier = FusedAttributionVerifier(fast)",
        ),
        (
            "            for optimizer in optimizers:\n                optimizer.step()\n",
            "            with outer_verifier.observe() if outer_verifier is not None else nullcontext():\n"
            "                for optimizer in optimizers:\n                    optimizer.step()\n",
        ),
    )
    for before, after in replacements:
        text = replace_once(text, before, after)
    return text.encode()


def contents(previous: Path) -> dict[str, bytes]:
    directory = Path(__file__).parent
    return {"experiment/train.py": transformed_train(previous)} | {
        destination: (directory / source).read_bytes() for source, destination in COPIES.items()
    }


def helper_hashes() -> dict:
    directory = Path(__file__).parent
    names = helper_names(directory, (Path(__file__).name, *COPIES, *TESTS))
    return {name: sha256(directory / name) for name in names}


def preparation_record(root: Path, receipt_path: Path, candidate: Path, receipt: dict, parent: dict) -> dict:
    previous = Path(receipt["candidate"])
    for spec, trial in zip(trial_specs(), parent["trials"], strict=True):
        if (
            any(trial[key] != spec[key] for key in ("name", "family", "lr_label", "matrix_lr"))
            or trial["config"] != str(previous / spec["config_name"])
            or (candidate / spec["config_name"]).read_bytes() != (previous / spec["config_name"]).read_bytes()
        ):
            raise ValueError("Validation successor changed the parent's exact nine full-schedule configs")
    return {
        "status": PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_cpu_receipt": str(receipt_path),
        "parent_cpu_receipt_sha256": sha256(receipt_path),
        "parent_preparation_sha256": sha256(previous / PARENT_PREPARATION),
        "parent_metadata_sha256": {path.name: sha256(path) for path in previous.glob("*.json")},
        "scientific_parent_release_sha256": parent["parent_release_sha256"],
        "data_receipt_sha256": parent["data_receipt_sha256"],
        "files_sha256": candidate_inventory(candidate),
        "changed_files": sorted(CHANGES),
        "helpers_sha256": helper_hashes(),
        "allocation_predecessor": failed_profiler_evidence(root),
        "plan": plan(candidate, root, locations(root)[3]),
        "expansion_policy": copy.deepcopy(EXPANSION_POLICY),
        "policy": (
            "Validation-only successor; parent/config/optimizer bytes preserved. No CPU acceptance, freeze or launch."
        ),
    }


def prepare(root: Path, receipt_path: Path, candidate: Path) -> dict:
    root, receipt_path, candidate = root.resolve(), receipt_path.resolve(), candidate.resolve()
    receipt, parent = parent_cpu(root, receipt_path)
    previous = Path(receipt["candidate"])
    if candidate.exists() or candidate.is_relative_to(previous) or previous.is_relative_to(candidate):
        raise FileExistsError("Attribution validation requires an unused separate successor candidate")
    payload, predecessor = contents(previous), failed_profiler_evidence(root)
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, raw in payload.items():
        (candidate / name).parent.mkdir(parents=True, exist_ok=True)
        (candidate / name).write_bytes(raw)
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | receipt["files_sha256"].keys()
        if files.get(name) != receipt["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError("Attribution validation exceeded checks/owner/trainer/tests-only scope")
    result = preparation_record(root, receipt_path, candidate, receipt, parent)
    if result["allocation_predecessor"] != predecessor:
        raise ValueError("Attribution predecessor changed during successor preparation")
    write(candidate / PREPARATION, result)
    require_preparation(root, candidate)
    return result


def require_preparation(root: Path, candidate: Path) -> dict:
    root, candidate = root.resolve(), candidate.resolve()
    prepared = read(candidate / PREPARATION)
    receipt_path = Path(prepared["parent_cpu_receipt"])
    receipt, parent = parent_cpu(root, receipt_path)
    expected = preparation_record(root, receipt_path, candidate, receipt, parent)
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | receipt["files_sha256"].keys()
        if files.get(name) != receipt["files_sha256"].get(name)
    }
    if (
        prepared.keys() != expected.keys() | {"time"}
        or any(prepared[key] != value for key, value in expected.items())
        or changed != CHANGES
        or any((candidate / name).read_bytes() != raw for name, raw in contents(Path(receipt["candidate"])).items())
        or any(sha256(candidate / name) != digest for name, digest in expected["parent_metadata_sha256"].items())
    ):
        raise ValueError("Attribution validation source, parent, keeper evidence or exact transformation differs")
    return prepared


def archive_inventory(prepared: dict, digest: str) -> dict:
    """Bind all candidate bytes, preserved parent metadata and the complete local helper closure."""
    return (
        prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: digest}
        | {f"tracking/{name}": value for name, value in prepared["helpers_sha256"].items()}
    )


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    """Run candidate CPU tests with CUDA hidden and preserve exact source evidence, including failures."""
    root, candidate, out = root.resolve(), candidate.resolve(), out.resolve()
    prepared = require_preparation(root, candidate)
    directory = Path(__file__).parent.resolve()
    if out.exists() or any(
        out.is_relative_to(path) for path in (candidate, Path(prepared["parent_candidate"]), directory)
    ):
        raise FileExistsError("Attribution validation CPU evidence requires an unused external directory")
    digest, hashes = sha256(candidate / PREPARATION), prepared["helpers_sha256"]
    out.mkdir(parents=True, exist_ok=False)
    helpers = out / "tracking"
    helpers.mkdir()
    for name, value in hashes.items():
        shutil.copy2(directory / name, helpers / name)
        if sha256(helpers / name) != value:
            raise ValueError("Attribution validation helper changed before candidate CPU execution")
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
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Attribution validation candidate CPU timeout at 900 seconds"
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
            raise ValueError("Attribution validation source or predecessor evidence changed during CPU execution")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": CPU_STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
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
        "archive_sha256": sha256(archive) if archive.is_file() else None,
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CPU_STATUS:
        raise RuntimeError(f"Attribution validation CPU/archive failure preserved at {out}")
    try:
        return require_cpu(root, out / "receipt.json")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as failure:
        result.update(status="needs_attention", error=str(failure))
        write(out / "receipt.json", result)
        raise RuntimeError(f"Attribution validation final CPU/archive verification failed at {out}") from failure


def require_cpu(root: Path, receipt_path: Path) -> dict:
    """Reconstruct CPU acceptance from exact current helpers, source, logs and archived bytes."""
    root, receipt_path = root.resolve(), receipt_path.resolve()
    receipt = read(receipt_path)
    candidate = Path(receipt["candidate"])
    prepared = require_preparation(root, candidate)
    helpers = Path(receipt["helpers"])
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root)
        or receipt["candidate"] != str(candidate.resolve())
        or receipt["files_sha256"] != prepared["files_sha256"]
        or receipt["helpers_sha256"] != prepared["helpers_sha256"]
        or receipt["preparation_sha256"] != sha256(candidate / PREPARATION)
        or receipt["command"] != cpu_command()
        or receipt["cwd"] != str(candidate)
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
        or helpers.is_symlink()
        or not helpers.is_dir()
        or {path.name for path in helpers.iterdir()} != set(prepared["helpers_sha256"])
        or any(
            (helpers / name).is_symlink() or not (helpers / name).is_file() or sha256(helpers / name) != value
            for name, value in prepared["helpers_sha256"].items()
        )
    ):
        raise ValueError("Attribution validation CPU command, helpers, archive or outcome differs")
    require_archive(Path(receipt["archive"]), archive_inventory(prepared, receipt["preparation_sha256"]), alias=True)
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
    result = (
        prepare(args.root, args.parent_cpu_receipt, args.candidate)
        if args.phase == "prepare"
        else (
            cpu(args.root, args.candidate, args.out)
            if args.phase == "cpu"
            else require_preparation(args.root, args.candidate)
        )
    )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
