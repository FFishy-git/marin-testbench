# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare actual full-model observations without altering the queued synthetic candidate."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from diagnose_upstream_outer_gpu import cpu_source
from launch_wave5 import PYTHON, read, sha256
from prepare_upstream_outer import replace_once
from watch_wave5 import write
from wave7_release import candidate_inventory

ARMS = ("muont", "outer_leftdiag_fused", "outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused")
EXTRA_FILES = {
    "experiment/upstream_full_model_checks.py",
    "experiment/tests/test_upstream_full_model_checks.py",
    "experiment/train.py",
    "experiment/gpu_checks.py",
    "experiment/cluster.json",
}


def cpu_test_commands(candidate: Path, helpers: Path) -> list[tuple[str, Path, list[str]]]:
    """Define the distinct candidate and actual-path observer evidence commands."""
    prefix = [
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
    ]
    return [
        ("candidate", candidate, [*prefix, "experiment/tests"]),
        ("observer", helpers.parent, [*prefix, "tracking/test_upstream_full_model_checks.py"]),
    ]


def prepare(receipt: Path, candidate: Path) -> dict:
    """Copy the CPU-verified prototype and wire its actual-path observers, keeping CUDA denied."""
    previous, cpu = cpu_source(receipt)
    if candidate.exists() or candidate.resolve().is_relative_to(previous.resolve()):
        raise FileExistsError("Full-model follow-up requires a new candidate; preserve existing evidence")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(Path(__file__).with_name("upstream_full_model_checks.py"), candidate / "experiment")
    shutil.copy2(
        Path(__file__).with_name("test_upstream_full_model_checks.py"),
        candidate / "experiment/tests",
    )
    train_path = candidate / "experiment/train.py"
    train = replace_once(
        train_path.read_text(),
        "from .outer_diagonal_checks import OuterDiagonalVerifier",
        "from .outer_diagonal_checks import OuterDiagonalVerifier\n"
        "from .upstream_full_model_checks import UpstreamProductVerifier",
    )
    train = replace_once(
        train,
        '        "outer_leftdiag_fused",\n        "outer_leftdiag_combined_fused",',
        '        "outer_leftdiag_fused",\n'
        '        "outer_leftdiag_upstream_fused",\n        "null_leftdiag_upstream_fused",',
    )
    train = replace_once(
        train,
        "OuterDiagonalVerifier(fast)",
        "UpstreamProductVerifier(fast) if fast.arm.upstream_product_kernels else OuterDiagonalVerifier(fast)",
    )
    train_path.write_text(train)
    gpu_path = candidate / "experiment/gpu_checks.py"
    gpu = replace_once(
        gpu_path.read_text(),
        "from .outer_diagonal_checks import compare_outer_update, fp64_diagonal_factors",
        "from .outer_diagonal_checks import compare_outer_update, fp64_diagonal_factors\n"
        "from .upstream_full_model_checks import test_upstream_product_updates, test_upstream_scalar_step",
    )
    gpu = replace_once(
        gpu,
        '            ("gpu_combined_outer_updates", lambda: test_combined_outer_updates(device)),',
        '            ("gpu_upstream_product_updates", lambda: test_upstream_product_updates(device)),\n'
        '            ("gpu_upstream_scalar_step", lambda: test_upstream_scalar_step(device)),',
    )
    gpu_path.write_text(gpu)
    cluster_path = candidate / "experiment/cluster.json"
    cluster = read(cluster_path)
    cluster["wave_name"] = "upstream-full-model-observers-unreleased"
    cluster["wave"] = [{"arm": arm, "seed": 0, "batch": "0p5m"} for arm in ARMS]
    write(cluster_path, cluster)
    files = candidate_inventory(candidate)
    changed = {name for name in files.keys() | cpu["files_sha256"] if files.get(name) != cpu["files_sha256"].get(name)}
    if changed != EXTRA_FILES:
        raise ValueError(f"Full-model observer changes differ: {sorted(changed ^ EXTRA_FILES)}")
    result = {
        "status": "full_model_observers_prepared_gpu_execution_denied",
        "candidate": str(candidate.resolve()),
        "prototype_cpu_receipt": str(receipt.resolve()),
        "prototype_cpu_receipt_sha256": sha256(receipt),
        "prototype_candidate": str(previous),
        "prototype_files_sha256": cpu["files_sha256"],
        "files_sha256": files,
        "changed_files": sorted(changed),
        "helper_sha256": sha256(Path(__file__)),
        "arms": ARMS,
        "policy": (
            "CPU-only actual-path observer preparation. Existing queued synthetic source unchanged. "
            "No GPU release, controller, queue, validation result or training authorization."
        ),
    }
    write(candidate / "full-model-preparation.json", result)
    return result


def cpu_evidence(receipt: Path, candidate: Path, output: Path) -> dict:
    """Bind fresh candidate and observer CPU results without issuing any execution release."""
    previous, original = cpu_source(receipt)
    initial_path = candidate / "full-model-preparation.json"
    initial = read(initial_path)
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | original["files_sha256"] if files.get(name) != original["files_sha256"].get(name)
    }
    helpers = Path(__file__).parent.resolve()
    helper_files = {
        name: sha256(helpers / name)
        for name in (
            "prepare_upstream_full_model.py",
            "upstream_full_model_checks.py",
            "test_upstream_full_model_checks.py",
        )
    }
    if (
        initial["candidate"] != str(candidate.resolve())
        or initial["prototype_cpu_receipt"] != str(receipt.resolve())
        or initial["prototype_cpu_receipt_sha256"] != sha256(receipt)
        or initial["prototype_candidate"] != str(previous)
        or initial["prototype_files_sha256"] != original["files_sha256"]
        or initial["files_sha256"] != files
        or changed != EXTRA_FILES
        or initial["helper_sha256"] != helper_files["prepare_upstream_full_model.py"]
        or files["experiment/upstream_full_model_checks.py"] != helper_files["upstream_full_model_checks.py"]
        or files["experiment/tests/test_upstream_full_model_checks.py"]
        != helper_files["test_upstream_full_model_checks.py"]
    ):
        raise ValueError("Full-model preparation, predecessor or actual-path observers changed")
    if output.exists() or output.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Fresh CPU evidence requires a new directory outside its candidate")
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(helpers)}
    results = []
    for label, cwd, command in cpu_test_commands(candidate, helpers):
        log = output / f"{label}.log"
        with log.open("x") as stream:
            completed = subprocess.run(
                command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=600
            )
        results.append(
            {
                "command": command,
                "cwd": str(cwd),
                "exit_code": completed.returncode,
                "log": str(log),
                "log_sha256": sha256(log),
            }
        )
        if completed.returncode:
            raise RuntimeError(f"Full-model observer CPU tests failed; inspect {log}")
    if files != candidate_inventory(candidate) or any(
        sha256(helpers / name) != digest for name, digest in helper_files.items()
    ):
        raise ValueError("Full-model observer sources changed during CPU verification")
    archive = output / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in files:
            bundle.add(candidate / name, arcname=f"candidate/{name}", recursive=False)
        bundle.add(
            candidate / "artifacts/external-sources", arcname="candidate/artifacts/external-sources", recursive=False
        )
        bundle.add(initial_path, arcname="candidate/full-model-preparation.json", recursive=False)
        for name in helper_files:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    record = {
        **initial,
        "status": "full_model_observer_cpu_verified_gpu_execution_denied",
        "initial_preparation": str(initial_path),
        "initial_preparation_sha256": sha256(initial_path),
        "helpers": str(helpers),
        "helpers_sha256": helper_files,
        "commands": results,
        "cuda_visible_devices": "",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "scope": (
            "CPU actual-step/scalar dispatch, independent output equation, observer behavior and denied ownership. "
            "No executed compiled GPU scalar transition, actual captured CUDA pipeline, all-rank restart, "
            "production release or training result."
        ),
    }
    write(output / "receipt.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "cpu"), default="prepare")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "cpu":
        if args.out is None:
            parser.error("--out is required for a fresh CPU receipt")
        result = cpu_evidence(args.cpu_receipt, args.candidate, args.out)
    else:
        result = prepare(args.cpu_receipt, args.candidate)
    print(json.dumps({"status": result["status"], "candidate": result["candidate"]}, indent=2))


if __name__ == "__main__":
    main()
