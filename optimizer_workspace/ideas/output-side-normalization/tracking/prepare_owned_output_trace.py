# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare full-shaped trace reduction while preserving the failed owned validator."""

import argparse
import ast
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

PREVIOUS = "run-preparation/owned-output-handoff-binding.json"
PREPARATION = "owned-output-trace-preparation.json"
STATUS = "owned_output_trace_prepared_gpu_denied"
CPU_STATUS = "owned_output_trace_cpu_verified_gpu_denied"
HELPERS = (
    "prepare_owned_output_trace.py",
    "test_prepare_owned_output_trace.py",
    "owned_output_handoff.py",
    "test_owned_output_handoff.py",
    "owned_output_trace_gpu_checks.py",
    "owned_output_trace_factor_checks.py",
    "owned_output_trace_handoff.py",
)
CHANGES = {
    "experiment/execution_owner.py",
    "experiment/tests/test_owned_output_execution_owner.py",
    "experiment/cluster.json",
    "experiment/block_toner.py",
    "experiment/owned_output_gpu_checks.py",
    "experiment/tests/test_owned_output_trace_factors.py",
}
REPLACEMENTS = (
    ("run_owned_output_handoff.py", "run_owned_output_trace.py"),
    ("owned-output-handoff-binding.json", "owned-output-trace-binding.json"),
    ("owned-output-handoff-state.json", "owned-output-trace-state.json"),
    ("owned-output-handoff-validation-20261117", "owned-output-trace-validation-20261117"),
    ("owned_output_handoff", "owned_output_trace"),
    ("validation-owned-output-handoff-gpu", "validation-owned-output-trace-gpu"),
)
OLD_TRACE = "            trace = moment.sum(-1, keepdim=True)"
NEW_TRACE = (
    "            # Retain the full batch shape used by the original GPU trace reduction.\n"
    "            trace = self.moment[width].sum(-1, keepdim=True)[selection]"
)


def require_original(root: Path) -> dict:
    """Require immutable failed source and terminal evidence before preparing a successor."""
    frozen = read(root / PREVIOUS)
    helpers = Path(frozen["helpers"])
    state = read(root / "run-preparation/owned-output-handoff-state.json")
    evidence = read(root / "run-preparation/owned-output-handoff-failure-evidence-20261117T1533.json")
    if (
        frozen["root"] != str(root)
        or frozen["helpers"] != str(root / "controllers/owned-output-handoff-validation-20261117/tracking")
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or frozen["files_sha256"] != candidate_inventory(Path(frozen["candidate"]))
        or state["binding_sha256"] != sha256(root / PREVIOUS)
        or state["status"] != "needs_attention"
        or type(state.get("exit_code")) is not int
        or state["exit_code"] != 1
        or type(state.get("cleanup_exit_code")) is not int
        or state["cleanup_exit_code"] != 1
        or type(state.get("keeper_restore")) is not int
        or state["keeper_restore"] != 0
        or "cleanup_error" in state
        or evidence["status"] != "owned_handoff_gpu_failure_source_and_terminal_evidence_bound"
        or evidence["exit_codes"] != [0, 1]
        or not evidence["files_sha256"]
        or any(sha256(root / name) != digest for name, digest in evidence["files_sha256"].items())
    ):
        raise ValueError("Failed owned-output predecessor source or terminal evidence changed")
    subprocess.run(
        [PYTHON, str(helpers / "run_owned_output_handoff.py"), "--root", str(root), "--phase", "check"],
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
            raise ValueError(f"Trace successor ownership source identity is missing: {old}")
        text = text.replace(old, new)
    return text


def trace_block(text: str) -> str:
    """Change only the reduction shape inside the existing owned-factor method."""
    tree = ast.parse(text)
    owned = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OwnedOutputMoments")
    method = next(node for node in owned.body if isinstance(node, ast.FunctionDef) and node.name == "factors")
    lines = text.splitlines(keepends=True)
    selected = "".join(lines[method.lineno - 1 : method.end_lineno])
    if selected.count(OLD_TRACE) != 1:
        raise ValueError("Owned factor trace replacement is not unique")
    revised = selected.replace(OLD_TRACE, NEW_TRACE)
    return "".join(lines[: method.lineno - 1]) + revised + "".join(lines[method.end_lineno :])


def expected_cluster(previous: Path) -> dict:
    cluster = read(previous / "experiment/cluster.json")
    cluster["wave_name"] = "owned-output-trace-validation-unreleased"
    return cluster


def prepare(root: Path, candidate: Path) -> dict:
    root = root.resolve()
    frozen = require_original(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.resolve().is_relative_to(previous.resolve()):
        raise FileExistsError("Trace successor requires a distinct unused candidate")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("experiment/execution_owner.py", "experiment/tests/test_owned_output_execution_owner.py"):
        (candidate / name).write_text(transformed((previous / name).read_text()))
    (candidate / "experiment/block_toner.py").write_text(
        trace_block((previous / "experiment/block_toner.py").read_text())
    )
    helpers = Path(__file__).parent
    shutil.copy2(helpers / "owned_output_trace_gpu_checks.py", candidate / "experiment/owned_output_gpu_checks.py")
    shutil.copy2(
        helpers / "owned_output_trace_factor_checks.py",
        candidate / "experiment/tests/test_owned_output_trace_factors.py",
    )
    write(candidate / "experiment/cluster.json", expected_cluster(previous))
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | frozen["files_sha256"] if files.get(name) != frozen["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError("Trace successor exceeds its numerical/diagnostic/ownership scope")
    result = {
        "status": STATUS,
        "root": str(root),
        "candidate": str(candidate.resolve()),
        "original_binding_sha256": sha256(root / PREVIOUS),
        "original_candidate": str(previous),
        "failure_evidence_sha256": sha256(
            root / "run-preparation/owned-output-handoff-failure-evidence-20261117T1533.json"
        ),
        "files_sha256": files,
        "changed_files": sorted(CHANGES),
        "preparation_helper_sha256": sha256(Path(__file__)),
        "diagnostic_helper_sha256": {
            name: sha256(helpers / name)
            for name in ("owned_output_trace_gpu_checks.py", "owned_output_trace_factor_checks.py")
        },
        "policy": (
            "Full-shaped trace before owned selection; unchanged pooled moments, configs, training, kernels, "
            "exact assertions and tolerances. Separate validation-only ownership; no production release."
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
    helpers = Path(__file__).parent
    if (
        initial["status"] != STATUS
        or initial["candidate"] != str(candidate.resolve())
        or initial["original_binding_sha256"] != sha256(root / PREVIOUS)
        or initial["original_candidate"] != str(previous)
        or initial["failure_evidence_sha256"]
        != sha256(root / "run-preparation/owned-output-handoff-failure-evidence-20261117T1533.json")
        or initial["preparation_helper_sha256"] != sha256(Path(__file__))
        or initial["files_sha256"] != files
        or initial["changed_files"] != sorted(CHANGES)
        or changed != CHANGES
        or any(
            (candidate / name).read_text() != transformed((previous / name).read_text())
            for name in ("experiment/execution_owner.py", "experiment/tests/test_owned_output_execution_owner.py")
        )
        or (candidate / "experiment/block_toner.py").read_text()
        != trace_block((previous / "experiment/block_toner.py").read_text())
        or initial["diagnostic_helper_sha256"]
        != {
            name: sha256(helpers / name)
            for name in ("owned_output_trace_gpu_checks.py", "owned_output_trace_factor_checks.py")
        }
        or sha256(candidate / "experiment/owned_output_gpu_checks.py")
        != initial["diagnostic_helper_sha256"]["owned_output_trace_gpu_checks.py"]
        or sha256(candidate / "experiment/tests/test_owned_output_trace_factors.py")
        != initial["diagnostic_helper_sha256"]["owned_output_trace_factor_checks.py"]
        or {key: value for key, value in cluster.items() if key != "time"}
        != {key: value for key, value in expected_cluster(previous).items() if key != "time"}
    ):
        raise ValueError("Trace successor preparation or source changed")
    return initial


def require_cpu_archive(archive: Path, initial: dict, preparation_sha256: str) -> None:
    expected = {f"candidate/{name}": digest for name, digest in initial["files_sha256"].items()}
    expected.update(
        {
            f"candidate/{PREPARATION}": preparation_sha256,
            "tracking/prepare_owned_output_trace.py": initial["preparation_helper_sha256"],
        }
    )
    alias = "candidate/artifacts/external-sources"
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) + 1 or {member.name for member in members} != set(expected) | {alias}:
            raise ValueError("Trace successor CPU archive inventory differs")
        link = bundle.getmember(alias)
        if not link.issym() or link.linkname != "pinned-sources-materialized":
            raise ValueError("Trace successor archived pinned alias differs")
        for name, digest in expected.items():
            member = bundle.getmember(name)
            if not member.isfile():
                raise ValueError("Trace successor archived source is not a regular file")
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("Trace successor archive bytes differ from tested source")


def cpu(candidate: Path, out: Path) -> dict:
    initial = require_preparation(candidate)
    if out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Trace successor CPU evidence needs a distinct external directory")
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
        bundle.add(Path(__file__), arcname="tracking/prepare_owned_output_trace.py", recursive=False)
    try:
        if require_preparation(candidate) != initial or sha256(preparation) != preparation_hash:
            raise ValueError("Trace successor source changed during CPU testing or archival")
        require_cpu_archive(archive, initial, preparation_hash)
    except ValueError as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        **initial,
        "status": CPU_STATUS if code == 0 and error is None else "owned_output_trace_cpu_failed",
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
        raise RuntimeError(f"Trace successor CPU failure preserved at {out}")
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
