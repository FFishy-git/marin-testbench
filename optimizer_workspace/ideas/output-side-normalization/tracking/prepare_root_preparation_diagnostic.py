# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a distinct scratch-readiness experiment only after authenticated saved-NS evidence."""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import root_preparation_contract as contract
from prepare_fused_attribution_production import replace_once
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write

PARENT_BINDING = "run-preparation/fused-continuation-binding-20261118.json"
REPLAY_BINDING = "run-preparation/saved-ns-replay-binding-20261118.json"
REPLAY_STATE = "run-preparation/saved-ns-replay-state-20261118.json"
REPLAY_OBSERVATION = "run-preparation/saved-ns-terminal-result-observation-20261118T1212.json"
REPLAY_OBSERVATION_SHA256 = "32c974881142da7b300046552363c5c4c8c05598f4707d06e414b7d7a418d42a"
ROOT_HELPER_SHA256 = "65236cbebf6bda7cef685600722927adac330d2267713df9604022adad1fd835"
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
COPIES = {
    "root_preparation_contract.py": "experiment/root_preparation_contract.py",
    "root_preparation_checked.py": "experiment/root_preparation_checked.py",
    "prepare_restored_roots.py": "experiment/prepare_restored_roots.py",
    "test_root_preparation_checked.py": "experiment/tests/test_root_preparation_checked.py",
}
LIFECYCLE_TEST = "experiment/tests/test_prepare_restored_roots.py"
OWNER_TEST = "experiment/tests/test_fused_attribution_validation_owner.py"
CHANGES = {"experiment/train.py", "experiment/fused_attribution_owner.py", LIFECYCLE_TEST, *COPIES.values()}
EXTERNAL = {"torch", "triton", "pytest", "pytest_timeout", "numpy", "swanlab", "dotenv"}


def parent(root):
    """Require terminal reconstructed replay evidence, not a live allocation claim."""
    observation_path = root / REPLAY_OBSERVATION
    if sha256(observation_path) != REPLAY_OBSERVATION_SHA256:
        raise ValueError("Root experiment lacks its independently accepted saved-NS result")
    observation = read(observation_path)
    state, replay = read(root / REPLAY_STATE), read(root / REPLAY_BINDING)
    frozen = read(root / PARENT_BINDING)
    previous = Path(frozen["candidate"])
    if (
        observation["status"] != "saved_ns_cuda_replay_terminal_reconstructed_exact_same_input_and_saved_outputs"
        or observation["comparison_independently_reconstructed"] is not True
        or observation["same_input_repeats"] != {"comparisons": 4, "all_exact_finite": True}
        or observation["own_saved_outputs"] != {"comparisons": 8, "all_exact_finite": True}
        or observation["all_cross_process_exact_finite"] is not True
        or len(observation["cross_process"]) != 4
        or state["status"] != "diagnostic_complete"
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or "error" in state
        or "cleanup_error" in state
        or state["binding_sha256"] != sha256(root / REPLAY_BINDING)
        or replay["predecessor"]["evidence_sha256"][PARENT_BINDING] != sha256(root / PARENT_BINDING)
        or state["restored_keeper"]["devices"] != frozen["allocation_predecessor"]["restored_keeper"]["devices"]
        or candidate_files(previous) != frozen["files_sha256"]
        or any(sha256(root / name) != value for name, value in observation["evidence_sha256"].items())
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
    ):
        raise ValueError("Root diagnostic predecessor source or reconstructed/restored replay evidence differs")
    prior_prepared = read(previous / "fused-continuation-preparation.json")
    prior_inventory = (
        frozen["files_sha256"]
        | prior_prepared["parent_metadata_sha256"]
        | {
            "fused-continuation-preparation.json": frozen["preparation_sha256"],
            "candidate-cpu-receipt.json": frozen["cpu_receipt_sha256"],
            "controller-cpu-receipt.json": frozen["controller_cpu_receipt_sha256"],
        }
        | {f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()}
    )
    require_archive(Path(frozen["archive"]), prior_inventory, alias=True)
    predecessor = {
        "status": "saved_ns_reconstructed_exact_outputs_restored_pending_live_root_handoff",
        "evidence_sha256": observation["evidence_sha256"] | {REPLAY_OBSERVATION: sha256(observation_path)},
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
        "child_pids": [item["pid"] for item in state["executions"]],
        "restored_keeper": state["restored_keeper"],
        "replay_binding_sha256": sha256(root / REPLAY_BINDING),
    }
    return frozen, predecessor


def helper_sources():
    """Archive the flat preparer closure plus the original timeout guard."""
    directory = Path(__file__).parent
    pending = [Path(__file__).name, *COPIES, "test_prepare_root_preparation_diagnostic.py"]
    result = {}
    while pending:
        name = pending.pop()
        if name in result:
            continue
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Root diagnostic helper absent or aliased: {name}")
        result[name] = path
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                modules = [node.module] if node.module else []
            else:
                continue
            for module in modules:
                head = module.split(".")[0]
                dependency = directory / f"{head}.py"
                if dependency.is_file():
                    pending.append(dependency.name)
                elif head not in sys.stdlib_module_names | EXTERNAL | {"experiment"}:
                    raise ValueError(f"Unbound root diagnostic helper: {name}: {module}")
    guard = directory / "root_diagnostic_pytest_guard.py"
    if not guard.is_file():
        guard = Path(__file__).resolve().parents[4] / "infra/ci/pytest_timeout_guard.py"
    result["root_diagnostic_pytest_guard.py"] = guard
    return result


def transformed_train(previous):
    text = (previous / "experiment/train.py").read_text()
    text = replace_once(
        text,
        "from .fused_attribution_owner import require_fused_attribution_owner\n",
        "from .fused_attribution_owner import require_fused_attribution_owner\n"
        "from .root_preparation_checked import checked_preparation\n",
    )
    text = replace_once(
        text,
        '    parser.add_argument("--trace-continuation", action="store_true")\n',
        '    parser.add_argument("--trace-continuation", action="store_true")\n'
        '    parser.add_argument("--prepare-restored-roots", action="store_true")\n',
    )
    text = replace_once(
        text,
        "    diagonal_reference = None\n",
        "    root_preparation_receipt = None\n"
        "    if args.prepare_restored_roots:\n"
        "        if args.resume is None:\n"
        '            raise ValueError("Root preparation is restricted to an owned checkpoint resume")\n'
        "        root_preparation_receipt = checked_preparation(fast, rank=rank, step=start_step)\n"
        "    diagonal_reference = None\n",
    )
    text = replace_once(
        text,
        '    metric_path = args.out / f"metrics_from_{start_step:06d}.jsonl"\n',
        "    if root_preparation_receipt is not None:\n"
        '        receipt_path = args.out / f"root_preparation_rank_{rank}.json"\n'
        "        if receipt_path.exists():\n"
        "            raise FileExistsError(receipt_path)\n"
        '        root_preparation_receipt["identity"] = identity\n'
        "        atomic_json(receipt_path, root_preparation_receipt)\n"
        '    metric_path = args.out / f"metrics_from_{start_step:06d}.jsonl"\n',
    )
    return text.encode()


def contents(previous):
    directory = Path(__file__).parent
    helper = directory / "prepare_restored_roots.py"
    if sha256(helper) != ROOT_HELPER_SHA256:
        raise ValueError("Root diagnostic must retain the accepted scratch-only helper")
    lifecycle = replace_once(
        (directory / "test_prepare_restored_roots.py").read_text(),
        "import prepare_restored_roots as preparation\n",
        "from experiment import prepare_restored_roots as preparation\n",
    )
    lifecycle = replace_once(
        lifecycle, "TASK = Path(__file__).resolve().parents[1]\n", "TASK = Path(__file__).resolve().parents[2]\n"
    )
    owner = replace_once(
        (previous / "experiment/fused_attribution_owner.py").read_text(),
        "from .fused_continuation_contract import (",
        "from .root_preparation_contract import (",
    )
    return {
        "experiment/train.py": transformed_train(previous),
        "experiment/fused_attribution_owner.py": owner.encode(),
        LIFECYCLE_TEST: lifecycle.encode(),
        **{destination: (directory / name).read_bytes() for name, destination in COPIES.items()},
    }


def record(root, candidate, frozen, predecessor):
    previous = Path(frozen["candidate"])
    sources = helper_sources()
    sources["test_prepare_restored_roots.py"] = Path(__file__).parent / "test_prepare_restored_roots.py"
    return {
        "status": contract.PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_binding_sha256": sha256(root / PARENT_BINDING),
        "parent_files_sha256": frozen["files_sha256"],
        "parent_metadata_sha256": {path.name: sha256(path) for path in previous.glob("*.json")},
        "files_sha256": candidate_files(candidate),
        "changed_files": sorted(CHANGES),
        "helpers_sha256": {name: sha256(path) for name, path in sources.items()},
        "data_receipt_sha256": frozen["data_receipt_sha256"],
        "allocation_predecessor": predecessor,
        "plan": contract.plan(candidate, root),
        "policy": (
            "CPU/source preparation only; no GPU release, controller, numerical correction or production acceptance."
        ),
    }


def prepare(root, candidate):
    root, candidate = root.resolve(strict=True), candidate.resolve()
    frozen, predecessor = parent(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.is_relative_to(previous) or previous.is_relative_to(candidate):
        raise FileExistsError("Root preparation diagnostic needs an unused independent candidate")
    payload = contents(previous)
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, raw in payload.items():
        path = candidate / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    prepared = record(root, candidate, frozen, predecessor)
    changed = {
        name
        for name in prepared["files_sha256"].keys() | frozen["files_sha256"].keys()
        if prepared["files_sha256"].get(name) != frozen["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError("Root diagnostic exceeded readiness/owner/trainer/test scope")
    write(candidate / contract.PREPARATION, prepared)
    return prepared


def require_preparation(root, candidate):
    frozen, predecessor = parent(root)
    prepared = read(candidate / contract.PREPARATION)
    expected = record(root, candidate, frozen, predecessor)
    changed = {
        name
        for name in expected["files_sha256"].keys() | frozen["files_sha256"].keys()
        if expected["files_sha256"].get(name) != frozen["files_sha256"].get(name)
    }
    if (
        set(prepared) != set(expected) | {"time"}
        or any(prepared[key] != value for key, value in expected.items())
        or changed != CHANGES
        or any((candidate / name).read_bytes() != value for name, value in contents(Path(frozen["candidate"])).items())
        or any(sha256(candidate / name) != value for name, value in prepared["parent_metadata_sha256"].items())
    ):
        raise ValueError("Root diagnostic candidate, actual schedule or exact transformation differs")
    return prepared


def cpu_command(candidate):
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        contract.PYTHON,
        "--with",
        "pytest-timeout",
        "python",
        "-P",
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "--rootdir",
        str(candidate),
        "-p",
        "no:cacheprovider",
        "-p",
        "root_diagnostic_pytest_guard",
        "--timeout=60",
        "--session-timeout=600",
        "--import-mode=importlib",
        "-m",
        SAFE_MARKERS,
        "-q",
        "experiment/tests",
    ]


def archive_inventory(prepared, preparation_sha256):
    return (
        prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {contract.PREPARATION: preparation_sha256}
        | {f"tracking/{name}": value for name, value in prepared["helpers_sha256"].items()}
    )


def cpu(root, candidate, out):
    prepared = require_preparation(root, candidate)
    if out.exists() or out.is_relative_to(candidate) or candidate.is_relative_to(out):
        raise FileExistsError("Root candidate CPU evidence needs a new external directory")
    sources = helper_sources() | {
        "test_prepare_restored_roots.py": Path(__file__).parent / "test_prepare_restored_roots.py"
    }
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, source in sources.items():
        shutil.copy2(source, helpers / name)
    command, log, archive = cpu_command(candidate), out / "candidate.log", out / "source.tar"
    with log.open("x") as stream:
        code = subprocess.run(
            [*command, "--junitxml", str(out / "tests.xml")],
            cwd=candidate,
            env=os.environ
            | {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "UV_OFFLINE": "1",
                "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers))),
            },
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=900,
        ).returncode
    preparation_hash = sha256(candidate / contract.PREPARATION)
    inventory = archive_inventory(prepared, preparation_hash)
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            source = out / name if name.startswith("tracking/") else candidate / name
            bundle.add(source, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    require_archive(archive, inventory, alias=True)
    unchanged = (
        require_preparation(root, candidate) == prepared
        and prepared["helpers_sha256"] == {name: sha256(source) for name, source in sources.items()}
        and prepared["helpers_sha256"] == {name: sha256(helpers / name) for name in sources}
    )
    result = {
        "status": contract.CPU_STATUS if type(code) is int and code == 0 and unchanged else "needs_attention",
        "root": str(root),
        "candidate": str(candidate),
        "cwd": str(candidate),
        "command": [*command, "--junitxml", str(out / "tests.xml")],
        "exit_code": code,
        "cuda_visible_devices": "",
        "files_sha256": prepared["files_sha256"],
        "preparation_sha256": preparation_hash,
        "helpers": str(helpers),
        "helpers_sha256": prepared["helpers_sha256"],
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "tests": str(out / "tests.xml"),
        "tests_sha256": sha256(out / "tests.xml"),
    }
    write(out / "receipt.json", result)
    if result["status"] != contract.CPU_STATUS:
        raise RuntimeError(f"Root diagnostic candidate CPU failure preserved at {out}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "check", "cpu"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root, candidate = args.root.resolve(strict=True), args.candidate.resolve()
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU verification requires --out")
    if args.phase == "prepare":
        result = prepare(root, candidate)
    elif args.phase == "check":
        result = require_preparation(root, candidate)
    else:
        result = cpu(root, candidate, args.out.resolve())
    print(json.dumps({"status": result["status"], "candidate": str(candidate)}))


if __name__ == "__main__":
    main()
