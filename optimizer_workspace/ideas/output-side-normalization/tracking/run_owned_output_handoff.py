# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze a separate owned-output validator; never issue a production release."""

import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

import owned_output_handoff as handoff
import torch
from analyze_tuning import executed_sources, normalized_sources
from benchmark_diagonal import source_definitions
from compare_runs import history, window_means
from launch_wave5 import NODES, PYTHON, node_action, read, sha256
from owned_output_validation import ARMS, GATES, SYNTHETIC_GATES, VALIDATION_ENVIRONMENT, validation_commands
from prepare_owned_output_handoff import HELPERS as PREPARATION_HELPERS
from prepare_owned_output_handoff import cpu_command, require_cpu_archive, require_preparation
from prepare_owned_output_observer import HELPERS as OBSERVER_HELPERS
from run_diagonal_diagnostic import PROCESS_MARKER, finish_child, tagged_processes
from watch_wave5 import process, write
from wave7_release import candidate_inventory, idle_training, local_node, node_guard, release_keeper

NODE = "28.216.17.139"
MASK = "0,1,2,3,4,5,6,7"
STATUS = "frozen_pending_completed_ceiling_and_owned_gpu_validation"
CEILING = "campaigns/lr-fixed-age-ceiling-visible8-recovery-20261117"
CEILING_FAMILIES = ("muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused")
WINDOW_TOKENS = 33554432
TRIAL_TOKENS = 167772160
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
CONTROLLER_FILES = ("run_owned_output_handoff.py", "test_owned_output_handoff_controller.py")
CONTROLLER_MARKER = "OUTPUT_MARIN_OWNED_HANDOFF_CONTROLLER"


def locations(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "run-preparation/owned-output-handoff-binding.json",
        root / "controllers/owned-output-handoff-validation-20261117/tracking",
        root / "run-preparation/owned-output-handoff-state.json",
    )


def helper_names(directory: Path) -> tuple[str, ...]:
    """Bind the exact transitive local import graph, including preparation and CPU tests."""
    pending = set((*OBSERVER_HELPERS, *PREPARATION_HELPERS, *CONTROLLER_FILES))
    found = set()
    while pending:
        name = pending.pop()
        path = directory / name
        if Path(name).name != name or not path.is_file():
            raise ValueError(f"Owned-output controller dependency is missing or not local: {name}")
        found.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                dependency = f"{module.split('.')[0]}.py"
                if (directory / dependency).is_file() and dependency not in found:
                    pending.add(dependency)
    return tuple(sorted(found))


def require_cpu(receipt_path: Path) -> tuple[Path, dict]:
    """Accept only the preparer's successful single candidate pass with CUDA hidden."""
    record = read(receipt_path)
    candidate = Path(record["candidate"])
    initial = require_preparation(candidate)
    if (
        record["status"] != "owned_output_handoff_cpu_verified_gpu_denied"
        or type(record["exit_code"]) is not int
        or record["exit_code"] != 0
        or "error" in record
        or "errors" in record
        or record["cuda_visible_devices"] != ""
        or record["command"] != cpu_command()
        or record["cwd"] != str(candidate)
        or record["preparation"] != str(candidate / "owned-output-handoff-preparation.json")
        or record["preparation_sha256"] != sha256(Path(record["preparation"]))
        or record["archive_sha256"] != sha256(Path(record["archive"]))
        or record["log_sha256"] != sha256(Path(record["log"]))
        or any(record[key] != value for key, value in initial.items() if key not in ("status", "time"))
    ):
        raise ValueError("Owned-output successful CPU command, source or archived evidence changed")
    require_cpu_archive(Path(record["archive"]), initial, record["preparation_sha256"])
    return candidate, record


def controller_cpu_command() -> list[str]:
    return [
        PYTHON,
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "-p",
        "no:cacheprovider",
        "-m",
        SAFE_MARKERS,
        "tracking/test_owned_output_handoff_controller.py",
        "tracking/test_owned_output_handoff.py",
        "tracking/test_prepare_owned_output_handoff.py",
        "-q",
    ]


def require_controller_archive(archive: Path, hashes: dict) -> None:
    """Require exactly the helper bytes whose CPU command was verified."""
    expected = {f"tracking/{name}": digest for name, digest in hashes.items()}
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise ValueError("Owned-output controller CPU archive inventory differs")
        for member in members:
            if not member.isfile():
                raise ValueError("Owned-output controller CPU archive source is not a regular file")
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected[member.name]:
                    raise ValueError("Owned-output controller CPU archive bytes differ from tested helpers")


def cpu(receipt_path: Path, out: Path) -> dict:
    """Test controller contracts separately from the canonical candidate pass."""
    candidate, record = require_cpu(receipt_path)
    helpers = Path(__file__).parent.resolve()
    names = helper_names(helpers)
    hashes = {name: sha256(helpers / name) for name in names}
    if out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Controller CPU evidence needs a new external directory")
    out.mkdir(parents=True, exist_ok=False)
    log = out / "controller.log"
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                controller_cpu_command(),
                cwd=helpers.parent,
                env=os.environ
                | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers)))},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code = 124
            error = "Controller CPU command exceeded 600 seconds; exit 124 is the timeout sentinel"
    archive = out / "controller-source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in names:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    try:
        if require_cpu(receipt_path) != (candidate, record) or hashes != {
            name: sha256(helpers / name) for name in names
        }:
            raise ValueError("Owned-output source or controller changed during CPU verification or archival")
        require_controller_archive(archive, hashes)
    except ValueError as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": (
            "owned_output_controller_cpu_verified_gpu_denied"
            if type(code) is int and code == 0 and error is None
            else "owned_output_controller_cpu_failed"
        ),
        "candidate_cpu_receipt": str(receipt_path.resolve()),
        "candidate_cpu_receipt_sha256": sha256(receipt_path),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "command": controller_cpu_command(),
        "cwd": str(helpers.parent),
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
    if type(code) is not int or code != 0 or error is not None:
        raise RuntimeError(f"Owned-output controller CPU failure preserved at {out}")
    return result


def require_controller_cpu(receipt_path: Path, candidate_receipt: Path) -> dict:
    """Require the separately tested controller bytes and its canonical CPU outcome."""
    record = read(receipt_path)
    helpers = Path(__file__).parent.resolve()
    names = helper_names(helpers)
    if (
        record["status"] != "owned_output_controller_cpu_verified_gpu_denied"
        or type(record["exit_code"]) is not int
        or record["exit_code"] != 0
        or "error" in record
        or "errors" in record
        or record["cuda_visible_devices"] != ""
        or record["candidate_cpu_receipt"] != str(candidate_receipt.resolve())
        or record["candidate_cpu_receipt_sha256"] != sha256(candidate_receipt)
        or record["command"] != controller_cpu_command()
        or record["cwd"] != str(Path(record["helpers"]).parent)
        or set(record["helpers_sha256"]) != set(names)
        or any(sha256(helpers / name) != digest for name, digest in record["helpers_sha256"].items())
        or record["log_sha256"] != sha256(Path(record["log"]))
        or record["archive_sha256"] != sha256(Path(record["archive"]))
    ):
        raise ValueError("Owned-output separate controller CPU success or tested helper closure changed")
    require_controller_archive(Path(record["archive"]), record["helpers_sha256"])
    return record


def require_ceiling_binding(root: Path) -> dict:
    """Inspect only the original recovery campaign; completion is deliberately not a freeze gate."""
    campaign_path = root / CEILING / "campaign.json"
    campaign = read(campaign_path)
    path = campaign_path.parent / "execution-binding.json"
    frozen = read(path)
    helpers = campaign_path.parent / "execution-helpers"
    required = {"run_fixed_age_tuning.py", "analyze_lr_ceiling.py", "prepare_lr_ceiling.py"}
    if (
        campaign["root"] != str(root)
        or campaign["predecessor_wave"] != "wave9"
        or "recovery" not in campaign
        or campaign["predecessor_release_sha256"] != sha256(root / "run-preparation/wave9-launch-binding.json")
        or campaign["selection"]["training_window_tokens"] != WINDOW_TOKENS
        or frozen["campaign"] != str(campaign_path)
        or frozen["campaign_sha256"] != sha256(campaign_path)
        or frozen["helpers"] != str(helpers)
        or not required <= frozen["helpers_sha256"].keys()
        or any(
            Path(name).name != name or sha256(helpers / name) != digest
            for name, digest in frozen["helpers_sha256"].items()
        )
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["validation_binding_sha256"] != sha256(campaign_path.parent / "validation-binding.json")
        or frozen["environment"] != VALIDATION_ENVIRONMENT
        or [(item["family"], item["node"]) for item in frozen["plan"]] != list(zip(CEILING_FAMILIES, NODES, strict=True))
        or len(campaign["trials"]) != 8
    ):
        raise ValueError("Original visible-eight LR recovery campaign/helper/archive binding changed")
    for assignment in frozen["plan"]:
        expected = [trial for trial in campaign["trials"] if trial["family"] == assignment["family"]]
        if len(expected) != 2 or [trial["name"] for trial in assignment["trials"]] != [
            trial["name"] for trial in expected
        ]:
            raise ValueError("LR recovery requires both exact prepared LR points per family")
        for planned, trial in zip(assignment["trials"], expected, strict=True):
            if (
                planned["config_sha256"] != trial["config_sha256"]
                or sha256(Path(trial["config"])) != trial["config_sha256"]
                or read(Path(trial["config"]))["optimizer"]["lr"] != trial["matrix_lr"]
                or trial["tokens"] != TRIAL_TOKENS
                or trial["steps"] != 320
                or planned["tokens"] != TRIAL_TOKENS
            ):
                raise ValueError("LR recovery actual LR, budget or configuration identity changed")
    release_path = root / "run-preparation/wave9-launch-binding.json"
    release = read(release_path)
    if (
        release["status"] != "frozen_validated_launch_release"
        or release["root"] != str(root)
        or release["helpers"] != str(root / "controllers/wave9-launch-20261117/tracking")
        or any(sha256(Path(release["helpers"]) / name) != digest for name, digest in release["helpers_sha256"].items())
        or sha256(Path(release["archive"])) != release["archive_sha256"]
    ):
        raise ValueError("Owned-output prerequisite requires the accepted original Wave-9 release")
    subprocess.run(
        [PYTHON, str(helpers / "run_fixed_age_tuning.py"), "--campaign", str(campaign_path), "--phase", "check"],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return {
        "campaign": str(campaign_path),
        "campaign_sha256": sha256(campaign_path),
        "execution_binding_sha256": sha256(path),
        "wave9_release_sha256": sha256(release_path),
    }


def freeze(root: Path, receipt_path: Path, *, controller_cpu_receipt: Path) -> dict:
    """Archive a distinct validation-only identity while LR recovery may still be pending."""
    root = root.resolve()
    candidate, record = require_cpu(receipt_path)
    require_controller_cpu(controller_cpu_receipt, receipt_path)
    prerequisite = require_ceiling_binding(root)
    binding_path, helpers, _ = locations(root)
    archive = root / "run-preparation/source-owned-output-handoff-validation-20261117.tar"
    if any(path.exists() for path in (binding_path, helpers, archive)):
        raise FileExistsError("Owned-output binding exists; preserve it without refreezing")
    source_helpers = Path(__file__).parent.resolve()
    names = helper_names(source_helpers)
    hashes = {name: sha256(source_helpers / name) for name in names}
    helpers.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "x") as bundle:
        for name in names:
            shutil.copy2(source_helpers / name, helpers / name)
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        for name in record["files_sha256"]:
            bundle.add(candidate / name, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
        bundle.add(candidate / "owned-output-handoff-preparation.json", arcname="preparation.json", recursive=False)
        bundle.add(receipt_path, arcname="cpu-receipt.json", recursive=False)
        bundle.add(controller_cpu_receipt, arcname="controller-cpu-receipt.json", recursive=False)
    if (
        hashes != {name: sha256(helpers / name) for name in names}
        or require_cpu(receipt_path) != (candidate, record)
        or require_controller_cpu(controller_cpu_receipt, receipt_path)["helpers_sha256"] != hashes
    ):
        raise ValueError("Owned-output source/helpers changed during freeze; preserve partial evidence")
    result = {
        "status": STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "node": NODE,
        "arms": ARMS,
        "required_gates": GATES,
        "environment": VALIDATION_ENVIRONMENT,
        "cpu_receipt": str(receipt_path.resolve()),
        "cpu_receipt_sha256": sha256(receipt_path),
        "controller_cpu_receipt": str(controller_cpu_receipt.resolve()),
        "controller_cpu_receipt_sha256": sha256(controller_cpu_receipt),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "files_sha256": record["files_sha256"],
        "source_sha256": {
            name: digest
            for name, digest in record["files_sha256"].items()
            if name.startswith("experiment/") and name.endswith(".py")
        },
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "commands": validation_commands(PYTHON, candidate, root, root / "validation/owned_output_handoff"),
        "ceiling": prerequisite,
        "policy": (
            "Validation only. Pending ceiling blocks GPU transfer. "
            "No queue, production release, retry or parent mutation."
        ),
    }
    write(binding_path, result)
    return result


def binding(root: Path) -> dict:
    """Reconstruct the exact frozen helper allowlist, candidate CPU evidence and command plan."""
    root = root.resolve()
    path, helpers, _ = locations(root)
    frozen = read(path)
    if Path(__file__).resolve() != (helpers / CONTROLLER_FILES[0]).resolve() or frozen["status"] != STATUS:
        raise ValueError("Owned-output validation requires its separate original frozen controller binding")
    candidate, record = require_cpu(Path(frozen["cpu_receipt"]))
    require_controller_cpu(Path(frozen["controller_cpu_receipt"]), Path(frozen["cpu_receipt"]))
    names = helper_names(helpers)
    if (
        frozen["root"] != str(root)
        or frozen["candidate"] != str(candidate)
        or frozen["node"] != NODE
        or frozen["helpers"] != str(helpers)
        or tuple(frozen["arms"]) != ARMS
        or tuple(frozen["required_gates"]) != GATES
        or frozen["environment"] != VALIDATION_ENVIRONMENT
        or frozen["cpu_receipt_sha256"] != sha256(Path(frozen["cpu_receipt"]))
        or frozen["controller_cpu_receipt_sha256"] != sha256(Path(frozen["controller_cpu_receipt"]))
        or frozen["files_sha256"] != record["files_sha256"]
        or frozen["files_sha256"] != candidate_inventory(candidate)
        or frozen["source_sha256"]
        != {
            name: digest
            for name, digest in record["files_sha256"].items()
            if name.startswith("experiment/") and name.endswith(".py")
        }
        or set(frozen["helpers_sha256"]) != set(names)
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or frozen["archive"] != str(root / "run-preparation/source-owned-output-handoff-validation-20261117.tar")
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["commands"] != validation_commands(PYTHON, candidate, root, root / "validation/owned_output_handoff")
        or len(frozen["commands"]) != 14
        or frozen["ceiling"] != require_ceiling_binding(root)
    ):
        raise ValueError("Owned-output frozen source, allowlist, CPU evidence or fourteen-command plan changed")
    return frozen


def require_completed_ceiling(root: Path, frozen: dict) -> dict:
    """Rederive completed LR histories and freshly probe every original worker over SSH."""
    if frozen["ceiling"] != require_ceiling_binding(root):
        raise ValueError("Owned-output LR prerequisite is not the original recovery binding")
    campaign_path = root / CEILING / "campaign.json"
    directory = campaign_path.parent
    campaign, execution = read(campaign_path), read(directory / "execution-binding.json")
    helpers = directory / "execution-helpers"
    # snapshot performs its own original-bound SSH probes. Do not persist observation in the parent campaign.
    script = (
        "import json,sys\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\n"
        "from run_fixed_age_tuning import snapshot\n"
        "print(json.dumps(snapshot(Path(sys.argv[2])),allow_nan=False))\n"
    )
    observed = json.loads(
        subprocess.run(
            [PYTHON, "-c", script, str(helpers), str(campaign_path)],
            env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
            check=True,
            capture_output=True,
            text=True,
            timeout=1500,
        ).stdout.splitlines()[-1]
    )
    if (
        observed["status"] != "complete"
        or observed["all_workers_terminal"] is not True
        or observed["campaign_sha256"] != sha256(campaign_path)
        or observed["binding_sha256"] != sha256(directory / "execution-binding.json")
        or len(observed["workers"]) != 4
        or {worker["family"] for worker in observed["workers"]} != set(CEILING_FAMILIES)
        or "error" in observed
    ):
        raise RuntimeError("Original LR recovery workers are active, unknown or incomplete")
    for worker in observed["workers"]:
        receipt = worker["receipt"]
        assignment = next(item for item in execution["plan"] if item["family"] == worker["family"])
        if (
            worker["status"] != "complete"
            or receipt["status"] != "complete"
            or receipt["family"] != assignment["family"]
            or receipt["node"] != assignment["node"]
            or receipt["binding_sha256"] != observed["binding_sha256"]
            or type(receipt.get("keeper_restore")) is not int
            or receipt["keeper_restore"] != 0
            or worker.get("descendants")
            or any(key in item for item in (worker, receipt) for key in ("error", "cleanup_error"))
            or len(receipt["trials"]) != 2
        ):
            raise RuntimeError("LR recovery original workers, child cleanup or keeper restoration are unverified")
        for trial, planned in zip(receipt["trials"], assignment["trials"], strict=True):
            child_path = directory / "workers" / f"{trial['name']}.execution.json"
            child = read(child_path)
            if (
                trial["name"] != planned["name"]
                or trial["command"] != planned["command"]
                or trial["status"] != "complete"
                or type(trial.get("exit_code")) is not int
                or trial["exit_code"] != 0
                or trial["log_sha256"] != sha256(Path(trial["log"]))
                or trial["execution_sha256"] != sha256(child_path)
                or child["command"] != planned["command"]
                or type(child.get("cleanup_exit_code")) is not int
                or child["cleanup_exit_code"] != 0
                or "cleanup_error" in child
            ):
                raise RuntimeError("LR recovery exact trial exit/log or bounded cleanup receipt changed")
    handoffs = []
    for assignment in execution["plan"]:
        command = [
            PYTHON,
            str(Path(__file__).parent / "owned_output_handoff.py"),
            "--root",
            str(root),
            "--family",
            assignment["family"],
        ]
        absence = json.loads(
            subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    f"root@{assignment['node']}",
                    shlex.join(["env", "CUDA_VISIBLE_DEVICES=", *command]),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            ).stdout.splitlines()[-1]
        )
        if (
            absence.get("status") != "original_training_absent_exact_keeper_verified"
            or absence.get("family") != assignment["family"]
            or absence.get("node") != assignment["node"]
            or absence.get("original_handles_absent") is not True
            or absence.get("training_child_markers_absent") is not True
            or len(absence.get("keeper", {}).get("generation", {})) != 9
            or len(absence.get("keeper", {}).get("occupied", [])) != 8
        ):
            raise RuntimeError("LR recovery original-bound SSH absence is unverified")
        handoffs.append(absence)
    analysis_script = (
        "import json,sys\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\n"
        "from analyze_lr_ceiling import analyze\nprint(json.dumps(analyze(Path(sys.argv[2])),allow_nan=False))\n"
    )
    actual = json.loads(
        subprocess.run(
            [PYTHON, "-c", analysis_script, str(helpers), str(campaign_path)],
            env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        ).stdout
    )
    analysis_path, cloud_path = directory / "analysis.json", directory / "swanlab-upload.json"
    saved, cloud = read(analysis_path), read(cloud_path)
    if (
        actual["status"] != "complete"
        or {key: value for key, value in saved.items() if key != "time"}
        != {key: value for key, value in actual.items() if key != "time"}
        or len(actual["trials"]) != 8
        or {trial["name"] for trial in actual["trials"]} != {trial["name"] for trial in campaign["trials"]}
        or cloud["status"] != "verified"
        or cloud["visibility"] != "PRIVATE"
        or not cloud["project"].endswith("/output-side-normalization-marin")
        or cloud["campaign_sha256"] != sha256(campaign_path)
        or cloud["execution_binding_sha256"] != sha256(directory / "execution-binding.json")
        or set(cloud["runs"]) != {trial["name"] for trial in campaign["trials"]}
    ):
        raise ValueError(
            "Completed exact-LR five-window analysis and private full-resolution cloud evidence are required"
        )
    for trial in campaign["trials"]:
        measured = next(row for row in actual["trials"] if row["name"] == trial["name"])
        run = directory / "runs" / trial["name"]
        rows = history(run).rows
        evaluation_tokens = [step * 524288 for step in (1, 41, 81, 121, 161, 201, 241, 281, 320)]
        late = [tokens for tokens in evaluation_tokens if tokens > TRIAL_TOKENS // 2]
        if (
            measured["complete"] is not True
            or measured["observed_tokens"] != TRIAL_TOKENS
            or measured["matrix_lr"] != trial["matrix_lr"]
            or len(measured["training_windows"]) != 5
            or [window["end_tokens"] for window in measured["training_windows"]]
            != [(index + 1) * WINDOW_TOKENS for index in range(5)]
            or measured["training_windows"]
            != [{"end_tokens": endpoint, "mean": mean} for endpoint, mean in window_means(rows, WINDOW_TOKENS).items()]
            or len(rows) != 320
            or [row["step"] for row in rows] != list(range(1, 321))
            or any(row["tokens"] != row["step"] * 524288 for row in rows)
            or any(
                [row["tokens"] for row in rows if metric in row] != evaluation_tokens
                or measured["late_evaluation_tokens"][metric] != late
                for metric in ("selection_loss", "c4_loss")
            )
        ):
            raise ValueError("LR recovery requires five full shared windows, actual LR and exact late evaluations")
        checkpoint = run / "checkpoints/step_000320"
        entry = cloud["runs"][trial["name"]]
        verification = entry["verification"]
        provenance = {
            "manifest_sha256": sha256(run / "run.json"),
            "metrics_sha256": sha256(run / "metrics_from_000000.jsonl"),
            "checkpoint_receipt_sha256": sha256(checkpoint / "complete.json"),
            "checkpoint_rank_bytes": [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)],
            "campaign_sha256": sha256(campaign_path),
            "execution_binding_sha256": cloud["execution_binding_sha256"],
            "trial_config_sha256": trial["config_sha256"],
        }
        if (
            entry["status"] != "verified"
            or verification["state"] != "FINISHED"
            or any(entry.get(key) != value for key, value in provenance.items())
            or read(checkpoint / "complete.json")["step"] != 320
            or read(checkpoint / "complete.json")["world"] != 8
            or any(size <= 0 for size in provenance["checkpoint_rank_bytes"])
            or any(
                verification["point_counts"].get(key) != count
                for key, count in (("tokens", 320), ("train/loss", 320), ("eval/selection_loss", 9), ("eval/c4_loss", 9))
            )
        ):
            raise ValueError("LR recovery cloud history differs from its final local checkpoint/history provenance")
    return {
        "status": "completed_ceiling_workers_restored_absent_cloud_verified",
        **frozen["ceiling"],
        "analysis_sha256": sha256(analysis_path),
        "cloud_sha256": sha256(cloud_path),
        "fresh_handoffs": handoffs,
    }


def require_engine_state(engine: dict, step: int, decay: float, *, root_only: bool) -> None:
    """Check actual owned root caches and counters independently of continuation agreement."""
    schema = {"groups", "momentum", "weight", "steps", "t", "fallbacks", "extra", "refresh_install_age"}
    if (
        set(engine) != schema
        or engine["steps"] != step
        or type(engine["steps"]) is not int
        or engine["t"] != step
        or type(engine["t"]) is not int
        or engine["refresh_install_age"] != 1
        or type(engine["weight"]) is not float
        or abs(engine["weight"] - (1 - decay**step)) > 1e-12
        or set(engine["groups"]) != {"d", "f"}
        or (root_only and engine["momentum"] != {})
    ):
        raise ValueError("Owned-output root-only engine schema, EMA weight or update counters differ")
    group_schema = {"C", "tol", "basis_step", "warm", "cached_root_graph_active", "SB", "S2", "pending"}
    for name, count, width in (("d", 12, 512), ("f", 4, 2048)):
        group = engine["groups"][name]
        if (
            set(group) != group_schema
            or tuple(group["C"].shape) != (count, width, width)
            or group["C"].dtype != torch.float32
            or len(group["basis_step"]) != count
            or len(group["pending"]) != count
            or any(type(value) is not int or not 0 <= value < step for value in group["basis_step"])
            or group["cached_root_graph_active"] is not True
            or group["SB"] is None
            or group["S2"] is None
        ):
            raise ValueError("Owned-output all-rank root cache inventory differs")


def require_checkpoint_identity(directory: Path, config: dict, source: Path, frozen: dict, step: int) -> dict:
    """Reconstruct every rank's identity and full retained numerical state before comparison."""
    completion = read(directory / "complete.json")
    if completion["step"] != step or completion["world"] != 8:
        raise ValueError("Owned-output checkpoint completion step or world differs")
    if set(path.name for path in directory.glob("rank_*.pt")) != {f"rank_{rank}.pt" for rank in range(8)}:
        raise ValueError("Owned-output checkpoint lacks the exact eight-rank inventory")
    dimensions = (
        config["model"]["layers"],
        config["model"]["hidden"],
        config["model"]["mlp"],
        config["training"]["world_size"],
    )
    if dimensions != (32, 512, 2048, 8):
        raise ValueError("Owned-output checkpoint requires the actual full-model geometry")
    geometry = source_definitions(
        source / "experiment/block_toner.py", ("BLOCK_KINDS", "HIDDEN_OUTPUTS", "MLP_OUTPUTS"), {}
    )
    if (
        geometry["BLOCK_KINDS"] != ("q", "k", "v", "o", "gate", "up", "down")
        or geometry["HIDDEN_OUTPUTS"] != ("q", "k", "v", "o", "down")
        or geometry["MLP_OUTPUTS"] != ("gate", "up")
    ):
        raise ValueError("Owned-output frozen projection-key order changed")
    expected_config = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    expected_sources = executed_sources(source, frozen)
    arm = config["experiment"]["arm"]
    baseline = arm == ARMS[0]
    if arm not in ARMS:
        raise ValueError("Owned-output checkpoint belongs to an unplanned arm")
    reference_moments = None
    for rank in range(8):
        saved = torch.load(directory / f"rank_{rank}.pt", map_location="cpu", weights_only=False)
        identity = saved["identity"]
        if (
            any(
                type(saved[key]) is not int or saved[key] != value
                for key, value in (("rank", rank), ("world", 8), ("step", step))
            )
            or saved["config"] != config
            or saved["next_global_sequence"] != step * config["training"]["global_batch_sequences"]
            or set(identity) != {"schema", "config_sha256", "data_receipt_sha256", "source_sha256"}
            or type(identity["schema"]) is not int
            or identity["schema"] != 1
            or identity["config_sha256"] != expected_config
            or identity["data_receipt_sha256"] != frozen["data_receipt_sha256"]
            or normalized_sources(source, identity["source_sha256"]) != expected_sources
        ):
            raise ValueError(f"Owned-output expected checkpoint rank/config/source identity differs for rank {rank}")
        fast = saved["fast"]
        if baseline:
            require_engine_state(fast, step, config["statistics"]["momentum"], root_only=False)
            continue
        if (
            set(fast) != {"lagged", "input", "output", "input_diagonal", "output_diagonal", "momentum", "steps"}
            or type(fast["steps"]) is not int
            or fast["steps"] != step
            or fast["lagged"] != {}
            or fast["output"] is not None
            or fast["input_diagonal"] is not None
        ):
            raise ValueError(f"Owned-output block state schema or steps differ for rank {rank}")
        require_engine_state(fast["input"], step, config["statistics"]["momentum"], root_only=True)
        momentum_shapes = {(512, 512): (16, 512, 512), (2048, 512): (8, 2048, 512), (512, 2048): (4, 512, 2048)}
        if set(fast["momentum"]) != set(momentum_shapes) or any(
            tuple(fast["momentum"][shape].shape) != size or fast["momentum"][shape].dtype != torch.float32
            for shape, size in momentum_shapes.items()
        ):
            raise ValueError(f"Owned-output rank {rank} lacks its 28 locally owned momentum projections")
        diagonal = fast["output_diagonal"]
        if (
            set(diagonal) != {"moment", "weight"}
            or set(diagonal["moment"]) != {512, 2048}
            or type(diagonal["weight"]) is not float
            or abs(diagonal["weight"] - (1 - config["statistics"]["momentum"] ** step)) > 1e-12
        ):
            raise ValueError(f"Owned-output full output-diagonal schema or EMA weight differs for rank {rank}")
        for width, size in ((512, (160, 512)), (2048, (64, 2048))):
            moment = diagonal["moment"][width]
            if (
                not isinstance(moment, torch.Tensor)
                or tuple(moment.shape) != size
                or moment.dtype != torch.float32
                or not torch.isfinite(moment).all()
                or (moment < 0).any()
                or (reference_moments is not None and not torch.equal(moment, reference_moments[width]))
            ):
                raise ValueError(
                    f"Owned-output full pooled moment inventory or all-rank equality differs for rank {rank}"
                )
        if reference_moments is None:
            reference_moments = diagonal["moment"]
    return {
        "ranks": 8,
        "step": step,
        "pooled_keys_per_rank": 0 if baseline else 224,
        "owned_keys_per_rank": 28,
        "full_moment_shapes": [] if baseline else [[160, 512], [64, 2048]],
    }


def owned_contract_functions(source: Path) -> dict:
    """Extract the source-bound contract, including tuple-assigned tolerance constants."""
    path = source / "experiment/owned_output_validation.py"
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "Path": Path,
        "ARMS": ARMS,
        "VALIDATION_ENVIRONMENT": VALIDATION_ENVIRONMENT,
    }
    namespace["require_upstream_capture_history"] = source_definitions(
        source / "experiment/upstream_validation.py",
        ("require_upstream_capture_history",),
        {"json": json, "math": math, "Path": Path, "ARMS": ARMS},
    )["require_upstream_capture_history"]
    constants = {
        "WORLD",
        "LAYERS",
        "EMA_DECAY",
        "POOLED_MOMENT_SHAPES",
        "OWNED_PROJECTION_SHAPES",
        "PROJECTION_KINDS",
        "RTOL",
        "ATOL",
        "OWNED_RANK_CHECKS",
        "OWNED_SYNTHETIC_CHECKS",
        "OWNED_STACK_CHECKS",
        "OWNED_CAPTURE_CHECKS",
        "OWNED_CAPTURE_ERRORS",
    }
    selected, found = [], set()
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        targets = {name.id for target in node.targets for name in ast.walk(target) if isinstance(name, ast.Name)}
        if targets & constants:
            if not targets <= constants:
                raise ValueError("Owned-output contract constant assignment exceeds its allowlist")
            selected.append(node)
            found.update(targets)
    if found != constants:
        raise ValueError("Owned-output source-bound contract constants are incomplete")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return source_definitions(
        path,
        (
            "owned_config_sha256",
            "require_owned_config",
            "owned_rank_coverage",
            "require_rank_records",
            "require_owned_update_stack_records",
            "require_owned_synthetic_receipt",
            "require_owned_capture_history",
        ),
        namespace,
    )


def require_validation(root: Path, frozen: dict) -> dict:
    """Rederive all fourteen commands, every rank checkpoint, and source-owned histories."""
    work = root / "validation/owned_output_handoff"
    receipt = read(root / "run-preparation/validation-owned-output-handoff-gpu.json")
    state = read(locations(root)[2])
    source = Path(frozen["candidate"])
    commands = receipt["evidence"]["gpu_commands"]
    owner = {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
    if (
        receipt["owner"] != owner
        or receipt["source_sha256"] != frozen["source_sha256"]
        or receipt["environment"] != VALIDATION_ENVIRONMENT
        or receipt["data_receipt_sha256"] != frozen["data_receipt_sha256"]
        or receipt["config_sha256"]
        != {arm: sha256(source / f"experiment/configs/{arm}_0p5m_seed0.json") for arm in ARMS}
        or not all(receipt["checks"].get(gate) is True for gate in GATES)
        or set(receipt["evidence"]["real_data_smokes"]) != set(ARMS)
        or len(commands) != 14
        or [item["command"] for item in commands] != frozen["commands"]
        or frozen["commands"] != validation_commands(PYTHON, source, root, work)
        or any(
            type(item["exit_code"]) is not int
            or item["exit_code"] != 0
            or sha256(Path(item["log"])) != item["log_sha256"]
            for item in commands
        )
    ):
        raise ValueError("Owned-output validation gates, command coverage or log identity differ")
    synthetic = read(work / "gpu_checks.json")
    if (
        type(synthetic["world"]) is not int
        or synthetic["world"] != 2
        or synthetic["source_sha256"] != frozen["source_sha256"]
        or not all(synthetic["checks"].get(gate) is True for gate in SYNTHETIC_GATES)
    ):
        raise ValueError("Owned-output original two-rank synthetic source/gates differ")
    owned = owned_contract_functions(source)
    owned_path = work / "owned_output_gpu_checks.json"
    owned_record = read(owned_path)
    summary = owned["require_owned_synthetic_receipt"](
        owned_record, frozen["source_sha256"], read(source / f"experiment/configs/{ARMS[2]}_0p5m_seed0.json")
    )
    if (
        owned_record["owner"] != owner
        or receipt["evidence"]["owned_gpu_receipt"] != str(owned_path)
        or receipt["evidence"]["owned_gpu_receipt_sha256"] != sha256(owned_path)
        or receipt["evidence"]["owned_gpu_summary"] != summary
    ):
        raise ValueError("Owned-output distinct eight-rank synthetic receipt or owner differs")
    comparisons = source_definitions(
        source / "experiment/records.py", ("compare_tree", "compare_continuation"), {"torch": torch}
    )
    histories = source_definitions(
        source / "experiment/validate.py",
        ("require_cached_root_checkpoint", "require_fixed_age_history", "require_outer_capture_history"),
        {"torch": torch, "json": json},
    )
    for arm in ARMS:
        smoke = receipt["evidence"]["real_data_smokes"][arm]
        config = read(source / f"experiment/configs/{arm}_0p5m_seed0.json")
        full, resumed = work / f"{arm}_continuous", work / f"{arm}_resumed"
        if smoke["out"] != str(full) or smoke["cached_root_checkpoint"] != histories["require_cached_root_checkpoint"](
            resumed / "checkpoints/step_000022", 8
        ):
            raise ValueError("Owned-output cached-root checkpoint identity differs")
        for step in (22, 24):
            for directory in (full, resumed):
                require_checkpoint_identity(directory / f"checkpoints/step_{step:06d}", config, source, frozen, step)
        for label, directory in (("continuous", full), ("resumed", resumed)):
            manifest = read(directory / "run.json")
            if (
                manifest["status"] != "execution_check_complete"
                or manifest["completed_steps"] != 24
                or manifest["config"] != config
                or manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
                or normalized_sources(source, manifest["source_sha256"]) != executed_sources(source, frozen)
                or smoke["fixed_age_history"][label] != histories["require_fixed_age_history"](directory)
            ):
                raise ValueError("Owned-output actual trajectory/source or fixed-age history changed")
            if arm == ARMS[1]:
                histories["require_outer_capture_history"](directory)
            elif arm in ARMS[2:]:
                if smoke["owned_output_observations"][label] != owned["require_owned_capture_history"](directory, arm):
                    raise ValueError("Owned-output explicit all-rank captured history differs")
    for arm in ARMS:
        smoke = receipt["evidence"]["real_data_smokes"][arm]
        full, resumed = work / f"{arm}_continuous", work / f"{arm}_resumed"
        for label, step in (("boundary_continuation", 22), ("continuation", 24)):
            actual = comparisons["compare_continuation"](
                full / f"checkpoints/step_{step:06d}", resumed / f"checkpoints/step_{step:06d}", 8
            )
            if smoke[label] != actual or actual["tensor_comparisons"] <= 0:
                raise ValueError("Owned-output all-rank continuation evidence changed")
    return receipt


def bounded_validator(command: list[str], candidate: Path, log: Path, state: dict, state_path: Path) -> int:
    """Acknowledge the validator generation before CUDA ownership can be granted."""
    marker, generation = uuid4().hex, {}
    state.update(command=command, child_marker=marker, log=str(log), spawn_status="intent")
    write(state_path, state)
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=candidate,
            env=os.environ | VALIDATION_ENVIRONMENT | {PROCESS_MARKER: marker},
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            observed = process(child.pid)
            if observed is None or observed[0] == "Z":
                raise RuntimeError("Owned-output validator exited before generation acknowledgment")
            state.update(child_pid=child.pid, child_start_ticks=observed[1], spawn_status="acknowledged")
            write(state_path, state)
            deadline = time.monotonic() + 14400
            while True:
                generation.update(tagged_processes(marker))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, 14400)
                try:
                    return child.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            try:
                finish_child(child, marker, generation)
                state["cleanup_exit_code"] = child.returncode
            except Exception as error:
                state["cleanup_error"] = str(error)
                raise
            finally:
                state["child_generation"] = generation
                write(state_path, state)


def start_controller(root: Path, frozen: dict, predecessor: dict) -> dict:
    """Persist one detached launch intent and exact generation acknowledgment without retry."""
    binding_path, helpers, state_path = locations(root)
    directory = helpers.parent
    intent_path = directory / "controller.json"
    log = directory / "controller.log"
    evidence = (
        intent_path,
        log,
        state_path,
        root / "validation/owned_output_handoff",
        root / "run-preparation/validation-owned-output-handoff-gpu.json",
        root / "run-preparation/validation-owned-output-handoff-gpu.log",
    )
    with (directory / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.exists() for path in evidence):
            raise FileExistsError("Owned handoff already has intent or execution evidence; no retry")
        command = [PYTHON, str(helpers / CONTROLLER_FILES[0]), "--root", str(root), "--phase", "validate"]
        intent = {
            "status": "starting",
            "command": command,
            "binding_sha256": sha256(binding_path),
            "candidate": frozen["candidate"],
            "marker": uuid4().hex,
            "cuda_visible_devices": MASK,
            "predecessor": predecessor,
            "log": str(log),
        }
        write(intent_path, intent)
        try:
            with log.open("x") as stream:
                child = subprocess.Popen(
                    command,
                    cwd=helpers,
                    env=os.environ | VALIDATION_ENVIRONMENT | {CONTROLLER_MARKER: intent["marker"]},
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            intent["pid"] = child.pid
            write(intent_path, intent)
            generation = handoff.observed(child.pid)
            if generation is None or generation["status"] in ("Z", "X", "x"):
                raise RuntimeError("Owned handoff controller exited before acknowledgment; preserve intent")
            intent["start_ticks"] = generation["start_ticks"]
            write(intent_path, intent)
            marker = f"{CONTROLLER_MARKER}={intent['marker']}".encode()
            environment = (handoff.PROC_ROOT / str(child.pid) / "environ").read_bytes().split(b"\0")
            if (
                generation["command"] != command
                or generation["group"] != child.pid
                or generation["session"] != child.pid
                or marker not in environment
                or f"CUDA_VISIBLE_DEVICES={MASK}".encode() not in environment
                or not handoff.same_generation(generation, handoff.observed(child.pid))
            ):
                raise RuntimeError("Owned handoff controller generation, command, marker or mask differs")
            intent.update(status="acknowledged", generation=generation)
            write(intent_path, intent)
            return intent
        except BaseException as error:
            intent.update(status="unacknowledged", error=f"{type(error).__name__}: {error}")
            write(intent_path, intent)
            raise


def launch(root: Path) -> dict:
    """Reconstruct completed prerequisites before starting the one frozen GPU validator."""
    root = root.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != MASK:
        raise RuntimeError("Owned handoff launch requires the explicit eight-device mask before intent")
    frozen = binding(root)
    if local_node() != NODE:
        raise RuntimeError("Owned handoff launch requires its original .139 allocation")
    predecessor = require_completed_ceiling(root, frozen)
    return start_controller(root, frozen, predecessor)


def validate(root: Path) -> dict:
    """Require completed ceiling recovery before the only authorized allocation transfer."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != MASK:
        raise RuntimeError("Owned-output validation requires the explicit eight-device mask before allocation transfer")
    frozen = binding(root)
    if local_node() != NODE:
        raise RuntimeError("Owned-output validation requires its original .139 allocation")
    _, _, state_path = locations(root)
    work = root / "validation/owned_output_handoff"
    receipt = root / "run-preparation/validation-owned-output-handoff-gpu.json"
    log = root / "run-preparation/validation-owned-output-handoff-gpu.log"
    if any(path.exists() for path in (state_path, work, receipt, log)):
        raise FileExistsError("Owned-output validation has evidence; preserve it without retry")
    with node_guard(root, NODE) as signals:
        predecessor = require_completed_ceiling(root, frozen)
        idle_training()
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        if len(devices) != 8 or any("A100-SXM4-40GB" not in name for name in devices):
            raise ValueError("Owned-output validation requires the original eight A100 GPUs")
        state = {
            "status": "preflight",
            "node": NODE,
            "candidate": frozen["candidate"],
            "controller_pid": os.getpid(),
            "controller_start_ticks": process(os.getpid())[1],
            "binding_sha256": sha256(locations(root)[0]),
            "predecessor": predecessor,
            "cuda_visible_devices": MASK,
        }
        write(state_path, state)
        try:
            state["keeper"] = release_keeper(state, state_path)
            state["status"] = "validating"
            command = [
                PYTHON,
                "-m",
                "experiment.validate",
                "--gpu",
                "--data",
                str(root / "data"),
                "--data-receipt",
                str(root / "data/experiment_receipt.json"),
                "--work",
                str(work),
                "--receipt",
                str(receipt),
            ]
            code = bounded_validator(command, Path(frozen["candidate"]), log, state, state_path)
            state["exit_code"] = code
            if (
                type(code) is not int
                or code != 0
                or type(state.get("cleanup_exit_code")) is not int
                or state["cleanup_exit_code"] != 0
                or "cleanup_error" in state
            ):
                raise RuntimeError("Owned-output actual full-model validation failed")
            require_validation(root, frozen)
            binding(root)
            state.update(status="passed", validation_sha256=sha256(receipt))
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if state.get("keeper_release_attempted"):
                try:
                    if "cleanup_error" in state or tagged_processes(state.get("child_marker", "")):
                        raise RuntimeError("Unverified marked-child cleanup prevents keeper restoration")
                    node_action(root, "restore")
                    state["keeper_restore"] = 0
                except Exception as error:
                    state.update(status="needs_attention", keeper_restore=f"refused: {error}")
            if signals:
                state["deferred_signals"] = signals
            write(state_path, state)
        if type(state.get("keeper_restore")) is not int or state["keeper_restore"] != 0:
            raise RuntimeError("Owned-output full-model keeper restoration failed")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cpu", "freeze", "check", "launch", "validate"), required=True)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--controller-cpu-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase in ("cpu", "freeze") and args.cpu_receipt is None:
        parser.error("cpu and freeze require --cpu-receipt")
    if args.phase == "cpu" and args.out is None:
        parser.error("cpu requires --out")
    if args.phase == "freeze" and args.controller_cpu_receipt is None:
        parser.error("freeze requires --controller-cpu-receipt")
    if args.phase == "cpu":
        result = cpu(args.cpu_receipt, args.out)
    elif args.phase == "freeze":
        result = freeze(args.root, args.cpu_receipt, controller_cpu_receipt=args.controller_cpu_receipt)
    elif args.phase == "check":
        result = binding(args.root)
    elif args.phase == "launch":
        result = launch(args.root)
    else:
        result = validate(args.root)
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
