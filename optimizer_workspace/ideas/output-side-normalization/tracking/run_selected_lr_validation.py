# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze and own distinct actual-config Wave-11 validation after completed Wave 10."""

import argparse
import ast
import fcntl
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

from launch_wave5 import node_action, read, sha256
from owned_output_handoff import keeper_observation, observed, same_generation
from prepare_selected_lr_validation import CPU_STATUS, HELPERS, require_archive, require_candidate
from run_diagonal_diagnostic import finish_child, tagged_processes
from selected_lr_validation import (
    CHILD_MARKER,
    CONTROLLER_MARKER,
    ENVIRONMENT,
    MASK,
    NODE,
    PREPARATION,
    PYTHON,
    STATUS,
    controller_command,
    locations,
    plan,
    validation_commands,
    validator_command,
)
from selected_lr_wave10_handoff import probe
from upstream_production import cpu_command
from validate_selected_lr_full_budget import reconstruct
from watch_wave5 import write
from wave7_release import candidate_inventory, idle_training, local_node, node_guard, release_keeper

TESTS = (
    "test_selected_lr_execution_owner.py",
    "test_selected_lr_validation.py",
    "test_selected_lr_validation_controller.py",
)
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)


def helper_names(directory: Path) -> tuple[str, ...]:
    pending = set(
        (*HELPERS, *TESTS, Path(__file__).name, "selected_lr_wave10_handoff.py", "validate_selected_lr_full_budget.py")
    )
    found = set()
    while pending:
        name = pending.pop()
        path = directory / name
        if Path(name).name != name or not path.is_file():
            raise ValueError(f"Selected-LR controller dependency missing: {name}")
        found.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            for module in names:
                dependency = f"{module.split('.')[0]}.py"
                if (directory / dependency).is_file() and dependency not in found:
                    pending.add(dependency)
    return tuple(sorted(found))


def require_cpu(receipt_path: Path) -> tuple[Path, dict]:
    receipt = read(receipt_path)
    candidate = Path(receipt["candidate"])
    prepared = require_candidate(candidate)
    if (
        receipt["status"] != CPU_STATUS
        or receipt["files_sha256"] != prepared["files_sha256"]
        or receipt["preparation_sha256"] != sha256(candidate / PREPARATION)
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["cuda_visible_devices"] != ""
        or receipt["command"] != cpu_command()
        or receipt["cwd"] != str(candidate)
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
        or receipt["helpers_sha256"] != prepared["helpers_sha256"]
        or "error" in receipt
    ):
        raise ValueError("Selected-LR validation candidate CPU evidence differs")
    require_archive(
        Path(receipt["archive"]),
        prepared["files_sha256"]
        | {PREPARATION: receipt["preparation_sha256"]}
        | {f"tracking/{name}": digest for name, digest in prepared["helpers_sha256"].items()},
        alias=True,
    )
    return candidate, receipt


def controller_cpu_command() -> list[str]:
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
        SAFE_MARKERS,
        *(f"tracking/{name}" for name in TESTS),
    ]


def controller_cpu(receipt_path: Path, out: Path) -> dict:
    candidate, receipt = require_cpu(receipt_path)
    if out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Selected-LR controller CPU evidence requires an unused directory")
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    names = helper_names(Path(__file__).parent)
    for name in names:
        shutil.copy2(Path(__file__).parent / name, helpers / name)
    hashes = {name: sha256(helpers / name) for name in names}
    log, archive = out / "controller.log", out / "controller-source.tar"
    command = controller_cpu_command()
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                command,
                cwd=out,
                env=os.environ
                | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers)))},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Controller CPU timeout at 600 seconds"
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in names:
                bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        require_archive(archive, {f"tracking/{name}": digest for name, digest in hashes.items()}, alias=False)
        if hashes != {name: sha256(helpers / name) for name in names} or require_cpu(receipt_path)[1] != receipt:
            raise ValueError("Selected-LR controller or candidate changed during CPU verification")
    except (ValueError, OSError, tarfile.TarError, subprocess.SubprocessError) as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": (
            "selected_lr_controller_cpu_verified_no_gpu_binding" if code == 0 and error is None else "needs_attention"
        ),
        "candidate_cpu_receipt": str(receipt_path.resolve()),
        "candidate_cpu_receipt_sha256": sha256(receipt_path),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "command": command,
        "exit_code": code,
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive) if archive.is_file() else None,
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] == "needs_attention":
        raise RuntimeError(f"Selected-LR controller CPU evidence failed at {out}")
    return result


def require_controller_cpu(receipt_path: Path, candidate_receipt: Path) -> dict:
    receipt = read(receipt_path)
    helpers = Path(receipt["helpers"])
    names = helper_names(helpers)
    if (
        receipt["status"] != "selected_lr_controller_cpu_verified_no_gpu_binding"
        or receipt["candidate_cpu_receipt"] != str(candidate_receipt.resolve())
        or receipt["candidate_cpu_receipt_sha256"] != sha256(candidate_receipt)
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["cuda_visible_devices"] != ""
        or receipt["command"] != controller_cpu_command()
        or receipt["helpers_sha256"] != {name: sha256(helpers / name) for name in names}
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
        or "error" in receipt
    ):
        raise ValueError("Selected-LR controller CPU archive or command evidence changed")
    require_archive(
        Path(receipt["archive"]),
        {f"tracking/{name}": digest for name, digest in receipt["helpers_sha256"].items()},
        alias=False,
    )
    return receipt


def freeze(root: Path, cpu_receipt: Path, controller_receipt: Path) -> dict:
    candidate, cpu = require_cpu(cpu_receipt)
    controller = require_controller_cpu(controller_receipt, cpu_receipt)
    binding_path, helpers, _, work, receipt = locations(root)
    archive = root / "run-preparation/source-wave11-selected-lr-validation-20261117.tar"
    if any(path.exists() for path in (binding_path, helpers, archive)):
        raise FileExistsError("Selected-LR validation identity already exists; preserve it")
    source_helpers = Path(__file__).parent
    names = helper_names(source_helpers)
    hashes = {name: sha256(source_helpers / name) for name in names}
    if hashes != controller["helpers_sha256"]:
        raise ValueError("Selected-LR current controller differs from its CPU-verified archive")
    helpers.mkdir(parents=True)
    with tarfile.open(archive, "x") as bundle:
        for name in names:
            shutil.copy2(source_helpers / name, helpers / name)
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        for name in cpu["files_sha256"]:
            bundle.add(candidate / name, arcname=name, recursive=False)
        for path, name in (
            (candidate / PREPARATION, PREPARATION),
            (cpu_receipt, "candidate-cpu-receipt.json"),
            (controller_receipt, "controller-cpu-receipt.json"),
        ):
            bundle.add(path, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    result = {
        "status": STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "node": NODE,
        "python": PYTHON,
        "environment": ENVIRONMENT,
        "files_sha256": cpu["files_sha256"],
        "preparation_sha256": sha256(candidate / PREPARATION),
        "cpu_receipt": str(cpu_receipt.resolve()),
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "controller_cpu_receipt": str(controller_receipt.resolve()),
        "controller_cpu_receipt_sha256": sha256(controller_receipt),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "predecessor_release_sha256": sha256(root / "run-preparation/wave10-launch-binding.json"),
        "commands": validation_commands(PYTHON, candidate, root, work),
        "plan": plan(candidate, root, work),
        "work": str(work),
        "receipt": str(receipt),
        "policy": (
            "Validation only; completed Wave10/all-four exact keeper/cloud acceptance precedes GPU transfer. "
            "No queue or production release."
        ),
    }
    expected = (
        cpu["files_sha256"]
        | {
            PREPARATION: result["preparation_sha256"],
            "candidate-cpu-receipt.json": result["cpu_receipt_sha256"],
            "controller-cpu-receipt.json": result["controller_cpu_receipt_sha256"],
        }
        | {f"tracking/{name}": digest for name, digest in hashes.items()}
    )
    require_archive(archive, expected, alias=True)
    if require_cpu(cpu_receipt)[1] != cpu or require_controller_cpu(controller_receipt, cpu_receipt) != controller:
        raise ValueError("Selected-LR acceptance changed during freeze")
    write(binding_path, result)
    return result


def binding(root: Path) -> dict:
    binding_path, helpers, _, work, receipt = locations(root)
    frozen = read(binding_path)
    candidate, cpu = require_cpu(Path(frozen["cpu_receipt"]))
    controller = require_controller_cpu(Path(frozen["controller_cpu_receipt"]), Path(frozen["cpu_receipt"]))
    if (
        Path(__file__).resolve() != helpers / "run_selected_lr_validation.py"
        or frozen["status"] != STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(candidate)
        or frozen["node"] != NODE
        or frozen["python"] != PYTHON
        or frozen["environment"] != ENVIRONMENT
        or frozen["files_sha256"] != cpu["files_sha256"]
        or frozen["files_sha256"] != candidate_inventory(candidate)
        or frozen["helpers"] != str(helpers)
        or frozen["helpers_sha256"] != controller["helpers_sha256"]
        or frozen["helpers_sha256"] != {name: sha256(helpers / name) for name in helper_names(helpers)}
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["preparation_sha256"] != sha256(candidate / PREPARATION)
        or frozen["cpu_receipt_sha256"] != sha256(Path(frozen["cpu_receipt"]))
        or frozen["controller_cpu_receipt_sha256"] != sha256(Path(frozen["controller_cpu_receipt"]))
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or frozen["predecessor_release_sha256"] != sha256(root / "run-preparation/wave10-launch-binding.json")
        or frozen["commands"] != validation_commands(PYTHON, candidate, root, work)
        or frozen["plan"] != plan(candidate, root, work)
        or frozen["work"] != str(work)
        or frozen["receipt"] != str(receipt)
    ):
        raise ValueError("Selected-LR validation frozen identity or actual-config plan differs")
    return frozen


def require_completed_wave10(root: Path) -> dict:
    """Reconstruct the completed broad-window result and each imported scalar history."""
    path = root / "run-preparation/wave10-launch-binding.json"
    release = read(path)
    helpers = Path(release["helpers"])
    script = (
        "import json,sys\nfrom pathlib import Path\nfrom owned_output_production import snapshot\n"
        "from analyze_owned_output_run import analyze\nfrom owned_output_run_history import binding,completed_run\n"
        "root=Path(sys.argv[1]); release=binding(root); observed=snapshot(root)\n"
        "print(json.dumps({'snapshot':observed,'analysis':analyze(root),"
        "'histories':{arm:completed_run(root,arm,release) for arm in release['arms']}},allow_nan=False))"
    )
    result = subprocess.run(
        [PYTHON, "-P", "-c", script, str(root)],
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        check=True,
        timeout=1500,
    )
    actual = json.loads(result.stdout.splitlines()[-1])
    snapshot = actual["snapshot"]
    saved, cloud = read(root / "run-preparation/analysis-wave10.json"), read(
        root / "run-preparation/swanlab-upload-wave10.json"
    )
    if (
        snapshot["status"] != "complete"
        or snapshot["all_workers_terminal"] is not True
        or snapshot["binding_sha256"] != sha256(path)
        or actual["analysis"] != saved
        or cloud["status"] != "verified"
        or cloud["visibility"] != "PRIVATE"
        or cloud["project"] != "siyuc3141/output-side-normalization-marin"
        or set(cloud["runs"]) != set(release["arms"])
        or len(snapshot["workers"]) != 4
    ):
        raise RuntimeError(
            "Wave10 complete original workers, broad-window analysis or cloud verification remain pending"
        )
    for analysis in saved["analyses"]:
        for item in analysis["comparisons"].values():
            if not item["final_comparison_eligible"] or len(item["training_windows"]) != 5:
                raise ValueError("Wave10 final comparison lacks five complete large windows")
    columns_script = (
        "import json,sys\nfrom pathlib import Path\nfrom swanlab_refresh_upload import load_history,expected_columns\n"
        "root=Path(sys.argv[1]); arms=json.loads(sys.argv[2]); print(json.dumps("
        "{arm:{'point_counts':{k:len(v) for k,v in expected_columns(load_history(root,'wave10',arm)).items()},"
        "'final_metrics':load_history(root,'wave10',arm).metrics[-1]} for arm in arms},allow_nan=False))"
    )
    columns = json.loads(
        subprocess.run(
            [str(root / ".venv-logging/bin/python"), "-P", "-c", columns_script, str(root), json.dumps(release["arms"])],
            cwd=helpers,
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(helpers)},
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        ).stdout
    )
    for worker in snapshot["workers"]:
        arm, state = worker["arm"], worker["receipt"]
        imported, local = cloud["runs"][arm], actual["histories"][arm]
        verification = imported["verification"]
        if (
            worker["status"] != "complete"
            or state["status"] != "complete"
            or state["completed_run"] != local
            or state["command"] != release["commands"][state["node"]]
            or state["release_binding_sha256"] != snapshot["binding_sha256"]
            or any(
                type(state.get(key)) is not int or state[key] != 0
                for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
            )
            or any(key in state for key in ("error", "cleanup_error"))
            or imported["status"] != "verified"
            or any(
                imported[key] != local[key]
                for key in ("manifest_sha256", "metrics_sha256", "checkpoint_receipt_sha256", "checkpoint_rank_bytes")
            )
            or verification["state"] != "FINISHED"
            or verification["configuration_verified"] is not True
            or verification["token_axis_verified"] is not True
            or verification["integer_values_exact"] is not True
            or verification["csv_export_max_ulp"] != 2
            or verification["point_counts"] != columns[arm]["point_counts"]
            or verification["final_metrics"] != columns[arm]["final_metrics"]
            or set(verification["serialization_errors"]) != set(verification["point_counts"])
            or any(
                not math.isfinite(error[field]) or error[field] < 0 or (field == "max_ulp" and error[field] > 2)
                for error in verification["serialization_errors"].values()
                for field in ("max_absolute", "max_ulp")
            )
        ):
            raise ValueError("Wave10 full-resolution cloud, checkpoint or original worker identity differs")
    handoffs = []
    for arm, node in zip(release["arms"], release["nodes"], strict=True):
        command = [
            PYTHON,
            str(Path(__file__).parent / "selected_lr_wave10_handoff.py"),
            "--root",
            str(root),
            "--arm",
            arm,
        ]
        observed_handoff = json.loads(
            subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    f"root@{node}",
                    shlex.join(["env", "CUDA_VISIBLE_DEVICES=", *command]),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            ).stdout.splitlines()[-1]
        )
        if (
            observed_handoff["status"] != "wave10_original_training_absent_exact_keeper_verified"
            or observed_handoff["arm"] != arm
            or observed_handoff["node"] != node
            or observed_handoff["original_handles_absent"] is not True
            or observed_handoff["training_child_markers_absent"] is not True
        ):
            raise RuntimeError("Wave10 original process absence or exact restored keeper is not established")
        handoffs.append(observed_handoff)
    return {
        "status": "wave10_complete_original_workers_absent_restored_analyzed_cloud_verified",
        "release_sha256": sha256(path),
        "analysis_sha256": sha256(root / "run-preparation/analysis-wave10.json"),
        "cloud_sha256": sha256(root / "run-preparation/swanlab-upload-wave10.json"),
        "handoffs": handoffs,
    }


def start(root: Path) -> dict:
    binding(root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != MASK:
        raise RuntimeError("Selected-LR validation requires the explicit eight-device mask")
    predecessor = require_completed_wave10(root)
    _, helpers, state_path, work, receipt = locations(root)
    intent_path, log = helpers.parent / "controller.json", helpers.parent / "controller.log"
    with (helpers.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.exists() for path in (intent_path, log, state_path, work, receipt)):
            raise FileExistsError("Selected-LR validator already has intent or evidence; no retry")
        intent = {
            "status": "starting",
            "binding_sha256": sha256(locations(root)[0]),
            "command": controller_command(root),
            "marker": uuid4().hex,
            "predecessor": predecessor,
            "log": str(log),
        }
        write(intent_path, intent)
        with log.open("x") as stream:
            child = subprocess.Popen(
                intent["command"],
                cwd=helpers,
                env=os.environ | ENVIRONMENT | {CONTROLLER_MARKER: intent["marker"]},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        generation = observed(child.pid)
        if (
            generation is None
            or generation["command"] != intent["command"]
            or generation["group"] != child.pid
            or generation["session"] != child.pid
        ):
            raise RuntimeError("Selected-LR controller startup generation unknown; preserve original handle")
        intent.update(status="acknowledged", pid=child.pid, start_ticks=generation["start_ticks"], generation=generation)
        write(intent_path, intent)
    return intent


def bounded_validator(root: Path, frozen: dict, state: dict) -> int:
    _, helpers, state_path, _, _ = locations(root)
    candidate, command = Path(frozen["candidate"]), validator_command(root, Path(frozen["candidate"]))
    marker, generation = uuid4().hex, {}
    log = helpers.parent / "validator.log"
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=candidate,
            env=os.environ
            | ENVIRONMENT
            | {
                CHILD_MARKER: marker,
                "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers))),
            },
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process = observed(child.pid)
            if process is None or process["command"] != command:
                raise RuntimeError("Selected-LR validator generation unknown")
            state.update(
                status="validating",
                spawn_status="acknowledged",
                command=command,
                child_pid=child.pid,
                child_start_ticks=process["start_ticks"],
                child_marker=marker,
                log=str(log),
            )
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
            except BaseException as error:
                state["cleanup_error"] = str(error)
                raise
            finally:
                state.update(child_generation=generation, log_sha256=sha256(log))
                write(state_path, state)


def validate(root: Path) -> dict:
    frozen = binding(root)
    _, helpers, state_path, _, receipt = locations(root)
    intent_path = helpers.parent / "controller.json"
    deadline = time.monotonic() + 30
    intent = read(intent_path)
    while intent.get("status") == "starting" and time.monotonic() < deadline:
        time.sleep(0.05)
        intent = read(intent_path)
    own = observed(os.getpid())
    if (
        own is None
        or intent["status"] != "acknowledged"
        or intent["pid"] != os.getpid()
        or intent["start_ticks"] != own["start_ticks"]
        or own["command"] != controller_command(root)
        or os.environ.get(CONTROLLER_MARKER) != intent["marker"]
        or os.environ.get("CUDA_VISIBLE_DEVICES") != MASK
        or local_node() != NODE
        or state_path.exists()
    ):
        raise RuntimeError("Selected-LR validator has no original controller ownership")
    candidate = Path(frozen["candidate"])
    state = {
        "status": "preflight",
        "binding_sha256": sha256(locations(root)[0]),
        "controller_pid": os.getpid(),
        "controller_start_ticks": own["start_ticks"],
        "node": NODE,
        "candidate": str(candidate),
    }
    write(state_path, state)
    signals = []
    try:
        with node_guard(root, NODE) as signals:
            state["predecessor"] = require_completed_wave10(root)
            idle_training()
            original_arm = "muont"
            accepted_keeper = probe(root, original_arm)["keeper"]
            subprocess.run(
                [PYTHON, "-m", "experiment.bootstrap", "--check-only"],
                cwd=candidate,
                env=os.environ | ENVIRONMENT,
                check=True,
                capture_output=True,
                text=True,
                timeout=240,
            )
            current = probe(root, original_arm)["keeper"]
            if set(current["generation"]) != set(accepted_keeper["generation"]) or any(
                not same_generation(process, current["generation"][pid])
                for pid, process in accepted_keeper["generation"].items()
            ):
                raise RuntimeError("Selected-LR predecessor keeper changed before release")
            try:
                state["keeper"] = release_keeper(state, state_path)
                code = bounded_validator(root, frozen, state)
                state["exit_code"] = code
                if type(code) is not int or code != 0:
                    raise RuntimeError("Selected-LR actual-config validator failed")
                actual = reconstruct(root, candidate, receipt)
                if actual["owner"] != {"controller_pid": os.getpid(), "controller_start_ticks": own["start_ticks"]}:
                    raise ValueError("Selected-LR receipt belongs to a different controller generation")
                state.update(status="passed", validation_sha256=sha256(receipt))
            finally:
                if state.get("keeper_release_attempted"):
                    try:
                        if "cleanup_error" in state or tagged_processes(state.get("child_marker", "")):
                            raise RuntimeError("Selected-LR marked descendants prevent keeper restoration")
                        node_action(root, "restore")
                        state.update(restored_keeper=keeper_observation(), keeper_restore=0)
                    except BaseException as error:
                        state.update(status="needs_attention", keeper_restore=f"refused: {error}")
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if signals:
            state["deferred_signals"] = signals
        write(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cpu", "freeze", "check", "start", "validate"), required=True)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--controller-cpu-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase in ("cpu", "freeze") and args.cpu_receipt is None:
        parser.error("cpu/freeze require --cpu-receipt")
    if args.phase == "cpu" and args.out is None:
        parser.error("cpu requires --out")
    if args.phase == "freeze" and args.controller_cpu_receipt is None:
        parser.error("freeze requires --controller-cpu-receipt")
    root = args.root.resolve()
    if args.phase == "cpu":
        result = controller_cpu(args.cpu_receipt, args.out)
    elif args.phase == "freeze":
        result = freeze(root, args.cpu_receipt, args.controller_cpu_receipt)
    elif args.phase == "check":
        result = binding(root)
    elif args.phase == "start":
        result = start(root)
    else:
        result = validate(root)
    print(json.dumps({"status": result["status"]}, allow_nan=False))


if __name__ == "__main__":
    main()
