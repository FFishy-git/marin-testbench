# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze, launch and observe the twelve equal-budget fixed-age LR trials."""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

import boundary_handoff
from analyze_tuning import executed_sources, normalized_sources
from compare_runs import completion, history
from fixed_age_protocol import environment, is_boundary, multipliers, predecessor_wave, require_gpu_environment
from launch_wave5 import NODES, PYTHON, node_action, read, sha256
from prepare_fixed_age_tuning import require_source
from prepare_tuning import TRIAL_TOKENS, trial_config
from run_diagonal_diagnostic import tagged_processes
from run_fixed_age_validation import VALIDATION_HELPERS, require_actual_receipt, require_completed_wave
from watch_wave5 import write
from watch_wave7_release import observed_process
from wave7_release import ARMS, bounded_execution, defer_signals, idle_training, local_node, node_guard, release_keeper

EXTRA_HELPERS = ("run_fixed_age_tuning.py", "swanlab_fixed_age_tuning.py", "monitor_fixed_age_tuning.sh")
HELPERS = (*VALIDATION_HELPERS, *EXTRA_HELPERS)
WORKER_MARKER = "OUTPUT_MARIN_TUNING_WORKER_ID"
TRIAL_TIMEOUT = 7200


def plan(campaign_path: Path) -> list[dict]:
    """Assign one family per original node and retain three complete short schedules."""
    campaign = read(campaign_path)
    root, source = Path(campaign["root"]), Path(campaign["source"])
    numerical = require_source(root, Path(campaign["source_receipt"]))
    grid = multipliers(campaign_path)
    protocol = {(family, label): multiplier for family in ARMS for label, multiplier in grid}
    if (
        campaign["source_receipt_sha256"] != sha256(Path(campaign["source_receipt"]))
        or campaign["source"] != numerical["candidate"]
        or campaign["data_receipt_sha256"] != numerical["data_receipt_sha256"]
        or campaign["families"] != list(ARMS)
        or campaign["batch"] != "0p5m"
        or campaign["seed"] != 0
        or campaign["initial_tokens_per_family"] != len(grid) * TRIAL_TOKENS
        or campaign["initial_trials_per_family"] != len(grid)
        or campaign["selection"]["training_window_tokens"] != TRIAL_TOKENS // 5
        or [trial["name"] for trial in campaign["trials"]]
        != [f"{family}_{label}" for label, _ in grid for family in ARMS]
    ):
        raise ValueError("Execution campaign differs from the fixed equal-budget LR protocol")
    for trial in campaign["trials"]:
        family = trial["family"]
        label = trial["name"].removeprefix(f"{family}_")
        multiplier = protocol[(family, label)]
        reference_path = source / f"experiment/configs/{family}_0p5m_seed0.json"
        reference = read(reference_path)
        expected = trial_config(reference, family, "0p5m", label, multiplier)
        config_path = campaign_path.parent / "configs" / f"{trial['name']}.json"
        if (
            trial["config"] != str(config_path)
            or trial["config_sha256"] != sha256(config_path)
            or read(config_path) != expected
            or trial["reference_config"] != str(reference_path)
            or trial["reference_config_sha256"] != sha256(reference_path)
            or trial["tokens"] != TRIAL_TOKENS
            or trial["steps"] != 320
            or trial["lr_multiplier"] != multiplier
            or trial["matrix_lr"] != expected["optimizer"]["lr"]
            or trial["batch"] != "0p5m"
            or trial["seed"] != 0
            or campaign["selection"]["weights"] != reference["data"]["mixture_weights"]
        ):
            raise ValueError("Execution trial differs from its exact prepared configuration")
    result = []
    for family, node in zip(ARMS, NODES, strict=True):
        trials = []
        for trial in campaign["trials"]:
            if trial["family"] != family:
                continue
            out = campaign_path.parent / "runs" / trial["name"]
            trials.append(
                {
                    "name": trial["name"],
                    "config_sha256": trial["config_sha256"],
                    "tokens": trial["tokens"],
                    "out": str(out),
                    "command": [
                        PYTHON,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node=8",
                        "-m",
                        "experiment.train",
                        "--config",
                        trial["config"],
                        "--data",
                        str(Path(campaign["root"]) / "data"),
                        "--data-receipt",
                        str(Path(campaign["root"]) / "data/experiment_receipt.json"),
                        "--out",
                        str(out),
                    ],
                }
            )
        result.append({"family": family, "node": node, "trials": trials})
    return result


def validation_binding(campaign_path: Path) -> dict:
    """Check the original frozen validator without invoking CUDA or changing its evidence."""
    path = campaign_path.parent / "validation-binding.json"
    frozen = read(path)
    if (
        frozen["campaign"] != str(campaign_path)
        or frozen["campaign_sha256"] != sha256(campaign_path)
        or set(frozen["helpers_sha256"]) != set(VALIDATION_HELPERS)
        or any(sha256(Path(frozen["helpers"]) / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
    ):
        raise ValueError("Actual-config validation campaign, helper or archive identity changed")
    return frozen


def freeze(campaign_path: Path) -> dict:
    """Archive execution helpers separately, even while predecessor training is running."""
    campaign_path = campaign_path.resolve()
    campaign = read(campaign_path)
    require_source(Path(campaign["root"]), Path(campaign["source_receipt"]))
    validation = validation_binding(campaign_path)
    subprocess.run(
        [
            PYTHON,
            str(Path(validation["helpers"]) / "run_fixed_age_validation.py"),
            "--campaign",
            str(campaign_path),
            "--phase",
            "check",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    execution_plan = plan(campaign_path)
    directory = campaign_path.parent
    helpers = directory / "execution-helpers"
    archive = directory / "execution-source.tar"
    path = directory / "execution-binding.json"
    if any(item.exists() for item in (helpers, archive, path)):
        raise FileExistsError("Execution binding already exists; preserve its exact evidence")
    for name in VALIDATION_HELPERS:
        if sha256(Path(__file__).parent / name) != validation["helpers_sha256"][name]:
            raise ValueError("Execution must retain the exact already frozen numerical and validation helpers")
    helpers.mkdir()
    with tarfile.open(archive, "x") as stream:
        for name in HELPERS:
            destination = helpers / name
            destination.write_bytes((Path(__file__).parent / name).read_bytes())
            stream.add(destination, arcname=f"tracking/{name}", recursive=False)
        stream.add(campaign_path, arcname="campaign.json", recursive=False)
        for trial in campaign["trials"]:
            stream.add(trial["config"], arcname=f"configs/{trial['name']}.json", recursive=False)
    result = {
        "status": (
            f"frozen_pending_completed_{predecessor_wave(campaign_path)}_and_actual_gpu_validation"
            if is_boundary(campaign_path)
            else "frozen_pending_completed_wave7_and_actual_gpu_validation"
        ),
        "campaign": str(campaign_path),
        "campaign_sha256": sha256(campaign_path),
        "validation_binding_sha256": sha256(directory / "validation-binding.json"),
        "source_release_sha256": campaign["source_receipt_sha256"],
        "helpers": str(helpers),
        "helpers_sha256": {name: sha256(helpers / name) for name in HELPERS},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "plan": execution_plan,
        "environment": environment(campaign_path),
        "trial_timeout_seconds": TRIAL_TIMEOUT,
        "monitor_interval_seconds": 900,
        "policy": (
            "No retry or backend substitution. Complete predeclared schedules per family. Failed tokens remain charged. "
            "Completed mixture loss selects LR; C4 is reporting only. No GPU job or queue is created by freezing."
        ),
    }
    write(path, result)
    return result


def binding(campaign_path: Path) -> dict:
    """Require the separately archived execution copy and unchanged original validator."""
    campaign_path = campaign_path.resolve()
    path = campaign_path.parent / "execution-binding.json"
    frozen = read(path)
    helpers = Path(frozen["helpers"])
    if (
        Path(__file__).resolve() != (helpers / "run_fixed_age_tuning.py").resolve()
        or frozen["campaign"] != str(campaign_path)
        or frozen["campaign_sha256"] != sha256(campaign_path)
        or frozen["validation_binding_sha256"] != sha256(campaign_path.parent / "validation-binding.json")
        or set(frozen["helpers_sha256"]) != set(HELPERS)
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["environment"] != environment(campaign_path)
        or frozen["trial_timeout_seconds"] != TRIAL_TIMEOUT
        or frozen["monitor_interval_seconds"] != 900
    ):
        raise ValueError("Fixed-age execution helper, archive, campaign or validator identity changed")
    campaign = read(campaign_path)
    source = require_source(Path(campaign["root"]), Path(campaign["source_receipt"]))
    if frozen["source_release_sha256"] != sha256(Path(campaign["source_receipt"])):
        raise ValueError("Execution source release changed")
    validation = validation_binding(campaign_path)
    if any(frozen["helpers_sha256"][name] != validation["helpers_sha256"][name] for name in VALIDATION_HELPERS):
        raise ValueError("Execution numerical helpers differ from actual-config validation")
    if source["candidate"] != campaign["source"] or frozen["plan"] != plan(campaign_path):
        raise ValueError("Execution trial plan or source differs from the equal-budget campaign")
    return frozen


def require_ready(campaign_path: Path) -> dict:
    """Accept only actual twelve-config validation, cleanup/restoration and completed Wave 7."""
    frozen = binding(campaign_path)
    directory = campaign_path.parent
    state = read(directory / "validation-state.json")
    receipt = directory / "validation-gpu.json"
    if (
        state["status"] != "passed"
        or type(state.get("keeper_restore")) is not int
        or state["keeper_restore"] != 0
        or state.get("cleanup_exit_code") != 0
        or "cleanup_error" in state
        or state["binding_sha256"] != frozen["validation_binding_sha256"]
        or state["validation_sha256"] != sha256(receipt)
    ):
        raise ValueError("Actual twelve-config GPU validation and keeper restoration have not passed")
    require_actual_receipt(read(directory / "validation-binding.json"))
    require_completed_wave(Path(read(campaign_path)["root"]), campaign_path=campaign_path)
    return frozen


def worker_command(campaign_path: Path, frozen: dict, family: str) -> list[str]:
    return [
        PYTHON,
        str(Path(frozen["helpers"]) / "run_fixed_age_tuning.py"),
        "--campaign",
        str(campaign_path),
        "--phase",
        "worker",
        "--family",
        family,
    ]


def completed_trial(campaign_path: Path, name: str) -> dict:
    """Validate one finished trial without reading any concurrently appended sibling history."""
    campaign = read(campaign_path)
    trial = next(item for item in campaign["trials"] if item["name"] == name)
    source = Path(campaign["source"])
    numerical = require_source(Path(campaign["root"]), Path(campaign["source_receipt"]))
    directory = campaign_path.parent / "runs" / name
    run = history(directory)
    if (
        run.manifest["config"] != read(Path(trial["config"]))
        or run.manifest["data_receipt_sha256"] != campaign["data_receipt_sha256"]
        or normalized_sources(source, run.manifest["source_sha256"]) != executed_sources(source, numerical)
        or completion(run)["complete"] is not True
    ):
        raise ValueError("Short-LR trial is incomplete or differs from its executed source/config/data")
    checkpoint = directory / f"checkpoints/step_{trial['steps']:06d}"
    receipt = read(checkpoint / "complete.json")
    sizes = [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)]
    final = run.rows[-1]
    weights = campaign["selection"]["weights"]
    recomputed = sum(weight * final["heldout_losses"][f"heldout/{family}"] for family, weight in weights.items())
    recomputed /= sum(weights.values())
    if (
        receipt["step"] != 320
        or receipt["world"] != 8
        or not all(size > 0 for size in sizes)
        or recomputed != final["selection_loss"]
        or "c4_loss" not in final
    ):
        raise ValueError("Short-LR final checkpoint or heldout-mixture endpoint is incomplete")
    return {"name": name, "observed_tokens": final["tokens"], "complete": True}


def allocation(campaign_path: Path, frozen: dict, family: str) -> dict:
    assignment = next(item for item in frozen["plan"] if item["family"] == family)
    if local_node() != assignment["node"]:
        raise RuntimeError("Family worker is on the wrong original A100 allocation")
    root = Path(read(campaign_path)["root"])
    observation = campaign_path.parent / "workers" / f"{family}.preflight.json"
    if is_boundary(campaign_path):
        if predecessor_wave(campaign_path) == "wave9":
            boundary_handoff.snapshot(root, observation, wave="wave9")
        else:
            boundary_handoff.snapshot(root, observation)
    else:
        source = read(root / "run-preparation/wave7-release-binding.json")
        subprocess.run(
            [
                PYTHON,
                str(Path(source["helpers"]) / "snapshot_wave7_release.py"),
                "--root",
                str(root),
                "--out",
                str(observation),
            ],
            check=True,
            timeout=240,
        )
        if read(observation)["status"] != "complete":
            raise RuntimeError("Wave-7 parents or keeper restoration are still live, unknown or incomplete")
    idle_training()
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(devices) != 8 or any("A100-SXM4-40GB" not in device for device in devices):
        raise ValueError("Short-LR study requires the original eight A100 40GB GPUs per family")
    subprocess.run(
        [PYTHON, "-c", "import torch; assert torch.cuda.device_count() == 8, 'Expected eight visible CUDA GPUs'"],
        env=os.environ | frozen["environment"],
        check=True,
        timeout=60,
    )
    subprocess.run([PYTHON, "-m", "experiment.bootstrap", "--check-only"], cwd=read(campaign_path)["source"], check=True)
    return assignment


def started_worker(campaign_path: Path, frozen: dict, family: str) -> tuple[str, str, list[str]]:
    """Verify the persisted spawn generation and unique marker, including the startup-write race."""
    path = campaign_path.parent / "workers" / f"{family}.start.json"
    deadline = time.monotonic() + 30
    intent = read(path)
    while "pid" not in intent and time.monotonic() < deadline:
        time.sleep(0.1)
        intent = read(path)
    own = observed_process(os.getpid())
    launch = read(campaign_path.parent / "launch-state.json")
    if (
        own is None
        or intent.get("pid") != os.getpid()
        or intent.get("start_ticks") != own[1]
        or intent.get("command") != worker_command(campaign_path, frozen, family)
        or own[2] != intent["command"]
        or intent.get("binding_sha256") != sha256(campaign_path.parent / "execution-binding.json")
        or intent.get("worker_marker") != launch["worker_markers"][family]
        or os.environ.get(WORKER_MARKER) != intent["worker_marker"]
    ):
        raise RuntimeError("Worker generation differs from its persisted bound spawn acknowledgment")
    return own


def run_family(campaign_path: Path, family: str) -> dict:
    """Hold one node lock through three trials and bounded cleanup, then restore its keeper."""
    require_gpu_environment(campaign_path)
    frozen = require_ready(campaign_path)
    root = Path(read(campaign_path)["root"])
    launch = read(campaign_path.parent / "launch-state.json")
    marker = launch["worker_markers"][family]
    if (
        launch["binding_sha256"] != sha256(campaign_path.parent / "execution-binding.json")
        or launch["status"] not in ("launching", "launched", "needs_attention")
        or os.environ.get(WORKER_MARKER) != marker
    ):
        raise RuntimeError("Family worker must be started by its bound launch generation")
    directory = campaign_path.parent / "workers"
    state_path = directory / f"{family}.json"
    if state_path.exists():
        raise FileExistsError("Family already has execution evidence; no automatic retry")
    with node_guard(root, NODES[ARMS.index(family)]) as signals:
        assignment = allocation(campaign_path, frozen, family)
        own = started_worker(campaign_path, frozen, family)
        state = {
            "status": "preflight",
            "family": family,
            "node": assignment["node"],
            "pid": os.getpid(),
            "start_ticks": own[1],
            "command": own[2],
            "worker_marker": marker,
            "binding_sha256": sha256(campaign_path.parent / "execution-binding.json"),
            "trials": [],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        write(state_path, state)
        try:
            state["keeper"] = release_keeper(state, state_path)
            state["status"] = "running"
            write(state_path, state)
            for trial in assignment["trials"]:
                if signals:
                    raise InterruptedError("Worker interrupted before the next trial")
                require_ready(campaign_path)
                if Path(trial["out"]).exists():
                    raise FileExistsError("Trial output already exists; preserve it without retry")
                log = directory / f"{trial['name']}.torchrun.log"
                execution = {"name": trial["name"], "status": "running", "command": trial["command"], "log": str(log)}
                state["trials"].append(execution)
                write(state_path, state)
                child_state = {"status": "running"}
                child_path = directory / f"{trial['name']}.execution.json"
                try:
                    code = bounded_execution(
                        trial["command"],
                        Path(read(campaign_path)["source"]),
                        log,
                        child_state,
                        child_path,
                        timeout=TRIAL_TIMEOUT,
                    )
                    execution.update(exit_code=code, log_sha256=sha256(log), execution_sha256=sha256(child_path))
                    if code:
                        raise RuntimeError(f"Short-LR trial failed: {trial['name']}")
                    measured = completed_trial(campaign_path, trial["name"])
                    execution.update(status="complete", observed_tokens=measured["observed_tokens"])
                except BaseException:
                    execution["status"] = "needs_attention"
                    if "cleanup_error" in child_state:
                        state["cleanup_error"] = child_state["cleanup_error"]
                    raise
                finally:
                    write(state_path, state)
            state["status"] = "complete"
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if state.get("keeper_release_attempted"):
                try:
                    if "cleanup_error" in state:
                        raise RuntimeError("Unverified trial child cleanup prevents keeper restoration")
                    node_action(root, "restore")
                    state["keeper_restore"] = 0
                except Exception as error:
                    state.update(status="needs_attention", keeper_restore=f"refused: {error}")
            if signals:
                state["deferred_signals"] = signals
            write(state_path, state)
        if type(state.get("keeper_restore")) is not int or state["keeper_restore"] != 0:
            raise RuntimeError("Family keeper restoration failed")
        return state


def start_family(campaign_path: Path, family: str) -> dict:
    """Start a detached worker once; preserve it even when acknowledgment times out."""
    require_gpu_environment(campaign_path)
    frozen = require_ready(campaign_path)
    assignment = next(item for item in frozen["plan"] if item["family"] == family)
    if local_node() != assignment["node"]:
        raise RuntimeError("Worker start requested on the wrong node")
    directory = campaign_path.parent / "workers"
    path = directory / f"{family}.json"
    with (directory / f"{family}.start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent_path = directory / f"{family}.start.json"
        if path.exists() or intent_path.exists():
            raise FileExistsError("Worker has prior launch evidence; inspect it without retry")
        launch = read(campaign_path.parent / "launch-state.json")
        if launch["status"] != "launching" or launch["binding_sha256"] != sha256(
            campaign_path.parent / "execution-binding.json"
        ):
            raise ValueError("Worker launch is not authorized by the bound campaign controller")
        command = worker_command(campaign_path, frozen, family)
        intent = {
            "status": "starting",
            "family": family,
            "node": assignment["node"],
            "command": command,
            "binding_sha256": sha256(campaign_path.parent / "execution-binding.json"),
            "worker_marker": launch["worker_markers"][family],
        }
        write(intent_path, intent)
        try:
            with (directory / f"{family}.log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    cwd=read(campaign_path)["source"],
                    env={**os.environ, **frozen["environment"], WORKER_MARKER: launch["worker_markers"][family]},
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except Exception as error:
            intent.update(status="spawn_failed", error=str(error))
            write(intent_path, intent)
            raise
        observed = observed_process(child.pid)
        intent.update(pid=child.pid, start_ticks=None if observed is None else observed[1])
        write(intent_path, intent)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if path.exists():
                state = read(path)
                if state["pid"] != child.pid or state["start_ticks"] != intent["start_ticks"]:
                    raise RuntimeError("Worker acknowledgment differs from the spawned generation")
                return state
            if child.poll() is not None:
                raise RuntimeError("Family worker exited before its startup acknowledgment; inspect its log")
            time.sleep(0.1)
        raise TimeoutError("Worker startup observation timed out; preserve its PID without retry")


def remote(campaign_path: Path, frozen: dict, node: str, phase: str, family: str) -> dict:
    command = [
        PYTHON,
        str(Path(frozen["helpers"]) / "run_fixed_age_tuning.py"),
        "--campaign",
        str(campaign_path),
        "--phase",
        phase,
        "--family",
        family,
    ]
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            f"root@{node}",
            shlex.join(["env", *(f"{key}={value}" for key, value in frozen["environment"].items()), *command]),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode:
        raise RuntimeError(f"{node} {phase}: {result.stderr[-4000:]}")
    return json.loads(result.stdout.splitlines()[-1])


def probe_worker(campaign_path: Path, family: str) -> dict:
    frozen = binding(campaign_path)
    assignment = next(item for item in frozen["plan"] if item["family"] == family)
    if local_node() != assignment["node"]:
        raise RuntimeError("Worker observation requested on the wrong node")
    path = campaign_path.parent / "workers" / f"{family}.json"
    intent_path = path.with_name(f"{family}.start.json")
    state = read(path) if path.exists() else read(intent_path) if intent_path.exists() else None
    if state is None:
        return {"status": "not_launched", "family": family}
    launch = read(campaign_path.parent / "launch-state.json")
    if (
        state.get("family") != family
        or state.get("node") != assignment["node"]
        or state.get("binding_sha256") != sha256(campaign_path.parent / "execution-binding.json")
        or state.get("worker_marker") != launch["worker_markers"][family]
    ):
        raise ValueError("Worker observation has a different bound family or launch generation")
    if state["status"] == "spawn_failed":
        return {"status": "needs_attention", "family": family, "receipt": state}
    if "pid" not in state or state.get("start_ticks") is None:
        return {"status": "unknown", "family": family, "receipt": state}
    observed = observed_process(state["pid"])
    if observed is not None and observed[0] != "Z":
        if observed[1] != state["start_ticks"] or observed[2] != worker_command(campaign_path, frozen, family):
            return {"status": "needs_attention", "family": family, "error": "Worker PID reused or command changed"}
        return {"status": "running", "family": family, "receipt": state}
    if state.get("trials"):
        child_path = campaign_path.parent / "workers" / f"{state['trials'][-1]['name']}.execution.json"
        if child_path.exists():
            child = read(child_path)
            descendants = tagged_processes(child["child_marker"]) if child.get("child_marker") else {}
            if descendants:
                return {"status": "live_orphaned_trial", "family": family, "receipt": state, "descendants": descendants}
    restored = type(state.get("keeper_restore")) is int and state["keeper_restore"] == 0
    terminal = "complete" if state["status"] == "complete" and restored else "needs_attention"
    if terminal == "complete":
        if (
            [trial["name"] for trial in state["trials"]] != [trial["name"] for trial in assignment["trials"]]
            or any(
                trial["status"] != "complete"
                or trial["exit_code"] != 0
                or trial["command"] != expected["command"]
                or trial["log_sha256"] != sha256(Path(trial["log"]))
                for trial, expected in zip(state["trials"], assignment["trials"], strict=True)
            )
            or "cleanup_error" in state
        ):
            raise ValueError("Completed worker lacks its three successful exact trial command/log receipts")
        for trial in assignment["trials"]:
            completed_trial(campaign_path, trial["name"])
    return {"status": terminal, "family": family, "receipt": state}


def snapshot(campaign_path: Path) -> dict:
    """Save process evidence and exact consecutive scalar prefixes; never infer missing rows."""
    frozen = binding(campaign_path)
    workers = []
    trials = []
    for assignment in frozen["plan"]:
        try:
            worker = remote(campaign_path, frozen, assignment["node"], "probe", assignment["family"])
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            worker = {"status": "unknown", "family": assignment["family"], "error": str(error)}
        if (
            worker["status"] == "not_launched"
            and read(campaign_path.parent / "launch-state.json")["status"] == "needs_attention"
        ):
            worker["status"] = "not_launched_after_launch_failure"
        workers.append(worker)
        for trial in assignment["trials"]:
            out = Path(trial["out"])
            item = {
                "name": trial["name"],
                "family": assignment["family"],
                "observed_tokens": 0,
                "status": "not_launched",
            }
            if (out / "run.json").exists():
                item["status"] = read(out / "run.json")["status"]
                segments = sorted(out.glob("metrics_from_*.jsonl"))
                if segments and all(path.stat().st_size for path in segments):
                    try:
                        raw = [path.read_bytes() for path in segments]
                        if any(not payload.endswith(b"\n") for payload in raw):
                            raise ValueError("Unfinished scalar row; prefix accounting is unknown")
                        rows = [json.loads(line) for payload in raw for line in payload.splitlines()]
                        if (
                            not 0 < len(rows) <= 320
                            or [row["step"] for row in rows] != list(range(1, len(rows) + 1))
                            or any(row["tokens"] != row["step"] * 524288 for row in rows)
                        ):
                            raise ValueError("Missing, overlapping or inconsistent short-trial token history")
                        item["observed_tokens"] = rows[-1]["tokens"]
                        item["observed_updates"] = len(rows)
                        item["history_sha256"] = {
                            path.name: hashlib.sha256(payload).hexdigest()
                            for path, payload in zip(segments, raw, strict=True)
                        }
                    except (ValueError, KeyError) as error:
                        item.update(observed_tokens=None, accounting_error=str(error))
            trials.append(item)
    statuses = {worker["status"] for worker in workers}
    terminal = statuses <= {"complete", "needs_attention", "not_launched_after_launch_failure"}
    return {
        "status": "complete" if statuses == {"complete"} else "needs_attention" if terminal else "active",
        "campaign_sha256": frozen["campaign_sha256"],
        "binding_sha256": sha256(campaign_path.parent / "execution-binding.json"),
        "workers": workers,
        "trials": trials,
        "all_workers_terminal": terminal,
        "token_accounting": "Actual consecutive prefixes, including failed runs; unknown prefixes are not zero charges.",
    }


def install_monitor(campaign_path: Path, frozen: dict) -> int:
    directory = campaign_path.parent / "monitor"
    directory.mkdir(exist_ok=True)
    if (directory / "pid").exists():
        raise FileExistsError("Campaign monitor already has evidence; inspect it without restart")
    with (directory / "monitor.log").open("x") as log:
        command = ["bash", str(Path(frozen["helpers"]) / "monitor_fixed_age_tuning.sh"), str(campaign_path)]
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if (directory / "pid").exists():
            pid = int((directory / "pid").read_text())
            observed = observed_process(pid)
            if observed is not None and observed[0] != "Z" and pid == child.pid and observed[2] == command:
                return pid
        if child.poll() is not None:
            raise RuntimeError("Campaign monitor exited before startup acknowledgment")
        time.sleep(0.1)
    raise TimeoutError("Campaign monitor startup is unacknowledged; preserve its child without retry")


def launch(campaign_path: Path) -> dict:
    require_gpu_environment(campaign_path)
    frozen = require_ready(campaign_path)
    directory = campaign_path.parent
    state_path = directory / "launch-state.json"
    if any(path.exists() for path in (state_path, directory / "workers", directory / "runs", directory / "monitor")):
        raise FileExistsError("Campaign already has launch evidence; preserve it without retry")
    (directory / "workers").mkdir()
    state = {
        "status": "launching",
        "binding_sha256": sha256(directory / "execution-binding.json"),
        "worker_markers": {family: uuid4().hex for family in ARMS},
        "workers": {},
    }
    write(state_path, state)
    try:
        state["monitor_pid"] = install_monitor(campaign_path, frozen)
        write(state_path, state)
        for assignment in frozen["plan"]:
            family = assignment["family"]
            acknowledgment = remote(campaign_path, frozen, assignment["node"], "start", family)
            state["workers"][family] = acknowledgment
            write(state_path, state)
            if (
                acknowledgment.get("family") != family
                or acknowledgment.get("node") != assignment["node"]
                or acknowledgment.get("binding_sha256") != state["binding_sha256"]
                or acknowledgment.get("worker_marker") != state["worker_markers"][family]
                or acknowledgment.get("command") != worker_command(campaign_path, frozen, family)
                or type(acknowledgment.get("pid")) is not int
                or acknowledgment["pid"] <= 0
                or not isinstance(acknowledgment.get("start_ticks"), str)
                or acknowledgment.get("status") not in ("preflight", "running", "complete")
            ):
                raise RuntimeError("Family worker acknowledgment differs or reports failure; preserve partial launch")
        state["status"] = "launched"
        write(state_path, state)
        return state
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(state_path, state)
        # Workers own their locks, cleanup and restoration. Unknown remote state never authorizes restoration here.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("freeze", "check", "launch", "start", "worker", "probe", "snapshot"), required=True
    )
    parser.add_argument("--family", choices=ARMS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    campaign_path = args.campaign.resolve()
    if args.phase in ("start", "worker", "probe") and args.family is None:
        parser.error("Node operation requires --family")
    if args.phase == "freeze":
        result = freeze(campaign_path)
    elif args.phase == "check":
        result = binding(campaign_path)
    elif args.phase == "launch":
        with defer_signals():
            with (campaign_path.parent / ".execution-controller.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = launch(campaign_path)
    elif args.phase == "start":
        result = start_family(campaign_path, args.family)
    elif args.phase == "worker":
        result = run_family(campaign_path, args.family)
    elif args.phase == "probe":
        result = probe_worker(campaign_path, args.family)
    else:
        if args.out is None:
            parser.error("Snapshot requires --out")
        result = snapshot(campaign_path)
        write(args.out, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
