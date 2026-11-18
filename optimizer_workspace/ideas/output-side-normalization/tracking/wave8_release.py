# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch and monitor source-bound Wave-8 workers after numerical gates and the short LR campaign."""

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

import wave8_history
from launch_wave5 import NODES, PYTHON, node_action, read, sha256
from run_diagonal_diagnostic import PROCESS_MARKER, finish_child, tagged_processes
from validate_wave5 import ENVIRONMENT
from watch_wave5 import write
from watch_wave7_release import observed_process
from wave7_release import defer_signals, idle_training, local_node, node_guard, release_keeper
from wave8_history import ARMS, EXTRA_HELPERS, commands, completed_run, measured_run, paths

WORKER_MARKER = "OUTPUT_MARIN_WAVE8_WORKER_ID"


def require_numerical(root: Path) -> dict:
    """Rederive GPU evidence through the exact original controller and dependencies."""
    numerical = read(root / "run-preparation/wave8-candidate.json")
    command = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import wave8
root=Path(sys.argv[2])
frozen=wave8.binding(root)
wave8.require_validation(root, frozen)
print(json.dumps({"status":"rederived_all_original_wave8_gpu_gates"}))
"""
    subprocess.run(
        [PYTHON, "-c", command, numerical["helpers"], str(root)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return numerical


def freeze(root: Path) -> dict:
    """Bind launch helpers only after rederiving actual gates and verifying controller cleanup."""
    numerical_path = root / "run-preparation/wave8-candidate.json"
    numerical = read(numerical_path)
    original = Path(numerical["helpers"])
    subprocess.run(
        [PYTHON, str(original / "wave8.py"), "--root", str(root), "--phase", "check"],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    state_path = root / "run-preparation/wave8-validation-state.json"
    state = read(state_path)
    validation = root / "run-preparation/validation-wave8-gpu.json"
    if (
        state["status"] != "passed"
        or type(state.get("keeper_restore")) is not int
        or state["keeper_restore"] != 0
        or state.get("cleanup_exit_code") != 0
        or "cleanup_error" in state
        or state["validation_sha256"] != sha256(validation)
        or state["binding_sha256"] != sha256(numerical_path)
    ):
        raise ValueError("Wave-8 numerical validation, cleanup or restoration remains incomplete")
    if observed_validation(root, state)["status"] != "exited_clean":
        raise RuntimeError("Wave-8 validator controller or marked children remain live")
    require_numerical(root)
    release_path, helpers = paths(root)
    archive = root / "run-preparation/source-wave8-launch-20261117.tar"
    if any(path.exists() for path in (release_path, helpers, archive)):
        raise FileExistsError("Wave-8 launch evidence already exists; preserve without refreezing")
    helpers.mkdir(parents=True)
    with tarfile.open(archive, "x") as stream:
        for name in (*numerical["helpers_sha256"], *EXTRA_HELPERS):
            source = original / name if name in numerical["helpers_sha256"] else Path(__file__).parent / name
            if name in numerical["helpers_sha256"] and sha256(source) != numerical["helpers_sha256"][name]:
                raise ValueError("Original Wave-8 numerical helper changed")
            shutil.copy2(source, helpers / name)
            stream.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        stream.add(Path(__file__).parents[1] / "ENDEAVOR.md", arcname="ENDEAVOR.md", recursive=False)
    release = {
        "status": "frozen_validated_launch_release",
        "root": str(root),
        "candidate": numerical["candidate"],
        "helpers": str(helpers),
        "helpers_sha256": {name: sha256(helpers / name) for name in (*numerical["helpers_sha256"], *EXTRA_HELPERS)},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "candidate_binding_sha256": sha256(numerical_path),
        "validation_sha256": sha256(validation),
        "validation_state_sha256": sha256(state_path),
        "data_receipt_sha256": numerical["data_receipt_sha256"],
        "environment": ENVIRONMENT,
        "arms": ARMS,
        "nodes": NODES,
        "commands": commands(root, Path(numerical["candidate"])),
        "monitor_interval_seconds": 900,
        "training_timeout_seconds": 21600,
        "policy": (
            "Full original schedules, no resume/retry; LR campaign completes first; failures retain token charges."
        ),
    }
    write(release_path, release)
    return release


def observed_validation(root: Path, state: dict) -> dict:
    probe = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from watch_wave7_release import observed_process
from run_diagonal_diagnostic import tagged_processes
state=json.loads(Path(sys.argv[2]).read_text())
print(json.dumps({"controller": observed_process(state["controller_pid"]),
                  "children": tagged_processes(state["child_marker"])}))
"""
    numerical = read(root / "run-preparation/wave8-candidate.json")
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "root@28.216.17.139",
            shlex.join(
                [PYTHON, "-c", probe, numerical["helpers"], str(root / "run-preparation/wave8-validation-state.json")]
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    observed = json.loads(result.stdout)
    controller = observed["controller"]
    terminal = (controller is None or controller[0] == "Z") and not observed["children"]
    return {"status": "exited_clean" if terminal else "live_or_unknown", "observation": observed}


def binding(root: Path) -> dict:
    release = wave8_history.binding(root)
    if Path(__file__).resolve() != (Path(release["helpers"]) / "wave8_release.py").resolve():
        raise ValueError("Use the bound Wave-8 launch controller")
    return release


def helper_binding(root: Path) -> dict:
    release = wave8_history.helper_binding(root)
    if Path(__file__).resolve() != (Path(release["helpers"]) / "wave8_release.py").resolve():
        raise ValueError("Use the bound Wave-8 process observer")
    return release


def require_lr_complete(root: Path) -> None:
    """Give the already prepared equal-budget LR study first use of all original nodes."""
    directory = root / "campaigns/lr-fixed-age-0p5m-20261116"
    monitor = read(directory / "monitor/latest.json")
    cloud = read(directory / "swanlab-upload.json")
    analysis = read(directory / "analysis.json")
    campaign = read(directory / "campaign.json")
    execution = read(directory / "execution-binding.json")
    if (
        monitor["status"] != "complete"
        or monitor["all_workers_terminal"] is not True
        or monitor["binding_sha256"] != sha256(directory / "execution-binding.json")
        or len(monitor["workers"]) != 4
        or any(worker["status"] != "complete" for worker in monitor["workers"])
        or analysis["status"] != "complete"
        or len(analysis["completed_family_lr_choices"]) != 4
        or analysis["campaign_sha256"] != sha256(directory / "campaign.json")
        or cloud["status"] != "verified"
        or cloud["visibility"] != "PRIVATE"
        or cloud["campaign_sha256"] != sha256(directory / "campaign.json")
        or cloud["execution_binding_sha256"] != sha256(directory / "execution-binding.json")
        or set(cloud["runs"]) != {trial["name"] for trial in campaign["trials"]}
        or any(entry["status"] != "verified" for entry in cloud["runs"].values())
    ):
        raise RuntimeError("Completed, analyzed and cloud-verified short-LR study is required before Wave-8 launch")
    subprocess.run(
        [
            PYTHON,
            str(Path(execution["helpers"]) / "run_fixed_age_tuning.py"),
            "--campaign",
            str(directory / "campaign.json"),
            "--phase",
            "check",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    command = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from run_fixed_age_tuning import snapshot
observed=snapshot(Path(sys.argv[2]))
assert observed["status"] == "complete" and observed["all_workers_terminal"] is True
"""
    subprocess.run(
        [PYTHON, "-c", command, execution["helpers"], str(directory / "campaign.json")],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )


def worker_command(root: Path, release: dict, arm: str) -> list[str]:
    return [
        PYTHON,
        str(Path(release["helpers"]) / "wave8_release.py"),
        "--root",
        str(root),
        "--phase",
        "worker",
        "--arm",
        arm,
    ]


def arm_node(arm: str) -> str:
    return NODES[ARMS.index(arm)]


def marked_workers(marker: str) -> dict[int, str]:
    """Find a lost worker acknowledgment using its distinct inherited marker key."""
    wanted = f"{WORKER_MARKER}={marker}".encode()
    found = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            environment = (directory / "environ").read_bytes().split(b"\0")
            observed = observed_process(int(directory.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if wanted in environment and observed is not None and observed[0] != "Z":
            found[int(directory.name)] = observed[1]
    return found


def worker_path(root: Path, arm: str) -> Path:
    return root / "run-preparation" / f"wave8-worker-{arm_node(arm)}.json"


def allocation(root: Path, release: dict, arm: str) -> None:
    if local_node() != arm_node(arm):
        raise RuntimeError("Wave-8 worker is on the wrong original node")
    require_lr_complete(root)
    idle_training()
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(devices) != 8 or any("A100-SXM4-40GB" not in device for device in devices):
        raise ValueError("Wave 8 requires original eight A100 40GB GPUs")
    subprocess.run([PYTHON, "-m", "experiment.bootstrap", "--check-only"], cwd=release["candidate"], check=True)


def started_worker(root: Path, release: dict, arm: str) -> tuple:
    intent_path = worker_path(root, arm).with_suffix(".start.json")
    deadline = time.monotonic() + 30
    intent = read(intent_path)
    while "pid" not in intent and time.monotonic() < deadline:
        time.sleep(0.1)
        intent = read(intent_path)
    own = observed_process(os.getpid())
    launch = read(root / "run-preparation/wave8-launch-state.json")
    if (
        own is None
        or intent.get("pid") != os.getpid()
        or intent.get("start_ticks") != own[1]
        or intent["command"] != worker_command(root, release, arm)
        or own[2] != intent["command"]
        or intent["release_binding_sha256"] != sha256(paths(root)[0])
        or intent["worker_marker"] != launch["worker_markers"][arm]
        or os.environ.get(WORKER_MARKER) != intent["worker_marker"]
        or launch["status"] not in ("launching", "launched", "needs_attention")
        or launch["binding_sha256"] != sha256(paths(root)[0])
    ):
        raise RuntimeError("Wave-8 worker spawn generation or launch release changed")
    return own


def train_child(root: Path, release: dict, state: dict, state_path: Path) -> int:
    """Persist exact torchrun generation before ranks reach the frozen pre-CUDA guard."""
    command = release["commands"][state["node"]]
    marker, generation = uuid4().hex, {}
    log = root / "runs/wave8" / f"{state['arm']}.torchrun.log"
    state.update(status="starting", command=command, child_marker=marker, log=str(log))
    write(state_path, state)
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=release["candidate"],
            env={**os.environ, **ENVIRONMENT, PROCESS_MARKER: marker},
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            observed = observed_process(child.pid)
            if observed is None or observed[2] != command:
                raise RuntimeError("Wave-8 torchrun generation was not acknowledged")
            state.update(
                status="running",
                command=command,
                child_pid=child.pid,
                child_start_ticks=observed[1],
                child_marker=marker,
                log=str(log),
            )
            write(state_path, state)
            deadline = time.monotonic() + release["training_timeout_seconds"]
            while time.monotonic() < deadline:
                generation.update(tagged_processes(marker))
                try:
                    return child.wait(timeout=min(1, max(0.001, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            raise subprocess.TimeoutExpired(command, release["training_timeout_seconds"])
        finally:
            try:
                finish_child(child, marker, generation)
                state["cleanup_exit_code"] = child.returncode
            except Exception as error:
                state["cleanup_error"] = str(error)
                raise
            finally:
                state.update(child_generation=generation, log_sha256=sha256(log))
                write(state_path, state)


def run_arm(root: Path, arm: str) -> dict:
    release = binding(root)
    own = started_worker(root, release, arm)
    state_path = worker_path(root, arm)
    if state_path.exists():
        raise FileExistsError("Preserve previous Wave-8 worker evidence without retry")
    with node_guard(root, arm_node(arm)) as signals:
        allocation(root, release, arm)
        state = {
            "status": "starting",
            "arm": arm,
            "node": arm_node(arm),
            "worker_pid": os.getpid(),
            "worker_start_ticks": own[1],
            "worker_command": own[2],
            "worker_marker": os.environ[WORKER_MARKER],
            "release_binding_sha256": sha256(paths(root)[0]),
            "out": str(root / "runs/wave8" / arm),
        }
        write(state_path, state)
        try:
            if Path(state["out"]).exists():
                raise FileExistsError("Preserve existing full-budget output without retry")
            state["keeper"] = release_keeper(state, state_path)
            code = train_child(root, release, state, state_path)
            state["exit_code"] = code
            if code:
                raise RuntimeError("Wave-8 full-budget command failed")
            state["completed_run"] = completed_run(root, arm, release)
            state["status"] = "complete"
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if state.get("keeper_release_attempted"):
                try:
                    if "cleanup_error" in state:
                        raise RuntimeError("Unverified marked-child cleanup prevents keeper restoration")
                    node_action(root, "restore")
                    state["keeper_restore"] = 0
                except Exception as error:
                    state.update(status="needs_attention", keeper_restore=f"refused: {error}")
            if signals:
                state["deferred_signals"] = signals
            write(state_path, state)
        return state


def start_arm(root: Path, arm: str) -> dict:
    release = binding(root)
    if local_node() != arm_node(arm):
        raise RuntimeError("Wave-8 worker start is on the wrong node")
    path = worker_path(root, arm)
    intent_path = path.with_suffix(".start.json")
    with path.with_suffix(".start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() or intent_path.exists():
            raise FileExistsError("Wave-8 worker has previous spawn evidence; no retry")
        launch = read(root / "run-preparation/wave8-launch-state.json")
        if launch["status"] != "launching" or launch["binding_sha256"] != sha256(paths(root)[0]):
            raise ValueError("Wave-8 arm spawn is not authorized by its exact release")
        command = worker_command(root, release, arm)
        intent = {
            "status": "starting",
            "arm": arm,
            "node": arm_node(arm),
            "command": command,
            "release_binding_sha256": sha256(paths(root)[0]),
            "worker_marker": launch["worker_markers"][arm],
        }
        write(intent_path, intent)
        try:
            with path.with_suffix(".log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    cwd=release["candidate"],
                    env={**os.environ, **ENVIRONMENT, WORKER_MARKER: intent["worker_marker"]},
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except Exception as error:
            intent.update(status="spawn_failed", error=str(error))
            write(intent_path, intent)
            raise
        own = observed_process(child.pid)
        intent.update(pid=child.pid, start_ticks=None if own is None else own[1])
        write(intent_path, intent)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if path.exists():
                state = read(path)
                if state["worker_pid"] != child.pid or state["worker_start_ticks"] != intent["start_ticks"]:
                    raise RuntimeError("Wave-8 worker acknowledgment generation differs")
                return state
            if child.poll() is not None:
                raise RuntimeError("Wave-8 worker exited before acknowledgment; inspect saved log")
            time.sleep(0.1)
        raise TimeoutError("Wave-8 worker acknowledgment timed out; preserve its PID without retry")


def remote(root: Path, release: dict, arm: str, phase: str) -> dict:
    command = [
        PYTHON,
        str(Path(release["helpers"]) / "wave8_release.py"),
        "--root",
        str(root),
        "--phase",
        phase,
        "--arm",
        arm,
    ]
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"root@{arm_node(arm)}", shlex.join(command)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode:
        raise RuntimeError(f"{arm} {phase}: {result.stderr[-4000:]}")
    return json.loads(result.stdout.splitlines()[-1])


def probe_arm(root: Path, arm: str) -> dict:
    release = helper_binding(root)
    if local_node() != arm_node(arm):
        raise RuntimeError("Wave-8 probe is on the wrong node")
    path = worker_path(root, arm)
    intent_path = path.with_suffix(".start.json")
    state = read(path) if path.exists() else read(intent_path) if intent_path.exists() else None
    if state is None:
        return {"arm": arm, "status": "not_launched"}
    launch = read(root / "run-preparation/wave8-launch-state.json")
    if (
        state["arm"] != arm
        or state["node"] != arm_node(arm)
        or state["release_binding_sha256"] != sha256(paths(root)[0])
        or state["worker_marker"] != launch["worker_markers"][arm]
    ):
        raise ValueError("Wave-8 worker receipt is not bound to this arm/release")
    pid = state.get("worker_pid", state.get("pid"))
    ticks = state.get("worker_start_ticks", state.get("start_ticks"))
    if state["status"] == "spawn_failed":
        return {"arm": arm, "status": "needs_attention", "receipt": state}
    if pid is None or ticks is None:
        descendants = marked_workers(state["worker_marker"])
        if descendants:
            return {"arm": arm, "status": "unacknowledged_live_worker", "receipt": state, "descendants": descendants}
        return {"arm": arm, "status": "unknown", "receipt": state}
    observed = observed_process(pid)
    descendants = tagged_processes(state["child_marker"]) if state.get("child_marker") else {}
    if observed is not None and observed[0] != "Z":
        if observed[1] != ticks or observed[2] != worker_command(root, release, arm):
            return {
                "arm": arm,
                "status": "live_orphaned_training" if descendants else "worker_generation_changed",
                "error": "Worker PID reused or argv changed; unrelated process is not owned",
                "receipt": state,
                "descendants": descendants,
            }
        return {"arm": arm, "status": "running", "receipt": state}
    if descendants:
        return {"arm": arm, "status": "live_orphaned_training", "receipt": state, "descendants": descendants}
    terminal = (
        state["status"] == "complete"
        and type(state.get("keeper_restore")) is int
        and state["keeper_restore"] == 0
        and state.get("exit_code") == 0
        and state.get("cleanup_exit_code") == 0
        and "cleanup_error" not in state
    )
    if terminal:
        if (
            state["command"] != release["commands"][arm_node(arm)]
            or state["log_sha256"] != sha256(Path(state["log"]))
            or state["completed_run"] != completed_run(root, arm, release)
        ):
            raise ValueError("Wave-8 completed worker history/command/log evidence changed")
    return {"arm": arm, "status": "complete" if terminal else "needs_attention", "receipt": state}


def snapshot(root: Path) -> dict:
    release = helper_binding(root)
    workers, runs = [], []
    report_path = root / "run-preparation/wave8.json"
    entries = {}
    if report_path.exists():
        for entry in read(report_path)["launches"]:
            arm = entry["arm"]
            if (
                arm not in ARMS
                or arm in entries
                or entry["node"] != arm_node(arm)
                or entry["out"] != str(root / "runs/wave8" / arm)
                or entry["command"] != release["commands"][arm_node(arm)]
                or type(entry["pid"]) is not int
                or entry["pid"] <= 0
            ):
                raise ValueError("Saved Wave-8 launch inventory changed")
            entries[arm] = entry
    launch = read(root / "run-preparation/wave8-launch-state.json")
    for arm in ARMS:
        try:
            worker = remote(root, release, arm, "probe")
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            worker = {"arm": arm, "status": "unknown", "error": str(error)}
        if worker["status"] == "not_launched" and launch["status"] == "needs_attention":
            worker["status"] = "not_launched_after_launch_failure"
        workers.append(worker)
        state = worker.get("receipt", {})
        run = {
            "name": arm,
            "arm": arm,
            "node": arm_node(arm),
            "status": worker["status"],
            "planned_steps": 5120,
            "latest_metrics": None,
            "observed_tokens": 0 if worker["status"] in ("not_launched", "not_launched_after_launch_failure") else None,
            "keeper_watcher": {
                "status": "restored" if worker["status"] == "complete" else state.get("status", "unknown")
            },
        }
        directory = root / "runs/wave8" / arm
        if (directory / "run.json").exists():
            try:
                measured = measured_run(root, arm, release)
                run.update(
                    latest_metrics={"step": measured["observed_updates"]},
                    observed_tokens=measured["observed_tokens"],
                    manifest_status=read(directory / "run.json")["status"],
                )
            except (ValueError, OSError, KeyError) as error:
                run.update(observed_tokens=None, accounting_error=str(error))
        if state.get("child_pid"):
            entries[arm] = {
                "arm": arm,
                "node": arm_node(arm),
                "seed": 0,
                "batch": "0p5m",
                "out": str(directory),
                "pid": state["child_pid"],
                "start_ticks": state["child_start_ticks"],
                "command": state["command"],
            }
        runs.append(run)
    statuses = {worker["status"] for worker in workers}
    terminal = statuses <= {"complete", "needs_attention", "not_launched_after_launch_failure"}
    write(report_path, {"expected_launches": 4, "launches": list(entries.values())})
    return {
        "status": "complete" if statuses == {"complete"} else "needs_attention" if terminal else "active",
        "binding_sha256": sha256(paths(root)[0]),
        "workers": workers,
        "runs": runs,
        "all_workers_terminal": terminal,
        "token_accounting": "Consecutive actual prefixes; failed tokens charged; unknown scalar reads are not zero.",
    }


def install_monitor(root: Path, release: dict) -> int:
    directory = root / "run-preparation/monitor-wave8"
    directory.mkdir(exist_ok=True)
    if (directory / "pid").exists():
        raise FileExistsError("Wave-8 monitor already has evidence; preserve without restart")
    command = ["bash", str(Path(release["helpers"]) / "monitor_wave8.sh"), str(root)]
    with (directory / "monitor.log").open("x") as log:
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if (directory / "pid").exists():
            pid = int((directory / "pid").read_text())
            observed = observed_process(pid)
            if pid == child.pid and observed is not None and observed[0] != "Z" and observed[2] == command:
                return pid
        if child.poll() is not None:
            raise RuntimeError("Wave-8 monitor exited before acknowledgment")
        time.sleep(0.1)
    raise TimeoutError("Wave-8 monitor startup timed out; preserve its child without retry")


def launch(root: Path) -> dict:
    release = binding(root)
    require_lr_complete(root)
    require_numerical(root)
    preparation = root / "run-preparation"
    state_path = preparation / "wave8-launch-state.json"
    if any(path.exists() for path in (state_path, root / "runs/wave8", preparation / "monitor-wave8")):
        raise FileExistsError("Wave-8 launch evidence exists; preserve without retry")
    for arm in ARMS:
        if worker_path(root, arm).exists() or worker_path(root, arm).with_suffix(".start.json").exists():
            raise FileExistsError("Wave-8 worker has earlier evidence")
        remote(root, release, arm, "inspect")
    (root / "runs/wave8").mkdir(parents=True)
    state = {
        "status": "launching",
        "binding_sha256": sha256(paths(root)[0]),
        "workers": {},
        "worker_markers": {arm: uuid4().hex for arm in ARMS},
    }
    write(state_path, state)
    try:
        state["monitor_pid"] = install_monitor(root, release)
        write(state_path, state)
        for arm in ARMS:
            acknowledgment = remote(root, release, arm, "start")
            state["workers"][arm] = acknowledgment
            write(state_path, state)
            if (
                acknowledgment["arm"] != arm
                or acknowledgment["node"] != arm_node(arm)
                or acknowledgment["release_binding_sha256"] != state["binding_sha256"]
                or acknowledgment["worker_marker"] != state["worker_markers"][arm]
                or acknowledgment["worker_command"] != worker_command(root, release, arm)
                or type(acknowledgment.get("worker_pid")) is not int
                or not isinstance(acknowledgment.get("worker_start_ticks"), str)
                or acknowledgment["status"] not in ("starting", "running", "complete")
            ):
                raise RuntimeError("Wave-8 worker acknowledgment differs; preserve partial launch")
        state["status"] = "launched"
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("freeze", "check", "launch", "inspect", "start", "worker", "probe", "snapshot"),
        required=True,
    )
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase in ("inspect", "start", "worker", "probe") and args.arm is None:
        parser.error("Node operation requires --arm")
    if args.phase == "freeze":
        result = freeze(root)
    elif args.phase == "check":
        result = binding(root)
    elif args.phase == "launch":
        with defer_signals():
            with (root / "run-preparation/.wave8-launch-controller.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = launch(root)
    elif args.phase == "start":
        result = start_arm(root, args.arm)
    elif args.phase == "inspect":
        release = binding(root)
        with node_guard(root, arm_node(args.arm)):
            allocation(root, release, args.arm)
        result = {"status": "allocation_preflight_passed", "arm": args.arm, "node": arm_node(args.arm)}
    elif args.phase == "worker":
        result = run_arm(root, args.arm)
    elif args.phase == "probe":
        result = probe_arm(root, args.arm)
    else:
        if args.out is None:
            parser.error("Snapshot requires --out")
        result = snapshot(root)
        write(args.out, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
