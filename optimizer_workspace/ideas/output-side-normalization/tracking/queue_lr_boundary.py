# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind and advance the upper-LR handoff without retrying an acknowledged controller."""

import argparse
import fcntl
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

from boundary_handoff import ARMS, require_completed
from fixed_age_protocol import environment, predecessor_wave, require_gpu_environment
from launch_wave5 import PYTHON, read, sha256
from prepare_lr_boundary import STUDY
from prepare_lr_ceiling import STUDY as CEILING_STUDY
from watch_wave5 import write
from watch_wave7_release import observed_process

EXTRA_HELPERS = ("queue_lr_boundary.py", "queue_lr_boundary.sh")
MARKER = "OUTPUT_MARIN_BOUNDARY_QUEUE_CHILD"
QUEUE_MARKER = "OUTPUT_MARIN_BOUNDARY_QUEUE"
INTERVAL = 900
NODE = "28.216.17.9"


def directory(campaign: Path) -> Path:
    return campaign.parent / "queue"


def handoff_wave(campaign: Path) -> str:
    return "wave9" if read(campaign).get("study") == CEILING_STUDY else "wave8"


def handoff_arms(campaign: Path) -> tuple[str, ...]:
    if handoff_wave(campaign) == "wave9":
        return ("muont", "outer_leftdiag_fused", "outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused")
    return ARMS


def command(campaign: Path, phase: str) -> list[str]:
    if phase not in ("validate", "launch"):
        raise ValueError("Unknown boundary controller phase")
    kind = "validation" if phase == "validate" else "execution"
    module = "run_fixed_age_validation.py" if phase == "validate" else "run_fixed_age_tuning.py"
    return [PYTHON, str(campaign.parent / f"{kind}-helpers" / module), "--campaign", str(campaign), "--phase", phase]


def check_release(path: Path) -> dict:
    frozen = read(path)
    if frozen["archive_sha256"] != sha256(Path(frozen["archive"])) or any(
        sha256(Path(frozen["helpers"]) / name) != digest for name, digest in frozen["helpers_sha256"].items()
    ):
        raise ValueError("Boundary validation/execution release changed before handoff")
    return frozen


def freeze(campaign: Path) -> dict:
    campaign = campaign.resolve()
    planned = read(campaign)
    if planned["study"] not in (STUDY, CEILING_STUDY):
        raise ValueError("Queue accepts only the separately identified upper-boundary study")
    wave = predecessor_wave(campaign)
    validation = check_release(campaign.parent / "validation-binding.json")
    execution = check_release(campaign.parent / "execution-binding.json")
    if (
        validation["campaign_sha256"] != sha256(campaign)
        or execution["campaign_sha256"] != sha256(campaign)
        or execution["validation_binding_sha256"] != sha256(campaign.parent / "validation-binding.json")
    ):
        raise ValueError("Boundary queue release does not match its campaign")
    for phase in ("validate", "launch"):
        subprocess.run(
            [*command(campaign, phase)[:-1], "check"], check=True, capture_output=True, text=True, timeout=240
        )
    target = directory(campaign)
    if target.exists():
        raise FileExistsError("Boundary queue already has evidence; preserve it without refreezing")
    target.mkdir()
    helpers = target / "helpers"
    helpers.mkdir()
    archive = target / "source.tar"
    names = (*execution["helpers_sha256"], *EXTRA_HELPERS)
    with tarfile.open(archive, "x") as stream:
        for name in names:
            destination = helpers / name
            source = (
                Path(execution["helpers"]) / name
                if name in execution["helpers_sha256"]
                else Path(__file__).parent / name
            )
            destination.write_bytes(source.read_bytes())
            stream.add(destination, arcname=f"tracking/{name}", recursive=False)
    result = {
        "status": f"frozen_waiting_for_{wave}",
        "campaign": str(campaign),
        "campaign_sha256": sha256(campaign),
        "validation_binding_sha256": sha256(campaign.parent / "validation-binding.json"),
        "execution_binding_sha256": sha256(campaign.parent / "execution-binding.json"),
        "predecessor_release_sha256": validation["predecessor_release_sha256"],
        "helpers": str(helpers),
        "helpers_sha256": {name: sha256(helpers / name) for name in names},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "node": NODE,
        "interval_seconds": INTERVAL,
        "environment": environment(campaign),
        "commands": {phase: command(campaign, phase) for phase in ("validate", "launch")},
        "policy": (
            f"Fresh complete {wave} process/history/analysis/private-cloud evidence before validation; "
            "actual eight-config gates and restoration before launch. Controller intents are never retried; "
            "unknown observations wait without releasing keepers."
        ),
    }
    write(target / "binding.json", result)
    return result


def binding(campaign: Path) -> dict:
    frozen = read(directory(campaign) / "binding.json")
    execution = check_release(campaign.parent / "execution-binding.json")
    if (
        Path(__file__).resolve() != (Path(frozen["helpers"]) / "queue_lr_boundary.py").resolve()
        or frozen["campaign"] != str(campaign.resolve())
        or frozen["campaign_sha256"] != sha256(campaign)
        or frozen["validation_binding_sha256"] != sha256(campaign.parent / "validation-binding.json")
        or frozen["execution_binding_sha256"] != sha256(campaign.parent / "execution-binding.json")
        or set(frozen["helpers_sha256"]) != set(execution["helpers_sha256"]) | set(EXTRA_HELPERS)
        or any(frozen["helpers_sha256"].get(name) != digest for name, digest in execution["helpers_sha256"].items())
        or any(sha256(Path(frozen["helpers"]) / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["interval_seconds"] != INTERVAL
        or frozen["node"] != NODE
        or frozen["environment"] != environment(campaign)
        or frozen["commands"] != {phase: command(campaign, phase) for phase in ("validate", "launch")}
    ):
        raise ValueError("Boundary queue helper, campaign, release, archive or command identity changed")
    validation = check_release(campaign.parent / "validation-binding.json")
    root = Path(read(campaign)["root"])
    wave = handoff_wave(campaign)
    if frozen["predecessor_release_sha256"] != validation["predecessor_release_sha256"] or frozen[
        "predecessor_release_sha256"
    ] != sha256(root / f"run-preparation/{wave}-launch-binding.json"):
        raise ValueError(f"Boundary queue has a changed {wave} predecessor")
    return frozen


def install(campaign: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("CPU-only CUDA visibility prevents GPU queue installation")
    require_gpu_environment(campaign)
    frozen = binding(campaign)
    target = directory(campaign)
    path = target / "installation.json"
    with (target / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((target / name).exists() for name in ("installation.json", "pid", "queue.log", "stopped_at")):
            raise FileExistsError("Queue already has installation evidence; inspect its generation without retry")
        command = ["bash", str(Path(frozen["helpers"]) / "queue_lr_boundary.sh"), str(campaign)]
        intent = {
            "status": "starting",
            "binding_sha256": sha256(target / "binding.json"),
            "command": command,
            "marker": uuid4().hex,
        }
        write(path, intent)
        try:
            with (target / "queue.log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    cwd=read(campaign)["source"],
                    env={**os.environ, **environment(campaign), QUEUE_MARKER: intent["marker"]},
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            observed = observed_process(child.pid)
            intent.update(pid=child.pid, start_ticks=None if observed is None else observed[1])
            write(path, intent)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if (target / "pid").exists():
                    observed = observed_process(child.pid)
                    token = f"{QUEUE_MARKER}={intent['marker']}".encode()
                    if (
                        int((target / "pid").read_text()) != child.pid
                        or observed is None
                        or observed[0] == "Z"
                        or observed[1] != intent["start_ticks"]
                        or observed[2] != command
                        or token not in (Path("/proc") / str(child.pid) / "environ").read_bytes().split(b"\0")
                    ):
                        raise RuntimeError("Boundary queue acknowledgment differs from its spawned generation")
                    intent["status"] = "acknowledged"
                    write(path, intent)
                    return intent
                if child.poll() is not None:
                    raise RuntimeError("Boundary queue exited before its startup acknowledgment")
                time.sleep(0.1)
            raise TimeoutError("Boundary queue startup is unacknowledged; preserve its child without retry")
        except BaseException as error:
            intent.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            write(path, intent)
            raise


def spawn(campaign: Path, state: dict, phase: str) -> dict:
    """Persist intent before spawning; missing acknowledgments never authorize a second spawn."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("CPU-only CUDA visibility prevents GPU controller spawn")
    require_gpu_environment(campaign)
    target = directory(campaign)
    intent = target / f"{phase}.json"
    if intent.exists():
        raise FileExistsError("Controller already has an intent; inspect its exact generation without retry")
    payload = {"status": "starting", "command": command(campaign, phase), "marker": uuid4().hex}
    write(intent, payload)
    with (target / f"{phase}.log").open("x") as log:
        child = subprocess.Popen(
            payload["command"],
            cwd=read(campaign)["source"],
            env={**os.environ, **environment(campaign), MARKER: payload["marker"]},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    observed = observed_process(child.pid)
    payload.update(pid=child.pid, start_ticks=None if observed is None else observed[1])
    write(intent, payload)
    state["status"] = "validating" if phase == "validate" else "launching"
    return payload


def observe_controller(campaign: Path, phase: str) -> str:
    intent = read(directory(campaign) / f"{phase}.json")
    if "pid" not in intent or intent["start_ticks"] is None:
        return "unknown"
    observed = observed_process(intent["pid"])
    if observed is None:
        return "exited"
    if observed[1] != intent["start_ticks"]:
        raise RuntimeError("Queue controller PID was reused or argv changed; no retry or allocation transfer")
    if observed[0] == "Z":
        return "exited"
    if observed[2] != intent["command"]:
        raise RuntimeError("Queue controller PID was reused or argv changed; no retry or allocation transfer")
    token = f"{MARKER}={intent['marker']}".encode()
    if token not in (Path("/proc") / str(intent["pid"]) / "environ").read_bytes().split(b"\0"):
        raise RuntimeError("Queue controller marker changed")
    return "running"


def validation_ready(campaign: Path) -> bool:
    intent = directory(campaign) / "validate.json"
    if not intent.exists():
        return False
    if observe_controller(campaign, "validate") != "exited":
        return False
    state_path = campaign.parent / "validation-state.json"
    if not state_path.exists():
        raise RuntimeError("Validation controller exited without execution state; preserve intent and log")
    state = read(state_path)
    expected = read(intent)
    if (
        state["controller_pid"] != expected["pid"]
        or state["controller_start_ticks"] != expected["start_ticks"]
        or state["status"] != "passed"
        or type(state.get("keeper_restore")) is not int
        or state["keeper_restore"] != 0
        or state.get("cleanup_exit_code") != 0
        or state.get("exit_code") != 0
        or "cleanup_error" in state
    ):
        raise RuntimeError("Actual boundary validation, owned generation, cleanup or restoration did not pass")
    execution = read(campaign.parent / "execution-binding.json")
    script = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from run_fixed_age_tuning import require_ready
from run_diagonal_diagnostic import tagged_processes
campaign=Path(sys.argv[2])
state=json.loads((campaign.parent/"validation-state.json").read_text())
assert not tagged_processes(state["child_marker"]), "Marked validation children remain live"
require_ready(campaign)
print("actual_boundary_gates_and_cleanup_rederived")
"""
    subprocess.run(
        [PYTHON, "-c", script, execution["helpers"], str(campaign)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return True


def require_launch_acknowledgments(campaign: Path, frozen: dict) -> dict:
    execution = read(campaign.parent / "execution-binding.json")
    launch = read(campaign.parent / "launch-state.json")
    if (
        launch["status"] != "launched"
        or launch["binding_sha256"] != frozen["execution_binding_sha256"]
        or set(launch["workers"]) != set(read(campaign)["families"])
        or set(launch.get("worker_markers", {})) != set(launch["workers"])
        or type(launch.get("monitor_pid")) is not int
        or launch["monitor_pid"] <= 0
        or int((campaign.parent / "monitor/pid").read_text()) != launch["monitor_pid"]
    ):
        raise RuntimeError("Boundary launcher exited without all bound worker and monitor acknowledgments")
    for assignment in execution["plan"]:
        family = assignment["family"]
        worker = launch["workers"][family]
        expected = [
            PYTHON,
            str(Path(execution["helpers"]) / "run_fixed_age_tuning.py"),
            "--campaign",
            str(campaign),
            "--phase",
            "worker",
            "--family",
            family,
        ]
        if (
            worker.get("family") != family
            or worker.get("node") != assignment["node"]
            or worker.get("binding_sha256") != frozen["execution_binding_sha256"]
            or worker.get("worker_marker") != launch["worker_markers"][family]
            or worker.get("command") != expected
            or type(worker.get("pid")) is not int
            or worker["pid"] <= 0
            or not isinstance(worker.get("start_ticks"), str)
            or not worker["start_ticks"]
            or worker.get("status") not in ("preflight", "running", "complete")
        ):
            raise RuntimeError("Boundary worker acknowledgment differs; preserve the partial launch without retry")
    return launch


def advance(campaign: Path) -> dict:
    frozen = binding(campaign)
    if NODE not in subprocess.check_output(["hostname", "-I"], text=True).split():
        raise RuntimeError("Boundary queue must run on its original designated .9 allocation")
    target = directory(campaign)
    state_path = target / "state.json"
    wave = handoff_wave(campaign)
    state = read(state_path) if state_path.exists() else {"status": f"waiting_for_{wave}"}
    if state.get("binding_sha256", sha256(target / "binding.json")) != sha256(target / "binding.json"):
        raise ValueError("Queue execution state belongs to a different binding")
    if state["status"] in ("needs_attention", "launched"):
        return state
    state["binding_sha256"] = sha256(target / "binding.json")
    try:
        if (target / "launch.json").exists():
            live = observe_controller(campaign, "launch")
            state["status"] = "launching" if live == "running" else "unknown_launch" if live == "unknown" else "launched"
            if live == "exited":
                require_launch_acknowledgments(campaign, frozen)
                state["launch_state_sha256"] = sha256(campaign.parent / "launch-state.json")
        elif (target / "validate.json").exists():
            if validation_ready(campaign):
                if any(
                    path.exists()
                    for path in (
                        campaign.parent / "launch-state.json",
                        campaign.parent / "workers",
                        campaign.parent / "runs",
                        campaign.parent / "monitor",
                    )
                ):
                    raise FileExistsError("Unowned boundary launch evidence prevents launcher spawn")
                spawn(campaign, state, "launch")
            else:
                state["status"] = "validating"
        else:
            root = Path(read(campaign)["root"])
            release = read(root / f"run-preparation/{wave}-launch-binding.json")
            snapshot_path = target / f"{wave}-{time.time_ns()}.json"
            observer = "upstream_production.py" if wave == "wave9" else "wave8_release.py"
            analyzer = "analyze_upstream_run.py" if wave == "wave9" else "analyze_wave8.py"
            try:
                subprocess.run(
                    [
                        PYTHON,
                        str(Path(release["helpers"]) / observer),
                        "--root",
                        str(root),
                        "--phase",
                        "snapshot",
                        "--out",
                        str(snapshot_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                state.update(status=f"waiting_for_{wave}", observation_error=f"{type(error).__name__}: {error}")
                write(state_path, state)
                return state
            observed = read(snapshot_path)
            state["predecessor_snapshot"] = str(snapshot_path)
            if observed["binding_sha256"] != frozen["predecessor_release_sha256"]:
                raise ValueError(f"Queue predecessor observation belongs to a different {wave} binding")
            if observed["status"] == "needs_attention":
                raise RuntimeError(f"{wave} predecessor requires attention; no subsequent GPU execution")
            if (
                observed["status"] != "complete"
                or observed["all_workers_terminal"] is not True
                or len(observed["workers"]) != 4
                or {worker["arm"] for worker in observed["workers"]} != set(handoff_arms(campaign))
                or any(worker["status"] != "complete" for worker in observed["workers"])
            ):
                state["status"] = f"waiting_for_{wave}"
            else:
                subprocess.run(
                    [PYTHON, str(Path(release["helpers"]) / analyzer), "--root", str(root)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                cloud_path = root / f"run-preparation/swanlab-upload-{wave}.json"
                if not cloud_path.exists() or read(cloud_path).get("status") != "verified":
                    state["status"] = f"waiting_for_{wave}_upload"
                else:
                    state["predecessor_results"] = (
                        require_completed(root, wave=wave) if wave == "wave9" else require_completed(root)
                    )
                    if any(
                        path.exists()
                        for path in (
                            campaign.parent / "validation-state.json",
                            campaign.parent / "validation-gpu.json",
                            campaign.parent / "validation",
                            campaign.parent / "launch-state.json",
                            campaign.parent / "workers",
                            campaign.parent / "runs",
                            campaign.parent / "monitor",
                        )
                    ):
                        raise FileExistsError("Unowned boundary execution evidence prevents controller spawn")
                    spawn(campaign, state, "validate")
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(state_path, state)
        raise
    write(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "check", "install", "step"), required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.phase == "freeze":
        result = freeze(campaign)
    elif args.phase == "check":
        result = binding(campaign)
    elif args.phase == "install":
        result = install(campaign)
    else:
        with (directory(campaign) / ".step.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = advance(campaign)
    print(json.dumps({"status": result["status"], "campaign": str(campaign)}))


if __name__ == "__main__":
    main()
