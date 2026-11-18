# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Queue one frozen upstream-product diagnostic after the entire upper-LR study."""

import argparse
import fcntl
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

from launch_wave5 import PYTHON, read, sha256
from run_diagonal_diagnostic import keeper_generation, tagged_processes
from run_upstream_outer_gpu import NODE, completed_predecessor, require_result
from validate_wave5 import ENVIRONMENT
from watch_wave5 import write
from watch_wave7_release import observed_process

EXTRA_HELPERS = ("queue_upstream_outer_gpu.py", "queue_upstream_outer_gpu.sh")
QUEUE_MARKER = "OUTPUT_MARIN_UPSTREAM_PROBE_QUEUE"
CONTROLLER_MARKER = "OUTPUT_MARIN_UPSTREAM_PROBE_CONTROLLER"
INTERVAL = 900


def command(root: Path, receipt: Path, out: Path) -> list[str]:
    return [
        PYTHON,
        str(out / "helpers/run_upstream_outer_gpu.py"),
        "--root",
        str(root),
        "--cpu-receipt",
        str(receipt),
        "--out",
        str(out),
        "--phase",
        "run",
    ]


def diagnostic_binding(root: Path, receipt: Path, out: Path) -> dict:
    subprocess.run(
        [*command(root, receipt, out)[:-1], "check"],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return read(out / "binding.json")


def freeze(root: Path, receipt: Path, out: Path) -> dict:
    """Bind a new queue without changing the diagnostic, candidate or predecessor releases."""
    original = diagnostic_binding(root, receipt, out)
    target = out / "queue"
    if target.exists() or any((out / name).exists() for name in ("state.json", "result.json", "diagnostic.log")):
        raise FileExistsError("Upstream probe already has queue/execution evidence; preserve it without refreezing")
    helpers = target / "helpers"
    helpers.mkdir(parents=True)
    archive = target / "source.tar"
    names = (*original["helpers_sha256"], *EXTRA_HELPERS)
    with tarfile.open(archive, "x") as bundle:
        for name in names:
            source = out / "helpers" / name if name in original["helpers_sha256"] else Path(__file__).parent / name
            destination = helpers / name
            destination.write_bytes(source.read_bytes())
            bundle.add(destination, arcname=f"tracking/{name}", recursive=False)
    frozen = {
        "status": "frozen_waiting_entire_upper_lr_study",
        "root": str(root),
        "out": str(out),
        "cpu_receipt": str(receipt),
        "cpu_receipt_sha256": sha256(receipt),
        "diagnostic_binding_sha256": sha256(out / "binding.json"),
        "campaign": original["campaign"],
        "campaign_sha256": original["campaign_sha256"],
        "execution_binding_sha256": original["execution_binding_sha256"],
        "helpers": str(helpers),
        "helpers_sha256": {name: sha256(helpers / name) for name in names},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "node": NODE,
        "interval_seconds": INTERVAL,
        "environment": ENVIRONMENT,
        "command": command(root, receipt, out),
        "policy": (
            "Entire upper-LR study complete, rederived, cloud-verified and freshly restored before one "
            "unchanged diagnostic controller. No keeper operations in queue. No retries after controller "
            "intent, no full-model validation or training authorization."
        ),
    }
    write(target / "binding.json", frozen)
    return frozen


def binding(root: Path, receipt: Path, out: Path) -> dict:
    frozen = read(out / "queue/binding.json")
    original = diagnostic_binding(root, receipt, out)
    helpers = out / "queue/helpers"
    if (
        Path(__file__).resolve() != (helpers / "queue_upstream_outer_gpu.py").resolve()
        or frozen["status"] != "frozen_waiting_entire_upper_lr_study"
        or frozen["root"] != str(root)
        or frozen["out"] != str(out)
        or frozen["cpu_receipt"] != str(receipt)
        or frozen["cpu_receipt_sha256"] != sha256(receipt)
        or frozen["diagnostic_binding_sha256"] != sha256(out / "binding.json")
        or frozen["campaign"] != original["campaign"]
        or frozen["campaign_sha256"] != original["campaign_sha256"]
        or frozen["execution_binding_sha256"] != original["execution_binding_sha256"]
        or frozen["helpers"] != str(helpers)
        or set(frozen["helpers_sha256"]) != set(original["helpers_sha256"]) | set(EXTRA_HELPERS)
        or any(frozen["helpers_sha256"].get(name) != value for name, value in original["helpers_sha256"].items())
        or any(sha256(helpers / name) != value for name, value in frozen["helpers_sha256"].items())
        or frozen["archive"] != str(out / "queue/source.tar")
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["node"] != NODE
        or frozen["interval_seconds"] != INTERVAL
        or frozen["environment"] != ENVIRONMENT
        or frozen["command"] != command(root, receipt, out)
    ):
        raise ValueError("Upstream queue source, diagnostic, campaign or command identity changed")
    return frozen


def install(root: Path, receipt: Path, out: Path) -> dict:
    frozen = binding(root, receipt, out)
    target = out / "queue"
    with (target / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if NODE not in subprocess.check_output(["hostname", "-I"], text=True).split():
            raise RuntimeError("Upstream diagnostic queue must run on its original .139 allocation")
        if any((target / name).exists() for name in ("installation.json", "pid", "queue.log", "stopped_at")):
            raise FileExistsError("Upstream queue installation evidence exists; inspect its generation without retry")
        argv = [
            "bash",
            str(Path(frozen["helpers"]) / "queue_upstream_outer_gpu.sh"),
            str(root),
            str(receipt),
            str(out),
        ]
        intent = {
            "status": "starting",
            "binding_sha256": sha256(target / "binding.json"),
            "command": argv,
            "marker": uuid4().hex,
        }
        write(target / "installation.json", intent)
        try:
            with (target / "queue.log").open("x") as log:
                child = subprocess.Popen(
                    argv,
                    cwd=out,
                    env={**os.environ, **ENVIRONMENT, QUEUE_MARKER: intent["marker"]},
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            observed = observed_process(child.pid)
            intent.update(pid=child.pid, start_ticks=None if observed is None else observed[1])
            write(target / "installation.json", intent)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if (target / "pid").exists():
                    current = observed_process(child.pid)
                    token = f"{QUEUE_MARKER}={intent['marker']}".encode()
                    if (
                        int((target / "pid").read_text()) != child.pid
                        or current is None
                        or current[0] == "Z"
                        or current[1] != intent["start_ticks"]
                        or current[2] != argv
                        or token not in (Path("/proc") / str(child.pid) / "environ").read_bytes().split(b"\0")
                    ):
                        raise RuntimeError("Upstream queue acknowledgment differs from its spawned generation")
                    intent["status"] = "acknowledged"
                    write(target / "installation.json", intent)
                    return intent
                if child.poll() is not None:
                    raise RuntimeError("Upstream queue exited before acknowledgment")
                time.sleep(0.1)
            raise TimeoutError("Upstream queue startup is unacknowledged; preserve its child without retry")
        except BaseException as error:
            intent.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            write(target / "installation.json", intent)
            raise


def observe_controller(out: Path) -> str:
    intent = read(out / "queue/controller.json")
    if "pid" not in intent or intent.get("start_ticks") is None:
        return "unknown"
    observed = observed_process(intent["pid"])
    if observed is None:
        return "exited"
    if observed[1] != intent["start_ticks"]:
        raise RuntimeError("Upstream controller PID reused; no retry or allocation transfer")
    if observed[0] == "Z":
        return "exited"
    token = f"{CONTROLLER_MARKER}={intent['marker']}".encode()
    if observed[2] != intent["command"] or token not in (
        Path("/proc") / str(intent["pid"]) / "environ"
    ).read_bytes().split(b"\0"):
        raise RuntimeError("Upstream controller argv or marker changed")
    return "running"


def spawn(root: Path, receipt: Path, out: Path) -> dict:
    """Persist one intent before spawning the unchanged controller; ambiguity never permits retry."""
    target = out / "queue"
    path = target / "controller.json"
    if path.exists():
        raise FileExistsError("Upstream diagnostic controller intent already exists")
    intent = {
        "status": "starting",
        "command": command(root, receipt, out),
        "marker": uuid4().hex,
        "diagnostic_binding_sha256": sha256(out / "binding.json"),
    }
    write(path, intent)
    with (target / "controller.log").open("x") as log:
        child = subprocess.Popen(
            intent["command"],
            cwd=out / "helpers",
            env={**os.environ, **ENVIRONMENT, CONTROLLER_MARKER: intent["marker"]},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    observed = observed_process(child.pid)
    intent.update(pid=child.pid, start_ticks=None if observed is None else observed[1])
    write(path, intent)
    if observe_controller(out) == "running":
        intent["status"] = "acknowledged"
        write(path, intent)
    return intent


def completed_diagnostic(out: Path, frozen: dict, keeper_file: Path = Path("/root/dl/runtime/gpu_matmul.pid")) -> dict:
    """Require final owned parity evidence, absent children and a currently healthy restored keeper."""
    intent = read(out / "queue/controller.json")
    state = read(out / "state.json")
    result = read(out / "result.json")
    if (
        state.get("controller_pid") != intent["pid"]
        or state.get("controller_start_ticks") != intent["start_ticks"]
        or state.get("status") != "measured_upstream_probe_not_launch_gate"
        or state.get("binding_sha256") != sha256(out / "binding.json")
        or state.get("result_sha256") != sha256(out / "result.json")
        or state.get("helpers_sha256") != frozen["helpers_sha256"]
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or "cleanup_error" in state
        or tagged_processes(state["child_marker"])
    ):
        raise RuntimeError("Upstream diagnostic failed, changed owner or has incomplete cleanup/restoration")
    require_result(frozen, result, state)
    keeper = int(keeper_file.read_text())
    generation = keeper_generation(keeper)
    owned = {pid for pid, _, _ in generation if pid != keeper}
    occupied = {
        int(pid)
        for pid in subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).split()
    }
    if len(owned) != 8 or occupied != owned:
        raise RuntimeError("Upstream diagnostic final keeper generation or eight-GPU occupancy is unhealthy")
    return {
        "state_sha256": sha256(out / "state.json"),
        "result_sha256": sha256(out / "result.json"),
        "keeper_generation": generation,
    }


def advance(root: Path, receipt: Path, out: Path) -> dict:
    frozen = binding(root, receipt, out)
    if NODE not in subprocess.check_output(["hostname", "-I"], text=True).split():
        raise RuntimeError("Upstream diagnostic queue is on the wrong allocation")
    target = out / "queue"
    state_path = target / "state.json"
    digest = sha256(target / "binding.json")
    state = read(state_path) if state_path.exists() else {"status": "waiting_for_upper_lr"}
    if state.get("binding_sha256", digest) != digest:
        raise ValueError("Upstream queue state has a changed binding")
    if state["status"] in ("needs_attention", "measured"):
        return state
    state["binding_sha256"] = digest
    try:
        if (target / "controller.json").exists():
            intent = read(target / "controller.json")
            if (
                intent["command"] != frozen["command"]
                or intent.get("diagnostic_binding_sha256") != frozen["diagnostic_binding_sha256"]
            ):
                raise ValueError("Upstream controller intent differs from its frozen command or diagnostic")
            live = observe_controller(out)
            state["status"] = (
                "running_probe" if live == "running" else "unknown_probe" if live == "unknown" else "measured"
            )
            if live == "exited":
                state["diagnostic"] = completed_diagnostic(out, read(out / "binding.json"))
        else:
            if any(
                (out / name).exists() for name in ("state.json", "result.json", "diagnostic.log", "predecessor.json")
            ):
                raise FileExistsError("Unowned upstream diagnostic evidence prevents controller spawn")
            campaign = Path(frozen["campaign"])
            launch_path = campaign.parent / "launch-state.json"
            validation_path = campaign.parent / "validation-state.json"
            if validation_path.exists() and read(validation_path)["status"] == "needs_attention":
                raise RuntimeError("Upper-LR actual-config validation failed; no following GPU diagnostic")
            if not launch_path.exists():
                state["status"] = "waiting_for_upper_lr_launch"
            elif read(launch_path)["status"] == "needs_attention":
                raise RuntimeError("Upper-LR launch requires attention; no following GPU diagnostic")
            else:
                execution = read(campaign.parent / "execution-binding.json")
                observation = target / f"upper-lr-{time.time_ns()}.json"
                try:
                    subprocess.run(
                        [
                            PYTHON,
                            str(Path(execution["helpers"]) / "run_fixed_age_tuning.py"),
                            "--campaign",
                            str(campaign),
                            "--phase",
                            "snapshot",
                            "--out",
                            str(observation),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                    state.update(status="waiting_for_upper_lr", observation_error=f"{type(error).__name__}: {error}")
                    write(state_path, state)
                    return state
                observed = read(observation)
                state["predecessor_snapshot"] = str(observation)
                if (
                    observed["campaign_sha256"] != frozen["campaign_sha256"]
                    or observed["binding_sha256"] != frozen["execution_binding_sha256"]
                ):
                    raise ValueError("Upper-LR observation belongs to a different campaign/release")
                if observed["status"] == "needs_attention":
                    raise RuntimeError("Upper-LR worker execution/restoration requires attention")
                if observed["status"] != "complete" or observed["all_workers_terminal"] is not True:
                    state["status"] = "waiting_for_upper_lr"
                else:
                    cloud_path = campaign.parent / "swanlab-upload.json"
                    analysis_path = campaign.parent / "analysis.json"
                    if (
                        not cloud_path.exists()
                        or read(cloud_path).get("status") != "verified"
                        or not analysis_path.exists()
                        or read(analysis_path).get("status") != "complete"
                    ):
                        state["status"] = "waiting_for_upper_lr_results"
                    else:
                        state["predecessor"] = completed_predecessor(read(out / "binding.json"))
                        intent = spawn(root, receipt, out)
                        state["status"] = "running_probe" if intent["status"] == "acknowledged" else "unknown_probe"
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(state_path, state)
        raise
    write(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "check", "install", "step"), required=True)
    args = parser.parse_args()
    root, receipt, out = args.root.resolve(), args.cpu_receipt.resolve(), args.out.resolve()
    if args.phase == "freeze":
        result = freeze(root, receipt, out)
    elif args.phase == "check":
        result = binding(root, receipt, out)
    elif args.phase == "install":
        result = install(root, receipt, out)
    else:
        with (out / "queue/.step.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = advance(root, receipt, out)
    print(json.dumps({"status": result["status"], "out": str(out)}))


if __name__ == "__main__":
    main()
