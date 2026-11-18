# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hand off the corrected upstream probe to one separately frozen full-model validator."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

from launch_wave5 import PYTHON, read, sha256
from run_diagonal_diagnostic import keeper_generation, tagged_processes
from validate_wave5 import ENVIRONMENT
from watch_wave5 import write
from watch_wave7_release import observed_process

NODE = "28.216.17.139"
INTERVAL = 900
QUEUE_MARKER = "OUTPUT_MARIN_UPSTREAM_FULL_MODEL_QUEUE"
CONTROLLER_MARKER = "OUTPUT_MARIN_UPSTREAM_FULL_MODEL_CONTROLLER"
EXTRA_HELPERS = ("queue_upstream_full_model.py", "queue_upstream_full_model.sh")


def paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "run-preparation/upstream-full-model-corrected-binding.json",
        root / "controllers/upstream-full-model-corrected-20261117/tracking",
        root / "run-preparation/upstream-full-model-corrected-queue",
    )


def command(root: Path) -> list[str]:
    return [PYTHON, str(paths(root)[1] / "run_upstream_full_model.py"), "--root", str(root), "--phase", "validate"]


def validation_binding(root: Path) -> dict:
    subprocess.run([*command(root)[:-1], "check"], check=True, capture_output=True, text=True, timeout=240)
    return read(paths(root)[0])


def probe_queue_binding(root: Path, validation: dict) -> dict:
    probe = Path(validation["probe"])
    diagnostic = read(probe / "binding.json")
    subprocess.run(
        [
            PYTHON,
            str(probe / "queue/helpers/queue_upstream_outer_gpu.py"),
            "--root",
            str(root),
            "--cpu-receipt",
            diagnostic["cpu_receipt"],
            "--out",
            str(probe),
            "--phase",
            "check",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return read(probe / "queue/binding.json")


def freeze(root: Path) -> dict:
    """Keep both existing releases unchanged and bind a new one-shot handoff."""
    validation = validation_binding(root)
    probe_queue = probe_queue_binding(root, validation)
    binding_path, _, directory = paths(root)
    if directory.exists():
        raise FileExistsError("Full-model queue evidence exists; preserve it without refreezing")
    helpers = directory / "helpers"
    helpers.mkdir(parents=True, exist_ok=False)
    names = (*validation["helpers_sha256"], *EXTRA_HELPERS)
    if len(names) != len(set(names)):
        raise ValueError("Full-model handoff helper inventory has duplicate names")
    archive = directory / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in names:
            source = (
                Path(validation["helpers"]) / name
                if name in validation["helpers_sha256"]
                else Path(__file__).parent / name
            )
            shutil.copy2(source, helpers / name)
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    frozen = {
        "status": "frozen_waiting_owned_probe_completion",
        "root": str(root),
        "node": NODE,
        "interval_seconds": INTERVAL,
        "environment": ENVIRONMENT,
        "validation_binding_sha256": sha256(binding_path),
        "probe": validation["probe"],
        "probe_queue_binding_sha256": sha256(Path(validation["probe"]) / "queue/binding.json"),
        "probe_queue_helpers_sha256": probe_queue["helpers_sha256"],
        "helpers": str(helpers),
        "helpers_sha256": {name: sha256(helpers / name) for name in names},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "command": command(root),
        "policy": (
            "Original synthetic queue terminal and measured; owned result, marked-child cleanup, healthy keeper "
            "and entire upper-LR study rederived/cloud-verified/restored before one unchanged full-model validator. "
            "Queue never performs keeper operations. Persisted launch intents prohibit retries. No production release."
        ),
    }
    write(directory / "binding.json", frozen)
    return frozen


def binding(root: Path) -> dict:
    validation = validation_binding(root)
    probe_queue = probe_queue_binding(root, validation)
    validation_path, _, directory = paths(root)
    frozen = read(directory / "binding.json")
    helpers = directory / "helpers"
    if (
        Path(__file__).resolve() != (helpers / "queue_upstream_full_model.py").resolve()
        or frozen["status"] != "frozen_waiting_owned_probe_completion"
        or frozen["root"] != str(root)
        or frozen["node"] != NODE
        or frozen["interval_seconds"] != INTERVAL
        or frozen["environment"] != ENVIRONMENT
        or frozen["validation_binding_sha256"] != sha256(validation_path)
        or frozen["probe"] != validation["probe"]
        or frozen["probe_queue_binding_sha256"] != sha256(Path(frozen["probe"]) / "queue/binding.json")
        or frozen["probe_queue_helpers_sha256"] != probe_queue["helpers_sha256"]
        or frozen["helpers"] != str(helpers)
        or set(frozen["helpers_sha256"]) != set(validation["helpers_sha256"]) | set(EXTRA_HELPERS)
        or any(frozen["helpers_sha256"][name] != digest for name, digest in validation["helpers_sha256"].items())
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or frozen["archive"] != str(directory / "source.tar")
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["command"] != command(root)
    ):
        raise ValueError("Full-model handoff source, predecessor or command identity changed")
    return frozen


def observe(intent: dict, marker_name: str) -> str:
    """Distinguish unknown startup from a terminal generation without authorizing retries."""
    if type(intent.get("pid")) is not int or not isinstance(intent.get("start_ticks"), str):
        return "unknown"
    observed = observed_process(intent["pid"])
    if observed is None:
        return "exited"
    if observed[1] != intent["start_ticks"]:
        raise RuntimeError("Full-model handoff PID reused; allocation transfer and retry denied")
    if observed[0] == "Z":
        return "exited"
    token = f"{marker_name}={intent['marker']}".encode()
    try:
        environment = (Path("/proc") / str(intent["pid"]) / "environ").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return "unknown"
    if observed[2] != intent["command"] or token not in environment:
        raise RuntimeError("Full-model handoff live process command or marker differs")
    return "running"


def install(root: Path) -> dict:
    frozen = binding(root)
    directory = paths(root)[2]
    with (directory / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_node()
        if any(
            (directory / name).exists()
            for name in ("installation.json", "pid", "queue.log", "stopped_at", "controller.json", "state.json")
        ):
            raise FileExistsError("Full-model handoff installation evidence exists; do not install again")
        intent = {
            "status": "starting",
            "command": ["bash", str(Path(frozen["helpers"]) / "queue_upstream_full_model.sh"), str(root)],
            "marker": uuid4().hex,
            "binding_sha256": sha256(directory / "binding.json"),
        }
        write(directory / "installation.json", intent)
        try:
            with (directory / "queue.log").open("x") as log:
                child = subprocess.Popen(
                    intent["command"],
                    cwd=directory,
                    env=os.environ | ENVIRONMENT | {QUEUE_MARKER: intent["marker"]},
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            observed = observed_process(child.pid)
            intent.update(pid=child.pid, start_ticks=None if observed is None else observed[1])
            write(directory / "installation.json", intent)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if (directory / "pid").exists() and intent["start_ticks"] is not None:
                    if int((directory / "pid").read_text()) != child.pid or observe(intent, QUEUE_MARKER) != "running":
                        raise RuntimeError("Full-model queue acknowledgment differs from its spawned generation")
                    intent["status"] = "acknowledged"
                    write(directory / "installation.json", intent)
                    return intent
                if child.poll() is not None:
                    raise RuntimeError("Full-model queue exited before acknowledgment")
                time.sleep(0.1)
            raise TimeoutError("Full-model queue startup unacknowledged; preserve its child without retry")
        except BaseException as error:
            intent.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            write(directory / "installation.json", intent)
            raise


def require_node() -> None:
    if NODE not in subprocess.check_output(["hostname", "-I"], text=True).split():
        raise RuntimeError("Full-model handoff must run on the original .139 allocation")


def prerequisite(root: Path, frozen: dict) -> dict | None:
    """Wait for the original queue to exit, then let both frozen owners rederive their evidence."""
    probe = Path(frozen["probe"])
    directory = probe / "queue"
    installation = read(directory / "installation.json")
    original = read(directory / "binding.json")
    expected_queue = [
        "bash",
        str(Path(original["helpers"]) / "queue_upstream_outer_gpu.sh"),
        str(root),
        original["cpu_receipt"],
        str(probe),
    ]
    if (
        installation.get("status") != "acknowledged"
        or installation.get("binding_sha256") != frozen["probe_queue_binding_sha256"]
        or installation.get("command") != expected_queue
    ):
        raise RuntimeError("Original upstream probe queue installation identity differs")
    state = read(directory / "state.json")
    if state.get("binding_sha256") != frozen["probe_queue_binding_sha256"] or state["status"] == "needs_attention":
        raise RuntimeError("Original upstream probe queue failed or changed binding")
    live = observe(installation, "OUTPUT_MARIN_UPSTREAM_PROBE_QUEUE")
    if live != "exited":
        return None
    if state["status"] != "measured" or not (directory / "stopped_at").exists():
        raise RuntimeError("Original probe queue exited without a measured completion")
    script = """
import json,sys
from pathlib import Path
root,probe=Path(sys.argv[2]),Path(sys.argv[3])
sys.path.insert(0,str(probe/"queue/helpers"))
from queue_upstream_outer_gpu import binding,completed_diagnostic,observe_controller
original=json.load(open(probe/"binding.json"))
binding(root,Path(original["cpu_receipt"]),probe)
if observe_controller(probe)!="exited":
    raise RuntimeError("Original probe controller remains live or unacknowledged")
diagnostic=completed_diagnostic(probe,original)
sys.path.insert(0,sys.argv[1])
from run_upstream_full_model import binding as validation_binding,require_probe
frozen=validation_binding(root)
print(json.dumps({"diagnostic":diagnostic,"rederived":require_probe(root,frozen)},allow_nan=False))
"""
    result = subprocess.run(
        [PYTHON, "-c", script, str(paths(root)[1]), str(root), str(probe)],
        check=True,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    evidence = json.loads(result.stdout)
    # Rederivation may be long; a restarted predecessor must not receive an allocation handoff.
    if observe(installation, "OUTPUT_MARIN_UPSTREAM_PROBE_QUEUE") != "exited":
        raise RuntimeError("Original upstream queue changed during prerequisite reconstruction")
    return evidence


def spawn(root: Path, frozen: dict) -> dict:
    directory = paths(root)[2]
    intent_path = directory / "controller.json"
    evidence = (
        root / "run-preparation/upstream-full-model-corrected-state.json",
        root / "run-preparation/validation-upstream-full-model-corrected-gpu.json",
        root / "run-preparation/validation-upstream-full-model-corrected-gpu.log",
        root / "validation/upstream_full_model_corrected",
    )
    if intent_path.exists() or any(path.exists() for path in evidence):
        raise FileExistsError("Full-model controller already has intent or execution evidence; no retry")
    intent = {
        "status": "starting",
        "command": frozen["command"],
        "marker": uuid4().hex,
        "validation_binding_sha256": frozen["validation_binding_sha256"],
        "queue_binding_sha256": sha256(directory / "binding.json"),
    }
    write(intent_path, intent)
    with (directory / "controller.log").open("x") as log:
        child = subprocess.Popen(
            intent["command"],
            cwd=paths(root)[1],
            env=os.environ | ENVIRONMENT | {CONTROLLER_MARKER: intent["marker"]},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    observed = observed_process(child.pid)
    intent.update(pid=child.pid, start_ticks=None if observed is None else observed[1])
    write(intent_path, intent)
    if observe(intent, CONTROLLER_MARKER) == "running":
        intent["status"] = "acknowledged"
        write(intent_path, intent)
    return intent


def completed(
    root: Path, frozen: dict, intent: dict, keeper_file: Path = Path("/root/dl/runtime/gpu_matmul.pid")
) -> dict:
    state_path = root / "run-preparation/upstream-full-model-corrected-state.json"
    receipt_path = root / "run-preparation/validation-upstream-full-model-corrected-gpu.json"
    state = read(state_path)
    if (
        state.get("status") != "passed"
        or state.get("controller_pid") != intent["pid"]
        or state.get("controller_start_ticks") != intent["start_ticks"]
        or state.get("binding_sha256") != frozen["validation_binding_sha256"]
        or state.get("validation_sha256") != sha256(receipt_path)
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or "cleanup_error" in state
        or tagged_processes(state["child_marker"])
    ):
        raise RuntimeError("Full-model validation failed, changed owner or incomplete cleanup/restoration")
    script = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from run_upstream_full_model import binding,require_validation
root=Path(sys.argv[2])
result=require_validation(root,binding(root))
print(json.dumps({"checks":result["checks"]}))
"""
    result = subprocess.run(
        [PYTHON, "-c", script, str(paths(root)[1]), str(root)],
        check=True,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    keeper = int(keeper_file.read_text())
    generation = keeper_generation(keeper)
    occupied = {
        int(pid)
        for pid in subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).split()
    }
    owned = {pid for pid, _, _ in generation if pid != keeper}
    if len(owned) != 8 or occupied != owned:
        raise RuntimeError("Full-model validation restored keeper or eight-GPU occupancy differs")
    if observe(intent, CONTROLLER_MARKER) != "exited" or tagged_processes(state["child_marker"]):
        raise RuntimeError("Full-model controller or marked children changed during final reconstruction")
    return {
        "state_sha256": sha256(state_path),
        "receipt_sha256": sha256(receipt_path),
        "checks": json.loads(result.stdout)["checks"],
        "keeper_generation": generation,
    }


def advance(root: Path) -> dict:
    frozen = binding(root)
    require_node()
    directory = paths(root)[2]
    state_path = directory / "state.json"
    digest = sha256(directory / "binding.json")
    state = read(state_path) if state_path.exists() else {"status": "waiting_for_probe"}
    if state.get("binding_sha256", digest) != digest:
        raise ValueError("Full-model handoff state differs from its binding")
    if state["status"] in ("needs_attention", "validated"):
        return state
    state["binding_sha256"] = digest
    try:
        intent_path = directory / "controller.json"
        if intent_path.exists():
            intent = read(intent_path)
            if (
                intent["command"] != frozen["command"]
                or intent.get("validation_binding_sha256") != frozen["validation_binding_sha256"]
                or intent.get("queue_binding_sha256") != digest
            ):
                raise ValueError("Full-model controller intent differs from its frozen command or release")
            live = observe(intent, CONTROLLER_MARKER)
            state["status"] = "validating" if live == "running" else "unknown_controller"
            if live == "exited":
                try:
                    state["validation"] = completed(root, frozen, intent)
                except subprocess.TimeoutExpired as error:
                    state.update(status="unknown_controller", observation_error=str(error))
                    write(state_path, state)
                    return state
                state["status"] = "validated"
        else:
            try:
                predecessor = prerequisite(root, frozen)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                state.update(status="waiting_for_probe", observation_error=str(error))
                write(state_path, state)
                return state
            if predecessor is None:
                state["status"] = "waiting_for_probe"
            else:
                state["predecessor"] = predecessor
                write(state_path, state)
                intent = spawn(root, frozen)
                state["status"] = "validating" if intent["status"] == "acknowledged" else "unknown_controller"
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(state_path, state)
        raise
    write(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "check", "install", "step"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "freeze":
        result = freeze(root)
    elif args.phase == "check":
        result = binding(root)
    elif args.phase == "install":
        result = install(root)
    else:
        with (paths(root)[2] / ".step.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = advance(root)
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
