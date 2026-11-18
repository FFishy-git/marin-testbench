# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accept only the exact restored allocations of the preserved pre-update Wave11 failure."""

import argparse
import ctypes
import fcntl
import json
import os
import select
import signal
import subprocess
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from launch_wave5 import KEEPER, NODES, PYTHON, WORKLOAD, read, sha256
from owned_output_handoff import (
    KEEPER_PID,
    keeper_observation,
    marker_processes,
    observed,
    require_absent,
    same_generation,
)
from selected_lr_validation import CHILD_MARKER, CONTROLLER_MARKER, NODE, locations
from watch_wave5 import write
from wave7_release import candidate_inventory, local_node

ARMS = ("muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused")
WORKER_MARKER = "OUTPUT_MARIN_WAVE11_WORKER_ID"
FAILURE = "Wave11 training rank or torchrun child generation differs"
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.syscall.restype = ctypes.c_long
PIDFD_OPEN_X86_64 = 434
MANAGER_LOCK = KEEPER_PID.with_name("gpu_resource_manager.lock")
MANAGER_SOURCE = Path(__file__).with_name("recovery_keeper_manager.sh")


def require_manager_source() -> None:
    if not MANAGER_SOURCE.is_file() or sha256(Path(KEEPER)) != sha256(MANAGER_SOURCE):
        raise RuntimeError("Recovery requires the exact archived serialized keeper manager")


def require_manager_lock(handle: int) -> None:
    actual, expected = os.fstat(handle), MANAGER_LOCK.stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError("Recovery manager descriptor does not identify the node lock")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require_manager_source()


@contextmanager
def keeper_guard():
    """Reserve manager actions through training exit and restoration on this node."""
    with MANAGER_LOCK.open("a") as lock:
        require_manager_lock(lock.fileno())
        yield lock.fileno()


def restore_accepted_keeper(handle: int) -> None:
    """Restore under the worker's inherited lock without reacquiring a different descriptor."""
    require_manager_lock(handle)
    processes = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
    if any("-m torch.distributed.run" in row or "-m experiment.train" in row for row in processes):
        raise RuntimeError("Live training prevents serialized keeper restoration")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("CUDA occupancy prevents serialized keeper restoration")
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if WORKLOAD in (directory / "cmdline").read_bytes().split(b"\0"):
                raise RuntimeError("Existing keeper process prevents serialized restoration")
        except FileNotFoundError:
            continue
    if KEEPER_PID.exists():
        raise RuntimeError("Unexpected keeper pointer prevents serialized restoration")
    subprocess.run(
        ["bash", KEEPER, "start"],
        env=os.environ | {"GPU_RESOURCE_MANAGER_LOCAL_ONLY": "1", "GPU_RESOURCE_MANAGER_LOCK_FD": str(handle)},
        pass_fds=(handle,),
        check=True,
        timeout=120,
    )


def same_keeper(actual: dict, expected: dict) -> bool:
    before = {int(pid): process for pid, process in expected["generation"].items()}
    after = {int(pid): process for pid, process in actual["generation"].items()}
    return (
        actual["pid"] == expected["pid"]
        and actual["devices"] == expected["devices"]
        and set(before) == set(after)
        and all(same_generation(process, after[pid]) for pid, process in before.items())
    )


def released_keeper(compact: dict, accepted: dict) -> bool:
    processes = accepted["generation"].values()
    return (
        compact["pid"] == accepted["pid"]
        and compact["start_ticks"] == accepted["generation"][str(accepted["pid"])]["start_ticks"]
        and sorted(compact["generation"])
        == sorted([process["pid"], process["start_ticks"], process["group"]] for process in processes)
    )


def failure_evidence(root: Path) -> dict[str, str]:
    """Bind the failed one-shot release, all original acknowledgments, logs and terminal monitor."""
    release_path = root / "run-preparation/wave11-launch-binding.json"
    launch_path = root / "run-preparation/wave11-launch-state.json"
    snapshot_path = root / "run-preparation/monitor-wave11/latest.json"
    stopped_path = snapshot_path.parent / "stopped_at"
    release, launch, snapshot = read(release_path), read(launch_path), read(snapshot_path)
    digest = sha256(release_path)
    if (
        release["root"] != str(root)
        or tuple(release["arms"]) != ARMS
        or tuple(release["nodes"]) != NODES
        or launch["status"] != "launched"
        or launch["binding_sha256"] != digest
        or set(launch["workers"]) != set(ARMS)
        or set(launch["worker_markers"]) != set(ARMS)
        or len(set(launch["worker_markers"].values())) != len(ARMS)
        or snapshot["status"] != "needs_attention"
        or snapshot["all_workers_terminal"] is not True
        or snapshot["binding_sha256"] != digest
        or len(snapshot["workers"]) != len(ARMS)
        or {worker["arm"] for worker in snapshot["workers"]} != set(ARMS)
        or not stopped_path.is_file()
        or candidate_inventory(Path(release["candidate"])) != release["files_sha256"]
        or any(sha256(Path(release["helpers"]) / name) != digest for name, digest in release["helpers_sha256"].items())
    ):
        raise RuntimeError("Recovery requires the original four-arm failed release and terminal monitor")
    files = [release_path, launch_path, snapshot_path, stopped_path, Path(release["archive"])]
    if release["archive_sha256"] != sha256(Path(release["archive"])):
        raise RuntimeError("Failed release source archive changed")
    for arm, node in zip(ARMS, NODES, strict=True):
        path = root / f"run-preparation/wave11-workers/{arm}.json"
        intent_path, worker_log = path.with_suffix(".start.json"), path.with_suffix(".log")
        state, intent = read(path), read(intent_path)
        worker = next(item for item in snapshot["workers"] if item["arm"] == arm)
        child_log = root / f"runs/wave11/{arm}.torchrun.log"
        if (
            state["status"] != "needs_attention"
            or state["arm"] != arm
            or state["node"] != node
            or state["out"] != str(root / f"runs/wave11/{arm}")
            or Path(state["out"]).exists()
            or state["release_binding_sha256"] != digest
            or state["command"] != release["commands"][node]
            or state["log"] != str(child_log)
            or state["log_sha256"] != sha256(child_log)
            or FAILURE not in child_log.read_text()
            or any(type(state.get(key)) is not int or state[key] != 1 for key in ("exit_code", "cleanup_exit_code"))
            or type(state.get("keeper_restore")) is not int
            or state["keeper_restore"] != 0
            or "cleanup_error" in state
            or state["keeper_release_attempted"] is not True
            or not state["child_marker"]
            or intent["status"] != "acknowledged"
            or intent["arm"] != arm
            or intent["node"] != node
            or intent["release_binding_sha256"] != digest
            or intent["pid"] != state["worker_pid"]
            or intent["start_ticks"] != state["worker_start_ticks"]
            or intent["worker_marker"] != state["worker_marker"]
            or intent["command"] != state["worker_command"]
            or state["worker_marker"] != launch["worker_markers"][arm]
            or worker["status"] != "needs_attention"
            or worker["receipt"] != state
            or not released_keeper(state["keeper"], release["allocation_by_node"][node]["keeper"])
            or state["restored_keeper"]["devices"] != release["allocation_by_node"][node]["keeper"]["devices"]
        ):
            raise RuntimeError("Recovery failed-worker provenance, pre-update failure or restoration differs")
        files.extend((path, intent_path, worker_log, child_log))
    return {str(path.relative_to(root)): sha256(path) for path in files}


def require_original_binding(root: Path) -> None:
    """Check the failed release with its own frozen helpers, including its untouched CPU/source archives."""
    release = read(root / "run-preparation/wave11-launch-binding.json")
    helpers = Path(release["helpers"])
    command = [
        PYTHON,
        "-P",
        "-c",
        "from pathlib import Path\nimport sys\nfrom selected_lr_run_history import binding\nbinding(Path(sys.argv[1]))",
        str(root),
    ]
    subprocess.run(
        command,
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )


def probe(root: Path, arm: str, manager_lock: int | None = None) -> dict:
    """Observe the exact post-failure keeper without stopping, adopting or replacing a process."""
    if manager_lock is not None:
        require_manager_lock(manager_lock)
        return probe_locked(root, arm)
    with keeper_guard():
        return probe_locked(root, arm)


def probe_locked(root: Path, arm: str) -> dict:
    if arm not in ARMS or local_node() != NODES[ARMS.index(arm)]:
        raise RuntimeError("Recovery handoff requires the original arm's node")
    handle = pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(handle, 0)
    finally:
        os.close(handle)
    identities = failure_evidence(root)
    path = root / f"run-preparation/wave11-workers/{arm}.json"
    state, intent = read(path), read(path.with_suffix(".start.json"))
    require_absent(state["worker_pid"])
    require_absent(state["child_pid"])
    require_absent(intent["starter"]["pid"])
    if arm == ARMS[0]:
        require_absent(read(root / "run-preparation/wave11-launch-state.json")["monitor_pid"])
    if marker_processes(CHILD_MARKER, state["child_marker"]):
        raise RuntimeError("Failed Wave11 training descendants remain")
    keeper = keeper_observation()
    if not same_keeper(keeper, state["restored_keeper"]):
        raise RuntimeError("Recovery keeper differs from the exact post-failure restoration")
    marked = marker_processes(WORKER_MARKER, state["worker_marker"])
    if any(
        pid not in keeper["generation"] or not same_generation(process, keeper["generation"][pid])
        for pid, process in marked.items()
    ):
        raise RuntimeError("Failed Wave11 worker marker exists outside its exact restored keeper")
    if identities != failure_evidence(root) or any(
        not same_generation(process, observed(pid)) for pid, process in keeper["generation"].items()
    ):
        raise RuntimeError("Recovery failure evidence or keeper changed during handoff")
    return {
        "status": "failed_wave11_absent_exact_post_failure_keeper_verified",
        "arm": arm,
        "node": NODES[ARMS.index(arm)],
        "failure_evidence": identities,
        "keeper": keeper,
        "pidfd_signal_zero_verified": True,
    }


def validation_terminal(root: Path) -> dict:
    """Retain historical numerical restoration separately from the current post-failure allocation."""
    if local_node() != NODE:
        raise RuntimeError("Original numerical validation must be inspected on its original node")
    original_path, helpers, state_path, _, receipt_path = locations(root)
    state, intent = read(state_path), read(helpers.parent / "controller.json")
    failure = read(root / "run-preparation/wave11-workers/muont.json")
    if (
        state["status"] != "passed"
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or state["binding_sha256"] != sha256(original_path)
        or state["validation_sha256"] != sha256(receipt_path)
        or any(key in state for key in ("error", "cleanup_error"))
        or intent["status"] != "acknowledged"
        or intent["pid"] != state["controller_pid"]
        or intent["start_ticks"] != state["controller_start_ticks"]
        or not released_keeper(failure["keeper"], state["restored_keeper"])
    ):
        raise RuntimeError("Original numerical validation or keeper succession differs")
    require_absent(state["controller_pid"])
    require_absent(state["child_pid"])
    if marker_processes(CHILD_MARKER, state["child_marker"]):
        raise RuntimeError("Original numerical validation descendants remain")
    current = probe(root, ARMS[0])
    marked = marker_processes(CONTROLLER_MARKER, intent["marker"])
    keeper = current["keeper"]
    if any(
        pid not in keeper["generation"] or not same_generation(process, keeper["generation"][pid])
        for pid, process in marked.items()
    ):
        raise RuntimeError("Original validation marker exists outside the declared keeper succession")
    return {
        "status": "original_validation_absent_declared_post_failure_succession_verified",
        "node": NODE,
        "keeper": state["restored_keeper"],
        "current_keeper": keeper,
        "failure_evidence": current["failure_evidence"],
    }


def pidfd_open(pid: int) -> int:
    """Open a generation-stable handle on the explicitly supported Linux x86-64 nodes."""
    if os.uname().sysname != "Linux" or os.uname().machine != "x86_64":
        raise RuntimeError("Recovery release requires the validated Linux x86-64 pidfd ABI")
    handle = LIBC.syscall(ctypes.c_long(PIDFD_OPEN_X86_64), ctypes.c_int(pid), ctypes.c_uint(0))
    if handle < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return handle


def release_accepted_keeper(accepted: dict, state: dict, state_path: Path, manager_lock: int) -> dict:
    """Signal generation-stable handles for the accepted keeper, never numeric replacement IDs."""
    require_manager_lock(manager_lock)
    if not same_keeper(keeper_observation(), accepted):
        raise RuntimeError("Accepted recovery keeper changed before release")
    generation = {int(pid): process for pid, process in accepted["generation"].items()}
    compact = {
        "pid": accepted["pid"],
        "start_ticks": generation[accepted["pid"]]["start_ticks"],
        "generation": [[pid, process["start_ticks"], process["group"]] for pid, process in generation.items()],
    }
    with ExitStack() as stack:
        handles, pending, poll = {}, set(), select.poll()
        for pid, expected in generation.items():
            handle = pidfd_open(pid)
            stack.callback(os.close, handle)
            if not same_generation(expected, observed(pid)):
                raise RuntimeError("Accepted recovery keeper changed while binding signal handles")
            handles[pid] = handle
            pending.add(handle)
            poll.register(handle, select.POLLIN)
        if not same_keeper(keeper_observation(), accepted):
            raise RuntimeError("Accepted recovery allocation changed before signaling")
        state.update(keeper_release_attempted=True, keeper=compact, keeper_release_complete=False)
        write(state_path, state)
        order = [pid for pid in generation if pid != accepted["pid"]] + [accepted["pid"]]
        for pid in order:
            try:
                signal.pidfd_send_signal(handles[pid], signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 45
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Accepted recovery keeper did not exit")
            for handle, events in poll.poll(max(1, int(remaining * 1000))):
                if not events & select.POLLIN:
                    raise RuntimeError("Accepted recovery process handle has an unexpected poll state")
                pending.discard(handle)
                poll.unregister(handle)
    for pid in generation:
        current = observed(pid)
        if current is not None and current["status"] not in ("Z", "X", "x"):
            raise RuntimeError("Recovery keeper PID was replaced during release")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("CUDA occupancy remains after accepted recovery keeper release")
    if KEEPER_PID.exists():
        if KEEPER_PID.read_text().strip() != str(accepted["pid"]):
            raise RuntimeError("Recovery keeper pointer changed during release")
        KEEPER_PID.unlink()
    state["keeper_release_complete"] = True
    write(state_path, state)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.root.resolve(), args.arm), allow_nan=False))


if __name__ == "__main__":
    main()
