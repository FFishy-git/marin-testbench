# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover a completed related command's refused restoration without rerunning it."""

import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

from owned_output_handoff import KEEPER_PID, keeper_observation, marker_processes, observed
from related_allocation import LOCK, MANAGER, MARKER, ROOT, defer_signals, digest, local_node, now, require_no_training


def restore(original: Path, receipt: Path) -> dict:
    if receipt.exists():
        raise FileExistsError(receipt)
    state = json.loads(original.read_text())
    if (
        state["status"] != "needs_attention"
        or state.get("exit_code") != 0
        or state.get("cleanup_exit_code") != 0
        or state.get("keeper_release_complete") is not True
        or state.get("restoration_error") != "RuntimeError: CUDA occupancy prevents keeper restoration"
    ):
        raise ValueError("Recovery requires the specific completed-command occupancy refusal")
    record = {"time": now(), "original": str(original), "original_sha256": digest(original), "status": "starting"}
    node_lock_path = ROOT / "run-preparation" / f".wave7-node-{local_node()}.lock"
    with node_lock_path.open("a") as node_lock, LOCK.open("a") as lock, defer_signals():
        fcntl.flock(node_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if digest(original) != record["original_sha256"] or digest(MANAGER) != state["manager_sha256"]:
            raise ValueError("Original evidence or manager changed")
        for pid in (state["pid"], state["child_pid"], *map(int, state["keeper"]["generation"])):
            process = observed(pid)
            if process and process["status"] not in ("Z", "X", "x"):
                raise RuntimeError("Original command or keeper handle is still live")
        if marker_processes(MARKER, state["child_marker"]):
            raise RuntimeError("Original marked descendants remain")
        require_no_training(keeper_absent=True)
        if KEEPER_PID.exists():
            raise RuntimeError("A keeper pointer already exists; recovery refused")
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip():
            raise RuntimeError("CUDA occupancy remains; recovery refused")
        environment = {key: value for key, value in os.environ.items() if key != MARKER}
        environment.update(GPU_RESOURCE_MANAGER_LOCAL_ONLY="1", GPU_RESOURCE_MANAGER_LOCK_FD=str(lock.fileno()))
        subprocess.run(
            ["bash", str(MANAGER), "start"],
            env=environment,
            pass_fds=(lock.fileno(),),
            check=True,
            timeout=120,
        )
        keeper = keeper_observation()
        if keeper["devices"] != state["keeper"]["devices"]:
            raise RuntimeError("Recovered keeper does not own the original physical GPUs")
        record.update(time=now(), status="complete", keeper_restore=0, restored_keeper=keeper)
        with receipt.open("x") as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = restore(args.original.resolve(), args.receipt.resolve())
    print(json.dumps({"status": result["status"], "keeper_pid": result["restored_keeper"]["pid"]}), flush=True)


if __name__ == "__main__":
    main()
