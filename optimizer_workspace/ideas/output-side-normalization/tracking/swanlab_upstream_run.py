# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import private full-resolution Wave-9 histories only after verified worker restoration."""

import json
import sys
from pathlib import Path

import swanlab_refresh_upload
from upstream_run_history import ARMS, completed_run, paths, verify_histories


def require_completed_snapshot(root: Path, release: dict) -> dict:
    """Bind saved terminal worker evidence to all four current full histories."""
    snapshot = json.loads((root / "run-preparation/monitor-wave9/latest.json").read_text())
    workers = snapshot["workers"]
    if (
        snapshot["status"] != "complete"
        or snapshot["all_workers_terminal"] is not True
        or snapshot["binding_sha256"] != swanlab_refresh_upload.sha256(paths(root)[0])
        or len(workers) != len(ARMS)
        or {worker["arm"] for worker in workers} != set(ARMS)
        or any(worker["status"] != "complete" for worker in workers)
    ):
        raise RuntimeError("Wave-9 worker cleanup/restoration remains incomplete; no cloud import")
    for worker in workers:
        arm = worker["arm"]
        node = release["nodes"][ARMS.index(arm)]
        state = worker["receipt"]
        if (
            state["arm"] != arm
            or state["node"] != node
            or state["release_binding_sha256"] != snapshot["binding_sha256"]
            or state["status"] != "complete"
            or any(
                type(state.get(key)) is not int or state[key] != 0
                for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
            )
            or "cleanup_error" in state
            or state["command"] != release["commands"][node]
            or state["log_sha256"] != swanlab_refresh_upload.sha256(Path(state["log"]))
            or state["completed_run"] != completed_run(root, arm, release)
        ):
            raise RuntimeError("Wave-9 terminal worker history, command or restoration evidence changed")
    return snapshot


def main() -> None:
    arguments = sys.argv[1:]
    index = arguments.index("--root")
    root = Path(arguments[index + 1])
    release = verify_histories(root)
    require_completed_snapshot(root, release)
    swanlab_refresh_upload.WAVES["wave9"] = ARMS
    sys.argv = [sys.argv[0], *arguments, "--wave", "wave9"]
    swanlab_refresh_upload.main()


if __name__ == "__main__":
    main()
