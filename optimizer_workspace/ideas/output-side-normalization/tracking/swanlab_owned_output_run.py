# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import Wave-10 histories after all original workers complete and restore."""

import json
import sys
from pathlib import Path

import swanlab_refresh_upload
from owned_output_run_history import ARMS, WAVE, completed_run, paths, verify_histories


def require_completed_snapshot(root: Path, release: dict) -> dict:
    snapshot = json.loads((root / f"run-preparation/monitor-{WAVE}/latest.json").read_text())
    workers = snapshot["workers"]
    if (
        snapshot["status"] != "complete"
        or snapshot["all_workers_terminal"] is not True
        or snapshot["binding_sha256"] != swanlab_refresh_upload.sha256(paths(root)[0])
        or len(workers) != len(ARMS)
        or {worker["arm"] for worker in workers} != set(ARMS)
        or any(worker["status"] != "complete" for worker in workers)
    ):
        raise RuntimeError("Wave-10 original workers have not completed and restored")
    for worker in workers:
        arm, state = worker["arm"], worker["receipt"]
        node = release["nodes"][ARMS.index(arm)]
        if (
            state["arm"] != arm
            or state["node"] != node
            or state["status"] != "complete"
            or state["release_binding_sha256"] != snapshot["binding_sha256"]
            or any(
                type(state.get(key)) is not int or state[key] != 0
                for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
            )
            or "cleanup_error" in state
            or "error" in state
            or state["command"] != release["commands"][node]
            or state["log_sha256"] != swanlab_refresh_upload.sha256(Path(state["log"]))
            or state["completed_run"] != completed_run(root, arm, release)
        ):
            raise RuntimeError("Wave-10 terminal history, cleanup or restoration evidence changed")
    return snapshot


def main() -> None:
    arguments = sys.argv[1:]
    root = Path(arguments[arguments.index("--root") + 1])
    release = verify_histories(root)
    require_completed_snapshot(root, release)
    swanlab_refresh_upload.WAVES[WAVE] = ARMS
    sys.argv = [sys.argv[0], *arguments, "--wave", WAVE]
    swanlab_refresh_upload.main()


if __name__ == "__main__":
    main()
