# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload complete Wave11 scalar histories after original-worker restoration."""

import json
import sys
from pathlib import Path

import swanlab_refresh_upload
from selected_lr_run_history import ARMS, WAVE, completed_run, paths, verify_histories


def require_completed_snapshot(root: Path, release: dict) -> dict:
    snapshot = json.loads((root / f"run-preparation/monitor-{WAVE}/latest.json").read_text())
    workers = snapshot["workers"]
    if (
        snapshot["status"] != "complete"
        or snapshot["all_workers_terminal"] is not True
        or snapshot["binding_sha256"] != swanlab_refresh_upload.sha256(paths(root)[0])
        or len(workers) != len(ARMS)
        or {worker["arm"] for worker in workers} != set(ARMS)
    ):
        raise RuntimeError("Wave11 original workers have not completed and restored")
    for worker in workers:
        state, arm = worker["receipt"], worker["arm"]
        node = release["nodes"][ARMS.index(arm)]
        if (
            worker["status"] != "complete"
            or state["status"] != "complete"
            or state["arm"] != arm
            or state["node"] != node
            or state["release_binding_sha256"] != snapshot["binding_sha256"]
            or any(
                type(state.get(key)) is not int or state[key] != 0
                for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
            )
            or any(key in state for key in ("error", "cleanup_error"))
            or state["command"] != release["commands"][node]
            or state["log_sha256"] != swanlab_refresh_upload.sha256(Path(state["log"]))
            or state["completed_run"] != completed_run(root, arm, release)
        ):
            raise RuntimeError("Wave11 completed history or cleanup/restoration evidence differs")
    analysis = json.loads((root / "run-preparation/analysis-wave11.json").read_text())
    if (
        analysis["wave"] != WAVE
        or analysis["source_release_sha256"] != snapshot["binding_sha256"]
        or any(
            not item["final_comparison_eligible"] or len(item["training_windows"]) != 5
            for result in analysis["analyses"]
            for item in result["comparisons"].values()
        )
        or len(analysis["analyses"]) != 1
        or set(analysis["analyses"][0]["comparisons"]) != {f"{WAVE}/{arm}" for arm in ARMS}
    ):
        raise RuntimeError("Wave11 complete five-window analysis is not available")
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
