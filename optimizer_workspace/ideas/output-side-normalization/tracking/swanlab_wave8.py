# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import Wave-8 full histories only after all workers and keeper restorations are verified."""

import json
import sys
from pathlib import Path

import swanlab_refresh_upload
from wave8_history import ARMS, completed_run, verify_histories


def main() -> None:
    arguments = sys.argv[1:]
    index = arguments.index("--root")
    root = Path(arguments[index + 1])
    release = verify_histories(root)
    snapshot = json.loads((root / "run-preparation/monitor-wave8/latest.json").read_text())
    if (
        snapshot["status"] != "complete"
        or snapshot["all_workers_terminal"] is not True
        or snapshot["binding_sha256"]
        != swanlab_refresh_upload.sha256(root / "run-preparation/wave8-launch-binding.json")
        or {worker["arm"] for worker in snapshot["workers"]} != set(ARMS)
        or any(worker["status"] != "complete" for worker in snapshot["workers"])
    ):
        raise RuntimeError("Wave-8 worker cleanup/restoration remains incomplete; no cloud import")
    for arm in ARMS:
        completed_run(root, arm, release)
    swanlab_refresh_upload.WAVES["wave8"] = ARMS
    sys.argv = [sys.argv[0], *arguments, "--wave", "wave8"]
    swanlab_refresh_upload.main()


if __name__ == "__main__":
    main()
