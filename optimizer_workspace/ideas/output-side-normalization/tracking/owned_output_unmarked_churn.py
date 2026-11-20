# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce unmarked-process environment disappearance against the issued scanner."""

import argparse
import json
import shutil
import tempfile
import types
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from owned_output_helper_churn import MARKER, SOURCE, SOURCE_SHA256, fake_process, source_bytes

UNMARKED_PID = 950
SCOPE = (
    "CPU-only fake-process-tree study using exact issued ownership bytes. "
    "No live /proc changes, cause attribution, acceptance-policy change, GPU or launch authority."
)


def unmarked_process(root: Path, pid: int) -> None:
    fake_process(root, pid)
    (root / str(pid) / "environ").write_bytes(b"UNRELATED_PROCESS=fixture\0")


def trial(exits: int, directory: Path = SOURCE) -> dict:
    """Observe complete-scan exhaustion when an unmarked process disappears before marker reading."""
    sources = source_bytes(directory)
    scan = types.ModuleType("unmarked_churn_issued_scan")
    scan.__file__ = str(directory / "owned_output_handoff.py")
    exec(compile(sources["owned_output_handoff.py"], scan.__file__, "exec"), scan.__dict__)
    retired, returned, failure = [], None, None
    with tempfile.TemporaryDirectory(prefix="owned-output-unmarked-churn-") as temporary:
        proc = Path(temporary)
        scan.PROC_ROOT = proc
        fake_process(proc, 900)
        fake_process(proc, 901)
        unmarked_process(proc, UNMARKED_PID)
        read_bytes = Path.read_bytes

        # Remove the unmarked directory between the scan's process and environment observations.
        def environment_disappearance(path):
            if (
                path.parent.parent == proc
                and path.name == "environ"
                and int(path.parent.name) >= UNMARKED_PID
                and len(retired) < exits
            ):
                unmarked_process(proc, UNMARKED_PID + len(retired) + 1)
                retired.append(int(path.parent.name))
                shutil.rmtree(path.parent)
            return read_bytes(path)

        with patch.object(Path, "read_bytes", environment_disappearance):
            try:
                returned = sorted(scan.marker_processes(*MARKER.split("=", 1)))
            except scan.attribution_handler.OwnershipObservationError as error:
                failure = error.evidence
        remaining = sorted(int(path.name) for path in proc.iterdir())
    return {
        "requested_exits": exits,
        "retired_pids": retired,
        "remaining_pids": remaining,
        "returned_pids": returned,
        "status": "complete_scan_returned" if failure is None else "ownership_refused",
        "failure_stage": None if failure is None else failure["stage"],
        "failure_snapshots": None if failure is None else failure["snapshots"],
        "scope": SCOPE,
    }


def study(directory: Path = SOURCE) -> dict:
    source_bytes(directory)
    return {
        "status": "cpu_unmarked_process_churn_observed",
        "time": datetime.now(UTC).isoformat(),
        "source_directory": str(directory),
        "sources_sha256": SOURCE_SHA256,
        "trials": [trial(exits, directory) for exits in range(5)],
        "scope": SCOPE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = study(args.source)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
