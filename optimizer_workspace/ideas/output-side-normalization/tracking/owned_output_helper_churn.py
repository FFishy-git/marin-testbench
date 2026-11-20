# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce bounded helper churn against the unchanged issued ownership scan."""

import argparse
import hashlib
import json
import shutil
import tempfile
import types
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/"
    "controllers/ns-training-path-recheck-exit-20261119/tracking"
)
SOURCE_SHA256 = {
    "owned_output_handoff.py": "7c653b0a08e3159599ef00ba4767ef3291b7d581744581d000ca14c3a6ad4e24",
    "owned_output_recheck_exit.py": "59bef187d102f41367062437c8b15f563a7268d6ad5b233c3af49d43773e60f3",
    "owned_output_missing_environment.py": "06c18d6387607339ceaa320b8ae4a156c172f0054b93fb32c5e2b67b1feb7dde",
    "owned_output_exit_transition.py": "e3ae48eea72cd3d9828f06c407551b8f316716498994f54b9d5bb77a55fa9a59",
    "owned_output_exit_drain.py": "981deaf0ec73f43c69e64c0409212100afafb0929e6c0cb11ccf8892edd17db6",
    "owned_output_attribution.py": "c999637ad12729d767f2d9539e7adeb359fd38260f6e09b089b35d00be235653",
}
MARKER = "HELPER_CHURN_STUDY=owned"
SCOPE = (
    "CPU fake-process-tree experiment against exact issued scan bytes. "
    "No live /proc mutation, numerical acceptance, GPU launch or causal attribution."
)


def source_bytes(directory: Path) -> dict[str, bytes]:
    """Authenticate the retained ownership sources before loading the scan."""
    result = {}
    for name, expected in SOURCE_SHA256.items():
        path = directory / name
        payload = path.read_bytes()
        if path.is_symlink() or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Helper-churn issued source differs: {name}")
        result[name] = payload
    return result


def fake_process(root: Path, pid: int) -> None:
    directory = root / str(pid)
    directory.mkdir()
    fields = ["0"] * 21
    fields[:4] = ["S", "1", str(pid), str(pid)]
    fields[19:21] = [str(pid), "4096"]
    (directory / "stat").write_text(f"{pid} (helper churn fixture) {' '.join(fields)}")
    (directory / "cmdline").write_bytes(b"fixture-helper\0")
    (directory / "environ").write_bytes((MARKER + "\0").encode())


def trial(exits: int, directory: Path = SOURCE) -> dict:
    """Observe the scan result while new marked helpers replace proven absent helpers."""
    sources = source_bytes(directory)
    scan = types.ModuleType("helper_churn_issued_scan")
    scan.__file__ = str(directory / "owned_output_handoff.py")
    exec(compile(sources["owned_output_handoff.py"], scan.__file__, "exec"), scan.__dict__)
    if scan.MARKER_SCAN_ATTEMPTS != 3:
        raise ValueError("Helper-churn issued scan budget differs")
    retired, returned = [], None
    error = None
    with tempfile.TemporaryDirectory(prefix="owned-output-helper-churn-") as temporary:
        proc = Path(temporary)
        scan.PROC_ROOT = proc
        fake_process(proc, 900)
        fake_process(proc, 901)
        read_bytes = Path.read_bytes

        # Remove a helper after reading its marker, then expose a new helper to the next scan.
        def environment_exit(path):
            payload = read_bytes(path)
            if (
                path.parent.parent == proc
                and path.name == "environ"
                and path.parent.name != "901"
                and len(retired) < exits
            ):
                replacement = 902 + len(retired)
                fake_process(proc, replacement)
                retired.append(int(path.parent.name))
                shutil.rmtree(path.parent)
            return payload

        with patch.object(Path, "read_bytes", environment_exit):
            try:
                returned = sorted(scan.marker_processes("HELPER_CHURN_STUDY", "owned"))
            except scan.attribution_handler.OwnershipObservationError as failure:
                error = failure.evidence
        remaining = sorted(int(path.name) for path in proc.iterdir())
    return {
        "requested_exits": exits,
        "retired_pids": retired,
        "remaining_pids": remaining,
        "returned_pids": returned,
        "status": "complete_scan_returned" if error is None else "ownership_refused",
        "failure_stage": None if error is None else error["stage"],
        "failure_snapshots": None if error is None else error["snapshots"],
        "scope": SCOPE,
    }


def study(directory: Path = SOURCE) -> dict:
    return {
        "status": "cpu_helper_churn_observed",
        "time": datetime.now(UTC).isoformat(),
        "source_directory": str(directory),
        "sources_sha256": SOURCE_SHA256,
        "scan_attempts": 3,
        "trials": [trial(exits, directory) for exits in range(5)],
        "scope": SCOPE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    print(json.dumps(study(args.source), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
