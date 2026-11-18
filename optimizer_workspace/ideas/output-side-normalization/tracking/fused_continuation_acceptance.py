# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate issued continuation CPU evidence using its own archived preparer."""

import json
import os
import subprocess
from pathlib import Path

from fused_continuation_contract import CPU_STATUS, PREPARATION, PYTHON
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import candidate_files, read, sha256
from upstream_production import cpu_command


def require_cpu(root: Path, receipt_path: Path) -> dict:
    """Verify source and archive before invoking the issued CPU-only preparation check."""
    raw = receipt_path.read_bytes()
    cpu = json.loads(raw)
    candidate, helpers = Path(cpu["candidate"]), Path(cpu["helpers"])
    prepared = read(candidate / PREPARATION)
    hashes = cpu["helpers_sha256"]
    if (
        root != root.resolve()
        or candidate != candidate.resolve()
        or helpers != helpers.resolve()
        or helpers.is_symlink()
        or cpu["status"] != CPU_STATUS
        or cpu["root"] != str(root)
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["cuda_visible_devices"] != ""
        or cpu["command"] != cpu_command()
        or cpu["files_sha256"] != candidate_files(candidate)
        or cpu["files_sha256"] != prepared["files_sha256"]
        or cpu["preparation_sha256"] != sha256(candidate / PREPARATION)
        or hashes != prepared["helpers_sha256"]
        or "prepare_fused_continuation.py" not in hashes
        or any(
            Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != value
            for name, value in hashes.items()
        )
        or cpu["log_sha256"] != sha256(Path(cpu["log"]))
        or cpu["archive_sha256"] != sha256(Path(cpu["archive"]))
        or "error" in cpu
    ):
        raise ValueError("Continuation CPU receipt, candidate, issued helpers or outcome differs")
    expected = (
        cpu["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: cpu["preparation_sha256"]}
        | {f"tracking/{name}": value for name, value in hashes.items()}
    )
    require_archive(Path(cpu["archive"]), expected, alias=True)
    checked = subprocess.run(
        [
            PYTHON,
            "-P",
            str(helpers / "prepare_fused_continuation.py"),
            "--root",
            str(root),
            "--candidate",
            str(candidate),
            "--phase",
            "check",
        ],
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        check=True,
        timeout=1500,
    )
    if (
        json.loads(checked.stdout)["status"] != prepared["status"]
        or receipt_path.read_bytes() != raw
        or sha256(candidate / PREPARATION) != cpu["preparation_sha256"]
        or candidate_files(candidate) != cpu["files_sha256"]
        or any(sha256(helpers / name) != value for name, value in hashes.items())
    ):
        raise ValueError("Continuation issued CPU evidence changed during independent authentication")
    return cpu
