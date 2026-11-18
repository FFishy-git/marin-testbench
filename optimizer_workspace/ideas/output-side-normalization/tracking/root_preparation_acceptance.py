# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate root diagnostic CPU evidence with its issued preparer in a fresh process."""

import json
import os
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from prepare_root_preparation_diagnostic import archive_inventory, cpu_command
from prepare_selected_lr_validation import require_archive
from root_preparation_contract import CPU_STATUS, PREPARATION, PYTHON
from selected_lr_release_owner import candidate_files, read, sha256


def require_cpu(root: Path, receipt_path: Path) -> dict:
    raw = receipt_path.read_bytes()
    cpu = json.loads(raw)
    candidate, helpers = Path(cpu["candidate"]), Path(cpu["helpers"])
    prepared = read(candidate / PREPARATION)
    hashes = cpu["helpers_sha256"]
    suites = list(ET.parse(cpu["tests"]).getroot().iter("testsuite"))
    counts = {
        name: sum(int(suite.attrib[name]) for suite in suites) for name in ("tests", "failures", "errors", "skipped")
    }
    if (
        root != root.resolve()
        or candidate != candidate.resolve()
        or helpers != receipt_path.parent / "tracking"
        or helpers.is_symlink()
        or helpers != helpers.resolve()
        or cpu["status"] != CPU_STATUS
        or cpu["root"] != str(root)
        or cpu["cwd"] != str(candidate)
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or cpu["cuda_visible_devices"] != ""
        or cpu["command"] != [*cpu_command(candidate), "--junitxml", cpu["tests"]]
        or cpu["files_sha256"] != candidate_files(candidate)
        or cpu["files_sha256"] != prepared["files_sha256"]
        or cpu["preparation_sha256"] != sha256(candidate / PREPARATION)
        or hashes != prepared["helpers_sha256"]
        or "prepare_root_preparation_diagnostic.py" not in hashes
        or any(
            Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != value
            for name, value in hashes.items()
        )
        or any(cpu[f"{name}_sha256"] != sha256(Path(cpu[name])) for name in ("log", "archive", "tests"))
        or counts["tests"] == 0
        or any(counts[name] for name in ("failures", "errors", "skipped"))
        or "error" in cpu
    ):
        raise ValueError("Root CPU source, issued command/archive or complete test outcome differs")
    require_archive(Path(cpu["archive"]), archive_inventory(prepared, cpu["preparation_sha256"]), alias=True)
    checked = subprocess.run(
        [
            PYTHON,
            "-P",
            str(helpers / "prepare_root_preparation_diagnostic.py"),
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
        json.loads(checked.stdout) != {"status": prepared["status"], "candidate": str(candidate)}
        or receipt_path.read_bytes() != raw
        or candidate_files(candidate) != cpu["files_sha256"]
        or sha256(candidate / PREPARATION) != cpu["preparation_sha256"]
        or any(sha256(helpers / name) != value for name, value in hashes.items())
    ):
        raise ValueError("Root CPU evidence changed during its archived fresh-process check")
    return cpu
