# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct completed Wave12 results and exact restored allocations before profiling."""

import json
import math
import os
import shlex
import subprocess
from pathlib import Path

from analyze_upstream_snapshot import verify_production_binding
from launch_wave5 import NODES, PYTHON, read, sha256
from selected_lr_stage_profile_contract import ARMS


def frozen_result(python: str, helpers: Path, script: str, arguments: list[str]) -> dict:
    checked = subprocess.run(
        [python, "-P", "-c", script, *arguments],
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        check=True,
        timeout=1500,
    )
    return json.loads(checked.stdout)


def require_completed_wave12(root: Path) -> dict:
    """Require original-worker exit, five-window results and full-resolution verified import."""
    release, release_digest = verify_production_binding(root, "wave12")
    helpers = Path(release["helpers"])
    saved_path = root / "run-preparation/analysis-wave12.json"
    cloud_path = root / "run-preparation/swanlab-upload-wave12.json"
    cloud = read(cloud_path)
    actual = frozen_result(
        PYTHON,
        helpers,
        "import json,sys;from pathlib import Path;"
        "from selected_lr_run_history import binding,completed_run;"
        "from analyze_selected_lr_run import analyze;"
        "root=Path(sys.argv[1]);release=binding(root);"
        "print(json.dumps({'analysis':analyze(root),"
        "'histories':{arm:completed_run(root,arm,release) for arm in release['arms']}},allow_nan=False))",
        [str(root)],
    )
    saved = read(saved_path)
    if (
        tuple(release["arms"]) != ARMS
        or tuple(release["nodes"]) != NODES
        or actual["analysis"] != {key: value for key, value in saved.items() if key != "time"}
        or saved["source_release_sha256"] != release_digest
        or cloud["status"] != "verified"
        or cloud["visibility"] != "PRIVATE"
        or cloud["project"] != "siyuc3141/output-side-normalization-marin"
        or set(cloud["runs"]) != set(ARMS)
        or any(
            not item["final_comparison_eligible"]
            or len(item["training_windows"]) != 5
            or len(item["evaluations"]["selection_loss"]["shared_tokens"]) != 22
            for analysis in saved["analyses"]
            for item in analysis["comparisons"].values()
        )
    ):
        raise RuntimeError("Wave12 full-budget analysis or verified four-arm cloud import remains pending")
    columns = frozen_result(
        str(root / ".venv-logging/bin/python"),
        helpers,
        "import json,sys;from pathlib import Path;"
        "from swanlab_refresh_upload import load_history,expected_columns;"
        "root=Path(sys.argv[1]);arms=json.loads(sys.argv[2]);"
        "print(json.dumps({arm:{'point_counts':{k:len(v) for k,v in expected_columns("
        "load_history(root,'wave12',arm)).items()},'final_metrics':"
        "load_history(root,'wave12',arm).metrics[-1]} for arm in arms},allow_nan=False))",
        [str(root), json.dumps(ARMS)],
    )
    workers = {}
    for arm, node in zip(ARMS, NODES, strict=True):
        command = [
            PYTHON,
            "-P",
            str(helpers / "selected_lr_production.py"),
            "--root",
            str(root),
            "--phase",
            "probe",
            "--arm",
            arm,
        ]
        response = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                f"root@{node}",
                shlex.join(["env", "CUDA_VISIBLE_DEVICES=", f"PYTHONPATH={helpers}", *command]),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=900,
        )
        observed = json.loads(response.stdout.splitlines()[-1])
        state = observed["receipt"]
        local, imported = actual["histories"][arm], cloud["runs"][arm]
        verification = imported["verification"]
        if (
            observed["status"] != "complete"
            or state["status"] != "complete"
            or state["arm"] != arm
            or state["node"] != node
            or state["completed_run"] != local
            or state["command"] != release["commands"][node]
            or state["release_binding_sha256"] != release_digest
            or any(
                type(state.get(key)) is not int or state[key] != 0
                for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
            )
            or any(key in state for key in ("error", "cleanup_error"))
            or imported["status"] != "verified"
            or any(
                imported[key] != local[key]
                for key in ("manifest_sha256", "metrics_sha256", "checkpoint_receipt_sha256", "checkpoint_rank_bytes")
            )
            or verification["state"] != "FINISHED"
            or verification["configuration_verified"] is not True
            or verification["token_axis_verified"] is not True
            or verification["integer_values_exact"] is not True
            or verification["csv_export_max_ulp"] != 2
            or verification["point_counts"] != columns[arm]["point_counts"]
            or verification["final_metrics"] != columns[arm]["final_metrics"]
            or set(verification["serialization_errors"]) != set(verification["point_counts"])
            or any(
                not math.isfinite(error[key]) or error[key] < 0 or (key == "max_ulp" and error[key] > 2)
                for error in verification["serialization_errors"].values()
                for key in ("max_absolute", "max_ulp")
            )
        ):
            raise RuntimeError("Wave12 original completion, restoration or full-resolution cloud identity differs")
        workers[arm] = state
    if sha256(root / "run-preparation/wave12-launch-binding.json") != release_digest:
        raise RuntimeError("Wave12 release changed during completed-wave reconstruction")
    return {
        "status": "wave12_completed_original_workers_restored_analyzed_cloud_verified",
        "release_sha256": release_digest,
        "analysis_sha256": sha256(saved_path),
        "cloud_sha256": sha256(cloud_path),
        "workers": workers,
    }
