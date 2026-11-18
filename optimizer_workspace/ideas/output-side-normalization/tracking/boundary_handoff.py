# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require completed frozen loss/cloud/process evidence before upper-boundary GPU execution."""

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Literal

from launch_wave5 import NODES, PYTHON, read, sha256

ARMS = ("muont", "outer_leftdiag", "outer_leftdiag_fused", "outer_leftdiag_combined_fused")
WINDOW_TOKENS = 536870912
TOTAL_TOKENS = 2684354560
EVALUATION_TOKENS = [step * 524288 for step in (*range(1, 5120, 250), 5120)]


def release(root: Path, *, wave: Literal["wave8", "wave9"] = "wave8") -> dict:
    """Check the selected predecessor's original helper and archive identity."""
    if wave not in ("wave8", "wave9"):
        raise ValueError("Boundary handoff supports only literal wave8 or wave9")
    path = root / f"run-preparation/{wave}-launch-binding.json"
    frozen = read(path)
    helpers = root / f"controllers/{wave}-launch-20261117/tracking"
    if (
        frozen["root"] != str(root)
        or frozen["status"] != "frozen_validated_launch_release"
        or frozen["helpers"] != str(helpers)
        or (wave == "wave8" and tuple(frozen["arms"]) != ARMS)
        or tuple(frozen["nodes"]) != NODES
        or (
            wave == "wave9"
            and not {
                "upstream_validation.py",
                "upstream_production.py",
                "analyze_upstream_run.py",
                "upstream_run_history.py",
            }
            <= frozen["helpers_sha256"].keys()
        )
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or sha256(Path(frozen["archive"])) != frozen["archive_sha256"]
    ):
        raise ValueError(f"Boundary-study Wave-{wave[-1]} predecessor release or helpers changed")
    if wave == "wave9":
        # Wave-8-only archives do not contain this optional observation contract.
        spec = importlib.util.spec_from_file_location("upstream_validation", helpers / "upstream_validation.py")
        assert spec is not None and spec.loader is not None
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        if tuple(frozen["arms"]) != upstream.ARMS:
            raise ValueError("Boundary-study Wave-9 predecessor arms changed")
    return frozen


def snapshot(root: Path, out: Path, *, wave: Literal["wave8", "wave9"] = "wave8") -> dict:
    """Fresh SSH observations are mandatory; a saved terminal receipt alone cannot free a node."""
    frozen = release(root, wave=wave)
    arms = tuple(frozen["arms"])
    controller = "wave8_release.py" if wave == "wave8" else "upstream_production.py"
    subprocess.run(
        [
            PYTHON,
            str(Path(frozen["helpers"]) / controller),
            "--root",
            str(root),
            "--phase",
            "snapshot",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    observed = read(out)
    if (
        observed["status"] != "complete"
        or observed["all_workers_terminal"] is not True
        or observed["binding_sha256"] != sha256(root / f"run-preparation/{wave}-launch-binding.json")
        or len(observed["workers"]) != len(arms)
        or {worker["arm"] for worker in observed["workers"]} != set(arms)
        or any(
            worker["status"] != "complete"
            or type(worker["receipt"].get("keeper_restore")) is not int
            or worker["receipt"]["keeper_restore"] != 0
            for worker in observed["workers"]
        )
    ):
        raise RuntimeError(
            f"Wave-{wave[-1]} workers, marked children or keeper restoration remain active, unknown or incomplete"
        )
    if wave == "wave9":
        for worker in observed["workers"]:
            receipt = worker["receipt"]
            if (
                receipt.get("status") != "complete"
                or receipt.get("arm") != worker["arm"]
                or receipt.get("node") != NODES[arms.index(worker["arm"])]
                or receipt.get("release_binding_sha256") != observed["binding_sha256"]
                or any(
                    type(receipt.get(key)) is not int or receipt[key] != 0
                    for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
                )
                or any(key in evidence for evidence in (worker, receipt) for key in ("error", "cleanup_error"))
                or worker.get("descendants")
            ):
                raise RuntimeError("Wave-9 workers or original marked handles remain active, unknown or incomplete")
        if "error" in observed or any("accounting_error" in run for run in observed.get("runs", [])):
            raise RuntimeError("Wave-9 fresh observation contains errors")
    return observed


def _require_wave9_analysis(analysis: dict, arms: tuple[str, ...]) -> None:
    late_tokens = [tokens for tokens in EVALUATION_TOKENS if tokens > TOTAL_TOKENS // 2]
    if (
        analysis.get("wave") != "wave9"
        or analysis.get("window_tokens") != WINDOW_TOKENS
        or not analysis["analyses"]
        or {name for pair in analysis["analyses"] for name in pair["comparisons"]} != {f"wave9/{arm}" for arm in arms}
    ):
        raise ValueError("Wave-9 analysis lacks the complete four-arm full-window comparison")
    for pair in analysis["analyses"]:
        for comparison in pair["comparisons"].values():
            if (
                pair.get("window_tokens") != WINDOW_TOKENS
                or comparison.get("common_tokens") != TOTAL_TOKENS
                or any(
                    type(window.get(key)) is not int or window[key] != expected
                    for index, window in enumerate(comparison["training_windows"])
                    for key, expected in (
                        ("start_tokens", index * WINDOW_TOKENS),
                        ("end_tokens", (index + 1) * WINDOW_TOKENS),
                    )
                )
                or any(
                    evaluation.get("shared_tokens") != EVALUATION_TOKENS
                    or evaluation.get("late_tokens") != late_tokens
                    or evaluation.get("late_start_tokens_exclusive") != TOTAL_TOKENS // 2
                    or evaluation.get("late_end_tokens") != TOTAL_TOKENS
                    or evaluation.get("last_shared_tokens") != TOTAL_TOKENS
                    or evaluation.get("final_tokens") != TOTAL_TOKENS
                    or any(evaluation.get(key) is None for key in ("mean_late_delta", "final_value", "final_delta"))
                    for evaluation in (comparison["evaluations"].get(key, {}) for key in ("selection_loss", "c4_loss"))
                )
            ):
                raise ValueError("Wave-9 requires five full token windows and shared late/final evaluation coverage")


def require_completed(root: Path, *, wave: Literal["wave8", "wave9"] = "wave8") -> dict:
    """Rederive five-window results with the original frozen analyzer and check cloud provenance."""
    frozen = release(root, wave=wave)
    arms = tuple(frozen["arms"])
    analyzer = "analyze_wave8" if wave == "wave8" else "analyze_upstream_run"
    histories = "wave8_history" if wave == "wave8" else "upstream_run_history"
    script = f"""
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from {analyzer} import analyze
from {histories} import ARMS,binding,completed_run
root=Path(sys.argv[2])
release=binding(root)
print(json.dumps({{"analysis":analyze(root),
                  "runs":{{arm:completed_run(root,arm,release) for arm in ARMS}}}},allow_nan=False))
"""
    process = subprocess.run(
        [PYTHON, "-c", script, frozen["helpers"], str(root)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    rederived = json.loads(process.stdout)
    analysis_path = root / f"run-preparation/analysis-{wave}.json"
    cloud_path = root / f"run-preparation/swanlab-upload-{wave}.json"
    cloud = read(cloud_path)
    if (
        rederived["analysis"] != read(analysis_path)
        or rederived["analysis"]["source_release_sha256"] != sha256(root / f"run-preparation/{wave}-launch-binding.json")
        or any(
            comparison["final_comparison_eligible"] is not True or len(comparison["training_windows"]) != 5
            for pair in rederived["analysis"]["analyses"]
            for comparison in pair["comparisons"].values()
        )
        or cloud["status"] != "verified"
        or cloud["visibility"] != "PRIVATE"
        or not cloud["project"].endswith("/output-side-normalization-marin")
        or set(cloud["runs"]) != set(arms)
        or set(rederived["runs"]) != set(arms)
    ):
        raise ValueError(
            f"Complete Wave-{wave[-1]} five-window analysis " "and full-resolution cloud receipts are required"
        )
    if wave == "wave9":
        _require_wave9_analysis(rederived["analysis"], arms)
    for arm, measured in rederived["runs"].items():
        imported = cloud["runs"][arm]
        verification = imported["verification"]
        if (
            measured["complete"] is not True
            or imported["status"] != "verified"
            or any(
                imported[key] != measured[key]
                for key in ("manifest_sha256", "metrics_sha256", "checkpoint_receipt_sha256", "checkpoint_rank_bytes")
            )
            or verification["state"] != "FINISHED"
            or verification["configuration_verified"] is not True
            or verification["token_axis_verified"] is not True
            or verification["integer_values_exact"] is not True
            or verification["point_counts"]["train/loss"] != 5120
            or verification["point_counts"]["tokens"] != 5120
            or verification["point_counts"]["eval/selection_loss"] != 22
            or verification["point_counts"]["eval/c4_loss"] != 22
        ):
            raise ValueError(
                f"Wave-{wave[-1]} cloud history is incomplete " f"or differs from its final local evidence: {arm}"
            )
        if wave == "wave9" and (
            measured.get("observed_updates") != 5120
            or measured.get("observed_tokens") != TOTAL_TOKENS
            or len(measured["checkpoint_rank_bytes"]) != 8
            or any(type(size) is not int or size <= 0 for size in measured["checkpoint_rank_bytes"])
        ):
            raise ValueError(f"Wave-9 final history or eight-rank checkpoint evidence is incomplete: {arm}")
    return {
        "analysis_sha256": sha256(analysis_path),
        "cloud_sha256": sha256(cloud_path),
        "release_sha256": sha256(root / f"run-preparation/{wave}-launch-binding.json"),
    }
