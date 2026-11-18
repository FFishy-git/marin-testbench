# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise full-budget attribution reports and succession without live infrastructure."""

import json
import os
import subprocess
import sys
from pathlib import Path

import analyze_fused_attribution_run as analysis
import fused_attribution_run_history as histories
import pytest
from fused_attribution_production_contract import FAMILIES, STEPS, config_name, paths, wave_spec


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture_histories(root: Path, steps: int = STEPS) -> dict:
    source = root / "candidate"
    release = {"candidate": str(source), "files_sha256": {}, "data_receipt_sha256": "paired-data"}
    for index, family in enumerate(FAMILIES):
        config = {
            "experiment": {"arm": family, "refresh_install_age": 1, "diagonal_backend": "triton"},
            "model": {"sequence_length": 4096},
            "optimizer": {"lr": wave_spec("wave13")["matrix_lr"], "ns_mode": "tpu", "ns_steps": 10},
            "root": {"graphs": True},
            "schedule": {
                "warmup_steps": 0,
                "stable_steps": 1024,
                "decay_steps": 4096,
                "kind": "linear",
                "min_lr_ratio": 0.0,
            },
            "data": {"mixture_weights": {"dclm": 2.0, "c4": 1.0}},
            "training": {
                "steps": STEPS,
                "tokens": 2684354560,
                "batch_tokens": 524288,
                "global_batch_sequences": 128,
                "world_size": 8,
                "seed": 0,
                "heldout_windows": 64,
            },
            "runtime": {
                "eval_max_windows": 0,
                "eval_every": 250,
                "compile": "default",
                "attention": "FA2",
                "head": "compiled",
            },
        }
        directory = root / "runs/wave13" / family
        record(source / config_name(family, "lower"), config)
        record(
            directory / "run.json",
            {
                "status": "complete" if steps == STEPS else "running",
                "completed_steps": steps,
                "config": config,
                "data_receipt_sha256": "paired-data",
                "source_sha256": {},
                "execution_check": False,
                "elapsed_seconds": float(steps * 2),
            },
        )
        rows = []
        for step in range(1, steps + 1):
            row = {
                "step": step,
                "tokens": step * 524288,
                "train_loss": 5.0 - index * 0.01,
                "step_seconds": 2.0,
                "elapsed_seconds": step * 2.0,
                "end_to_end_seconds": step * 2.0,
            }
            if (step - 1) % 250 == 0 or step == STEPS:
                dclm, c4 = 3.0 - index * 0.01, 3.5 - index * 0.01
                row.update(
                    c4_loss=c4,
                    selection_loss=(2.0 * dclm + c4) / 3.0,
                    heldout_losses={"heldout/dclm": dclm, "heldout/c4": c4},
                )
            rows.append(row)
        (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        if steps == STEPS:
            checkpoint = directory / "checkpoints/step_005120"
            record(checkpoint / "complete.json", {"step": STEPS, "world": 8})
            for rank in range(8):
                (checkpoint / f"rank_{rank}.pt").write_bytes(f"rank {rank}".encode())
    record(paths(root, "wave13")[0], release)
    return release


def bind_fixture(monkeypatch, release):
    monkeypatch.setattr(histories, "candidate_inventory", lambda source: release["files_sha256"])
    monkeypatch.setattr(histories, "normalized_sources", lambda *args: {})
    monkeypatch.setattr(histories, "executed_sources", lambda *args: {})


@pytest.mark.parametrize("steps", (1536, STEPS))
def test_analysis_retains_only_complete_shared_windows_and_matched_null(tmp_path, monkeypatch, steps):
    fixture_histories(tmp_path, steps)
    monkeypatch.setattr(analysis, "verify_histories", lambda *args: {})
    result = analysis.analyze(tmp_path, "wave13")
    candidate, null = (cohort["comparisons"]["wave13/outer_leftdiag_fused"] for cohort in result["analyses"])
    count = 5 if steps == STEPS else 1
    assert [window["end_tokens"] for window in candidate["training_windows"]] == [
        index * 536870912 for index in range(1, count + 1)
    ]
    assert [window["delta"] for window in candidate["training_windows"]] == pytest.approx([-0.01] * count)
    assert [window["delta"] for window in null["training_windows"]] == pytest.approx([0.01] * count)
    assert result["analyses"][0]["baseline"] == "wave13/muont"
    assert result["analyses"][1]["baseline"] == "wave13/null_leftdiag_fused"
    assert candidate["final_comparison_eligible"] is (steps == STEPS)
    assert len(candidate["evaluations"]["selection_loss"]["shared_tokens"]) == (22 if steps == STEPS else 7)
    if steps != STEPS:
        assert candidate["evaluations"]["selection_loss"]["final_delta"] is None


@pytest.mark.parametrize("fault", ("none", "missing_rank", "missing_evaluation", "mixture", "resume_segment"))
def test_history_checks_actual_budget_checkpoint_and_selection(tmp_path, monkeypatch, fault):
    release = fixture_histories(tmp_path)
    bind_fixture(monkeypatch, release)
    family = FAMILIES[1]
    directory = tmp_path / "runs/wave13" / family
    expected = histories.completed_run(tmp_path, "wave13", family, release)
    assert expected["observed_updates"] == STEPS
    assert expected["observed_tokens"] == 2684354560
    assert len(expected["evaluation_tokens"]) == 22
    assert len(expected["checkpoint_rank_bytes"]) == 8
    if fault == "none":
        return
    if fault == "missing_rank":
        (directory / "checkpoints/step_005120/rank_7.pt").unlink()
    else:
        path = directory / "metrics_from_000000.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "missing_evaluation":
            rows[250].pop("selection_loss")
        elif fault == "mixture":
            rows[250]["selection_loss"] += 0.01
        else:
            (directory / "metrics_from_000001.jsonl").write_text(json.dumps(rows[-1]) + "\n")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((ValueError, FileNotFoundError)):
        histories.completed_run(tmp_path, "wave13", family, release)


def cloud_fixture(root: Path, release: dict) -> dict:
    runs = {}
    for family in FAMILIES:
        original = histories.completed_run(root, "wave13", family, release)
        runs[family] = {
            key: original[key]
            for key in ("manifest_sha256", "metrics_sha256", "checkpoint_receipt_sha256", "checkpoint_rank_bytes")
        } | {
            "status": "verified",
            "verification": {
                "state": "FINISHED",
                "configuration_verified": True,
                "token_axis_verified": True,
                "integer_values_exact": True,
                "csv_export_max_ulp": 2,
                "serialization_errors": {"train/loss": {"max_ulp": 1}},
                "point_counts": {"tokens": STEPS, "eval/selection_loss": 22, "eval/c4_loss": 22},
            },
        }
    return {
        "status": "verified",
        "visibility": "PRIVATE",
        "project": "siyuc3141/output-side-normalization-marin",
        "runs": runs,
    }


@pytest.mark.parametrize("fault", ("none", "partial", "public", "truncated", "changed_history", "float_error"))
def test_succession_requires_private_full_resolution_cloud(tmp_path, monkeypatch, fault):
    release = fixture_histories(tmp_path)
    bind_fixture(monkeypatch, release)
    monkeypatch.setattr(histories, "verify_histories", lambda *args: release)
    monkeypatch.setattr(histories, "require_completed_snapshot", lambda *args: {})
    monkeypatch.setattr(histories, "require_analysis", lambda *args: {})
    record(tmp_path / "run-preparation/monitor-wave13/latest.json", {"status": "complete"})
    record(tmp_path / "run-preparation/analysis-wave13.json", {"wave": "wave13"})
    cloud = cloud_fixture(tmp_path, release)
    if fault == "partial":
        cloud["runs"].pop(FAMILIES[-1])
    elif fault == "public":
        cloud["visibility"] = "PUBLIC"
    elif fault == "truncated":
        cloud["runs"][FAMILIES[1]]["verification"]["point_counts"]["tokens"] -= 1
    elif fault == "changed_history":
        cloud["runs"][FAMILIES[1]]["metrics_sha256"] = "changed"
    elif fault == "float_error":
        cloud["runs"][FAMILIES[1]]["verification"]["serialization_errors"]["train/loss"]["max_ulp"] = 3
    record(tmp_path / "run-preparation/swanlab-upload-wave13.json", cloud)
    if fault == "none":
        result = histories.require_completed_wave(tmp_path, "wave13")
        assert result["status"] == "attribution_wave_completed_analyzed_cloud_verified"
        assert result["complete_windows"] == 5
        assert len(result["evidence_sha256"]) == 4
        return
    with pytest.raises(ValueError):
        histories.require_completed_wave(tmp_path, "wave13")


def test_completed_analysis_rejects_changed_baseline_cohorts(tmp_path, monkeypatch):
    fixture_histories(tmp_path)
    monkeypatch.setattr(analysis, "verify_histories", lambda *args: {})
    result = analysis.analyze(tmp_path, "wave13")
    record(tmp_path / "run-preparation/analysis-wave13.json", result)
    histories.require_analysis(tmp_path, "wave13", {})
    result["analyses"][1] = result["analyses"][0]
    record(tmp_path / "run-preparation/analysis-wave13.json", result)
    with pytest.raises(ValueError):
        histories.require_analysis(tmp_path, "wave13", {})


def test_reporting_imports_without_training_dependencies():
    directory = Path(histories.__file__).parent
    script = (
        "import sys\n"
        "class NoTrainingImports:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'torch','pytest','experiment'}:\n"
        "            raise RuntimeError('reporting imported training dependencies')\n"
        "sys.meta_path.insert(0, NoTrainingImports())\n"
        "import fused_attribution_run_history, analyze_fused_attribution_run\n"
    )
    result = subprocess.run(
        [sys.executable, "-P", "-B", "-c", script],
        cwd=directory,
        env=os.environ | {"PYTHONPATH": str(directory)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
