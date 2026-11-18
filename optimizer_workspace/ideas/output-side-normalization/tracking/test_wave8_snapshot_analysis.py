# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict Wave-8 captures use whole common windows and preserve malformed-history evidence."""

import copy
import json
import subprocess
from datetime import UTC, datetime

import analyze_wave8_snapshot
import pytest
from analyze_tuning import executed_sources
from compare_runs import WINDOW
from launch_wave5 import read, sha256
from test_wave8_release import release_fixture as source_fixture
from update_endeavor import ANCHOR, update


@pytest.fixture
def running_wave8(tmp_path, monkeypatch):
    root, candidate, release = source_fixture.__wrapped__(tmp_path, monkeypatch)
    base = {
        "model": {"sequence_length": 4096},
        "data": {"mixture": "same"},
        "experiment": {"arm": "muont"},
        "training": {"seed": 0, "heldout_windows": 64, "steps": 5120, "tokens": 2684354560, "batch_tokens": 524288},
    }
    numerical = read(root / "run-preparation/wave8-candidate.json")
    source = executed_sources(candidate, numerical)
    for index, arm in enumerate(analyze_wave8_snapshot.ARMS):
        config = copy.deepcopy(base)
        config["experiment"]["arm"] = arm
        (candidate / f"experiment/configs/{arm}_0p5m_seed0.json").write_text(json.dumps(config))
        directory = root / "runs/wave8" / arm
        directory.mkdir(parents=True)
        manifest = {
            "status": "running",
            "config": config,
            "data_receipt_sha256": release["data_receipt_sha256"],
            "source_sha256": source,
        }
        (directory / "run.json").write_text(json.dumps(manifest))
        rows = [{"step": step, "tokens": step * 524288, "train_loss": 5.0 - index / 100} for step in range(1, 2050)]
        (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(analyze_wave8_snapshot, "release", lambda root: release)
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "bound fixture", "")
    )
    return root


def test_wave8_snapshot_compares_immutable_common_large_windows(running_wave8):
    root = running_wave8
    observation = root / "observation"
    analyze_wave8_snapshot.capture(root, observation)
    result = analyze_wave8_snapshot.analyze_capture(observation)
    comparison = result["analyses"][0]["comparisons"]["wave8/outer_leftdiag_combined_fused"]
    assert len(comparison["training_windows"]) == 2
    assert [window["end_tokens"] for window in comparison["training_windows"]] == [WINDOW, 2 * WINDOW]
    assert all(window["delta"] == pytest.approx(-0.03) for window in comparison["training_windows"])
    assert comparison["final_comparison_eligible"] is False
    path = root / "runs/wave8/muont/metrics_from_000000.jsonl"
    path.write_text(path.read_text() + '{"unfinished":')
    assert analyze_wave8_snapshot.analyze_capture(observation) == result
    captured = observation / "muont/metrics_from_000000.jsonl"
    captured.write_text(captured.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after capture"):
        analyze_wave8_snapshot.analyze_capture(observation)


def test_recurring_document_reconstructs_strict_wave8_results_and_withholds_changed_capture(running_wave8):
    root = running_wave8
    observation = root / "run-preparation/wave8-observations/capture"
    analyze_wave8_snapshot.capture(root, observation)
    result = analyze_wave8_snapshot.analyze_capture(observation)
    (observation / "analysis.json").write_text(json.dumps(result))
    for name in ("wave8-candidate.json", "wave8-validation-state.json", "validation-wave8-gpu.json"):
        (root / "run-preparation" / name).unlink()
    document = root / "ENDEAVOR.md"
    document.write_text(f"# Manual research\n\n{ANCHOR}\n")
    update(document, root, datetime.now(UTC))
    text = document.read_text()
    assert "Wave-8 Strict Captured Comparison" in text
    assert "Complete 536,870,912-token training-window deltas: -0.030000, -0.030000" in text
    assert "interim; no final outcome" in text
    path = observation / "muont/metrics_from_000000.jsonl"
    path.write_text(path.read_text() + "\n")
    update(document, root, datetime.now(UTC))
    assert "captured results withheld" in document.read_text()
    assert "Complete 536,870,912-token training-window deltas" not in document.read_text()


@pytest.mark.parametrize("fault", ["partial", "blank", "missing_update", "manifest", "source", "complete"])
def test_wave8_capture_refuses_bad_evidence_without_creating_observation(running_wave8, fault):
    root = running_wave8
    directory = root / "runs/wave8" / analyze_wave8_snapshot.ARMS[-1]
    path = directory / "metrics_from_000000.jsonl"
    if fault == "partial":
        path.write_text(path.read_text() + '{"unfinished":')
    elif fault == "blank":
        path.write_text(path.read_text() + "\n")
    elif fault == "missing_update":
        path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    else:
        path = directory / "run.json"
        manifest = read(path)
        if fault == "manifest":
            manifest["data_receipt_sha256"] = "changed"
        elif fault == "source":
            manifest["source_sha256"] = {}
        else:
            manifest["status"] = "complete"
        path.write_text(json.dumps(manifest))
    observation = root / "observation"
    original_hash = sha256(path)
    with pytest.raises(ValueError):
        analyze_wave8_snapshot.capture(root, observation)
    assert not observation.exists()
    assert sha256(path) == original_hash
