# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable history comparisons remain stable after live appends and reject altered captures."""

import json

import pytest
from analyze_wave7_snapshot import analyze_capture
from compare_runs import WINDOW
from watch_wave5 import write
from wave7_release import ARMS, sha256


def test_snapshot_comparison_uses_fixed_large_windows_and_rejects_changed_observation(tmp_path):
    observation = tmp_path / "observation"
    identities = {}
    for index, arm in enumerate(ARMS):
        directory = observation / arm
        directory.mkdir(parents=True)
        write(
            directory / "run.json",
            {
                "status": "running",
                "data_receipt_sha256": "matched-data",
                "config": {
                    "model": {"sequence_length": 4096},
                    "data": {"mixture": "matched"},
                    "training": {
                        "seed": 0,
                        "heldout_windows": 64,
                        "steps": 5120,
                        "batch_tokens": 524288,
                        "tokens": 2684354560,
                    },
                },
            },
        )
        rows = [{"step": step, "tokens": step * 524288, "train_loss": 5.0 - 0.01 * index} for step in range(1, 2050)]
        metrics = directory / "metrics_from_000000.jsonl"
        metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
        identities[arm] = {
            "observed_updates": len(rows),
            "manifest_sha256": sha256(directory / "run.json"),
            "history_sha256": {metrics.name: sha256(metrics)},
        }
    write(observation / "observation.json", {"runs": identities})
    result = analyze_capture(observation)
    pair = result["analyses"][0]["comparisons"]["wave7/outer_leftdiag"]
    assert len(pair["training_windows"]) == 2
    assert [window["end_tokens"] for window in pair["training_windows"]] == [WINDOW, 2 * WINDOW]
    assert all(window["delta"] == pytest.approx(-0.02) for window in pair["training_windows"])
    assert pair["final_comparison_eligible"] is False
    assert pair["evaluations"]["c4_loss"]["final_delta"] is None
    metrics = observation / ARMS[0] / "metrics_from_000000.jsonl"
    metrics.write_text(
        metrics.read_text() + json.dumps({"step": 2050, "tokens": 2050 * 524288, "train_loss": 5.0}) + "\n"
    )
    with pytest.raises(ValueError, match="observation changed after capture"):
        analyze_capture(observation)
    manifest_path = observation / ARMS[0] / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    write(manifest_path, manifest)
    identities[ARMS[0]]["manifest_sha256"] = sha256(manifest_path)
    identities[ARMS[0]]["history_sha256"][metrics.name] = sha256(metrics)
    write(observation / "observation.json", {"runs": identities})
    with pytest.raises(ValueError, match="requires running manifests"):
        analyze_capture(observation)
