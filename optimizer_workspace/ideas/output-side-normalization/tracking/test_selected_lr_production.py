# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise selected-LR launch failures and full-budget reporting without GPUs."""

import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path

import analyze_selected_lr_run as analysis
import pytest
import selected_lr_production as production
import selected_lr_run_history as histories
from launch_wave5 import read, sha256


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture_histories(root: Path, steps: int = 5120) -> dict:
    source = root / "candidate"
    release = {"candidate": str(source), "files_sha256": {}, "data_receipt_sha256": "paired-data"}
    for index, arm in enumerate(production.ARMS):
        directory = root / "runs/wave11" / arm
        config = {
            "experiment": {"arm": arm},
            "model": {"sequence_length": 4096},
            "data": {"mixture_weights": {"dclm": 1.0}},
            "training": {
                "steps": 5120,
                "tokens": 2684354560,
                "batch_tokens": 524288,
                "world_size": 8,
                "seed": 0,
                "heldout_windows": 64,
            },
            "runtime": {"eval_max_windows": 0, "eval_every": 250},
        }
        record(source / f"experiment/configs/{arm}_0p5m_seed0.json", config)
        record(
            directory / "run.json",
            {
                "status": "complete" if steps == 5120 else "running",
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
            if (step - 1) % 250 == 0 or step == 5120:
                row.update(
                    c4_loss=3.5 - index * 0.01,
                    selection_loss=3.0 - index * 0.01,
                    heldout_losses={"heldout/dclm": 3.0 - index * 0.01},
                )
            rows.append(row)
        (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    record(production.paths(root)[0], release)
    return release


def test_analysis_retains_five_large_windows_and_tuned_baseline(tmp_path, monkeypatch):
    fixture_histories(tmp_path)
    monkeypatch.setattr(analysis, "verify_histories", lambda root: {})
    result = analysis.analyze(tmp_path)
    comparisons = result["analyses"][0]["comparisons"]
    assert result["analyses"][0]["baseline"] == "wave11/muont"
    assert set(comparisons) == {f"wave11/{arm}" for arm in production.ARMS}
    candidate = comparisons["wave11/outer_leftdiag"]
    assert [item["end_tokens"] for item in candidate["training_windows"]] == [index * 536870912 for index in range(1, 6)]
    assert [item["delta"] for item in candidate["training_windows"]] == pytest.approx([-0.02] * 5)
    assert candidate["final_comparison_eligible"] is True
    assert len(candidate["evaluations"]["c4_loss"]["shared_tokens"]) == 22
    assert result["source_release_sha256"] == sha256(production.paths(tmp_path)[0])


def test_running_analysis_withholds_final_and_incomplete_window(tmp_path, monkeypatch):
    fixture_histories(tmp_path, 1536)
    monkeypatch.setattr(analysis, "verify_histories", lambda root: {})
    result = analysis.analyze(tmp_path)
    candidate = result["analyses"][0]["comparisons"]["wave11/outer_leftdiag_fused"]
    assert [item["end_tokens"] for item in candidate["training_windows"]] == [536870912]
    assert candidate["final_comparison_eligible"] is False
    assert candidate["evaluations"]["selection_loss"]["final_delta"] is None


def test_original_pending_validation_refuses_acceptance_before_node_lock(tmp_path):
    record(production.locations(tmp_path)[2], {"status": "validating"})
    with pytest.raises(RuntimeError, match="has not passed"):
        production.validation_acceptance(tmp_path)
    assert not production.paths(tmp_path)[0].exists()
    assert not (tmp_path / "runs/wave11").exists()


def test_preflight_failure_preserves_no_launch_intent_or_output(tmp_path, monkeypatch):
    (tmp_path / "run-preparation").mkdir()
    release = {"candidate": str(tmp_path), "helpers": str(tmp_path)}
    monkeypatch.setattr(production, "binding", lambda root: release)
    monkeypatch.setattr(production, "validation_acceptance", lambda root: {})
    monkeypatch.setattr(production, "predecessor_results", lambda root: None)

    def occupied(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="predecessor allocation changed")

    monkeypatch.setattr(production.subprocess, "run", occupied)
    with pytest.raises(subprocess.CalledProcessError):
        production.launch(tmp_path)
    assert not (tmp_path / "run-preparation/wave11-launch-state.json").exists()
    assert not (tmp_path / "runs/wave11").exists()
    assert not (tmp_path / "run-preparation/monitor-wave11").exists()


@pytest.mark.parametrize("reparented", (False, True))
def test_unverified_cleanup_preserves_failure_and_prevents_restoration(tmp_path, monkeypatch, reparented):
    release = {"helpers": str(tmp_path)}
    release_path = production.paths(tmp_path)[0]
    record(release_path, {"status": "bound"})
    arm = production.ARMS[0]
    command = production.worker_command(tmp_path, release, arm)
    generation = {
        "pid": os.getpid(),
        "status": "S",
        "parent": os.getppid(),
        "group": os.getpgrp(),
        "session": os.getsid(0),
        "start_ticks": "original",
        "command": command,
    }
    path = production.worker_path(tmp_path, arm)
    record(
        path.with_suffix(".start.json"),
        {
            "status": "acknowledged",
            "generation": generation,
            "worker_marker": "worker",
            "release_binding_sha256": sha256(release_path),
        },
    )
    record(
        tmp_path / "run-preparation/wave11-launch-state.json",
        {
            "status": "launching",
            "binding_sha256": sha256(release_path),
            "worker_markers": {arm: "worker"},
        },
    )
    keeper = {
        "pid": 11,
        "generation": {11: {"pid": 11, "status": "R", "start_ticks": "keeper"}},
        "devices": [["0", "uuid", "A100"]],
    }
    monkeypatch.setenv(production.WORKER_MARKER, "worker")
    monkeypatch.setattr(production, "binding", lambda root: release)
    current = generation | {"parent": 1} if reparented else generation
    monkeypatch.setattr(production, "observed", lambda pid: current)
    monkeypatch.setattr(production, "node_guard", lambda *args: nullcontext([]))
    monkeypatch.setattr(production, "allocation", lambda *args: keeper)
    monkeypatch.setattr(production, "keeper_observation", lambda: keeper)
    monkeypatch.setattr(production, "release_keeper", lambda state, path: state.update(keeper_release_attempted=True))
    restored = []
    monkeypatch.setattr(production, "node_action", lambda root, action: restored.append(action))

    def failed(root, release, state, path):
        state["cleanup_error"] = "marked child still live"
        raise RuntimeError("cleanup unverified")

    monkeypatch.setattr(production, "run_child", failed)
    with pytest.raises(RuntimeError, match="cleanup unverified"):
        production.run_arm(tmp_path, arm)
    state = read(path)
    assert state["status"] == "needs_attention"
    assert state["keeper_restore"].startswith("refused:")
    assert state["cleanup_error"] == "marked child still live"
    assert restored == []


def test_keeper_succession_rejects_reused_generation_but_allows_scheduler_state():
    previous = {
        "pid": 11,
        "devices": [["0", "uuid", "A100"]],
        "generation": {"11": {"pid": 11, "status": "S", "parent": 1, "start_ticks": "original"}},
    }
    current = previous | {"generation": {11: previous["generation"]["11"] | {"status": "R"}}}
    assert production.same_keeper(current, previous)
    serialized = json.loads(json.dumps(current))
    assert production.same_keeper(serialized, serialized)
    assert production.same_keeper(serialized, previous)
    changed = current | {"generation": {11: current["generation"][11] | {"start_ticks": "reused"}}}
    assert not production.same_keeper(changed, previous)


@pytest.mark.parametrize(
    "fault", ("none", "shifted_evaluation", "empty_heldout", "selection_mismatch", "nonfinite_heldout")
)
def test_completed_history_requires_exact_evaluation_schedule_and_mixture(tmp_path, monkeypatch, fault):
    release = fixture_histories(tmp_path)
    arm = production.ARMS[0]
    directory = tmp_path / "runs/wave11" / arm
    checkpoint = directory / "checkpoints/step_005120"
    record(checkpoint / "complete.json", {"step": 5120, "world": 8})
    for rank in range(8):
        (checkpoint / f"rank_{rank}.pt").write_bytes(b"checkpoint")
    monkeypatch.setattr(histories, "candidate_inventory", lambda path: {})
    monkeypatch.setattr(histories, "executed_sources", lambda path, frozen: {})
    path = directory / "metrics_from_000000.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "shifted_evaluation":
        for key in ("c4_loss", "selection_loss", "heldout_losses"):
            rows[1][key] = rows[0].pop(key)
    elif fault == "empty_heldout":
        rows[-1]["heldout_losses"] = {}
    elif fault == "selection_mismatch":
        rows[-1]["heldout_losses"]["heldout/dclm"] += 0.1
    elif fault == "nonfinite_heldout":
        rows[-1]["heldout_losses"]["heldout/dclm"] = float("nan")
    if fault != "none":
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError):
            histories.completed_run(tmp_path, arm, release)
        return
    result = histories.completed_run(tmp_path, arm, release)
    assert result["complete"] is True
    assert result["observed_updates"] == 5120
    assert result["checkpoint_rank_bytes"] == [10] * 8
