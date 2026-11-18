# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Crossings use actual completed evaluation points and explicit censoring."""

from types import SimpleNamespace

import pytest
from compare_runs import RunHistory
from time_to_target import crossings


def test_crossing_does_not_interpolate_or_credit_training_loss():
    run = SimpleNamespace(
        rows=[
            {"tokens": 100, "c4_loss": 4.0, "end_to_end_seconds": 10},
            {"tokens": 200, "train_loss": 2.0, "end_to_end_seconds": 20},
            {"tokens": 300, "c4_loss": 3.5, "end_to_end_seconds": 32},
        ]
    )
    result = crossings(run, 3.7)
    assert result["crossing_tokens"] == 300
    assert result["recorded_main_seconds"] == 32
    assert result["previous_evaluation_tokens"] == 100
    assert result["previous_recorded_main_seconds"] == 10
    assert result["censored"] is False


@pytest.mark.parametrize("status", ["running", "complete"])
def test_unreached_target_is_censored_without_final_training_loss_fallback(status):
    rows = [
        {"step": 1, "tokens": 100, "c4_loss": 4.0, "end_to_end_seconds": 10},
        {"step": 2, "tokens": 200, "train_loss": 2.0, "end_to_end_seconds": 20},
        {"step": 3, "tokens": 300, "train_loss": 1.0, "end_to_end_seconds": 34},
    ]
    run = RunHistory(
        manifest={"status": status, "completed_steps": 3, "config": {"training": {"steps": 3, "tokens": 300}}},
        rows=rows,
        segments=[rows],
    )
    result = crossings(run, 3.7)
    assert result["censored"] is True
    assert result["observed_tokens"] == 300
    assert result["last_evaluation_tokens"] == 100
    assert result["last_evaluation_c4"] == 4.0
    assert result["recorded_main_seconds"] == 34
    assert result["run_complete"] is (status == "complete")
    assert "crossing_tokens" not in result


def test_missing_evaluations_remain_censored_after_completion():
    rows = [{"step": 1, "tokens": 100, "train_loss": 2.0, "end_to_end_seconds": 10}]
    run = RunHistory(
        manifest={"status": "complete", "completed_steps": 1, "config": {"training": {"steps": 1, "tokens": 100}}},
        rows=rows,
        segments=[rows],
    )
    result = crossings(run, 3.7)
    assert result["censored"] is True
    assert result["run_complete"] is True
    assert result["last_evaluation_tokens"] is None
    assert result["last_evaluation_c4"] is None


def test_crossing_without_main_timer_does_not_use_training_only_time():
    run = SimpleNamespace(rows=[{"tokens": 100, "c4_loss": 3.5, "elapsed_seconds": 10}])
    result = crossings(run, 3.7)
    assert result["censored"] is False
    assert result["crossing_tokens"] == 100
    assert result["recorded_main_seconds"] is None
