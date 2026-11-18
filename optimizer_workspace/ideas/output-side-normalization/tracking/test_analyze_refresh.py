# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recorded refresh windows count events once and preserve sparse diagnostics."""

import copy

import pytest
from analyze_refresh import refresh_windows, sha256_rows, summarize


@pytest.fixture
def rows():
    history = [
        {
            "step": step,
            "tokens": step * 8,
            "train_loss": 4.0,
            "step_seconds": float(step),
            "input_fallbacks": count,
            "input_refresh_installations": [],
        }
        for step, count in enumerate((0, 1, 1, 3), start=1)
    ]
    for step, installed in ((2, True), (3, False), (4, True)):
        history[step - 1]["input_refresh_installations"].append(
            {"group": "d", "index": 0, "source_step": step - 2, "installation_step": step - 1, "installed": installed}
        )
    history[-1]["input_refresh_installations"].append(
        {"group": "f", "index": 0, "source_step": 2, "installation_step": 3, "installed": True}
    )
    history[2]["input_max_basis_age"] = 2
    history[3]["input_max_basis_age"] = 4
    return history


def test_complete_shared_windows_count_once_and_use_previous_counter(rows):
    original = copy.deepcopy(rows)
    windows = refresh_windows(rows, common_tokens=32, window_tokens=16, age=1)
    assert [(window["start_tokens"], window["end_tokens"]) for window in windows] == [(0, 16), (16, 32)]
    assert [window["events"] for window in windows] == [1, 3]
    assert [window["counter_deltas"]["input_fallbacks"] for window in windows] == [1, 2]
    assert windows[1]["installed"] == 2
    assert windows[1]["discarded"] == 1
    assert windows[1]["groups"]["input_refresh_installations/d"]["installation_ages"] == {1: 2}
    assert windows[1]["step_timing"] == {"rows": 2, "sum": 7.0, "median": 3.5, "p95": 4.0}
    assert windows[1]["recorded_samples"]["input_max_basis_age"] == {"rows": 2, "mean": 3, "max": 4}
    assert "input_extra_roots" not in windows[1]["counter_deltas"]
    assert "input_max_basis_age" not in windows[0]["recorded_samples"]
    assert rows == original


def test_incomplete_shared_tail_is_not_counted_as_a_complete_window(rows):
    windows = refresh_windows(rows, common_tokens=24, window_tokens=16, age=1)
    assert len(windows) == 1
    assert windows[0]["updates"] == 2
    assert windows[0]["events"] == 1


@pytest.mark.parametrize("fault", ("age", "duplicate", "counter_reset"))
def test_invalid_observed_scheduling_or_counter_history_refuses_summary(rows, fault):
    if fault == "age":
        rows[2]["input_refresh_installations"][0]["source_step"] = 0
    elif fault == "duplicate":
        rows[2]["input_refresh_installations"].append(copy.deepcopy(rows[2]["input_refresh_installations"][0]))
    else:
        rows[2]["input_fallbacks"] = 0
    with pytest.raises(ValueError):
        summarize(rows, None, 1)


def test_empty_event_lists_and_absent_counters_remain_explicit(rows):
    stripped = [
        {key: value for key, value in row.items() if key not in ("input_fallbacks", "input_max_basis_age")}
        for row in rows
    ]
    for row in stripped:
        row["input_refresh_installations"] = []
    result = summarize(stripped, None, 1)
    assert result["events"] == result["installed"] == result["discarded"] == 0
    assert result["counter_deltas"] == result["recorded_samples"] == {}
    assert result["event_field_rows"] == {"input_refresh_installations": 4, "output_refresh_installations": 0}


def test_observed_row_identity_survives_json_key_order_but_changes_with_values(rows):
    reversed_keys = [dict(reversed(list(row.items()))) for row in rows]
    assert sha256_rows(rows) == sha256_rows(reversed_keys)
    reversed_keys[-1]["train_loss"] += 0.001
    assert sha256_rows(rows) != sha256_rows(reversed_keys)
