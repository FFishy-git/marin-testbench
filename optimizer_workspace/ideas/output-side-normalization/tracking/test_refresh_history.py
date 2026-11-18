# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise structured refresh telemetry without importing SwanLab."""

import copy

import pytest
from refresh_history import scalar_history_row


def test_refresh_projection_preserves_original_scalars_and_raw_records():
    row = {
        "step": 24,
        "tokens": 12582912,
        "train_loss": 6.0,
        "input_gpu_ms_previous": {"roots": 0.2},
        "input_refresh_installations": [
            {"group": "d", "index": 2, "source_step": 22, "installation_step": 23, "installed": True},
            {"group": "f", "index": 0, "source_step": 22, "installation_step": 23, "installed": True},
            {"group": "f", "index": 1, "source_step": 22, "installation_step": 23, "installed": False},
        ],
    }
    original = copy.deepcopy(row)
    scalar = scalar_history_row(row)
    assert row == original
    for key in ("step", "tokens", "train_loss", "input_gpu_ms_previous"):
        assert scalar[key] == row[key]
    totals = scalar["input_refresh_installations"]
    assert totals["count"] == 3 and totals["installed"] == 2 and totals["discarded"] == 1
    assert totals["d/count"] == 1
    assert totals["f/count"] == 2 and totals["f/installed"] == 1 and totals["f/discarded"] == 1
    assert totals["f/min_age"] == totals["f/max_age"] == 1
    assert totals["f/min_source_step"] == totals["f/max_source_step"] == 22


def test_empty_refresh_list_remains_zero_count_and_missing_lists_add_no_columns():
    assert scalar_history_row({"step": 1, "tokens": 1}) == {"step": 1, "tokens": 1}
    result = scalar_history_row({"step": 1, "input_refresh_installations": []})
    assert result["input_refresh_installations"] == {"count": 0, "installed": 0, "discarded": 0}


def test_invalid_refresh_event_is_rejected_before_it_becomes_cloud_telemetry():
    row = {
        "step": 24,
        "input_refresh_installations": [
            {"group": "f", "index": 0, "source_step": 23, "installation_step": 23, "installed": True}
        ],
    }
    with pytest.raises(ValueError, match="Invalid recorded"):
        scalar_history_row(row)
