# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify finite and sustained churn through the unchanged issued scan."""

import pytest
from owned_output_helper_churn import trial


@pytest.mark.parametrize("exits", (0, 1, 2))
def test_finite_helper_churn_returns_survivor_and_new_marked_helper(exits):
    result = trial(exits)
    expected = [900, 901] if exits == 0 else [901, 901 + exits]
    assert result["status"] == "complete_scan_returned"
    assert result["returned_pids"] == result["remaining_pids"] == expected
    assert len(result["retired_pids"]) == exits
    assert result["failure_stage"] is None


@pytest.mark.parametrize("exits", (3, 4))
def test_sustained_helper_churn_refuses_without_returning_partial_ownership(exits):
    result = trial(exits)
    assert result["status"] == "ownership_refused"
    assert result["returned_pids"] is None
    assert result["failure_stage"] == "complete_scan_exhausted"
    assert result["retired_pids"] == [900, 902, 903]
    assert result["remaining_pids"] == [901, 904]
    assert result["failure_snapshots"]["unstable_pids"] == [903]
    assert result["failure_snapshots"]["previously_confirmed_marked"] == {"903": True}
