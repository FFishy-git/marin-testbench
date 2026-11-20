# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test unmarked disappearance through the unchanged external scanner boundary."""

import types

import owned_output_unmarked_churn as study
import pytest


@pytest.mark.parametrize("exits", (0, 1, 2))
def test_finite_unmarked_churn_returns_every_marked_survivor(exits):
    result = study.trial(exits)
    assert result["status"] == "complete_scan_returned"
    assert result["returned_pids"] == [900, 901]
    assert result["remaining_pids"] == [900, 901, study.UNMARKED_PID + exits]
    assert result["retired_pids"] == list(range(study.UNMARKED_PID, study.UNMARKED_PID + exits))
    assert result["failure_stage"] is None


@pytest.mark.parametrize("exits", (3, 4))
def test_sustained_unmarked_churn_refuses_without_partial_ownership(exits):
    result = study.trial(exits)
    assert result["status"] == "ownership_refused"
    assert result["returned_pids"] is None
    assert result["failure_stage"] == "complete_scan_exhausted"
    assert result["remaining_pids"] == [900, 901, study.UNMARKED_PID + 3]
    assert result["retired_pids"] == list(range(study.UNMARKED_PID, study.UNMARKED_PID + 3))
    assert result["failure_snapshots"]["previously_confirmed_marked"] == {str(study.UNMARKED_PID + 2): False}
    assert result["failure_snapshots"]["unstable_pids"] == [study.UNMARKED_PID + 2]


def test_present_unmarked_process_with_missing_environment_never_proves_absence(tmp_path):
    sources = study.source_bytes(study.SOURCE)
    scan = types.ModuleType("unmarked_missing_environment_issued_scan")
    scan.__file__ = str(study.SOURCE / "owned_output_handoff.py")
    exec(compile(sources["owned_output_handoff.py"], scan.__file__, "exec"), scan.__dict__)
    scan.PROC_ROOT = tmp_path
    study.fake_process(tmp_path, 900)
    study.unmarked_process(tmp_path, study.UNMARKED_PID)
    (tmp_path / str(study.UNMARKED_PID) / "environ").unlink()
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(RuntimeError):
        scan.marker_processes(*study.MARKER.split("=", 1))
    assert (tmp_path / str(study.UNMARKED_PID)).is_dir()
    assert {path: path.read_bytes() for path in before} == before
    assert not (tmp_path / str(study.UNMARKED_PID) / "environ").exists()
