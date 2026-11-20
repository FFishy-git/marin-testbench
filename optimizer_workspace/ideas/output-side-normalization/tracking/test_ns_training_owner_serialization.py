# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check lock contention, cleanup and the unchanged scan against transient query helpers."""

import fcntl
import hashlib
import json
import os
import shutil
import types
from pathlib import Path

import ns_training_owner_serialization as serialization
import owned_output_helper_churn as churn
import pytest


def test_serialized_inspection_waits_for_other_rank_and_releases_after_failure(tmp_path):
    path = tmp_path / "inspection.lock"
    other = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(other, fcntl.LOCK_EX)
    elapsed = 0

    def clock():
        return elapsed

    def pause(interval):
        nonlocal elapsed
        elapsed += interval
        fcntl.flock(other, fcntl.LOCK_UN)

    with pytest.raises(ValueError, match="inspection failure"):
        with serialization.inspection_lock(path, clock=clock, pause=pause):
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise ValueError("inspection failure")
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(other)
    assert elapsed > 0


def test_unfinished_other_rank_preserves_bounded_wait_and_cannot_enter(tmp_path):
    path = tmp_path / "inspection.lock"
    other = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(other, fcntl.LOCK_EX)
    elapsed, entered = 0, False

    def clock():
        return elapsed

    def pause(interval):
        nonlocal elapsed
        elapsed += 1

    try:
        with pytest.raises(RuntimeError, match="deadline exhausted"):
            with serialization.inspection_lock(path, clock=clock, pause=pause):
                entered = True
    finally:
        os.close(other)
    assert not entered
    assert elapsed == serialization.TIMEOUT


@pytest.mark.parametrize("fault", ("alias", "replacement"))
def test_alias_or_replaced_lock_cannot_supply_serialization(tmp_path, fault):
    path = tmp_path / "inspection.lock"
    if fault == "alias":
        (tmp_path / "foreign").touch()
        path.symlink_to("foreign")
        with pytest.raises(OSError):
            with serialization.inspection_lock(path):
                pytest.fail("Aliased lock entered")
    else:
        with pytest.raises(RuntimeError, match="identity changed during"):
            with serialization.inspection_lock(path):
                path.unlink()
                path.touch()
    assert path.exists()


def test_sustained_query_helper_churn_refuses_then_serialized_quiescent_scan_retains_ranks(tmp_path, monkeypatch):
    sources = churn.source_bytes(churn.SOURCE)
    scan = types.ModuleType("serialized_issued_scan")
    scan.__file__ = str(churn.SOURCE / "owned_output_handoff.py")
    exec(compile(sources["owned_output_handoff.py"], scan.__file__, "exec"), scan.__dict__)
    proc = tmp_path / "proc"
    proc.mkdir()
    scan.PROC_ROOT = proc
    for pid in range(900, 908):
        churn.fake_process(proc, pid)
    command = ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]
    command_bytes = ("\0".join(command) + "\0").encode()
    churn.fake_process(proc, 1000)
    (proc / "1000/cmdline").write_bytes(command_bytes)
    retired = []
    read_bytes = Path.read_bytes

    def other_rank_query(path):
        payload = read_bytes(path)
        if path.parent.parent == proc and path.name == "environ" and int(path.parent.name) >= 1000:
            pid = int(path.parent.name)
            churn.fake_process(proc, pid + 1)
            (proc / str(pid + 1) / "cmdline").write_bytes(command_bytes)
            retired.append(pid)
            shutil.rmtree(path.parent)
        return payload

    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", other_rank_query)
        with pytest.raises(scan.attribution_handler.OwnershipObservationError) as failed:
            scan.marker_processes("HELPER_CHURN_STUDY", "owned")
    assert failed.value.evidence["stage"] == "complete_scan_exhausted"
    assert retired == [1000, 1001, 1002]
    digest = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
    assert failed.value.evidence["snapshots"]["retained"]["1002"]["command"]["sha256"] == digest
    shutil.rmtree(proc / "1003")
    with serialization.inspection_lock(tmp_path / "inspection.lock"):
        result = scan.marker_processes("HELPER_CHURN_STUDY", "owned")
    assert set(result) == set(range(900, 908))
