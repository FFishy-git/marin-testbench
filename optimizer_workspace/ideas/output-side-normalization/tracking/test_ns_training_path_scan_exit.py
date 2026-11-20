# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify retained failed evidence and numerical-contract preservation before a distinct launch."""

import copy
import importlib.util
from pathlib import Path

import ns_training_path_scan_exit_predecessor as predecessor
import ns_training_path_scan_exit_source as source
import prepare_ns_training_path_scan_exit as preparation
import pytest

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def test_scan_exit_preserves_training_contract_and_changes_only_owner_execution_namespace(tmp_path):
    parent = source.recorded_parent(ROOT)
    generated = source.derived_sources(parent)
    path = tmp_path / "contract.py"
    path.write_bytes(generated["ns_training_path_scan_exit_contract.py"])
    spec = importlib.util.spec_from_file_location("scan_exit_contract", path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    assert contract.specs() == predecessor.original.specs()
    assert contract.TOLERANCE == predecessor.original.TOLERANCE
    assert len(contract.commands(Path(parent["candidate"]), ROOT, contract.locations(ROOT)[3])) == 54
    assert set(contract.locations(ROOT)).isdisjoint(predecessor.original.locations(ROOT))
    sources = preparation.helper_bytes(parent)
    hashes = preparation.digest_bytes(sources)
    assert all(hashes[name] == digest for name, digest in predecessor.original.NUMERICAL_READERS.items())
    changed = {name for name, digest in parent["helpers_sha256"].items() if hashes[name] != digest}
    assert changed == {"owned_output_handoff.py"}
    assert hashes["owned_output_handoff.py"] == source.HANDOFF_SHA256


@pytest.mark.parametrize(
    "fault", ("none", "coverage", "decision", "cleanup", "owner", "failed_command", "waiter", "queue")
)
def test_scan_exit_preservation_rejects_altered_partial_failure_and_followups(monkeypatch, fault):
    frozen, intent, state, result = copy.deepcopy(predecessor.recorded(ROOT))
    if fault == "coverage":
        result["gpu_commands"].append(copy.deepcopy(result["gpu_commands"][-1]))
    elif fault == "decision":
        result["results"][6]["accepted"] = True
    elif fault == "cleanup":
        state["cleanup_exit_code"] = 0
    elif fault == "owner":
        state["child_generation"]["parent"] += 1
    elif fault == "failed_command":
        frozen["commands"][23] = ["unaccepted"]
    elif fault in ("waiter", "queue"):
        original = predecessor.read

        def changed(path):
            value = original(path)
            if path == ROOT / (predecessor.WAITER if fault == "waiter" else predecessor.QUEUE) / "state.json":
                value["status"] = "complete"
            return value

        monkeypatch.setattr(predecessor, "read", changed)
        monkeypatch.setattr(predecessor.parent, "read", changed)
    if fault == "none":
        saved = predecessor.transition_structure(ROOT, frozen, intent, state, result)
        assert (saved["completed_commands"], saved["case_decisions"]) == (23, 7)
        assert len(saved["accepted_cases"]) == 6 and len(saved["rejected_cases"]) == 1
        assert saved["decisions"] == result["results"]
        assert not saved["coverage_complete"] and not saved["numerically_accepted"]
        assert saved["queue"]["phases"] == []
    else:
        with pytest.raises(ValueError):
            predecessor.transition_structure(ROOT, frozen, intent, state, result)


@pytest.mark.parametrize("owner", ("controller", "validator", "waiter", "queue", "failed", "marked"))
def test_scan_exit_refuses_present_consumed_owner_directory_or_descendant(tmp_path, monkeypatch, owner):
    frozen, intent, state, result = predecessor.recorded(ROOT)
    monkeypatch.setattr(predecessor.previous.handoff, "PROC_ROOT", tmp_path)
    handles = {
        "controller": intent["pid"],
        "validator": state["child_pid"],
        "waiter": predecessor.read(ROOT / predecessor.WAITER / "intent.json")["pid"],
        "queue": predecessor.read(ROOT / predecessor.QUEUE / "intent.json")["pid"],
        "failed": predecessor.read(ROOT / predecessor.FAILED_LOG.replace(".log", ".execution.json"))["generation"][
            "pid"
        ],
    }
    if owner == "marked":
        monkeypatch.setattr(predecessor.previous.handoff, "marker_processes", lambda *args: {999999: {}})
    else:
        (tmp_path / str(handles[owner])).mkdir()
    with pytest.raises(RuntimeError):
        predecessor.require_absent(ROOT, frozen, intent, state, result)
