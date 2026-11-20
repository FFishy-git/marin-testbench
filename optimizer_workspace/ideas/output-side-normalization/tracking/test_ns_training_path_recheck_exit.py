# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain failed scan-exit evidence and unchanged original numerical contracts."""

import copy
import importlib.util
from pathlib import Path

import ns_training_path_recheck_exit_predecessor as predecessor
import ns_training_path_recheck_exit_source as source
import prepare_ns_training_path_recheck_exit as preparation
import pytest

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def test_recheck_exit_preserves_numerical_contract_and_only_replaces_owner_handoff(tmp_path):
    parent = source.recorded_parent(ROOT)
    generated = source.derived_sources(parent)
    path = tmp_path / "contract.py"
    path.write_bytes(generated["ns_training_path_recheck_exit_contract.py"])
    spec = importlib.util.spec_from_file_location("recheck_exit_contract", path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    assert contract.specs() == predecessor.original.specs()
    assert contract.TOLERANCE == predecessor.original.TOLERANCE
    assert len(contract.commands(Path(parent["candidate"]), ROOT, contract.locations(ROOT)[3])) == 54
    assert set(contract.locations(ROOT)).isdisjoint(predecessor.original.locations(ROOT))
    hashes = preparation.digest_bytes(preparation.helper_bytes(parent))
    assert all(hashes[name] == digest for name, digest in predecessor.original.NUMERICAL_READERS.items())
    assert {name for name, digest in parent["helpers_sha256"].items() if hashes[name] != digest} == {
        "owned_output_handoff.py",
    }
    assert hashes["owned_output_handoff.py"] == source.HANDOFF_SHA256
    assert hashes["test_owned_output_handoff.py"] == source.HANDOFF_TEST_SHA256


@pytest.mark.parametrize("fault", ("none", "coverage", "decision", "cleanup", "owner", "failed_command", "waiter"))
def test_recheck_exit_retains_partial_failure_and_refuses_substitution(monkeypatch, fault):
    frozen, intent, state, result = copy.deepcopy(predecessor.recorded(ROOT))
    if fault == "coverage":
        result["gpu_commands"].append(copy.deepcopy(result["gpu_commands"][-1]))
    elif fault == "decision":
        result["results"][1]["accepted"] = False
    elif fault == "cleanup":
        state["cleanup_exit_code"] = 0
    elif fault == "owner":
        state["child_generation"]["parent"] += 1
    elif fault == "failed_command":
        frozen["commands"][8] = ["unaccepted"]
    elif fault == "waiter":
        original = predecessor.read

        def changed(path):
            value = original(path)
            if path == ROOT / predecessor.WAITER / "state.json":
                value["status"] = "complete"
            return value

        monkeypatch.setattr(predecessor, "read", changed)
    if fault == "none":
        saved = predecessor.transition_structure(ROOT, frozen, intent, state, result)
        assert (saved["completed_commands"], saved["case_decisions"]) == (8, 2)
        assert len(saved["accepted_cases"]) == 2 and saved["rejected_cases"] == []
        assert saved["decisions"] == result["results"]
        assert not saved["coverage_complete"] and not saved["numerically_accepted"]
        assert saved["queue"]["status"] == "cpu_archive_accepted_unissued"
    else:
        with pytest.raises(ValueError):
            predecessor.transition_structure(ROOT, frozen, intent, state, result)


@pytest.mark.parametrize("owner", ("controller", "validator", "waiter", "failed", "marked"))
def test_recheck_exit_refuses_consumed_owner_or_marked_descendant(tmp_path, monkeypatch, owner):
    frozen, intent, state, result = predecessor.recorded(ROOT)
    monkeypatch.setattr(predecessor.previous.handoff, "PROC_ROOT", tmp_path)
    failed = predecessor.read(ROOT / predecessor.FAILED_LOG.replace(".log", ".execution.json"))
    handles = {
        "controller": intent["pid"],
        "validator": state["child_pid"],
        "waiter": predecessor.read(ROOT / predecessor.WAITER / "intent.json")["pid"],
        "failed": failed["generation"]["pid"],
    }
    if owner == "marked":
        monkeypatch.setattr(predecessor.previous.handoff, "marker_processes", lambda *args: {999999: {}})
    else:
        (tmp_path / str(handles[owner])).mkdir()
    with pytest.raises(RuntimeError):
        predecessor.require_absent(ROOT, frozen, intent, state, result)


def test_recheck_exit_refuses_issuing_preserved_actual_lr_queue(tmp_path):
    (tmp_path / predecessor.QUEUE).mkdir(parents=True)
    with pytest.raises(ValueError):
        predecessor.queue_structure(tmp_path)
    assert (tmp_path / predecessor.QUEUE).is_dir()
