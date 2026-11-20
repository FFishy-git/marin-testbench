# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise preserved pre-command failure and distinct unchanged numerical protocols."""

import ast
import copy
import importlib.util
from pathlib import Path

import ns_training_path_query_serialized_source as parent_source
import ns_training_path_streaming_predecessor as predecessor
import ns_training_path_streaming_source as source
import prepare_ns_training_path_streaming as preparation
import pytest

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def test_streaming_source_retains_numeric_protocol_query_lock_and_complete_scan(tmp_path):
    parent = source.recorded_parent(ROOT)
    generated = source.derived_sources(parent)
    path = tmp_path / "contract.py"
    path.write_bytes(generated["ns_training_path_streaming_contract.py"])
    spec = importlib.util.spec_from_file_location("streaming_contract_under_test", path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    assert contract.specs() == predecessor.original.specs()
    assert contract.TOLERANCE == predecessor.original.TOLERANCE
    assert len(contract.commands(Path(parent["candidate"]), ROOT, contract.locations(ROOT)[3])) == 54
    assert set(contract.locations(ROOT)).isdisjoint(predecessor.original.locations(ROOT))
    hashes = preparation.digest_bytes(preparation.helper_bytes(parent))
    assert all(hashes[name] == digest for name, digest in predecessor.original.NUMERICAL_READERS.items())
    assert hashes["owned_output_handoff.py"] == source.HANDOFF_SHA256 != parent_source.HANDOFF_SHA256
    assert hashes["test_owned_output_handoff.py"] == source.HANDOFF_TEST_SHA256
    before = source.StreamingNamespace().visit(
        ast.parse(parent_source.derived_sources(parent)["ns_training_path_query_serialized_gpu_owner.py"])
    )
    after = ast.parse(generated["ns_training_path_streaming_gpu_owner.py"])
    assert ast.dump(before) == ast.dump(after)
    assert all(
        preparation.sha256(Path(parent["candidate"]) / name) == digest
        for name, digest in predecessor.original.TRAINING_FILES.items()
    )


@pytest.mark.parametrize("fault", ("none", "coverage", "decision", "cleanup", "owner", "attribution", "waiter"))
def test_streaming_predecessor_retains_six_commands_two_controls_and_refuses_substitution(monkeypatch, fault):
    frozen, intent, state, result = copy.deepcopy(predecessor.recorded(ROOT))
    if fault == "coverage":
        result["gpu_commands"].pop()
    elif fault == "decision":
        result["results"][1]["accepted"] = False
    elif fault == "cleanup":
        state["cleanup_exit_code"] = 0
    elif fault == "owner":
        state["child_generation"]["parent"] += 1
    elif fault == "attribution":
        result["error"] = "RuntimeError: unrecorded training failure"
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
        assert (saved["completed_commands"], saved["case_decisions"]) == (6, 2)
        assert saved["accepted_cases"] == [trial["name"] for trial in frozen["plan"]["trials"][:2]]
        assert saved["failed_command"] is None
        assert saved["precommand_failure"]["spawn_status"] == "not_spawned"
        assert saved["precommand_failure"]["execution_receipt"] is None
        assert not saved["coverage_complete"] and not saved["numerically_accepted"]
        assert saved["queue"]["status"] == "cpu_archive_accepted_unissued"
    else:
        with pytest.raises(ValueError):
            predecessor.transition_structure(ROOT, frozen, intent, state, result)


@pytest.mark.parametrize("owner", ("controller", "validator", "waiter", "completed", "marked"))
def test_streaming_predecessor_refuses_live_consumed_owners_or_descendants(tmp_path, monkeypatch, owner):
    frozen, intent, state, result = predecessor.recorded(ROOT)
    monkeypatch.setattr(predecessor.previous.handoff, "PROC_ROOT", tmp_path)
    handles = {
        "controller": intent["pid"],
        "validator": state["child_pid"],
        "waiter": predecessor.read(ROOT / predecessor.WAITER / "intent.json")["pid"],
        "completed": result["gpu_commands"][-1]["generation"]["pid"],
    }
    if owner == "marked":
        monkeypatch.setattr(predecessor.previous.handoff, "marker_processes", lambda *args: {999999: {}})
    else:
        (tmp_path / str(handles[owner])).mkdir()
    with pytest.raises(RuntimeError):
        predecessor.require_absent(ROOT, frozen, intent, state, result)


@pytest.mark.parametrize("artifact", ("directory", "log", "execution_sidecar"))
def test_streaming_predecessor_refuses_fictional_unspawned_execution_evidence(tmp_path, artifact):
    output = tmp_path / "continuous"
    trial = {"name": "unspawned", "continuous": str(output), "resumed": str(tmp_path / "resumed")}
    frozen = {"commands": [[] for _ in range(7)], "plan": {"trials": [{}, {}, trial]}}
    log = tmp_path / predecessor.WORK / "unspawned_continuous.log"
    if artifact == "directory":
        output.mkdir()
        retained = output
    else:
        retained = log if artifact == "log" else log.with_suffix(".execution.json")
        retained.parent.mkdir(parents=True)
        retained.write_text("unaccepted invented evidence\n")
    with pytest.raises(ValueError):
        predecessor.require_unspawned(tmp_path, frozen)
    assert retained.exists()


def test_streaming_predecessor_keeps_accepted_actual_lr_queue_unissued(tmp_path):
    (tmp_path / predecessor.QUEUE).mkdir(parents=True)
    with pytest.raises(ValueError):
        predecessor.queue_structure(tmp_path)
    assert (tmp_path / predecessor.QUEUE).is_dir()
