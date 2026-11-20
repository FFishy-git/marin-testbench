# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve failed evidence and every numerical/ownership check under serialization."""

import ast
import copy
import importlib.util
from pathlib import Path

import ns_training_path_recheck_exit_source as parent_source
import ns_training_path_serialized_predecessor as predecessor
import ns_training_path_serialized_source as source
import prepare_ns_training_path_serialized as preparation
import pytest

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def test_serialized_source_preserves_every_numeric_command_and_full_owner_check(tmp_path):
    parent = source.recorded_parent(ROOT)
    generated = source.derived_sources(parent)
    path = tmp_path / "contract.py"
    path.write_bytes(generated["ns_training_path_serialized_contract.py"])
    spec = importlib.util.spec_from_file_location("serialized_contract", path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    assert contract.specs() == predecessor.original.specs()
    assert contract.TOLERANCE == predecessor.original.TOLERANCE
    assert len(contract.commands(Path(parent["candidate"]), ROOT, contract.locations(ROOT)[3])) == 54
    assert set(contract.locations(ROOT)).isdisjoint(predecessor.original.locations(ROOT))
    hashes = preparation.digest_bytes(preparation.helper_bytes(parent))
    assert all(hashes[name] == digest for name, digest in predecessor.original.NUMERICAL_READERS.items())
    assert hashes["owned_output_handoff.py"] == source.HANDOFF_SHA256
    assert hashes["test_owned_output_handoff.py"] == source.HANDOFF_TEST_SHA256
    before = ast.parse(parent_source.derived_sources(parent)["ns_training_path_recheck_exit_gpu_owner.py"])
    before = source.SerializedNamespace().visit(before)
    after = ast.parse(generated["ns_training_path_serialized_gpu_owner.py"])
    original = next(
        node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == "require_controller"
    )
    serialized = next(
        node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == "require_controller"
    )
    assert len(serialized.body) == 1 and isinstance(serialized.body[0], ast.With)
    assert ast.dump(ast.Module(body=serialized.body[0].body, type_ignores=[])) == ast.dump(
        ast.Module(body=original.body, type_ignores=[])
    )


@pytest.mark.parametrize("fault", ("none", "coverage", "decision", "cleanup", "owner", "failed_command", "waiter"))
def test_serialized_predecessor_preserves_failure_and_refuses_substitution(monkeypatch, fault):
    frozen, intent, state, result = copy.deepcopy(predecessor.recorded(ROOT))
    if fault == "coverage":
        result["gpu_commands"].append(copy.deepcopy(result["gpu_commands"][-1]))
    elif fault == "decision":
        result["results"].append({"accepted": True})
    elif fault == "cleanup":
        state["cleanup_exit_code"] = 0
    elif fault == "owner":
        state["child_generation"]["parent"] += 1
    elif fault == "failed_command":
        frozen["commands"][1] = ["unaccepted"]
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
        assert (saved["completed_commands"], saved["case_decisions"]) == (1, 0)
        assert saved["accepted_cases"] == saved["rejected_cases"] == []
        assert not saved["coverage_complete"] and not saved["numerically_accepted"]
        assert saved["queue"]["status"] == "cpu_archive_accepted_unissued"
    else:
        with pytest.raises(ValueError):
            predecessor.transition_structure(ROOT, frozen, intent, state, result)


@pytest.mark.parametrize("owner", ("controller", "validator", "waiter", "failed", "marked"))
def test_serialized_predecessor_refuses_original_consumed_owner_or_descendant(tmp_path, monkeypatch, owner):
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


def test_serialized_predecessor_refuses_launching_preserved_actual_lr_queue(tmp_path):
    (tmp_path / predecessor.QUEUE).mkdir(parents=True)
    with pytest.raises(ValueError):
        predecessor.queue_structure(tmp_path)
    assert (tmp_path / predecessor.QUEUE).is_dir()
