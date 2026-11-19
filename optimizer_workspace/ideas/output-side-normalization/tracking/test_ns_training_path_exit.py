# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reuse accepted recovery evidence tests against the distinct exit-drain namespace."""

import ast
import copy
import importlib.util
from pathlib import Path

import ns_training_path_exit_predecessor as predecessor
import ns_training_path_exit_source as exit_source
import pytest
from selected_lr_release_owner import sha256

PARENT_TEST_SHA256 = "a12cfa424692829f48a0e558da5ed2514afc82593a9e870e796098a6d85ae148"
path = Path(__file__).with_name("test_ns_training_path_recovery.py")
ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def inherited_tests() -> bytes:
    if sha256(path) != PARENT_TEST_SHA256:
        raise ValueError("Exit-drain inherited recovery tests changed")
    tree = exit_source.ExitNamespace().visit(ast.parse(path.read_bytes()))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == (
            "test_derived_recovery_keeps_numerical_protocol_and_changes_execution_namespace"
        ):
            tree.body.remove(node)
            break
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())


def test_exit_drain_protocol_retains_all54_commands_and_original_numerical_settings(tmp_path):
    parent = exit_source.recorded_parent(ROOT)
    generated = exit_source.derived_sources(parent)
    path = tmp_path / "contract.py"
    path.write_bytes(generated["ns_training_path_exit_contract.py"])
    spec = importlib.util.spec_from_file_location("exit_contract_test", path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    candidate = Path(parent["candidate"])
    commands = contract.commands(candidate, ROOT, contract.locations(ROOT)[3])
    assert len(commands) == 54
    assert contract.specs() == predecessor.original.specs()
    assert contract.TOLERANCE == predecessor.original.TOLERANCE
    assert set(contract.locations(ROOT)).isdisjoint(predecessor.original.locations(ROOT))


@pytest.mark.parametrize("fault", ("none", "completion", "decision", "restoration", "command", "owner"))
def test_recovery_preservation_rejects_changed_coverage_or_execution_evidence(fault):
    frozen, intent, state, result = copy.deepcopy(predecessor.recorded(ROOT))
    if fault == "completion":
        result["gpu_commands"].append(copy.deepcopy(result["gpu_commands"][-1]))
    elif fault == "decision":
        result["results"] = [{"accepted": True}]
    elif fault == "restoration":
        state["keeper_restore"] = 1
    elif fault == "command":
        frozen["commands"][2] = ["substituted-command"]
    elif fault == "owner":
        state["child_generation"]["parent"] += 1
    if fault == "none":
        saved = predecessor.recovery_structure(ROOT, frozen, intent, state, result)
        assert saved["completed_commands"] == 2
        assert saved["case_decisions"] == 0
        assert saved["coverage_complete"] is False
    else:
        with pytest.raises(ValueError):
            predecessor.recovery_structure(ROOT, frozen, intent, state, result)


def test_recovery_owner_directory_with_unreadable_files_is_not_absence(tmp_path, monkeypatch):
    frozen, intent, state, result = predecessor.recorded(ROOT)
    directory = tmp_path / str(intent["pid"])
    directory.mkdir()
    monkeypatch.setattr(predecessor.handoff, "PROC_ROOT", tmp_path)
    with pytest.raises(RuntimeError):
        predecessor.require_absent(ROOT, frozen, intent, state, result)
    assert directory.exists()
