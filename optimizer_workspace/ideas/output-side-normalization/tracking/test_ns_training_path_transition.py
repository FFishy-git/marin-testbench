# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain source/owner checks and reject altered partial diagnostic evidence."""

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path

import ns_training_path_transition_predecessor as predecessor
import ns_training_path_transition_source as source
import pytest

PARENT_TEST_SHA256 = "a12cfa424692829f48a0e558da5ed2514afc82593a9e870e796098a6d85ae148"
ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_ns_training_path_recovery.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_TEST_SHA256:
        raise ValueError("Exit-transition inherited source acceptance tests changed")
    from_namespace = source.parent_source.ExitNamespace()
    tree = source.TransitionNamespace().visit(from_namespace.visit(ast.parse(payload)))
    for node in list(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "test_derived_recovery_keeps_numerical_protocol_and_changes_execution_namespace",
            "test_failed_attempt_owner_absence_rejects_present_handles_and_descendants",
        ):
            tree.body.remove(node)
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())


def test_transition_keeps_original_numerical_contract_and_fresh_execution_identity(tmp_path):
    parent = source.recorded_parent(ROOT)
    generated = source.derived_sources(parent)
    path = tmp_path / "contract.py"
    path.write_bytes(generated["ns_training_path_transition_contract.py"])
    spec = importlib.util.spec_from_file_location("transition_contract_test", path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    commands = contract.commands(Path(parent["candidate"]), ROOT, contract.locations(ROOT)[3])
    assert len(commands) == 54
    assert contract.specs() == predecessor.original.specs()
    assert contract.TOLERANCE == predecessor.original.TOLERANCE
    assert set(contract.locations(ROOT)).isdisjoint(predecessor.original.locations(ROOT))
    assert set(contract.locations(ROOT)).isdisjoint(
        Path(ROOT / path)
        for path in (
            f"run-preparation/{predecessor.OLD_NAMESPACE}-binding.json",
            f"run-preparation/{predecessor.OLD_NAMESPACE}-state.json",
        )
    )


@pytest.mark.parametrize("fault", ("none", "coverage", "decision", "cleanup", "owner", "failed_command", "waiter"))
def test_failed_exit_and_waiter_preservation_rejects_altered_evidence(monkeypatch, fault):
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
        frozen["commands"][6] = ["unaccepted-command"]
    elif fault == "waiter":
        real_read = predecessor.read

        def changed(path):
            value = real_read(path)
            if path == ROOT / predecessor.WAITER / "state.json":
                value["status"] = "terminal_reconstructed"
            return value

        monkeypatch.setattr(predecessor, "read", changed)
    if fault == "none":
        saved = predecessor.transition_structure(ROOT, frozen, intent, state, result)
        assert (saved["completed_commands"], saved["case_decisions"]) == (6, 2)
        assert saved["coverage_complete"] is False
        assert saved["numerically_accepted"] is False
        assert saved["decisions"] == result["results"]
    else:
        with pytest.raises(ValueError):
            predecessor.transition_structure(ROOT, frozen, intent, state, result)


@pytest.mark.parametrize("fault", ("owner_directory", "marked_descendant"))
def test_failed_exit_absence_refuses_incomplete_procfs_or_surviving_descendant(tmp_path, monkeypatch, fault):
    frozen, intent, state, result = predecessor.recorded(ROOT)
    monkeypatch.setattr(predecessor.previous.handoff, "PROC_ROOT", tmp_path)
    if fault == "owner_directory":
        (tmp_path / str(intent["pid"])).mkdir()
    else:
        monkeypatch.setattr(predecessor.previous.handoff, "marker_processes", lambda *args: {999999: {}})
    with pytest.raises(RuntimeError):
        predecessor.require_absent(ROOT, frozen, intent, state, result)
