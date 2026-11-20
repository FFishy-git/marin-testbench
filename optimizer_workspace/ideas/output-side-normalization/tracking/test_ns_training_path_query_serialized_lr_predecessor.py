# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the query terminal handoff against executable persisted boundary fixtures."""

import ast
import hashlib
import json
from pathlib import Path

import ns_training_path_query_serialized_lr_predecessor as query
import prepare_ns_training_path_query_serialized_lr_predecessor as preparation
import pytest

PARENT_TEST_SHA256 = "b82a51ac493ea6313c05e5c4d5bb62412df5f261f1d07c3c16b6240ade020ef8"
FIXTURE_LINEAGE = (
    ("serialized", 1, 0, (8, 2)),
    ("recheck", 8, 2, (23, 7)),
    ("scan", 23, 7, (6, 2)),
    ("transition", 6, 2, None),
    ("exit", 2, 0, None),
)


def fixture_lineage(root, failed, monkeypatch, fault):
    """Model the nonflattened historical seals independently of the production table."""
    pins = []
    for name, commands, decisions, previous in FIXTURE_LINEAGE:
        summaries = {"retained_summary": (commands, decisions)}
        ancestor = {
            "root": str(root),
            "status": f"failed_{name}_fixture",
            "gpu_execution_authorized": False,
            "production_authorized": False,
            "original_retained_summary": {
                "completed_commands": 44,
                "case_decisions": 14,
                "coverage_complete": False,
                "numerically_accepted": False,
            },
            "recovery_retained_summary": {
                "completed_commands": 2,
                "case_decisions": 0,
                "coverage_complete": False,
                "numerically_accepted": False,
            },
        }
        if previous is not None:
            summaries["previous_retained_summary"] = previous
        for key, (completed, count) in summaries.items():
            ancestor[key] = {
                "completed_commands": completed + (1 if fault == name and key == "retained_summary" else 0),
                "case_decisions": count,
                "coverage_complete": False,
                "numerically_accepted": False,
            }
        path = root / f"lineage-{name}.json"
        path.write_text(json.dumps(ancestor) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        failed["evidence"][path.name] = {"sha256": digest, "identity": query.file_identity(path)}
        pins.append((path.name, digest, ancestor["status"], summaries))
    failed["retained_summary"]["completed_commands"] = 1 if fault == "query" else 0
    failed["retained_summary"]["case_decisions"] = 0
    failed["previous_retained_summary"] = {
        "completed_commands": 1,
        "case_decisions": 0,
        "coverage_complete": False,
        "numerically_accepted": False,
    }
    failed["recovery_retained_summary"] = {
        "completed_commands": 2,
        "case_decisions": 0,
        "coverage_complete": False,
        "numerically_accepted": False,
    }
    monkeypatch.setattr(query, "FAILED_LINEAGE", tuple(pins))


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_ns_training_path_transition_lr_predecessor.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_TEST_SHA256:
        raise ValueError("Query-serialized predecessor pinned external-boundary tests changed")
    parent = ast.parse(payload)
    parent.body = [
        node
        for node in parent.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "exec"
        )
    ]
    namespace = {"__file__": str(path)}
    exec(compile(parent, str(path), "exec"), namespace)
    tree = ast.parse(namespace["inherited_tests"]())
    tree.body.extend(
        node
        for node in parent.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_cpu_publication_preserves_real_staged_source_validation_outcome"
    )
    tree.body[:0] = ast.parse(
        "import tarfile\nfrom selected_lr_release_owner import sha256\nfrom watch_wave5 import write\n"
    ).body
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "transition":
            node.id = "query"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_transition_lr_predecessor":
                    alias.name = "ns_training_path_query_serialized_lr_predecessor"
                elif alias.name == "prepare_ns_training_path_transition_lr_predecessor":
                    alias.name = "prepare_ns_training_path_query_serialized_lr_predecessor"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_transition", "ns_training_path_query_serialized")
                .replace("ns-training-path-exit-transition-20261119", "ns-training-path-query-serialized-20261120")
                .replace(
                    "ns-training-path-exit-transition-terminal-followup-20261119",
                    "ns-training-path-query-serialized-terminal-followup-20261120",
                )
                .replace(
                    "failed_exit_drain_and_waiter_sealed_for_distinct_transition_not_numerical_acceptance",
                    "failed_whole_inspection_and_waiter_sealed_for_query_serialization_not_numerical_acceptance",
                )
                .replace("transition_owners", "query_serialized_owners")
                .replace("Transition", "Query-serialized")
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            if len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 174:
                node.args[0].value = 194
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "retained_summary" and isinstance(value, ast.Call):
                    for keyword in value.keywords:
                        if keyword.arg == "completed_commands":
                            keyword.value = ast.Constant(0)
    fixture = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "completed")
    fixture.args.args.append(ast.arg(arg="lineage_fault"))
    index = next(
        index
        for index, node in enumerate(fixture.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "record"
        and ast.unparse(node.value.args[0]) == "root / predecessor.FAILED_SEAL"
    )
    fixture.body[index:index] = ast.parse("fixture_lineage(root, failed, monkeypatch, lineage_fault)").body
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


@pytest.fixture
def lineage_fault():
    return None


exec(compile(inherited_tests(), __file__, "exec"), globals())


@pytest.mark.parametrize("lineage_fault", ("query", "serialized", "recheck", "scan", "transition", "exit"))
def test_rebound_ancestor_coverage_refuses_complete_terminal_without_publication(completed, lineage_fault):
    root, _ = completed
    before = {path: query.file_identity(path) for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        query.evidence(root)
    assert before == {path: query.file_identity(path) for path in before}
    assert not (root / query.SEAL).exists()


@pytest.mark.parametrize("fault", ("missing", "substituted", "unretained"))
def test_lost_ancestor_evidence_refuses_source_preparation_without_publication(completed, fault, monkeypatch):
    root, snapshot = completed
    numerical = root / "fixture-numerical-cpu.json"
    numerical.write_text('{"helpers_sha256":{}}\n')
    monkeypatch.setattr(preparation, "NUMERICAL_CPU", numerical.name)
    monkeypatch.setattr(preparation, "NUMERICAL_CPU_SHA256", hashlib.sha256(numerical.read_bytes()).hexdigest())
    path = root / "lineage-scan.json"
    if fault == "missing":
        path.unlink()
    elif fault == "substituted":
        path.write_text('{"status":"substituted"}\n')
    else:
        failed = snapshot["frozen"]["allocation_predecessor"]
        failed["evidence"].pop(path.name)
        (root / query.FAILED_SEAL).write_text(json.dumps(failed) + "\n")
        query_digest = hashlib.sha256((root / query.FAILED_SEAL).read_bytes()).hexdigest()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(query, "FAILED_SEAL_SHA256", query_digest)
            with pytest.raises(ValueError):
                preparation.source_plan(root)
        assert not (root / query.SEAL).exists()
        return
    with pytest.raises((OSError, ValueError)):
        preparation.source_plan(root)
    assert not (root / query.SEAL).exists()


def test_source_plan_retains_conflicting_numerical_and_execution_archives(completed, monkeypatch):
    root, snapshot = completed
    numerical = root / "old-numerical-cpu.json"
    reader = root / "reader/tracking/ns_training_path_query_serialized_followup.py"
    numerical.write_text(json.dumps({"helpers_sha256": {reader.name: "0" * 64}}) + "\n")
    monkeypatch.setattr(preparation, "NUMERICAL_CPU", numerical.name)
    monkeypatch.setattr(preparation, "NUMERICAL_CPU_SHA256", hashlib.sha256(numerical.read_bytes()).hexdigest())
    before = {path: query.file_identity(path) for path in root.rglob("*") if path.is_file()}
    plan = preparation.source_plan(root)
    assert plan["conflicting_helpers"] == {
        reader.name: {
            "numerical_sha256": "0" * 64,
            "execution_sha256": snapshot["cpu"]["helpers_sha256"][reader.name],
        }
    }
    assert plan["execution_closure_merged"] is False
    assert plan["gpu_execution_authorized"] is False
    assert before == {path: query.file_identity(path) for path in before}
