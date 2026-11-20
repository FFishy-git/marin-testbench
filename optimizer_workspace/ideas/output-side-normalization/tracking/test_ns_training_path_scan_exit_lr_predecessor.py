# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise scan-exit handoff boundaries through persisted executable evidence fixtures."""

import ast
import hashlib
import json
from pathlib import Path

import ns_training_path_scan_exit_lr_predecessor as transition
import prepare_ns_training_path_scan_exit_lr_predecessor as preparation
import pytest

PARENT_TEST_SHA256 = "b82a51ac493ea6313c05e5c4d5bb62412df5f261f1d07c3c16b6240ade020ef8"


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_ns_training_path_transition_lr_predecessor.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_TEST_SHA256:
        raise ValueError("Scan-exit actual-LR pinned external-boundary tests changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    tree = ast.parse(namespace["inherited_tests"]())
    parent = ast.parse(payload)
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
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_transition_lr_predecessor":
                    alias.name = "ns_training_path_scan_exit_lr_predecessor"
                elif alias.name == "prepare_ns_training_path_transition_lr_predecessor":
                    alias.name = "prepare_ns_training_path_scan_exit_lr_predecessor"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = transition.literal(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            if len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 174:
                node.args[0].value = 179
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "retained_summary" and isinstance(value, ast.Call):
                    for keyword in value.keywords:
                        if keyword.arg == "completed_commands":
                            keyword.value = ast.Constant(value=23)
                        elif keyword.arg == "case_decisions":
                            keyword.value = ast.Constant(value=7)
                    node.keys.append(ast.Constant(value="previous_retained_summary"))
                    node.values.append(
                        ast.parse(
                            "dict(completed_commands=6,case_decisions=2,coverage_complete=False,numerically_accepted=False)",
                            mode="eval",
                        ).body
                    )
                    break
    fixture = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "completed")
    fixture.args.args.append(ast.arg(arg="failed_summary"))
    index = next(
        index
        for index, node in enumerate(fixture.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "record"
        and ast.unparse(node.value.args[0]) == "root / predecessor.FAILED_SEAL"
    )
    fixture.body[index:index] = ast.parse(
        """if failed_summary is not None:
    failed[failed_summary]["completed_commands"] += 1
"""
    ).body
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


@pytest.fixture
def failed_summary():
    return None


exec(compile(inherited_tests(), __file__, "exec"), globals())


@pytest.mark.parametrize(
    "failed_summary",
    ("retained_summary", "previous_retained_summary", "recovery_retained_summary", "original_retained_summary"),
)
def test_authenticated_rebound_failed_coverage_refuses_without_publication(completed, failed_summary):
    root, _ = completed
    before = {path: transition.file_identity(path) for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        transition.evidence(root)
    assert before == {path: transition.file_identity(path) for path in before}
    assert not (root / transition.SEAL).exists()


def test_source_plan_retains_conflicting_numerical_and_execution_archives(completed, monkeypatch):
    root, snapshot = completed
    numerical = root / "old-numerical-cpu.json"
    reader = root / "reader/tracking/ns_training_path_scan_exit_followup.py"
    numerical.write_text(json.dumps({"helpers_sha256": {reader.name: "0" * 64}}) + "\n")
    monkeypatch.setattr(preparation, "NUMERICAL_CPU", numerical.name)
    monkeypatch.setattr(preparation, "NUMERICAL_CPU_SHA256", hashlib.sha256(numerical.read_bytes()).hexdigest())
    before = {path: transition.file_identity(path) for path in root.rglob("*") if path.is_file()}
    plan = preparation.source_plan(root)
    assert plan["conflicting_helpers"] == {
        reader.name: {
            "numerical_sha256": "0" * 64,
            "execution_sha256": snapshot["cpu"]["helpers_sha256"][reader.name],
        }
    }
    assert plan["execution_closure_merged"] is False
    assert plan["gpu_execution_authorized"] is False
    assert before == {path: transition.file_identity(path) for path in before}
