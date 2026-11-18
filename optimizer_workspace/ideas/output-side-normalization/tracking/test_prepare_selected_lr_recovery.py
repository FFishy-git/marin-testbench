# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the generated recovery pipeline preserves original numerical-validation identity."""

import ast
from pathlib import Path

import prepare_selected_lr_recovery as recovery


def test_recovery_namespaces_are_distinct_and_validation_paths_preserved():
    directory = Path(recovery.__file__).parent
    for name in recovery.OWNED_HELPERS:
        source = (directory / name).read_bytes()
        generated = recovery.transformed(name, source).decode()
        if name.endswith(".py"):
            ast.parse(generated)
        assert "wave11-launch-binding.json" not in generated or name in (
            "selected_lr_production.py",
            "selected_lr_run_history.py",
            "selected_lr_release_owner.py",
            "test_selected_lr_release_owner.py",
        )
        assert "runs/wave11" not in generated
    original = (directory / "selected_lr_validation.py").read_bytes()
    assert recovery.transformed("selected_lr_validation.py", original) == original
    production = recovery.transformed(
        "selected_lr_production.py", (directory / "selected_lr_production.py").read_bytes()
    )
    tree = ast.parse(production)
    functions = {node.name: ast.unparse(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "failure_probe(root, arm, manager_lock)" in functions["allocation"]
    assert "validation_terminal(root)" in functions["terminal_validation_observation"]
    assert "wave12-launch-state.json" in functions["launch"]
    assert "release_accepted_keeper(keeper, state, path, manager_lock)" in functions["run_arm"]
    assert "restore_accepted_keeper(manager_lock)" in functions["run_arm"]
    assert "state.get('keeper_release_complete') is not True" in functions["run_arm"]
