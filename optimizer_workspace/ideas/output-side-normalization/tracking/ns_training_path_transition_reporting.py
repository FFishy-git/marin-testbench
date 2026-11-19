# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project the distinct transition diagnostic from authenticated saved receipts."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "56858d0deab1af2c76cefdac6d462eaf1c597b237d636788e1b4215e914f87b8"
TITLE_VALUE = "NS Mixed-Exit-Bracket Diagnostic"
CPU_PATH_VALUE = (
    "run-preparation/ns-training-path-exit-transition-20261119-cpu/closure-corrected-20261119T1700/receipt.json"
)
PINS = {
    "TITLE": TITLE_VALUE,
    "CPU_PATH": CPU_PATH_VALUE,
    "CPU_SHA256": "ad18edf9b4f230e78bd7eef2a97f44bde307a1c524192d37f23491257b6fc392",
    "SEAL_SHA256": "070227bc90c040368f7ce2a28ef897425a1f24639d870377954827dbe20f9b49",
    "BINDING_SHA256": "18c9b21d16ff817059177e6746c3ae2246d39232396220c2b1deec805a33ca3d",
    "INTENT_SHA256": "42a901376f00df7b1672ab5d9dad610f669097ca0e48c16690ea4e48b2fb54e1",
}


def runtime_source() -> bytes:
    """Retain the proven receipt projection while changing namespace and predecessor coverage."""
    path = Path(__file__).with_name("ns_training_path_exit_reporting.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Transition reporting pinned exit-drain projection changed")
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_exit_source":
                    alias.name = "ns_training_path_transition_source"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_exit_drain", "ns_training_path_transition")
                .replace("prepare_ns_training_path_exit", "prepare_ns_training_path_transition")
                .replace("exit_drain_saved_", "exit_transition_saved_")
                .replace("Exit-drain reporting", "Transition reporting")
                .replace("Exit-drain evidence withheld:", "Transition evidence withheld:")
            )
    changes = 0
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name != "report_module":
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "tree" for target in node.targets
            ):
                node.value = ast.parse(
                    "source.TransitionNamespace().visit(source.parent_source.ExitNamespace().visit(ast.parse(payload)))",
                    mode="eval",
                ).body
                changes += 1
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value == "125 passing cases":
                    node.value = "129 passing cases"
                elif node.value == (
                    "Both failures remain original44/54commands14/18decisions and recovery2/54commands0/18decisions; "
                ):
                    node.value = (
                        "Preserved failures remain original44/54commands14/18decisions, "
                        "recovery2/54commands0/18decisions and exit-drain6/54commands2/18decisions; "
                    )
                elif node.value == "Exit-drain saved status:":
                    node.value = "Transition saved status:"
        for node in ast.walk(function):
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.BoolOp):
                continue
            comparison = node.test.values[-1]
            if not isinstance(comparison, ast.Compare) or not comparison.comparators:
                continue
            value = comparison.comparators[0]
            if isinstance(value, ast.Constant) and value.value in (44, 14):
                replacement = 6 if value.value == 44 else 2
                assignment = node.body[0]
                if not isinstance(assignment, ast.Assign) or not isinstance(assignment.value, ast.Call):
                    raise ValueError("Transition reporting parent coverage replacement changed")
                assignment.value.keywords[0].value = ast.Constant(value=replacement)
                changes += 1
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in PINS:
                    node.value = ast.Constant(value=PINS[target.id])
    if changes != 3:
        raise ValueError("Transition reporting namespace or retained-coverage projection differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
