# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project the distinct scan-exit diagnostic from authenticated saved receipts."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "9e8db207b04b033740187dee1c186228ff54f60d9a13391811fe71f62eee6a54"
TITLE_VALUE = "NS Missing-Environment Scan-Exit Diagnostic"
CPU_PATH_VALUE = "run-preparation/ns-training-path-scan-exit-20261119-cpu/reviewed-20261119T2058/receipt.json"
PINS = {
    "TITLE": TITLE_VALUE,
    "CPU_PATH": CPU_PATH_VALUE,
    "CPU_SHA256": "eec8b5ec97f6531bec4447078ff99b823d20458038574099fe68106b7b7bc599",
    "SEAL_SHA256": "59012d6f15ef1a3b4fb5cddb8049e8dd18fd63efe3f7348a2a01a923ccc1d42d",
    "BINDING_SHA256": "7af74fd70d7ea1fbb68f09209f93ef62e19a86c17d1cb200dde99872a1b27f98",
    "INTENT_SHA256": "0020cfa6f52eea4259612d17d080029df07d1151fda088a0655117b4aa97ca22",
}


def runtime_source() -> bytes:
    """Keep the accepted projection and bind the new namespace and failed predecessor."""
    path = Path(__file__).with_name("ns_training_path_transition_reporting.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit reporting pinned transition projection changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    tree = ast.parse(namespace["runtime_source"]())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_transition_source":
                    alias.name = "ns_training_path_scan_exit_source"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_transition", "ns_training_path_scan_exit")
                .replace("prepare_ns_training_path_transition", "prepare_ns_training_path_scan_exit")
                .replace("exit_transition_saved_", "scan_exit_saved_")
                .replace("Transition reporting", "Scan-exit reporting")
                .replace("Transition evidence withheld:", "Scan-exit evidence withheld:")
                .replace("Transition saved status:", "Scan-exit saved status:")
                .replace(
                    "and exit-drain6/54commands2/18decisions; ",
                    "exit-drain6/54commands2/18decisions and transition23/54commands7/18decisions; ",
                )
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
                    "source.ScanExitNamespace().visit(source.parent_source.TransitionNamespace().visit("
                    "source.parent_source.parent_source.ExitNamespace().visit(ast.parse(payload))))",
                    mode="eval",
                ).body
                changes += 1
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.BoolOp):
                continue
            comparison = node.test.values[-1]
            if not isinstance(comparison, ast.Compare) or not comparison.comparators:
                continue
            value = comparison.comparators[0]
            if isinstance(value, ast.Constant) and value.value in (44, 14):
                assignment = node.body[0]
                if not isinstance(assignment, ast.Assign) or not isinstance(assignment.value, ast.Call):
                    raise ValueError("Scan-exit reporting parent coverage replacement changed")
                assignment.value.keywords[0].value = ast.Constant(value=23 if value.value == 44 else 7)
                changes += 1
    pins = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in PINS:
                    node.value = ast.Constant(value=PINS[target.id])
                    pins.add(target.id)
    if changes != 3 or pins != set(PINS):
        raise ValueError("Scan-exit reporting namespace, pins or retained coverage differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
