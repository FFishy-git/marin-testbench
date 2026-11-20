# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project only receipt-authenticated serialized diagnostic progress into both journals."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_recheck_exit_reporting as parent

PARENT_SOURCE_SHA256 = "696b19372f97c91257a47869a03b979b01afec92db80ceec6199079e349b0a1d"
PINS = {
    "TITLE": "NS Serialized Ownership Diagnostic",
    "CPU_PATH": "run-preparation/ns-training-path-serialized-20261119-cpu/reviewed-20261119T2340/receipt.json",
    "CPU_SHA256": "293daf7c2a91b5ba2e2081646521f44bd85092d7f89b2a1f1b256a8b4a1aeda2",
    "SEAL_SHA256": "dd3d333dfbe00686d8a25da358ae783c11cc0163a679c750338076b5c9a0bdfe",
    "BINDING_SHA256": "5789abe0b6c47fd910917cba35e00b2b2d519e66c867dab2214418327bbddc68",
    "INTENT_SHA256": "e782a7fcdc67c137c64c4d65799eb45b56c4800f969d0f1cb0640ed52295bef1",
}


def runtime_source() -> bytes:
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Serialized reporting pinned recheck projection changed")
    tree = ast.parse(payload)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "exec"
        )
    ]
    namespace = {"__file__": str(path)}
    exec(compile(tree, str(path), "exec"), namespace)
    tree = ast.parse(namespace["runtime_source"]())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_recheck_exit_source":
                    alias.name = "ns_training_path_serialized_source"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_recheck_exit", "ns_training_path_serialized")
                .replace("recheck_exit_saved_", "serialized_saved_")
                .replace("Recheck-exit reporting", "Serialized reporting")
                .replace("Recheck-exit evidence withheld:", "Serialized evidence withheld:")
                .replace("Recheck-exit saved status:", "Serialized saved status:")
                .replace("136 passing cases", "141 passing cases")
                .replace(
                    "Source/configuration changes are restricted to the owner adapter and the measured handoff helper.",
                    "Source/configuration changes are restricted to the owner adapter and full ownership-inspection "
                    "serialization; the accepted handoff helper and trainer/configuration bytes are retained.",
                )
                .replace(
                    "and scan-exit8/54commands2/18decisions; ",
                    "and scan-exit8/54commands2/18decisions and recheck1/54commands0/18decisions; ",
                )
            )
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "report_module")
    assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "tree" for target in node.targets)
    )
    assignment.value = ast.parse(
        "serialization_report_tree(source.SerializedNamespace().visit(source.parent_source.RecheckExitNamespace().visit("
        "source.parent_source.parent_source.ScanExitNamespace().visit("
        "source.parent_source.parent_source.parent_source.TransitionNamespace().visit("
        "source.parent_source.parent_source.parent_source.parent_source.ExitNamespace().visit(ast.parse(payload)))))))",
        mode="eval",
    ).body
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.BoolOp):
            continue
        comparison = node.test.values[-1]
        if not isinstance(comparison, ast.Compare) or not comparison.comparators:
            continue
        value = comparison.comparators[0]
        if isinstance(value, ast.Constant) and value.value in (44, 14):
            node.body[0].value.keywords[0].value = ast.Constant(1 if value.value == 44 else 0)
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in PINS:
                    node.value = ast.Constant(PINS[target.id])
                    changed.add(target.id)
    if changed != set(PINS):
        raise ValueError("Serialized reporting accepted receipt pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


def serialization_report_tree(tree):
    """Describe the accepted serialization change in the derived receipt projection."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace(
                "Source/configuration changes are restricted to the owner adapter and the measured handoff helper.",
                "Source/configuration changes are restricted to the owner adapter and full ownership-inspection "
                "serialization; the accepted handoff helper and trainer/configuration bytes are retained.",
            )
    return tree


exec(compile(runtime_source(), __file__, "exec"), globals())
