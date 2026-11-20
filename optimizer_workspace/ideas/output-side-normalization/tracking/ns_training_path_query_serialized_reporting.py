# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project only accepted query-serialization receipts into the research journals."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_serialized_reporting as parent

PARENT_SOURCE_SHA256 = "d399387d91006cd0c0ed4242dcdfb664b3b29c26dde1edb4460c00bd5f064034"
PINS = {
    "TITLE": "NS Query-Serialized Ownership Diagnostic",
    "CPU_PATH": "run-preparation/ns-training-path-query-serialized-20261120-cpu/reviewed-20261120T0012/receipt.json",
    "CPU_SHA256": "dee0f1f2794237af02cfc0ac0f966a8a30a08f55a489d6ac7ef05291aebf3dc8",
    "SEAL_SHA256": "ca766ffc6298c7bb9284a7d514c44b875d0a80bfb53f9029f7794eae64388373",
    "BINDING_SHA256": "8e6760d8a0a679e057ad7434df2e969925f055c57912e271b42ff45fa1cfe24e",
    "INTENT_SHA256": "43e810371a8441b56f75c740ffc8ec126473acaaf360873647dd9d9aaffef3fe",
}


def runtime_source() -> bytes:
    """Retain the saved-evidence protocol and pin the separately accepted candidate."""
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Query reporting pinned whole-inspection projection changed")
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
                if alias.name == "ns_training_path_serialized_source":
                    alias.name = "ns_training_path_query_serialized_source"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_serialized", "ns_training_path_query_serialized")
                .replace("serialized_saved_", "query_serialized_saved_")
                .replace("Serialized reporting", "Query reporting")
                .replace("Serialized evidence withheld:", "Query evidence withheld:")
                .replace("Serialized saved status:", "Query-serialized saved status:")
                .replace("141 passing cases", "142 passing cases")
                .replace(
                    "and recheck1/54commands0/18decisions; ",
                    "and recheck1/54commands0/18decisions and whole-inspection0/54commands0/18decisions; ",
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
        "query_report_tree(source.QuerySerializedNamespace().visit(source.parent_source.SerializedNamespace().visit("
        "source.parent_source.parent_source.RecheckExitNamespace().visit("
        "source.parent_source.parent_source.parent_source.ScanExitNamespace().visit("
        "source.parent_source.parent_source.parent_source.parent_source.TransitionNamespace().visit("
        "source.parent_source.parent_source.parent_source.parent_source.parent_source.ExitNamespace().visit("
        "ast.parse(payload))))))))",
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
            node.body[0].value.keywords[0].value = ast.Constant(0)
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in PINS:
                    node.value = ast.Constant(PINS[target.id])
                    changed.add(target.id)
    if changed != set(PINS):
        raise ValueError("Query reporting accepted receipt pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


def query_report_tree(tree):
    """Describe the exact lock-boundary change without rewriting a historical projection."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace(
                "Source/configuration changes are restricted to the owner adapter and the measured handoff helper.",
                "The owner adapter and query/marker-scan lock boundary change; source authentication remains "
                "outside the unchanged 30-second lock. The handoff helper, trainer/configuration bytes "
                "and every ownership statement/order are retained.",
            )
    return tree


exec(compile(runtime_source(), __file__, "exec"), globals())
