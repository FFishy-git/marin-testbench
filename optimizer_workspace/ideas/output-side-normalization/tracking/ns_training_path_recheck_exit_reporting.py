# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report the recheck-exit successor through its independently accepted receipts."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_scan_exit_reporting as parent

PARENT_SOURCE_SHA256 = "1c0e5da291eb04847bd8a4dcff7073544b3dc9db36cb261a6d40335b50fa30c9"
TITLE_VALUE = "NS Whole-Process Recheck-Exit Diagnostic"
CPU_PATH_VALUE = "run-preparation/ns-training-path-recheck-exit-20261119-cpu/closure-reviewed-20261119T2241/receipt.json"
PINS = {
    "TITLE": TITLE_VALUE,
    "CPU_PATH": CPU_PATH_VALUE,
    "CPU_SHA256": "5d085c3b8df3750ae7a4b488cc87cdf6b2ec6a4e275767bfc441e4858bce007f",
    "SEAL_SHA256": "10c189fc543b9d9e1ec4d8d28bf590700948ed108dbd02e8bfd1174fe66193b6",
    "BINDING_SHA256": "ae7757c3010f88724a036d1b9f0d8b5c11b3abb53b97ca2300090f1973d1456f",
    "INTENT_SHA256": "9502a4d1d2ce65f0ff7e23f1c9bccf62572cd5430b100d76150cad46ceebb607",
}


def runtime_source() -> bytes:
    """Retain the accepted receipt projection and bind the distinct successor namespace."""
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Recheck-exit pinned scan-exit reporting changed")
    tree = ast.parse(parent.runtime_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_scan_exit_source":
                    alias.name = "ns_training_path_recheck_exit_source"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_scan_exit", "ns_training_path_recheck_exit")
                .replace("scan_exit_saved_", "recheck_exit_saved_")
                .replace("Scan-exit reporting", "Recheck-exit reporting")
                .replace("Scan-exit evidence withheld:", "Recheck-exit evidence withheld:")
                .replace("Scan-exit saved status:", "Recheck-exit saved status:")
                .replace("129 passing cases", "136 passing cases")
                .replace(
                    "transition23/54commands7/18decisions; ",
                    "transition23/54commands7/18decisions and scan-exit8/54commands2/18decisions; ",
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
        "source.RecheckExitNamespace().visit(source.parent_source.ScanExitNamespace().visit("
        "source.parent_source.parent_source.TransitionNamespace().visit("
        "source.parent_source.parent_source.parent_source.ExitNamespace().visit(ast.parse(payload)))))",
        mode="eval",
    ).body
    changes = 0
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.BoolOp):
            continue
        comparison = node.test.values[-1]
        if not isinstance(comparison, ast.Compare) or not comparison.comparators:
            continue
        value = comparison.comparators[0]
        if isinstance(value, ast.Constant) and value.value in (44, 14):
            node.body[0].value.keywords[0].value = ast.Constant(8 if value.value == 44 else 2)
            changes += 1
    pins = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in PINS:
                    node.value = ast.Constant(PINS[target.id])
                    pins.add(target.id)
    if changes != 2 or pins != set(PINS):
        raise ValueError("Recheck-exit reporting retained coverage or receipt pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
