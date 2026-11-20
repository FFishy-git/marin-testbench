# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate the recheck-exit waiter's saved generation-bound observations."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_scan_exit_followup_reporting as parent

PARENT_SOURCE_SHA256 = "fa4cf860a7f302c4df1a04030acd9c50e057ac47b9f047ce595b3019035dcf58"
VALUES = {
    "TITLE": "NS Recheck-Exit Independent Terminal Waiter",
    "CPU": "run-preparation/ns-training-path-recheck-exit-followup-cpu/20261119T2253/receipt.json",
    "CPU_SHA256": "33528ba3d167f4d59b34d3586cc135fde0adc07e404bd369f61947eac2784ebf",
    "INTENT_SHA256": "84ae50084083fa775b37023824016336473d27e1b3c66b6a42938d808a81e6ac",
    "RUNNER_SHA256": "063d62973d32efe745776f48407b06266107514ba95b1ba0ac21172082a6de88",
}


def runtime_source() -> bytes:
    """Retain the pinned saved-waiter predicate with the distinct accepted identities."""
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Recheck-exit pinned scan-exit waiter projection changed")
    parent_tree = ast.parse(payload)
    parent_tree.body = [
        node
        for node in parent_tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "exec"
        )
    ]
    namespace = {"__file__": str(path)}
    exec(compile(parent_tree, str(path), "exec"), namespace)
    tree = ast.parse(namespace["runtime_source"]())
    modules = {
        "ns_training_path_scan_exit_reporting": "ns_training_path_recheck_exit_reporting",
        "prepare_ns_training_path_scan_exit_followup": "prepare_ns_training_path_recheck_exit_followup",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                alias.name = modules.get(alias.name, alias.name)
        if isinstance(node, ast.ImportFrom):
            node.module = modules.get(node.module, node.module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_scan_exit", "ns_training_path_recheck_exit")
                .replace("scan_exit_saved_followup", "recheck_exit_saved_followup")
                .replace("Scan-exit waiter", "Recheck-exit waiter")
                .replace(
                    "CPU/source archive:24 passing cases,180 helpers;",
                    "CPU/source archive:24 passing cases,185 helpers;",
                )
            )
    changed = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.Constant(VALUES[target.id])
                    changed.add(target.id)
    if changed != set(VALUES):
        raise ValueError("Recheck-exit waiter accepted receipt pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
