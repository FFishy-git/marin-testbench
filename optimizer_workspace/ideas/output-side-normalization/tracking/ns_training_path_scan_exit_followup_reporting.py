# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate saved observations of the distinct scan-exit terminal waiter."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "647b4a05997c18ad88d444a7ea18df3572ee646435dd9387b8518eb1c72b865c"
VALUES = {
    "TITLE": "NS Scan-Exit Independent Terminal Waiter",
    "CPU": "run-preparation/ns-training-path-scan-exit-followup-cpu/20261119T2110/receipt.json",
    "CPU_SHA256": "86d6984b2166c079bced8d3b69709c2f07ac37b92dcb9f3e89360f7735f70f0c",
    "INTENT_SHA256": "9ddc9c73d53e45779208e90259ab6d391f38484372e75f29c077c148be1ef1c7",
    "RUNNER_SHA256": "b8a69286240ba12190f73f2120094b2e11f6bdb1b5e37ef34a5f12d887548840",
}


def runtime_source() -> bytes:
    """Retain the pinned waiter projection with distinct source and acknowledgment pins."""
    path = Path(__file__).with_name("ns_training_path_transition_followup_reporting.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit waiter pinned transition projection changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    tree = ast.parse(namespace["runtime_source"]())
    modules = {
        "ns_training_path_transition_reporting": "ns_training_path_scan_exit_reporting",
        "prepare_ns_training_path_transition_followup": "prepare_ns_training_path_scan_exit_followup",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                alias.name = modules.get(alias.name, alias.name)
        if isinstance(node, ast.ImportFrom):
            node.module = modules.get(node.module, node.module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_transition", "ns_training_path_scan_exit")
                .replace("exit_transition_saved_followup", "scan_exit_saved_followup")
                .replace("Transition waiter", "Scan-exit waiter")
                .replace(
                    "CPU/source archive:24 passing cases,175 helpers;",
                    "CPU/source archive:24 passing cases,180 helpers;",
                )
            )
    changed = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.Constant(value=VALUES[target.id])
                    changed.add(target.id)
    if changed != set(VALUES):
        raise ValueError("Scan-exit waiter projection pin replacement differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
