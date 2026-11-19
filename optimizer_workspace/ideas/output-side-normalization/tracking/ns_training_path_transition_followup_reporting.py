# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render saved observations of the distinct transition terminal waiter."""

import ast
import hashlib
from pathlib import Path

PARENT_REPORT_SHA256 = "01d66fe771d00176f32090b9493cc17d8b9185d664a1115d9d58d85d490ba9c9"
VALUES = {
    "TITLE": "NS Transition Independent Terminal Waiter",
    "CPU": "run-preparation/ns-training-path-exit-transition-followup-cpu/20261119T1718/receipt.json",
    "CPU_SHA256": "6a0994915727bf06c606ff0a2bfcece8e4070045ac1f0dcb5f2e79d6b3041906",
    "INTENT_SHA256": "1641a7cfd73b11d71a91220c7aab6b9aa6e9601d3804237ce4d89abaef0aae0d",
    "RUNNER_SHA256": "9beaf16b3df4347bbbcca05339689ee0baef25afb81e2dd1b8ed9fff67bdacb8",
}
MODULES = {
    "ns_training_path_exit_reporting": "ns_training_path_transition_reporting",
    "prepare_ns_training_path_exit_followup": "prepare_ns_training_path_transition_followup",
}


def runtime_source() -> bytes:
    """Retain the pinned saved-receipt predicate with this waiter's exact generation."""
    path = Path(__file__).with_name("ns_training_path_exit_followup_reporting.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_REPORT_SHA256:
        raise ValueError("Transition waiter pinned reporter changed")
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                alias.name = MODULES.get(alias.name, alias.name)
        if isinstance(node, ast.ImportFrom):
            node.module = MODULES.get(node.module, node.module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_exit_drain", "ns_training_path_transition")
                .replace("ns_training_path_exit_followup", "ns_training_path_transition_followup")
                .replace("exit_drain_saved_followup", "exit_transition_saved_followup")
                .replace("Exit-drain waiter", "Transition waiter")
                .replace(
                    "CPU/source archive:24 passing cases,169 helpers;",
                    "CPU/source archive:24 passing cases,175 helpers;",
                )
            )
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.Constant(value=VALUES[target.id])
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
