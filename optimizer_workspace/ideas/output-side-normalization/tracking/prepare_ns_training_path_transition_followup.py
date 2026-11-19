# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a fresh read-only terminal waiter for the issued transition generation."""

import ast
import hashlib
from pathlib import Path

PARENT_PREPARER_SHA256 = "90219a26493dc4773ad60e559d84dd76fe221daccc50f56163ff287d3bd2a685"
VALUES = {
    "BINDING_SHA256": "18c9b21d16ff817059177e6746c3ae2246d39232396220c2b1deec805a33ca3d",
    "INTENT_SHA256": "42a901376f00df7b1672ab5d9dad610f669097ca0e48c16690ea4e48b2fb54e1",
    "ARCHIVE_SHA256": "5defe28731900a9c1e9a9d7fa08abd298e932475a0520c58345f0067718bb32b",
    "SCOPE": (
        "One distinct read-only waiter for the acknowledged mixed-exit-bracket controller "
        "and validator, checking exact original generations every900seconds. Invoke only "
        "their issued canonical independent terminal after full54-command18-decision "
        "coverage, actual owner/descendant absence and clean exact keeper restoration. "
        "Retain rejected cases and all consumed failed predecessors. No keeper transfer, "
        "new GPU work, numerical waiver, production, quality/efficiency or SwanLab authority."
    ),
}


def runtime_source() -> bytes:
    """Reuse the accepted waiter protocol, changing only bound execution identities."""
    path = Path(__file__).with_name("prepare_ns_training_path_exit_followup.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_PREPARER_SHA256:
        raise ValueError("Transition waiter pinned exit-drain preparer changed")
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_exit_source":
                    alias.name = "ns_training_path_transition_source"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_exit_drain", "ns_training_path_transition")
                .replace("ns_training_path_exit_", "ns_training_path_transition_")
                .replace("ns-training-path-exit-drain", "ns-training-path-exit-transition")
                .replace("OUTPUT_MARIN_NS_TRAINING_PATH_EXIT", "OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION")
                .replace("exit_drain_read_only_waiter", "exit_transition_read_only_waiter")
                .replace("Exit-drain waiter", "Transition waiter")
                .replace("Exit-drain terminal waiter", "Transition terminal waiter")
            )
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.Constant(value=VALUES[target.id])
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
