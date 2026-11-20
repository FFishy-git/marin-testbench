# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observe the separate scan-exit terminal reader with acknowledged, reaped CPU children."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "2c14caa054f7c3b7478af3e037522c72a3f7d494c0998d93bafbde6e9274df1d"
OBSERVER_MARKER_VALUE = "OUTPUT_MARIN_NS_SCAN_EXIT_LR_QUEUE_OBSERVER"


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_transition_lr_queue_parent.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit LR queue pinned external observer changed")
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_transition_lr_parent":
                    alias.name = "ns_training_path_scan_exit_lr_parent"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("ns_training_path_transition_lr", "ns_training_path_scan_exit_lr").replace(
                "Transition-LR", "Scan-exit LR"
            )
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "OBSERVER_MARKER":
                    node.value = ast.Constant(value=OBSERVER_MARKER_VALUE)
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
