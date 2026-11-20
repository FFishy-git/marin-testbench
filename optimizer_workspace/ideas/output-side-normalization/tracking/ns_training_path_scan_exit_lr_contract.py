# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reserve a distinct scan-exit LR execution namespace with unchanged numerical settings."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "1d51d28530607690c3716885f18df79d9bdb0bd0608e405d8f6df510ccf420e2"
NAMESPACE = "ns-training-path-scan-exit-actual-lr-20261119"


def literal(value: str) -> str:
    return (
        value.replace("ns-training-path-transition-actual-lr-20261119", NAMESPACE)
        .replace("ns_training_path_transition_lr_20261119", "ns_training_path_scan_exit_lr_20261119")
        .replace("OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION_LR_", "OUTPUT_MARIN_NS_TRAINING_PATH_SCAN_EXIT_LR_")
        .replace("ns_training_path_transition_actual_lr_", "ns_training_path_scan_exit_actual_lr_")
    )


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_transition_lr_contract.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit LR pinned namespace contract changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    tree = ast.parse(namespace["runtime_source"]())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = literal(node.value)
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
