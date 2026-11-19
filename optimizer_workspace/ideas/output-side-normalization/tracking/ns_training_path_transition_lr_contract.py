# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reserve a distinct actual-LR namespace without changing the accepted numerical protocol."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "0e91e0c921032b3799ca8f6b9f0066e31487dfde237c02252706a004818b5fe0"
NAMESPACE = "ns-training-path-transition-actual-lr-20261119"


def literal(value: str) -> str:
    return (
        value.replace("ns-training-path-actual-lr-binding-20261119.json", f"{NAMESPACE}-binding.json")
        .replace("ns-training-path-actual-lr-state-20261119.json", f"{NAMESPACE}-state.json")
        .replace("validation-ns-training-path-actual-lr-gpu-20261119.json", f"validation-{NAMESPACE}-gpu.json")
        .replace("controllers/ns-training-path-actual-lr-20261119", f"controllers/{NAMESPACE}")
        .replace("validation/ns_training_path_actual_lr_20261119", "validation/ns_training_path_transition_lr_20261119")
        .replace("OUTPUT_MARIN_NS_TRAINING_PATH_LR_", "OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION_LR_")
        .replace("ns_training_path_actual_lr_", "ns_training_path_transition_actual_lr_")
    )


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_lr_contract.py")
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != PARENT_SHA256:
        raise ValueError("Transition-LR pinned numerical contract changed")
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = literal(node.value)
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
