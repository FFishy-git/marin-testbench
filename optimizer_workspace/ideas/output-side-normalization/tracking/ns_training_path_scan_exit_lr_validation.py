# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain actual-LR numerical orchestration under the distinct scan-exit execution identity."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_scan_exit_lr_contract as scan_contract

PARENT_SHA256 = "56817963148654c7c964ddfb3b729346627270e7400e5c32398a120b6820ae58"


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_transition_lr_validation.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit LR pinned numerical orchestration changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    tree = ast.parse(namespace["runtime_source"]())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_transition_lr_contract":
                    alias.name = "ns_training_path_scan_exit_lr_contract"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = scan_contract.literal(node.value)
        if isinstance(node, ast.Attribute) and node.attr == "require_ns_training_path_transition_lr_gpu_owner":
            node.attr = "require_ns_training_path_scan_exit_lr_gpu_owner"
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
