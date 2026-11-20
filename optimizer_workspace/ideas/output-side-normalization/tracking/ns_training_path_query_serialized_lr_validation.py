# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain all actual-LR numerical readers under a distinct query execution namespace."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_query_serialized_lr_contract as query_contract

DERIVATION_SHA256 = "4e56cd744e0e593b68afa6a2654e93144e0361e5c4bb747642c475298532608b"


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_lr_validation.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != DERIVATION_SHA256:
        raise ValueError("Query actual-LR pinned numerical orchestration changed")
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_lr_contract":
                    alias.name = "ns_training_path_query_serialized_lr_contract"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = query_contract.literal(node.value)
        if isinstance(node, ast.Attribute) and node.attr == "require_ns_training_path_lr_gpu_owner":
            node.attr = "require_ns_training_path_query_serialized_lr_gpu_owner"
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
