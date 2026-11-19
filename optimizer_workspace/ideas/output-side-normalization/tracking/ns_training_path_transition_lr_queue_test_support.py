# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain the queue's real CPU lifecycle fixtures in the transition namespace."""

import ast
from pathlib import Path

import ns_training_path_transition_lr_queue as derivation


def runtime_source() -> bytes:
    tree = derivation.transform(Path(__file__).with_name("ns_training_path_lr_queue_test_support.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace(
                "ns_training_path_transition_lr_parent", "ns_training_path_transition_lr_queue_parent"
            )
        if isinstance(node, ast.Subscript) and ast.unparse(node) == "queue.NEW_HELPERS[-1]":
            node.slice = ast.Constant(value=3)
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
