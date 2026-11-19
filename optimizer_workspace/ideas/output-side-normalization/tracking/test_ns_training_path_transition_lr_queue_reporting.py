# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain queue waiting, failure and numerical rejection summaries in both journals."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_transition_lr_queue as derivation

PARENT_SHA256 = "a7c0ffa2d02e25ad5df0ad9737d8483a35e7ce68859ba72d38b6258747613070"


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_ns_training_path_lr_queue_reporting.py")
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != PARENT_SHA256:
        raise ValueError("Transition-LR queue reporting tests parent changed")
    tree = derivation.Namespace().visit(ast.parse(raw))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_lr_queue_reporting":
                    alias.name = "ns_training_path_transition_lr_queue_reporting"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace(
                    "ns_training_path_lr_queue_reporting", "ns_training_path_transition_lr_queue_reporting"
                )
                .replace("NS Actual-LR Training-Path Queue", "NS Transition Actual-LR Queue")
                .replace("waiting_for_original_training_path_owners", "waiting_for_transition_owners")
            )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())
