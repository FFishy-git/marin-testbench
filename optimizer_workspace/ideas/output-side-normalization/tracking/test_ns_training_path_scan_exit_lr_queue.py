# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise isolated terminal observation and the one-shot scan-exit actual-LR queue."""

import ast
from pathlib import Path

import ns_training_path_scan_exit_lr_queue as derivation


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_ns_training_path_transition_lr_queue.py")
    tree = derivation.transform(path, "inherited_tests")
    parent = ast.parse(path.read_bytes())
    for node in parent.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXCLUDED" for target in node.targets
        ):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "exec":
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "inherited_tests":
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            tree.body.append(derivation.Namespace().visit(node))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if node.value == 198:
                node.value = 214
            elif isinstance(node.value, str):
                node.value = node.value.replace(
                    "OUTPUT_MARIN_NS_TRANSITION_LR_QUEUE_OBSERVER", "OUTPUT_MARIN_NS_SCAN_EXIT_LR_QUEUE_OBSERVER"
                )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())
