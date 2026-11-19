# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observe an identity-bound live/terminal bracket until whole-process exit."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "981deaf0ec73f43c69e64c0409212100afafb0929e6c0cb11ccf8892edd17db6"


def runtime_source() -> bytes:
    """Change only mixed-bracket classification and this successor's helper identity."""
    path = Path(__file__).with_name("owned_output_exit_drain.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Exit-transition pinned exit-drain parent changed")
    tree = ast.parse(payload)
    old = ast.parse('any(sample["status"] in ("Z", "X", "x") for sample in snapshots)').body[0].value
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("owned_output_exit_drain.py", "owned_output_exit_transition.py")
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name != "exit_sample":
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
                if len(node.test.values) == 3 and ast.dump(node.test.values[0]) == ast.dump(old):
                    # This only permits another bounded observation; drain_exit still
                    # requires directory absence or an identity-bound readable pidfd.
                    node.test.values.pop(0)
                    changes += 1
    if changes != 1:
        raise ValueError("Exit-transition parent bracket boundary differs")
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name != "main":
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Tuple) and ast.dump(node) == ast.dump(
                ast.parse("(Path(__file__), Path(attribution.__file__))", mode="eval").body
            ):
                node.elts.append(ast.parse('Path(__file__).with_name("owned_output_exit_drain.py")', mode="eval").body)
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
