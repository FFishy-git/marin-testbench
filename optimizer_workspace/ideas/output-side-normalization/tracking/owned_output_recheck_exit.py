# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require whole-process exit proof for marked-process recheck disappearance."""

import ast
import hashlib
from pathlib import Path

import owned_output_missing_environment as parent

PARENT_SOURCE_SHA256 = "06c18d6387607339ceaa320b8ae4a156c172f0054b93fb32c5e2b67b1feb7dde"
SCOPE = (
    "Ownership-only marked-process recheck correction. Retain the three complete-scan "
    "attempts and existing exit deadline. Require directory absence or identity-bound "
    "whole-process exit before a fresh complete scan. Missing files, PID reuse, surviving "
    "threads and live ownership drift remain refusals. No numerical or launch acceptance."
)


def runtime_source() -> bytes:
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Recheck-exit accepted missing-environment parent changed")
    tree = ast.parse(parent.runtime_source())
    changed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MARKER_SCAN" for target in node.targets
        ):
            scan = ast.parse(ast.literal_eval(node.value))
            for branch in ast.walk(scan):
                if not isinstance(branch, ast.If):
                    continue
                condition = ast.unparse(branch.test)
                if condition == "current is None":
                    branch.body[:0] = ast.parse(
                        """exit_drain_handler.missing_environment_exit(PROC_ROOT, process)
drained.add(pid)
"""
                    ).body
                    changed.add("disappeared")
                elif condition == "current['status'] in EXIT_STATES":
                    branch.body[-1:-1] = ast.parse(
                        """exit_drain_handler.drain_exit(PROC_ROOT, process)
drained.add(pid)
"""
                    ).body
                    changed.add("terminal")
            node.value = ast.Constant(ast.unparse(ast.fix_missing_locations(scan)) + "\n")
    if changed != {"disappeared", "terminal"}:
        raise ValueError("Recheck-exit accepted scan branches differ")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("owned_output_missing_environment.py", "owned_output_recheck_exit.py")
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    for node in ast.walk(main):
        if isinstance(node, ast.Tuple) and any(
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "Path"
            and any(isinstance(argument, ast.Name) and argument.id == "__file__" for argument in item.args)
            for item in node.elts
        ):
            node.elts.append(
                ast.parse('Path(__file__).with_name("owned_output_missing_environment.py")', mode="eval").body
            )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
