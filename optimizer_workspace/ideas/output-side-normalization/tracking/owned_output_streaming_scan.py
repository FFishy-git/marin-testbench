# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Interleave process and environment reads without changing complete-scan acceptance."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "7c653b0a08e3159599ef00ba4767ef3291b7d581744581d000ca14c3a6ad4e24"
SCOPE = (
    "Ownership-observation ordering only. Enumerate every original process candidate and "
    "read its identity immediately before its environment; retain every complete-scan body "
    "statement, final marked-process recheck, three attempts and identity-bound exit proof. "
    "No missing-file waiver, numerical-policy change, GPU or production authority."
)
ITERATOR = '''def scan_processes():
    """Yield the complete process namespace one identity observation at a time."""
    for path in PROC_ROOT.iterdir():
        if path.name.isdigit():
            process = observed(int(path.name))
            if process is not None and process["status"] != "Z":
                yield process["pid"], process
'''


def derived_source(parent: Path) -> bytes:
    """Retain the issued scanner except for the observation iterator's materialization."""
    payload = parent.read_bytes()
    if parent.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Streaming scan pinned query handoff changed")
    tree = ast.parse(payload)
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "live_processes")
    iterator = ast.parse(ITERATOR).body[0]
    before = original.body[1]
    after = next(node for node in iterator.body if isinstance(node, ast.For))
    if not isinstance(before, ast.For) or not isinstance(after, ast.For):
        raise ValueError("Streaming scan original namespace iteration changed")
    if ast.dump(before.iter) != ast.dump(after.iter) or ast.dump(before.body[0].test) != ast.dump(after.body[0].test):
        raise ValueError("Streaming scan process namespace or identity filter changed")
    prior = before.body[0].body
    current = after.body[0].body
    if ast.dump(prior[0]) != ast.dump(current[0]) or ast.dump(prior[1].test) != ast.dump(current[1].test):
        raise ValueError("Streaming scan original identity observation or terminal filter changed")
    scanner = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "marker_processes")
    changed = 0
    for node in ast.walk(scanner):
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "live_processes().items()":
            node.iter = ast.parse("scan_processes()", mode="eval").body
            changed += 1
    if changed != 1:
        raise ValueError("Streaming scan original complete ownership loop changed")
    tree.body.insert(tree.body.index(scanner), iterator)
    return (
        "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
        + ast.unparse(ast.fix_missing_locations(tree))
        + "\n"
    ).encode()
