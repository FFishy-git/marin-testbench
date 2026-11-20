# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a separately accepted read-only waiter for serialized ownership."""

import ast
import hashlib
from pathlib import Path

PARENT_SOURCE_SHA256 = "65f93c3b4a9801055e0fab0a71203fc71cb7a829228beb51f4aa7c4984896da6"
VALUES = {
    "BINDING_SHA256": "5789abe0b6c47fd910917cba35e00b2b2d519e66c867dab2214418327bbddc68",
    "INTENT_SHA256": "e782a7fcdc67c137c64c4d65799eb45b56c4800f969d0f1cb0640ed52295bef1",
    "ARCHIVE_SHA256": "d7d3d70714d6919dc96a0cf112bea2a320bfdc2b74d80cadc0470d6108144f8f",
    "SCOPE": (
        "One distinct read-only waiter for the acknowledged serialized-ownership controller "
        "and validator, checking exact original generations every900seconds. Invoke only "
        "their issued canonical independent terminal after full54-command18-decision "
        "coverage, actual owner/descendant absence and clean exact keeper restoration. "
        "Preserve rejected cases, all failed owners/waiters and accepted unissued queues. "
        "No keeper transfer, new GPU work, numerical waiver, production, optimizer quality "
        "or efficiency claim or SwanLab authority."
    ),
}


def runtime_source() -> bytes:
    """Retain the pinned waiter protocol with separate execution and source identities."""
    path = Path(__file__).with_name("prepare_ns_training_path_recheck_exit_followup.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Serialized waiter pinned recheck preparer changed")
    tree = ast.parse(payload)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "exec"
        )
    ]
    namespace = {"__file__": str(path)}
    exec(compile(tree, str(path), "exec"), namespace)
    tree = ast.parse(namespace["runtime_source"]())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_recheck_exit_source":
                    alias.name = "ns_training_path_serialized_source"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_recheck_exit", "ns_training_path_serialized")
                .replace("ns-training-path-recheck-exit", "ns-training-path-serialized")
                .replace("OUTPUT_MARIN_NS_TRAINING_PATH_RECHECK_EXIT", "OUTPUT_MARIN_NS_TRAINING_PATH_SERIALIZED")
                .replace("recheck_exit_read_only_waiter", "serialized_read_only_waiter")
                .replace("Recheck-exit waiter", "Serialized waiter")
                .replace("Recheck-exit terminal waiter", "Serialized terminal waiter")
            )
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.Constant(VALUES[target.id])
                    changed.add(target.id)
    if changed != set(VALUES):
        raise ValueError("Serialized waiter execution identity pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
