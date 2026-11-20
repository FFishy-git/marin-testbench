# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a separately accepted read-only waiter for query-serialized ownership."""

import ast
import hashlib
from pathlib import Path

PARENT_SOURCE_SHA256 = "d22a0f41855fdd562d6893a1581b1832d60682c9bf980ad2f078cc4791430cf0"
VALUES = {
    "BINDING_SHA256": "8e6760d8a0a679e057ad7434df2e969925f055c57912e271b42ff45fa1cfe24e",
    "INTENT_SHA256": "43e810371a8441b56f75c740ffc8ec126473acaaf360873647dd9d9aaffef3fe",
    "ARCHIVE_SHA256": "a64d07b32e383e062ad4985f89a337f3d14b8a2e12bcd1d099cc9d346b7d1c82",
    "SCOPE": (
        "One distinct read-only waiter for the acknowledged query-serialized ownership "
        "controller and validator, checking exact original generations every900seconds. "
        "Invoke only their issued canonical independent terminal after full54-command "
        "18-decision coverage, actual owner/descendant absence and clean exact keeper "
        "restoration. Preserve rejected cases, all failed owners/waiters and accepted "
        "unissued queues. No keeper transfer, new GPU work, numerical waiver, production, "
        "optimizer quality/efficiency or SwanLab authority."
    ),
}


def runtime_source() -> bytes:
    """Retain the pinned waiter protocol with separate execution and source identities."""
    path = Path(__file__).with_name("prepare_ns_training_path_serialized_followup.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Query-serialized waiter pinned whole-inspection preparer changed")
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
                if alias.name == "ns_training_path_serialized_source":
                    alias.name = "ns_training_path_query_serialized_source"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_serialized", "ns_training_path_query_serialized")
                .replace("ns-training-path-serialized-20261119", "ns-training-path-query-serialized-20261120")
                .replace(
                    "ns-training-path-serialized-terminal-followup-20261119",
                    "ns-training-path-query-serialized-terminal-followup-20261120",
                )
                .replace("ns-training-path-serialized", "ns-training-path-query-serialized")
                .replace("OUTPUT_MARIN_NS_TRAINING_PATH_SERIALIZED", "OUTPUT_MARIN_NS_TRAINING_PATH_QUERY_SERIALIZED")
                .replace("serialized_read_only_waiter", "query_serialized_read_only_waiter")
                .replace("Serialized waiter", "Query-serialized waiter")
                .replace("Serialized terminal waiter", "Query-serialized terminal waiter")
            )
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.Constant(VALUES[target.id])
                    changed.add(target.id)
    if changed != set(VALUES):
        raise ValueError("Query-serialized waiter execution identity pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
