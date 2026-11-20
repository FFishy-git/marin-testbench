# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate the query-serialized waiter's saved original-generation observations."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_serialized_followup_reporting as parent

PARENT_SOURCE_SHA256 = "5f18af223aa5dceb1c870c9c0e88cb190f3fbba0ccc0d66bb91e4fbfaebc7f0a"
VALUES = {
    "TITLE": "NS Query-Serialized Independent Terminal Waiter",
    "CPU": "run-preparation/ns-training-path-query-serialized-followup-cpu/20261120T0026/receipt.json",
    "CPU_SHA256": "65e5133be24ea2d2ab1fbb4155c7c25d1c34d9ddc06ce08020694bde6d362e4d",
    "INTENT_SHA256": "3679f0cbca16b619995e76b70950951348db6ff4bb9592f435ccbc329f391666",
    "RUNNER_SHA256": "58365100058d55822eedcf35439b7ed9e4d6d1ccb51f90f3e63e9c01ef73ea7f",
}


def runtime_source() -> bytes:
    """Retain the saved-waiter predicate with separately accepted receipt identities."""
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Query waiter pinned whole-inspection projection changed")
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
    modules = {
        "ns_training_path_serialized_reporting": "ns_training_path_query_serialized_reporting",
        "prepare_ns_training_path_serialized_followup": "prepare_ns_training_path_query_serialized_followup",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                alias.name = modules.get(alias.name, alias.name)
        if isinstance(node, ast.ImportFrom):
            node.module = modules.get(node.module, node.module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_serialized", "ns_training_path_query_serialized")
                .replace("serialized_saved_followup", "query_serialized_saved_followup")
                .replace("Serialized waiter", "Query-serialized waiter")
                .replace(
                    "CPU/source archive:24 passing cases,191 helpers;",
                    "CPU/source archive:24 passing cases,195 helpers;",
                )
            )
    changed = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.Constant(VALUES[target.id])
                    changed.add(target.id)
    if changed != set(VALUES):
        raise ValueError("Query waiter accepted receipt pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
