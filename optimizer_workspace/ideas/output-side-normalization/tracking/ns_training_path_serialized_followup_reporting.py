# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate the serialized waiter's saved original-generation observations."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_recheck_exit_followup_reporting as parent

PARENT_SOURCE_SHA256 = "d5fa65f71879855f915588984c1de7606387acb9a9248a6fce85cda5c8dda983"
VALUES = {
    "TITLE": "NS Serialized Independent Terminal Waiter",
    "CPU": "run-preparation/ns-training-path-serialized-followup-cpu/20261119T2353/receipt.json",
    "CPU_SHA256": "632f9296c004585213712462828068aabde2013ed97a3d3555ecde9c62e726ff",
    "INTENT_SHA256": "7c4115103b9e07596480936a3da39aa9113aa8d179374cffbe9dce2de8b860ad",
    "RUNNER_SHA256": "1b284f406151344dbb4246617d7ef22d9917e450f2f4dd94a6e032bc4d4b5f3a",
}


def runtime_source() -> bytes:
    """Retain the saved-waiter predicate and pin the separately accepted source archive."""
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Serialized waiter pinned recheck projection changed")
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
        "ns_training_path_recheck_exit_reporting": "ns_training_path_serialized_reporting",
        "prepare_ns_training_path_recheck_exit_followup": "prepare_ns_training_path_serialized_followup",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                alias.name = modules.get(alias.name, alias.name)
        if isinstance(node, ast.ImportFrom):
            node.module = modules.get(node.module, node.module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_recheck_exit", "ns_training_path_serialized")
                .replace("recheck_exit_saved_followup", "serialized_saved_followup")
                .replace("Recheck-exit waiter", "Serialized waiter")
                .replace(
                    "CPU/source archive:24 passing cases,185 helpers;",
                    "CPU/source archive:24 passing cases,191 helpers;",
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
        raise ValueError("Serialized waiter accepted receipt pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
