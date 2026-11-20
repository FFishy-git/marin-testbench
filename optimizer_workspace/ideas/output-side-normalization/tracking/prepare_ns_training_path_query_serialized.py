# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive and issue a distinct query-serialization diagnostic without changing numeric gates."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_query_serialized_source as successor

PARENT_SOURCE_SHA256 = "4a1e21aab5212a238cf709e89a88e23df6b66cc5c8a9b7d5ed8721aa7b7be2de"
CPU_PATH_VALUE = "run-preparation/ns-training-path-query-serialized-20261120-cpu/reviewed-20261120T0012/receipt.json"


def runtime_source() -> bytes:
    path = Path(__file__).with_name("prepare_ns_training_path_serialized.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Query diagnostic pinned whole-inspection preparer changed")
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
    inherited = ast.parse(namespace["runtime_source"]())
    parent_helpers = next(
        ast.literal_eval(node.value)
        for node in inherited.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "LOCAL_HELPERS" for target in node.targets)
    )
    tree = successor.QuerySerializedNamespace().visit(inherited)
    replacements = {
        "CPU_PATH": CPU_PATH_VALUE,
        "LOCAL_HELPERS": tuple(
            dict.fromkeys(
                (
                    *parent_helpers,
                    "prepare_ns_training_path_serialized.py",
                    "ns_training_path_query_serialized_predecessor.py",
                    "ns_training_path_query_serialized_source.py",
                    "prepare_ns_training_path_query_serialized.py",
                    "test_ns_training_path_query_serialized.py",
                )
            )
        ),
        "TEST_COUNTS": {
            "test_ns_training_path_query_serialized_gpu": 45,
            "test_owned_output_handoff": 36,
            "test_owned_output_recheck_exit": 41,
            "test_ns_training_owner_serialization": 5,
        },
        "RECOVERY_TEST_COUNT": 15,
    }
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
                    changed.add(target.id)
    if changed != set(replacements):
        raise ValueError("Query serialization archive protocol differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
