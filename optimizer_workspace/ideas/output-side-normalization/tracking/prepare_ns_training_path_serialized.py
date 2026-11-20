# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive a distinct serialization diagnostic using the accepted preparation protocol."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_serialized_source as successor

PARENT_SOURCE_SHA256 = "99c024c5173921b1f7f337da82fa90e49ec0069adafc792aa68ce027fec977eb"
CPU_PATH_VALUE = "run-preparation/ns-training-path-serialized-20261119-cpu/reviewed-20261119T2340/receipt.json"


def runtime_source() -> bytes:
    path = Path(__file__).with_name("prepare_ns_training_path_recheck_exit.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Serialized diagnostic pinned recheck preparer changed")
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
    tree = successor.SerializedNamespace().visit(ast.parse(namespace["runtime_source"]()))
    replacements = {
        "CPU_PATH": CPU_PATH_VALUE,
        "LOCAL_HELPERS": (
            "ns_training_path_serialized_predecessor.py",
            "ns_training_path_serialized_source.py",
            "prepare_ns_training_path_serialized.py",
            "prepare_ns_training_path_recheck_exit.py",
            "prepare_ns_training_path_scan_exit.py",
            "prepare_ns_training_path_transition.py",
            "prepare_ns_training_path_exit.py",
            "test_ns_training_path_serialized.py",
            "ns_training_owner_serialization.py",
            "test_ns_training_owner_serialization.py",
            "owned_output_helper_churn.py",
            "owned_output_attribution.py",
            "owned_output_exit_drain.py",
            "owned_output_exit_transition.py",
            "owned_output_missing_environment.py",
            "owned_output_recheck_exit.py",
            "test_owned_output_exit_drain.py",
            "test_owned_output_exit_transition.py",
            "test_owned_output_missing_environment.py",
            "test_owned_output_recheck_exit.py",
        ),
        "TEST_COUNTS": {
            "test_ns_training_path_serialized_gpu": 45,
            "test_owned_output_handoff": 36,
            "test_owned_output_recheck_exit": 41,
            "test_ns_training_owner_serialization": 5,
        },
        "RECOVERY_TEST_COUNT": 14,
    }
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
                    changed.add(target.id)
    if changed != set(replacements):
        raise ValueError("Serialized diagnostic preparation contract differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
