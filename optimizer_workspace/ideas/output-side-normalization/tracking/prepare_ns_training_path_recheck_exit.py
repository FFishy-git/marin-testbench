# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive and issue the distinct original-training-path recheck-exit correction."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_recheck_exit_source as successor

PARENT_SOURCE_SHA256 = "2151a8f90df287e300970b922bf7179f4e90fda7800f605033d9e7158375d233"
CPU_PATH_VALUE = "run-preparation/ns-training-path-recheck-exit-20261119-cpu/closure-reviewed-20261119T2241/receipt.json"


def runtime_source() -> bytes:
    path = Path(__file__).with_name("prepare_ns_training_path_scan_exit.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Recheck-exit accepted scan-exit preparer changed")
    namespace = {"__file__": str(path)}
    parent_tree = ast.parse(payload)
    parent_tree.body = [
        node
        for node in parent_tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "exec"
        )
    ]
    exec(compile(parent_tree, str(path), "exec"), namespace)
    tree = successor.RecheckExitNamespace().visit(ast.parse(namespace["runtime_source"]()))
    replacements = {
        "CPU_PATH": CPU_PATH_VALUE,
        "LOCAL_HELPERS": (
            "ns_training_path_recheck_exit_predecessor.py",
            "ns_training_path_recheck_exit_source.py",
            "prepare_ns_training_path_recheck_exit.py",
            "prepare_ns_training_path_scan_exit.py",
            "prepare_ns_training_path_transition.py",
            "prepare_ns_training_path_exit.py",
            "test_ns_training_path_recheck_exit.py",
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
            "test_ns_training_path_recheck_exit_gpu": 45,
            "test_owned_output_handoff": 36,
            "test_owned_output_recheck_exit": 41,
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
        raise ValueError("Recheck-exit orchestration receipt constants differ")
    helpers = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "helper_bytes")
    start = next(
        index
        for index, node in enumerate(helpers.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "measured"
    )
    end = next(
        index
        for index, node in enumerate(helpers.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "sources[test_path.name]"
    )
    helpers.body[start : end + 1] = ast.parse(
        """sources["test_owned_output_handoff.py"] = source.handoff_test_bytes(parent)
if hashlib.sha256(sources["test_owned_output_handoff.py"]).hexdigest() != source.HANDOFF_TEST_SHA256:
    raise ValueError("Recheck-exit derived handoff test evidence changed")
"""
    ).body
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
