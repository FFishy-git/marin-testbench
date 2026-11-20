# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive and issue the distinct streaming scanner through the accepted protocol."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_streaming_source as successor

PARENT_SOURCE_SHA256 = "59594e2b7ea37aab31ff08ec5f121702d3b8b1479e7dd843b5a9b5ef309ad4d4"
CPU_PATH_VALUE = "run-preparation/ns-training-path-streaming-20261120-cpu/20261120T0205/receipt.json"


def runtime_source() -> bytes:
    path = Path(__file__).with_name("prepare_ns_training_path_query_serialized.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Streaming diagnostic pinned query preparer changed")
    tree = ast.parse(payload)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "exec"
        )
    ]
    namespace = {"__file__": str(path)}
    exec(compile(tree, str(path), "exec"), namespace)
    inherited = ast.parse(namespace["runtime_source"]())
    helpers = next(
        ast.literal_eval(node.value)
        for node in inherited.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "LOCAL_HELPERS" for target in node.targets)
    )
    tree = successor.StreamingNamespace().visit(inherited)
    replacements = {
        "CPU_PATH": CPU_PATH_VALUE,
        "LOCAL_HELPERS": tuple(
            dict.fromkeys(
                (
                    *helpers,
                    "ns_training_path_streaming_predecessor.py",
                    "ns_training_path_streaming_source.py",
                    "prepare_ns_training_path_streaming.py",
                    "test_ns_training_path_streaming.py",
                    "owned_output_streaming_scan.py",
                    "test_owned_output_streaming_scan.py",
                    "owned_output_unmarked_churn.py",
                    "test_owned_output_unmarked_churn.py",
                    "test_owned_output_helper_churn.py",
                )
            )
        ),
        "TEST_COUNTS": {
            "test_ns_training_path_streaming_gpu": 45,
            "test_owned_output_handoff": 36,
            "test_owned_output_recheck_exit": 41,
            "test_ns_training_owner_serialization": 5,
            "test_owned_output_streaming_scan": 46,
            "test_owned_output_unmarked_churn": 6,
            "test_owned_output_helper_churn": 5,
        },
        "RECOVERY_TEST_COUNT": 17,
    }
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
                    changed.add(target.id)
    if changed != set(replacements):
        raise ValueError("Streaming source archive protocol changed")
    boundary = next(index for index, node in enumerate(tree.body) if isinstance(node, ast.FunctionDef))
    tree.body[boundary:boundary] = ast.parse("import ns_training_path_query_serialized_source as query_source\n").body
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "helper_bytes")
    merge = next(
        index
        for index, node in enumerate(helper.body)
        if isinstance(node, ast.AugAssign) and ast.unparse(node.value) == "source.derived_sources(parent)"
    )
    helper.body[merge:merge] = ast.parse("sources |= query_source.derived_sources(parent)").body
    helper.body[merge + 2 : merge + 2] = ast.parse("sources |= source.study_sources(parent)").body
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
