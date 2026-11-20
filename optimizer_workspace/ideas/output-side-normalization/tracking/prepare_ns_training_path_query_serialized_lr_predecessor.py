# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a separate query-serialized actual-LR terminal reader without GPU authority."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "49472305ac5b0f9a30f0e29ba2c40aa7f2f2419363f955463aadeb317f5cb726"
VALUES = {
    "STATUS": "query_serialized_actual_lr_predecessor_cpu_archive_verified_execution_archives_separate",
    "PENDING_STATUS": "query_serialized_actual_lr_predecessor_staged_acceptance_pending",
    "TEST_MODULE": "test_ns_training_path_query_serialized_lr_predecessor",
    "TEST_COUNT": 39,
    "NEW_HELPERS": (
        "ns_training_path_query_serialized_lr_predecessor.py",
        "prepare_ns_training_path_query_serialized_lr_predecessor.py",
        "test_ns_training_path_query_serialized_lr_predecessor.py",
    ),
    "PARENTS": {
        "ns_training_path_exit_lr_predecessor.py": "e0724c9b0e3021b74d0824083fd2a125452f240bd858a349c3a4e4245c3d9aad",
        "test_ns_training_path_exit_lr_predecessor.py": (
            "0a59d82243719252e9dcb31cb1d4261f2afcacb08da14ef7d72de937a0410bea"
        ),
        "prepare_ns_training_path_exit_lr_predecessor.py": (
            "624500764687f1a08f09f6a704907a1f1e06d3b9753a3069964f2e27a9c613bc"
        ),
        "ns_training_path_transition_lr_predecessor.py": (
            "f38cf4e4c991ed960b20f9d6849e248100bc96cfb5672362c080dc39078bc5f9"
        ),
        "prepare_ns_training_path_transition_lr_predecessor.py": PARENT_SHA256,
        "test_ns_training_path_transition_lr_predecessor.py": (
            "b82a51ac493ea6313c05e5c4d5bb62412df5f261f1d07c3c16b6240ade020ef8"
        ),
    },
}


def runtime_source() -> bytes:
    """Keep separate flat source closure and independently accepted staged CPU artifacts."""
    path = Path(__file__).with_name("prepare_ns_training_path_transition_lr_predecessor.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Query-serialized actual-LR preparer pinned staged protocol changed")
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_transition_lr_predecessor":
                    alias.name = "ns_training_path_query_serialized_lr_predecessor"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace(
                "ns_training_path_transition_reporting", "ns_training_path_query_serialized_reporting"
            ).replace("Transition", "Query-serialized")
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "source_plan")
    index = next(
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "waiter" for target in node.targets)
    )
    function.body[index:index] = ast.parse(
        "predecessor.validate_failed_lineage(root, predecessor.saved(root / predecessor.FAILED_SEAL, "
        "predecessor.FAILED_SEAL_SHA256))"
    ).body
    changed = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.parse(repr(VALUES[target.id]), mode="eval").body
                    changed.add(target.id)
    if changed != set(VALUES):
        raise ValueError("Query-serialized preparer source-closure pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
