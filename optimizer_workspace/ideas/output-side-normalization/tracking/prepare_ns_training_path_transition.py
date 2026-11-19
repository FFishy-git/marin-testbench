# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive and issue a fresh diagnostic with the measured mixed-bracket correction."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_transition_source as transition_source

PARENT_PREPARER_SHA256 = "f85ff94a72820b04970c1e756920690bcde3726ce3f0e1183673ba0907a9ffcc"
CPU_PATH_VALUE = (
    "run-preparation/ns-training-path-exit-transition-20261119-cpu/closure-corrected-20261119T1700/receipt.json"
)


def runtime_source() -> bytes:
    path = Path(__file__).with_name("prepare_ns_training_path_exit.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_PREPARER_SHA256:
        raise ValueError("Exit-transition accepted parent preparer changed")
    tree = ast.parse(payload)
    namespace = {"__file__": str(path)}
    exec(compile(tree, str(path), "exec"), namespace)
    tree = transition_source.TransitionNamespace().visit(ast.parse(namespace["runtime_source"]()))
    replacements = {
        "CPU_PATH": repr(CPU_PATH_VALUE),
        "LOCAL_HELPERS": repr(
            (
                "ns_training_path_transition_predecessor.py",
                "ns_training_path_transition_source.py",
                "prepare_ns_training_path_transition.py",
                "prepare_ns_training_path_exit.py",
                "test_ns_training_path_transition.py",
                "owned_output_attribution.py",
                "owned_output_exit_drain.py",
                "owned_output_exit_transition.py",
                "test_owned_output_exit_drain.py",
                "test_owned_output_exit_transition.py",
            )
        ),
        "TEST_COUNTS": repr(
            {
                "test_ns_training_path_transition_gpu": 45,
                "test_owned_output_handoff": 36,
                "test_owned_output_exit_transition": 21,
            }
        ),
        "RECOVERY_TEST_COUNT": "27",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(replacements[target.id], mode="eval").body
    # The parent preparer retains inherited original-failure regression functions;
    # the successor's own file runs the new retained-exit failure regressions.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "test_ns_training_path_recovery":
            node.value = "test_ns_training_path_transition"
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
