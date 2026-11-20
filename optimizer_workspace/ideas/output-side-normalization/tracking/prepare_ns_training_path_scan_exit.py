# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive and issue a distinct diagnostic using the missing-environment ownership correction."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_scan_exit_source as successor

PARENT_SHA256 = "e46a66d3a20e6927f61cdd6f3234e41f5599ef1c1a936119adb50457f464380f"
CPU_PATH_VALUE = "run-preparation/ns-training-path-scan-exit-20261119-cpu/reviewed-20261119T2058/receipt.json"


def runtime_source() -> bytes:
    path = Path(__file__).with_name("prepare_ns_training_path_transition.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit accepted orchestration preparer changed")
    namespace = {"__file__": str(path)}
    exec(compile(ast.parse(payload), str(path), "exec"), namespace)
    tree = successor.ScanExitNamespace().visit(ast.parse(namespace["runtime_source"]()))
    replacements = {
        "CPU_PATH": CPU_PATH_VALUE,
        "LOCAL_HELPERS": (
            "ns_training_path_scan_exit_predecessor.py",
            "ns_training_path_scan_exit_source.py",
            "prepare_ns_training_path_scan_exit.py",
            "prepare_ns_training_path_transition.py",
            "prepare_ns_training_path_exit.py",
            "test_ns_training_path_scan_exit.py",
            "owned_output_attribution.py",
            "owned_output_exit_drain.py",
            "owned_output_exit_transition.py",
            "owned_output_missing_environment.py",
            "test_owned_output_exit_drain.py",
            "test_owned_output_exit_transition.py",
            "test_owned_output_missing_environment.py",
        ),
        "TEST_COUNTS": {
            "test_ns_training_path_scan_exit_gpu": 45,
            "test_owned_output_handoff": 36,
            "test_owned_output_missing_environment": 33,
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
        raise ValueError("Scan-exit orchestration successor constants differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
