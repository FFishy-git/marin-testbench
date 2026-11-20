# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate the distinct scan-exit terminal reader without merging numerical archives."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "58ccfde4c5d4683162204bc438186f538824b3df9f01a74c2c76e34990904274"
VALUES = {
    "CPU": "run-preparation/ns-training-path-scan-exit-actual-lr-predecessor-cpu/staged-20261119T2132/receipt.json",
    "CPU_SHA256": "66bb686faee018b44d8c1b3371a5576bb14a912933f034e6048945d132a66f6b",
    "STATUS": "scan_exit_actual_lr_predecessor_cpu_archive_verified_execution_archives_separate",
    "VERIFIER": "ns_training_path_scan_exit_lr_predecessor.py",
    "VERIFIER_SHA256": "de1b55a2174406b40d0c78e6441a205c0c0421ae7efc2dbb0576597c0814b3a3",
    "CHECKER": "prepare_ns_training_path_scan_exit_lr_predecessor.py",
    "CHECKER_SHA256": "a257ae0d5eb176db9d0d249a555b0aef3d2e41c8fdea5fb3fe9c639d33048de3",
    "SEAL": "run-preparation/ns-training-path-scan-exit-actual-lr-predecessor-20261119-seal.json",
}


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_transition_lr_parent.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit LR parent pinned isolated reader changed")
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("ns_training_path_transition_lr", "ns_training_path_scan_exit_lr").replace(
                "Transition-LR", "Scan-exit LR"
            )
    changed = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.parse(repr(VALUES[target.id]), mode="eval").body
                    changed.add(target.id)
    if changed != set(VALUES):
        raise ValueError("Scan-exit LR parent receipt or source pin replacement differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
