# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect the separately accepted query terminal reader without merging numerical archives."""

import ast
import hashlib
from pathlib import Path

DERIVATION_SHA256 = "58ccfde4c5d4683162204bc438186f538824b3df9f01a74c2c76e34990904274"
VALUES = {
    "CPU": (
        "run-preparation/ns-training-path-query-serialized-actual-lr-predecessor-cpu/staged-20261120T0053/receipt.json"
    ),
    "CPU_SHA256": "06533d588b4681413028cdfe7c8830c02adc1ae6572beab6e59bfe2b52684006",
    "STATUS": "query_serialized_actual_lr_predecessor_cpu_archive_verified_execution_archives_separate",
    "VERIFIER": "ns_training_path_query_serialized_lr_predecessor.py",
    "VERIFIER_SHA256": "198c5d5d5cd4c26f89d9f72cb135997bfedf887e577a53d5ed1f095e33d496c8",
    "CHECKER": "prepare_ns_training_path_query_serialized_lr_predecessor.py",
    "CHECKER_SHA256": "9af1f97bb28827adf1fbda0a3c25bdc68595be23cb69ca0386045522a8720a54",
    "SEAL": "run-preparation/ns-training-path-query-serialized-actual-lr-predecessor-20261120-seal.json",
}


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_transition_lr_parent.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != DERIVATION_SHA256:
        raise ValueError("Query actual-LR bridge pinned isolated parent changed")
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace(
                "ns_training_path_transition_lr", "ns_training_path_query_serialized_lr"
            ).replace("Transition-LR", "Query actual-LR")
    changed = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.parse(repr(VALUES[target.id]), mode="eval").body
                    changed.add(target.id)
    if changed != set(VALUES):
        raise ValueError("Query actual-LR bridge receipt/source pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
