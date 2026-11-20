# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind scan-exit LR ownership to its separately accepted canonical terminal reader."""

import ast
from pathlib import Path

import ns_training_path_scan_exit_lr_gpu_source as derivation
import ns_training_path_scan_exit_lr_parent as bridge

SOURCE_SHA256 = "4399c9ff5aeff03c15033d34668bba46566d6b628d36d0f5130d5be786682679"


def runtime_source() -> bytes:
    tree = derivation.transform(Path(__file__).with_name("ns_training_path_transition_lr_gpu_owner.py"), SOURCE_SHA256)
    replacements = {
        "PREDECESSOR_RECEIPT": bridge.CPU,
        "PREDECESSOR_RECEIPT_SHA256": bridge.CPU_SHA256,
        "PREDECESSOR_VERIFIER_SHA256": bridge.VERIFIER_SHA256,
        "PREDECESSOR_SEALER_SHA256": bridge.CHECKER_SHA256,
        "ARCHIVE": f"run-preparation/source-{derivation.adapter_source.NAMESPACE}.tar",
    }
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
                    changed.add(target.id)
    if changed != set(replacements):
        raise ValueError("Scan-exit LR execution owner predecessor replacement differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
