# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project the distinct scan-exit actual-LR queue from authenticated saved receipts."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_scan_exit_lr_queue as derivation

PARENT_SHA256 = "282efacbbe881ed3e76cebabeefd3587a9940ba5399a3e07c2bae4d162ac320b"
CPU_PATH = "run-preparation/ns-training-path-scan-exit-actual-lr-queue-cpu/20261119T2202/receipt.json"
CPU_PIN: str | None = "8142d861674a5ac95003e25a60da86e478934aa7f2a431bf132538b5c2d117fa"
INTENT_PIN: str | None = None
TITLE = "NS Scan-Exit Actual-LR Queue"


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_transition_lr_queue_reporting.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit LR queue pinned reporting source changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    tree = derivation.Namespace().visit(ast.parse(namespace["runtime_source"]()))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("waiting_for_transition_owners", "waiting_for_scan_exit_owners").replace(
                "NS Transition Actual-LR Queue", TITLE
            )
    replacements = {"CPU": CPU_PATH, "CPU_SHA256": CPU_PIN, "INTENT_SHA256": INTENT_PIN}
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
                    changed.add(target.id)
    if changed != set(replacements):
        raise ValueError("Scan-exit LR queue reporting receipt replacement differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
