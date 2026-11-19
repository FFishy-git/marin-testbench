# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report the distinct transition-LR queue from authenticated saved receipts."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_transition_lr_queue as derivation

PARENT_SHA256 = "c5264e454fcec70e70fe60f342f6af7093a356e02770191d6cce347204e17a54"
CPU_PATH = "run-preparation/ns-training-path-transition-actual-lr-queue-cpu/reviewed-20261119T1855/receipt.json"
CPU_PIN: str | None = "ed283127735f2c23648fb14b8342aad46ffb112aabc29dd059e221562c37dac5"
INTENT_PIN: str | None = "95a67ee0e8cf787344215cfe50413379aac84565573a12de86ee067a3575bd38"
TITLE = "NS Transition Actual-LR Queue"


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_lr_queue_reporting.py")
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != PARENT_SHA256:
        raise ValueError("Transition-LR queue reporting parent changed")
    tree = derivation.Namespace().visit(ast.parse(raw))
    replacements = {"CPU": repr(CPU_PATH), "CPU_SHA256": repr(CPU_PIN), "INTENT_SHA256": repr(INTENT_PIN)}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(replacements[target.id], mode="eval").body
                elif isinstance(target, ast.Name) and target.id == "STATUSES":
                    node.value.elts.append(ast.Constant(value="waiting_for_transition_owners"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "### NS Actual-LR Training-Path Queue":
            node.value = f"### {TITLE}"
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
