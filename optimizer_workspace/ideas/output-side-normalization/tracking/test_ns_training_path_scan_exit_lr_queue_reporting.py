# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain saved queue waiting, failure and rejection summaries in the split journals."""

import ast
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import ns_training_path_scan_exit_lr_queue as derivation
import ns_training_path_scan_exit_lr_queue_reporting as reporting
import update_endeavor as journal
from watch_wave5 import write

PARENT_SHA256 = "95b02374ac884332e17fa1c19f7782be74cbf305a20778f22d82401a356a3353"


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_ns_training_path_transition_lr_queue_reporting.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit LR queue pinned reporting tests changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    tree = derivation.Namespace().visit(ast.parse(namespace["inherited_tests"]()))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_transition_lr_queue_reporting":
                    alias.name = "ns_training_path_scan_exit_lr_queue_reporting"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace(
                    "ns_training_path_transition_lr_queue_reporting", "ns_training_path_scan_exit_lr_queue_reporting"
                )
                .replace("NS Transition Actual-LR Queue", "NS Scan-Exit Actual-LR Queue")
                .replace("waiting_for_transition_owners", "waiting_for_scan_exit_owners")
            )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())


def test_scan_exit_queue_refresh_publishes_both_journals_and_withholds_changed_receipt(queued):
    root, work, cpu_path, state, _ = queued
    saved_state = (work / "state.json").read_bytes()
    research, details = root / "ENDEAVOR.md", root / "DETAILS.md"
    research.write_text("Retained scientific decision.\n\n[`DETAILS.md`](DETAILS.md)\n\n## Document Management\n")
    details.write_text("Retained execution evidence.\n\n## Verification and Remaining Work\n")
    journal.update(research, root, datetime.now(UTC))
    for document, retained in (
        (research, "Retained scientific decision."),
        (details, "Retained execution evidence."),
    ):
        text = document.read_text()
        assert retained in text
        assert "### NS Scan-Exit Actual-LR Queue" in text
        assert "0/6 completed phases" in text and state["status"] in text
    write(cpu_path, {"fixture": "substituted queue acceptance"})
    journal.update(details, root, datetime.now(UTC))
    for document in (research, details):
        section = document.read_text().split("### NS Scan-Exit Actual-LR Queue", 1)[1].split("### ", 1)[0]
        assert "evidence refused" in section and "0/6 completed phases" not in section
    assert (work / "state.json").read_bytes() == saved_state
    assert not reporting.queue.contract.locations(root)[0].exists()
