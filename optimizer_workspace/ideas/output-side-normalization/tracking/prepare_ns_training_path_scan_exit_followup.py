# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a distinct read-only terminal waiter for the acknowledged scan-exit generation."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "0074627deb21ab3ce8165fd869bbd4d58cbfebea65c2e1cc55fd9ef74cd0fb65"
VALUES = {
    "BINDING_SHA256": "7af74fd70d7ea1fbb68f09209f93ef62e19a86c17d1cb200dde99872a1b27f98",
    "INTENT_SHA256": "0020cfa6f52eea4259612d17d080029df07d1151fda088a0655117b4aa97ca22",
    "ARCHIVE_SHA256": "ccc4ae7be2ce30218bed1054a764af084457648b5b32757da0452495caed82a4",
    "SCOPE": (
        "One distinct read-only waiter for the acknowledged missing-environment scan-exit "
        "controller and validator, checking exact original generations every900seconds. "
        "Invoke only their issued canonical independent terminal after full54-command "
        "18-decision coverage, actual owner/descendant absence and clean exact keeper "
        "restoration. Retain rejected cases and all consumed failed predecessors, waiters "
        "and queues. No keeper transfer, new GPU work, numerical waiver, production, "
        "optimizer quality/efficiency or SwanLab authority."
    ),
}


def runtime_source() -> bytes:
    """Retain the accepted terminal-waiter protocol with distinct bound execution identities."""
    path = Path(__file__).with_name("prepare_ns_training_path_transition_followup.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit waiter pinned transition preparer changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    tree = ast.parse(namespace["runtime_source"]())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_transition_source":
                    alias.name = "ns_training_path_scan_exit_source"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_transition", "ns_training_path_scan_exit")
                .replace("ns-training-path-exit-transition", "ns-training-path-scan-exit")
                .replace("OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION", "OUTPUT_MARIN_NS_TRAINING_PATH_SCAN_EXIT")
                .replace("exit_transition_read_only_waiter", "scan_exit_read_only_waiter")
                .replace("Transition waiter", "Scan-exit waiter")
                .replace("Transition terminal waiter", "Scan-exit terminal waiter")
            )
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.Constant(value=VALUES[target.id])
                    changed.add(target.id)
    if changed != set(VALUES):
        raise ValueError("Scan-exit waiter binding replacement differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
