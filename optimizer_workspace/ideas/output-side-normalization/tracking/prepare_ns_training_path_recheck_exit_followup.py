# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a separately accepted read-only waiter for the recheck-exit generation."""

import ast
import hashlib
from pathlib import Path

PARENT_SOURCE_SHA256 = "faabf74da5d0166b005f33ff131a5addaa12c7a4176331ec767d3e9dd35e6d9f"
VALUES = {
    "BINDING_SHA256": "ae7757c3010f88724a036d1b9f0d8b5c11b3abb53b97ca2300090f1973d1456f",
    "INTENT_SHA256": "9502a4d1d2ce65f0ff7e23f1c9bccf62572cd5430b100d76150cad46ceebb607",
    "ARCHIVE_SHA256": "e6bce471412924b48981ea4301f05e19a9899a83bdb0c59cb8549c744b8d6b05",
    "SCOPE": (
        "One distinct read-only waiter for the acknowledged recheck-exit controller and "
        "validator, checking exact original generations every900seconds. Invoke only their "
        "issued canonical independent terminal after full54-command18-decision coverage, "
        "actual owner/descendant absence and clean exact keeper restoration. Preserve "
        "rejected cases and all consumed failures, waiters and accepted unissued queues. "
        "No keeper transfer, new GPU work, numerical waiver, production, quality/efficiency "
        "or SwanLab authority."
    ),
}


def runtime_source() -> bytes:
    """Derive the pinned waiter protocol without invoking the parent's command entry point."""
    path = Path(__file__).with_name("prepare_ns_training_path_scan_exit_followup.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Recheck-exit pinned scan-exit waiter preparer changed")
    tree = ast.parse(payload)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "exec"
        )
    ]
    namespace = {"__file__": str(path)}
    exec(compile(tree, str(path), "exec"), namespace)
    tree = ast.parse(namespace["runtime_source"]())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_scan_exit_source":
                    alias.name = "ns_training_path_recheck_exit_source"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("ns_training_path_scan_exit", "ns_training_path_recheck_exit")
                .replace("ns-training-path-scan-exit", "ns-training-path-recheck-exit")
                .replace("OUTPUT_MARIN_NS_TRAINING_PATH_SCAN_EXIT", "OUTPUT_MARIN_NS_TRAINING_PATH_RECHECK_EXIT")
                .replace("scan_exit_read_only_waiter", "recheck_exit_read_only_waiter")
                .replace("Scan-exit waiter", "Recheck-exit waiter")
                .replace("Scan-exit terminal waiter", "Recheck-exit terminal waiter")
            )
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.Constant(VALUES[target.id])
                    changed.add(target.id)
    if changed != set(VALUES):
        raise ValueError("Recheck-exit waiter execution identity pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
