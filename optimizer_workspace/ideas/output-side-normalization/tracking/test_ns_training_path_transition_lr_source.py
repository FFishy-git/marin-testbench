# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise preserved actual-LR numerics and refusal at the fresh unissued owner boundary."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_lr_contract as numerical_contract
import ns_training_path_transition_lr_source as source
from selected_lr_release_owner import sha256

PARENT_SHA256 = "e9a0130cf288cc00d6424e3b9538c93885802d287edc328c58223b7e007cca6c"


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_ns_training_path_lr_execution_source.py")
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != PARENT_SHA256:
        raise ValueError("Transition-LR pinned adapter regressions changed")
    tree = source.Namespace().visit(ast.parse(raw))
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        if function.name == "test_execution_closure_keeps_archived_helpers_and_refuses_an_unaccepted_import":
            for node in ast.walk(function):
                if isinstance(node, ast.GeneratorExp):
                    node.generators[0].ifs.append(ast.parse("name != source.HANDOFF", mode="eval").body)
        if function.name == "test_adapter_cpu_archive_retains_parent_adapter_and_refuses_rehashed_incomplete_evidence":
            position = next(index for index, node in enumerate(function.body) if isinstance(node, ast.FunctionDef))
            function.body.insert(position, ast.parse("real_run = subprocess.run").body[0])
            completed = function.body[position + 1]
            completed.body[:0] = ast.parse(
                """if kwargs.get("capture_output"):
    return real_run(command,cwd=cwd,**kwargs)
"""
            ).body
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())


def test_measured_handoff_changes_only_named_execution_source_and_preserves_all_actual_lrs(candidate):
    _, _, successor, prepared, previous, _ = candidate
    inherited = previous["helpers_sha256"]
    executed = prepared["helpers_sha256"]
    assert {name for name, digest in inherited.items() if executed[name] != digest} == {source.HANDOFF}
    assert prepared["helper_transform"] == {
        source.HANDOFF: {
            "original_sha256": source.NUMERICAL_HANDOFF_SHA256,
            "executed_sha256": source.HANDOFF_SHA256,
        }
    }
    assert prepared["terminal_archive_merged"] is False
    assert prepared["predecessor_cpu_receipt_sha256"] == source.bridge.CPU_SHA256
    assert sha256(Path(previous["helpers"]) / source.HANDOFF) == source.NUMERICAL_HANDOFF_SHA256
    assert len(prepared["plan"]["trials"]) == 54 and len(prepared["plan"]["commands"]) == 162
    assert source.contract.specs() == numerical_contract.specs()
    assert sha256(successor / source.OWNER) != previous["files_sha256"][source.OWNER]
