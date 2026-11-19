# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the transition terminal to a distinct actual-LR handoff without merging archives."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "e0724c9b0e3021b74d0824083fd2a125452f240bd858a349c3a4e4245c3d9aad"
VALUES = {
    "NAMESPACE": "ns-training-path-transition-actual-lr-predecessor-20261119",
    "STATUS": "transition_terminal_sealed_actual_lr_owner_unissued",
    "EXIT_CPU": (
        "run-preparation/ns-training-path-exit-transition-20261119-cpu/closure-corrected-20261119T1700/receipt.json"
    ),
    "EXIT_CPU_SHA256": "ad18edf9b4f230e78bd7eef2a97f44bde307a1c524192d37f23491257b6fc392",
    "FOLLOWUP_CPU": "run-preparation/ns-training-path-exit-transition-followup-cpu/20261119T1718/receipt.json",
    "FOLLOWUP_CPU_SHA256": "6a0994915727bf06c606ff0a2bfcece8e4070045ac1f0dcb5f2e79d6b3041906",
    "FOLLOWUP_SHA256": "9beaf16b3df4347bbbcca05339689ee0baef25afb81e2dd1b8ed9fff67bdacb8",
    "FOLLOWUP_WORK": "run-preparation/ns-training-path-exit-transition-terminal-followup-20261119",
    "FOLLOWUP_INTENT_SHA256": "1641a7cfd73b11d71a91220c7aab6b9aa6e9601d3804237ce4d89abaef0aae0d",
    "BOOTSTRAP": "run-preparation/ns-training-path-exit-transition-followup-bootstrap/20261119T1718/source.json",
    "BOOTSTRAP_SHA256": "2a042f189b20fe1199e28415616b75ca6971cecfc75a258c745dcd5022d970a9",
    "REPORTER_SHA256": "9e8db207b04b033740187dee1c186228ff54f60d9a13391811fe71f62eee6a54",
    "REPORT_SOURCES": {
        "ns_training_path_reporting.py": "89ece70437d29bc03b85c05fb60b870b2933b213295dfc156dbffb5d81ccc1db",
        "ns_training_path_recovery_reporting.py": "0a6cd49db261f2eff62da8dfe35a6f7c85aaea608b1eb9c8171dfd6034b6d2e9",
        "ns_training_path_exit_reporting.py": "56858d0deab1af2c76cefdac6d462eaf1c597b237d636788e1b4215e914f87b8",
    },
    "BINDING": "run-preparation/ns-training-path-exit-transition-20261119-binding.json",
    "BINDING_SHA256": "18c9b21d16ff817059177e6746c3ae2246d39232396220c2b1deec805a33ca3d",
    "INTENT_SHA256": "42a901376f00df7b1672ab5d9dad610f669097ca0e48c16690ea4e48b2fb54e1",
    "FAILED_SEAL": "run-preparation/ns-training-path-exit-transition-predecessor-seal-20261119.json",
    "FAILED_SEAL_SHA256": "070227bc90c040368f7ce2a28ef897425a1f24639d870377954827dbe20f9b49",
    "SCOPE": (
        "Distinct file-only transition54-command18-decision canonical-terminal handoff "
        "to an unissued actual-LR owner. Require the pinned independent waiter/terminal, "
        "exact controller/validator/waiter/terminal-child and marked descendant absence, "
        "clean restoration and exact current keeper. Preserve original44/54commands14/18decisions, "
        "recovery2/54commands0decisions, exit-drain6/54commands2decisions and failed waiter "
        "lineage. Retain all accepted/rejected decisions and checkpoint digests/current "
        "identities without new tensor hashing. Keep numerical and execution archives "
        "separate; no merged owner, GPU transfer, launch, numerical waiver, production, "
        "quality/efficiency or SwanLab authority."
    ),
}


def literal(value: str) -> str:
    return (
        value.replace("ns_training_path_exit_drain", "ns_training_path_transition")
        .replace("ns_training_path_exit_", "ns_training_path_transition_")
        .replace("ns-training-path-exit-drain", "ns-training-path-exit-transition")
        .replace("exit_drain_owners", "transition_owners")
        .replace("Exit-drain", "Transition")
        .replace(
            "both_failed_training_path_attempts_sealed_for_distinct_exit_drain_not_numerical_acceptance",
            "failed_exit_drain_and_waiter_sealed_for_distinct_transition_not_numerical_acceptance",
        )
    )


def runtime_source() -> bytes:
    """Retain the isolated file-reader protocol and extend its failed-lineage binding."""
    path = Path(__file__).with_name("ns_training_path_exit_lr_predecessor.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Transition actual-LR predecessor pinned parent changed")
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = literal(node.value)
    changes = 0
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name == "waiter_cpu":
            for node in ast.walk(function):
                if isinstance(node, ast.Compare):
                    for value in node.comparators:
                        if isinstance(value, ast.Constant) and value.value == 169:
                            value.value = 175
                            changes += 1
        if isinstance(function, ast.FunctionDef) and function.name == "evidence":
            for node in ast.walk(function):
                if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
                    continue
                target = ast.unparse(node.left)
                if target == "failed['retained_summary']['completed_commands']":
                    node.comparators[0] = ast.Constant(value=6)
                    changes += 1
                elif target == "failed['retained_summary']['case_decisions']":
                    node.comparators[0] = ast.Constant(value=2)
                    changes += 1
            index = next(
                index
                for index, node in enumerate(function.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "chronology" for target in node.targets)
            )
            function.body[index:index] = ast.parse(
                """if (
    failed["recovery_retained_summary"]["completed_commands"] != 2
    or failed["recovery_retained_summary"]["case_decisions"] != 0
    or failed["recovery_retained_summary"]["coverage_complete"] is not False
    or failed["recovery_retained_summary"]["numerically_accepted"] is not False
):
    raise ValueError("Transition actual-LR predecessor recovery lineage differs")
"""
            ).body
            changes += 1
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.parse(repr(VALUES[target.id]), mode="eval").body
    if changes != 4:
        raise ValueError("Transition actual-LR predecessor parent coverage boundary differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
