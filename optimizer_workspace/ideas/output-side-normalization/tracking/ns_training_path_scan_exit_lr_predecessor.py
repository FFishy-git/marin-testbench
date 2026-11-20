# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the scan-exit terminal to an isolated actual-LR handoff, preserving every failed attempt."""

import ast
import hashlib
from pathlib import Path

PARENT_SHA256 = "f38cf4e4c991ed960b20f9d6849e248100bc96cfb5672362c080dc39078bc5f9"
VALUES = {
    "NAMESPACE": "ns-training-path-scan-exit-actual-lr-predecessor-20261119",
    "STATUS": "scan_exit_terminal_sealed_actual_lr_owner_unissued",
    "EXIT_CPU": "run-preparation/ns-training-path-scan-exit-20261119-cpu/reviewed-20261119T2058/receipt.json",
    "EXIT_CPU_SHA256": "eec8b5ec97f6531bec4447078ff99b823d20458038574099fe68106b7b7bc599",
    "FOLLOWUP_CPU": "run-preparation/ns-training-path-scan-exit-followup-cpu/20261119T2110/receipt.json",
    "FOLLOWUP_CPU_SHA256": "86d6984b2166c079bced8d3b69709c2f07ac37b92dcb9f3e89360f7735f70f0c",
    "FOLLOWUP_SHA256": "b8a69286240ba12190f73f2120094b2e11f6bdb1b5e37ef34a5f12d887548840",
    "FOLLOWUP_WORK": "run-preparation/ns-training-path-scan-exit-terminal-followup-20261119",
    "FOLLOWUP_INTENT_SHA256": "9ddc9c73d53e45779208e90259ab6d391f38484372e75f29c077c148be1ef1c7",
    "BOOTSTRAP": "candidates/ns-training-path-scan-exit-followup-20261119T2110/source.json",
    "BOOTSTRAP_SHA256": "1fed47a754f4113e3a8973a0dd11e658429e281ae0abf48ede525b4552f04c50",
    "REPORTER_SHA256": "1c0e5da291eb04847bd8a4dcff7073544b3dc9db36cb261a6d40335b50fa30c9",
    "REPORT_SOURCES": {
        "ns_training_path_reporting.py": "89ece70437d29bc03b85c05fb60b870b2933b213295dfc156dbffb5d81ccc1db",
        "ns_training_path_recovery_reporting.py": "0a6cd49db261f2eff62da8dfe35a6f7c85aaea608b1eb9c8171dfd6034b6d2e9",
        "ns_training_path_exit_reporting.py": "56858d0deab1af2c76cefdac6d462eaf1c597b237d636788e1b4215e914f87b8",
        "ns_training_path_transition_reporting.py": "9e8db207b04b033740187dee1c186228ff54f60d9a13391811fe71f62eee6a54",
    },
    "BINDING": "run-preparation/ns-training-path-scan-exit-20261119-binding.json",
    "BINDING_SHA256": "7af74fd70d7ea1fbb68f09209f93ef62e19a86c17d1cb200dde99872a1b27f98",
    "INTENT_SHA256": "0020cfa6f52eea4259612d17d080029df07d1151fda088a0655117b4aa97ca22",
    "FAILED_SEAL": "run-preparation/ns-training-path-scan-exit-predecessor-seal-20261119.json",
    "FAILED_SEAL_SHA256": "59012d6f15ef1a3b4fb5cddb8049e8dd18fd63efe3f7348a2a01a923ccc1d42d",
    "SCOPE": (
        "Distinct file-only scan-exit54-command18-decision canonical-terminal handoff "
        "to an unissued actual-LR owner. Require the pinned independent waiter/terminal, "
        "exact controller/validator/waiter/terminal-child and marked descendant absence, "
        "clean restoration and exact current keeper. Preserve original44/54commands14/18decisions, "
        "recovery2/54commands0decisions, exit-drain6/54commands2decisions and failed "
        "transition23/54commands7decisions, six accepted controls, fused rejection and "
        "stopped waiter/queue lineage. Retain all accepted/rejected decisions and checkpoint "
        "digests/current identities without fresh tensor hashing. Keep numerical and execution "
        "archives separate; no merged owner, GPU transfer, launch, numerical waiver, production, "
        "quality/efficiency or SwanLab authority."
    ),
}


def literal(value: str) -> str:
    return (
        value.replace("ns_training_path_transition", "ns_training_path_scan_exit")
        .replace("ns-training-path-exit-transition", "ns-training-path-scan-exit")
        .replace("transition_owners", "scan_exit_owners")
        .replace("Transition", "Scan-exit")
        .replace(
            "failed_exit_drain_and_waiter_sealed_for_distinct_transition_not_numerical_acceptance",
            "failed_transition_and_followups_sealed_for_distinct_scan_exit_not_numerical_acceptance",
        )
    )


def runtime_source() -> bytes:
    """Keep the pinned isolated reader and strengthen its failed-lineage coverage."""
    path = Path(__file__).with_name("ns_training_path_transition_lr_predecessor.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit actual-LR predecessor pinned transition parent changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    tree = ast.parse(namespace["runtime_source"]())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = literal(node.value)
    changes = 0
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        if function.name == "waiter_cpu":
            for node in ast.walk(function):
                if isinstance(node, ast.Compare):
                    for value in node.comparators:
                        if isinstance(value, ast.Constant) and value.value == 175:
                            value.value = 180
                            changes += 1
        if function.name == "evidence":
            for node in ast.walk(function):
                if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
                    continue
                target = ast.unparse(node.left)
                if target == "failed['retained_summary']['completed_commands']":
                    node.comparators[0] = ast.Constant(value=23)
                    changes += 1
                elif target == "failed['retained_summary']['case_decisions']":
                    node.comparators[0] = ast.Constant(value=7)
                    changes += 1
            index = next(
                index
                for index, node in enumerate(function.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "chronology" for target in node.targets)
            )
            function.body[index:index] = ast.parse(
                """if (
    failed["previous_retained_summary"]["completed_commands"] != 6
    or failed["previous_retained_summary"]["case_decisions"] != 2
    or failed["previous_retained_summary"]["coverage_complete"] is not False
    or failed["previous_retained_summary"]["numerically_accepted"] is not False
):
    raise ValueError("Scan-exit actual-LR predecessor exit-drain lineage differs")
"""
            ).body
            changes += 1
    changed = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.parse(repr(VALUES[target.id]), mode="eval").body
                    changed.add(target.id)
    if changes != 4 or changed != set(VALUES):
        raise ValueError("Scan-exit actual-LR predecessor pins or retained coverage boundary differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
