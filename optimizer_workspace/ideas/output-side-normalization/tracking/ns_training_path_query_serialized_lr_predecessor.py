# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the original query-serialized terminal to a separate actual-LR handoff."""

import ast
import hashlib
import json
from pathlib import Path

from ns_training_path_exit_lr_predecessor import checked, relative

PARENT_SHA256 = "e0724c9b0e3021b74d0824083fd2a125452f240bd858a349c3a4e4245c3d9aad"
VALUES = {
    "NAMESPACE": "ns-training-path-query-serialized-actual-lr-predecessor-20261120",
    "STATUS": "query_serialized_terminal_sealed_actual_lr_owner_unissued",
    "EXIT_CPU": "run-preparation/ns-training-path-query-serialized-20261120-cpu/reviewed-20261120T0012/receipt.json",
    "EXIT_CPU_SHA256": "dee0f1f2794237af02cfc0ac0f966a8a30a08f55a489d6ac7ef05291aebf3dc8",
    "FOLLOWUP_CPU": "run-preparation/ns-training-path-query-serialized-followup-cpu/20261120T0026/receipt.json",
    "FOLLOWUP_CPU_SHA256": "65e5133be24ea2d2ab1fbb4155c7c25d1c34d9ddc06ce08020694bde6d362e4d",
    "FOLLOWUP_SHA256": "58365100058d55822eedcf35439b7ed9e4d6d1ccb51f90f3e63e9c01ef73ea7f",
    "FOLLOWUP_WORK": "run-preparation/ns-training-path-query-serialized-terminal-followup-20261120",
    "FOLLOWUP_INTENT_SHA256": "3679f0cbca16b619995e76b70950951348db6ff4bb9592f435ccbc329f391666",
    "BOOTSTRAP": "run-preparation/ns-training-path-query-serialized-followup-bootstrap-20261120T0026/source.json",
    "BOOTSTRAP_SHA256": "fc0d8c842f2f768726f82eaad154bf4df8f6dd2c195e11d3d6f2375aa6c9ff9d",
    "REPORTER_SHA256": "94d7fc2cc18b1ad3f37d8a2db05a7b3016ee82c4a8468993ddac28393db28652",
    "REPORT_SOURCES": {
        "ns_training_path_reporting.py": "89ece70437d29bc03b85c05fb60b870b2933b213295dfc156dbffb5d81ccc1db",
        "ns_training_path_recovery_reporting.py": "0a6cd49db261f2eff62da8dfe35a6f7c85aaea608b1eb9c8171dfd6034b6d2e9",
        "ns_training_path_exit_reporting.py": "56858d0deab1af2c76cefdac6d462eaf1c597b237d636788e1b4215e914f87b8",
        "ns_training_path_transition_reporting.py": "9e8db207b04b033740187dee1c186228ff54f60d9a13391811fe71f62eee6a54",
        "ns_training_path_scan_exit_reporting.py": "1c0e5da291eb04847bd8a4dcff7073544b3dc9db36cb261a6d40335b50fa30c9",
        "ns_training_path_recheck_exit_reporting.py": "696b19372f97c91257a47869a03b979b01afec92db80ceec6199079e349b0a1d",
        "ns_training_path_serialized_reporting.py": "d399387d91006cd0c0ed4242dcdfb664b3b29c26dde1edb4460c00bd5f064034",
    },
    "BINDING": "run-preparation/ns-training-path-query-serialized-20261120-binding.json",
    "BINDING_SHA256": "8e6760d8a0a679e057ad7434df2e969925f055c57912e271b42ff45fa1cfe24e",
    "INTENT_SHA256": "43e810371a8441b56f75c740ffc8ec126473acaaf360873647dd9d9aaffef3fe",
    "FAILED_SEAL": "run-preparation/ns-training-path-query-serialized-predecessor-seal-20261120.json",
    "FAILED_SEAL_SHA256": "ca766ffc6298c7bb9284a7d514c44b875d0a80bfb53f9029f7794eae64388373",
    "SCOPE": (
        "Distinct file-only query-serialized54-command18-decision canonical-terminal handoff "
        "to an unissued actual-LR owner. Require the pinned original independent waiter/terminal, "
        "exact controller/validator/waiter/terminal-child and marked descendant absence, "
        "clean restoration and exact current keeper. Preserve whole-inspection0/0, recheck1/0, "
        "scan-exit8/2, transition23/7, exit-drain6/2, recovery2/0 and original44/14 failures, "
        "all accepted/rejected decisions, stopped owners/waiters and accepted unissued queues. "
        "Reuse checkpoint digests and inspect current identities without tensor replay. Keep "
        "numerical and execution archives separate; no GPU transfer, launch, numerical waiver, "
        "production, optimizer quality/efficiency or SwanLab authority."
    ),
}
FAILED_LINEAGE = (
    (
        "run-preparation/ns-training-path-serialized-predecessor-seal-20261119.json",
        "dd3d333dfbe00686d8a25da358ae783c11cc0163a679c750338076b5c9a0bdfe",
        "failed_recheck_and_waiter_sealed_for_serialized_owner_not_numerical_acceptance",
        {"retained_summary": (1, 0), "previous_retained_summary": (8, 2)},
    ),
    (
        "run-preparation/ns-training-path-recheck-exit-predecessor-seal-20261119.json",
        "10c189fc543b9d9e1ec4d8d28bf590700948ed108dbd02e8bfd1174fe66193b6",
        "failed_scan_exit_and_waiter_sealed_for_distinct_recheck_exit_not_numerical_acceptance",
        {"retained_summary": (8, 2), "previous_retained_summary": (23, 7)},
    ),
    (
        "run-preparation/ns-training-path-scan-exit-predecessor-seal-20261119.json",
        "59012d6f15ef1a3b4fb5cddb8049e8dd18fd63efe3f7348a2a01a923ccc1d42d",
        "failed_transition_and_followups_sealed_for_distinct_scan_exit_not_numerical_acceptance",
        {"retained_summary": (23, 7), "previous_retained_summary": (6, 2)},
    ),
    (
        "run-preparation/ns-training-path-exit-transition-predecessor-seal-20261119.json",
        "070227bc90c040368f7ce2a28ef897425a1f24639d870377954827dbe20f9b49",
        "failed_exit_drain_and_waiter_sealed_for_distinct_transition_not_numerical_acceptance",
        {"retained_summary": (6, 2)},
    ),
    (
        "run-preparation/ns-training-path-exit-drain-predecessor-seal-20261119.json",
        "07980e69f353192141e52d7dba5e98ea9fda35841f5d1c7be683408fb16237e3",
        "both_failed_training_path_attempts_sealed_for_distinct_exit_drain_not_numerical_acceptance",
        {"retained_summary": (2, 0)},
    ),
)


def literal(value: str) -> str:
    return (
        value.replace("ns_training_path_exit_drain", "ns_training_path_query_serialized")
        .replace("ns_training_path_exit_", "ns_training_path_query_serialized_")
        .replace("ns-training-path-exit-drain-20261119", "ns-training-path-query-serialized-20261120")
        .replace("ns-training-path-exit-drain", "ns-training-path-query-serialized")
        .replace("exit_drain_owners", "query_serialized_owners")
        .replace("Exit-drain", "Query-serialized")
        .replace(
            "both_failed_training_path_attempts_sealed_for_distinct_exit_drain_not_numerical_acceptance",
            "failed_whole_inspection_and_waiter_sealed_for_query_serialization_not_numerical_acceptance",
        )
    )


def validate_failed_lineage(root: Path, failed: dict) -> None:
    """Authenticate ancestor seals directly; the latest seal does not flatten their summaries."""
    ancestors = []
    for name, digest, status, summaries in FAILED_LINEAGE:
        path = relative(root, name)
        payload, identity = checked(path, digest)
        if failed["evidence"].get(name) != identity:
            raise ValueError("Query-serialized predecessor lost an authenticated ancestor seal")
        ancestor = json.loads(payload)
        if ancestor["status"] != status or ancestor["root"] != str(root):
            raise ValueError("Query-serialized predecessor failed ancestor attribution differs")
        ancestors.append((ancestor, summaries))
    ancestors.append((failed, {"retained_summary": (0, 0), "previous_retained_summary": (1, 0)}))
    for record, summaries in ancestors:
        if record["gpu_execution_authorized"] is not False or record["production_authorized"] is not False:
            raise ValueError("Query-serialized predecessor failed lineage grants execution")
        expected = summaries | {"original_retained_summary": (44, 14)}
        if "recovery_retained_summary" in record:
            expected["recovery_retained_summary"] = (2, 0)
        for name, (commands, decisions) in expected.items():
            summary = record[name]
            if (
                type(summary["completed_commands"]) is not int
                or summary["completed_commands"] != commands
                or type(summary["case_decisions"]) is not int
                or summary["case_decisions"] != decisions
                or summary["coverage_complete"] is not False
                or summary["numerically_accepted"] is not False
            ):
                raise ValueError("Query-serialized predecessor failed coverage lineage differs")


def runtime_source() -> bytes:
    """Retain the original terminal/ownership protocol with separately pinned query receipts."""
    path = Path(__file__).with_name("ns_training_path_exit_lr_predecessor.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Query-serialized actual-LR predecessor pinned isolated reader changed")
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = literal(node.value)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    changes = 0
    for node in ast.walk(functions["waiter_cpu"]):
        if isinstance(node, ast.Compare):
            for value in node.comparators:
                if isinstance(value, ast.Constant) and value.value == 169:
                    value.value = 195
                    changes += 1
    for node in ast.walk(functions["evidence"]):
        if (
            isinstance(node, ast.Compare)
            and ast.unparse(node.left) == "failed['retained_summary']['completed_commands']"
        ):
            node.comparators[0] = ast.Constant(0)
            changes += 1
    function = functions["evidence"]
    index = next(
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "chronology" for target in node.targets)
    )
    function.body[index:index] = ast.parse("validate_failed_lineage(root, failed)").body
    changed = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in VALUES:
                    node.value = ast.parse(repr(VALUES[target.id]), mode="eval").body
                    changed.add(target.id)
    if changes != 2 or changed != set(VALUES):
        raise ValueError("Query-serialized predecessor coverage boundary or receipt pins differ")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
