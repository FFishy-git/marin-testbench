# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain failed scan-exit coverage and the accepted unissued actual-LR queue."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_scan_exit_predecessor as parent

PARENT_SOURCE_SHA256 = "9177fd2fd657c0fb18dbc22e1511e7547aef956b959da1cd0f98faf18de22f13"
OLD_NAMESPACE = "ns-training-path-scan-exit-20261119"
WORK = "validation/ns_training_path_scan_exit_20261119"
WAITER = "run-preparation/ns-training-path-scan-exit-terminal-followup-20261119"
QUEUE = "run-preparation/ns-training-path-scan-exit-actual-lr-queue-20261119"
CPU_SPEC = (
    "run-preparation/ns-training-path-scan-exit-20261119-cpu/reviewed-20261119T2058/receipt.json",
    "eec8b5ec97f6531bec4447078ff99b823d20458038574099fe68106b7b7bc599",
    "ns_training_path_scan_exit_cpu_archive_verified_gpu_unissued",
    129,
)
WAITER_CPU_SPEC = (
    "run-preparation/ns-training-path-scan-exit-followup-cpu/20261119T2110/receipt.json",
    "86d6984b2166c079bced8d3b69709c2f07ac37b92dcb9f3e89360f7735f70f0c",
    "ns_training_path_scan_exit_read_only_followup_cpu_archive_verified",
    24,
)
QUEUE_CPU_SPEC = (
    "run-preparation/ns-training-path-scan-exit-actual-lr-queue-cpu/20261119T2202/receipt.json",
    "8142d861674a5ac95003e25a60da86e478934aa7f2a431bf132538b5c2d117fa",
    "ns_training_path_scan_exit_actual_lr_queue_cpu_archive_verified_gpu_unissued",
    47,
)
FAILED_LOG = f"{WORK}/muont_ns_tpu_q10_full_lr_lower_resumed.log"
SEAL = "run-preparation/ns-training-path-recheck-exit-predecessor-seal-20261119.json"
STATUS = "failed_scan_exit_and_waiter_sealed_for_distinct_recheck_exit_not_numerical_acceptance"
PREVIOUS_SEAL_SHA256 = "59012d6f15ef1a3b4fb5cddb8049e8dd18fd63efe3f7348a2a01a923ccc1d42d"
RECORDS = {
    f"run-preparation/{OLD_NAMESPACE}-binding.json": "7af74fd70d7ea1fbb68f09209f93ef62e19a86c17d1cb200dde99872a1b27f98",
    f"run-preparation/{OLD_NAMESPACE}-state.json": "1e5553c5be5f9e74cfc5b95bd25b21001eae82572d19ef040657255cd69a1c94",
    f"run-preparation/validation-{OLD_NAMESPACE}-gpu.json": (
        "b08076ccab68d23dcae34cd1f135c750d9586664b56d5bd6c490974b6a17cb37"
    ),
    f"controllers/{OLD_NAMESPACE}/controller.json": "0020cfa6f52eea4259612d17d080029df07d1151fda088a0655117b4aa97ca22",
    f"controllers/{OLD_NAMESPACE}/controller.log": "2bdce3fbd962151e95cde9ae849ff5ff170c8af4b58c12d4880e2d953bd7576d",
    f"controllers/{OLD_NAMESPACE}/validator.log": "40ef15a525dd38254de0391a3c09e169c4527118054921c91daf37547445c4c3",
    FAILED_LOG: "28d768197c1090eadefaa843cf9622d3c267c7a0fe82ffdfb93efc3f0c1bc95d",
    FAILED_LOG.replace(".log", ".execution.json"): "1e5bbaa373579f7b86a91f8173cdb81e8467c28cedcab0f5414633bbf8ea24a1",
    f"{WAITER}/intent.json": "9ddc9c73d53e45779208e90259ab6d391f38484372e75f29c077c148be1ef1c7",
    f"{WAITER}/state.json": "a83799e14d82f694d4c0f363b5ee13c8531b3ad91f15a60501b968c437d10b86",
    f"{WAITER}/followup.log": "4a527b743157518270053d775c2f8b965e2a8e39c89e69b53e9e068b23efd09e",
    "run-preparation/endeavor-scan-exit-failure-restoration-observation-20261119T2213.json": (
        "abceb5f9f538619419c550226cc43fc1f7af76d3c12097d01efbe30a5c4cae2f"
    ),
}
SCOPE = (
    "Preserve scan-exit8/54commands2/18accepted-controls, failed ten-step resumed command "
    "and stopped four-observation waiter. Preserve accepted214-helper actual-LR queue "
    "unissued, all transition and older failures/rejections/seals, exact source/checkpoint "
    "identities and restored eight-GPU keeper. No complete terminal, numerical waiver, "
    "quality/efficiency, GPU or production authority. Only a new accepted ownership-only "
    "recheck diagnostic may proceed."
)


def queue_structure(root: Path) -> dict:
    if (root / QUEUE).exists() or (root / QUEUE).is_symlink():
        raise ValueError("Recheck-exit preserved actual-LR queue is no longer unissued")
    return {"status": "cpu_archive_accepted_unissued", "cpu_receipt_sha256": QUEUE_CPU_SPEC[1]}


def runtime_source() -> bytes:
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Recheck-exit accepted scan-exit preservation parent changed")
    tree = ast.parse(parent.runtime_source())
    replacements = {
        name: globals()[name]
        for name in (
            "OLD_NAMESPACE",
            "WORK",
            "WAITER",
            "CPU_SPEC",
            "WAITER_CPU_SPEC",
            "FAILED_LOG",
            "SEAL",
            "STATUS",
            "PREVIOUS_SEAL_SHA256",
            "RECORDS",
            "SCOPE",
        )
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_transition_predecessor":
                    alias.name = "ns_training_path_scan_exit_predecessor"
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    structure = functions["transition_structure"]
    for node in ast.walk(structure):
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "frozen['commands']":
            if isinstance(node.slice, ast.Constant) and node.slice.value == 23:
                node.slice.value = 8
        if isinstance(node, ast.Compare) and isinstance(node.comparators[0], ast.Constant):
            left = ast.unparse(node.left)
            if left == "len(result['gpu_commands'])":
                node.comparators[0].value = 8
            elif left == "len(result['results'])":
                node.comparators[0].value = 2
            elif left == "waiter_state['observations']":
                node.comparators[0].value = 4
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_decisions":
            node.args[-1] = ast.Constant(8)
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in ("completed_commands", "case_decisions"):
                    value.value = 8 if key.value == "completed_commands" else 2
    index = next(
        index
        for index, node in enumerate(structure.body)
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("decisions['rejected_cases']")
    )
    structure.body[index] = ast.parse(
        """if decisions["rejected_cases"] or decisions["accepted_cases"] != [
    item["name"] for item in frozen["plan"]["trials"][:2]
]:
    raise ValueError("Recheck-exit preserved accepted five/seven-step controls differ")
"""
    ).body[0]
    absent = functions["require_absent"]
    absent.body = [
        node for node in absent.body if not (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "queue")
    ]
    for node in ast.walk(absent):
        if isinstance(node, ast.List):
            node.elts = [
                item
                for item in node.elts
                if ast.unparse(item) != "queue['generation']"
                and not (isinstance(item, ast.Tuple) and "LR_QUEUE" in ast.unparse(item))
            ]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace(
                "OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION", "OUTPUT_MARIN_NS_TRAINING_PATH_SCAN_EXIT"
            )
    for node in ast.walk(functions["accepted_waiter"]):
        if isinstance(node, ast.Constant) and node.value == "test_ns_training_path_transition_followup":
            node.value = "test_ns_training_path_scan_exit_followup"
    evidence = functions["evidence"]
    for node in ast.walk(evidence):
        if isinstance(node, ast.Constant) and node.value == "ns_training_path_transition_followup.py":
            node.value = "ns_training_path_scan_exit_followup.py"
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            if any(isinstance(item, ast.Tuple) and ast.unparse(item.elts[0]) == "WAITER" for item in node.iter.elts):
                node.iter = ast.parse('((WAITER, 4, "observation-*.json"),)', mode="eval").body
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "previous_retained_summary":
                    value.slice = ast.Constant("retained_summary")
                elif isinstance(key, ast.Constant) and key.value == "recovery_retained_summary":
                    value.slice = ast.Constant("recovery_retained_summary")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
