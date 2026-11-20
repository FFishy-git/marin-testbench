# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the failed recheck diagnostic before serialized ownership inspection."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_recheck_exit_predecessor as parent

PARENT_SOURCE_SHA256 = "25c127dea7bc4e5931efcaf81cf15cc429822e62e98902aff3f3603f102f01c5"
OLD_NAMESPACE = "ns-training-path-recheck-exit-20261119"
WORK = "validation/ns_training_path_recheck_exit_20261119"
WAITER = "run-preparation/ns-training-path-recheck-exit-terminal-followup-20261119"
QUEUE = parent.QUEUE
QUEUE_CPU_SPEC = parent.QUEUE_CPU_SPEC
queue_structure = parent.queue_structure
CPU_SPEC = (
    "run-preparation/ns-training-path-recheck-exit-20261119-cpu/closure-reviewed-20261119T2241/receipt.json",
    "5d085c3b8df3750ae7a4b488cc87cdf6b2ec6a4e275767bfc441e4858bce007f",
    "ns_training_path_recheck_exit_cpu_archive_verified_gpu_unissued",
    136,
)
WAITER_CPU_SPEC = (
    "run-preparation/ns-training-path-recheck-exit-followup-cpu/20261119T2253/receipt.json",
    "33528ba3d167f4d59b34d3586cc135fde0adc07e404bd369f61947eac2784ebf",
    "ns_training_path_recheck_exit_read_only_followup_cpu_archive_verified",
    24,
)
FAILED_LOG = f"{WORK}/muont_ns_tpu_q5_full_lr_lower_interrupted.log"
SEAL = "run-preparation/ns-training-path-serialized-predecessor-seal-20261119.json"
STATUS = "failed_recheck_and_waiter_sealed_for_serialized_owner_not_numerical_acceptance"
PREVIOUS_SEAL_SHA256 = "10c189fc543b9d9e1ec4d8d28bf590700948ed108dbd02e8bfd1174fe66193b6"
RECORDS = {
    f"run-preparation/{OLD_NAMESPACE}-binding.json": "ae7757c3010f88724a036d1b9f0d8b5c11b3abb53b97ca2300090f1973d1456f",
    f"run-preparation/{OLD_NAMESPACE}-state.json": "2e5493b27459198b1d933cae63ad7f99662cda7561f7e53b775e23a8f3d3cbaa",
    f"run-preparation/validation-{OLD_NAMESPACE}-gpu.json": (
        "1e3081d43464f066939f1aa2a0b26bc874913f9f67583a55eea5643928ff58d0"
    ),
    f"controllers/{OLD_NAMESPACE}/controller.json": "9502a4d1d2ce65f0ff7e23f1c9bccf62572cd5430b100d76150cad46ceebb607",
    f"controllers/{OLD_NAMESPACE}/controller.log": "15216db0eefb413a9883b010d2d9108633ada4d450d02f30a03f26643a835907",
    f"controllers/{OLD_NAMESPACE}/validator.log": "19b3c0ccb2c651c4c0ab759d14061f4b27f0122cc2a7176014b79d9ebf34268d",
    FAILED_LOG: "0da08ee82c2985017a42a1439971517317f7070561d3443fc862c235cf68c37e",
    FAILED_LOG.replace(".log", ".execution.json"): "4ab5c13013d587a385f9957716eb6f945273b06c85773b7cfd34cb588f27690a",
    f"{WAITER}/intent.json": "84ae50084083fa775b37023824016336473d27e1b3c66b6a42938d808a81e6ac",
    f"{WAITER}/state.json": "0df698ee770a4be94a3ce64dae200f8a80942635738aee240f236c5ee86426aa",
    f"{WAITER}/followup.log": "968adcbcc22eb8572b299a896a2b20bc043a9fd5c2f272dfa43e7c111543f6b6",
    "run-preparation/recheck-exit-waiter-stopped-observation-20261119T2314.json": (
        "dc6cd0ad998d8149f9f37b1ba5befd956befef814921081769ee5e3009c4cbff"
    ),
    "run-preparation/owned-output-helper-churn-20261119T2309.json": (
        "bf078c5eb7dbbd0f3ab467c98af5edb164dabd67cf8bb91ac57fd53962a4de65"
    ),
    "run-preparation/owned-output-helper-churn-archived-repeat-20261119T2316.json": (
        "197a604681ce8000354cfe43e50bea4023feb1169dad528086611b2d9a8f1a94"
    ),
}
SCOPE = (
    "Preserve failed recheck1/54commands0/18decisions, failed interrupted command and "
    "stopped original waiter, all older failures and accepted unissued actual-LR queue. "
    "Require exact consumed-owner/marker absence and original eight-A100 keeper. "
    "A distinct serialized-owner diagnostic must retain all checks, scan attempts, "
    "exit deadlines, numerical readers/tolerances, trainer bytes, compilation and graphs. "
    "No complete terminal, numerical acceptance, production or SwanLab authority."
)


def runtime_source() -> bytes:
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Serialized predecessor pinned recheck source changed")
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
                if alias.name == "ns_training_path_scan_exit_predecessor":
                    alias.name = "ns_training_path_recheck_exit_predecessor"
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    structure = functions["transition_structure"]
    for node in ast.walk(structure):
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "frozen['commands']":
            if isinstance(node.slice, ast.Constant) and node.slice.value == 8:
                node.slice.value = 1
        if isinstance(node, ast.Compare) and isinstance(node.comparators[0], ast.Constant):
            counts = {
                "len(result['gpu_commands'])": 1,
                "len(result['results'])": 0,
                "waiter_state['observations']": 1,
            }
            if ast.unparse(node.left) in counts:
                node.comparators[0].value = counts[ast.unparse(node.left)]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_decisions":
            node.args[-1] = ast.Constant(1)
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in ("completed_commands", "case_decisions"):
                    value.value = 1 if key.value == "completed_commands" else 0
    decision = next(
        index
        for index, node in enumerate(structure.body)
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("decisions['rejected_cases']")
    )
    structure.body[decision] = ast.parse(
        """if decisions["rejected_cases"] or decisions["accepted_cases"]:
    raise ValueError("Serialized predecessor recheck has no completed numerical case")
"""
    ).body[0]
    for node in ast.walk(functions["require_absent"]):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace(
                "OUTPUT_MARIN_NS_TRAINING_PATH_SCAN_EXIT", "OUTPUT_MARIN_NS_TRAINING_PATH_RECHECK_EXIT"
            )
    for node in ast.walk(functions["accepted_waiter"]):
        if isinstance(node, ast.Constant) and node.value == "test_ns_training_path_scan_exit_followup":
            node.value = "test_ns_training_path_recheck_exit_followup"
    for node in ast.walk(functions["evidence"]):
        if isinstance(node, ast.Constant) and node.value == "ns_training_path_scan_exit_followup.py":
            node.value = "ns_training_path_recheck_exit_followup.py"
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            if any(isinstance(item, ast.Tuple) and ast.unparse(item.elts[0]) == "WAITER" for item in node.iter.elts):
                node.iter = ast.parse('((WAITER, 1, "observation-*.json"),)', mode="eval").body
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
