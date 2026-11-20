# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the failed whole-inspection attempt before a smaller query critical section."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_serialized_predecessor as parent

PARENT_SOURCE_SHA256 = "041fe8b29cb4b4a25411d2092db4b72a0c20c37061bacf28c6cead387d81fa2d"
OLD_NAMESPACE = "ns-training-path-serialized-20261119"
WORK = "validation/ns_training_path_serialized_20261119"
WAITER = "run-preparation/ns-training-path-serialized-terminal-followup-20261119"
QUEUE = parent.QUEUE
QUEUE_CPU_SPEC = parent.QUEUE_CPU_SPEC
queue_structure = parent.queue_structure
CPU_SPEC = (
    "run-preparation/ns-training-path-serialized-20261119-cpu/reviewed-20261119T2340/receipt.json",
    "293daf7c2a91b5ba2e2081646521f44bd85092d7f89b2a1f1b256a8b4a1aeda2",
    "ns_training_path_serialized_cpu_archive_verified_gpu_unissued",
    141,
)
WAITER_CPU_SPEC = (
    "run-preparation/ns-training-path-serialized-followup-cpu/20261119T2353/receipt.json",
    "632f9296c004585213712462828068aabde2013ed97a3d3555ecde9c62e726ff",
    "ns_training_path_serialized_read_only_followup_cpu_archive_verified",
    24,
)
FAILED_LOG = f"{WORK}/muont_ns_tpu_q5_full_lr_lower_continuous.log"
SEAL = "run-preparation/ns-training-path-query-serialized-predecessor-seal-20261120.json"
STATUS = "failed_whole_inspection_and_waiter_sealed_for_query_serialization_not_numerical_acceptance"
PREVIOUS_SEAL_SHA256 = "dd3d333dfbe00686d8a25da358ae783c11cc0163a679c750338076b5c9a0bdfe"
RECORDS = {
    f"run-preparation/{OLD_NAMESPACE}-binding.json": "5789abe0b6c47fd910917cba35e00b2b2d519e66c867dab2214418327bbddc68",
    f"run-preparation/{OLD_NAMESPACE}-state.json": "b9d3f34834e757d1b4e1d03879a8f1fae4eaed2f511fff4f91a919434d010e39",
    f"run-preparation/validation-{OLD_NAMESPACE}-gpu.json": (
        "2ab91be805a1671762ec1796b7fd56330ba110f4815fa4ce2f152b0332b692c2"
    ),
    f"controllers/{OLD_NAMESPACE}/controller.json": "e782a7fcdc67c137c64c4d65799eb45b56c4800f969d0f1cb0640ed52295bef1",
    f"controllers/{OLD_NAMESPACE}/controller.log": "5fb245284f4524bc196b3273822985f0b3c089b1b9a667db9bde763a1e667d0b",
    f"controllers/{OLD_NAMESPACE}/validator.log": "4fff906e45ff5a83545678dd0f19343171f49233ab1f7184893ace7cd4f1fe05",
    FAILED_LOG: "48b1425c6912c7d38aaa79c7291341ead7399b31b7a8b0a989fcd02e7f8c0ac8",
    FAILED_LOG.replace(".log", ".execution.json"): "324f94c3eccbac0d716b96faf81a92daf2c81b765ab2f8eef6c85aeafc712472",
    f"{WAITER}/intent.json": "7c4115103b9e07596480936a3da39aa9113aa8d179374cffbe9dce2de8b860ad",
    f"{WAITER}/state.json": "ba1fac09df0442cd69a695a4673bc30164234381f018ff4c478a0c82478992de",
    f"{WAITER}/followup.log": "61254590d435003dc37d2dfa02b47aad51b7669393b46f831e7f12322663b961",
    "run-preparation/serialized-failure-restored-observation-20261119T2356.json": (
        "ab7c5b4acaf5b25c76c61b8fd19ada2b0d6b69a3bd4560af54653d7cf95a964b"
    ),
}
SCOPE = (
    "Preserve failed whole-inspection0/54commands0/18decisions, failed continuous command "
    "and original stopped waiter; preserve all older failures/seals and accepted unissued "
    "actual-LR queue. Require exact consumed-owner/marker absence and restored original "
    "eight-A100 keeper. New query-only serialization retains every statement/order, "
    "30-second lock wait, three complete scans, one-second exit proof, numerical readers/"
    "tolerances, trainer/config bytes, compilation and graphs. No complete terminal, "
    "numerical acceptance, production, optimizer quality/efficiency or SwanLab authority."
)


def runtime_source() -> bytes:
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Query predecessor pinned serialized preservation source changed")
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
                if alias.name == "ns_training_path_recheck_exit_predecessor":
                    alias.name = "ns_training_path_serialized_predecessor"
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(functions["transition_structure"]):
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "frozen['commands']":
            if isinstance(node.slice, ast.Constant) and node.slice.value == 1:
                node.slice.value = 0
        if isinstance(node, ast.Compare) and isinstance(node.comparators[0], ast.Constant):
            if ast.unparse(node.left) == "len(result['gpu_commands'])":
                node.comparators[0].value = 0
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_decisions":
            node.args[-1] = ast.Constant(0)
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "completed_commands":
                    value.value = 0
    for function in ("require_absent", "accepted_waiter", "evidence"):
        for node in ast.walk(functions[function]):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = node.value.replace(
                    "OUTPUT_MARIN_NS_TRAINING_PATH_RECHECK_EXIT", "OUTPUT_MARIN_NS_TRAINING_PATH_SERIALIZED"
                ).replace("ns_training_path_recheck_exit_followup", "ns_training_path_serialized_followup")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
