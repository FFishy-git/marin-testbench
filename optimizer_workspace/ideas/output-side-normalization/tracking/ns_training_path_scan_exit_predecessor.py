# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the failed transition and both stopped follow-ups before a fresh scan-exit diagnostic."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_transition_predecessor as parent
from selected_lr_release_owner import read

PARENT_SHA256 = "f79eedf305eff558c63df5d98eabe54806ddae63cc0e248e16f166bfb31c1120"
OLD_NAMESPACE = "ns-training-path-exit-transition-20261119"
WORK = "validation/ns_training_path_transition_20261119"
WAITER = "run-preparation/ns-training-path-exit-transition-terminal-followup-20261119"
QUEUE = "run-preparation/ns-training-path-transition-actual-lr-queue-20261119"
CPU_SPEC = (
    "run-preparation/ns-training-path-exit-transition-20261119-cpu/closure-corrected-20261119T1700/receipt.json",
    "ad18edf9b4f230e78bd7eef2a97f44bde307a1c524192d37f23491257b6fc392",
    "ns_training_path_exit_transition_cpu_archive_verified_gpu_unissued",
    129,
)
WAITER_CPU_SPEC = (
    "run-preparation/ns-training-path-exit-transition-followup-cpu/20261119T1718/receipt.json",
    "6a0994915727bf06c606ff0a2bfcece8e4070045ac1f0dcb5f2e79d6b3041906",
    "ns_training_path_transition_read_only_followup_cpu_archive_verified",
    24,
)
QUEUE_CPU_SPEC = (
    "run-preparation/ns-training-path-transition-actual-lr-queue-cpu/reviewed-20261119T1855/receipt.json",
    "ed283127735f2c23648fb14b8342aad46ffb112aabc29dd059e221562c37dac5",
    "ns_training_path_transition_actual_lr_queue_cpu_archive_verified_gpu_unissued",
    47,
)
FAILED_LOG = f"{WORK}/outer_leftdiag_fused_ns_tpu_q7_full_lr_lower_resumed.log"
SEAL = "run-preparation/ns-training-path-scan-exit-predecessor-seal-20261119.json"
STATUS = "failed_transition_and_followups_sealed_for_distinct_scan_exit_not_numerical_acceptance"
PREVIOUS_SEAL_SHA256 = "070227bc90c040368f7ce2a28ef897425a1f24639d870377954827dbe20f9b49"
RECORDS = {
    f"run-preparation/{OLD_NAMESPACE}-binding.json": "18c9b21d16ff817059177e6746c3ae2246d39232396220c2b1deec805a33ca3d",
    f"run-preparation/{OLD_NAMESPACE}-state.json": "89544593f96b24a8188138a585d8d14821ed548b4ac6360ddc5bcd27c62c79ce",
    f"run-preparation/validation-{OLD_NAMESPACE}-gpu.json": (
        "e3cf909c5e9fa728389f554640125154efa36633da34dbc8809fbe247a6d00c0"
    ),
    f"controllers/{OLD_NAMESPACE}/controller.json": "42a901376f00df7b1672ab5d9dad610f669097ca0e48c16690ea4e48b2fb54e1",
    f"controllers/{OLD_NAMESPACE}/controller.log": "24128da0d0d99641a6041c575c56ac8b4bdc9ab07a3b771a25b6a31095d59fb3",
    f"controllers/{OLD_NAMESPACE}/validator.log": "ed7f3b6e33e2671ea92a79747b363d94d4ef70a196f7b891482eab0abf1374b4",
    FAILED_LOG: "9becaaf5b84ae61267e1043df435a714fa63d1b18070eb89170a4e02cf783895",
    FAILED_LOG.replace(".log", ".execution.json"): "34de7bc7a30cb84ec616b57918d080c55f311576eb392489b16c2f38e3f8f76e",
    f"{WAITER}/intent.json": "1641a7cfd73b11d71a91220c7aab6b9aa6e9601d3804237ce4d89abaef0aae0d",
    f"{WAITER}/state.json": "6b43259b415b930889a20a9d9ce6fd960de292abc2343f865d69c7547f7a1159",
    f"{WAITER}/followup.log": "983ec81174aaadf8a45e9a3153c1574e038493e84a1bee645a5b6a01862ad50e",
    f"{QUEUE}/intent.json": "95a67ee0e8cf787344215cfe50413379aac84565573a12de86ee067a3575bd38",
    f"{QUEUE}/state.json": "7051ca0c7b5ea41a8e41b9d283402b3294909ffa741812e703e6041a6d665cb0",
    f"{QUEUE}/queue.log": "9591a0a7324aaea2899fe38dac1bcc05e30e02488cb1ab3a20f0c54472392b9a",
    "run-preparation/endeavor-transition-failure-restoration-observation-20261119T2006.json": (
        "fbdfac0fd043478a8f91fe92f8fec6e98ff274fbaf21839aa826871477ae9761"
    ),
}
SCOPE = (
    "Preserve transition23/54commands7/18decisions, six accepted MuonT controls and fused "
    "BF16/five-step rejection, failed seven-step resumed command, stopped11-observation "
    "terminal waiter and0-phase actual-LR queue, plus all older failures and seals. "
    "Authenticate exact absent owners/markers and restored eight-GPU keeper. Retain "
    "checkpoint digests and current identities without fresh large-payload hashing or "
    "numerical reconstruction. No complete terminal, numerical waiver or GPU/production "
    "authority; only a separately accepted distinct ownership-only diagnostic may proceed."
)


def queue_structure(root: Path) -> dict:
    intent, state = (read(root / QUEUE / name) for name in ("intent.json", "state.json"))
    if (
        intent["status"] != "acknowledged"
        or state["status"] != "needs_attention"
        or state["phases"] != []
        or state["original_observations"] != 4
        or not state["error"].startswith("RuntimeError: Transition-LR external evidence reader refused evidence")
        or "FileNotFoundError" not in state["error"]
        or intent["cpu_receipt_sha256"] != QUEUE_CPU_SPEC[1]
        or state["cpu_receipt_sha256"] != QUEUE_CPU_SPEC[1]
        or state["runner_sha256"] != intent["runner_sha256"]
        or state["pid"] != intent["pid"]
        or state["start_ticks"] != intent["start_ticks"]
        or any(key in state for key in ("controller_generation", "terminal_generation", "terminal_sha256"))
    ):
        raise ValueError("Scan-exit preserved failed actual-LR queue differs")
    parent.reporting._generation(intent["generation"], intent["command"], intent["pid"], intent["start_ticks"])
    return state


def runtime_source() -> bytes:
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit accepted preservation parent changed")
    tree = ast.parse(payload)
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
                if alias.name == "ns_training_path_exit_predecessor":
                    alias.name = "ns_training_path_transition_predecessor"
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
    structure = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "transition_structure"
    )
    for node in ast.walk(structure):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == 6:
            if ast.unparse(node.value) == "frozen['commands']":
                node.slice.value = 23
        if isinstance(node, ast.Compare) and isinstance(node.comparators[0], ast.Constant):
            left = ast.unparse(node.left)
            if left == "len(result['gpu_commands'])":
                node.comparators[0].value = 23
            elif left == "len(result['results'])":
                node.comparators[0].value = 7
            elif left == "waiter_state['observations']":
                node.comparators[0].value = 11
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_decisions":
            node.args[-1] = ast.Constant(23)
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in ("completed_commands", "case_decisions"):
                    value.value = 23 if key.value == "completed_commands" else 7
    decision_check = next(
        index
        for index, node in enumerate(structure.body)
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("decisions['rejected_cases']")
    )
    structure.body[decision_check] = ast.parse(
        """if decisions["rejected_cases"] != [frozen["plan"]["trials"][6]["name"]] or decisions["accepted_cases"] != [
    item["name"] for item in frozen["plan"]["trials"][:6]
]:
    raise ValueError("Scan-exit preserved six accepted controls and fused rejection differ")
"""
    ).body[0]
    structure.body.insert(len(structure.body) - 1, ast.parse("queue_structure(root)").body[0])
    returned = structure.body[-1].value
    returned.keys.extend([ast.Constant("rejected_cases"), ast.Constant("queue")])
    returned.values.extend(
        [
            ast.parse('decisions["rejected_cases"]', mode="eval").body,
            ast.parse("queue_structure(root)", mode="eval").body,
        ]
    )
    absent = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "require_absent")
    for node in ast.walk(absent):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace(
                "OUTPUT_MARIN_NS_TRAINING_PATH_EXIT", "OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION"
            )
    absent.body.insert(2, ast.parse('queue = read(root / QUEUE / "intent.json")').body[0])
    assignment = next(
        node for node in absent.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "handles"
    )
    assignment.value.elts.append(ast.parse('queue["generation"]', mode="eval").body)
    marker_assignment = next(
        node for node in absent.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "markers"
    )
    marker_assignment.value.elts.append(
        ast.parse('("OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION_LR_QUEUE", queue["marker"], set())', mode="eval").body
    )
    waiter = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "accepted_waiter")
    for node in ast.walk(waiter):
        if isinstance(node, ast.Constant) and node.value == "test_ns_training_path_exit_followup":
            node.value = "test_ns_training_path_transition_followup"
    evidence = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evidence")
    for node in ast.walk(evidence):
        if isinstance(node, ast.Constant) and node.value == "ns_training_path_exit_followup.py":
            node.value = "ns_training_path_transition_followup.py"
    evidence.body.insert(3, ast.parse("queue_cpu = accepted_archive(root, QUEUE_CPU_SPEC)").body[0])
    evidence.body.insert(
        len(evidence.body) - 2,
        ast.parse(
            """for relative, count, pattern in (
    (WAITER, 11, "observation-*.json"), (QUEUE, 4, "original-observation-*.json")
):
    paths = sorted((root / relative).glob(pattern))
    if len(paths) != count:
        raise ValueError("Scan-exit retained follow-up observation coverage differs")
    for path in paths:
        records[str(path.relative_to(root))] = checked(path, sha256(path))
"""
        ).body[0],
    )
    # Add both loops rather than relying on the template's existing waiter inventory loop.
    extra = ast.parse(
        """for path, digest in (
    (root / QUEUE_CPU_SPEC[0], QUEUE_CPU_SPEC[1]),
    *((Path(queue_cpu[name]), queue_cpu[f"{name}_sha256"]) for name in ("archive", "tests", "log")),
):
    records[str(path.relative_to(root))] = checked(path, digest)
for name, digest in queue_cpu["inventory_sha256"].items():
    path = Path(queue_cpu["helpers"]).parent / name
    if name.startswith("artifacts/external-sources/"):
        path = Path(queue_cpu["helpers"]).parent / "artifacts/pinned-sources-materialized" / name.removeprefix(
            "artifacts/external-sources/"
        )
    records[str(path.relative_to(root))] = checked(path, digest)
"""
    ).body
    identity_check = next(
        index
        for index, node in enumerate(evidence.body)
        if isinstance(node, ast.If) and "records | payloads" in ast.unparse(node.test)
    )
    evidence.body[identity_check:identity_check] = extra
    output = evidence.body[-1].value
    for key, value in zip(output.keys, output.values, strict=True):
        if isinstance(key, ast.Constant) and key.value == "recovery_retained_summary":
            value.slice = ast.Constant("recovery_retained_summary")
    output.keys.append(ast.Constant("previous_retained_summary"))
    output.values.append(ast.parse('older["retained_summary"]', mode="eval").body)
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
