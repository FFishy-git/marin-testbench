# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the failed query's unspawned command before a distinct streaming scan."""

import ast
import hashlib
import json
from pathlib import Path

import ns_training_path_query_serialized_predecessor as parent
import ns_training_path_reporting as reporting
from selected_lr_release_owner import read

original = parent.original
previous = parent

PARENT_SOURCE_SHA256 = "d9954df17ea9aaa918e579877c01a6690802b49b7465b582eb1c2baa4118cf3d"
OLD_NAMESPACE = "ns-training-path-query-serialized-20261120"
WORK = "validation/ns_training_path_query_serialized_20261119"
WAITER = "run-preparation/ns-training-path-query-serialized-terminal-followup-20261120"
FAILED_LOG = f"{WORK}/muont_ns_tpu_q10_full_lr_lower_continuous.log"
SEAL = "run-preparation/ns-training-path-streaming-predecessor-seal-20261120.json"
STATUS = "failed_query_precommand_and_waiter_sealed_for_streaming_scan_not_numerical_acceptance"
PREVIOUS_SEAL_SHA256 = "ca766ffc6298c7bb9284a7d514c44b875d0a80bfb53f9029f7794eae64388373"
CPU_SPEC = (
    "run-preparation/ns-training-path-query-serialized-20261120-cpu/reviewed-20261120T0012/receipt.json",
    "dee0f1f2794237af02cfc0ac0f966a8a30a08f55a489d6ac7ef05291aebf3dc8",
    "ns_training_path_query_serialized_cpu_archive_verified_gpu_unissued",
    142,
)
WAITER_CPU_SPEC = (
    "run-preparation/ns-training-path-query-serialized-followup-cpu/20261120T0026/receipt.json",
    "65e5133be24ea2d2ab1fbb4155c7c25d1c34d9ddc06ce08020694bde6d362e4d",
    "ns_training_path_query_serialized_read_only_followup_cpu_archive_verified",
    24,
)
RECORDS = {
    f"run-preparation/{OLD_NAMESPACE}-binding.json": "8e6760d8a0a679e057ad7434df2e969925f055c57912e271b42ff45fa1cfe24e",
    f"run-preparation/{OLD_NAMESPACE}-state.json": "e7529dcdf00438ca777f3f74b4607345290f5e20b014ff2050db734c2417fcf7",
    f"run-preparation/validation-{OLD_NAMESPACE}-gpu.json": (
        "83d3083f314c3f9d7e173d69e0a93af243b1669eb6a923105a9aeecd8bf891a3"
    ),
    f"controllers/{OLD_NAMESPACE}/controller.json": "43e810371a8441b56f75c740ffc8ec126473acaaf360873647dd9d9aaffef3fe",
    f"controllers/{OLD_NAMESPACE}/controller.log": "182c0398a352b281800780381b3475384c7c4da87a93c3caacbe8c50576052ef",
    f"controllers/{OLD_NAMESPACE}/validator.log": "699de40330dc96c3eba2753343f79eae6534458b8299903052c9818c80d7ce72",
    f"{WAITER}/intent.json": "3679f0cbca16b619995e76b70950951348db6ff4bb9592f435ccbc329f391666",
    f"{WAITER}/state.json": "964c69e153290f17ab939963ca18de9ca5192c8e57f31a6a2eb0f73151382559",
    f"{WAITER}/followup.log": "1455e5f9e47af533425542a5302d7419ae66ec615e12b59616a2da68adb68d08",
    "run-preparation/query-and-waiter-stopped-observation-20261120T0132.json": (
        "34bdb242d4697be517920517bfa81f0d2dddf774bf98ad743a8bfb386ee0286f"
    ),
    "run-preparation/unmarked-churn-study-20261120T0124.json": (
        "6784a60834a5be390d5036166ec92474b4f5b90f99b3d4923e95aec6ea00ae3c"
    ),
}
QUEUE = parent.QUEUE
QUEUE_CPU_SPEC = parent.QUEUE_CPU_SPEC
queue_structure = parent.queue_structure
SCOPE = (
    "Preserve failed query6/54commands2/18accepted MuonT controls, unspawned ten-step "
    "continuous command and original stopped four-observation waiter. Require all remaining "
    "command outputs/logs/sidecars absent; never invent a spawned-failure receipt. Retain "
    "all prior failures/rejections/seals, checkpoint digests and metadata without tensor "
    "replay, accepted unissued actual-LR preparations and queues, original exact eight-A100 "
    "keeper and consumed-owner/marker absence. Distinct streaming observation order changes "
    "no namespace filter, complete-scan statement/recheck, three attempts, one-second exit "
    "proof, query lock, numerical reader/tolerance, trainer/config bytes, compilation or "
    "CUDA graph. No complete terminal, numerical waiver, GPU/production, optimizer "
    "quality/efficiency or SwanLab authority."
)


def require_unspawned(root: Path, frozen: dict) -> None:
    """Reject evidence for any command after the six acknowledged completions."""
    for index in range(6, len(frozen["commands"])):
        trial = frozen["plan"]["trials"][index // 3]
        mode = original.MODES[index % 3]
        output = Path(trial["continuous"] if mode == "continuous" else trial["resumed"])
        log = root / WORK / f"{trial['name']}_{mode}.log"
        if any(path.exists() or path.is_symlink() for path in (output, log, log.with_suffix(".execution.json"))):
            raise ValueError("Streaming predecessor has evidence for an unspawned query command")


def transition_structure(root: Path, frozen: dict, intent: dict, state: dict, result: dict) -> dict:
    """Retain accepted controls and the pre-command refusal without fabricating a child."""
    waiter = read(root / WAITER / "intent.json")
    waiter_state = read(root / WAITER / "state.json")
    digest = RECORDS[f"run-preparation/{OLD_NAMESPACE}-binding.json"]
    if (
        frozen["root"] != str(root)
        or len(frozen["commands"]) != 54
        or len(frozen["plan"]["trials"]) != 18
        or frozen["commands"] != frozen["plan"]["commands"]
        or state["status"] != "needs_attention"
        or state["error"] != "RuntimeError: NS training-path bounded validator failed"
        or any(type(state[key]) is not int or state[key] != 1 for key in ("exit_code", "cleanup_exit_code"))
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or any(
            state[key] is not True
            for key in ("marked_descendants_absent", "keeper_release_attempted", "keeper_release_complete")
        )
        or any(key in state for key in ("cleanup_error", "deferred_signals"))
        or intent["status"] != "acknowledged"
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or state["command"] != frozen["validator_command"]
        or result["status"] != "needs_attention"
        or len(result["gpu_commands"]) != 6
        or len(result["results"]) != 2
        or result["trials"] != frozen["plan"]["trials"]
        or result["owner"]
        != {
            "controller_pid": intent["pid"],
            "controller_start_ticks": intent["start_ticks"],
            "validator_pid": state["child_pid"],
            "validator_start_ticks": state["child_start_ticks"],
        }
        or any(item["binding_sha256"] != digest for item in (state, intent, result, waiter, waiter_state))
        or waiter_state["status"] != "needs_attention"
        or waiter_state["error"]
        != "RuntimeError: Training-path original owners exited without complete cleanup/restoration"
        or type(waiter_state["observations"]) is not int
        or waiter_state["observations"] != 4
        or waiter_state["pid"] != waiter["pid"]
        or waiter_state["start_ticks"] != waiter["start_ticks"]
        or waiter["status"] != "acknowledged"
        or any(item["cpu_receipt_sha256"] != WAITER_CPU_SPEC[1] for item in (waiter, waiter_state))
        or waiter_state["runner_sha256"] != waiter["runner_sha256"]
        or any(key in waiter_state for key in ("terminal_generation", "terminal_sha256"))
    ):
        raise ValueError("Streaming predecessor failed query coverage, owner or waiter differs")
    error, separator, payload = result["error"].partition(" ownership_evidence=")
    attribution = json.loads(payload) if separator else {}
    if (
        not error.startswith(
            "OwnershipObservationError: Marked process scan remained unstable during handoff after 3 attempts:"
        )
        or attribution.get("stage") != "complete_scan_exhausted"
        or attribution.get("schema") != "owned_output_failure_attribution_v1"
        or attribution.get("snapshots", {}).get("previously_confirmed_marked")
        != {str(pid): False for pid in (3730921, 3730943, 3730947, 3730948, 3730949, 3730952)}
    ):
        raise ValueError("Streaming predecessor original pre-command scan attribution differs")
    reporting._generation(intent["generation"], frozen["controller_command"], intent["pid"], intent["start_ticks"])
    reporting._generation(
        state["child_generation"],
        frozen["validator_command"],
        state["child_pid"],
        state["child_start_ticks"],
        intent["pid"],
    )
    reporting._generation(waiter["generation"], waiter["command"], waiter["pid"], waiter["start_ticks"])
    records = {}
    for index, entry in enumerate(result["gpu_commands"]):
        trial, mode = frozen["plan"]["trials"][index // 3], original.MODES[index % 3]
        if entry["trial"] != trial["name"] or entry["mode"] != mode:
            raise ValueError("Streaming predecessor completed command order differs")
        reporting._command(
            entry,
            frozen["commands"][index],
            root / WORK / f"{trial['name']}_{mode}.log",
            state["child_pid"],
            records,
        )
    decisions = reporting._decisions(result, frozen["plan"], 6)
    if decisions["rejected_cases"] or decisions["accepted_cases"] != [
        trial["name"] for trial in frozen["plan"]["trials"][:2]
    ]:
        raise ValueError("Streaming predecessor accepted five/seven-step controls differ")
    require_unspawned(root, frozen)
    return {
        "completed_commands": 6,
        "planned_commands": 54,
        "case_decisions": 2,
        "planned_cases": 18,
        "accepted_cases": decisions["accepted_cases"],
        "rejected_cases": decisions["rejected_cases"],
        "decisions": result["results"],
        "coverage_complete": False,
        "numerically_accepted": False,
        "failed_command": None,
        "precommand_failure": {
            "command": frozen["commands"][6],
            "spawn_status": "not_spawned",
            "error": result["error"],
            "time": result["time"],
            "stage": attribution["stage"],
            "execution_receipt": None,
        },
        "waiter": waiter_state,
        "queue": queue_structure(root),
    }


def require_absent(root: Path, frozen: dict, intent: dict, state: dict, result: dict) -> None:
    waiter = read(root / WAITER / "intent.json")
    handles = [intent["generation"], state["child_generation"], waiter["generation"]]
    handles += [entry["generation"] for entry in result["gpu_commands"]]
    if any((previous.handoff.PROC_ROOT / str(item["pid"])).exists() for item in handles):
        raise RuntimeError("Streaming predecessor consumed owner or completed launcher remains")
    permitted = {int(pid) for pid in state["restored_keeper"]["generation"]}
    markers = [
        ("OUTPUT_MARIN_NS_TRAINING_PATH_QUERY_SERIALIZED_CONTROLLER", intent["marker"], permitted),
        ("OUTPUT_MARIN_NS_TRAINING_PATH_QUERY_SERIALIZED_VALIDATOR", state["child_marker"], set()),
        ("OUTPUT_MARIN_NS_TRAINING_PATH_QUERY_SERIALIZED_FOLLOWUP", waiter["marker"], set()),
    ]
    markers += [
        ("OUTPUT_MARIN_NS_TRAINING_PATH_QUERY_SERIALIZED_COMMAND", entry["marker"], set())
        for entry in result["gpu_commands"]
    ]
    for name, value, allowed in markers:
        if not set(previous.handoff.marker_processes(name, value)).issubset(allowed):
            raise RuntimeError("Streaming predecessor marked descendant remains")
    directory = Path(frozen["helpers"]).parent
    if any(
        path.exists() or path.is_symlink()
        for path in (directory / "terminal-intent.json", directory / "terminal-outcome.json")
    ):
        raise ValueError("Streaming predecessor unexpectedly claimed a complete terminal")
    require_unspawned(root, frozen)


def runtime_source() -> bytes:
    path = Path(parent.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Streaming predecessor pinned query preservation source changed")
    tree = ast.parse(payload)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "exec"
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
            "FAILED_LOG",
            "SEAL",
            "STATUS",
            "PREVIOUS_SEAL_SHA256",
            "CPU_SPEC",
            "WAITER_CPU_SPEC",
            "RECORDS",
            "SCOPE",
        )
    }
    own = ast.parse(Path(__file__).read_bytes())
    functions = {
        node.name: node
        for node in own.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("transition_structure", "require_absent", "require_unspawned")
    }
    boundary = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == "transition_structure"
    )
    tree.body.insert(boundary, functions.pop("require_unspawned"))
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_serialized_predecessor":
                    alias.name = "ns_training_path_query_serialized_predecessor"
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            tree.body[index] = functions[node.name]
        if isinstance(node, ast.FunctionDef) and node.name in ("evidence", "accepted_waiter"):
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    child.value = child.value.replace(
                        "ns_training_path_serialized_followup", "ns_training_path_query_serialized_followup"
                    )
                if isinstance(child, ast.For) and isinstance(child.iter, ast.Tuple):
                    if any(
                        isinstance(item, ast.Tuple) and ast.unparse(item.elts[0]) == "WAITER" for item in child.iter.elts
                    ):
                        child.iter = ast.parse('((WAITER, 4, "observation-*.json"),)', mode="eval").body
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
