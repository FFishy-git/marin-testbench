# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate saved observations of the distinct exit-drain terminal waiter."""

import ast
import hashlib
import json
from pathlib import Path
from types import ModuleType

import ns_training_path_exit_reporting as exit_reporting
import ns_training_path_reporting as original
import prepare_ns_training_path_exit_followup as preparation
from prepare_ns_training_path_exit_followup import WaiterNamespace

TITLE = "NS Exit-Drain Independent Terminal Waiter"
CPU = "run-preparation/ns-training-path-exit-drain-followup-cpu/20261119T1620/receipt.json"
CPU_SHA256 = "5d94403384b5f6bf9d79dc1438cfec8ae025de267ee76b971345c1cc37146462"
INTENT_SHA256: str | None = "457a7e76095697d1c11a26e5b870b0c1888549a46da4cd255e3ee2aaae4d2c88"
RUNNER_SHA256 = "528c58152a93c6728b9b9d559dd400c84353b36c2dc1bd0941b490f9c9f5f190"
PARENT_SHA256 = "89ece70437d29bc03b85c05fb60b870b2933b213295dfc156dbffb5d81ccc1db"


def evidence(root: Path) -> dict:
    """Use the pinned saved-receipt predicate without importing a mutable execution owner."""
    intent_path = root / preparation.WORK / "intent.json"
    if not intent_path.exists() and not intent_path.is_symlink():
        return {"status": "not_recorded"}
    if INTENT_SHA256 is None:
        raise ValueError("Exit-drain waiter reporting acknowledgment is not pinned")
    path = Path(original.__file__)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Exit-drain waiter reporting parent changed")
    tree = ast.parse(payload)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_followup")
    function = WaiterNamespace().visit(function)
    function.name = "saved_followup"
    for node in ast.walk(function):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace(
                "ns_training_path_read_only_followup_cpu_archive_verified",
                "ns_training_path_exit_drain_read_only_followup_cpu_archive_verified",
            )
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "test_ns_training_path_exit_followup":
                    if not isinstance(value, ast.Constant) or value.value != 19:
                        raise ValueError("Exit-drain waiter parent test projection differs")
                    value.value = 24
    cpu = original._receipt(root / CPU, CPU_SHA256)
    if cpu["helpers_sha256"][preparation.RUNNER] != RUNNER_SHA256:
        raise ValueError("Exit-drain waiter accepted runner differs")
    binding = original._receipt(
        root / exit_reporting.source.LITERALS["run-preparation/ns-training-path-binding-20261119.json"],
        exit_reporting.BINDING_SHA256,
    )
    contract = ModuleType("exit_drain_saved_followup_contract")
    contract.__dict__.update(original.contract.__dict__)
    contract.locations = lambda base: (
        base / exit_reporting.source.LITERALS["run-preparation/ns-training-path-binding-20261119.json"],
        base / "controllers" / exit_reporting.source.NAMESPACE / "tracking",
        base / exit_reporting.source.LITERALS["run-preparation/ns-training-path-state-20261119.json"],
        base / "validation/ns_training_path_exit_drain_20261119",
        base / exit_reporting.source.LITERALS["run-preparation/validation-ns-training-path-gpu-20261119.json"],
    )
    module = ModuleType("exit_drain_saved_followup_projection")
    module.__dict__.update(original.__dict__)
    module.__dict__.update(
        contract=contract,
        FOLLOWUP_WORK=preparation.WORK,
        FOLLOWUP_CPU=CPU,
        FOLLOWUP_CPU_SHA256=CPU_SHA256,
        FOLLOWUP_INTENT_SHA256=INTENT_SHA256,
        BINDING_SHA256=exit_reporting.BINDING_SHA256,
        INTENT_SHA256=exit_reporting.INTENT_SHA256,
        ISSUED_ARCHIVE_SHA256=preparation.ARCHIVE_SHA256,
        FOLLOWUP_ENVIRONMENT=preparation.ENVIRONMENT,
        FOLLOWUP_SCOPE=preparation.SCOPE,
    )
    records = {}
    intent = original._receipt(intent_path, INTENT_SHA256)
    state = original._saved(contract.locations(root)[2], records)
    waiter_state = original._saved(root / preparation.WORK / "state.json", records)
    progress = {"terminal": {}}
    if waiter_state["status"] == "terminal_reconstructed":
        projection = exit_reporting.report_module(root)
        complete = projection.evidence(root)["progress"]["terminal"]
        if complete["status"] != projection.TERMINAL_STATUS:
            raise ValueError("Exit-drain waiter completed without the independent canonical terminal")
        generation = {
            "binding_sha256": exit_reporting.BINDING_SHA256,
            "controller_pid": state["controller_pid"],
            "controller_start_ticks": state["controller_start_ticks"],
            "controller_marker": original._receipt(
                Path(binding["helpers"]).parent / "controller.json", exit_reporting.INTENT_SHA256
            )["marker"],
        }
        token = hashlib.sha256(json.dumps(generation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        terminal = root / "run-preparation" / f"{exit_reporting.source.NAMESPACE}-terminal-{token}.json"
        progress["terminal"] = {
            "status": "complete",
            "receipt": str(terminal),
            "receipt_sha256": original._sha256(terminal),
            **{name: complete[name] for name in ("accepted_cases", "rejected_cases", "all_cases_accepted")},
        }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"),
        module.__dict__,
    )
    saved = module.saved_followup(root, binding, progress, records)
    if intent["runner_sha256"] != RUNNER_SHA256:
        raise ValueError("Exit-drain waiter acknowledgment runner differs")
    if any(original._sha256(path) != digest for path, (digest, _) in records.items()):
        raise ValueError("Exit-drain waiter saved evidence changed during reporting")
    return saved


def lines(root: Path, local_time) -> list[str]:
    if not (root / CPU).is_file():
        return []
    try:
        saved = evidence(root)
    except original.EVIDENCE_ERRORS as failure:
        return [f"### {TITLE}", "", f"Exit-drain waiter evidence withheld: `{type(failure).__name__}: {failure}`.", ""]
    return [
        f"### {TITLE}",
        "",
        f"Saved waiter status `{saved['status']}`; original-owner checks every900seconds.",
        f"CPU/source archive:24 passing cases,169 helpers; `{CPU}`.",
        f"Saved waiter PID/start `{saved.get('pid', 'not recorded')}/{saved.get('start_ticks', 'not recorded')}`; "
        f"observation time {local_time(saved.get('time'))}.",
        f"Saved original owner states: `{saved.get('saved_original_owners', {})}`.",
        (
            f"Retained waiter error: `{saved.get('error')}`."
            if saved.get("error")
            else "Independent terminal completion is required; saved observations do not prove current liveness."
        ),
        "No keeper transfer, new GPU command, numerical waiver, production or SwanLab authority.",
        "",
    ]
