# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the distinct source-bound exit-drain attempt from saved receipts only."""

import ast
import hashlib
from pathlib import Path
from types import ModuleType

import ns_training_path_exit_source as source
import ns_training_path_reporting as original_report

TITLE = "NS Whole-Process Exit-Drain Diagnostic"
CPU_SHA256 = "941990541351584ed2d6b2a4e0302f7e3cb5159e851442f8832ce60ed9c50f69"
SEAL_SHA256 = "07980e69f353192141e52d7dba5e98ea9fda35841f5d1c7be683408fb16237e3"
BINDING_SHA256 = "80032984ac27c3cd79aa72a579e0b213190146e57c8a87c6b5f9fd1e216095c7"
INTENT_SHA256 = "64ec305891cdc9cfa1626db79364bf6f3d1a272e6a251ab9485a71e95114f610"
REPORT_PARENT_SHA256 = "0a6cd49db261f2eff62da8dfe35a6f7c85aaea608b1eb9c8171dfd6034b6d2e9"
CPU_PATH = "run-preparation/ns-training-path-exit-drain-20261119-cpu/pidfd-bound-20261119T1542/receipt.json"


def preparation_module(cpu_path: Path, local_preparer: Path) -> ModuleType:
    """Execute one retained byte buffer only after checking the pinned acceptance digest."""
    cpu = original_report._receipt(cpu_path, CPU_SHA256)
    helper = cpu_path.parent / "tracking" / local_preparer.name
    expected = cpu["helpers_sha256"][local_preparer.name]
    before = original_report._identity(local_preparer)
    payload = local_preparer.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected or original_report._identity(local_preparer) != before:
        raise ValueError("Exit-drain reporting local preparer differs from pinned acceptance")
    original_report._checked(helper, expected, {})
    implementation = ModuleType("exit_drain_saved_cpu_preparation")
    implementation.__file__ = str(local_preparer)
    exec(compile(payload, str(local_preparer), "exec"), implementation.__dict__)
    implementation.__file__ = str(helper)
    return implementation


def require_saved_generations(frozen: dict, intent: dict, state: dict, result: dict | None = None) -> None:
    """Bind mutable owner receipts to the acknowledged controller and direct validator."""
    if state.get("spawn_status") == "acknowledged":
        original_report._generation(
            state["child_generation"],
            frozen["validator_command"],
            state["child_pid"],
            state["child_start_ticks"],
            intent["pid"],
        )
        if state["command"] != frozen["validator_command"]:
            raise ValueError("Exit-drain reporting validator command differs")
    if result is not None:
        if state.get("spawn_status") != "acknowledged":
            raise ValueError("Exit-drain reporting result lacks an acknowledged validator")
        expected = {
            "controller_pid": intent["pid"],
            "controller_start_ticks": intent["start_ticks"],
            "validator_pid": state["child_pid"],
            "validator_start_ticks": state["child_start_ticks"],
        }
        if result["owner"] != expected:
            raise ValueError("Exit-drain reporting retained owner generation differs")


def failed_command_evidence(root: Path, frozen: dict, state: dict, result: dict, records: dict) -> dict:
    """Keep proven completed progress while describing a separately authenticated failure."""
    index = len(result["gpu_commands"])
    if index == len(frozen["commands"]):
        return {}
    item = frozen["plan"]["trials"][index // 3]
    mode = ("continuous", "interrupted", "resumed")[index % 3]
    log = root / "validation/ns_training_path_exit_drain_20261119" / f"{item['name']}_{mode}.log"
    path = log.with_suffix(".execution.json")
    if not path.exists() and not path.is_symlink():
        return {"trial": item["name"], "mode": mode, "detail": "No next-command execution receipt is recorded."}
    entry = original_report._saved(path, records)
    command = frozen["commands"][index]
    if entry["command"] != command or entry["log"] != str(log):
        raise ValueError("Exit-drain reporting failed command identity differs")
    if entry.get("spawn_status") == "acknowledged":
        generation = entry["generation"]
        original_report._generation(
            generation, command, generation["pid"], generation["start_ticks"], state["child_pid"]
        )
    elif entry.get("spawn_status") not in ("starting", "needs_attention"):
        raise ValueError("Exit-drain reporting failed command spawn state differs")
    for name in ("exit_code", "cleanup_exit_code"):
        if name in entry and type(entry[name]) is not int:
            raise ValueError("Exit-drain reporting failed command exit type differs")
    if "log_sha256" in entry and entry["log_sha256"] is not None:
        original_report._checked(log, entry["log_sha256"], records)
    return entry | {"trial": item["name"], "mode": mode}


def failed_command_line(entry: dict, local_time) -> str:
    return (
        f"Next-command receipt: `{entry['trial']}/{entry['mode']}` at {local_time(entry.get('time'))}; "
        f"execution exit `{entry.get('exit_code', 'not recorded')}`, "
        f"cleanup exit `{entry.get('cleanup_exit_code', 'not recorded')}`, "
        f"marked-descendant absence `{entry.get('marked_descendants_absent', 'not recorded')}`. "
        f"{entry.get('detail', 'Excluded from completed trajectory counts.')}"
    )


def report_module(root: Path) -> ModuleType:
    """Derive the accepted receipt projection with a read-only archived CPU entry."""
    path = Path(__file__).with_name("ns_training_path_recovery_reporting.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != REPORT_PARENT_SHA256:
        raise ValueError("Exit-drain reporting parent source changed")
    tree = source.ExitNamespace().visit(ast.parse(payload))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Import) and any(alias.name == "prepare_ns_training_path_exit" for alias in node.names)
        )
    ]
    failure_test = ast.dump(
        ast.parse("result['status'] == 'needs_attention' and len(result['gpu_commands']) < 54").body[0].value
    )
    failure_changes, state_changes, result_changes, line_changes = 0, 0, 0, 0
    # The pinned parent remains intact; replace only the failure projection in this successor.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evidence":
            for child in ast.walk(node):
                if isinstance(child, ast.If) and ast.dump(child.test) == failure_test:
                    child.body = ast.parse(
                        "progress['failed_command'] = failed_command_evidence(root, frozen, state, result, records)"
                    ).body
                    failure_changes += 1
            for child in ast.walk(node):
                if isinstance(child, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "state" for target in child.targets
                ):
                    child.value = ast.Call(
                        func=ast.Name(id="checked_state", ctx=ast.Load()),
                        args=[child.value, ast.Name(id="frozen", ctx=ast.Load()), ast.Name(id="intent", ctx=ast.Load())],
                        keywords=[],
                    )
                    state_changes += 1
                if isinstance(child, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "result" for target in child.targets
                ):
                    child.value = ast.Call(
                        func=ast.Name(id="checked_result", ctx=ast.Load()),
                        args=[
                            child.value,
                            ast.Name(id="frozen", ctx=ast.Load()),
                            ast.Name(id="intent", ctx=ast.Load()),
                            ast.Name(id="state", ctx=ast.Load()),
                        ],
                        keywords=[],
                    )
                    result_changes += 1
        if isinstance(node, ast.FunctionDef) and node.name == "lines":
            for child in ast.walk(node):
                if isinstance(child, ast.If) and isinstance(child.test, ast.Name) and child.test.id == "failed_command":
                    child.body = ast.parse("result.append(failed_command_line(failed_command, local_time))").body
                    line_changes += 1
    if (failure_changes, state_changes, result_changes, line_changes) != (1, 1, 1, 1):
        raise ValueError("Exit-drain reporting successor projection differs from pinned parent")
    values = {
        "TITLE": TITLE,
        "CPU_SHA256": CPU_SHA256,
        "SEAL_SHA256": SEAL_SHA256,
        "BINDING_SHA256": BINDING_SHA256,
        "INTENT_SHA256": INTENT_SHA256,
    }
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id in values:
                node.value = ast.Constant(value=values[target.id])
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for index, comparator in enumerate(node.comparators):
                if isinstance(comparator, ast.Constant) and comparator.value == 44:
                    node.comparators[index] = ast.Constant(value=2)
                elif isinstance(comparator, ast.Constant) and comparator.value == 14:
                    node.comparators[index] = ast.Constant(value=0)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = (
                node.value.replace("103 passing cases", "125 passing cases")
                .replace(
                    "The failed original remains 44/54 commands, 14/18 decisions, six accepted/eight rejected; ",
                    "Both failures remain original44/54commands14/18decisions and recovery2/54commands0/18decisions; ",
                )
                .replace("its separate preservation seal", "the separate preservation seal")
                .replace("Recovery saved status:", "Exit-drain saved status:")
                .replace("Recovery evidence withheld:", "Exit-drain evidence withheld:")
            )
    cpu_path = root / CPU_PATH
    local_preparer = Path(__file__).with_name("prepare_ns_training_path_exit.py")
    implementation = preparation_module(cpu_path, local_preparer)
    report = ModuleType("exit_drain_saved_reporting")
    report.__file__ = str(path)
    report.preparation = implementation
    report.checked_state = checked_state
    report.checked_result = checked_result
    report.failed_command_evidence = failed_command_evidence
    report.failed_command_line = failed_command_line
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), report.__dict__)
    return report


def checked_state(state: dict, frozen: dict, intent: dict) -> dict:
    require_saved_generations(frozen, intent, state)
    return state


def checked_result(result: dict, frozen: dict, intent: dict, state: dict) -> dict:
    require_saved_generations(frozen, intent, state, result)
    return result


def lines(root: Path, local_time) -> list[str]:
    if not (root / CPU_PATH).is_file():
        return []
    try:
        return report_module(root).lines(root, local_time)
    except original_report.EVIDENCE_ERRORS as error:
        return [f"### {TITLE}", "", f"Exit-drain evidence withheld: `{type(error).__name__}: {error}`.", ""]
