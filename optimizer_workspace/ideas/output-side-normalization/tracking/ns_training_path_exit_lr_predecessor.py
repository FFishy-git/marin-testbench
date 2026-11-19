# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seal the distinct exit-drain terminal, preserving failed lineage without tensor replay."""

import fcntl
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

NAMESPACE = "ns-training-path-exit-drain-actual-lr-predecessor-20261119"
SEAL = f"run-preparation/{NAMESPACE}-seal.json"
STATUS = "exit_drain_terminal_sealed_actual_lr_owner_unissued"
PYTHON = "/opt/conda/envs/torch-base/bin/python"
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONFAULTHANDLER": "1"}
READ_TIMEOUT = 900
EXIT_CPU = "run-preparation/ns-training-path-exit-drain-20261119-cpu/pidfd-bound-20261119T1542/receipt.json"
EXIT_CPU_SHA256 = "941990541351584ed2d6b2a4e0302f7e3cb5159e851442f8832ce60ed9c50f69"
FOLLOWUP_CPU = "run-preparation/ns-training-path-exit-drain-followup-cpu/20261119T1620/receipt.json"
FOLLOWUP_CPU_SHA256 = "5d94403384b5f6bf9d79dc1438cfec8ae025de267ee76b971345c1cc37146462"
FOLLOWUP_SHA256 = "528c58152a93c6728b9b9d559dd400c84353b36c2dc1bd0941b490f9c9f5f190"
FOLLOWUP_WORK = "run-preparation/ns-training-path-exit-drain-terminal-followup-20261119"
FOLLOWUP_INTENT_SHA256: str | None = "457a7e76095697d1c11a26e5b870b0c1888549a46da4cd255e3ee2aaae4d2c88"
BOOTSTRAP = "run-preparation/ns-training-path-exit-drain-followup-bootstrap-20261119T1620/source.json"
BOOTSTRAP_SHA256 = "5a8cbfbda91a385777c290409dc780aa7a9ef6e4a808f0a68752137863dc4bfe"
REPORTER_SHA256 = "56858d0deab1af2c76cefdac6d462eaf1c597b237d636788e1b4215e914f87b8"
REPORT_SOURCES = {
    "ns_training_path_reporting.py": "89ece70437d29bc03b85c05fb60b870b2933b213295dfc156dbffb5d81ccc1db",
    "ns_training_path_recovery_reporting.py": "0a6cd49db261f2eff62da8dfe35a6f7c85aaea608b1eb9c8171dfd6034b6d2e9",
}
BINDING = "run-preparation/ns-training-path-exit-drain-20261119-binding.json"
BINDING_SHA256 = "80032984ac27c3cd79aa72a579e0b213190146e57c8a87c6b5f9fd1e216095c7"
INTENT_SHA256 = "64ec305891cdc9cfa1626db79364bf6f3d1a272e6a251ab9485a71e95114f610"
FAILED_SEAL = "run-preparation/ns-training-path-exit-drain-predecessor-seal-20261119.json"
FAILED_SEAL_SHA256 = "07980e69f353192141e52d7dba5e98ea9fda35841f5d1c7be683408fb16237e3"
MANAGER_LOCK = Path("/root/dl/runtime/gpu_resource_manager.lock")
MODES = ("continuous", "interrupted", "resumed")
CHECKPOINTS = {"continuous": {"22", "23", "24"}, "interrupted": {"22"}, "resumed": {"22", "23", "24"}}
SCOPE = (
    "Distinct file-only exit-drain54-command18-decision terminal handoff to an unissued actual-LR owner. "
    "Require the pinned independent waiter/terminal, exact controller/validator/waiter/terminal-child "
    "and marked descendant absence, clean restoration and exact current keeper. Preserve failed original "
    "and recovery lineage, all accepted and rejected decisions, retained checkpoint digests/current "
    "identities without fresh tensor hashing. Separate source archives remain unmerged; no owner, GPU "
    "transfer, launch, numerical waiver, production, quality/efficiency or SwanLab authority."
)

# Each child imports just one pinned closure. In particular, it never imports the old LR handoff.
OBSERVER_PROGRAM = """import json,sys
from pathlib import Path
import ns_training_path_exit_followup as followup
root,cpu_path,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
cpu=followup.accepted(root,cpu_path)
frozen=followup.frozen(root)
binding,helpers,state_path,work,result_path=followup.contract.locations(root)
intent=followup.read(helpers.parent/'controller.json')
state=followup.read(state_path)
result=followup.read(result_path)
waiter=followup.read(root/followup.WORK/'intent.json')
waiter_state=followup.read(root/followup.WORK/'state.json')
contexts=[
 ('controller',intent['generation'],helpers,followup.contract.CONTROLLER_MARKER,intent['marker'],
  frozen['controller_environment']),
 ('validator',state['child_generation'],Path(frozen['candidate']),followup.contract.CHILD_MARKER,
  state['child_marker'],frozen['environment']),
 ('terminal_waiter',waiter['generation'],root/followup.WORK,followup.MARKER,waiter['marker'],followup.ENVIRONMENT)]
if waiter_state.get('terminal_generation'):
 contexts.append(('terminal_child',waiter_state['terminal_generation'],helpers,followup.CHILD_MARKER,
  waiter_state['terminal_marker'],followup.ENVIRONMENT))
owners={name:followup.handle(generation,cwd,marker,value,env)
 for name,generation,cwd,marker,value,env in contexts}
marks=[(marker,value,set()) for _,_,_,marker,value,_ in contexts]
permitted={int(pid) for pid in state.get('restored_keeper',{}).get('generation',{})}
marks[0]=(followup.contract.CONTROLLER_MARKER,intent['marker'],permitted)
for entry in result.get('gpu_commands',[]):
 owners['command_'+str(entry['generation']['pid'])]=followup.handle(entry['generation'],
  Path(frozen['candidate']),followup.contract.COMMAND_MARKER,entry['marker'],frozen['environment'])
 marks.append((followup.contract.COMMAND_MARKER,entry['marker'],set()))
for pid,generation in waiter_state.get('terminal_generations',{}).items():
 owners['terminal_descendant_'+str(pid)]=followup.handle(generation,helpers,followup.CHILD_MARKER,
  waiter_state['terminal_marker'],followup.ENVIRONMENT)
marked=set()
for marker,value,allowed in marks:
 marked.update(set(followup.handoff.marker_processes(marker,value))-allowed)
ready=not marked and all(item['status']=='absent' for item in owners.values())
keeper={}
if mode=='keeper':
 if not ready:
  raise RuntimeError('Exit-drain predecessor owner appeared during keeper verification')
 keeper=followup.handoff.keeper_observation()
terminal,attempt,outcome,_=followup.controller.terminal_paths(root,intent)
print(json.dumps(dict(cpu=cpu,frozen=frozen,intent=intent,state=state,result=result,waiter=waiter,
 waiter_state=waiter_state,owners=owners,marked_pids=sorted(marked),ready=ready,keeper=keeper,
 terminal=str(terminal),attempt=str(attempt),outcome=str(outcome),
 paths=dict(binding=str(binding),helpers=str(helpers),state=str(state_path),work=str(work),
 result=str(result_path)),waiter_command=followup.command(root,cpu_path)),allow_nan=False))
"""
REPORT_PROGRAM = """import json,sys
from pathlib import Path
import ns_training_path_exit_reporting as reporting
print(json.dumps(reporting.report_module(Path(sys.argv[1])).evidence(Path(sys.argv[1])),allow_nan=False))
"""


def require_root(root: Path) -> None:
    if root != root.resolve(strict=True) or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Exit-drain actual-LR predecessor requires canonical root and hidden CUDA")


def file_identity(path: Path) -> dict[str, int]:
    info = path.lstat()
    if path != path.resolve(strict=True) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Exit-drain handoff requires canonical singly linked regular evidence")
    return dict(
        device=info.st_dev,
        inode=info.st_ino,
        bytes=info.st_size,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
        links=info.st_nlink,
    )


def checked(path: Path, digest: str | None = None) -> tuple[bytes, dict]:
    if path.suffix in (".pt", ".bin", ".safetensors"):
        raise ValueError("Exit-drain handoff may not hash tensor payloads")
    identity = file_identity(path)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if (digest is not None and actual != digest) or file_identity(path) != identity:
        raise ValueError(f"Exit-drain handoff evidence changed: {path}")
    return raw, {"sha256": actual, "identity": identity}


def saved(path: Path, digest: str | None = None) -> dict:
    return json.loads(checked(path, digest)[0])


def relative(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("Exit-drain handoff evidence path escapes its root")
    return root / name


def run_saved(directory: Path, program: str, arguments: list[str]) -> dict:
    """Run one read-only pinned closure with hidden CUDA and bounded, reaped execution."""
    result = subprocess.run(
        [PYTHON, "-B", "-P", "-c", program, *arguments],
        cwd=directory,
        env=os.environ | ENVIRONMENT | {"PYTHONPATH": str(directory), "UV_OFFLINE": "1"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=READ_TIMEOUT,
    )
    if result.returncode != 0:
        # Do not reflect arbitrary child stdout/stderr or inherited environment into receipts.
        raise RuntimeError(f"Exit-drain pinned read-only child refused evidence (exit {result.returncode})")
    return json.loads(result.stdout)


def waiter_cpu(root: Path) -> dict:
    """Authenticate the exact accepted waiter package, without merging helper namespaces."""
    cpu = saved(root / FOLLOWUP_CPU, FOLLOWUP_CPU_SHA256)
    bootstrap = saved(root / BOOTSTRAP, BOOTSTRAP_SHA256)
    helpers = (root / FOLLOWUP_CPU).parent / "tracking"
    if (
        cpu["root"] != str(root)
        or cpu["status"] != "ns_training_path_exit_drain_read_only_followup_cpu_archive_verified"
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or "error" in cpu
        or cpu["environment"] != ENVIRONMENT
        or cpu["helpers"] != str(helpers)
        or len(cpu["helpers_sha256"]) != 169
        or cpu["helpers_sha256"]["ns_training_path_exit_followup.py"] != FOLLOWUP_SHA256
        or cpu["binding_sha256"] != BINDING_SHA256
        or cpu["original_intent_sha256"] != INTENT_SHA256
        or bootstrap["root"] != str(root)
        or bootstrap["new_helpers_sha256"]["ns_training_path_exit_followup.py"] != FOLLOWUP_SHA256
        or bootstrap["issued_helpers_sha256"]
        != {
            name: digest for name, digest in cpu["helpers_sha256"].items() if name not in bootstrap["new_helpers_sha256"]
        }
    ):
        raise ValueError("Exit-drain predecessor accepted waiter or bootstrap provenance differs")
    for name, digest in cpu["helpers_sha256"].items():
        if Path(name).name != name or not name.endswith((".py", ".sh")):
            raise ValueError("Exit-drain predecessor waiter inventory is not a flat source closure")
        checked(helpers / name, digest)
    for key in ("archive", "tests", "log"):
        checked(Path(cpu[key]), cpu[f"{key}_sha256"])
    suites = list(ET.fromstring(checked(Path(cpu["tests"]))[0]).iter("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        len(cases) != 24
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != 24
        or sum(int(suite.attrib["tests"]) for suite in suites) != 24
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(key) is not None for case in cases for key in ("failure", "error", "skipped"))
    ):
        raise ValueError("Exit-drain predecessor waiter CPU suite is incomplete")
    return cpu


def observation(root: Path, mode: str = "owners") -> dict:
    require_root(root)
    if FOLLOWUP_INTENT_SHA256 is None:
        raise RuntimeError("Exit-drain predecessor waiter intent is not pinned")
    waiter_cpu(root)
    waiter = saved(root / FOLLOWUP_WORK / "intent.json", FOLLOWUP_INTENT_SHA256)
    if waiter["runner_sha256"] != FOLLOWUP_SHA256 or waiter["cpu_receipt_sha256"] != FOLLOWUP_CPU_SHA256:
        raise ValueError("Exit-drain predecessor waiter acknowledgment source differs")
    checked(root / BINDING, BINDING_SHA256)
    result = run_saved(
        (root / FOLLOWUP_CPU).parent / "tracking", OBSERVER_PROGRAM, [str(root), str(root / FOLLOWUP_CPU), mode]
    )
    frozen, intent, state = result["frozen"], result["intent"], result["state"]
    paths = result["paths"]
    if (
        result["waiter"] != waiter
        or paths["binding"] != str(root / BINDING)
        or frozen != saved(root / BINDING, BINDING_SHA256)
        or intent != saved(Path(paths["helpers"]).parent / "controller.json", INTENT_SHA256)
        or state != saved(Path(paths["state"]))
        or result["result"] != saved(Path(paths["result"]))
        or result["waiter_state"] != saved(root / FOLLOWUP_WORK / "state.json")
        or frozen["root"] != str(root)
        or frozen["helpers"] != paths["helpers"]
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or state["binding_sha256"] != BINDING_SHA256
        or state.get("spawn_status") != "acknowledged"
        or state["child_generation"]["pid"] != state["child_pid"]
        or state["child_generation"]["start_ticks"] != state["child_start_ticks"]
        or state["child_generation"]["parent"] != intent["pid"]
        or state["child_generation"]["command"] != frozen["validator_command"]
        or state["command"] != frozen["validator_command"]
        or intent["command"] != frozen["controller_command"]
        or waiter["status"] != "acknowledged"
        or waiter["binding_sha256"] != BINDING_SHA256
        or waiter["command"] != result["waiter_command"]
        or result["waiter_state"]["pid"] != waiter["pid"]
        or result["waiter_state"]["start_ticks"] != waiter["start_ticks"]
        or result["waiter_state"]["binding_sha256"] != BINDING_SHA256
        or result["waiter_state"]["runner_sha256"] != FOLLOWUP_SHA256
        or result["waiter_state"]["cpu_receipt_sha256"] != FOLLOWUP_CPU_SHA256
    ):
        raise ValueError("Exit-drain predecessor original owner/waiter attribution differs")
    for record in (intent, waiter):
        generation = record["generation"]
        if any(generation[key] != record[key] for key in ("pid", "start_ticks", "command")):
            raise ValueError("Exit-drain predecessor acknowledged process generation differs")
    for generation in (intent["generation"], state["child_generation"], waiter["generation"]):
        if generation["group"] != generation["pid"] or generation["session"] != generation["pid"]:
            raise ValueError("Exit-drain predecessor original process session differs")
    ready = not result["marked_pids"] and all(item["status"] == "absent" for item in result["owners"].values())
    if result["ready"] is not ready:
        raise ValueError("Exit-drain predecessor owner absence predicate differs")
    return result


def readiness(root: Path) -> dict:
    observed = observation(root)
    return {
        "status": "exit_drain_owners_absent" if observed["ready"] else "waiting_for_exit_drain_owners",
        "ready": observed["ready"],
        "owners": observed["owners"],
        "marked_pids": observed["marked_pids"],
    }


def exit_report(root: Path) -> dict:
    directory = Path(__file__).resolve().parent
    pins = REPORT_SOURCES | {"ns_training_path_exit_reporting.py": REPORTER_SHA256}
    before = {name: checked(directory / name, digest)[1] for name, digest in pins.items()}
    result = run_saved(directory, REPORT_PROGRAM, [str(root)])
    if any(checked(directory / name, item["sha256"])[1] != item for name, item in before.items()):
        raise ValueError("Exit-drain predecessor reporter changed during inspection")
    return result


def same_keeper(actual: dict, expected: dict) -> bool:
    before, after = expected["generation"], actual["generation"]
    return (
        actual["pid"] == expected["pid"]
        and actual["devices"] == expected["devices"]
        and actual["occupied"] == expected["occupied"]
        and len(before) == len(after) == 9
        and set(before) == set(after)
        and all(
            process["status"] not in ("Z", "X", "x")
            and after[pid]["status"] not in ("Z", "X", "x")
            and {key: value for key, value in process.items() if key != "status"}
            == {key: value for key, value in after[pid].items() if key != "status"}
            for pid, process in before.items()
        )
    )


def instant(record: dict) -> datetime:
    result = datetime.fromisoformat(record["time"])
    if result.tzinfo is None:
        raise ValueError("Exit-drain predecessor evidence lacks absolute time")
    return result


def retain(root: Path, records: dict, path: Path, digest: str | None = None) -> None:
    name = str(path.relative_to(root))
    item = checked(relative(root, name), digest)[1]
    if name in records and records[name] != item:
        raise ValueError("Exit-drain predecessor retained evidence identities conflict")
    records[name] = item


def checkpoint_payloads(root: Path, cases: list[dict], claim: datetime, records: dict) -> dict:
    """Reuse reconstructor digests; inspect only payload metadata and chronology."""
    payloads = {}
    claim_ns = int(claim.timestamp() * 1_000_000_000)
    for case in cases:
        if set(case["trajectories"]) != set(MODES):
            raise ValueError("Exit-drain predecessor trajectory coverage differs")
        for mode, trajectory in case["trajectories"].items():
            directory = Path(case["continuous"] if mode == "continuous" else case["resumed"])
            directory.relative_to(root)
            if set(trajectory["checkpoints"]) != CHECKPOINTS[mode]:
                raise ValueError("Exit-drain predecessor checkpoint endpoint coverage differs")
            manifest = "attribution-interrupted-manifest.json" if mode == "interrupted" else "run.json"
            retain(root, records, directory / manifest, trajectory["manifest_sha256"])
            retain(root, records, directory / "config.json")
            for name, digest in trajectory["history_sha256"].items():
                if Path(name).name != name:
                    raise ValueError("Exit-drain predecessor scalar history escapes trajectory")
                retain(root, records, directory / name, digest)
            for step, checkpoint in trajectory["checkpoints"].items():
                folder = directory / f"checkpoints/step_{int(step):06d}"
                retain(root, records, folder / "complete.json", checkpoint["receipt_sha256"])
                if len(checkpoint["rank_bytes"]) != 8 or set(checkpoint["rank_sha256"]) != {
                    f"rank_{rank}.pt" for rank in range(8)
                }:
                    raise ValueError("Exit-drain predecessor checkpoint rank coverage differs")
                for rank, size in enumerate(checkpoint["rank_bytes"]):
                    path = folder / f"rank_{rank}.pt"
                    identity = file_identity(path)
                    if identity["bytes"] != size or identity["ctime_ns"] > claim_ns:
                        raise ValueError("Exit-drain predecessor payload changed after independent claim")
                    name = str(path.relative_to(root))
                    item = {"sha256": checkpoint["rank_sha256"][path.name], "identity": identity}
                    if name in payloads and payloads[name] != item:
                        raise ValueError("Exit-drain predecessor shared checkpoint identities conflict")
                    payloads[name] = item
    return payloads


def evidence(root: Path) -> dict:
    """Bind complete independent receipts; this function never publishes a seal."""
    observed = observation(root)
    if not observed["ready"]:
        raise RuntimeError("Exit-drain predecessor controller/validator/waiter/descendant remains")
    report = exit_report(root)
    frozen, intent, state, result, waiter_state = (
        observed[name] for name in ("frozen", "intent", "state", "result", "waiter_state")
    )
    terminal_path, attempt_path, outcome_path = (Path(observed[name]) for name in ("terminal", "attempt", "outcome"))
    generation = {
        "binding_sha256": BINDING_SHA256,
        "controller_pid": intent["pid"],
        "controller_start_ticks": intent["start_ticks"],
        "controller_marker": intent["marker"],
    }
    token = hashlib.sha256(json.dumps(generation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (
        terminal_path != root / "run-preparation" / f"ns-training-path-exit-drain-20261119-terminal-{token}.json"
        or attempt_path != Path(frozen["helpers"]).parent / "terminal-intent.json"
        or outcome_path != Path(frozen["helpers"]).parent / "terminal-outcome.json"
    ):
        raise ValueError("Exit-drain predecessor terminal is not its canonical generation-bound path")
    terminal, claim, outcome = (saved(path) for path in (terminal_path, attempt_path, outcome_path))
    progress, failed = report["progress"], report["sealed"]
    cases = result["results"]
    decisions = {
        "accepted_cases": [case["name"] for case in cases if case["accepted"]],
        "rejected_cases": [case["name"] for case in cases if not case["accepted"]],
        "all_cases_accepted": all(case["accepted"] for case in cases),
    }
    controller_digest = frozen["helpers_sha256"]["ns_training_path_exit_gpu_controller.py"]
    terminal_command = [
        PYTHON,
        "-P",
        str(Path(frozen["helpers"]) / "ns_training_path_exit_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "terminal",
        "--out",
        str(terminal_path),
    ]
    if (
        set(report) != {"cpu", "sealed", "progress"}
        or progress["state"] != state
        or report["cpu"] != saved(root / EXIT_CPU, EXIT_CPU_SHA256)
        or frozen["cpu_receipt"] != str(root / EXIT_CPU)
        or frozen["cpu_receipt_sha256"] != EXIT_CPU_SHA256
        or frozen["helpers_sha256"] != report["cpu"]["helpers_sha256"]
        or frozen["files_sha256"] != report["cpu"]["files_sha256"]
        or progress["intent"] != intent
        or progress["result"] != result
        or progress["terminal"] != terminal
        or progress["status"] != "validation_complete"
        or progress["completed_commands"] != 54
        or progress["decisions"] != cases
        or len(cases) != 18
        or len({case["name"] for case in cases}) != 18
        or any(type(case["accepted"]) is not bool for case in cases)
        or len(result["gpu_commands"]) != 54
        or len(frozen["commands"]) != 54
        or len(frozen["plan"]["trials"]) != 18
        or frozen["commands"] != frozen["plan"]["commands"]
        or result["trials"] != frozen["plan"]["trials"]
        or result["status"] != "ns_training_path_exit_drain_gates_reconstructed_not_production"
        or result["binding_sha256"] != BINDING_SHA256
        or state["result_sha256"] != checked(Path(observed["paths"]["result"]))[1]["sha256"]
        or state["status"] != "validation_complete"
        or any(
            type(state[key]) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or state["marked_descendants_absent"] is not True
        or state["keeper_release_complete"] is not True
        or any(
            key in value
            for value in (state, result, waiter_state)
            for key in ("error", "cleanup_error", "deferred_signals")
        )
        or waiter_state["status"] != "terminal_reconstructed"
        or waiter_state["runner_sha256"] != FOLLOWUP_SHA256
        or waiter_state["cpu_receipt_sha256"] != FOLLOWUP_CPU_SHA256
        or waiter_state["terminal_command"] != terminal_command
        or waiter_state["terminal_generation"]["command"] != terminal_command
        or waiter_state["terminal_generation"]["parent"] != observed["waiter"]["pid"]
        or waiter_state["terminal_generation"]["pid"] != waiter_state["terminal_pid"]
        or waiter_state["terminal_generation"]["group"] != waiter_state["terminal_pid"]
        or waiter_state["terminal_generation"]["session"] != waiter_state["terminal_pid"]
        or any(
            type(waiter_state[key]) is not int or waiter_state[key] != 0
            for key in ("terminal_exit_code", "terminal_cleanup_exit_code")
        )
        or waiter_state["terminal_descendants_absent"] is not True
        or waiter_state["terminal"] != str(terminal_path)
        or waiter_state["terminal_sha256"] != checked(terminal_path)[1]["sha256"]
        or waiter_state["terminal_outcome_sha256"] != checked(outcome_path)[1]["sha256"]
        or claim["status"] != "reconstruction_claimed"
        or any(claim[name] != value for name, value in generation.items())
        or claim["output"] != str(terminal_path)
        or claim["verifier_sha256"] != controller_digest
        or claim["reconstructor_pid"] != waiter_state["terminal_pid"]
        or outcome["status"] != "complete"
        or outcome["output"] != str(terminal_path)
        or outcome["terminal_sha256"] != checked(terminal_path)[1]["sha256"]
        or outcome["reconstruction_intent_sha256"] != checked(attempt_path)[1]["sha256"]
        or terminal["reconstruction_intent_sha256"] != outcome["reconstruction_intent_sha256"]
        or terminal["status"] != "ns_training_path_exit_drain_terminal_independently_reconstructed_not_production"
        or terminal["binding_sha256"] != BINDING_SHA256
        or terminal["controller_pid_absent"] != intent["pid"]
        or terminal["controller_start_ticks"] != intent["start_ticks"]
        or terminal["controller_marker"] != intent["marker"]
        or terminal["validator_pid_absent"] != state["child_pid"]
        or terminal["verifier_sha256"] != controller_digest
        or terminal["result_sha256"] != state["result_sha256"]
        or terminal["terminal_time"] != state["time"]
        or terminal["production_authorized"] is not False
        or terminal["scope"] != frozen["scope"]
        or terminal["predecessor_acceptance"] != frozen["predecessor_acceptance"]
        or frozen["predecessor_acceptance"]["seal_sha256"] != FAILED_SEAL_SHA256
        or frozen["allocation_predecessor"] != failed
        or failed != saved(root / FAILED_SEAL, FAILED_SEAL_SHA256)
        or failed["status"]
        != "both_failed_training_path_attempts_sealed_for_distinct_exit_drain_not_numerical_acceptance"
        or failed["root"] != str(root)
        or failed["production_authorized"] is not False
        or failed["gpu_execution_authorized"] is not False
        or failed["retained_summary"]["completed_commands"] != 2
        or failed["retained_summary"]["case_decisions"] != 0
        or failed["original_retained_summary"]["completed_commands"] != 44
        or failed["original_retained_summary"]["case_decisions"] != 14
        or any(
            value[key] is not False
            for value in (failed["retained_summary"], failed["original_retained_summary"])
            for key in ("coverage_complete", "numerically_accepted")
        )
        or any(
            value[name] != expected
            for value in (result, state, terminal, waiter_state)
            for name, expected in decisions.items()
        )
        or any(type(value["all_cases_accepted"]) is not bool for value in (result, state, terminal, waiter_state))
        or any(progress[name] != decisions[name] for name in ("accepted_cases", "rejected_cases"))
        or not same_keeper(terminal["keeper"], state["restored_keeper"])
        or terminal["keeper"]["devices"] != frozen["devices"]
    ):
        raise ValueError("Exit-drain predecessor complete terminal, waiter, decisions or failed lineage differs")
    chronology = [instant(value) for value in (result, state, claim, terminal, outcome, waiter_state)]
    if chronology != sorted(chronology):
        raise ValueError("Exit-drain predecessor independent reconstruction chronology differs")
    records = {}
    for name, digest in terminal["evidence_sha256"].items():
        retain(root, records, relative(root, name), digest)
    for path in (
        root / BINDING,
        Path(observed["paths"]["state"]),
        Path(observed["paths"]["result"]),
        Path(frozen["helpers"]).parent / "controller.json",
    ):
        retain(root, records, path)
    for path, digest in (
        (root / FOLLOWUP_CPU, FOLLOWUP_CPU_SHA256),
        (root / BOOTSTRAP, BOOTSTRAP_SHA256),
        (root / FOLLOWUP_WORK / "intent.json", FOLLOWUP_INTENT_SHA256),
        (root / FAILED_SEAL, FAILED_SEAL_SHA256),
        (terminal_path, None),
        (attempt_path, None),
        (outcome_path, None),
        (root / FOLLOWUP_WORK / "state.json", None),
        (Path(waiter_state["terminal_log"]), waiter_state["terminal_log_sha256"]),
    ):
        retain(root, records, path, digest)
    for entry in result["gpu_commands"]:
        log = Path(entry["log"])
        retain(root, records, log, entry["log_sha256"])
        retain(root, records, log.with_suffix(".execution.json"))
    for name, item in frozen.get("authenticated_files", {}).items():
        path = Path(name)
        relative(root, str(path.relative_to(root)))
        if file_identity(path) != item["identity"]:
            raise ValueError("Exit-drain predecessor operation-authenticated input identity changed")
        if path.suffix in (".pt", ".bin", ".safetensors"):
            name = str(path.relative_to(root))
            if name in records and records[name] != item:
                raise ValueError("Exit-drain predecessor operation-authenticated payload identity conflicts")
            records[name] = item
            continue
        retain(root, records, path, item["sha256"])
        if records[str(path.relative_to(root))] != item:
            raise ValueError("Exit-drain predecessor operation-authenticated input digest changed")
    for path, digest in (
        (Path(frozen["archive"]), frozen["archive_sha256"]),
        (Path(frozen["cpu_receipt"]), frozen["cpu_receipt_sha256"]),
        (root / FOLLOWUP_WORK / "followup.log", None),
    ):
        retain(root, records, path, digest)
    for name, item in failed["evidence"].items():
        retain(root, records, relative(root, name), item["sha256"])
        if records[name] != item:
            raise ValueError("Exit-drain predecessor failed lineage identity changed")
    payloads = checkpoint_payloads(root, cases, chronology[2], records)
    for name, item in failed["checkpoint_payloads"].items():
        if file_identity(relative(root, name)) != item["identity"]:
            raise ValueError("Exit-drain predecessor failed checkpoint identity changed")
    current = observation(root, "keeper")
    if (
        not current["ready"]
        or not same_keeper(current["keeper"], terminal["keeper"])
        or any(
            current[key] != observed[key] for key in ("frozen", "intent", "state", "result", "waiter", "waiter_state")
        )
    ):
        raise RuntimeError("Exit-drain predecessor exact current keeper or owner absence differs")
    result = {
        "status": STATUS,
        "root": str(root),
        "scope": SCOPE,
        "evidence": records,
        "checkpoint_payloads": payloads,
        "failed_checkpoint_payloads": failed["checkpoint_payloads"],
        "failed_lineage": failed,
        "original_terminal": str(terminal_path),
        "original_terminal_sha256": checked(terminal_path)[1]["sha256"],
        "original_decisions": decisions,
        "restored_keeper": terminal["keeper"],
        "controller_marker": intent["marker"],
        "waiter_intent_sha256": FOLLOWUP_INTENT_SHA256,
        "reporter_sha256": REPORTER_SHA256,
        "verifier_sha256": checked(Path(__file__))[1]["sha256"],
        "execution_closure_merged": False,
        "actual_lr_execution_authorized": False,
        "production_authorized": False,
    }
    require_identities(root, result)
    return result


def require_identities(root: Path, record: dict) -> None:
    for group in ("evidence", "checkpoint_payloads", "failed_checkpoint_payloads"):
        for name, item in record[group].items():
            if file_identity(relative(root, name)) != item["identity"]:
                raise ValueError("Exit-drain predecessor sealed evidence identity changed")


def seal(root: Path) -> dict:
    """Publish one immutable seal under the existing manager lock, without manager actions."""
    require_root(root)
    path = root / SEAL
    if path.exists() or path.is_symlink():
        raise FileExistsError("Exit-drain actual-LR predecessor seal is one-shot")
    with MANAGER_LOCK.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (os.fstat(lock.fileno()).st_dev, os.fstat(lock.fileno()).st_ino) != (
            MANAGER_LOCK.stat().st_dev,
            MANAGER_LOCK.stat().st_ino,
        ):
            raise RuntimeError("Exit-drain predecessor manager lock identity changed")
        record = evidence(root) | {"time": datetime.now(UTC).isoformat()}
        require_identities(root, record)
        with tempfile.TemporaryDirectory(prefix=f".{NAMESPACE}-", dir=path.parent) as directory:
            staged = Path(directory) / "seal.json"
            with staged.open("x") as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(staged, path)
            staged.unlink()
    return saved(path)


def require_seal(root: Path, digest: str) -> dict:
    """Check a caller-bound seal after allocation transfer without demanding the old keeper."""
    require_root(root)
    path = root / SEAL
    raw, identity = checked(path, digest)
    record = json.loads(raw)
    if (
        record["status"] != STATUS
        or record["root"] != str(root)
        or record["scope"] != SCOPE
        or record["verifier_sha256"] != checked(Path(__file__))[1]["sha256"]
        or record["reporter_sha256"] != REPORTER_SHA256
        or record["waiter_intent_sha256"] != FOLLOWUP_INTENT_SHA256
        or any(
            record[key] is not False
            for key in ("execution_closure_merged", "actual_lr_execution_authorized", "production_authorized")
        )
        or len(record["original_decisions"]["accepted_cases"]) + len(record["original_decisions"]["rejected_cases"])
        != 18
        or record["original_decisions"]["all_cases_accepted"] != (not record["original_decisions"]["rejected_cases"])
        or len(set(record["original_decisions"]["accepted_cases"]) | set(record["original_decisions"]["rejected_cases"]))
        != 18
        or type(record["original_decisions"]["all_cases_accepted"]) is not bool
        or not record["checkpoint_payloads"]
        or record["failed_lineage"] != saved(root / FAILED_SEAL, FAILED_SEAL_SHA256)
        or record["evidence"].get(str(Path(record["original_terminal"]).relative_to(root)), {}).get("sha256")
        != record["original_terminal_sha256"]
    ):
        raise ValueError("Exit-drain predecessor caller-bound seal differs")
    require_identities(root, record)
    if not readiness(root)["ready"] or checked(path, digest)[1] != identity:
        raise RuntimeError("Exit-drain predecessor owners or seal changed")
    return record
