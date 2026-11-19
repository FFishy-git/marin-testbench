# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real file/process fixtures at the accepted terminal reporting boundary."""

import json
import shutil
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_compiler_localization_predecessor as predecessor
import ns_training_path_lr_predecessor_test_support as original_support
from fused_attribution_validation_test_support import record as fixture_record
from ns_training_path_lr_gpu_test_support import generation
from ns_training_path_lr_test_support import PRESERVED
from selected_lr_release_owner import read, sha256

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def record(path: Path, value: dict) -> None:
    value.setdefault("time", datetime.now(UTC).isoformat())
    fixture_record(path, value)


def archive(root: Path, spec: tuple, candidate: Path, files: dict, helpers: dict) -> tuple[tuple, dict]:
    path = root / spec[0]
    output = path.parent
    output.mkdir(parents=True)
    hashes = {}
    for name, source in helpers.items():
        target = output / "tracking" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = sha256(target)
    inventory = dict(files) | {f"tracking/{name}": digest for name, digest in hashes.items()}
    for name in files:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate / name, target)
    (output / "artifacts").mkdir(exist_ok=True)
    (output / "artifacts/pinned-sources-materialized").mkdir(exist_ok=True)
    (output / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    bundle_path = output / "source.tar"
    with tarfile.open(bundle_path, "x") as bundle:
        for name in inventory:
            bundle.add(output / name, arcname=name, recursive=False)
        bundle.add(output / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    log, tests = output / "cpu.log", output / "tests.xml"
    log.write_text("External accepted CPU receipt fixture; no GPU execution\n")
    xml = ET.Element("testsuites")
    suite = ET.SubElement(xml, "testsuite", tests=str(spec[3]), errors="0", failures="0", skipped="0", time="0")
    for index in range(spec[3]):
        ET.SubElement(suite, "testcase", classname="external_accepted_boundary", name=f"case_{index}")
    ET.ElementTree(xml).write(tests)
    cpu = {
        "status": spec[2],
        "root": str(root),
        "candidate": str(candidate),
        "exit_code": 0,
        "cuda_visible_devices": "",
        "helpers": str(output / "tracking"),
        "helpers_sha256": hashes,
        "files_sha256": files,
        "inventory_sha256": inventory,
        **{name: str(item) for name, item in (("archive", bundle_path), ("tests", tests), ("log", log))},
        **{f"{name}_sha256": sha256(item) for name, item in (("archive", bundle_path), ("tests", tests), ("log", log))},
    }
    record(path, cpu)
    return (spec[0], sha256(path), spec[2], spec[3]), cpu


def sync_report(root: Path) -> None:
    queue = predecessor.queue
    work = root / queue.WORK
    intent, state = read(work / "intent.json"), read(work / "state.json")
    terminal = read(Path(state["terminal"]))
    cpu = read(root / predecessor.reporting.CPU)
    record(root / "external-queue-report.json", {"report": [cpu, intent, state, terminal]})


def rebind(root: Path) -> None:
    """Update outer fixture hashes so mutations must fail on semantic or retained evidence."""
    queue = predecessor.queue
    binding, helpers, state_path, _, result_path = queue.contract.locations(root)
    state = read(state_path)
    state["result_sha256"] = sha256(result_path)
    record(state_path, state)
    intent = read(helpers.parent / "controller.json")
    output, attempt, outcome, _ = queue.controller.terminal_paths(root, intent)
    terminal = read(output)
    terminal.update(result_sha256=sha256(result_path), terminal_time=state["time"])
    terminal["evidence_sha256"] = {
        str(path.relative_to(root)): sha256(path)
        for path in (
            binding,
            state_path,
            helpers.parent / "controller.json",
            result_path,
            helpers.parent / "controller.log",
            helpers.parent / "validator.log",
        )
    }
    terminal["time"] = (datetime.fromisoformat(read(attempt)["time"]) + timedelta(seconds=1)).isoformat()
    output.write_text(json.dumps(terminal, indent=2) + "\n")
    finished = read(outcome)
    finished.update(terminal_sha256=sha256(output), reconstruction_intent_sha256=sha256(attempt))
    record(outcome, finished)
    work = root / queue.WORK
    saved = read(work / "state.json")
    saved.update(terminal_sha256=sha256(output), terminal_outcome_sha256=sha256(outcome))
    saved["time"] = (datetime.fromisoformat(terminal["time"]) + timedelta(seconds=1)).isoformat()
    (work / "state.json").write_text(json.dumps(saved, indent=2) + "\n")
    sync_report(root)


def completed(root: Path, monkeypatch) -> tuple[Path, dict]:
    """Persist complete original and actual-LR evidence without reading checkpoint tensors."""
    queue = predecessor.queue
    root, restored = original_support.completed(root, monkeypatch)
    monkeypatch.setattr(predecessor.handoff, "keeper_observation", lambda: restored)
    previous = queue.predecessor.seal(root)
    candidate = root / "candidates/actual-lr-fixture"
    names = {
        *(item["config_name"] for item in queue.contract.specs()),
        *(
            f"experiment/configs/{family}_full_lr_lower_0p5m_seed0.json"
            for family in queue.contract.study.contract.FAMILIES
        ),
    }
    archived = Path(__file__).resolve().parents[1]
    source = archived if (archived / next(iter(names))).is_file() else PRESERVED
    for name in names:
        target = candidate / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    files = {name: sha256(candidate / name) for name in names}
    helper_source = Path(queue.controller.__file__)
    helper_files = {helper_source.name: helper_source}
    owner_spec, owner_cpu = archive(root, predecessor.OWNER_SPEC, candidate, files, helper_files)
    execution_spec, _ = archive(
        root,
        predecessor.EXECUTION_SPEC,
        root / "candidate",
        {"train.py": sha256(root / "candidate/train.py")},
        helper_files,
    )
    queue_spec, _ = archive(root, predecessor.QUEUE_SPEC, candidate, files, helper_files)
    monkeypatch.setattr(predecessor, "OWNER_SPEC", owner_spec)
    monkeypatch.setattr(predecessor, "EXECUTION_SPEC", execution_spec)
    monkeypatch.setattr(predecessor, "QUEUE_SPEC", queue_spec)
    monkeypatch.setattr(queue, "PARENT_SHA256", owner_spec[1])
    monkeypatch.setattr(predecessor.reporting, "CPU_SHA256", queue_spec[1])
    binding, helpers, state_path, work, result_path = queue.contract.locations(root)
    helpers.mkdir(parents=True)
    for name, path in helper_files.items():
        shutil.copy2(path, helpers / name)
    plan = queue.controller.validation.plan(root, candidate)
    frozen = {
        "status": queue.contract.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "helpers_sha256": owner_cpu["helpers_sha256"],
        "files_sha256": files,
        "cpu_receipt": str(root / queue.PARENT),
        "cpu_receipt_sha256": owner_spec[1],
        "archive": owner_cpu["archive"],
        "archive_sha256": owner_cpu["archive_sha256"],
        "allocation_predecessor": previous,
        "predecessor_acceptance": {"seal_sha256": sha256(root / queue.predecessor.SEAL)},
        "devices": restored["devices"],
        "controller_command": queue.owner.controller_command(root),
        "validator_command": queue.owner.validator_command(root, candidate),
        "plan": plan,
        "commands": plan["commands"],
        "authenticated_files": {},
    }
    record(binding, frozen)
    controller = generation(811001, 1, frozen["controller_command"])
    validator = generation(811002, controller["pid"], frozen["validator_command"])
    issued = {
        "status": "acknowledged",
        "pid": controller["pid"],
        "start_ticks": controller["start_ticks"],
        "command": controller["command"],
        "generation": controller,
        "marker": "actual-controller",
        "parent_proof": "external-accepted-parent",
        "binding_sha256": sha256(binding),
    }
    record(helpers.parent / "controller.json", issued)
    cases, entries = [], []
    for index, item in enumerate(plan["trials"]):
        trajectories = {}
        for mode, command in zip(queue.contract.MODES, item["commands"], strict=True):
            directory = Path(item["continuous"] if mode == "continuous" else item["resumed"])
            record(directory / "config.json", read(candidate / item["config_name"]))
            manifest = predecessor.INTERRUPTED_MANIFEST if mode == "interrupted" else "run.json"
            record(directory / manifest, {"status": "external retained trajectory"})
            history = directory / f"metrics_{mode}.jsonl"
            history.write_text('{"external":"retained scalar series"}\n')
            checkpoints = {}
            for step in (22,) if mode == "interrupted" else (22, 23, 24):
                folder = directory / f"checkpoints/step_{step:06d}"
                if not (folder / "complete.json").exists():
                    record(folder / "complete.json", {"step": step, "world": 8})
                    for rank in range(8):
                        (folder / f"rank_{rank}.pt").write_bytes(
                            f"opaque actual-LR fixture {index}:{step}:{rank}".encode()
                        )
                ranks = {f"rank_{rank}.pt": sha256(folder / f"rank_{rank}.pt") for rank in range(8)}
                checkpoints[str(step)] = {
                    "receipt_sha256": sha256(folder / "complete.json"),
                    "rank_sha256": ranks,
                    "rank_bytes": [(folder / name).stat().st_size for name in ranks],
                }
            updates = 22 if mode == "interrupted" else 24
            trajectories[mode] = {
                "manifest_sha256": sha256(directory / manifest),
                "history_sha256": {history.name: sha256(history)},
                "checkpoints": checkpoints,
                "updates": updates,
                "tokens": updates * queue.contract.BATCH_TOKENS,
            }
            log = work / f"{item['name']}_{mode}.log"
            log.write_text("External retained command evidence; no GPU execution\n")
            pid = 820000 + len(entries)
            entry = {
                "command": command,
                "marker": f"actual-command-{pid}",
                "spawn_status": "acknowledged",
                "generation": generation(pid, validator["pid"], command),
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "log": str(log),
                "log_sha256": sha256(log),
            }
            record(log.with_suffix(".execution.json"), entry)
            entries.append(entry | {"trial": item["name"], "mode": mode, "trajectory": trajectories[mode]})
        gates = dict.fromkeys(predecessor.GATES, True)
        if index == 0:
            gates["post_cached_root_all_rank_continuation"] = False
        cases.append(item | {"gates": gates, "accepted": index != 0, "trajectories": trajectories})
    decisions = predecessor.decision_summary(cases, [item["name"] for item in plan["trials"]])
    result = plan | {
        "status": queue.controller.validation.STATUS,
        "owner": {
            "controller_pid": controller["pid"],
            "controller_start_ticks": controller["start_ticks"],
            "validator_pid": validator["pid"],
            "validator_start_ticks": validator["start_ticks"],
        },
        "binding_sha256": sha256(binding),
        "gpu_commands": entries,
        "results": cases,
        **decisions,
    }
    record(result_path, result)
    state = {
        "status": "validation_complete",
        "binding_sha256": sha256(binding),
        "controller_pid": controller["pid"],
        "controller_start_ticks": controller["start_ticks"],
        "child_pid": validator["pid"],
        "child_start_ticks": validator["start_ticks"],
        "child_generation": validator,
        "child_marker": "actual-validator",
        "spawn_status": "acknowledged",
        "command": validator["command"],
        "allocation_predecessor": previous,
        "result_sha256": sha256(result_path),
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "marked_descendants_absent": True,
        "keeper_release_complete": True,
        "restored_keeper": restored,
        **decisions,
    }
    record(state_path, state)
    for name in ("controller.log", "validator.log"):
        (helpers.parent / name).write_text("External retained controller log\n")
    output, attempt, outcome, _ = queue.controller.terminal_paths(root, issued)
    record(
        attempt,
        {
            "status": "reconstruction_claimed",
            "binding_sha256": sha256(binding),
            "output": str(output),
            "controller_pid": issued["pid"],
            "controller_start_ticks": issued["start_ticks"],
            "controller_marker": issued["marker"],
            "verifier_sha256": sha256(helpers / helper_source.name),
        },
    )
    terminal = {
        "status": queue.controller.TERMINAL_STATUS,
        "binding_sha256": sha256(binding),
        "result_sha256": sha256(result_path),
        "terminal_time": state["time"],
        "predecessor_acceptance": frozen["predecessor_acceptance"],
        "reconstruction_intent_sha256": sha256(attempt),
        "production_authorized": False,
        "keeper": restored,
        "controller_pid_absent": issued["pid"],
        "validator_pid_absent": validator["pid"],
        "controller_start_ticks": issued["start_ticks"],
        "verifier_sha256": sha256(helpers / helper_source.name),
        "evidence_sha256": {},
        **decisions,
    }
    record(output, terminal)
    record(
        outcome,
        {
            "status": "complete",
            "output": str(output),
            "terminal_sha256": sha256(output),
            "reconstruction_intent_sha256": sha256(attempt),
        },
    )
    queue_work = root / queue.WORK
    queue_work.mkdir(parents=True)
    argv = queue.command(root, root / predecessor.reporting.CPU)
    own = generation(811003, 1, argv)
    intent = {
        "status": "acknowledged",
        "pid": own["pid"],
        "start_ticks": own["start_ticks"],
        "command": argv,
        "generation": own,
        "marker": "original-actual-lr-queue",
        "cpu_receipt": str(root / predecessor.reporting.CPU),
        "cpu_receipt_sha256": queue_spec[1],
        "parent_cpu_receipt_sha256": owner_spec[1],
        "runner_sha256": sha256(Path(queue.__file__)),
        "interval": queue.INTERVAL,
        "scope": queue.SCOPE,
    }
    record(queue_work / "intent.json", intent)
    monkeypatch.setattr(predecessor.reporting, "INTENT_SHA256", sha256(queue_work / "intent.json"))
    queued = {
        "status": "actual_lr_training_path_terminal_reconstructed",
        **{
            name: intent[name]
            for name in (
                "pid",
                "start_ticks",
                "cpu_receipt_sha256",
                "parent_cpu_receipt_sha256",
                "runner_sha256",
                "interval",
                "scope",
            )
        },
        "issued_controller": issued,
        "issued_binding_sha256": sha256(binding),
        "predecessor_seal_sha256": frozen["predecessor_acceptance"]["seal_sha256"],
        "original_decisions": previous["original_decisions"],
        "phases": [],
        "terminal": str(output),
        "terminal_sha256": sha256(output),
        "terminal_outcome_sha256": sha256(outcome),
        **decisions,
    }
    for index, phase in enumerate(predecessor.reporting.PHASES):
        argv, cwd = predecessor.reporting.phase_command(root, phase, queued)
        log = queue_work / f"{phase}.log"
        log.write_text("External retained queue phase\n")
        child = generation(830000 + index, own["pid"], argv)
        queued["phases"].append(
            {
                "phase": phase,
                "command": argv,
                "cwd": str(cwd),
                "spawn_status": "acknowledged",
                "pid": child["pid"],
                "start_ticks": child["start_ticks"],
                "generation": child,
                "marker": f"phase-{index}",
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "log": str(log),
                "log_sha256": sha256(log),
            }
        )
    (queue_work / "queue.log").write_text("External retained queue log\n")
    record(queue_work / "state.json", queued)
    reporter = root / "external-report-reader/ns_training_path_lr_queue_reporting.py"
    reporter.parent.mkdir()
    reporter.write_text(
        "import json\ndef evidence(root):\n"
        " return json.loads((root/'external-queue-report.json').read_text())['report']\n"
    )
    monkeypatch.setattr(predecessor.reporting, "__file__", str(reporter))
    monkeypatch.setattr(predecessor, "REPORTER_SHA256", sha256(reporter))
    rebind(root)
    return root, restored


def bind_queue_generation(root: Path, monkeypatch, own: dict) -> None:
    queue = predecessor.queue
    work = root / queue.WORK
    intent, state = read(work / "intent.json"), read(work / "state.json")
    intent.update(pid=own["pid"], start_ticks=own["start_ticks"], command=own["command"], generation=own)
    state.update(pid=own["pid"], start_ticks=own["start_ticks"])
    for entry in state["phases"]:
        entry["generation"]["parent"] = own["pid"]
    record(work / "intent.json", intent)
    record(work / "state.json", state)
    monkeypatch.setattr(predecessor.reporting, "INTENT_SHA256", sha256(work / "intent.json"))
