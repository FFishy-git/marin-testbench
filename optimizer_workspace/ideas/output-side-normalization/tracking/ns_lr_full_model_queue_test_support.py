# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU process and external receipt fixtures for the actual-LR one-shot queue."""

import os
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_budget_gpu_owner as process_owner
import ns_lr_full_model_queue as queue
from ns_lr_full_model_predecessor_test_support import generation
from selected_lr_release_owner import sha256
from watch_wave5 import write

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def original_queue(root: Path, monkeypatch) -> tuple[Path, dict]:
    work = root / queue.predecessor.queue.WORK
    work.mkdir(parents=True)
    intent = {
        "status": "acknowledged",
        "generation": generation(910001, 1, [queue.contract.PYTHON, "absent-original-queue"]),
        "marker": "original-queue",
    }
    write(work / "intent.json", intent)
    write(work / "state.json", {"phases": []})
    monkeypatch.setattr(queue.predecessor.reporting, "INTENT_SHA256", sha256(work / "intent.json"))
    return work, intent


def phase_program(helpers: Path) -> Path:
    program = helpers / "phase.py"
    program.write_text(
        "import json,os,select,signal,subprocess,sys\nfrom pathlib import Path\n"
        f"work=Path(sys.argv[1])/{queue.WORK!r};outcome=sys.argv[2];poll=select.poll()\n"
        "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
        "if outcome=='timeout':signal.pause()\n"
        "if outcome in ('detached','new_owner'):\n"
        " r,w=os.pipe();env=dict(os.environ)\n"
        f" if outcome=='new_owner':env[{queue.PHASE_MARKER!r}]='separate-controller'\n"
        " child=subprocess.Popen([sys.executable,'-c',"
        "'import os,signal,sys;os.write(int(sys.argv[1]),b\"R\");signal.pause()',str(w)],"
        "env=env,pass_fds=(w,),start_new_session=True)\n"
        " os.close(w);assert os.read(r,1)==b'R';os.close(r)\n"
        " (work/'detached.pid').write_text(str(child.pid))\n"
        "sys.exit(7 if outcome in ('nonzero','detached') else 0)\n"
    )
    return program


def issued_owner(root: Path, monkeypatch, outcome: str) -> tuple[dict, Path]:
    binding, helpers, state_path, _, _ = queue.contract.locations(root)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    program = helpers / "ns_lr_full_model_gpu_controller.py"
    program.write_text("# issued actual-LR external fixture\n")
    candidate = root / "candidate"
    candidate.mkdir()
    frozen = {
        "candidate": str(candidate),
        "helpers_sha256": {program.name: sha256(program)},
        "controller_command": queue.owner.controller_command(root),
        "cpu_receipt_sha256": queue.PARENT_SHA256,
        "environment": queue.contract.ENVIRONMENT,
    }
    write(binding, frozen)
    own = generation(920001, 1, frozen["controller_command"])
    child = generation(920002, own["pid"], queue.owner.validator_command(root, candidate))
    intent = {
        "status": "acknowledged",
        "pid": own["pid"],
        "start_ticks": own["start_ticks"],
        "binding_sha256": sha256(binding),
        "generation": own,
        "marker": "controller",
    }
    write(helpers.parent / "controller.json", intent)
    saved = {
        "status": "needs_attention" if outcome == "failed" else "validation_complete",
        "controller_pid": own["pid"],
        "controller_start_ticks": own["start_ticks"],
        "binding_sha256": sha256(binding),
        "spawn_status": "acknowledged",
        "child_generation": child,
        "child_marker": "validator",
    }
    write(state_path, saved)
    live = {"controller_live": {own["pid"]: own}, "validator_live": {child["pid"]: child}}.get(outcome, {})
    monkeypatch.setattr(queue.handoff, "observed", lambda pid: live.get(pid))
    monkeypatch.setattr(process_owner, "process_context", lambda pid, *args: live[pid])
    monkeypatch.setattr(
        queue.handoff,
        "marker_processes",
        lambda *args: {930001: {}} if outcome == "marked_rank" else {},
    )
    return {"issued_binding_sha256": sha256(binding), "issued_controller": intent}, state_path


def terminal_program(root: Path, changed: bool) -> tuple[Path, dict]:
    binding, helpers, _, _, _ = queue.contract.locations(root)
    helpers.mkdir(parents=True)
    (root / queue.WORK).mkdir(parents=True)
    program = helpers / "ns_lr_full_model_gpu_controller.py"
    program.write_text(
        "import argparse,hashlib,json,os,select\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--phase');p.add_argument('--out');a=p.parse_args()\n"
        f"root=Path(a.root);work=root/{queue.WORK!r};poll=select.poll()\n"
        "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
        "assert a.phase=='terminal' and os.environ['CUDA_VISIBLE_DEVICES']==''\n"
        "state=json.loads((work/'state.json').read_text())\n"
        f"Path(a.out).write_text(json.dumps({{'status':{queue.controller.TERMINAL_STATUS!r},"
        "'binding_sha256':state['issued_binding_sha256'],"
        "'controller_pid_absent':state['issued_controller']['pid'],"
        f"'verifier_sha256':{'\"changed\"' if changed else 'hashlib.sha256(Path(__file__).read_bytes()).hexdigest()'},"
        "'evidence_sha256':{},'accepted_cases':['accepted'],'rejected_cases':['rejected'],"
        "'all_cases_accepted':False}))\n"
    )
    write(binding, {"helpers_sha256": {program.name: sha256(program)}})
    return program, {
        "status": "actual_lr_terminal_ready",
        "phases": [],
        "issued_binding_sha256": sha256(binding),
        "issued_controller": {"pid": 940001},
    }


def completed_cpu(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    out = Path(kwargs["cwd"])
    xml = ET.Element("testsuites")
    suite = ET.SubElement(xml, "testsuite", tests=str(queue.TEST_COUNT), failures="0", errors="0", skipped="0")
    for index in range(queue.TEST_COUNT):
        ET.SubElement(suite, "testcase", classname=Path(queue.NEW_HELPERS[-1]).stem, name=f"external_case_{index}")
    ET.ElementTree(xml).write(out / "tests.xml")
    kwargs["stdout"].write("External CPU runner complete\n")
    return subprocess.CompletedProcess(command, 0)


def archive_probe(root: Path, out: Path) -> Path:
    mutable = root / "mutable_predecessor.py"
    mutable.write_bytes((out / "tracking/ns_lr_full_model_predecessor.py").read_bytes())
    probe = root / "archive_probe.py"
    probe.write_text(
        "import importlib.util,json,sys\nfrom pathlib import Path\n"
        "root,receipt,parent,mutable=map(Path,sys.argv[1:5]);fault=sys.argv[5]\n"
        "import ns_lr_full_model_queue as queue\n"
        "queue.parent=lambda root:queue.read(parent)\n"
        "if fault=='mutable_import':\n"
        " spec=importlib.util.spec_from_file_location('ns_lr_full_model_predecessor',mutable)\n"
        " mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)\n"
        "cpu=queue.accepted(root,receipt)\n"
        "print(json.dumps({'status':'archived_actual_lr_queue_roundtrip','helpers':len(cpu['helpers_sha256'])}))\n"
    )
    return probe


def fail_spawn(*args, **kwargs):
    raise OSError("External queue phase spawn failed")


def process_environment(helpers: Path) -> dict:
    return (
        os.environ
        | queue.ENVIRONMENT
        | {
            "PYTHONPATH": f"{helpers}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
        }
    )
