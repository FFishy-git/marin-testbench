# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External CPU process and archive fixtures for the training-path actual-LR queue."""

import os
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_lr_queue as queue
from ns_training_path_lr_gpu_test_support import generation
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


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


def launch_program(root: Path, sealed: dict) -> tuple[Path, Path]:
    inherited = (root / queue.PARENT).parent / "tracking"
    sealer = (root / queue.owner.PREDECESSOR_RECEIPT).parent / "tracking"
    inherited.mkdir(parents=True)
    sealer.mkdir(parents=True)
    write(root / queue.PARENT, {"external_fixture": "accepted-owner"})
    write(root / "fixture-seal-input.json", sealed)
    script = (
        "import argparse,hashlib,json,os,select,shutil,signal,subprocess,sys,uuid\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--phase');"
        "p.add_argument('--cpu-receipt');p.add_argument('--seal-sha256');a=p.parse_args();root=Path(a.root)\n"
        f"work=root/{queue.WORK!r};issued=root/{str(queue.contract.locations(root)[1].relative_to(root))!r}\n"
        f"binding=root/{str(queue.contract.locations(root)[0].relative_to(root))!r};seal=root/{queue.predecessor.SEAL!r}\n"
        "digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()\n"
        "if a.phase=='execute':signal.pause()\n"
        "poll=select.poll()\n"
        "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
        "if a.phase=='seal':shutil.copy2(root/'fixture-seal-input.json',seal)\n"
        "elif a.phase=='check-seal':assert digest(seal)==a.seal_sha256\n"
        "elif a.phase=='freeze':\n"
        f" assert a.cpu_receipt==str(root/{queue.PARENT!r}) and digest(seal)==a.seal_sha256\n"
        " issued.mkdir(parents=True);program=issued/'ns_training_path_lr_gpu_controller.py'\n"
        " shutil.copy2(__file__,program)\n"
        " command=[sys.executable,'-P',str(program),'--root',str(root),'--phase','execute']\n"
        " saved={'cpu_receipt_sha256':"
        f"{queue.PARENT_SHA256!r},'controller_command':command,'helpers_sha256':{{program.name:digest(program)}},"
        f"'environment':{queue.contract.ENVIRONMENT!r},'predecessor_acceptance':{{'seal_sha256':a.seal_sha256}},"
        "'allocation_predecessor':json.loads(seal.read_text())}\n"
        " binding.write_text(json.dumps(saved));(work/'freeze-argv.json').write_text(json.dumps(vars(a)))\n"
        "elif a.phase=='check':assert binding.is_file()\n"
        "elif a.phase=='start':\n"
        " saved=json.loads(binding.read_text());marker=uuid.uuid4().hex\n"
        " proof='external-hidden-parent';env=dict(os.environ)\n"
        f" env[{queue.contract.CONTROLLER_MARKER!r}]=marker;env[{queue.owner.PARENT_PROOF_MARKER!r}]=proof\n"
        " child=subprocess.Popen(saved['controller_command'],cwd=issued,env=env,start_new_session=True)\n"
        " proc=Path('/proc')/str(child.pid)\n"
        " while [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\\0') if x]"
        "!=saved['controller_command']:poll.poll(1)\n"
        " fields=(proc/'stat').read_text().rsplit(')',1)[1].split()\n"
        " generation={'pid':child.pid,'parent':int(fields[1]),'group':int(fields[2]),'session':int(fields[3]),"
        "'start_ticks':fields[19],'status':fields[0],'command':saved['controller_command']}\n"
        " intent={'status':'acknowledged','pid':child.pid,'start_ticks':fields[19],'generation':generation,"
        "'binding_sha256':digest(binding),'command':saved['controller_command'],'marker':marker,'parent_proof':proof}\n"
        " (issued.parent/'controller.json').write_text(json.dumps(intent))\n"
        " (work/'owner.pid').write_text(str(child.pid))\n"
    )
    controller = inherited / "ns_training_path_lr_gpu_controller.py"
    controller.write_text(script)
    (sealer / "prepare_ns_training_path_lr_predecessor.py").write_text(script)
    return inherited, sealer


def issued_owner(root: Path, monkeypatch, outcome: str) -> tuple[dict, Path]:
    binding, helpers, state_path, _, _ = queue.contract.locations(root)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    program = helpers / "ns_training_path_lr_gpu_controller.py"
    program.write_text("# issued actual-LR external fixture\n")
    candidate = root / "candidate"
    candidate.mkdir()
    original_decisions = {
        "accepted_cases": ["original-accepted"],
        "rejected_cases": ["original-rejected"],
        "all_cases_accepted": False,
    }
    frozen = {
        "candidate": str(candidate),
        "helpers_sha256": {program.name: sha256(program)},
        "controller_command": queue.owner.controller_command(root),
        "cpu_receipt_sha256": queue.PARENT_SHA256,
        "environment": queue.contract.ENVIRONMENT,
        "predecessor_acceptance": {"seal_sha256": "original-seal"},
        "allocation_predecessor": {"original_decisions": original_decisions},
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
        "parent_proof": "accepted-hidden-parent",
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
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "marked_descendants_absent": True,
        "keeper_release_complete": True,
        "restored_keeper": {"external_fixture": "restored"},
    }
    if outcome == "cleanup_failed":
        saved["cleanup_exit_code"] = 7
    write(state_path, saved)
    live = {"controller_live": {own["pid"]: own}, "validator_live": {child["pid"]: child}}.get(outcome, {})
    if outcome == "substituted":
        live = {own["pid"]: own | {"start_ticks": "substituted-generation"}}
    monkeypatch.setattr(queue.handoff, "observed", lambda pid: live.get(pid))
    monkeypatch.setattr(queue.predecessor.followup, "process_context", lambda pid, *args: live[pid])
    monkeypatch.setattr(
        queue.handoff, "marker_processes", lambda *args: {930001: {}} if outcome == "marked_rank" else {}
    )
    if outcome == "consumed":
        _, attempt, _, _ = queue.controller.terminal_paths(root, intent)
        write(attempt, {"status": "consumed-original-claim"})
    return {
        "issued_binding_sha256": sha256(binding),
        "issued_controller": intent,
        "predecessor_seal_sha256": "original-seal",
        "original_decisions": original_decisions,
        "phases": [],
    }, state_path


def terminal_program(root: Path, state: dict, fault: str) -> Path:
    binding, helpers, state_path, _, _ = queue.contract.locations(root)
    program = helpers / "ns_training_path_lr_gpu_controller.py"
    _, attempt, outcome, _ = queue.controller.terminal_paths(root, state["issued_controller"])
    program.write_text(
        "import argparse,hashlib,json,os,select\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--phase');p.add_argument('--out');a=p.parse_args()\n"
        f"root=Path(a.root);work=root/{queue.WORK!r};poll=select.poll()\n"
        "while json.loads((work/'state.json').read_text())['active_phase'].get('pid')!=os.getpid():poll.poll(1)\n"
        "state=json.loads((work/'state.json').read_text())\n"
        "assert a.phase=='terminal' and a.out==state['terminal_expected'] and os.environ['CUDA_VISIBLE_DEVICES']==''\n"
        "digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()\n"
        f"attempt=Path({str(attempt)!r});outcome=Path({str(outcome)!r})\n"
        "attempt.write_text(json.dumps({'status':'reconstruction_claimed'}))\n"
        f"value={{'status':{queue.controller.TERMINAL_STATUS!r},'binding_sha256':state['issued_binding_sha256'],"
        "'controller_pid_absent':state['issued_controller']['pid'],"
        "'controller_start_ticks':state['issued_controller']['start_ticks'],"
        f"'verifier_sha256':{'\"changed\"' if fault == 'verifier' else 'digest(__file__)'},"
        "'predecessor_acceptance':{'seal_sha256':state['predecessor_seal_sha256']},'production_authorized':False,"
        "'evidence_sha256':{},'accepted_cases':['accepted'],'rejected_cases':['rejected'],"
        "'all_cases_accepted':False,'reconstruction_intent_sha256':digest(attempt)}\n"
        "Path(a.out).write_text(json.dumps(value))\n"
        f"outcome.write_text(json.dumps({{'status':{'\"needs_attention\"' if fault == 'outcome' else '\"complete\"'},"
        "'output':a.out,'terminal_sha256':digest(a.out),'reconstruction_intent_sha256':digest(attempt)}))\n"
    )
    frozen = read(binding)
    frozen["helpers_sha256"] = {program.name: sha256(program)}
    write(binding, frozen)
    state["issued_binding_sha256"] = sha256(binding)
    state["issued_controller"]["binding_sha256"] = sha256(binding)
    write(helpers.parent / "controller.json", state["issued_controller"])
    saved = read(state_path)
    write(state_path, saved | {"binding_sha256": sha256(binding)})
    state["terminal_expected"] = str(queue.controller.terminal_paths(root, state["issued_controller"])[0])
    return program


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
    shutil.copy2(out / "tracking/ns_training_path_lr_predecessor.py", mutable)
    probe = root / "archive_probe.py"
    probe.write_text(
        "import importlib.util,json,sys\nfrom pathlib import Path\n"
        "root,receipt,parent,mutable=map(Path,sys.argv[1:5]);fault=sys.argv[5]\n"
        "import ns_training_path_lr_queue as queue\n"
        "queue.parent=lambda root:queue.read(parent)\n"
        "if fault=='mutable_import':\n"
        " spec=importlib.util.spec_from_file_location('ns_training_path_lr_predecessor',mutable)\n"
        " mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)\n"
        "cpu=queue.accepted(root,receipt)\n"
        "print(json.dumps({'status':'archived_training_path_lr_queue_roundtrip','helpers':len(cpu['helpers_sha256'])}))\n"
    )
    return probe


def fail_spawn(*args, **kwargs):
    raise OSError("External training-path queue phase spawn failed")


def process_environment(helpers: Path) -> dict:
    return (
        os.environ
        | queue.ENVIRONMENT
        | {"PYTHONPATH": f"{helpers}:/opt/conda/envs/torch-base/lib/python3.13/site-packages"}
    )
