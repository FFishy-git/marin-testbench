# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise historical acceptance and fresh physical-node succession without CUDA."""

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import fused_attribution_production_contract as contract
import fused_attribution_production_handoff as handoff
import fused_attribution_validation_contract as validation
import owned_output_handoff as processes
import pytest
import selected_lr_failure_handoff as manager


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(path: Path, files: dict[str, Path], candidate: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as bundle:
        for name, source in files.items():
            bundle.add(source, arcname=name, recursive=False)
        if candidate is not None:
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)


def candidate_sources(root: Path, name: str) -> tuple[Path, dict]:
    candidate = root / name
    text(candidate / "experiment/train.py", "def train():\n    return None\n")
    text(candidate / "artifacts/pinned-sources-materialized/numerics.py", "VALUE = 1\n")
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized")
    return candidate, {
        "experiment/train.py": digest(candidate / "experiment/train.py"),
        "artifacts/pinned-sources-materialized/numerics.py": digest(
            candidate / "artifacts/pinned-sources-materialized/numerics.py"
        ),
    }


def process_file(proc: Path, pid: int, command: list[str], parent: int = 1, marker: dict | None = None) -> None:
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["S", str(parent), str(pid), str(pid), *(["0"] * 15), str(100000 + pid)]
    (directory / "stat").write_text(f"{pid} (cpu fixture) {' '.join(fields)}")
    (directory / "cmdline").write_bytes(b"\0".join(argument.encode() for argument in command) + b"\0")
    (directory / "environ").write_bytes(
        b"\0".join(f"{name}={value}".encode() for name, value in (marker or {}).items()) + b"\0"
    )


@pytest.fixture
def machine(tmp_path, monkeypatch):
    root, proc = tmp_path / "campaign", tmp_path / "proc"
    (root / "run-preparation").mkdir(parents=True)
    proc.mkdir()
    pointer = text(tmp_path / "keeper.pid", "7000")
    source = text(tmp_path / "manager-source.sh", "# preserved CPU manager fixture\n")
    current = text(tmp_path / "manager-current.sh", source.read_text())
    lock = text(tmp_path / "manager.lock", "")
    monkeypatch.setattr(processes, "PROC_ROOT", proc)
    monkeypatch.setattr(processes, "KEEPER_PID", pointer)
    monkeypatch.setattr(manager, "MANAGER_LOCK", lock)
    monkeypatch.setattr(manager, "MANAGER_SOURCE", source)
    monkeypatch.setattr(manager, "KEEPER", str(current))
    machine = SimpleNamespace(
        root=root,
        proc=proc,
        lock=lock,
        node=validation.NODE,
        devices=[[str(index), f"GPU-{index}", "NVIDIA A100-SXM4-40GB"] for index in range(8)],
        occupied=[[str(7001 + index), f"GPU-{index}"] for index in range(8)],
    )

    def output(command, **kwargs):
        if command == ["hostname", "-I"]:
            return f"{machine.node}\n"
        if command == ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"]:
            return "\n".join(", ".join(row) for row in machine.devices)
        if command == ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]:
            return "\n".join(", ".join(row) for row in machine.occupied)
        raise AssertionError(f"Unexpected external command in CPU handoff: {command}")

    monkeypatch.setattr(subprocess, "check_output", output)
    launcher = [
        processes.PYTHON,
        processes.TORCHRUN,
        "--standalone",
        "--nproc-per-node=8",
        processes.WORKLOAD,
        *processes.WORKLOAD_ARGS,
    ]
    process_file(proc, 7000, launcher)
    for pid in range(7001, 7009):
        process_file(proc, pid, [processes.PYTHON, "-u", processes.WORKLOAD, *processes.WORKLOAD_ARGS], parent=7000)
    machine.keeper = json.loads(json.dumps(processes.keeper_observation()))
    return machine


def parent_validation(machine) -> SimpleNamespace:
    root = machine.root
    binding_path, helpers, state_path, work, receipt_path = validation.locations(root)
    candidate, files = candidate_sources(root, "validated-candidate")
    helpers.mkdir(parents=True)
    controller = text(
        helpers / "fused_attribution_validation_controller.py",
        "import json,os,sys\nfrom pathlib import Path\n"
        "def binding(root):\n"
        "    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':\n"
        "        raise RuntimeError('CPU reader requires CUDA hidden')\n"
        "    with (root/'executions.log').open('a') as stream: stream.write('binding\\n')\n"
        f"    return json.loads((root/{str(binding_path.relative_to(root))!r}).read_text())\n"
        "if __name__=='__main__':\n"
        "    print(json.dumps(binding(Path(sys.argv[sys.argv.index('--root')+1]))))\n",
    )
    text(
        helpers / "fused_attribution_validation.py",
        "import json,os\nfrom pathlib import Path\n"
        "def reconstruct(root,candidate,receipt):\n"
        "    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise RuntimeError('CUDA visible')\n"
        "    with (root/'executions.log').open('a') as stream: stream.write('reconstruct\\n')\n"
        "    action=root/'replace-source.json'\n"
        "    if action.exists():\n"
        "        target=Path(json.loads(action.read_text())['path'])\n"
        "        target.write_text(target.read_text()+'\\n# replaced during reconstruction\\n')\n"
        "    return json.loads(receipt.read_text())\n",
    )
    hashes = {path.name: digest(path) for path in helpers.iterdir()}
    prepared = {
        "parent_metadata_sha256": {},
        "helpers_sha256": hashes,
    }
    prep_path = candidate / validation.PREPARATION
    save(prep_path, prepared)
    data = root / "data/experiment_receipt.json"
    save(data, {"fixture": "CPU"})
    cpu_path, controller_cpu_path = root / "cpu/receipt.json", root / "controller-cpu/receipt.json"
    cpu_archive, controller_archive = cpu_path.parent / "source.tar", controller_cpu_path.parent / "source.tar"
    cpu_log, controller_log = text(cpu_path.parent / "cpu.log", "passed\n"), text(
        controller_cpu_path.parent / "cpu.log", "passed\n"
    )
    contents = {name: candidate / name for name in files} | {validation.PREPARATION: prep_path}
    helper_contents = {f"tracking/{name}": helpers / name for name in hashes}
    archive(cpu_archive, contents | helper_contents, candidate)
    archive(controller_archive, helper_contents)
    common_cpu = {"root": str(root), "exit_code": 0, "cuda_visible_devices": ""}
    save(
        cpu_path,
        common_cpu
        | {
            "status": "fused_attribution_validation_cpu_verified_gpu_denied",
            "candidate": str(candidate),
            "files_sha256": files,
            "preparation_sha256": digest(prep_path),
            "helpers_sha256": hashes,
            "log": str(cpu_log),
            "log_sha256": digest(cpu_log),
            "archive": str(cpu_archive),
            "archive_sha256": digest(cpu_archive),
        },
    )
    save(
        controller_cpu_path,
        common_cpu
        | {
            "status": "fused_attribution_validation_controller_cpu_verified_no_gpu_binding",
            "helpers_sha256": hashes,
            "fixture_files_sha256": {},
            "log": str(controller_log),
            "log_sha256": digest(controller_log),
            "archive": str(controller_archive),
            "archive_sha256": digest(controller_archive),
        },
    )
    source_archive = root / "run-preparation/source-validation.tar"
    archive(
        source_archive,
        contents
        | helper_contents
        | {"candidate-cpu-receipt.json": cpu_path, "controller-cpu-receipt.json": controller_cpu_path},
        candidate,
    )
    frozen = {
        "status": validation.FROZEN_STATUS,
        "root": str(root),
        "helpers": str(helpers),
        "candidate": str(candidate),
        "helpers_sha256": hashes,
        "files_sha256": files,
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": digest(cpu_path),
        "controller_cpu_receipt": str(controller_cpu_path),
        "controller_cpu_receipt_sha256": digest(controller_cpu_path),
        "archive": str(source_archive),
        "archive_sha256": digest(source_archive),
        "preparation_sha256": digest(prep_path),
        "data_receipt_sha256": digest(data),
        "receipt": str(receipt_path),
    }
    save(binding_path, frozen)
    owner = {"controller_pid": 8000, "controller_start_ticks": "108000"}
    gpu_commands = []
    for index in range(27):
        log = text(work / f"command-{index}.log", f"CPU fixture original command {index}\n")
        gpu_commands.append({"exit_code": 0, "log": str(log), "log_sha256": digest(log)})
    trials = [{"checks": {f"check-{index}": True for index in range(count)}} for _ in range(3) for count in (8, 11, 12)]
    receipt = {
        "status": handoff.NUMERICAL_STATUS,
        "owner": owner,
        "execution_binding_sha256": digest(binding_path),
        "gpu_commands": gpu_commands,
        "trials": trials,
    }
    save(receipt_path, receipt)
    log = text(root / "validation/controller.log", "original controller terminal\n")
    state = {
        "status": "passed",
        "node": validation.NODE,
        "candidate": str(candidate),
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "keeper_release_complete": True,
        "marked_descendants_absent": True,
        "binding_sha256": digest(binding_path),
        "validation_sha256": digest(receipt_path),
        "command": validation.validator_command(root, candidate),
        **owner,
        "child_pid": 8100,
        "child_marker": "original-validator-child",
        "restored_keeper": machine.keeper,
        "log": str(log),
    }
    save(state_path, state)
    intent = {
        "status": "acknowledged",
        "pid": owner["controller_pid"],
        "start_ticks": owner["controller_start_ticks"],
        "command": validation.controller_command(root),
        "binding_sha256": digest(binding_path),
        "marker": "original-controller",
    }
    intent["generation"] = {
        "pid": 8000,
        "start_ticks": "108000",
        "group": 8000,
        "session": 8000,
        "parent": 1,
        "status": "S",
        "command": intent["command"],
    }
    save(helpers.parent / "controller.json", intent)
    for pid in range(7000, 7009):
        (machine.proc / str(pid) / "environ").write_bytes(
            f"{validation.CONTROLLER_MARKER}={intent['marker']}\0".encode()
        )
    return SimpleNamespace(
        root=root,
        frozen=frozen,
        state=state,
        receipt=receipt,
        intent=intent,
        binding=binding_path,
        state_path=state_path,
        receipt_path=receipt_path,
        helpers=helpers,
        controller=controller,
        archive=source_archive,
        out=root / "acceptance/receipt.json",
    )


def test_validation_acceptance_reconstructs_and_remains_historical_after_allocation_changes(machine):
    parent = parent_validation(machine)
    accepted = handoff.validation_acceptance(parent.root, parent.out)
    assert accepted["owner"] == {"controller_pid": 8000, "controller_start_ticks": "108000"}
    assert (accepted["commands"], accepted["checks"]) == (27, 93)
    assert parent.out.is_file() and accepted["time"]
    assert (parent.root / "executions.log").read_text().splitlines() == ["binding", "binding", "reconstruct"]
    # Numerical acceptance must remain usable after its historical keeper has been released.
    for path in machine.proc.iterdir():
        for entry in path.iterdir():
            entry.unlink()
        path.rmdir()
    assert handoff.require_acceptance(parent.root, parent.out) == accepted


@pytest.mark.parametrize(
    "failure", ["terminal_failure", "partial_release", "stale_owner", "live_controller", "child_marker"]
)
def test_validation_acceptance_refuses_failed_or_live_original_before_executing_helpers(machine, failure):
    parent = parent_validation(machine)
    if failure == "terminal_failure":
        parent.state.update(status="needs_attention", exit_code=1, cleanup_exit_code=1, error="continuation mismatch")
        parent.receipt.update(status="needs_attention", error="rank_0.model.body.emb.weight")
        save(parent.receipt_path, parent.receipt)
        parent.state["validation_sha256"] = digest(parent.receipt_path)
    elif failure == "partial_release":
        parent.state["keeper_release_complete"] = False
    elif failure == "stale_owner":
        parent.receipt["owner"]["controller_start_ticks"] = "stale"
        save(parent.receipt_path, parent.receipt)
        parent.state["validation_sha256"] = digest(parent.receipt_path)
    elif failure == "live_controller":
        process_file(machine.proc, 8000, parent.intent["command"])
    else:
        process_file(machine.proc, 8200, ["unrelated"], marker={validation.CHILD_MARKER: parent.state["child_marker"]})
    save(parent.state_path, parent.state)
    with pytest.raises(RuntimeError):
        handoff.validation_acceptance(parent.root, parent.out)
    assert not parent.out.exists()
    assert not (parent.root / "executions.log").exists()


@pytest.mark.parametrize("drift", ["helper", "archive", "undeclared_import"])
def test_validation_acceptance_authenticates_archive_and_import_closure_before_execution(machine, drift):
    parent = parent_validation(machine)
    if drift == "helper":
        text(parent.controller, parent.controller.read_text() + "\nraise RuntimeError('untrusted import')\n")
    elif drift == "archive":
        parent.archive.write_bytes(b"replaced archive")
    else:
        text(parent.controller, parent.controller.read_text() + "\nimport unarchived_helper\n")
        parent.frozen["helpers_sha256"][parent.controller.name] = digest(parent.controller)
        save(parent.binding, parent.frozen)
        parent.state["binding_sha256"] = digest(parent.binding)
        parent.receipt["execution_binding_sha256"] = digest(parent.binding)
        save(parent.receipt_path, parent.receipt)
        parent.state["validation_sha256"] = digest(parent.receipt_path)
        save(parent.state_path, parent.state)
        parent.intent["binding_sha256"] = digest(parent.binding)
        save(parent.helpers.parent / "controller.json", parent.intent)
    with pytest.raises(ValueError):
        handoff.validation_acceptance(parent.root, parent.out)
    assert not parent.out.exists() and not (parent.root / "executions.log").exists()


def test_validation_acceptance_refuses_helper_replacement_during_independent_reconstruction(machine):
    parent = parent_validation(machine)
    save(parent.root / "replace-source.json", {"path": str(parent.controller)})
    with pytest.raises(subprocess.CalledProcessError):
        handoff.validation_acceptance(parent.root, parent.out)
    assert not parent.out.exists()
    assert (parent.root / "executions.log").read_text().splitlines()[-1] == "reconstruct"


def test_saved_acceptance_reader_has_no_torch_or_preparer_import_prerequisite(machine):
    parent = parent_validation(machine)
    accepted = handoff.validation_acceptance(parent.root, parent.out)
    script = (
        "import importlib.abc,json,sys\nfrom pathlib import Path\n"
        "class DenyNumericalImports(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self,fullname,path=None,target=None):\n"
        "        if fullname.split('.')[0] in ('torch','prepare_fused_attribution_production'):\n"
        "            raise RuntimeError('Saved reader imported a numerical/preparer prerequisite')\n"
        "sys.meta_path.insert(0,DenyNumericalImports())\n"
        "from fused_attribution_production_handoff import require_acceptance\n"
        "print(json.dumps(require_acceptance(Path(sys.argv[1]),Path(sys.argv[2]))))\n"
    )
    result = subprocess.run(
        [validation.PYTHON, "-P", "-B", "-c", script, str(parent.root), str(parent.out)],
        env=os.environ
        | {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(Path(handoff.__file__).parent),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == accepted


def test_saved_acceptance_cannot_rebind_changed_evidence_by_dropping_it(machine):
    parent = parent_validation(machine)
    accepted = handoff.validation_acceptance(parent.root, parent.out)
    log = Path(parent.receipt["gpu_commands"][0]["log"])
    text(log, "changed original log\n")
    accepted["evidence_sha256"].pop(str(log.relative_to(parent.root)))
    save(parent.out, accepted)
    with pytest.raises(ValueError):
        handoff.require_acceptance(parent.root, parent.out)
    assert log.read_text() == "changed original log\n"


def production_predecessor(machine, wave: str) -> SimpleNamespace:
    root = machine.root
    if wave == "wave12":
        binding_path = root / "run-preparation/wave12-launch-binding.json"
        helpers = root / "controllers/wave12-launch-20261117/tracking"
        launch_path = root / "run-preparation/wave12-launch-state.json"
        spec = {
            "families": ["muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused"],
            "nodes": ["28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139"],
        }
    else:
        binding_path, helpers, launch_path, _ = contract.paths(root, wave)
        spec = contract.wave_spec(wave)
    candidate, files = candidate_sources(root, f"{wave}-candidate")
    helpers.mkdir(parents=True)
    text(
        helpers / "fused_attribution_production.py",
        "import json,os\nfrom pathlib import Path\n"
        "def probe(root,wave,family):\n"
        "    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise RuntimeError('CUDA visible')\n"
        "    with (root/'probes.log').open('a') as stream: stream.write(wave+' '+family+'\\n')\n"
        "    receipt=json.loads((root/f'run-preparation/{wave}-workers/{family}.json').read_text())\n"
        "    return {'status':'complete','receipt':receipt,'original_worker_absent':True,\n"
        "            'original_child_absent':True,'keeper':receipt['restored_keeper']}\n",
    )
    if wave == "wave12":
        text(
            helpers / "selected_lr_production.py",
            "import json,os\n"
            "def probe_arm(root,arm):\n"
            "    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise RuntimeError('CUDA visible')\n"
            "    with (root/'probes.log').open('a') as stream: stream.write('wave12 '+arm+'\\n')\n"
            "    receipt=json.loads((root/f'run-preparation/wave12-workers/{arm}.json').read_text())\n"
            "    return {'status':'complete','receipt':receipt}\n",
        )
        text(
            helpers / "selected_lr_run_history.py",
            "import json\n"
            "def binding(root):\n"
            "    return json.loads((root/'run-preparation/wave12-launch-binding.json').read_text())\n",
        )
    text(
        helpers / "fused_attribution_run_history.py",
        "import json\n"
        "def require_completed_wave(root,wave):\n"
        "    result=json.loads((root/f'completion-{wave}.json').read_text())\n"
        "    if result['status']!='attribution_wave_completed_analyzed_cloud_verified':\n"
        "        raise RuntimeError('Completion remains pending')\n"
        "    return result\n",
    )
    hashes = {path.name: digest(path) for path in helpers.iterdir()}
    prep_path = candidate / ("preparation.json" if wave == "wave12" else contract.PREPARATION)
    save(prep_path, {"files_sha256": files, "parent_metadata_sha256": {}, "helpers_sha256": {}})
    receipts = {}
    for key in ("cpu_receipt", "controller_cpu_receipt", "validation_acceptance"):
        path = root / f"{wave}-evidence/{key}.json"
        save(path, {"fixture": key})
        receipts[key] = str(path)
        receipts[f"{key}_sha256"] = digest(path)
    source_archive = root / f"{wave}-evidence/source.tar"
    members = {name: candidate / name for name in files} | {prep_path.name: prep_path}
    members.update({f"tracking/{name}": helpers / name for name in hashes})
    if wave != "wave12":
        members.update(
            {
                "candidate-cpu-receipt.json": Path(receipts["cpu_receipt"]),
                "controller-cpu-receipt.json": Path(receipts["controller_cpu_receipt"]),
                "validation-acceptance.json": Path(receipts["validation_acceptance"]),
            }
        )
    archive(source_archive, members, candidate)
    release = {
        "status": contract.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "files_sha256": files,
        "helpers_sha256": hashes,
        "archive": str(source_archive),
        "archive_sha256": digest(source_archive),
        "preparation_sha256": digest(prep_path),
        "families": spec["families"],
        "nodes": spec["nodes"],
        "commands": {
            node: ["original-training", family] for family, node in zip(spec["families"], spec["nodes"], strict=True)
        },
        **receipts,
    }
    if wave == "wave12":
        release["arms"] = spec["families"]
    save(binding_path, release)
    family = spec["families"][spec["nodes"].index(machine.node)]
    state_path = (
        root / f"run-preparation/wave12-workers/{family}.json"
        if wave == "wave12"
        else contract.worker_path(root, wave, family)
    )
    worker_command = (
        [
            validation.PYTHON,
            str(helpers / "selected_lr_production.py"),
            "--root",
            str(root),
            "--phase",
            "worker",
            "--arm",
            family,
        ]
        if wave == "wave12"
        else contract.worker_command(root, wave, family)
    )
    log = text(root / f"runs/{wave}/{family}.log", "original complete training\n")
    state = {
        "status": "complete",
        "arm": family,
        "node": machine.node,
        "command": release["commands"][machine.node],
        "release_binding_sha256": digest(binding_path),
        "worker_command": worker_command,
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "worker_pid": 9000,
        "worker_start_ticks": "109000",
        "worker_marker": f"{wave}-worker",
        "child_pid": 9100,
        "child_marker": f"{wave}-child",
        "restored_keeper": machine.keeper,
        "log": str(log),
        "log_sha256": digest(log),
    }
    save(state_path, state)
    intent = {
        "status": "acknowledged",
        "pid": 9000,
        "start_ticks": "109000",
        "command": state["worker_command"],
        "worker_marker": state["worker_marker"],
        "release_binding_sha256": state["release_binding_sha256"],
        "generation": {
            "pid": 9000,
            "start_ticks": "109000",
            "group": 9000,
            "session": 9000,
            "parent": 1,
            "status": "S",
            "command": state["worker_command"],
        },
        "starter": {"pid": 9200},
    }
    save(state_path.with_suffix(".start.json"), intent)
    save(
        launch_path,
        {
            "status": "launched",
            "binding_sha256": digest(binding_path),
            "worker_markers": {family: state["worker_marker"]},
        },
    )
    completion_files = [
        root / f"run-preparation/monitor-{wave}/latest.json",
        root / f"run-preparation/analysis-{wave}.json",
        root / f"run-preparation/swanlab-upload-{wave}.json",
    ]
    for previous_family in spec["families"]:
        directory = root / "runs" / wave / previous_family
        completion_files.extend(
            [
                directory / "run.json",
                directory / "metrics_from_000000.jsonl",
                directory / "checkpoints/step_005120/complete.json",
            ]
        )
    for path in completion_files:
        text(path, "{}\n")
    completion = {
        "status": handoff.COMPLETION_STATUS,
        "wave": wave,
        "evidence_sha256": {str(path.relative_to(root)): digest(path) for path in completion_files},
    }
    save(root / f"completion-{wave}.json", completion)
    for pid in range(7000, 7009):
        marker = "OUTPUT_MARIN_WAVE12_WORKER_ID" if wave == "wave12" else contract.WORKER_MARKER
        (machine.proc / str(pid) / "environ").write_bytes(f"{marker}={state['worker_marker']}\0".encode())
    return SimpleNamespace(
        root=root,
        family=family,
        release=release,
        binding=binding_path,
        helpers=helpers,
        state=state,
        state_path=state_path,
        intent=intent,
        archive=source_archive,
    )


@pytest.mark.parametrize("wave,previous", [("wave14", "wave13"), ("wave15", "wave14")])
def test_allocation_follows_rotated_same_node_predecessor_with_inherited_keeper_marker(machine, wave, previous):
    family = "muont"
    machine.node = contract.node_for(wave, family)
    predecessor = production_predecessor(machine, previous)
    # The shared descriptor is already held; observation must not reacquire a second flock.
    with manager.keeper_guard() as lock:
        actual = handoff.allocation_observation(machine.root, wave, family, manager_lock=lock)
        checked = handoff.allocation_observation(machine.root, wave, family, expected=actual, manager_lock=lock)
    assert actual == checked
    assert actual["predecessor_family"] == predecessor.family != family
    assert actual["node"] == machine.node
    assert json.loads(json.dumps(actual["keeper"])) == machine.keeper
    assert not contract.paths(machine.root, wave)[0].exists()
    assert (machine.root / "probes.log").read_text().splitlines() == [f"{previous} {predecessor.family}"] * 2


@pytest.mark.parametrize(
    "failure", ["mapping", "marker", "live_worker", "stale_ack", "terminal_failure", "pending_cloud"]
)
def test_allocation_refuses_predecessor_drift_without_changing_keeper_or_issued_files(machine, failure):
    machine.node = contract.node_for("wave14", "muont")
    predecessor = production_predecessor(machine, "wave13")
    if failure == "mapping":
        machine.occupied[0][1], machine.occupied[1][1] = machine.occupied[1][1], machine.occupied[0][1]
    elif failure == "marker":
        process_file(
            machine.proc, 9300, ["unrelated"], marker={contract.WORKER_MARKER: predecessor.state["worker_marker"]}
        )
    elif failure == "live_worker":
        process_file(machine.proc, 9000, predecessor.state["worker_command"])
    elif failure == "stale_ack":
        predecessor.intent["generation"]["start_ticks"] = "reused"
        save(predecessor.state_path.with_suffix(".start.json"), predecessor.intent)
    elif failure == "terminal_failure":
        predecessor.state.update(status="needs_attention", exit_code=1, error="original training failed")
        save(predecessor.state_path, predecessor.state)
    else:
        save(machine.root / "completion-wave13.json", {"status": "cloud_pending"})
    source_before = predecessor.archive.read_bytes()
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        handoff.allocation_observation(machine.root, "wave14", "muont")
    assert predecessor.archive.read_bytes() == source_before
    assert processes.keeper_observation()["pid"] == 7000
    assert not contract.paths(machine.root, "wave14")[0].exists()


def test_allocation_rejects_hash_valid_but_wrong_archive_members_before_predecessor_import(machine):
    machine.node = contract.node_for("wave14", "muont")
    predecessor = production_predecessor(machine, "wave13")
    archive(predecessor.archive, {"unrelated.json": predecessor.binding})
    predecessor.release["archive_sha256"] = digest(predecessor.archive)
    save(predecessor.binding, predecessor.release)
    with pytest.raises(ValueError):
        handoff.allocation_observation(machine.root, "wave14", "muont")
    assert not (machine.root / "probes.log").exists()
    assert processes.keeper_observation()["occupied"] == machine.keeper["occupied"]


def test_allocation_refuses_validation_failure_instead_of_adopting_old_muont_keeper(machine):
    parent = parent_validation(machine)
    parent.state.update(status="needs_attention", exit_code=1, cleanup_exit_code=1, error="cached-root continuation")
    save(parent.state_path, parent.state)
    with pytest.raises(RuntimeError):
        handoff.allocation_observation(machine.root, "wave13", "muont")
    assert not (machine.root / "executions.log").exists()
    assert processes.keeper_observation()["pid"] == 7000


def test_allocation_rejects_wrong_manager_inode_while_original_lock_stays_held(machine):
    machine.node = contract.node_for("wave14", "muont")
    production_predecessor(machine, "wave13")
    unrelated = text(machine.root / "unrelated.lock", "")
    with manager.keeper_guard(), unrelated.open("a") as wrong:
        with pytest.raises(RuntimeError):
            handoff.allocation_observation(machine.root, "wave14", "muont", manager_lock=wrong.fileno())
    assert not (machine.root / "probes.log").exists()
    assert processes.keeper_observation()["pid"] == 7000


@pytest.mark.parametrize("family", ["outer_leftdiag_fused", "null_leftdiag_fused"])
def test_wave13_allocation_uses_frozen_wave12_physical_node_predecessor(machine, family):
    machine.node = contract.node_for("wave13", family)
    predecessor = production_predecessor(machine, "wave12")
    actual = handoff.allocation_observation(machine.root, "wave13", family)
    assert actual["predecessor_wave"] == "wave12"
    assert actual["predecessor_family"] == predecessor.family
    assert actual["node"] == machine.node and actual["keeper"]["occupied"] == machine.keeper["occupied"]
    assert (machine.root / "probes.log").read_text().splitlines() == [f"wave12 {predecessor.family}"]
    assert not contract.paths(machine.root, "wave13")[0].exists()


def test_wave13_muont_allocation_uses_new_validation_restoration_without_prior_wave12_binding(machine):
    parent = parent_validation(machine)
    actual = handoff.allocation_observation(machine.root, "wave13", "muont")
    assert actual["predecessor_wave"] == "attribution_validation"
    assert json.loads(json.dumps(actual["keeper"])) == parent.state["restored_keeper"]
    assert not (machine.root / "run-preparation/wave12-launch-binding.json").exists()
    assert not parent.out.exists()


@pytest.mark.parametrize("change", ["generation", "physical_mapping", "source_addition"])
def test_saved_identity_or_keeper_reuse_cannot_cross_allocation_acceptance(machine, change):
    machine.node = contract.node_for("wave14", "muont")
    predecessor = production_predecessor(machine, "wave13")
    accepted = handoff.allocation_observation(machine.root, "wave14", "muont")
    if change == "generation":
        path = machine.proc / "7001/stat"
        path.write_text(path.read_text().replace("107001", "207001"))
    elif change == "physical_mapping":
        machine.occupied[0][1], machine.occupied[1][1] = machine.occupied[1][1], machine.occupied[0][1]
    else:
        text(Path(predecessor.release["candidate"]) / "experiment/extra.py", "VALUE = 'unbound'\n")
    with pytest.raises((ValueError, RuntimeError)):
        handoff.allocation_observation(machine.root, "wave14", "muont", expected=accepted)
    assert not contract.paths(machine.root, "wave14")[0].exists()
