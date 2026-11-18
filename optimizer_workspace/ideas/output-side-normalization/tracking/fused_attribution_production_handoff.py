# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate immutable attribution numerical acceptance from fresh keeper succession."""

import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import owned_output_handoff as processes
from fused_attribution_production_contract import (
    ACCEPTANCE_STATUS,
    CHILD_MARKER,
    FROZEN_STATUS,
    PREPARATION,
    PYTHON,
    WORKER_MARKER,
    node_for,
    paths,
    wave_spec,
    worker_command,
    worker_path,
)
from fused_attribution_validation_contract import CHILD_MARKER as VALIDATION_CHILD_MARKER
from fused_attribution_validation_contract import CONTROLLER_MARKER as VALIDATION_CONTROLLER_MARKER
from fused_attribution_validation_contract import FROZEN_STATUS as VALIDATION_STATUS
from fused_attribution_validation_contract import NODE, controller_command, locations, validator_command
from launch_wave5 import read, sha256
from selected_lr_failure_handoff import keeper_guard, require_manager_lock, same_keeper
from watch_wave5 import write
from wave7_release import candidate_inventory, local_node, node_guard

NUMERICAL_STATUS = "passed_fused_attribution_actual_config_continuation_and_work_gates"
COMMANDS, CHECKS = 27, 93
COMPLETION_STATUS = "attribution_wave_completed_analyzed_cloud_verified"
EXTERNAL_MODULES = sys.stdlib_module_names | {"torch", "pytest", "swanlab", "dotenv", "experiment"}


def evidence_file(root: Path, path: Path) -> Path:
    if path != path.resolve() or not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError("Attribution handoff evidence must be a canonical regular file inside its root")
    return path


def evidence_hashes(root: Path, files: list[Path]) -> dict:
    return {str(path.relative_to(root)): sha256(evidence_file(root, path)) for path in files}


def require_evidence(root: Path, evidence: dict) -> None:
    if not evidence:
        raise ValueError("Attribution handoff has no bound evidence")
    for name, digest in evidence.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
            raise ValueError("Attribution evidence contains a noncanonical relative path")
        if sha256(evidence_file(root, root / relative)) != digest:
            raise ValueError("Attribution bound evidence changed")


def require_helper_closure(helpers: Path, hashes: dict, seeds: tuple[str, ...], candidate: Path) -> None:
    """Reject undeclared flat imports before executing any frozen helper."""
    if (
        helpers != helpers.resolve()
        or helpers.is_symlink()
        or not set(seeds) <= hashes.keys()
        or {path.name for path in helpers.iterdir() if path.name != "__pycache__"} != set(hashes)
    ):
        raise ValueError("Attribution handoff helper origin or required entry points differ")
    for name, digest in hashes.items():
        if Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != digest:
            raise ValueError("Attribution handoff helper bytes differ")
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse((helpers / name).read_bytes())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module else [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                head = module.split(".")[0]
                dependency = f"{head}.py"
                if isinstance(node, ast.ImportFrom) and node.level:
                    if dependency not in hashes and not (candidate / "experiment" / dependency).is_file():
                        raise ValueError(f"Attribution frozen relative import is not archived: {name}: {module}")
                    continue
                if dependency not in hashes and head not in EXTERNAL_MODULES:
                    raise ValueError(f"Attribution frozen helper has an unbound import: {name}: {module}")
                if dependency in hashes and head != module:
                    raise ValueError("Attribution flat helper imports a local package")


def frozen_call(root: Path, frozen: dict, script: str, arguments: list[str], evidence: dict) -> dict:
    require_evidence(root, evidence)
    helpers = Path(frozen["helpers"])
    guard = (
        "import hashlib,importlib.util,json,sys\nfrom pathlib import Path\n"
        "helpers=Path(sys.argv[1]); hashes=json.loads(sys.argv[2]); evidence=json.loads(sys.argv[3])\n"
        "root=Path(sys.argv[4])\n"
        "def require_bytes():\n"
        "    for name,digest in evidence.items():\n"
        "        path=root/name\n"
        "        actual=hashlib.sha256(path.read_bytes()).hexdigest()\n"
        "        if path.resolve()!=path or path.is_symlink() or actual!=digest:\n"
        "            raise RuntimeError('Frozen handoff evidence changed before or during execution')\n"
        "require_bytes()\n"
        "for name in hashes:\n"
        "    if name.endswith('.py'):\n"
        "        spec=importlib.util.find_spec(Path(name).stem)\n"
        "        if spec is None or spec.origin is None or Path(spec.origin).resolve()!=helpers/name:\n"
        "            raise RuntimeError('Frozen handoff helper resolves outside its archive')\n"
        "def require_origins():\n"
        "    for name in hashes:\n"
        "        module=sys.modules.get(Path(name).stem)\n"
        "        if name.endswith('.py') and module is not None:\n"
        "            if Path(module.__file__).resolve()!=helpers/name:\n"
        "                raise RuntimeError('Frozen handoff imported a different helper origin')\n"
        "sys.argv=[sys.argv[0],*sys.argv[5:]]\n"
    )
    checked = subprocess.run(
        [
            PYTHON,
            "-P",
            "-B",
            "-c",
            guard + script + "\nrequire_bytes()\n",
            str(helpers),
            json.dumps(frozen["helpers_sha256"]),
            json.dumps(evidence),
            str(root),
            *arguments,
        ],
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        check=True,
        timeout=3600,
    )
    require_evidence(root, evidence)
    return json.loads(checked.stdout)


def require_same_keeper(actual: dict, expected: dict) -> None:
    """Retain exact generations and canonical physical PID/GPU assignments."""
    if not same_keeper(actual, expected) or sorted(actual["occupied"]) != sorted(expected["occupied"]):
        raise RuntimeError("Attribution keeper generation or physical GPU assignment changed")


def require_markers_on_keeper(keeper: dict, markers: list[dict], absent_pids: list[int]) -> None:
    """Allow inherited predecessor markers only on the exact current keeper."""
    generation = {int(pid): process for pid, process in keeper["generation"].items()}
    for pid in absent_pids:
        processes.require_absent(pid)
    for marker in markers:
        if not marker["name"] or not marker["value"]:
            raise ValueError("Attribution predecessor marker identity is empty")
        for pid, process in processes.marker_processes(marker["name"], marker["value"]).items():
            if pid not in generation or not processes.same_generation(generation[pid], process):
                raise RuntimeError("Attribution predecessor marker exists outside its exact restored keeper")
    if any(not processes.same_generation(process, processes.observed(pid)) for pid, process in generation.items()):
        raise RuntimeError("Attribution keeper changed during predecessor inspection")


def validation_terminal(root: Path, frozen: dict) -> tuple[dict, dict]:
    state, receipt, intent = validation_records(root, frozen)
    if local_node() != NODE:
        raise RuntimeError("Attribution original validation must be inspected on its physical node")
    keeper = processes.keeper_observation()
    require_same_keeper(keeper, state["restored_keeper"])
    if not state["child_marker"] or processes.marker_processes(VALIDATION_CHILD_MARKER, state["child_marker"]):
        raise RuntimeError("Attribution original validator descendants remain")
    require_markers_on_keeper(
        keeper,
        [{"name": VALIDATION_CONTROLLER_MARKER, "value": intent["marker"]}],
        [state["controller_pid"], state["child_pid"]],
    )
    return state, receipt


def validation_records(root: Path, frozen: dict) -> tuple[dict, dict, dict]:
    """Authenticate historical terminal ownership without probing a superseded allocation."""
    binding_path, helpers, state_path, _, receipt_path = locations(root)
    state, receipt, intent = read(state_path), read(receipt_path), read(helpers.parent / "controller.json")
    owner = {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
    generation = intent["generation"]
    if (
        state["status"] != "passed"
        or state["node"] != NODE
        or state["candidate"] != frozen["candidate"]
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or any(key in state for key in ("error", "errors", "cleanup_error"))
        or state.get("keeper_release_complete") is not True
        or state.get("marked_descendants_absent") is not True
        or state["binding_sha256"] != sha256(binding_path)
        or state["validation_sha256"] != sha256(receipt_path)
        or state["command"] != validator_command(root, Path(frozen["candidate"]))
        or intent["status"] != "acknowledged"
        or not intent["marker"]
        or intent["pid"] != owner["controller_pid"]
        or intent["start_ticks"] != owner["controller_start_ticks"]
        or intent["command"] != controller_command(root)
        or intent["binding_sha256"] != state["binding_sha256"]
        or generation["pid"] != intent["pid"]
        or generation["start_ticks"] != intent["start_ticks"]
        or generation["command"] != intent["command"]
        or generation["group"] != intent["pid"]
        or generation["session"] != intent["pid"]
        or receipt["status"] != NUMERICAL_STATUS
        or any(key in receipt or key in intent for key in ("error", "errors", "cleanup_error"))
        or receipt["owner"] != owner
        or receipt["execution_binding_sha256"] != state["binding_sha256"]
        or len(receipt["gpu_commands"]) != COMMANDS
        or sum(value is True for trial in receipt["trials"] for value in trial["checks"].values()) != CHECKS
        or any(type(item["exit_code"]) is not int or item["exit_code"] != 0 for item in receipt["gpu_commands"])
    ):
        raise RuntimeError("Attribution original validation is live, incomplete or has stale terminal ownership")
    return state, receipt, intent


def validation_source(root: Path) -> dict:
    binding_path, helpers, _, _, _ = locations(root)
    frozen = read(binding_path)
    candidate = Path(frozen["candidate"])
    if (
        frozen["status"] != VALIDATION_STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or candidate != candidate.resolve()
        or candidate_inventory(candidate) != frozen["files_sha256"]
    ):
        raise ValueError("Attribution validation helper directory or frozen identity differs")
    require_helper_closure(
        helpers,
        frozen["helpers_sha256"],
        ("fused_attribution_validation_controller.py", "fused_attribution_validation.py"),
        candidate,
    )
    return frozen


def require_validation_source(root: Path, frozen: dict, evidence: dict) -> None:
    # Preparation is optional for saved acceptance readers; import it only for CPU source authentication.
    require_evidence(root, evidence)
    preparation = importlib.import_module("prepare_fused_attribution_production")
    if preparation.original(root) != frozen:
        raise ValueError("Attribution validation binding changed during source authentication")
    require_evidence(root, evidence)


def validation_evidence(root: Path, frozen: dict, receipt: dict) -> dict:
    binding_path, helpers, state_path, work, receipt_path = locations(root)
    candidate = Path(frozen["candidate"])
    files = [
        binding_path,
        state_path,
        receipt_path,
        helpers.parent / "controller.json",
        Path(frozen["archive"]),
        Path(frozen["cpu_receipt"]),
        Path(frozen["controller_cpu_receipt"]),
        root / "data/experiment_receipt.json",
        *[helpers / name for name in frozen["helpers_sha256"]],
        *[candidate / name for name in frozen["files_sha256"]],
        *candidate.glob("*.json"),
    ]
    state = read(state_path)
    files.append(Path(state["log"]))
    for key in ("cpu_receipt", "controller_cpu_receipt"):
        cpu = read(Path(frozen[key]))
        files.extend((Path(cpu["archive"]), Path(cpu["log"])))
    for command in receipt["gpu_commands"]:
        files.append(Path(command["log"]))
    # Rank tensor hashes are carried by the reconstructed receipt, not reloaded by every history reader.
    files.extend(path for path in work.rglob("*") if path.is_file() and path.suffix != ".pt")
    return evidence_hashes(root, files)


def validation_acceptance(root: Path, out: Path) -> dict:
    """Publish numerical acceptance only after original exit and locked exact restoration."""
    root, out = root.resolve(), out.resolve()
    if out.exists() or not out.is_relative_to(root):
        raise FileExistsError("Attribution acceptance needs an unused receipt inside the campaign root")
    with node_guard(root, NODE), keeper_guard():
        frozen = validation_source(root)
        helpers = Path(frozen["helpers"])
        require_helper_closure(
            helpers,
            frozen["helpers_sha256"],
            ("fused_attribution_validation_controller.py", "fused_attribution_validation.py"),
            Path(frozen["candidate"]),
        )
        state, receipt = validation_terminal(root, frozen)
        evidence = validation_evidence(root, frozen, receipt)
        require_validation_source(root, frozen, evidence)
        rebuilt = frozen_call(
            root,
            frozen,
            "from fused_attribution_validation_controller import binding\n"
            "from fused_attribution_validation import reconstruct\n"
            "require_origins()\n"
            "root=Path(sys.argv[1]); frozen=binding(root)\n"
            "actual=reconstruct(root,Path(frozen['candidate']),Path(frozen['receipt']))\n"
            "require_origins()\nprint(json.dumps(actual,allow_nan=False))\n",
            [str(root)],
            evidence,
        )
        if rebuilt != receipt:
            raise ValueError("Attribution independent numerical receipt differs")
        require_evidence(root, evidence)
        final_state, final_receipt = validation_terminal(root, frozen)
        if final_state != state or final_receipt != receipt:
            raise RuntimeError("Attribution terminal ownership changed during reconstruction")
        binding_path, _, state_path, _, receipt_path = locations(root)
        result = {
            "status": ACCEPTANCE_STATUS,
            "root": str(root),
            "binding_sha256": evidence[str(binding_path.relative_to(root))],
            "state_sha256": evidence[str(state_path.relative_to(root))],
            "receipt_sha256": evidence[str(receipt_path.relative_to(root))],
            "intent_sha256": evidence[str((helpers.parent / "controller.json").relative_to(root))],
            "owner": receipt["owner"],
            "commands": COMMANDS,
            "checks": CHECKS,
            "numerical_status": NUMERICAL_STATUS,
            "keeper": state["restored_keeper"],
            "evidence_sha256": evidence,
        }
        if out.is_relative_to(Path(frozen["candidate"])) or out.is_relative_to(helpers) or out.exists():
            raise FileExistsError("Attribution acceptance must not overwrite frozen source or prior evidence")
        out.parent.mkdir(parents=True, exist_ok=True)
        write(out, result)
    return require_acceptance(root, out)


def require_acceptance(root: Path, path: Path) -> dict:
    """Verify immutable numerical evidence without requiring a superseded keeper to remain current."""
    root = root.resolve()
    evidence_file(root, path)
    raw = path.read_bytes()
    result = json.loads(raw)
    require_evidence(root, result["evidence_sha256"])
    binding_path, helpers, state_path, _, receipt_path = locations(root)
    frozen = validation_source(root)
    state, receipt, _ = validation_records(root, frozen)
    identities = {
        "binding_sha256": binding_path,
        "state_sha256": state_path,
        "receipt_sha256": receipt_path,
        "intent_sha256": helpers.parent / "controller.json",
    }
    if (
        result["status"] != ACCEPTANCE_STATUS
        or result["root"] != str(root)
        or result["commands"] != COMMANDS
        or result["checks"] != CHECKS
        or result["numerical_status"] != NUMERICAL_STATUS
        or result["owner"] != receipt["owner"]
        or result["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or result["keeper"] != state["restored_keeper"]
        or result["evidence_sha256"] != validation_evidence(root, frozen, receipt)
        or any(
            result[key] != sha256(file) or result["evidence_sha256"].get(str(file.relative_to(root))) != result[key]
            for key, file in identities.items()
        )
        or not result.get("time")
        or path.read_bytes() != raw
    ):
        raise ValueError("Attribution saved numerical acceptance identity differs")
    return result


def production_source(root: Path, wave: str) -> tuple[dict, dict]:
    preparation = importlib.import_module("prepare_fused_attribution_production")
    if wave == "wave12":
        binding_path = root / "run-preparation/wave12-launch-binding.json"
        helpers = root / "controllers/wave12-launch-20261117/tracking"
        controller = "selected_lr_production.py"
    else:
        binding_path, helpers, _, _ = paths(root, wave)
        controller = "fused_attribution_production.py"
    raw = binding_path.read_bytes()
    release = json.loads(raw)
    candidate = Path(release["candidate"])
    if (
        release["root"] != str(root)
        or release["helpers"] != str(helpers)
        or sha256(Path(release["archive"])) != release["archive_sha256"]
        or candidate != candidate.resolve()
        or candidate_inventory(candidate) != release["files_sha256"]
        or (wave != "wave12" and release["status"] != FROZEN_STATUS)
    ):
        raise ValueError("Attribution predecessor frozen source identity differs")
    require_helper_closure(helpers, release["helpers_sha256"], (controller,), candidate)
    if wave == "wave12":
        expected = release["files_sha256"] | {"preparation.json": release["preparation_sha256"]}
    else:
        prepared = read(candidate / PREPARATION)
        if sha256(candidate / PREPARATION) != release["preparation_sha256"]:
            raise ValueError("Attribution predecessor preparation bytes differ")
        expected = preparation.archive_inventory(prepared, release["preparation_sha256"])
        for key, member in (
            ("cpu_receipt", "candidate-cpu-receipt.json"),
            ("controller_cpu_receipt", "controller-cpu-receipt.json"),
            ("validation_acceptance", "validation-acceptance.json"),
        ):
            if sha256(Path(release[key])) != release[f"{key}_sha256"]:
                raise ValueError("Attribution predecessor archived receipt identity differs")
            expected[member] = release[f"{key}_sha256"]
    expected.update({f"tracking/{name}": digest for name, digest in release["helpers_sha256"].items()})
    preparation.require_archive(Path(release["archive"]), expected, alias=True)
    files = [
        binding_path,
        Path(release["archive"]),
        *[helpers / name for name in release["helpers_sha256"]],
        *[candidate / name for name in release["files_sha256"]],
        *candidate.glob("*.json"),
    ]
    if wave != "wave12":
        files.extend(Path(release[key]) for key in ("cpu_receipt", "controller_cpu_receipt", "validation_acceptance"))
    evidence = evidence_hashes(root, files)
    if sha256(binding_path) != hashlib.sha256(raw).hexdigest():
        raise ValueError("Attribution predecessor binding changed during authentication")
    return release, evidence


def predecessor_worker(root: Path, wave: str, release: dict, family: str) -> tuple[dict, dict, dict]:
    path = (
        root / f"run-preparation/wave12-workers/{family}.json" if wave == "wave12" else worker_path(root, wave, family)
    )
    state, intent = read(path), read(path.with_suffix(".start.json"))
    node = local_node()
    command = (
        worker_command(root, wave, family)
        if wave != "wave12"
        else [
            PYTHON,
            str(Path(release["helpers"]) / "selected_lr_production.py"),
            "--root",
            str(root),
            "--phase",
            "worker",
            "--arm",
            family,
        ]
    )
    binding_path = root / f"run-preparation/{wave}-launch-binding.json"
    launch_path = root / f"run-preparation/{wave}-launch-state.json"
    launch = read(launch_path)
    generation = intent["generation"]
    if (
        state["status"] != "complete"
        or state["arm"] != family
        or state["node"] != node
        or state["command"] != release["commands"][node]
        or state["release_binding_sha256"] != sha256(binding_path)
        or state["worker_command"] != command
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or any(key in state for key in ("error", "errors", "cleanup_error"))
        or intent["status"] != "acknowledged"
        or intent["pid"] != state["worker_pid"]
        or intent["start_ticks"] != state["worker_start_ticks"]
        or intent["command"] != command
        or intent["worker_marker"] != state["worker_marker"]
        or intent["release_binding_sha256"] != state["release_binding_sha256"]
        or generation["pid"] != intent["pid"]
        or generation["start_ticks"] != intent["start_ticks"]
        or generation["command"] != command
        or generation["group"] != intent["pid"]
        or generation["session"] != intent["pid"]
        or launch["status"] != "launched"
        or launch["binding_sha256"] != state["release_binding_sha256"]
        or launch["worker_markers"][family] != state["worker_marker"]
    ):
        raise RuntimeError("Attribution predecessor worker completion or original acknowledgment differs")
    files = [path, path.with_suffix(".start.json"), launch_path, Path(state["log"])]
    if state["log_sha256"] != sha256(Path(state["log"])):
        raise ValueError("Attribution predecessor training log changed")
    return state, intent, evidence_hashes(root, files)


def allocation_observation(
    root: Path, wave: str, family: str, expected: dict | None = None, manager_lock: int | None = None
) -> dict:
    """Observe the immediate physical-node predecessor without releasing or adopting processes."""
    root = root.resolve()
    spec, node = wave_spec(wave), node_for(wave, family)
    if local_node() != node:
        raise RuntimeError("Attribution allocation observation is on the wrong rotated physical node")
    if manager_lock is not None:
        require_manager_lock(manager_lock)
    with nullcontext(manager_lock) if manager_lock is not None else keeper_guard():
        if wave == "wave13" and node == NODE:
            frozen = validation_source(root)
            state, receipt = validation_terminal(root, frozen)
            keeper = processes.keeper_observation()
            evidence = validation_evidence(root, frozen, receipt)
            require_validation_source(root, frozen, evidence)
            predecessor = "attribution_validation"
            predecessor_family = None
        else:
            predecessor = spec["predecessor_wave"] or "wave12"
            release, evidence = production_source(root, predecessor)
            families = release["arms"] if predecessor == "wave12" else release["families"]
            predecessor_family = families[release["nodes"].index(node)]
            state, intent, worker_evidence = predecessor_worker(root, predecessor, release, predecessor_family)
            evidence.update(worker_evidence)
            if predecessor == "wave12":
                script = (
                    "from selected_lr_run_history import binding\n"
                    "from selected_lr_production import probe_arm\n"
                    "require_origins()\nroot=Path(sys.argv[1]); release=binding(root)\n"
                    "result=probe_arm(root,sys.argv[2]); require_origins()\n"
                    "print(json.dumps(result,allow_nan=False))\n"
                )
                probe = frozen_call(root, release, script, [str(root), predecessor_family], evidence)
            else:
                completed_files = [
                    root / f"run-preparation/monitor-{predecessor}/latest.json",
                    root / f"run-preparation/analysis-{predecessor}.json",
                    root / f"run-preparation/swanlab-upload-{predecessor}.json",
                ]
                for previous_family in release["families"]:
                    directory = root / "runs" / predecessor / previous_family
                    completed_files.extend(
                        (
                            directory / "run.json",
                            directory / "metrics_from_000000.jsonl",
                            directory / "checkpoints/step_005120/complete.json",
                        )
                    )
                evidence.update(evidence_hashes(root, completed_files))
                script = (
                    "from fused_attribution_run_history import require_completed_wave\n"
                    "from fused_attribution_production import probe\n"
                    "require_origins()\nroot=Path(sys.argv[1]); wave=sys.argv[2]\n"
                    "complete=require_completed_wave(root,wave)\n"
                    "result=probe(root,wave,sys.argv[3]); require_origins()\n"
                    "print(json.dumps({'completion':complete,'probe':result},allow_nan=False))\n"
                )
                result = frozen_call(root, release, script, [str(root), predecessor, predecessor_family], evidence)
                complete, probe = result["completion"], result["probe"]
                if complete["status"] != COMPLETION_STATUS or complete["wave"] != predecessor:
                    raise RuntimeError("Attribution predecessor full history, analysis or cloud remains pending")
                require_evidence(root, complete["evidence_sha256"])
                evidence.update(complete["evidence_sha256"])
                if probe.get("original_worker_absent") is not True or probe.get("original_child_absent") is not True:
                    raise RuntimeError("Attribution predecessor original handles remain")
            if probe["status"] != "complete" or probe["receipt"] != state:
                raise RuntimeError("Attribution frozen predecessor probe refuses completed ownership")
            keeper = processes.keeper_observation()
            require_same_keeper(keeper, state["restored_keeper"])
            if predecessor != "wave12":
                require_same_keeper(keeper, probe["keeper"])
            child_marker = "OUTPUT_MARIN_DIAGNOSTIC_ID" if predecessor == "wave12" else CHILD_MARKER
            worker_marker = "OUTPUT_MARIN_WAVE12_WORKER_ID" if predecessor == "wave12" else WORKER_MARKER
            if not state["child_marker"] or processes.marker_processes(child_marker, state["child_marker"]):
                raise RuntimeError("Attribution predecessor training descendants remain")
            require_markers_on_keeper(
                keeper,
                [{"name": worker_marker, "value": state["worker_marker"]}],
                [state["worker_pid"], state["child_pid"], intent["starter"]["pid"]],
            )
        require_same_keeper(keeper, state["restored_keeper"])
        require_evidence(root, evidence)
        result = {
            "status": "fused_attribution_exact_predecessor_keeper_verified",
            "wave": wave,
            "family": family,
            "arm": family,
            "node": node,
            "predecessor_wave": predecessor,
            "predecessor_family": predecessor_family,
            "keeper": keeper,
            "evidence_sha256": evidence,
        }
        if expected is not None:
            require_same_keeper(keeper, expected["keeper"])
            if any(result[key] != expected[key] for key in result if key != "keeper"):
                raise RuntimeError("Attribution accepted predecessor evidence or rotated allocation changed")
        return result
