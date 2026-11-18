# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check release archives and acknowledged rank ownership through filesystem boundaries."""

import importlib
import json
import os
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

package = ModuleType("_fused_attribution_production_cpu")
package.__path__ = [str(Path(__file__).parent)]
sys.modules[package.__name__] = package
owner = importlib.import_module(f"{package.__name__}.fused_attribution_production_owner")
contract = sys.modules[owner.wave_spec.__module__]
process_reader = sys.modules[owner.observed.__module__]


def record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def archive(path: Path, entries: dict[str, Path], candidate: Path | None = None) -> None:
    with tarfile.open(path, "w") as bundle:
        for name, source in entries.items():
            bundle.add(source, arcname=name, recursive=False)
        if candidate is not None:
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)


def process(proc: Path, pid: int, parent: int, ticks: str, command: list, cwd: Path, environment: dict) -> None:
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[:4], fields[19] = ["S", str(parent), str(pid), str(pid)], ticks
    (directory / "stat").write_text(f"{pid} (owned rank fixture) {' '.join(fields)}\n")
    (directory / "cmdline").write_bytes(("\0".join(command) + "\0").encode())
    (directory / "environ").write_bytes(
        ("\0".join(f"{key}={value}" for key, value in environment.items()) + "\0").encode()
    )
    link = directory / "cwd"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(cwd, target_is_directory=True)


@pytest.fixture
def released(tmp_path, monkeypatch):
    root, proc = tmp_path / "root", tmp_path / "proc"
    previous, candidate = root / "candidates/original", root / "candidates/production"
    directory = Path(owner.__file__).parent
    original_experiment = previous / "experiment"
    (original_experiment / "tests").mkdir(parents=True)
    (original_experiment / "__init__.py").write_text("")
    (original_experiment / "train.py").write_text(
        "from .fused_attribution_owner import require_fused_attribution_owner\n\n"
        "def main(args):\n"
        "    require_fused_attribution_owner(args)\n"
        "    return args.config.read_bytes()\n"
    )
    (original_experiment / "fused_attribution_owner.py").write_text("ORIGINAL_VALIDATION_OWNER = True\n")
    for name in ("selected_lr_release_owner.py", "fused_attribution_validation_contract.py"):
        (original_experiment / name).write_bytes((directory / name).read_bytes())
    (original_experiment / "numerical.py").write_text("NUMERICAL_TRAJECTORY_UNCHANGED = True\n")
    for wave in owner.WAVES:
        spec = owner.wave_spec(wave)
        for family in owner.FAMILIES:
            record(
                previous / owner.config_name(family, spec["lr_label"]),
                {
                    "model": {"sequence_length": 4096},
                    "training": {
                        "steps": 5120,
                        "tokens": 2684354560,
                        "batch_tokens": 524288,
                        "global_batch_sequences": 128,
                        "world_size": 8,
                        "seed": 0,
                    },
                    "experiment": {"arm": family, "refresh_install_age": 1, "diagonal_backend": "triton"},
                    "optimizer": {"lr": spec["matrix_lr"], "ns_mode": "tpu", "ns_steps": 10},
                    "runtime": {"compile": "default", "attention": "FA2", "head": "compiled"},
                    "root": {"graphs": True},
                    "schedule": contract.SCHEDULE,
                },
            )
    pinned = previous / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    (pinned / "numerical.bin").write_bytes(b"original numerical source")
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    record(previous / owner.VALIDATION_PREPARATION, {"status": "original_validation"})
    record(root / "data/experiment_receipt.json", {"fixture": "prepared-data"})
    binding, original_helpers, state_path, work, receipt_path = owner.locations(root)
    original_helpers.mkdir(parents=True)
    (original_helpers / "frozen-validator.py").write_text("# preserved validator\n")
    cpu_parent = root / "original-cpu.json"
    controller_parent = root / "original-controller.json"
    record(cpu_parent, {"status": "original_cpu_preserved"})
    record(controller_parent, {"status": "original_controller_cpu_preserved"})
    source_parent = root / "source-original.tar"
    archive(
        source_parent,
        {name: previous / name for name in owner.candidate_files(previous)}
        | {owner.VALIDATION_PREPARATION: previous / owner.VALIDATION_PREPARATION},
        previous,
    )
    original = {
        "status": owner.VALIDATION_STATUS,
        "root": str(root),
        "candidate": str(previous),
        "helpers": str(original_helpers),
        "helpers_sha256": {path.name: owner.sha256(path) for path in original_helpers.iterdir()},
        "files_sha256": owner.candidate_files(previous),
        "preparation_sha256": owner.sha256(previous / owner.VALIDATION_PREPARATION),
        "data_receipt_sha256": owner.sha256(root / "data/experiment_receipt.json"),
        "cpu_receipt": str(cpu_parent),
        "cpu_receipt_sha256": owner.sha256(cpu_parent),
        "controller_cpu_receipt": str(controller_parent),
        "controller_cpu_receipt_sha256": owner.sha256(controller_parent),
        "archive": str(source_parent),
        "archive_sha256": owner.sha256(source_parent),
        "commands": owner.validation_commands(previous, root, work),
    }
    record(binding, original)
    original_owner = {"controller_pid": 18001, "controller_start_ticks": "801"}
    gpu_commands = []
    for index, command in enumerate(original["commands"]):
        log = work / f"command-{index}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("original validation command completed\n")
        gpu_commands.append(
            {
                "trial": owner.trial_specs()[index // 3]["name"],
                "mode": owner.MODES[index % 3],
                "command": command,
                "exit_code": 0,
                "log": str(log),
                "log_sha256": owner.sha256(log),
            }
        )
    trials = []
    for index, spec in enumerate(owner.trial_specs()):
        checks = owner.BASE_CHECKS | (owner.FUSED_CHECKS if spec["family"] != "muont" else set())
        if spec["family"] == "null_leftdiag_fused":
            checks = checks | {"null_identity_outer"}
        trials.append(
            spec
            | {
                "config": str(previous / spec["config_name"]),
                "config_sha256": owner.sha256(previous / spec["config_name"]),
                "checks": dict.fromkeys(checks, True),
                "commands": original["commands"][index * 3 : (index + 1) * 3],
            }
        )
    receipt = {
        "status": owner.NUMERICAL_STATUS,
        "owner": original_owner,
        "execution_binding_sha256": owner.sha256(binding),
        "candidate_preparation_sha256": original["preparation_sha256"],
        "gpu_commands": gpu_commands,
        "trials": trials,
    }
    record(receipt_path, receipt)
    keeper = {
        "pid": 17001,
        "generation": {
            str(pid): {
                "pid": pid,
                "start_ticks": str(900 + index),
                "parent": 1 if index == 0 else 17001,
                "group": pid,
                "session": pid,
                "status": "S",
                "command": ["python", "keeper"],
            }
            for index, pid in enumerate(range(17001, 17010))
        },
    }
    validation_state = {
        "status": "passed",
        **original_owner,
        "child_pid": 18002,
        "child_start_ticks": "802",
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "keeper_release_complete": True,
        "command": owner.validator_command(root, previous),
        "binding_sha256": owner.sha256(binding),
        "validation_sha256": owner.sha256(receipt_path),
        "restored_keeper": keeper,
    }
    record(state_path, validation_state)
    intent_path = original_helpers.parent / "controller.json"
    record(
        intent_path,
        {
            "status": "acknowledged",
            "pid": original_owner["controller_pid"],
            "start_ticks": original_owner["controller_start_ticks"],
            "binding_sha256": owner.sha256(binding),
            "command": owner.controller_command(root),
            "generation": {
                "pid": original_owner["controller_pid"],
                "start_ticks": original_owner["controller_start_ticks"],
                "group": original_owner["controller_pid"],
                "session": original_owner["controller_pid"],
                "command": owner.controller_command(root),
            },
        },
    )
    acceptance_path = root / "run-preparation/fused-attribution-production-validation-acceptance-20261118.json"
    evidence_paths = (
        binding,
        state_path,
        receipt_path,
        intent_path,
        source_parent,
        cpu_parent,
        controller_parent,
        previous / owner.VALIDATION_PREPARATION,
        root / "data/experiment_receipt.json",
        *original_helpers.iterdir(),
        *(Path(x["log"]) for x in gpu_commands),
    )
    evidence_paths = tuple(path for path in evidence_paths if path.is_file())
    acceptance = {
        "status": owner.ACCEPTANCE_STATUS,
        "root": str(root),
        "binding_sha256": owner.sha256(binding),
        "state_sha256": owner.sha256(state_path),
        "receipt_sha256": owner.sha256(receipt_path),
        "intent_sha256": owner.sha256(intent_path),
        "owner": original_owner,
        "commands": 27,
        "checks": 93,
        "numerical_status": owner.NUMERICAL_STATUS,
        "keeper": keeper,
        "evidence_sha256": {str(path.relative_to(root)): owner.sha256(path) for path in evidence_paths},
    }
    record(acceptance_path, acceptance)
    for name in owner.candidate_files(previous) | {owner.VALIDATION_PREPARATION: original["preparation_sha256"]}:
        target = candidate / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((previous / name).read_bytes())
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    train = (
        (candidate / "experiment/train.py")
        .read_text()
        .replace(
            "from .fused_attribution_owner import require_fused_attribution_owner",
            "from .fused_attribution_production_owner import require_fused_attribution_production_owner",
        )
    )
    (candidate / "experiment/train.py").write_text(
        train.replace(
            "    require_fused_attribution_owner(args)\n", "    require_fused_attribution_production_owner(args)\n"
        )
    )
    for name in ("fused_attribution_production_owner.py", "fused_attribution_production_contract.py"):
        (candidate / "experiment" / name).write_bytes((directory / name).read_bytes())
    test_name = "experiment/tests/test_fused_attribution_production_owner.py"
    (candidate / test_name).parent.mkdir(parents=True, exist_ok=True)
    (candidate / test_name).write_text("# source-bound production owner test\n")
    files = owner.candidate_files(candidate)
    helpers_source = tmp_path / "tested-helpers"
    helpers_source.mkdir()
    (helpers_source / "prepare_fused_attribution_production.py").write_text("# CPU-tested preparation helper\n")
    for name in ("fused_attribution_production_owner.py", "fused_attribution_production_contract.py"):
        (helpers_source / name).write_bytes((directory / name).read_bytes())
    hashes = {path.name: owner.sha256(path) for path in helpers_source.iterdir()}
    metadata = {owner.VALIDATION_PREPARATION: original["preparation_sha256"]}
    prepared = {
        "status": owner.PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "files_sha256": files,
        "changed_files": sorted(owner.CHANGES),
        "validation_binding_sha256": owner.sha256(binding),
        "helpers_sha256": hashes,
        "parent_candidate": str(previous),
        "parent_cpu_receipt": str(cpu_parent),
        "parent_cpu_receipt_sha256": original["cpu_receipt_sha256"],
        "parent_controller_cpu_receipt": str(controller_parent),
        "parent_controller_cpu_receipt_sha256": original["controller_cpu_receipt_sha256"],
        "parent_archive": str(source_parent),
        "parent_archive_sha256": original["archive_sha256"],
        "parent_preparation_sha256": original["preparation_sha256"],
        "parent_metadata_sha256": metadata,
        "data_receipt_sha256": original["data_receipt_sha256"],
    }
    record(candidate / owner.PREPARATION, prepared)
    preparation_digest = owner.sha256(candidate / owner.PREPARATION)
    cpu_out, controller_out = root / "production-cpu", root / "production-controller-cpu"
    for out in (cpu_out, controller_out):
        (out / "tracking").mkdir(parents=True)
        for name in hashes:
            (out / "tracking" / name).write_bytes((helpers_source / name).read_bytes())
    cpu_log, controller_log = cpu_out / "candidate.log", controller_out / "controller.log"
    cpu_log.write_text("candidate CPU passed\n")
    controller_log.write_text("controller CPU passed\n")
    cpu_archive = cpu_out / "source.tar"
    archive(
        cpu_archive,
        {name: candidate / name for name in files | metadata | {owner.PREPARATION: preparation_digest}}
        | {f"tracking/{name}": cpu_out / "tracking" / name for name in hashes},
        candidate,
    )
    cpu_path = cpu_out / "receipt.json"
    cpu = {
        "status": owner.CPU_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "files_sha256": files,
        "helpers_sha256": hashes,
        "helpers": str(cpu_out / "tracking"),
        "preparation_sha256": preparation_digest,
        "validation_binding_sha256": owner.sha256(binding),
        "command": owner.cpu_command(),
        "cwd": str(candidate),
        "exit_code": 0,
        "cuda_visible_devices": "",
        "log": str(cpu_log),
        "log_sha256": owner.sha256(cpu_log),
        "archive": str(cpu_archive),
        "archive_sha256": owner.sha256(cpu_archive),
    }
    record(cpu_path, cpu)
    controller_archive = controller_out / "source.tar"
    archive(controller_archive, {f"tracking/{name}": controller_out / "tracking" / name for name in hashes})
    controller_path = controller_out / "receipt.json"
    controller = {
        "status": owner.CONTROLLER_CPU_STATUS,
        "root": str(root),
        "candidate_cpu_receipt": str(cpu_path),
        "candidate_cpu_receipt_sha256": owner.sha256(cpu_path),
        "helpers_sha256": hashes,
        "helpers": str(controller_out / "tracking"),
        "fixture_files_sha256": {},
        "command": owner.controller_cpu_command(),
        "cwd": str(controller_out),
        "exit_code": 0,
        "cuda_visible_devices": "",
        "log": str(controller_log),
        "log_sha256": owner.sha256(controller_log),
        "archive": str(controller_archive),
        "archive_sha256": owner.sha256(controller_archive),
    }
    record(controller_path, controller)
    monkeypatch.setattr(owner, "TASK", candidate)
    monkeypatch.setattr(process_reader, "PROC_ROOT", proc)
    return SimpleNamespace(
        root=root,
        previous=previous,
        candidate=candidate,
        proc=proc,
        files=files,
        hashes=hashes,
        prepared=prepared,
        preparation_digest=preparation_digest,
        helpers_source=helpers_source,
        cpu_path=cpu_path,
        controller_path=controller_path,
        acceptance_path=acceptance_path,
        acceptance=acceptance,
        keeper=keeper,
    )


def bind_worker(item, wave, family, rank, monkeypatch):
    binding, helpers, launch_path, _ = owner.paths(item.root, wave)
    helpers.mkdir(parents=True)
    for name in item.hashes:
        (helpers / name).write_bytes((item.helpers_source / name).read_bytes())
    source_archive = item.root / f"source-{wave}.tar"
    archive(
        source_archive,
        {name: item.candidate / name for name in item.files | item.prepared["parent_metadata_sha256"]}
        | {
            owner.PREPARATION: item.candidate / owner.PREPARATION,
            "candidate-cpu-receipt.json": item.cpu_path,
            "controller-cpu-receipt.json": item.controller_path,
            "validation-acceptance.json": item.acceptance_path,
        }
        | {f"tracking/{name}": helpers / name for name in item.hashes},
        item.candidate,
    )
    spec = owner.wave_spec(wave)
    allocations = {}
    for current_family, node in zip(owner.FAMILIES, spec["nodes"], strict=True):
        predecessor = spec["predecessor_wave"] or ("attribution_validation" if node.endswith(".9") else "wave12")
        predecessor_family = None
        if predecessor == "attribution_validation":
            evidence = item.acceptance["evidence_sha256"]
        else:
            nodes = (
                ["28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139"]
                if predecessor == "wave12"
                else owner.wave_spec(predecessor)["nodes"]
            )
            families = (
                ["muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused"]
                if predecessor == "wave12"
                else list(owner.FAMILIES)
            )
            predecessor_family = families[nodes.index(node)]
            previous_binding = item.root / f"run-preparation/{predecessor}-launch-binding.json"
            record(previous_binding, {"nodes": nodes, "arms" if predecessor == "wave12" else "families": families})
            path = item.root / f"run-preparation/{predecessor}-workers/{predecessor_family}.json"
            log = path.with_suffix(".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("predecessor completed and restored\n")
            record(
                path,
                {
                    "status": "complete",
                    "node": node,
                    "release_binding_sha256": owner.sha256(previous_binding),
                    "exit_code": 0,
                    "cleanup_exit_code": 0,
                    "keeper_restore": 0,
                    "restored_keeper": item.keeper,
                    "log": str(log),
                },
            )
            record(path.with_suffix(".start.json"), {"status": "acknowledged"})
            evidence = {
                str(path.relative_to(item.root)): owner.sha256(path)
                for path in (previous_binding, path, path.with_suffix(".start.json"), log)
            }
        allocations[node] = {
            "status": "fused_attribution_exact_predecessor_keeper_verified",
            "wave": wave,
            "family": current_family,
            "node": node,
            "predecessor_wave": predecessor,
            "predecessor_family": predecessor_family,
            "keeper": item.keeper,
            "evidence_sha256": evidence,
        }
    release = {
        "status": owner.FROZEN_STATUS,
        "root": str(item.root),
        "candidate": str(item.candidate),
        "wave": wave,
        "files_sha256": item.files,
        "preparation_sha256": item.preparation_digest,
        "helpers": str(helpers),
        "helpers_sha256": item.hashes,
        "families": spec["families"],
        "nodes": spec["nodes"],
        "commands": owner.commands(item.root, item.candidate, wave),
        "environment": owner.ENVIRONMENT,
        "monitor_interval_seconds": 900,
        "training_timeout_seconds": 21600,
        "allocation_by_node": allocations,
        "archive": str(source_archive),
        "archive_sha256": owner.sha256(source_archive),
        "validation_binding": str(owner.locations(item.root)[0]),
        "validation_binding_sha256": item.prepared["validation_binding_sha256"],
        "validation_acceptance": str(item.acceptance_path),
        "validation_acceptance_sha256": owner.sha256(item.acceptance_path),
        "cpu_receipt": str(item.cpu_path),
        "cpu_receipt_sha256": owner.sha256(item.cpu_path),
        "controller_cpu_receipt": str(item.controller_path),
        "controller_cpu_receipt_sha256": owner.sha256(item.controller_path),
    }
    record(binding, release)
    command = release["commands"][owner.node_for(wave, family)]
    marker, child_marker = f"{wave}-{family}", f"{wave}-{family}-child"
    state = {
        "status": "running",
        "wave": wave,
        "family": family,
        "node": owner.node_for(wave, family),
        "worker_pid": 21001,
        "worker_start_ticks": "101",
        "worker_marker": marker,
        "worker_command": owner.worker_command(item.root, wave, family),
        "release_binding_sha256": owner.sha256(binding),
        "child_pid": 21002,
        "child_start_ticks": "102",
        "child_marker": child_marker,
        "command": command,
        "keeper_release_complete": True,
        "keeper": {
            "pid": item.keeper["pid"],
            "start_ticks": item.keeper["generation"][str(item.keeper["pid"])]["start_ticks"],
            "generation": [
                [process["pid"], process["start_ticks"], process["group"]]
                for process in item.keeper["generation"].values()
            ],
        },
    }
    state_path = owner.worker_path(item.root, wave, family)
    record(state_path, state)
    environment = owner.ENVIRONMENT | {owner.WORKER_MARKER: marker, owner.CHILD_MARKER: child_marker}
    rank_env = environment | {"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8", "RANK": str(rank), "LOCAL_RANK": str(rank)}
    process(item.proc, 21001, 1, "101", state["worker_command"], item.candidate, environment)
    process(item.proc, 21002, 21001, "102", command, item.candidate, environment)
    process(item.proc, os.getpid(), 21002, "103", [command[0], "-u", *command[5:]], item.candidate, rank_env)
    record(
        state_path.with_suffix(".start.json"),
        {
            "status": "acknowledged",
            "pid": 21001,
            "start_ticks": "101",
            "wave": wave,
            "family": family,
            "node": state["node"],
            "command": state["worker_command"],
            "worker_marker": marker,
            "release_binding_sha256": owner.sha256(binding),
            "generation": owner.observed(21001),
        },
    )
    record(
        launch_path,
        {
            "status": "launched",
            "binding_sha256": owner.sha256(binding),
            "worker_markers": {name: f"{wave}-{name}" for name in owner.FAMILIES},
        },
    )
    monkeypatch.chdir(item.candidate)
    for key, value in rank_env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(owner.sys, "argv", ["experiment.train", *command[7:]])
    monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kwargs: f"{state['node']}\n")
    args = SimpleNamespace(
        launch_release=binding,
        config=Path(command[command.index("--config") + 1]),
        data=item.root / "data",
        data_receipt=item.root / "data/experiment_receipt.json",
        out=owner.output_path(item.root, wave, family),
        max_steps=None,
        resume=None,
        execution_boundary=None,
        verify_diagonal_capture=False,
        verify_outer_diagonal_updates=False,
    )
    return SimpleNamespace(
        args=args, release=release, binding=binding, state=state, state_path=state_path, launch_path=launch_path
    )


@pytest.mark.parametrize(
    ("wave", "family", "rank"),
    (("wave13", "muont", 0), ("wave14", "outer_leftdiag_fused", 7), ("wave15", "null_leftdiag_fused", 3)),
)
def test_released_rank_accepts_rotated_nodes_without_output_or_process_mutation(
    released, monkeypatch, wave, family, rank
):
    item = bind_worker(released, wave, family, rank, monkeypatch)
    before = {str(path): path.read_bytes() for path in released.proc.rglob("*") if path.is_file()}
    owner.require_fused_attribution_production_owner(item.args)
    assert owner.require_release(item.binding) == item.release
    assert not item.args.out.exists()
    assert before == {str(path): path.read_bytes() for path in released.proc.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    "fault", ("worker_reuse", "ack_generation", "child_parent", "rank_session", "keeper", "wave", "mask", "terminal")
)
def test_released_rank_refuses_generation_or_trial_drift(released, monkeypatch, fault):
    item = bind_worker(released, "wave13", "muont", 0, monkeypatch)
    owner.require_fused_attribution_production_owner(item.args)
    if fault == "worker_reuse":
        path = released.proc / "21001/stat"
        path.write_text(path.read_text().replace("101", "999"))
    elif fault == "ack_generation":
        path = item.state_path.with_suffix(".start.json")
        intent = owner.read(path)
        intent["generation"]["session"] = 1
        record(path, intent)
    elif fault == "child_parent":
        path = released.proc / "21002/stat"
        fields = path.read_text().rsplit(")", 1)[1].split()
        fields[1] = "1"
        path.write_text(f"21002 (fixture) {' '.join(fields)}\n")
    elif fault == "rank_session":
        path = released.proc / str(os.getpid()) / "stat"
        fields = path.read_text().rsplit(")", 1)[1].split()
        fields[3] = "21002"
        path.write_text(f"{os.getpid()} (fixture) {' '.join(fields)}\n")
    elif fault == "keeper":
        state = owner.read(item.state_path)
        state["keeper"]["generation"][1][1] = "replaced"
        record(item.state_path, state)
    elif fault == "wave":
        record(item.state_path, item.state | {"wave": "wave14"})
    elif fault == "mask":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    else:
        record(item.launch_path, owner.read(item.launch_path) | {"status": "needs_attention"})
    with pytest.raises((RuntimeError, FileNotFoundError)):
        owner.require_fused_attribution_production_owner(item.args)
    assert not item.args.out.exists()


@pytest.mark.parametrize("fault", ("acceptance", "cpu_log", "extra_archive", "numerical", "original_log"))
def test_release_refuses_rehashed_acceptance_source_or_cpu_archive(released, monkeypatch, fault):
    item = bind_worker(released, "wave13", "muont", 0, monkeypatch)
    owner.require_release(item.binding)
    if fault == "acceptance":
        acceptance = owner.read(released.acceptance_path)
        acceptance["evidence_sha256"].pop(next(name for name in acceptance["evidence_sha256"] if name.endswith(".log")))
        record(released.acceptance_path, acceptance)
        item.release["validation_acceptance_sha256"] = owner.sha256(released.acceptance_path)
    elif fault == "cpu_log":
        cpu = owner.read(released.cpu_path)
        Path(cpu["log"]).write_text("changed CPU evidence\n")
    elif fault == "extra_archive":
        extra = released.root / "foreign.py"
        extra.write_text("FOREIGN = True\n")
        with tarfile.open(item.release["archive"], "a") as bundle:
            bundle.add(extra, arcname="foreign.py")
        item.release["archive_sha256"] = owner.sha256(Path(item.release["archive"]))
    elif fault == "numerical":
        (released.candidate / "experiment/numerical.py").write_text("CHANGED = True\n")
        item.release["files_sha256"] = owner.candidate_files(released.candidate)
        prepared = owner.read(released.candidate / owner.PREPARATION)
        prepared["files_sha256"] = item.release["files_sha256"]
        record(released.candidate / owner.PREPARATION, prepared)
        item.release["preparation_sha256"] = owner.sha256(released.candidate / owner.PREPARATION)
    else:
        acceptance = owner.read(released.acceptance_path)
        log = next(name for name in acceptance["evidence_sha256"] if name.endswith(".log"))
        (released.root / log).write_text("changed original numerical trajectory log\n")
    record(item.binding, item.release)
    with pytest.raises(RuntimeError):
        owner.require_release(item.binding)
