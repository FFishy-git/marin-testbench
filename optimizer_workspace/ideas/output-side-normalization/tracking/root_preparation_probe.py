# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain cold-process root capture/replay evidence under a separate diagnostic owner."""

import argparse
import importlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import torch
from diagnose_continuation import compare_saved
from fused_attribution_validation import numerical_modules
from root_preparation_contract import CHILD_MARKER, CONFIG, ENVIRONMENT, PREPARATION, PYTHON, locations
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

ORDER = ("continuous", "resumed", "continuous", "resumed")
GROUPS = ("d", "f")
STATUS = "saved_root_cold_capture_replay_diagnostic_not_continuation_acceptance"
ORIGINAL = "run-preparation/diagnostic-fused-continuation-gpu-20261118.json"


def command(root: Path, candidate: Path, index: int) -> list[str]:
    return [
        PYTHON,
        "-P",
        str(locations(root)[1] / "root_preparation_probe.py"),
        "--root",
        str(root),
        "--candidate",
        str(candidate),
        "--index",
        str(index),
    ]


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def finite_tree(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(item) for item in value)
    return True


def comparison(actual, expected, name) -> dict:
    changed, scalars = [], []
    count = compare_saved(actual, expected, name, changed, scalars)
    finite = finite_tree(actual) and finite_tree(expected)
    return {
        "tensor_comparisons": count,
        "exact": not changed and not scalars,
        "finite": finite,
        "within_tolerance": finite and not scalars and all(item["within_tolerance"] for item in changed),
        "changed_tensors": changed,
        "scalar_or_structure_differences": scalars,
    }


def retained_order() -> list[dict]:
    result, seen = [], set()
    for mode in ORDER:
        if mode not in seen:
            result.extend({"mode": mode, "group": name, "phase": "eager"} for name in GROUPS)
        phase = "replay" if mode in seen else "capture"
        result.extend({"mode": mode, "group": name, "phase": phase} for name in GROUPS)
        seen.add(mode)
    return result


def expected_lifecycle() -> list[dict]:
    seen, result = set(), []
    for mode in ORDER:
        fresh = mode not in seen
        result.extend(
            {"mode": mode, "group": name, "phase": "capture" if fresh else "replay", "fresh": fresh, "active": True}
            for name in GROUPS
        )
        seen.add(mode)
    return result


def output_checks(retained: list, sources: dict) -> tuple[list, list]:
    if [{key: value for key, value in item.items() if key != "output"} for item in retained] != retained_order():
        raise ValueError("Root cold retained eager/capture/replay order differs")
    historical = [
        comparison(
            item["output"],
            sources[item["mode"]][item["group"]]["saved_output"],
            f"{item['mode']}.{item['group']}.{item['phase']}.saved",
        )
        for item in retained
        if item["phase"] != "eager"
    ]
    safety = []
    for mode in ("continuous", "resumed"):
        for name in GROUPS:
            outputs = {
                item["phase"]: item["output"] for item in retained if item["mode"] == mode and item["group"] == name
            }
            safety.extend(
                comparison(outputs[phase], outputs["eager"], f"{mode}.{name}.{phase}.eager")
                for phase in ("capture", "replay")
            )
            safety.append(comparison(outputs["replay"], outputs["capture"], f"{mode}.{name}.repeat"))
    return historical, safety


def input_state(engine) -> dict:
    return cpu_tree(
        {
            name: {"C": group.C, "tol": group.tol, "SB": group.basis.SB, "S2": group.basis.S2}
            for name, group in engine.groups.items()
        }
    )


def rng_state() -> dict:
    return cpu_tree(
        {"torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate()}
    )


def saved_inputs(root: Path, prepared: dict) -> tuple[dict, dict]:
    """Read original full-shaped rank-zero inputs and candidates only after hash authentication."""
    replay = read(root / "run-preparation/saved-ns-replay-binding-20261118.json")
    expected = replay["predecessor"]["evidence_sha256"][ORIGINAL]
    path = root / ORIGINAL
    if sha256(path) != expected:
        raise ValueError("Root cold probe original diagnostic receipt changed")
    original = read(path)
    if not original["analysis"]["trace_transparent"] or not original["analysis"]["extra_checkpoint_transparent"]:
        raise ValueError("Root cold probe requires the original trace transparency gates")
    result, evidence = {}, {ORIGINAL: expected}
    work = root / "validation/fused_continuation_20261118"
    for mode in ("continuous", "resumed"):
        trace = work / f"trace_{mode}/continuation-trace/rank_0/update_000023.pt"
        inventory = original["analysis"]["traces"][mode]["files_sha256"]
        sidecar = trace.with_suffix(".json")
        if sha256(trace) != inventory[str(trace)] or sha256(sidecar) != inventory[str(sidecar)]:
            raise ValueError("Root cold probe saved input trace changed")
        saved = torch.load(trace, map_location="cpu", mmap=True, weights_only=False)
        if (
            saved["rank"] != 0
            or saved["update"] != 23
            or saved["provenance"]["start_step"] != (22 if mode == "resumed" else 0)
            or original["candidate"] != prepared["parent_candidate"]
        ):
            raise ValueError("Root cold probe trace rank/update/source differs")
        groups = {}
        for name, shape in (("d", (12, 512, 512)), ("f", (4, 2048, 2048))):
            before = [item["values"] for item in saved["records"] if item["stage"] == f"input/{name}/before_root"]
            outputs = [item["values"] for item in saved["records"] if item["stage"] == f"input/{name}/candidate"]
            if len(before) != 1 or len(outputs) != 1 or tuple(before[0]["C"].shape) != shape:
                raise ValueError("Root cold probe needs both complete original input groups")
            fields = {key: before[0][key] for key in ("C", "tol", "SB", "S2")}
            if not finite_tree(fields) or not finite_tree(outputs[0]):
                raise ValueError("Root cold probe saved inputs or outputs are nonfinite")
            groups[name] = {"inputs": fields, "saved_output": outputs[0]}
        result[mode] = groups
        evidence.update({str(source.relative_to(root)): inventory[str(source)] for source in (trace, sidecar)})
    return result, evidence


def require_owner(root: Path, candidate: Path, index: int) -> tuple[dict, dict]:
    """Authenticate a directly owned probe descendant before initializing CUDA."""
    if index not in (0, 1):
        raise ValueError("Root cold probe index differs")
    _, _, owner = numerical_modules(candidate)
    state, frozen = owner.require_controller(locations(root)[0])
    own = owner.require_process(os.getpid(), CHILD_MARKER, state["child_marker"], candidate)
    if (
        Path(__file__).resolve() != locations(root)[1] / "root_preparation_probe.py"
        or state.get("active_probe_command") != command(root, candidate, index)
        or own["command"] != command(root, candidate, index)
        or own["parent"] != state["child_pid"]
        or own["group"] != own["pid"]
        or own["session"] != own["pid"]
        or frozen["helpers_sha256"]["root_preparation_probe.py"] != sha256(Path(__file__))
    ):
        raise RuntimeError("Root cold probe has no exact acknowledged validator ancestry or source")
    return state, frozen


def run(root: Path, candidate: Path, index: int) -> dict:
    state, _ = require_owner(root, candidate, index)
    prepared = read(candidate / PREPARATION)
    sources, evidence = saved_inputs(root, prepared)
    out = locations(root)[3] / "cold-root"
    report_path, tensor_path = out / f"process-{index}.json", out / f"process-{index}.pt"
    if report_path.exists() or tensor_path.exists() or not out.is_dir():
        raise FileExistsError("Root cold probe requires new externally owned evidence")
    if any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()):
        raise RuntimeError("Root cold probe environment differs")
    config = read(candidate / CONFIG)
    _, records, _ = numerical_modules(candidate)
    identity = records.checkpoint_identity(config, root / "data/experiment_receipt.json")
    # Delay numerical imports until candidate ownership and source checks have passed.
    prepare_restored_roots = importlib.import_module("experiment.prepare_restored_roots").prepare_restored_roots
    exact_tree = importlib.import_module("experiment.root_preparation_checked").exact_tree
    fast_arguments = importlib.import_module("reproduce.marin.optimizer").fast_arguments
    roots = importlib.import_module("toner.fast.roots")
    FastTonerCore = importlib.import_module("toner.fast.engine").FastTonerCore

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(config["runtime"]["deterministic"])
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    expected_uuid = prepared["allocation_predecessor"]["restored_keeper"]["devices"][0][1]
    if UUID(str(properties.uuid).removeprefix("GPU-")) != UUID(expected_uuid.removeprefix("GPU-")):
        raise RuntimeError("Root cold probe actual physical GPU differs")
    roots.set_powers((config["statistics"]["inner_strength"], config["statistics"]["outer_strength"]))
    engines, retained, preparation, lifecycle = {}, [], {}, []
    for mode in ORDER:
        if mode not in engines:
            engine = FastTonerCore.__new__(FastTonerCore)
            engine.args = fast_arguments(config)
            engine.split, engine.p_in, engine.p_out = True, 0.5, 0.125
            engine.froot, engine.iters = roots, config["root"]["iterations"]
            engine.steps, engine.begun, engine.device = 22, False, "cuda:0"
            engine.side, engine.graph_pool = torch.cuda.Stream(), torch.cuda.graph_pool_handle()
            engine.groups = {}
            for name in GROUPS:
                fields = sources[mode][name]["inputs"]
                resident = {key: value.to("cuda:0", copy=True) for key, value in fields.items()}
                if any(resident[key].stride() != value.stride() for key, value in fields.items()):
                    raise ValueError("Root cold upload changed an input layout")
                engine.groups[name] = SimpleNamespace(
                    C=resident["C"],
                    tol=resident["tol"],
                    basis=SimpleNamespace(SB=resident["SB"], S2=resident["S2"]),
                    per=resident["C"].shape[0],
                    warm=False,
                    graph=None,
                )
            before, rng_before = input_state(engine), rng_state()
            prepared_groups = prepare_restored_roots(engine)
            after, rng_after = input_state(engine), rng_state()
            exact_tree(after, before)
            exact_tree(rng_after, rng_before)
            if prepared_groups != list(GROUPS) or any(
                group.graph is not None or group.warm is not True for group in engine.groups.values()
            ):
                raise ValueError("Root cold scratch preparation consumed or created a graph")
            preparation[mode] = {
                "inputs_before": before,
                "inputs_after": after,
                "rng_before": rng_before,
                "rng_after": rng_after,
                "prepared_groups": prepared_groups,
                "readiness": {
                    name: {"warm": group.warm, "graph_active": group.graph is not None}
                    for name, group in engine.groups.items()
                },
            }
            engines[mode] = engine
            for name, group in engine.groups.items():
                with torch.cuda.stream(engine.side), torch.no_grad():
                    eager = cpu_tree(engine._roots(group))
                engine.side.synchronize()
                retained.append({"mode": mode, "group": name, "phase": "eager", "output": eager})
        engine = engines[mode]
        engine.side.wait_stream(torch.cuda.current_stream())
        for name, group in engine.groups.items():
            phase = "capture" if group.graph is None else "replay"
            before_graph = group.graph
            with torch.cuda.stream(engine.side), torch.no_grad():
                output = engine._roots_graphed(group)
            engine.side.synchronize()
            copied = cpu_tree(output)
            retained.append({"mode": mode, "group": name, "phase": phase, "output": copied})
            lifecycle.append(
                {
                    "mode": mode,
                    "group": name,
                    "phase": phase,
                    "fresh": before_graph is None,
                    "active": group.graph is not None,
                }
            )
    final_inputs = {}
    for mode in ("continuous", "resumed"):
        final_inputs[mode] = input_state(engines[mode])
        for name in GROUPS:
            exact_tree(final_inputs[mode][name], sources[mode][name]["inputs"])
    historical, safety = output_checks(retained, sources)
    checks = historical + safety
    if lifecycle != expected_lifecycle():
        raise ValueError("Root cold actual capture/replay lifecycle differs")
    if require_owner(root, candidate, index)[0] != state or saved_inputs(root, prepared)[1] != evidence:
        raise RuntimeError("Root cold owner or original saved source changed during execution")
    with tensor_path.open("xb") as stream:
        torch.save({"retained": retained, "preparation": preparation, "final_inputs": final_inputs}, stream)
    result = {
        "status": STATUS,
        "index": index,
        "identity": identity,
        "binding_sha256": sha256(locations(root)[0]),
        "owner": {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]},
        "physical_uuid": expected_uuid,
        "saved_inputs_sha256": evidence,
        "order": list(ORDER),
        "groups": list(GROUPS),
        "lifecycle": lifecycle,
        "checks": checks,
        "safety_within_tolerance": all(item["within_tolerance"] for item in safety),
        "historical_saved_within_tolerance": all(item["within_tolerance"] for item in historical),
        "state_rng_exact": True,
        "scratch_outputs_consumed": False,
        "outputs": str(tensor_path),
        "outputs_sha256": sha256(tensor_path),
        "all_within_tolerance": all(item["within_tolerance"] for item in checks),
    }
    write(report_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.root.resolve(), args.candidate.resolve(), args.index)["status"]}))


if __name__ == "__main__":
    main()
