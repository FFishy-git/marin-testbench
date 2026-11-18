# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute and reconstruct the distinct restored-root intervention without production acceptance."""

import argparse
import os
import subprocess
from pathlib import Path

import torch
from analyze_tuning import executed_sources, normalized_sources
from compare_runs import history
from fused_attribution_validation import (
    INTERRUPTED_MANIFEST,
    numerical_modules,
    require_capture_history,
    require_evaluations,
    require_numerical_state,
    save_interrupted_manifest,
)
from fused_continuation_contract import PREPARATION as PARENT_PREPARATION
from fused_continuation_validation import checkpoint_pair, loss_pair, traces
from root_preparation_contract import (
    BATCH_TOKENS,
    BOUNDARY,
    CONFIG,
    ENDPOINT,
    ENVIRONMENT,
    FIRST_RESUMED,
    MODES,
    PREPARATION,
    PREPARED_STATUS,
    PROTOCOLS,
    REGIMES,
    WORLD,
    locations,
    plan,
    require_actual_config,
)
from root_preparation_probe import GROUPS, ORDER, comparison, expected_lifecycle, output_checks, saved_inputs
from root_preparation_probe import STATUS as PROBE_STATUS
from root_preparation_probe import command as probe_command
from selected_lr_release_owner import candidate_files, read, sha256
from selected_lr_stage_profile_validation import complete_checkpoint_evidence
from watch_wave5 import write

COMPLETE_STATUS = "completed_root_preparation_diagnostic_not_production_gate"
COMMAND_TIMEOUT = 900
PROBE_TIMEOUT = 900
STEPS = (BOUNDARY, FIRST_RESUMED, ENDPOINT)


def source(candidate: Path, root: Path) -> dict:
    prepared = read(candidate / PREPARATION)
    require_actual_config(candidate)
    if (
        prepared["status"] != PREPARED_STATUS
        or prepared["candidate"] != str(candidate)
        or prepared["root"] != str(root)
        or prepared["files_sha256"] != candidate_files(candidate)
        or prepared["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or prepared["plan"] != plan(candidate, root)
        or any(sha256(candidate / name) != value for name, value in prepared["parent_metadata_sha256"].items())
        or any(
            sha256(root / name) != value for name, value in prepared["allocation_predecessor"]["evidence_sha256"].items()
        )
    ):
        raise ValueError("Root diagnostic candidate, actual schedule or saved predecessor changed")
    return prepared


def coordinates(index: int) -> tuple[str, str, str]:
    if not 0 <= index < len(REGIMES) * len(PROTOCOLS) * len(MODES):
        raise ValueError("Root diagnostic command index differs")
    return REGIMES[index // 6], PROTOCOLS[index % 6 // 3], MODES[index % 3]


def directory(root: Path, regime: str, protocol: str, mode: str) -> Path:
    return locations(root)[3] / f"{regime}_{protocol}_{'continuous' if mode == 'continuous' else 'resumed'}"


def graph_lifecycle(saved: dict, mode: str, regime: str, step: int) -> dict:
    groups = saved["fast"]["input"]["groups"]
    expected = not (regime == "unprepared" and mode == "resumed" and step == FIRST_RESUMED)
    if set(groups) != set(GROUPS) or any(
        item["warm"] is not True or item["cached_root_graph_active"] is not expected for item in groups.values()
    ):
        raise ValueError("Root diagnostic checkpoint does not reflect the required fresh graph lifecycle")
    return {
        name: {"warm": item["warm"], "graph_active": item["cached_root_graph_active"]} for name, item in groups.items()
    }


def preparation_receipts(out: Path, candidate: Path, prepared: dict, regime: str, mode: str) -> dict:
    paths = sorted(out.glob("root_preparation_rank_*.json"))
    expected = regime == "prepared" and mode == "resumed"
    if not expected:
        if paths:
            raise ValueError("Root preparation leaked into an unprepared or continuous command")
        return {}
    _, records, _ = numerical_modules(candidate)
    identity = records.checkpoint_identity(
        read(candidate / CONFIG), Path(prepared["root"]) / "data/experiment_receipt.json"
    )
    if [path.name for path in paths] != [f"root_preparation_rank_{rank}.json" for rank in range(WORLD)]:
        raise ValueError("Root preparation lacks all-rank state/RNG evidence")
    hashes = {}
    for rank, path in enumerate(paths):
        receipt = read(path)
        if path.is_symlink() or receipt != {
            "status": "restored_root_scratch_preparation_state_rng_exact_not_continuation_acceptance",
            "rank": rank,
            "checkpoint_step": BOUNDARY,
            "groups": list(GROUPS),
            "numerical_state_exact": True,
            "torch_rng_exact": True,
            "cuda_rng_exact": True,
            "python_rng_exact": True,
            "fresh_graphs": True,
            "scratch_outputs_consumed": False,
            "identity": identity,
        }:
            raise ValueError("Root preparation per-rank state/RNG/isolation receipt differs")
        hashes[path.name] = sha256(path)
    return hashes


def trajectory(out: Path, candidate: Path, prepared: dict, regime: str, mode: str) -> dict:
    config = read(candidate / CONFIG)
    run = history(out)
    endpoint = BOUNDARY if mode == "interrupted" else ENDPOINT
    rows = run.segments[0] if mode == "interrupted" else run.rows
    manifest_path = out / INTERRUPTED_MANIFEST if mode == "interrupted" else out / "run.json"
    manifest = read(manifest_path)
    segments = sorted(out.glob("metrics_from_*.jsonl"))
    expected_segments = ["metrics_from_000000.jsonl"]
    if mode == "resumed":
        expected_segments.append(f"metrics_from_{BOUNDARY:06d}.jsonl")
    if (
        mode not in MODES
        or regime not in REGIMES
        or manifest["config"] != config
        or read(out / "config.json") != config
        or manifest["data_receipt_sha256"] != prepared["data_receipt_sha256"]
        or normalized_sources(candidate, manifest["source_sha256"]) != executed_sources(candidate, prepared)
        or manifest["status"] != "execution_check_complete"
        or manifest.get("execution_check") is not True
        or type(manifest["completed_steps"]) is not int
        or manifest["completed_steps"] != endpoint
        or manifest.get("resume") != (str(out / f"checkpoints/step_{BOUNDARY:06d}") if mode == "resumed" else None)
        or [row["step"] for row in rows] != list(range(1, endpoint + 1))
        or any(row["tokens"] != row["step"] * BATCH_TOKENS for row in rows)
        or (mode != "interrupted" and [path.name for path in segments] != expected_segments)
        or any(path.is_symlink() or path.stat().st_nlink != 1 for path in (manifest_path, *segments))
    ):
        raise ValueError("Root diagnostic history/source/configuration or independent segments differ")
    require_evaluations(rows, config, endpoint)
    installations = [entry for row in rows for entry in row["input_refresh_installations"]]
    if not installations or any(
        entry["installation_step"] != row["step"] - 1 or entry["installation_step"] - entry["source_step"] != 1
        for row in rows
        for entry in row["input_refresh_installations"]
    ):
        raise ValueError("Root diagnostic lacks its fixed-age installation history")
    checkpoints, graphs = {}, {}
    for step in STEPS:
        if step > endpoint:
            continue
        checkpoints[str(step)] = complete_checkpoint_evidence(out, config, candidate, prepared, step)
        graphs[str(step)] = {}
        for rank in range(WORLD):
            saved = torch.load(
                out / f"checkpoints/step_{step:06d}/rank_{rank}.pt", map_location="cpu", weights_only=False, mmap=True
            )
            require_numerical_state(
                saved,
                config,
                step,
                rank,
                require_cached_graph=not (regime == "unprepared" and mode == "resumed" and step == FIRST_RESUMED),
            )
            graphs[str(step)][str(rank)] = graph_lifecycle(saved, mode, regime, step)
    if mode != "interrupted":
        require_capture_history(out, config["experiment"]["arm"])
    return {
        "manifest_sha256": sha256(manifest_path),
        "history_sha256": {path.name: sha256(path) for path in (segments[:1] if mode == "interrupted" else segments)},
        "updates": len(rows),
        "tokens": rows[-1]["tokens"],
        "fixed_age_installations": len(installations),
        "checkpoints": checkpoints,
        "graph_lifecycle": graphs,
        "root_preparation": (
            preparation_receipts(out, candidate, prepared, regime, mode) if mode != "interrupted" else {}
        ),
    }


def checkpoint_pass(pair: dict, step: int) -> bool:
    ranks = pair.get("ranks", [])
    return (
        pair.get("step") == step
        and type(pair.get("step")) is int
        and pair.get("within_tolerance") is True
        and len(ranks) == WORLD
        and [item.get("rank") for item in ranks] == list(range(WORLD))
        and all(
            type(item["rank"]) is int
            and type(item.get("tensor_comparisons")) is int
            and item["tensor_comparisons"] > 0
            and type(item.get("failed_tensor_count")) is int
            and item["failed_tensor_count"] == 0
            and item.get("scalar_or_structure_differences") == []
            and isinstance(item.get("changed_tensors"), list)
            and len(item["changed_tensors"]) <= item["tensor_comparisons"]
            and all(
                changed["within_tolerance"] is True
                and not any(f".{name}" in changed["name"] for name in ("torch_rng", "cuda_rng", "python_rng"))
                for changed in item["changed_tensors"]
            )
            and isinstance(item.get("checkpoint_sha256"), dict)
            and len(item["checkpoint_sha256"]) == 2
            for item in ranks
        )
    )


def pair_set(actual: Path, expected: Path) -> dict:
    return {
        "checkpoints": {str(step): checkpoint_pair(actual, expected, step) for step in STEPS},
        "loss": loss_pair(actual, expected),
    }


def pair_set_pass(value: dict) -> bool:
    checkpoints, loss = value["checkpoints"], value["loss"]
    return (
        set(checkpoints) == {str(step) for step in STEPS}
        and all(checkpoint_pass(checkpoints[str(step)], step) for step in STEPS)
        and loss.get("within_tolerance") is True
        and type(loss.get("comparisons")) is int
        and loss["comparisons"] >= ENDPOINT
        and loss.get("failures") == []
    )


def root_trace_lifecycle(out: Path) -> dict:
    result = {}
    for rank in range(WORLD):
        for step in STEPS:
            value = torch.load(
                out / f"continuation-trace/rank_{rank}/update_{step:06d}.pt",
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            groups = {}
            for name in GROUPS:
                matches = [item["values"] for item in value["records"] if item["stage"] == f"input/{name}/before_root"]
                if len(matches) != 1:
                    raise ValueError("Root trace missed an actual group call")
                groups[name] = {"warm": matches[0]["warm"], "graph_active": matches[0]["graph_active"]}
            result[f"{rank}:{step}"] = groups
    return result


def cold_probe_evidence(root: Path, candidate: Path, prepared: dict, executions: list) -> dict:
    work = locations(root)[3] / "cold-root"
    saved, saved_hashes = saved_inputs(root, prepared)
    _, numerical_records, _ = numerical_modules(candidate)
    identity = numerical_records.checkpoint_identity(read(candidate / CONFIG), root / "data/experiment_receipt.json")
    state = read(locations(root)[2])
    owner = {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
    records, reconstructed, historical_checks = [], [], []
    if len(executions) != 2:
        raise ValueError("Root cold safety requires two fresh processes")
    for index, execution in enumerate(executions):
        path, tensors = work / f"process-{index}.json", work / f"process-{index}.pt"
        report = read(path)
        log = locations(root)[3] / f"cold-root-{index}.log"
        if (
            index not in (0, 1)
            or type(execution["index"]) is not int
            or execution["index"] != index
            or execution["command"] != probe_command(root, candidate, index)
            or type(execution["exit_code"]) is not int
            or execution["exit_code"] != 0
            or execution["log"] != str(log)
            or execution["log_sha256"] != sha256(log)
            or report["status"] != PROBE_STATUS
            or type(report["index"]) is not int
            or report["index"] != index
            or report["identity"] != identity
            or report["owner"] != owner
            or report["binding_sha256"] != sha256(locations(root)[0])
            or report["saved_inputs_sha256"] != saved_hashes
            or report["order"] != list(ORDER)
            or report["groups"] != list(GROUPS)
            or report["lifecycle"] != expected_lifecycle()
            or report["state_rng_exact"] is not True
            or report["scratch_outputs_consumed"] is not False
            or report["outputs"] != str(tensors)
            or report["outputs_sha256"] != sha256(tensors)
            or report["physical_uuid"] != prepared["allocation_predecessor"]["restored_keeper"]["devices"][0][1]
            or any(source.is_symlink() or source.stat().st_nlink != 1 for source in (path, tensors, log))
        ):
            raise ValueError("Root cold probe command/source/retained-output provenance differs")
        bundle = torch.load(tensors, map_location="cpu", weights_only=False, mmap=True)
        if set(bundle) != {"retained", "preparation", "final_inputs"}:
            raise ValueError("Root cold probe lacks retained state/RNG/input evidence")
        retained = bundle["retained"]
        historical, safety = output_checks(retained, saved)
        if (
            report["checks"] != historical + safety
            or report["safety_within_tolerance"] is not all(item["within_tolerance"] for item in safety)
            or report["historical_saved_within_tolerance"] is not all(item["within_tolerance"] for item in historical)
            or report["all_within_tolerance"] is not all(item["within_tolerance"] for item in historical + safety)
            or set(bundle["preparation"]) != {"continuous", "resumed"}
            or set(bundle["final_inputs"]) != {"continuous", "resumed"}
        ):
            raise ValueError("Root cold reported checks differ from retained output reconstruction")
        for mode in ("continuous", "resumed"):
            preparation = bundle["preparation"][mode]
            inputs = {name: saved[mode][name]["inputs"] for name in GROUPS}
            if (
                set(preparation)
                != {"inputs_before", "inputs_after", "rng_before", "rng_after", "prepared_groups", "readiness"}
                or preparation["prepared_groups"] != list(GROUPS)
                or preparation["readiness"] != {name: {"warm": True, "graph_active": False} for name in GROUPS}
                or set(preparation["rng_before"]) != {"torch_rng", "cuda_rng", "python_rng"}
                or not comparison(preparation["rng_after"], preparation["rng_before"], f"{mode}.rng")["exact"]
                or any(
                    not comparison(value, inputs, f"{mode}.inputs")["exact"]
                    for value in (
                        preparation["inputs_before"],
                        preparation["inputs_after"],
                        bundle["final_inputs"][mode],
                    )
                )
            ):
                raise ValueError("Root cold retained preparation changed resident inputs or RNG")
        records.append(retained)
        reconstructed.append(safety)
        historical_checks.append(historical)
    cross = []
    for first, second in zip(records[0], records[1], strict=True):
        if {key: value for key, value in first.items() if key != "output"} != {
            key: value for key, value in second.items() if key != "output"
        }:
            raise ValueError("Root cold process output order differs")
        cross.append(comparison(second["output"], first["output"], f"{first['mode']}.{first['group']}.{first['phase']}"))
    if saved_inputs(root, prepared)[1] != saved_hashes:
        raise ValueError("Root cold original saved inputs changed during reconstruction")
    return {
        "reports_sha256": {
            str(work / f"process-{index}.json"): sha256(work / f"process-{index}.json") for index in range(2)
        },
        "outputs_sha256": {
            str(work / f"process-{index}.pt"): sha256(work / f"process-{index}.pt") for index in range(2)
        },
        "reconstructed_checks": reconstructed,
        "historical_saved_checks": historical_checks,
        "historical_saved_within_tolerance": all(
            item["within_tolerance"] for checks in historical_checks for item in checks
        ),
        "cross_process": cross,
        "within_tolerance": (
            all(item["within_tolerance"] for checks in reconstructed for item in checks)
            and all(item["within_tolerance"] for item in cross)
        ),
    }


def original_evidence(root: Path, prepared: dict) -> dict:
    _, saved_hashes = saved_inputs(root, prepared)
    original = read(root / "run-preparation/diagnostic-fused-continuation-gpu-20261118.json")
    prior_work = root / "validation/fused_continuation_20261118"
    expected_coordinates = [(protocol, mode) for protocol in PROTOCOLS for mode in MODES]
    if [(entry["protocol"], entry["mode"]) for entry in original["gpu_commands"]] != expected_coordinates:
        raise ValueError("Root original diagnostic trajectory coverage differs")
    parent_candidate = Path(prepared["parent_candidate"])
    parent_preparation = read(parent_candidate / PARENT_PREPARATION)
    config = read(parent_candidate / CONFIG)
    if (
        config != read(Path(prepared["candidate"]) / CONFIG)
        or candidate_files(parent_candidate) != parent_preparation["files_sha256"]
    ):
        raise ValueError("Root original diagnostic numerical source/configuration changed")
    for execution in original["gpu_commands"]:
        out = prior_work / f"{execution['protocol']}_{'continuous' if execution['mode'] == 'continuous' else 'resumed'}"
        retained = execution["trajectory"]
        manifest = out / (INTERRUPTED_MANIFEST if execution["mode"] == "interrupted" else "run.json")
        saved_manifest = read(manifest)
        if (
            retained["manifest_sha256"] != sha256(manifest)
            or saved_manifest["config"] != config
            or saved_manifest["data_receipt_sha256"] != prepared["data_receipt_sha256"]
            or normalized_sources(parent_candidate, saved_manifest["source_sha256"])
            != executed_sources(parent_candidate, parent_preparation)
            or any(sha256(out / name) != value for name, value in retained["history_sha256"].items())
        ):
            raise ValueError("Root unprepared reproduction original trajectory changed")
        expected_steps = [str(step) for step in STEPS if execution["mode"] != "interrupted" or step <= BOUNDARY]
        if set(retained["checkpoints"]) != set(expected_steps):
            raise ValueError("Root original diagnostic checkpoint coverage differs")
        for step, checkpoint in retained["checkpoints"].items():
            folder = out / f"checkpoints/step_{int(step):06d}"
            if (
                set(checkpoint["rank_sha256"]) != {f"rank_{rank}.pt" for rank in range(WORLD)}
                or checkpoint["receipt_sha256"] != sha256(folder / "complete.json")
                or any(sha256(folder / name) != value for name, value in checkpoint["rank_sha256"].items())
            ):
                raise ValueError("Root unprepared reproduction original checkpoint changed")
    return {"saved_inputs_sha256": saved_hashes, "trajectories": original["gpu_commands"]}


def analysis(root: Path, candidate: Path, prepared: dict, probes: list) -> dict:
    original = original_evidence(root, prepared)
    outputs = {
        f"{regime}_{protocol}_{mode}": directory(root, regime, protocol, mode)
        for regime in REGIMES
        for protocol in PROTOCOLS
        for mode in ("continuous", "resumed")
    }
    continuation = {
        regime: {
            protocol: pair_set(outputs[f"{regime}_{protocol}_resumed"], outputs[f"{regime}_{protocol}_continuous"])
            for protocol in PROTOCOLS
        }
        for regime in REGIMES
    }
    transparency = {
        regime: {
            mode: pair_set(outputs[f"{regime}_trace_{mode}"], outputs[f"{regime}_control_{mode}"])
            for mode in ("continuous", "resumed")
        }
        for regime in REGIMES
    }
    controls = {
        protocol: pair_set(outputs[f"prepared_{protocol}_continuous"], outputs[f"unprepared_{protocol}_continuous"])
        for protocol in PROTOCOLS
    }
    prior_work = root / "validation/fused_continuation_20261118"
    reproduction = {
        f"{protocol}_{mode}": pair_set(outputs[f"unprepared_{protocol}_{mode}"], prior_work / f"{protocol}_{mode}")
        for protocol in PROTOCOLS
        for mode in ("continuous", "resumed")
    }
    trace_evidence, lifecycle = {}, {}
    for regime in REGIMES:
        for mode in ("continuous", "resumed"):
            out = outputs[f"{regime}_trace_{mode}"]
            trace_evidence[f"{regime}_{mode}"] = traces(out, candidate, prepared)
            lifecycle[f"{regime}_{mode}"] = root_trace_lifecycle(out)
            for rank in range(WORLD):
                for step in STEPS:
                    expected_graph = not (mode == "resumed" and step == FIRST_RESUMED)
                    expected_warm = not (regime == "unprepared" and mode == "resumed" and step == FIRST_RESUMED)
                    if any(
                        item != {"warm": expected_warm, "graph_active": expected_graph}
                        for item in lifecycle[f"{regime}_{mode}"][f"{rank}:{step}"].values()
                    ):
                        raise ValueError("Root trace first-capture/next-replay lifecycle differs")
    cold = cold_probe_evidence(root, candidate, prepared, probes)
    if original_evidence(root, prepared) != original or source(candidate, root) != prepared:
        raise ValueError("Root original or candidate source changed during cross-source comparisons")
    gates = {
        "cold_capture_repeat_safety": cold["within_tolerance"],
        "trace_transparency": all(pair_set_pass(value) for modes in transparency.values() for value in modes.values()),
        "continuous_control_unchanged": all(pair_set_pass(value) for value in controls.values()),
        "unprepared_original_reproduced": all(pair_set_pass(value) for value in reproduction.values()),
        "prepared_all_rank_continuation": all(pair_set_pass(value) for value in continuation["prepared"].values()),
    }
    return {
        "continuation": continuation,
        "transparency": transparency,
        "continuous_controls": controls,
        "unprepared_original_reproduction": reproduction,
        "traces": trace_evidence,
        "trace_root_lifecycle": lifecycle,
        "cold_root": cold,
        "gates": gates,
        "diagnostic_intervention_accepted": all(gates.values()),
        "scope": (
            "Bounded diagnostic only; production needs all original numerical/configuration gates "
            "and a separate release."
        ),
    }


def run(root: Path, candidate: Path, receipt_path: Path) -> dict:
    work = locations(root)[3]
    prepared = source(candidate, root)
    numerical, _, owner = numerical_modules(candidate)
    owned = owner.require_gpu_owner(root)
    if work.exists() or receipt_path.exists() or receipt_path != locations(root)[4]:
        raise FileExistsError("Root diagnostic execution identity already has evidence")
    numerical.require_receipt(root / "data", root / "data/experiment_receipt.json")
    if any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()):
        raise RuntimeError("Root diagnostic GPU environment differs")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Root diagnostic refuses occupied GPUs")
    result = plan(candidate, root) | {
        "status": "diagnosing",
        "owner": owned,
        "execution_binding_sha256": sha256(locations(root)[0]),
        "candidate_preparation_sha256": sha256(candidate / PREPARATION),
        "gpu_commands": [],
        "cold_probes": [],
    }
    work.mkdir(parents=True)
    (work / "cold-root").mkdir()
    write(receipt_path, result)
    try:
        for index in range(2):
            command = probe_command(root, candidate, index)
            state = read(locations(root)[2]) | {"active_probe_command": command}
            write(locations(root)[2], state)
            log = work / f"cold-root-{index}.log"
            with log.open("x") as stream:
                code = subprocess.run(
                    command,
                    cwd=candidate,
                    env=os.environ | ENVIRONMENT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    timeout=PROBE_TIMEOUT,
                ).returncode
            result["cold_probes"].append(
                {"index": index, "command": command, "exit_code": code, "log": str(log), "log_sha256": sha256(log)}
            )
            write(receipt_path, result)
            if code != 0:
                raise RuntimeError(f"Root cold probe failed; preserve {log}")
        result["cold_probe_evidence"] = cold_probe_evidence(root, candidate, prepared, result["cold_probes"])
        write(receipt_path, result)
        for index, command in enumerate(result["commands"]):
            regime, protocol, mode = coordinates(index)
            if owner.require_gpu_owner(root) != owned or source(candidate, root) != prepared:
                raise RuntimeError("Root diagnostic source or acknowledged owner changed")
            state = read(locations(root)[2]) | {"active_command": command, "active_probe_command": None}
            write(locations(root)[2], state)
            log = work / f"{regime}_{protocol}_{mode}.log"
            with log.open("x") as stream:
                code = subprocess.run(
                    command,
                    cwd=candidate,
                    env=os.environ | ENVIRONMENT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    timeout=COMMAND_TIMEOUT,
                ).returncode
            execution = {
                "regime": regime,
                "protocol": protocol,
                "mode": mode,
                "command": command,
                "exit_code": code,
                "log": str(log),
                "log_sha256": sha256(log),
            }
            result["gpu_commands"].append(execution)
            write(receipt_path, result)
            if code != 0:
                raise RuntimeError(f"Root diagnostic command failed; preserve {log}")
            out = directory(root, regime, protocol, mode)
            if mode == "interrupted":
                save_interrupted_manifest(out)
            execution["trajectory"] = trajectory(out, candidate, prepared, regime, mode)
            write(receipt_path, result)
        result["analysis"] = analysis(root, candidate, prepared, result["cold_probes"])
        if source(candidate, root) != prepared or owner.require_gpu_owner(root) != owned:
            raise RuntimeError("Root diagnostic source or original owner changed at completion")
        result["status"] = COMPLETE_STATUS
        write(receipt_path, result)
        return result
    except BaseException as error:
        result.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(receipt_path, result)
        raise


def reconstruct(root: Path, candidate: Path, receipt_path: Path) -> dict:
    raw = receipt_path.read_bytes()
    result, prepared = read(receipt_path), source(candidate, root)
    frozen, state = read(locations(root)[0]), read(locations(root)[2])
    expected = plan(candidate, root)
    if (
        result["status"] != COMPLETE_STATUS
        or "error" in result
        or receipt_path != locations(root)[4]
        or any(result[key] != value for key, value in expected.items() if key != "status")
        or result["execution_binding_sha256"] != sha256(locations(root)[0])
        or state["binding_sha256"] != result["execution_binding_sha256"]
        or frozen["preparation_sha256"] != result["candidate_preparation_sha256"]
        or result["candidate_preparation_sha256"] != sha256(candidate / PREPARATION)
        or result["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or len(result["gpu_commands"]) != len(expected["commands"])
    ):
        raise ValueError("Root diagnostic original owner, plan or commands differ")
    for index, (execution, command) in enumerate(zip(result["gpu_commands"], expected["commands"], strict=True)):
        regime, protocol, mode = coordinates(index)
        log = locations(root)[3] / f"{regime}_{protocol}_{mode}.log"
        if (
            (execution["regime"], execution["protocol"], execution["mode"]) != (regime, protocol, mode)
            or execution["command"] != command
            or type(execution["exit_code"]) is not int
            or execution["exit_code"] != 0
            or execution["log"] != str(log)
            or execution["log_sha256"] != sha256(log)
            or execution["trajectory"]
            != trajectory(directory(root, regime, protocol, mode), candidate, prepared, regime, mode)
        ):
            raise ValueError("Root diagnostic history/checkpoint/command reconstruction differs")
    cold = cold_probe_evidence(root, candidate, prepared, result["cold_probes"])
    reconstructed = analysis(root, candidate, prepared, result["cold_probes"])
    if (
        result["cold_probe_evidence"] != cold
        or result["analysis"] != reconstructed
        or source(candidate, root) != prepared
        or read(locations(root)[0]) != frozen
        or read(locations(root)[2]) != state
        or receipt_path.read_bytes() != raw
    ):
        raise ValueError("Root diagnostic independently reconstructed result differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reconstruct", action="store_true")
    args = parser.parse_args()
    operation = reconstruct if args.reconstruct else run
    print(operation(args.root.resolve(), args.candidate.resolve(), args.receipt.resolve())["status"])


if __name__ == "__main__":
    main()
