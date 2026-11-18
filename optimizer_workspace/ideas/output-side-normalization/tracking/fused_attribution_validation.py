# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute and reconstruct nine independently owned bounded attribution validations."""

import argparse
import importlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import torch
from analyze_tuning import executed_sources, normalized_sources
from compare_runs import history
from fused_attribution_validation_contract import (
    BATCH_TOKENS,
    BOUNDARY,
    CONTINUATION_TOLERANCE,
    ENDPOINT,
    ENVIRONMENT,
    FROZEN_STATUS,
    MODES,
    NODE,
    PREPARATION,
    PREPARED_STATUS,
    WORLD,
    locations,
    predecessor_locations,
    require_config,
    trial_specs,
    validation_commands,
)
from launch_wave5 import read, sha256
from owned_output_handoff import same_generation
from selected_lr_failure_handoff import same_keeper
from selected_lr_stage_profile_contract import CHILD_MARKER as PREDECESSOR_CHILD_MARKER
from selected_lr_stage_profile_contract import CONTROLLER_MARKER as PREDECESSOR_CONTROLLER_MARKER
from selected_lr_stage_profile_validation import complete_checkpoint_evidence
from selected_lr_validation import FUSED_GATES, GATES
from watch_wave5 import write
from wave7_release import candidate_inventory

HANDOFF_STATUS = "failed_profiler_exact_restored_keeper_pending_live_handoff"
PASS_STATUS = "passed_fused_attribution_actual_config_continuation_and_work_gates"
INTERRUPTED_MANIFEST = "attribution-interrupted-manifest.json"
COMMAND_TIMEOUT = 900


def plan(candidate: Path, root: Path, work: Path) -> dict:
    commands = validation_commands(candidate, root, work)
    trials = []
    for index, spec in enumerate(trial_specs()):
        checks = (*GATES, "loss_continuation")
        if spec["family"] != "muont":
            checks = (*checks, *FUSED_GATES, "output_work_retained")
        if spec["family"] == "null_leftdiag_fused":
            checks = (*checks, "null_identity_outer")
        trials.append(
            spec
            | {
                "config": str(candidate / spec["config_name"]),
                "config_sha256": sha256(candidate / spec["config_name"]),
                "continuous": str(work / f"{spec['name']}_continuous"),
                "resumed": str(work / f"{spec['name']}_resumed"),
                "commands": commands[index * len(MODES) : (index + 1) * len(MODES)],
                "checks": dict.fromkeys(checks, False),
            }
        )
    return {
        "status": "fused_attribution_validation_planned_not_executed",
        "root": str(root),
        "candidate": str(candidate),
        "work": str(work),
        "environment": ENVIRONMENT,
        "world": WORLD,
        "continuous_steps": ENDPOINT,
        "interrupted_steps": BOUNDARY,
        "resumed_total_steps": ENDPOINT,
        "continuation_tolerance": CONTINUATION_TOLERANCE,
        "planned_validation_tokens": len(trials) * 2 * ENDPOINT * BATCH_TOKENS,
        "trials": trials,
        "gpu_commands": [],
        "scope": "Full schedules retained; independent 24 versus 22+resume24. No full-budget quality or timing result.",
    }


def require_keeper_shape(keeper: dict) -> None:
    generation = {int(pid): process for pid, process in keeper["generation"].items()}
    pid = keeper["pid"]
    devices, occupied = keeper["devices"], keeper["occupied"]
    if (
        type(pid) is not int
        or len(generation) != 9
        or pid not in generation
        or any(
            process["pid"] != key
            or process["status"] in ("Z", "X", "x")
            or process["group"] != key
            or process["session"] != key
            or (key != pid and process["parent"] != pid)
            for key, process in generation.items()
        )
        or len(devices) != WORLD
        or any(len(row) != 3 or "A100" not in row[2] for row in devices)
        or {int(row[0]) for row in devices} != set(range(WORLD))
        or len({row[1] for row in devices}) != WORLD
        or len(occupied) != WORLD
        or any(len(row) != 2 for row in occupied)
        or {int(row[0]) for row in occupied} != set(generation) - {pid}
        or {row[1] for row in occupied} != {row[1] for row in devices}
    ):
        raise ValueError("Failed profiler restoration lacks its exact nine processes/eight physical devices")


def failed_profiler_evidence(root: Path) -> dict:
    """Serialize terminal files only; live keeper/marker acceptance is a later locked gate."""
    binding_path, helpers, state_path, receipt_path = predecessor_locations(root)
    intent_path = helpers.parent / "controller.json"
    paths = [binding_path, state_path, intent_path, receipt_path]
    before = {str(path.relative_to(root)): sha256(path) for path in paths}
    binding, state, intent, receipt = (read(path) for path in paths)
    if (
        binding["root"] != str(root)
        or binding["helpers"] != str(helpers)
        or binding["node"] != NODE
        or state["status"] != "needs_attention"
        or receipt["status"] != "needs_attention"
        or state["node"] != NODE
        or state["candidate"] != binding["candidate"]
        or state["binding_sha256"] != before[str(binding_path.relative_to(root))]
        or intent["binding_sha256"] != state["binding_sha256"]
        or intent["status"] != "acknowledged"
        or not intent["marker"]
        or not state["child_marker"]
        or intent["pid"] != state["controller_pid"]
        or intent["start_ticks"] != state["controller_start_ticks"]
        or receipt["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or type(state["exit_code"]) is not int
        or state["exit_code"] == 0
        or type(state["cleanup_exit_code"]) is not int
        or type(state["keeper_restore"]) is not int
        or state["keeper_restore"] != 0
        or state.get("marked_descendants_absent") is not True
        or state.get("keeper_release_complete") is not True
        or "cleanup_error" in state
        or not state.get("error")
        or not receipt.get("error")
        or binding["archive_sha256"] != sha256(Path(binding["archive"]))
        or any(
            Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != digest
            for name, digest in binding["helpers_sha256"].items()
        )
    ):
        raise ValueError("Attribution requires the preserved failed-profiler terminal ownership and restoration")
    require_keeper_shape(state["restored_keeper"])
    for path, digest in ((Path(state["log"]), state["log_sha256"]),):
        if not path.is_relative_to(root) or path.is_symlink() or sha256(path) != digest:
            raise ValueError("Failed profiler original log differs")
        before[str(path.relative_to(root))] = digest
    if any(sha256(root / name) != digest for name, digest in before.items()):
        raise ValueError("Failed profiler evidence changed during serialization")
    return {
        "status": HANDOFF_STATUS,
        "node": NODE,
        "evidence_sha256": before,
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
        "child_pid": state["child_pid"],
        "child_start_ticks": state["child_start_ticks"],
        "controller_marker_name": PREDECESSOR_CONTROLLER_MARKER,
        "controller_marker": intent["marker"],
        "child_marker_name": PREDECESSOR_CHILD_MARKER,
        "child_marker": state["child_marker"],
        "restored_keeper": state["restored_keeper"],
    }


def require_failed_profiler_handoff(
    expected: dict,
    current_keeper: dict,
    controller_marked: dict,
    child_marked: dict,
    original_handles: dict,
) -> None:
    """Pure predicate for fresh observations made by a future controller under the manager lock."""
    require_keeper_shape(current_keeper)
    keeper = expected["restored_keeper"]
    generation = {int(pid): process for pid, process in keeper["generation"].items()}
    if (
        expected["status"] != HANDOFF_STATUS
        or expected["node"] != NODE
        or not same_keeper(current_keeper, keeper)
        or child_marked
        or sorted(current_keeper["occupied"]) != sorted(keeper["occupied"])
        or set(original_handles) != {expected["controller_pid"], expected["child_pid"]}
        or any(process is not None and process["status"] not in ("Z", "X", "x") for process in original_handles.values())
        or any(
            int(pid) not in generation or not same_generation(generation[int(pid)], process)
            for pid, process in controller_marked.items()
        )
    ):
        raise RuntimeError("Failed profiler handoff has live descendants, unrelated markers or keeper drift")


def checkpoint_evidence(directory: Path, config: dict, candidate: Path, frozen: dict, step: int) -> dict:
    if step not in (BOUNDARY, ENDPOINT):
        raise ValueError("Attribution checkpoint is outside the exact bounded protocol")
    result = complete_checkpoint_evidence(directory, config, candidate, frozen, step)
    for rank in range(WORLD):
        state = torch.load(
            directory / f"checkpoints/step_{step:06d}/rank_{rank}.pt", map_location="cpu", weights_only=False
        )
        require_numerical_state(state, config, step, rank)
    return result


def require_tensor(value: object, shape: tuple, dtype: torch.dtype) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or not torch.isfinite(value).all()
    ):
        raise ValueError("Attribution checkpoint numerical tensor is missing, nonfinite or has different geometry")


def require_engine_state(
    engine: dict, config: dict, step: int, *, root_only: bool, require_cached_graph: bool = True
) -> None:
    """Check every restore-engine input independently of agreement between two checkpoints."""
    schema = {"groups", "momentum", "weight", "steps", "t", "fallbacks", "extra", "refresh_install_age"}
    if (
        set(engine) != schema
        or any(type(engine[key]) is not int or engine[key] != step for key in ("steps", "t"))
        or engine["refresh_install_age"] != 1
        or type(engine["weight"]) is not float
        or abs(engine["weight"] - (1 - config["statistics"]["momentum"] ** step)) > 1e-12
        or any(type(engine[key]) is not int or engine[key] < 0 for key in ("fallbacks", "extra"))
        or set(engine["groups"]) != {"d", "f"}
    ):
        raise ValueError("Attribution checkpoint root-engine schema, counters or EMA weight differs")
    layers, hidden, mlp = (config["model"][key] for key in ("layers", "hidden", "mlp"))
    local = layers // WORLD
    group_schema = {"C", "tol", "basis_step", "warm", "cached_root_graph_active", "SB", "S2", "pending"}
    for name, count, width in (("d", 3 * local, hidden), ("f", local, mlp)):
        group = engine["groups"][name]
        if (
            set(group) != group_schema
            or len(group["basis_step"]) != count
            or len(group["pending"]) != count
            or any(type(value) is not int or not 0 <= value < step for value in group["basis_step"])
            or type(group["cached_root_graph_active"]) is not bool
            or (require_cached_graph and group["cached_root_graph_active"] is not True)
            or type(group["warm"]) is not bool
        ):
            raise ValueError("Attribution checkpoint cached-root group or pending-refresh coverage differs")
        for field, shape, dtype in (
            ("C", (count, width, width), torch.float32),
            ("tol", (count, 3), torch.float32),
            ("SB", (count, 2 * width, width), torch.float16),
            ("S2", (count, width, 2 * width), torch.float16),
        ):
            require_tensor(group[field], shape, dtype)
        for pending in group["pending"]:
            if pending is None:
                continue
            if (
                set(pending) != {"step", "basis", "tol"}
                or type(pending["step"]) is not int
                or not 0 <= pending["step"] < step
            ):
                raise ValueError("Attribution checkpoint pending refresh lost its restore inputs")
            require_tensor(pending["basis"], (width, width), torch.float32)
            require_tensor(pending["tol"], (3,), torch.float32)
    shapes = {(hidden, hidden): 4 * local, (mlp, hidden): 2 * local, (hidden, mlp): local}
    if root_only:
        if engine["momentum"] != {}:
            raise ValueError("Attribution block input root-only engine has unexpected optimizer momentum")
    else:
        if set(engine["momentum"]) != set(shapes):
            raise ValueError("Attribution baseline checkpoint lacks its owned momentum shapes")
        for shape, count in shapes.items():
            momentum = engine["momentum"][shape]
            if set(momentum) != {"m"}:
                raise ValueError("Attribution baseline checkpoint lacks its MuonT momentum state")
            require_tensor(momentum["m"], (count, *shape), torch.float32)


def model_shapes(config: dict) -> dict[str, tuple]:
    model = config["model"]
    hidden, mlp, layers, vocab = (model[key] for key in ("hidden", "mlp", "layers", "vocab_size"))
    result = {"body.emb.weight": (vocab, hidden), "body.norm.weight": (hidden,), "head.weight": (vocab, hidden)}
    for layer in range(layers):
        for kind in ("ln1", "ln2", "q", "k", "v", "o", "gate", "up", "down"):
            shape = (hidden,)
            if kind in ("q", "k", "v", "o"):
                shape = (hidden, hidden)
            elif kind in ("gate", "up"):
                shape = (mlp, hidden)
            elif kind == "down":
                shape = (hidden, mlp)
            result[f"body.blocks.{layer}.{kind}.weight"] = shape
    return result


def require_numerical_state(
    state: dict, config: dict, step: int, rank: int, *, require_cached_graph: bool = True
) -> None:
    """Require complete fast, AdamW, model and RNG state trees needed for a real resume."""
    layers, hidden, mlp = (config["model"][key] for key in ("layers", "hidden", "mlp"))
    if layers % WORLD:
        raise ValueError("Attribution checkpoint requires equal full-layer ownership")
    family, fast = config["experiment"]["arm"], state["fast"]
    if family == "muont":
        require_engine_state(fast, config, step, root_only=False, require_cached_graph=require_cached_graph)
    else:
        if (
            set(fast) != {"lagged", "input", "output", "input_diagonal", "output_diagonal", "momentum", "steps"}
            or type(fast["steps"]) is not int
            or fast["steps"] != step
            or fast["lagged"] != {}
            or fast["input_diagonal"] is not None
            or fast["output"] is not None
        ):
            raise ValueError("Attribution checkpoint fused/null fast-state restore schema differs")
        require_engine_state(fast["input"], config, step, root_only=True, require_cached_graph=require_cached_graph)
        local = layers // WORLD
        shapes = {(hidden, hidden): 4 * local, (mlp, hidden): 2 * local, (hidden, mlp): local}
        if set(fast["momentum"]) != set(shapes):
            raise ValueError("Attribution checkpoint fused/null momentum coverage differs")
        for shape, count in shapes.items():
            require_tensor(fast["momentum"][shape], (count, *shape), torch.float32)
        diagonal = fast["output_diagonal"]
        if (
            set(diagonal) != {"moment", "weight"}
            or set(diagonal["moment"]) != {hidden, mlp}
            or type(diagonal["weight"]) is not float
            or abs(diagonal["weight"] - (1 - config["statistics"]["momentum"] ** step)) > 1e-12
        ):
            raise ValueError("Attribution checkpoint output moments or EMA weight differs")
        for width, count in ((hidden, 5 * layers), (mlp, 2 * layers)):
            require_tensor(diagonal["moment"][width], (count, width), torch.float32)
            if (diagonal["moment"][width] < 0).any():
                raise ValueError("Attribution checkpoint has negative output second moments")
    if state["output"] is not None:
        raise ValueError("Attribution checkpoint retained an unrelated standalone output engine")
    optimizers = state["optimizers"]
    if len(optimizers) != 2 or optimizers[0] is not None or set(optimizers[1]) != {"state", "param_groups"}:
        raise ValueError("Attribution checkpoint auxiliary optimizer restore schema differs")
    auxiliary = optimizers[1]
    groups = auxiliary["param_groups"]
    shapes = model_shapes(config)
    expected_auxiliary = [[shapes["head.weight"]], [shapes["body.emb.weight"]]]
    expected_auxiliary[1].extend(
        shapes[f"body.blocks.{layer}.{kind}.weight"] for layer in range(layers) for kind in ("ln1", "ln2")
    )
    expected_auxiliary[1].append(shapes["body.norm.weight"])
    if len(groups) != 2:
        raise ValueError("Attribution checkpoint auxiliary parameter groups differ")
    parameters = []
    for index, (group, dimensions) in enumerate(zip(groups, expected_auxiliary, strict=True)):
        settings = config["optimizer"]
        if (
            len(group["params"]) != len(dimensions)
            or group["lr"] != settings["auxiliary_lr"]
            or group["base_lr"] != settings["auxiliary_lr"]
            or tuple(group["betas"]) != tuple(settings["betas"])
            or group["eps"] != settings["eps"]
            or group["weight_decay"] != (settings["head_weight_decay"] if index == 0 else 0.0)
            or group["fused"] is not True
            or group["amsgrad"] is not False
        ):
            raise ValueError("Attribution checkpoint auxiliary optimization settings or coverage differs")
        parameters.extend(group["params"])
        for parameter, shape in zip(group["params"], dimensions, strict=True):
            saved = auxiliary["state"][parameter]
            if set(saved) != {"step", "exp_avg", "exp_avg_sq"} or saved["step"].item() != step:
                raise ValueError("Attribution checkpoint auxiliary optimizer moment state differs")
            require_tensor(saved["exp_avg"], shape, torch.float32)
            require_tensor(saved["exp_avg_sq"], shape, torch.float32)
    if len(set(parameters)) != len(parameters) or set(auxiliary["state"]) != set(parameters):
        raise ValueError("Attribution checkpoint auxiliary optimizer has missing or duplicated parameter state")
    if rank == 0:
        if set(state["model"]) != set(shapes):
            raise ValueError("Attribution checkpoint model parameter coverage differs")
        for name, shape in shapes.items():
            require_tensor(state["model"][name], shape, torch.float32)
    require_tensor(state["torch_rng"], tuple(torch.get_rng_state().shape), torch.uint8)
    torch.Generator(device="cpu").set_state(state["torch_rng"])
    # The pinned CUDA generator serializes its Philox seed and offset as two uint64 values.
    require_tensor(state["cuda_rng"], (16,), torch.uint8)
    offset = int.from_bytes(bytes(state["cuda_rng"][8:].tolist()), byteorder=sys.byteorder)
    if offset % 4:
        raise ValueError("Attribution checkpoint CUDA RNG has an unrestorable Philox offset")
    random.Random(0).setstate(state["python_rng"])


def require_capture_history(directory: Path, family: str) -> dict:
    rows = [
        json.loads(line)
        for path in sorted(directory.glob("metrics_from_*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    keys = ["diagonal_capture_parity_passed", "outer_diagonal_update_parity_passed", "output_work_retained_passed"]
    if family == "null_leftdiag_fused":
        keys.append("null_identity_outer_passed")
    if (
        family not in ("outer_leftdiag_fused", "null_leftdiag_fused")
        or [row["step"] for row in rows] != list(range(1, ENDPOINT + 1))
        or any(
            row["tokens"] != row["step"] * BATCH_TOKENS or any(row.get(key) is not True for key in keys) for row in rows
        )
    ):
        raise ValueError("Attribution strict capture/update/work evidence is incomplete")
    return {"updates": ENDPOINT, "checks": keys}


def numerical_modules(candidate: Path) -> tuple:
    """Load only the issued candidate's numerical readers and execution owner."""
    pinned = candidate / "artifacts/pinned-sources-materialized"
    for name, module in sys.modules.items():
        if name == "experiment" or name.startswith("experiment."):
            if not Path(module.__file__).resolve().is_relative_to(candidate):
                raise RuntimeError("Attribution validation has cached experiment code from another candidate")
        if name in ("reproduce", "toner") or name.startswith(("reproduce.", "toner.")):
            origin = module.__spec__.origin
            paths = list(module.__spec__.submodule_search_locations or ()) if origin is None else [origin]
            if not paths or any(not Path(path).resolve().is_relative_to(pinned) for path in paths):
                raise RuntimeError("Attribution validation has cached upstream code from another candidate")
    sys.path.insert(0, str(candidate))
    names = ("experiment.validate", "experiment.records", "experiment.fused_attribution_owner")
    modules = tuple(importlib.import_module(name) for name in names)
    if any(
        not Path(module.__file__).resolve().is_relative_to(candidate)
        for name, module in sys.modules.items()
        if (name == "experiment" or name.startswith("experiment."))
    ):
        raise RuntimeError("Attribution validation imported numerical code from another candidate")
    return modules


def require_source(candidate: Path, root: Path) -> dict:
    prepared = read(candidate / PREPARATION)
    if (
        prepared["status"] != PREPARED_STATUS
        or prepared["candidate"] != str(candidate)
        or prepared["root"] != str(root)
        or candidate_inventory(candidate) != prepared["files_sha256"]
        or prepared["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
        or any(sha256(candidate / name) != digest for name, digest in prepared["parent_metadata_sha256"].items())
    ):
        raise ValueError("Attribution validation candidate or prepared data changed")
    return prepared


def trajectory_evidence(directory: Path, config: dict, candidate: Path, frozen: dict, *, mode: str) -> dict:
    """Bind the independent prefix and complete trajectories to their actual source and checkpoints."""
    run = history(directory)
    interrupted = mode == "interrupted"
    endpoint = BOUNDARY if interrupted else ENDPOINT
    manifest_path = directory / INTERRUPTED_MANIFEST if interrupted else directory / "run.json"
    manifest = read(manifest_path)
    segments = sorted(directory.glob("metrics_from_*.jsonl"))
    prefix = directory / "metrics_from_000000.jsonl"
    expected_segments = ["metrics_from_000000.jsonl"]
    if mode == "resumed":
        expected_segments.append(f"metrics_from_{BOUNDARY:06d}.jsonl")
    rows = run.segments[0] if interrupted else run.rows
    resume = str(directory / f"checkpoints/step_{BOUNDARY:06d}") if mode == "resumed" else None
    if (
        mode not in MODES
        or manifest["config"] != config
        or read(directory / "config.json") != config
        or manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
        or normalized_sources(candidate, manifest["source_sha256"]) != executed_sources(candidate, frozen)
        or manifest["status"] != "execution_check_complete"
        or manifest.get("execution_check") is not True
        or type(manifest["completed_steps"]) is not int
        or manifest["completed_steps"] != endpoint
        or manifest.get("resume") != resume
        or len(rows) != endpoint
        or (not interrupted and [path.name for path in segments] != expected_segments)
        or (mode == "resumed" and [len(segment) for segment in run.segments] != [BOUNDARY, ENDPOINT - BOUNDARY])
        or prefix.name != segments[0].name
        or any(
            path.is_symlink() or path.stat().st_nlink != 1 or not path.read_bytes().endswith(b"\n")
            for path in [manifest_path, directory / "config.json", *segments]
        )
    ):
        raise ValueError("Attribution trajectory source, independent prefix or manifest coverage differs")
    installations = 0
    for row in rows:
        for entry in row["input_refresh_installations"]:
            if entry["installation_step"] != row["step"] - 1 or entry["installation_step"] - entry["source_step"] != 1:
                raise ValueError("Attribution trajectory did not retain fixed installation age 1")
            installations += 1
    if not installations:
        raise ValueError("Attribution trajectory did not exercise host-refresh installation")
    require_evaluations(rows, config, endpoint)
    paths = [prefix] if interrupted else segments
    return {
        "manifest_sha256": sha256(manifest_path),
        "history_sha256": {path.name: sha256(path) for path in paths},
        "observed_updates": len(rows),
        "observed_tokens": rows[-1]["tokens"],
        "fixed_age_history": {"rank": 0, "updates": len(rows), "installation_age": 1, "installations": installations},
        "checkpoints": {
            str(step): checkpoint_evidence(directory, config, candidate, frozen, step)
            for step in (BOUNDARY, ENDPOINT)
            if step <= endpoint
        },
    }


def require_evaluations(rows: list[dict], config: dict, endpoint: int) -> None:
    expected = {
        step
        for step in range(1, endpoint + 1)
        if (step - 1) % config["runtime"]["eval_every"] == 0 or step in (BOUNDARY, endpoint)
    }
    heldouts = {f"heldout/{name}" for name in config["data"]["mixture_weights"]}
    if {
        row["step"] for row in rows if any(key in row for key in ("c4_loss", "selection_loss", "heldout_losses"))
    } != expected:
        raise ValueError("Attribution trajectory lacks its configured boundary/endpoint evaluation coverage")
    for row in rows:
        if row["step"] not in expected:
            continue
        if (
            any(key not in row for key in ("c4_loss", "selection_loss", "heldout_losses"))
            or set(row["heldout_losses"]) != heldouts
            or any(not math.isfinite(value) for value in row["heldout_losses"].values())
        ):
            raise ValueError("Attribution trajectory omitted expected held-out evaluation values")


def save_interrupted_manifest(directory: Path) -> None:
    original = directory / "run.json"
    if original.is_symlink() or original.stat().st_nlink != 1:
        raise ValueError("Attribution interrupted manifest is aliased")
    raw = original.read_bytes()
    with (directory / INTERRUPTED_MANIFEST).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if original.read_bytes() != raw:
        raise ValueError("Attribution interrupted manifest changed during preservation")


def loss_continuation_evidence(continuous: Path, resumed: Path) -> dict:
    """Compare full observed loss histories, including held-out values, at unchanged tolerances."""
    expected, actual = history(continuous).rows, history(resumed).rows
    if len(expected) != ENDPOINT or len(actual) != ENDPOINT:
        raise ValueError("Attribution loss continuation lacks both complete bounded trajectories")
    config = read(continuous / "config.json")
    require_evaluations(expected, config, ENDPOINT)
    require_evaluations(actual, config, ENDPOINT)
    errors = []
    for reference, observed in zip(expected, actual, strict=True):
        for key in ("train_loss", "c4_loss", "selection_loss", "heldout_losses"):
            if (key in reference) != (key in observed):
                raise ValueError("Attribution loss continuation evaluation coverage differs")
            if key not in reference:
                continue
            first, second = reference[key], observed[key]
            if key == "heldout_losses":
                if first.keys() != second.keys():
                    raise ValueError("Attribution loss continuation held-out identities differ")
                values = [(first[name], second[name]) for name in first]
            else:
                values = [(first, second)]
            for baseline, value in values:
                if (
                    isinstance(baseline, bool)
                    or isinstance(value, bool)
                    or not math.isfinite(baseline)
                    or not math.isfinite(value)
                    or not math.isclose(
                        value, baseline, rel_tol=CONTINUATION_TOLERANCE["rtol"], abs_tol=CONTINUATION_TOLERANCE["atol"]
                    )
                ):
                    raise ValueError("Attribution loss continuation differs beyond unchanged tolerances")
                errors.append(abs(value - baseline))
    return {"scalar_comparisons": len(errors), "max_absolute_error": max(errors, default=0.0)}


def trial_evidence(item: dict, candidate: Path, frozen: dict) -> dict:
    """Reconstruct actual all-rank continuation and retained output-work evidence."""
    root = Path(frozen["root"])
    if require_source(candidate, root) != frozen:
        raise ValueError("Attribution trial numerical source changed before import")
    validation, records, _ = numerical_modules(candidate)
    config = read(Path(item["config"]))
    require_config(config, {key: item[key] for key in trial_specs()[0]})
    continuous, resumed = Path(item["continuous"]), Path(item["resumed"])
    evidence = {
        "trajectory": {
            "continuous": trajectory_evidence(continuous, config, candidate, frozen, mode="continuous"),
            "interrupted": trajectory_evidence(resumed, config, candidate, frozen, mode="interrupted"),
            "resumed": trajectory_evidence(resumed, config, candidate, frozen, mode="resumed"),
        },
        "cached_root_checkpoint": {
            name: validation.require_cached_root_checkpoint(directory / f"checkpoints/step_{BOUNDARY:06d}", WORLD)
            for name, directory in (("continuous", continuous), ("resumed", resumed))
        },
        "boundary_continuation": records.compare_continuation(
            continuous / f"checkpoints/step_{BOUNDARY:06d}", resumed / f"checkpoints/step_{BOUNDARY:06d}", WORLD
        ),
        "post_cached_root_continuation": records.compare_continuation(
            continuous / f"checkpoints/step_{ENDPOINT:06d}", resumed / f"checkpoints/step_{ENDPOINT:06d}", WORLD
        ),
        "loss_continuation": loss_continuation_evidence(continuous, resumed),
    }
    if item["family"] != "muont":
        evidence["capture_history"] = {
            name: require_capture_history(directory, item["family"])
            for name, directory in (("continuous", continuous), ("resumed", resumed))
        }
    return item | {"evidence": evidence, "checks": dict.fromkeys(item["checks"], True)}


def reconstruct(root: Path, candidate: Path, receipt_path: Path) -> dict:
    """Accept only original complete command histories and independently reproduced numerical evidence."""
    if receipt_path != locations(root)[4]:
        raise ValueError("Attribution reconstruction requires its canonical GPU receipt")
    prepared = require_source(candidate, root)
    expected = plan(candidate, root, locations(root)[3])
    actual = read(receipt_path)
    state = read(locations(root)[2])
    frozen = read(locations(root)[0])
    owner = {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
    if (
        actual["status"] != PASS_STATUS
        or "error" in actual
        or actual["owner"] != owner
        or frozen["status"] != FROZEN_STATUS
        or frozen["candidate"] != str(candidate)
        or frozen["files_sha256"] != prepared["files_sha256"]
        or frozen["preparation_sha256"] != sha256(candidate / PREPARATION)
        or frozen["commands"] != validation_commands(candidate, root, locations(root)[3])
        or state["binding_sha256"] != sha256(locations(root)[0])
        or actual["execution_binding_sha256"] != state["binding_sha256"]
        or actual["candidate_preparation_sha256"] != frozen["preparation_sha256"]
        or any(
            actual[key] != value for key, value in expected.items() if key not in ("status", "trials", "gpu_commands")
        )
        or len(actual["trials"]) != len(expected["trials"])
        or len(actual["gpu_commands"]) != len(expected["trials"]) * len(MODES)
    ):
        raise ValueError("Attribution GPU receipt has incomplete execution or a different bounded protocol")
    executions = iter(actual["gpu_commands"])
    for item, measured in zip(expected["trials"], actual["trials"], strict=True):
        rebuilt = trial_evidence(item, candidate, prepared)
        if measured != rebuilt:
            raise ValueError("Attribution checkpoint/history evidence differs from independent reconstruction")
        for mode, command in zip(MODES, item["commands"], strict=True):
            execution = next(executions)
            log = locations(root)[3] / f"{item['name']}_{mode}.log"
            if (
                execution["trial"] != item["name"]
                or execution["mode"] != mode
                or execution["command"] != command
                or type(execution["exit_code"]) is not int
                or execution["exit_code"] != 0
                or execution["log"] != str(log)
                or execution["log_sha256"] != sha256(log)
                or execution["trajectory"] != rebuilt["evidence"]["trajectory"][mode]
            ):
                raise ValueError("Attribution original command, trajectory or log identity differs")
    if require_source(candidate, root) != prepared:
        raise ValueError("Attribution source changed during independent reconstruction")
    return actual


def run(root: Path, candidate: Path, work: Path, receipt_path: Path) -> dict:
    """Execute once under the acknowledged owner, preserving every failure and original history."""
    result = plan(candidate, root, work)
    if receipt_path != locations(root)[4] or work.exists() or receipt_path.exists():
        raise FileExistsError("Attribution GPU validation requires unused canonical evidence")
    prepared = require_source(candidate, root)
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("Attribution validator explicit CUDA environment differs")
    validation, _, owner = numerical_modules(candidate)
    result["owner"] = owner.require_gpu_owner(root)
    result["execution_binding_sha256"] = sha256(locations(root)[0])
    result["candidate_preparation_sha256"] = sha256(candidate / PREPARATION)
    validation.require_receipt(root / "data", root / "data/experiment_receipt.json")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Attribution validator refused occupied GPUs")
    work.mkdir(parents=True, exist_ok=False)
    result["status"] = "validating"
    write(receipt_path, result)
    try:
        for item in result["trials"]:
            config = read(Path(item["config"]))
            for mode, command in zip(MODES, item["commands"], strict=True):
                if owner.require_gpu_owner(root) != result["owner"]:
                    raise RuntimeError("Attribution validator ownership changed between commands")
                state_path = locations(root)[2]
                state = read(state_path)
                state["active_command"] = command
                write(state_path, state)
                log = work / f"{item['name']}_{mode}.log"
                with log.open("x") as stream:
                    try:
                        code = subprocess.run(
                            command,
                            cwd=candidate,
                            env=os.environ | ENVIRONMENT,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            timeout=COMMAND_TIMEOUT,
                        ).returncode
                    except subprocess.TimeoutExpired:
                        code = 124
                execution = {
                    "trial": item["name"],
                    "mode": mode,
                    "command": command,
                    "exit_code": code,
                    "log": str(log),
                    "log_sha256": sha256(log),
                }
                result["gpu_commands"].append(execution)
                write(receipt_path, result)
                if code:
                    raise RuntimeError(f"Attribution bounded command failed; preserve {log}")
                directory = Path(item["continuous"] if mode == "continuous" else item["resumed"])
                if mode == "interrupted":
                    save_interrupted_manifest(directory)
                execution["trajectory"] = trajectory_evidence(directory, config, candidate, prepared, mode=mode)
                write(receipt_path, result)
            item.update(trial_evidence(item, candidate, prepared))
            write(receipt_path, result)
        result["status"] = PASS_STATUS
        write(receipt_path, result)
        return reconstruct(root, candidate, receipt_path)
    except BaseException as error:
        result.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        write(receipt_path, result)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--reconstruct", action="store_true")
    args = parser.parse_args()
    root, candidate, work = args.root.resolve(), args.candidate.resolve(), args.work.resolve()
    if args.reconstruct and args.receipt is None:
        parser.error("Reconstruction requires --receipt")
    if args.receipt is None:
        result = plan(candidate, root, work)
    elif args.reconstruct:
        result = reconstruct(root, candidate, args.receipt.resolve())
    else:
        result = run(root, candidate, work, args.receipt.resolve())
    print(json.dumps({"status": result["status"], "trials": len(result["trials"])}))


if __name__ == "__main__":
    main()
