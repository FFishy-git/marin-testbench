# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan and compare full-budget NS quality/cost studies without granting execution authority."""

import copy
import math
from pathlib import Path

import ns_full_model_contract as contract
from analyze_tuning import executed_sources, normalized_sources
from compare_runs import WINDOW, RunHistory, compare, completion, evaluations, history, runtime, window_means
from fused_attribution_validation_contract import require_config as require_reference
from fused_attribution_validation_contract import trial_specs as reference_specs
from selected_lr_release_owner import candidate_files, read, sha256

STEPS = 5120
TOKENS = STEPS * contract.BATCH_TOKENS
SEEDS = (3, 4, 5)
BASELINE = "muont_ns_tpu_q10"
PLAN_STATUS = "ns_budget_full_budget_study_prepared_execution_unissued"
ANALYSIS_STATUS = "ns_budget_full_budget_history_analysis_not_production_acceptance"
SCOPE = (
    "Source-bound study protocol and history comparisons only. No allocation, execution owner, "
    "GPU release, production acceptance, cloud verification or optimizer-superiority claim. "
    "The current eighteen-case lower-LR diagnostic does not authorize center/upper LR. "
    "Each actual LR configuration requires its own unchanged-tolerance compiled/cached-root "
    "all-rank boundary22/post-cached-root24, scalar-loss and fused/null captured-row/update/work "
    "gates under the eventual production source. History rankings are provisional until exact "
    "worker cleanup/restoration and private full-resolution cloud evidence are verified. "
    "All cases receive equal LR search allowance; C4 is reporting-only. Boundary winners require "
    "symmetric expansion for every compared case, followed by paired fresh seeds3/4/5."
)


def case_name(family: str, operand: str, budget: int) -> str:
    return f"{family}_ns_{operand}_q{budget}"


def trial_specs() -> tuple[dict, ...]:
    """Retain every family/mode/budget and its equal three-point LR allowance."""
    return tuple(
        {
            "name": f"{case_name(family, operand, budget)}_full_lr_{label}",
            "case": case_name(family, operand, budget),
            "family": family,
            "operand": operand,
            "budget": budget,
            "lr_label": label,
            "matrix_lr": lr,
            "config_name": f"experiment/configs/{case_name(family, operand, budget)}_full_lr_{label}_0p5m_seed0.json",
            "lower_lr_screen": f"{case_name(family, operand, budget)}_full_lr_lower",
        }
        for operand in contract.OPERANDS
        for budget in contract.BUDGETS
        for label, lr in contract.LR_GRID
        for family in contract.FAMILIES
    )


def actual_config(candidate: Path, spec: dict) -> dict:
    """Allow only LR, operand mode and NS budget changes from the preserved lower-LR family."""
    if spec not in trial_specs():
        raise ValueError("NS study trial is outside its equal-budget protocol")
    reference_spec = next(
        item for item in reference_specs() if item["family"] == spec["family"] and item["lr_label"] == "lower"
    )
    paths = (candidate / reference_spec["config_name"], candidate / spec["config_name"])
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("NS study actual or preserved reference configuration is absent or aliased")
    reference, config = (read(path) for path in paths)
    require_reference(reference, reference_spec)
    expected = copy.deepcopy(reference)
    expected["optimizer"].update(lr=spec["matrix_lr"], ns_mode=spec["operand"], ns_steps=spec["budget"])
    if config != expected:
        raise ValueError("NS study changed settings outside the exact LR/operand/budget transformation")
    return config


def plan(root: Path, candidate: Path, frozen: dict) -> dict:
    """Prepare all comparisons and required gates without copying or changing training source."""
    if (
        root != root.resolve()
        or candidate != candidate.resolve()
        or candidate.is_symlink()
        or frozen["candidate"] != str(candidate)
        or frozen["files_sha256"] != candidate_files(candidate)
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
    ):
        raise ValueError("NS study source or data is not its authenticated parent")
    trials = []
    for spec in trial_specs():
        config = actual_config(candidate, spec)
        trials.append(
            spec
            | {
                "config": config,
                "config_sha256": sha256(candidate / spec["config_name"]),
                "out": str(root / "runs/ns-budget-study" / spec["name"]),
                "required_actual_config_gate": {
                    "configuration_sha256": sha256(candidate / spec["config_name"]),
                    "lr_label": spec["lr_label"],
                    "production_source_revalidation_required": True,
                    "continuous_endpoint": 24,
                    "interrupted_endpoint": 22,
                    "resumed_endpoint": 24,
                    "checkpoint_steps": [22, 23, 24],
                    "first_resumed_update_diagnostic_only": True,
                    "tolerance": contract.TOLERANCE,
                    "capture_update_work_required": spec["family"] != "muont",
                    "null_identity_required": spec["family"] == "null_leftdiag_fused",
                    "current_state": "unissued",
                },
            }
        )
    return {
        "status": PLAN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "files_sha256": frozen["files_sha256"],
        "data_receipt_sha256": frozen["data_receipt_sha256"],
        "executed_source_sha256": executed_sources(candidate, frozen),
        "trials": trials,
        "canonical_baseline_case": BASELINE,
        "training_steps": STEPS,
        "tokens_per_trial": TOKENS,
        "planned_grid_tokens": len(trials) * TOKENS,
        "selection": {
            "metric": "mean_exact_final_half_selection_loss",
            "training_window_tokens": WINDOW,
            "report_final_two_complete_windows": True,
            "c4_reporting_only": True,
            "tie_break": "trial_name",
        },
        "expansion": {
            "trigger": "any completed case winner is lower or upper",
            "factor": math.sqrt(2),
            "additional_matrix_lrs": [contract.LR_GRID[0][1] / math.sqrt(2), contract.LR_GRID[-1][1] * math.sqrt(2)],
            "apply_equally_to_all_compared_cases": True,
            "new_actual_config_gates_required": True,
        },
        "confirmation_seeds": list(SEEDS),
        "pending_lower_lr_terminal": str(root / "run-preparation/ns-full-model-terminal-independent-20261119.json"),
        "execution_authorized": False,
        "scope": SCOPE,
    }


def require_plan(record: dict) -> dict:
    """Authenticate the current source and full protocol before reading any run history."""
    root, candidate = Path(record["root"]), Path(record["candidate"])
    frozen = {key: record[key] for key in ("candidate", "files_sha256", "data_receipt_sha256")}
    if {key: value for key, value in record.items() if key != "time"} != plan(root, candidate, frozen):
        raise ValueError("NS study plan, equal LR allowance or original source changed")
    return record


def measured_history(record: dict, trial: dict, run: RunHistory) -> dict:
    """Summarize complete common-token metrics without interpreting them as owner acceptance."""
    config, candidate = trial["config"], Path(record["candidate"])
    if (
        run.manifest["config"] != config
        or run.manifest["data_receipt_sha256"] != record["data_receipt_sha256"]
        or normalized_sources(candidate, run.manifest["source_sha256"]) != record["executed_source_sha256"]
        or run.manifest.get("execution_check")
        or len(run.segments) != 1
    ):
        raise ValueError("NS study history configuration, source, data or uninterrupted provenance differs")
    train = window_means(run.rows, WINDOW)
    progress = completion(run)
    expected_steps = sorted(set(range(1, min(len(run.rows), STEPS) + 1, config["runtime"]["eval_every"])))
    if progress["complete"]:
        expected_steps = sorted(set(expected_steps) | {STEPS})
    selected = evaluations(run.rows, "selection_loss")
    c4 = evaluations(run.rows, "c4_loss")
    if (
        sorted(selected) != [step * contract.BATCH_TOKENS for step in expected_steps]
        or sorted(c4) != sorted(selected)
        or (run.manifest.get("status") == "complete" and not progress["complete"])
    ):
        raise ValueError("NS study history is truncated or lacks exact held-out evaluation coverage")
    weights = config["data"]["mixture_weights"]
    for row in run.rows:
        if "selection_loss" not in row:
            continue
        heldout = row.get("heldout_losses", {})
        if set(heldout) != {f"heldout/{name}" for name in weights} or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in heldout.values()
        ):
            raise ValueError("NS study reserved mixture is missing sources or contains nonfinite values")
        expected = sum(weight * heldout[f"heldout/{name}"] for name, weight in weights.items()) / sum(weights.values())
        if row["selection_loss"] != expected:
            raise ValueError("NS study selection scalar differs from its complete held-out mixture")
    late = [token for token in selected if token > TOKENS // 2]
    if progress["complete"] and (len(train) != 5 or len(selected) != 22 or len(late) != 11):
        raise ValueError("NS study completed history lacks five large windows and 22/11 exact evaluations")
    return {
        "name": trial["name"],
        "case": trial["case"],
        "family": trial["family"],
        "operand": trial["operand"],
        "budget": trial["budget"],
        "lr_label": trial["lr_label"],
        "matrix_lr": trial["matrix_lr"],
        "status": "complete_history_not_production_accepted" if progress["complete"] else "incomplete",
        "history_complete": progress["complete"],
        "observed_tokens": progress["observed_tokens"],
        "training_windows": [{"end_tokens": token, "mean": mean} for token, mean in train.items()],
        "late_selection_tokens": late,
        "mean_late_selection_loss": math.fsum(selected[token] for token in late) / len(late) if late else None,
        "final_selection_loss": selected.get(TOKENS) if progress["complete"] else None,
        "mean_late_c4_loss_reporting_only": math.fsum(c4[token] for token in late) / len(late) if late else None,
        "runtime": runtime(run, progress["observed_tokens"]),
        "nominal_ns_products": 3 * trial["budget"],
        "nominal_ns_product_ratio_to_q10": trial["budget"] / 10,
    }


def comparisons_for(record: dict, runs: dict[str, RunHistory]) -> list[dict]:
    reports = []
    for operand in contract.OPERANDS:
        for budget in contract.BUDGETS:
            for label, _ in contract.LR_GRID:
                names = {family: f"{case_name(family, operand, budget)}_full_lr_{label}" for family in contract.FAMILIES}
                canonical = f"{BASELINE}_full_lr_{label}"
                available = {name: runs[name] for name in (*names.values(), canonical) if name in runs}
                baselines = {
                    "canonical_muont_same_lr": canonical,
                    "same_finite_map_muont": names["muont"],
                    "matched_null_same_lr": names["null_leftdiag_fused"],
                }
                for role, baseline in baselines.items():
                    if baseline not in available or len(available) < 2:
                        continue
                    result = compare(available, baseline, record["selection"]["training_window_tokens"])
                    for item in result["comparisons"].values():
                        item["final_two_complete_training_windows"] = item["training_windows"][-2:]
                    reports.append({"operand": operand, "budget": budget, "lr_label": label, "role": role} | result)
    return reports


def analyze_histories(record: dict, runs: dict[str, RunHistory]) -> dict:
    """Withhold provisional LR choices until every planned trial in that case completes."""
    specs = {trial["name"]: trial for trial in record["trials"]}
    if set(runs) - specs.keys():
        raise ValueError("NS study received a run outside its fixed equal-budget grid")
    summaries = {name: measured_history(record, specs[name], run) for name, run in runs.items()}
    choices, incomplete = {}, {}
    for case in dict.fromkeys(trial["case"] for trial in record["trials"]):
        planned = [trial["name"] for trial in record["trials"] if trial["case"] == case]
        missing = [name for name in planned if name not in summaries or not summaries[name]["history_complete"]]
        if missing:
            incomplete[case] = missing
            continue
        best = min(planned, key=lambda name: (summaries[name]["mean_late_selection_loss"], name))
        choices[case] = {
            key: summaries[best][key]
            for key in ("name", "lr_label", "matrix_lr", "mean_late_selection_loss", "final_selection_loss", "runtime")
        }
    boundary = [case for case, choice in choices.items() if choice["lr_label"] != "center"]
    tuned = []
    if BASELINE in choices:
        baseline = choices[BASELINE]["name"]
        selected = {choice["name"]: runs[choice["name"]] for choice in choices.values()}
        tuned.append(
            {"role": "provisional_selected_case_vs_selected_canonical_muont"} | compare(selected, baseline, WINDOW)
        )
        for operand in contract.OPERANDS:
            for budget in contract.BUDGETS:
                fused, null = (case_name(family, operand, budget) for family in contract.FAMILIES[1:])
                if fused in choices and null in choices:
                    names = (choices[fused]["name"], choices[null]["name"])
                    tuned.append(
                        {"role": "provisional_independently_tuned_fused_vs_matched_null"}
                        | compare({name: runs[name] for name in names}, choices[null]["name"], WINDOW)
                    )
    return {
        "status": ANALYSIS_STATUS,
        "planned_trials": len(specs),
        "observed_trials": len(runs),
        "complete_histories": sum(item["history_complete"] for item in summaries.values()),
        "runs": summaries,
        "incomplete_case_grids": incomplete,
        "provisional_history_lr_choices": choices,
        "boundary_winners": boundary,
        "symmetric_expansion_required": bool(boundary),
        "expansion": record["expansion"] if boundary else None,
        "same_lr_comparisons": comparisons_for(record, runs),
        "provisional_tuned_comparisons": tuned,
        "confirmation_seeds": record["confirmation_seeds"],
        "production_acceptance": False,
        "private_cloud_verification": False,
        "superiority_claim_permitted": False,
        "scope": SCOPE,
    }


def analyze(record: dict) -> dict:
    """Read stable local history bytes, never a tensor reader, uploader or GPU controller."""
    require_plan(record)
    runs, evidence = {}, {}
    for trial in record["trials"]:
        directory = Path(trial["out"])
        if not (directory / "run.json").exists():
            continue
        paths = [directory / "run.json", directory / "config.json", *sorted(directory.glob("metrics_from_*.jsonl"))]
        if (
            directory.is_symlink()
            or [path.name for path in paths[2:]] != ["metrics_from_000000.jsonl"]
            or any(path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 for path in paths)
            or read(directory / "config.json") != trial["config"]
        ):
            raise ValueError("NS study saved history files are shared, aliased or lack their original configuration")
        hashes = {str(path): sha256(path) for path in paths}
        run = history(directory)
        if completion(run)["complete"]:
            folder = directory / "checkpoints/step_005120"
            expected = {"complete.json", *(f"rank_{rank}.pt" for rank in range(contract.WORLD))}
            if (
                folder.is_symlink()
                or {path.name for path in folder.iterdir()} != expected
                or any(
                    path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size == 0
                    for path in folder.iterdir()
                )
            ):
                raise ValueError("NS study complete history lacks private complete eight-rank checkpoints")
            receipt = read(folder / "complete.json")
            if type(receipt["step"]) is not int or receipt["step"] != STEPS or receipt["world"] != contract.WORLD:
                raise ValueError("NS study checkpoint receipt differs from the full budget")
            hashes[str(folder / "complete.json")] = sha256(folder / "complete.json")
        runs[trial["name"]] = run
        evidence.update(hashes)
    result = analyze_histories(record, runs)
    if any(sha256(Path(path)) != digest for path, digest in evidence.items()):
        raise RuntimeError("NS study saved histories changed during comparison")
    require_plan(record)
    return result | {
        "history_evidence_sha256": evidence,
        "checkpoint_scope": "Complete rank-file layout and receipt only; tensor numerics and owners are not rederived.",
    }
