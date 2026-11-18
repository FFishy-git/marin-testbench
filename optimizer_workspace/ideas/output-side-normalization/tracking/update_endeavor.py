# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Refresh a bounded ENDEAVOR ledger from saved monitoring and result receipts."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

from analyze_upstream_snapshot import analyze_capture as analyze_wave9_capture
from analyze_upstream_snapshot import captured_cohort, refresh_running_capture, verify_production_binding
from analyze_wave7_snapshot import analyze_capture
from analyze_wave8_snapshot import analyze_capture as analyze_wave8_capture
from fused_attribution_validation import PASS_STATUS as ATTRIBUTION_VALIDATION_PASS_STATUS
from fused_attribution_validation import failed_profiler_evidence
from fused_attribution_validation import plan as attribution_validation_plan
from fused_attribution_validation_contract import FROZEN_STATUS as ATTRIBUTION_VALIDATION_FROZEN_STATUS
from fused_attribution_validation_contract import MODES as ATTRIBUTION_VALIDATION_MODES
from fused_attribution_validation_contract import NODE as ATTRIBUTION_VALIDATION_NODE
from fused_attribution_validation_contract import PREPARATION as ATTRIBUTION_VALIDATION_PREPARATION
from fused_attribution_validation_contract import PREPARED_STATUS as ATTRIBUTION_VALIDATION_PREPARED_STATUS
from fused_attribution_validation_contract import PYTHON as ATTRIBUTION_VALIDATION_PYTHON
from fused_attribution_validation_contract import controller_command as attribution_validation_controller_command
from fused_attribution_validation_contract import locations as attribution_validation_locations
from fused_attribution_validation_contract import trial_specs as attribution_validation_trial_specs
from fused_attribution_validation_contract import validator_command as attribution_validation_validator_command
from fused_continuation_contract import CPU_STATUS as CONTINUATION_CPU_STATUS
from fused_continuation_contract import FROZEN_STATUS as CONTINUATION_FROZEN_STATUS
from fused_continuation_contract import PREPARATION as CONTINUATION_PREPARATION
from fused_continuation_contract import PREPARED_STATUS as CONTINUATION_PREPARED_STATUS
from fused_continuation_contract import controller_command as continuation_controller_command
from fused_continuation_contract import locations as continuation_locations
from fused_continuation_contract import plan as continuation_plan
from fused_continuation_contract import validator_command as continuation_validator_command
from fused_continuation_controller import CONTROLLER_CPU_STATUS as CONTINUATION_CONTROLLER_CPU_STATUS
from fused_continuation_controller import cpu_command as continuation_controller_cpu_command
from fused_continuation_controller import fixture_names as continuation_controller_fixture_names
from fused_continuation_controller import inventory as continuation_inventory
from fused_continuation_validation import COMPLETE_STATUS as CONTINUATION_COMPLETE_STATUS
from owned_output_production import require_archive as require_owned_production_archive
from owned_output_run_history import CHANGES as OWNED_PRODUCTION_CHANGES
from prepare_fixed_age_tuning import require_source as require_selected_lr_source
from prepare_fused_attribution import CPU_STATUS as ATTRIBUTION_CPU_STATUS
from prepare_fused_attribution import PREPARATION as ATTRIBUTION_PREPARATION
from prepare_fused_attribution import STATUS as ATTRIBUTION_STATUS
from prepare_fused_attribution import archive_inventory as attribution_archive_inventory
from prepare_fused_attribution import transformed_files as attribution_transformed_files
from prepare_fused_attribution_validation import CHANGES as ATTRIBUTION_VALIDATION_CHANGES
from prepare_fused_attribution_validation import COPIES as ATTRIBUTION_VALIDATION_COPIES
from prepare_fused_attribution_validation import CPU_STATUS as ATTRIBUTION_VALIDATION_CPU_STATUS
from prepare_fused_attribution_validation import EXPANSION_POLICY as ATTRIBUTION_VALIDATION_EXPANSION_POLICY
from prepare_fused_attribution_validation import archive_inventory as attribution_validation_archive_inventory
from prepare_fused_attribution_validation import transformed_train as attribution_validation_transformed_train
from prepare_lr_ceiling import require_recovery
from prepare_owned_output_handoff import CHANGES as HANDOFF_CHANGES
from prepare_owned_output_handoff import PREPARATION as HANDOFF_PREPARATION
from prepare_owned_output_handoff import require_cpu_archive as require_handoff_archive
from prepare_owned_output_trace import CHANGES as TRACE_CHANGES
from prepare_owned_output_trace import PREPARATION as TRACE_PREPARATION
from prepare_owned_output_trace import require_cpu_archive as require_trace_archive
from prepare_owned_output_validation import cpu_command as owned_output_cpu_command
from prepare_root_preparation_diagnostic import CHANGES as ROOT_PREPARATION_CHANGES
from prepare_root_preparation_diagnostic import REPLAY_OBSERVATION as ROOT_PREPARATION_PREDECESSOR
from prepare_root_preparation_diagnostic import REPLAY_OBSERVATION_SHA256 as ROOT_PREPARATION_PREDECESSOR_SHA256
from prepare_root_preparation_diagnostic import archive_inventory as root_preparation_archive_inventory
from prepare_root_preparation_diagnostic import cpu_command as root_preparation_cpu_command
from prepare_selected_lr_full_budget import CHANGES as SELECTED_LR_CHANGES
from prepare_selected_lr_full_budget import PREPARATION as SELECTED_LR_PREPARATION
from prepare_selected_lr_full_budget import full_config as selected_lr_full_config
from prepare_selected_lr_production import CHANGES as SELECTED_LR_DENIED_CHANGES
from prepare_selected_lr_validation import require_archive as require_selected_lr_archive
from prepare_upstream_full_model import cpu_test_commands
from root_preparation_contract import CPU_STATUS as ROOT_PREPARATION_CPU_STATUS
from root_preparation_contract import PREPARATION as ROOT_PREPARATION
from root_preparation_contract import PREPARED_STATUS as ROOT_PREPARATION_STATUS
from root_preparation_contract import plan as root_preparation_plan
from root_preparation_reporting import executor_lines as root_executor_lines
from saved_ns_gpu_owner import CPU_STATUS as SAVED_NS_CPU_STATUS
from saved_ns_gpu_owner import WORKER_SOURCES as SAVED_NS_WORKER_SOURCES
from saved_ns_gpu_owner import cpu_command as saved_ns_cpu_command
from selected_lr_release_owner import CHANGES as SELECTED_LR_RELEASE_CHANGES
from selected_lr_release_owner import sha256
from selected_lr_run_history import STATUS as SELECTED_LR_PRODUCTION_STATUS
from selected_lr_stage_profile_contract import FROZEN_STATUS as PROFILE_FROZEN_STATUS
from selected_lr_stage_profile_contract import controller_command as profile_controller_command
from selected_lr_stage_profile_contract import locations as profile_locations
from selected_lr_stage_profile_contract import validator_command as profile_validator_command
from selected_lr_stage_profile_validation import plan as profile_validation_plan
from selected_lr_validation import NODE as SELECTED_LR_NODE
from selected_lr_validation import PREPARATION as SELECTED_LR_VALIDATION_PREPARATION
from selected_lr_validation import STATUS as SELECTED_LR_VALIDATION_STATUS
from selected_lr_validation import controller_command as selected_lr_controller_command
from selected_lr_validation import locations as selected_lr_locations
from selected_lr_validation import plan as selected_lr_validation_plan
from selected_lr_validation import validator_command as selected_lr_validator_command
from upstream_production import cpu_command
from upstream_run_history import CHANGES
from wave7_release import candidate_inventory

START = "<!-- BEGIN AUTOMATED WAVE LEDGER -->"
END = "<!-- END AUTOMATED WAVE LEDGER -->"
ANCHOR = "## Verification and Remaining Work"
LOCAL_ZONE = ZoneInfo("Asia/Shanghai")
OWNED_VALIDATORS = {
    "owned-output-full-model": ("Owned-Output Full-Model Validation", "run_owned_output_validation.py"),
    "owned-output-handoff": ("Owned-Output Exact-Keeper Handoff Validation", "run_owned_output_handoff.py"),
    "owned-output-trace": ("Owned-Output Full-Trace Validation", "run_owned_output_trace.py"),
}
ATTRIBUTION_VALIDATION_CONTROLLER_CPU_TESTS = (
    "test_fused_attribution_validation_controller.py",
    "test_fused_attribution_validation_owner.py",
    "test_selected_lr_failure_handoff.py",
)
ATTRIBUTION_VALIDATION_TIMEOUT = 27 * 900 + 8100


def read_record(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def local_time(value: str | None) -> str:
    if value is None:
        return "not recorded"
    return datetime.fromisoformat(value).astimezone(LOCAL_ZONE).strftime("%Y-%m-%d %H:%M:%S")


def number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.6f}"


def comparison_lines(record: dict) -> list[str]:
    lines = []
    for analysis in record["analyses"]:
        baseline = analysis["baseline"]
        for name, comparison in analysis["comparisons"].items():
            if name == baseline:
                continue
            eligible = comparison["final_comparison_eligible"]
            evaluations = comparison["evaluations"]
            c4, selection = evaluations["c4_loss"], evaluations["selection_loss"]
            windows = comparison["training_windows"]
            deltas = ", ".join(number(window["delta"]) for window in windows) or "none complete"
            lines.extend(
                [
                    f"Comparison `{name}` minus `{baseline}`: "
                    f"{'full-budget eligible' if eligible else 'interim; no final outcome'}.",
                    f"Complete {analysis['window_tokens']:,}-token training-window deltas: {deltas}.",
                    f"Exact-shared late C4 delta {number(c4['mean_late_delta'])} "
                    f"({len(c4['late_tokens'])} evaluations); "
                    f"selection delta {number(selection['mean_late_delta'])} "
                    f"({len(selection['late_tokens'])} evaluations).",
                ]
            )
            if eligible:
                lines.append(
                    f"Final C4 delta {number(c4['final_delta'])}; "
                    f"final selection delta {number(selection['final_delta'])}."
                )
            runtime = comparison["runtime"]
            arm = runtime["arm"]["steady_step_seconds_per_million_tokens"]
            reference = runtime["baseline"]["steady_step_seconds_per_million_tokens"]
            if arm is not None and reference is not None:
                lines.append(f"Steady seconds/MTok: candidate {arm:.3f}, reference {reference:.3f}.")
            final_duration = runtime["delta"]["final_total_seconds"]
            if final_duration is not None:
                lines.append(f"Full-budget recorded total-time delta: {final_duration:+.1f} seconds.")
            lines.append("")
    for name in record.get("unpaired_runs", {}):
        lines.extend([f"`{name}` remains unpaired in this analysis; no optimizer improvement claim.", ""])
    return lines


def validation_progress_lines(campaign_path: Path, frozen: dict, state: dict, receipt: dict) -> list[str]:
    """Show provisional saved progress without treating it as final launch approval."""
    plan = frozen.get("plan")
    if state.get("status") not in ("validating", "needs_attention") or not plan:
        return ["Actual-config GPU receipt is not fully hash-bound; no passed trial count published."]
    helpers, archive = Path(frozen["helpers"]), Path(frozen["archive"])
    matches = (
        frozen["campaign"] == str(campaign_path)
        and frozen["campaign_sha256"] == hashlib.sha256(campaign_path.read_bytes()).hexdigest()
        and archive.is_file()
        and frozen["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
        and frozen["helpers_sha256"]
        and all(
            (helpers / name).is_file() and hashlib.sha256((helpers / name).read_bytes()).hexdigest() == digest
            for name, digest in frozen["helpers_sha256"].items()
        )
        and state.get("binding_sha256")
        == hashlib.sha256((campaign_path.parent / "validation-binding.json").read_bytes()).hexdigest()
        and receipt.get("owner")
        == {
            "controller_pid": state.get("controller_pid"),
            "controller_start_ticks": state.get("controller_start_ticks"),
        }
        and all(
            receipt.get(key) == value for key, value in plan.items() if key not in ("status", "trials", "gpu_commands")
        )
    )
    trials, planned = receipt.get("trials", []), plan["trials"]
    matches = (
        matches
        and len(trials) == len(planned)
        and all(
            all(
                actual.get(key) == expected[key]
                for key in ("name", "family", "config", "config_sha256", "commands", "continuous", "resumed")
            )
            and set(actual.get("checks", {})) == set(expected["checks"])
            and all(type(value) is bool for value in actual["checks"].values())
            for actual, expected in zip(trials, planned, strict=False)
        )
    )
    commands = receipt.get("gpu_commands", [])
    expected_commands = [(trial["name"], command) for trial in planned for command in trial["commands"]]
    matches = (
        matches
        and len(commands) <= len(expected_commands)
        and all(
            actual.get("trial") == name
            and actual.get("command") == command
            and type(actual.get("exit_code")) is int
            and Path(actual["log"]).is_file()
            and hashlib.sha256(Path(actual["log"]).read_bytes()).hexdigest() == actual.get("log_sha256")
            for actual, (name, command) in zip(commands, expected_commands, strict=False)
        )
    )
    if not matches:
        return ["Saved provisional validation identity differs; command and trial progress summaries withheld."]
    complete = {
        trial["name"]
        for trial in trials
        if all(trial["checks"].values())
        and sum(command["trial"] == trial["name"] and command["exit_code"] == 0 for command in commands)
        == len(trial["commands"])
    }
    passed = len(complete)
    successful = sum(command["exit_code"] == 0 for command in commands)
    lines = [
        f"Provisional saved GPU progress at {local_time(receipt.get('time'))}: "
        f"{successful} / {len(expected_commands)} commands exited zero; "
        f"{passed} / {len(planned)} trial check sets report complete.",
        "Frozen plan, helper/archive hashes, command-log prefix and recorded controller generation match.",
        "This is not a fresh liveness check or a final hash-bound validation receipt; "
        "checkpoint reconstruction, cleanup and keeper restoration remain required before launch.",
        "",
        "| Actual LR Trial | Saved Check Set |",
        "| --- | --- |",
    ]
    if state["status"] == "needs_attention":
        lines.insert(0, "The saved validator failed or requires attention; no further execution is authorized.")
    for trial in trials:
        status = "complete; provisional" if trial["name"] in complete else "pending or incomplete"
        lines.append(f"| `{trial['name']}` | {status} |")
    numerical = []
    for trial in trials:
        if trial["name"] not in complete:
            continue
        for key, label in (
            ("boundary_continuation", "Update-22 boundary"),
            ("post_cached_root_continuation", "Update-24 post-resume"),
        ):
            summary = trial.get(key)
            if not trial["checks"].get(key) or not isinstance(summary, dict):
                continue
            count, error = summary.get("tensor_comparisons"), summary.get("max_absolute_error")
            if type(count) is int and count > 0 and type(error) in (int, float) and math.isfinite(error) and error >= 0:
                numerical.append(f"| `{trial['name']}` | {label} | {count:,} | {error:.6g} |")
    if numerical:
        lines.extend(
            [
                "",
                "Saved numerical summaries for provisional complete check sets:",
                "",
                "| Actual LR Trial | Saved Comparison | Tensor Comparisons | Maximum Absolute Error |",
                "| --- | --- | ---: | ---: |",
                *numerical,
                "",
                "These recorded summaries are not an independent checkpoint reconstruction or final launch approval.",
            ]
        )
    return lines


def tuning_trial_identities_match(campaign: dict, analysis: dict) -> bool:
    planned = {trial["name"]: trial for trial in campaign["trials"]}
    trials = analysis["trials"]
    identity_fields = ("family", "lr_multiplier", "matrix_lr")
    return (
        len(planned) == len(campaign["trials"])
        and len(trials) == len(planned)
        and {trial["name"] for trial in trials} == set(planned)
        and all(all(trial.get(key) == planned[trial["name"]].get(key) for key in identity_fields) for trial in trials)
    )


def tuning_result_lines(campaign: dict, analysis: dict) -> list[str]:
    """Compare saved short-study results only at matching LRs and complete token windows."""
    if not campaign.get("selection") or not campaign.get("trials"):
        return []
    if not tuning_trial_identities_match(campaign, analysis):
        return ["", "Saved tuning trial identities differ from the campaign plan; loss/runtime summaries withheld."]
    planned = {trial["name"]: trial for trial in campaign["trials"]}
    trials = analysis["trials"]
    references = {(trial["lr_multiplier"], trial["matrix_lr"]): trial for trial in trials if trial["family"] == "muont"}
    window = campaign["selection"]["training_window_tokens"]
    recorded_coverage = all("late_evaluation_tokens" in trial for trial in trials)
    coverage_lines = (
        ["Late-mean deltas require identical nonempty recorded evaluation-token lists for each metric."]
        if recorded_coverage
        else [
            "The saved analysis does not record late evaluation tokens/counts; their matching coverage is unverified.",
            "These aggregate late-mean deltas are not verified paired held-out estimates.",
        ]
    )
    lines = [
        "",
        "Saved short-study results; these are not a fresh history read or a full-budget optimizer ranking.",
        "Deltas compare the same planned matrix LR with MuonT. Late-mean deltas require both trials complete.",
        *coverage_lines,
        "Training deltas use only complete shared token windows, never partial windows or interpolation.",
        "",
        "| Trial | Saved State | Final Selection | Late Selection Mean Delta | Late C4 Mean Delta | "
        "Final Total Seconds | Extra Final Time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trial in trials:
        reference = references.get((trial["lr_multiplier"], trial["matrix_lr"]))
        complete = trial["complete"] and trial["status"] == "complete"
        paired = complete and reference is not None and reference["complete"] and reference["status"] == "complete"
        final = trial["final_selection_loss"] if complete else None
        late = {}
        for metric, coverage_key in (
            ("mean_late_selection_loss", "selection_loss"),
            ("mean_late_c4_loss_reporting_only", "c4_loss"),
        ):
            value, baseline = trial.get(metric), reference.get(metric) if reference else None
            coverage_matches = True
            if recorded_coverage and paired:
                tokens = trial["late_evaluation_tokens"].get(coverage_key)
                coverage_matches = bool(tokens) and tokens == reference["late_evaluation_tokens"].get(coverage_key)
            late[metric] = (
                value - baseline if paired and coverage_matches and value is not None and baseline is not None else None
            )
        runtime, baseline_runtime = trial.get("runtime") or {}, (reference.get("runtime") or {}) if reference else {}
        total = runtime.get("total_seconds") if complete and runtime.get("total_is_final") is True else None
        baseline_total = (
            baseline_runtime.get("total_seconds") if baseline_runtime.get("total_is_final") is True else None
        )
        extra = (
            100 * (total / baseline_total - 1)
            if paired and total is not None and baseline_total is not None and baseline_total > 0
            else None
        )
        lines.append(
            f"| `{trial['name']}` | `{trial['status']}` | "
            f"{'n/a' if final is None else f'{final:.6f}'} | "
            f"{number(late['mean_late_selection_loss'])} | "
            f"{number(late['mean_late_c4_loss_reporting_only'])} | "
            f"{'n/a' if total is None else f'{total:.2f}'} | "
            f"{'n/a' if extra is None else f'{extra:+.2f}%'} |"
        )
    lines.extend(
        [
            "",
            f"| Candidate Trial | Complete Shared {window:,}-Token Training-Window Deltas |",
            "| --- | --- |",
        ]
    )
    for trial in trials:
        if trial["family"] == "muont":
            continue
        reference = references.get((trial["lr_multiplier"], trial["matrix_lr"]))
        baseline_windows = (
            {item["end_tokens"]: item["mean"] for item in reference["training_windows"]} if reference else {}
        )
        deltas = []
        for item in trial["training_windows"]:
            endpoint = item["end_tokens"]
            if (
                endpoint in baseline_windows
                and endpoint % window == 0
                and endpoint
                <= min(trial["observed_tokens"], reference["observed_tokens"], planned[trial["name"]]["tokens"])
            ):
                deltas.append(f"W{endpoint // window}: {number(item['mean'] - baseline_windows[endpoint])}")
        lines.append(f"| `{trial['name']}` | {', '.join(deltas) or 'none shared; no value inferred'} |")
    lines.extend(
        [
            "",
            "C4 is reporting only. LR selection still requires all "
            f"{campaign.get('initial_trials_per_family', 3)} completed trials per family.",
            "Short-schedule loss and final timers do not establish 5,120-update ranking "
            "or independently tuned superiority.",
        ]
    )
    if campaign.get("study") == "wave7_fixed_age_matrix_lr_upper_boundary":
        lines.append(
            "New-grid choices are not cumulative choices; all five parent-plus-new endpoints "
            "must be complete before selecting across the expanded grid."
        )
    return lines


def tuning_execution_lines(root: Path, campaign_path: Path, campaign: dict) -> list[str]:
    """Report saved execution evidence only under its unchanged campaign and helper binding."""
    directory = campaign_path.parent
    binding_path = directory / "execution-binding.json"
    if not binding_path.exists():
        return ["No separately frozen tuning execution binding or launch is recorded."]
    frozen = read_record(binding_path)
    lines = [
        f"Separate execution binding: `{frozen['status']}` at {local_time(frozen.get('time'))}; "
        f"`PVC/{binding_path.relative_to(root)}`.",
        "Execution freezing alone creates no GPU job, worker, monitor or queue.",
    ]
    archive = Path(frozen["archive"])
    helpers = Path(frozen["helpers"])
    validation_path = directory / "validation-binding.json"
    matches = (
        frozen["campaign"] == str(campaign_path)
        and frozen["campaign_sha256"] == hashlib.sha256(campaign_path.read_bytes()).hexdigest()
        and validation_path.is_file()
        and frozen["validation_binding_sha256"] == hashlib.sha256(validation_path.read_bytes()).hexdigest()
        and archive.is_file()
        and frozen["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
        and frozen["helpers_sha256"]
        and all(
            (helpers / name).is_file() and hashlib.sha256((helpers / name).read_bytes()).hexdigest() == digest
            for name, digest in frozen["helpers_sha256"].items()
        )
    )
    if not matches:
        return [*lines, "Execution binding identity differs or is incomplete; worker and cloud summaries withheld."]
    digest = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    launch_path = directory / "launch-state.json"
    launch = read_record(launch_path)
    if launch:
        if launch.get("binding_sha256") != digest:
            return [*lines, "Saved tuning launch differs from the execution binding; execution summaries withheld."]
        lines.append(
            f"Saved tuning launch: `{launch['status']}` at {local_time(launch.get('time'))}; "
            f"`PVC/{launch_path.relative_to(root)}`."
        )
        if "monitor_pid" in launch:
            lines.append(f"Recorded tuning monitor PID: `{launch['monitor_pid']}`; not a new liveness check.")
    else:
        lines.append("No tuning worker launch is recorded.")
    snapshot_path = directory / "monitor/latest.json"
    snapshot = read_record(snapshot_path)
    if snapshot:
        if snapshot.get("binding_sha256") != digest or snapshot.get("campaign_sha256") != frozen["campaign_sha256"]:
            lines.append("Saved tuning monitor identity differs; process and token summaries withheld.")
        else:
            lines.extend(
                [
                    f"Saved tuning monitor: `{snapshot['status']}` at {local_time(snapshot.get('time'))}; "
                    f"`PVC/{snapshot_path.relative_to(root)}`.",
                    "",
                    "| Family | Saved Worker State | Keeper Restoration |",
                    "| --- | --- | --- |",
                ]
            )
            for worker in snapshot["workers"]:
                receipt = worker.get("receipt", {})
                restoration = receipt.get("keeper_restore", "not recorded")
                lines.append(f"| `{worker['family']}` | `{worker['status']}` | `{restoration}` |")
            lines.extend(["", "| Trial | Saved Manifest State | Observed Tokens |", "| --- | --- | ---: |"])
            for trial in snapshot["trials"]:
                tokens = trial.get("observed_tokens")
                charged = "unknown; not zero" if tokens is None else f"{tokens:,}"
                lines.append(f"| `{trial['name']}` | `{trial['status']}` | {charged} |")
            lines.append("Actual failed-run prefixes remain charged; unknown SSH or scalar reads are not completion.")
    cloud_path = directory / "swanlab-upload.json"
    cloud = read_record(cloud_path)
    if not cloud:
        return [*lines, "No verified tuning cloud receipt is recorded."]
    expected = {trial["name"] for trial in campaign["trials"]}
    if (
        cloud.get("campaign_sha256") != frozen["campaign_sha256"]
        or cloud.get("execution_binding_sha256") != digest
        or cloud.get("visibility") != "PRIVATE"
        or not cloud.get("project", "").endswith("/output-side-normalization-marin")
        or not set(cloud.get("runs", {})) <= expected
    ):
        return [*lines, "Saved tuning cloud identity differs; verified-history counts withheld."]
    verified = sum(
        entry.get("status") == "verified"
        and entry.get("campaign_sha256") == frozen["campaign_sha256"]
        and entry.get("execution_binding_sha256") == digest
        for entry in cloud["runs"].values()
    )
    lines.append(
        f"Saved tuning cloud verification: {verified} / {len(expected)} trial histories at "
        f"{local_time(cloud.get('verified_at'))}; `PVC/{cloud_path.relative_to(root)}`."
    )
    if cloud.get("status") == "verified" and verified == len(expected):
        lines.append("All tuning histories have a saved verified receipt; this is not an independently tuned win.")
    else:
        lines.append("Tuning cloud verification remains partial or incomplete; no complete-study import claimed.")
    return lines


def boundary_queue_lines(root: Path, campaign_path: Path) -> list[str]:
    """Publish saved queue state only when its frozen release identities still match."""
    directory = campaign_path.parent / "queue"
    binding_path = directory / "binding.json"
    if not binding_path.exists():
        return ["No separately frozen upper-boundary handoff queue is recorded."]
    frozen = read_record(binding_path)
    lines = [
        f"Separate boundary handoff queue: `{frozen['status']}` at {local_time(frozen.get('time'))}; "
        f"`PVC/{binding_path.relative_to(root)}`."
    ]
    validation_path = campaign_path.parent / "validation-binding.json"
    execution_path = campaign_path.parent / "execution-binding.json"
    predecessor_wave = (
        "wave9" if read_record(campaign_path).get("study") == "wave7_fixed_age_matrix_lr_ceiling" else "wave8"
    )
    predecessor_path = root / f"run-preparation/{predecessor_wave}-launch-binding.json"
    execution = read_record(execution_path)
    helpers, archive = Path(frozen["helpers"]), Path(frozen["archive"])
    matches = (
        frozen["campaign"] == str(campaign_path)
        and frozen["campaign_sha256"] == hashlib.sha256(campaign_path.read_bytes()).hexdigest()
        and validation_path.is_file()
        and frozen["validation_binding_sha256"] == hashlib.sha256(validation_path.read_bytes()).hexdigest()
        and execution_path.is_file()
        and frozen["execution_binding_sha256"] == hashlib.sha256(execution_path.read_bytes()).hexdigest()
        and predecessor_path.is_file()
        and frozen["predecessor_release_sha256"] == hashlib.sha256(predecessor_path.read_bytes()).hexdigest()
        and archive.is_file()
        and frozen["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
        and set(frozen["helpers_sha256"])
        == set(execution.get("helpers_sha256", {})) | {"queue_lr_boundary.py", "queue_lr_boundary.sh"}
        and all(frozen["helpers_sha256"].get(name) == digest for name, digest in execution["helpers_sha256"].items())
        and all(
            (helpers / name).is_file() and hashlib.sha256((helpers / name).read_bytes()).hexdigest() == digest
            for name, digest in frozen["helpers_sha256"].items()
        )
    )
    digest = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    state = read_record(directory / "state.json")
    installation = read_record(directory / "installation.json")
    if not matches or any(record and record.get("binding_sha256") != digest for record in (state, installation)):
        return [*lines, "Queue binding identity differs or is incomplete; queue/controller summaries withheld."]
    lines.extend(
        [
            f"Bound Bash observation interval: {frozen['interval_seconds']} seconds.",
            f"GPU execution waits for completed five-window {predecessor_wave} analysis, verified cloud histories "
            "and fresh terminal process/restoration checks, then actual new-LR validation and cleanup.",
            "Persisted controller intents are one-shot; unknown observations do not authorize allocation changes.",
        ]
    )
    if installation:
        lines.append(
            f"Saved queue installation: `{installation['status']}` at {local_time(installation.get('time'))}; "
            f"recorded PID `{installation.get('pid', 'unacknowledged')}`. This is not a fresh liveness check."
        )
    else:
        lines.append("No acknowledged queue installation is recorded.")
    if state:
        lines.append(
            f"Saved boundary handoff state: `{state['status']}` at {local_time(state.get('time'))}; "
            f"`PVC/{(directory / 'state.json').relative_to(root)}`."
        )
        if state.get("error") or state.get("observation_error"):
            lines.append("The saved queue reports an error or unknown observation; inspect its evidence without retry.")
    for phase in ("validate", "launch"):
        intent = read_record(directory / f"{phase}.json")
        if intent:
            lines.append(
                f"Saved `{phase}` controller intent: `{intent['status']}`; "
                f"PID `{intent.get('pid', 'unacknowledged')}`, start ticks "
                f"`{intent.get('start_ticks', 'unacknowledged')}`. Intent is not a passed GPU or launch result."
            )
    stopped = directory / "stopped_at"
    if stopped.exists():
        lines.append(f"Queue exit recorded at {local_time(stopped.read_text().strip())}; no live queue inferred.")
    return lines


def combined_probe_lines(root: Path, binding_path: Path) -> list[str]:
    """Keep separately bound GPU diagnostics visible without promoting them to training gates."""
    frozen = read_record(binding_path)
    directory = binding_path.parent
    lines = [
        f"### Combined-Product GPU Probe: {directory.name}",
        "",
        f"Binding state: `{frozen['status']}` at {local_time(frozen.get('time'))};",
        f"`PVC/{binding_path.relative_to(root)}`.",
        "Synthetic update parity and timing only; never full-model validation or an optimizer efficiency win.",
    ]
    release = root / "run-preparation/wave7-release-binding.json"
    guard = root / "campaigns/lr-fixed-age-0p5m-20261116/validation-binding.json"
    archive = directory / "source.tar"
    matches = (
        frozen["root"] == str(root.resolve())
        and frozen["out"] == str(directory.resolve())
        and release.is_file()
        and frozen["source_release_sha256"] == hashlib.sha256(release.read_bytes()).hexdigest()
        and guard.is_file()
        and frozen["original_guard_binding_sha256"] == hashlib.sha256(guard.read_bytes()).hexdigest()
        and archive.is_file()
        and frozen["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
        and frozen["helpers_sha256"]
        and all(
            (directory / "helpers" / name).is_file()
            and hashlib.sha256((directory / "helpers" / name).read_bytes()).hexdigest() == digest
            for name, digest in frozen["helpers_sha256"].items()
        )
    )
    state = read_record(directory / "state.json")
    if not state:
        return [*lines, "No GPU execution receipt is recorded.", ""]
    matches = matches and state.get("binding_sha256") == hashlib.sha256(binding_path.read_bytes()).hexdigest()
    if not matches:
        return [*lines, "Probe binding identity differs; execution and numerical summaries withheld.", ""]
    lines.append(f"Saved execution: `{state['status']}` at {local_time(state.get('time'))}.")
    if "keeper_restore" in state:
        lines.append(f"Recorded keeper restoration: `{state['keeper_restore']}`.")
    if "error" in state:
        lines.append(f"Recorded failure: `{state['error']}`.")
    result_path = directory / "result.json"
    result = read_record(result_path)
    if not result:
        return [*lines, "No numerical result receipt is recorded.", ""]
    owner = {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
    if (
        state.get("result_sha256") != hashlib.sha256(result_path.read_bytes()).hexdigest()
        or result.get("owner") != owner
        or result.get("helpers_sha256") != frozen["helpers_sha256"]
    ):
        return [*lines, "Numerical result is not controller hash-bound; measured summaries withheld.", ""]
    cases = result["cases"]
    passed = sum(all(case["exact"].values()) and case["parameter_delta"]["within_tolerance"] for case in cases)
    lines.extend(
        [
            f"Saved result: `{result['status']}`; {passed} / {len(cases)} recorded cases pass.",
            f"`PVC/{result_path.relative_to(root)}`.",
            "A failed or partial fixture set is not a passed production diagnostic.",
        ]
    )
    for measurement in result["timing"]:
        mode = "CUDA graph" if measurement["graphs"] else "eager"
        lines.append(
            f"Shape `{measurement['shape']}`, {mode}: combined/reference GPU-event median ratio "
            f"{measurement['ratio']:.4f}; synthetic single-device update only."
        )
    return [*lines, ""]


def upstream_cpu_lines(root: Path) -> list[str]:
    """Report separately recorded CPU evidence only while its source and log identities match."""
    lines = []
    for path in sorted((root / "run-preparation/upstream-outer-cpu").glob("*/receipt.json")):
        record = read_record(path)
        candidate = Path(record["candidate"])
        initial = Path(record["initial_preparation"])
        helpers = Path(record["helpers"])
        identities = [
            (initial, record["initial_preparation_sha256"]),
            (root / "run-preparation/wave8-launch-binding.json", record["predecessor_release_sha256"]),
            (root / "run-preparation/wave8-candidate.json", record["predecessor_candidate_sha256"]),
            (Path(record["archive"]), record["archive_sha256"]),
            *((helpers / name, digest) for name, digest in record["helpers_sha256"].items()),
            *((Path(command["log"]), command["log_sha256"]) for command in record["commands"]),
        ]
        bound = (
            record.get("status") == "cpu_verified_gpu_execution_denied"
            and record.get("cuda_visible_devices") == ""
            and all(
                source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == digest
                for source, digest in identities
            )
            and len(record["commands"]) == 2
            and all(type(command["exit_code"]) is int and command["exit_code"] == 0 for command in record["commands"])
        )
        try:
            bound = bound and candidate_inventory(candidate) == record["files_sha256"]
        except ValueError:
            bound = False
        lines.extend(
            [
                "### Upstream-Product / Null CPU Follow-Up",
                "",
                f"Saved receipt at {local_time(record.get('time'))}; `PVC/{path.relative_to(root)}`.",
            ]
        )
        if not bound:
            lines.extend(["Saved CPU source or log identity differs; verification summaries withheld.", ""])
            continue
        lines.extend(
            [
                f"Saved status: `{record['status']}`; {len(record['files_sha256'])} candidate files bound.",
                "Candidate and equation CPU commands exited zero with CUDA devices hidden.",
                "Arms: " + ", ".join(f"`{arm}`" for arm in record["arms"]) + ".",
                "Direct-product CPU references do not emulate pinned CUDA split kernels.",
                "This CPU receipt establishes no CUDA parity, all-rank continuation, measured GPU timing "
                "or training authorization; separate GPU diagnostic evidence is reported below.",
                "",
            ]
        )
    return lines


def upstream_full_model_cpu_lines(root: Path) -> list[str]:
    """Report separately prepared actual-path observers without implying a GPU release."""
    lines = []
    for path in sorted((root / "run-preparation/upstream-full-model-cpu").glob("*/receipt.json")):
        record = read_record(path)
        candidate = Path(record["candidate"])
        prototype = Path(record["prototype_candidate"])
        prototype_receipt = Path(record["prototype_cpu_receipt"])
        initial = Path(record["initial_preparation"])
        helpers = Path(record["helpers"])
        identities = [
            (prototype_receipt, record["prototype_cpu_receipt_sha256"]),
            (initial, record["initial_preparation_sha256"]),
            (Path(record["archive"]), record["archive_sha256"]),
            *((helpers / name, digest) for name, digest in record["helpers_sha256"].items()),
            *((Path(command["log"]), command["log_sha256"]) for command in record["commands"]),
        ]
        matches = (
            record["status"] == "full_model_observer_cpu_verified_gpu_execution_denied"
            and record["cuda_visible_devices"] == ""
            and len(record["commands"]) == 2
            and [(command.get("cwd"), command.get("command")) for command in record["commands"]]
            == [(str(cwd), command) for _, cwd, command in cpu_test_commands(candidate, helpers)]
            and record["helpers_sha256"]
            and all(type(command["exit_code"]) is int and command["exit_code"] == 0 for command in record["commands"])
            and all(
                source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == digest
                for source, digest in identities
            )
        )
        try:
            matches = (
                matches
                and candidate_inventory(candidate) == record["files_sha256"]
                and candidate_inventory(prototype) == record["prototype_files_sha256"]
            )
        except (ValueError, FileNotFoundError):
            matches = False
        lines.extend(
            [
                "### Upstream-Product Full-Model Observer Preparation",
                "",
                f"Saved receipt at {local_time(record.get('time'))}; `PVC/{path.relative_to(root)}`.",
            ]
        )
        if not matches:
            lines.extend(["Saved observer CPU source or log identity differs; verification summaries withheld.", ""])
            continue
        lines.extend(
            [
                f"Saved status: `{record['status']}`; {len(record['files_sha256'])} candidate files bound.",
                "Arms: " + ", ".join(f"`{arm}`" for arm in record["arms"]) + ".",
                "Candidate and actual-step observer CPU commands exited zero with CUDA devices hidden.",
                "Prepared changing-scalar dispatch, exact pinned null/input and independent FP64 output checks.",
                "No actual GPU scalar transition, full-model CUDA continuation, release, queue or training result.",
                "",
            ]
        )
    return lines


def owned_output_validation_cpu_lines(root: Path) -> list[str]:
    """Report distinct owned-validation CPU outcomes without promoting them to GPU gates."""
    lines = []
    for path in sorted((root / "run-preparation/owned-output-validation-cpu").glob("*/receipt.json")):
        record = read_record(path)
        candidate = Path(record["candidate"])
        initial_path = candidate / "owned-output-validation-preparation.json"
        initial = read_record(initial_path)
        code = record.get("exit_code")
        identities = [
            (initial_path, record.get("preparation_sha256")),
            (Path(record["predecessor_cpu_receipt"]), record.get("predecessor_cpu_receipt_sha256")),
            (Path(record["archive"]), record.get("archive_sha256")),
            (Path(record["log"]), record.get("log_sha256")),
        ]
        expected_status = (
            "owned_output_validation_cpu_verified_gpu_denied"
            if code == 0 and "error" not in record
            else "owned_output_validation_cpu_failed"
        )
        files = initial.get("files_sha256", {})
        previous_files = initial.get("predecessor_files_sha256", {})
        changed = {name for name in files.keys() | previous_files.keys() if files.get(name) != previous_files.get(name)}
        bound = (
            type(code) is int
            and record.get("status") == expected_status
            and initial.get("status") == "owned_output_validation_prepared_gpu_denied"
            and initial.get("candidate") == str(candidate.resolve())
            and record.get("preparation") == str(initial_path)
            and record.get("cuda_visible_devices") == ""
            and record.get("command") == owned_output_cpu_command()
            and record.get("cwd") == str(candidate.resolve())
            and bool(files)
            and set(initial.get("changed_files", [])) == changed
            and all(record.get(key) == value for key, value in initial.items() if key not in ("status", "time"))
            and all(
                source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == digest
                for source, digest in identities
            )
        )
        try:
            bound = bound and candidate_inventory(candidate) == files
        except (ValueError, FileNotFoundError):
            bound = False
        lines.extend(
            [
                "### Owned-Output Full-Model Validation CPU Successor",
                "",
                f"Saved receipt at {local_time(record.get('time'))}; `PVC/{path.relative_to(root)}`.",
            ]
        )
        if not bound:
            lines.extend(["Saved owned-validation source or CPU outcome differs; summaries withheld.", ""])
            continue
        lines.extend(
            [
                f"Saved status: `{record['status']}`; integer exit `{code}`; "
                f"{len(files)} candidate files and {len(changed)} changed files bound.",
                "CPU execution hides CUDA devices and retains its original log/archive, including failed outcomes.",
                "Numerical block, kernel, training and config preservation belongs to the preparation receipt.",
                "No GPU gate, training-quality result, whole-training saving or production authorization "
                "is established by this CPU receipt.",
                "",
            ]
        )
        if "error" in record:
            lines.extend([f"Recorded CPU evidence failure: `{record['error']}`.", ""])
    return lines


def owned_output_handoff_cpu_lines(root: Path, identity: str = "owned-output-handoff") -> list[str]:
    """Keep ownership-only CPU successors and their issued predecessors separately visible."""
    if identity == "owned-output-handoff":
        original_identity, preparation_name = "owned-output-full-model", HANDOFF_PREPARATION
        changes, require_archive = HANDOFF_CHANGES, require_handoff_archive
        label = "Exact-Keeper Handoff"
        scope = "Original validator preserved; numerical training, kernel, config and validation bytes unchanged."
    elif identity == "owned-output-trace":
        original_identity, preparation_name = "owned-output-handoff", TRACE_PREPARATION
        changes, require_archive = TRACE_CHANGES, require_trace_archive
        label = "Full-Trace"
        scope = "Failed validator preserved; only owned trace reduction, diagnostic/tests and ownership differ."
    else:
        raise ValueError("Unknown owned CPU successor")
    status_prefix = identity.replace("-", "_")
    lines = []
    original_path = root / f"run-preparation/{original_identity}-binding.json"
    original = read_record(original_path)
    for path in sorted((root / f"run-preparation/{identity}-cpu").glob("*/receipt.json")):
        record = read_record(path)
        candidate = Path(record["candidate"])
        preparation = candidate / preparation_name
        initial = read_record(preparation)
        files = initial.get("files_sha256", {})
        previous = original.get("files_sha256", {})
        changed = {name for name in files.keys() | previous.keys() if files.get(name) != previous.get(name)}
        code = record.get("exit_code")
        expected_status = (
            f"{status_prefix}_cpu_verified_gpu_denied"
            if code == 0 and "error" not in record
            else f"{status_prefix}_cpu_failed"
        )
        identities = [
            (original_path, initial.get("original_binding_sha256")),
            (preparation, record.get("preparation_sha256")),
            (Path(record["archive"]), record.get("archive_sha256")),
            (Path(record["log"]), record.get("log_sha256")),
        ]
        bound = (
            type(code) is int
            and record.get("status") == expected_status
            and initial.get("status") == f"{status_prefix}_prepared_gpu_denied"
            and initial.get("root") == str(root)
            and initial.get("candidate") == str(candidate.resolve())
            and initial.get("original_candidate") == original.get("candidate")
            and set(initial.get("changed_files", [])) == changed == changes
            and record.get("preparation") == str(preparation)
            and record.get("command") == owned_output_cpu_command()
            and record.get("cwd") == str(candidate.resolve())
            and record.get("cuda_visible_devices") == ""
            and bool(files)
            and all(record.get(key) == value for key, value in initial.items() if key not in ("status", "time"))
            and all(
                source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == digest
                for source, digest in identities
            )
        )
        try:
            bound = (
                bound
                and candidate_inventory(candidate) == files
                and candidate_inventory(Path(original["candidate"])) == previous
            )
            if bound:
                require_archive(Path(record["archive"]), initial, record["preparation_sha256"])
        except (ValueError, FileNotFoundError):
            bound = False
        lines.extend(
            [
                f"### Owned-Output {label} CPU Successor",
                "",
                f"Saved receipt at {local_time(record.get('time'))}; `PVC/{path.relative_to(root)}`.",
            ]
        )
        if not bound:
            lines.extend(["Saved handoff source, predecessor or CPU outcome differs; summaries withheld.", ""])
            continue
        lines.extend(
            [
                f"Saved status: `{record['status']}`; integer exit `{code}`; "
                f"{len(files)} candidate files; "
                + (
                    "exactly three ownership changes bound."
                    if identity == "owned-output-handoff"
                    else "six scoped changes bound."
                ),
                scope,
                "CPU outcome retains hidden CUDA devices and original log/archive. No GPU gate or production release.",
                "",
            ]
        )
        if "error" in record:
            lines.extend([f"Recorded handoff CPU failure: `{record['error']}`.", ""])
    return lines


def owned_output_validation_lines(root: Path, identity: str = "owned-output-full-model") -> list[str]:
    """Track a separately bound validation controller using the existing saved-progress contract."""
    title, controller = OWNED_VALIDATORS[identity]
    path = root / f"run-preparation/{identity}-binding.json"
    frozen = read_record(path)
    if not frozen:
        return []
    lines = [f"### {title}", "", f"`PVC/{path.relative_to(root)}`."]
    candidate = Path(frozen["candidate"])
    identities = [
        (Path(frozen["archive"]), frozen["archive_sha256"]),
        (Path(frozen["cpu_receipt"]), frozen["cpu_receipt_sha256"]),
        (root / "data/experiment_receipt.json", frozen["data_receipt_sha256"]),
        *((Path(frozen["helpers"]) / name, digest) for name, digest in frozen["helpers_sha256"].items()),
    ]
    bound = (
        frozen["status"] == "frozen_pending_completed_ceiling_and_owned_gpu_validation"
        and frozen["root"] == str(root)
        and frozen["candidate"] == str(candidate.resolve())
        and len(frozen["commands"]) == 14
        and bool(frozen["helpers_sha256"])
        and all(
            source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == digest
            for source, digest in identities
        )
    )
    try:
        bound = bound and candidate_inventory(candidate) == frozen["files_sha256"]
    except (ValueError, FileNotFoundError):
        bound = False
    if not bound:
        return [*lines, "Saved owned-validation binding identity differs; execution summaries withheld.", ""]
    lines.extend(
        [
            f"Saved binding `{frozen['status']}`; 14 planned commands. Validation-only, not production release.",
            "GPU transfer requires completed, restored, analyzed and cloud-verified original ceiling-LR recovery.",
        ]
    )
    intent_path = Path(frozen["helpers"]).parent / "controller.json"
    intent = read_record(intent_path) if identity != "owned-output-full-model" else {}
    if intent:
        command = [
            "/opt/conda/envs/torch-base/bin/python",
            str(Path(frozen["helpers"]) / controller),
            "--root",
            str(root),
            "--phase",
            "validate",
        ]
        if (
            intent.get("binding_sha256") == hashlib.sha256(path.read_bytes()).hexdigest()
            and intent.get("command") == command
            and intent.get("candidate") == str(candidate)
            and intent.get("cuda_visible_devices") == "0,1,2,3,4,5,6,7"
        ):
            lines.append(
                f"Saved one-shot controller intent `{intent['status']}` at {local_time(intent.get('time'))}; "
                f"PID `{intent.get('pid', 'unacknowledged')}`. No retry or fresh liveness inferred."
            )
        else:
            lines.append("Saved controller intent identity differs; launch summaries withheld.")
    state = read_record(root / f"run-preparation/{identity}-state.json")
    if not state:
        message = (
            "No saved GPU controller state; detached startup remains unverified."
            if intent
            else "No saved owned-validation controller execution; remains unlaunched."
        )
        return [*lines, message, ""]
    if state.get("binding_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        return [*lines, "Saved owned-validation controller binding differs; progress withheld.", ""]
    lines.append(
        f"Saved controller `{state['status']}` at {local_time(state.get('time'))}; "
        f"PID `{state.get('controller_pid')}`. Not fresh liveness or GPU approval."
    )
    for name in ("exit_code", "cleanup_exit_code", "keeper_restore"):
        if name in state:
            lines.append(f"Recorded `{name}`: `{state[name]}`.")
    if state.get("error"):
        lines.append(f"Recorded validation failure: `{state['error']}`. This identity cannot be retried.")
    lines.extend(upstream_full_model_validation_lines(root, path, frozen, state))
    return [*lines, "Independent frozen-controller reconstruction and cleanup acceptance remain required.", ""]


def upstream_gpu_lines(root: Path, binding_path: Path) -> list[str]:
    """Keep the frozen synthetic probe and its owned saved execution distinct from training."""
    frozen = read_record(binding_path)
    directory = binding_path.parent
    lines = [
        f"### Upstream-Product / Null GPU Diagnostic: {directory.name}",
        "",
        f"Saved binding at {local_time(frozen.get('time'))}; `PVC/{binding_path.relative_to(root)}`.",
        "Synthetic single-device diagnostic only; no whole-model validation, training authorization "
        "or optimizer efficiency claim.",
    ]
    cpu_path, campaign_path = Path(frozen["cpu_receipt"]), Path(frozen["campaign"])
    cpu = read_record(cpu_path)
    execution_path = campaign_path.parent / "execution-binding.json"
    execution = read_record(execution_path)
    helpers = directory / "helpers"
    identities = [
        (cpu_path, frozen["cpu_receipt_sha256"]),
        (campaign_path, frozen["campaign_sha256"]),
        (execution_path, frozen["execution_binding_sha256"]),
        (root / "run-preparation/wave8-launch-binding.json", frozen["predecessor_release_sha256"]),
        (Path(frozen["archive"]), frozen["archive_sha256"]),
        *((helpers / name, digest) for name, digest in frozen["helpers_sha256"].items()),
    ]
    matches = (
        frozen["root"] == str(root.resolve())
        and frozen["out"] == str(directory.resolve())
        and frozen["helpers"] == str(helpers)
        and frozen["status"] == "frozen_waiting_upper_lr_completion_not_launch_gate"
        and frozen["candidate_files_sha256"] == cpu.get("files_sha256")
        and frozen["plan"]["cpu_receipt_sha256"] == frozen["cpu_receipt_sha256"]
        and frozen["helpers_sha256"]
        and execution.get("campaign") == str(campaign_path)
        and execution.get("campaign_sha256") == frozen["campaign_sha256"]
        and all(
            frozen["helpers_sha256"].get(name) == digest for name, digest in execution.get("helpers_sha256", {}).items()
        )
        and all(
            path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in identities
        )
    )
    try:
        matches = matches and candidate_inventory(Path(cpu["candidate"])) == frozen["candidate_files_sha256"]
    except (ValueError, KeyError, FileNotFoundError):
        matches = False
    if not matches:
        return [*lines, "Saved probe source or binding identity differs; execution summaries withheld.", ""]
    plan = frozen["plan"]
    lines.extend(
        [
            f"Saved binding state: `{frozen['status']}`; {len(frozen['helpers_sha256'])} helpers bound.",
            f"Planned coverage: {plan['cases']} production-shape cases; "
            f"{2 * len(plan['shapes'])} eager/CUDA-graph workloads.",
            "GPU execution waits for the entire upper-LR study's completed rederived results, "
            "verified cloud histories, fresh terminal/restored workers and exclusive node ownership.",
        ]
    )
    state = read_record(directory / "state.json")
    result_path = directory / "result.json"
    result = read_record(result_path)
    if not state:
        message = (
            "A result exists without an owned controller state; execution summaries withheld."
            if result
            else "No saved GPU execution state or result; this diagnostic remains unlaunched."
        )
        return [*lines, message, ""]
    if state.get("binding_sha256") != hashlib.sha256(binding_path.read_bytes()).hexdigest():
        return [*lines, "Saved controller binding differs; execution summaries withheld.", ""]
    lines.append(f"Saved controller state: `{state['status']}` at {local_time(state.get('time'))}.")
    for key in ("exit_code", "cleanup_exit_code", "keeper_restore", "error"):
        if key in state:
            lines.append(f"Recorded `{key}`: `{state[key]}`.")
    if result:
        owner = {
            "controller_pid": state.get("controller_pid"),
            "controller_start_ticks": state.get("controller_start_ticks"),
        }
        if (
            state.get("result_sha256") == hashlib.sha256(result_path.read_bytes()).hexdigest()
            and result.get("owner") == owner
            and result.get("binding_sha256") == state["binding_sha256"]
            and result.get("helpers_sha256") == frozen["helpers_sha256"]
        ):
            lines.append(
                f"Saved controller-bound result: `{result['status']}`; `PVC/{result_path.relative_to(root)}`. "
                "This ledger does not independently rederive numerical parity or measured timing."
            )
        else:
            lines.append("Saved result ownership or hash differs; numerical summaries withheld.")
    else:
        lines.append("No saved numerical result receipt.")
    return [*lines, ""]


def upstream_queue_lines(root: Path, probe: Path) -> list[str]:
    """Report the separately bound one-shot queue without inferring live or passed GPU execution."""
    directory = probe / "queue"
    path = directory / "binding.json"
    if not path.exists():
        return []
    frozen = read_record(path)
    diagnostic_path = probe / "binding.json"
    diagnostic = read_record(diagnostic_path)
    helpers = directory / "helpers"
    lines = ["### Upstream-Product GPU Handoff Queue", "", f"`PVC/{path.relative_to(root)}`."]
    matches = (
        frozen["root"] == str(root.resolve())
        and frozen["out"] == str(probe.resolve())
        and frozen["diagnostic_binding_sha256"] == hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
        and frozen["cpu_receipt_sha256"] == diagnostic.get("cpu_receipt_sha256")
        and frozen["campaign"] == diagnostic.get("campaign")
        and frozen["campaign_sha256"] == diagnostic.get("campaign_sha256")
        and frozen["execution_binding_sha256"] == diagnostic.get("execution_binding_sha256")
        and frozen["helpers"] == str(helpers)
        and set(frozen["helpers_sha256"])
        == set(diagnostic.get("helpers_sha256", {})) | {"queue_upstream_outer_gpu.py", "queue_upstream_outer_gpu.sh"}
        and all(frozen["helpers_sha256"].get(name) == digest for name, digest in diagnostic["helpers_sha256"].items())
        and Path(frozen["archive"]).is_file()
        and frozen["archive_sha256"] == hashlib.sha256(Path(frozen["archive"]).read_bytes()).hexdigest()
        and all(
            (helpers / name).is_file() and hashlib.sha256((helpers / name).read_bytes()).hexdigest() == digest
            for name, digest in frozen["helpers_sha256"].items()
        )
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    state = read_record(directory / "state.json")
    installation = read_record(directory / "installation.json")
    if not matches or any(record and record.get("binding_sha256") != digest for record in (state, installation)):
        return [*lines, "Saved upstream queue source or binding differs; execution summaries withheld.", ""]
    lines.extend(
        [
            f"Saved queue binding: `{frozen['status']}` at {local_time(frozen.get('time'))}.",
            f"Bound Bash interval {frozen['interval_seconds']} seconds; designated node `{frozen['node']}`.",
            "The entire upper-LR study must complete, rederive, cloud-verify and restore before one diagnostic.",
            "No keeper operations in this queue, no controller retries and no training authorization.",
        ]
    )
    if installation:
        lines.append(
            f"Saved installation: `{installation['status']}` at {local_time(installation.get('time'))}; "
            f"PID `{installation.get('pid', 'unacknowledged')}`. This is not a fresh liveness check."
        )
    else:
        lines.append("No saved queue installation acknowledgment.")
    if state:
        lines.append(f"Saved handoff: `{state['status']}` at {local_time(state.get('time'))}.")
        if state.get("error") or state.get("observation_error"):
            lines.append("Saved failure/unknown observation remains preserved; inspect without restarting controllers.")
    intent = read_record(directory / "controller.json")
    if intent:
        if (
            intent.get("diagnostic_binding_sha256") == frozen["diagnostic_binding_sha256"]
            and intent.get("command") == frozen["command"]
        ):
            lines.append(
                f"Saved one-shot diagnostic intent: `{intent['status']}`, "
                f"PID `{intent.get('pid', 'unacknowledged')}`; never a passed GPU gate."
            )
        else:
            lines.append("Saved controller intent differs from the frozen command; controller summaries withheld.")
    stopped = directory / "stopped_at"
    if stopped.exists():
        lines.append(f"Queue exit recorded at {local_time(stopped.read_text().strip())}.")
    return [*lines, ""]


def upstream_full_model_validation_lines(root: Path, validation_path: Path, frozen: dict, state: dict) -> list[str]:
    """Report only source-bound saved command progress, never independently accepted GPU gates."""
    receipt_path = validation_path.with_name(
        "validation-" + validation_path.name.removesuffix("-binding.json") + "-gpu.json"
    )
    receipt = read_record(receipt_path)
    if not receipt:
        return ["No saved full-model validation receipt; GPU progress remains unreported."]
    candidate = Path(frozen["candidate"])
    commands = receipt.get("evidence", {}).get("gpu_commands", [])
    planned = frozen["commands"]
    gates = frozen["required_gates"]
    checks = receipt.get("checks", {})
    withheld = "Saved full-model receipt source, owner or command-log identity differs; progress counts withheld."
    if (
        not isinstance(commands, list)
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("command"), list)
            or not isinstance(item.get("log"), str)
            or not isinstance(item.get("log_sha256"), str)
            or type(item.get("exit_code")) is not int
            for item in commands
        )
        or not isinstance(checks, dict)
    ):
        return [withheld]
    identities = [
        (root / "data/experiment_receipt.json", frozen["data_receipt_sha256"]),
        (Path(frozen["archive"]), frozen["archive_sha256"]),
        *((Path(frozen["helpers"]) / name, digest) for name, digest in frozen["helpers_sha256"].items()),
        *((Path(item["log"]), item["log_sha256"]) for item in commands),
    ]
    matches = (
        state.get("binding_sha256") == hashlib.sha256(validation_path.read_bytes()).hexdigest()
        and receipt.get("owner")
        == {
            "controller_pid": state.get("controller_pid"),
            "controller_start_ticks": state.get("controller_start_ticks"),
        }
        and receipt.get("candidate_path") == str(candidate.resolve())
        and receipt.get("environment") == frozen["environment"]
        and receipt.get("source_sha256") == frozen["source_sha256"]
        and receipt.get("data_receipt_sha256") == frozen["data_receipt_sha256"]
        and receipt.get("config_sha256")
        == {arm: frozen["files_sha256"][f"experiment/configs/{arm}_0p5m_seed0.json"] for arm in frozen["arms"]}
        and len(commands) <= len(planned)
        and [item.get("command") for item in commands] == planned[: len(commands)]
        and all(type(item.get("exit_code")) is int for item in commands)
        and all(type(value) is bool for value in checks.values())
        and all(
            path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in identities
        )
    )
    try:
        matches = matches and candidate_inventory(candidate) == frozen["files_sha256"]
    except (ValueError, FileNotFoundError):
        matches = False
    if not matches:
        return [withheld]
    successful = sum(item["exit_code"] == 0 for item in commands)
    failed = len(commands) - successful
    passed = sum(checks.get(gate) is True for gate in gates)
    return [
        f"Saved full-model receipt: `PVC/{receipt_path.relative_to(root)}`; "
        f"initialized or finalized at {local_time(receipt.get('time'))}.",
        f"Source-bound saved GPU commands: {successful} / {len(planned)} exited zero; {failed} failed. "
        f"Saved required check flags: {passed} / {len(gates)} true (including CPU checks).",
        "Saved command/check progress is not fresh liveness, independent tensor reconstruction, "
        "completed cleanup/restoration or production-launch approval.",
    ]


def upstream_full_model_queue_lines(root: Path, path: Path) -> list[str]:
    """Report saved handoff ownership while leaving numerical rederivation to its frozen owner."""
    directory = path.parent
    frozen = read_record(path)
    if not frozen:
        return []
    lines = [f"### Upstream Full-Model Handoff: {directory.name}", "", f"`PVC/{path.relative_to(root)}`."]
    validation_path = directory.with_name(directory.name.removesuffix("-queue") + "-binding.json")
    validation = read_record(validation_path)
    probe = Path(frozen["probe"])
    matches = (
        frozen["root"] == str(root)
        and frozen["validation_binding_sha256"] == hashlib.sha256(validation_path.read_bytes()).hexdigest()
        and frozen["probe_queue_binding_sha256"]
        == hashlib.sha256((probe / "queue/binding.json").read_bytes()).hexdigest()
        and frozen["command"]
        == [
            "/opt/conda/envs/torch-base/bin/python",
            str(Path(validation["helpers"]) / "run_upstream_full_model.py"),
            "--root",
            str(root),
            "--phase",
            "validate",
        ]
        and all(
            (Path(frozen["helpers"]) / name).is_file()
            and hashlib.sha256((Path(frozen["helpers"]) / name).read_bytes()).hexdigest() == digest
            for name, digest in frozen["helpers_sha256"].items()
        )
        and Path(frozen["archive"]).is_file()
        and hashlib.sha256(Path(frozen["archive"]).read_bytes()).hexdigest() == frozen["archive_sha256"]
    )
    if not matches:
        return [*lines, "Saved queue release identity differs; handoff summaries withheld.", ""]
    lines.append(
        f"Saved binding `{frozen['status']}`; {len(frozen['helpers_sha256'])} helpers; "
        f"Bash interval {frozen['interval_seconds']} seconds. No production release."
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    installation = read_record(directory / "installation.json")
    if installation:
        if installation.get("binding_sha256") != digest:
            return [*lines, "Saved installation binding differs; progress withheld.", ""]
        lines.append(
            f"Saved installation `{installation['status']}` at {local_time(installation.get('time'))}; "
            f"PID `{installation.get('pid')}`. This is not a fresh liveness check."
        )
    state = read_record(directory / "state.json")
    if state:
        if state.get("binding_sha256") != digest:
            return [*lines, "Saved queue state binding differs; progress withheld.", ""]
        lines.append(f"Saved handoff `{state['status']}` at {local_time(state.get('time'))}.")
    intent = read_record(directory / "controller.json")
    if intent:
        if (
            intent.get("command") != frozen["command"]
            or intent.get("queue_binding_sha256") != digest
            or intent.get("validation_binding_sha256") != frozen["validation_binding_sha256"]
        ):
            return [*lines, "Saved controller intent differs; validation summaries withheld.", ""]
        lines.append(f"Saved one-shot controller `{intent['status']}`; PID `{intent.get('pid')}`.")
        execution = read_record(directory.with_name(directory.name.removesuffix("-queue") + "-state.json"))
        if execution:
            if (
                execution.get("binding_sha256") != frozen["validation_binding_sha256"]
                or execution.get("controller_pid") != intent.get("pid")
                or execution.get("controller_start_ticks") != intent.get("start_ticks")
            ):
                return [*lines, "Saved validator owner differs; execution summaries withheld.", ""]
            lines.append(f"Saved validator `{execution['status']}` at {local_time(execution.get('time'))}.")
            if "keeper_restore" in execution:
                lines.append(f"Recorded keeper restoration `{execution['keeper_restore']}`.")
            lines.extend(upstream_full_model_validation_lines(root, validation_path, validation, execution))
    return [
        *lines,
        "Numerical receipts require independent frozen-controller reconstruction; saved state is insufficient.",
        "",
    ]


def production_release_matches(root: Path, release: dict, wave: str = "wave9") -> bool:
    """Check the release artifacts before publishing source-bound production evidence."""
    if not release:
        return False
    preparation = root / "run-preparation"
    helpers = Path(release["helpers"])
    identity = (
        "wave11-selected-lr-validation"
        if wave in ("wave11", "wave12")
        else "owned-output-trace" if wave == "wave10" else "upstream-full-model-corrected"
    )
    gpu_receipt = (
        "validation-wave11-selected-lr-gpu.json" if wave in ("wave11", "wave12") else f"validation-{identity}-gpu.json"
    )
    identities = [
        (Path(release["archive"]), release["archive_sha256"]),
        (preparation / f"{identity}-binding.json", release["validation_binding_sha256"]),
        (preparation / gpu_receipt, release["validation_sha256"]),
        (preparation / f"{identity}-state.json", release["state_sha256"]),
        (Path(release["preparation"]), release["preparation_sha256"]),
        (Path(release["cpu_receipt"]), release["cpu_receipt_sha256"]),
        *((helpers / name, digest) for name, digest in release["helpers_sha256"].items()),
    ]
    expected_status = (
        SELECTED_LR_PRODUCTION_STATUS if wave in ("wave11", "wave12") else "frozen_validated_launch_release"
    )
    if wave == "wave12":
        if release.get("recovery_of") != str(preparation / "wave11-launch-binding.json") or not release.get(
            "failure_evidence"
        ):
            return False
        identities.extend((root / name, digest) for name, digest in release["failure_evidence"].items())
    return release["status"] == expected_status and all(
        item.is_file() and hashlib.sha256(item.read_bytes()).hexdigest() == digest for item, digest in identities
    )


def production_lines(root: Path, wave: str = "wave9") -> list[str]:
    """Report preserved production preparation separately from release and worker acknowledgments."""
    preparation_root = root / "run-preparation"
    owned = wave == "wave10"
    identity = "owned-output-trace" if owned else "upstream-full-model-corrected"
    cpu_directory = "owned-output-production-cpu" if owned else "upstream-production-cpu"
    owner_test = "test_owned_output_release.py" if owned else "test_upstream_release.py"
    owner_source = "owned_production_owner.py" if owned else "production_owner.py"
    allowed_changes = OWNED_PRODUCTION_CHANGES if owned else CHANGES
    title = "Wave-10" if owned else "Wave-9"
    numerical_path = preparation_root / f"{identity}-binding.json"
    numerical = read_record(numerical_path)
    receipts = sorted(preparation_root.glob(f"{cpu_directory}/*/receipt.json"))
    lines = []
    for path in receipts:
        receipt = read_record(path)
        candidate = Path(receipt["candidate"])
        prepared_path = candidate / "production-preparation.json"
        prepared = read_record(prepared_path)
        lines.extend(
            [
                f"### {title} Ownership-Only Production Preparation",
                "",
                f"Saved CPU receipt: `PVC/{path.relative_to(root)}` at {local_time(receipt.get('time'))}.",
            ]
        )
        files = prepared.get("files_sha256", {})
        identities = [
            (prepared_path, receipt.get("preparation_sha256")),
            (Path(receipt["log"]), receipt.get("log_sha256")),
            (Path(receipt["archive"]), receipt.get("archive_sha256")),
        ]
        changed = {
            name
            for name in files.keys() | numerical.get("files_sha256", {}).keys()
            if files.get(name) != numerical.get("files_sha256", {}).get(name)
        }
        matches = (
            bool(numerical)
            and prepared.get("status") == "prepared_owner_only_no_production_release"
            and prepared.get("candidate") == str(candidate.resolve())
            and prepared.get("validation_binding_sha256") == hashlib.sha256(numerical_path.read_bytes()).hexdigest()
            and set(prepared.get("changed_files", [])) == allowed_changes
            and changed == allowed_changes
            and prepared.get("owner_sha256") == files.get(f"experiment/{owner_source}")
            and prepared.get("test_sha256") == files.get(f"experiment/tests/{owner_test}")
            and receipt.get("status") in ("production_cpu_verified_gpu_release_pending", "production_cpu_failed")
            and receipt.get("files_sha256") == files
            and receipt.get("command") == cpu_command()
            and receipt.get("cwd") == str(candidate.resolve())
            and receipt.get("cuda_visible_devices") == ""
            and type(receipt.get("exit_code")) is int
            and (
                receipt["exit_code"] != 0 or "error" in receipt
                if receipt.get("status") == "production_cpu_failed"
                else receipt["exit_code"] == 0 and "error" not in receipt
            )
            and all(
                item.is_file() and hashlib.sha256(item.read_bytes()).hexdigest() == digest for item, digest in identities
            )
        )
        try:
            matches = matches and candidate_inventory(candidate) == files
            if matches and owned:
                require_owned_production_archive(Path(receipt["archive"]), files, receipt["preparation_sha256"])
        except (ValueError, FileNotFoundError):
            matches = False
        if not matches:
            lines.extend(["Saved production source or CPU evidence differs; preparation summaries withheld.", ""])
            continue
        if receipt["status"] == "production_cpu_failed":
            lines.extend(
                [
                    f"Saved CPU failure: integer exit `{receipt['exit_code']}`; source/log/archive preserved.",
                    "This candidate cannot authorize a production release or launch.",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"CPU command exited zero with CUDA hidden; {len(files)} source files, "
                f"exactly {len(changed)} ownership/test/cluster differences from the corrected validator.",
                "Numerical sources and original full-budget configurations are unchanged.",
                "CPU preparation is not GPU acceptance, production release or a training launch.",
                "",
            ]
        )
    release_path = preparation_root / f"{wave}-launch-binding.json"
    release = read_record(release_path)
    if not release:
        if lines:
            lines.extend([f"No saved {title} production release or launch is recorded.", ""])
        return lines
    lines.extend([f"### {title} Production Release", "", f"`PVC/{release_path.relative_to(root)}`."])
    if not production_release_matches(root, release, wave):
        return [*lines, f"Saved {title} release identity differs; release/launch summaries withheld.", ""]
    lines.extend(
        [
            f"Saved release `{release['status']}` at {local_time(release.get('time'))}; "
            f"{len(release['helpers_sha256'])} helpers, Bash interval {release['monitor_interval_seconds']} seconds.",
            "Release evidence is distinct from original GPU gate reconstruction and fresh worker liveness.",
        ]
    )
    launch = read_record(preparation_root / f"{wave}-launch-state.json")
    if not launch:
        return [*lines, f"No saved {title} launch acknowledgment.", ""]
    if launch.get("binding_sha256") != hashlib.sha256(release_path.read_bytes()).hexdigest():
        return [*lines, f"Saved {title} launch binding differs; worker acknowledgments withheld.", ""]
    lines.append(
        f"Saved launch `{launch['status']}` at {local_time(launch.get('time'))}; "
        f"{len(launch.get('workers', {}))} / {len(release['arms'])} worker acknowledgments."
    )
    if "error" in launch:
        lines.append(f"Saved launch failure: `{launch['error']}`. Original intents are preserved without retry.")
    if "monitor_pid" in launch:
        lines.append(f"Saved Bash monitor PID `{launch['monitor_pid']}`; not a fresh liveness check.")
    return [*lines, ""]


def selected_lr_cpu_lines(root: Path) -> list[str]:
    """Report preserved full-budget LR preparations only when source/log/archive identities match."""
    lines = []
    for path in sorted((root / "run-preparation/selected-lr-full-budget-cpu").glob("*/receipt.json")):
        receipt = read_record(path)
        lines.extend(
            [
                "### Wave-11 Selected-LR Full-Budget Preparation",
                "",
                f"Saved CPU receipt: `PVC/{path.relative_to(root)}` at {local_time(receipt.get('time'))}.",
            ]
        )
        try:
            candidate = Path(receipt["candidate"])
            prepared_path = candidate / SELECTED_LR_PREPARATION
            prepared = read_record(prepared_path)
            source = require_selected_lr_source(root, Path(prepared["source_receipt"]))
            previous = Path(source["candidate"])
            files = prepared["files_sha256"]
            changed = {
                name
                for name in files.keys() | source["files_sha256"].keys()
                if files.get(name) != source["files_sha256"].get(name)
            }
            identities = [
                (prepared_path, receipt["preparation_sha256"]),
                (Path(receipt["log"]), receipt["log_sha256"]),
                (Path(receipt["archive"]), receipt["archive_sha256"]),
                *((Path(name), digest) for name, digest in prepared["selection"]["files_sha256"].items()),
            ]
            if (
                prepared["status"] != "selected_lr_full_budget_prepared_gpu_denied"
                or prepared["root"] != str(root)
                or prepared["candidate"] != str(candidate.resolve())
                or files != candidate_inventory(candidate)
                or source["files_sha256"] != candidate_inventory(previous)
                or receipt["files_sha256"] != files
                or set(prepared["changed_files"]) != SELECTED_LR_CHANGES
                or changed != SELECTED_LR_CHANGES
                or receipt["command"] != cpu_command()
                or receipt["cwd"] != str(candidate.resolve())
                or receipt["cuda_visible_devices"] != ""
                or type(receipt["exit_code"]) is not int
                or receipt["status"] not in ("selected_lr_full_budget_cpu_verified_gpu_denied", "selected_lr_cpu_failed")
                or (
                    receipt["status"] == "selected_lr_full_budget_cpu_verified_gpu_denied"
                    and (receipt["exit_code"] != 0 or "error" in receipt)
                )
                or any(
                    read_record(candidate / f"experiment/configs/{arm}_0p5m_seed0.json")
                    != selected_lr_full_config(
                        read_record(previous / f"experiment/configs/{arm}_0p5m_seed0.json"), choice
                    )
                    for arm, choice in prepared["selection"]["choices"].items()
                )
                or any(
                    not item.is_file() or hashlib.sha256(item.read_bytes()).hexdigest() != digest
                    for item, digest in identities
                )
            ):
                raise ValueError("Selected-LR preparation source or CPU identities differ")
            expected = files | {
                SELECTED_LR_PREPARATION: receipt["preparation_sha256"],
                "tracking/prepare_selected_lr_full_budget.py": prepared["preparation_helper_sha256"],
            }
            with tarfile.open(receipt["archive"]) as bundle:
                members = bundle.getmembers()
                alias = bundle.getmember("artifacts/external-sources")
                if (
                    len(members) != len(expected) + 1
                    or {member.name for member in members} != set(expected) | {"artifacts/external-sources"}
                    or not alias.issym()
                    or alias.linkname != "pinned-sources-materialized"
                ):
                    raise ValueError("Selected-LR archived source inventory differs")
                for name, digest in expected.items():
                    if not bundle.getmember(name).isfile():
                        raise ValueError("Selected-LR archive has an unexpected source alias")
                    with bundle.extractfile(name) as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                            raise ValueError("Selected-LR archived source bytes differ")
            lines.extend(
                [
                    f"Saved status `{receipt['status']}`; integer exit `{receipt['exit_code']}`; "
                    f"{len(files)} source files and exactly {len(changed)} configuration/denial/cluster changes.",
                    f"Planned {prepared['steps_per_arm']:,} updates and {prepared['tokens_per_arm']:,} tokens per arm.",
                    "Short-study LR transfer uses exploratory seed 0; full-budget optima remain unknown.",
                    "GPU execution remains denied; no validation/production release or training outcome is implied.",
                    "",
                ]
            )
        except (ValueError, KeyError, OSError, tarfile.TarError) as error:
            lines.extend(
                [
                    f"Saved preparation evidence differs or is incomplete: `{type(error).__name__}: {error}`.",
                    "Source/log/failure receipts remain preserved; CPU/launch acceptance is withheld.",
                    "",
                ]
            )
    return lines


def selected_lr_validation_lines(root: Path) -> list[str]:
    """Report saved validation progress without launching or reconstructing GPU tensors."""
    binding_path, helpers, state_path, work, receipt_path = selected_lr_locations(root)
    if not binding_path.exists():
        return []
    lines = [
        "### Wave-11 Selected-LR Actual-Config Validation",
        "",
        f"Saved binding: `PVC/{binding_path.relative_to(root)}`.",
    ]
    try:
        frozen = read_record(binding_path)
        candidate = Path(frozen["candidate"])
        planned = selected_lr_validation_plan(candidate, root, work)
        identities = [
            (Path(frozen["archive"]), frozen["archive_sha256"]),
            (Path(frozen["cpu_receipt"]), frozen["cpu_receipt_sha256"]),
            (Path(frozen["controller_cpu_receipt"]), frozen["controller_cpu_receipt_sha256"]),
            (candidate / SELECTED_LR_VALIDATION_PREPARATION, frozen["preparation_sha256"]),
            (root / "data/experiment_receipt.json", frozen["data_receipt_sha256"]),
            (root / "run-preparation/wave10-launch-binding.json", frozen["predecessor_release_sha256"]),
            *((helpers / name, digest) for name, digest in frozen["helpers_sha256"].items()),
        ]
        expected_commands = [(trial["name"], command) for trial in planned["trials"] for command in trial["commands"]]
        if (
            frozen["status"] != SELECTED_LR_VALIDATION_STATUS
            or frozen["root"] != str(root)
            or frozen["candidate"] != str(candidate.resolve())
            or frozen["helpers"] != str(helpers)
            or frozen["work"] != str(work)
            or frozen["receipt"] != str(receipt_path)
            or frozen["node"] != SELECTED_LR_NODE
            or frozen["environment"] != planned["environment"]
            or frozen["plan"] != planned
            or frozen["commands"] != [command for _, command in expected_commands]
            or frozen["files_sha256"] != candidate_inventory(candidate)
            or not frozen["helpers_sha256"]
            or any(
                not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest
                for path, digest in identities
            )
        ):
            raise ValueError("Saved selected-LR binding, source or archive identity differs")
        lines.extend(
            [
                f"Source-bound validation-only status `{frozen['status']}`; "
                f"{len(frozen['files_sha256'])} candidate files and {len(frozen['helpers_sha256'])} helpers.",
                f"Planned {len(expected_commands)} eight-rank commands, "
                f"{sum(len(item['checks']) for item in planned['trials'])} check flags and "
                f"{planned['planned_validation_tokens']:,} validation tokens.",
                "Full schedules remain 5,120 updates; bounded trajectories are continuous 24 "
                "versus interrupted 22 and resumed to 24.",
            ]
        )
        intent = read_record(helpers.parent / "controller.json")
        state, receipt = read_record(state_path), read_record(receipt_path)
        if not intent:
            if state or receipt:
                raise ValueError("Saved selected-LR execution lacks its original controller acknowledgment")
            return [*lines, "No saved controller intent or GPU execution. No production release.", ""]
        digest = hashlib.sha256(binding_path.read_bytes()).hexdigest()
        if intent["binding_sha256"] != digest or intent["command"] != selected_lr_controller_command(root):
            raise ValueError("Saved selected-LR controller intent differs")
        lines.append(
            f"Saved controller intent `{intent['status']}`; "
            f"PID `{intent.get('pid', 'unacknowledged')}`. This is not fresh liveness."
        )
        if not state:
            if receipt:
                raise ValueError("Saved selected-LR GPU receipt has no controller state")
            return [*lines, "No saved validation execution state; GPU outcomes remain unreported.", ""]
        if (
            intent["status"] != "acknowledged"
            or state["binding_sha256"] != digest
            or state["controller_pid"] != intent["pid"]
            or state["controller_start_ticks"] != intent["start_ticks"]
            or state["candidate"] != str(candidate)
            or state["node"] != frozen["node"]
        ):
            raise ValueError("Saved selected-LR execution owner differs")
        lines.append(f"Saved controller state `{state['status']}`.")
        if "keeper_restore" in state:
            lines.append(f"Recorded keeper restoration `{state['keeper_restore']}`.")
        if "error" in state or "cleanup_error" in state:
            lines.append("Saved controller failure is preserved; inspect the original evidence without retrying.")
        if not receipt:
            return [*lines, "No saved actual-config GPU receipt; no passed command count.", ""]
        trials, commands = receipt["trials"], receipt["gpu_commands"]
        if (
            receipt["status"] not in ("validating", "needs_attention", "passed_actual_full_budget_config_gates")
            or state["command"] != selected_lr_validator_command(root, candidate)
            or receipt["owner"]
            != {
                "controller_pid": state["controller_pid"],
                "controller_start_ticks": state["controller_start_ticks"],
            }
            or any(
                receipt[key] != value
                for key, value in planned.items()
                if key not in ("status", "trials", "gpu_commands")
            )
            or len(trials) != len(planned["trials"])
            or any(
                any(actual[key] != value for key, value in expected.items() if key != "checks")
                or set(actual["checks"]) != set(expected["checks"])
                or any(type(value) is not bool for value in actual["checks"].values())
                for actual, expected in zip(trials, planned["trials"], strict=True)
            )
            or len(commands) > len(expected_commands)
            or any(
                execution["trial"] != arm
                or execution["command"] != command
                or execution["log"] != str(work / f"{arm}_{('continuous', 'interrupted', 'resumed')[index % 3]}.log")
                or type(execution["exit_code"]) is not int
                or not Path(execution["log"]).is_file()
                or hashlib.sha256(Path(execution["log"]).read_bytes()).hexdigest() != execution["log_sha256"]
                for index, (execution, (arm, command)) in enumerate(zip(commands, expected_commands, strict=False))
            )
            or (
                state["status"] == "passed"
                and state["validation_sha256"] != hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            )
        ):
            raise ValueError("Saved selected-LR GPU source, owner, command or log identity differs")
        passed = sum(execution["exit_code"] == 0 for execution in commands)
        flags = sum(value is True for trial in trials for value in trial["checks"].values())
        lines.append(f"Saved GPU receipt status `{receipt['status']}`.")
        if receipt.get("error"):
            lines.append(f"Saved GPU validation failure: `{receipt['error']}`.")
        lines.extend(
            [
                f"Source-bound saved GPU commands: {passed} / {len(expected_commands)} exited zero; "
                f"{len(commands) - passed} failed. Saved required check flags: {flags} / "
                f"{sum(len(item['checks']) for item in planned['trials'])} true.",
                "Saved progress does not independently reconstruct numerical tensors, verify original "
                "process absence or grant production ownership. Full-budget quality remains unmeasured.",
                "",
            ]
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        lines.extend(
            [
                f"Saved validation evidence differs or is incomplete: `{type(error).__name__}: {error}`.",
                "Source/log/failure receipts remain preserved; validation progress counts are withheld.",
                "",
            ]
        )
    return lines


def fused_attribution_lines(root: Path) -> list[str]:
    """Report exact saved attribution preparation/CPU evidence without tensors or GPU operations."""
    lines = []
    receipts = {
        entry["candidate"]: path
        for path in sorted(root.glob("run-preparation/fused-attribution-cpu/*/receipt.json"))
        if (entry := read_record(path)).get("candidate")
    }
    for path in sorted(root.glob(f"candidates/fused-attribution*/{ATTRIBUTION_PREPARATION}")):
        if path.parent.name.startswith("fused-attribution-validation-"):
            continue
        lines.extend(["### Fused Matched-Null Full-Schedule Neighborhood", "", f"`PVC/{path.relative_to(root)}`."])
        try:
            prepared = read_record(path)
            candidate, previous = Path(prepared["candidate"]), Path(prepared["parent_candidate"])
            parent_release = Path(prepared["parent_release"])
            frozen = read_record(parent_release)
            contents = attribution_transformed_files(previous)
            identities = [
                (parent_release, prepared["parent_release_sha256"]),
                (Path(prepared["parent_archive"]), prepared["parent_archive_sha256"]),
                (root / "data/experiment_receipt.json", prepared["data_receipt_sha256"]),
                (root / "run-preparation/analysis-wave12.json", prepared["predecessor_results"]["analysis_sha256"]),
                (root / "run-preparation/swanlab-upload-wave12.json", prepared["predecessor_results"]["cloud_sha256"]),
                *((candidate / name, digest) for name, digest in prepared["parent_metadata_sha256"].items()),
            ]
            files = candidate_inventory(candidate)
            changed = {
                name
                for name in files.keys() | frozen["files_sha256"].keys()
                if files.get(name) != frozen["files_sha256"].get(name)
            }
            if (
                prepared["status"] != ATTRIBUTION_STATUS
                or prepared["root"] != str(root)
                or candidate != path.parent
                or prepared["files_sha256"] != files
                or candidate_inventory(previous) != frozen["files_sha256"]
                or changed != set(contents)
                or set(prepared["changed_files"]) != set(contents)
                or any((candidate / name).read_bytes() != content for name, content in contents.items())
                or any(hashlib.sha256(source.read_bytes()).hexdigest() != digest for source, digest in identities)
                or len(prepared["trials"]) != 9
                or prepared["total_training_tokens"] != 24159191040
                or prepared["reserved_confirmation_seeds"] != [3, 4, 5]
            ):
                raise ValueError("Saved attribution source, full schedules or predecessor identities differ")
            lines.extend(
                [
                    "Nine exploratory seed-0 full-budget runs: MuonT/fused/identity-outer null at "
                    "0.04525483392, 0.06399999989260598 and 0.09050966784.",
                    "Each retains 5,120 updates/2,684,354,560 tokens; total 24,159,191,040 tokens planned.",
                    "Five large training windows and all exact evaluations; eleven final-half selection "
                    "evaluations rank each family. Fresh seeds 3,4,5 remain reserved.",
                    "Identity-outer null retains output collection/moments/roots and factor application; "
                    "timing remains diagnostic.",
                ]
            )
            receipt_path = receipts.get(str(candidate))
            if receipt_path is None:
                lines.extend(["No saved candidate CPU receipt; no GPU binding or launch.", ""])
                continue
            cpu = read_record(receipt_path)
            if (
                cpu["candidate"] != str(candidate)
                or cpu["root"] != str(root)
                or cpu["preparation_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
                or cpu["files_sha256"] != files
                or cpu["helpers_sha256"] != prepared["helpers_sha256"]
                or cpu["command"] != cpu_command()
                or cpu["cwd"] != str(candidate)
                or type(cpu["exit_code"]) is not int
                or cpu["cuda_visible_devices"] != ""
                or cpu["log_sha256"] != hashlib.sha256(Path(cpu["log"]).read_bytes()).hexdigest()
            ):
                raise ValueError("Saved attribution CPU outcome belongs to different source or execution")
            if cpu["status"] == ATTRIBUTION_CPU_STATUS:
                if (
                    cpu["exit_code"] != 0
                    or "error" in cpu
                    or cpu["archive_sha256"] != hashlib.sha256(Path(cpu["archive"]).read_bytes()).hexdigest()
                ):
                    raise ValueError("Saved attribution CPU acceptance or archive differs")
                require_selected_lr_archive(
                    Path(cpu["archive"]),
                    attribution_archive_inventory(prepared, cpu["preparation_sha256"]),
                    alias=True,
                )
                lines.append(
                    f"Source/archive-bound CPU suite exited zero with CUDA hidden at {local_time(cpu.get('time'))}."
                )
                lines.append(
                    f"The CPU archive verifies all {len(cpu['helpers_sha256'])} issued helpers; "
                    "later working-helper edits do not replace that evidence."
                )
            else:
                lines.append(f"Preserved CPU failure: exit `{cpu['exit_code']}`, status `{cpu['status']}`.")
                if "error" in cpu:
                    lines.append(f"Preserved attribution CPU error: `{cpu['error']}`.")
            lines.extend(
                [
                    "GPU numerical/continuation gates and separate execution ownership remain required; "
                    "no new wave results or cloud imports.",
                    "",
                ]
            )
        except (ValueError, KeyError, TypeError, OSError, tarfile.TarError) as error:
            lines.extend(
                [
                    f"Saved attribution evidence differs or is incomplete: `{type(error).__name__}: {error}`.",
                    "CPU acceptance and launch/result claims withheld; original evidence is preserved.",
                    "",
                ]
            )
    return lines


def fused_continuation_lines(root: Path) -> list[str]:
    """Report saved diagnostic preparation and CPU evidence without GPU acceptance."""
    lines = []
    for path in sorted(root.glob(f"candidates/fused-continuation-*/{CONTINUATION_PREPARATION}")):
        current = ["### First-Resumed-Update Fused Diagnostic", "", f"`PVC/{path.relative_to(root)}`."]
        try:
            prepared = read_record(path)
            candidate, previous = path.parent, Path(prepared["parent_candidate"])
            files = candidate_inventory(candidate)
            changed = {
                name
                for name in files.keys() | prepared["parent_files_sha256"].keys()
                if files.get(name) != prepared["parent_files_sha256"].get(name)
            }
            identities = {
                root
                / "run-preparation/fused-attribution-validation-binding-20261118.json": prepared[
                    "parent_binding_sha256"
                ],
                root / "data/experiment_receipt.json": prepared["data_receipt_sha256"],
                **{candidate / name: digest for name, digest in prepared["parent_metadata_sha256"].items()},
                **{
                    root / name: digest for name, digest in prepared["allocation_predecessor"]["evidence_sha256"].items()
                },
            }
            if (
                path.is_symlink()
                or candidate != candidate.resolve()
                or prepared["status"] != CONTINUATION_PREPARED_STATUS
                or prepared["root"] != str(root)
                or prepared["candidate"] != str(candidate)
                or prepared["files_sha256"] != files
                or candidate_inventory(previous) != prepared["parent_files_sha256"]
                or changed != set(prepared["changed_files"])
                or any(
                    hashlib.sha256(source.read_bytes()).hexdigest() != digest for source, digest in identities.items()
                )
            ):
                raise ValueError("Saved continuation source or failed-parent evidence differs")
            current.extend(
                [
                    f"Saved diagnostic preparation at {local_time(prepared.get('time'))}; "
                    f"{len(files)} source files, {len(changed)} changes from the failed parent.",
                    "Paired trace-off/on uninterrupted and 22-to-24 resumed trajectories retain the lower-LR "
                    "5,120-update schedule; checkpoint 23 is diagnostic evidence.",
                ]
            )
            receipts = [
                (receipt_path, receipt)
                for receipt_path in sorted(root.glob("run-preparation/fused-continuation-cpu/**/receipt.json"))
                if (receipt := read_record(receipt_path)).get("candidate") == str(candidate)
            ]
            if not receipts:
                current.append("No saved candidate CPU acceptance; GPU execution remains unissued.")
            for receipt_path, cpu in receipts:
                if (
                    cpu["root"] != str(root)
                    or cpu["preparation_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
                    or cpu["files_sha256"] != files
                    or cpu["helpers_sha256"] != prepared["helpers_sha256"]
                    or cpu["command"] != cpu_command()
                    or type(cpu["exit_code"]) is not int
                    or cpu["cuda_visible_devices"] != ""
                    or cpu["log_sha256"] != hashlib.sha256(Path(cpu["log"]).read_bytes()).hexdigest()
                ):
                    raise ValueError("Saved continuation CPU receipt belongs to different source or execution")
                current.append(f"`PVC/{receipt_path.relative_to(root)}`.")
                if cpu["status"] != CONTINUATION_CPU_STATUS:
                    current.append(f"Preserved CPU failure: `{cpu['status']}`, exit `{cpu['exit_code']}`.")
                    if "error" in cpu:
                        current.append(f"Preserved diagnostic CPU error: `{cpu['error']}`.")
                    continue
                helpers = Path(cpu["helpers"])
                expected = (
                    files
                    | prepared["parent_metadata_sha256"]
                    | {CONTINUATION_PREPARATION: cpu["preparation_sha256"]}
                    | {f"tracking/{name}": digest for name, digest in cpu["helpers_sha256"].items()}
                )
                if (
                    cpu["exit_code"] != 0
                    or "error" in cpu
                    or helpers.is_symlink()
                    or helpers != helpers.resolve()
                    or any(
                        Path(name).name != name
                        or (helpers / name).is_symlink()
                        or hashlib.sha256((helpers / name).read_bytes()).hexdigest() != digest
                        for name, digest in cpu["helpers_sha256"].items()
                    )
                    or cpu["archive_sha256"] != hashlib.sha256(Path(cpu["archive"]).read_bytes()).hexdigest()
                ):
                    raise ValueError("Saved continuation CPU acceptance or archived helpers differ")
                require_selected_lr_archive(Path(cpu["archive"]), expected, alias=True)
                current.append(
                    f"Source/archive-bound diagnostic CPU suite exited zero with CUDA hidden "
                    f"at {local_time(cpu.get('time'))}; {len(cpu['helpers_sha256'])} issued helpers verified."
                )
            current.append(
                "Trace-off/on GPU parity, exact-keeper ownership and all original production gates remain required. "
                "No full-budget result, throughput claim or new cloud import is inferred from this diagnostic."
            )
        except (ValueError, KeyError, TypeError, OSError, tarfile.TarError) as error:
            current = [
                *current[:3],
                f"Saved continuation evidence differs or is incomplete: `{type(error).__name__}: {error}`.",
                "CPU acceptance and result claims withheld; original evidence is preserved.",
            ]
        lines.extend([*current, ""])
    return lines


def fused_continuation_controller_lines(root: Path) -> list[str]:
    """Retain separately identified executor outcomes and source/archive evidence."""
    lines = []
    for path in sorted(root.glob("run-preparation/fused-continuation-controller-cpu/**/receipt.json")):
        current = ["### Fused Continuation Diagnostic Executor", "", f"`PVC/{path.relative_to(root)}`."]
        try:
            receipt = read_record(path)
            cpu_path = Path(receipt["candidate_cpu_receipt"])
            cpu = read_record(cpu_path)
            candidate, helpers = Path(cpu["candidate"]), Path(receipt["helpers"])
            prepared = read_record(candidate / CONTINUATION_PREPARATION)
            if (
                receipt["root"] != str(root)
                or receipt["candidate_cpu_receipt_sha256"] != sha256(cpu_path)
                or cpu["root"] != str(root)
                or cpu["status"] != CONTINUATION_CPU_STATUS
                or type(cpu["exit_code"]) is not int
                or cpu["exit_code"] != 0
                or cpu["cuda_visible_devices"] != ""
                or "error" in cpu
                or cpu["files_sha256"] != candidate_inventory(candidate)
                or cpu["preparation_sha256"] != sha256(candidate / CONTINUATION_PREPARATION)
                or cpu["files_sha256"] != prepared["files_sha256"]
                or cpu["helpers_sha256"] != prepared["helpers_sha256"]
                or cpu["log_sha256"] != sha256(Path(cpu["log"]))
                or cpu["archive_sha256"] != sha256(Path(cpu["archive"]))
                or any(sha256(Path(cpu["helpers"]) / name) != value for name, value in cpu["helpers_sha256"].items())
                or receipt["command"] != continuation_controller_cpu_command()
                or type(receipt["exit_code"]) is not int
                or receipt["cuda_visible_devices"] != ""
                or receipt["log_sha256"] != sha256(Path(receipt["log"]))
            ):
                raise ValueError("Saved continuation executor belongs to different candidate or execution evidence")
            require_selected_lr_archive(
                Path(cpu["archive"]),
                cpu["files_sha256"]
                | prepared["parent_metadata_sha256"]
                | {CONTINUATION_PREPARATION: cpu["preparation_sha256"]}
                | {f"tracking/{name}": value for name, value in cpu["helpers_sha256"].items()},
                alias=True,
            )
            current.append(
                f"Saved executor CPU attempt at {local_time(receipt.get('time'))}; "
                f"status `{receipt['status']}`, exit `{receipt['exit_code']}`, CUDA hidden."
            )
            if receipt["status"] != CONTINUATION_CONTROLLER_CPU_STATUS:
                current.append("Preserved executor CPU failure; it authorizes no freeze or GPU execution.")
                if "error" in receipt:
                    current.append(f"Saved executor error: `{receipt['error']}`.")
                if receipt.get("archive_sha256") is not None:
                    if receipt["archive_sha256"] != sha256(Path(receipt["archive"])):
                        raise ValueError("Saved failed executor source archive changed")
                    current.append("The failed attempt's source archive and log hashes remain verified.")
            else:
                fixtures = {name: cpu["files_sha256"][name] for name in continuation_controller_fixture_names()}
                if (
                    receipt["exit_code"] != 0
                    or "error" in receipt
                    or helpers.is_symlink()
                    or helpers != helpers.resolve()
                    or any(
                        Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != value
                        for name, value in receipt["helpers_sha256"].items()
                    )
                    or any(receipt["helpers_sha256"].get(name) != value for name, value in cpu["helpers_sha256"].items())
                    or receipt["fixture_files_sha256"] != fixtures
                    or any(sha256(helpers.parent / name) != value for name, value in fixtures.items())
                    or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
                ):
                    raise ValueError("Saved continuation executor acceptance or archived bytes differ")
                require_selected_lr_archive(
                    Path(receipt["archive"]),
                    {f"tracking/{name}": value for name, value in receipt["helpers_sha256"].items()} | fixtures,
                    alias=False,
                )
                current.append(
                    f"Source/archive-bound executor CPU suite exited zero; "
                    f"{len(receipt['helpers_sha256'])} helpers and {len(fixtures)} configuration fixtures verified."
                )
                current.append("This saved CPU outcome establishes no GPU transparency or production acceptance.")
        except (ValueError, KeyError, TypeError, OSError, tarfile.TarError) as error:
            current = [
                *current[:3],
                f"Saved executor evidence differs or is incomplete: `{type(error).__name__}: {error}`.",
                "Executor acceptance and GPU/result claims withheld; original evidence is preserved.",
            ]
        lines.extend([*current, ""])
    binding = continuation_locations(root)[0]
    if lines and not binding.exists():
        lines.extend(["No saved continuation execution binding; the six-command GPU diagnostic remains unissued.", ""])
    elif binding.exists():
        current = ["### Fused Continuation GPU Diagnostic", "", f"`PVC/{binding.relative_to(root)}`."]
        try:
            frozen = read_record(binding)
            candidate = Path(frozen["candidate"])
            binding_path, helpers, state_path, _, gpu_path = continuation_locations(root)
            digest = sha256(binding_path)
            if (
                frozen["status"] != CONTINUATION_FROZEN_STATUS
                or frozen["root"] != str(root)
                or frozen["files_sha256"] != candidate_inventory(candidate)
                or frozen["helpers"] != str(helpers)
                or any(sha256(helpers / name) != value for name, value in frozen["helpers_sha256"].items())
                or frozen["preparation_sha256"] != sha256(candidate / CONTINUATION_PREPARATION)
                or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
                or frozen["plan"] != continuation_plan(candidate, root)
                or frozen["commands"] != frozen["plan"]["commands"]
                or frozen["cpu_receipt_sha256"] != sha256(Path(frozen["cpu_receipt"]))
                or frozen["controller_cpu_receipt_sha256"] != sha256(Path(frozen["controller_cpu_receipt"]))
                or any(
                    sha256(root / name) != value
                    for name, value in frozen["allocation_predecessor"]["evidence_sha256"].items()
                )
            ):
                raise ValueError("Saved diagnostic binding/source/commands or original failure evidence changed")
            require_selected_lr_archive(Path(frozen["archive"]), continuation_inventory(frozen), alias=True)
            current.append(
                f"Saved diagnostic freeze at {local_time(frozen.get('time'))}; six bounded commands, "
                "original lower LR/full schedule and unchanged continuation tolerances."
            )
            intent = read_record(helpers.parent / "controller.json")
            state = read_record(state_path)
            gpu = read_record(gpu_path)
            if intent:
                if intent["binding_sha256"] != digest or intent["command"] != continuation_controller_command(root):
                    raise ValueError("Saved diagnostic launch intent belongs to a different generation")
                current.append(f"Saved controller intent: `{intent['status']}` at {local_time(intent.get('time'))}.")
            if state:
                if (
                    state["binding_sha256"] != digest
                    or state["candidate"] != str(candidate)
                    or state["controller_pid"] != intent["pid"]
                    or state["controller_start_ticks"] != intent["start_ticks"]
                    or state["allocation_predecessor"] != frozen["allocation_predecessor"]
                    or ("command" in state and state["command"] != continuation_validator_command(root, candidate))
                ):
                    raise ValueError("Saved diagnostic state or validator ownership differs")
                current.append(f"Saved executor state: `{state['status']}` at {local_time(state.get('time'))}.")
                if "keeper_restore" in state:
                    current.append(f"Recorded diagnostic keeper restoration: `{state['keeper_restore']}`.")
                if "error" in state:
                    current.append(f"Preserved executor error: `{state['error']}`.")
            if gpu:
                if (
                    gpu["execution_binding_sha256"] != digest
                    or gpu["candidate_preparation_sha256"] != frozen["preparation_sha256"]
                    or gpu["owner"]
                    != {
                        "controller_pid": state["controller_pid"],
                        "controller_start_ticks": state["controller_start_ticks"],
                    }
                    or gpu["commands"] != frozen["commands"]
                    or len(gpu["gpu_commands"]) > len(frozen["commands"])
                    or any(
                        item["command"] != frozen["commands"][index]
                        or type(item["exit_code"]) is not int
                        or item["log_sha256"] != sha256(Path(item["log"]))
                        for index, item in enumerate(gpu["gpu_commands"])
                    )
                ):
                    raise ValueError("Saved diagnostic command/log/owner evidence differs")
                complete = sum(item["exit_code"] == 0 for item in gpu["gpu_commands"])
                current.append(
                    f"Saved GPU receipt `{gpu['status']}` at {local_time(gpu.get('time'))}; "
                    f"{complete}/6 integer-zero command exits."
                )
                if "error" in gpu:
                    current.append(f"Preserved GPU diagnostic error: `{gpu['error']}`.")
                if gpu["status"] == CONTINUATION_COMPLETE_STATUS:
                    current.append(
                        f"Saved transparency predicates: trace `{gpu['analysis']['trace_transparent']}`, "
                        f"extra checkpoint `{gpu['analysis']['extra_checkpoint_transparent']}`."
                    )
                    current.append("These are saved diagnostic predicates; independent full reconstruction is required.")
            elif not state:
                current.append("No diagnostic GPU execution state or command receipt exists.")
            current.append(
                "No full-budget optimizer result or cloud import is inferred. "
                "All original production gates remain required."
            )
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, tarfile.TarError) as error:
            current = [
                *current[:3],
                f"Saved GPU diagnostic evidence differs or is incomplete: `{type(error).__name__}: {error}`.",
                "Diagnostic progress and acceptance claims withheld; original evidence is preserved.",
            ]
        lines.extend([*current, ""])
    return lines


def saved_ns_replay_cpu_lines(root: Path) -> list[str]:
    """Keep archived replay preparation separate from live GPU or training acceptance."""
    lines = []
    for path in sorted(root.glob("run-preparation/saved-ns-gpu-adapter-cpu/**/receipt.json")):
        current = ["### Saved-NS Replay CPU Preparation", "", f"`PVC/{path.relative_to(root)}`."]
        try:
            receipt = read_record(path)
            helpers = Path(receipt["helpers"])
            hashes = receipt["helpers_sha256"]
            fixtures = receipt["fixtures_sha256"]
            previous = read_record(continuation_locations(root)[0])
            inherited = previous["helpers_sha256"]
            required = {
                *SAVED_NS_WORKER_SOURCES,
                "saved_ns_gpu_adapter.py",
                "replay_pytest_guard.py",
                "recovery_keeper_manager.sh",
                "test_saved_ns_gpu_owner.py",
                "test_saved_ns_gpu_adapter.py",
                "test_saved_ns_gpu_worker.py",
                "test_replay_saved_ns.py",
            }
            manifest_name = "experiment/upstream_manifest.json"
            manifest = read_record(helpers.parent / manifest_name)
            muon_name = f"artifacts/external-sources/Toner-{manifest['revision']}/src/toner/fast/muon.py"
            pinned_name = muon_name.replace("external-sources", "pinned-sources-materialized")
            if (
                path.is_symlink()
                or helpers.is_symlink()
                or helpers != path.parent / "tracking"
                or helpers != helpers.resolve()
                or receipt["root"] != str(root)
                or receipt["cwd"] != str(path.parent)
                or receipt["command"] != saved_ns_cpu_command(helpers)
                or type(receipt["exit_code"]) is not int
                or receipt["cuda_visible_devices"] != ""
                or not required <= hashes.keys()
                or any(hashes.get(name) != value for name, value in inherited.items())
                or any(
                    Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != value
                    for name, value in hashes.items()
                )
                or set(fixtures) != {manifest_name, muon_name}
                or fixtures[manifest_name] != previous["files_sha256"][manifest_name]
                or fixtures[muon_name] != previous["files_sha256"][pinned_name]
                or fixtures[muon_name] != manifest["files"]["src/toner/fast/muon.py"]
                or any(
                    (helpers.parent / name).is_symlink() or sha256(helpers.parent / name) != value
                    for name, value in fixtures.items()
                )
                or receipt["log_sha256"] != sha256(Path(receipt["log"]))
                or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
            ):
                raise ValueError("Saved replay CPU command, archived sources or executed NS fixtures differ")
            require_selected_lr_archive(
                Path(receipt["archive"]),
                {f"tracking/{name}": value for name, value in hashes.items()} | fixtures,
                alias=False,
            )
            current.append(
                f"Saved replay CPU attempt at {local_time(receipt.get('time'))}; "
                f"status `{receipt['status']}`, exit `{receipt['exit_code']}`, CUDA hidden."
            )
            if receipt["status"] != SAVED_NS_CPU_STATUS:
                current.append("Preserved CPU failure; no replay acceptance or GPU authorization.")
                if "error" in receipt:
                    current.append(f"Preserved replay CPU error: `{receipt['error']}`.")
            elif receipt["exit_code"] != 0 or "error" in receipt:
                raise ValueError("Saved replay CPU acceptance contradicts its execution outcome")
            else:
                current.append(
                    f"Source/archive-bound replay CPU suite exited zero; "
                    f"{len(hashes)} helpers and two executed pinned-source fixtures verified."
                )
            current.append(
                "This is archived CPU evidence, not a fresh process/device observation, "
                "GPU replay launch, numerical continuation pass, optimizer result or cloud import. "
                "Original terminal reconstruction and exact healthy keeper restoration remain "
                "required before a separately issued replay owner can start."
            )
        except (ValueError, KeyError, TypeError, OSError, tarfile.TarError) as error:
            current = [
                *current[:3],
                f"Saved replay CPU evidence differs or is incomplete: `{type(error).__name__}: {error}`.",
                "Replay CPU acceptance and GPU/result claims withheld; original evidence is preserved.",
            ]
        lines.extend([*current, ""])
    return lines


def root_preparation_cpu_lines(root: Path) -> list[str]:
    """Report archived scratch-readiness attempts without issuing GPU ownership."""
    lines = []
    for path in sorted(root.glob(f"candidates/root-preparation-*/{ROOT_PREPARATION}")):
        current = ["### Restored-Root Preparation Diagnostic", "", f"`PVC/{path.relative_to(root)}`."]
        try:
            prepared = read_record(path)
            candidate, previous = path.parent, Path(prepared["parent_candidate"])
            files = candidate_inventory(candidate)
            changed = {
                name
                for name in files.keys() | prepared["parent_files_sha256"].keys()
                if files.get(name) != prepared["parent_files_sha256"].get(name)
            }
            identities = {
                continuation_locations(root)[0]: prepared["parent_binding_sha256"],
                root / "data/experiment_receipt.json": prepared["data_receipt_sha256"],
                **{candidate / name: value for name, value in prepared["parent_metadata_sha256"].items()},
                **{root / name: value for name, value in prepared["allocation_predecessor"]["evidence_sha256"].items()},
            }
            if (
                path.is_symlink()
                or candidate != candidate.resolve()
                or prepared["status"] != ROOT_PREPARATION_STATUS
                or prepared["root"] != str(root)
                or prepared["candidate"] != str(candidate)
                or prepared["files_sha256"] != files
                or candidate_inventory(previous) != prepared["parent_files_sha256"]
                or changed != ROOT_PREPARATION_CHANGES
                or changed != set(prepared["changed_files"])
                or prepared["plan"] != root_preparation_plan(candidate, root)
                or prepared["allocation_predecessor"]["evidence_sha256"].get(ROOT_PREPARATION_PREDECESSOR)
                != ROOT_PREPARATION_PREDECESSOR_SHA256
                or any(sha256(source) != value for source, value in identities.items())
            ):
                raise ValueError("Saved root preparation scope, schedule or accepted replay predecessor differs")
            current.extend(
                [
                    f"Saved scratch-readiness preparation at {local_time(prepared.get('time'))}; "
                    f"{len(files)} source files and {len(changed)} readiness/owner/trainer/test changes.",
                    "Twelve planned unprepared/prepared control/trace commands retain the full lower-LR schedule. "
                    "Only prepared checkpoint resumes request scratch-kernel preparation.",
                ]
            )
            receipts = [
                (receipt_path, receipt)
                for receipt_path in sorted(root.glob("run-preparation/root-preparation-cpu/**/receipt.json"))
                if (receipt := read_record(receipt_path)).get("candidate") == str(candidate)
            ]
            if not receipts:
                current.append("No saved candidate CPU acceptance; GPU execution remains unissued.")
            for receipt_path, cpu in receipts:
                helpers = Path(cpu["helpers"])
                command = root_preparation_cpu_command(candidate)
                legacy_command = command.copy()
                position = legacy_command.index("--rootdir")
                del legacy_command[position : position + 2]
                failed = cpu["status"] != ROOT_PREPARATION_CPU_STATUS
                allowed_commands = [[*command, "--junitxml", cpu["tests"]]]
                if failed:
                    allowed_commands.append([*legacy_command, "--junitxml", cpu["tests"]])
                if (
                    receipt_path.is_symlink()
                    or cpu["root"] != str(root)
                    or cpu["cwd"] != str(candidate)
                    or cpu["preparation_sha256"] != sha256(path)
                    or cpu["files_sha256"] != files
                    or cpu["helpers_sha256"] != prepared["helpers_sha256"]
                    or cpu["command"] not in allowed_commands
                    or type(cpu["exit_code"]) is not int
                    or cpu["cuda_visible_devices"] != ""
                    or helpers != receipt_path.parent / "tracking"
                    or helpers.is_symlink()
                    or helpers != helpers.resolve()
                    or any(
                        Path(name).name != name or (helpers / name).is_symlink() or sha256(helpers / name) != value
                        for name, value in cpu["helpers_sha256"].items()
                    )
                    or any(
                        Path(cpu[name]).is_symlink() or cpu[f"{name}_sha256"] != sha256(Path(cpu[name]))
                        for name in ("log", "tests", "archive")
                    )
                ):
                    raise ValueError("Saved root CPU command, source, log, tests or archive differs")
                require_selected_lr_archive(
                    Path(cpu["archive"]),
                    root_preparation_archive_inventory(prepared, cpu["preparation_sha256"]),
                    alias=True,
                )
                suites = list(ET.parse(cpu["tests"]).getroot().iter("testsuite"))
                totals = {
                    name: sum(int(suite.attrib[name]) for suite in suites)
                    for name in ("tests", "failures", "errors", "skipped")
                }
                current.append(f"`PVC/{receipt_path.relative_to(root)}`.")
                if failed:
                    current.append(
                        f"Preserved CPU failure at {local_time(cpu.get('time'))}: exit `{cpu['exit_code']}`; "
                        f"{totals['tests']} tests, {totals['failures']} failures, {totals['errors']} errors. "
                        "Original source/log/XML/archive verified; no candidate acceptance."
                    )
                elif (
                    cpu["exit_code"] != 0
                    or "error" in cpu
                    or totals["tests"] <= 0
                    or any(totals[name] for name in ("failures", "errors", "skipped"))
                ):
                    raise ValueError("Saved root CPU acceptance contradicts its test outcome")
                else:
                    current.append(
                        f"Source/archive-bound root diagnostic CPU suite exited zero at {local_time(cpu.get('time'))}; "
                        f"{totals['tests']} tests passed and {len(cpu['helpers_sha256'])} helpers verified, CUDA hidden."
                    )
            current.append(
                "Root-repeat/cold-capture safety, fresh graph capture/replay, trace transparency, all-rank "
                "22/23/24 continuation at original tolerance, separate owner and exact restoration remain required. "
                "CPU evidence authorizes no GPU launch, production acceptance, optimizer result or cloud import."
            )
        except (ValueError, KeyError, TypeError, OSError, tarfile.TarError, ET.ParseError) as error:
            current = [
                *current[:3],
                f"Saved root preparation evidence differs or is incomplete: `{type(error).__name__}: {error}`.",
                "Root CPU acceptance and GPU/result claims withheld; original evidence is preserved.",
            ]
        lines.extend([*current, ""])
    return lines


def fused_attribution_validation_lines(root: Path) -> list[str]:
    """Render saved attribution validation identities, never tensor or live allocation acceptance."""
    binding_path, helpers, state_path, work, gpu_path = attribution_validation_locations(root)
    paths = set(root.glob(f"candidates/fused-attribution-validation-*/{ATTRIBUTION_VALIDATION_PREPARATION}"))
    if binding_path.exists():
        try:
            paths.add(Path(read_record(binding_path)["candidate"]) / ATTRIBUTION_VALIDATION_PREPARATION)
        except (ValueError, KeyError, TypeError, OSError) as error:
            return [
                "### Fused Attribution Actual-Config Validation",
                "",
                f"Saved attribution binding is unreadable: `{type(error).__name__}: {error}`.",
                "CPU acceptance and attribution progress counts withheld; original evidence is preserved.",
                "",
            ]
    receipts = sorted(root.glob("run-preparation/fused-attribution-validation-cpu/**/receipt.json"))
    lines = []
    for path in sorted(paths):
        heading = [
            "### Fused Attribution Actual-Config Validation",
            "",
            f"Saved preparation: `{path}`.",
        ]
        current, identities = list(heading), []

        def checked_file(source: Path, digest: str, captured: list = identities) -> None:
            if source != source.resolve() or not source.is_file() or source.is_symlink():
                raise ValueError(f"Saved attribution file is not canonical and regular: {source}")
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Saved attribution file identity differs: {source}")
            captured.append((source, digest))

        def saved(source: Path) -> dict:
            raw = source.read_bytes()
            checked_file(source, hashlib.sha256(raw).hexdigest())
            return json.loads(raw)

        def cpu_evidence(receipt_path: Path, prepared: dict, preparation_path: Path) -> dict:
            cpu = saved(receipt_path)
            candidate = preparation_path.parent
            if (
                cpu["root"] != str(root)
                or cpu["candidate"] != str(candidate)
                or cpu["files_sha256"] != prepared["files_sha256"]
                or cpu["helpers_sha256"] != prepared["helpers_sha256"]
                or cpu["preparation_sha256"] != hashlib.sha256(preparation_path.read_bytes()).hexdigest()
                or cpu["command"] != cpu_command()
                or cpu["cwd"] != str(candidate)
                or type(cpu["exit_code"]) is not int
                or cpu["cuda_visible_devices"] != ""
            ):
                raise ValueError("Saved attribution CPU receipt belongs to different source or execution")
            checked_file(Path(cpu["log"]), cpu["log_sha256"])
            if cpu["status"] == ATTRIBUTION_VALIDATION_CPU_STATUS or cpu["status"] == ATTRIBUTION_CPU_STATUS:
                if cpu["exit_code"] != 0 or "error" in cpu:
                    raise ValueError("Saved attribution CPU acceptance has a failed outcome")
                checked_file(Path(cpu["archive"]), cpu["archive_sha256"])
                inventory = (
                    attribution_validation_archive_inventory
                    if preparation_path.name == ATTRIBUTION_VALIDATION_PREPARATION
                    else attribution_archive_inventory
                )
                require_selected_lr_archive(
                    Path(cpu["archive"]), inventory(prepared, cpu["preparation_sha256"]), alias=True
                )
                if preparation_path.name == ATTRIBUTION_VALIDATION_PREPARATION:
                    issued_helpers = Path(cpu["helpers"])
                    if (
                        issued_helpers != issued_helpers.resolve()
                        or issued_helpers.is_symlink()
                        or {entry.name for entry in issued_helpers.iterdir()} != set(cpu["helpers_sha256"])
                    ):
                        raise ValueError("Saved attribution CPU helpers are not canonical")
                    for name, digest in cpu["helpers_sha256"].items():
                        if Path(name).name != name:
                            raise ValueError("Saved attribution CPU helper name is not local")
                        checked_file(issued_helpers / name, digest)
            return cpu

        try:
            prepared = saved(path)
            candidate, previous = path.parent, Path(prepared["parent_candidate"])
            files = candidate_inventory(candidate)
            parent_path = previous / ATTRIBUTION_PREPARATION
            parent = saved(parent_path)
            parent_cpu_path = Path(prepared["parent_cpu_receipt"])
            checked_file(parent_cpu_path, prepared["parent_cpu_receipt_sha256"])
            checked_file(parent_path, prepared["parent_preparation_sha256"])
            parent_cpu = cpu_evidence(parent_cpu_path, parent, parent_path)
            planned = attribution_validation_plan(candidate, root, work)
            predecessor = failed_profiler_evidence(root)
            changed = {
                name
                for name in files.keys() | parent_cpu["files_sha256"].keys()
                if files.get(name) != parent_cpu["files_sha256"].get(name)
            }
            if (
                root != root.resolve()
                or prepared["status"] != ATTRIBUTION_VALIDATION_PREPARED_STATUS
                or prepared["root"] != str(root)
                or prepared["candidate"] != str(candidate)
                or candidate != candidate.resolve()
                or previous != previous.resolve()
                or parent["status"] != ATTRIBUTION_STATUS
                or parent_cpu["status"] != ATTRIBUTION_CPU_STATUS
                or candidate_inventory(previous) != parent_cpu["files_sha256"]
                or prepared["files_sha256"] != files
                or changed != ATTRIBUTION_VALIDATION_CHANGES
                or set(prepared["changed_files"]) != ATTRIBUTION_VALIDATION_CHANGES
                or prepared["parent_metadata_sha256"]
                != {entry.name: hashlib.sha256(entry.read_bytes()).hexdigest() for entry in previous.glob("*.json")}
                or prepared["scientific_parent_release_sha256"] != parent["parent_release_sha256"]
                or prepared["data_receipt_sha256"] != parent["data_receipt_sha256"]
                or prepared["allocation_predecessor"] != predecessor
                or prepared["plan"] != planned
                or prepared["expansion_policy"] != ATTRIBUTION_VALIDATION_EXPANSION_POLICY
                or (candidate / "experiment/train.py").read_bytes() != attribution_validation_transformed_train(previous)
                or any(
                    files[destination] != prepared["helpers_sha256"][source]
                    for source, destination in ATTRIBUTION_VALIDATION_COPIES.items()
                )
                or any(
                    (candidate / spec["config_name"]).read_bytes() != (previous / spec["config_name"]).read_bytes()
                    for spec in attribution_validation_trial_specs()
                )
            ):
                raise ValueError("Saved attribution validation source, actual configs or predecessor differs")
            for source, digest in (
                (root / "run-preparation/wave12-launch-binding.json", prepared["scientific_parent_release_sha256"]),
                (root / "data/experiment_receipt.json", prepared["data_receipt_sha256"]),
                (root / "run-preparation/analysis-wave12.json", parent["predecessor_results"]["analysis_sha256"]),
                (root / "run-preparation/swanlab-upload-wave12.json", parent["predecessor_results"]["cloud_sha256"]),
                *((candidate / name, digest) for name, digest in prepared["parent_metadata_sha256"].items()),
                *((root / name, digest) for name, digest in predecessor["evidence_sha256"].items()),
            ):
                checked_file(source, digest)
            current.extend(
                [
                    f"Source-bound nine-config validation: 27 eight-rank commands; "
                    f"{planned['planned_validation_tokens']:,} validation tokens planned.",
                    "Actual 5,120-update schedules remain fixed; bounded trajectories are independent "
                    "continuous 24 versus interrupted 22 and resumed to 24.",
                    "The allocation predecessor remains the preserved failed profiler, not a passed profiler "
                    "or a new production result.",
                ]
            )
            frozen = saved(binding_path) if binding_path.exists() else {}
            bound = frozen.get("candidate") == str(candidate)
            receipt_path = Path(frozen["cpu_receipt"]) if bound else None
            if receipt_path is None:
                for entry in receipts:
                    if saved(entry).get("candidate") == str(candidate):
                        receipt_path = entry
            if receipt_path is None:
                current.append("No saved candidate CPU acceptance; no attribution GPU progress published.")
            else:
                cpu = cpu_evidence(receipt_path, prepared, path)
                if cpu["status"] != ATTRIBUTION_VALIDATION_CPU_STATUS:
                    current.append(
                        f"Preserved attribution validation CPU failure: `{cpu['status']}`, exit "
                        f"`{cpu['exit_code']}`; no CPU acceptance or GPU progress published."
                    )
                    if cpu.get("error"):
                        current.append(f"Preserved attribution validation CPU error: `{cpu['error']}`.")
                    if bound:
                        raise ValueError(
                            "Frozen attribution validation refers to a failed candidate CPU receipt: "
                            f"{cpu['status']}, exit {cpu['exit_code']}, error {cpu.get('error', 'unrecorded')}"
                        )
                else:
                    current.append(
                        f"Source/archive-bound validation CPU suite exited zero with CUDA hidden at "
                        f"{local_time(cpu.get('time'))}; `{receipt_path}`."
                    )
                    current.append("CPU acceptance uses its issued archive, not later mutable working helpers.")
            if bound:
                controller_path = Path(frozen["controller_cpu_receipt"])
                controller = saved(controller_path)
                fixtures = {
                    spec["config_name"]: files[spec["config_name"]] for spec in attribution_validation_trial_specs()
                }
                controller_helpers = Path(controller["helpers"])
                controller_command = [
                    ATTRIBUTION_VALIDATION_PYTHON,
                    "-P",
                    "-B",
                    "-m",
                    "pytest",
                    "-c",
                    "/dev/null",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    "-m",
                    "not slow and not integration and not data_integration and not cluster "
                    "and not requires_cluster and not docker and not manual",
                    *(f"tracking/{name}" for name in ATTRIBUTION_VALIDATION_CONTROLLER_CPU_TESTS),
                ]
                if (
                    frozen["status"] != ATTRIBUTION_VALIDATION_FROZEN_STATUS
                    or frozen["root"] != str(root)
                    or frozen["node"] != ATTRIBUTION_VALIDATION_NODE
                    or frozen["python"] != ATTRIBUTION_VALIDATION_PYTHON
                    or frozen["helpers"] != str(helpers)
                    or frozen["work"] != str(work)
                    or frozen["receipt"] != str(gpu_path)
                    or frozen["validation_timeout"] != ATTRIBUTION_VALIDATION_TIMEOUT
                    or frozen["files_sha256"] != files
                    or frozen["environment"] != planned["environment"]
                    or frozen["plan"] != planned
                    or frozen["commands"] != [command for trial in planned["trials"] for command in trial["commands"]]
                    or frozen["data_receipt_sha256"] != prepared["data_receipt_sha256"]
                    or frozen["allocation_predecessor"] != predecessor
                    or controller["status"] != "fused_attribution_validation_controller_cpu_verified_no_gpu_binding"
                    or controller["root"] != str(root)
                    or controller["candidate_cpu_receipt"] != str(receipt_path)
                    or controller["candidate_cpu_receipt_sha256"] != frozen["cpu_receipt_sha256"]
                    or controller["helpers_sha256"] != frozen["helpers_sha256"]
                    or not frozen["helpers_sha256"]
                    or any(
                        frozen["helpers_sha256"].get(name) != digest for name, digest in cpu["helpers_sha256"].items()
                    )
                    or type(controller["exit_code"]) is not int
                    or controller["exit_code"] != 0
                    or controller["cuda_visible_devices"] != ""
                    or controller["command"] != controller_command
                    or controller["cwd"] != str(controller_helpers.parent)
                    or controller["fixture_files_sha256"] != fixtures
                    or "error" in controller
                    or frozen["archive"] != str(root / "run-preparation" / f"source-{helpers.parent.name}.tar")
                    or Path(controller["archive"]) != controller_helpers.parent / "controller-source.tar"
                    or Path(controller["log"]) != controller_helpers.parent / "controller.log"
                ):
                    raise ValueError("Saved attribution frozen source or controller CPU binding differs")
                for source, digest in (
                    (path, frozen["preparation_sha256"]),
                    (receipt_path, frozen["cpu_receipt_sha256"]),
                    (controller_path, frozen["controller_cpu_receipt_sha256"]),
                    (Path(frozen["archive"]), frozen["archive_sha256"]),
                    (Path(controller["archive"]), controller["archive_sha256"]),
                    (Path(controller["log"]), controller["log_sha256"]),
                    *((controller_helpers.parent / name, digest) for name, digest in fixtures.items()),
                ):
                    checked_file(source, digest)
                for name, digest in frozen["helpers_sha256"].items():
                    if Path(name).name != name:
                        raise ValueError("Saved attribution controller helper name is not local")
                    checked_file(helpers / name, digest)
                    checked_file(controller_helpers / name, digest)
                require_selected_lr_archive(
                    Path(controller["archive"]),
                    {f"tracking/{name}": digest for name, digest in frozen["helpers_sha256"].items()} | fixtures,
                    alias=False,
                )
                require_selected_lr_archive(
                    Path(frozen["archive"]),
                    files
                    | prepared["parent_metadata_sha256"]
                    | {
                        ATTRIBUTION_VALIDATION_PREPARATION: frozen["preparation_sha256"],
                        "candidate-cpu-receipt.json": frozen["cpu_receipt_sha256"],
                        "controller-cpu-receipt.json": frozen["controller_cpu_receipt_sha256"],
                    }
                    | {f"tracking/{name}": digest for name, digest in frozen["helpers_sha256"].items()},
                    alias=True,
                )
                current.append(
                    f"Saved validation-only binding: `{binding_path}`; SHA256 "
                    f"`{hashlib.sha256(binding_path.read_bytes()).hexdigest()}`."
                )
                intent_path = helpers.parent / "controller.json"
                intent = saved(intent_path) if intent_path.exists() else {}
                state = saved(state_path) if state_path.exists() else {}
                receipt = saved(gpu_path) if gpu_path.exists() else {}
                digest = hashlib.sha256(binding_path.read_bytes()).hexdigest()
                if not intent:
                    if state or receipt:
                        raise ValueError("Saved attribution execution has no controller intent")
                    current.append("No saved attribution controller intent or GPU execution.")
                else:
                    if (
                        intent["binding_sha256"] != digest
                        or intent["command"] != attribution_validation_controller_command(root)
                        or intent["allocation_predecessor"] != predecessor
                        or not intent["marker"]
                        or intent["log"] != str(helpers.parent / "controller.log")
                    ):
                        raise ValueError("Saved attribution controller intent differs")
                    if intent["status"] == "acknowledged":
                        generation = intent["generation"]
                        if (
                            type(intent["pid"]) is not int
                            or not str(intent["start_ticks"]).isdigit()
                            or any(
                                generation[key] != value
                                for key, value in {
                                    "pid": intent["pid"],
                                    "start_ticks": intent["start_ticks"],
                                    "group": intent["pid"],
                                    "session": intent["pid"],
                                    "command": intent["command"],
                                }.items()
                            )
                        ):
                            raise ValueError("Saved attribution controller acknowledgment differs")
                    current.append(
                        f"Saved attribution controller intent `{intent['status']}`; owner "
                        f"`{intent.get('pid', 'unacknowledged')}/{intent.get('start_ticks', 'unrecorded')}`; "
                        "not fresh process liveness."
                    )
                    if "log_sha256" in intent:
                        checked_file(Path(intent["log"]), intent["log_sha256"])
                        current.append(f"Saved controller log SHA256 `{intent['log_sha256']}`.")
                    if state:
                        if (
                            intent["status"] != "acknowledged"
                            or state["binding_sha256"] != digest
                            or state["controller_pid"] != intent["pid"]
                            or state["controller_start_ticks"] != intent["start_ticks"]
                            or state["candidate"] != str(candidate)
                            or state["node"] != ATTRIBUTION_VALIDATION_NODE
                            or state["allocation_predecessor"] != predecessor
                        ):
                            raise ValueError("Saved attribution execution owner differs")
                        current.append(
                            f"Saved attribution state `{state['status']}` at {local_time(state.get('time'))}; "
                            f"spawn status `{state.get('spawn_status', 'unrecorded')}`."
                        )
                        if state.get("spawn_status") == "acknowledged":
                            if (
                                state["command"] != attribution_validation_validator_command(root, candidate)
                                or type(state["child_pid"]) is not int
                                or not str(state["child_start_ticks"]).isdigit()
                                or not state["child_marker"]
                                or state["log"] != str(helpers.parent / "validator.log")
                            ):
                                raise ValueError("Saved attribution validator spawn identity differs")
                            current.append(
                                f"Saved validator owner `{state['child_pid']}/{state['child_start_ticks']}`; "
                                f"log `{state['log']}`."
                            )
                        if "log_sha256" in state:
                            if state["log"] != str(helpers.parent / "validator.log"):
                                raise ValueError("Saved attribution validator log path differs")
                            checked_file(Path(state["log"]), state["log_sha256"])
                            current.append(f"Saved validator log SHA256 `{state['log_sha256']}`.")
                        if "active_command" in state:
                            if state["active_command"] not in frozen["commands"]:
                                raise ValueError("Saved attribution active command is outside its frozen plan")
                            current.append(f"Saved active command: `{json.dumps(state['active_command'])}`.")
                        for key in ("error", "cleanup_error", "keeper_restore"):
                            if key in state:
                                current.append(f"Preserved attribution {key}: `{state[key]}`.")
                    elif receipt:
                        raise ValueError("Saved attribution GPU receipt has no execution state")
                    if receipt:
                        commands = [
                            (trial["name"], mode, command)
                            for trial in planned["trials"]
                            for mode, command in zip(ATTRIBUTION_VALIDATION_MODES, trial["commands"], strict=True)
                        ]
                        executed, trials = receipt["gpu_commands"], receipt["trials"]
                        if (
                            state.get("spawn_status") != "acknowledged"
                            or receipt["status"]
                            not in ("validating", "needs_attention", ATTRIBUTION_VALIDATION_PASS_STATUS)
                            or receipt["owner"]
                            != {
                                "controller_pid": state["controller_pid"],
                                "controller_start_ticks": state["controller_start_ticks"],
                            }
                            or receipt["execution_binding_sha256"] != digest
                            or receipt["candidate_preparation_sha256"] != frozen["preparation_sha256"]
                            or any(
                                receipt[key] != value
                                for key, value in planned.items()
                                if key not in ("status", "trials", "gpu_commands")
                            )
                            or len(trials) != len(planned["trials"])
                            or len(executed) > len(commands)
                            or any(
                                any(actual[key] != value for key, value in expected.items() if key != "checks")
                                or set(actual["checks"]) != set(expected["checks"])
                                or any(type(value) is not bool for value in actual["checks"].values())
                                for actual, expected in zip(trials, planned["trials"], strict=True)
                            )
                        ):
                            raise ValueError("Saved attribution GPU plan, owner or receipt binding differs")
                        for entry, (trial, mode, command) in zip(executed, commands, strict=False):
                            if (
                                entry["trial"] != trial
                                or entry["mode"] != mode
                                or entry["command"] != command
                                or type(entry["exit_code"]) is not int
                                or entry["log"] != str(work / f"{trial}_{mode}.log")
                            ):
                                raise ValueError("Saved attribution completed command prefix differs")
                            checked_file(Path(entry["log"]), entry["log_sha256"])
                            current.append(f"Saved command `{trial}/{mode}` log SHA256 `{entry['log_sha256']}`.")
                        active_indices = {max(0, len(executed) - 1), min(len(executed), len(commands) - 1)}
                        if "active_command" in state and state["active_command"] not in [
                            commands[index][2] for index in active_indices
                        ]:
                            raise ValueError("Saved attribution active command differs from its completed prefix")
                        successful = sum(entry["exit_code"] == 0 for entry in executed)
                        current.append(
                            f"Source-bound saved attribution GPU commands: {successful} / {len(commands)} "
                            f"exited zero; {len(executed) - successful} failed at "
                            f"{local_time(receipt.get('time'))}; `{gpu_path}`; "
                            f"SHA256 `{hashlib.sha256(gpu_path.read_bytes()).hexdigest()}`."
                        )
                        current.extend(
                            [
                                f"Saved attribution GPU receipt status `{receipt['status']}`.",
                                "Partial progress counts exited-zero commands only. "
                                "Check flags below are saved reports, "
                                "not independent numerical acceptance.",
                                "",
                                "| Actual Config Trial | Saved Check Flags |",
                                "| --- | --- |",
                            ]
                        )
                        for trial in trials:
                            flags = ", ".join(f"{key}={str(value).lower()}" for key, value in trial["checks"].items())
                            current.append(f"| `{trial['name']}` | {flags} |")
                        if receipt.get("error"):
                            current.append(f"Preserved attribution GPU error: `{receipt['error']}`.")
                        if state["status"] == "passed":
                            if (
                                receipt["status"] != ATTRIBUTION_VALIDATION_PASS_STATUS
                                or successful != len(commands)
                                or not all(all(trial["checks"].values()) for trial in trials)
                                or state["validation_sha256"] != hashlib.sha256(gpu_path.read_bytes()).hexdigest()
                                or any(
                                    type(state.get(key)) is not int or state[key] != 0
                                    for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
                                )
                                or state.get("marked_descendants_absent") is not True
                                or "error" in state
                                or "cleanup_error" in state
                                or "error" in receipt
                            ):
                                raise ValueError("Saved attribution terminal receipt or cleanup/restoration differs")
                            current.append(
                                "Saved controller records reconstructed attribution validation gates and "
                                "successful cleanup/restoration; this renderer does not repeat tensor checks."
                            )
                    else:
                        current.append("No saved attribution GPU command receipt; no numerical acceptance.")
            else:
                current.append("No canonical execution binding for this candidate; no GPU progress published.")
            if (
                any(
                    source != source.resolve()
                    or source.is_symlink()
                    or not source.is_file()
                    or hashlib.sha256(source.read_bytes()).hexdigest() != digest
                    for source, digest in identities
                )
                or candidate_inventory(candidate) != files
                or candidate_inventory(previous) != parent_cpu["files_sha256"]
                or failed_profiler_evidence(root) != predecessor
            ):
                raise ValueError("Saved attribution evidence changed during rendering")
            current.extend(
                [
                    "Saved evidence does not inspect live ranks, authorize production or establish "
                    "full-budget optimizer quality, timing or cloud results.",
                    "",
                ]
            )
        except (ValueError, KeyError, TypeError, OSError, tarfile.TarError) as error:
            current = [
                *heading,
                f"Saved attribution validation evidence differs or is incomplete: `{type(error).__name__}: {error}`.",
                "CPU acceptance and attribution progress counts withheld; "
                "original source/log/failure evidence preserved.",
                "",
            ]
        lines.extend(current)
    return lines


def selected_lr_stage_profile_reader_lines(root: Path) -> list[str]:
    """Report additive CPU reader receipts while preserving the original failed GPU status."""
    lines = []
    scope = "MuonT saved rank traces only; no tensor/trial reconstruction, global acceptance, or GPU grant."
    for path in sorted(root.glob("run-preparation/selected-lr-stage-profile-reader-cpu/*/receipt.json")):
        lines.extend(["### Saved MuonT Trace Reader", "", f"`PVC/{path.relative_to(root)}`."])
        try:
            receipt = read_record(path)
            directory = path.parent
            prepared_path = directory / "preparation.json"
            prepared = read_record(prepared_path)
            commands = [receipt[key] for key in ("frozen_binding_check", "cpu", "lint", "saved_trace_reconstruction")]
            inventory = receipt["archive_inventory_sha256"]
            failed_path = profile_locations(root)[4]
            evidence = receipt["rank_traces"]
            identities = [
                (prepared_path, receipt["preparation_sha256"]),
                (Path(receipt["archive"]), receipt["archive_sha256"]),
                *((Path(name), digest) for name, digest in receipt["inputs_sha256"].items()),
                *((directory / name, digest) for name, digest in inventory.items()),
            ]
            if (
                receipt["status"] != "selected_lr_stage_profile_muont_reader_cpu_verified_gpu_denied"
                or receipt["root"] != str(root)
                or receipt["out"] != str(directory)
                or receipt["scope"] != scope
                or prepared["scope"] != scope
                or receipt["helpers_sha256"] != prepared["helpers_sha256"]
                or receipt["inputs_sha256"] != prepared["inputs_sha256"]
                or receipt["failed_receipt_status_preserved"] != "needs_attention"
                or read_record(failed_path)["status"] != "needs_attention"
                or str(failed_path) not in receipt["inputs_sha256"]
                or any(
                    type(command["exit_code"]) is not int
                    or command["exit_code"] != 0
                    or command["cuda_visible_devices"] != ""
                    or hashlib.sha256(Path(command["log"]).read_bytes()).hexdigest() != command["log_sha256"]
                    for command in commands
                )
                or any(hashlib.sha256(source.read_bytes()).hexdigest() != digest for source, digest in identities)
                or any(inventory.get(f"tracking/{name}") != digest for name, digest in receipt["helpers_sha256"].items())
                or read_record(directory / "saved-trace-evidence.json") != evidence
                or set(evidence) != {str(rank) for rank in range(8)}
                or any(
                    any(
                        type(item[key]) is not int or item[key] != 20
                        for key in (
                            "training_update_events",
                            "optimizer_step_events",
                            "cuda_training_update_events",
                            "cuda_optimizer_step_events",
                        )
                    )
                    or type(item["device_kernel_events"]) is not int
                    or item["device_kernel_events"] <= 0
                    for item in evidence.values()
                )
            ):
                raise ValueError("Saved reader source, protected input, command or trace evidence differs")
            require_selected_lr_archive(Path(receipt["archive"]), inventory, alias=False)
            lines.extend(
                [
                    f"Archive/input-bound CPU reader accepted at {local_time(receipt.get('time'))}; "
                    f"{len(receipt['helpers_sha256'])} issued helpers.",
                    "Eight saved MuonT ranks each reconstruct 20 CPU updates and 20 matched CUDA update "
                    "projections, with corresponding optimizer scopes and actual CUDA kernels.",
                    "Saved traces only: no checkpoint-tensor/trial reconstruction, global acceptance or GPU grant. "
                    "The original profiling receipt remains `needs_attention`.",
                    "",
                ]
            )
        except (ValueError, KeyError, TypeError, OSError, tarfile.TarError) as error:
            lines.extend(
                [
                    f"Saved reader evidence differs or is incomplete: `{type(error).__name__}: {error}`.",
                    "Reader acceptance/counts withheld; original failed evidence is preserved.",
                    "",
                ]
            )
    return lines


def selected_lr_stage_profile_lines(root: Path) -> list[str]:
    """Report saved bounded-profiler execution without reading tensors or making live node changes."""
    binding_path, helpers, state_path, work, receipt_path = profile_locations(root)
    if not binding_path.exists():
        return []
    lines = [
        "### Selected-LR Bounded Stage Profiling",
        "",
        f"Saved validation-only binding: `PVC/{binding_path.relative_to(root)}`.",
    ]
    try:
        frozen = read_record(binding_path)
        candidate = Path(frozen["candidate"])
        planned = profile_validation_plan(candidate, root, work)
        cpu, controller = read_record(Path(frozen["cpu_receipt"])), read_record(Path(frozen["controller_cpu_receipt"]))
        prepared = read_record(candidate / "selected-lr-stage-profile-validation-preparation.json")
        commands = [
            (trial["arm"], mode, command)
            for trial in planned["trials"]
            for mode, command in zip(("continuous", "off_resumed", "on_resumed"), trial["commands"], strict=True)
        ]
        identities = [
            (Path(frozen["archive"]), frozen["archive_sha256"]),
            (Path(frozen["cpu_receipt"]), frozen["cpu_receipt_sha256"]),
            (Path(frozen["controller_cpu_receipt"]), frozen["controller_cpu_receipt_sha256"]),
            (candidate / "selected-lr-stage-profile-validation-preparation.json", frozen["preparation_sha256"]),
            (root / "data/experiment_receipt.json", frozen["data_receipt_sha256"]),
            (root / "run-preparation/wave12-launch-binding.json", frozen["predecessor_release_sha256"]),
            *((helpers / name, digest) for name, digest in frozen["helpers_sha256"].items()),
            *((candidate / name, digest) for name, digest in prepared["parent_metadata_sha256"].items()),
        ]
        if (
            frozen["status"] != PROFILE_FROZEN_STATUS
            or frozen["root"] != str(root)
            or frozen["candidate"] != str(candidate.resolve())
            or frozen["helpers"] != str(helpers)
            or frozen["work"] != str(work)
            or frozen["receipt"] != str(receipt_path)
            or frozen["environment"] != planned["environment"]
            or frozen["plan"] != planned
            or frozen["commands"] != [command for _, _, command in commands]
            or frozen["files_sha256"] != candidate_inventory(candidate)
            or cpu["status"] != "selected_lr_stage_profile_validation_cpu_verified_no_gpu_binding"
            or cpu["files_sha256"] != frozen["files_sha256"]
            or controller["status"] != "selected_lr_stage_profile_controller_cpu_verified_no_gpu_binding"
            or controller["helpers_sha256"] != frozen["helpers_sha256"]
            or any(
                type(item["exit_code"]) is not int or item["exit_code"] != 0 or item["cuda_visible_devices"] != ""
                for item in (cpu, controller)
            )
            or any(frozen["helpers_sha256"].get(name) != digest for name, digest in cpu["helpers_sha256"].items())
            or any(Path(name).name != name or (helpers / name).is_symlink() for name in frozen["helpers_sha256"])
            or any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path, digest in identities)
        ):
            raise ValueError("Saved profiler source, actual schedule or CPU/controller binding differs")
        require_selected_lr_archive(
            Path(frozen["archive"]),
            frozen["files_sha256"]
            | prepared["parent_metadata_sha256"]
            | {
                "selected-lr-stage-profile-validation-preparation.json": frozen["preparation_sha256"],
                "candidate-cpu-receipt.json": frozen["cpu_receipt_sha256"],
                "controller-cpu-receipt.json": frozen["controller_cpu_receipt_sha256"],
            }
            | {f"tracking/{name}": digest for name, digest in frozen["helpers_sha256"].items()},
            alias=True,
        )
        lines.extend(
            [
                f"Source-bound {len(frozen['files_sha256'])}-file/{len(frozen['helpers_sha256'])}-helper protocol; "
                f"{len(commands)} eight-rank commands; {planned['planned_executed_tokens']:,} tokens planned.",
                "Continuous to update 220; private checkpoint-200 resumes off/on to 220; CUDA traces cover 201-220.",
                "The full 5,120-update schedule and compilation/graphs remain enabled. Resumed traces include "
                "graph rewarming; neither validation loss nor profiled time is a full-budget optimizer result.",
            ]
        )
        intent, state, receipt = (
            read_record(helpers.parent / "controller.json"),
            read_record(state_path),
            read_record(receipt_path),
        )
        if not intent:
            if state or receipt:
                raise ValueError("Saved profiling execution has no controller intent")
            return [*lines, "No saved profiler controller intent or GPU execution.", ""]
        digest = hashlib.sha256(binding_path.read_bytes()).hexdigest()
        if intent["binding_sha256"] != digest or intent["command"] != profile_controller_command(root):
            raise ValueError("Saved profiler controller intent differs")
        lines.append(
            f"Saved controller intent `{intent['status']}`; PID `{intent.get('pid', 'unacknowledged')}`. "
            "This is receipt-backed status, not fresh process liveness."
        )
        if not state:
            if receipt:
                raise ValueError("Saved profiling GPU receipt has no execution state")
            return [*lines, "No saved profiler execution state or GPU result.", ""]
        if (
            intent["status"] != "acknowledged"
            or state["binding_sha256"] != digest
            or state["controller_pid"] != intent["pid"]
            or state["controller_start_ticks"] != intent["start_ticks"]
            or state["candidate"] != str(candidate)
            or state["node"] != frozen["node"]
        ):
            raise ValueError("Saved profiler controller generation differs")
        lines.append(f"Saved profiler controller state `{state['status']}` at {local_time(state.get('time'))}.")
        if "keeper_restore" in state:
            lines.append(f"Recorded keeper restoration `{state['keeper_restore']}`.")
        for key in ("error", "cleanup_error"):
            if key in state:
                lines.append(f"Preserved profiler controller {key}: `{state[key]}`.")
        if not receipt:
            return [*lines, "No saved GPU command result; no numerical or CUDA-trace acceptance.", ""]
        executed = receipt["gpu_commands"]
        if (
            state["command"] != profile_validator_command(root, candidate)
            or receipt["owner"]
            != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
            or any(
                receipt[key] != value
                for key, value in planned.items()
                if key not in ("status", "trials", "gpu_commands")
            )
            or len(executed) > len(commands)
            or any(
                item["arm"] != arm
                or item["mode"] != mode
                or item["command"] != command
                or item["log"] != str(work / f"{arm}_{mode}.log")
                or type(item["exit_code"]) is not int
                or hashlib.sha256(Path(item["log"]).read_bytes()).hexdigest() != item["log_sha256"]
                for item, (arm, mode, command) in zip(executed, commands, strict=False)
            )
        ):
            raise ValueError("Saved profiler command, owner, plan or completed log identity differs")
        passed = sum(item["exit_code"] == 0 for item in executed)
        lines.append(
            f"Source-bound saved GPU commands: {passed} / {len(commands)} exited zero; "
            f"{len(executed) - passed} failed. Receipt status `{receipt['status']}`."
        )
        if receipt.get("error"):
            lines.append(f"Preserved profiler GPU failure: `{receipt['error']}`.")
        if state["status"] == "passed":
            if (
                receipt["status"] != "passed_selected_lr_stage_profile_continuation_and_trace_gates"
                or passed != len(commands)
                or state["validation_sha256"] != hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                or any(
                    type(state.get(key)) is not int or state[key] != 0
                    for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
                )
                or "error" in state
                or "cleanup_error" in state
            ):
                raise ValueError("Saved profiler terminal receipt or cleanup/restoration differs")
            lines.append("Saved controller records reconstructed GPU parity/traces and successful cleanup/restoration.")
        lines.append(
            "This saved progress does not independently reconstruct tensors, inspect live ranks or grant a release."
        )
        lines.append("")
    except (ValueError, KeyError, TypeError, OSError, tarfile.TarError) as error:
        lines = [line for line in lines if not line.startswith("Source-bound saved GPU commands:")]
        lines.extend(
            [
                f"Saved profiler evidence differs or is incomplete: `{type(error).__name__}: {error}`.",
                "Original source/log/failure evidence is preserved; profiler progress counts are withheld.",
                "",
            ]
        )
    return lines


def selected_lr_production_cpu_lines(root: Path) -> list[str]:
    """Report source-bound denied and release-owner CPU outcomes without changing GPU ownership."""
    lines = []
    kinds = (
        (
            "selected-lr-production-cpu",
            "selected-lr-production-preparation.json",
            "production",
            SELECTED_LR_DENIED_CHANGES,
            "wave11",
        ),
        (
            "selected-lr-release-cpu",
            "selected-lr-release-preparation.json",
            "release",
            SELECTED_LR_RELEASE_CHANGES,
            "wave11",
        ),
        (
            "selected-lr-recovery-cpu",
            "selected-lr-release-preparation.json",
            "release",
            SELECTED_LR_RELEASE_CHANGES,
            "wave12",
        ),
    )
    binding_path = selected_lr_locations(root)[0]
    if not binding_path.exists():
        return lines
    frozen = read_record(binding_path)
    for folder, preparation_name, label, expected_changes, wave in kinds:
        for receipt_path in sorted(root.glob(f"run-preparation/{folder}/*/receipt.json")):
            lines.extend(
                [
                    f"### Wave-{wave.removeprefix('wave')} Selected-LR Production CPU",
                    "",
                    f"Saved receipt: `PVC/{receipt_path.relative_to(root)}`.",
                ]
            )
            try:
                receipt = read_record(receipt_path)
                candidate = Path(receipt["candidate"])
                prepared = read_record(candidate / preparation_name)
                previous = Path(frozen["candidate"])
                files = candidate_inventory(candidate)
                changed = {
                    name
                    for name in files.keys() | frozen["files_sha256"].keys()
                    if files.get(name) != frozen["files_sha256"].get(name)
                }
                success = (
                    "selected_lr_production_cpu_verified_gpu_denied"
                    if label == "production"
                    else "selected_lr_release_cpu_verified_gpu_pending"
                )
                failed = (
                    "selected_lr_production_cpu_failed" if label == "production" else "selected_lr_release_cpu_failed"
                )
                if (
                    prepared["root"] != str(root)
                    or prepared["candidate"] != str(candidate.resolve())
                    or prepared["parent_candidate"] != str(previous)
                    or prepared["files_sha256"] != files
                    or changed != set(prepared["changed_files"])
                    or changed != expected_changes
                    or prepared["status"]
                    != (
                        "selected_lr_production_prepared_gpu_denied"
                        if label == "production"
                        else "prepared_release_owner_gpu_pending"
                    )
                    or prepared["validation_binding_sha256"] != hashlib.sha256(binding_path.read_bytes()).hexdigest()
                    or candidate_inventory(previous) != frozen["files_sha256"]
                    or receipt["root"] != str(root)
                    or receipt["files_sha256"] != files
                    or receipt["helpers_sha256"] != prepared["helpers_sha256"]
                    or receipt["validation_binding_sha256"] != prepared["validation_binding_sha256"]
                    or receipt["preparation_sha256"]
                    != hashlib.sha256((candidate / preparation_name).read_bytes()).hexdigest()
                    or receipt["command"] != cpu_command()
                    or receipt["cwd"] != str(candidate)
                    or receipt["cuda_visible_devices"] != ""
                    or type(receipt["exit_code"]) is not int
                    or receipt["status"] not in (success, failed)
                    or (receipt["status"] == success and (receipt["exit_code"] != 0 or "error" in receipt))
                    or receipt["log_sha256"] != hashlib.sha256(Path(receipt["log"]).read_bytes()).hexdigest()
                    or receipt["archive_sha256"] != hashlib.sha256(Path(receipt["archive"]).read_bytes()).hexdigest()
                ):
                    raise ValueError(f"Saved {wave} production CPU/source/log identity differs")
                require_selected_lr_archive(
                    Path(receipt["archive"]),
                    files
                    | {preparation_name: receipt["preparation_sha256"]}
                    | {f"tracking/{name}": digest for name, digest in prepared["helpers_sha256"].items()},
                    alias=True,
                )
                lines.extend(
                    [
                        f"Source-bound saved CPU status `{receipt['status']}`; integer exit `{receipt['exit_code']}`.",
                        f"{len(files)} files and {len(changed)} ownership-only changes; original log/archive verified.",
                        "CPU evidence grants no GPU allocation or production launch. "
                        "Full-budget LR-transfer quality remains unmeasured.",
                        "",
                    ]
                )
            except (ValueError, KeyError, TypeError, OSError, tarfile.TarError) as error:
                lines.extend(
                    [
                        f"Saved CPU/source/archive evidence differs: `{type(error).__name__}: {error}`.",
                        "Outcome counts withheld; original receipts remain preserved.",
                        "",
                    ]
                )
    return lines


def selected_lr_release_lines(root: Path, wave: str = "wave11") -> tuple[list[str], str | None]:
    """Use frozen read-only history helpers to bind saved production status."""
    path = root / f"run-preparation/{wave}-launch-binding.json"
    if not path.exists():
        return [], None
    lines = [
        f"### Wave-{wave.removeprefix('wave')} Selected-LR Production Release",
        "",
        f"Saved release: `PVC/{path.relative_to(root)}`.",
    ]
    try:
        release, binding_digest = verify_production_binding(root, wave)
        status = release["status"]
        launch = read_record(root / f"run-preparation/{wave}-launch-state.json")
        lines.append(f"Source-bound saved release `{status}`; {len(release['files_sha256'])} candidate files.")
        if launch:
            if launch["binding_sha256"] != binding_digest:
                raise ValueError(f"Saved {wave} launch belongs to a different release")
            lines.append(f"Saved original launch `{launch['status']}`; {len(launch['workers'])} worker acknowledgments.")
        else:
            lines.append("No saved launch intent or worker acknowledgment.")
        lines.extend(["Saved receipts do not establish fresh process liveness or completed full-budget quality.", ""])
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding_digest:
            raise ValueError(f"{wave} verified release bytes changed during publication")
        return lines, binding_digest
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        lines.extend(
            [
                f"Saved production evidence differs: `{type(error).__name__}: {error}`.",
                "Release/launch counts withheld.",
                "",
            ]
        )
    return lines, None


def production_capture_lines(root: Path, wave: str) -> list[str]:
    """Select receipt-time freshness and rederive only correctly labelled, release-bound captures."""
    preparation = root / "run-preparation"
    lines = []
    title = f"Wave-{wave.removeprefix('wave')}"
    try:
        observations = sorted(
            (preparation / f"{wave}-observations").glob("*/observation.json"),
            key=lambda path: datetime.fromisoformat(read_record(path)["time"]),
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        return [f"### {title} Strict Captured Comparison", "", f"Saved capture discovery refused: `{error}`.", ""]
    for observation_path in observations[-1:]:
        lines.extend(
            [
                f"### {title} Strict Captured Comparison",
                "",
                f"`PVC/{observation_path.relative_to(root)}`.",
            ]
        )
        try:
            record = read_record(observation_path)
            release_path = preparation / f"{wave}-launch-binding.json"
            if (
                record.get("status") != f"captured_strict_{wave}_observation"
                or not production_release_matches(root, read_record(release_path), wave)
                or record.get("source_release_sha256") != hashlib.sha256(release_path.read_bytes()).hexdigest()
            ):
                raise ValueError(f"Captured {title} production release identity differs")
            _, binding_digest = verify_production_binding(root, wave)
            if record["source_release_sha256"] != binding_digest:
                raise ValueError(f"Captured {title} verified release bytes differ")
            actual = analyze_wave9_capture(observation_path.parent)
            if actual != read_record(observation_path.parent / "analysis.json"):
                raise ValueError(f"Captured {title} analysis differs from immutable histories")
            cohort = captured_cohort(observation_path.parent)
            baseline = cohort["comparisons"][cohort["baseline"]]
            endpoints = [window["end_tokens"] for window in baseline["training_windows"]]
            cohort_lines = [
                f"All-four common observed prefix: {cohort['common_tokens']:,} tokens; "
                f"{len(endpoints)} complete {cohort['window_tokens']:,}-token windows.",
                "Original captured updates: "
                + ", ".join(f"{arm} {updates:,}" for arm, updates in cohort["captured_observed_updates"].items())
                + ".",
                "The following cohort summaries use the same four-arm prefix, exact evaluation token sets "
                "and training-step timer endpoint. This capture does not establish final losses or efficiency.",
                "",
            ]
            for key, tokens in cohort["common_evaluations"].items():
                labels = ", ".join(f"{token:,}" for token in tokens["late_tokens"]) or "none"
                cohort_lines.append(f"Cohort late `{key}` tokens: {labels}.")
            cohort_lines.extend(["", *comparison_lines({"analyses": [cohort]})])
            lines.extend(
                [
                    f"Saved observation at {local_time(record.get('time'))}; all bytes parsed without omitted rows.",
                    "This historical running-prefix capture does not establish final results or cloud verification.",
                    "",
                    *cohort_lines,
                    "The pairwise summaries below can cover different progress and late evaluation sets.",
                    "",
                    *comparison_lines(actual),
                ]
            )
        except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
            lines.extend([f"Saved {title} capture refused: `{type(error).__name__}: {error}`.", ""])
    return lines


def render(root: Path, now: datetime) -> str:
    preparation = root / "run-preparation"
    lines = [
        START,
        "## Automated Wave Ledger",
        "",
        f"Refreshed {local_time(now.isoformat())}, Asia/Shanghai, from saved artifacts.",
        "Observation and cloud-verification times below are the receipt times, not new live checks.",
        "Loss deltas are candidate minus reference; negative is better. Late evaluation means",
        "cover the final half of pairwise common observed tokens, not a completed budget",
        "when interim. These are exploratory comparisons, not tuned superiority or",
        "independent-replication claims. Historical narrative above remains unchanged.",
        "",
    ]
    lines.extend(upstream_cpu_lines(root))
    lines.extend(upstream_full_model_cpu_lines(root))
    lines.extend(owned_output_validation_cpu_lines(root))
    lines.extend(owned_output_handoff_cpu_lines(root))
    lines.extend(owned_output_handoff_cpu_lines(root, "owned-output-trace"))
    for identity in OWNED_VALIDATORS:
        lines.extend(owned_output_validation_lines(root, identity))
    for binding_path in sorted(preparation.glob("upstream-outer-gpu-*/binding.json")):
        lines.extend(upstream_gpu_lines(root, binding_path))
        lines.extend(upstream_queue_lines(root, binding_path.parent))
    for path in sorted(preparation.glob("upstream-full-model*-queue/binding.json")):
        lines.extend(upstream_full_model_queue_lines(root, path))
    lines.extend(production_lines(root))
    lines.extend(production_lines(root, "wave10"))
    lines.extend(selected_lr_cpu_lines(root))
    lines.extend(selected_lr_validation_lines(root))
    lines.extend(selected_lr_stage_profile_lines(root))
    lines.extend(selected_lr_stage_profile_reader_lines(root))
    lines.extend(fused_attribution_lines(root))
    lines.extend(fused_attribution_validation_lines(root))
    lines.extend(fused_continuation_lines(root))
    lines.extend(fused_continuation_controller_lines(root))
    lines.extend(saved_ns_replay_cpu_lines(root))
    lines.extend(root_preparation_cpu_lines(root))
    lines.extend(root_executor_lines(root, local_time))
    lines.extend(selected_lr_production_cpu_lines(root))
    selected_lr_bound = {}
    for wave in ("wave11", "wave12"):
        release_lines, selected_lr_bound[wave] = selected_lr_release_lines(root, wave)
        lines.extend(release_lines)
    for wave in ("wave10", "wave12"):
        if wave != "wave12" or selected_lr_bound[wave]:
            lines.extend(production_capture_lines(root, wave))
    receipts = sorted(
        path for path in preparation.glob("wave*.json") if re.fullmatch(r"wave\d+(?:-baseline)?", path.stem)
    )
    for path in receipts:
        wave = path.stem
        launch = read_record(path)
        snapshot = read_record(preparation / f"monitor-{wave}/latest.json")
        upload = read_record(preparation / f"swanlab-upload-{wave}.json")
        production_wave = wave in ("wave9", "wave10", "wave11", "wave12")
        title = f"Wave-{wave.removeprefix('wave')}"
        release_path = preparation / f"{wave}-launch-binding.json"
        release_matches = not production_wave or (
            production_release_matches(root, read_record(release_path), wave)
            and (
                wave not in selected_lr_bound
                or selected_lr_bound[wave] == hashlib.sha256(release_path.read_bytes()).hexdigest()
            )
        )
        release_digest = (
            selected_lr_bound[wave]
            if release_matches and wave in selected_lr_bound
            else hashlib.sha256(release_path.read_bytes()).hexdigest() if release_matches and production_wave else None
        )
        snapshot_withheld = (
            production_wave and snapshot and (not release_matches or snapshot.get("binding_sha256") != release_digest)
        )
        if snapshot_withheld:
            snapshot = {}
        state = snapshot.get("status", "no saved process observation")
        lines.extend(
            [
                f"### {wave}",
                "",
                f"Launch receipt: `PVC/run-preparation/{path.name}`; recorded {local_time(launch.get('time'))}.",
                f"Saved monitor state: `{state}` at {local_time(snapshot.get('time'))}.",
            ]
        )
        if snapshot_withheld:
            lines.append(f"Saved {title} monitor identity differs; progress and keeper summaries withheld.")
        if snapshot:
            lines.extend(["", "| Run | Node | State | Updates / Budget |", "| --- | --- | --- | ---: |"])
            for run in snapshot["runs"]:
                observed = (run.get("latest_metrics") or {}).get("step", "unknown")
                lines.append(
                    f"| `{run['name']}` | `{run['node']}` | {run['status']} | " f"{observed} / {run['planned_steps']} |"
                )
            for run in snapshot["runs"]:
                if "keeper_watcher" in run:
                    watcher = run["keeper_watcher"]
                    lines.append(f"Keeper watcher for `{run['name']}`: `{watcher['status']}`.")
        names = {Path(entry["out"]).name for entry in launch["launches"]}
        verified = (
            release_matches
            and bool(names)
            and upload.get("status") == "verified"
            and names == set(upload.get("runs", {}))
            and all(run["status"] == "verified" for run in upload["runs"].values())
        )
        if verified:
            lines.extend(
                [
                    "",
                    f"SwanLab receipt: all {len(names)} histories verified at {local_time(upload.get('verified_at'))};",
                    f"`PVC/run-preparation/swanlab-upload-{wave}.json`.",
                ]
            )
        else:
            lines.extend(["", "No complete-wave verified SwanLab receipt is recorded."])
        # Earlier waves already have detailed manually interpreted result sections.
        if int(re.match(r"wave(\d+)", wave).group(1)) >= 4:
            analysis = read_record(preparation / f"analysis-{wave}.json")
            if (
                analysis
                and production_wave
                and (not release_matches or analysis.get("source_release_sha256") != release_digest)
            ):
                lines.extend(["", f"Saved {title} analysis identity differs; loss and runtime summaries withheld."])
            elif analysis:
                modified = datetime.fromtimestamp((preparation / f"analysis-{wave}.json").stat().st_mtime, UTC)
                lines.extend(
                    [
                        "",
                        f"Analysis artifact modified {local_time(modified.isoformat())}:",
                        f"`PVC/run-preparation/analysis-{wave}.json`.",
                        "",
                        *comparison_lines(analysis),
                    ]
                )
        lines.append("")
    observations = sorted((preparation / "wave7-observations").glob("*/observation.json"))
    if observations:
        observation_path = observations[-1]
        observation = read_record(observation_path)
        analysis_path = observation_path.parent / "analysis.json"
        analysis = read_record(analysis_path)
        release_path = preparation / "wave7-release-binding.json"
        lines.extend(
            [
                "### Wave-7 Strict Captured Comparison",
                "",
                f"Saved observation at {local_time(observation.get('time'))};",
                f"`PVC/{observation_path.relative_to(root)}`.",
                "Immutable running-history comparison only; not completed-wave results or a cloud import.",
            ]
        )
        matches = (
            observation.get("status") == "captured_strict_wave7_observation"
            and release_path.is_file()
            and observation.get("source_release_sha256") == hashlib.sha256(release_path.read_bytes()).hexdigest()
            and set(observation.get("runs", {}))
            == {"muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused"}
        )
        if matches and analysis:
            try:
                matches = analysis == analyze_capture(observation_path.parent)
            except (ValueError, KeyError, FileNotFoundError):
                matches = False
        else:
            matches = False
        if matches:
            lines.extend(
                [
                    f"Reconstructed comparison: `PVC/{analysis_path.relative_to(root)}`.",
                    "",
                    "| Arm | Captured Updates |",
                    "| --- | ---: |",
                ]
            )
            for arm, run in observation["runs"].items():
                lines.append(f"| `{arm}` | {run['observed_updates']} |")
            lines.extend(["", *comparison_lines(analysis)])
        else:
            lines.extend(
                [
                    "Saved observation source, history or comparison identity differs or is incomplete; "
                    "captured result summaries withheld.",
                    "",
                ]
            )
    wave8_observations = sorted((preparation / "wave8-observations").glob("*/observation.json"))
    if wave8_observations:
        observation_path = wave8_observations[-1]
        observation = read_record(observation_path)
        analysis_path = observation_path.parent / "analysis.json"
        analysis = read_record(analysis_path)
        release_path = preparation / "wave8-launch-binding.json"
        lines.extend(
            [
                "### Wave-8 Strict Captured Comparison",
                "",
                f"Saved observation at {local_time(observation.get('time'))}; "
                f"`PVC/{observation_path.relative_to(root)}`.",
                "Immutable running-history comparison, not completed-wave results or cloud verification.",
            ]
        )
        matches = (
            observation.get("status") == "captured_strict_wave8_observation"
            and release_path.is_file()
            and observation.get("source_release_sha256") == hashlib.sha256(release_path.read_bytes()).hexdigest()
        )
        if matches and analysis:
            try:
                matches = analysis == analyze_wave8_capture(observation_path.parent)
            except (ValueError, KeyError, FileNotFoundError):
                matches = False
        else:
            matches = False
        if matches:
            lines.extend(
                [
                    f"Reconstructed comparison: `PVC/{analysis_path.relative_to(root)}`.",
                    "",
                    "| Arm | Captured Updates |",
                    "| --- | ---: |",
                ]
            )
            for arm, run in observation["runs"].items():
                lines.append(f"| `{arm}` | {run['observed_updates']} |")
            lines.extend(["", *comparison_lines(analysis)])
        else:
            lines.append("Saved Wave-8 source, history or comparison identity differs; captured results withheld.")
        lines.append("")
    try:
        wave9_observations = sorted(
            (preparation / "wave9-observations").glob("*/observation.json"),
            key=lambda path: datetime.fromisoformat(read_record(path)["time"]),
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        wave9_observations = []
        lines.extend(["### Wave-9 Strict Captured Comparison", "", f"Saved capture discovery refused: `{error}`.", ""])
    if wave9_observations:
        observation_path = wave9_observations[-1]
        observation = read_record(observation_path)
        analysis_path = observation_path.parent / "analysis.json"
        analysis = read_record(analysis_path)
        release_path = preparation / "wave9-launch-binding.json"
        lines.extend(
            [
                "### Wave-9 Strict Captured Comparison",
                "",
                f"Saved observation at {local_time(observation.get('time'))}; "
                f"`PVC/{observation_path.relative_to(root)}`.",
                "Immutable running-history comparison, not completed-wave results or cloud verification.",
            ]
        )
        matches = (
            release_path.is_file()
            and production_release_matches(root, read_record(release_path))
            and observation.get("status") == "captured_strict_wave9_observation"
            and observation.get("source_release_sha256") == hashlib.sha256(release_path.read_bytes()).hexdigest()
        )
        if matches and analysis:
            try:
                matches = analysis == analyze_wave9_capture(observation_path.parent)
            except (ValueError, KeyError, FileNotFoundError):
                matches = False
        else:
            matches = False
        if matches:
            lines.extend(
                [
                    f"Reconstructed comparison: `PVC/{analysis_path.relative_to(root)}`.",
                    "",
                    "| Arm | Captured Updates |",
                    "| --- | ---: |",
                ]
            )
            for arm, run in observation["runs"].items():
                lines.append(f"| `{arm}` | {run['observed_updates']} |")
            lines.extend(["", *comparison_lines(analysis)])
        else:
            lines.append("Saved Wave-9 source, history or comparison identity differs; captured results withheld.")
        lines.append("")
    for title, filename in (
        ("Seed-2 Companion", "seed2-companion-state.json"),
        ("Wave-5 Candidate", "wave5-candidate.json"),
        ("Wave-5 Validation", "wave5-validation-state.json"),
        ("Wave-5 Launch Queue", "wave5-launch-queue.json"),
        ("Wave-5 Launch", "wave5-launch-state.json"),
        ("Wave-6 Candidate", "wave6-candidate.json"),
        (
            "Wave-6 Preserved Preflight Failure",
            "wave6-validation-preflight-missing-keeper-20261116T133349Z.json",
        ),
        ("Wave-6 Validation", "wave6-validation-state.json"),
        ("Wave-6 Launch", "wave6-launch-state.json"),
        ("Wave-7 Candidate", "wave7-candidate.json"),
        ("Wave-7 Logging Release", "wave7-release-binding.json"),
        ("Wave-7 Validation", "wave7-validation-state.json"),
        ("Wave-7 Launch", "wave7-launch-state.json"),
        ("Wave-8 Candidate", "wave8-candidate.json"),
        ("Wave-8 Validation", "wave8-validation-state.json"),
        ("Wave-8 Full-Budget Release", "wave8-launch-binding.json"),
        ("Wave-8 Launch", "wave8-launch-state.json"),
    ):
        record = read_record(preparation / filename)
        if record:
            label = (
                "Binding state"
                if filename
                in (
                    "wave5-candidate.json",
                    "wave5-launch-queue.json",
                    "wave6-candidate.json",
                    "wave7-candidate.json",
                    "wave7-release-binding.json",
                    "wave8-candidate.json",
                    "wave8-launch-binding.json",
                )
                else "Receipt state"
            )
            lines.extend(
                [
                    f"### {title}",
                    "",
                    f"{label}: `{record['status']}` at {local_time(record.get('time'))};",
                    f"`PVC/run-preparation/{filename}`.",
                    "",
                ]
            )
            if filename == "wave6-candidate.json":
                lines.append(
                    "New outer-only geometry binding; not approval of failed inner-fused Wave 5 "
                    "and not evidence of GPU validation or training launch."
                )
                lines.append("")
            if filename == "wave7-candidate.json":
                lines.extend(
                    [
                        "Fixed-age host-refresh scheduling control; not approval of failed Waves 5/6.",
                        f"Bound installation age: `{record['refresh_install_age']}` optimizer update.",
                        "",
                    ]
                )
            if (
                filename in ("wave6-validation-state.json", "wave7-validation-state.json", "wave8-validation-state.json")
                and "keeper_restore" in record
            ):
                lines.append(f"Recorded keeper restoration: `{record['keeper_restore']}`.")
                lines.append("")
            if filename in ("wave6-validation-state.json", "wave7-validation-state.json", "wave8-validation-state.json"):
                wave = filename.split("-")[0]
                validation = read_record(preparation / f"validation-{wave}-gpu.json")
                required = read_record(preparation / f"{wave}-candidate.json").get("required_gates", [])
                if validation and required:
                    passed = [name for name in required if validation.get("checks", {}).get(name) is True]
                    pending = [name for name in required if name not in passed]
                    lines.extend(
                        [
                            f"Saved execution receipt: {len(passed)} / {len(required)} required gates passed;",
                            f"`PVC/run-preparation/validation-{wave}-gpu.json`.",
                            "Pending or failed gates: " + (", ".join(f"`{name}`" for name in pending) or "none") + ".",
                        ]
                    )
                    commands = validation.get("evidence", {}).get("gpu_commands", [])
                    failed = sum(command["exit_code"] != 0 for command in commands)
                    lines.append(f"Completed GPU command receipts: {len(commands)}; failed commands: {failed}.")
                    lines.append("Gate receipts are validation evidence, not full-budget optimizer performance.")
                    lines.append("")
            if (
                filename
                in (
                    "wave6-validation-state.json",
                    "wave6-validation-preflight-missing-keeper-20261116T133349Z.json",
                    "wave6-launch-state.json",
                    "wave7-validation-state.json",
                    "wave7-launch-state.json",
                    "wave8-validation-state.json",
                    "wave8-launch-state.json",
                )
                and "error" in record
            ):
                lines.extend([f"Recorded failure: `{record['error']}`.", ""])
            if filename == "wave8-candidate.json":
                lines.extend(
                    [
                        "Separate combined outer-diagonal arithmetic; not a whole-training speedup claim.",
                        "Full-model continuation, captured updates and distributed gates remain required before launch.",
                        "",
                    ]
                )
            if filename == "wave8-launch-binding.json":
                lines.extend(
                    [
                        "Separate launch/result helper binding; freezing does not create a training worker.",
                        "Original full schedules remain fixed; the short-LR campaign must complete, "
                        "be analyzed and be cloud-verified before full-budget launch.",
                        "",
                    ]
                )
            if filename == "wave5-launch-queue.json":
                log = preparation / "queue-wave5-launch.log"
                events = [
                    event
                    for row in (log.read_text().splitlines() if log.exists() else [])
                    if (event := re.fullmatch(r"(\S+) launch_status=(\w+)", row))
                ]
                if events:
                    event = events[-1]
                    lines.append(
                        f"Last execution event: `{event.group(2)}` at {local_time(event.group(1))}; "
                        "`PVC/run-preparation/queue-wave5-launch.log`."
                    )
                stopped = preparation / "queue-wave5-launch/stopped_at"
                if stopped.exists():
                    lines.append(
                        f"Queue exited at {local_time(stopped.read_text().strip())}; "
                        "the original binding is not evidence of a live queue."
                    )
                lines.append("")
    for path in sorted((root / "validation").glob("diagonal_diagnostic_*/state.json")):
        record = read_record(path)
        lines.extend(
            [
                f"### Direction Diagnostic: {path.parent.name}",
                "",
                f"Receipt state: `{record['status']}` at {local_time(record.get('time'))};",
                f"`PVC/{path.relative_to(root)}`.",
                "Synthetic direction diagnosis only; this is not a passed launch gate or training result.",
            ]
        )
        if "keeper_restore" in record:
            lines.append(f"Recorded keeper restoration: `{record['keeper_restore']}`.")
        result = read_record(path.parent / "result.json")
        identity_matches = (
            record.get("result_sha256") == hashlib.sha256((path.parent / "result.json").read_bytes()).hexdigest()
            if result
            else False
        )
        if result and identity_matches:
            lines.append(f"Result receipt: `PVC/{(path.parent / 'result.json').relative_to(root)}`.")
        elif result:
            lines.append("Result identity does not match the controller receipt; no measured result published here.")
        lines.append("")
    for path in sorted((root / "validation").glob("combined_outer_gpu_*/binding.json")):
        lines.extend(combined_probe_lines(root, path))
    checkpoint_path = preparation / "diagnostic-wave6-muont-continuation-20261116.json"
    checkpoint = read_record(checkpoint_path)
    if checkpoint:
        lines.extend(
            [
                "### Wave-6 CPU Continuation Diagnostic",
                "",
                f"Receipt state: `{checkpoint['status']}` at {local_time(checkpoint.get('time'))};",
                f"`PVC/run-preparation/{checkpoint_path.name}`.",
                "Saved-state CPU diagnosis only; not a passed GPU gate or causal intervention.",
            ]
        )
        identities_match = all(
            (preparation / name).is_file() and hashlib.sha256((preparation / name).read_bytes()).hexdigest() == expected
            for name, expected in checkpoint.get("validation_receipts_sha256", {}).items()
        )
        if checkpoint.get("validation_receipts_sha256") and identities_match:
            lines.extend(
                [
                    "",
                    "| Boundary | Tensor Comparisons | Changed Tensors | Failed Tensors | Scalar / Structure Diffs |",
                    "| ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for step in sorted({pair["step"] for pair in checkpoint["pairs"]}):
                pairs = [pair for pair in checkpoint["pairs"] if pair["step"] == step]
                count = sum(pair["tensor_comparisons"] for pair in pairs)
                changed = sum(pair["changed_tensor_count"] for pair in pairs)
                failed = sum(pair["failed_tensor_count"] for pair in pairs)
                scalars = sum(len(pair["scalar_or_structure_differences"]) for pair in pairs)
                lines.append(f"| {step} | {count} | {changed} | {failed} | {scalars} |")
        else:
            lines.append("Validation receipt identities differ; saved diagnostic conclusions are withheld.")
        lines.append("")
    refresh_path = preparation / "analysis-wave7-refresh.json"
    refresh = read_record(refresh_path)
    if refresh:
        lines.extend(
            [
                "### Wave-7 Recorded Refresh Diagnostic",
                "",
                f"Saved diagnostic: `{refresh['status']}` at {local_time(refresh.get('time'))};",
                f"`PVC/run-preparation/{refresh_path.name}`.",
                "Rank-zero scheduling telemetry only; not a passed launch gate, loss or efficiency result.",
            ]
        )
        release_path = preparation / "wave7-release-binding.json"
        identity_matches = (
            release_path.is_file()
            and refresh.get("source_release_sha256") == hashlib.sha256(release_path.read_bytes()).hexdigest()
        )
        if identity_matches:
            shared_windows = min(len(run["complete_shared_windows"]) for run in refresh["runs"].values())
            lines.extend(
                [
                    f"Common observed tokens: {refresh['common_observed_tokens']:,}; "
                    f"complete shared {refresh['window_tokens']:,}-token windows: {shared_windows}.",
                    "Counts below cover unequal all-observed prefixes; detailed shared windows remain in the receipt.",
                    "",
                    "| Arm | Observed Updates | Recorded Events | Installed | Discarded |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for arm, run in refresh["runs"].items():
                observed = run["all_observed"]
                lines.append(
                    f"| `{arm}` | {observed['updates']} | {observed['events']} | "
                    f"{observed['installed']} | {observed['discarded']} |"
                )
            lines.append(
                f"Validated recorded installation age: {refresh['refresh_install_age']} optimizer update. "
                "Missing telemetry is not inferred, and step timing cannot isolate host-refresh waits."
            )
        else:
            lines.append("Saved diagnostic does not match the source-release identity; measured summaries withheld.")
        lines.append("")
    for path in sorted((root / "campaigns").glob("*/campaign.json")):
        campaign = read_record(path)
        analysis_path = path.parent / "analysis.json"
        analysis = read_record(analysis_path)
        verified_identity = analysis.get("campaign_sha256") == hashlib.sha256(path.read_bytes()).hexdigest()
        fixed_age_study = campaign.get("study") in (
            "wave7_fixed_age_matrix_lr",
            "wave7_fixed_age_matrix_lr_upper_boundary",
            "wave7_fixed_age_matrix_lr_ceiling",
        )
        if verified_identity and fixed_age_study:
            verified_identity = tuning_trial_identities_match(campaign, analysis)
        lines.extend(
            [
                f"### Tuning: {path.parent.name}",
                "",
                f"Preparation state: `{campaign['status']}` at {local_time(campaign.get('time'))}.",
                f"Receipt: `PVC/{path.relative_to(root)}`.",
                "Prepared configurations and saved analysis do not prove GPU launch or cloud verification.",
            ]
        )
        if "recovery" in campaign:
            try:
                require_recovery(campaign)
            except (ValueError, KeyError, OSError) as error:
                lines.append(f"Recovery provenance differs; recovery/accounting summaries withheld: `{error}`.")
                verified_identity = False
            else:
                recovery = campaign["recovery"]
                lines.extend(
                    [
                        "Distinct recovery of a preserved pre-update CUDA-mask failure, not an extra LR point.",
                        f"Failed campaign: `PVC/campaigns/"
                        f"{Path(recovery['failed_campaign']).parent.name}/campaign.json`.",
                        f"Terminal failure evidence SHA256: `{recovery['terminal_observation_sha256']}`.",
                        f"Bound CUDA visibility: `{recovery['cuda_visible_devices']}`.",
                        "One failed validation command, zero validation updates/tokens and zero tuning attempts/tokens "
                        "per family; original search allowances and all eight config byte identities unchanged.",
                        "All 24 actual-config commands and fresh original-node cleanup/restoration remain required.",
                    ]
                )
        expanded_study = campaign.get("study") in (
            "wave7_fixed_age_matrix_lr_upper_boundary",
            "wave7_fixed_age_matrix_lr_ceiling",
        )
        cumulative_points = 7 if campaign.get("study") == "wave7_fixed_age_matrix_lr_ceiling" else 5
        points_label = "seven" if cumulative_points == 7 else "five"
        predecessor_wave = "wave9" if cumulative_points == 7 else "wave8"
        predecessor_label = predecessor_wave.replace("wave", "Wave-")
        if expanded_study:
            lines.extend(
                [
                    f"Separate upper-boundary expansion of `PVC/campaigns/"
                    f"{Path(campaign['parent_campaign']).parent.name}/campaign.json`.",
                    f"New trials per family: {campaign['initial_trials_per_family']}; "
                    f"new planned tokens/family: {campaign['initial_tokens_per_family']:,}; "
                    f"cumulative planned tokens/family: {campaign['cumulative_tokens_per_family']:,}.",
                    f"No cumulative {points_label}-point LR choice until both new trials in a family complete.",
                    f"GPU validation and launch wait for complete {predecessor_label} analysis, "
                    "cloud verification and restoration.",
                ]
            )
        if analysis and verified_identity:
            lines.extend(
                [
                    f"Saved analysis: `{analysis['status']}` at {local_time(analysis.get('time'))}; "
                    f"`PVC/{analysis_path.relative_to(root)}`.",
                    "",
                    "| Family | Complete / Prepared Trials | Observed Tokens | Completed LR Choice |",
                    "| --- | ---: | ---: | --- |",
                ]
            )
            for family in campaign["families"]:
                trials = [trial for trial in analysis["trials"] if trial["family"] == family]
                choice = analysis["completed_family_lr_choices"].get(family)
                selected = "none; incomplete family sweep" if choice is None else f"`{choice['trial']}`"
                completed = sum(trial["complete"] for trial in trials)
                observed = analysis["observed_tokens_by_family"][family]
                lines.append(f"| `{family}` | {completed} / {len(trials)} | {observed:,} | {selected} |")
            if expanded_study:
                lines.extend(
                    [
                        "",
                        f"Cumulative expanded-grid choices include the {cumulative_points - 2} completed parent trials:",
                        "",
                        f"| Family | Cumulative Observed Tokens | {points_label.title()}-Point LR Choice "
                        "| Upper Boundary Selected |",
                        "| --- | ---: | --- | --- |",
                    ]
                )
                for family in campaign["families"]:
                    choice = analysis["completed_cumulative_family_lr_choices"].get(family)
                    selected = (
                        f"none; incomplete {points_label}-point sweep"
                        if choice is None
                        else f"`{choice['trial']}` (LR {choice['matrix_lr']:.8f})"
                    )
                    boundary = "n/a" if choice is None else str(choice["upper_boundary_selected"]).lower()
                    observed = analysis["cumulative_observed_tokens_by_family"][family]
                    lines.append(f"| `{family}` | {observed:,} | {selected} | {boundary} |")
            lines.append(
                "LR choices use only completed held-out-mixture selection loss; "
                "no independently tuned superiority or efficiency claim."
            )
            if fixed_age_study:
                lines.extend(tuning_result_lines(campaign, analysis))
        elif analysis:
            lines.append("Saved analysis does not match the campaign identity or trial plan; no results published here.")
        else:
            lines.append("No saved tuning result analysis is recorded.")
        validation_binding_path = path.parent / "validation-binding.json"
        validation_binding = read_record(validation_binding_path)
        if validation_binding:
            lines.extend(
                [
                    "",
                    f"Actual-config validation binding: `{validation_binding['status']}` at "
                    f"{local_time(validation_binding.get('time'))}; "
                    f"`PVC/{validation_binding_path.relative_to(root)}`.",
                    "A frozen command plan is not GPU execution or training-launch approval.",
                ]
            )
            state_path = path.parent / "validation-state.json"
            validation_state = read_record(state_path)
            if validation_state:
                lines.append(
                    f"Saved validation execution: `{validation_state['status']}` at "
                    f"{local_time(validation_state.get('time'))}; `PVC/{state_path.relative_to(root)}`."
                )
                if "keeper_restore" in validation_state:
                    lines.append(f"Recorded validation keeper restoration: `{validation_state['keeper_restore']}`.")
                receipt_path = path.parent / "validation-gpu.json"
                validation_receipt = read_record(receipt_path)
                matches = (
                    (
                        validation_state.get("binding_sha256")
                        == hashlib.sha256(validation_binding_path.read_bytes()).hexdigest()
                        and validation_state.get("validation_sha256")
                        == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                    )
                    if validation_receipt
                    else False
                )
                if matches:
                    trials = validation_receipt["trials"]
                    passed = sum(all(value is True for value in trial["checks"].values()) for trial in trials)
                    lines.append(f"Saved actual-config GPU gates: {passed} / {len(trials)} trials passed.")
                elif validation_receipt:
                    lines.extend(
                        validation_progress_lines(path, validation_binding, validation_state, validation_receipt)
                    )
            else:
                lines.append("No actual-config GPU validation execution is recorded.")
        if fixed_age_study:
            lines.extend(["", *tuning_execution_lines(root, path, campaign)])
        if expanded_study:
            lines.extend(["", *boundary_queue_lines(root, path)])
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def update(document: Path, root: Path, now: datetime) -> None:
    with (root / ".endeavor-update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = document.read_text()
        payload = render(root, now)
        if START in original or END in original:
            if original.count(START) != 1 or original.count(END) != 1:
                raise ValueError("Ambiguous automated ledger markers")
            begin, end = original.index(START), original.index(END)
            if end < begin:
                raise ValueError("Reversed automated ledger markers")
            revised = original[:begin] + payload + original[end + len(END) :]
        else:
            if original.count(ANCHOR) != 1:
                raise ValueError("Cannot locate the existing endeavor's ledger insertion point")
            revised = original.replace(ANCHOR, payload + "\n\n" + ANCHOR, 1)
        with tempfile.NamedTemporaryFile(mode="w", dir=document.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(revised)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            temporary.chmod(document.stat().st_mode & 0o777)
            if document.read_text() != original:
                raise RuntimeError("ENDEAVOR changed during refresh; refusing to replace concurrent edits")
            temporary.replace(document)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    args = parser.parse_args()
    waves = ["wave9"]
    for wave in ("wave10", "wave12"):
        if (args.root / f"run-preparation/{wave}-launch-binding.json").is_file():
            waves.append(wave)
    for wave in waves:
        try:
            print(json.dumps(refresh_running_capture(args.root, datetime.now(UTC), wave=wave)))
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
            print(json.dumps({"status": f"{wave}_capture_refused", "error": f"{type(error).__name__}: {error}"}))
    update(args.document, args.root, datetime.now(UTC))


if __name__ == "__main__":
    main()
