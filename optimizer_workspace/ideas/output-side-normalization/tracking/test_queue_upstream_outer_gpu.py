# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check one-shot upstream diagnostic handoff without CUDA or keeper operations."""

import importlib.util
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest
import queue_upstream_outer_gpu as queue
from launch_wave5 import read, sha256
from update_endeavor import upstream_queue_lines


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def bound_queue(tmp_path):
    root = tmp_path
    out = root / "run-preparation/probe"
    helpers = out / "helpers"
    helpers.mkdir(parents=True)
    controller = helpers / "run_upstream_outer_gpu.py"
    controller.write_text("import json\nprint(json.dumps({'status':'fixture_check'}))\n")
    archive = out / "source.tar"
    archive.write_bytes(b"diagnostic source fixture")
    receipt = root / "cpu/receipt.json"
    record(receipt, {"status": "cpu_fixture"})
    campaign = root / "campaigns/upper/campaign.json"
    record(campaign, {"root": str(root)})
    execution = campaign.parent / "execution-binding.json"
    record(execution, {"helpers": str(campaign.parent / "execution-helpers")})
    frozen = {
        "cpu_receipt_sha256": sha256(receipt),
        "campaign": str(campaign),
        "campaign_sha256": sha256(campaign),
        "execution_binding_sha256": sha256(execution),
        "helpers_sha256": {controller.name: sha256(controller)},
    }
    record(out / "binding.json", frozen)
    saved = queue.freeze(root, receipt, out)
    spec = importlib.util.spec_from_file_location(
        "frozen_upstream_queue_test", Path(saved["helpers"]) / "queue_upstream_outer_gpu.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, receipt, out, module


def test_frozen_queue_preserves_original_diagnostic_and_refuses_changed_source(bound_queue):
    root, receipt, out, module = bound_queue
    diagnostic_hash = sha256(out / "binding.json")
    frozen = module.binding(root, receipt, out)
    assert frozen["status"] == "frozen_waiting_entire_upper_lr_study"
    with tarfile.open(frozen["archive"]) as bundle:
        for member in bundle.getmembers():
            assert bundle.extractfile(member).read() == (Path(frozen["helpers"]) / Path(member.name).name).read_bytes()
    with pytest.raises(FileExistsError):
        module.freeze(root, receipt, out)
    assert sha256(out / "binding.json") == diagnostic_hash
    assert not (out / "state.json").exists()
    (Path(frozen["helpers"]) / "queue_upstream_outer_gpu.sh").write_text("changed\n")
    with pytest.raises(ValueError, match="identity changed"):
        module.binding(root, receipt, out)


@pytest.fixture
def handoff(bound_queue, monkeypatch):
    root, receipt, out, module = bound_queue
    campaign = Path(read(out / "binding.json")["campaign"])
    observed = {
        "status": "active",
        "all_workers_terminal": False,
        "campaign_sha256": sha256(campaign),
        "binding_sha256": sha256(campaign.parent / "execution-binding.json"),
    }
    launches = []
    fail_observation = []

    def run(argv, **kwargs):
        if "--phase" in argv and argv[argv.index("--phase") + 1] == "snapshot":
            if fail_observation:
                raise subprocess.TimeoutExpired(argv, 600)
            record(Path(argv[argv.index("--out") + 1]), observed)
            return subprocess.CompletedProcess(argv, 0, "{}", "")
        # The frozen completion gate's external subprocess normally rederives trial and cloud provenance.
        return subprocess.CompletedProcess(
            argv,
            0,
            json.dumps({"snapshot": observed, "analysis": read(campaign.parent / "analysis.json"), "trials": {}}),
            "",
        )

    def spawn(argv, **kwargs):
        launches.append({"argv": argv, "kwargs": kwargs})
        return type("Child", (), {"pid": 987654})()

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: queue.NODE + "\n")
    monkeypatch.setattr(module, "observed_process", lambda pid: None)
    monkeypatch.setattr(module, "diagnostic_binding", lambda root, receipt, out: read(out / "binding.json"))
    return root, receipt, out, module, campaign, observed, launches, fail_observation


def test_missing_launch_and_active_study_wait_without_controller_or_gpu_commands(handoff):
    root, receipt, out, module, campaign, _, launches, _ = handoff
    assert module.advance(root, receipt, out)["status"] == "waiting_for_upper_lr_launch"
    record(campaign.parent / "launch-state.json", {"status": "launched"})
    assert module.advance(root, receipt, out)["status"] == "waiting_for_upper_lr"
    assert read(out / "queue/state.json")["predecessor_snapshot"]
    assert not launches
    assert not (out / "queue/controller.json").exists()
    assert not (out / "state.json").exists()


def test_unknown_predecessor_observation_waits_without_restarting_worker(handoff):
    root, receipt, out, module, campaign, _, launches, failure = handoff
    record(campaign.parent / "launch-state.json", {"status": "launched"})
    failure.append(True)
    state = module.advance(root, receipt, out)
    assert state["status"] == "waiting_for_upper_lr"
    assert "TimeoutExpired" in state["observation_error"]
    assert not launches
    assert not (out / "queue/controller.json").exists()


@pytest.mark.parametrize("fault", ["validation", "launch", "workers", "release", "unowned_probe"])
def test_failed_predecessor_or_unowned_probe_evidence_stops_queue_without_spawn(handoff, fault):
    root, receipt, out, module, campaign, observed, launches, _ = handoff
    record(campaign.parent / "launch-state.json", {"status": "launched"})
    if fault == "validation":
        record(campaign.parent / "validation-state.json", {"status": "needs_attention"})
    elif fault == "launch":
        record(campaign.parent / "launch-state.json", {"status": "needs_attention"})
    elif fault == "workers":
        observed["status"] = "needs_attention"
    elif fault == "release":
        observed["binding_sha256"] = "different"
    else:
        record(out / "state.json", {"status": "unowned"})
    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        module.advance(root, receipt, out)
    assert read(out / "queue/state.json")["status"] == "needs_attention"
    assert module.advance(root, receipt, out)["status"] == "needs_attention"
    assert not launches


def test_complete_study_with_pending_cloud_waits_for_full_verified_receipt(handoff):
    root, receipt, out, module, campaign, observed, launches, _ = handoff
    record(campaign.parent / "launch-state.json", {"status": "launched"})
    observed.update(status="complete", all_workers_terminal=True)
    record(campaign.parent / "analysis.json", {"status": "complete"})
    record(campaign.parent / "swanlab-upload.json", {"status": "partial"})
    assert module.advance(root, receipt, out)["status"] == "waiting_for_upper_lr_results"
    assert not launches
    assert not (out / "queue/controller.json").exists()


def test_controller_spawn_preserves_intent_and_uncertain_generation_never_retries(handoff):
    root, receipt, out, module, _, _, launches, _ = handoff
    intent = module.spawn(root, receipt, out)
    assert intent["start_ticks"] is None
    assert launches[0]["argv"] == module.command(root, receipt, out)
    assert launches[0]["kwargs"]["start_new_session"] is True
    assert module.observe_controller(out) == "unknown"
    assert module.advance(root, receipt, out)["status"] == "unknown_probe"
    assert module.advance(root, receipt, out)["status"] == "unknown_probe"
    assert len(launches) == 1
    with pytest.raises(FileExistsError):
        module.spawn(root, receipt, out)
    assert len(launches) == 1


def test_controller_spawn_failure_retains_intent_for_no_retry(bound_queue, monkeypatch):
    root, receipt, out, module = bound_queue

    def fail(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(OSError):
        module.spawn(root, receipt, out)
    assert read(out / "queue/controller.json")["status"] == "starting"
    with pytest.raises(FileExistsError):
        module.spawn(root, receipt, out)


def test_observer_tracks_real_marked_controller_until_terminal_and_refuses_reused_pid(bound_queue, monkeypatch):
    _, _, out, module = bound_queue
    argv = [queue.PYTHON, "-c", "import sys; print('ready',flush=True); sys.stdin.read()"]
    with subprocess.Popen(
        argv,
        env={**os.environ, queue.CONTROLLER_MARKER: "test-owner"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        assert child.stdout.readline().strip() == "ready"
        observed = module.observed_process(child.pid)
        record(
            out / "queue/controller.json",
            {"pid": child.pid, "start_ticks": observed[1], "command": argv, "marker": "test-owner"},
        )
        try:
            assert module.observe_controller(out) == "running"
        finally:
            child.communicate(input="", timeout=10)
    assert module.observe_controller(out) == "exited"
    monkeypatch.setattr(module, "observed_process", lambda pid: ("Z", "reused", []))
    with pytest.raises(RuntimeError, match="PID reused"):
        module.observe_controller(out)


def test_exited_controller_without_owned_result_cannot_complete_or_spawn_again(handoff):
    root, receipt, out, module, _, _, launches, _ = handoff
    record(
        out / "queue/controller.json",
        {
            "pid": 987654,
            "start_ticks": "1234",
            "command": module.command(root, receipt, out),
            "marker": "owner",
            "diagnostic_binding_sha256": sha256(out / "binding.json"),
        },
    )
    with pytest.raises(FileNotFoundError):
        module.advance(root, receipt, out)
    assert read(out / "queue/state.json")["status"] == "needs_attention"
    assert not launches


def test_failed_queue_installation_retains_intent_and_never_installs_twice(bound_queue, monkeypatch):
    root, receipt, out, module = bound_queue
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: queue.NODE + "\n")
    launches = []

    def fail(*args, **kwargs):
        launches.append(args)
        raise OSError("queue spawn failed")

    monkeypatch.setattr(subprocess, "Popen", fail)
    # Source checking itself uses subprocess.Popen; fail only the actual queue boundary.
    original_check = module.diagnostic_binding

    def validated(root, receipt, out):
        return read(out / "binding.json")

    monkeypatch.setattr(module, "diagnostic_binding", validated)
    with pytest.raises(OSError):
        module.install(root, receipt, out)
    assert read(out / "queue/installation.json")["status"] == "needs_attention"
    with pytest.raises(FileExistsError):
        module.install(root, receipt, out)
    assert len(launches) == 1
    monkeypatch.setattr(module, "diagnostic_binding", original_check)


def test_full_study_provenance_rederives_then_launches_unchanged_probe_exactly_once(handoff):
    root, receipt, out, module, campaign, observed, launches, _ = handoff
    record(campaign.parent / "launch-state.json", {"status": "launched"})
    observed.update(status="complete", all_workers_terminal=True)
    families = ("muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused")
    trials = [
        {"name": f"{family}_{label}", "config_sha256": f"{family}-{label}"}
        for family in families
        for label in ("a", "b")
    ]
    record(campaign, {"trials": trials})
    diagnostic = read(out / "binding.json")
    diagnostic["campaign_sha256"] = sha256(campaign)
    record(out / "binding.json", diagnostic)
    queue_binding = read(out / "queue/binding.json")
    queue_binding["campaign_sha256"] = sha256(campaign)
    queue_binding["diagnostic_binding_sha256"] = sha256(out / "binding.json")
    record(out / "queue/binding.json", queue_binding)
    observed["campaign_sha256"] = sha256(campaign)
    analysis = {
        "time": "2026-11-17T00:00:00+00:00",
        "status": "complete",
        "completed_cumulative_family_lr_choices": {family: {} for family in families},
    }
    record(campaign.parent / "analysis.json", analysis)
    cloud = {
        "status": "verified",
        "visibility": "PRIVATE",
        "project": "fixture/output-side-normalization-marin",
        "campaign_sha256": sha256(campaign),
        "execution_binding_sha256": sha256(campaign.parent / "execution-binding.json"),
        "runs": {},
    }
    for trial in trials:
        run = campaign.parent / "runs" / trial["name"]
        record(run / "run.json", {"status": "complete"})
        record(run / "metrics_from_000000.jsonl", {"step": 320})
        checkpoint = run / "checkpoints/step_000320"
        record(checkpoint / "complete.json", {"world": 8})
        for rank in range(8):
            (checkpoint / f"rank_{rank}.pt").write_bytes(b"checkpoint fixture")
        cloud["runs"][trial["name"]] = {
            "status": "verified",
            "manifest_sha256": sha256(run / "run.json"),
            "metrics_sha256": sha256(run / "metrics_from_000000.jsonl"),
            "checkpoint_receipt_sha256": sha256(checkpoint / "complete.json"),
            "checkpoint_rank_bytes": [(checkpoint / f"rank_{rank}.pt").stat().st_size for rank in range(8)],
            "campaign_sha256": sha256(campaign),
            "execution_binding_sha256": sha256(campaign.parent / "execution-binding.json"),
            "trial_config_sha256": trial["config_sha256"],
            "verification": {
                "state": "FINISHED",
                "configuration_verified": True,
                "token_axis_verified": True,
                "integer_values_exact": True,
                "point_counts": {"train/loss": 320, "tokens": 320, "eval/selection_loss": 9, "eval/c4_loss": 9},
            },
        }
    record(campaign.parent / "swanlab-upload.json", cloud)
    assert module.advance(root, receipt, out)["status"] == "unknown_probe"
    assert len(launches) == 1
    assert launches[0]["argv"] == module.command(root, receipt, out)
    assert read(out / "queue/state.json")["predecessor"]["cloud_sha256"] == sha256(
        campaign.parent / "swanlab-upload.json"
    )
    assert module.advance(root, receipt, out)["status"] == "unknown_probe"
    assert len(launches) == 1


def test_saved_controller_intent_cannot_authorize_changed_command(handoff):
    root, receipt, out, module, _, _, launches, _ = handoff
    intent = module.spawn(root, receipt, out)
    intent["command"] = [queue.PYTHON, "-c", "raise SystemExit(0)"]
    record(out / "queue/controller.json", intent)
    with pytest.raises(ValueError, match="intent differs"):
        module.advance(root, receipt, out)
    assert read(out / "queue/state.json")["status"] == "needs_attention"
    assert len(launches) == 1


def test_queue_ledger_shows_bound_wait_and_withholds_changed_controller_or_source(bound_queue):
    root, _, out, _ = bound_queue
    lines = "\n".join(upstream_queue_lines(root, out))
    assert "frozen_waiting_entire_upper_lr_study" in lines
    assert "No saved queue installation acknowledgment" in lines
    digest = sha256(out / "queue/binding.json")
    record(
        out / "queue/state.json",
        {"status": "waiting_for_upper_lr", "binding_sha256": digest, "time": "2026-11-17T00:00:00+00:00"},
    )
    lines = "\n".join(upstream_queue_lines(root, out))
    assert "Saved handoff: `waiting_for_upper_lr`" in lines
    record(
        out / "queue/controller.json",
        {
            "status": "starting",
            "command": ["changed"],
            "diagnostic_binding_sha256": sha256(out / "binding.json"),
        },
    )
    assert "controller intent differs" in "\n".join(upstream_queue_lines(root, out))
    (out / "queue/helpers/queue_upstream_outer_gpu.sh").write_text("changed\n")
    lines = "\n".join(upstream_queue_lines(root, out))
    assert "queue source or binding differs" in lines
    assert "Saved handoff" not in lines


@pytest.mark.parametrize("fault", [None, "owner", "coverage", "parity", "cleanup", "restore", "keeper"])
def test_final_probe_requires_owned_parity_complete_cleanup_and_healthy_keeper(bound_queue, monkeypatch, fault):
    _, _, out, module = bound_queue
    plan = {"shapes": [[1, 2, 2]], "patterns": ["zero"], "nesterov": [True], "learning_rates": [0.02], "cases": 1}
    frozen = read(out / "binding.json")
    frozen["plan"] = plan
    record(out / "binding.json", frozen)
    intent = {"pid": 1234, "start_ticks": "5678"}
    record(out / "queue/controller.json", intent)
    exact = {
        "momentum",
        "ns_source",
        "ns_output",
        "parameter_delta",
        "outer_momentum",
        "outer_ns_source",
        "outer_ns_output",
    }
    result = {
        **plan,
        "status": "passed_upstream_production_diagnostic_not_launch_gate",
        "owner": {"controller_pid": 1234, "controller_start_ticks": "5678"},
        "binding_sha256": sha256(out / "binding.json"),
        "helpers_sha256": frozen["helpers_sha256"],
        "cases": [
            {
                "shape": [1, 2, 2],
                "pattern": "zero",
                "nesterov": True,
                "lr": 0.02,
                "exact_null_and_unchanged_input": dict.fromkeys(exact, True),
                "same_ns_outer_equation": {
                    "within_tolerance": True,
                    "finite": True,
                    "mismatched_elements": 0,
                    "elements": 4,
                },
            }
        ],
        "timing": [
            {
                "shape": [1, 2, 2],
                "graphs": graph,
                "paired_median_ms": {"legacy_direct": 2.0, "upstream_outer": 1.0, "upstream_null": 0.5},
                "upstream_outer_over_legacy": 0.5,
                "upstream_outer_over_null": 2.0,
            }
            for graph in (False, True)
        ],
    }
    state = {
        "status": "measured_upstream_probe_not_launch_gate",
        "controller_pid": 1234,
        "controller_start_ticks": "5678",
        "binding_sha256": sha256(out / "binding.json"),
        "helpers_sha256": frozen["helpers_sha256"],
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "child_marker": "unassigned-test-marker",
    }
    if fault == "owner":
        result["owner"]["controller_pid"] = 9876
    elif fault == "coverage":
        result["cases"] = []
    elif fault == "parity":
        result["cases"][0]["exact_null_and_unchanged_input"]["parameter_delta"] = False
    elif fault == "cleanup":
        state["cleanup_exit_code"] = 1
    elif fault == "restore":
        state["keeper_restore"] = False
    record(out / "result.json", result)
    state["result_sha256"] = sha256(out / "result.json")
    record(out / "state.json", state)
    keeper_file = out / "keeper.pid"
    keeper_file.write_text("9000\n")
    generation = [(9000 + rank, str(rank), 9000 + rank) for rank in range(9)]
    monkeypatch.setattr(module, "keeper_generation", lambda pid: generation)
    occupied = list(range(9001, 9009))
    if fault == "keeper":
        occupied[-1] = 9999
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "\n".join(map(str, occupied)))
    if fault:
        with pytest.raises((RuntimeError, ValueError)):
            module.completed_diagnostic(out, frozen, keeper_file)
    else:
        complete = module.completed_diagnostic(out, frozen, keeper_file)
        assert complete["result_sha256"] == sha256(out / "result.json")
        assert complete["keeper_generation"] == generation
