# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check diagnostic calculations and allocation refusal before any CUDA execution."""

import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
import run_diagonal_diagnostic as controller
import torch
from diagnose_fused_diagonal import differences, fp64_diagonal_factors, source_and_trace, trace_sources
from experiment.block_toner import MU


def test_direction_error_report_preserves_original_gate_failure():
    expected = torch.tensor([1.0, 0.0, -0.01])
    actual = torch.tensor([1.001, 0.0003, -0.01])
    result = differences(actual, expected, {"rtol": 0.003, "atol": 5e-5})
    assert result["finite"]
    assert not result["within_tolerance"]
    assert result["mismatched_elements"] == 1
    assert result["max_absolute_error"] == pytest.approx(0.001000046730041504)
    assert result["relative_frobenius_error"] > 0


def test_nonfinite_direction_errors_remain_recordable_as_strict_json():
    result = differences(torch.tensor([float("nan"), float("inf")]), torch.ones(2), {"rtol": 0.003, "atol": 5e-5})
    serialized = json.dumps(result, allow_nan=False)
    assert not result["finite"]
    assert not result["within_tolerance"]
    assert result["actual_nonfinite_elements"] == 2
    assert json.loads(serialized)["max_absolute_error"] is None


def test_mixed_dtype_direction_comparison_retains_fp64_reference_at_tolerance_boundary():
    actual = torch.tensor([0.0, 1.0], dtype=torch.float32)
    expected = torch.tensor([1.00001000012e-6, 1.000000001], dtype=torch.float64)
    result = differences(actual, expected, {"rtol": 1e-5, "atol": 1e-6})
    assert result["finite"]
    assert not result["within_tolerance"]
    assert result["mismatched_elements"] == 1
    assert result["max_absolute_error"] == expected[0].item()
    assert result["mean_absolute_error"] == pytest.approx(
        (expected[0].item() + expected[1].item() - 1.0) / 2, rel=1e-9, abs=0
    )
    within = differences(actual, torch.tensor([0.5e-6, 1.0], dtype=torch.float64), {"rtol": 1e-5, "atol": 1e-6})
    assert within["within_tolerance"]
    assert within["mismatched_elements"] == 0


def test_mixed_dtype_nonfinite_direction_errors_retain_failure_and_json_receipt():
    actual = torch.tensor([float("nan"), float("inf"), 0.0], dtype=torch.float32)
    expected = torch.tensor([0.0, float("inf"), float("-inf")], dtype=torch.float64)
    result = json.loads(json.dumps(differences(actual, expected, {"rtol": 1e-5, "atol": 1e-6}), allow_nan=False))
    assert not result["finite"]
    assert not result["within_tolerance"]
    assert result["mismatched_elements"] == 2
    assert result["actual_nonfinite_elements"] == 2
    assert result["reference_nonfinite_elements"] == 2
    assert result["max_absolute_error"] is None


def test_preconditioned_source_matches_explicit_nesterov_right_and_left_products():
    gradient = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])
    momentum = torch.tensor([[[0.1, 0.2], [0.3, 0.5]]])
    left = (torch.tensor([[2.0, 0.5]]), None)
    right = (torch.tensor([[[1.0, 0.2], [0.2, 2.0]]]), None)
    args = SimpleNamespace(muon_momentum=0.95, muon_nesterov=True)
    actual, condition = source_and_trace(torch.zeros_like(gradient), gradient, momentum, left, right, args)
    update_momentum = 0.95 * momentum + gradient
    expected = (gradient + 0.95 * update_momentum) @ right[0]
    expected = left[0][:, :, None] * expected
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert condition["first_matrix_condition_number"] > 1
    assert condition["weights_shape"] == [1, 2, 2]


def test_fp64_diagonal_reference_retains_zero_identity_and_trace_scaling():
    moment = torch.tensor([[0, 0], [1, 3]], dtype=torch.float64)
    inner, outer = fp64_diagonal_factors(moment, 0.01, 0.5, 0.125)
    expected_inner = torch.tensor([[1, 1], [0.51**-0.5, 1.51**-0.5]], dtype=torch.float64)
    expected_outer = torch.tensor([[1, 1], [0.51**-0.125, 1.51**-0.125]], dtype=torch.float64)
    torch.testing.assert_close(inner, expected_inner, rtol=0, atol=1e-15)
    torch.testing.assert_close(outer, expected_outer, rtol=0, atol=1e-15)
    scaled = fp64_diagonal_factors(moment * 16, 0.01, 0.5, 0.125)
    torch.testing.assert_close(scaled[0], inner, rtol=0, atol=0)
    torch.testing.assert_close(scaled[1], outer, rtol=0, atol=0)


def test_ns_trace_reports_identical_repeat_then_detects_perturbed_upstream_output():
    source = torch.tensor([[[1.0, 0.2], [-0.1, 0.3], [0.2, 0.4]]])
    args = SimpleNamespace(muon_eps=1e-5, muon_ns_steps=3)
    repeated = trace_sources(MU, source, source.clone(), args, "f32")
    assert repeated["normalized_source"]["max_absolute_error"] == 0
    assert all(
        row[name]["max_absolute_error"] == 0
        for row in repeated["iterations"]
        for name in ("input", "gram", "polynomial", "output")
    )
    changed = source.clone()
    changed[0, 0, 0] += 0.01
    actual = trace_sources(MU, changed, source, args, "f32")
    expected = differences(
        MU.newton_schulz_batched(changed, 3, 1e-5, "f32"),
        MU.newton_schulz_batched(source, 3, 1e-5, "f32"),
        {"rtol": 0.003, "atol": 5e-5},
    )
    assert actual["iterations"][-1]["output"]["max_absolute_error"] == expected["max_absolute_error"]
    assert expected["max_absolute_error"] > 0


@pytest.fixture
def diagnostic_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "bound.py").write_text("frozen")
    prep = root / "run-preparation"
    prep.mkdir(parents=True)
    (prep / "wave5-candidate.json").write_text(
        json.dumps({"candidate": str(source), "files_sha256": {"bound.py": controller.sha256(source / "bound.py")}})
    )
    (prep / "wave5-launch-queue.json").write_text(json.dumps({"helpers_sha256": {}}))
    predecessor = root / "runs/wave4/lag"
    predecessor.mkdir(parents=True)
    (predecessor / "run.json").write_text(
        json.dumps({"status": "complete", "completed_steps": 5120, "config": {"training": {"steps": 5120}}})
    )
    (predecessor.parent / "keeper-restore-28.216.17.9.log").write_text("keeper start exit 0")
    (prep / "wave4.json").write_text(
        json.dumps({"launches": [{"node": controller.NODE, "pid": 12345, "out": str(predecessor)}]})
    )
    monkeypatch.setattr(controller, "process", lambda pid: None)
    return root


def test_controller_refuses_unacknowledged_training_before_releasing_keeper(diagnostic_root, monkeypatch):
    def check_output(command, **kwargs):
        if command == ["hostname", "-I"]:
            return controller.NODE
        if command == ["ps", "-eo", "args="]:
            return "python -m torch.distributed.run --out /different/live/run"
        raise AssertionError("Unexpected GPU or release command before refusal")

    monkeypatch.setattr(controller.subprocess, "check_output", check_output)
    out = diagnostic_root / "diagnostic"
    with pytest.raises(RuntimeError, match="Existing training"):
        controller.run(diagnostic_root, out)
    assert not out.exists()


def test_child_ignoring_termination_is_reaped_with_bounded_escalation():
    command = [
        sys.executable,
        "-c",
        "import signal, sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); sys.stdin.read()",
    ]
    marker = "cpu-cleanup-test"
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        start_new_session=True,
        env={**os.environ, controller.PROCESS_MARKER: marker},
    ) as child:
        assert child.stdout.readline() == b"ready\n"
        controller.finish_child(child, marker, {}, terminate_timeout=0.01, kill_timeout=5)
        assert child.returncode == -signal.SIGKILL


def test_independent_rank_session_is_stopped_after_launcher_exit():
    worker = (
        "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('rank-ready', flush=True); signal.pause()"
    )
    command = [
        sys.executable,
        "-c",
        "import subprocess, sys; child = subprocess.Popen([sys.executable, '-c', sys.argv[1]], "
        "stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True); "
        "assert child.stdout.readline() == b'rank-ready\\n'; print(child.pid, flush=True); sys.stdin.read()",
        worker,
    ]
    marker = "cpu-independent-rank-test"
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        start_new_session=True,
        env={**os.environ, controller.PROCESS_MARKER: marker},
    ) as child:
        rank_pid = int(child.stdout.readline())
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        generation = {}
        controller.finish_child(child, marker, generation, terminate_timeout=0.01, kill_timeout=5)
        assert rank_pid in generation
        assert not controller.tagged_processes(marker)


def test_keeper_inventory_records_independent_rank_session_with_start_identity(monkeypatch):
    workload = "keeper-rank-cpu-test"
    worker = "import signal; print('rank-ready', flush=True); signal.pause()"
    command = [
        sys.executable,
        "-c",
        "import subprocess, sys; child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], "
        "stdout=subprocess.PIPE, start_new_session=True); "
        "assert child.stdout.readline() == b'rank-ready\\n'; print(child.pid, flush=True); sys.stdin.read()",
        worker,
        workload,
    ]
    marker = "cpu-keeper-topology-test"
    monkeypatch.setattr(controller, "WORKLOAD", workload.encode())
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        start_new_session=True,
        env={**os.environ, controller.PROCESS_MARKER: marker},
    ) as child:
        rank_pid = int(child.stdout.readline())
        try:
            generation = controller.keeper_generation(child.pid)
            assert {pid for pid, _, _ in generation} == {child.pid, rank_pid}
            assert all(pid == group and started == controller.process(pid)[1] for pid, started, group in generation)
        finally:
            controller.finish_child(child, marker, {}, terminate_timeout=1, kill_timeout=5)
