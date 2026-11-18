# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU parity and retention regressions; CUDA root-stage transparency remains unissued."""

import hashlib
import io
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import replay_saved_roots as replay
import root_stage_trace as tracing
import torch
from benchmark_diagonal import source_definitions
from saved_ns_test_support import saved_evidence
from selected_lr_release_owner import sha256


@pytest.fixture
def roots():
    task = Path(__file__).resolve().parents[1]
    manifest = json.loads((task / "experiment/upstream_manifest.json").read_text())
    source = task / f"artifacts/external-sources/Toner-{manifest['revision']}/src/toner/fast/roots.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["files"]["src/toner/fast/roots.py"]
    module = ModuleType("cpu_root_stage_fixture")
    module.__dict__.update(
        torch=torch,
        EPS=1e-3,
        QUARTER=0.25,
        U32=2.0**-24,
        LOWP=torch.float16,
        ALPHA=0.25,
        POWERS=(0.125, 0.5),
        SQRT_ITERS=2,
        _sym_ok=lambda width: False,
    )
    names = (
        "_start",
        "_iter_mid",
        "_iter_w",
        "_iter_post",
        "_square",
        "_sqrt_start",
        "_sqrt_post",
        "_estimate",
        "_iterate",
        "froot_state",
        "froot_more",
        "basis_transform",
        "_formation",
        "form_powers",
        "ysqrt",
    )
    source_definitions(source, names, module.__dict__)
    for name in ("start", "iter_mid", "iter_w", "iter_post", "square", "sqrt_start", "sqrt_post"):
        setattr(module, name, getattr(module, f"_{name}"))
    # CPU products reconstruct split operands in FP32; they are not the pinned CUDA GEMMs.
    module._bmm32 = lambda left, right: left.float() @ right.float()
    module.mm_sym = module._bmm32
    module.rows_hi = lambda value: value
    module.rows_lo = lambda value: value
    module.cols_lo = lambda value: value
    module.scaled_cols_hi = lambda value: (
        value / value.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None],
        value.diagonal(dim1=-2, dim2=-1).mean(-1),
    )
    module.mm16 = lambda left, right, low_first: left.float() @ right.float()
    return module


@pytest.fixture
def inputs():
    values = torch.tensor(
        [
            [[2.0, 0.01, -0.03], [0.01, 3.0, 0.02], [-0.03, 0.02, 4.0]],
            [[1.5, -0.02, 0.01], [-0.02, 2.5, 0.02], [0.01, 0.02, 3.5]],
        ]
    )
    vectors = torch.linalg.qr(
        torch.tensor(
            [
                [[1.0, 0.2, 0.3], [0.1, 1.0, 0.2], [0.2, 0.3, 1.0]],
                [[1.0, -0.1, 0.2], [0.2, 1.0, 0.1], [0.1, -0.2, 1.0]],
            ]
        )
    )[0]
    return values, SimpleNamespace(SB=vectors, S2=vectors)


def root_call(roots, inputs):
    covariance, basis = inputs
    transformed, scale = roots.basis_transform(covariance, basis)
    lam, off, rho, estimate, state = roots.froot_state(transformed, 2, 0.0)
    factors, sqrt_estimate = roots.form_powers(basis, lam, off)
    return {
        "B": transformed,
        "mu": scale,
        "rho": rho,
        "estimate": estimate,
        "state": state,
        "factors": factors,
        "sqrt_estimate": sqrt_estimate,
    }


def assert_exact_tree(actual, expected):
    result = tracing.comparison(actual, expected, "test")
    assert result["exact"]
    assert result["finite"]
    assert result["tensor_comparisons"] > 0


def test_pinned_cpu_root_equations_preserve_outputs_inputs_and_rng_with_stage_recording(roots, inputs):
    reference = root_call(roots, inputs)
    before = tracing.cpu_tree({"covariance": inputs[0], "basis": vars(inputs[1])})
    rng = torch.random.get_rng_state().clone()
    functions = {name: getattr(roots, name) for name in tracing.OUTPUT_FIELDS}
    with tracing.observe_root_stages(roots) as trace:
        observed = root_call(roots, inputs)
        saved = trace.snapshot(observed)
    assert_exact_tree(observed, reference)
    assert_exact_tree({"covariance": inputs[0], "basis": vars(inputs[1])}, before)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(getattr(roots, name) is function for name, function in functions.items())
    starts = [record for record in saved["records"] if record["stage"] == "start"]
    assert len(starts) == 1
    assert_exact_tree(starts[0]["values"]["E"].half(), starts[0]["values"]["Eb"])
    iterations = [record for record in saved["records"] if record["stage"] == "iter_post"]
    assert [record["occurrence"] for record in iterations] == [0, 1]
    assert_exact_tree(iterations[-1]["values"]["lam"], observed["state"][0])
    assert_exact_tree(iterations[-1]["values"]["O"], observed["state"][1])
    formations = [record for record in saved["records"] if record["stage"] == "_formation"]
    assert len(formations) == 2
    for record, power in zip(formations, (0.125, 0.5), strict=True):
        assert_exact_tree(record["values"]["root"], observed["factors"][power])


def test_recording_retains_original_output_objects_without_numeric_copies(roots, inputs):
    numerical = roots.iter_mid
    outputs = []

    def observed_numerical(*args):
        value = numerical(*args)
        outputs.append(value)
        return value

    roots.iter_mid = observed_numerical
    with tracing.observe_root_stages(roots) as trace:
        root_call(roots, inputs)
        records = [record for record in trace.records if record["stage"] == "iter_mid"]
        for record, output in zip(records, outputs, strict=True):
            assert record["values"]["P"] is output[0]
            assert record["values"]["Pb"] is output[1]
    assert roots.iter_mid is observed_numerical


def test_root_failure_restores_kernels_and_preserves_already_recorded_intermediates(roots, inputs):
    numerical = roots.iter_post
    calls = 0

    def failing(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second fixed-point iteration failed")
        return numerical(*args)

    roots.iter_post = failing
    originals = {name: getattr(roots, name) for name in tracing.OUTPUT_FIELDS}
    with pytest.raises(RuntimeError, match="second fixed-point"), tracing.observe_root_stages(roots) as trace:
        root_call(roots, inputs)
    assert all(getattr(roots, name) is function for name, function in originals.items())
    completed = [record for record in trace.records if record["stage"] == "iter_post"]
    assert len(completed) == 1
    assert torch.isfinite(completed[0]["values"]["O"]).all()


def test_independent_cpu_snapshots_round_trip_and_localize_a_changed_iteration(roots, inputs):
    snapshots = []
    for _ in range(2):
        with tracing.observe_root_stages(roots) as trace:
            snapshots.append(trace.snapshot(root_call(roots, inputs)))
    stream = io.BytesIO()
    torch.save(snapshots[0], stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=True)
    exact = tracing.compare_root_stages(restored, snapshots[1])
    assert exact["complete_stage_coverage"]
    assert all(stage["exact"] and stage["finite"] for stage in exact["stages"])
    assert exact["output"]["exact"]
    stage = next(record for record in restored["records"] if record["stage"] == "iter_post")
    stage["values"]["O"][0, 0, 1] += 0.25
    changed = tracing.compare_root_stages(restored, snapshots[1])
    failures = [item for item in changed["stages"] if not item["within_tolerance"]]
    assert [(item["stage"], item["occurrence"]) for item in failures] == [("iter_post", 0)]
    assert failures[0]["changed_tensors"][0]["max_absolute_error"] == pytest.approx(0.25)
    assert changed["output"]["exact"]
    assert json.loads(json.dumps(changed, allow_nan=False))["complete_stage_coverage"]


def test_different_iteration_counts_are_not_prefix_paired(roots, inputs):
    covariance, basis = inputs
    transformed, _ = roots.basis_transform(covariance, basis)
    snapshots = []
    for count in (1, 2):
        with tracing.observe_root_stages(roots) as trace:
            snapshots.append(trace.snapshot(roots.froot_state(transformed, count)))
    result = tracing.compare_root_stages(*snapshots)
    assert not result["complete_stage_coverage"]
    assert result["stages"] == []
    assert result["stage_coordinates"]["actual"] != result["stage_coordinates"]["expected"]


def test_nested_recording_refuses_and_leaves_outer_recorder_usable(roots, inputs):
    with tracing.observe_root_stages(roots) as outer:
        with pytest.raises(RuntimeError, match="unwrapped"):
            with tracing.observe_root_stages(roots):
                pytest.fail("nested root observation must not start")
        snapshot = outer.snapshot(root_call(roots, inputs))
    assert tracing.comparison(snapshot["output"], root_call(roots, inputs), "root")["exact"]


def test_snapshot_refuses_cuda_capture_before_copying_tensor_values(roots, inputs, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with tracing.observe_root_stages(roots) as trace:
        output = root_call(roots, inputs)
        with pytest.raises(RuntimeError, match="captured CUDA graph"):
            trace.snapshot(output)
    assert output["B"].shape == (2, 3, 3)


def replay_groups(roots, inputs):
    covariance, basis = inputs
    groups = {}
    for mode in ("continuous", "resumed"):
        values = covariance.clone()
        if mode == "resumed":
            values[1, 0, 0] += 0.25
        group_inputs = {"C": values, "tol": torch.ones(2, 3), "SB": basis.SB.clone(), "S2": basis.S2.clone()}
        result = root_call(roots, (values, SimpleNamespace(SB=group_inputs["SB"], S2=group_inputs["S2"])))
        groups[mode] = {"inputs": group_inputs, "saved_output": result}
    return groups


def test_crossed_replay_retains_independent_stages_and_original_source_values(roots, inputs):
    groups = replay_groups(roots, inputs)
    before = tracing.cpu_tree(groups)
    calls = []

    def numerical(mode, values):
        calls.append(mode)
        with tracing.observe_root_stages(roots) as trace:
            return trace.snapshot(root_call(roots, (values["C"], SimpleNamespace(SB=values["SB"], S2=values["S2"]))))

    result = replay.replay_root_pair(groups, numerical)
    assert calls == ["continuous", "resumed", "continuous", "resumed"]
    assert all(item["own_saved_output"]["exact"] for item in result.report["historical_comparisons"])
    assert all(not item["other_saved_output"]["exact"] for item in result.report["historical_comparisons"])
    for repeat in result.report["repeats"].values():
        assert repeat["complete_stage_coverage"]
        assert repeat["output"]["exact"]
        assert all(stage["exact"] and stage["finite"] for stage in repeat["stages"])
    result.snapshots[0]["output"]["B"].zero_()
    assert not tracing.comparison(result.snapshots[0]["output"], result.snapshots[2]["output"], "separate")["exact"]
    assert_exact_tree(groups, before)
    assert result.report["owner_context"] is None
    assert json.loads(json.dumps(result.report, allow_nan=False))["status"] == "saved_root_stage_replay_gpu_unissued"


@pytest.mark.parametrize("field", ("inputs", "saved_output"))
def test_crossed_replay_rejects_mutation_of_another_mode_evidence(roots, inputs, field):
    groups = replay_groups(roots, inputs)

    def numerical(mode, values):
        key = "C" if field == "inputs" else "B"
        groups["resumed"][field][key].zero_()
        with tracing.observe_root_stages(roots) as trace:
            return trace.snapshot(root_call(roots, (values["C"], SimpleNamespace(SB=values["SB"], S2=values["S2"]))))

    with pytest.raises(ValueError, match="mutated"):
        replay.replay_root_pair(groups, numerical)


def test_crossed_replay_nonfinite_stage_values_remain_explicit_failures(roots, inputs):
    groups = replay_groups(roots, inputs)

    def numerical(mode, values):
        with tracing.observe_root_stages(roots) as trace:
            snapshot = trace.snapshot(root_call(roots, (values["C"], SimpleNamespace(SB=values["SB"], S2=values["S2"]))))
        next(record for record in snapshot["records"] if record["stage"] == "start")["values"]["O"].fill_(float("inf"))
        return snapshot

    report = replay.replay_root_pair(groups, numerical).report
    serialized = json.loads(json.dumps(report, allow_nan=False))
    start = next(item for item in serialized["repeats"]["continuous"]["stages"] if item["stage"] == "start")
    assert start["exact"]
    assert not start["finite"]
    assert not start["within_tolerance"]


def test_root_source_clone_preserves_transposes_and_refuses_nonzero_offset_normalization(inputs):
    covariance, basis = inputs
    values = {"C": covariance.mT, "tol": torch.ones(2, 3), "SB": basis.SB.mT, "S2": basis.S2.mT}
    cloned = replay.clone_inputs(values)
    assert_exact_tree(cloned, values)
    assert all(cloned[name].stride() == value.stride() for name, value in values.items())
    values["C"] = torch.ones(3, 3, 3)[1:]
    with pytest.raises(ValueError, match="raw layout"):
        replay.clone_inputs(values)


def test_stage_comparison_refuses_matching_omitted_kernel_outputs(roots, inputs):
    with tracing.observe_root_stages(roots) as trace:
        snapshot = trace.snapshot(root_call(roots, inputs))
    snapshot["records"][0]["values"].clear()
    with pytest.raises(ValueError, match="complete named"):
        tracing.compare_root_stages(snapshot, snapshot)


@pytest.fixture
def bound_roots(tmp_path, monkeypatch, request):
    group, fault = getattr(request, "param", ("d", None))
    root = tmp_path / "root"
    receipt_path, paths = saved_evidence(root)
    candidate = root / "candidate"
    task = Path(__file__).resolve().parents[1]
    revision = json.loads((task / "experiment/upstream_manifest.json").read_text())["revision"]
    upstream = candidate / "artifacts/pinned-sources-materialized/Toner-synthetic"
    alias = candidate / "artifacts/external-sources"
    alias.symlink_to("pinned-sources-materialized", target_is_directory=True)
    numerical = task / f"artifacts/external-sources/Toner-{revision}"
    shutil.copy2(numerical / "src/toner/fast/roots.py", upstream / "src/toner/fast/roots.py")
    arguments = upstream / "reproduce/marin/optimizer.py"
    arguments.parent.mkdir(parents=True)
    shutil.copy2(numerical / "reproduce/marin/optimizer.py", arguments)
    records_path = candidate / "experiment/records.py"
    shutil.copy2(task / "experiment/records.py", records_path)
    config_path = candidate / "experiment/config.json"
    config_path.write_bytes((task / "experiment/configs/inner_outer_leftdiag_0p5m_seed0.json").read_bytes())
    manifest_path = candidate / "experiment/upstream_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["src/toner/fast/roots.py"] = sha256(upstream / "src/toner/fast/roots.py")
    manifest_path.write_text(json.dumps(manifest))
    data_path = root / "data/experiment_receipt.json"
    data_path.parent.mkdir()
    data_path.write_text(json.dumps({"status": "synthetic_cpu_fixture"}))
    records = source_definitions(
        records_path,
        ("sha256", "checkpoint_identity", "CHECKPOINT_SCHEMA"),
        {"TASK": candidate, "UPSTREAM": alias / upstream.name, "Path": Path, "hashlib": hashlib, "json": json},
    )
    identity = records["checkpoint_identity"](json.loads(config_path.read_bytes()), data_path)
    if fault == "shared_wrong_identity":
        identity["config_sha256"] = "0" * 64
    binding_path = paths[0]
    binding = json.loads(binding_path.read_bytes())
    binding["files_sha256"] = {
        str(path.relative_to(candidate)): sha256(path)
        for directory in (candidate / "experiment", upstream)
        for path in directory.rglob("*")
        if path.is_file()
    }
    binding["data_receipt_sha256"] = sha256(data_path)
    binding_path.write_text(json.dumps(binding))
    receipt = json.loads(receipt_path.read_bytes())
    receipt["execution_binding_sha256"] = sha256(binding_path)
    shape = (2, 3, 3) if group == "d" else (3, 4, 4)
    batch, width, _ = shape
    for mode in ("continuous", "resumed"):
        values = torch.arange(batch * width * width, dtype=torch.float32).reshape(shape).mT
        before = {
            "C": values,
            "tol": torch.ones(batch, 3),
            "SB": torch.ones(batch, 2 * width, width, dtype=torch.float16),
            "S2": torch.ones(batch, width, 2 * width, dtype=torch.float16),
        }
        output = (
            (values.clone(), values.clone()),
            values.clone(),
            torch.zeros(batch, dtype=torch.bool),
            (torch.ones(batch, width), values.clone(), torch.ones(batch)),
        )
        if fault == "partial_output":
            output = (output[0][:1], *output[1:])
        elif fault == "nonfinite_output":
            output[3][1][0, 0, 0] = float("inf")
        elif fault == "short_basis":
            before["S2"] = before["S2"][..., :-1]
        provenance = {"identity": identity, "start_step": 0 if mode == "continuous" else 22}
        trace_records = [
            {"stage": f"input/{group}/before_root", "values": before},
            {"stage": f"input/{group}/candidate", "values": output},
        ]
        path = replay.analyzer._trace_path(root, mode, 0, 23)
        torch.save({"schema": 1, "rank": 0, "update": 23, "provenance": provenance, "records": trace_records}, path)
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "status": "saved_continuation_intermediates_not_numerical_acceptance",
                    "rank": 0,
                    "update": 23,
                    "provenance": provenance,
                    "records": len(trace_records),
                    "stages": [record["stage"] for record in trace_records],
                    "trace_sha256": sha256(path),
                }
            )
        )
        receipt["analysis"]["traces"][mode]["files_sha256"].update({str(item): sha256(item) for item in (path, sidecar)})
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(replay, "ROOT", root)
    monkeypatch.setattr(replay, "RECEIPT_SHA256", sha256(receipt_path))
    monkeypatch.setattr(replay, "BINDING_SHA256", sha256(binding_path))
    monkeypatch.setattr(replay, "SHAPES", {group: shape})
    return group, root, identity, records_path


@pytest.mark.parametrize("bound_roots", (("d", None), ("f", None)), indirect=True)
def test_saved_root_loader_authenticates_identity_split_layout_and_independent_outputs(bound_roots):
    group, root, identity, _ = bound_roots
    path = replay.analyzer._trace_path(root, "continuous", 0, 23)
    before = sha256(path)
    with ExitStack() as stack:
        groups, evidence = replay.load_saved_roots(stack, group)
        assert evidence["identity"] == identity
        assert evidence["root_execution"] == {
            "mode": "ref",
            "split": True,
            "graphs": True,
            "iterations": 2,
            "inner_power": 0.5,
            "outer_power": 0.125,
        }
        raw = torch.load(path, map_location="cpu", weights_only=True)["records"]
        assert_exact_tree(groups["continuous"]["inputs"], raw[0]["values"])
        assert_exact_tree(groups["continuous"]["saved_output"], raw[1]["values"])
        covariance = groups["continuous"]["inputs"]["C"]
        assert covariance.stride() == raw[0]["values"]["C"].stride()
        assert not covariance.is_contiguous()
        assert evidence["trace_pair"]["continuous"]["input_layouts"]["SB"]["dtype"] == "torch.float16"
        groups["continuous"]["saved_output"][0][0].zero_()
        groups["continuous"]["inputs"]["C"].zero_()
        assert not tracing.comparison(groups["continuous"], groups["resumed"], "isolated")["exact"]
    assert sha256(path) == before


@pytest.mark.parametrize(
    "bound_roots, message",
    (
        (("d", "shared_wrong_identity"), "identity differs"),
        (("f", "partial_output"), "output structure"),
        (("d", "nonfinite_output"), "finite full-shaped"),
        (("f", "short_basis"), "finite full-shaped"),
    ),
    indirect=("bound_roots",),
)
def test_saved_root_loader_refuses_authenticated_but_invalid_historical_evidence(bound_roots, message):
    group, _, _, _ = bound_roots
    with ExitStack() as stack, pytest.raises(ValueError, match=message):
        replay.load_saved_roots(stack, group)


def test_saved_root_loader_detects_source_replacement_during_analysis(bound_roots):
    group, _, _, source = bound_roots
    with pytest.raises(ValueError, match="changed during"):
        with ExitStack() as stack:
            replay.load_saved_roots(stack, group)
            replacement = source.with_suffix(".replacement")
            replacement.write_bytes(source.read_bytes())
            replacement.replace(source)
