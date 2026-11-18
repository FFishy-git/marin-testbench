# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU numerical and wire-contract regressions for the distinct owned-output observer."""

import copy
import importlib
import importlib.util
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch


def observer_module():
    if Path(__file__).parent.name == "tests":
        return importlib.import_module("experiment.owned_output_observer")
    candidate = Path(
        "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
        "/candidates/owned-output-observed-formatted-20261117/experiment"
    )
    package_name = "_owned_output_validation_cpu"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name, candidate / "__init__.py", submodule_search_locations=[str(candidate)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    name = f"{package_name}.owned_output_observer"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("owned_output_validation_observer.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OBSERVER = observer_module()
BLOCK = importlib.import_module(f"{OBSERVER.__package__}.block_toner")
MODEL = importlib.import_module("reproduce.marin.model").Model
CONTRACT = importlib.import_module(
    "experiment.owned_output_validation" if Path(__file__).parent.name == "tests" else "owned_output_validation"
)
WORLD = 8
MODEL_CONFIG = {
    "layers": 8,
    "hidden": 4,
    "mlp": 8,
    "heads": 2,
    "sequence_length": 8,
    "vocab_size": 16,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "residual_dtype": "bf16",
}
DIMENSIONS = {name: MODEL_CONFIG[name] for name in ("layers", "hidden", "mlp")}
SETTINGS = {
    "decay": 0.938,
    "damping": 0.001,
    "inner_power": 0.5,
    "outer_power": 0.125,
    "rows_total": 6 * WORLD,
    "device": "cpu",
    "backend": "torch",
}


@pytest.fixture
def collectives(monkeypatch):
    """Replace only distributed I/O; equal rank rows make pooled sums exact."""
    monkeypatch.setattr(OBSERVER.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(OBSERVER.dist, "new_group", lambda backend: None)
    monkeypatch.setattr(OBSERVER.dist, "all_reduce", lambda tensor, **kwargs: tensor.mul_(WORLD))


def captured_fast(rank=1, apply_left=True):
    network = MODEL(MODEL_CONFIG)
    rows, by_shape = {}, {}
    for layer, block in enumerate(network.body.blocks):
        for kind in BLOCK.BLOCK_KINDS:
            parameter = getattr(block, kind).weight
            rows[layer, kind] = torch.zeros(6, parameter.shape[0], dtype=torch.bfloat16)
            by_shape.setdefault(tuple(parameter.shape), []).append((layer, kind, parameter))
    counts = BLOCK.layer_counts(MODEL_CONFIG["layers"], WORLD)
    owner = [r for r, count in enumerate(counts) for _ in range(count)]
    arm = "outer_leftdiag_upstream_fused" if apply_left else "null_leftdiag_upstream_fused"
    return SimpleNamespace(
        arm=BLOCK.BLOCK_ARMS[arm],
        input=SimpleNamespace(
            L=len(network.body.blocks),
            groups={"d": SimpleNamespace(n=MODEL_CONFIG["hidden"]), "f": SimpleNamespace(n=MODEL_CONFIG["mlp"])},
        ),
        output=None,
        output_diagonal=BLOCK.OwnedOutputMoments(
            rows, rank=rank, layers=len(network.body.blocks), world=WORLD, **SETTINGS
        ),
        device=torch.device("cpu"),
        shape_groups=[
            (shape, [[entry for entry in entries if owner[entry[0]] == r] for r in range(WORLD)])
            for shape, entries in by_shape.items()
        ],
        layer_counts=counts,
        rank=rank,
        world=WORLD,
        steps=0,
        update_scalars=torch.tensor([0.02, 0.1]),
        args=SimpleNamespace(
            muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32"
        ),
    )


def update_rows(fast, step):
    for (layer, kind), rows in fast.output_diagonal.bufs.items():
        values = torch.arange(rows.numel()).reshape(rows.shape).float()
        rows.copy_(torch.sin(values * 0.13 + layer * 0.1 + BLOCK.BLOCK_KINDS.index(kind) * 0.01 + step * 0.03))
        if layer == 0:
            rows.zero_()


def observed_update(verifier, step):
    update_rows(verifier.fast, step)
    verifier.begin()
    verifier.actual.update()
    verifier.check_factors()
    verifier.fast.steps += 1


def stack_arguments(verifier, group):
    shape, per_rank = group
    mine = per_rank[verifier.fast.rank]
    generator = torch.Generator().manual_seed(sum(shape))
    weights, gradients, momentum = [torch.randn(len(mine), *shape, generator=generator) * 0.01 for _ in range(3)]
    right = tuple(
        torch.diag(torch.linspace(0.8 + side * 0.1, 1.2 + side * 0.1, shape[1])).repeat(len(mine), 1, 1)
        for side in (0, 1)
    )
    outer = (
        torch.stack([verifier.actual.factors()[layer, kind][1] for layer, kind, _ in mine])
        if verifier.fast.arm.apply_left
        else None
    )
    trace = {}
    updated, delta = BLOCK.block_update_stack(
        weights,
        gradients,
        momentum,
        (None, outer),
        right,
        args=verifier.fast.args,
        lr=verifier.fast.update_scalars[0],
        weight_decay=verifier.fast.update_scalars[1],
        trace=trace,
    )
    return mine, weights, gradients, momentum, updated, delta, right, trace, 0.02, 0.1


def gather_equal_rank_records(received, record):
    """Collective-I/O substitute; separate tests collect each rank's actual result."""
    received[:] = [copy.deepcopy(record) for _ in range(WORLD)]
    for rank, item in enumerate(received):
        item["rank"] = rank
        item.update(OBSERVER.owned_output_layout(record["model"], WORLD, rank))


def capture_update(verifier, step):
    update_rows(verifier.fast, step)
    verifier.begin()
    verifier.actual.update()
    for group in verifier.fast.shape_groups:
        verifier(*stack_arguments(verifier, group))
    verifier.fast.steps += 1


def complete_update(verifier, step):
    capture_update(verifier, step)
    return verifier.finish()


@pytest.mark.parametrize("apply_left", [False, True])
def test_owned_observer_begin_update_finish_gathers_real_all_rank_results(collectives, monkeypatch, apply_left):
    verifiers = [OBSERVER.OwnedOutputVerifier(captured_fast(rank, apply_left)) for rank in range(WORLD)]
    submissions = [None] * WORLD
    barrier = Barrier(WORLD, timeout=30)

    def gather(received, record):
        submissions[record["rank"]] = copy.deepcopy(record)
        barrier.wait()
        received[:] = copy.deepcopy(submissions)

    monkeypatch.setattr(OBSERVER.dist, "all_gather_object", gather)
    for step in (1, 2):
        submissions[:] = [None] * WORLD
        for verifier in verifiers:
            capture_update(verifier, step)
        with ThreadPoolExecutor(max_workers=WORLD) as pool:
            results = list(pool.map(lambda verifier: verifier.finish(), verifiers))
        for result in results:
            assert result["owned_output_step"] == step
            assert result["owned_output_parity_passed"] is True
            assert result["upstream_null_exact"] is (not apply_left)
            assert result["owned_output_records"] == submissions
        result = validate(json.loads(json.dumps(submissions)), step=step, apply_left=apply_left)
        assert [record["rank"] for record in result["owned_output_records"]] == list(range(WORLD))
        assert all(record["owned_count"] == 7 for record in submissions)
        assert result["owned_output_max_errors"]["factor_same_state"] == 0
    verifier = verifiers[-1]
    update_rows(verifier.fast, 3)
    verifier.begin()
    verifier.actual.update()
    verifier(*stack_arguments(verifier, verifier.fast.shape_groups[0]))
    verifier.fast.steps += 1
    with pytest.raises(ValueError, match="every owned shape"):
        verifier.finish()


@pytest.mark.parametrize("corruption", ["pinned", "null", "coverage", "stale", "nonfinite"])
def test_owned_observer_finish_rejects_corrupt_remote_rank_proof(collectives, monkeypatch, corruption):
    verifier = OBSERVER.OwnedOutputVerifier(captured_fast())

    def gather(received, record):
        gather_equal_rank_records(received, record)
        remote = received[7]
        if corruption == "pinned":
            remote["upstream"]["upstream_pinned_input_exact"] = False
        elif corruption == "null":
            remote["upstream"]["upstream_null_exact"] = 0
        elif corruption == "coverage":
            remote["owned_keys"].pop()
        elif corruption == "stale":
            remote["step"] -= 1
        else:
            remote["upstream"]["upstream_output_equation_max_error"] = float("inf")

    monkeypatch.setattr(OBSERVER.dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match="rank 7"):
        complete_update(verifier, 1)


@pytest.mark.parametrize("corruption", ["omitted", "duplicate", "repeated", "momentum", "ns", "delta", "scalars"])
def test_owned_observer_rejects_incomplete_or_corrupt_local_update(collectives, corruption):
    verifier = OBSERVER.OwnedOutputVerifier(captured_fast())
    update_rows(verifier.fast, 1)
    verifier.begin()
    verifier.actual.update()
    arguments = list(stack_arguments(verifier, verifier.fast.shape_groups[0]))
    if corruption == "omitted":
        arguments[0] = arguments[0][:-1]
    elif corruption == "duplicate":
        arguments[0] = [arguments[0][0]] * len(arguments[0])
    elif corruption == "repeated":
        verifier(*arguments)
    elif corruption == "momentum":
        arguments[4][0, 0, 0] += 0.01
    elif corruption == "ns":
        arguments[7]["ns_source"][0, 0, 0] += 0.01
    elif corruption == "delta":
        arguments[5][0, 0, 0] += 0.01
    else:
        verifier.fast.update_scalars[0] += 0.01
    with pytest.raises((ValueError, AssertionError)):
        verifier(*arguments)


@pytest.mark.parametrize("apply_left", [False, True])
def test_owned_observer_rejects_truncated_tensors_with_complete_local_key_inventory(collectives, apply_left):
    verifier = OBSERVER.OwnedOutputVerifier(captured_fast(apply_left=apply_left))
    update_rows(verifier.fast, 1)
    verifier.begin()
    verifier.actual.update()
    arguments = list(stack_arguments(verifier, verifier.fast.shape_groups[0]))
    for index in range(1, 6):
        arguments[index] = arguments[index][:1]
    arguments[6] = tuple(factor[:1] for factor in arguments[6])
    arguments[7] = {name: tensor[:1] for name, tensor in arguments[7].items()}
    with pytest.raises(ValueError, match="complete local projection stack"):
        verifier(*arguments)


@pytest.mark.parametrize("corruption", ["unowned", "weight", "truncated", "coverage"])
def test_owned_observer_finish_rechecks_retained_state_after_all_callbacks(collectives, corruption):
    verifier = OBSERVER.OwnedOutputVerifier(captured_fast())
    capture_update(verifier, 1)
    if corruption == "unowned":
        verifier.actual.moment[4][-1, 0] += 0.1
    elif corruption == "weight":
        verifier.actual.weight += 0.01
    elif corruption == "truncated":
        verifier.actual.moment[4] = verifier.actual.moment[4][:-1]
    else:
        verifier.fast.shape_groups[0][1][7].pop()
    with pytest.raises((ValueError, AssertionError)):
        verifier.finish()


@pytest.mark.parametrize("rank", [0, 7])
@pytest.mark.parametrize("apply_left", [False, True])
def test_owned_observer_full_constructor_checks_model_and_exact_same_state_factors(collectives, rank, apply_left):
    fast = captured_fast(rank, apply_left)
    verifier = OBSERVER.OwnedOutputVerifier(fast)
    observed_update(verifier, 1)
    original = BLOCK.DiagonalMoments(fast.output_diagonal.bufs, world=1, **SETTINGS)
    original.load_state_dict(fast.output_diagonal.state_dict())
    expected = {(layer, kind) for _, per_rank in fast.shape_groups for layer, kind, _ in per_rank[rank]}
    actual = fast.output_diagonal.factors()
    assert set(actual) == expected
    for key, (inner, outer) in actual.items():
        assert inner is None
        torch.testing.assert_close(outer, original.factors()[key][1], rtol=0, atol=0)
    assert {width: list(moment.shape) for width, moment in fast.output_diagonal.moment.items()} == {
        4: [40, 4],
        8: [16, 8],
    }


def test_owned_observer_exact_factor_reference_does_not_require_exact_independent_accumulation(collectives):
    verifier = OBSERVER.OwnedOutputVerifier(captured_fast())
    observed_update(verifier, 1)
    # A representable accumulation difference within the unchanged tolerance must
    # not contaminate the separate exact same-state factoring check.
    moment = verifier.actual.moment[4]
    moment[5, 0] = torch.nextafter(moment[5, 0], torch.tensor(float("inf")))
    verifier.check_factors()
    original = BLOCK.DiagonalMoments(verifier.actual.bufs, world=1, **SETTINGS)
    original.load_state_dict(verifier.actual.state_dict())
    torch.testing.assert_close(verifier.actual.factors()[1, "q"][1], original.factors()[1, "q"][1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "corruption", ["unowned", "weight", "truncated_width", "truncated_rows", "local_coverage", "owned_slice"]
)
def test_owned_observer_rejects_corrupt_retained_state_and_projection_coverage(collectives, corruption):
    verifier = OBSERVER.OwnedOutputVerifier(captured_fast())
    update_rows(verifier.fast, 1)
    verifier.begin()
    verifier.actual.update()
    if corruption == "unowned":
        verifier.actual.moment[4][-1, 0] += 0.1
    elif corruption == "weight":
        verifier.actual.weight += 0.01
    elif corruption == "truncated_width":
        verifier.actual.moment.pop(8)
    elif corruption == "truncated_rows":
        verifier.actual.moment[4] = verifier.actual.moment[4][:-1]
    elif corruption == "local_coverage":
        verifier.fast.shape_groups[0][1][verifier.fast.rank].pop()
    else:
        selection, keys = verifier.actual.owned[4]
        verifier.actual.owned[4] = (slice(selection.start + 1, selection.stop + 1), keys)
    with pytest.raises((AssertionError, ValueError)):
        verifier.check_factors()


def test_owned_observer_constructor_rejects_jointly_truncated_buffers_keys_and_state(collectives):
    fast = captured_fast()
    actual = fast.output_diagonal
    key = actual.keys[4].pop()
    actual.bufs.pop(key)
    actual.moment[4] = actual.moment[4][:-1]
    with pytest.raises(ValueError, match="global inventory"):
        OBSERVER.OwnedOutputVerifier(fast)


def test_owned_observer_continuous24_and_restored22_have_exact_full_moments_and_factors(collectives, monkeypatch):
    monkeypatch.setattr(OBSERVER.dist, "all_gather_object", gather_equal_rank_records)
    continuous = OBSERVER.OwnedOutputVerifier(captured_fast(rank=7))
    for step in range(1, 23):
        assert complete_update(continuous, step)["owned_output_step"] == step
    stream = io.BytesIO()
    torch.save(continuous.actual.state_dict(), stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    fast = captured_fast(rank=7)
    fast.steps = 22
    fast.output_diagonal.load_state_dict(saved)
    resumed = OBSERVER.OwnedOutputVerifier(fast)
    for step in (23, 24):
        assert complete_update(continuous, step)["owned_output_step"] == step
        assert complete_update(resumed, step)["owned_output_step"] == step
        assert resumed.actual.weight == continuous.actual.weight
        assert resumed.actual.state_dict().keys() == {"moment", "weight"}
        for width, moment in continuous.actual.moment.items():
            torch.testing.assert_close(resumed.actual.moment[width], moment, rtol=0, atol=0)
        for key, (inner, outer) in resumed.actual.factors().items():
            assert inner is None
            torch.testing.assert_close(outer, continuous.actual.factors()[key][1], rtol=0, atol=0)


def test_owned_observer_finish_rejects_unobserved_upstream_stacks_after_successful_factor_checks(collectives):
    verifier = OBSERVER.OwnedOutputVerifier(captured_fast())
    observed_update(verifier, 1)
    with pytest.raises(ValueError, match="every owned shape"):
        verifier.finish()
    update_rows(verifier.fast, 2)
    verifier.begin()
    with pytest.raises(ValueError, match="every owned shape"):
        verifier.finish()


def gathered_records(step=24, apply_left=True, model=None, backend="torch", device_type="cpu"):
    model = DIMENSIONS if model is None else model
    weight = 0.0
    for _ in range(step):
        weight = SETTINGS["decay"] * weight + 1 - SETTINGS["decay"]
    return [
        {
            "schema": 1,
            "rank": rank,
            "world": WORLD,
            "step": step,
            "model": dict(model),
            "backend": backend,
            "device_type": device_type,
            **OBSERVER.owned_output_layout(model, WORLD, rank),
            "state_schema": ["moment", "weight"],
            "weight": weight,
            "checks": dict.fromkeys(OBSERVER.CHECKS, True),
            "max_errors": dict.fromkeys(OBSERVER.ERRORS, 0.0),
            "upstream": {
                **dict.fromkeys(OBSERVER.UPSTREAM_CHECKS, True),
                "upstream_null_exact": not apply_left,
                "upstream_owned_stacks_checked": 3,
                "upstream_output_equation_max_error": rank * 1e-8,
            },
        }
        for rank in range(WORLD)
    ]


def validate(records, step=24, apply_left=True):
    return OBSERVER.validate_owned_output_records(
        records,
        model=DIMENSIONS,
        world=WORLD,
        step=step,
        apply_left=apply_left,
        backend="torch",
        device_type="cpu",
    )


@pytest.mark.parametrize("apply_left", [False, True])
def test_owned_gathered_records_round_trip_preserves_all_rank_inventory_and_error_maxima(apply_left):
    records = json.loads(json.dumps(gathered_records(apply_left=apply_left), allow_nan=False))
    records[7]["max_errors"]["moment_torch"] = 1e-7
    result = validate(records, apply_left=apply_left)
    assert [record["rank"] for record in result["owned_output_records"]] == list(range(WORLD))
    assert result["owned_output_global_count"] == 56
    assert result["owned_output_max_errors"]["moment_torch"] == 1e-7
    assert result["owned_output_max_errors"]["upstream_output_equation"] == 7e-8
    assert result["owned_output_parity_passed"] is True
    with pytest.raises(ValueError):
        OBSERVER.validate_owned_output_records(
            records,
            model=DIMENSIONS,
            world=WORLD,
            step=24,
            apply_left=apply_left,
            backend="triton",
            device_type="cuda",
        )


@pytest.mark.parametrize("apply_left", [False, True])
def test_owned_observer_wire_schema_matches_live_continuous_and_resumed_history_contract(tmp_path, apply_left):
    """Production-shape wire fixtures test serialization, not GPU numerical approval."""
    model = {"layers": 32, "hidden": 512, "mlp": 2048}
    assert OBSERVER.CHECKS == CONTRACT.OWNED_CAPTURE_CHECKS
    assert OBSERVER.ERRORS == CONTRACT.OWNED_CAPTURE_ERRORS
    rows = []
    for step in range(1, 25):
        records = gathered_records(step, apply_left, model, backend="triton", device_type="cuda")
        records[7]["max_errors"]["moment_fp64"] = 2e-7
        records[7]["max_errors"]["factor_fp64"] = 3e-7
        result = OBSERVER.validate_owned_output_records(
            records,
            model=model,
            world=WORLD,
            step=step,
            apply_left=apply_left,
            backend="triton",
            device_type="cuda",
        )
        upstream = dict(records[0]["upstream"])
        upstream["upstream_output_equation_max_error"] = result["owned_output_max_errors"]["upstream_output_equation"]
        rows.append({"step": step, "diagonal_capture_parity_passed": True, **upstream, **result})
    serialized = [json.dumps(row, allow_nan=False) for row in rows]
    arm = CONTRACT.ARMS[2] if apply_left else CONTRACT.ARMS[3]
    histories = []
    for name, checkpoint in (("continuous", 24), ("resumed", 22)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "metrics_from_000000.jsonl").write_text("\n".join(serialized[:checkpoint]))
        if checkpoint == 22:
            (directory / "metrics_from_000022.jsonl").write_text("\n".join(serialized[22:]))
        summary = CONTRACT.require_owned_capture_history(directory, arm)
        assert summary["rank_updates"] == 192
        assert summary["max_moment_error"] == 2e-7
        assert summary["max_factor_error"] == 3e-7
        histories.append(summary)
    assert histories[0] == histories[1]
    rows[-1]["owned_output_records"][7]["upstream"]["upstream_pinned_input_exact"] = False
    (tmp_path / "resumed" / "metrics_from_000022.jsonl").write_text("\n".join(json.dumps(row) for row in rows[22:]))
    with pytest.raises(ValueError, match="upstream parity"):
        CONTRACT.require_owned_capture_history(tmp_path / "resumed", arm)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_rank",
        "duplicate_rank",
        "step",
        "global",
        "owned",
        "schema",
        "weight",
        "nan",
        "flag",
        "upstream",
        "exact",
    ],
)
def test_owned_gathered_records_reject_rank7_or_resumed_tail_corruption(corruption):
    records = copy.deepcopy(gathered_records())
    record = records[7]
    if corruption == "missing_rank":
        records.pop()
    elif corruption == "duplicate_rank":
        record["rank"] = 6
    elif corruption == "step":
        record["step"] = 23
    elif corruption == "global":
        record["global_shapes"][0]["moment_shape"][0] -= 1
    elif corruption == "owned":
        record["owned_keys"].pop()
    elif corruption == "schema":
        record["state_schema"] = ["moment"]
    elif corruption == "weight":
        record["weight"] += 1e-6
    elif corruption == "nan":
        record["max_errors"]["factor_fp64"] = float("nan")
    elif corruption == "flag":
        record["checks"]["full_moment_parity_passed"] = 1
    elif corruption == "upstream":
        record["upstream"]["upstream_pinned_input_exact"] = False
    else:
        record["max_errors"]["factor_same_state"] = 1e-8
    with pytest.raises(ValueError):
        validate(records)
