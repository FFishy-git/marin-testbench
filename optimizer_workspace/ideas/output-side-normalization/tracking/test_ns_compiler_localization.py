# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check source-derived finite maps and trace-transparency refusal on CPU."""

import copy
from contextlib import contextmanager
from pathlib import Path

import ns_compiler_localization as localization
import pytest
import torch
from benchmark_diagonal import source_definitions

PINNED = "artifacts/pinned-sources-materialized/Toner-dab7f3777c75e740536df822f50ffa1d929a340c" "/src/toner/fast/muon.py"
PARENT = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/run-preparation/ns-training-path-owner-cpu/closure-corrected-20261119T0738"
)


@pytest.fixture
def source_path():
    archived = Path(__file__).resolve().parents[1] / PINNED
    return archived if archived.is_file() else PARENT / PINNED


def uncompiled(numerical):
    return numerical


def matrix_batch(shape):
    count = shape[0] * shape[1] * shape[2]
    return torch.sin(torch.arange(count, dtype=torch.float32).reshape(shape) * 0.73) + 0.1


@pytest.mark.parametrize("shape", ((2, 3, 5), (2, 5, 3), (2, 4, 4), (1, 1, 3)))
@pytest.mark.parametrize("steps", (5, 10))
def test_localization_source_splits_and_traces_match_unmodified_pinned_map(source_path, shape, steps):
    maps = localization.compiler_maps(source_path, uncompiled, mode="f32", steps=steps, eps=1e-5)
    original = source_definitions(source_path, ("ABC", "_bmm", "newton_schulz_batched"), {"torch": torch})
    source = matrix_batch(shape)
    before, rng = source.clone(), torch.get_rng_state().clone()
    expected = original["newton_schulz_batched"](source, steps, 1e-5, mode="f32")
    for boundary in localization.BOUNDARIES:
        result, traced = maps.controls[boundary](source), maps.traces[boundary](source)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(traced["output"], expected, rtol=0, atol=0)
        localization.require_trace(traced, steps, "f32")
        normalized = source / (torch.linalg.matrix_norm(source, dim=(-2, -1), keepdim=True) + 1e-5)
        torch.testing.assert_close(traced["records"][0]["values"]["normalized"], normalized, rtol=0, atol=0)
        for record in traced["records"]:
            if record["stage"] in localization.PRODUCT_STAGES:
                values = record["values"]
                torch.testing.assert_close(
                    values["product"], torch.bmm(values["left_operand"], values["right_operand"]), rtol=0, atol=0
                )
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("fault", ("output", "missing_stage", "operand_dtype", "operand_shape", "nonfinite"))
def test_localization_withholds_missing_or_nontransparent_stage_evidence(source_path, fault):
    maps = localization.compiler_maps(source_path, uncompiled, mode="f32", steps=5, eps=1e-5)
    source = matrix_batch((2, 3, 5))
    control, expected = maps.controls["eager"](source), maps.traces["eager"](source)
    actual = copy.deepcopy(expected)
    if fault == "output":
        actual["output"][0, 0, 0] += 0.01
    elif fault == "missing_stage":
        actual["records"].pop(3)
    elif fault == "operand_dtype":
        actual["records"][3]["values"]["left_operand"] = actual["records"][3]["values"]["left_operand"].double()
    elif fault == "operand_shape":
        actual["records"][3]["values"]["left_operand"] = actual["records"][3]["values"]["left_operand"][:, :1]
    else:
        actual["records"][3]["values"]["product"][0, 0, 0] = float("nan")
    if fault in ("missing_stage", "operand_dtype", "operand_shape"):
        with pytest.raises(ValueError):
            localization.guarded_comparison(control, actual, control, expected, steps=5, mode="f32")
    else:
        report = localization.guarded_comparison(control, actual, control, expected, steps=5, mode="f32")
        assert report["stage_interpretation_permitted"] is False
        assert report["stage_comparison"] is None
        assert report["controls"]["exact"] is True


def test_localization_reconstructs_first_changed_bf16_operand_and_preserves_failed_output(source_path):
    maps = localization.compiler_maps(source_path, uncompiled, mode="f32", steps=5, eps=1e-5)
    expected = copy.deepcopy(maps.traces["eager"](matrix_batch((2, 3, 5))))
    for record in expected["records"]:
        if record["stage"] in localization.PRODUCT_STAGES:
            for name in ("left_operand", "right_operand"):
                record["values"][name] = record["values"][name].to(torch.bfloat16)
    actual = copy.deepcopy(expected)
    first_product = actual["records"][3]
    first_product["values"]["left_operand"][0, 0, 0] += 0.0625
    actual["output"][0, 0, 0] += 0.125
    report = localization.guarded_comparison(actual["output"], actual, expected["output"], expected, steps=5, mode="tpu")
    assert report["stage_interpretation_permitted"] is True
    changed = report["stage_comparison"]
    assert changed["first_changed_stage"]["stage"] == "gram"
    assert changed["first_changed_bf16_operand"] == {"stage": "gram", "iteration": 0, "operand": "left_operand"}
    assert changed["output"]["within_tolerance"] is False
    assert changed["output"]["changed_elements"] == 1


def test_localization_real_dynamo_compilation_retains_control_and_trace_outputs(source_path):
    maps = localization.compiler_maps(
        source_path,
        lambda fn: torch.compile(fn, backend="eager", fullgraph=True, dynamic=False),
        mode="f32",
        steps=5,
        eps=1e-5,
    )
    source = matrix_batch((2, 3, 5))
    expected, trace = maps.controls["eager"](source), maps.traces["eager"](source)
    for boundary in localization.BOUNDARIES[1:]:
        actual, observed = maps.controls[boundary](source), maps.traces[boundary](source)
        report = localization.guarded_comparison(actual, observed, expected, trace, steps=5, mode="f32")
        assert report["stage_interpretation_permitted"] is True
        assert report["controls"]["exact"] is True
        assert report["stage_comparison"]["first_changed_stage"] is None


def test_localization_rejects_changed_pinned_source_before_building_maps(source_path, tmp_path):
    path = tmp_path / "muon.py"
    path.write_bytes(source_path.read_bytes().replace(b"3.4445", b"3.4446"))
    with pytest.raises(ValueError):
        localization.compiler_maps(path, uncompiled, mode="f32", steps=5, eps=1e-5)


@pytest.mark.parametrize("timing", ("during_guard", "after_guard"))
def test_localization_path_replacement_never_executes_unverified_definitions(source_path, tmp_path, monkeypatch, timing):
    path, retained = tmp_path / "muon.py", tmp_path / "retained.py"
    replacement, side_effect = tmp_path / "replacement.py", tmp_path / "unverified-write"
    source = source_path.read_bytes()
    path.write_bytes(source)
    malicious = (
        f"ABC = (__import__('pathlib').Path({str(side_effect)!r}).write_text('unverified'), (9.0, 9.0, 9.0))[1]"
    ).encode()
    replacement.write_bytes(source.replace(b"ABC = (3.4445, -4.7750, 2.0315)", malicious))
    reference = source_definitions(source_path, ("ABC", "_bmm", "newton_schulz_batched"), {"torch": torch})
    bound_file = localization.replay.analyzer._bound_file

    @contextmanager
    def replace_at_boundary(path, expected_sha256):
        with bound_file(path, expected_sha256) as stream:
            if timing == "during_guard":
                path.rename(retained)
                replacement.rename(path)
            yield stream
        if timing == "after_guard":
            path.rename(retained)
            replacement.rename(path)

    monkeypatch.setattr(localization.replay.analyzer, "_bound_file", replace_at_boundary)
    try:
        if timing == "during_guard":
            with pytest.raises(ValueError):
                localization.source_namespace(path)
        else:
            namespace, *_ = localization.source_namespace(path)
            batch = matrix_batch((2, 3, 5))
            expected = reference["newton_schulz_batched"](batch, 5, 1e-5, mode="f32")
            actual = namespace["newton_schulz_batched"](batch, 5, 1e-5, mode="f32")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert namespace["ABC"] == reference["ABC"]
            for name in ("_bmm", "newton_schulz_batched"):
                assert namespace[name].__code__.co_filename == str(path)
        assert not side_effect.exists()
    finally:
        if retained.exists():
            path.unlink()
            retained.rename(path)


def protocol_fixture(source_path):
    maps = localization.compiler_maps(source_path, uncompiled, mode="f32", steps=10, eps=1e-5)
    source = matrix_batch((1, 2, 3))
    groups = {
        "continuous": {"source": source, "output": maps.controls["eager"](source)},
        "resumed": {"source": source + 0.02, "output": maps.controls["eager"](source + 0.02)},
    }

    def retain(coordinate, source):
        traced = copy.deepcopy(maps.traces["eager"](source))
        if coordinate["mode"] == "tpu":
            for record in traced["records"]:
                if record["stage"] in localization.PRODUCT_STAGES:
                    for name in ("left_operand", "right_operand"):
                        record["values"][name] = record["values"][name].to(torch.bfloat16)
        return [{"phase": phase, "control": traced["output"], "trace": traced} for phase in localization.PHASES]

    return groups, retain


@pytest.mark.parametrize("index", (0, 1))
def test_localization_opposite_order_protocol_retains_independent_phases_and_repeat_comparisons(source_path, index):
    groups, retain = protocol_fixture(source_path)
    records = localization.retain_protocol(groups, index, retain)
    assert len(records) == 128
    assert records[0]["source"] == ("resumed" if index else "continuous")
    assert records[0]["boundary"] == (localization.BOUNDARIES[-1] if index else "eager")
    report = localization.summarize_protocol(records, index)
    assert report["all_traces_transparent_exact_finite"]
    assert report["all_control_phases_exact_finite"]
    assert report["all_same_input_repeats_exact_finite"]
    records[3]["control"][0, 0, 0] += 0.25
    changed = localization.summarize_protocol(records, index)
    assert changed["all_traces_transparent_exact_finite"] is False
    assert changed["all_control_phases_exact_finite"] is False
    assert changed["all_same_input_repeats_exact_finite"] is False
    torch.testing.assert_close(
        records[0]["control"], retain(records[0], groups[records[0]["source"]]["source"])[0]["control"]
    )


@pytest.mark.parametrize("fault", ("source", "storage", "phase", "partial"))
def test_localization_protocol_refuses_mutation_and_incomplete_retained_batches(source_path, fault):
    groups, original = protocol_fixture(source_path)

    def retain(coordinate, source):
        result = original(coordinate, source)
        if fault == "source":
            groups["resumed"]["source"].add_(1)
        elif fault == "storage":
            groups["resumed"]["source"] = groups["resumed"]["source"].clone()
        elif fault == "phase":
            result.pop()
        else:
            result[0]["control"] = result[0]["control"][:, :1]
        return result

    with pytest.raises(ValueError):
        localization.retain_protocol(groups, 0, retain)
