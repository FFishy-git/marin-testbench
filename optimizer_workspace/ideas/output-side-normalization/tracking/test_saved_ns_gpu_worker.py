# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU staging and owner-refusal regressions; no CUDA replay execution."""

import hashlib
from pathlib import Path

import pytest
import saved_ns_gpu_owner as owner
import saved_ns_gpu_worker as worker
import torch


def test_source_clone_preserves_noncontiguous_batch_layout_and_independent_storage():
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
    result = worker.clone_source(source)
    torch.testing.assert_close(result, source, rtol=0, atol=0)
    assert result.stride() == source.stride()
    assert result.storage_offset() == source.storage_offset()
    assert result.data_ptr() != source.data_ptr()
    result[1, 3, 2] = -1
    assert source[1, 3, 2] == 23
    assert worker.tensor_layout(result)["stride"] == [12, 1, 4]
    assert worker.tensor_layout(result)["storage_nbytes"] == 24 * 4


def test_resident_selection_uses_identity_when_continuous_and_resumed_values_match():
    source = torch.ones(2, 3, 4)
    groups = {"continuous": {"source": source}, "resumed": {"source": source.clone()}}
    assert worker.source_mode(groups, source) == "continuous"
    assert worker.source_mode(groups, groups["resumed"]["source"]) == "resumed"
    with pytest.raises(ValueError, match="identity"):
        worker.source_mode(groups, source.clone())


def test_resident_value_check_detects_mutation_anywhere_in_the_full_batch():
    source = torch.ones(worker.replay.SAVED_SHAPE)
    actual = source.clone()
    worker.require_unchanged_source(actual, source)
    actual[-1, -1, -1] += 0.125
    with pytest.raises(ValueError, match="changed"):
        worker.require_unchanged_source(actual, source)
    actual[-1, -1, -1] = float("nan")
    with pytest.raises(ValueError, match="changed"):
        worker.require_unchanged_source(actual, source)


def test_output_evidence_round_trip_preserves_four_batches_and_never_overwrites(tmp_path):
    base = torch.zeros(worker.replay.SAVED_SHAPE)
    outputs = (base.clone(), base.clone(), base.clone(), base.clone())
    outputs[3][-1, -1, -1] = 0.125
    path = tmp_path / "process-0.pt"
    digest = worker.write_outputs(path, outputs)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    for actual, expected in zip(saved, outputs, strict=True):
        assert worker.replay.compare_outputs(actual, expected)["exact"]
    saved[0].zero_()
    assert saved[3][-1, -1, -1] == 0.125
    with pytest.raises(FileExistsError):
        worker.write_outputs(path, tuple(base.clone() for _ in range(4)))
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_owner_denial_precedes_cuda_and_saved_evidence_io(tmp_path, monkeypatch):
    def deny(*args, **kwargs):
        raise PermissionError("Synthetic unowned worker")

    def refuse_boundary(*args, **kwargs):
        raise AssertionError("Unowned worker reached CUDA or evidence I/O")

    monkeypatch.setattr(owner, "require_gpu_owner", deny)
    monkeypatch.setattr(torch.cuda, "device_count", refuse_boundary)
    monkeypatch.setattr(torch.cuda, "get_device_properties", refuse_boundary)
    monkeypatch.setattr(torch.cuda, "set_device", refuse_boundary)
    monkeypatch.setattr(Path, "read_bytes", refuse_boundary)
    monkeypatch.setattr(Path, "open", refuse_boundary)
    with pytest.raises(PermissionError):
        worker.run(tmp_path, "0" * 64, tmp_path / "state.json", 0)
    assert list(tmp_path.iterdir()) == []
