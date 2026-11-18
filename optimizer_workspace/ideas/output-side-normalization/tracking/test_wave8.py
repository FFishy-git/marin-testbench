# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check that the combined candidate cannot transfer GPUs from stale or wrong-node evidence."""

import json
from pathlib import Path

import pytest
import torch
import wave8
from benchmark_diagonal import source_definitions


def test_combined_controller_wrong_node_refuses_before_keeper_transfer(tmp_path):
    helper = tmp_path / "wave8.py"
    helper.write_bytes(Path(wave8.__file__).read_bytes())
    definitions = source_definitions(
        helper,
        ("validate",),
        {
            "binding": lambda root: {"candidate": str(tmp_path)},
            "local_node": lambda: "28.216.17.9",
            "NODE": wave8.NODE,
            "Path": Path,
        },
    )
    with pytest.raises(RuntimeError, match=r"original \.139"):
        definitions["validate"](tmp_path)
    assert not (tmp_path / "run-preparation/wave8-validation-state.json").exists()


def test_combined_candidate_refuses_changed_uncombined_reference(tmp_path):
    previous, candidate = tmp_path / "previous", tmp_path / "candidate"
    (previous / "experiment").mkdir(parents=True)
    (candidate / "experiment").mkdir(parents=True)
    before = previous / "experiment/block_toner.py"
    after = candidate / "experiment/block_toner.py"
    before.write_text(
        "def stack_unit_rms(x):\n    return x\n"
        "def block_update_stack(x):\n    return x\n"
        "def block_update(x):\n    return x\n"
    )
    after.write_text(before.read_text().replace("return x", "return -x", 1))
    original = {"candidate": str(previous), "files_sha256": {"experiment/block_toner.py": wave8.sha256(before)}}
    definitions = source_definitions(
        Path(wave8.__file__),
        ("definition", "require_candidate"),
        {
            "ast": wave8.ast,
            "Path": Path,
            "CHANGED_FILES": {"experiment/block_toner.py"},
            "require_source": lambda root, receipt: original,
            "candidate_inventory": lambda path: {"experiment/block_toner.py": wave8.sha256(after)},
        },
    )
    with pytest.raises(ValueError, match="Uncombined reference"):
        definitions["require_candidate"](tmp_path, candidate)
    assert before.read_text().count("return x") == 3


def test_actual_validation_rejects_changed_command_even_with_success_flags(tmp_path):
    preparation = tmp_path / "run-preparation"
    preparation.mkdir()
    source = tmp_path / "candidate"
    configurations = {}
    for arm in wave8.ARMS:
        path = source / f"experiment/configs/{arm}_0p5m_seed0.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"arm": arm}))
        configurations[arm] = wave8.sha256(path)
    owner = {"controller_pid": 1234, "controller_start_ticks": "5678"}
    (preparation / "wave8-validation-state.json").write_text(json.dumps(owner))
    receipt = {
        "owner": owner,
        "source_sha256": {},
        "data_receipt_sha256": "data",
        "environment": wave8.ENVIRONMENT,
        "config_sha256": configurations,
        "checks": dict.fromkeys(wave8.GATES, True),
        "evidence": {
            "real_data_smokes": {arm: {} for arm in wave8.ARMS},
            "gpu_commands": [{"command": ["unrelated"], "exit_code": 0}] * 13,
        },
    }
    (preparation / "validation-wave8-gpu.json").write_text(json.dumps(receipt))
    frozen = {"candidate": str(source), "source_sha256": {}, "data_receipt_sha256": "data"}
    with pytest.raises(ValueError, match="coverage is incomplete"):
        wave8.require_validation(tmp_path, frozen)


def test_frozen_checkpoint_comparator_checks_all_eight_ranks(tmp_path):
    source = Path(wave8.__file__).parents[1] / "experiment/records.py"
    comparisons = source_definitions(source, ("compare_tree", "compare_continuation"), {"torch": torch})
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full.mkdir()
    resumed.mkdir()
    for rank in range(8):
        checkpoint = {
            "step": 24,
            "next_global_sequence": 3072,
            "identity": {"source": "bound"},
            "model": {"weight": torch.tensor([float(rank)])},
        }
        torch.save(checkpoint, full / f"rank_{rank}.pt")
        torch.save(checkpoint, resumed / f"rank_{rank}.pt")
    result = comparisons["compare_continuation"](full, resumed, 8)
    assert result == {"tensor_comparisons": 8, "max_absolute_error": 0.0}
    changed = torch.load(resumed / "rank_7.pt", weights_only=False)
    changed["model"]["weight"].add_(0.01)
    torch.save(changed, resumed / "rank_7.pt")
    with pytest.raises(AssertionError):
        comparisons["compare_continuation"](full, resumed, 8)
