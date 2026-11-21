# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json

import pytest
import torch
from experiment import launch, related_configure
from experiment.capture import build_model
from experiment.configure import configuration
from experiment.launch import continuation_steps, wave_entries
from experiment.optimizer import build, fast_runtime_state, restore_fast, toner_engines
from experiment.records import compare_tree, sha256
from experiment.related_configure import comparison_configuration, prepare_comparison


@pytest.mark.parametrize("arm", ["muon_control", "mousse_control", "mousse_paper", "go_muon_control", "go_muon_paper"])
def test_runner_routing_and_checkpoint_restore_preserve_updates(arm):
    config = comparison_configuration(arm, batch="0p5m", seed=0, microbatch=4, lr_multiplier=1.0, stage="short")
    config["model"].update(hidden=4, mlp=8, heads=1, layers=1, sequence_length=4, vocab_size=16)
    config["optimizer"]["ns_mode"] = "f32"
    model = build_model(config, positions=[0, 2], row_capacity=2, rows_per_microbatch=2, device="cpu")
    optimizers, fast, clipping = build(
        model,
        config,
        None,
        positions=[0, 2],
        rows_per_microbatch=2,
        capacity=2,
        rows_total=2,
        world=1,
        rank=0,
        device="cpu",
    )
    for parameter in fast.owned.values():
        parameter.grad = torch.arange(parameter.numel()).reshape(parameter.shape).float() / 10 + 0.1
    if arm.startswith("go_muon"):
        fast.begin_step(0)
        fast.select_microbatch(0)
        for buffer in (*fast.input_rows.values(), *fast.output_rows.values()):
            buffer.copy_(torch.arange(buffer.numel()).reshape(buffer.shape).to(buffer.dtype) / 10 + 1)
        fast.finish_backward()
    fast.step()
    stream = io.BytesIO()
    torch.save(fast_runtime_state(fast), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=False)
    before = {key: parameter.detach().clone() for key, parameter in fast.matrices.items()}
    fast.begin_step(1)
    fast.select_microbatch(0)
    fast.finish_backward()
    fast.step()
    for key, parameter in fast.matrices.items():
        with torch.no_grad():
            parameter.copy_(before[key])
    restore_fast(fast, checkpoint)
    comparisons = []
    compare_tree(fast_runtime_state(fast), checkpoint, "baseline", comparisons)
    assert comparisons
    assert not toner_engines(fast)
    matrix_ids = {id(parameter) for parameter in fast.params}
    assert matrix_ids.isdisjoint({id(parameter) for parameter in clipping})
    assert {id(parameter) for parameter in clipping} | matrix_ids == {id(parameter) for parameter in model.parameters()}
    assert optimizers[0] is fast and isinstance(optimizers[1], torch.optim.AdamW)
    for key, parameter in fast.matrices.items():
        torch.testing.assert_close(parameter, before[key], rtol=0, atol=0)


def test_comparison_configs_preserve_workload_and_complete_short_schedule():
    configs = [
        comparison_configuration(arm, batch="0p5m", seed=2, microbatch=4, lr_multiplier=1.0, stage="short")
        for arm in ("muon_control", "inner_outer_mlpdiag", "go_muon_control", "mousse_control")
    ]
    for config in configs:
        assert config["training"]["steps"] * config["training"]["batch_tokens"] == config["training"]["tokens"]
        assert config["schedule"]["stable_steps"] + config["schedule"]["decay_steps"] == config["training"]["steps"]
    for field in ("model", "training", "data", "schedule", "runtime"):
        assert all(config[field] == configs[0][field] for config in configs[1:])
    historical = configuration("inner_outer_mlpdiag", "0p5m", 2, 4)
    assert "matrix_decay" not in historical["experiment"]
    for config in configs:
        assert config["experiment"]["matrix_decay"] == "isotropic"
        assert config["optimizer"]["auxiliary_lr"] == historical["optimizer"]["auxiliary_lr"]
    assert continuation_steps(configs[2]) == (10, 5)


def test_prepared_wave_resolves_configs_and_refuses_mismatched_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(related_configure, "TASK", tmp_path)
    monkeypatch.setattr(launch, "TASK", tmp_path)
    directory = tmp_path / "experiment" / "comparisons"
    artifacts = prepare_comparison(
        directory, profile="controlled", stage="short", batch="0p5m", seeds=[0], microbatch=4, lr_multipliers=[1.0]
    )
    manifest = directory / "wave_seed0_lr0.json"
    _, entries = wave_entries(manifest)
    assert len(entries) == len(artifacts)
    path = entries[2]["path"]
    original = path.read_text()
    changed = json.loads(original)
    changed["training"]["seed"] = 3
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs from its configuration"):
        wave_entries(manifest)


def test_remote_preflight_rejects_stale_selected_comparison_config(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "TASK", tmp_path)
    source = tmp_path / "experiment" / "train.py"
    config = tmp_path / "experiment" / "comparisons" / "go.json"
    config.parent.mkdir(parents=True)
    source.write_text("pass\n")
    config.write_text('{"optimizer": "go_muon"}\n')
    expected = {str(path.relative_to(tmp_path)): sha256(path) for path in (source, config)}
    stale = dict(expected)
    stale[str(config.relative_to(tmp_path))] = "0" * 64

    def remote(_node, arguments):
        if arguments[0] == "nvidia-smi":
            if "--query-compute-apps=pid,process_name,used_gpu_memory" in arguments:
                return ""
            return "\n".join(f"{index}, NVIDIA A100-SXM4-40GB, 0, 0" for index in range(8))
        return json.dumps(stale) if "hashlib" in arguments[-1] else "present"

    monkeypatch.setattr(launch, "remote", remote)
    inspected = launch.inspect("node", None, None, [{"path": config}])
    assert "Remote experiment code/configuration hashes differ" in inspected["blockers"]
    stale.update(expected)
    inspected = launch.inspect("node", None, None, [{"path": config}])
    assert "Remote experiment code/configuration hashes differ" not in inspected["blockers"]
