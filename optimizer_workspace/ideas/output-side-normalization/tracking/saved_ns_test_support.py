# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist synthetic saved traces for authenticated loader and result reconstruction tests."""

from pathlib import Path

import analyze_fused_continuation_stages as analyzer
import torch
from selected_lr_release_owner import sha256
from watch_wave5 import write

SAVED_SHAPE = (16, 512, 512)


def saved_evidence(root: Path, layout: str = "contiguous") -> tuple[Path, list[Path]]:
    candidate = root / "candidate"
    muon = candidate / "artifacts/pinned-sources-materialized/Toner-synthetic/src/toner/fast/muon.py"
    muon.parent.mkdir(parents=True)
    muon.write_text("ABC = (3.4445, -4.7750, 2.0315)\n")
    upstream_path = candidate / "experiment/upstream_manifest.json"
    config_path = candidate / "experiment/config.json"
    upstream_path.parent.mkdir(parents=True)
    write(upstream_path, {"revision": "synthetic", "files": {"src/toner/fast/muon.py": sha256(muon)}})
    write(
        config_path,
        {"optimizer": {"ns_mode": "tpu", "ns_steps": 10, "ns_eps": 1e-5}, "runtime": {"deterministic": True}},
    )
    binding_path = root / "run-preparation/fused-continuation-binding-20261118.json"
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    write(
        binding_path,
        {
            "candidate": str(candidate),
            "files_sha256": {
                str(path.relative_to(candidate)): sha256(path) for path in (muon, upstream_path, config_path)
            },
        },
    )
    inventories, selected = {}, []
    for mode in analyzer.MODES:
        inventory = {
            str(path): "0" * 64
            for rank in range(8)
            for update in (22, 23, 24)
            for trace in (analyzer._trace_path(root, mode, rank, update),)
            for path in (trace, trace.with_suffix(".json"))
        }
        path = analyzer._trace_path(root, mode, 0, 23)
        path.parent.mkdir(parents=True)
        value = torch.full(SAVED_SHAPE, 1.0 if mode == "continuous" else 2.0)
        if layout == "transposed":
            value[-1, -1, -2] = 7
            value = value.transpose(-2, -1)
        elif layout == "offset":
            value = torch.full((17, 512, 512), 1.0)[1:]
        elif layout == "wrong_shape":
            value = value[:, :, :-1].clone()
        records = [
            {"stage": "matrix/after_update", "values": {"trace": {"ns_source": value, "ns_output": value.clone()}}}
            for _ in range(3)
        ]
        provenance = {
            "identity": {"source": "synthetic", "config": "same", "data": "same"},
            "start_step": 0 if mode == "continuous" else 22,
        }
        torch.save({"schema": 1, "rank": 0, "update": 23, "provenance": provenance, "records": records}, path)
        write(
            path.with_suffix(".json"),
            {
                "status": "saved_continuation_intermediates_not_numerical_acceptance",
                "rank": 0,
                "update": 23,
                "provenance": provenance,
                "records": 3,
                "stages": ["matrix/after_update"] * 3,
                "trace_sha256": sha256(path),
            },
        )
        inventory.update({str(item): sha256(item) for item in (path, path.with_suffix(".json"))})
        inventories[mode] = {"files_sha256": inventory}
        selected.extend((path, path.with_suffix(".json")))
    commands = [
        {"protocol": protocol, "mode": mode, "exit_code": 0, "command": [protocol, mode], "trajectory": {}}
        for protocol in ("control", "trace")
        for mode in ("continuous", "interrupted", "resumed")
    ]
    receipt_path = root / analyzer.RECEIPT_RELATIVE
    write(
        receipt_path,
        {
            "status": analyzer.COMPLETE_STATUS,
            "root": str(root),
            "candidate": str(candidate),
            "config": str(config_path),
            "execution_binding_sha256": sha256(binding_path),
            "world": 8,
            "boundary": 22,
            "first_resumed": 23,
            "endpoint": 24,
            "tolerance": {"rtol": 1e-5, "atol": 1e-6},
            "gpu_commands": commands,
            "commands": [command["command"] for command in commands],
            "analysis": {"trace_transparent": True, "extra_checkpoint_transparent": True, "traces": inventories},
        },
    )
    return receipt_path, [binding_path, muon, upstream_path, config_path, *selected]
