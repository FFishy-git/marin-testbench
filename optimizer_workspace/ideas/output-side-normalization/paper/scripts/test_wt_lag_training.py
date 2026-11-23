# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise exact export, offline rendering, and receipt integrity through the CLI."""

import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def test_training_export_preserves_every_loss_and_rebuilds_offline(tmp_path):
    script = Path(__file__).with_name("plot_wt_lag_training.py")
    runs = tmp_path / "runs"
    arms = ("inner_outer_mlpdiag_lag", "inner_outer_sym_mlpdiag_lag")
    losses = []
    for index, arm in enumerate(arms):
        directory = runs / "wave_sym_fastlag" / arm
        directory.mkdir(parents=True)
        history = [
            {
                "step": step,
                "tokens": step * 524288,
                "train_loss": 3.0 + math.sin(step) * 0.01 + index * (0.02 if step == 713 else 0.001),
            }
            for step in range(1, 5121)
        ]
        losses.append([row["train_loss"] for row in history])
        manifest = {
            "status": "complete",
            "completed_steps": 5120,
            "execution_check": False,
            "config": {
                "experiment": {"arm": arm},
                "training": {"steps": 5120, "batch_tokens": 524288, "seed": 0},
                "schedule": {"stable_steps": 1024},
            },
            "command": ["trainer", "--arm", arm],
            "source_revision": "fixture",
            "source_sha256": {"train.py": "frozen-source"},
            "data": "shared-token-order",
            "data_receipt_sha256": "frozen-data",
            "gpu": "A100",
            "world_size": 8,
        }
        (directory / "run.json").write_text(json.dumps(manifest))
        (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    data = tmp_path / "training.dat"
    receipt_path = tmp_path / "training.json"
    figure = tmp_path / "training.pdf"
    command = [
        sys.executable,
        "-I",
        str(script),
        "--runs-root",
        str(runs),
        "--data",
        str(data),
        "--receipt",
        str(receipt_path),
        "--figure",
        str(figure),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    with data.open() as handle:
        exported = list(csv.DictReader(handle, delimiter=" "))
    assert len(exported) == 5120
    assert [int(row["step"]) for row in exported] == list(range(1, 5121))
    assert [int(row["tokens"]) for row in exported] == [step * 524288 for step in range(1, 5121)]
    for values, field in zip(losses, ("wt_mlp_lag", "wt_sym_lag"), strict=True):
        assert [float(row[field]) for row in exported] == values
    expected_differences = [b - a for a, b in zip(*losses, strict=True)]
    assert [float(row["sym_minus_mlp"]) for row in exported] == expected_differences
    receipt = json.loads(receipt_path.read_text())
    assert [window["sym_minus_mlp"] for window in receipt["paired_window_means"]] == [
        statistics.fmean(expected_differences[start : start + 1024]) for start in range(0, 5120, 1024)
    ]
    assert [source["command"] for source in receipt["sources"]] == [["trainer", "--arm", arm] for arm in arms]
    assert figure.read_bytes().startswith(b"%PDF-")

    original_data, original_receipt = data.read_bytes(), receipt_path.read_bytes()
    shutil.rmtree(runs)
    figure.unlink()
    subprocess.run([*command, "--from-data"], check=True, capture_output=True, text=True)
    assert figure.read_bytes().startswith(b"%PDF-")
    assert data.read_bytes() == original_data
    assert receipt_path.read_bytes() == original_receipt

    offline_figure = figure.read_bytes()
    with data.open("a") as handle:
        handle.write("5121 0 3 3 0\n")
    rejected = subprocess.run([*command, "--from-data"], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert figure.read_bytes() == offline_figure
