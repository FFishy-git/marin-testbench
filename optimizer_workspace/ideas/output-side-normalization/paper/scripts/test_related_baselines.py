# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise related-baseline export, offline rebuild, and receipt integrity."""

import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ARMS = (
    "muon_control",
    "inner_outer_mlpdiag",
    "mousse_control",
    "mousse_paper",
    "go_muon_control",
    "go_muon_paper",
)


def _write_run(directory: Path, arm: str, seed: int, steps: int, lr: float, offset: float, crossover: bool = False):
    directory.mkdir(parents=True)
    batch_tokens = 524288
    stable_steps = steps // 5
    eval_steps = {1, 251, steps} if steps == 5120 else {1, steps}
    if steps == 5120:
        eval_steps |= set(range(2751, 5120, 250))
    history = []
    for step in range(1, steps + 1):
        delta = offset
        if crossover:
            delta = -0.20 if step < 180 else +0.035
        loss = 4.5 - 1.2 * (step / steps) + math.sin(step) * 0.005 + delta
        row = {
            "step": step,
            "tokens": step * batch_tokens,
            "train_loss": loss,
            "step_seconds": 1.2 + (0.3 if "go_muon" in arm else 0.0),
            "peak_allocated_bytes": 20 * (1024**3),
        }
        if step in eval_steps:
            row["selection_loss"] = loss + 0.02
            row["c4_loss"] = loss + 0.25
        if step == steps:
            row["end_to_end_seconds"] = steps * row["step_seconds"]
        history.append(row)
    manifest = {
        "status": "complete",
        "completed_steps": steps,
        "execution_check": False,
        "elapsed_seconds": history[-1]["end_to_end_seconds"],
        "config": {
            "experiment": {"arm": arm},
            "optimizer": {"lr": lr},
            "training": {"steps": steps, "batch_tokens": batch_tokens, "tokens": steps * batch_tokens, "seed": seed},
            "schedule": {"stable_steps": stable_steps},
        },
        "source_revision": "fixture",
        "data_receipt_sha256": "frozen-data",
        "gpu": "A100",
        "world_size": 8,
    }
    (directory / "run.json").write_text(json.dumps(manifest))
    (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))


def test_related_baselines_export_and_offline_rebuild(tmp_path):
    script = Path(__file__).with_name("plot_related_baselines.py")
    short_root = tmp_path / "short"
    full_root = tmp_path / "full"

    lrs = (0.016, 0.02262741696, 0.032)
    offsets = {
        "muon_control": 0.0,
        "inner_outer_mlpdiag": -0.010,
        "mousse_control": -0.001,
        "mousse_paper": 0.003,
        "go_muon_control": 0.031,
        "go_muon_paper": 0.032,
    }

    for lr_idx, lr in enumerate(lrs):
        for arm in ARMS:
            _write_run(
                short_root / "runs" / f"{arm}_0p5m_seed0_lr{lr_idx}",
                arm,
                seed=0,
                steps=320,
                lr=lr,
                offset=offsets[arm] - 0.05 * lr_idx,
            )

    for seed in (1, 2):
        for arm in ARMS:
            _write_run(
                full_root / "runs" / f"{arm}_0p5m_seed{seed}_lr2",
                arm,
                seed=seed,
                steps=5120,
                lr=lrs[2],
                offset=offsets[arm],
                crossover=arm.startswith("go_muon"),
            )

    data = tmp_path / "related.dat"
    receipt_path = tmp_path / "related.json"
    macros = tmp_path / "related_numbers.tex"
    table = tmp_path / "table_related.tex"
    figure = tmp_path / "related.pdf"

    cmd = [
        sys.executable,
        "-I",
        str(script),
        "--short-root",
        str(short_root),
        "--full-root",
        str(full_root),
        "--data",
        str(data),
        "--receipt",
        str(receipt_path),
        "--macros",
        str(macros),
        "--table",
        str(table),
        "--figure",
        str(figure),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    with data.open() as handle:
        exported = list(csv.DictReader(handle, delimiter=" "))
    assert len(exported) == 5120
    assert figure.read_bytes().startswith(b"%PDF-")
    assert r"\newcommand{\relGoCtrlFinalSelDeltaMean}" in macros.read_text()
    assert r"\begin{tabular}" in table.read_text()

    receipt = json.loads(receipt_path.read_text())
    assert receipt["full_runs"]["go_muon_control"]["per_seed"]["1"]["early_diag"]["sustained_positive_step"] == 180

    # Offline rebuild after removing raw runs
    shutil.rmtree(short_root)
    shutil.rmtree(full_root)
    figure.unlink()
    subprocess.run([*cmd, "--from-data"], check=True, capture_output=True, text=True)
    assert figure.read_bytes().startswith(b"%PDF-")

    # Tamper detection
    with data.open("a") as handle:
        handle.write("5121 0 " + " ".join(["0"] * 12) + "\n")
    rejected = subprocess.run([*cmd, "--from-data"], capture_output=True, text=True)
    assert rejected.returncode != 0
