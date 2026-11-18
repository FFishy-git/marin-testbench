# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for JSONL import and cloud-export validation."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from swanlab_refresh_upload import (
    TRAINING_METADATA,
    expected_columns,
    load_history,
    metric_row,
    run_name,
    verify_history,
)


class CloudExport:
    def __init__(self, history):
        self.history = history
        self.state = "FINISHED"
        self.url = "cloud-run"
        self.columns = expected_columns(history)
        self.profile = {
            "config": {
                **{key: {"value": value} for key, value in history.manifest["config"].items()},
                "training_record": {"value": {key: history.manifest[key] for key in TRAINING_METADATA}},
            }
        }

    def wrapper(self):
        return SimpleNamespace(ok=True)

    def metrics(self, *, keys, x_axis="step", **query):
        columns = []
        for key in keys:
            points = []
            for step, value in self.columns[key]:
                point = {"step": step, "value": value}
                if x_axis == "tokens":
                    point["index"] = self.history.metrics[step - 1]["tokens"]
                points.append(point)
            columns.append({"key": key, "metrics": points})
        return {"list": columns}


def fixture_history(root: Path, *, seed=0, batch="0p5m", wave="wave0", arm="muon"):
    batch_tokens = 524288 if batch == "0p5m" else 1048576
    name = run_name(arm, seed, batch)
    directory = root / "runs" / wave / name
    checkpoint = directory / "checkpoints/step_000002"
    checkpoint.mkdir(parents=True)
    manifest = {key: "recorded" for key in TRAINING_METADATA}
    manifest.update(
        status="complete",
        completed_steps=2,
        config={
            "training": {
                "steps": 2,
                "batch_tokens": batch_tokens,
                "tokens": 2 * batch_tokens,
                "world_size": 2,
                "seed": seed,
            },
            "experiment": {"arm": arm},
        },
    )
    rows = [
        {"step": 1, "tokens": batch_tokens, "train_loss": 4.0, "step_seconds": 0.2},
        {
            "step": 2,
            "tokens": 2 * batch_tokens,
            "train_loss": 3.0,
            "c4_loss": 3.5,
            "selection_loss": 3.25,
            "heldout_losses": {"heldout/dclm": 3.4},
            "input_gpu_ms_previous": {"roots": 1.2},
        },
    ]
    (directory / "run.json").write_text(json.dumps(manifest))
    (directory / "metrics_from_000000.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (checkpoint / "complete.json").write_text(json.dumps({"step": 2, "world": 2}))
    for rank in range(2):
        (checkpoint / f"rank_{rank}.pt").write_bytes(b"checkpoint fixture")
    return load_history(root, wave, name)


class UploadTest(unittest.TestCase):
    def test_refresh_events_cloud_roundtrip_preserves_all_original_scalars(self):
        with tempfile.TemporaryDirectory() as temporary:
            history = fixture_history(Path(temporary))
            history.metrics[1] = metric_row(
                {
                    "step": 2,
                    "tokens": 1048576,
                    "train_loss": 3.0,
                    "c4_loss": 3.5,
                    "selection_loss": 3.25,
                    "input_refresh_installations": [
                        {"group": "f", "index": 0, "source_step": 0, "installation_step": 1, "installed": True}
                    ],
                }
            )
            remote = CloudExport(history)
            result = verify_history(SimpleNamespace(run=lambda path: remote), "test/project/run", history)
            self.assertEqual(result["point_counts"]["train/loss"], 2)
            self.assertEqual(history.metrics[1]["runtime/input_refresh_installations/f/max_age"], 1)
            self.assertEqual(history.metrics[1]["runtime/input_refresh_installations/installed"], 1)
            self.assertTrue(result["all_scalar_values_exact"])

    def test_fused_wave5_histories_cloud_roundtrip_with_executed_arm_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm in ("inner_outer_mlpdiag", "inner_outer_mlpdiag_fused", "inner_outer_leftdiag_fused", "muont"):
                history = fixture_history(root, wave="wave5", arm=arm)
                remote = CloudExport(history)
                result = verify_history(SimpleNamespace(run=lambda path, run=remote: run), "test/project/run", history)
                self.assertEqual(result["point_counts"]["train/loss"], 2)
                self.assertEqual(history.manifest["config"]["experiment"]["arm"], arm)
                self.assertTrue(result["configuration_verified"])

    def test_seed_and_batch_histories_keep_distinct_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            histories = [fixture_history(root), fixture_history(root, seed=1), fixture_history(root, batch="1m")]
            self.assertEqual(len({history.run_id for history in histories}), 3)
            self.assertEqual(histories[1].arm, "muon_seed1")
            self.assertEqual(histories[2].arm, "muon_1m")
            self.assertEqual(histories[1].manifest["config"]["training"]["seed"], 1)
            for history in histories:
                remote = CloudExport(history)
                result = verify_history(SimpleNamespace(run=lambda path, run=remote: run), "test/project/run", history)
                self.assertTrue(result["token_axis_verified"])

    def test_sparse_history_and_cloud_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            history = fixture_history(Path(temporary))
            remote = CloudExport(history)
            value = 4.0
            for _ in range(2):
                value = math.nextafter(value, math.inf)
            remote.columns["train/loss"][0] = (1, value)
            api = SimpleNamespace(run=lambda path: remote)
            result = verify_history(api, "test/project/run", history)
            self.assertEqual(result["point_counts"]["train/loss"], 2)
            self.assertEqual(result["point_counts"]["eval/c4_loss"], 1)
            self.assertEqual(history.metrics[1]["eval/heldout/dclm"], 3.4)
            self.assertEqual(history.metrics[1]["runtime/input_gpu_ms_previous/roots"], 1.2)
            self.assertNotIn("eval/c4_loss", history.metrics[0])
            self.assertEqual(result["serialization_errors"]["train/loss"]["max_ulp"], 2)
            self.assertTrue(result["configuration_verified"])
            self.assertTrue(result["token_axis_verified"])
            self.assertFalse(result["all_scalar_values_exact"])

    def test_cloud_corruption_and_missing_points_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            history = fixture_history(Path(temporary))
            for damaged in (
                [(1, 4.0)],
                [(1, 4.0), (1, 3.0)],
                [(1, 4.0), (2, 3.01)],
            ):
                with self.subTest(damaged=damaged):
                    remote = CloudExport(history)
                    remote.columns["train/loss"] = damaged
                    api = SimpleNamespace(run=lambda path, run=remote: run)
                    with self.assertRaises(RuntimeError):
                        verify_history(api, "test/project/run", history)
            remote = CloudExport(history)
            remote.profile["config"]["training_record"]["value"]["data_receipt_sha256"] = "different data"
            with self.assertRaises(RuntimeError):
                verify_history(SimpleNamespace(run=lambda path: remote), "test/project/run", history)

    def test_incomplete_training_never_becomes_upload_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_history(root)
            metric_path = root / "runs/wave0/muon/metrics_from_000000.jsonl"
            rows = metric_path.read_text().splitlines()
            metric_path.write_text("\n".join((rows[0], rows[0])))
            with self.assertRaises(ValueError):
                load_history(root, "wave0", "muon")

    def test_short_lr_identity_preserves_full_scalars_and_trial_config_in_cloud_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = fixture_history(Path(temporary))
            short = replace(
                original,
                wave="fixed-age-study-lr_down",
                arm="muon_lr_down",
                run_id="fixed-age-study-muon-lr_down-source",
                provenance={**original.provenance, "campaign_sha256": "fixed-campaign"},
                manifest={
                    **original.manifest,
                    "config": {
                        **original.manifest["config"],
                        "experiment": {
                            **original.manifest["config"]["experiment"],
                            "tuning": {"trial": "lr_down", "lr_multiplier": 0.7071067811865475},
                        },
                    },
                },
            )
            cloud = CloudExport(short)
            result = verify_history(SimpleNamespace(run=lambda path: cloud), "test/project/run", short)
            self.assertEqual(result["point_counts"], {key: len(values) for key, values in cloud.columns.items()})
            self.assertEqual(result["final_metrics"], original.metrics[-1])
            self.assertTrue(result["configuration_verified"])
            cloud.profile["config"] = json.loads(json.dumps(cloud.profile["config"]))
            cloud.profile["config"]["experiment"]["value"]["tuning"]["trial"] = "lr_up"
            with self.assertRaises(RuntimeError):
                verify_history(SimpleNamespace(run=lambda path: cloud), "test/project/run", short)


if __name__ == "__main__":
    unittest.main()
