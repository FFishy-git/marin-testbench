# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor state follows fresh process observations, not stale manifests."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from snapshot import snapshot


def fixture(root: Path, *, manifest_status="complete") -> Path:
    directory = root / "runs/wave/arm"
    directory.mkdir(parents=True)
    preparation = root / "run-preparation"
    preparation.mkdir()
    (preparation / "wave.json").write_text(
        json.dumps(
            {
                "launches": [{"out": str(directory), "arm": "arm", "node": "test-node", "pid": 123}],
            }
        )
    )
    (directory / "run.json").write_text(
        json.dumps(
            {
                "status": manifest_status,
                "completed_steps": 2,
                "config": {"training": {"steps": 2, "seed": 0, "batch_tokens": 524288}},
            }
        )
    )
    (directory / "metrics_from_000000.jsonl").write_text(
        json.dumps({"step": 1, "tokens": 524288, "train_loss": 3.0})
        + "\n"
        + json.dumps({"step": 2, "tokens": 1048576, "train_loss": 2.9})
        + "\n"
    )
    return directory


class SnapshotTest(unittest.TestCase):
    def test_partial_wave_never_triggers_completed_wave_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            path = root / "run-preparation/wave.json"
            record = json.loads(path.read_text())
            record["expected_launches"] = 4
            path.write_text(json.dumps(record))
            with patch("snapshot.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "")):
                result = snapshot(root, "wave")
            self.assertEqual(result["runs"][0]["status"], "complete")
            self.assertEqual(result["status"], "partial")

    def test_live_process_overrides_complete_manifest_and_ignores_partial_last_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = fixture(root)
            with (directory / "metrics_from_000000.jsonl").open("a") as stream:
                stream.write('{"step":3,"tokens":')
            observed = subprocess.CompletedProcess(
                [], 0, f"123 Sl 01:00 python -m torch.distributed.run --out {directory}", ""
            )
            with patch("snapshot.subprocess.run", return_value=observed):
                result = snapshot(root, "wave")
            self.assertEqual(result["status"], "active")
            self.assertEqual(result["runs"][0]["status"], "running")
            self.assertEqual(result["runs"][0]["latest_metrics"]["step"], 2)

    def test_network_failure_or_reused_pid_never_proves_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            for observed in (
                subprocess.CompletedProcess([], 255, "", "Connection failed"),
                subprocess.CompletedProcess([], 0, "123 Sl 01:00 python unrelated", ""),
                subprocess.TimeoutExpired("ssh", 30),
            ):
                with self.subTest(observed=observed):
                    options = (
                        {"side_effect": observed} if isinstance(observed, Exception) else {"return_value": observed}
                    )
                    with patch("snapshot.subprocess.run", **options):
                        result = snapshot(root, "wave")
                    self.assertEqual(result["status"], "active")
                    self.assertEqual(result["runs"][0]["status"], "unknown")

    def test_confirmed_missing_process_distinguishes_complete_and_failed_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = fixture(root)
            missing = subprocess.CompletedProcess([], 1, "", "")
            with patch("snapshot.subprocess.run", return_value=missing):
                result = snapshot(root, "wave")
            self.assertEqual(result["status"], "complete")
            manifest = json.loads((directory / "run.json").read_text())
            manifest["status"] = "failed"
            (directory / "run.json").write_text(json.dumps(manifest))
            with patch("snapshot.subprocess.run", return_value=missing):
                result = snapshot(root, "wave")
            self.assertEqual(result["status"], "needs_attention")
            self.assertEqual(result["runs"][0]["status"], "stopped_incomplete")


if __name__ == "__main__":
    unittest.main()
