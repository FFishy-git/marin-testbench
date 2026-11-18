# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise deferred-launch refusal before any CUDA workload can be replaced."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("torch"):
    import seed2_companion


@unittest.skipUnless(importlib.util.find_spec("torch"), "Companion gates require the Torch validation environment")
class CompanionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        preparation = self.root / "run-preparation"
        preparation.mkdir()
        (preparation / "validation-wave4.json").write_text(json.dumps({}))
        if importlib.util.find_spec("torch"):
            config = seed2_companion.config_path("muont", 2, "0p5m")
            (preparation / "seed2-companion-queue.json").write_text(json.dumps({
                "node": "28.216.17.144", "config_sha256": seed2_companion.sha256(config),
                "validation_sha256": seed2_companion.sha256(preparation / "validation-wave4.json"),
                "helpers_sha256": {},
            }))
        self.previous = self.root / "runs/wave4/muont_1m"
        self.previous.mkdir(parents=True)
        (preparation / "wave4.json").write_text(json.dumps({
            "launches": [{
                "node": "28.216.17.144", "arm": "muont", "seed": 0, "batch": "1m",
                "pid": 10, "out": str(self.previous),
            }],
        }))

    def test_live_predecessor_refuses_before_gpu_commands(self):
        with patch(
            "seed2_companion.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "Ssl live", ""),
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "still running"):
                seed2_companion.launch(self.root)
        self.assertEqual([call.args[0][0] for call in run.call_args_list], ["ps"])
        self.assertFalse((self.root / "run-preparation/wave4-baseline.json").exists())
        state = json.loads((self.root / "run-preparation/seed2-companion-state.json").read_text())
        self.assertEqual(state["status"], "preflight")

    def test_failed_predecessor_never_releases_keeper(self):
        (self.previous / "run.json").write_text(json.dumps({
            "status": "failed", "completed_steps": 20, "config": {"training": {"steps": 2560}},
        }))
        with patch("seed2_companion.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "")) as run:
            with self.assertRaisesRegex(ValueError, "did not complete"):
                seed2_companion.launch(self.root)
        self.assertEqual([call.args[0][0] for call in run.call_args_list], ["ps"])
        self.assertFalse((self.root / "validation").exists())

    def test_unfinished_keeper_restoration_refuses_transfer(self):
        (self.previous / "run.json").write_text(json.dumps({
            "status": "complete", "completed_steps": 2560, "config": {"training": {"steps": 2560}},
        }))
        (self.previous.parent / "keeper-restore-28.216.17.144.log").write_text("training exited\n")
        with patch("seed2_companion.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "")) as run:
            with self.assertRaisesRegex(RuntimeError, "keeper watcher"):
                seed2_companion.launch(self.root)
        self.assertEqual([call.args[0][0] for call in run.call_args_list], ["ps"])
        self.assertFalse((self.root / "run-preparation/wave4-baseline.json").exists())


if __name__ == "__main__":
    unittest.main()
