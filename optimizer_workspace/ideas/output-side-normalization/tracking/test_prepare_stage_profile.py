# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute the prepared CLI denial and cleanup boundary against real source transformations."""

import ast
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import prepare_stage_profile
import pytest
from launch_wave5 import NODES, PYTHON, sha256
from wave7_release import candidate_inventory

PRODUCTION = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/candidates/upstream-full-model-production-20261117"
)


@pytest.fixture
def profiled_candidate(tmp_path):
    previous = tmp_path / "production"
    shutil.copytree(
        PRODUCTION / "experiment",
        previous / "experiment",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        PRODUCTION / "artifacts/pinned-sources-materialized",
        previous / "artifacts/pinned-sources-materialized",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    root = tmp_path / "root"
    (root / "run-preparation").mkdir(parents=True)
    helpers = root / "controllers/wave9-launch-20261117/tracking"
    helpers.mkdir(parents=True)
    for name in ("upstream_production.py", "analyze_upstream_run.py"):
        (helpers / name).write_text("# Frozen fixture boundary\n")
    (helpers / "upstream_validation.py").write_text('ARMS = ("muont",)\n')
    (helpers / "upstream_run_history.py").write_text(
        "import json\n"
        "def binding(root):\n"
        "    return json.loads((root / 'run-preparation/wave9-launch-binding.json').read_text())\n"
    )
    archive = root / "run-preparation/helpers.tar"
    with tarfile.open(archive, "x") as stream:
        for path in helpers.iterdir():
            stream.add(path, arcname=path.name)
    binding = {
        "status": "frozen_validated_launch_release",
        "root": str(root),
        "nodes": list(NODES),
        "candidate": str(previous),
        "files_sha256": candidate_inventory(previous),
        "helpers": str(helpers),
        "helpers_sha256": {path.name: sha256(path) for path in helpers.iterdir()},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "arms": ["muont"],
    }
    (root / "run-preparation/wave9-launch-binding.json").write_text(json.dumps(binding))
    candidate = tmp_path / "profile"
    prepare_stage_profile.prepare(root, candidate)
    return candidate


def test_profile_candidate_cli_refuses_execution_before_output_or_cuda(profiled_candidate):
    candidate = profiled_candidate
    completed = subprocess.run(
        [
            PYTHON,
            "-m",
            "experiment.train",
            "--config",
            str(candidate / "experiment/configs/muont_0p5m_seed0.json"),
            "--data",
            str(candidate / "missing-data"),
            "--data-receipt",
            str(candidate / "missing-receipt.json"),
            "--out",
            str(candidate / "uncreated-run"),
            "--max-steps",
            "220",
            "--profile-updates",
            "201",
            "220",
        ],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "Stage-profile candidate has no distinct GPU execution binding" in completed.stderr
    assert not (candidate / "uncreated-run").exists()


def test_profile_export_failure_still_cleans_engines_and_process_group(profiled_candidate):
    tree = ast.parse((profiled_candidate / "experiment/train.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    training_try = next(node for node in main.body if isinstance(node, ast.Try))
    statements = ast.Module(body=training_try.finalbody, type_ignores=[])
    events = []

    class BrokenProfiler:
        def finish(self):
            events.append("profile")
            raise OSError("Trace export failed")

    engines = [
        SimpleNamespace(
            side=SimpleNamespace(synchronize=lambda: events.append("side")),
            pool=SimpleNamespace(shutdown=lambda wait: events.append("pool")),
        )
        for _ in range(2)
    ]
    namespace = {
        "stage_profile": BrokenProfiler(),
        "toner_engines": lambda fast: engines,
        "fast": object(),
        "world": 8,
        "dist": SimpleNamespace(destroy_process_group=lambda: events.append("process_group")),
    }
    with pytest.raises(OSError, match="Trace export failed"):
        exec(compile(statements, "prepared_cleanup", "exec"), namespace)
    assert events == ["profile", "side", "pool", "side", "pool", "process_group"]
