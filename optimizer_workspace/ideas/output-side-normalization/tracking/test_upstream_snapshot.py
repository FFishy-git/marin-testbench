# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify strict live captures and receipt-backed publication without touching GPUs."""

import copy
import json
import os
import shutil
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import analyze_upstream_snapshot
import pytest
from analyze_tuning import executed_sources
from compare_runs import WINDOW
from launch_wave5 import NODES, PYTHON, read, sha256
from selected_lr_validation import ARMS as SELECTED_LR_ARMS
from update_endeavor import ANCHOR, END, START, production_capture_lines, update
from upstream_validation import ARMS
from wave7_release import candidate_inventory


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def running_wave9(tmp_path):
    root = tmp_path
    candidate = root / "candidate"
    record(candidate / "experiment/upstream_manifest.json", {"revision": "fixture", "files": {}})
    (candidate / "experiment/train.py").write_text("# Frozen numerical source fixture\n")
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    base = {
        "model": {"sequence_length": 4096},
        "data": {"mixture": "same"},
        "experiment": {"arm": "muont"},
        "training": {
            "seed": 0,
            "world_size": 8,
            "heldout_windows": 64,
            "steps": 5120,
            "tokens": 2684354560,
            "batch_tokens": 524288,
        },
    }
    for arm in ARMS:
        config = copy.deepcopy(base)
        config["experiment"]["arm"] = arm
        record(candidate / f"experiment/configs/{arm}_0p5m_seed0.json", config)
    helpers = root / "controllers/wave9-launch-20261117/tracking"
    helpers.mkdir(parents=True)
    for name in ("upstream_production.py", "analyze_upstream_run.py"):
        (helpers / name).write_text("# Frozen observation fixture\n")
    (helpers / "upstream_validation.py").write_text(f"ARMS = {ARMS!r}\n")
    (helpers / "upstream_run_history.py").write_text(
        "import json\n"
        "def binding(root):\n"
        "    return json.loads((root / 'run-preparation/wave9-launch-binding.json').read_text())\n"
    )
    preparation = root / "run-preparation"
    preparation.mkdir()
    archive = preparation / "source.tar"
    with tarfile.open(archive, "x") as stream:
        for path in sorted(helpers.iterdir()):
            stream.add(path, arcname=path.name)
    provenance = {}
    for name in (
        "upstream-full-model-corrected-binding.json",
        "validation-upstream-full-model-corrected-gpu.json",
        "upstream-full-model-corrected-state.json",
        "prepared.json",
        "cpu.json",
        "data.json",
    ):
        path = preparation / name
        record(path, {"status": "fixture"})
        provenance[name] = path
    release_path = preparation / "wave9-launch-binding.json"
    frozen = {
        "status": "frozen_validated_launch_release",
        "root": str(root),
        "candidate": str(candidate),
        "helpers": str(helpers),
        "helpers_sha256": {path.name: sha256(path) for path in helpers.iterdir()},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "arms": list(ARMS),
        "nodes": list(NODES),
        "files_sha256": candidate_inventory(candidate),
        "data_receipt_sha256": sha256(provenance["data.json"]),
        "validation_binding_sha256": sha256(provenance["upstream-full-model-corrected-binding.json"]),
        "validation_sha256": sha256(provenance["validation-upstream-full-model-corrected-gpu.json"]),
        "state_sha256": sha256(provenance["upstream-full-model-corrected-state.json"]),
        "preparation": str(provenance["prepared.json"]),
        "preparation_sha256": sha256(provenance["prepared.json"]),
        "cpu_receipt": str(provenance["cpu.json"]),
        "cpu_receipt_sha256": sha256(provenance["cpu.json"]),
        "monitor_interval_seconds": 900,
    }
    record(release_path, frozen)
    sources = executed_sources(candidate, frozen)
    for index, arm in enumerate(ARMS):
        config = read(candidate / f"experiment/configs/{arm}_0p5m_seed0.json")
        directory = root / "runs/wave9" / arm
        record(
            directory / "run.json",
            {
                "status": "running",
                "config": config,
                "data_receipt_sha256": frozen["data_receipt_sha256"],
                "source_sha256": sources,
            },
        )
        (directory / "metrics_from_000000.jsonl").write_text(
            "".join(
                json.dumps({"step": step, "tokens": step * 524288, "train_loss": 5.0 - index / 100}) + "\n"
                for step in range(1, 2050 - index)
            )
        )
    return root


@pytest.fixture
def running_wave12(running_wave9):
    root = running_wave9
    frozen = read(root / "run-preparation/wave9-launch-binding.json")
    helpers = root / "controllers/wave12-launch-20261117/tracking"
    shutil.copytree(frozen["helpers"], helpers)
    frozen["helpers"] = str(helpers)
    candidate = Path(frozen["candidate"])
    helper = helpers / "selected_lr_run_history.py"
    helper.write_text(
        "import json\n"
        "def binding(root):\n"
        "    return json.loads((root/'run-preparation/wave12-launch-binding.json').read_text())\n"
    )
    frozen["helpers_sha256"][helper.name] = sha256(helper)
    frozen["arms"] = list(SELECTED_LR_ARMS)
    frozen["status"] = "frozen_selected_lr_production_after_original_gpu_validation"
    for old, arm in zip(ARMS, SELECTED_LR_ARMS, strict=True):
        config = read(candidate / f"experiment/configs/{old}_0p5m_seed0.json")
        config["experiment"]["arm"] = arm
        record(candidate / f"experiment/configs/{arm}_0p5m_seed0.json", config)
        directory = root / "runs/wave9" / old
        manifest = read(directory / "run.json")
        manifest["config"] = config
        record(directory / "run.json", manifest)
        if old != arm:
            directory.rename(root / "runs/wave9" / arm)
    frozen["files_sha256"] = candidate_inventory(candidate)
    for key, name in (
        ("validation_binding_sha256", "wave11-selected-lr-validation-binding.json"),
        ("validation_sha256", "validation-wave11-selected-lr-gpu.json"),
        ("state_sha256", "wave11-selected-lr-validation-state.json"),
    ):
        path = root / "run-preparation" / name
        record(path, {"status": "fixture"})
        frozen[key] = sha256(path)
    failed = root / "run-preparation/wave11-launch-binding.json"
    record(failed, {"status": "fixture"})
    frozen["recovery_of"] = str(failed)
    frozen["failure_evidence"] = {str(failed.relative_to(root)): sha256(failed)}
    record(root / "run-preparation/wave12-launch-binding.json", frozen)
    (root / "runs/wave9").rename(root / "runs/wave12")
    return root


def test_wave9_capture_reuses_exact_common_windows_after_live_histories_change(running_wave9):
    root = running_wave9
    observation = root / "observation"
    analyze_upstream_snapshot.capture(root, observation)
    result = analyze_upstream_snapshot.analyze_capture(observation)
    comparisons = result["analyses"][0]["comparisons"]
    upstream = comparisons["wave9/outer_leftdiag_upstream_fused"]
    assert [window["end_tokens"] for window in upstream["training_windows"]] == [WINDOW]
    assert upstream["training_windows"][0]["delta"] == pytest.approx(-0.02)
    assert upstream["final_comparison_eligible"] is False
    matched_null = result["analyses"][2]["comparisons"]["wave9/outer_leftdiag_upstream_fused"]
    assert matched_null["training_windows"][0]["delta"] == pytest.approx(0.01)
    path = root / "runs/wave9/muont/metrics_from_000000.jsonl"
    path.write_text(path.read_text() + '{"unfinished":')
    assert analyze_upstream_snapshot.analyze_capture(observation) == result
    with pytest.raises(FileExistsError):
        analyze_upstream_snapshot.capture(root, observation)


def test_wave10_capture_uses_bound_owner_and_retains_exact_large_windows(running_wave9):
    root = running_wave9
    release_path = root / "run-preparation/wave9-launch-binding.json"
    frozen = read(release_path)
    helpers = root / "controllers/wave10-launch-20261117/tracking"
    shutil.copytree(frozen["helpers"], helpers)
    (helpers / "owned_output_run_history.py").write_text(
        "import json\n"
        "def binding(root):\n"
        "    return json.loads((root/'run-preparation/wave10-launch-binding.json').read_text())\n"
    )
    frozen["helpers"] = str(helpers)
    frozen["helpers_sha256"]["owned_output_run_history.py"] = sha256(helpers / "owned_output_run_history.py")
    record(root / "run-preparation/wave10-launch-binding.json", frozen)
    (root / "runs/wave9").rename(root / "runs/wave10")
    observation = root / "observation10"
    analyze_upstream_snapshot.capture(root, observation, wave="wave10")
    result = analyze_upstream_snapshot.analyze_capture(observation)
    candidate = result["analyses"][0]["comparisons"]["wave10/outer_leftdiag_upstream_fused"]
    assert [window["end_tokens"] for window in candidate["training_windows"]] == [WINDOW]
    assert candidate["training_windows"][0]["delta"] == pytest.approx(-0.02)
    assert candidate["final_comparison_eligible"] is False
    assert read(observation / "observation.json")["source_release_sha256"] == sha256(
        root / "run-preparation/wave10-launch-binding.json"
    )
    path = root / "runs/wave10/muont/metrics_from_000000.jsonl"
    path.write_text(path.read_text() + "\n")
    assert analyze_upstream_snapshot.analyze_capture(observation) == result
    with pytest.raises(ValueError):
        analyze_upstream_snapshot.capture(root, root / "rejected10", wave="wave10")
    assert not (root / "rejected10").exists()


@pytest.mark.parametrize("wave", ("wave9", "wave10", "wave12"))
def test_captured_cohort_uses_common_prefix_evaluations_and_runtime(request, wave):
    root = request.getfixturevalue("running_wave12" if wave == "wave12" else "running_wave9")
    arms = SELECTED_LR_ARMS if wave == "wave12" else ARMS
    if wave == "wave10":
        frozen = read(root / "run-preparation/wave9-launch-binding.json")
        helpers = root / "controllers/wave10-launch-20261117/tracking"
        shutil.copytree(frozen["helpers"], helpers)
        frozen["helpers"] = str(helpers)
        (helpers / "owned_output_run_history.py").write_text(
            "import json\n"
            "def binding(root):\n"
            "    return json.loads((root/'run-preparation/wave10-launch-binding.json').read_text())\n"
        )
        frozen["helpers_sha256"]["owned_output_run_history.py"] = sha256(helpers / "owned_output_run_history.py")
        record(root / "run-preparation/wave10-launch-binding.json", frozen)
        (root / "runs/wave9").rename(root / "runs/wave10")
    for index, arm in enumerate(arms):
        rows = []
        for step in range(1, 2101 - 16 * index):
            row = {
                "step": step,
                "tokens": step * 524288,
                "train_loss": 5 - index / 100 if step <= 1024 else 4 + index / 10,
                "step_seconds": 1 + index / 20 if step <= 2052 else 100,
            }
            if step in (1025, 1200, 1600, 2000):
                row.update(c4_loss=2 + index / 100, selection_loss=3 + index / 50)
            if arm != arms[-1] and step == 1800:
                row["c4_loss"] = 1000 if arm == arms[2] else 2
            rows.append(row)
        root.joinpath("runs", wave, arm, "metrics_from_000000.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    observation = root / "cohort"
    analyze_upstream_snapshot.capture(root, observation, wave=wave)
    original = {path.relative_to(observation): path.read_bytes() for path in observation.rglob("*") if path.is_file()}
    result = analyze_upstream_snapshot.captured_cohort(observation)
    assert result["common_tokens"] == 2052 * 524288
    assert result["captured_observed_updates"] == dict(zip(arms, (2100, 2084, 2068, 2052), strict=True))
    for index, arm in enumerate(arms):
        item = result["comparisons"][f"{wave}/{arm}"]
        assert [window["end_tokens"] for window in item["training_windows"]] == [WINDOW, 2 * WINDOW]
        assert [window["delta"] for window in item["training_windows"]] == pytest.approx([-index / 100, index / 10])
        assert item["runtime"]["arm"]["shared_step_seconds"] == pytest.approx(2052 * (1 + index / 20))
        assert item["final_comparison_eligible"] is False
        for key, delta in (("c4_loss", index / 100), ("selection_loss", index / 50)):
            evaluation = item["evaluations"][key]
            assert evaluation["late_tokens"] == [step * 524288 for step in (1200, 1600, 2000)]
            assert evaluation["mean_late_delta"] == pytest.approx(delta)
            assert evaluation["final_value"] is None
    assert {path.relative_to(observation): path.read_bytes() for path in observation.rglob("*") if path.is_file()} == (
        original
    )


def test_wave10_ledger_uses_receipt_time_and_rejects_mislabelled_capture(running_wave9):
    root = running_wave9
    frozen = read(root / "run-preparation/wave9-launch-binding.json")
    for name in (
        "owned-output-trace-binding.json",
        "validation-owned-output-trace-gpu.json",
        "owned-output-trace-state.json",
    ):
        record(root / "run-preparation" / name, {"status": "fixture"})
    frozen["validation_binding_sha256"] = sha256(root / "run-preparation/owned-output-trace-binding.json")
    frozen["validation_sha256"] = sha256(root / "run-preparation/validation-owned-output-trace-gpu.json")
    frozen["state_sha256"] = sha256(root / "run-preparation/owned-output-trace-state.json")
    helpers = root / "controllers/wave10-launch-20261117/tracking"
    shutil.copytree(frozen["helpers"], helpers)
    frozen["helpers"] = str(helpers)
    (helpers / "owned_output_run_history.py").write_text(
        "import json\n"
        "def binding(root):\n"
        "    return json.loads((root/'run-preparation/wave10-launch-binding.json').read_text())\n"
    )
    frozen["helpers_sha256"]["owned_output_run_history.py"] = sha256(helpers / "owned_output_run_history.py")
    record(root / "run-preparation/wave10-launch-binding.json", frozen)
    (root / "runs/wave9").rename(root / "runs/wave10")
    latest = None
    for label, observed_time in (("z-older", "2026-11-17T10:00:00+00:00"), ("a-newer", "2026-11-17T10:01:00+00:00")):
        output = root / "run-preparation/wave10-observations" / label
        analyze_upstream_snapshot.capture(root, output, wave="wave10")
        observed = read(output / "observation.json")
        observed["time"] = observed_time
        record(output / "observation.json", observed)
        record(output / "analysis.json", analyze_upstream_snapshot.analyze_capture(output))
        latest = output
    published = "\n".join(production_capture_lines(root, "wave10"))
    assert "wave10-observations/a-newer/observation.json" in published
    assert "wave10-observations/z-older/observation.json" not in published
    assert "All-four common observed prefix: 1,072,693,248 tokens; 1 complete" in published
    observed = read(latest / "observation.json")
    observed["status"] = "captured_strict_wave9_observation"
    record(latest / "observation.json", observed)
    withheld = "\n".join(production_capture_lines(root, "wave10"))
    assert "Saved Wave-10 capture refused" in withheld
    assert "Complete 536,870,912-token training-window deltas" not in withheld


@pytest.mark.parametrize("wave", ("wave9", "wave12"))
@pytest.mark.parametrize("fault", ["partial", "blank", "gap", "source", "data", "complete", "resumed"])
def test_production_capture_rejects_bad_live_evidence_without_creating_output(request, wave, fault):
    root = request.getfixturevalue("running_wave12" if wave == "wave12" else "running_wave9")
    arms = SELECTED_LR_ARMS if wave == "wave12" else ARMS
    directory = root / "runs" / wave / arms[-1]
    path = directory / "metrics_from_000000.jsonl"
    if fault == "partial":
        path.write_text(path.read_text() + '{"unfinished":')
    elif fault == "blank":
        path.write_text(path.read_text() + "\n")
    elif fault == "gap":
        path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    elif fault == "resumed":
        (directory / "metrics_from_000022.jsonl").write_text("{}\n")
    else:
        path = directory / "run.json"
        manifest = read(path)
        key, value = {
            "source": ("source_sha256", {}),
            "data": ("data_receipt_sha256", "changed"),
            "complete": ("status", "complete"),
        }[fault]
        manifest[key] = value
        record(path, manifest)
    original = sha256(path)
    output = root / "observation"
    with pytest.raises(ValueError):
        analyze_upstream_snapshot.capture(root, output, wave=wave)
    assert not output.exists()
    assert sha256(path) == original


@pytest.mark.parametrize("fault", ["history", "source", "comparison"])
def test_wave9_ledger_rederives_capture_and_withholds_altered_evidence(running_wave9, fault):
    root = running_wave9
    observation = root / "run-preparation/wave9-observations/capture"
    analyze_upstream_snapshot.capture(root, observation)
    result = analyze_upstream_snapshot.analyze_capture(observation)
    analysis_path = observation / "analysis.json"
    record(analysis_path, result)
    document = root / "ENDEAVOR.md"
    prose = f"# Manual research\n\n{ANCHOR}\n"
    document.write_text(prose)
    update(document, root, datetime.now(UTC))
    text = document.read_text()
    assert "Wave-9 Strict Captured Comparison" in text
    assert "536,870,912-token training-window deltas: -0.020000" in text
    assert "Final C4 delta" not in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    if fault == "history":
        path = observation / "muont/metrics_from_000000.jsonl"
        path.write_text(path.read_text() + "\n")
    elif fault == "source":
        path = root / "run-preparation/wave9-launch-binding.json"
        path.write_text(path.read_text() + "\n")
    else:
        result["analyses"][0]["comparisons"]["wave9/outer_leftdiag_upstream_fused"]["training_windows"][0]["delta"] = -1
        record(analysis_path, result)
    update(document, root, datetime.now(UTC))
    text = document.read_text()
    assert "captured results withheld" in text
    assert "Reconstructed comparison:" not in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose


def test_document_entrypoint_captures_each_cycle_and_preserves_previous_capture_on_refusal(running_wave9):
    root = running_wave9
    document = root / "ENDEAVOR.md"
    document.write_text(f"# Manual research\n\n{ANCHOR}\n")
    old_capture = root / "run-preparation/wave9-observations/capture-9999"
    analyze_upstream_snapshot.capture(root, old_capture)
    recorded = read(old_capture / "observation.json")
    recorded["time"] = "2026-11-17T00:00:00+00:00"
    record(old_capture / "observation.json", recorded)
    record(old_capture / "analysis.json", analyze_upstream_snapshot.analyze_capture(old_capture))
    command = [
        PYTHON,
        str(Path(__file__).with_name("update_endeavor.py")),
        "--root",
        str(root),
        "--document",
        str(document),
    ]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run(command, env=environment, check=True, capture_output=True, text=True, timeout=30)
    observations = sorted((root / "run-preparation/wave9-observations").glob("*/observation.json"))
    assert len(observations) == 2
    previous = observations[0].parent
    assert previous != old_capture
    assert str(observations[0].relative_to(root)) in document.read_text()
    assert str((old_capture / "observation.json").relative_to(root)) not in document.read_text()
    recorded_hash = sha256(previous / "analysis.json")
    path = root / "runs/wave9/muont/metrics_from_000000.jsonl"
    path.write_text(path.read_text() + '{"unfinished":')
    refused = subprocess.run(command, env=environment, check=True, capture_output=True, text=True, timeout=30)
    assert json.loads(refused.stdout)["status"] == "wave9_capture_refused"
    assert sorted((root / "run-preparation/wave9-observations").glob("*/observation.json")) == observations
    assert sha256(previous / "analysis.json") == recorded_hash
    assert "Wave-9 Strict Captured Comparison" in document.read_text()
    path.write_text(path.read_text().removesuffix('{"unfinished":'))
    subprocess.run(command, env=environment, check=True, capture_output=True, text=True, timeout=30)
    observations = sorted((root / "run-preparation/wave9-observations").glob("*/observation.json"))
    assert len(observations) == 3
    assert sha256(previous / "analysis.json") == recorded_hash
    assert str(observations[-2].relative_to(root)) in document.read_text()


def test_document_capture_skips_finished_histories_without_relabeling_them(running_wave9):
    root = running_wave9
    manifest = root / "runs/wave9/muont/run.json"
    state = read(manifest)
    state["status"] = "complete"
    record(manifest, state)
    result = analyze_upstream_snapshot.refresh_running_capture(root, datetime.now(UTC))
    assert result["status"] == "wave9_capture_not_all_running"
    assert not (root / "run-preparation/wave9-observations").exists()


def test_wave12_document_cycle_preserves_manual_prose_and_last_capture_on_partial_write(running_wave12):
    root = running_wave12
    document = root / "ENDEAVOR.md"
    prose = f"# Manual research\n\n{ANCHOR}\n"
    document.write_text(prose)
    command = [
        PYTHON,
        str(Path(__file__).with_name("update_endeavor.py")),
        "--root",
        str(root),
        "--document",
        str(document),
    ]
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    subprocess.run(command, env=environment, check=True, capture_output=True, text=True, timeout=30)
    observations = list((root / "run-preparation/wave12-observations").glob("*/observation.json"))
    assert len(observations) == 1
    observation = observations[0].parent
    cohort = analyze_upstream_snapshot.captured_cohort(observation)
    assert [item["end_tokens"] for item in cohort["comparisons"]["wave12/muont"]["training_windows"]] == [WINDOW]
    text = document.read_text()
    assert "### Wave-12 Strict Captured Comparison" in text
    assert "wave12/inner_outer_leftdiag" in text
    assert "Final C4 delta" not in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose
    captured_hash = sha256(observation / "analysis.json")
    path = root / "runs/wave12/muont/metrics_from_000000.jsonl"
    path.write_text(path.read_text() + '{"unfinished":')
    refused = subprocess.run(command, env=environment, check=True, capture_output=True, text=True, timeout=30)
    statuses = [json.loads(line)["status"] for line in refused.stdout.splitlines()]
    assert "wave12_capture_refused" in statuses
    assert list((root / "run-preparation/wave12-observations").glob("*/observation.json")) == observations
    assert sha256(observation / "analysis.json") == captured_hash
    assert str(observations[0].relative_to(root)) in document.read_text()
    assert "Final C4 delta" not in document.read_text()


def test_wave12_capture_rejects_changed_helper_before_import_side_effects(running_wave12):
    root = running_wave12
    release = read(root / "run-preparation/wave12-launch-binding.json")
    marker = root / "unexpected-import"
    helper = Path(release["helpers"]) / "selected_lr_run_history.py"
    helper.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    output = root / "rejected"
    with pytest.raises(ValueError):
        analyze_upstream_snapshot.capture(root, output, wave="wave12")
    assert not marker.exists()
    assert not output.exists()


@pytest.mark.parametrize("wave", ("wave9", "wave12"))
def test_bad_capture_timestamp_refuses_only_its_ledger_section(running_wave12, wave):
    root = running_wave12
    output = root / f"run-preparation/{wave}-observations/capture"
    analyze_upstream_snapshot.refresh_running_capture(root, datetime.now(UTC), wave="wave12")
    record(output / "observation.json", {"status": f"captured_strict_{wave}_observation", "time": "invalid"})
    document = root / "ENDEAVOR.md"
    prose = f"# Manual research\n\n{ANCHOR}\n"
    document.write_text(prose)
    update(document, root, datetime.now(UTC))
    text = document.read_text()
    assert "Saved capture discovery refused" in text
    assert "Source-bound saved release" in text
    assert text.split(START)[0] + text.split(END)[1].lstrip("\n") == prose


def test_wave12_matching_helper_hashes_cannot_publish_after_binding_refusal(running_wave12):
    root = running_wave12
    analyze_upstream_snapshot.refresh_running_capture(root, datetime.now(UTC), wave="wave12")
    release_path = root / "run-preparation/wave12-launch-binding.json"
    frozen = read(release_path)
    helper = Path(frozen["helpers"]) / "selected_lr_run_history.py"
    helper.write_text("def binding(root):\n    raise ValueError('saved source evidence refused')\n")
    frozen["helpers_sha256"][helper.name] = sha256(helper)
    record(release_path, frozen)
    digest = sha256(release_path)
    record(root / "run-preparation/wave12.json", {"launches": [{"out": str(root / "runs/wave12/muont")}]})
    record(
        root / "run-preparation/monitor-wave12/latest.json",
        {
            "binding_sha256": digest,
            "status": "complete",
            "runs": [
                {
                    "name": "muont",
                    "node": "node",
                    "status": "complete",
                    "latest_metrics": {"step": 5120},
                    "planned_steps": 5120,
                }
            ],
        },
    )
    record(root / "run-preparation/analysis-wave12.json", {"source_release_sha256": digest, "analyses": []})
    record(
        root / "run-preparation/swanlab-upload-wave12.json",
        {"status": "verified", "runs": {"muont": {"status": "verified"}}},
    )
    document = root / "ENDEAVOR.md"
    document.write_text(f"# Manual research\n\n{ANCHOR}\n")
    update(document, root, datetime.now(UTC))
    text = document.read_text()
    assert "Release/launch counts withheld" in text
    assert "monitor identity differs; progress and keeper summaries withheld" in text
    assert "analysis identity differs; loss and runtime summaries withheld" in text
    assert "all 1 histories verified" not in text
    assert "### Wave-12 Strict Captured Comparison" not in text
    assert "5120 / 5120" not in text


def test_wave12_capture_rejects_release_replacement_during_frozen_verification(running_wave12):
    root = running_wave12
    release_path = root / "run-preparation/wave12-launch-binding.json"
    frozen = read(release_path)
    helper = Path(frozen["helpers"]) / "selected_lr_run_history.py"
    helper.write_text(
        "import json\n"
        "def binding(root):\n"
        "    path=root/'run-preparation/wave12-launch-binding.json'\n"
        "    saved=json.loads(path.read_text())\n"
        "    path.write_text(json.dumps(saved)+'\\n\\n')\n"
        "    return saved\n"
    )
    frozen["helpers_sha256"][helper.name] = sha256(helper)
    record(release_path, frozen)
    before = sha256(release_path)
    output = root / "rejected"
    with pytest.raises(ValueError):
        analyze_upstream_snapshot.capture(root, output, wave="wave12")
    assert read(release_path) == frozen
    assert sha256(release_path) != before
    assert not output.exists()
