# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check selected-LR schedule preservation and generated pre-CUDA execution denial."""

import json
import os
import subprocess
import sys
from pathlib import Path

import prepare_selected_lr_full_budget as preparation
import pytest
from launch_wave5 import read, sha256
from update_endeavor import selected_lr_cpu_lines
from wave7_release import ARMS, candidate_inventory

TASK = Path(__file__).resolve().parents[1]


def test_full_budget_transfer_retains_schedule_auxiliary_settings_and_holdouts():
    reference = json.loads((TASK / "experiment/configs/muont_0p5m_seed0.json").read_text())
    reference["experiment"]["refresh_install_age"] = 1
    before = json.loads(json.dumps(reference))
    result = preparation.full_config(reference, {"matrix_lr": 0.06399999989260598})
    assert reference == before
    assert result["optimizer"]["lr"] == 0.06399999989260598
    assert {key: value for key, value in result["optimizer"].items() if key != "lr"} == {
        key: value for key, value in reference["optimizer"].items() if key != "lr"
    }
    assert result["schedule"] == reference["schedule"]
    assert result["training"] == reference["training"]
    assert result["data"] == reference["data"]
    assert result["root"] == reference["root"]
    assert result["statistics"] == reference["statistics"]
    assert "reference_validation_loss" not in result


def test_frozen_analysis_ignores_shadow_module_in_callers_directory(tmp_path, monkeypatch):
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    helpers.joinpath("analyzer.py").write_text("def analyze(): return {'source':'frozen'}\n")
    caller = tmp_path / "caller"
    caller.mkdir()
    caller.joinpath("analyzer.py").write_text("raise RuntimeError('mutable analyzer imported')\n")
    monkeypatch.chdir(caller)
    result = preparation.frozen_result(
        sys.executable,
        helpers,
        "import json; from analyzer import analyze; print(json.dumps(analyze()))",
        tmp_path,
    )
    assert result == {"source": "frozen"}


@pytest.mark.parametrize("module", ("train", "validate", "gpu_checks"))
def test_generated_gpu_entrypoint_refuses_before_execution_side_effect(tmp_path, module):
    source = tmp_path / "source/experiment"
    source.mkdir(parents=True)
    source.joinpath("__init__.py").write_text("")
    source.joinpath("records.py").write_text(
        "atomic_json=checkpoint_identity=manifest=restore_checkpoint=save_checkpoint=utc_now=None\n"
    )
    source.joinpath("reference.py").write_text("TASK=UPSTREAM=None\n")
    anchor = {
        "train": preparation.TRAIN_IMPORT,
        "validate": preparation.VALIDATE_IMPORT,
        "gpu_checks": preparation.GPU_IMPORT,
    }[module]
    source.joinpath(f"{module}.py").write_text(
        "import argparse\nfrom pathlib import Path\n" + anchor + "\ndef main():\n"
        "    parser=argparse.ArgumentParser()\n"
        "    parser.add_argument('--gpu',action='store_true')\n"
        "    args = parser.parse_args()\n"
        "    Path('gpu_started').touch()\n"
        "if __name__=='__main__': main()\n"
    )
    candidate = tmp_path / "candidate/experiment"
    candidate.mkdir(parents=True)
    for name in ("__init__.py", "records.py", "reference.py"):
        candidate.joinpath(name).write_bytes(source.joinpath(name).read_bytes())
    candidate.joinpath("selected_lr_owner.py").write_text(preparation.OWNER)
    candidate.joinpath(f"{module}.py").write_text(preparation.guarded_source(source.parent, f"experiment/{module}.py"))
    arguments = ["--gpu"] if module == "validate" else []
    result = subprocess.run(
        [sys.executable, "-m", f"experiment.{module}", *arguments],
        cwd=candidate.parent,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate.parent)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "RuntimeError: Selected-LR full-budget candidate has no GPU execution binding" in result.stderr
    assert not (candidate.parent / "gpu_started").exists()
    assert source.joinpath(f"{module}.py").read_text().count("require_selected_lr_owner") == 0
    if module == "validate":
        cpu = subprocess.run(
            [sys.executable, "-m", "experiment.validate"],
            cwd=candidate.parent,
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate.parent)},
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert cpu.returncode == 0
        assert (candidate.parent / "gpu_started").exists()


@pytest.fixture
def selected_source(tmp_path, monkeypatch):
    root = tmp_path / "output"
    previous = tmp_path / "source"
    choices = {arm: {"matrix_lr": 0.064 if arm in ("muont", "outer_leftdiag_fused") else 0.09050966784} for arm in ARMS}
    source = previous / "experiment"
    (source / "configs").mkdir(parents=True)
    reference = read(TASK / "experiment/configs/inner_outer_leftdiag_0p5m_seed0.json")
    reference["experiment"]["refresh_install_age"] = 1
    for arm in ARMS:
        config = json.loads(json.dumps(reference))
        config["experiment"]["arm"] = arm
        source.joinpath(f"configs/{arm}_0p5m_seed0.json").write_text(json.dumps(config))
    source.joinpath("cluster.json").write_text('{"wave_name":"wave7","environment":{}}')
    for module in ("train", "validate", "gpu_checks"):
        anchor = {
            "train": preparation.TRAIN_IMPORT,
            "validate": preparation.VALIDATE_IMPORT,
            "gpu_checks": preparation.GPU_IMPORT,
        }[module]
        source.joinpath(f"{module}.py").write_text(
            anchor + "\ndef main():\n    args = parser.parse_args()\n    return 42\n"
        )
    (source / "tests").mkdir()
    source.joinpath("tests/test_cpu_output.py").write_text(
        "from pathlib import Path\n"
        "def test_cpu_output():\n"
        "    output=Path('cpu_output')\n"
        "    output.write_text('done')\n"
        "    assert output.read_text()=='done'\n"
    )
    pinned = previous / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    pinned.joinpath("source.py").write_text("def numerical(): return 42\n")
    previous.joinpath("artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    frozen = {"candidate": str(previous), "files_sha256": candidate_inventory(previous), "data_receipt_sha256": "data"}
    parent = root / preparation.PARENT
    boundary = root / "campaigns/boundary/campaign.json"
    initial = root / "campaigns/initial/campaign.json"
    for path, value in (
        (parent, {"root": str(root), "parent_campaign": str(boundary)}),
        (boundary, {"parent_campaign": str(initial)}),
        (initial, {}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        for name in ("analysis.json", "swanlab-upload.json", "execution-binding.json"):
            path.with_name(name).write_text("{}")
    parent.with_name("analysis.json").write_text(
        json.dumps({"status": "complete", "completed_cumulative_family_lr_choices": choices})
    )
    (root / "run-preparation").mkdir()
    for name in ("wave10-launch-binding.json", "wave7-release-binding.json"):
        root.joinpath("run-preparation", name).write_text(json.dumps(frozen) if name.startswith("wave7") else "{}")
    selection = {"choices": choices, "files_sha256": {str(p): sha256(p) for p in preparation.evidence_paths(root)}}
    monkeypatch.setattr(preparation, "require_choices", lambda root: (frozen, selection))
    monkeypatch.setattr(preparation, "require_source", lambda *args: frozen)
    return root, previous, frozen


@pytest.mark.parametrize("fault", ("choice", "numerical", "evidence"))
def test_isolated_preparation_retains_predecessor_and_rejects_changed_evidence(selected_source, tmp_path, fault):
    root, previous, frozen = selected_source
    candidate = tmp_path / "candidate"
    result = preparation.prepare(root, candidate)
    assert candidate_inventory(previous) == frozen["files_sha256"]
    assert set(result["changed_files"]) == preparation.CHANGES
    assert preparation.require_preparation(candidate) == result
    assert (
        candidate.joinpath("artifacts/pinned-sources-materialized/source.py").read_bytes()
        == (previous / "artifacts/pinned-sources-materialized/source.py").read_bytes()
    )
    assert not (root / "runs").exists()
    with pytest.raises(FileExistsError):
        preparation.prepare(root, candidate)
    if fault == "choice":
        path = candidate / preparation.PREPARATION
        record = read(path)
        record["selection"]["choices"]["muont"]["matrix_lr"] = 0.128
        config_path = candidate / "experiment/configs/muont_0p5m_seed0.json"
        config = read(config_path)
        config["optimizer"]["lr"] = 0.128
        config_path.write_text(json.dumps(config))
        record["files_sha256"] = candidate_inventory(candidate)
        path.write_text(json.dumps(record))
    elif fault == "numerical":
        candidate.joinpath("artifacts/pinned-sources-materialized/source.py").write_text("def numerical(): return 99\n")
    else:
        (root / preparation.PARENT).parent.joinpath("swanlab-upload.json").write_text('{"status":"partial"}')
    with pytest.raises(ValueError):
        preparation.require_preparation(candidate)


@pytest.mark.parametrize("failure", ("archive", "post_cpu_provenance"))
def test_cpu_failure_preserves_log_and_receipt(selected_source, tmp_path, monkeypatch, failure):
    root, _, _ = selected_source
    candidate = tmp_path / "candidate"
    preparation.prepare(root, candidate)
    out = tmp_path / "cpu_evidence"
    run = preparation.subprocess.run
    archive = preparation.tarfile.open

    def run_cpu(*args, **kwargs):
        result = run(*args, **kwargs)
        if failure == "post_cpu_provenance":
            (root / preparation.PARENT).parent.joinpath("swanlab-upload.json").write_text('{"status":"partial"}')
        return result

    def open_archive(*args, **kwargs):
        if failure == "archive" and args[1] == "x":
            raise OSError("archive storage unavailable")
        return archive(*args, **kwargs)

    monkeypatch.setattr(preparation.subprocess, "run", run_cpu)
    monkeypatch.setattr(preparation.tarfile, "open", open_archive)
    with pytest.raises(RuntimeError):
        preparation.cpu(candidate, out)
    record = read(out / "receipt.json")
    assert record["status"] == "selected_lr_cpu_failed"
    assert record["exit_code"] == 0
    assert record["log_sha256"] == sha256(out / "candidate.log")
    assert (candidate / "cpu_output").read_text() == "done"
    assert record["archive_sha256"] == (None if failure == "archive" else sha256(out / "source.tar"))
    assert record["error"]


def test_document_ledger_retains_cpu_preparation_and_withholds_changed_archive(selected_source, tmp_path, monkeypatch):
    root, _, frozen = selected_source
    monkeypatch.setattr("update_endeavor.require_selected_lr_source", lambda *args: frozen)
    candidate = tmp_path / "candidate"
    preparation.prepare(root, candidate)
    out = root / "run-preparation/selected-lr-full-budget-cpu/prepared"
    preparation.cpu(candidate, out)
    published = "\n".join(selected_lr_cpu_lines(root))
    assert "selected_lr_full_budget_cpu_verified_gpu_denied" in published
    assert "GPU execution remains denied" in published
    assert str((out / "receipt.json").relative_to(root)) in published
    out.joinpath("source.tar").write_bytes(b"changed archive")
    withheld = "\n".join(selected_lr_cpu_lines(root))
    assert "acceptance is withheld" in withheld
    assert "selected_lr_full_budget_cpu_verified_gpu_denied" not in withheld
