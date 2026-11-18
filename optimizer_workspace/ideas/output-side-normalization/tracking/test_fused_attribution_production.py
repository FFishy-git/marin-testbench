# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise persisted production controller archives and refusal before GPU execution."""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import fused_attribution_production as production
import pytest
from fused_attribution_production_test_support import archive, parent, record
from launch_wave5 import read, sha256


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    original = parent(tmp_path)
    candidate, cpu_out = tmp_path / "production", tmp_path / "candidate-cpu"
    real_run = subprocess.run
    outcome = {"code": 0, "failure": None, "disrupt": None}

    def boundary(command, **kwargs):
        if command not in (production.preparation.cpu_command(), production.cpu_command()):
            return real_run(command, **kwargs)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        kwargs["stdout"].write("synthetic CPU process outcome\n")
        if command == production.cpu_command():
            if outcome["disrupt"] is not None:
                outcome["disrupt"](Path(kwargs["cwd"]))
            if outcome["failure"] is not None:
                raise outcome["failure"]
            return subprocess.CompletedProcess(command, outcome["code"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", boundary)
    production.preparation.prepare(original.root, candidate)
    production.preparation.cpu(original.root, candidate, cpu_out)
    return SimpleNamespace(
        root=original.root,
        candidate=candidate,
        cpu=cpu_out / "receipt.json",
        out=tmp_path / "controller-cpu",
        outcome=outcome,
        real_run=real_run,
    )


@pytest.mark.parametrize("code", (0, 7))
def test_controller_cpu_preserves_outcome_and_nine_candidate_bound_fixtures(accepted, code):
    item = accepted
    item.outcome["code"] = code
    if code:
        with pytest.raises(RuntimeError):
            production.controller_cpu(item.root, item.cpu, item.out)
        result = read(item.out / "receipt.json")
    else:
        result = production.controller_cpu(item.root, item.cpu, item.out)
        assert production.require_controller_cpu(item.root, item.out / "receipt.json", item.cpu) == result
    assert result["status"] == (production.CONTROLLER_CPU_STATUS if code == 0 else "needs_attention")
    assert result["exit_code"] == code
    assert len(result["fixture_files_sha256"]) == 9
    assert result["fixture_files_sha256"] == {name: sha256(item.candidate / name) for name in production.fixture_names()}
    assert result["archive_sha256"] == sha256(item.out / "controller-source.tar")
    assert result["log_sha256"] == sha256(item.out / "controller.log")
    assert not production.paths(item.root, "wave13")[0].exists()
    assert not (item.root / "runs").exists()


@pytest.mark.parametrize(
    "fault", ("spawn", "timeout", "archive", "helper", "fixture", "source", "receipt", "runtime_verification")
)
def test_controller_cpu_preserves_spawn_archive_and_source_failures(accepted, fault):
    item = accepted
    original_digest = sha256(item.cpu)
    if fault == "spawn":
        item.outcome["failure"] = OSError("CPU executable unavailable")
    elif fault == "timeout":
        item.outcome["failure"] = subprocess.TimeoutExpired(production.cpu_command(), 900)
    else:

        def disrupt(directory):
            if fault == "archive":
                (directory / "controller-source.tar").mkdir()
            elif fault == "helper":
                (directory / "tracking/fused_attribution_production_contract.py").write_text("# changed\n")
            elif fault == "fixture":
                (directory / production.fixture_names()[0]).write_text("{}\n")
            elif fault == "source":
                (item.candidate / "experiment/numerical.py").write_text("CHANGED = True\n")
            elif fault == "receipt":
                with item.cpu.open("a") as stream:
                    stream.write("\n")
            else:
                frozen = read(item.root / "run-preparation/fused-attribution-validation-binding-20261118.json")
                helper = Path(frozen["helpers"]) / "fused_attribution_validation_controller.py"
                helper.write_text("raise RuntimeError('source verifier failed')\n")

        item.outcome["disrupt"] = disrupt
    with pytest.raises(RuntimeError):
        production.controller_cpu(item.root, item.cpu, item.out)
    result = read(item.out / "receipt.json")
    assert result["status"] == "needs_attention"
    assert result["error"]
    assert result["exit_code"] == {"spawn": 127, "timeout": 124}.get(fault, 0)
    assert result["candidate_cpu_receipt_sha256"] == original_digest
    assert result["log_sha256"] == sha256(item.out / "controller.log")
    assert result["archive_sha256"] == (None if fault == "archive" else sha256(item.out / "controller-source.tar"))
    assert not production.paths(item.root, "wave13")[0].exists()
    assert not (item.root / "runs").exists()


@pytest.mark.parametrize("fault", ("candidate", "fixture", "omitted_helper", "extra_helper"))
def test_controller_cpu_refuses_repaired_receipts_with_wrong_independent_inventory(accepted, fault):
    item = accepted
    result = production.controller_cpu(item.root, item.cpu, item.out)
    if fault == "candidate":
        result["candidate"] = str(item.candidate.parent / "wrong-candidate")
    elif fault == "fixture":
        name = production.fixture_names()[0]
        record(item.out / name, {"wrong_lr": True})
        result["fixture_files_sha256"][name] = sha256(item.out / name)
    elif fault == "omitted_helper":
        name = "fused_attribution_validation_controller.py"
        result["helpers_sha256"].pop(name)
        (item.out / "tracking" / name).unlink()
    else:
        (item.out / "tracking/unbound.py").write_text("UNBOUND = True\n")
    entries = {f"tracking/{name}": item.out / "tracking" / name for name in result["helpers_sha256"]}
    entries.update({name: item.out / name for name in result["fixture_files_sha256"]})
    archive(item.out / "controller-source.tar", entries)
    result["archive_sha256"] = sha256(item.out / "controller-source.tar")
    record(item.out / "receipt.json", result)
    with pytest.raises(ValueError):
        production.require_controller_cpu(item.root, item.out / "receipt.json", item.cpu)
    assert not production.paths(item.root, "wave13")[0].exists()


def test_controller_cpu_rewrites_provisional_success_after_log_replacement(accepted, monkeypatch):
    item = accepted
    real_replace = Path.replace
    changed = []

    def replace(source, target):
        result = real_replace(source, target)
        if Path(target) == item.out / "receipt.json" and not changed:
            changed.append(True)
            (item.out / "controller.log").write_text("replaced provisional evidence\n")
        return result

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(RuntimeError):
        production.controller_cpu(item.root, item.cpu, item.out)
    result = read(item.out / "receipt.json")
    assert result["status"] == "needs_attention"
    assert result["log_sha256"] != sha256(item.out / "controller.log")
    assert result["exit_code"] == 0


def test_archived_controller_collects_in_fresh_cuda_hidden_process(accepted, tmp_path):
    item = accepted
    result = production.controller_cpu(item.root, item.cpu, item.out)
    extracted = tmp_path / "extracted"
    with tarfile.open(result["archive"]) as bundle:
        bundle.extractall(extracted, filter="data")
    checked = item.real_run(
        [*production.cpu_command(), "--collect-only"],
        cwd=extracted,
        env=os.environ
        | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(extracted / "tracking")},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert not (item.root / "runs").exists()


def test_conflicting_parent_helper_is_refused_before_controller_evidence(accepted, tmp_path):
    item = accepted
    sources = production.helper_sources(item.root, Path(production.__file__).parent)
    directory = tmp_path / "conflicting"
    directory.mkdir()
    for name, source in sources.items():
        shutil.copy2(source, directory / name)
    helper = directory / "fused_attribution_validation_controller.py"
    helper.write_text("CONFLICTING_NUMERICAL_VERIFIER = True\n")
    entry = directory / "fused_attribution_production.py"
    with entry.open("a") as stream:
        stream.write("\nimport fused_attribution_validation_controller\n")
    with pytest.raises(ValueError, match="conflicts"):
        production.helper_sources(item.root, directory)
    assert not item.out.exists()
