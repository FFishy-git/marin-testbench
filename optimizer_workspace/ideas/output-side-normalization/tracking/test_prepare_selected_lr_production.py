# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise isolated preparation, exact source scope and CPU failure archives."""

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import prepare_selected_lr_production as preparation
import pytest
from launch_wave5 import read, sha256
from selected_lr_validation import ARMS, locations
from selected_lr_validation import PREPARATION as PARENT_PREPARATION
from selected_lr_validation import STATUS as PARENT_STATUS
from wave7_release import candidate_inventory


def record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


@pytest.fixture
def frozen_parent(tmp_path, monkeypatch):
    root, previous = tmp_path / "output", tmp_path / "validation-candidate"
    binding_path, helpers, _, _, _ = locations(root)
    helpers.mkdir(parents=True)
    binding_path.parent.mkdir(parents=True)
    experiment = previous / "experiment"
    (experiment / "tests").mkdir(parents=True)
    (experiment / "__init__.py").write_text("")
    (experiment / "train.py").write_text(
        "import argparse\nimport json\nfrom pathlib import Path\n"
        "from .selected_lr_owner import require_selected_lr_training\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--config', type=Path, required=True)\n"
        "    parser.add_argument('--data', type=Path, required=True)\n"
        "    parser.add_argument('--data-receipt', type=Path, required=True)\n"
        "    parser.add_argument('--out', type=Path, required=True)\n"
        "    args = parser.parse_args()\n"
        "    require_selected_lr_training(args)\n"
        "    config = json.loads(args.config.read_text())\n"
        "    args.out.mkdir()\n"
        "    return config['optimizer']['lr']\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    (experiment / "selected_lr_owner.py").write_text(
        "def require_selected_lr_training(args):\n"
        "    raise RuntimeError('original validation ownership is preserved')\n"
    )
    (experiment / "production_owner.py").write_text("HISTORICAL_OWNER = 'untouched'\n")
    (experiment / "selected_lr_validation.py").write_text("VALIDATION_STEPS = (24, 22, 24)\n")
    (experiment / "tests/test_historical_owners.py").write_text(
        "import pytest\n"
        "from experiment.production_owner import HISTORICAL_OWNER\n"
        "from experiment.selected_lr_owner import require_selected_lr_training\n\n"
        "def test_original_owners_remain_available():\n"
        "    assert HISTORICAL_OWNER == 'untouched'\n"
        "    with pytest.raises(RuntimeError, match='original validation ownership'):\n"
        "        require_selected_lr_training(None)\n"
    )
    for arm in ARMS:
        record(
            experiment / f"configs/{arm}_0p5m_seed0.json",
            {
                "experiment": {"arm": arm, "refresh_install_age": 1},
                "training": {"steps": 5120, "world_size": 8, "tokens": 2684354560},
                "optimizer": {"lr": 0.064 if arm == "muont" else 0.09050966784},
            },
        )
    record(
        experiment / "cluster.json",
        {
            "wave_name": "wave11-selected-lr-full-budget-unreleased",
            "nodes": ["28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139"],
            "gpus_per_run": 8,
            "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        },
    )
    pinned = previous / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    (pinned / "numerical.py").write_text("ORIGINAL_NUMERICAL_IDENTITY = 42\n")
    (pinned / "source.bin").write_bytes(b"\x00\x01\x02original pinned bytes")
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    record(previous / PARENT_PREPARATION, {"candidate": str(previous), "status": "original_validation_prepared"})
    (helpers / "run_selected_lr_validation.py").write_text("# original frozen check boundary\n")
    (helpers / "selected_lr_validation.py").write_text("# original frozen validation contract\n")
    record(root / "data/experiment_receipt.json", {"status": "original_data_verified"})
    record(root / "run-preparation/wave10-launch-binding.json", {"status": "original_wave10_release"})
    cpu_log = root / "candidate-cpu.log"
    cpu_log.write_text("original candidate CPU passed\n")
    record(
        root / "candidate-cpu.json",
        {"status": "selected_lr_validation_cpu_verified_no_gpu_binding", "exit_code": 0, "log_sha256": sha256(cpu_log)},
    )
    record(
        root / "controller-cpu.json", {"status": "selected_lr_controller_cpu_verified_no_gpu_binding", "exit_code": 0}
    )
    archive = root / "validation-source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in candidate_inventory(previous):
            bundle.add(previous / name, arcname=name, recursive=False)
        bundle.add(previous / PARENT_PREPARATION, arcname=PARENT_PREPARATION, recursive=False)
        bundle.add(previous / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    frozen = {
        "status": PARENT_STATUS,
        "root": str(root),
        "candidate": str(previous),
        "helpers": str(helpers),
        "helpers_sha256": {path.name: sha256(path) for path in helpers.iterdir()},
        "files_sha256": candidate_inventory(previous),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "cpu_receipt": str(root / "candidate-cpu.json"),
        "cpu_receipt_sha256": sha256(root / "candidate-cpu.json"),
        "controller_cpu_receipt": str(root / "controller-cpu.json"),
        "controller_cpu_receipt_sha256": sha256(root / "controller-cpu.json"),
        "preparation_sha256": sha256(previous / PARENT_PREPARATION),
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "predecessor_release_sha256": sha256(root / "run-preparation/wave10-launch-binding.json"),
    }
    record(binding_path, frozen)
    real_run = subprocess.run
    boundary = {"failure": None, "checks": []}

    def external(command, **kwargs):
        if len(command) > 2 and command[2] == str(helpers / "run_selected_lr_validation.py"):
            boundary["checks"].append((command, kwargs))
            if boundary["failure"] is not None:
                raise subprocess.CalledProcessError(1, command, stderr=boundary["failure"])
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"status": PARENT_STATUS}), stderr="")
        return real_run(command, **kwargs)

    monkeypatch.setattr(preparation.subprocess, "run", external)
    return root, previous, frozen, boundary, real_run


def test_preparation_preserves_original_and_limits_successor_to_four_exact_changes(frozen_parent, tmp_path):
    root, previous, frozen, boundary, real_run = frozen_parent
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    original_files = candidate_inventory(previous)
    candidate = tmp_path / "production-candidate"
    result = preparation.prepare(root, candidate)
    actual = candidate_inventory(candidate)
    changed = {name for name in actual.keys() | original_files.keys() if actual.get(name) != original_files.get(name)}
    assert changed == {
        "experiment/train.py",
        "experiment/selected_lr_production_owner.py",
        "experiment/cluster.json",
        "experiment/tests/test_selected_lr_production_owner.py",
    }
    assert candidate_inventory(previous) == original_files == frozen["files_sha256"]
    assert all(path.read_bytes() == content for path, content in before.items())
    assert (candidate / PARENT_PREPARATION).read_bytes() == (previous / PARENT_PREPARATION).read_bytes()
    for name in original_files.keys() - changed:
        assert (candidate / name).read_bytes() == (previous / name).read_bytes()
    cluster = read(candidate / "experiment/cluster.json")
    assert cluster.pop("wave_name") == "wave11"
    original_cluster = read(previous / "experiment/cluster.json")
    original_cluster.pop("wave_name")
    assert cluster == original_cluster
    assert preparation.require_preparation(root, candidate) == result
    assert all(kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "" for _, kwargs in boundary["checks"])
    assert all(command[-2:] == ["--phase", "check"] for command, _ in boundary["checks"])
    assert not (root / "run-preparation/wave11-launch-binding.json").exists()
    assert not (root / "controllers/wave11-launch-20261117").exists()
    assert not (root / "runs/wave11").exists()
    denied = real_run(
        [
            sys.executable,
            "-B",
            "-m",
            "experiment.train",
            "--config",
            str(tmp_path / "missing-config.json"),
            "--data",
            str(root / "data"),
            "--data-receipt",
            str(root / "data/experiment_receipt.json"),
            "--out",
            str(tmp_path / "uncreated-run"),
        ],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert denied.returncode != 0
    assert "separate generation-bound production release" in denied.stderr
    assert "FileNotFoundError" not in denied.stderr
    assert not (tmp_path / "uncreated-run").exists()
    with pytest.raises(FileExistsError):
        preparation.prepare(root, candidate)


@pytest.mark.parametrize("fault", ("frozen_check", "config", "archive", "cpu", "controller_cpu"))
def test_changed_original_source_or_frozen_evidence_prevents_copy(frozen_parent, tmp_path, fault):
    root, previous, frozen, boundary, _ = frozen_parent
    if fault == "frozen_check":
        boundary["failure"] = "original frozen CPU check refused"
    elif fault == "config":
        (previous / f"experiment/configs/{ARMS[0]}_0p5m_seed0.json").write_text("{}\n")
    else:
        key = {"archive": "archive", "cpu": "cpu_receipt", "controller_cpu": "controller_cpu_receipt"}[fault]
        Path(frozen[key]).write_bytes(b"replaced original evidence")
    candidate = tmp_path / "refused-candidate"
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        preparation.prepare(root, candidate)
    assert not candidate.exists()
    assert not (root / "runs/wave11").exists()


@pytest.mark.parametrize("scope", ("train", "owner", "cluster", "config", "parent_preparation"))
def test_preparation_rejects_rehashed_allowed_file_or_numerical_drift(frozen_parent, tmp_path, scope):
    root, _, _, _, _ = frozen_parent
    candidate = tmp_path / "production-candidate"
    preparation.prepare(root, candidate)
    if scope == "train":
        path = candidate / "experiment/train.py"
        path.write_text(path.read_text().replace("return config['optimizer']['lr']", "return 99"))
    elif scope == "owner":
        (candidate / "experiment/selected_lr_production_owner.py").write_text(
            "def require_selected_lr_production_owner(args):\n    return None\n"
        )
    elif scope == "cluster":
        cluster_path = candidate / "experiment/cluster.json"
        cluster = read(cluster_path)
        cluster["gpus_per_run"] = 1
        record(cluster_path, cluster)
    elif scope == "config":
        (candidate / f"experiment/configs/{ARMS[0]}_0p5m_seed0.json").write_text('{"optimizer":{"lr":999}}\n')
    else:
        record(candidate / PARENT_PREPARATION, {"status": "replaced_original_preparation"})
    prepared_path = candidate / preparation.PREPARATION
    prepared = read(prepared_path)
    prepared["files_sha256"] = candidate_inventory(candidate)
    record(prepared_path, prepared)
    with pytest.raises(ValueError, match="scope"):
        preparation.require_preparation(root, candidate)
    assert not (root / "run-preparation/wave11-launch-binding.json").exists()


def test_real_temporary_candidate_cpu_suite_archives_exact_copied_owner_and_preserves_gpu_denial(
    frozen_parent, tmp_path
):
    root, _, _, _, real_run = frozen_parent
    candidate, out = tmp_path / "production-candidate", tmp_path / "cpu"
    preparation.prepare(root, candidate)
    receipt = preparation.cpu(root, candidate, out)
    assert receipt["status"] == "selected_lr_production_cpu_verified_gpu_denied"
    assert receipt["exit_code"] == 0
    assert "2 passed" in (out / "candidate.log").read_text()
    assert preparation.require_cpu(root, out / "receipt.json") == receipt
    with tarfile.open(out / "source.tar") as bundle:
        expected = set(receipt["files_sha256"]) | {
            preparation.PREPARATION,
            "artifacts/external-sources",
            "tracking/prepare_selected_lr_production.py",
            "tracking/selected_lr_production_owner.py",
            "tracking/test_selected_lr_production_owner.py",
        }
        assert len(bundle.getmembers()) == len(expected)
        assert set(bundle.getnames()) == expected
        assert bundle.getmember("artifacts/external-sources").linkname == "pinned-sources-materialized"
    check = real_run(
        [
            sys.executable,
            "-B",
            "-c",
            "from experiment.selected_lr_production_owner import require_selected_lr_production_owner\n"
            "require_selected_lr_production_owner(None)\n",
        ],
        cwd=candidate,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert check.returncode != 0
    assert "separate generation-bound production release" in check.stderr
    assert not (root / "runs/wave11").exists()


@pytest.mark.parametrize("failure", ("exit", "timeout", "source_drift", "spawn"))
def test_cpu_failure_is_preserved_with_archive_and_rejected_as_success(frozen_parent, tmp_path, monkeypatch, failure):
    root, _, _, _, _ = frozen_parent
    candidate, out = tmp_path / "production-candidate", tmp_path / "cpu"
    preparation.prepare(root, candidate)
    external = preparation.subprocess.run

    def failed(command, **kwargs):
        if command != preparation.cpu_command():
            return external(command, **kwargs)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        kwargs["stdout"].write("preserved CPU boundary failure\n")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 600)
        if failure == "spawn":
            raise OSError("CPU executable unavailable")
        if failure == "source_drift":
            (candidate / f"experiment/configs/{ARMS[0]}_0p5m_seed0.json").write_text("{}\n")
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(preparation.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="preserved"):
        preparation.cpu(root, candidate, out)
    receipt = read(out / "receipt.json")
    assert receipt["status"] == "selected_lr_production_cpu_failed"
    assert receipt["exit_code"] == {"exit": 1, "timeout": 124, "source_drift": 0, "spawn": 127}[failure]
    assert receipt["log_sha256"] == sha256(out / "candidate.log")
    assert receipt["archive_sha256"] == sha256(out / "source.tar")
    with pytest.raises(ValueError):
        preparation.require_cpu(root, out / "receipt.json")
    assert not (root / "run-preparation/wave11-launch-binding.json").exists()


@pytest.mark.parametrize(
    "fault",
    (
        "status",
        "command",
        "cwd",
        "cuda",
        "exit_bool",
        "exit_nonzero",
        "error",
        "log",
        "archive_bytes",
        "archive_extra",
        "archive_duplicate",
        "archive_alias",
    ),
)
def test_cpu_rejects_rehashed_archive_payload_scope_or_log_drift(frozen_parent, tmp_path, monkeypatch, fault):
    root, _, _, _, _ = frozen_parent
    candidate, out = tmp_path / "production-candidate", tmp_path / "cpu"
    preparation.prepare(root, candidate)
    external = preparation.subprocess.run

    def passed(command, **kwargs):
        if command != preparation.cpu_command():
            return external(command, **kwargs)
        kwargs["stdout"].write("CPU boundary succeeded\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(preparation.subprocess, "run", passed)
    receipt = preparation.cpu(root, candidate, out)
    archive = out / "source.tar"
    if fault in ("status", "command", "cwd", "cuda", "exit_bool", "exit_nonzero", "error"):
        changes = {
            "status": ("status", "selected_lr_production_cpu_failed"),
            "command": ("command", [sys.executable, "-c", "pass"]),
            "cwd": ("cwd", str(tmp_path / "unrelated-candidate")),
            "cuda": ("cuda_visible_devices", "0,1,2,3,4,5,6,7"),
            "exit_bool": ("exit_code", False),
            "exit_nonzero": ("exit_code", 1),
            "error": ("error", "preserved post-command failure"),
        }
        key, value = changes[fault]
        receipt[key] = value
        record(out / "receipt.json", receipt)
    elif fault == "log":
        (out / "candidate.log").write_text("changed CPU outcome\n")
    else:
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            payloads = {member.name: bundle.extractfile(member).read() for member in members if member.isfile()}
        with tarfile.open(archive, "w") as bundle:
            for member in members:
                if member.isfile():
                    content = payloads[member.name]
                    if fault == "archive_bytes" and member.name == "experiment/train.py":
                        content = b"replaced trainer bytes"
                    member.size = len(content)
                    bundle.addfile(member, io.BytesIO(content))
                else:
                    if fault == "archive_alias":
                        member.linkname = "outside-candidate"
                    bundle.addfile(member)
            if fault in ("archive_extra", "archive_duplicate"):
                content = b"unexpected source"
                member = tarfile.TarInfo("extra.py" if fault == "archive_extra" else "experiment/train.py")
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
        receipt["archive_sha256"] = sha256(archive)
        record(out / "receipt.json", receipt)
    with pytest.raises(ValueError):
        preparation.require_cpu(root, out / "receipt.json")
    assert not (root / "runs/wave11").exists()
