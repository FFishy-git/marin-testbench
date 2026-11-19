# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test accepted-parent preservation and real refusal at the unissued execution boundary."""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ns_training_path_lr_execution_source as source
import pytest
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    previous = source.inherited(ROOT)
    archived, original = tmp_path / "parent-archive", tmp_path / "original"
    retained = Path(previous["helpers"]).parent
    shutil.copytree(retained, archived, symlinks=True)
    for directory in ("experiment", "artifacts/pinned-sources-materialized"):
        shutil.copytree(archived / directory, original / directory)
    (original / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for path in Path(previous["candidate"]).glob("*.json"):
        shutil.copy2(archived / path.name, original / path.name)
    # Original archive authentication is the external filesystem boundary; the fixture
    # retains real candidate, inherited helper and configuration bytes.
    previous = previous | {"candidate": str(original), "helpers": str(archived / "tracking")}
    helpers = Path(source.__file__).resolve().parent
    root, successor = tmp_path / "root", tmp_path / "successor"
    (root / "data").mkdir(parents=True)
    write(root / "data/experiment_receipt.json", {"fixture": "accepted source-only data"})
    monkeypatch.setattr(source, "inherited", lambda root: previous)
    for name, value in source.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    prepared = source.prepare(root, successor)
    return root, original, successor, prepared, previous, helpers


def test_owner_only_preparation_preserves_all_numerical_bytes_and_real_unowned_execution_creates_no_output(candidate):
    root, original, successor, prepared, _, helpers = candidate
    before = candidate_files(original)
    assert source.require_preparation(root, successor) == prepared
    assert prepared["files_sha256"] == before | {source.OWNER: sha256(successor / source.OWNER)}
    assert len(prepared["plan"]["trials"]) == 54 and len(prepared["plan"]["commands"]) == 162
    assert not (root / source.predecessor.SEAL).exists()
    command = prepared["plan"]["commands"][0]
    invocations = (
        [command[0], *command[5:]],
        [
            source.contract.PYTHON,
            "-B",
            "-P",
            str(helpers / "ns_training_path_lr_validation.py"),
            "--root",
            str(root),
            "--candidate",
            str(successor),
            "--phase",
            "execute",
        ],
    )
    for argv in invocations:
        denied = subprocess.run(
            argv,
            cwd=successor,
            env=os.environ | {"PYTHONPATH": f"{helpers}:{successor}", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert denied.returncode != 0
        assert "execution owner is unissued" in denied.stderr
        assert not source.contract.locations(root)[3].exists() and not source.contract.locations(root)[4].exists()
    assert candidate_files(original) == before and candidate_files(successor) == prepared["files_sha256"]
    with pytest.raises(FileExistsError):
        source.prepare(root, successor)


@pytest.mark.parametrize(
    "fault", ("trainer", "pinned_engine", "config", "both_configs", "extra", "removed", "adapter", "metadata")
)
def test_rehashed_candidate_cannot_expand_owner_only_accepted_parent_scope(candidate, fault):
    root, original, successor, prepared, _, _ = candidate
    if fault in ("config", "both_configs"):
        name = source.contract.specs()[0]["config_name"]
        for directory in (successor, original) if fault == "both_configs" else (successor,):
            config = read(directory / name)
            config["optimizer"]["lr"] *= 1.1
            write(directory / name, config)
    elif fault == "extra":
        (successor / "experiment/unaccepted.py").write_text("UNACCEPTED = True\n")
    elif fault == "removed":
        (successor / "experiment/capture.py").unlink()
    else:
        name = {
            "trainer": "experiment/train.py",
            "pinned_engine": next(name for name in source.contract.TRAINING_FILES if name.endswith("/engine.py")),
            "adapter": source.OWNER,
            "metadata": next(iter(prepared["parent_metadata_sha256"])),
        }[fault]
        with (successor / name).open("ab") as stream:
            stream.write(b"\n")
    write(successor / source.PREPARATION, prepared | {"files_sha256": candidate_files(successor)})
    with pytest.raises((ValueError, KeyError)):
        source.require_preparation(root, successor)
    assert not source.contract.locations(root)[0].exists() and not source.contract.locations(root)[3].exists()


def test_execution_closure_keeps_archived_helpers_and_refuses_an_unaccepted_import(candidate, monkeypatch, tmp_path):
    _, _, _, _, previous, _ = candidate
    sources = source.execution_sources(previous)
    assert all(sha256(sources[name]) == digest for name, digest in previous["helpers_sha256"].items())
    local = tmp_path / "new"
    local.mkdir()
    for name in source.NEW_HELPERS:
        shutil.copy2(sources[name], local / name)
    with (local / source.NEW_HELPERS[0]).open("a") as stream:
        stream.write("\nimport unaccepted_mutable_reader\n")
    monkeypatch.setattr(source, "__file__", str(local / source.NEW_HELPERS[0]))
    with pytest.raises(ValueError):
        source.execution_sources(previous)


@pytest.mark.parametrize("fault", ("none", "xml", "archive", "parent_fixture"))
def test_adapter_cpu_archive_retains_parent_adapter_and_refuses_rehashed_incomplete_evidence(
    candidate, monkeypatch, fault
):
    root, original, successor, prepared, _, _ = candidate

    def completed(command, cwd, **kwargs):
        suite = ET.Element("testsuite", tests=str(source.TEST_COUNT), failures="0", errors="0", skipped="0")
        for index in range(source.TEST_COUNT):
            ET.SubElement(suite, "testcase", classname=source.TEST_MODULE, name=f"external_cpu_case_{index}")
        tree = ET.Element("testsuites")
        tree.append(suite)
        ET.ElementTree(tree).write(Path(cwd) / "tests.xml")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(source.subprocess, "run", completed)
    out = root / "cpu"
    cpu = source.cpu(root, successor, out)
    path = out / "receipt.json"
    assert cpu["helpers_sha256"] == prepared["helpers_sha256"]
    assert sha256(out / source.PARENT_OWNER) == sha256(original / source.OWNER)
    if fault == "xml":
        tree = ET.parse(out / "tests.xml")
        suite = tree.getroot().find("testsuite")
        suite.remove(suite.findall("testcase")[-1])
        suite.set("tests", str(source.TEST_COUNT - 1))
        tree.write(out / "tests.xml")
        write(path, cpu | {"tests_sha256": sha256(out / "tests.xml")})
    elif fault == "archive":
        with tarfile.open(out / "source.tar", "a") as bundle:
            bundle.add(out / source.PARENT_OWNER, arcname="unaccepted-parent.py")
        write(path, cpu | {"archive_sha256": sha256(out / "source.tar")})
    elif fault == "parent_fixture":
        with (out / source.PARENT_OWNER).open("ab") as stream:
            stream.write(b"\nchanged retained parent\n")
    if fault == "none":
        assert source.require_cpu(root, path) == cpu
        assert not source.contract.locations(root)[0].exists()
    else:
        with pytest.raises(ValueError):
            source.require_cpu(root, path)
