# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise archived acceptance and fresh-process import provenance on persisted evidence."""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from xml.etree import ElementTree as ET

import prepare_ns_training_path_predecessor as preparation
import pytest
from fused_attribution_validation_test_support import record
from selected_lr_release_owner import read, sha256

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def archived_files(directory: Path, inventory: dict) -> None:
    path = directory / "source.tar"
    with tarfile.open(path, "x") as bundle:
        for name in inventory:
            bundle.add(directory / name, arcname=name, recursive=False)
        alias = tarfile.TarInfo("artifacts/external-sources")
        alias.type = tarfile.SYMTYPE
        alias.linkname = "pinned-sources-materialized"
        bundle.addfile(alias)


def passing_xml(path: Path, counts: dict) -> None:
    suites = ET.Element("testsuites")
    suite = ET.SubElement(suites, "testsuite", tests=str(sum(counts.values())), failures="0", errors="0", skipped="0")
    for classname, count in counts.items():
        for index in range(count):
            ET.SubElement(suite, "testcase", classname=classname, name=f"case_{index}")
    ET.ElementTree(suites).write(path)


@pytest.fixture
def accepted_component(tmp_path, monkeypatch):
    inherited = read(ROOT / preparation.predecessor.TRAINING_CPU)
    original_sources = Path(inherited["helpers"])
    local = Path(__file__).resolve().parent
    parent = tmp_path / "parent"
    helpers = parent / "tracking"
    helpers.mkdir(parents=True)
    for name in inherited["helpers_sha256"]:
        source = local / name
        if not source.is_file():
            source = original_sources / name
        shutil.copy2(source, helpers / name)
    hashes = {name: sha256(helpers / name) for name in inherited["helpers_sha256"]}
    inventory = {f"tracking/{name}": digest for name, digest in hashes.items()}
    archived_files(parent, inventory)
    (parent / "cpu.log").write_text("Completed independent parent CPU fixture\n")
    passing_xml(parent / "tests.xml", {"accepted_original_parent": 69})
    previous = {
        "status": preparation.PARENT_SPEC[2],
        "root": str(tmp_path),
        "candidate": str(tmp_path / "candidate"),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "inventory_sha256": inventory,
        "exit_code": 0,
        "cuda_visible_devices": "",
        **{
            name: str(parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        },
        **{
            f"{name}_sha256": sha256(parent / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        },
    }
    record(parent / "receipt.json", previous)
    spec = ("parent/receipt.json", sha256(parent / "receipt.json"), previous["status"], 69)
    monkeypatch.setattr(preparation, "PARENT_SPEC", spec)
    comparison = local.parent / preparation.COMPARISON
    if not comparison.is_file():
        comparison = ROOT / preparation.COMPARISON_SOURCE
    target = tmp_path / preparation.COMPARISON_SOURCE
    target.parent.mkdir(parents=True)
    shutil.copy2(comparison, target)
    out = tmp_path / "accepted"
    (out / "tracking").mkdir(parents=True)
    sources = preparation.helper_sources(previous)
    expected = preparation.inventory(tmp_path, previous, sources)
    for name in expected:
        source = sources[Path(name).name] if name.startswith("tracking/") else target
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    archived_files(out, expected)
    (out / "cpu.log").write_text("Completed archived component fixture\n")
    passing_xml(out / "tests.xml", preparation.TEST_COUNTS)
    receipt = {
        "status": preparation.STATUS,
        "root": str(tmp_path),
        "candidate": previous["candidate"],
        "parent_cpu_receipt_sha256": spec[1],
        "helpers": str(out / "tracking"),
        "helpers_sha256": {name: sha256(source) for name, source in sources.items()},
        "inventory_sha256": expected,
        "command": preparation.command(out),
        "cwd": str(out),
        "environment": preparation.ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": 0,
        "scope": preparation.predecessor.SCOPE,
        **{
            name: str(out / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        },
        **{
            f"{name}_sha256": sha256(out / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        },
    }
    path = out / "receipt.json"
    record(path, receipt)
    return tmp_path, path, spec


@pytest.mark.parametrize("fault", ("none", "changed_helper", "incomplete_xml", "extra_archive"))
def test_cpu_archive_acceptance_preserves_source_and_rejects_incomplete_evidence(accepted_component, fault):
    root, path, _ = accepted_component
    if fault == "changed_helper":
        (path.parent / "tracking/ns_training_path_predecessor.py").write_text("# unaccepted change\n")
    elif fault == "incomplete_xml":
        passing_xml(path.parent / "tests.xml", {"test_ns_training_path_predecessor": 1})
    elif fault == "extra_archive":
        with tarfile.open(path.parent / "source.tar", "a") as archive:
            archive.add(path.parent / "cpu.log", arcname="unaccepted-source.py", recursive=False)
    if fault == "none":
        receipt = preparation.require_cpu(root, path)
        assert receipt == read(path)
        assert receipt["status"] == preparation.STATUS
        assert not (root / preparation.predecessor.SEAL).exists()
        with pytest.raises(RuntimeError, match="archived entry"):
            preparation.accepted(root, path)
    else:
        with pytest.raises(ValueError):
            preparation.require_cpu(root, path)


def test_fresh_archived_process_accepts_complete_evidence_with_only_archived_imports(accepted_component):
    root, path, spec = accepted_component
    helpers = path.parent / "tracking"
    result = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            "from pathlib import Path; import prepare_ns_training_path_predecessor as p; "
            f"p.PARENT_SPEC={spec!r}; "
            f"receipt=p.accepted(Path({str(root)!r}),Path({str(path)!r})); "
            "print(receipt['status'])",
        ],
        cwd=helpers,
        env=os.environ | preparation.ENVIRONMENT | {"PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == preparation.STATUS
    assert not (root / preparation.predecessor.SEAL).exists()
