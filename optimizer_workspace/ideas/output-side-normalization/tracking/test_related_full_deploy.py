# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real deployment filesystem effects behind a fake SSH boundary."""

import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import related_campaign as short
import related_full_campaign as full
import related_full_deploy as deployment


def save(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


@pytest.fixture
def prepared(tmp_path):
    source, root = tmp_path / "source", tmp_path / "output"
    for path in (short.TASK / "experiment").rglob("*.py"):
        target = source / path.relative_to(short.TASK)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for folder, _ in short.PROFILES.values():
        shutil.copytree(short.TASK / "experiment/comparisons" / folder, source / "experiment/comparisons" / folder)
    manifest = short.read(short.TASK / "experiment/upstream_manifest.json")
    save(source / "experiment/upstream_manifest.json", manifest)
    shutil.copyfile(short.TASK / "experiment/cluster.json", source / "experiment/cluster.json")
    prefix = f"artifacts/external-sources/Toner-{manifest['revision']}"
    inputs = [
        name
        for name in manifest["files"]
        if Path(name).suffix == ".py"
        and (Path(name).parent == Path("reproduce/marin") or Path(name).is_relative_to("src/toner"))
    ]
    inputs += [
        "reproduce/marin/data_plan.json",
        "reproduce/marin/configs/130m/muon_0p5m.json",
        "reproduce/marin/configs/130m/muont_0p5m.json",
    ]
    for name in [*(f"{prefix}/{name}" for name in inputs), short.MOUSSE_REFERENCE]:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(short.TASK / name, target)
    shard = root / "data/dclm/tiny.bin"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"\x00\x00\x00\x00")
    save(
        root / "data/experiment_receipt.json",
        {
            "root": str(root / "data"),
            "data_plan_sha256": short.sha256(source / prefix / "reproduce/marin/data_plan.json"),
            "files": [
                {
                    "file": "dclm/tiny.bin",
                    "bytes": 4,
                    "mtime_ns": shard.stat().st_mtime_ns,
                    "sha256": short.sha256(shard),
                }
            ],
        },
    )
    campaign = short.prepare(root, source=source)
    for trial in campaign["trials"]:
        config, directory = short.read(Path(trial["config"])), Path(trial["out"])
        save(
            directory / "run.json",
            {
                "status": "complete",
                "completed_steps": short.STEPS,
                "execution_check": False,
                "resume": None,
                "config": config,
                "data": campaign["data"],
                "data_receipt_sha256": campaign["data_receipt_sha256"],
                "source_revision": campaign["source_revision"],
                "source_sha256": campaign["source_sha256"],
                "world_size": short.WORLD,
            },
        )
        rows = [
            {"step": step, "tokens": step * short.BATCH_TOKENS, "train_loss": 4.0} for step in range(1, short.STEPS + 1)
        ]
        weights = config["data"]["mixture_weights"]
        heldout = {f"heldout/{name}": 3.0 for name in weights}
        mixture = sum(weight * heldout[f"heldout/{name}"] for name, weight in weights.items()) / sum(weights.values())
        rows[-1].update(selection_loss=mixture, heldout_losses=heldout)
        (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        checkpoint = directory / f"checkpoints/step_{short.STEPS:06d}"
        save(checkpoint / "complete.json", {"step": short.STEPS, "world": short.WORLD})
        for rank in range(short.WORLD):
            (checkpoint / f"rank_{rank}.pt").write_bytes(b"offline nonempty endpoint evidence")
    return root, source, campaign, full.prepare(root)


class RemoteNodes:
    def __init__(self, tmp_path, monkeypatch):
        self.paths = {node: tmp_path / node for node in deployment.NODES[1:]}
        for path in self.paths.values():
            path.mkdir(parents=True)
        self.before = None
        self.after = None
        self.remote_io = ""
        self.calls = []
        self.run_process = subprocess.run
        monkeypatch.setattr(deployment.subprocess, "run", self.run)

    def run(self, command, **kwargs):
        node = command[-2].removeprefix("root@")
        arguments = shlex.split(command[-1])
        assert node in self.paths
        assert arguments[:3] == [deployment.PYTHON, "-B", "-c"]
        phase = arguments[5]
        self.calls.append((node, phase))
        payload = kwargs["input"]
        if self.before is not None:
            payload = self.before(node, phase, payload)
        arguments[0], arguments[4] = sys.executable, str(self.paths[node])
        if phase == "write":
            arguments[3] = self.remote_io + arguments[3]
        result = self.run_process(
            arguments, input=payload, capture_output=kwargs["capture_output"], timeout=kwargs["timeout"]
        )
        if self.after is not None:
            self.after(node, phase)
        return result


@pytest.fixture
def nodes(tmp_path, monkeypatch):
    return RemoteNodes(tmp_path / "remotes", monkeypatch)


def snapshots(directory: Path) -> dict:
    result = {}
    for path in directory.rglob("*"):
        information = path.lstat()
        identity = (
            information.st_dev,
            information.st_ino,
            information.st_size,
            information.st_mtime_ns,
            information.st_ctime_ns,
        )
        payload = path.read_bytes() if path.is_file() and not path.is_symlink() else None
        result[str(path.relative_to(directory))] = identity, payload
    return result


def receipt_path(root: Path) -> Path:
    return root / "run-preparation/full-config-deployment.json"


def test_deploy_creates_only_bound_configs_preserves_identical_and_unrelated_metadata(prepared, nodes):
    root, source, _, campaign = prepared
    expected, _ = deployment.inventory(campaign)
    before = {}
    existing = next(iter(expected))
    for node, path in nodes.paths.items():
        for relative in ("experiment/train.py", "experiment/comparisons/short/config.json", "cache/verified.json"):
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"existing source or short verification evidence")
        target = path / existing
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / existing).read_bytes())
        before[node] = snapshots(path)
    local_before = snapshots(source)
    result = deployment.deploy(root, receipt_path(root))
    assert short.read(receipt_path(root)) == result
    assert result["status"] == "deployed"
    assert len(expected) == 16
    assert all(len(record["created"]) == 15 for record in result["writes"])
    for node, path in nodes.paths.items():
        observed = snapshots(path)
        for name, information in before[node].items():
            if (path / name).is_file():
                assert observed[name] == information
        assert {name: short.sha256(path / name) for name in expected} == expected
        assert all((path / name).read_bytes() == (source / name).read_bytes() for name in expected)
    assert snapshots(source) == local_before
    assert nodes.calls[:3] == [(node, "check") for node in deployment.NODES[1:]]


def test_identical_redeployment_with_fresh_receipt_changes_no_remote_stat_identity(prepared, nodes):
    root, _, _, _ = prepared
    deployment.deploy(root, receipt_path(root))
    before = {node: snapshots(path) for node, path in nodes.paths.items()}
    result = deployment.deploy(root, root / "run-preparation/full-config-deployment-second.json")
    assert all(record["created"] == [] for record in result["writes"])
    assert {node: snapshots(path) for node, path in nodes.paths.items()} == before


def test_last_node_divergence_refuses_before_any_remote_write(prepared, nodes):
    root, _, _, campaign = prepared
    target = nodes.paths[deployment.NODES[-1]] / Path(campaign["waves"][-1]["manifest"]).relative_to(campaign["source"])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"divergent remote wave")
    before = {node: snapshots(path) for node, path in nodes.paths.items()}
    with pytest.raises(RuntimeError, match="divergent"):
        deployment.deploy(root, receipt_path(root))
    assert {node: snapshots(path) for node, path in nodes.paths.items()} == before
    assert not receipt_path(root).exists()


@pytest.mark.parametrize("redirect", ["config", "dangling_config", "parent", "source"])
def test_remote_symlinks_refuse_without_following_or_writing_targets(prepared, nodes, tmp_path, redirect):
    root, source, _, campaign = prepared
    remote = nodes.paths[deployment.NODES[-1]]
    name = Path(campaign["trials"][0]["config"]).relative_to(source)
    outside = tmp_path / "outside"
    outside.mkdir()
    if redirect in ("config", "dangling_config"):
        target = outside / "config.json"
        if redirect == "config":
            target.write_bytes((source / name).read_bytes())
        leaf = remote / name
        leaf.parent.mkdir(parents=True)
        leaf.symlink_to(target)
    elif redirect == "parent":
        parent = remote / full.CONFIG_DIRECTORY
        parent.parent.mkdir(parents=True)
        parent.symlink_to(outside, target_is_directory=True)
    else:
        remote.rmdir()
        remote.symlink_to(outside, target_is_directory=True)
    before = snapshots(outside)
    with pytest.raises(RuntimeError):
        deployment.deploy(root, receipt_path(root))
    assert snapshots(outside) == before
    assert not receipt_path(root).exists()
    assert all(
        not (path / full.CONFIG_DIRECTORY).exists() for node, path in nodes.paths.items() if node != deployment.NODES[-1]
    )


def test_check_only_never_creates_configs_directories_or_receipt(prepared, nodes):
    root, _, _, _ = prepared
    before = {node: snapshots(path) for node, path in nodes.paths.items()}
    result = deployment.deploy(root, receipt_path(root), check_only=True)
    assert result["status"] == "checked"
    assert all(set(record["files"].values()) == {"absent"} for record in result["preflight"])
    assert {node: snapshots(path) for node, path in nodes.paths.items()} == before
    assert not receipt_path(root).parent.exists()


@pytest.mark.parametrize("damage", ["short_rank", "short_history", "full_config"])
def test_stale_completed_campaign_refuses_without_remote_effects(prepared, nodes, damage):
    root, source, campaign, selected = prepared
    if damage == "short_rank":
        (Path(campaign["trials"][0]["out"]) / "checkpoints/step_000320/rank_7.pt").unlink()
    elif damage == "short_history":
        path = Path(campaign["trials"][0]["out"]) / "metrics_from_000000.jsonl"
        path.write_text(path.read_text().replace('"train_loss": 4.0', '"train_loss": 4.1'))
    else:
        Path(selected["trials"][0]["config"]).write_bytes(b"changed selected config")
    before = snapshots(source)
    with pytest.raises((ValueError, FileNotFoundError)):
        deployment.deploy(root, receipt_path(root))
    assert all(snapshots(path) == {} for path in nodes.paths.values())
    assert snapshots(source) == before
    assert not receipt_path(root).exists()


def test_local_mutation_during_all_node_preflight_blocks_write_phase(prepared, nodes):
    root, _, _, campaign = prepared

    def damage(node, phase, payload):
        if node == deployment.NODES[-1] and phase == "check":
            Path(campaign["waves"][-1]["manifest"]).write_bytes(b"changed during remote preflight")
        return payload

    nodes.before = damage
    with pytest.raises(ValueError):
        deployment.deploy(root, receipt_path(root))
    assert all(snapshots(path) == {} for path in nodes.paths.values())
    assert not receipt_path(root).exists()


@pytest.mark.parametrize("damage", ["divergent_file", "parent_redirect"])
def test_changed_remote_after_preflight_refuses_before_creating_configs(prepared, nodes, tmp_path, damage):
    root, source, _, campaign = prepared
    remote = nodes.paths[deployment.NODES[1]]
    name = Path(campaign["waves"][-1]["manifest"]).relative_to(source)
    outside = tmp_path / "outside"
    outside.mkdir()

    def intervene(node, phase, payload):
        if node == deployment.NODES[1] and phase == "write":
            parent = remote / full.CONFIG_DIRECTORY
            parent.parent.mkdir(parents=True)
            if damage == "divergent_file":
                parent.mkdir()
                (remote / name).write_bytes(b"raced divergent file")
            else:
                parent.symlink_to(outside, target_is_directory=True)
        return payload

    nodes.before = intervene
    with pytest.raises(RuntimeError):
        deployment.deploy(root, receipt_path(root))
    assert snapshots(outside) == {}
    assert not receipt_path(root).exists()
    assert not (nodes.paths[deployment.NODES[2]] / full.CONFIG_DIRECTORY).exists()
    assert not (nodes.paths[deployment.NODES[3]] / full.CONFIG_DIRECTORY).exists()
    if damage == "divergent_file":
        assert list((remote / full.CONFIG_DIRECTORY).iterdir()) == [remote / name]


def test_postwrite_hash_failure_preserves_files_but_never_publishes_success_receipt(prepared, nodes):
    root, source, _, campaign = prepared
    missing = nodes.paths[deployment.NODES[-1]] / Path(campaign["waves"][-1]["manifest"]).relative_to(source)

    def damage(node, phase):
        if node == deployment.NODES[-1] and phase == "write":
            missing.unlink()

    nodes.after = damage
    with pytest.raises(RuntimeError, match="Missing deployed config"):
        deployment.deploy(root, receipt_path(root))
    assert not receipt_path(root).exists()
    assert (nodes.paths[deployment.NODES[1]] / full.CONFIG_DIRECTORY).is_dir()
    assert not missing.exists()


def test_remote_exclusive_open_preserves_file_that_appears_after_scan(prepared, nodes):
    root, _, _, campaign = prepared
    files, _ = deployment.inventory(campaign)
    name = next(iter(files))
    nodes.remote_io = f"""
import os
original_open = os.open
def race_open(path, flags, mode=0o777, *, dir_fd=None):
    if flags & os.O_EXCL and path == {Path(name).name!r}:
        descriptor = original_open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode, dir_fd=dir_fd)
        os.write(descriptor, b'appeared after preflight')
        os.close(descriptor)
    return original_open(path, flags, mode, dir_fd=dir_fd)
os.open = race_open
"""
    with pytest.raises(RuntimeError, match="FileExistsError"):
        deployment.deploy(root, receipt_path(root))
    remote = nodes.paths[deployment.NODES[1]]
    assert (remote / name).read_bytes() == b"appeared after preflight"
    assert list((remote / full.CONFIG_DIRECTORY).iterdir()) == [remote / name]
    assert not receipt_path(root).exists()
    assert all(not (nodes.paths[node] / full.CONFIG_DIRECTORY).exists() for node in deployment.NODES[2:])


def test_parent_redirect_during_remote_creation_cannot_write_through_symlink(prepared, nodes, tmp_path):
    root, _, _, _ = prepared
    remote = nodes.paths[deployment.NODES[1]]
    outside, moved = tmp_path / "outside", tmp_path / "moved-configs"
    outside.mkdir()
    nodes.remote_io = f"""
import os
original_open = os.open
def redirect_open(path, flags, mode=0o777, *, dir_fd=None):
    if flags & os.O_EXCL and not os.path.exists({str(moved)!r}):
        os.rename({str(remote / full.CONFIG_DIRECTORY)!r}, {str(moved)!r})
        os.symlink({str(outside)!r}, {str(remote / full.CONFIG_DIRECTORY)!r})
    return original_open(path, flags, mode, dir_fd=dir_fd)
os.open = redirect_open
"""
    with pytest.raises(RuntimeError):
        deployment.deploy(root, receipt_path(root))
    assert snapshots(outside) == {}
    assert moved.is_dir()
    assert not receipt_path(root).exists()
    assert all(not (nodes.paths[node] / full.CONFIG_DIRECTORY).exists() for node in deployment.NODES[2:])


def test_archive_content_mismatch_refuses_before_remote_directory_creation(prepared, nodes):
    root, _, _, _ = prepared

    def damage(node, phase, payload):
        if phase != "write":
            return payload
        output = io.BytesIO()
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as original:
            with tarfile.open(fileobj=output, mode="w:gz") as changed:
                for index, member in enumerate(original.getmembers()):
                    data = b"corrupt selected config" if index == 0 else original.extractfile(member).read()
                    member.size = len(data)
                    changed.addfile(member, io.BytesIO(data))
        return output.getvalue()

    nodes.before = damage
    with pytest.raises(RuntimeError, match="Archive content"):
        deployment.deploy(root, receipt_path(root))
    assert all(snapshots(path) == {} for path in nodes.paths.values())
    assert not receipt_path(root).exists()


@pytest.mark.parametrize("location", ["existing", "source", "cache", "symlink_parent"])
def test_receipt_refusal_cannot_overwrite_evidence_or_redirect_output(prepared, nodes, tmp_path, location):
    root, source, _, _ = prepared
    receipt = receipt_path(root)
    if location == "existing":
        receipt.parent.mkdir()
        receipt.write_bytes(b"existing deployment evidence")
    elif location == "source":
        receipt = source / "new-receipt.json"
    elif location == "cache":
        receipt = root / "run-preparation/related-verified-cache/new.json"
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        receipt.parent.symlink_to(outside, target_is_directory=True)
    before = snapshots(root)
    with pytest.raises((ValueError, FileExistsError)):
        deployment.deploy(root, receipt)
    assert snapshots(root) == before
    assert all(snapshots(path) == {} for path in nodes.paths.values())


@pytest.mark.parametrize("change", ["cluster", "imported_nodes", "both", "empty_remotes"])
@pytest.mark.parametrize("check_only", [False, True])
def test_unapproved_nodes_refuse_before_remote_io(prepared, nodes, monkeypatch, change, check_only):
    root, source, _, _ = prepared
    unapproved = (deployment.APPROVED_NODES[0], "28.216.17.200")
    if change == "empty_remotes":
        unapproved = (deployment.APPROVED_NODES[0],)
    if change in ("cluster", "both", "empty_remotes"):
        path = source / "experiment/cluster.json"
        cluster = short.read(path)
        cluster["nodes"] = list(unapproved)
        save(path, cluster)
    if change in ("imported_nodes", "both", "empty_remotes"):
        monkeypatch.setattr(deployment, "NODES", unapproved)
    before = snapshots(root)
    with pytest.raises(ValueError, match="independently approved"):
        deployment.deploy(root, receipt_path(root), check_only=check_only)
    assert snapshots(root) == before
    assert nodes.calls == []
    assert all(snapshots(path) == {} for path in nodes.paths.values())


@pytest.mark.parametrize(
    "change", ["parent_replace", "parent_symlink", "leaf_missing", "leaf_replace", "leaf_symlink", "leaf_content"]
)
def test_receipt_retarget_during_publication_refuses_success_preserves_created_files(
    prepared, nodes, monkeypatch, tmp_path, change
):
    root, _, _, campaign = prepared
    receipt, moved = receipt_path(root), tmp_path / "moved-evidence"
    original_fsync = os.fsync
    interrupted = False

    def retarget(descriptor):
        nonlocal interrupted
        original_fsync(descriptor)
        if interrupted or Path(f"/proc/self/fd/{descriptor}").readlink() != receipt:
            return
        interrupted = True
        if change.startswith("parent_"):
            receipt.parent.rename(moved)
            if change == "parent_replace":
                receipt.parent.mkdir()
            else:
                receipt.parent.symlink_to(moved, target_is_directory=True)
        elif change == "leaf_content":
            moved.write_bytes(receipt.read_bytes())
            receipt.write_bytes(b"changed published evidence")
        else:
            receipt.rename(moved)
            if change == "leaf_replace":
                receipt.write_bytes(moved.read_bytes())
            elif change == "leaf_symlink":
                receipt.symlink_to(moved)

    monkeypatch.setattr(deployment.os, "fsync", retarget)
    with pytest.raises((OSError, ValueError)):
        deployment.deploy(root, receipt)
    preserved = moved / receipt.name if change.startswith("parent_") else moved
    assert interrupted
    assert short.read(preserved)["status"] == "deployed"
    files, _ = deployment.inventory(campaign)
    assert all({name: short.sha256(path / name) for name in files} == files for path in nodes.paths.values())
    if change == "parent_replace":
        assert snapshots(receipt.parent) == {}
    elif change == "leaf_missing":
        assert not receipt.exists()
    elif change == "leaf_content":
        assert receipt.read_bytes() == b"changed published evidence"
