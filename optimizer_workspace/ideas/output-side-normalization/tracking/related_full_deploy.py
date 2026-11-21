# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy only bound full configs, preserving every identical remote file."""

import argparse
import hashlib
import io
import json
import os
import shlex
import stat
import subprocess
import tarfile
from pathlib import Path

import related_full_campaign as full_campaign
from experiment.launch import NODES, PYTHON, SSH
from experiment.records import utc_now
from related_campaign import read, sha256

APPROVED_NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")

REMOTE_SCRIPT = r"""
import hashlib, io, json, os, pathlib, stat, sys, tarfile

root, phase, inventory = pathlib.Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
directory = pathlib.Path(sys.argv[4])
if not root.is_absolute() or ".." in root.parts or phase not in ("check", "write", "verify"):
    raise ValueError("Invalid remote deployment request")
if directory.is_absolute() or ".." in directory.parts:
    raise ValueError("Invalid config directory")
for name in inventory:
    path = pathlib.Path(name)
    if path.parent != directory or path.suffix != ".json" or str(path) != name:
        raise ValueError("Deployment inventory is outside the exact config directory")

payloads = {}
if phase == "write":
    with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.name not in inventory or member.name in payloads:
                raise ValueError("Archive differs from the bound regular-file inventory")
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != inventory[member.name]:
                raise ValueError("Archive content differs from the campaign hash")
            payloads[member.name] = data
    if set(payloads) != set(inventory):
        raise ValueError("Archive omits bound configs")

def folder(create):
    handle = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in (*root.parts[1:], *directory.parts):
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=handle)
            except FileNotFoundError:
                if not create:
                    os.close(handle)
                    return None
                try:
                    os.mkdir(part, dir_fd=handle)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=handle)
            os.close(handle)
            handle = child
        return handle
    except BaseException:
        os.close(handle)
        raise

def scan(handle):
    result = {}
    for name, expected in inventory.items():
        leaf = pathlib.Path(name).name
        try:
            before = os.stat(leaf, dir_fd=handle, follow_symlinks=False) if handle is not None else None
        except FileNotFoundError:
            before = None
        if before is None:
            if phase == "verify":
                raise FileNotFoundError("Missing deployed config: " + name)
            result[name] = "absent"
            continue
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Nonregular or redirected remote config: " + name)
        descriptor = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=handle)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("Remote config changed file type: " + name)
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.stat(leaf, dir_fd=handle, follow_symlinks=False)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        if identity(before) != identity(opened) or identity(before) != identity(after):
            raise ValueError("Remote config changed during hashing: " + name)
        if actual != expected:
            raise ValueError("Refusing divergent existing remote config: " + name)
        result[name] = "identical"
    return result

handle = folder(phase == "write")
try:
    observed = scan(handle)
    created = []
    if phase == "write":
        for name, state in observed.items():
            if state == "identical":
                continue
            descriptor = os.open(
                pathlib.Path(name).name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o644,
                dir_fd=handle,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payloads[name])
                stream.flush()
                os.fsync(stream.fileno())
            created.append(name)
        scan(handle)
    current = folder(False)
    try:
        if handle is not None and (
            current is None
            or (os.fstat(handle).st_dev, os.fstat(handle).st_ino)
            != (os.fstat(current).st_dev, os.fstat(current).st_ino)
        ):
            raise ValueError("Remote config directory changed during deployment")
    finally:
        if current is not None:
            os.close(current)
    print(json.dumps({"phase": phase, "files": observed, "created": created}, sort_keys=True))
finally:
    if handle is not None:
        os.close(handle)
"""


def inventory(campaign: dict) -> tuple[dict, bytes]:
    """Build an archive containing exactly the generated configs and four waves."""
    source = Path(campaign["source"])
    directory = Path(full_campaign.CONFIG_DIRECTORY)
    expected = {str(Path(trial["config"]).relative_to(source)): trial["config_sha256"] for trial in campaign["trials"]}
    expected.update(
        {str(Path(wave["manifest"]).relative_to(source)): wave["manifest_sha256"] for wave in campaign["waves"]}
    )
    selected = {
        name: digest for name, digest in campaign["files_sha256"].items() if Path(name).is_relative_to(directory)
    }
    if selected != expected or source.resolve() != source:
        raise ValueError("Full deployment inventory differs from the exact campaign files")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, expected_hash in sorted(selected.items()):
            path = source / name
            if Path(name).parent != directory or path.suffix != ".json" or path.resolve() != path:
                raise ValueError(f"Redirected or out-of-scope full config: {path}")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ValueError(f"Full config changed while archiving: {path}")
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(payload), 0o644
            archive.addfile(member, io.BytesIO(payload))
    return dict(sorted(selected.items())), buffer.getvalue()


def remote(node: str, source: Path, phase: str, files: dict, payload: bytes = b"") -> dict:
    """Run the same no-follow filesystem protocol on one bound SSH node."""
    command = [
        PYTHON,
        "-B",
        "-c",
        REMOTE_SCRIPT,
        str(source),
        phase,
        json.dumps(files, sort_keys=True),
        full_campaign.CONFIG_DIRECTORY,
    ]
    result = subprocess.run([*SSH, f"root@{node}", shlex.join(command)], input=payload, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{node} {phase}: {result.stderr.decode(errors='replace').strip()}")
    record = json.loads(result.stdout)
    if (
        record["phase"] != phase
        or set(record["files"]) != set(files)
        or any(value not in ("absent", "identical") for value in record["files"].values())
        or (phase == "verify" and any(value != "identical" for value in record["files"].values()))
        or not set(record["created"]).issubset(files)
    ):
        raise ValueError(f"{node}: Remote deployment response differs from the bound inventory")
    return {"node": node, **record}


def require_receipt_path(root: Path, path: Path) -> None:
    if path.parent != root / "run-preparation" or path.resolve() != path:
        raise ValueError("Deployment receipt must be canonical and directly under root/run-preparation")
    if os.path.lexists(path):
        raise FileExistsError(f"Preserve existing deployment receipt: {path}")


def require_approved_nodes(source: Path) -> None:
    if NODES != APPROVED_NODES or read(source / "experiment/cluster.json")["nodes"] != list(APPROVED_NODES):
        raise ValueError("Deployment nodes differ from the independently approved allocation")


def open_directory(path: Path) -> int:
    """Open a canonical directory without following any ancestor symlink."""
    handle = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=handle)
            os.close(handle)
            handle = child
        return handle
    except BaseException:
        os.close(handle)
        raise


def require_publication(path: Path, parent: int, published: os.stat_result, payload: bytes) -> None:
    """Verify the canonical receipt still names the exact file just written."""
    expected_parent = os.fstat(parent)
    current = open_directory(path.parent)
    try:
        information = os.fstat(current)
        if (information.st_dev, information.st_ino) != (expected_parent.st_dev, expected_parent.st_ino):
            raise ValueError("Receipt parent changed during publication")
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
        with os.fdopen(descriptor, "rb") as stream:
            information = os.fstat(stream.fileno())
            if not stat.S_ISREG(information.st_mode) or (information.st_dev, information.st_ino) != (
                published.st_dev,
                published.st_ino,
            ):
                raise ValueError("Receipt leaf changed during publication")
            if stream.read() != payload:
                raise ValueError("Receipt content changed during publication")
            after = os.fstat(stream.fileno())
        leaf = os.stat(path.name, dir_fd=current, follow_symlinks=False)
        if (
            (leaf.st_dev, leaf.st_ino, leaf.st_size, leaf.st_mtime_ns, leaf.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            != (information.st_size, information.st_mtime_ns, information.st_ctime_ns)
            or path.resolve() != path
        ):
            raise ValueError("Receipt target changed while verifying publication")
    finally:
        os.close(current)
    current = open_directory(path.parent)
    try:
        information = os.fstat(current)
        if (information.st_dev, information.st_ino) != (expected_parent.st_dev, expected_parent.st_ino):
            raise ValueError("Receipt parent changed while verifying publication")
    finally:
        os.close(current)


def deploy(root: Path, receipt: Path | None = None, *, check_only: bool = False) -> dict:
    """Preflight every remote before exclusive creation, then hash-verify all three."""
    root = root.resolve()
    if not check_only and receipt is None:
        raise ValueError("Deployment requires a fresh exclusive receipt")
    if not check_only:
        receipt = receipt.absolute()
        require_receipt_path(root, receipt)
    campaign = full_campaign._bound_campaign(root)
    campaign_path = full_campaign.campaign_path(root)
    campaign_hash = sha256(campaign_path)
    source = Path(campaign["source"])
    require_approved_nodes(source)
    files, payload = inventory(campaign)
    checked = [remote(node, source, "check", files) for node in APPROVED_NODES[1:]]
    if full_campaign._bound_campaign(root) != campaign or sha256(campaign_path) != campaign_hash:
        raise ValueError("Full campaign changed during remote preflight")
    require_approved_nodes(source)
    result = {
        "schema_version": 1,
        "time": utc_now(),
        "status": "checked" if check_only else "deployed",
        "campaign": str(campaign_path),
        "campaign_sha256": campaign_hash,
        "source": str(source),
        "helper_sha256": sha256(Path(__file__)),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
        "preflight": checked,
    }
    if check_only:
        return result
    result["writes"] = [remote(node, source, "write", files, payload) for node in APPROVED_NODES[1:]]
    result["verification"] = [remote(node, source, "verify", files) for node in APPROVED_NODES[1:]]
    if full_campaign._bound_campaign(root) != campaign or sha256(campaign_path) != campaign_hash:
        raise ValueError("Full campaign changed during deployment; preserve created files without a success receipt")
    require_approved_nodes(source)
    require_receipt_path(root, receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    descriptor = open_directory(receipt.parent)
    try:
        require_receipt_path(root, receipt)
        parent = receipt.parent.stat()
        opened = os.fstat(descriptor)
        if (parent.st_dev, parent.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("Receipt parent changed during deployment")
        output = os.open(receipt.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=descriptor)
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        with os.fdopen(output, "w") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
            published = os.fstat(stream.fileno())
        os.fsync(descriptor)
        require_publication(receipt, descriptor, published, serialized.encode())
    finally:
        os.close(descriptor)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and args.receipt is None:
        parser.error("Deployment requires --receipt; --check-only never writes a receipt")
    print(json.dumps(deploy(args.root, args.receipt, check_only=args.check_only), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
