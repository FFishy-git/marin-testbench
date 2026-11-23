# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore and verify the pinned Toner artifact on a clean GitHub checkout."""

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from .reference import REVISION, TASK, UPSTREAM

ARCHIVE_SHA256 = "fcdbbbdbaf07b6af7acab6250bd206e97724e65706d77e3cab4cbba34974b9ba"
ARCHIVE_URL = f"https://codeload.github.com/mohamad-amin/Toner/tar.gz/{REVISION}"
SOURCE_MANIFEST = TASK / "experiment" / "upstream_manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_snapshot(root=UPSTREAM):
    """Refuse missing or modified pinned source files."""
    expected = json.loads(SOURCE_MANIFEST.read_text())["files"]
    for name, sha256 in expected.items():
        path = root / name
        if not path.is_file() or digest(path) != sha256:
            raise ValueError(f"Pinned Toner source missing or modified: {path}")
    return len(expected)


def install_archive(archive_path, destination):
    """Extract only a verified archive and never overwrite an existing snapshot."""
    if digest(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Toner archive SHA256 differs from the recorded source")
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(directory, filter="data")
        source = Path(directory) / f"Toner-{REVISION}"
        verify_snapshot(source)
        source.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default="http://star-proxy.oa.com:3128")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if UPSTREAM.exists():
        print(f"Verified {verify_snapshot()} pinned Toner files at {UPSTREAM}")
        return
    if args.check_only:
        raise FileNotFoundError(UPSTREAM)
    UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
    if args.archive:
        install_archive(args.archive, UPSTREAM)
    else:
        opener = build_opener(ProxyHandler({"http": args.proxy, "https": args.proxy}))
        with tempfile.NamedTemporaryFile(dir=UPSTREAM.parent, suffix=".tar.gz") as stream:
            with opener.open(ARCHIVE_URL, timeout=120) as response:
                while chunk := response.read(1 << 20):
                    stream.write(chunk)
            stream.flush()
            install_archive(Path(stream.name), UPSTREAM)
    print(f"Installed and verified {verify_snapshot()} pinned Toner files at {UPSTREAM}")


if __name__ == "__main__":
    main()
