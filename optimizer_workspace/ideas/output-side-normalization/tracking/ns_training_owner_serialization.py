# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serialize complete rank ownership inspections without changing any gate."""

import fcntl
import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path

TIMEOUT = 30
INTERVAL = 0.01


@contextmanager
def inspection_lock(path: Path, *, clock=time.monotonic, pause=time.sleep):
    """Hold one canonical lock through the unchanged complete ownership inspection."""
    if path.parent != path.parent.resolve(strict=True):
        raise RuntimeError("Owner inspection lock directory is not canonical")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    deadline = clock() + TIMEOUT
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode):
            raise RuntimeError("Owner inspection lock is not a direct regular file")
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if clock() >= deadline:
                    raise RuntimeError("Owner inspection lock deadline exhausted") from None
                pause(INTERVAL)
        linked = path.lstat()
        if (linked.st_dev, linked.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError("Owner inspection lock identity changed")
        yield
        linked = path.lstat()
        if (linked.st_dev, linked.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError("Owner inspection lock identity changed during inspection")
    finally:
        os.close(descriptor)
