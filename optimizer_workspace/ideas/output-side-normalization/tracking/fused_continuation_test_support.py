# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persisted CPU fixtures for the continuation controller's archived execution paths."""

import importlib
import shutil
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path

from fused_attribution_validation_test_support import keeper, record
from fused_continuation_contract import CPU_STATUS, PREPARATION, PREPARED_STATUS
from owned_output_handoff import PYTHON, TORCHRUN, WORKLOAD, WORKLOAD_ARGS
from selected_lr_release_owner import candidate_files, sha256
from upstream_production import cpu_command

ISSUED_CANDIDATE = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/candidates/fused-continuation-owner-fixture-20261118T0634"
)
CPU_VERIFIER = """
def fixture_check():
    import hashlib
    import sys
    root = Path(sys.argv[sys.argv.index("--root") + 1])
    candidate = Path(sys.argv[sys.argv.index("--candidate") + 1])
    prepared = read(candidate / PREPARATION)
    assert prepared["files_sha256"] == candidate_files(candidate)
    assert prepared["data_receipt_sha256"] == sha256(root / "data/experiment_receipt.json")
    assert all(sha256(root / name) == value
               for name, value in prepared["allocation_predecessor"]["evidence_sha256"].items())
    print(json.dumps({"status": prepared["status"]}))

if __name__ == "__main__":
    fixture_check()
"""


def write_process(proc: Path, process: dict, cwd: Path, environment: dict) -> None:
    directory = proc / str(process["pid"])
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[:4] = [process["status"], str(process["parent"]), str(process["group"]), str(process["session"])]
    fields[19] = process["start_ticks"]
    (directory / "stat").write_text(f"{process['pid']} (continuation cpu fixture) {' '.join(fields)}\n")
    (directory / "cmdline").write_bytes(("\0".join(process["command"]) + "\0").encode())
    (directory / "environ").write_bytes(
        ("\0".join(f"{key}={value}" for key, value in environment.items()) + "\0").encode()
    )
    link = directory / "cwd"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(cwd, target_is_directory=True)


def failed_attribution_fixture(root: Path, controller) -> dict:
    binding_path, helpers, state_path, _, gpu_path = controller.parent_locations(root)
    helpers.mkdir(parents=True)
    manager = helpers / controller.MANAGER
    sibling = Path(controller.__file__).with_name(controller.MANAGER)
    manager.write_bytes(sibling.read_bytes() if sibling.is_file() else b"#!/bin/bash\nexit 0\n")
    binding = {
        "root": str(root),
        "helpers": str(helpers),
        "helpers_sha256": {manager.name: sha256(manager)},
    }
    record(binding_path, binding)
    restored = keeper()
    for pid, process in restored["generation"].items():
        process["command"] = (
            [PYTHON, TORCHRUN, "--standalone", "--nproc-per-node=8", WORKLOAD, *WORKLOAD_ARGS]
            if int(pid) == restored["pid"]
            else [PYTHON, "-u", WORKLOAD, *WORKLOAD_ARGS]
        )
    log = helpers.parent / "validator.log"
    log.write_text("preserved fused checkpoint continuation failure\n")
    owner = {"controller_pid": 201, "controller_start_ticks": "301"}
    state = owner | {
        "status": "needs_attention",
        "node": controller.NODE,
        "binding_sha256": sha256(binding_path),
        "child_pid": 202,
        "child_start_ticks": "302",
        "child_marker": "failed-child",
        "exit_code": 1,
        "cleanup_exit_code": 1,
        "keeper_restore": 0,
        "marked_descendants_absent": True,
        "keeper_release_complete": True,
        "restored_keeper": restored,
        "log": str(log),
        "log_sha256": sha256(log),
    }
    record(state_path, state)
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "pid": 201,
            "start_ticks": "301",
            "marker": "failed-controller",
            "binding_sha256": sha256(binding_path),
        },
    )
    commands = []
    for index in range(6):
        path = helpers.parent / f"command-{index}.log"
        path.write_text(f"completed bounded fixture command {index}\n")
        commands.append({"exit_code": 0, "log": str(path), "log_sha256": sha256(path)})
    record(
        gpu_path,
        {
            "status": "needs_attention",
            "owner": owner,
            "execution_binding_sha256": sha256(binding_path),
            "error": "AssertionError: rank_0.model.body.emb.weight",
            "gpu_commands": commands,
        },
    )
    record(root / "data/experiment_receipt.json", {"cpu_fixture": True})
    return controller.predecessor(root)


def candidate_fixture(root: Path, candidate: Path, helpers: Path, controller) -> Path:
    previous = failed_attribution_fixture(root, controller)
    helpers.mkdir()
    for name, source in controller.helper_sources(root, Path(controller.__file__).parent).items():
        shutil.copy2(source, helpers / name)
    verifier = helpers / "prepare_fused_continuation.py"
    text = verifier.read_text()
    original = 'if __name__ == "__main__":\n    main()\n'
    assert text.count(original) == 1
    # CPU acceptance is an external fixture boundary; controller/archive predicates remain real.
    verifier.write_text(text.replace(original, CPU_VERIFIER))
    for name in controller.fixture_names():
        path = candidate / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ISSUED_CANDIDATE / name, path)
    (candidate / "experiment/numerical_fixture.py").write_text("# unchanged numerical fixture\n")
    (candidate / "artifacts/pinned-sources-materialized").mkdir(parents=True)
    (candidate / "artifacts/pinned-sources-materialized/fixture.txt").write_text("pinned fixture\n")
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    prepared = {
        "root": str(root),
        "candidate": str(candidate),
        "status": PREPARED_STATUS,
        "files_sha256": candidate_files(candidate),
        "parent_metadata_sha256": {},
        "allocation_predecessor": previous,
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "helpers_sha256": {verifier.name: sha256(verifier)},
    }
    record(candidate / PREPARATION, prepared)
    out = root.parent / "candidate-cpu"
    out.mkdir()
    log = out / "candidate.log"
    log.write_text("synthetic external candidate acceptance\n")
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in (*prepared["files_sha256"], PREPARATION):
            bundle.add(candidate / name, arcname=name, recursive=False)
        bundle.add(verifier, arcname=f"tracking/{verifier.name}", recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    receipt = out / "receipt.json"
    record(
        receipt,
        {
            "root": str(root),
            "candidate": str(candidate),
            "status": CPU_STATUS,
            "exit_code": 0,
            "cuda_visible_devices": "",
            "command": cpu_command(),
            "files_sha256": prepared["files_sha256"],
            "preparation_sha256": sha256(candidate / PREPARATION),
            "helpers": str(helpers),
            "helpers_sha256": prepared["helpers_sha256"],
            "log": str(log),
            "log_sha256": sha256(log),
            "archive": str(archive),
            "archive_sha256": sha256(archive),
        },
    )
    return receipt


@contextmanager
def imported_controller(helpers: Path):
    """Load the archived helper graph without checkout imports leaking across the boundary."""
    names = {path.stem for path in helpers.glob("*.py")}
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    old_path = list(sys.path)
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(helpers))
        yield importlib.import_module("fused_continuation_controller")
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
