# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise file-backed precision evidence and public reconstruction without initializing CUDA."""

import ast
import copy
import json
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import ns_precision_placement_execution as execution
import pytest
import selected_lr_release_owner as release_owner
import torch
from selected_lr_release_owner import sha256


def helper_hashes():
    helpers = Path(execution.__file__).resolve().parent
    pending, hashes = list(execution.REQUIRED_HELPERS), {}
    while pending:
        name = pending.pop()
        if name in hashes:
            continue
        path = helpers / name
        hashes[name] = sha256(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.ImportFrom) and node.level:
                dependencies = [node.module] if node.module else [entry.name for entry in node.names]
                pending.extend(f"{module.split('.')[0]}.py" for module in dependencies)
                continue
            imports = (
                [entry.name for entry in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            pending.extend(
                f"{module.split('.')[0]}.py"
                for module in imports
                if module.split(".")[0] not in sys.stdlib_module_names | {"torch", "triton", "numpy"}
            )
    return hashes


@pytest.fixture
def evidence_fixture():
    source = torch.sin(torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) * 0.73) + 0.1
    groups = {
        "continuous": {"source": source, "output": source * 0.5},
        "resumed": {"source": source + 0.02, "output": (source + 0.02) * 0.5},
    }
    references = {
        name: {"5": (pair["source"] * 0.5).double(), "10": (pair["source"] * 0.5).double()}
        for name, pair in groups.items()
    }

    def retain(call, source):
        output = source * 0.5
        phases = [{"phase": phase, "output": output.clone()} for phase in execution.PHASES]
        kernel = execution.trial.summarize_kernel(phases[1:], [0.5] * execution.trial.TIMING_REPLAYS)
        kernel.update(
            source_shape=list(source.shape),
            source_stride=list(source.stride()),
            source_storage_offset=source.storage_offset(),
            rng_exact=True,
            compilation_preserved=True,
            graphs_preserved=True,
            graph_replays=3 + execution.trial.TIMING_REPLAYS,
            input_exact_finite=execution.tracing.comparison(source, source, "input"),
        )
        return phases, kernel

    return groups, references, retain


@pytest.mark.parametrize("index", (0, 1))
def test_placement_files_round_trip_complete_orders_with_recomputed_timing(evidence_fixture, tmp_path, index):
    groups, references, retain = evidence_fixture
    directory = tmp_path / "phases"
    manifest, lifecycle = execution.retain_files(groups, index, directory, retain)
    analysis = execution.summarize(manifest, directory, groups, references, index)
    timing = execution.reconstruct_lifecycle(manifest, directory, lifecycle, groups, index)
    assert len(manifest) == len(analysis["comparisons"]) == 120
    assert len(analysis["source_pairs"]) == len(analysis["same_input_repeats"]) == 60
    assert len(timing) == 20
    assert analysis["all_ten_step_tpu_eager_reproduce_saved_outputs"]
    assert analysis["all_compiled_phases_match_eager_exact_finite"]
    assert analysis["all_same_input_repeats_exact_finite"]
    assert not analysis["all_source_pairs_within_tolerance"]
    assert all(row["median_milliseconds"] == 0.5 for row in timing)
    files = execution.PhaseFiles(manifest, directory, groups, index)
    first = files.load(execution.coordinates(index)[0])
    first.zero_()
    assert torch.count_nonzero(files.load(execution.coordinates(index)[0])) > 0


@pytest.mark.parametrize("fault", ("missing", "duplicate", "order", "alias", "changed"))
def test_placement_files_refuse_partial_or_substituted_inventory(evidence_fixture, tmp_path, fault):
    groups, references, retain = evidence_fixture
    directory = tmp_path / "phases"
    manifest, _ = execution.retain_files(groups, 0, directory, retain)
    if fault == "missing":
        manifest.pop()
    elif fault == "duplicate":
        manifest[-1] = copy.deepcopy(manifest[0])
    elif fault == "order":
        manifest.reverse()
    elif fault == "alias":
        alias = tmp_path / "alias.pt"
        alias.symlink_to(manifest[0]["path"])
        manifest[0]["path"] = str(alias)
    else:
        path = Path(manifest[0]["path"])
        row = torch.load(path, weights_only=True)
        row["output"].add_(0.125)
        torch.save(row, path)
    with pytest.raises(ValueError):
        execution.summarize(manifest, directory, groups, references, 0)


@pytest.mark.parametrize("fault", ("source", "storage", "phase", "partial"))
def test_placement_retention_refuses_mutation_and_missing_batch_evidence(evidence_fixture, tmp_path, fault):
    groups, _, original = evidence_fixture

    def retain(call, source):
        phases, observed = original(call, source)
        if fault == "source":
            groups["resumed"]["source"].add_(0.1)
        elif fault == "storage":
            groups["resumed"]["output"] = groups["resumed"]["output"].clone()
        elif fault == "phase":
            phases.pop()
        else:
            phases[0]["output"] = phases[0]["output"][:, :1]
        return phases, observed

    with pytest.raises(ValueError):
        execution.retain_files(groups, 0, tmp_path / "phases", retain)
    assert list(tmp_path.iterdir())


def test_placement_graph_drift_and_nonfinite_outputs_remain_failed_evidence(evidence_fixture, tmp_path):
    groups, references, original = evidence_fixture

    def retain(call, source):
        phases, observed = original(call, source)
        if call["schedule"] == "q5_tpu" and call["call_index"] == 0:
            phases[-1]["output"][0, 0, 0] = float("inf")
            observed.update(execution.trial.summarize_kernel(phases[1:], observed["cuda_event_milliseconds"]))
        return phases, observed

    directory = tmp_path / "phases"
    manifest, lifecycle = execution.retain_files(groups, 0, directory, retain)
    analysis = execution.summarize(manifest, directory, groups, references, 0)
    timing = execution.reconstruct_lifecycle(manifest, directory, lifecycle, groups, 0)
    assert not analysis["all_graph_phases_match_compiled_exact_finite"]
    assert not timing[0]["all_phase_outputs_exact_finite"]
    assert timing[0]["repeat_comparisons"][-1]["actual_nonfinite_elements"] == 1
    json.dumps(analysis, allow_nan=False)


def test_placement_reverse_order_process_pairing_detects_one_rehashed_late_phase(evidence_fixture, tmp_path):
    groups, _, retain = evidence_fixture
    directories = (tmp_path / "process-0", tmp_path / "process-1")
    manifests = tuple(
        execution.retain_files(groups, index, directory, retain)[0] for index, directory in enumerate(directories)
    )
    first = execution.compare_processes(manifests, directories, groups)
    assert first["all_cross_process_exact_finite"] and len(first["comparisons"]) == 120
    path = Path(manifests[1][5]["path"])
    row = torch.load(path, weights_only=True)
    row["output"][0, 0, 0] += 0.125
    torch.save(row, path)
    manifests[1][5]["sha256"] = sha256(path)
    changed = execution.compare_processes(manifests, directories, groups)
    failures = [row for row in changed["comparisons"] if not row["exact"]]
    assert len(failures) == 1
    assert failures[0]["source"] == "resumed"
    assert failures[0]["occurrence"] == 0
    assert failures[0]["phase"] == "timed_final"


def test_placement_loaded_output_is_independent_after_authenticated_file_is_mutated(evidence_fixture, tmp_path):
    groups, _, retain = evidence_fixture
    directory = tmp_path / "phases"
    manifest, _ = execution.retain_files(groups, 0, directory, retain)
    files = execution.PhaseFiles(manifest, directory, groups, 0)
    loaded = files.load(execution.coordinates(0)[0])
    before = loaded.clone()
    path = Path(manifest[0]["path"])
    path.write_bytes(bytes(path.stat().st_size))
    torch.testing.assert_close(loaded, before, rtol=0, atol=0)
    with pytest.raises(ValueError):
        files.load(execution.coordinates(0)[0])


@pytest.mark.parametrize("fault", ("omitted", "rebound"))
def test_placement_binding_refuses_incomplete_or_rebound_imported_hasher(monkeypatch, fault):
    helpers, hashes = Path(execution.__file__).resolve().parent, helper_hashes()
    if fault == "omitted":
        hashes.pop("selected_lr_release_owner.py")
    else:
        monkeypatch.setattr(release_owner, "__file__", str(helpers.parent / "unaccepted.py"))
    with ExitStack() as stack, pytest.raises((ValueError, RuntimeError)):
        execution._bind(stack, helpers, hashes)


@pytest.mark.parametrize("fault", ("missing_file_and_hash", "missing_hash", "relative"))
def test_placement_binding_rejects_missing_deferred_transitive_import_in_fresh_process(tmp_path, fault):
    helpers = Path(execution.__file__).resolve().parent
    hashes = helper_hashes()
    directory = tmp_path / "helpers"
    directory.mkdir()
    for name in hashes:
        shutil.copy2(helpers / name, directory / name)
    dependency = "from . import unaccepted_transitive" if fault == "relative" else "import unaccepted_transitive.numeric"
    addition = f"\ndef deferred_dependency():\n    {dependency}\n"
    command = (
        "import json\n"
        "from contextlib import ExitStack\n"
        "from pathlib import Path\n"
        "import ns_precision_placement_execution as execution\n"
        "from selected_lr_release_owner import sha256\n"
        f"hashes = {hashes!r}\n"
        "directory = Path(execution.__file__).resolve().parent\n"
        "path = directory / 'diagnose_fused_diagonal.py'\n"
        f"addition = {addition!r}\n"
        "path.write_bytes(path.read_bytes() + addition.encode())\n"
        "hashes[path.name] = sha256(path)\n"
        + ("(directory / 'unaccepted_transitive.py').write_text('VALUE = 1\\n')\n" if fault == "missing_hash" else "")
        + "try:\n"
        "    with ExitStack() as stack:\n"
        "        execution._bind(stack, directory, hashes)\n"
        "except ValueError as error:\n"
        "    print(json.dumps({'refused': str(error)}))\n"
        "else:\n"
        "    raise AssertionError('Accepted an unbound transitive import')\n"
    )
    result = subprocess.run(
        [sys.executable, "-P", "-c", command],
        cwd=directory,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": f"{directory}:{Path(torch.__file__).parents[1]}"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    refused = json.loads(result.stdout)["refused"]
    assert "local import closure is incomplete" in refused
    assert (directory / "unaccepted_transitive.py").exists() == (fault == "missing_hash")


def test_placement_owner_denial_precedes_cuda_initialization_and_output_creation(tmp_path, monkeypatch):
    def denied():
        raise RuntimeError("No accepted precision-placement owner")

    def forbidden(*args, **kwargs):
        pytest.fail("Initialized CUDA without owner acceptance")

    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    with pytest.raises(RuntimeError, match="accepted precision-placement owner"):
        execution.run(tmp_path, {}, 0, tmp_path / "report.json", denied)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fault", ("none", "analysis", "timing", "output", "scope", "device", "lifecycle"))
def test_placement_public_reconstruction_rejects_rehashed_report_substitution(
    evidence_fixture, tmp_path, monkeypatch, fault
):
    groups, references, retain = evidence_fixture
    hashes = helper_hashes()
    config_path, receipt_path = tmp_path / "config.json", tmp_path / "receipt.json"
    config = {"runtime": {"deterministic": True}}
    config_path.write_text(json.dumps(config))
    receipt_path.write_text(json.dumps({"config": str(config_path)}))
    provenance = {"config_sha256": sha256(config_path)}
    reference_provenance = {"scope": "Synthetic persisted-I/O boundary fixture"}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(execution.replay, "ROOT", tmp_path)
    monkeypatch.setattr(execution.replay.analyzer, "RECEIPT_RELATIVE", receipt_path.name)
    monkeypatch.setattr(execution.replay, "RECEIPT_SHA256", sha256(receipt_path))
    monkeypatch.setattr(execution.replay, "load_saved_group", lambda stack: (groups, provenance))
    monkeypatch.setattr(execution.placement, "spectral_reference", lambda stack: (references, reference_provenance))
    report_path = tmp_path / "process-0.json"
    directory = report_path.with_suffix(".phases")
    manifest, lifecycle = execution.retain_files(groups, 0, directory, retain)
    owner = {"physical_uuid": "GPU-00000000-0000-0000-0000-000000000001"}
    report = {
        "status": execution.STATUS,
        "worker_index": 0,
        "owner": owner,
        "report_path": str(report_path),
        "phase_directory": str(directory),
        "phase_files": manifest,
        "coordinates": execution.coordinates(0),
        "schedules": {name: list(schedule) for name, schedule in execution.maps.SCHEDULES.items()},
        "provenance": provenance,
        "reference_provenance": reference_provenance,
        "runtime": config["runtime"],
        "helper_sources_sha256": hashes,
        "physical_uuid": owner["physical_uuid"],
        "actual_uuid": owner["physical_uuid"],
        "lifecycle": lifecycle,
        "analysis": execution.summarize(manifest, directory, groups, references, 0),
        "timings": execution.reconstruct_lifecycle(manifest, directory, lifecycle, groups, 0),
        "rng_exact": True,
        "compilation_preserved": True,
        "graphs_preserved": True,
        "compile_options": {"dynamic": False},
        "scope": execution.SCOPE,
    }
    if fault == "analysis":
        report["analysis"]["all_source_pairs_within_tolerance"] = True
    elif fault == "timing":
        report["timings"][0]["median_milliseconds"] = 0.1
    elif fault == "output":
        path = Path(manifest[-1]["path"])
        row = torch.load(path, weights_only=True)
        row["output"].add_(0.125)
        torch.save(row, path)
        manifest[-1]["sha256"] = sha256(path)
    elif fault == "scope":
        report["scope"] = "production accepted"
    elif fault == "device":
        report["actual_uuid"] = "GPU-00000000-0000-0000-0000-000000000002"
    elif fault == "lifecycle":
        report["lifecycle"][0]["execution"]["graph_replays"] -= 1
    report_path.write_text(json.dumps(report, allow_nan=False))
    if fault == "none":
        rebuilt = execution.reconstruct(report_path, sha256(report_path), owner, hashes, 0)
        assert rebuilt["analysis"] == report["analysis"]
        assert rebuilt["timings"] == report["timings"]
        assert len(rebuilt["phase_files"]) == 120
    else:
        with pytest.raises(ValueError):
            execution.reconstruct(report_path, sha256(report_path), owner, hashes, 0)
