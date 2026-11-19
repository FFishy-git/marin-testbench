# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise complete archived-parent preservation and the unissued adapter boundary."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import ns_training_path_execution_source as source
import pytest
from ns_training_path_test_support import PRESERVED, record
from selected_lr_release_owner import candidate_files, sha256


@pytest.fixture
def owner_candidate(tmp_path):
    root, original, candidate, archived = (
        tmp_path / name for name in ("root", "original", "candidate", "parent-archive")
    )
    retained = Path(__file__).resolve().parents[1]
    if not (retained / source.contract.specs()[0]["config_name"]).is_file():
        retained = PRESERVED
    original.mkdir()
    for directory in ("experiment", "artifacts/pinned-sources-materialized"):
        shutil.copytree(
            retained / directory,
            original / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    if (retained / source.PARENT_OWNER).is_file():
        shutil.copy2(retained / source.PARENT_OWNER, original / source.OWNER)
    (original / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for path in retained.glob("*.json"):
        if path.name == source.PREPARATION:
            continue
        shutil.copy2(path, original / path.name)
    shutil.copytree(original, archived, symlinks=True)
    shutil.copytree(original, candidate, symlinks=True)
    (candidate / source.OWNER).write_text(source.ADAPTER)
    record(root / "data/experiment_receipt.json", {"data": "CPU owner-only inventory fixture"})
    record(root / source.predecessor.SEAL, {"seal": "CPU fixture, no ownership authority"})
    files = candidate_files(archived)
    previous = {
        "candidate": str(original),
        "helpers": str(archived / "tracking"),
        "helpers_sha256": {},
        "inventory_sha256": files | {path.name: sha256(path) for path in archived.glob("*.json")},
    }
    modules = (source.contract, source.validation, source.parent, source.predecessor)
    sources = {Path(module.__file__).name: Path(module.__file__).resolve() for module in modules}
    sources.update(
        {
            "ns_full_model_contract.py": Path(source.contract.specs.__code__.co_filename),
            "ns_full_model_validation.py": Path(source.validation.trial.__code__.co_filename),
        }
    )
    prepared = source.preparation_record(root, candidate, previous, sources)
    record(candidate / source.PREPARATION, prepared)
    return root, original, candidate, archived, previous, sources


@pytest.mark.parametrize(
    "fault",
    ("none", "trainer", "toner_state", "capture", "config", "config_both", "removed", "extra"),
)
def test_owner_only_candidate_retains_all_archived_source_and_config_bytes(owner_candidate, fault):
    root, original, candidate, archived, previous, sources = owner_candidate
    if fault in ("trainer", "toner_state", "capture"):
        name = {"trainer": "train", "toner_state": "toner_state", "capture": "capture"}[fault]
        with (candidate / f"experiment/{name}.py").open("a") as stream:
            stream.write("\n# unaccepted source change\n")
    elif fault in ("config", "config_both"):
        name = source.contract.specs()[0]["config_name"]
        paths = (candidate / name, original / name) if fault == "config_both" else (candidate / name,)
        for path in paths:
            config = json.loads(path.read_bytes())
            config["runtime"]["compile"] = not config["runtime"]["compile"]
            record(path, config)
        prepared = json.loads((candidate / source.PREPARATION).read_bytes())
        prepared["files_sha256"] = candidate_files(candidate)
        record(candidate / source.PREPARATION, prepared)
    elif fault == "removed":
        (candidate / "experiment/capture.py").unlink()
    elif fault == "extra":
        (candidate / "experiment/unaccepted_config.json").write_text("{}\n")
    archived_before = candidate_files(archived)
    if fault == "none":
        prepared = source.require_candidate(root, candidate, previous, sources)
        assert prepared["changed_files"] == [source.OWNER]
        assert prepared["files_sha256"] == candidate_files(archived) | {source.OWNER: sha256(candidate / source.OWNER)}
        assert len(prepared["commands"]) == 54
    else:
        with pytest.raises(ValueError):
            source.require_candidate(root, candidate, previous, sources)
    assert candidate_files(archived) == archived_before


def test_execution_closure_keeps_parent_bytes_and_discovers_executable_dependencies(tmp_path, monkeypatch):
    inherited, local = tmp_path / "inherited", tmp_path / "new"
    inherited.mkdir()
    local.mkdir()
    (inherited / "accepted_reader.py").write_text("VALUE = 'accepted original'\n")
    for name in source.contract.NUMERICAL_READERS:
        origin = Path(source.contract.specs.__code__.co_filename).parent / name
        shutil.copy2(origin, inherited / name)
    (local / "accepted_reader.py").write_text("raise RuntimeError('mutable shadow')\n")
    (local / "ns_training_path_execution_source.py").write_text(
        "import accepted_reader\nimport ns_training_path_extra_reader\n"
    )
    (local / "ns_training_path_extra_reader.py").write_text("import json\nVALUE = 42\n")
    hashes = {path.name: sha256(path) for path in inherited.iterdir()}
    previous = {"helpers": str(inherited), "helpers_sha256": hashes}
    monkeypatch.setattr(source, "__file__", str(local / "ns_training_path_execution_source.py"))
    monkeypatch.setattr(source, "NEW_HELPERS", ("ns_training_path_execution_source.py",))
    result = source.execution_sources(previous)
    assert result["accepted_reader.py"] == inherited / "accepted_reader.py"
    assert result["ns_training_path_extra_reader.py"] == local / "ns_training_path_extra_reader.py"
    assert {name: sha256(result[name]) for name in hashes} == hashes
    (local / "ns_training_path_extra_reader.py").write_text("import mutable_unaccepted_reader\n")
    with pytest.raises(ValueError):
        source.execution_sources(previous)


def test_unissued_candidate_adapter_denies_before_importing_an_execution_owner(tmp_path):
    package = tmp_path / "unissued_training_path"
    experiment = package / "experiment"
    experiment.mkdir(parents=True)
    (experiment / "__init__.py").write_text("")
    shutil.copy2(Path(source.sha256.__code__.co_filename), experiment / "selected_lr_release_owner.py")
    path = experiment / "fused_attribution_owner.py"
    path.write_text(source.ADAPTER)
    spec = importlib.util.spec_from_file_location(
        "unissued_training_path_experiment",
        experiment / "__init__.py",
        submodule_search_locations=[str(experiment)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    owner_spec = importlib.util.spec_from_file_location(f"{spec.name}.fused_attribution_owner", path)
    adapter = importlib.util.module_from_spec(owner_spec)
    owner_spec.loader.exec_module(adapter)
    before = sys.modules.get("ns_training_path_gpu_owner")
    try:
        with pytest.raises(RuntimeError, match="not separately prepared"):
            adapter.require_ns_training_path_gpu_owner(tmp_path)
        assert sys.modules.get("ns_training_path_gpu_owner") is before
        assert not (tmp_path / "run-preparation").exists()
    finally:
        sys.modules.pop(spec.name, None)
        sys.modules.pop(f"{spec.name}.selected_lr_release_owner", None)
