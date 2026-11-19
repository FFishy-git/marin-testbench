# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise retained failures and distinct source-bound recovery acceptance on CPU."""

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import ns_training_path_gpu_owner as original_owner
import ns_training_path_recovery_predecessor as predecessor
import ns_training_path_recovery_source as source
import prepare_ns_training_path_recovery as preparation
import pytest
from ns_training_path_test_support import record
from selected_lr_release_owner import candidate_files, read, sha256

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def original_records():
    binding, helpers, state, _, result = predecessor.original.locations(ROOT)
    return read(binding), read(helpers.parent / "controller.json"), read(state), read(result)


def test_derived_recovery_keeps_numerical_protocol_and_changes_execution_namespace(tmp_path):
    parent = source.recorded_parent(ROOT)
    generated = source.derived_sources(parent)
    for name, target in source.MODULES.items():
        original = ast.parse((Path(parent["helpers"]) / f"{name}.py").read_bytes())
        actual = ast.parse(generated[f"{target}.py"])
        functions = {node.name: node for node in actual.body if isinstance(node, ast.FunctionDef)}
        replacements = {
            "ns_training_path_gpu_owner": {"predecessor_binding", "require_sealed_predecessor", "require_binding"},
            "ns_training_path_gpu_controller": {"freeze"},
        }.get(name, set())
        for node in original.body:
            if isinstance(node, ast.FunctionDef) and node.name not in replacements:
                expected = source.RecoveryNamespace().visit(copy.deepcopy(node))
                assert ast.dump(functions[node.name]) == ast.dump(expected)
    contract_path = tmp_path / "contract.py"
    contract_path.write_bytes(generated["ns_training_path_recovery_contract.py"])
    spec = importlib.util.spec_from_file_location("recovery_contract_fixture", contract_path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    parent_contract = predecessor.original
    candidate = Path(parent["candidate"])
    commands = contract.commands(candidate, ROOT, contract.locations(ROOT)[3])
    assert len(commands) == 54
    assert contract.specs() == parent_contract.specs()
    assert contract.TOLERANCE == parent_contract.TOLERANCE
    assert set(contract.locations(ROOT)).isdisjoint(parent_contract.locations(ROOT))
    for actual, original in zip(
        commands, parent_contract.commands(candidate, ROOT, parent_contract.locations(ROOT)[3]), strict=True
    ):
        assert actual[actual.index("--config") + 1] == original[original.index("--config") + 1]
        assert actual[actual.index("--max-steps") + 1] == original[original.index("--max-steps") + 1]
    controller = ast.parse(generated["ns_training_path_recovery_gpu_controller.py"])
    literals = {
        node.value for node in ast.walk(controller) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert f"{source.NAMESPACE}-terminal-" in literals
    assert "ns-training-path-terminal-" not in literals


def test_recovery_closure_retains_parent_readers_and_only_replaces_measured_handoff():
    parent = source.recorded_parent(ROOT)
    contents = preparation.helper_bytes(parent)
    hashes = preparation.digest_bytes(contents)
    changed = {name for name, digest in parent["helpers_sha256"].items() if hashes[name] != digest}
    assert changed == {"owned_output_handoff.py"}
    assert hashes["owned_output_handoff.py"] == source.HANDOFF_SHA256
    assert hashes["test_owned_output_handoff.py"] == source.HANDOFF_TEST_SHA256
    assert all(hashes[name] == digest for name, digest in predecessor.original.NUMERICAL_READERS.items())


@pytest.mark.parametrize("fault", ("none", "decision", "coverage", "cleanup", "owner", "failed_command"))
def test_incomplete_failure_retains_rejections_and_refuses_completion_claims(fault):
    frozen, intent, state, result = original_records()
    if fault == "decision":
        result["results"][6]["accepted"] = True
    elif fault == "coverage":
        result["gpu_commands"].append(copy.deepcopy(result["gpu_commands"][-1]))
    elif fault == "cleanup":
        state["cleanup_exit_code"] = 0
    elif fault == "owner":
        state["child_generation"]["parent"] += 1
    elif fault == "failed_command":
        frozen["commands"][44] = frozen["commands"][43]
    if fault == "none":
        saved = predecessor.failed_structure(ROOT, frozen, intent, state, result)
        assert (saved["completed_commands"], saved["case_decisions"]) == (44, 14)
        assert len(saved["accepted_cases"]) == 6 and len(saved["rejected_cases"]) == 8
        assert saved["coverage_complete"] is False and saved["numerically_accepted"] is False
        assert saved["decisions"] == result["results"]
    else:
        with pytest.raises(ValueError):
            predecessor.failed_structure(ROOT, frozen, intent, state, result)


@pytest.mark.parametrize("fault", ("none", "original_pid", "followup_pid", "marked_descendant"))
def test_failed_attempt_owner_absence_rejects_present_handles_and_descendants(monkeypatch, fault):
    frozen, intent, state, result = original_records()
    followup = read(ROOT / predecessor.WORKS[0][0] / "intent.json")
    present = {"original_pid": intent["pid"], "followup_pid": followup["pid"]}.get(fault)
    monkeypatch.setattr(predecessor.handoff, "observed", lambda pid: {"pid": pid} if pid == present else None)
    monkeypatch.setattr(
        predecessor.handoff,
        "marker_processes",
        lambda *args: {990001: {"pid": 990001}} if fault == "marked_descendant" else {},
    )
    if fault == "none":
        saved = predecessor.require_absent_owners(ROOT, frozen, intent, state, result)
        assert len(saved) == 2
        assert all(item["state"]["status"] == "needs_attention" for item in saved)
        assert all(not item["state"].get("phases", []) for item in saved)
    else:
        with pytest.raises(RuntimeError):
            predecessor.require_absent_owners(ROOT, frozen, intent, state, result)


@pytest.fixture
def prepared_candidate(tmp_path, monkeypatch):
    for name, value in source.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    candidate = tmp_path / "candidate"
    prepared = preparation.prepare(ROOT, candidate)
    return candidate, prepared


@pytest.mark.parametrize("fault", ("none", "trainer", "config", "metadata", "extra"))
def test_recovery_candidate_acceptance_rejects_changes_outside_owner_adapter(prepared_candidate, fault):
    candidate, prepared = prepared_candidate
    if fault == "trainer":
        with (candidate / "experiment/train.py").open("a") as stream:
            stream.write("\n# unaccepted training change\n")
    elif fault == "config":
        name = predecessor.original.specs()[0]["config_name"]
        config = read(candidate / name)
        config["runtime"]["compile"] = not config["runtime"]["compile"]
        record(candidate / name, config)
        prepared["files_sha256"] = candidate_files(candidate)
        record(candidate / source.PREPARATION, prepared)
    elif fault == "metadata":
        prepared["parent_metadata_sha256"] = {}
        record(candidate / source.PREPARATION, prepared)
    elif fault == "extra":
        record(candidate / "experiment/configs/unaccepted.json", {})
    if fault == "none":
        checked = preparation.require_preparation(ROOT, candidate)
        assert checked["changed_files"] == [source.OWNER]
        assert checked["parent_files_sha256"] == source.recorded_parent(ROOT)["files_sha256"]
    else:
        with pytest.raises(ValueError):
            preparation.require_preparation(ROOT, candidate)


@pytest.fixture
def visible_binding(tmp_path, monkeypatch):
    candidate, parent, helpers, cpu_directory = (tmp_path / name for name in ("candidate", "parent", "helpers", "cpu"))
    for directory in (candidate, parent, helpers, cpu_directory):
        directory.mkdir()
    for directory, content in ((candidate, source.ADAPTER), (parent, "original adapter\n")):
        (directory / "experiment").mkdir()
        (directory / source.OWNER).write_text(content)
    record(candidate / source.PREPARATION, {"fixture": "prepared CPU recovery"})
    record(tmp_path / "data/experiment_receipt.json", {"fixture": "data receipt"})
    record(tmp_path / "old-evidence.json", {"fixture": "retained rejection"})
    (tmp_path / "old-rank.pt").write_bytes(b"CPU checkpoint identity fixture")
    for name in ("source.tar", "tests.xml", "cpu.log"):
        (cpu_directory / name).write_bytes(b"CPU accepted evidence fixture")
    (tmp_path / "issued.tar").write_bytes(b"CPU issued archive fixture")
    helper_hashes = {
        "ns_training_path_recovery_predecessor.py": sha256(Path(predecessor.__file__)),
        "prepare_ns_training_path_recovery.py": sha256(Path(preparation.__file__)),
    }
    for name in helper_hashes:
        (helpers / name).write_bytes(Path(preparation.__file__).with_name(name).read_bytes())
    cpu_path = cpu_directory / "receipt.json"
    monkeypatch.setattr(preparation, "CPU_PATH", str(cpu_path.relative_to(tmp_path)))
    cpu = {
        "status": source.CPU_STATUS,
        "root": str(tmp_path),
        "candidate": str(candidate),
        "cuda_visible_devices": "",
        "exit_code": 0,
        "parent_cpu_receipt_sha256": preparation.PARENT_SHA256,
        "helpers_sha256": helper_hashes,
        "files_sha256": {source.OWNER: sha256(candidate / source.OWNER)},
        "inventory_sha256": {},
        **{
            name: str(cpu_directory / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        },
        **{
            f"{name}_sha256": sha256(cpu_directory / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        },
    }
    record(cpu_path, cpu)
    sealed = {
        "restored_keeper": {"devices": [["0", "CPU-UUID", "CPU fixture"]]},
        "evidence": original_owner.authenticated_inventory(
            {str(tmp_path / "old-evidence.json"): sha256(tmp_path / "old-evidence.json")}
        ),
        "checkpoint_payloads": original_owner.authenticated_inventory(
            {str(tmp_path / "old-rank.pt"): sha256(tmp_path / "old-rank.pt")}
        ),
    }
    sealed["evidence"] = {Path(name).name: item for name, item in sealed["evidence"].items()}
    sealed["checkpoint_payloads"] = {Path(name).name: item for name, item in sealed["checkpoint_payloads"].items()}
    record(tmp_path / predecessor.SEAL, sealed)
    binding = tmp_path / "binding.json"
    commands = [[f"CPU-command-{index}"] for index in range(54)]
    plan = {"commands": commands, "trials": list(range(18)), "tolerance": {"rtol": 1e-5, "atol": 1e-6}}
    contract = SimpleNamespace(
        FROZEN_STATUS="CPU recovery binding",
        ENVIRONMENT={"CUDA_VISIBLE_DEVICES": "CPU fixture"},
        TOLERANCE=plan["tolerance"],
        locations=lambda root: (binding, helpers, root / "state.json", root / "work", root / "result.json"),
        commands=lambda *args: commands,
    )
    frozen = {
        "status": contract.FROZEN_STATUS,
        "root": str(tmp_path),
        "candidate": str(candidate),
        "parent_candidate": str(parent),
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": sha256(cpu_path),
        "helpers": str(helpers),
        "helpers_sha256": helper_hashes,
        "files_sha256": cpu["files_sha256"],
        "parent_files_sha256": {source.OWNER: sha256(parent / source.OWNER)},
        "preparation_sha256": sha256(candidate / source.PREPARATION),
        "allocation_predecessor": sealed,
        "predecessor_acceptance": preparation.predecessor_binding(tmp_path),
        "devices": sealed["restored_keeper"]["devices"],
        "node": predecessor.NODE,
        "environment": contract.ENVIRONMENT,
        "controller_environment": {"CUDA_VISIBLE_DEVICES": ""},
        "scope": source.SCOPE,
        "allowed_transform": source.OWNER,
        "controller_command": ["CPU-controller"],
        "validator_command": ["CPU-validator"],
        "validation_timeout": 10,
        "archive": str(tmp_path / "issued.tar"),
        "archive_sha256": sha256(tmp_path / "issued.tar"),
        "data_receipt_sha256": sha256(tmp_path / "data/experiment_receipt.json"),
        "commands": commands,
        "plan": plan,
    }
    paths = {
        str(cpu_path): sha256(cpu_path),
        frozen["archive"]: frozen["archive_sha256"],
        str(candidate / source.PREPARATION): frozen["preparation_sha256"],
        str(tmp_path / predecessor.SEAL): frozen["predecessor_acceptance"]["seal_sha256"],
        str(tmp_path / "data/experiment_receipt.json"): frozen["data_receipt_sha256"],
        str(candidate / source.OWNER): frozen["files_sha256"][source.OWNER],
        str(parent / source.OWNER): frozen["parent_files_sha256"][source.OWNER],
        **{str(helpers / name): digest for name, digest in helper_hashes.items()},
        **{cpu[name]: cpu[f"{name}_sha256"] for name in ("archive", "tests", "log")},
    }
    frozen["authenticated_files"] = original_owner.authenticated_inventory(paths)
    frozen["authenticated_files"] |= {
        str(tmp_path / name): item for name, item in (sealed["evidence"] | sealed["checkpoint_payloads"]).items()
    }
    record(binding, frozen)
    # This fixture represents a previously accepted parent. The inherited lifecycle
    # suite separately tests real parent proofs; this isolates frozen inventory checks.
    namespace = {
        "contract": contract,
        "validation": SimpleNamespace(plan=lambda *args: plan),
        "require_parent_proof": lambda *args: None,
        "require_operation_inventory": original_owner.require_operation_inventory,
        "inventory_record": original_owner.inventory_record,
        "candidate_file_names": lambda path: {source.OWNER},
        "CONTROLLER_ENVIRONMENT": frozen["controller_environment"],
        "controller_command": lambda root: frozen["controller_command"],
        "validator_command": lambda *args: frozen["validator_command"],
        "TIMEOUT": 10,
        "ARCHIVE": "issued.tar",
        "__file__": str(helpers / "ns_training_path_recovery_gpu_owner.py"),
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "CPU fixture")
    monkeypatch.setattr(preparation, "sys", SimpleNamespace(modules={}))
    return tmp_path, binding, frozen, namespace


@pytest.mark.parametrize("fault", ("none", "missing_cpu", "source_digest", "missing_seal", "payload_digest"))
def test_visible_recovery_rejects_frozen_fields_outside_authenticated_inventory(visible_binding, fault):
    root, binding, frozen, namespace = visible_binding
    inventory = frozen["authenticated_files"]
    if fault == "missing_cpu":
        inventory.pop(frozen["cpu_receipt"])
    elif fault == "source_digest":
        inventory[str(Path(frozen["candidate"]) / source.OWNER)]["sha256"] = "forged"
    elif fault == "missing_seal":
        inventory.pop(str(root / predecessor.SEAL))
    elif fault == "payload_digest":
        inventory[str(root / "old-rank.pt")]["sha256"] = "forged"
    record(binding, frozen)
    if fault == "none":
        saved = preparation.require_binding(root, namespace)
        assert saved == frozen
        assert len(saved["commands"]) == 54
    else:
        with pytest.raises(ValueError):
            preparation.require_binding(root, namespace)
