# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the transition handoff using isolated executable evidence fixtures."""

import ast
import hashlib
import json
import tarfile
from pathlib import Path

import ns_training_path_transition_lr_predecessor as transition
import prepare_ns_training_path_transition_lr_predecessor as preparation
import pytest
from selected_lr_release_owner import sha256
from watch_wave5 import write

PARENT_TEST_SHA256 = "0a59d82243719252e9dcb31cb1d4261f2afcacb08da14ef7d72de937a0410bea"
EXCLUDED = {
    "test_archive_merge_refusal_cannot_create_or_overwrite_evidence",
    "test_source_plan_retains_conflicting_handoffs_in_separate_parents",
    "test_cli_cpu_refusal_leaves_only_original_files",
}


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_ns_training_path_exit_lr_predecessor.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_TEST_SHA256:
        raise ValueError("Transition actual-LR handoff pinned external-boundary tests changed")
    tree = ast.parse(payload)
    tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name in EXCLUDED)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_training_path_exit_lr_predecessor":
                    alias.name = "ns_training_path_transition_lr_predecessor"
                elif alias.name == "prepare_ns_training_path_exit_lr_predecessor":
                    alias.name = "prepare_ns_training_path_transition_lr_predecessor"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = transition.literal(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            if len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 168:
                node.args[0].value = 174
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "retained_summary" and isinstance(value, ast.Call):
                    for keyword in value.keywords:
                        if keyword.arg == "completed_commands":
                            keyword.value = ast.Constant(value=6)
                        elif keyword.arg == "case_decisions":
                            keyword.value = ast.Constant(value=2)
                    node.keys.append(ast.Constant(value="recovery_retained_summary"))
                    node.values.append(
                        ast.parse(
                            "dict(completed_commands=2,case_decisions=0,coverage_complete=False,numerically_accepted=False)",
                            mode="eval",
                        ).body
                    )
                    break
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())


@pytest.mark.parametrize("changed_source", (False, True))
def test_cpu_publication_preserves_real_staged_source_validation_outcome(completed, monkeypatch, changed_source):
    root, _ = completed
    waiter_path = root / transition.FOLLOWUP_CPU
    waiter = json.loads(waiter_path.read_text())
    helpers = Path(waiter["helpers"])
    guard = helpers / "root_diagnostic_pytest_guard.py"
    guard.write_text("# CPU fixture uses pytest-timeout's bounded session option.\n")
    waiter["helpers_sha256"].pop("source_0.py")
    (helpers / "source_0.py").unlink()
    waiter["helpers_sha256"][guard.name] = sha256(guard)
    archive = Path(waiter["archive"])
    with tarfile.open(archive, "w") as bundle:
        for name in waiter["helpers_sha256"]:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    waiter["archive_sha256"] = sha256(archive)
    write(waiter_path, waiter)
    monkeypatch.setattr(transition, "FOLLOWUP_CPU_SHA256", sha256(waiter_path))
    bootstrap_path = root / transition.BOOTSTRAP
    bootstrap = json.loads(bootstrap_path.read_text())
    bootstrap["issued_helpers_sha256"] = {
        name: value for name, value in waiter["helpers_sha256"].items() if name not in bootstrap["new_helpers_sha256"]
    }
    write(bootstrap_path, bootstrap)
    monkeypatch.setattr(transition, "BOOTSTRAP_SHA256", sha256(bootstrap_path))
    numerical = root / "numerical-cpu.json"
    write(numerical, {"helpers_sha256": {}})
    monkeypatch.setattr(preparation, "NUMERICAL_CPU", numerical.name)
    monkeypatch.setattr(preparation, "NUMERICAL_CPU_SHA256", sha256(numerical))
    entry = helpers / "fixture_preparation.py"
    entry.write_text("# Separate CPU acceptance entry fixture\n")
    module = helpers / "test_fixture_acceptance.py"
    module.write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        f"@pytest.mark.parametrize('case', range({preparation.TEST_COUNT}))\n"
        "def test_staged_source(case):\n"
        "    path = Path(__file__).with_name('source_1.py')\n"
        "    assert path.is_file()\n"
        f"    if case == {preparation.TEST_COUNT - 1} and {changed_source!r}:\n"
        "        path.write_text('# Source changed after tests started\\n')\n"
    )
    monkeypatch.setattr(preparation, "__file__", str(entry))
    monkeypatch.setattr(preparation, "PARENTS", {})
    monkeypatch.setattr(preparation, "NEW_HELPERS", (entry.name, module.name))
    monkeypatch.setattr(preparation, "TEST_MODULE", module.stem)
    for name, value in transition.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    before = {path: transition.file_identity(path) for path in root.rglob("*") if path.is_file()}
    out = root / "staged-acceptance"
    staged = preparation.cpu(root, out)
    assert staged["status"] == preparation.PENDING_STATUS
    assert staged["exit_code"] == 0
    assert not (out / "receipt.json").exists()
    if changed_source:
        with pytest.raises(ValueError):
            preparation.accept_cpu(root, out)
        assert not (out / "receipt.json").exists()
        published = json.loads((out / "acceptance-failure.json").read_text())
        assert published["status"] == "needs_attention"
        assert published["gpu_execution_authorized"] is False
        assert published["staged_sha256"] == sha256(out / "staged.json")
    else:
        result = preparation.accept_cpu(root, out)
        assert preparation.require_cpu(root, out / "receipt.json") == result
        assert result["exit_code"] == 0 and result["status"] == preparation.STATUS
        assert "error" not in result
    assert staged["source_plan"]["execution_closure_merged"] is False
    assert staged["source_plan"]["gpu_execution_authorized"] is False
    with pytest.raises(FileExistsError):
        preparation.accept_cpu(root, out)
    assert before == {path: transition.file_identity(path) for path in before}
    assert not (root / transition.SEAL).exists()
