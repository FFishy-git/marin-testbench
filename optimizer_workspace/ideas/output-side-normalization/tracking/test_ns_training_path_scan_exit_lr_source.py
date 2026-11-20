# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise preserved actual-LR settings and refusal at the distinct scan-exit adapter boundary."""

import ast
import hashlib
import shutil
from pathlib import Path

import ns_training_path_scan_exit_lr_source as source
import pytest

PARENT_SHA256 = "71364d8e8932c34ee3642c0484a1cfb2f6860b3a450f41bfff248f3fdfd7e25b"


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_ns_training_path_transition_lr_source.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit LR pinned adapter regression source changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    tree = source.Namespace().visit(ast.parse(namespace["inherited_tests"]()))
    parent = ast.parse(payload)
    tree.body[:0] = ast.parse("import ns_training_path_lr_contract as numerical_contract\n").body
    tree.body.extend(
        source.Namespace().visit(node)
        for node in parent.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_measured_handoff_changes_only_named_execution_source_and_preserves_all_actual_lrs"
    )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())


def test_changed_derivation_parent_refuses_before_new_candidate_publication(candidate, monkeypatch, tmp_path):
    root, original, _, _, previous, _ = candidate
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in original.rglob("*") if path.is_file()}
    sources = source.execution_sources(previous)
    local = tmp_path / "rebound-sources"
    local.mkdir()
    for name in source.NEW_FILES:
        shutil.copy2(sources[name], local / name)
    parent = local / "ns_training_path_transition_lr_source.py"
    parent.write_text(parent.read_text().replace("162commands", "163commands"))
    monkeypatch.setattr(source, "__file__", str(local / Path(source.__file__).name))
    unpublished = tmp_path / "unpublished-candidate"
    with pytest.raises(ValueError):
        source.prepare(root, unpublished)
    assert not unpublished.exists()
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    assert not source.contract.locations(root)[0].exists()
