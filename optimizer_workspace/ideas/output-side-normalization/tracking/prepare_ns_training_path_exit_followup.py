# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a distinct read-only terminal waiter from the pinned original implementation."""

import argparse
import ast
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import ns_training_path_exit_source as source
from selected_lr_release_owner import read, sha256
from watch_wave5 import write

BINDING_SHA256 = "80032984ac27c3cd79aa72a579e0b213190146e57c8a87c6b5f9fd1e216095c7"
INTENT_SHA256 = "64ec305891cdc9cfa1626db79364bf6f3d1a272e6a251ab9485a71e95114f610"
ARCHIVE_SHA256 = "0f0446eeb701425299760e883ddf1849db810dc158a8f743d9a349579d56e734"
PARENT_SHA256 = "7b051e8d4d2ccf5aeac5226fb151aa739a7eb7a360bb4d405b63b9a656d232bb"
PARENT_TEST_SHA256 = "e10fc334ce6c0ffbdf78c041c12a6313367cd480b2c6d9b897a1c567d85d93f2"
RUNNER = "ns_training_path_exit_followup.py"
TEST = "test_ns_training_path_exit_followup.py"
WORK = "run-preparation/ns-training-path-exit-drain-terminal-followup-20261119"
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONFAULTHANDLER": "1"}
SCOPE = (
    "One distinct read-only waiter for the acknowledged exit-drain controller and validator, "
    "checking exact original generations every900seconds. Invoke only their issued canonical "
    "independent terminal once full54-command18-decision coverage, actual owner/descendant "
    "absence and clean exact keeper restoration are recorded. Retain rejected cases and all "
    "consumed failed predecessors. No keeper transfer, new GPU work, numerical waiver, "
    "production, optimizer quality/efficiency or SwanLab authority."
)
MODULES = {
    "ns_training_path_contract": "ns_training_path_exit_contract",
    "ns_training_path_gpu_controller": "ns_training_path_exit_gpu_controller",
    "ns_training_path_followup": "ns_training_path_exit_followup",
}


class WaiterNamespace(ast.NodeTransformer):
    def visit_Import(self, node):
        for alias in node.names:
            alias.name = MODULES.get(alias.name, alias.name)
        return node

    def visit_ImportFrom(self, node):
        node.module = MODULES.get(node.module, node.module)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value
            for previous, replacement in source.LITERALS.items():
                value = value.replace(previous, replacement)
            node.value = (
                value.replace("test_ns_training_path_followup", "test_ns_training_path_exit_followup")
                .replace("ns_training_path_followup", "ns_training_path_exit_followup")
                .replace("ns_training_path_gpu_controller", "ns_training_path_exit_gpu_controller")
            )
        return node


def parent_payload(filename: str, digest: str) -> bytes:
    path = Path(__file__).with_name(filename)
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Exit-drain waiter pinned parent differs")
    return payload


def derived_sources() -> dict[str, bytes]:
    """Change only execution identity and preclaim receipt-generation consistency."""
    tree = WaiterNamespace().visit(ast.parse(parent_payload("ns_training_path_followup.py", PARENT_SHA256)))
    values = {
        "BINDING_SHA256": BINDING_SHA256,
        "INTENT_SHA256": INTENT_SHA256,
        "ARCHIVE_SHA256": ARCHIVE_SHA256,
        "WORK": WORK,
        "MARKER": "OUTPUT_MARIN_NS_TRAINING_PATH_EXIT_FOLLOWUP",
        "CHILD_MARKER": "OUTPUT_MARIN_NS_TRAINING_PATH_EXIT_TERMINAL_CHILD",
        "CPU_STATUS": "ns_training_path_exit_drain_read_only_followup_cpu_archive_verified",
        "SCOPE": SCOPE,
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in values:
                node.value = ast.Constant(value=values[target.id])
        if isinstance(node, ast.FunctionDef) and node.name == "readiness":
            validation = ast.parse(
                """if state.get("spawn_status") != "acknowledged" or any(
    process["group"] != process["pid"] or process["session"] != process["pid"]
    for process in (child, original)
):
    raise ValueError("Exit-drain terminal waiter owner receipt lacks exact process sessions")
"""
            ).body
            index = next(
                i + 1
                for i, item in enumerate(node.body)
                if isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Tuple) for target in item.targets)
                and any(isinstance(value, ast.Name) and value.id == "child" for value in ast.walk(item))
            )
            node.body[index:index] = validation
        if isinstance(node, ast.FunctionDef) and node.name == "handle":
            node.body = ast.parse(
                '''"""Inspect exact handles; incomplete files and live replacement never prove absence."""
actual = handoff.observed(expected["pid"])
if actual is None:
    if (handoff.PROC_ROOT / str(expected["pid"])).exists():
        raise RuntimeError("Exit-drain waiter original process files are incomplete")
    return {"status": "absent", "expected": expected}
if any(actual[key] != expected[key] for key in ("pid", "start_ticks", "group", "session")):
    raise RuntimeError("Exit-drain waiter original process generation was substituted")
if actual["status"] in ("Z", "X", "x"):
    return {"status": "exited_handle_present", "expected": expected, "observed": actual}
if actual["command"] != expected["command"]:
    raise RuntimeError("Exit-drain waiter live command changed")
if marker_name == contract.CHILD_MARKER and actual["parent"] != expected["parent"]:
    raise RuntimeError("Exit-drain waiter validator ancestry changed")
verified = process_context(expected["pid"], expected["command"], cwd, marker_name, marker, environment)
if verified["start_ticks"] != expected["start_ticks"] or not handoff.same_generation(
    verified, handoff.observed(expected["pid"])
):
    raise RuntimeError("Exit-drain waiter owner changed during observation")
return {"status": "live_exact_generation", "expected": expected, "observed": verified}
'''
            ).body
    tests = WaiterNamespace().visit(ast.parse(parent_payload("test_ns_training_path_followup.py", PARENT_TEST_SHA256)))
    for node in ast.walk(tests):
        if isinstance(node, ast.Dict) and any(
            isinstance(key, ast.Constant) and key.value == "child_generation" for key in node.keys
        ):
            node.keys.append(ast.Constant(value="spawn_status"))
            node.values.append(ast.Constant(value="acknowledged"))
    extra = ast.parse(
        """@pytest.mark.parametrize("fault", ("unacknowledged", "controller_group", "validator_session"))
def test_changed_exit_owner_session_refuses_before_canonical_claim(original, fault):
    root, binding, state, _ = original
    helpers = Path(binding["helpers"])
    intent = read(helpers.parent / "controller.json")
    if fault == "unacknowledged":
        state["spawn_status"] = "starting"
    elif fault == "controller_group":
        intent["generation"]["group"] = 99
        write(helpers.parent / "controller.json", intent)
    else:
        state["child_generation"]["session"] = 99
    write(followup.contract.locations(root)[2], state)
    with pytest.raises(ValueError):
        followup.readiness(root, binding)
    terminal, attempt, outcome, _ = followup.controller.terminal_paths(root, intent)
    assert not any(path.exists() for path in (terminal, attempt, outcome))
"""
    )
    tests.body.extend(extra.body)
    tests.body.extend(
        ast.parse(
            """def test_incomplete_original_proc_files_cannot_authorize_terminal(original):
    root, binding, _, proc = original
    helpers = Path(binding["helpers"])
    intent = read(helpers.parent / "controller.json")
    directory = proc / str(intent["pid"])
    directory.mkdir()
    with pytest.raises(RuntimeError):
        followup.readiness(root, binding)
    terminal, attempt, outcome, _ = followup.controller.terminal_paths(root, intent)
    assert not any(path.exists() for path in (terminal, attempt, outcome))

def test_zombie_with_empty_command_is_waited_without_terminal_claim(original):
    root, binding, _, proc = original
    helpers = Path(binding["helpers"])
    intent = read(helpers.parent / "controller.json")
    put_process(proc, intent["generation"], helpers, {}, status="Z")
    (proc / str(intent["pid"]) / "cmdline").write_bytes(b"")
    observation = followup.readiness(root, binding)
    assert observation["ready"] is False
    assert observation["owners"]["controller"]["status"] == "exited_handle_present"
    terminal, attempt, outcome, _ = followup.controller.terminal_paths(root, intent)
    assert not any(path.exists() for path in (terminal, attempt, outcome))
"""
        ).body
    )
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TEST_COUNT" for target in node.targets
        ):
            node.value = ast.Constant(value=24)
    header = "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
    return {
        RUNNER: (header + ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode(),
        TEST: (header + ast.unparse(ast.fix_missing_locations(tests)) + "\n").encode(),
    }


def stage(root: Path, bootstrap: Path) -> dict:
    """Copy the issued helper closure unchanged and add the two derived waiter files."""
    binding = root / source.LITERALS["run-preparation/ns-training-path-binding-20261119.json"]
    if sha256(binding) != BINDING_SHA256:
        raise ValueError("Exit-drain waiter issued binding changed")
    frozen = read(binding)
    helpers = Path(frozen["helpers"])
    if sha256(helpers.parent / "controller.json") != INTENT_SHA256:
        raise ValueError("Exit-drain waiter acknowledgment changed")
    if frozen["archive_sha256"] != ARCHIVE_SHA256 or sha256(Path(frozen["archive"])) != ARCHIVE_SHA256:
        raise ValueError("Exit-drain waiter issued archive changed")
    payloads = derived_sources()
    if set(payloads) & set(frozen["helpers_sha256"]):
        raise ValueError("Exit-drain waiter new source collides with issued helper")
    bootstrap.mkdir(parents=True, exist_ok=False)
    target = bootstrap / "tracking"
    target.mkdir()
    for name, digest in frozen["helpers_sha256"].items():
        path = helpers / name
        if sha256(path) != digest:
            raise ValueError("Exit-drain waiter issued helper changed")
        shutil.copy2(path, target / name)
    for name, payload in payloads.items():
        with (target / name).open("xb") as stream:
            stream.write(payload)
    prepared = {
        "status": "exit_drain_read_only_waiter_bootstrap_prepared_not_accepted",
        "root": str(root),
        "helpers": str(target),
        "new_helpers_sha256": {name: sha256(target / name) for name in payloads},
        "issued_helpers_sha256": frozen["helpers_sha256"],
        "binding_sha256": BINDING_SHA256,
        "intent_sha256": INTENT_SHA256,
        "archive_sha256": ARCHIVE_SHA256,
        "parent_sha256": PARENT_SHA256,
        "parent_tests_sha256": PARENT_TEST_SHA256,
        "generator_sha256": sha256(Path(__file__)),
        "scope": SCOPE,
    }
    shutil.copy2(Path(__file__), bootstrap / Path(__file__).name)
    write(bootstrap / "source.json", prepared)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--bootstrap", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("Exit-drain waiter staging requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    prepared = stage(root, args.bootstrap.resolve())
    helpers = Path(prepared["helpers"])
    subprocess.run(
        [
            "/opt/conda/envs/torch-base/bin/python",
            "-P",
            str(helpers / RUNNER),
            "--root",
            str(root),
            "--phase",
            "cpu",
            "--out",
            str(args.out.resolve()),
        ],
        cwd=helpers,
        env=os.environ | ENVIRONMENT | {"PYTHONPATH": str(helpers), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        stdin=subprocess.DEVNULL,
        timeout=1000,
        check=True,
    )


if __name__ == "__main__":
    main()
