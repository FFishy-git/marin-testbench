# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a distinct Wave12 ownership pipeline while preserving all Wave11 evidence."""

import argparse
import ast
import json
import os
import subprocess
from pathlib import Path

from launch_wave5 import KEEPER, read, sha256
from prepare_upstream_outer import replace_once
from selected_lr_failure_handoff import failure_evidence, require_original_binding
from selected_lr_production import helper_sources
from selected_lr_validation import locations
from watch_wave5 import write

GENERATION = "selected-lr-recovery-generation.json"
OWNED_HELPERS = {
    "prepare_selected_lr_release.py",
    "selected_lr_release_owner.py",
    "selected_lr_production.py",
    "selected_lr_run_history.py",
    "test_prepare_selected_lr_release.py",
    "test_selected_lr_release_owner.py",
    "test_selected_lr_production.py",
    "analyze_selected_lr_run.py",
    "swanlab_selected_lr_run.py",
    "monitor_selected_lr_run.sh",
}
NAMESPACES = {
    "wave11-launch-binding.json": "wave12-launch-binding.json",
    "wave11-launch-state.json": "wave12-launch-state.json",
    "source-wave11-launch-20261117.tar": "source-wave12-launch-20261117.tar",
    "controllers/wave11-launch-20261117/tracking": "controllers/wave12-launch-20261117/tracking",
    "wave11-workers": "wave12-workers",
    "monitor-wave11": "monitor-wave12",
    "runs/wave11": "runs/wave12",
    "wave11/": "wave12/",
    "wave11.json": "wave12.json",
    ".wave11-": ".wave12-",
    '"wave11"': '"wave12"',
    "analysis-wave11.json": "analysis-wave12.json",
    "swanlab-upload-wave11.json": "swanlab-upload-wave12.json",
    "OUTPUT_MARIN_WAVE11_WORKER_ID": "OUTPUT_MARIN_WAVE12_WORKER_ID",
    "Wave11": "Wave12",
    "Wave 11": "Wave 12",
}


def replace_function(text: str, name: str, replacement: str) -> str:
    """Replace one top-level function using Python's parsed source locations."""
    matches = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"Recovery source requires one original {name} function")
    node = matches[0]
    lines = text.splitlines(keepends=True)
    return "".join(lines[: node.lineno - 1]) + replacement + "\n" + "".join(lines[node.end_lineno :])


def production_source(text: str) -> str:
    text = replace_once(
        text,
        "import selected_lr_run_history as run_history\n",
        "import selected_lr_run_history as run_history\n"
        "from selected_lr_failure_handoff import failure_evidence, probe as failure_probe, "
        "keeper_guard, release_accepted_keeper, restore_accepted_keeper, validation_terminal\n",
    )
    text = replace_function(
        text,
        "terminal_validation_observation",
        "def terminal_validation_observation(root: Path) -> dict:\n"
        '    """Keep original numerical restoration and declared current succession separate."""\n'
        "    return validation_terminal(root)\n",
    )
    text = replace_function(
        text,
        "initial_allocation",
        "def initial_allocation(root: Path, node: str) -> dict:\n"
        "    arm = ARMS[NODES.index(node)]\n"
        "    command = [PYTHON, str(Path(__file__).with_name('selected_lr_failure_handoff.py')),\n"
        "               '--root', str(root), '--arm', arm]\n"
        "    return json.loads(subprocess.run(\n"
        "        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', f'root@{node}', shlex.join(command)],\n"
        "        check=True, capture_output=True, text=True, timeout=600,\n"
        "    ).stdout.splitlines()[-1])\n",
    )
    text = replace_function(
        text,
        "allocation",
        "def allocation(root: Path, release: dict, arm: str, manager_lock: int | None = None) -> dict:\n"
        "    if local_node() != node_for(arm) or os.environ.get('CUDA_VISIBLE_DEVICES') != MASK:\n"
        "        raise RuntimeError('Recovery allocation requires its node and explicit eight-device mask')\n"
        "    idle_training()\n"
        "    observation = failure_probe(root, arm, manager_lock)\n"
        "    if observation['failure_evidence'] != release['failure_evidence'] or not same_keeper(\n"
        "        observation['keeper'], release['allocation_by_node'][node_for(arm)]['keeper']\n"
        "    ):\n"
        "        raise RuntimeError('Accepted post-failure evidence or allocation changed')\n"
        "    subprocess.run([PYTHON, '-m', 'experiment.bootstrap', '--check-only'],\n"
        "                   cwd=release['candidate'], env=os.environ | ENVIRONMENT,\n"
        "                   check=True, capture_output=True, text=True, timeout=240)\n"
        "    current = failure_probe(root, arm, manager_lock)\n"
        "    if current['failure_evidence'] != release['failure_evidence'] or not same_keeper(\n"
        "        current['keeper'], observation['keeper']\n"
        "    ):\n"
        "        raise RuntimeError('Post-failure keeper changed during bootstrap')\n"
        "    return current['keeper']\n",
    )
    text = replace_once(
        text,
        'not same_keeper(observation["keeper"], accepted["keeper"])',
        'not same_keeper(observation["current_keeper"], accepted["current_keeper"])',
    )
    text = replace_once(
        text,
        'not same_keeper(terminal_validation_observation(root)["keeper"], observation["keeper"])',
        'not same_keeper(terminal_validation_observation(root)["current_keeper"], observation["current_keeper"])',
    )
    text = replace_once(
        text,
        '        "keeper": observation["keeper"],\n',
        '        "keeper": observation["keeper"],\n'
        '        "current_keeper": observation["current_keeper"],\n'
        '        "failure_evidence": observation["failure_evidence"],\n',
    )
    text = replace_once(
        text,
        'not same_keeper(final["keeper"], accepted["keeper"])',
        'not same_keeper(final["current_keeper"], accepted["current_keeper"])',
    )
    text = replace_once(
        text,
        '        "validation_reconstruction": accepted,\n',
        '        "validation_reconstruction": accepted,\n'
        '        "recovery_of": str(root / "run-preparation/wave11-launch-binding.json"),\n'
        '        "failure_evidence": accepted["failure_evidence"],\n',
    )
    text = replace_once(
        text,
        "    with node_guard(root, NODE):\n        final = terminal_validation_observation(root)\n",
        "    if failure_evidence(root) != accepted['failure_evidence']:\n"
        "        raise RuntimeError('Original failed-wave evidence changed before publication')\n"
        "    with node_guard(root, NODE):\n        final = terminal_validation_observation(root)\n",
    )
    text = replace_once(
        text,
        "from owned_output_handoff import keeper_observation, marker_processes, "
        "observed, require_absent, same_generation",
        "from owned_output_handoff import keeper_observation, marker_processes, observed, same_generation",
    )
    for line in ("    CONTROLLER_MARKER,\n", "    controller_command,\n", "    validator_command,\n"):
        text = replace_once(text, line, "")
    text = replace_once(text, "from selected_lr_wave10_handoff import probe as wave10_handoff\n", "")
    text = replace_once(
        text,
        "from wave7_release import defer_signals, idle_training, local_node, node_guard, release_keeper",
        "from wave7_release import defer_signals, idle_training, local_node, node_guard",
    )
    text = replace_once(
        text,
        '            state["keeper"] = release_keeper(state, path)\n',
        '            state["keeper"] = release_accepted_keeper(keeper, state, path, manager_lock)\n',
    )
    text = replace_once(
        text,
        "    with node_guard(root, node_for(arm)) as signals:\n        keeper = allocation(root, release, arm)\n",
        "    with node_guard(root, node_for(arm)) as signals, keeper_guard() as manager_lock:\n"
        "        keeper = allocation(root, release, arm, manager_lock)\n",
    )
    text = replace_once(
        text,
        '                    node_action(root, "restore")\n',
        "                    restore_accepted_keeper(manager_lock)\n",
    )
    text = replace_once(
        text,
        "from launch_wave5 import NODES, PYTHON, node_action, read, sha256",
        "from launch_wave5 import NODES, PYTHON, read, sha256",
    )
    text = replace_once(
        text,
        '                    if "cleanup_error" in state or tagged_processes(state.get("child_marker", "")):\n',
        '                    if state.get("keeper_release_complete") is not True:\n'
        '                        raise RuntimeError("Unverified accepted keeper release prevents restoration")\n'
        '                    if "cleanup_error" in state or tagged_processes(state.get("child_marker", "")):\n',
    )
    return text


def transformed(name: str, content: bytes) -> bytes:
    if name not in OWNED_HELPERS:
        return content
    text = content.decode()
    for before, after in NAMESPACES.items():
        text = text.replace(before, after)
    if name == "selected_lr_production.py":
        text = production_source(text)
    if name in ("selected_lr_release_owner.py", "selected_lr_run_history.py"):
        prefix = "        " if name == "selected_lr_release_owner.py" else "        or "
        anchor = prefix + 'release["status"] != STATUS\n'
        text = replace_once(
            text,
            anchor,
            anchor
            + '        or release["recovery_of"] != str(root / "run-preparation/wave11-launch-binding.json")\n'
            + '        or not release["failure_evidence"]\n'
            + '        or any(sha256(root / name) != digest for name, digest in release["failure_evidence"].items())\n',
        )
    if name == "test_selected_lr_release_owner.py":
        text = replace_once(
            text,
            '    record(preparation / "wave10-launch-binding.json", {"status": "preserved_wave10"})\n',
            '    record(preparation / "wave10-launch-binding.json", {"status": "preserved_wave10"})\n'
            '    failed_path = preparation / "wave11-launch-binding.json"\n'
            '    record(failed_path, {"status": "preserved_failed_wave11"})\n',
        )
        text = replace_once(
            text,
            '        "status": owner.STATUS,\n',
            '        "status": owner.STATUS,\n'
            '        "recovery_of": str(failed_path),\n'
            '        "failure_evidence": {"run-preparation/wave11-launch-binding.json": owner.sha256(failed_path)},\n',
        )
    if name == "test_selected_lr_production.py":
        text = replace_once(
            text,
            '    monkeypatch.setattr(production, "release_keeper", '
            "lambda state, path: state.update(keeper_release_attempted=True))\n",
            "    monkeypatch.setattr(\n"
            '        production, "release_accepted_keeper",\n'
            "        lambda keeper, state, path, manager_lock: state.update(\n"
            "            keeper_release_attempted=True, keeper_release_complete=True\n"
            "        ),\n"
            "    )\n",
        )
        text = replace_once(
            text,
            '    monkeypatch.setattr(production, "node_action", lambda root, action: restored.append(action))\n',
            '    monkeypatch.setattr(production, "restore_accepted_keeper", lambda handle: restored.append("restore"))\n'
            '    monkeypatch.setattr(production, "keeper_guard", lambda: nullcontext(11))\n',
        )
    if name == "selected_lr_run_history.py":
        text = replace_once(
            text,
            '    "time_to_target.py",\n',
            '    "time_to_target.py",\n'
            '    "selected_lr_failure_handoff.py",\n'
            '    "test_selected_lr_failure_handoff.py",\n',
        )
        text = replace_once(
            text,
            '    "test_selected_lr_failure_handoff.py",\n',
            '    "test_selected_lr_failure_handoff.py",\n    "recovery_keeper_manager.sh",\n',
        )
    if name == "selected_lr_production.py":
        text = replace_once(
            text,
            '    "test_selected_lr_release_owner.py",\n)',
            '    "test_selected_lr_release_owner.py",\n    "test_selected_lr_failure_handoff.py",\n)',
        )
    if name == "swanlab_selected_lr_run.py":
        text = text.replace("original-worker restoration", "recovery-worker restoration")
    if name.endswith(".py"):
        ast.parse(text)
    return text.encode()


def generated_sources(root: Path) -> dict[str, bytes]:
    sources = helper_sources(read(locations(root)[0]))
    directory = Path(__file__).parent
    for name in ("selected_lr_failure_handoff.py", "test_selected_lr_failure_handoff.py"):
        sources[name] = directory / name
    original = read(locations(root)[0])
    if OWNED_HELPERS & original["helpers_sha256"].keys():
        raise ValueError("Recovery transformation overlaps original numerical validation helpers")
    return {name: transformed(name, path.read_bytes()) for name, path in sources.items()}


def prepare(root: Path, out: Path) -> dict:
    root, out = root.resolve(), out.resolve()
    require_original_binding(root)
    failures = failure_evidence(root)
    if out.exists():
        raise FileExistsError("Recovery generation requires a distinct unused directory")
    contents = generated_sources(root)
    contents["recovery_keeper_manager.sh"] = Path(KEEPER).read_bytes()
    directory = out / "tracking"
    directory.mkdir(parents=True)
    for name, content in contents.items():
        (directory / name).write_bytes(content)
    config = Path(__file__).resolve().parents[4] / "pyproject.toml"
    files = [str(directory / name) for name in sorted(OWNED_HELPERS) if name.endswith(".py")]
    for command in (
        ["uvx", "ruff@0.14.3", "check", "--fix", "--config", str(config), *files],
        ["uvx", "black@25.9.0", "--config", str(config), *files],
    ):
        subprocess.run(command, env=os.environ | {"UV_OFFLINE": "1"}, check=True, capture_output=True, text=True)
    if failure_evidence(root) != failures:
        raise RuntimeError("Original failed-wave evidence changed during generation")
    result = {
        "status": "wave12_recovery_helpers_prepared_no_release",
        "root": str(root),
        "helpers": str(directory),
        "helpers_sha256": {name: sha256(directory / name) for name in contents},
        "generator_sha256": sha256(Path(__file__)),
        "failure_evidence": failures,
        "validation_binding_sha256": sha256(locations(root)[0]),
        "wave": "wave12",
        "policy": (
            "Same numerical source/settings; distinct candidate CPU/archive, controller CPU/archive, "
            "original all-rank reconstruction and exact post-failure handoff precede a separate one-shot release."
        ),
    }
    write(out / GENERATION, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": prepare(args.root, args.out)["status"]}))


if __name__ == "__main__":
    main()
