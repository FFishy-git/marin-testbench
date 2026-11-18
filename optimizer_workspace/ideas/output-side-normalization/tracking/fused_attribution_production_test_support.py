# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist a synthetic attribution parent with real source and CPU archive identities."""

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import prepare_fused_attribution_production as preparation
from fused_attribution_production_contract import FAMILIES, WAVES, config_name, wave_spec
from fused_attribution_validation_contract import locations
from prepare_fused_attribution_production import candidate_inventory, sha256


def record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def archive(path: Path, entries: dict[str, Path], candidate: Path | None = None) -> None:
    with tarfile.open(path, "w") as bundle:
        for name, source in entries.items():
            bundle.add(source, arcname=name, recursive=False)
        if candidate is not None:
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)


def parent(directory: Path) -> SimpleNamespace:
    root, previous = directory / "root", directory / "original"
    experiment = previous / "experiment"
    (experiment / "tests").mkdir(parents=True)
    (experiment / "__init__.py").write_text("")
    (experiment / "train.py").write_text(
        "import argparse\nfrom pathlib import Path\n"
        "from .fused_attribution_owner import require_fused_attribution_owner\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        '    parser.add_argument("--config", type=Path, required=True)\n'
        '    parser.add_argument("--out", type=Path, required=True)\n'
        '    parser.add_argument("--launch-release", type=Path, required=True)\n'
        "    args = parser.parse_args()\n"
        "    require_fused_attribution_owner(args)\n"
        "    numerical_body = args.config.read_bytes()\n"
        "    args.out.mkdir()\n"
        "    return numerical_body\n\n"
        "if __name__ == '__main__':\n    main()\n"
    )
    (experiment / "fused_attribution_owner.py").write_text(
        "def require_fused_attribution_owner(args):\n    raise RuntimeError('validation only')\n"
    )
    sources = Path(preparation.__file__).parent
    for name in ("selected_lr_release_owner.py", "fused_attribution_validation_contract.py"):
        (experiment / name).write_bytes((sources / name).read_bytes())
    (experiment / "numerical.py").write_text("UNCHANGED_NUMERICAL_TRAJECTORY = True\n")
    record(experiment / "cluster.json", {"wave_name": "original_validation", "nodes": ["original"], "gpus_per_run": 8})
    for wave in WAVES:
        spec = wave_spec(wave)
        for family in FAMILIES:
            record(
                previous / config_name(family, spec["lr_label"]),
                {
                    "model": {"sequence_length": 4096},
                    "training": {
                        "steps": 5120,
                        "tokens": 2684354560,
                        "batch_tokens": 524288,
                        "global_batch_sequences": 128,
                        "world_size": 8,
                        "seed": 0,
                    },
                    "experiment": {"arm": family, "refresh_install_age": 1, "diagonal_backend": "triton"},
                    "optimizer": {"lr": spec["matrix_lr"], "ns_mode": "tpu", "ns_steps": 10},
                    "runtime": {"compile": "default", "attention": "FA2", "head": "compiled"},
                    "root": {"graphs": True},
                    "schedule": {
                        "warmup_steps": 0,
                        "stable_steps": 1024,
                        "decay_steps": 4096,
                        "kind": "linear",
                        "min_lr_ratio": 0.0,
                    },
                },
            )
    pinned = previous / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    (pinned / "numerical.bin").write_bytes(b"original numerical source")
    (previous / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    record(previous / "scientific-preparation.json", {"status": "preserved_scientific_parent"})
    record(root / "data/experiment_receipt.json", {"fixture": "prepared-data"})
    binding, helpers, _, _, _ = locations(root)
    helpers.mkdir(parents=True)
    marker = directory / "check-executed"
    helper = helpers / "fused_attribution_validation_controller.py"
    helper.write_text(
        "import json,os\nfrom pathlib import Path\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
        f"Path({str(marker)!r}).write_text('frozen check only')\n"
        f"print(json.dumps({{'status': {preparation.VALIDATION_STATUS!r}}}))\n"
    )
    hashes = {helper.name: sha256(helper)}
    record(
        previous / preparation.PARENT_PREPARATION,
        {
            "helpers_sha256": hashes,
            "parent_metadata_sha256": {"scientific-preparation.json": sha256(previous / "scientific-preparation.json")},
        },
    )
    files = candidate_inventory(previous)
    metadata = {
        name: sha256(previous / name) for name in ("scientific-preparation.json", preparation.PARENT_PREPARATION)
    }
    source_entries = {name: previous / name for name in files | metadata} | {f"tracking/{helper.name}": helper}
    cpu_out = root / "parent-cpu"
    cpu_out.mkdir()
    (cpu_out / "candidate.log").write_text("parent CPU passed\n")
    archive(cpu_out / "source.tar", source_entries, previous)
    cpu_path = cpu_out / "receipt.json"
    record(
        cpu_path,
        {
            "status": "fused_attribution_validation_cpu_verified_gpu_denied",
            "root": str(root),
            "candidate": str(previous),
            "files_sha256": files,
            "preparation_sha256": sha256(previous / preparation.PARENT_PREPARATION),
            "helpers_sha256": hashes,
            "exit_code": 0,
            "cuda_visible_devices": "",
            "log": str(cpu_out / "candidate.log"),
            "log_sha256": sha256(cpu_out / "candidate.log"),
            "archive": str(cpu_out / "source.tar"),
            "archive_sha256": sha256(cpu_out / "source.tar"),
        },
    )
    controller_out = root / "parent-controller-cpu"
    controller_out.mkdir()
    (controller_out / "controller.log").write_text("parent controller CPU passed\n")
    archive(controller_out / "source.tar", {f"tracking/{helper.name}": helper})
    controller_path = controller_out / "receipt.json"
    record(
        controller_path,
        {
            "status": "fused_attribution_validation_controller_cpu_verified_no_gpu_binding",
            "root": str(root),
            "helpers_sha256": hashes,
            "fixture_files_sha256": {},
            "exit_code": 0,
            "cuda_visible_devices": "",
            "log": str(controller_out / "controller.log"),
            "log_sha256": sha256(controller_out / "controller.log"),
            "archive": str(controller_out / "source.tar"),
            "archive_sha256": sha256(controller_out / "source.tar"),
        },
    )
    frozen_archive = root / "source-validation.tar"
    archive(
        frozen_archive,
        source_entries | {"candidate-cpu-receipt.json": cpu_path, "controller-cpu-receipt.json": controller_path},
        previous,
    )
    frozen = {
        "status": preparation.VALIDATION_STATUS,
        "root": str(root),
        "candidate": str(previous),
        "files_sha256": files,
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "preparation_sha256": sha256(previous / preparation.PARENT_PREPARATION),
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": sha256(cpu_path),
        "controller_cpu_receipt": str(controller_path),
        "controller_cpu_receipt_sha256": sha256(controller_path),
        "archive": str(frozen_archive),
        "archive_sha256": sha256(frozen_archive),
    }
    record(binding, frozen)
    return SimpleNamespace(root=root, previous=previous, frozen=frozen, binding=binding, marker=marker)
