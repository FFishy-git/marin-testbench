# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache completed continuation verification without rereading checkpoint tensors.

Only the final successful require_validation call may store its captured hash
identities. An ordinary receipt or an unkeyed cache checksum is not comparison
authority. Lookup never stores; the parent retains its normal source, campaign,
command, receipt and manifest gates. This is a local trusted-writer cache, not an
authentication boundary against a writer able to replace evidence and checksums.
"""

import hashlib
import json
import math
import os
import stat
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
INTERPRETATION_FILES = (
    "tracking/related_verified_cache.py",
    "tracking/related_execution.py",
    "tracking/related_sequence.py",
    "tracking/related_campaign.py",
    "experiment/records.py",
    "experiment/launch.py",
)
WORLD = 8
SCHEMA = 1


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(path: Path) -> tuple[int, ...]:
    information = path.stat()
    if not stat.S_ISREG(information.st_mode):
        raise ValueError(f"Verification evidence is not a regular file: {path}")
    return (
        information.st_dev,
        information.st_ino,
        information.st_size,
        information.st_mtime_ns,
        information.st_ctime_ns,
    )


def _snapshot(path: Path, expected: str) -> dict:
    before = _identity(path)
    if _sha256(path) != expected or _identity(path) != before:
        raise ValueError(f"Verification evidence changed: {path}")
    return {"sha256": expected, "stat": list(before)}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _interpretation() -> dict:
    return {name: _sha256(TASK / name) for name in INTERPRETATION_FILES}


def _controls(validation: dict, receipt: dict, interpretation: dict) -> dict:
    if (
        receipt["wave_name"] != validation["wave_name"]
        or receipt["config_sha256"] != {name: alias["config_sha256"] for name, alias in validation["aliases"].items()}
        or set(receipt["evidence"]["real_data_smokes"]) != set(validation["aliases"])
    ):
        raise ValueError("Receipt differs from the completed external validation")
    logs = {item["log"]: item["log_sha256"] for item in receipt["evidence"]["gpu_commands"]}
    probes = {receipt["evidence"][key] for key in ("gpu_receipt", "related_gpu_receipt")}
    if logs != validation["logs_sha256"] or probes != set(validation["gpu_receipts_sha256"]):
        raise ValueError("Validator log or accelerator evidence differs")
    controls = {
        validation["path"]: validation["sha256"],
        **validation["logs_sha256"],
        **validation["gpu_receipts_sha256"],
        **{alias["config"]: alias["config_sha256"] for alias in validation["aliases"].values()},
    }
    for name, digest in {**receipt["source_sha256"], **interpretation}.items():
        path = str(TASK / name)
        if path in controls and controls[path] != digest:
            raise ValueError("Validator interpretation conflicts with the accepted source")
        controls[path] = digest
    # Run manifests bind executed upstream sources not listed in the GPU receipt.
    for name in validation["smoke_artifacts_sha256"]:
        if Path(name).name != "run.json":
            continue
        manifest = _read(Path(name))
        if (
            manifest["data_receipt_sha256"] != receipt["data_receipt_sha256"]
            or {key: manifest["source_sha256"].get(key) for key in receipt["source_sha256"]} != receipt["source_sha256"]
        ):
            raise ValueError("Smoke manifest source or data differs from the verified receipt")
        for relative, digest in manifest["source_sha256"].items():
            path = str(TASK / relative)
            if path in controls and controls[path] != digest:
                raise ValueError("Smoke manifests disagree about executed source")
            controls[path] = digest
    return controls


def _checkpoint_paths(directory: Path, step: int) -> set[str]:
    checkpoint = directory / "checkpoints" / f"step_{step:06d}"
    return {str(checkpoint / "complete.json"), *(str(checkpoint / f"rank_{rank}.pt") for rank in range(WORLD))}


def _comparison_seeds(receipt: dict, hashes: dict) -> dict:
    comparisons, covered = {}, set()
    for name, smoke in receipt["evidence"]["real_data_smokes"].items():
        full = Path(smoke["out"]).resolve()
        if full.name != f"{name}_continuous":
            raise ValueError("Smoke does not identify its bound trial")
        resumed = full.with_name(f"{name}_resumed")
        complete, split = smoke["completed_steps"], smoke["split_steps"]
        paths = _checkpoint_paths(full, complete) | _checkpoint_paths(resumed, complete)
        paths |= _checkpoint_paths(resumed, split) | {str(full / "run.json"), str(resumed / "run.json")}
        pairs = [(full, resumed, complete, smoke["continuation"])]
        pre_refresh = smoke["pre_refresh_continuation"]
        if pre_refresh is not None:
            prefix = full.with_name(f"{name}_pre_refresh_continuous")
            paths |= _checkpoint_paths(prefix, split + 1) | _checkpoint_paths(resumed, split + 1)
            paths.add(str(prefix / "run.json"))
            pairs.append((prefix, resumed, split + 1, pre_refresh))
        artifacts = tuple(sorted((path, hashes[path]) for path in paths))
        for original, continued, step, summary in pairs:
            if (
                type(summary["tensor_comparisons"]) is not int
                or summary["tensor_comparisons"] <= 0
                or type(summary["max_absolute_error"]) not in (int, float)
                or not math.isfinite(summary["max_absolute_error"])
                or summary["max_absolute_error"] < 0
            ):
                raise ValueError("Verified comparison summary is invalid")
            comparisons[(str(original), str(continued), step, artifacts)] = summary
        covered.update(paths)
    if covered != set(hashes):
        raise ValueError("Verified smoke inventory differs from the comparison pairs")
    return comparisons


def cache_path(campaign: dict, receipt_path: Path) -> Path:
    """Return the campaign's stable cache location bound to the receipt bytes."""
    return Path(campaign["root"]) / "run-preparation" / "related-verified-cache" / f"{_sha256(receipt_path)}.json"


def store(cache_path: Path, validation: dict, receipt: dict, identities: dict) -> dict:
    """Publish captured identities only after the caller completed actual verification.

    Args:
        cache_path: Fresh immutable output path.
        validation: Successful require_validation result, not a raw GPU receipt.
        receipt: Exact receipt decoded by that successful call.
        identities: ARTIFACT_HASHES entries captured before/after content hashing.

    Checkpoint files are stat-checked, not read. Tiny receipts, configs, logs and
    validator/source files are hashed; any change refuses publication.
    """
    cache_path = cache_path.resolve()
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if os.path.lexists(cache_path) or os.path.lexists(temporary):
        raise FileExistsError("Preserve existing verification cache evidence")
    receipt_path = Path(validation["path"])
    _snapshot(receipt_path, validation["sha256"])
    if _read(receipt_path) != receipt:
        raise ValueError("Current receipt differs from the verified receipt")
    interpretation = _interpretation()
    controls = _controls(validation, receipt, interpretation)
    hashes = validation["smoke_artifacts_sha256"]
    _comparison_seeds(receipt, hashes)
    if str(cache_path) in {*controls, *hashes} or str(temporary) in {*controls, *hashes}:
        raise ValueError("Cache output overlaps original verification evidence")
    files = {}
    for name, digest in hashes.items():
        identity, verified_digest = identities[name]
        if verified_digest != digest or tuple(identity) != _identity(Path(name)):
            raise ValueError(f"Captured verified artifact identity changed: {name}")
        files[name] = {"sha256": digest, "stat": list(identity)}
    control_files = {name: _snapshot(Path(name), digest) for name, digest in controls.items()}
    for name, information in {**control_files, **files}.items():
        if _identity(Path(name)) != tuple(information["stat"]):
            raise ValueError("Verification evidence changed while storing the cache")
    record = {
        "schema_version": SCHEMA,
        "cache_path": str(cache_path),
        "source": str(TASK),
        "interpretation_sha256": interpretation,
        "validation": validation,
        "receipt": receipt,
        "control_sha256": controls,
        "files": files,
    }
    record["integrity_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x") as stream:
        try:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.chmod(temporary, 0o444)
            os.link(temporary, cache_path)
        finally:
            temporary.unlink()
    return record


def load(cache_path: Path, receipt_path: Path) -> dict | None:
    """Return exact worker seed maps, or None for normal uncached verification."""
    try:
        cache_path, receipt_path = cache_path.resolve(), receipt_path.resolve()
        before = _identity(cache_path)
        record = _read(cache_path)
        integrity = record.pop("integrity_sha256")
        if (
            record["schema_version"] != SCHEMA
            or record["cache_path"] != str(cache_path)
            or record["source"] != str(TASK)
            or hashlib.sha256(_canonical(record)).hexdigest() != integrity
            or record["interpretation_sha256"] != _interpretation()
            or record["validation"]["path"] != str(receipt_path)
        ):
            return None
        validation = record["validation"]
        receipt = _read(receipt_path)
        controls = _controls(validation, receipt, record["interpretation_sha256"])
        if record["receipt"] != receipt or record["control_sha256"] != controls:
            return None
        control_files = {name: _snapshot(Path(name), digest) for name, digest in controls.items()}
        files = record["files"]
        if {name: item["sha256"] for name, item in files.items()} != validation["smoke_artifacts_sha256"]:
            return None
        comparisons = _comparison_seeds(receipt, validation["smoke_artifacts_sha256"])
        for name, information in {**control_files, **files}.items():
            if _identity(Path(name)) != tuple(information["stat"]):
                return None
        if _identity(cache_path) != before:
            return None
        return {
            "validation": validation,
            "artifact_hashes": {name: (tuple(item["stat"]), item["sha256"]) for name, item in files.items()},
            "verified_comparisons": comparisons,
            "receipt": receipt,
            "cache_sha256": integrity,
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return None


def validation_cache(campaign: dict, receipt_path: Path) -> dict | None:
    """Locate unchanged campaign evidence without ever creating or priming a cache."""
    try:
        hit = load(cache_path(campaign, receipt_path), receipt_path)
        if hit is None:
            return None
        receipt, validation = hit["receipt"], hit["validation"]
        wave = next(item for item in campaign["waves"] if item["wave_name"] == validation["wave_name"])
        trials = {trial["name"]: trial for trial in campaign["trials"]}
        aliases = {
            name: next(alias for alias in trials[name]["aliases"] if alias["wave"] == wave["wave_name"])
            for name in wave["trials"]
        }
        if (
            Path(campaign["source"]).resolve() != TASK
            or validation["aliases"] != aliases
            or receipt["source_sha256"]
            != {name: digest for name, digest in campaign["source_sha256"].items() if name.startswith("experiment/")}
            or receipt["data_receipt_sha256"] != campaign["data_receipt_sha256"]
        ):
            return None
        return hit
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration):
        return None
