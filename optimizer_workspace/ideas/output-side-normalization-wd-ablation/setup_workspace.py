"""Bootstrap isolated workspace from the frozen lr-bracket snapshot."""

from __future__ import annotations

import shutil
from pathlib import Path

SRC = Path("/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization-lr-bracket")
DST = Path(__file__).resolve().parent


def copy_tree_clean(src: Path, dst: Path) -> int:
    count = 0
    for path in sorted(src.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(path.readlink())
            count += 1
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            count += 1
    return count


def main() -> None:
    n_exp = copy_tree_clean(SRC / "experiment", DST / "experiment")
    n_art = copy_tree_clean(SRC / "artifacts", DST / "artifacts")
    (DST / "campaign").mkdir(parents=True, exist_ok=True)
    alloc_text = (SRC / "campaign" / "allocation.py").read_text()
    alloc_text = alloc_text.replace(
        'if "allocation.py" in cmd or "run_campaign.py" in cmd or "run_lr_campaign.py" in cmd:',
        'if "allocation.py" in cmd or "run_campaign.py" in cmd or "run_lr_campaign.py" in cmd or "run_wd_campaign.py" in cmd:',
    )
    (DST / "campaign" / "allocation.py").write_text(alloc_text)

    eval_text = (SRC / "campaign" / "eval_paloma.py").read_text()
    eval_text = eval_text.replace(
        '"arm": config["experiment"]["arm"],',
        '"arm": config["experiment"]["arm"],\n        "arm_label": config["experiment"].get("arm_label"),\n        "decay_mode": config["experiment"].get("decay_mode", "default"),',
    )
    (DST / "campaign" / "eval_paloma.py").write_text(eval_text)
    print(f"Bootstrapped {n_exp} experiment files, {n_art} artifact entries, and allocation/eval_paloma helpers.")


if __name__ == "__main__":
    main()
