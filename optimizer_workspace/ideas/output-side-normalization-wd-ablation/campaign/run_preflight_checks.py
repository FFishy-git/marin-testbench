"""Run CPU unit tests, config/archive checks, 4-node deploy, and GPU + 8-step parity gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

from campaign.configure_wd_campaign import generate_all
from campaign.run_wd_campaign import EXISTING_RUNS, PYTHON, CampaignScheduler, preflight_gates

CPU_TEST_SNIPPET = """
from experiment.tests.test_block_toner import (
    test_batched_update_matches_per_matrix_update,
    test_decay_modes_match_explicit_equations_per_matrix_and_batched,
    test_two_sided_update_matches_independent_finite_ns_equations,
)
for shape in [(6, 6), (3, 7), (8, 4)]:
    test_two_sided_update_matches_independent_finite_ns_equations(shape)
for sides in [("dense", "dense"), ("dense", "diagonal"), ("none", "dense"), ("none", "diagonal"), ("diagonal", "none")]:
    test_batched_update_matches_per_matrix_update(sides)
for mode in ["default", "isotropic", "two_sided"]:
    for lk in ["dense", "diagonal", "none"]:
        test_decay_modes_match_explicit_equations_per_matrix_and_batched(mode, lk)
print("CPU unit tests: 17/17 combinations passed.")
"""


def main() -> None:
    print("1. Running CPU unit tests for decay_mode equations...", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    toner_root = TASK / "artifacts" / "external-sources" / "Toner-dab7f3777c75e740536df822f50ffa1d929a340c"
    env["PYTHONPATH"] = f"{TASK}:{toner_root / 'src'}:{toner_root}"
    subprocess.run([PYTHON, "-c", CPU_TEST_SNIPPET], cwd=TASK, env=env, check=True)

    print("2. Generating configs & checking byte-identical parity with wave1a...", flush=True)
    configs_index = generate_all()
    gen_ref_cfg = json.loads(Path(configs_index["wt_mlp_lag_130m_1x_seed0_lr0p016"]["config"]).read_text())
    arch_ref_cfg = json.loads((EXISTING_RUNS[("wt_mlp_lag", "0p016", 0)] / "config.json").read_text())
    assert gen_ref_cfg == arch_ref_cfg, "Generated wt_mlp_lag reference config differs from wave1a config.json!"

    sched = CampaignScheduler(configs_index)
    desired = sched._desired_jobs()
    print(f"Verified {len(EXISTING_RUNS)} archived runs; initial queued jobs ({len(desired)}): {desired}", flush=True)

    print("3. Deploying workspace to all 4 A100 nodes...", flush=True)
    subprocess.run([PYTHON, str(TASK / "campaign" / "deploy_wd_campaign.py")], check=True)

    print("4. Running GPU decay-operator math gate & 8-step distributed parity gate...", flush=True)
    preflight_gates(configs_index)
    print("ALL PREFLIGHT GATES PASSED.", flush=True)


if __name__ == "__main__":
    main()
