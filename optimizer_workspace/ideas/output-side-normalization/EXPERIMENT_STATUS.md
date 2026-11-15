# Experiment Status

Updated: 2026-11-15.

## Completed

- Removed the fresh `artifacts/external-sources/Toner` clone.
- Preserved the pinned `Toner-dab7f3777c75e740536df822f50ffa1d929a340c`
  snapshot. All 112 archived files match its downloaded archive.
- Wrote and cold-reader-reviewed `MARIN_EXPERIMENT_PLAN.md`, including the
  model, pinned datasets/tokenizer, batch-specific optimizer settings,
  controls, selection rules, budget, and 32-GPU allocation.
- Implemented `experiment/` with five first-stage arms: Muon, input MuonT,
  outer-left down, inner+outer down, and down hybrid.
- Generated 20 explicit seed-0 A100/FA2 configurations for 0.5M, 1M, 2M,
  and 4M batches. The first wave uses four independent eight-GPU jobs.
- Added cotangent collection, raw distributed sum/count pooling, output EMA
  and cached roots, down-free input collection, rank-local checkpoints,
  data checksums, source manifests, deployment, and launch preflight.
- Added a clean-checkout bootstrap with archive and 112-file source hash
  verification, an A100/NCCL validation driver, and real-data smoke and
  checkpoint-continuation orchestration. These accelerator checks are pending.
- Published the implementation to `FFishy-git/marin-testbench`, branch
  `output-side-normalization-marin`, commit
  `280d40ed7302e77a7117774de219832dc10ca244`. Remote-ref and commit-pinned
  file retrieval both verified publication.
- Created the approved PVC output directories under
  `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization`.
  Deployment receipts identify each source version; redeploy after edits.
- Deployed the current 154-file task snapshot to all three workers.
  Final preflight verified matching experiment source/configuration hashes
  on all four nodes.

## Executed Verification

The CPU suite passed 15 tests, including:

- exact-SVD direction and transpose orientation;
- inverse-factor eigenvalue and scale checks;
- unchanged parameter gradients with output observations;
- loss-scale correction and microbatch pooling;
- eager, AOT, and Inductor-compiled backward observation;
- unequal two-rank sample-count pooling using Gloo;
- CPU optimizer/RNG checkpoint continuation.
- actual finite-NS down direction/decay equations for all three candidate arms.

The constructed pinned architecture contains 265,585,152 parameters.
The full scoped repository lint entry point passed Ruff, Black, license headers,
size, syntax, conflict, whitespace, newline, and documentation-command checks.
These results do not validate A100 kernels or full-model training.

The full formatter pass removed trailing whitespace from the input proposal.
Its scientific text was not edited; checksum provenance is recorded in
`artifacts/SOURCES.md`.

Evidence:

- `artifacts/run-preparation/validation.json`
- `artifacts/run-preparation/deployment.json`
- `artifacts/run-preparation/preflight.json`

Current access and validation receipts are also written under the PVC
`output-side-normalization/run-preparation/` directory. The source bootstrap
passed a fresh temporary extraction and verified all 112 pinned files.

## Launch Blockers

No experiment training has started.

1. All 32 A100-SXM4-40GB GPUs are occupied at 100% utilization by the existing
   `hy_infuse_moe/gpu_matmul.py` workload. It has not been stopped. Owner
   confirmation is required before releasing these processes.
2. The working proxy reaches Hugging Face, but the exact pinned Llama
   tokenizer request returns HTTP 401 without authorized access. No approved
   exact prepared-data directory has been supplied. The datasets/tokenizer
   have not been substituted.
   StarCoderData and Paloma also report automatic gating. The PVC access
   receipt records the resolved full dataset revisions and HTTP status.
3. GPU cotangent tests, NCCL statistics parity, A100 FA2, compiled full-model
   checks, four-arm real-data smoke, memory fit, and GPU checkpoint
   continuation remain pending. The validation receipt marks these false,
   and the launcher refuses a full wave.

## Next Actions

After workload release and approved data access:

1. Prepare the pinned data once and verify names, token/document counts,
   token-ID range, and SHA256 receipts. Record full dataset revisions.
   Install missing preparation dependencies in a task-specific environment.
2. Execute GPU kernel/collector parity and two-rank NCCL checks. Compare
   candidate down updates with an independent FP32 matrix reference.
3. Run full-model/context real-data two-step smoke for all first-wave arms.
   Measure memory and lower microbatch identically only if required.
4. Compare uninterrupted and resumed GPU trajectories, including input bases,
   output moments/roots, optimizer state, and data position.
5. Issue an evidence-backed passing validation receipt, then launch Muon,
   input MuonT, outer-left down, and down hybrid across all four nodes.
6. Monitor logs, losses, checkpoints, root fallbacks, memory, and throughput.
   Treat the first wave as exploratory, then execute the plan's fair tuning
   and paired-seed confirmation gates.

Later control arms, full Bi-MuonT/full hybrid, detailed spectral/curvature
diagnostics, the tuning campaign, and larger-model extensions are planned.
They have not been implemented or executed in this first-stage runner.
