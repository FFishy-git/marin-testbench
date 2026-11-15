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

The CPU suite passed 18 tests, including:

- exact-SVD direction and transpose orientation;
- inverse-factor eigenvalue and scale checks;
- unchanged parameter gradients with output observations;
- loss-scale correction and microbatch pooling;
- eager, AOT, and Inductor-compiled backward observation;
- unequal two-rank sample-count pooling using Gloo;
- CPU optimizer/RNG checkpoint continuation.
- actual finite-NS down direction/decay equations for all three candidate arms.
- output EMA and cached-root continuation through an actual checkpoint;
- changed data-receipt rejection before checkpoint weights are loaded.
- corrupt rank-ownership rejection before checkpoint weights are loaded.

Checkpoint state now binds source hashes, full configuration, prepared-data
identity, and rank ownership. A new synthetic GPU gate saves after update 22
and continues through update 24, covering Toner's cached-root CUDA graphs
after its 20-update exact-root startup. This gate is implemented but unexecuted
while the GPUs remain occupied.
Restored input factors rewarm and recapture process-local CUDA graphs without
resetting numerical root or momentum state. The new GPU continuation gate
must verify this adaptation before training resumes are accepted.

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

## Data Preparation Attempt

On 2026-11-15, created `.venv-data` under the approved PVC output root with
system-site access to the existing Torch environment. Installed
`zstandard==0.25.0` there and verified imports of the preparation dependencies.
The shared Torch/CUDA installation was not changed.

The initial attempt executed the pinned `reproduce.marin.prepare_data` entry
point with 16 workers and the designated PVC `data/` output. It exited with code 1 during
tokenizer initialization: the pinned Llama repository returned HTTP 401.
That failed attempt created no prepared token shards.

Initial 1024-byte source-file probes returned HTTP 206 for DCLM and Proof-Pile,
and HTTP 401 for StarCoderData and Paloma C4. These probes check reachability;
they do not prepare or verify the corpus.

Failure log:
`/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/run-preparation/prepare-data-attempt-20261115.log`.

After the user supplied a local HF credential and granted the gated access,
all four required source-file probes returned HTTP 206. The pinned tokenizer
download succeeded with SHA256
`76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa`.
The first shard of every required component was prepared with Toner's unchanged
code and verified:

| Component | Tokens | Documents |
| --- | ---: | ---: |
| DCLM | 10,971,222 | 9,472 |
| StarCoder | 433,522 | 64 |
| Proof-Pile | 254,590 | 128 |
| Paloma C4 | 186,085 | 435 |

Full preparation started on 2026-11-15 for all 2,876 pinned shards
(101,782,148,820 bytes of token files). It was increased from 32 to 96 CPU
workers after stopping the initial preparation process and preserving its
verified shards. Small-write waits on the PVC's FUSE path prompted a switch to
local scratch with block copies to the PVC. Toner's tokenization code is
unchanged. Full preparation is still pending.
`experiment.prepare` loads only `HF_TOKEN` from the local credential file,
holds an output lock, and calls the unchanged upstream preparation entry point.
It checks the complete plan and creates the training/C4 content receipt only
after successful preparation. Partial data cannot authorize training.
The CPU suite was rerun and all 18 tests passed. No GPU smoke has run.

Current evidence under the PVC output root:

- `run-preparation/access.json`
- `run-preparation/source-file-access.json`
- `run-preparation/first-shards.json`
- `run-preparation/preparation-state.json`
- `run-preparation/prepare-full-scratch-20261115.log`

Authentication, isolated-environment, preparation, and verification commands
are in `experiment/README.md`. Credentials are not in the evidence or Git.

## Launch Blockers

No experiment training has started.

1. All 32 A100-SXM4-40GB GPUs are occupied at 100% utilization by the existing
   `hy_infuse_moe/gpu_matmul.py` workload. It has not been stopped. Owner
   confirmation is required before releasing these processes.
2. Access is resolved and full preparation is running. The exact data receipt
   is still pending; the first four verified shards are insufficient for
   training. No datasets, revisions, or tokenizer have been substituted.
3. GPU cotangent tests, NCCL statistics parity, A100 FA2, compiled full-model
   checks, four-arm real-data smoke, memory fit, and GPU checkpoint
   continuation remain pending. The validation receipt marks these false,
   and the launcher refuses a full wave.

## Next Actions

1. Await the existing preparation job. It verifies names, token/document counts,
   token-ID range, and SHA256 receipts automatically. Confirm
   `run-preparation/preparation-state.json` records `status: complete` and
   `data/experiment_receipt.json` exists. Do not start a duplicate job.
   Release only the identified GPU keepers after owner confirmation.
2. Execute GPU kernel/collector parity and two-rank NCCL checks. Compare
   candidate down updates with an independent FP32 matrix reference.
3. Run full-model/context real-data three-update smoke for all first-wave arms.
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
