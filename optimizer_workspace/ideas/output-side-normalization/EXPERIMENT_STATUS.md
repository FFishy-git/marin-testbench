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
  checkpoint-continuation orchestration.
- Published the implementation to `FFishy-git/marin-testbench`, branch
  `output-side-normalization-marin`. Latest published implementation revision:
  `daed57bf315a7cb8a841a75342fd42c082c0fed0`.
- Created the approved PVC output directories under
  `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization`.
  Deployment receipts identify each source version; redeploy after edits.
- Deployed the task snapshot to all three workers.
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
after its 20-update exact-root startup. This gate passed on all four nodes.
Restored input factors rewarm and recapture process-local CUDA graphs without
resetting numerical root or momentum state. The new GPU continuation gate
verified this adaptation on synthetic optimizer inputs; full-model checks
remain a separate requirement.

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
unchanged. Full preparation and verification completed on 2026-11-15 at
10:05:40 UTC (18:05:40 Asia/Shanghai), with process exit code 0.
`experiment.prepare` loads only `HF_TOKEN` from the local credential file,
holds an output lock, and calls the unchanged upstream preparation entry point.
Both scratch and PVC checks verified all 2,876 shards. The training/C4 receipt
records 2,311 required shards, 25,386,955,482 tokens, and 20,425,515 documents.
It verifies names, token/document counts, token-ID bounds, SHA256, size and
modification timestamps. Partial data cannot authorize training.
The CPU suite was rerun and all 18 tests passed.
The full-data CPU loader check passed the deterministic allocation of each
2,048-window block: DCLM 1,897, StarCoder 124, and Proof-Pile 27 sequence
windows. It also checked 64 held-out windows per source, held-out exclusion, 236 C4
evaluation windows, and packed batches at the beginning and end of the
configured training trajectory.

Prepared data:
`/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/data`.

Training receipt: `data/experiment_receipt.json`, SHA256
`e6ff883cb816c34cc502061dbc6a8239fc19dfc803588786758bb1020e1afe10`.
The data-ready preflight accepted this receipt on all four nodes and found
matching experiment code/configuration hashes. The GPU keepers were released
after owner confirmation; full-model continuation remains the launch gate.

Current evidence under the PVC output root:

- `run-preparation/access.json`
- `run-preparation/source-file-access.json`
- `run-preparation/first-shards.json`
- `run-preparation/preparation-state.json`
- `run-preparation/prepare-full-scratch-20261115.log`
- `run-preparation/full-loader.json`
- `run-preparation/preflight-data-ready.json`

Authentication, isolated-environment, preparation, and verification commands
are in `experiment/README.md`. Credentials are not in the evidence or Git.

## Launch Blockers

No full-budget experiment training has started. The owner confirmed keeper
release on 2026-11-15. Only the four verified keeper parents were sent
SIGTERM; all 32 GPUs became idle. The release receipt is
`run-preparation/keeper-release-20261115.json` under the PVC output root.

The synthetic two-rank GPU checks passed: compiled cotangent transparency
and independent sums, unequal-count NCCL pooling, FA2, down-update equations
in FP32 and configured BF16-input/FP32-output NS, compiled hybrid body parity,
and cached-root checkpoint continuation after 22 updates with CUDA graphs.
Evidence is in `validation/attempt_20261115_1348/gpu_checks.json` and its log
under the PVC output root.

The initial communicator initialization failed with SIGILL in the optional
TCCL Inspector plugin. Validation and launch now record
`NCCL_INSPECTOR_ENABLE=0`; NCCL and root CUDA graphs remain enabled.
Candidate down updates now use FP32 root products and upstream per-matrix
NS because split-product/batched BF16 updates exceeded the existing parity
tolerance. Compiled transparency probes compare compiled reference paths to
avoid conflating compiler BF16 rounding with collector changes.

All four arms completed eight-rank full-model/context real-data smoke and
checkpoints at microbatch 4. Later updates took approximately 1.07 seconds for
Muon, 1.50-2.59 seconds for MuonT, 1.52-2.62 seconds for outer-left down, and
1.35-1.94 seconds for down hybrid. Peak allocations were approximately
19-20.4 GiB per GPU. These are short execution probes, not steady-state
benchmarks or optimizer quality measurements. Their continuation comparisons
failed under the original DDP bucket behavior. The launcher still refuses
full-budget training until every required gate passes.

The first Muon full-model attempt exhausted A100 memory during compiled-head
autotuning at microbatch 8. All 20 active configs now use microbatch 4 and
preserve their original global batches and token budgets. The 0.5M first wave
uses accumulation 4 on eight GPUs. All synthetic GPU gates, including the
added partition gradient/statistic check, passed again in
`validation/attempt_20261115_1353/gpu_checks.json`. Full-model smoke passed
under the adapted configs.

The unrestricted-autotune Muon smoke completed three finite-loss real-data
updates and saved checkpoints, but its uninterrupted/resumed comparison
failed at embedding weights. The active configs now request deterministic
algorithms, fixed ATEN body/head GEMMs, explicit precision casts, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; validation must establish continuation
without changing tolerances.
Full-model output-arm backward also exposed shared-buffer version counters
across compiled segments. Observation buffers now have independent storage
per layer and stack only for pooling. CPU and synthetic GPU checks passed
after this change in `validation/attempt_20261115_1407`; full-model checks
completed locally and on each assigned worker.

Deterministic body/head compilation made printed training losses agree, but
all four full-model continuation checks still failed at
`body.blocks.0.gate.weight`. The Muon gate had maximum absolute error
`7.715e-6`. Disabling Dynamo DDP partitioning alone did not fix this.
The current local runner uses `find_unused_parameters=True` to keep the
initial DDP gradient bucket assignment across fresh and resumed processes.
The full eight-rank Muon continuation comparison passed, including model,
optimizer and RNG state. The previously failing gate weight is identical.
The remaining arms are being checked in
`validation/attempt_20261115_1418`. This source change is not yet published
or deployed to the workers. No tolerances or Newton-Schulz settings were
relaxed.

Data access, full preparation, and data verification are resolved.
No datasets, revisions, or tokenizer have been substituted.

## Next Actions

1. Reuse the completed prepared corpus and released GPUs.
2. Preserve the passing synthetic GPU checks when their source inputs match.
3. Complete fixed-bucket full-model/context continuation for all first-wave arms.
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
