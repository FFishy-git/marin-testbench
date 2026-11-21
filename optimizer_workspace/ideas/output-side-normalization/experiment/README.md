# Marin Output-Side Experiment

This runner imports the preserved Toner revision
`dab7f3777c75e740536df822f50ffa1d929a340c`. The experimental code and generated
configurations are separate from that snapshot. See
[ENDEAVOR.md](../ENDEAVOR.md) for the scientific design, executed results,
and complete model, data, optimizer, and allocation settings.

## Related-Method Controls

GO-MUON and Mousse now have separately labeled common-setting controls and
source-recipe adapters. These are prepared baselines, not executed quality
results or validated A100 implementations.

| Arm | Geometry | Numerical Profile |
| --- | --- | --- |
| `muon_control` | No factors; same owner/update runner | Common finite NS and isotropic decay |
| `go_muon_control` | Activation/cotangent moments; matched quarter powers; final norm graft | Common EMA/ridge/refresh and finite NS |
| `go_muon_paper` | OAS factors; held activations; sparse backward EMA | Compact SVD polar; source-statistics adapter |
| `mousse_control` | Weight-gradient Gram EMAs; matched eighth powers; post-NS norm graft | Common EMA/ridge/refresh and finite NS in eigen coordinates |
| `mousse_paper` | Same gradient-Gram information | Pinned author BF16 NS and shape scaling |

Both panels include `inner_outer_mlpdiag` with explicit isotropic matrix
decay. The original arm still defaults to its original input-shaped decay;
the isotropic configuration is a separate ablation, not a replacement.
Model, initialization seed, data order, global batch, auxiliary AdamW routing,
evaluation, token budget, and schedule are shared. The panels compare complete
recipes: norm grafting, powers, statistic sources, and root execution can still
differ. They do not isolate any single mechanism.

`go_muon_paper` is an implementation of arXiv:2608.09763v1 formulas, not
author-code parity. Its capture remains attached on reuse steps, and its decay
is an explicit choice because the paper does not specify matrix decay.
Quarter roots use FP64 eigendecomposition with FP32 returned factors;
the graft norm uses an FP64 scalar reduction. This reference implementation
is not a tuned throughput implementation.
`mousse_paper` adapts pinned revision
`d00c1bf17790fbe56424ee5567cce80d8e75f4b2`; architecture, auxiliary routing,
momentum, and additive update transport remain runner choices.
Neither adapter supports an unqualified source-paper throughput claim.

Prepared seed-0 short-budget configs are under
`experiment/comparisons/related_controlled_short/` and
`experiment/comparisons/related_paper_short/`. Each panel has three paired
four-arm waves at matrix LR multipliers `{1/sqrt(2), 1, sqrt(2)}`.
Each trial has 167,772,160 tokens and a completed 20%-stable/80%-decay
schedule, not an early stop of the full schedule. Shared Muon/ours configs
are identical across panels and need not be run twice.

Generate additional configurations in a new directory:

```bash
python -m experiment.related_configure \
  --out experiment/comparisons/related_full_seed12 \
  --profile controlled --stage full --seeds 1 2
```

Pass a prepared wave to the existing gates without changing `cluster.json`:

```bash
python -m experiment.validate --wave-manifest \
  experiment/comparisons/related_controlled_short/wave_seed0_lr1.json \
  --receipt artifacts/run-preparation/related_cpu_validation.json
```

CPU validation leaves all GPU gates pending. Once GPUs are available, use
the same manifest with `validate --gpu`, exact data/receipt, and a unique work
directory, then `launch --wave-manifest` with that passing receipt and a
separate report/output location. Related-method continuation checks cross a
periodic refresh: GO-MUON checks ten updates split after five; source Mousse
checks twenty-two split after eleven.
A pre-refresh comparison verifies that restored factors are consumed before
being replaced. Independent related-method GPU numerical and compiled-capture
checks must also pass. Do not bypass launch gates with direct training
commands. Choose full-budget rates using selection loss, then run paired seeds;
short trials and CPU parity tests do not establish optimizer superiority.

Implemented arms. Down-only: Muon, input MuonT, outer-left down,
inner+outer down, down hybrid. All-block (`block_toner.py`), where an arm
names an input side plus an output side per output width, and may withhold
the left factors as a control, drop the inner left factor, or apply the
left factors one update late:

| Arm | Input | Output at 512 | Output at 2048 | Note |
| --- | --- | --- | --- | --- |
| `inner_outer_all` | full | full | full | Wave 1 |
| `null_all` | full | full | full | Left factors withheld |
| `left_all` | none | full | full | Wave 1 |
| `inner_outer_all_diag` | diagonal | diagonal | diagonal | Wave 1 |
| `inner_outer_hidden` | full | full | none | Wave 2 |
| `inner_outer_mlpdiag` | full | full | diagonal | Wave 2; best selection loss |
| `inner_outer_leftdiag` | full | diagonal | diagonal | Wave 2; best cost trade |
| `outer_left_all` | full | full | full | Outer left factor only |
| `null_mlpdiag` | full | full | diagonal | Control for `inner_outer_mlpdiag` |
| `inner_outer_all_lag` | full | full | full | Left factors one update late |
| `inner_outer_mlpdiag_lag` | full | full | diagonal | Both of the above |

The launcher schedules the entries listed in `cluster.json` under
`runs/WAVE_NAME`. Each entry names an arm, seed and batch, so one wave can mix a
candidate, its matched control, and a second seed of the same arm; the run
directory is `ARM` for 0.5M seed 0, `ARM_seedN` for other 0.5M seeds,
`ARM_BATCH` for larger seed-0 batches, and `ARM_BATCH_seedN` otherwise.
Launch and full-model validation use exactly that entry's configuration;
larger-batch entries do not reuse a 0.5M smoke gate.

Run commands from the parent of `experiment/`. Use the existing Torch environment:

```bash
cd /root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization
uv run --no-project --python /opt/conda/envs/torch-base/bin/python \
  python -m experiment.bootstrap --check-only
uv run --no-project --python /opt/conda/envs/torch-base/bin/python \
  python -m pytest -c /dev/null experiment/tests -q
```

On a clean GitHub checkout, omit `--check-only` to restore the pinned source
archive through the explicit proxy. It verifies both the recorded archive
SHA256 and all 112 source-file hashes. Existing differing source files are
never overwritten. The large source artifacts remain outside Git.

Prepare the exact data with the pinned script after authorized Hugging Face
access is configured. Accept access terms for
`meta-llama/Meta-Llama-3.1-8B`, `bigcode/starcoderdata`, and `allenai/paloma`
using the account associated with the credential. The preparation wrapper loads
only `HF_TOKEN` from the designated local `.env` file. It does not export the
GitHub or W&B credentials. Never put credentials in tracked files or logs.
The source caches and token files are large. The script does not change
tokenizer or skip count checks.

```bash
export HTTPS_PROXY=http://star-proxy.oa.com:3128
export HTTP_PROXY=http://star-proxy.oa.com:3128
export ALL_PROXY=http://star-proxy.oa.com:3128
```

The task-specific preparation environment is on the approved PVC. It inherits
the inspected Torch environment and installs only the missing decompressor.
It was created and import-checked on 2026-11-15; the shared Torch/CUDA stack
was not modified.

```bash
ROOT=/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization
uv venv --system-site-packages --python /opt/conda/envs/torch-base/bin/python \
  "$ROOT/.venv-data"
uv pip install --python "$ROOT/.venv-data/bin/python" \
  zstandard==0.25.0 python-dotenv==1.2.1
```

For the existing environment, skip creation and proceed with:

```bash
ROOT=/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization
uv run --no-project --python "$ROOT/.venv-data/bin/python" \
  python -m experiment.prepare --out "$ROOT/data" --workers 96 \
  --scratch /tmp/output-side-normalization-data \
  --env-file /root/dl/runtime/script/.env \
  --proxy http://star-proxy.oa.com:3128 \
  --state "$ROOT/run-preparation/preparation-state.json" \
  --receipt "$ROOT/data/experiment_receipt.json"
```

The inspected versions are recorded in `experiment/requirements.txt`.
After access was granted on 2026-11-15, direct source-file requests returned
HTTP 206 for all four required components. The first shard of DCLM, StarCoder,
Proof-Pile and Paloma C4 was prepared and verified against the pinned counts.
Full preparation subsequently completed with exit code 0. All 2,876 shards
passed scratch and PVC verification. The existing prepared corpus is
`$ROOT/data`; skip preparation and reuse its `experiment_receipt.json`.
The receipt covers 2,311 required training/C4 shards and has SHA256
`e6ff883cb816c34cc502061dbc6a8239fc19dfc803588786758bb1020e1afe10`.
The wrapper calls Toner's unchanged preparation entry point, reuses only
verified existing shards, checks all 2,876 shards, then creates the full
training/C4 content receipt. New shards are written to local scratch to avoid
small-write latency on the PVC's FUSE mount, then copied in blocks to the PVC
and verified there. Scratch retains completed files for retries and needs
approximately 102 GB. A per-output filesystem lock prevents duplicate
preparation jobs. The state file records the PID and lifecycle; only
`status: complete` establishes successful preparation and verification.
Partial shards and access probes cannot authorize training.

A single-node execution check uses the full model/context and reference
schedule; `--max-steps` is an early-stop execution check.

```bash
export NCCL_INSPECTOR_ENABLE=0
export PYTHONFAULTHANDLER=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/opt/conda/envs/torch-base/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m experiment.train \
  --config experiment/configs/down_hybrid_0p5m_seed0.json \
  --data /APPROVED/DATA --data-receipt /APPROVED/DATA/experiment_receipt.json \
  --out /APPROVED/RUNS/smoke_down_hybrid --max-steps 2
```

Checkpoint directories have one atomic file per rank and a `complete.json`
receipt written only after all ranks finish saving. Resume requires the
original configuration/world size. Saved state includes input bases and
output roots, avoiding an implicit root reset.
Run the resume command in the same environment, including the exports above.
The execution check saves at its final update, so the command above creates
`checkpoints/step_000002`. `--max-steps` is a total update limit; resuming that
checkpoint with `--max-steps 3` executes one additional update.
Checkpoint identities also bind source hashes, the full configuration, and
the prepared-data receipt. A resume refuses mismatched identities before
loading weights; changing source code requires a separately reviewed migration.

```bash
/opt/conda/envs/torch-base/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m experiment.train \
  --config experiment/configs/down_hybrid_0p5m_seed0.json \
  --data /APPROVED/DATA --data-receipt /APPROVED/DATA/experiment_receipt.json \
  --out /APPROVED/RUNS/smoke_down_hybrid \
  --resume /APPROVED/RUNS/smoke_down_hybrid/checkpoints/step_000002 \
  --max-steps 3
```

CPU validation and launch preflight do not stop or replace running workloads.
Deployment copies task-owned files to workers and may update files that match
the previous deployment receipt. The launch preflight records GPU occupancy on
all four nodes:

```bash
uv run --no-project --python /opt/conda/envs/torch-base/bin/python \
  python -m experiment.validate --receipt artifacts/run-preparation/validation.json
uv run --no-project --python /opt/conda/envs/torch-base/bin/python \
  python -m experiment.deploy
uv run --no-project --python /opt/conda/envs/torch-base/bin/python \
  python -m experiment.launch --report artifacts/run-preparation/preflight.json
```

`validate` executes CPU checks and records all accelerator/real-data gates
as pending. Its receipt cannot authorize a full training launch by itself.
`deploy` refuses to replace differing files on workers unless they match the
previous task deployment receipt. User edits remain protected.

After GPU release and full data verification, execute the actual GPU and
real-data gates:

```bash
uv run --no-project --python /opt/conda/envs/torch-base/bin/python \
  python -m experiment.validate --gpu \
  --data /APPROVED/DATA --data-receipt /APPROVED/DATA/experiment_receipt.json \
  --work /apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/validation/UNIQUE_ATTEMPT \
  --receipt artifacts/run-preparation/validation.json
```

This runs synthetic two-rank A100/NCCL correctness checks followed by all four
first-wave arms on eight GPUs with real data. Each arm runs three updates
continuously, then two updates plus a restored third update. Checkpoint
comparison includes model, auxiliary optimizer, fast input state, output
moments/roots, data position, and RNG state. A failure preserves command logs
and leaves the corresponding launch gates pending.
Down-update checks cover FP32 and the configured BF16-input/FP32-output
Newton-Schulz mode, using Toner's per-matrix routine as the reference.
Candidate down updates use FP32 root products and per-matrix Newton-Schulz:
the initial split-product/batched implementation exceeded the existing BF16
parity tolerance on A100. This increases launch overhead for candidate down
updates; include that cost in throughput comparisons. Non-down input updates
retain the pinned fast backend. The `f32` and `tpu` NS modes both use
`rtol=0.003, atol=5e-5` for the resulting down parameter delta.
Checkpoint tensor comparisons use `rtol=1e-5, atol=1e-6` for every tensor;
scalars, source/data identities and counters must match exactly.
The elementwise rule is `abs(actual - expected) <= atol + rtol * abs(expected)`.
Passing continuation receipts report the maximum absolute tensor error.
Collector and body transparency checks compare compiled observed and
compiled unobserved paths; eager BF16 accumulation differs from compiled
accumulation. The collector also has an independent raw-cotangent reference.
The synthetic continuation probe separately runs 24 optimizer updates with
the configured 20-update exact-root startup, saves after update 22, and
compares a restored two-update continuation. It requires that cached-root
CUDA graphs were active before saving and covers input MuonT, outer-left down,
and down hybrid. The three-update real-data smoke alone does not establish
post-warmup root-state continuation.
The numerical probe waits for pending host refreshes between updates so
thread scheduling does not change which bases are ready at the next update.
Production retains Toner's asynchronous host-refresh pipeline.
This probe establishes continuation of saved numerical state with controlled
refresh readiness. It does not establish bitwise production trajectories when
asynchronous host refreshes become ready at different updates.
CUDA graph objects are process-local. Resume retains the saved EMAs, bases,
tolerances, momenta, output roots, and pending host refreshes, then performs
one eager root warmup before capturing fresh graphs. The cached-root
continuation gate checks this runtime adaptation.

The full launch requires available GPUs, exact prepared data, and a validation
receipt for the current source hashes. The receipt must record passing
`gpu_collector`, `distributed_statistics`, `fa2`, `compiled_model`,
`real_data_four_arm_smoke`, and `checkpoint_continuation` checks with supporting
evidence paths. Do not mark unexecuted checks as passing. The launcher schedules
Muon, MuonT, outer-left down, and down hybrid on one eight-GPU node each.
The launcher also requires `gpu_down_updates`, `cached_root_continuation`,
the current data receipt, and source/configuration hashes.
`experiment/cluster.json` records the four nodes
and the approved PVC output root. Outputs default to
`/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/runs/wave0`.

The cluster runtime sets `NCCL_INSPECTOR_ENABLE=0` for validation and training.
The optional TCCL Inspector plugin caused SIGILL during communicator
initialization on these nodes; disabling its instrumentation allowed a GPU
all-reduce and the two-rank numerical checks to pass. NCCL communication,
model compilation, and root CUDA graphs remain enabled. The launcher records
these environment settings and requires the validation receipt to match.
Direct `torchrun` invocations on this cluster also need that environment setting.

The active generated configs use microbatch 4 across all arms. Microbatch 8
exhausted the A100 40GB memory during full-model compiled-head autotuning.
Global batches, schedules, contexts, model dimensions, and token budgets are
unchanged; the 0.5M configs accumulate four microbatches on each of eight GPUs.
The synthetic GPU partition check compares gradients and collected output
statistics for the same 16 sequences partitioned as two microbatches of 8
or four microbatches of 4, using the same effective batch and loss scaling.
It also checks the independent raw moment.

The active model/head compiler options retain compilation with CUDA graphs
off for those two surfaces, restrict GEMM selection to ATEN, and emulate
precision casts. Training enables deterministic algorithms and sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The original unrestricted autotune paths
produced different full-model trajectories across process starts and failed
the existing checkpoint-continuation tolerance. These options apply to every
arm and remain subject to full-model validation; root CUDA graphs are separate.
Output observation buffers have independent storage per layer so compiled
backward segments cannot invalidate other layers' saved-buffer versions.

Training also uses DDP `find_unused_parameters=True` to keep its initial
gradient bucket assignment across process restarts. Without this setting,
the startup and rebuilt buckets produced small gradient differences amplified
by BF16 Newton-Schulz, failing full-model continuation at unchanged tolerances.
All four eight-rank full-model continuation comparisons passed with fixed
buckets and zero maximum tensor error. DDP warns that no
parameters are unused; the flag deliberately prevents bucket rebuilding and
adds a graph traversal. Record the cost in throughput comparisons.

```bash
uv run --no-project --python /opt/conda/envs/torch-base/bin/python \
  python -m experiment.launch --launch \
  --data /APPROVED/DATA --data-receipt /APPROVED/DATA/experiment_receipt.json \
  --validation-receipt artifacts/run-preparation/validation.json \
  --out /APPROVED/RUNS/wave0 \
  --report artifacts/run-preparation/wave0.json
```

The CPU numerical suite and synthetic A100/NCCL checks have passed.
All four first-wave arms completed three-update full-model real-data smoke
runs and checkpoint continuation. All four full-budget wave-0 jobs completed
on 2026-11-16; see [ENDEAVOR.md](../ENDEAVOR.md#execution-identity) for launch
identities, results, and evidence.
The launcher requires a passing receipt for all current source/configuration
hashes. Short smoke losses do not establish optimizer quality.
