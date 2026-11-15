# Marin Output-Side Experiment

This runner imports the preserved Toner revision
`dab7f3777c75e740536df822f50ffa1d929a340c`. The experimental code and generated
configurations are separate from that snapshot. See
[`../MARIN_EXPERIMENT_PLAN.md`](../MARIN_EXPERIMENT_PLAN.md) for the scientific
design and complete model, data, optimizer, and allocation settings.

Implemented first-stage arms: Muon, input MuonT, outer-left down,
inner+outer down, and down hybrid. Later controls, full Bi-MuonT, larger
models, and bounded tuning are planned extensions.

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
  --work /apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/validation \
  --receipt artifacts/run-preparation/validation.json
```

This runs synthetic two-rank A100/NCCL correctness checks followed by all four
first-wave arms on eight GPUs with real data. Each arm runs three updates
continuously, then two updates plus a restored third update. Checkpoint
comparison includes model, auxiliary optimizer, fast input state, output
moments/roots, data position, and RNG state. A failure preserves command logs
and leaves the corresponding launch gates pending.
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

```bash
uv run --no-project --python /opt/conda/envs/torch-base/bin/python \
  python -m experiment.launch --launch \
  --data /APPROVED/DATA --data-receipt /APPROVED/DATA/experiment_receipt.json \
  --validation-receipt artifacts/run-preparation/validation.json \
  --out /APPROVED/RUNS/wave0 \
  --report artifacts/run-preparation/wave0.json
```

The code has CPU numerical coverage. Until the GPU/real-data gates execute,
compiled full-model correctness, A100 throughput, memory fit, and scientific
outcomes remain unverified. Synthetic correctness checks are never Marin
training evidence.
