# Marin Output-Side Experiment

This runner imports the preserved Toner revision
`dab7f3777c75e740536df822f50ffa1d929a340c`. The experimental code and generated
configurations are separate from that snapshot. See
[`../MARIN_EXPERIMENT_PLAN.md`](../MARIN_EXPERIMENT_PLAN.md) for the scientific
design and complete model, data, optimizer, and allocation settings.

Implemented first-stage arms: Muon, input MuonT, outer-left down,
inner+outer down, and down hybrid. Later controls, full Bi-MuonT, larger
models, and bounded tuning are planned extensions.

Run commands from this task directory. Use the existing Torch environment:

```bash
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
access is configured. The source caches and token files are large. The script
does not change tokenizer or skip count checks.

```bash
export HTTPS_PROXY=http://star-proxy.oa.com:3128
export HTTP_PROXY=http://star-proxy.oa.com:3128
export PYTHONPATH="$PWD/artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c"
uv run --no-project --python /opt/conda/envs/torch-base/bin/python \
  python -m reproduce.marin.prepare_data --out /APPROVED/DATA --workers 16
uv run --no-project --python /opt/conda/envs/torch-base/bin/python \
  python -m experiment.data_access --data /APPROVED/DATA \
  --receipt /APPROVED/DATA/experiment_receipt.json
```

The inspected versions are recorded in `experiment/requirements.txt`.
Preparation also requires `zstandard`, which was missing from the inspected
Torch environment. Install it in a task-specific environment without replacing
the shared Torch/CUDA stack. The exact pinned tokenizer returned HTTP 401
without configured Hugging Face authorization.

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

```bash
/opt/conda/envs/torch-base/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m experiment.train \
  --config experiment/configs/down_hybrid_0p5m_seed0.json \
  --data /APPROVED/DATA --data-receipt /APPROVED/DATA/experiment_receipt.json \
  --out /APPROVED/RUNS/smoke_down_hybrid \
  --resume /APPROVED/RUNS/smoke_down_hybrid/checkpoints/step_000002 \
  --max-steps 3
```

Preflight is read-only and records GPU occupancy on all four nodes:

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

The full launch requires available GPUs, exact prepared data, and a validation
receipt for the current source hashes. The receipt must record passing
`gpu_collector`, `distributed_statistics`, `fa2`, `compiled_model`,
`real_data_four_arm_smoke`, and `checkpoint_continuation` checks with supporting
evidence paths. Do not mark unexecuted checks as passing. The launcher schedules
Muon, MuonT, outer-left down, and down hybrid on one eight-GPU node each.
The launcher also requires `gpu_down_updates`, the current data receipt, and
source/configuration hashes. `experiment/cluster.json` records the four nodes
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
