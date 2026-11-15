# Output-Side Normalization: Marin Experiment Plan

Date: 2026-11-15. Status: implementation and launch preparation.

## Decision and Scope

Follow Toner's **Marin experiment**. The primary model is its Llama-style
130M non-embedding configuration, trained on the fixed
DCLM/StarCoderData/Proof-Pile-2 mixture and evaluated on Paloma C4.
nanoGPT is outside this program. The local six-layer Grug model is also
outside the controlled comparison.

Answer two questions separately:

1. Does output-cotangent preconditioning improve optimization beyond
   independently tuned input-only MuonT?
2. Does replacing the large down-projection input factor improve measured
   time-to-target and memory without materially degrading validation loss?

Use the existing, unchanged source snapshot:

`artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c`

Its `reproduce/marin/README.md`, configurations, model, data plan, and fast
backend are the reproduction reference. The fresh clone was removed at the
user's request. Source inspection and retrieved papers are indexed in
[artifacts/SOURCES.md](artifacts/SOURCES.md).

The experimental runner lives separately under `experiment/`; it imports
the pinned model/data/backend and leaves the reference files intact.
An A100 run is a hardware-adapted replication: FA2 replaces the recorded
H100 FA3 backend consistently in every arm. A native Marin/JAX port is a
later engineering task and must establish its own numerical parity.

Track the implementation on GitHub in the existing
`FFishy-git/marin-testbench` repository, branch
`output-side-normalization-marin`. Keep papers, downloaded sources,
datasets, checkpoints, and machine-specific receipts outside Git.
The tracked source bootstrap restores the exact Toner archive and verifies
all source hashes. Store run outputs on the approved PVC:
`/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization`.

## Hardware and Allocation

The available allocation contains four nodes, each with eight
A100-SXM4-40GB GPUs:

| Node | Address | GPUs |
| --- | --- | --- |
| Launcher | 28.216.17.9 | 8 |
| Worker 0 | 28.216.17.80 | 8 |
| Worker 1 | 28.216.17.144 | 8 |
| Worker 2 | 28.216.17.139 | 8 |

Use all 32 GPUs as **four concurrent eight-GPU experiments**. This keeps
the reference world size and batch accumulation semantics, gives four
comparisons per wave, and avoids inter-node communication in an individual
run. "Use 32 GPUs" does not mean multiplying the global batch by four.

First launch wave, after preflight and numerical tests:

| Node | Arm | Batch | Seed | Duration |
| --- | --- | --- | --- | --- |
| Launcher | Muon | 0.5M | 0 | Full 2.684B-token reference run |
| Worker 0 | Input-only MuonT | 0.5M | 0 | Full 2.684B-token reference run |
| Worker 1 | MuonT + outer-left down factors | 0.5M | 0 | Same budget and schedule |
| Worker 2 | Down-only hybrid | 0.5M | 0 | Same budget and schedule |

The first wave is exploratory and is not the three-seed confirmation.
If a candidate fails preflight, run an independent baseline seed in that slot;
do not launch a scientifically invalid arm merely to occupy GPUs.
Subsequent waves test inner-left, Bi-MuonT, full hybrid, controls, tuning,
and confirmation, subject to the gates below.

Existing workloads must be released with the owner's approval. The initial
inspection found `hy_infuse_moe/gpu_matmul.py` occupying the allocation.
Never kill unrelated training or infrastructure processes. Each experiment
must have a PID, log path, exact configuration, and source hash manifest.

## Model Configuration

All primary arms use these verified pinned settings:

| Field | Value |
| --- | --- |
| Model label | `130m`; excludes embeddings and final output head |
| Layers | 32 |
| Hidden dimension | 512 |
| MLP dimension | 2048 |
| Attention heads / head dimension | 8 / 64 |
| Context length | 4096 |
| Vocabulary | 128256 |
| Projections | Separate Q/K/V/O; separate gate/up/down; no linear biases |
| MLP | SwiGLU: `silu(gate(h)) * up(h)` |
| Normalization | RMSNorm, epsilon `1e-5`, learned gains |
| Position encoding | RoPE, theta 10000 |
| Residual computation | BF16; preserve the reference normalization rounding |
| Parameter storage | FP32; BF16 computation as in the pinned runner |
| Embedding initialization | `sqrt_hidden` |
| Initialization | Pinned truncated-normal rule; head initially copies the appropriately scaled embedding draw |
| Embedding/head tying | Separate trainable parameters |

The block matrix weights contain 134,217,728 parameters. Including
normalization, input embedding, and output head gives 265,585,152 trainable
parameters, calculated from the pinned architecture. Verify these counts
by constructing the actual model before launching.

Mathematical orientation is `y = W x`, `W in R^(m x n)`:

| Projection | Weight shape | Input factor size | Output factor size |
| --- | --- | --- | --- |
| Q/K/V/O | 512 x 512 | 512 x 512 | 512 x 512 |
| Gate/up | 2048 x 512 | 512 x 512 | 2048 x 2048 |
| Down | 512 x 2048 | 2048 x 2048 | 512 x 512 |

The first candidate wave only collects down output moments. This bounds
collector cost while directly testing the largest input-factor replacement.
Do not share O and down output factors: they have different cotangents.

## Training Data and Tokenizer

Use the **complete pinned Marin data plan**, not today's default mixture:

`artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/reproduce/marin/data_plan.json`

| Component | Dataset repository | Recorded revision | Mixture weight |
| --- | --- | --- | --- |
| DCLM | `mlfoundations/dclm-baseline-1.0` | `a3b142c` | 3.8 |
| Code | `bigcode/starcoderdata` | `9fc30b5` | 0.25 |
| Math | `EleutherAI/proof-pile-2` | `901a927` | 0.055 |
| Evaluation | `allenai/paloma` | `65cd6fc` | No training weight |

The abbreviated dataset revisions are exactly what the pinned plan records.
Resolve and record their full repository commits when preparing data.
The access probe resolved these commits on 2026-11-15:

| Component | Full Revision |
| --- | --- |
| DCLM | `a3b142c183aebe5af344955ae20836eb34dcf69b` |
| Code | `9fc30b578cedaec69e47302df72cf00feed7c8c4` |
| Math | `901a9273a770e9d4138c5ddd91802f9c5c6cdc4b` |
| Paloma | `65cd6fc59dba021b21db414fa5e8d7765ffbe5e6` |

StarCoderData and Paloma report automatic gating; readable repository metadata
does not establish source-file access. After authorized access was granted on
2026-11-15, the pinned tokenizer download and source-file probes for all four
required components succeeded. Full corpus preparation and verification are
still in progress.
The weights normalize approximately to 92.5700%, 6.0901%, and 1.3398%.
The actual loader uses 2048-sequence mixture blocks, integer allocation,
and remainder correction: 1897 DCLM, 124 code, and 27 math windows per block.
Preserve this implementation, including seed-dependent permutations.

| Tokenizer field | Value |
| --- | --- |
| Repository | `meta-llama/Meta-Llama-3.1-8B` |
| Revision | `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` |
| BOS | 128000 |
| EOS | 128001 |
| Format | Raw uint32 `.bin` token streams |
| Document encoding | Pinned tokenizer applied to `text + " " + eos_token` |

Each source file has a fixed document-prefix quota, output order, expected
document count, and expected token count in the plan. Use its preparation
script and count checks. The reference has no original cross-machine token
file hashes; generated checksum receipts prove later file integrity, not
bitwise agreement with an unavailable historical cache.

Expected prepared training data, from the plan's summary:

| Component | Tokens | Documents | Files |
| --- | --- | --- | --- |
| DCLM | 23,378,120,924 | 18,701,696 | 1200 |
| Code | 1,584,860,368 | 1,624,000 | 863 |
| Math | 423,006,576 | 97,728 | 242 |

The roughly 102 GB prepared training corpus is larger than an individual
2.684B-token run. Allow additional space for source-download caches,
validation, preparation intermediates, and checkpoints. Prepare once and
use identical verified receipts on all four nodes, or a task-owned shared
filesystem with adequate throughput. Do not copy data across cloud regions.

The loader reserves 64 windows per training component for each run seed.
These are excluded from that run's training. No arbitrary row shuffle,
new shard, new tokenizer, or substitute dataset is allowed in the main
comparison. C4 is an evaluation corpus here.

Access prerequisites: authorized Hugging Face access to the tokenizer and
any gated data, or an approved exact prepared-data directory. The initial
HTTP 401 was resolved with the supplied local HF credential and access grants.
The preparation wrapper loads only that credential, writes new shards to
local scratch, copies them to the approved PVC, and verifies the complete plan
before issuing a training content receipt.
Do not silently replace the tokenizer to work around this prerequisite.

## Objective and Evaluation

Preserve document-packed causal attention, EOS-separated attention segments,
next-token cross entropy, and `ignore_index=-100` on each window's final
target. The pinned packer only masks the final target; do not add new
cross-document target masks. No z-loss or new logits transform is introduced.

Train with the pinned microbatch mean loss divided by accumulation count,
then DDP gradient averaging. This differs from nanoGPT's summed gradients.
Use the pinned auxiliary clipping behavior described below.

Primary metric is token-weighted cross entropy, in nats, on
`paloma/c4_en`. The plan records 967,614 tokens and 2091 documents across six
files. The pinned evaluator drops the final incomplete 4096-token window:
236 full windows, with 966,420 scored targets after final-target masking.
Use all windows (`eval_max_windows=0`) and preserve padded evaluation
microbatch exclusion and FP64 loss/count reduction.

Keep the source evaluation cadence for the first reproduction wave:
`eval_every=250`, first update and final update included. For dedicated
time-to-target comparisons, rerun all compared arms with a common dense
token-based evaluation cadence; use 64M-token intervals initially and
report the actual rounded interval. Sparse evaluation gives crossing
intervals, not an exact crossing time.

Candidate hyperparameter selection uses reserved in-mixture windows,
with the same fixed selection windows across candidates. Freeze selected
settings before confirmation. Use seed 0's 64 reserved windows per component
for all tuning runs, and exclude this selection pool from every tuning run.
Compute each component's token-weighted mean loss, then aggregate using
the normalized training-mixture weights. The first exploratory wave uses
seed 0; confirmation reports its own seed-dependent holdouts without using
them to retune. Repeated exploratory C4 inspection must be
disclosed; C4 then remains a benchmark, not an untouched statistical test.
Final confirmation can additionally evaluate the remaining Paloma subsets
as secondary outcomes using a predeclared aggregation rule.

## Batch-Specific Reference Settings

Every row processes **2,684,354,560 input tokens**. Budget accounting counts
input positions, following the reference; scored targets are slightly fewer.
The numbers below come from `reproduce/marin/configs/130m/*.json`.

| Field | 0.5M | 1M | 2M | 4M extension |
| --- | --- | --- | --- | --- |
| Tokens/update | 524288 | 1048576 | 2097152 | 4194304 |
| Global sequences | 128 | 256 | 512 | 1024 |
| GPUs per run | 8 | 8 | 8 | 8 |
| Microbatch sequences/GPU | 8 | 8 | 8 | 8 |
| Accumulations | 2 | 4 | 8 | 16 |
| Updates | 5120 | 2560 | 1280 | 640 |
| Muon hidden LR | 0.016 | 0.016 | 0.032 | 0.0452548 |
| MuonT hidden LR | 0.02262741696 | 0.02262741696 | 0.04525483392 | 0.063999951815088 |
| Auxiliary AdamW LR | 0.0048 | 0.006788224 | 0.0096 | 0.0095999912336 |
| Hidden matrix decay | 0.1 | 0.2 | 0.141421 | 0.141421 |
| Auxiliary beta1 | 0.8 | 0.8 | 0.8 | 0.8 |
| Auxiliary beta2 | 0.98 | 0.9604 | 0.92236816 | 0.85076302 |
| Stable updates | 1024 | 512 | 256 | 128 |
| Linear-decay updates | 4096 | 2048 | 1024 | 512 |
| Reference Muon C4 loss | 3.462118 | 3.493352 | 3.561805 | 3.720991 |
| Reference MuonT C4 loss | 3.4505794265 | 3.4688706790 | 3.5159593145 | 3.6258195780 |

Reference losses are author-reported outcomes, not local measurements or
exact acceptance assertions. Initially use 0.5M; confirm at 0.5M, 1M, and
2M. The 4M extension tests stronger batch sensitivity after earlier gates.

Common optimizer rules:

- Hidden block matrices use Muon/MuonT; embeddings, head, and RMSNorm gains
  retain the reference AdamW treatment.
- Muon momentum 0.95, Nesterov enabled. Preserve the reference unnormalized
  momentum buffer: `buffer = 0.95 * buffer + gradient`.
- Hidden shape multiplier `sqrt(max(1, fan_out / fan_in))`.
- Newton-Schulz: 10 steps, `(3.4445, -4.7750, 2.0315)`, epsilon `1e-5`,
  `ns_mode="tpu"`: BF16 GEMM operands and FP32 output on CUDA.
- Auxiliary AdamW epsilon `1e-15`; head decay 0.1; embedding/norm decay 0.
- For Muon/MuonT, `clip_norm=1.0` clips the auxiliary group only in the
  pinned runner. Do not reinterpret it as hidden-matrix gradient clipping.
- No warmup. Linear decay after the stable portion; minimum LR ratio 0.
  Learning rates are explicit; no additional transfer multiplier.

## Input Factors and Runtime

Input-only MuonT reference settings:

| Field | Value |
| --- | --- |
| Inner / outer / decay-root strength | 0.5 / 0.125 / 0.125 |
| Input moment EMA | 0.938 |
| Trace-relative damping | 0.001 |
| Sample positions | 0 through 31, then 33,49,...,4081 |
| Positions per sequence | 286 |
| Moment collection | All accumulated microbatches/ranks |
| Sharing | Q/K/V inputs shared; gate/up inputs shared |
| Capture | Recompute sampled inputs as in Toner's Marin runner |
| Root mode | Fast cached-basis refinement with safeguarded exact fallback |
| Iterations / refresh / inline steps | 2 / 4 / 20 |
| Tolerance kind / fraction / minimum | floor / 1/15 / `5e-6` |
| Root numerical epsilon / rho limit | `1e-4` / 0.2 |
| Repair rounds | 3 |
| Host workers / threads | 2 / 4 |
| Refresh flagged / graphs / overlap | Enabled / enabled / enabled |
| Chunk | 0: full accumulation |
| Begin | Head-gradient readiness hook |

Inner roots are unnormalized in Toner's implementation; outer roots are
Frobenius-normalized. Keep this distinction for reference parity. The
fast backend's refine/refresh behavior is more detailed than "eigh every
four steps"; preserve the actual implementation.

On A100, use FA2 with the same document-packed kernel contract for every
arm. Keep model compilation `max-autotune-no-cudagraphs`, compiled head,
and the backend's separate root CUDA graphs. Record Python, Torch, CUDA,
Triton, driver, attention implementation, source hashes, and GPU model.
Initial environment inspection found Torch 2.10.0/CUDA 12.9; this is not a
claim of parity with the README's integration-check environment.

If microbatch 8 exceeds 40 GB, reduce it identically across all arms and
increase accumulation to preserve global batch. This is a recorded runtime
adaptation requiring gradient/statistic parity tests. Do not change model,
context, token budget, or data to get around memory pressure.

## Output Statistic and Update Definitions

Let `C = E[x x^T]` and `B_obs = E[delta delta^T]`, where `delta` is the
cotangent of the summed weighted training objective at a projection's
output. These are uncentered second moments. `B_obs` is not automatically
the Fisher or generalized Gauss-Newton factor.

For a PSD moment `F` of dimension `d`:

```text
A(F) = d F / tr(F) + epsilon I
inner(F,a) = A(F)^(-a)
outer(F,b) = sqrt(d) A(F)^(-b) / ||A(F)^(-b)||_F
```

Zero-trace moments use identity until a nonzero observation is available.
Symmetrize numeric moments; damp and clamp roundoff-negative eigenvalues.
Finite inputs producing invalid factors must fail or use a logged exact
fallback, never silently alter the exponent.

All actual updates use:

```text
actual_step = -learning_rate * (shape_scale * D + weight_decay * Q(W))
```

Default output settings are proposed choices, not a Toner result:
`a_out=0.5`, `b_out=0.125`, output EMA 0.938, damping 0.001.
Collect output moments every update; initially refresh their exact
eigendecomposition every four updates. The output backend is separate from
Toner's fast input-root algorithm, and its overhead must be reported.
Start with exact factors, then optimize only after numerical parity.
Update the raw output EMA after the completed accumulated backward, before
the same optimizer update. Construct factors on update 1 and then updates
5, 9, and so on. Keep both inner and outer factors fixed between refreshes;
continue collecting and updating the raw EMA on every update.

For each local backward scalar `loss = c * sum(w_t * loss_t)`, collect
`delta_sum = delta_autograd / c` **before forming outer products**.
With fixed microbatch token count `Z` and accumulation `A`, the pinned
mean/accumulated backward has `c=1/(Z*A)`, excluding ignored targets.
DDP's later parameter-gradient averaging does not change the already captured
activation cotangents. Preserve nonuniform weights in diagnostic variants.

Collect the same 286 valid activation positions per sequence as the input
recipe. A hidden activation with an unscored local target may affect later
scored positions; do not drop it solely on that basis. Form FP32 sums/counts,
pool them over all microbatches and devices, then update raw-moment EMA,
then trace-normalize. Do not average inverse roots across devices.
Freeze collection during evaluation and guard against rematerialization
double-counting. Moments, roots, and counters belong to checkpoint state.
Checkpoint identities bind source, configuration, prepared-data hashes,
and rank ownership. Continuation checks include both the three-update
full-model real-data smoke and a synthetic 24-update probe that saves after
cached-root CUDA graphs are active. The latter gate must pass before
accepting post-warmup resume behavior.

## Arms and Ablations

Use independent output factors per layer and projection. Define all
non-target matrices explicitly as retaining input-only MuonT.
In the equations, `buffer_t = 0.95*buffer_(t-1) + G_t` and
`M = G_t + 0.95*buffer_t` for Nesterov. Let
`R_a=inner(C,a_in)`, `R_b=outer(C,b_in)`,
`L_a=inner(B_obs,a_out)`, and `L_b=outer(B_obs,b_out)`.
The `epsilon` in `A(F)` is trace-relative damping, 0.001; it is separate
from Newton-Schulz epsilon 1e-5 and fast-root numerical tolerance settings.
Candidate hidden learning rates start from the batch's MuonT rate.

| Arm | Target | Direction D | Decay Q |
| --- | --- | --- | --- |
| Muon | All hidden matrices | `polar(M)` | `W` |
| Input MuonT | All hidden matrices | `polar(M R_a) R_b` | `W R_b^2` |
| Outer-left down | Down only | `L_b polar(M R_a) R_b` | `W R_b^2` initially |
| Inner+outer down | Down only | `L_b polar(L_a M R_a) R_b` | `W R_b^2` initially |
| Down hybrid | Down only | `L_b polar(L_a M)` | `L_b^2 W` |
| Output-only | All hidden matrices | `L_b polar(L_a M)` | `L_b^2 W` |
| Bi-MuonT | All hidden matrices | `L_b polar(L_a M R_a) R_b` | Factorial decay choices |
| Full hybrid | Q/K/V/gate/up input-only; O/down output-only | Respective one-sided formula | Respective shaped decay |

The first wave compares Muon, input MuonT, outer-left down, and down hybrid.
The additive down arm preserves input-shaped decay to isolate its direction.
Down hybrid changes direction, information source, and decay together:
it needs the decay/control studies before attributing an optimization gain.

Required survivor controls:

1. Output outer scaling x left-shaped decay factorial; preserve input decay
   in additive arms and include isotropic decay in replacement arms.
   For additive arms, cross `D={polar(L_a M R_a)R_b,
   L_b polar(L_a M R_a)R_b}` with
   `Q={W R_b^2, L_b^2 W R_b^2}`. For replacement arms, cross
   `D={polar(L_a M), L_b polar(L_a M)}` with `Q={W,L_b^2 W}`.
   Bi-MuonT uses the same four additive combinations.
2. Norm-grafted directions versus independently LR-tuned original directions.
   Normalizing individual factors does not fix final update norm.
   Graft each candidate matrix's direction to that matrix's input-only
   MuonT direction norm: `D_graft=D*||D_input||_F/||D||_F`.
   Leave decay unchanged. Use identical momentum and current input factors
   for this diagnostic; report its extra computation separately.
3. Diagonal B, shuffled eigenvalues/randomized basis, and gradient-Gram B
   controls. A Mousse-style control tests hook-free structural information.
   Define the hook-free control as `B_grad=G_t G_t^T / n`, with the
   already-DDP-averaged raw matrix gradient, the same raw EMA, damping,
   strengths, and refresh. It is inspired by Mousse and is not a reproduction
   of that optimizer. Randomize the orthogonal basis once per layer/seed
   while retaining observed eigenvalues; separately shuffle eigenvalues
   within the observed basis using a seeded permutation.
4. GO-MUON-form matched quarter powers and norm grafting under common factor
   settings. Label this as a formula control; its decay and exact author
   implementation remain unresolved.
   The local formula control is
   `D=L_(1/4) polar(L_(1/4) M R_(1/4)) R_(1/4)`,
   grafted per matrix to the input-only MuonT direction norm, with
   `Q=W R_b^2`. It does not establish parity with the author implementation.
5. Block-diagonal input-only down factors capped at 512 dimensions.
   This separates smaller-factor savings from output-information gains.
6. Exact SVD-polar versus finite Newton-Schulz on small diagnostic matrices.

## Experimental Sequence and Search Budget

1. Preflight: authorize data, release GPU workload, verify all node access,
   validate token receipts, attention backend, model count, and environment.
2. Correctness: CPU math tests, GPU collector tests, two-rank sum/count
   parity, compiled backward parity, and checkpoint continuation.
3. Three-update configured smoke: real data, full model/context, all four arms;
   these checks retain the full schedule and are not short training studies.
4. First four-node wave: full 0.5M-budget runs above, seed 0, with checkpoint
   spectra and common evaluation. Treat it as exploratory reproduction.
5. Bounded tuning: each family gets three LR multipliers
   `{1/sqrt(2),1,sqrt(2)}` and at most six additional short-budget trials.
   Use 167,772,160 tokens: 320/160/80 updates at 0.5M/1M/2M, completed
   20%-stable/80%-linear-decay schedules, and identical budgets within a batch.
   Explicitly separate these from early-stopped full-schedule trajectories.
   Here the families are Muon, input MuonT, and each surviving candidate.
   The nine-trial maximum applies separately per family and batch.
   Muon spends its six refinements on LR and scalar-decay sensitivity;
   input MuonT may refine input strengths, damping, EMA, and refresh within
   the same six-trial allowance. Non-0.001 damping requires an exact-root
   reference backend because the pinned fast implementation fixes damping.
6. Output refinement: consider `a_out={0,1/4,1/2}`,
   `b_out={1/16,1/8}`, damping `{1e-3,1e-2}`, EMA `{0.938,0.97,0.99}`,
   refresh `{1,4,8}`. Spend the six-trial allowance on diagnostic-motivated
   one-factor changes; do not run this entire Cartesian product.
7. Confirm selected settings with paired seeds `{0,1,2}` and new initializations
   at 0.5M, 1M, and 2M. Keep initialization/data order paired across arms.
   Freeze all settings before this stage.
8. If successful, evaluate full hybrid/Bi-MuonT, 4M batches, or a larger
   Toner Marin model; do not scale every unpromising variant.

For all LR sweeps, maintain the reference auxiliary rates and the chosen
scalar hidden decay initially. Run a separate matched `LR*decay` sensitivity
study on survivors; record both conventions because LR changes regularization.
Give baseline families equal tuning tokens and failed-run accounting.

## Diagnostics, Profiling, and Decision Gates

At baseline checkpoint fractions 0%, 10%, 50%, and 100%, collect input/output
spectra on two disjoint reserved sequence pools. Begin with 32 sequences per
pool; expand if moment/subspace estimates are unstable.
Log top/mean eigenvalue ratios, leading energy shares, covariance drift,
direction cosines, descent/decay norms, effective step ratios, and root age.

Compare local joint scores `E[(delta^T D x)^2]` with
`tr(B_obs D C D^T)` and inspect cross-token contributions. Separately compare
scalar Hessian/GGN contractions on fixed unit-Frobenius directions and fresh
batches. Poor curvature correspondence rejects that explanation, while
leaving an empirical optimizer hypothesis testable.
The explanation under test is that the factorized score ranks local
directions similarly to fresh-batch GGN/Hessian directional contractions.
Predeclare unit-Frobenius directions and report rank correlation and relative
error on independent pools. A correlation interval containing zero provides
no support for this explanation; there is no numerical-error cutoff that
proves curvature fidelity.

Track weights/activations/gradients/momentum/final updates separately for
stable rank and entropy rank. Exact polar's equal singular values are partly
an algorithmic construction. Off-diagonal energy in a weight singular basis
does not establish finite-step spectrum preservation.

Record steady-state median/p95 step latency, input and output collection,
communication, root refresh/refinement, fallback counts, matrix application,
peak device memory, host memory, and end-to-end wall time.
Exclude startup compile from steady-state timing but include it separately
in end-to-end results. No timing conclusions may use a diagnostic mode that
disables production checkpointing/rematerialization.

One FP32 down input moment is 16 MiB versus 1 MiB for its output moment:
16x moment-state reduction and 64x nominal dense eigendecomposition work
reduction at expansion 4. These are per-factor ratios, not measured
whole-training speedups; roots, workspaces, collection, and communication
must also be accounted for.

Suggested thresholds, frozen after baseline variability estimation:

- Optimization: at least 0.003-nat mean C4 improvement over tuned MuonT,
  consistent direction across seeds, and uncertainty reported.
- Efficiency: no more than 0.002-nat degradation and at least 10% measured
  time-to-target reduction over fast input MuonT on identical hardware.
- Loss margins refer to the final fixed-token-budget checkpoint.
  Efficiency also requires lower peak allocated device memory in the same
  full configuration. Report peak reserved memory and host memory separately.
- Three seeds are a minimum estimate of variation, not proof of significance.
  Increase seeds if the uncertainty overlaps the practical margin.
- Candidate gains that disappear after fair norm/LR/decay controls do not
  support an output-information mechanism.
- A failed data/numerical check invalidates a run; retain its failure record.
  Do not select validation minima from incomplete/diverged trials.

Predeclare a target loss per batch from the baseline-only pilot.
Use common token-based evaluation times and report crossing intervals.
If a target is unreached, report censoring. Do not infer token savings from
a single final loss gap or power-law extrapolation.
Time-to-target starts at launch, includes compilation, evaluation, and
checkpointing, and stops after completing the first evaluation below target.
Also report a steady-state training-only clock as a secondary diagnostic.

## Persistence, Run Records, and Launch Gate

The pinned Marin entry point only saves a final model and metrics; it is not
a training-resume interface. The experimental runner must save model,
auxiliary state, hidden momentum, input/output EMAs, EMA weights, root caches,
step, data position, seeds/RNG state, and configuration. Synchronize async
root work before saving. Recompute input root bases on resume only as an
explicit numerical adaptation, with continuation parity tested.

Checkpoint every 250 updates and at final completion initially; use
separate names and atomic replacement. Do not overwrite a completed run
or append a second launch's metrics to it. Preserve one readable run summary
and structured JSONL metrics; compare runs in one experiment namespace.

Every run record contains exact source hashes, pinned upstream revision,
dataset/tokenizer revisions and checksum receipts, model/optimizer settings,
hardware/runtime, command, start/end time, seed, status, checkpoint paths,
and reason for any restart. Keep this work private until publication is
authorized; no public logbooks issue or push is part of the current request.

The four-node launcher must refuse training until:

- all 32 GPUs are available for this task;
- prepared data has passed full-plan and checksum verification;
- FA2 and the compiled head run on A100;
- the four arms pass math/collector/distribution tests;
- a real-data smoke run succeeds with no nonfinite values or OOM;
- checkpoint save/restore is validated.

Current prerequisites are full prepared-data verification and release of the
identified GPU keepers with owner approval. Tokenizer/source access is resolved.
Synthetic data may be used only for labeled correctness checks; it cannot
be reported as the Marin experiment.

## Budget and Deliverables

The first wave is four full runs: 10,737,418,240 training tokens.
Three-arm confirmation at two batches and three seeds is 18 full runs:
48,318,382,080 tokens. Three batches make 27 runs:
72,477,573,120 tokens. The three confirmation arms are Muon, input MuonT,
and the selected candidate. A five-arm grid with Muon, input MuonT,
outer-left down, inner+outer down, and down hybrid at three batches and
three seeds would be 45 runs and 120,795,955,200 tokens before tuning.
Use the staged gates to avoid paying for the full grid prematurely.

Estimate GPU hours only after measured A100 throughput, including refresh
spikes and evaluation. The 32-GPU allocation reduces campaign elapsed time
through four simultaneous jobs; it does not reduce total GPU-hours by itself.

Deliver the plan, separately versioned experiment code/configs/tests,
verified data manifest, launch/preflight records, baseline reproduction
report, paired-seed comparisons, loss/time/memory curves, and a decision
about output information and hybrid efficiency. Successful scale-up uses
Toner's 300M or 520M Marin configuration and its own batch-specific settings.
