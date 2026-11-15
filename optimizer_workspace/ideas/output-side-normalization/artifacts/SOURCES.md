# Source Collection and Technical Review

Reviewed on 2026-11-15, Asia/Shanghai.

## Collection Status

The proposal was read in full. External retrieval works through
`http://star-proxy.oa.com:3128`. The earlier access limitation came from
commands that did not inherit the proxy exports in `/root/.bashrc`.
The built-in web search tool still returned no usable records, but direct
proxied requests retrieved primary sources from arXiv and author repositories.
No global shell or Git configuration was changed.

`papers/` contains the retrieved papers; `external-sources/` contains the
Toner article and pinned author code. `retrieval/` contains bibliographic
metadata, search responses, and alternate full-text captures. A search
response is discovery evidence, not a source supporting a scientific claim.
Versions, hashes, sizes, and validation results are in
`external_source_manifest.json`. The local snapshots have a separate manifest.
All ten PDFs passed full-page parsing and embedded arXiv-version checks.
The source manifest indexes 17 primary artifacts. The complete article
matches its pinned Git blob, and both pinned source archives parse correctly.

To repeat parsing and regenerate the provenance manifest without installing
dependencies into the repository:

```bash
uv run --offline --no-project --with pypdf==7.1.0 --with fonttools==4.68.0 \
  python optimizer_workspace/ideas/output-side-normalization/artifacts/verify_sources.py
```

The repository's Echo search also could not run: its CLI failed on a missing
`requests` dependency, and no cached Marin credential directory was present.
No repository environment was installed or changed to repair that service.

This source review preceded implementation. Current implementation and
execution status are recorded in `../EXPERIMENT_STATUS.md`; this review
contains no new training results.

## Local Primary Sources

Base revision: `6e09d9043eb71d7bd487b5e3d79e275b95c454fd`.

| Snapshot | Original Path | Relevant Evidence |
| --- | --- | --- |
| `local-sources/muon.py` | `lib/levanter/src/levanter/optim/muon.py` | Existing Muon momentum, Nesterov handling, matrix scaling, parameter routing, and isotropic weight decay. |
| `local-sources/grugmuon.py` | `lib/levanter/src/levanter/optim/grugmuon.py` | Grug matrices use `(fan_in, fan_out)` storage, the transpose of the proposal's convention. Includes replicated and stacked/sharded Newton-Schulz paths. |
| `local-sources/muon_util.py` | `lib/levanter/src/levanter/optim/util.py` | Matrix flattening and unflattening, scanned-layer handling, and finite-step Newton-Schulz implementations and coefficients. |
| `local-sources/soap.py` | `lib/levanter/src/levanter/optim/soap.py` | Existing gradient-derived matrix preconditioners, update cadence, damping, block partitioning, and eigenspace machinery. These factors are not activation or output-cotangent moments. |
| `local-sources/backward_flow.py` | `lib/levanter/src/levanter/analysis/backward_flow.py` | Identity custom-VJP instrumentation, diagnostic backward tracing, and optional gradient-scale correction. Does not already expose dense output-gradient factors as optimizer state. |
| `local-sources/grug_base_model.py` | `experiments/grug/base/model.py` | Separate Q/K/V/O projections, two-matrix ReLU MLP, sequential residual additions, and a checkpointing path that changes when backward-flow capture is active. |
| `local-sources/grug_base_train.py` | `experiments/grug/base/train.py` | Jitted loss/gradient/update boundary, mean-loss reduction, backward-flow capture scale, and training-state structure. |
| `local-sources/grug_base_launch.py` | `experiments/grug/base/launch.py` | Existing model labeled `GRUG_130M_MODEL`: hidden size 512, intermediate size 1792, six layers, sequence length 4096, vocabulary size 128256. Its default data is a Nemotron-based mixture, not the proposal's C4 configuration. |
| `local-sources/gpt2_small_fast_muon.yaml` | `lib/levanter/config/gpt2_small_fast_muon.yaml` | Existing GPT-2 Muon configuration on OpenWebText. This is another available recipe, not a Toner replication. |
| `local-sources/grad_accum.py` | `lib/levanter/src/levanter/grad_accum.py` | Existing microbatch reduction semantics. A new statistic collector must preserve these semantics and combine sums/counts correctly. |

A scoped search of `lib/levanter`, `experiments`, `docs`, and `.agents` found no
existing Toner/MuonT/FastMuonT implementation or references for the named
spectral variants. This does not establish anything about unsearched branches
or upstream repositories.

Snapshot provenance, byte counts, and SHA-256 hashes are recorded in
`local_source_manifest.json`.

## Retrieved Primary Sources

| Source | Local Artifact | Why It Matters |
| --- | --- | --- |
| Mohamadi, Wen, Wang, and Li, **Toner: Improving Muon and AdamW with Input Geometry** (2026) | `external-sources/toner-blog-pinned.html`; `external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/` | Direct reference for input factors, decoupled powers, shaped decay, measurements, and reproducible configurations. This is an author article, not an identified arXiv paper. |
| Tong Che, **Second-Order Muon Done Right: A Principled Marriage of Spectral Geometry and Curvature**, arXiv:2608.09763v1 | `papers/go-muon-2608.09763.pdf`; `retrieval/second-order-muon.html` | Closest output-cotangent two-sided comparator; matched quarter powers, shrinkage, norm graft, and explicit conditional curvature interpretation. |
| Keller Jordan, **Muon** author implementation | `external-sources/muon-f98f1cac.py` | Pinned implementation reference; the Toner reproduction runner determines the actual benchmark arithmetic. |
| Martens and Grosse, **Optimizing Neural Networks with Kronecker-factored Approximate Curvature**, arXiv:1503.05671v7 | `papers/kfac-1503.05671.pdf` | Fisher-factor semantics and Kronecker approximation assumptions. |
| Kunstner, Balles, and Hennig, **Limitations of the Empirical Fisher Approximation for Natural Gradient Descent**, arXiv:1905.12558v3 | `papers/empirical-fisher-limitations-1905.12558.pdf` | Observed-label score moments need not approximate loss curvature. |
| Gupta, Koren, and Singer, **Shampoo: Preconditioned Stochastic Tensor Optimization**, arXiv:1802.09568v2 | `papers/shampoo-1802.09568.pdf` | Gradient-derived matrix factors and a hook-free control. |
| Vyas et al., **SOAP: Improving and Stabilizing Shampoo using Adam**, arXiv:2409.11321v2 | `papers/soap-2409.11321.pdf`; `local-sources/soap.py` | Existing eigenspace machinery and refresh-frequency considerations. |
| Zhang et al., **Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning**, arXiv:2603.09697v2 | `papers/mousse-2603.09697.pdf`; `external-sources/Mousse-d00c1bf17790fbe56424ee5567cce80d8e75f4b2/` | Two-sided matched polar using Shampoo-derived factors; separates output-cotangent information from generic structural preconditioning. |
| Shumaylov et al., **Muon is Not That Special: Random or Inverted Spectra Work Just as Well**, arXiv:2605.11181v1 | `papers/freon-2605.11181.pdf`; `retrieval/freon-fulltext.html` | Challenges strict spectral-geometry explanations and motivates norm/alignment controls. |
| Goyal and Ramkumar, **Spectrally Targeted Muon**, arXiv:2610.10965v1 | `papers/spectrally-targeted-muon-2610.10965.pdf` | In the studied tasks, training benefits from small momentum directions differ from parameter-spectrum flattening. |
| Du and Su, **The Newton-Muon Optimizer**, arXiv:2604.01472v1 | `papers/newton-muon-2604.01472.pdf` | Closely related input-only preconditioning; its surrogate explicitly includes output curvature before further assumptions. |
| Dang, Wen, and Malladi, **The Best Optimizer Depends on Batch Size**, arXiv:2610.08975v2 | `papers/batch-size-optimizer-2610.08975.pdf` | Batch-dependent rankings and the need to tune baseline settings at each batch size. |

Canonical URLs and retrieved versions are recorded in `reference_queue.json`,
which now functions as the resolved reference index. The Toner code is pinned
at `dab7f3777c75e740536df822f50ffa1d929a340c`; its article is pinned at website
revision `453badd98fe389f009884063ca88378db15bf582`.
The complete article's Git blob hash is
`1db4aae72ab2e0deb979ce5228f273f82a4bac74` and its size is 3,644,748 bytes.
`toner-blog.html` is a byte-identical convenience copy.
The plain-text article capture uses an alternate host and can lose formulas;
use the complete canonical HTML and code when checking equations.

The Freon README from `John-HW-Cao/Freon` is a third-party discovery artifact.
Its ownership does not match the paper authors, and it is not used to verify
the paper's algorithm or results. No GO-MUON author code was identified in
the retrieved paper or scoped repository search. Do not call a new local
implementation an exact code reproduction without resolving that gap.

### Verified Toner Settings and Limits

- The article reports input top-to-mean eigenvalue ratios of 85--156 across
  32 Marin attention layers, and about 20--1850 in nanoGPT, median 129.
  These are author-reported measurements; none were independently replicated.
- The default MuonT strengths are `(a_in, b_in) = (0.5, 0.125)`,
  respectively the inner and outer powers. The inner inverse root in
  the reference code is unnormalized; the outer root is Frobenius-normalized.
  Normalizing both roots is therefore a new numerical variant. Exact polar
  ignores a positive scalar, but finite Newton-Schulz with epsilon need not.
- The Marin 130M/0.5M configuration uses hidden size 512, MLP size 2048,
  32 layers, context 4096, vocabulary 128256, and a gated MLP.
  The size label excludes embeddings and the output head. The local six-layer
  ReLU Grug model with a similar label is a separate integration benchmark.
- Marin trains on a DCLM/StarCoder/ProofPile mixture and evaluates on C4
  from Paloma. C4 is not the training corpus in this reproduction.
- That configuration processes 2,684,354,560 tokens in 5120 updates of
  524,288 tokens. It uses hidden LR 0.02262741696, auxiliary LR 0.0048,
  momentum 0.95, statistic momentum 0.938, damping 0.001, and decay 0.1.
  The schedule has no warmup, 1024 stable steps, and 4096 linear-decay steps.
- Its finite Newton-Schulz map has 10 steps and epsilon `1e-5`; the shared
  polynomial coefficients are `(3.4445, -4.7750, 2.0315)`. The configured
  TPU-like CUDA arithmetic uses BF16 operands and FP32 output.
  Fast roots refresh every four updates. These differ from local defaults.
- The general reference implementation recomputes exact eigendecompositions;
  the fast implementation uses asynchronous root updates. Match the fast
  reference before making wall-clock comparisons.
- The nanoGPT recipe uses 162M parameters, GPT-2-tokenized FineWeb10B at a
  pinned dataset revision, and an unshuffled loader. The 128K/512K/1M runs
  each process 1,703,936,000 tokens; 2M rounds to 1,704,984,576 tokens.
  The exact 512K and 1M batches are 524,288 and 1,048,576 tokens.
- Its 512K MuonT configuration uses hidden LR 0.035355, momentum and
  statistic momentum 0.95, decay 0.05, and 12 BF16 Newton-Schulz steps
  with coefficients `(2.0, -1.5, 0.5)` and epsilon `1e-7`.
  Gradient reduction is `sum` across accumulation/ranks, and auxiliary
  groups have distinct rates. Do not substitute the Marin arithmetic.
- Several curvature/noise probes in the article concern the earlier
  `(0.25, 0.25)` recipe. They do not establish the mechanism of the final
  `(0.5, 0.125)` recipe or output-side normalization.

### Verified GO-MUON Comparison

Section 3 forms observed-label activation/backward factors, applies
OAS shrinkage and inverse quarter powers, and computes
`D0 = P_B polar(P_B Mhat P_A) P_A`. It then grafts
`D = D0 * sqrt(min(m,n)) / norm(D0, Frobenius)`.
Its momentum buffer follows `Mbuf_t = beta Mbuf_(t-1) + (1-beta) G_t`,
and `Mhat_t = (1-beta) G_t + beta Mbuf_t` is the Nesterov-style source.
This `Mhat` plays the role of the orthogonalization source called `M`
in the notation below; it is not a matrix-norm normalization.
The stated refresh interval is four updates with backward EMA 0.97 and
sparse EMA coefficient `beta_B^4`; momentum remains fresh each update.
Its rank-deficient implementation convention allows a semi-orthogonal
completion, which must be distinguished from a support-only polar.

The paper's complete hidden-matrix update equation does not specify the
isotropic decay claimed in the proposal. That comparison is unverified.
Its main empirical settings include small character-language-model and
modular-addition tasks; they do not establish a large-LM output-factor gain.
For a matched benchmark, distinguish a GO-MUON-form formula control using
common factor settings from a faithful reproduction of its OAS/refresh/graft
recipe. Label any chosen decay rule explicitly.

## Technical Review of the Proposal

The following are algebraic checks and local implementation observations.
The source-specific findings above are separately attributed. No new
checkpoint measurements or training results were produced.

### Notation

Use the proposal's mathematical orientation: `y = W x`,
`W, M, D in R^(m x n)`, `x in R^n`, and `delta in R^m`. Here `M` is the
momentum or Nesterov matrix supplied to orthogonalization, `D` is the
unscaled descent direction, and `P` is a polar factor.

For a positive-semidefinite moment `F in R^(d x d)` with positive trace,
define `A_F = d F / tr(F) + epsilon_F I` and
`T_q(F) = sqrt(d) A_F^(-q) / norm(A_F^(-q), Frobenius)`.
Set `R_q = T_q(C)` and `L_q = T_q(B_obs)`. At `q = 0` the factor is identity.
A zero-trace factor needs a specified initialization/fallback, such as
identity, before this definition can be used.

The proposed full direction is
`D = L_bout polar(L_aout M R_ain) R_bin`. Its proposed decay operator is
`Q(W) = L_bout^2 W R_bin^2`; the left-only and right-only formulas omit the
unused factors. The proposal uses `s = sqrt(max(1, m/n))`.

### Output Cotangent Moments Are a Hypothesis, Not Established Curvature

For a local linear application, define
`C = E[x x^T]` and `B_obs = E[delta delta^T]` using the actual activation and
output cotangent. A local outer-product surrogate has a block moment
`E[(x x^T) kron (delta delta^T)]`. Replacing this with `C kron B_obs` requires
an approximation of the joint statistics.

There is a second distinction: cotangents of observed-label loss are not
automatically a true Fisher or generalized Gauss-Newton factor. In a
transformer, the cotangent at one token's hidden state can include effects
from many downstream loss positions. Outer products of token-local
cotangents also omit cross-position terms in a sequence gradient block.

Call
`tr(B_obs D C D^T)` a **Kronecker-factored cotangent-moment quadratic**.
The proposal's joint energy shares are contributions to that quadratic.
Calling them Gauss-Newton shares requires an independently justified
Gauss-Newton factor.

For a small diagnostic subset, compare the local joint statistic
`E[(delta^T D x)^2]` with `tr(B_obs D C D^T)` using the same uniformly
sampled valid positions, unreduced-objective cotangents, and fixed direction
`D`. Here unreduced cotangents mean a unit-seeded summed weighted loss,
not separately differentiated individual losses. This checks factorization
at a consistent scale.

Separately, for a fixed valid length `T` define
`a_t = delta_t^T D x_t`, `Q_seq = E[(sum_t a_t)^2] / T`, and
`Q_local = E[sum_t a_t^2] / T`. Their difference is the cross-position
contribution at this normalization. If the intended sequence objective is
a mean, its squared directional score instead scales the numerator by
`1/T^2`. For variable lengths, specify whether sequence or token weighting
defines the expectation.

Compare scalar contractions `vec(D)^T H vec(D)` and, when available,
`vec(D)^T GGN vec(D)` with the factored quadratic on the same objective.
Use unit-Frobenius directions and report the moment normalization explicitly.
Evaluate correlation and ranking on held-out batches and multiple directions;
do not assume equality between a score moment and Hessian curvature.
Anisotropy by itself does not show that inverse-power rescaling helps.

### Exact Polar Geometry Already Equalizes the Full-Rank Short Side

Use the canonical partial-isometry convention
`polar(M) = U_r V_r^T`, retaining only singular vectors with nonzero singular
values; numerical tests must state an absolute/relative rank threshold.
Untruncated reduced-SVD `U V^T` can extend into arbitrary null directions
and is a different rank-deficient convention.

Let `P = polar(M)` use that convention. With full row rank and
`m <= n`, `P P^T = I_m`; with full column rank and `m >= n`,
`P^T P = I_n`. With deficient rank the identities become projectors onto
the active subspaces.

Thus a wide down-projection already has isotropic row energy after exact
polarization. For a full-row-rank `M R_a`, where `R_a` is invertible, the
same identity applies to `polar(M R_a)`. Multiplication by a nontrivial
outer right factor can subsequently change that energy: the row Gram is
`P R_b R_b^T P^T`. Row energy is preserved when this is `I_m`, including
cases where a factor's changes are confined outside the row space.

The proposal's claim that preserving left singular vectors necessarily
leaves dominant output magnitudes is too broad. Nonzero singular values are
equalized by exact polar; retained subspaces, output metrics, outer factors,
and finite Newton-Schulz accuracy are separate issues.

An inner-left factor can sometimes have no effect. If
`M = S Q`, `S = (M M^T)^(1/2)` is positive definite, and a positive-definite
`L` commutes with `S`, then `L S` is positive definite and
`polar(L M) = Q`. This is a useful unit-test case.

Test **outer-left-only** scaling separately from inner-left and full
two-sided scaling. Use exact SVD-polar on small matrices to distinguish
geometric effects from finite-step Newton-Schulz effects.

### Weight Decay Is a Separate Choice

For frozen symmetric factors, `L_b^2 W R_b^2` is the gradient of
`0.5 * tr(W^T L_b^2 W R_b^2)`. This makes it a well-defined shaped penalty
at that step; it does not establish that it is the right regularizer for a
new output-side method. Factors changing over time define an adaptive
penalty, and the optimizer does not differentiate through those factors.

Left-only and two-sided shaped decay should be compared with isotropic
decay. Log descent and decay norms separately and compare decay strength
fairly. Preserving a Toner recipe for replication is distinct from assuming
its decay choice transfers to a new method.

### Normalize Global Update Scale Explicitly

Normalizing each factor to Frobenius norm `sqrt(d)` does not fix the
Frobenius norm of the final two-sided update.

Use one convention:

`U = s D + weight_decay * Q(W)`

`actual_step = -learning_rate * U`

Then effective step ratios are
`norm(actual_step) / norm(W)`, with analogous spectral ratios. The proposal
uses `Delta W` for an actual step and later multiplies it by learning rate
again. Record gradient-only and total-step ratios separately.

A norm-matched diagnostic ablation can compare directions at equal descent
norm. Report it separately from the unmodified, independently LR-tuned
optimizer: it changes the method.

### Shared Residual Dimension Does Not Imply Shared Cotangents

In `grug_base_model.py`, attention and MLP write to sequential residual
nodes. Their cotangents differ because the attention output also influences
the following MLP. They should have separate output factors initially.

Exact sharing is reasonable for projections with the same activation
tensor, such as Q/K/V inputs. Output-factor sharing requires the same
cotangent in the same coordinate system, or an explicit approximation
validated by covariance and update comparisons. Do not share factors across
layers merely because their dimension is equal.

### Gradient Collection Must Preserve Objective and Execution Semantics

The trainer already rescales diagnostic gradient RMS by the loss-weight sum.
A covariance collector must apply a global cotangent scale before forming
outer products, since scale enters second moments quadratically.

For uniform mean loss, removing its scalar normalizer recovers cotangents
of the summed objective. It does not recover independent per-loss scores
at every hidden position. Global scalar normalization cancels in a
trace-normalized factor for one batch, but changing scales still matter
when raw batches are mixed in an EMA.

Use valid activation positions when sampling statistics. A zero local loss
weight does not prove that the activation has no gradient: it can affect
later scored tokens. Preserve any nonuniform objective weights and document
whether their squared weighting is intended in the moment.

Aggregate raw statistic sums and sample counts over microbatches and devices
before trace normalization. Average factors after eigendecomposition is a
different operation. Test rematerialization, accumulation, distribution,
and restart behavior explicitly.

For each backward invocation with scalar loss
`ell = c * sum_t w_t ell_t`, collect `delta_sum = delta_autograd / c`
before outer products. Derive `c` from the actual local loss reduction,
accumulation multiplier, and mixed-precision loss scale. Device reduction
of parameter gradients does not itself rescale already-captured local
activation cotangents. For a local mean with denominator `Z`, no extra
scaling, and uniform reduction, `c = 1/Z`. Test an explicit summed-loss
backward against each accumulated/distributed collector.

The existing backward-flow mode disables normal block checkpointing.
Diagnostic timing from that mode cannot establish training throughput with
a production statistic collector.

### The Hybrid Cost Claim Is Per Factor, Not End-to-End

The proposal's hybrid retains input-only MuonT `(a_in,b_in)=(0.5,0.125)`
on Q/K/V and MLP gate/up projections. It replaces O and down projections
with left-only output factors `(a_out,b_out)=(0.5,0.125)` and left-shaped
decay, removing their right factors. Treat O and down cotangents separately.
A down-only replacement is a narrower, separately labeled pilot.

With `d_ff = k d_model`, replacing one dense down-projection input factor
with a `d_model`-dimensional output factor reduces that factor's matrix-state
size by `k^2` and nominal dense eigendecomposition work by `k^3`.

At `k = 4`, these algebraic ratios are 16 and 64. The local Grug recipe has
`k = 1792 / 512 = 3.5`, giving 12.25 and 42.875. Neither is a measured
end-to-end speedup. Collection, matrix application, communication,
refresh frequency, and backward capture must be timed separately.

The exact local parameter count should also be measured rather than
inferred from the `GRUG_130M_MODEL` label. Its separate vocabulary
projection and its ReLU MLP differ from the gated architecture described
in the supplied proposal. Toner's actual configuration is described above.

Compare the hybrid with a right-only dimension-capped, blocked, or
low-rank control. A systems improvement could come from factor
approximation alone.

### Rank Metrics Need Limited Interpretations

For distinct nonzero singular values, `u_i^T actual_step v_i` gives their
first-order change. Off-diagonal energy in the weight's singular basis is
not proof that a finite update preserves the spectrum. Repeated singular
values need block treatment; reduced SVDs can omit rectangular complement
directions and rank-creating components.

Exact polar factors have equal nonzero singular values by construction.
Their effective rank partly reports the algorithm's definition. This does
not hold generally for the final shaped direction `D` or `actual_step`
including decay. Record these as distinct objects. Track weight and
activation rank as secondary mechanistic metrics, with singular values and
stable rank. Do not make rank improvement the primary endpoint.

Define zero-matrix conventions for rank metrics and record whether entropy
uses singular values or squared singular values.

## Evidence Boundary

Retrieval and source inspection support
the experimental design; they do not establish that output-side normalization
works. External claims remain author-reported. Complete source artifacts are
indexed with their provenance and validation status.

The source-review manifest records the original proposal checksum
`2ab5d88ea85e0531526b31a5e800d625d017227483564ccd4d671220adbeaad2`.
The repository formatter subsequently removed trailing whitespace from the
working proposal; its checksum is now
`f8204370858b828e90e87a06f5658a9013880dbd002939c334e47b08c927e4b7`.
No scientific claims were revised in that formatting pass.

The first empirical question remains whether a held-out, repeatable
output-cotangent statistic improves useful update directions beyond input
preconditioning. A separate question is whether replacing the largest
input factors improves the measured compute/quality frontier.
