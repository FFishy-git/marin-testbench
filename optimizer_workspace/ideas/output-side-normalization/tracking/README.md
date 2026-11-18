# SwanLab Historical Import

Scientific setup, completed results, and experiment status are maintained in
[ENDEAVOR.md](../ENDEAVOR.md#progress-and-logging).

`swanlab_upload.py` imports the completed histories of one wave,
`runs/WAVE/`, selected with `--wave`, for example `--wave wave2`. Each
wave's arms are listed in the uploader's `WAVES` table.
Names and tags use the manifest's actual seed and batch. Directories are
`ARM`, `ARM_seedN`, `ARM_BATCH`, or `ARM_BATCH_seedN`.
It runs independently of Torch and the checkpoint-bound `experiment/`
package. The private project is
[output-side-normalization-marin](https://swanlab.cn/@siyuc3141/output-side-normalization-marin).

The isolated logging environment contains `swanlab==0.10.5` and
`python-dotenv==1.2.4`. It is under the approved PVC output root.
The uploader reads only `SWANLAB_API_KEY` from the designated credential
file and uses `swanlab.login(save=False)`. It does not source the file
or export its other credentials.

Run from
`/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization`:

```bash
PVC=/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization
HTTP_PROXY=http://star-proxy.oa.com:3128 \
HTTPS_PROXY=http://star-proxy.oa.com:3128 \
ALL_PROXY=http://star-proxy.oa.com:3128 \
"$PVC/.venv-logging/bin/python" tracking/swanlab_upload.py \
  --root "$PVC" \
  --credential-file /root/dl/runtime/script/.env \
  --wave wave1
```

Read-only cloud verification, with no metric upload:

```bash
PVC=/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization
HTTP_PROXY=http://star-proxy.oa.com:3128 \
HTTPS_PROXY=http://star-proxy.oa.com:3128 \
ALL_PROXY=http://star-proxy.oa.com:3128 \
"$PVC/.venv-logging/bin/python" tracking/swanlab_upload.py \
  --root "$PVC" \
  --credential-file /root/dl/runtime/script/.env \
  --wave wave1 \
  --verify-only
```

Both commands refresh the local receipt at
`$PVC/run-preparation/swanlab-upload-WAVE.json`. SDK output stays under
`$PVC/swanlab/`. Subsequent invocations verify existing uploads rather
than creating duplicate experiments. An interrupted import recorded as
`initialized` requires inspection with `--verify-only`. If all cloud points
are present and the run is `FINISHED`, verification marks it `verified`.
A partial history or other cloud state causes a nonzero exit with the
missing metric count, value mismatch, or state. Automatic recovery from
a partial upload is not implemented; preserve the receipt and original
JSONL for owner-assisted recovery. The uploader does not silently resend
a partial history or delete cloud runs.

Before authentication, validation requires completed manifests,
consecutive unique updates, matching token budgets, finite metrics,
and final checkpoint receipts with nonempty files for all ranks.
Provenance hashes bind each cloud run to its JSONL, manifest, and
checkpoint receipt. Configuration includes the original model, training,
optimizer, schedule, data, runtime, hardware, and source/data identities.

Metric mapping:

| JSONL | SwanLab |
| --- | --- |
| `tokens` | `tokens` |
| `train_loss` | `train/loss` |
| `c4_loss` | `eval/c4_loss` |
| `selection_loss` | `eval/selection_loss` |
| `heldout_losses["heldout/SOURCE"]` | `eval/heldout/SOURCE` |
| Other scalars | `runtime/KEY` |
| `input_gpu_ms_previous["KEY"]` | `runtime/input_gpu_ms_previous/KEY` |

`selection_loss` is the weighted average of the held-out DCLM,
StarCoder, and Proof-Pile losses, with weights 3.8, 0.25, and 0.055,
divided by their sum 4.105. C4 is evaluated separately and is not
included in this selection metric.

Each log call uses the original update number. Training, evaluation,
and runtime charts use `tokens` as their default X axis. Sparse
evaluation/timing data is not filled or interpolated. Full-resolution
cloud exports are checked against every local scalar column, including
exact update identities and integer values. Float64 values permit at
most two ULPs of CSV serialization error, measured in the first import;
the receipt records per-column maximum errors. This bound applies only
to cloud serialization, not to training numerics.

All experiments are labeled `historical-import`. Cloud timestamps and
duration refer to uploading; use `training_record` timestamps and
`runtime/end_to_end_seconds` for original training time. Automatic
hardware/environment/requirements/git monitoring and terminal capture
are disabled. No source files, datasets, checkpoint weights, or `.env`
files are uploaded.

Offline uploader tests:

```bash
PVC=/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization
"$PVC/.venv-logging/bin/python" -m unittest discover -s tracking -p 'test_swanlab*upload.py' -v
```

## Monitoring and Comparison

Wave 7's fixed-age refresh candidate writes structured installation events
beside the usual scalar metrics. Its original source and GPU-validation
binding stays frozen. The separate `wave7_release.py` binding uses
`swanlab_refresh_upload.py` plus `refresh_history.py` to project those
events into per-update/group counts and age/source-update extrema.
Every original scalar is retained unchanged; raw events remain local,
bound by the original JSONL hash. Only the scalar projection is sent to
SwanLab. The logging-release binding requires the original successful
four-arm GPU receipt, restoration state, and unchanged candidate,
helpers and archive before it can launch.

`prepare_fixed_age_tuning.py` creates a separate four-family LR study
outside the frozen Wave-7 candidate. `analyze_fixed_age_tuning.py` verifies
the original successful numerical evidence and ranks only complete
equal-budget family sweeps by held-out-mixture selection loss.
These offline helpers do not launch jobs or release GPU allocations.
The fixed-age study is distinct from the historical Wave-5-bound LR
campaign; its short configurations still require actual GPU validation.

`prepare_lr_boundary.py` extends the completed, verified initial fixed-age
grid using a separate eight-trial campaign. All four families receive
two new matrix LRs, 0.04525483392 and 0.06399999989260598, with the same
320-update schedules and controls. `analyze_lr_boundary.py` binds the
parent results/cloud provenance and permits five-point family choices
only after both new trials complete. Late evaluation-token lists are
retained for coverage-checked document comparisons. The original
campaign, numerical source and frozen helpers are unchanged.

```bash
/opt/conda/envs/torch-base/bin/python tracking/prepare_lr_boundary.py \
  --root "$PVC" \
  --parent "$PVC/campaigns/lr-fixed-age-0p5m-20261116/campaign.json" \
  --out "$PVC/campaigns/lr-fixed-age-upper-boundary-0p5m-20261117"
/opt/conda/envs/torch-base/bin/python tracking/analyze_lr_boundary.py \
  --campaign "$PVC/campaigns/lr-fixed-age-upper-boundary-0p5m-20261117/campaign.json" \
  --out "$PVC/campaigns/lr-fixed-age-upper-boundary-0p5m-20261117/analysis.json"
```

This campaign was prepared November 17 at 03:50:33 and has separately
frozen validation/execution helpers at 04:08. It is not GPU-validated or
queued. Do not repeat preparation/freezing or repoint the frozen original
three-LR validator/executor at it. The mutable shared controllers now
route only the explicit original/boundary protocols and reject arbitrary
campaign grids. `boundary_handoff.py` rederives completed Wave-8 results
using the original frozen analyzer, checks full-resolution cloud
provenance, then requires a fresh terminal-worker/restoration snapshot.
All eight new actual-config gates still precede any training launch.

```bash
campaign="$PVC/campaigns/lr-fixed-age-upper-boundary-0p5m-20261117/campaign.json"
/opt/conda/envs/torch-base/bin/python \
  "$(dirname -- "$campaign")/validation-helpers/run_fixed_age_validation.py" \
  --campaign "$campaign" --phase check
/opt/conda/envs/torch-base/bin/python \
  "$(dirname -- "$campaign")/execution-helpers/run_fixed_age_tuning.py" \
  --campaign "$campaign" --phase check
```

Only after completed Wave-8 analysis/upload/fresh restoration, run
the bound validator with `--phase validate` on its designated `.9`
allocation. Only after its successful actual gates and cleanup/restoration,
the bound executor's `--phase launch` is eligible. Frozen plans install
no queue and change no allocation; `monitor_endeavor.sh` discovers saved
validation/execution/analysis states on the existing 900-second cadence.

`analyze_wave8_snapshot.py` separately captures running histories once
as bytes and rejects unfinished/malformed rows before writing an observation.
It uses original-source identity checks and the existing comparator on
immutable copies, without editing any active/frozen trainer, JSONL or
monitor. Completed-wave claims and final cloud import still require the
original release's complete evidence.

```bash
/opt/conda/envs/torch-base/bin/python tracking/analyze_wave8_snapshot.py \
  --root "$PVC" \
  --observation "$PVC/run-preparation/wave8-observations/NEW_UNIQUE_OBSERVATION"
```

The document ledger reconstructs saved strict comparisons separately
from the canonical analyzer output and withholds changed observations.
`monitor_wave8_observations.sh` is the separately acknowledged reporting-only
900-second Bash loop, PID3600017 at November17 04:14. It saves bound
process snapshots under `monitor-wave8-observations/` and strict captures
under `wave8-observations/`; failed or unfinished reads remain refusals.
It neither replaces the frozen Wave-8 training/upload monitor nor changes
allocations, and stops on the original bound observer's terminal state.
Do not start a duplicate instance; inspect its process and saved evidence.

`analyze_refresh.py` checks the same frozen Wave-7 source/configuration/data
identities and summarizes actual rank-zero installation records, cumulative
counter deltas and recorded basis-age/pending-count samples. It retains
unequal observed-prefix counts separately from complete shared
536,870,912-token windows. Installation age is snapshot age at installation,
not current basis age across all ranks. Missing telemetry is not inferred;
step times include startup and cannot isolate host-refresh waits. This
read-only diagnostic neither compares loss nor approves a launch.

```bash
/opt/conda/envs/torch-base/bin/python tracking/prepare_fixed_age_tuning.py \
  --root "$PVC" --out "$PVC/campaigns/lr-fixed-age-0p5m-20261116"
/opt/conda/envs/torch-base/bin/python tracking/analyze_fixed_age_tuning.py \
  --campaign "$PVC/campaigns/lr-fixed-age-0p5m-20261116/campaign.json" \
  --out "$PVC/campaigns/lr-fixed-age-0p5m-20261116/analysis.json"
/opt/conda/envs/torch-base/bin/python tracking/analyze_refresh.py \
  --root "$PVC" --out "$PVC/run-preparation/analysis-wave7-refresh.json"
```

For Wave-7 interim or completed broad-window comparisons, run the existing
frozen logging-release analyzer. It verifies the four executed identities;
final losses remain unavailable while the runs are incomplete.

If streaming live JSONL is transiently unreadable, preserve the prior
analysis and use `analyze_wave7_snapshot.py` for a separate strict
observation. It rejects unterminated or malformed rows without creating
a partial comparison, hashes all captured files and compares each arm's
single immutable observation. It accepts only running manifests; it is
not a completed-wave import or a change to the frozen completion monitor.

```bash
/opt/conda/envs/torch-base/bin/python \
  "$PVC/controllers/wave7-release-20261116/tracking/analyze_wave7_release.py" \
  --root "$PVC"
/opt/conda/envs/torch-base/bin/python tracking/analyze_wave7_snapshot.py \
  --root "$PVC" \
  --observation "$PVC/run-preparation/wave7-observations/NEW_OBSERVATION"
```

The prepared fixed-age LR study now has a separate frozen actual-config
validator at `campaigns/lr-fixed-age-0p5m-20261116/validation-helpers/`.
`run_fixed_age_validation.py --phase check` checks the bound controller
without CUDA work; `--phase validate` first requires completed,
five-window analyzed, full-resolution cloud-verified Wave-7 histories,
fresh exited-parent/restored-watcher SSH evidence and the original idle
A100 allocation. It preserves continuous24 versus22+resume24 for all
twelve actual LR configs, including fused capture/update checks. Worker
startup requires its exact guarded controller generation and child marker.
Thirty-six commands cost301,989,888 validation tokens, separate from
tuning tokens; no validation loss is a selection result. The binding
creates no job or queue. Separate training launch/monitor/upload helpers
are implemented and require their own execution freeze and actual gates.

```bash
/opt/conda/envs/torch-base/bin/python \
  "$PVC/campaigns/lr-fixed-age-0p5m-20261116/validation-helpers/run_fixed_age_validation.py" \
  --campaign "$PVC/campaigns/lr-fixed-age-0p5m-20261116/campaign.json" \
  --phase check
```

`run_fixed_age_tuning.py --phase freeze` preserves those execution helpers
with the 22 byte-identical already-frozen validation helpers. Its bound
`--phase check` performs no GPU work; freezing installs no worker,
monitor or queue. `--phase launch` requires complete five-window,
cloud-verified Wave 7, actual twelve-config GPU validation and successful
hash-bound cleanup/restoration. It starts one detached family worker per
original node and three sequential full 320-update down/base/up trials
per worker. The separate monitor polls every 900 seconds and imports
completed trials with distinct LR identities and the full scalar verifier.

```bash
campaign="$PVC/campaigns/lr-fixed-age-0p5m-20261116/campaign.json"
/opt/conda/envs/torch-base/bin/python tracking/run_fixed_age_tuning.py \
  --campaign "$campaign" --phase freeze
/opt/conda/envs/torch-base/bin/python \
  "$(dirname -- "$campaign")/execution-helpers/run_fixed_age_tuning.py" \
  --campaign "$campaign" --phase check
```

The campaign execution binding is frozen at November 17 00:24:49 and
its bound check passes; all 22 reused validation helpers are unchanged.
No worker, tuning monitor or trial has launched. Do not repeat a freeze
or launch with existing evidence. After reviewing completed Wave 7 and
passing actual GPU gates, the same bound helper can
run `--phase launch`. Inspect `launch-state.json`, `workers/*.json`,
`monitor/latest.json`, `monitor/monitor.log` and `swanlab-upload.json`.
The monitor exits successfully only after all workers are terminal with
restored keepers, the complete selection analysis succeeds and all trial
histories are verified. Unknown SSH state and live orphaned children
remain nonterminal; failed prefixes stay charged and malformed token
prefixes stay unknown. Workers own cleanup/restoration under node locks;
the central controller never restores over unknown remote workers.

Wave 7 subsequently completes and is full-resolution cloud-verified at
November 17 00:49:03, Asia/Shanghai. Its final analyzer records five
complete token windows and eleven exact-shared late evaluations.
The separately frozen actual-LR validator starts on `.9` at 00:52:34
under controller PID `3281345`; `validation-state.json` and
`validation-gpu.json` are `validating`. This is gate execution, not
tuning training or an LR selection result. Inspect
`validation-controller.log`, `validation.log` and per-command
`validation/*.log`. Bash process monitor `3282367` polls every 900 seconds
under `run-preparation/monitor-validation-lr-fixed-age-20261117/`;
the existing endeavor monitor records campaign receipts on the same cadence.
All twelve actual-config gates and controller cleanup/restoration must
pass before the bound tuning launch command is eligible.
Completed-trial upload validates one history independently of streaming
siblings and reads only `SWANLAB_API_KEY`; no completed histories means
no authentication.

Run one Bash monitor for a launched wave. It takes fresh SSH process
observations, saves the current manifest state and final complete JSONL
row, then sleeps 900 seconds. A timeout is unknown, not terminal.
`flock` prevents duplicate monitors for the same wave. On confirmed
completion it saves broad-window analyses, uploads, and verifies cloud
histories. A failed run exits with its evidence preserved and is never
automatically restarted.

```bash
bash tracking/monitor.sh "$PVC" wave4
```

Snapshots and the monitor PID are under
`$PVC/run-preparation/monitor-WAVE/`. Launch it under a detached process
manager for persistence beyond the current shell. Check the PID with
`ps`, not only the PID file. The exit condition is
`analysis_and_upload_verified_at`; inspect logs if it is absent.

```bash
/opt/conda/envs/torch-base/bin/python tracking/analyze_wave.py \
  --root "$PVC" --wave wave3
/opt/conda/envs/torch-base/bin/python tracking/compare_runs.py \
  --baseline wave3/muont_seed1 wave3/inner_outer_mlpdiag_seed1 \
  --window-tokens 536870912
```

The default is five 536,870,912-token windows per full run: 1024 updates
at 0.5M and 512 at 1M. `--window` also means tokens, not updates.
`--json PATH` saves the structured comparison. The comparator requires
equal seeds, held-out-window counts, corpus receipt, model and data
configuration for paired labels. `--descriptive` explicitly permits an
unpaired comparison, but changing seed changes selection holdouts and
absolute selection losses are then not a controlled comparison.

Reports use exact common-token evaluation records without interpolation;
late means span the final half of the common observed token range. Final
fields require complete equal-budget runs with endpoint evaluations.
Overlapping/gapped histories are rejected. Adjacent evaluations are not
independent seeds. Original process totals include startup, evaluation
and checkpoints; sums of resumed segments exclude downtime.

## Deferred Companion

`queue_seed2.sh` watches the wave-4 1M MuonT process on `.144` every
900 seconds, then invokes `seed2_companion.py` remotely with the task
root in `PYTHONPATH`. The queue receipt binds the exact configuration,
validation and deployed helper hashes. The follow-up refuses a live or
incomplete predecessor, unfinished keeper restoration, unrelated CUDA
occupancy, or changed bound evidence. The actual 0.5M seed-2 configuration
must pass its continuous-three versus two-plus-resume comparison before
launch; unchanged synthetic gates are reused only with unchanged source,
environment, data and original command evidence.

```bash
bash tracking/queue_seed2.sh "$PVC"
```

Run this detached only once, then check both its saved PID and `ps`.
Snapshots/logs are under `run-preparation/queue-seed2-companion/`.
On success, `wave4-baseline.json` records the eight-GPU `muont_seed2`
launch; `monitor-wave4-baseline/` tracks completion, paired seed-2
analysis, import and cloud verification on the credential-owning host.
Failure preserves evidence and requires inspection, not an automatic retry.

Before the companion launches, `benchmark_diagonal.py` runs an isolated
row-square probe on the released A100. This does not change training source.
It derives full accumulation row capacities and group counts from the
deployed configs and pinned sampler, validates BF16 inputs against FP32
production sums and an independent FP64 reference, and measures eager
CUDA-event timing and allocator memory. Probe failure is recorded separately
and does not suppress a correctly gated MuonT baseline.

```bash
/opt/conda/envs/torch-base/bin/python tracking/benchmark_diagonal.py --describe
/opt/conda/envs/torch-base/bin/python tracking/benchmark_diagonal.py \
  --out "$PVC/run-preparation/benchmark-diagonal-20261116.json"
```

Do not run the GPU command while training or an unrelated keeper occupies
the node. The geometry command does not initialize CUDA. A passing
microbenchmark is not a whole-training efficiency result; communication,
EMA updates, compiled capture and full training continuation need separate
validation before integration.

The three queue refusal tests use the Torch environment:

```bash
PYTHONPATH="$PWD" /opt/conda/envs/torch-base/bin/python \
  -m unittest discover -s tracking -p 'test_seed2_companion.py' -v
```

### Durable Endeavor Ledger

`monitor_endeavor.sh` refreshes the bounded automated ledger in `ENDEAVOR.md`
every 900 seconds. It reads saved launch, monitor, comparison and cloud
verification receipts; it does not run GPU jobs, import credentials or replace
the scientific interpretation outside its marked section. Observation times
are explicit, and incomplete comparisons cannot publish final loss fields.
The ledger continues across waves until its saved PID is explicitly stopped.
It reports saved refresh-diagnostic times and unequal prefix counts only
when the diagnostic's source-release hash matches the current binding.
Rendering a saved receipt is not a new live-history read.
The newest strict observation under `run-preparation/wave7-observations/`
is shown separately from the canonical completed-wave analysis. Its current
source-release hash, captured manifests/history hashes and reconstructed
comparison must match before any summary is printed. Changed or incomplete
evidence withholds that summary; a running capture cannot become a final
result or cloud-upload receipt.
Wave-9 uses the same strict byte-capture policy under
`run-preparation/wave9-observations/`. A streaming analyzer can race an
appending JSONL file; capture rejects unfinished, blank or malformed rows
without trimming them. The captured comparison reuses the complete shared
536,870,912-token windows and all original matched pairs, and the ledger
rederives its immutable hashes before publishing. Capture does not change
the frozen production helpers or authorize uploads.
The existing Bash document cycle now attempts a fresh strict capture while
all four manifests are running. Refused captures are logged, the previous
valid comparison remains available, and the next 900-second cycle retries.
Completed or uninitialized histories are not recast as running captures.
Receipt times, not filename ordering, select the latest captured comparison.
Rendering through `update()` remains saved-artifact-only; its CLI owns the
bounded read-only capture attempt and does not load credentials or start jobs.

```bash
CUDA_VISIBLE_DEVICES= /opt/conda/envs/torch-base/bin/python \
  tracking/analyze_upstream_snapshot.py --root "$PVC" \
  --observation "$PVC/run-preparation/wave9-observations/NEW_ID"
```

Use a new observation identity; never overwrite an existing capture.
The ceiling-LR campaign retains the original four fixed-age families,
two equal-budget higher-LR points, and a separately frozen one-shot queue.
Its queue waits for completed Wave-9 process, five-window analysis and
private cloud evidence before actual-config validation or training.
The immutable campaign's preparation status does not replace its later
queue installation/state receipts.
Fixed-age tuning execution is tracked separately from preparation:
the campaign, execution archive/helpers and validation-binding hashes
must match before saved worker/token or cloud counts are rendered.
Partial cloud receipts remain partial, and unknown token prefixes are
not shown as zero charges. Saved PIDs are not new process observations.
The fixed-age study also renders saved final selection losses, late
held-out deltas and final timers, plus complete shared training-window
deltas against same-LR MuonT. Incomplete/failed endpoints cannot publish
final losses or timers, missing windows are not filled, and mismatched LRs
are not paired. These saved comparisons do not select an incomplete family
sweep or establish a full-budget optimizer ranking.
Each saved trial's name/family/LR must match the campaign plan. The existing
analysis does not store late evaluation tokens/counts, so its aggregate
late-mean deltas are explicitly coverage-unverified, not paired held-out
estimates. Manual exact-token comparisons require separate history checks.
During actual-LR validation, provisional command/trial progress is shown
only when the frozen plan, campaign/helper/archive hashes, recorded
controller generation and completed command-log prefix match. It is
explicitly distinct from the final controller-bound receipt and cannot
authorize launch. Final checkpoint reconstruction, marked-child cleanup
and keeper restoration remain required.

```bash
nohup bash tracking/monitor_endeavor.sh "$PVC" \
  > "$PVC/run-preparation/monitor-endeavor.log" 2>&1 < /dev/null &
```

Run only one instance and confirm its PID/log under
`run-preparation/monitor-endeavor/`. An invalid receipt or concurrent document
edit fails visibly and preserves the original document.

The ledger distinguishes immutable candidate/queue bindings from recorded
execution events and queue exits. It includes bounded direction-diagnostic
state, keeper restoration and hash-matched results from
`validation/diagonal_diagnostic_*/`; diagnostic completion does not count as
GPU launch-gate approval. Manual interpretation remains outside its markers.
Wave-8's separate `wave8-launch-binding.json` appears as a full-budget
release binding; its existence does not mean training has launched.
Subsequent `wave8.json`, monitor snapshots, broad-window comparisons
and verified SwanLab receipts are shown through the recurring wave ledger.
Separately bound combined-product probes under
`validation/combined_outer_gpu_*/` are also tracked. Numerical results
require the controller's result hash plus owner/helper identity; failed
and partial results remain failures or partial evidence.

### Combined-Product GPU Probe

`run_combined_outer_gpu.py` freezes a new isolated diagnostic, preserving
the original actual-LR allocation helpers byte-for-byte. Run `freeze`
from the checkout and `check`/`run` only from its bound helper copy.
GPU execution is restricted to the original restored `.139` allocation,
with the existing node lock, exact keeper generation release, marked-child
cleanup and verified restoration. It never changes a frozen trainer.

```bash
probe="$PVC/validation/NEW_COMBINED_OUTER_GPU_IDENTITY"
/opt/conda/envs/torch-base/bin/python tracking/run_combined_outer_gpu.py \
  --root "$PVC" --out "$probe" --phase freeze
/opt/conda/envs/torch-base/bin/python "$probe/helpers/run_combined_outer_gpu.py" \
  --root "$PVC" --out "$probe" --phase check
ssh -o BatchMode=yes -o ConnectTimeout=10 root@28.216.17.139 \
  "/opt/conda/envs/torch-base/bin/python $probe/helpers/run_combined_outer_gpu.py \
  --root $PVC --out $probe --phase run"
```

Do not repeat a probe identity or overwrite its execution receipts.
`combined_outer_gpu_20261117_0118` completed November 17 at 01:17:27:
54 production-shape `tpu` parity cases and six eager/graph timing workloads
passed; controller cleanup and keeper restoration both exit zero.
The detailed result and interpretation are in `ENDEAVOR.md`.
This is synthetic single-A100 update evidence, not full-model continuation,
a loss result, training launch approval or whole-training efficiency.

### Isolated Wave-5 Validation

`diagnose_combined_outer.py` is a separate CPU-only post-NS arithmetic
diagnostic for the three-product outer-left update hypothesis. It loads
the frozen Wave-7 equations using the existing AST loader and preserves
the reference NS output unchanged. Its CPU `f32` fixture parity and
algebraic FLOP count are not production `tpu` parity, GPU timing,
full-model continuation or launch approval. No candidate or allocation
is changed. Preserve each receipt rather than overwriting a previous run.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/conda/envs/torch-base/bin/python tracking/diagnose_combined_outer.py \
  --root "$PVC" --out "$PVC/run-preparation/NEW_COMBINED_OUTER_CPU_RECEIPT.json"
```

`freeze_wave5.py` creates the self-contained source archive and
`run-preparation/wave5-candidate.json` binding after the final candidate CPU
receipt matches its source. The queued validation runs on `.9` only after its
wave-4 lagged run completes and its keeper watcher restores the allocation.
It checks candidate/configuration/pinned-source/data/controller identities
before releasing only the known keeper. Full training is not launched here.

```bash
/opt/conda/envs/torch-base/bin/python tracking/freeze_wave5.py \
  --root "$PVC" --candidate "$PVC/candidates/fused-diagonal-20261116"
nohup bash tracking/queue_wave5_validation.sh "$PVC" \
  "$PVC/candidates/fused-diagonal-20261116" \
  > "$PVC/run-preparation/queue-wave5-validation.log" 2>&1 < /dev/null &
```

The November 16 freeze and queue have preserved evidence; inspect their
receipts and logs instead of recreating them. The queued A100 attempt failed
`fused_diagonal_pipeline` direction parity at 19:38:46, then restored the
keeper and exited. It has not passed real-data continuation or launch gates.
Validation evidence is
`validation-wave5-gpu.{json,log}` and `wave5-validation-state.json` under
`run-preparation/`. A failed gate preserves evidence and requires inspection.
Do not change bound source/helper files while the queue is active.

### Direction Diagnosis

`run_diagonal_diagnostic.py` runs an isolated two-A100 replay on `.9` only
after its predecessor is complete/restored and no training is present.
It verifies frozen candidate/restoration-helper identities, archives its
own source, and releases only identified keeper launcher/rank generations.
The child and independently sessioned ranks share a unique environment
marker used for bounded timeout cleanup. Keeper restoration refuses any
remaining CUDA workload and fails the CLI on error.

The completed November 16 attempt is preserved under
`validation/diagonal_diagnostic_20261116_2024/`. Both ranks reproduced the
original failure and failed production-shaped `tpu` updates with structured
and Gaussian inputs; FP32 controls passed. See the result table and next
controlled experiment in `ENDEAVOR.md`. This is diagnosis, not training or
launch approval. Do not rerun into its existing evidence directory or
overwrite the frozen failure; a follow-up needs a new source/output identity.

### Wave-5 Launch Queue

`launch_wave5.py --bind-queue` archives and binds nine launch/tracking helpers.
`queue_wave5_launch.sh` waits for candidate GPU validation/restoration and
complete cloud-verified wave-4/seed-2-baseline histories. Its controller
recomputes all five token-window comparisons before transferring allocations,
checks each predecessor and only its identified keeper generation, then
launches the frozen source and installs `watch_wave5.py` watchers and the
completion monitor. Partial launches stay observable without complete-wave
analysis/import; recovery cannot start a keeper over live training.

The November 16 queue ran as PID 2834304 and exited at 19:48:59 with
`needs_attention` after the failed candidate gate. No wave-5 training launched.
Inspect
`run-preparation/wave5-launch-queue.json`, `queue-wave5-launch/pid` and
`queue-wave5-launch.log`; do not recreate it or edit its bound helpers.
`wave5-launch-state.json` and `wave5.json` will record actual transfer/launch.
The queue leaves credentials on the current host. Live recovery has not
been exercised; source review and CPU refusal tests are not GPU evidence.

### Equal-Budget LR Preparation

`prepare_tuning.py` writes three short-budget matrix-LR configs
per family into a new campaign directory, outside the frozen source tree.
It verifies source/configuration/data bindings and leaves auxiliary LR,
decay coefficients and the seed-0 reserved holdouts unchanged. It creates
no GPU launch or queue. The November 16 0.5M study already exists at
`campaigns/lr-0p5m-20261116/`; inspect rather than overwrite it.

```bash
/opt/conda/envs/torch-base/bin/python tracking/prepare_tuning.py \
  --root "$PVC" --source-receipt "$PVC/run-preparation/wave5-candidate.json" \
  --out "$PVC/campaigns/NEW_CAMPAIGN" --batch 0p5m
/opt/conda/envs/torch-base/bin/python tracking/analyze_tuning.py \
  --campaign "$PVC/campaigns/lr-0p5m-20261116/campaign.json" \
  --out "$PVC/campaigns/lr-0p5m-20261116/analysis.json"
```

Each trial uses 167,772,160 tokens with a completed 20%-stable/80%-decay
schedule, not `--max-steps`. Original histories belong in
`CAMPAIGN/runs/TRIAL_NAME/`. A family is ranked only after all three initial
trials complete, their identities match, and final rank checkpoints exist.
The analyzer reconstructs complete trial configs from frozen references and
allowed LR multipliers, enforces fixed budgets/seeds/holdouts, and compares
the full executed source inventory, including pinned Toner modules. Source
symlink aliases must resolve within the frozen candidate; changed or missing
upstream modules cannot produce winners even when every trial shares them.
Only final held-out mixture `selection_loss` chooses an LR; C4 and broad
training windows are reporting data. Failed attempts retain observed-token
charges and cannot win. A validated backend choice, actual short-config GPU
gates, launch/monitor/upload support and fresh allocation are still required
before this preparation becomes an executed tuning study.

### Upper-Boundary Handoff Queue

The separate eight-trial fixed-age LR expansion is under
`PVC/campaigns/lr-fixed-age-upper-boundary-0p5m-20261117/`. Its validation,
execution and queue releases are independent immutable archives. The
queue installed November17 at04:36:41 as PID3641454; do not install it
again. Inspect its binding, installation, state, controller intents and
`queue.log` without changing frozen helpers or existing evidence.

```bash
campaign="$PVC/campaigns/lr-fixed-age-upper-boundary-0p5m-20261117/campaign.json"
/opt/conda/envs/torch-base/bin/python \
  "$(dirname "$campaign")/queue/helpers/queue_lr_boundary.py" \
  --campaign "$campaign" --phase check
```

`queue_lr_boundary.sh` uses a900-second Bash cycle and a process lock.
While Wave8 is active or unknown, it only performs original bound SSH
observations. Completed five-window results and the original monitor's
full-resolution verified SwanLab receipt must be rederived before the
frozen validator can start. Fresh terminal/restoration checks are also
required by the validator and each family worker before any keeper release.
Only actual new-LR continuation/capture gates, owned controller evidence,
marked-child cleanup and successful restoration permit the frozen launcher.

Installer and controller spawn intents are written before launching.
An unacknowledged generation, failed validator or partial launch is never
retried. Workers retain ownership of their cleanup/restoration; the queue
cannot restore a keeper over unknown or live training. The recurring
`ENDEAVOR.md` ledger reports saved bound queue state, not fresh liveness.

### Wave-9 Production Follow-Up

`upstream_production.py` prepares an ownership-only successor of the
corrected full-model validator. The four arms are `muont`,
`outer_leftdiag_fused`, `outer_leftdiag_upstream_fused` and
`null_leftdiag_upstream_fused`, with original seed-0/base-LR configurations
and 5,120 updates per arm. Numerical sources and configurations remain
unchanged; CPU preparation does not authorize release or launch.

Before freezing, the controller requires the original corrected queue's
`validated` state, recorded exit, absent controller and independent
tensor/history reconstruction with exact restored keeper occupancy.
Freeze rechecks candidate/CPU/helper hashes after reconstruction and stages
artifacts before publishing the separate release. Inspect an issued release
using its frozen controller, never the mutable checkout:

```bash
helpers="$PVC/controllers/wave9-launch-20261117/tracking"
/opt/conda/envs/torch-base/bin/python "$helpers/upstream_production.py" \
  --root "$PVC" --phase check
```

Node mutations require exact eight-worker/eight-distinct-GPU keeper
occupancy under the existing node lock. Per-node start intents are one-shot;
unknown acknowledgment never authorizes retry. Worker-generation and marker
descendant checks distinguish unknown/live startup from verified terminal
absence. Unverified child cleanup blocks keeper restoration.

The frozen `monitor_upstream_run.sh` records snapshots every 900 seconds
under `PVC/run-preparation/monitor-wave9/`, analyzes shared
536,870,912-token windows, and uploads only completed full-resolution
histories after terminal cleanup/restoration evidence. Its verifier checks
every scalar series in private SwanLab project
`output-side-normalization-marin`. The independent document monitor keeps
`ENDEAVOR.md` current across waves; saved observations retain their own times.
