# Output-Side Normalization in MuonT: Endeavor

Updated on 2026-11-20, Asia/Shanghai. This is the authoritative record for
the research question, executed experiments, mathematical definitions,
results, and remaining experimental program.
Unqualified clock times in the November 16-20 continuation use Asia/Shanghai
(UTC+08:00). Current running status refers to the stated observation time.

## Navigation

This file is the scientific record. Process receipts, source identities,
checksums, controller and keeper generations, and the automated wave ledger
live in [`DETAILS.md`](DETAILS.md), preserved verbatim and in original order.

| To read about | Go to |
| --- | --- |
| What was asked and what it means | [Summary](#summary), [Research Questions and Evidence](#research-questions-and-evidence) |
| Where the endeavor stands now | [Current Experiment State](#current-experiment-state), [Program Status](#program-status), [Recurring Experiment Status](#recurring-experiment-status) |
| What has been concluded | [Conclusion](#conclusion), [Results and the Small Loss Gap](#results-and-the-small-loss-gap) |
| The executed setup | [Model](#model), [Data and Tokenization](#data-and-tokenization), [Batch and Training Budget](#batch-and-training-budget), [Optimizer and Schedule](#optimizer-and-schedule) |
| The update equation for each arm | [Executed Runs Mapped to Their Update Equations](#executed-runs-mapped-to-their-update-equations) |
| Per-wave results | The wave sections below, and [Remaining Experimental Program](#remaining-experimental-program) for the later waves |
| Timing and efficiency | [Compute Interpretation](#compute-interpretation), [Exploratory Elapsed-Time Crossings](#exploratory-elapsed-time-crossings) |
| An artifact, hash, PID or receipt | [`DETAILS.md`](DETAILS.md) |

## Current Experiment State

Evidence checkpoint, November 20, 2026 02:01 Asia/Shanghai.
Receipt and explicit live-observation times are shown below. Saved
receipts alone do not establish current GPU occupancy.
`PVC` denotes
`/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization`.

Latest status check: Wave12 remains the latest full-budget, large-window
analyzed and privately SwanLab-verified wave. The newer query diagnostic
failed at01:14:48 with6/54completed commands and2/18accepted MuonT controls;
the ten-step command was never spawned. Its controller, validator and
waiter are absent. Read-only process/GPU inspection at01:58 verifies the
same restored eight-rank keeper on the original eight A100s, not training.
The distinct streaming successor was prepared at01:55:47 with353source
files and210declared helpers; its17-case local suite passes. Fresh manifest
checking finds no source drift. Its archived201-test acceptance,
failed-precommand predecessor seal, frozen binding and GPU launch remain
pending. No new full-budget result, cloud import or optimizer superiority
is established. The sole original monitor publishes both journals at
01:57:54 from its01:53:27report and resumes its900-second sleep.
The task controller currently records the broader goal as paused; this
status check updates the journals without starting a new experiment or
altering the existing monitor. See
[Streaming Diagnostic Prepared](#streaming-diagnostic-prepared).
Earlier checkpoints below remain preserved historical observations.

The distinct missing-environment scan-exit diagnostic now has accepted
CPU/source coverage: 129 cases, 178 helpers and 547 inventory entries
at 20:56:19. A fresh archived-entry check passes at 20:58, and its
separate failed-predecessor seal passes at 20:59. It retains all prior
failures, the six accepted MuonT controls and fused five-step rejection.
Only ownership observation changes; numerical readers, tolerances,
training/configuration bytes, compilation and CUDA graphs are unchanged.
Independent seal checking passes, freeze completes at 21:02:50, and
the issued-controller check passes before launch. Controller
`3403676/942778625` acknowledges at 21:07:10. Validator
`3405014/942784407` acknowledges at 21:08:36 as its direct child,
after exact keeper release. The 22:03:52 aggregate admits eight of
54 commands and two of eighteen case decisions. The BF16/five-step
and BF16/seven-step MuonT lower-LR controls pass all five existing
acceptance gates; each has 39 passing scalar-loss comparisons and
no failed tensor comparison at
updates 22/23/24. First-resumed graph-state scalars retain their
existing diagnostic-only treatment; no tolerance or graph waiver
is added. Ten-step continuous and interrupted commands are completed.
Its resumed command fails before training at 22:03:52 in the complete
ownership scan: a previously confirmed marked child remains unstable
through three rechecks. Execution and cleanup exits are 1/1; no ten-step
numerical decision is established. Controller and validator stop
`needs_attention`; the controller records keeper restoration at 22:04:12.
The independent waiter also stops `needs_attention` at 22:12:06 because
the complete cleanup/terminal requirements are unavailable. Independent
22:12:53 read-only verification finds all three original owners, failed
launcher/ranks and retained child absent. Keeper3517296 and its exact eight
ranks exclusively occupy the same eight physical A100s. These observations
do not waive either failed cleanup exit or supply the missing 54-command/
18-decision terminal. See
[Missing-Environment Scan-Exit Successor](#missing-environment-scan-exit-successor).
The original journal monitor remains live with its exclusive lock;
its latest completed publication is 23:00:58 from the 22:56:51 report.
Its 900-second loop now includes the receipt-authenticated scan-exit
and terminal-waiter reports alongside preserved failed diagnostics.
A manual refresh completes at 21:18 with a 21:14:07 report timestamp:
the waiter projection passes, while diagnostic counts are withheld
because mutable evidence changes during inspection. The next cycle
retries; no stable count or numerical result is inferred from that refusal.
The waiter acknowledges at 21:11:58. Its fourth 21:57:06 observation
still finds the exact original owners live; this retained observation is
historical. Its 24-case archive and fresh archived-entry check pass.
The failed waiter is not restarted and no complete terminal is issued.
A separate actual-LR predecessor reader is CPU/archive accepted at
21:32:26: 34 cases and 193 helpers. Fresh archived checking passes;
the 21:33:45 read-only observation records `waiting_for_scan_exit_owners`.
Its numerical and execution archives remain separate, including their
conflicting handoff identities. No canonical seal, successor GPU
command or production authority is issued.
A distinct actual-LR adapter candidate is now CPU/archive accepted
at 21:53:00: 16 archived tests, 185 helpers and 559 inventory entries.
Fresh archived checking passes. All 54 numerical configurations and
162 commands are preserved; only the candidate owner and declared
scan-exit execution-helper transformation change. The separate
execution-owner archive accepts at 21:59:22 with 53 passing cases,
203 helpers and 577 inventory entries. Fresh archived checking passes
at 22:01. The distinct queue now pins that receipt; all 47 local queue
tests pass, including the five previously deferred archive tests.
Its separate CPU/source archive accepts at 22:05:07 with 47 passing cases,
214 helpers and 588 inventory entries; fresh archived checking passes.
The reporter pins that accepted receipt and is included in both automated
blocks. The queue remains unlaunched because the original complete terminal
is unavailable. No queue intent, canonical seal or successor GPU binding
is created. The final integrated journal/reporting suite passes 267 cases.
Wave 12 remains the latest full-budget, large-window analyzed and
privately SwanLab-verified wave. No optimizer superiority is established.

The distinct recheck-exit ownership correction passes all 55 local
tests: 41 exit/recheck regressions and 14 failure-preservation cases.
It resolves a marked process disappearing or becoming terminal during
the final recheck only after existing identity-bound whole-process exit
proof, then repeats the complete scan. Three scan attempts and the
one-second exit deadline remain unchanged. Numerical configurations,
readers, tolerances, compilation and CUDA graphs are preserved.
The distinct source archive accepts at 22:42:24 with all 136 cases,
183 helpers and 552 inventory entries. Fresh archived-entry checking
passes at 22:43. The inherited zombie fixture is separately derived
with complete stat evidence and observed directory disappearance;
the original fixture and failed archives remain unchanged. The separate
predecessor seal passes at 22:44:56 and independent seal checking
passes. Freeze and the fresh issued-controller check pass by 22:49.
The controller acknowledges at 22:51:58 as `3557212/943407368`.
Its direct validator `3558375/943415476` acknowledges at 22:53:19
after exact keeper release. The diagnostic fails at 22:59:49 after
1/54 completed commands and 0/18 case decisions. The MuonT BF16/
five-step lower-LR continuous command completes; the interrupted
command fails before training in `complete_scan_exhausted` with two
previously marked helpers unstable across three complete scans.
Execution and cleanup exits are 1/1. The controller records failure
and keeper restoration at 23:00:07. Independent 23:00:38 inspection
finds controller and validator absent, no validator-marked descendants,
and exact restored keeper3564728 with its eight original ranks on the
same eight physical A100s. The recheck-exit correction did not resolve
this execution failure; no case is numerically accepted. The separate read-only
waiter accepts all 24 archived cases and passes fresh archived checking.
It acknowledges at 22:54:23 as `3559219/943421918`; its first
22:54:29 observation finds both exact owners live, 0/54 completed
commands and 0/18 decisions. That observation is historical. Its
original next 900-second cycle stops `needs_attention` at 23:09:30
because the complete cleanup/restoration predicates are unsatisfied.
No terminal command is issued. Fresh 23:11:03 read-only checking
finds controller, validator and waiter absent, no validator-marked
descendants, and the same restored keeper on all eight original A100s.
The failed diagnostic and consumed waiter are not restarted.
The integrated journal/reporting suite passes 284 test executions
(264 distinct testcase identities); the 136-case acceptance archive
separately requires and passes unique case identities.

The CPU-only helper-churn experiment completes at 23:09:30 against
the exact issued ownership-helper bytes. With zero, one or two
successive marked-helper exits, the original three-scan protocol
returns every surviving and newly discovered marked process.
Churn continuing through all three scans instead produces
`complete_scan_exhausted` and returns no partial ownership result.
All five regression cases pass. This reproduces the refusal mechanism
in a fake process tree; it does not establish the live failure's cause.
A later command-digest comparison identifies both retained helpers as
matching the rank's compute-occupancy `nvidia-smi` query. This supports
testing concurrent ownership inspections, but does not prove live
causality. The startup-ordering proposal below is superseded by a
distinct serialized-ownership diagnostic. Its CPU/source archive
accepts 141 unique cases at 23:35:02; fresh archived-entry and separate
failed-predecessor seal checks pass. Its 30-second canonical lock
encloses the entire unchanged ownership inspection. Trainer/configuration
bytes, numerical readers, tolerances, three complete scans, the
one-second exit proof, compilation and CUDA graphs remain unchanged.
Freeze completes November 19 at 23:46:29, and fresh issued checking
passes. Controller3595881 acknowledges at 23:51:10; its direct
validator3597661 acknowledges at 23:52:56 after exact keeper release.
The first continuous command fails before training at 23:54:13 when
waiting ranks exhaust the new 30-second inspection lock. The retained
result is 0/54 completed commands and 0/18 decisions, with execution/
cleanup exits1/1. Controller restoration is recorded at 23:54:30.
Independent 23:56:15 checking finds both owners absent, no validator
markers, and the exact restored keeper3598689/ranks3598829-3598836
exclusively on the original eight A100s. No numerical acceptance or
complete terminal follows. The separately accepted24-case read-only
waiter3598600 acknowledges at 23:54:12 and retains its23:54:17
historical owner observation; its original900-second schedule remains.
No failed controller or waiter is restarted. See
[Serialized Ownership Diagnostic](#serialized-ownership-diagnostic).
The sole original Bash monitor publishes both journals November20
at00:00:10 from its November19 23:56:01 report, then resumes `sleep 900`.
All259 affected CPU cases pass. Whole-inspection serialization fails
within its unchanged30-second lock bound; a smaller query/marker-scan
critical section is now separately CPU/source accepted at00:11:29 with
142unique cases,193helpers562inventory entries. Fresh archived checking
passes. The original waiter stops00:09:18 on its scheduled interval;
fresh00:10:16 checking confirms controller, validator and waiter absent,
no validator markers and the same restored keeper. Separate predecessor
sealing passes at00:13:48 with2746evidence files and1352checkpoint
identities; independent archived seal checking passes. Freeze publishes
at00:18:47, and the fresh issued-controller check passes. The distinct
controller3633332/start943964742 acknowledges at00:24:51; direct
validator3637693/start943978129 acknowledges at00:27:05 after exact
keeper release. The first MuonT BF16/five-step lower-LR continuous
command finishes24updates with zero execution/cleanup exits at00:30:14.
The interrupted22-update command also finishes with zero exits at
00:35:15. The resumed command finishes at00:38:20 with zero exits;
the00:40:46aggregate admits3/54commands and0/18decisions. That saved
observation is superseded by the00:56:09aggregate:4/54commands and
1/18decisions. The first five-step MuonT case passes all five unchanged
gates, checkpoints22/24 and39/39scalar-loss comparisons. Checkpoint23
retains16diagnostic-only graph-state scalar differences and no failed
tensor comparisons. The seven-step continuous command finishes cleanly
at00:53:40; its complete case decision remains pending. The interrupted
22-update command also finishes cleanly at00:58:38; aggregate00:59:29
admits5/54commands and1/18decisions. The resumed24-update command
finishes01:01:43 with integer execution/cleanup0/0 and marked descendants
absent. Aggregate01:04:16 admits6/54commands and1/18decisions, `validating`,
with no error. This observation is superseded by the01:14:48failed aggregate:
6/54commands and2/18decisions. Both five-step and seven-step MuonT lower-LR
controls pass all five unchanged continuation gates and39/39loss checks.
The validator fails before launching the ten-step command in
`complete_scan_exhausted`. None of six retained unstable PIDs was previously
confirmed marked. Controller and validator exit with recorded execution/
cleanup1/1; keeper restoration records0 at01:15:06. Independent01:16:42
read-only checking finds both owners absent, no validator markers, and exact
restored keeper3731474 with ranks3731705-3731712 on the original eight A100s.
The original waiter stops `needs_attention` at01:28:01 on its scheduled
900-second cycle, retaining four observations and no complete terminal.
Independent01:30:59 read-only checking finds controller, validator, waiter
and all six completed launcher handles absent, no validator markers, and
the same exact restored keeper on all eight original A100s. The failed
execution/cleanup exits remain1/1; absence does not supply missing coverage.
No GPU successor authority exists, and no failed one-shot is restarted.
The separate read-only waiter3638945/
start943982783 acknowledges at00:27:52 after24-case archive acceptance
and fresh archived checking. Its first00:27:57 observation verifies
both exact owners live,0/54completed commands and0/18decisions.
Its third00:57:59observation still waits for the original owners.
The original monitor publishes both journals at00:58:51 from its
00:54:14 report and returns to its900-second sleep. All242unique
integrated journal cases pass; source/content lint and diff-check pass.
A separate query-serialized actual-LR predecessor reader now accepts
39unique CPU/archive cases at00:55:08 with211authenticated helpers.
Fresh archived-entry checking passes; its00:57:24read-only observation
finds the exact controller, validator and waiter live and returns
`ready: false`. No seal, GPU handoff or successor launch is issued.
The distinct actual-LR adapter candidate is prepared at01:12:32 with187
helpers, preserving54configurations/162commands and the original5120
schedule. Three families retain all three actual LRs, BF16/FP32 and5/7/10
iterations. Only the candidate owner changes; the conflicting execution
handoff is explicitly transformed to the accepted query helper, with its
original numerical bytes retained in the planned archive inventory.
Its16-case local CPU suite passes in393.58seconds. Separate archived
acceptance at01:21:54 passes16unique cases in381.99seconds, with187helpers
and561inventory entries. Fresh archived-entry checking passes at01:23.
The distinct unmarked-process churn CPU study reproduces three-scan
exhaustion using exact issued scanner bytes; six new regressions and the
five earlier marked-churn cases pass. The distinct, unissued streaming
observation-order candidate subsequently passes46local scanner cases:
all36pinned handoff cases plus10ordering/lifecycle cases. Reading each
process environment immediately after its identity avoids the reproduced
unmarked exit race; continuous marked churn, PID reuse, missing files,
live marker loss and surviving threads still refuse. All57combined
scanner/churn cases and5focused journal cases pass. This is CPU mechanism
evidence, not live causal proof or archive/GPU acceptance. Complete scan
body/rechecks, namespace filters, three attempts and one-second exit proof
remain unchanged; accepted issued scanners are untouched. No separate
execution owner, queue, seal or launch is inferred. The original monitor
publishes both journals at01:38:23 from its01:34:01report and returns to
`sleep900`. See [Streaming Observation-Order Study](#streaming-observation-order-study).
No new full-budget loss result, SwanLab import or optimizer superiority exists.

The following 20:35 and earlier checkpoints are retained observations.
The mixed-exit-bracket diagnostic stops `needs_attention` at19:59:22
with23/54 completed commands and7/18 case decisions: six accepted
MuonT controls and the preserved fused BF16/five-step rejection.
The next fused BF16/seven-step resumed command fails before training
in a complete ownership-marker scan. Its execution/cleanup exits
are1/1, so it is excluded from completed trajectories and supplies
no seven-step numerical decision. This is an execution failure;
no source, command or log substitution is established by the
validator's generic identity-error message.
Independent20:06:15 read-only checking verifies controller3118453,
validator3118822 and all issued training-command handles absent,
with no validator or command markers. Restored keeper3313700 and
its eight ranks exclusively occupy the same eight physical A100s
with exact recorded generations. The terminal waiter stops
`needs_attention` at20:04:58; the actual-LR queue stops
`needs_attention` at20:05:59 with0/6 phases because the required
complete predecessor terminal is absent. Neither failed one-shot
is restarted. See [Transition Partial Failure And Restoration](#transition-partial-failure-and-restoration).
The original Bash journal monitor1588227 remains live, retains its
exclusive lock and900-second sleep between refreshes. Its last
completed publication is20:30:47, using its20:27:24 report observation.
Both automated blocks now retain the failed diagnostic and stopped
waiters; manual failure and review milestones remain preserved.
Wave12 remains
the latest full-budget, large-window analyzed and privately
SwanLab-verified production wave. No new production wave or cloud
import is inferred, and the optimizer-improvement objective remains open.

The distinct precision-placement execution archive
`PVC/run-preparation/ns-precision-placement-execution-cpu/closure-reviewed-20261119T2032/receipt.json`
records CPU/source acceptance at 20:31:01 with 49 staged cases,
163 helpers and 531 inventory entries. A fresh archived-entry check
passes by 20:32; the broader CPU numerical suite passes 113 cases
in 165.57 seconds. Missing transitive imports are now required even
when their sibling files do not exist, including same-package
relative imports; declared external modules remain separate.
All 158 CPU-study parent helpers and the earlier 43/46-case archives
remain unchanged. This closes the reported missing-local-import gap
for static helper imports, not dynamic-import authority, CUDA numerics
or live ownership. No precision CUDA owner or launch exists. See
[Precision Import-Closure Correction](#precision-import-closure-correction).

The following19:54 and earlier checkpoints are retained observations,
not current liveness claims.
At 19:50:23 the transition aggregate retains 21/54 completed commands
and 7/18 case decisions: six accepted and one rejected.
MuonT BF16 and FP32 with five, seven and
ten Newton-Schulz iterations each pass all five continuation gates
and 39/39 scalar-loss comparisons at matrix LR 0.04525483392.
The first fused BF16/five-step case is rejected: boundary22 and
captured-row/update-work pass, but post-cached-root24 and scalar-loss
continuation fail.419/2,199update24tensor comparisons fail across
all eight ranks; three of39scalar-loss comparisons fail.
See [Fused Five-Step Continuation Rejection](#fused-five-step-continuation-rejection).
All other fused and matched-null decisions remain pending. Fresh
19:47:28 process inspection finds both original owners, the original
terminal waiter, reviewed actual-LR queue waiter and sole Bash monitor
live. The queue's 19:35:14 observation remains
`waiting_for_transition_owners`, with zero of six handoff phases.
The monitor publishes both journals at 19:34:05 and its next
scheduled updater is verified live at19:52. It keeps its
900-second sleep between refreshes. No complete transition terminal,
actual-LR GPU launch or production result follows from partial coverage.

A new CPU-only precision-placement screen completes at19:41:16: eighteen
explicit five/seven/ten-step schedules compare matched one/two/three
FP32 prefix versus suffix iterations against homogeneous controls.
The reviewed source archive passes64CPUcases and fresh archived-entry
checking. Independent retained verification passes at19:42:28,
covering all18trials,36exact finite same-input repeats and exact
reproduction of all six homogeneous controls. Every C/R pair remains
rejected at the original tolerance. Suffix precision lowers pair error
versus matched prefix precision in all six comparisons, but no mixed
output passes its own FP64 finite-polynomial reference. The five-step
two-BF16/three-FP32 suffix is retained only as an exploratory endpoint
placement. A source audit's second-read race in the unlaunched
compiler-localization loader is corrected in this distinct archive.
Accepted parent archives and running GPU sources are unchanged.
See [NS Precision-Placement Screen](#ns-precision-placement-screen).
Wave 12 remains the latest full-budget, large-window analyzed and
privately SwanLab-verified production wave. Better quality and
efficiency than independently tuned MuonT remain unestablished.

The following 19:11 and earlier checkpoints are retained observations.
At 19:08:46 the transition aggregate retains 16/54 completed commands
and 5/18 accepted decisions: MuonT BF16 with five, seven and ten
Newton-Schulz iterations and MuonT FP32 with five and seven iterations.
Each passes all five continuation gates and 39/39 scalar-loss
comparisons. FP32/ten-iteration continuous24 completes with
execution/cleanup exits 0/0; its interrupted/resumed trajectories and
case comparison remain pending. Read-only process checks at
19:11 verify the controller, direct validator, independent terminal
waiter, actual-LR queue waiter and sole Bash journal monitor live.
The monitor publishes both
journals at 18:56:46 and continues its 900-second interval.

The distinct actual-LR execution archive remains accepted with
53 passing cases and 191 helpers. A 198-helper queue archive passes
42 cases at 18:47:34 and independent archived checking by 18:50,
but independent review withholds its waiter launch because terminal
admission omits controller-marked descendants and observer timeout
cleanup omits nested readers. Queue-only corrections pass all
47 local cases in 36.33 seconds. Their separate complete CPU archive
is accepted at 18:58:03 with 47 cases in 36.51 seconds, 198 helpers
and 572 inventory entries. Independent archived-entry checking and
source/archive re-review pass. Its hidden-CPU waiter acknowledges
at 19:04:03 as `3263097/942039903`. Its first saved observation
at 19:04:54 finds the original controller, validator and terminal
waiter exact/live, records `waiting_for_transition_owners`, and
retains zero of six handoff phases. The accepted
archives and failed attempts are preserved. The transition terminal,
actual-LR freeze, keeper transfer and successor GPU launch remain
unissued. Wave 12 remains the latest full-budget,
large-window analyzed and privately SwanLab-verified production wave.
See [Transition Queue Review Milestone](#transition-queue-review-milestone).

The following 18:20 and earlier checkpoints are retained observations.
The transition aggregate at18:18:59 has9/54 complete commands and
3/18 accepted decisions: MuonT BF16 with five/seven/ten NS iterations
passes all five required gates and39/39 scalar-loss comparisons per
case. Checkpoints22/24 pass; checkpoint23 remains diagnostic only.
Fresh18:15 process checks verify the original controller, validator,
independent waiter and sole journal monitor live. No complete terminal
or production acceptance follows from these partial diagnostic results.

The distinct transition actual-LR adapter passes complete archived
acceptance at18:15:56:15 passing cases,179 helpers and553 inventory
entries; independent archived-entry checking passes. It retains all
54 configurations/162 bounded commands and changes only the candidate
owner adapter plus the separately reviewed execution handoff helper,
whose original bytes are retained. The external terminal archive is
not merged. The separately implemented191-helper execution-owner
draft passes45 lifecycle cases and required source lint; its complete
source/ownership suite remains running, so no owner acceptance,
freeze, keeper transfer or GPU launch is claimed. Wave12 remains the
latest analyzed and privately SwanLab-verified production wave. See
[Transition Actual-LR Adapter Acceptance](#transition-actual-lr-adapter-acceptance).

The transition aggregate at 17:46:13 records 6/54 completed commands
and 1/18 case decisions. MuonT BF16/five-iteration passes all five
required gates and 39/39 scalar-loss comparisons at matrix LR
0.04525483392; checkpoints22/24 pass, while checkpoint23 remains
diagnostic only. Both MuonT five/seven-iteration triples have zero
execution/cleanup exits; the seven-iteration comparison is pending.
A fresh 17:47:40 file-only probe verifies controller
`3118453/941361276`, validator `3118822/941365079` and waiter
`3127027/941414079` exact/live and refuses successor readiness.
The original single Bash monitor refreshed both journals at17:42:20.

A distinct transition-to-actual-LR predecessor archive is accepted
at17:47:35 with29 passing cases and184 authenticated helpers;
independent archived-entry checking passes. It retains all175 issued
waiter helpers and keeps the numerical archive separate. Its saved
readiness observation is false; no predecessor seal or successor GPU
owner is issued. Two earlier artifact-authentication failures remain
preserved. Wave12 remains the latest complete, large-window analyzed
and privately SwanLab-verified production wave; no new production
result or cloud import is inferred. See
[Transition Actual-LR Handoff Acceptance](#transition-actual-lr-handoff-acceptance).

The distinct mixed-exit-bracket successor passes archived CPU/source
acceptance at 17:02:50: 129 unique passing cases, 173 helpers and
542 inventory entries. Its archived-entry check and required source lint
pass. Only the measured ownership scan and execution namespace change;
original training sources, 18 cases/54 commands, lower LR, full
5,120-update schedule, numerical readers and tolerances remain unchanged.
The separate preservation seal passes archived checking; issued freeze
at 17:08:05 binds 2,814 authenticated file identities. The issued check
passes and launch acknowledges at 17:10:57 as `3118453/941361276`.
The controller records complete keeper release and direct validator
`3118822/941365079` acknowledged at 17:11:58. Fresh 17:14 process checks
verify both exact owners and the sole journal monitor live; eight
diagnostic ranks occupy the original GPUs. Initial progress is 0/54
commands and 0/18 decisions, not numerical acceptance. Wave 12 remains the
latest complete, large-window analyzed and privately SwanLab-verified
production wave. See
[Mixed-Exit-Bracket Successor](#mixed-exit-bracket-successor).

The transition aggregate advances to 2/54 commands and 0/18 decisions
at 17:19:42: MuonT five-iteration continuous24 and interrupted22 both
record execution/cleanup exits 0/0. The resumed trajectory and first
case comparison remain pending. Its separate terminal-waiter CPU
archive passes 24 cases at 17:18:42 and archived checking; the waiter
acknowledges at 17:19:45 as `3127027/941414079`. The first immutable
17:19:49 observation records both exact original owners live and
2/54-command, 0/18-decision progress. It checks every 900 seconds
and has no keeper or new-GPU authority. The final journal suite passes
233 cases; source lint passes. No independent complete terminal,
new production wave or cloud import is recorded.

At 17:23:56 the transition aggregate records all three first-case
commands complete, with execution/cleanup exits 0/0 and no marked
descendants. Coverage is 3/54 commands and zero case decisions;
the independent checkpoint/loss comparison remains pending. Fresh
17:24 owner checking verifies the same controller, validator, waiter
and monitor generations live. The scheduled journal refresh completes
at 17:23:46 using its 17:20:13 receipt observation; it predates the
newest completed-command aggregate.

The exit-drain diagnostic stops `needs_attention` at 16:33:03 after
6/54 completed commands and 2/18 accepted MuonT decisions. The
BF16/seven-iteration case passes all five required gates and 39/39
scalar-loss comparisons. The next ten-iteration continuous command
fails before training: its process-exit check rejects a bracketed
`S`-to-`Z` transition, with both PF_EXITING flags set, zero virtual
memory and empty command/environment files. This is an execution
failure, not a ten-iteration numerical rejection.
The controller records execution/cleanup exits 1/1 and keeper
restoration exit 0 at 16:33:20. Read-only checks at 16:38 verify both
owners absent, no validator-marked descendants, and the exact restored
nine-process keeper occupying all eight original GPUs. The waiter and
sole journal monitor remain live; the waiter's 16:25 saved live-owner
observation predates this failure. The waiter subsequently stops
`needs_attention` at 16:40:32 and is absent at 16:41:24; the sole
journal monitor remains live. No complete terminal, production wave
or SwanLab import is recorded. See the
[16:39 failure milestone](#exit-drain-partial-failure-milestone).

The distinct exit-drain terminal waiter acknowledges at 16:25:28 as
`3071163/941088367`. Its first retained observation at 16:25:32 records
the exact original controller and direct validator live, with 6/54
completed commands and 1/18 case decisions. A separate 16:32:24 process
check verifies those same owners, the waiter and the sole journal monitor
live. The seven-iteration MuonT comparison is pending at that observation; no complete
diagnostic terminal, new production wave or SwanLab import is recorded.
Wave 12 remains the latest completed, analyzed and cloud-verified
production wave. See the
[16:33 waiter launch milestone](#exit-drain-waiter-launch-milestone).

Seventeen focused reporting/split-document/monitor tests pass, followed
by all 228 tests in the full journal suite. The scheduled Bash refresh
finishes at 16:28:45. It loaded the waiter reporter before acknowledgment
pinning and therefore preserves a withheld-evidence notice; the current
reporter independently authenticates the acknowledgment and observation.
The existing monitor continues its 900-second sleep between refreshes.

The final descriptor-bound exit-drain archive passes125 CPU cases in
70.94seconds and is accepted15:41:56. A separate preservation seal
at15:42:41 retains both consumed failures. The issued source freezes
15:44:16 and its distinct diagnostic launch acknowledges15:47:40.
Fresh15:51 process checks confirm the exact controller and validator
generations live. The first MuonT continuous24 and interrupted22
trajectories have zero execution/cleanup exits. The authenticated
aggregate at16:22:30 records6/54 completed commands and1/18 decisions.
The first MuonT BF16/five-iteration case passes all five required gates,
including boundary22/post-root24 all-rank continuation and39/39 scalar
loss comparisons. First-resumed checkpoint23 retains sixteen graph-state
flag differences and remains diagnostic only. The full diagnostic and
independent terminal are unfinished.
A retained synthetic Linux probe at15:26:56 confirms process-pidfd
unreadability while a zombie leader has a live thread and through
nonleader exec, then readability after whole-process exit. This does not
attribute the recovery failure or establish optimizer quality. See
[Whole-Process Exit-Drain Successor](#whole-process-exit-drain-successor).
The same Bash monitor publishes the reviewed diagnostic block in both
journals at16:10:23 and resumes its900-second sleep. No second monitor
is launched.

The separate ownership-failure attribution experiment passes71 CPU
cases, including the unchanged recovery acceptance checks. Its derived
scan also passes all36 inherited ownership cases; four split-journal/
monitor tests pass. It changes only rejection evidence, recording exact
identity fields, command hashes and marker presence without raw arguments
or environment values. The accepted recovery source remains unchanged;
there is no successor acceptance or GPU launch. The failed GPU PIDs'
transitions remain unattributed. See
[CPU Ownership Failure Attribution](#cpu-ownership-failure-attribution).
The scheduled single Bash monitor publishes both journals at14:57:38
and resumes its900-second sleep. Fresh14:54-14:55 checks find the recovery
owners absent and the same eight restored keeper contexts occupying the
original GPUs. At that observation no training diagnostic or production
wave is running; the separately accepted exit-drain launch above is later.

The distinct marker-exit recovery launches14:24:21 and fails14:33:11.
It retains two successful commands: MuonT with five Newton-Schulz
iterations, continuous through update24 and interrupted at22. Its
resumed24 command fails before training on marked-process generation/
environment changes for PIDs2873160/2873161. No case decision is retained.
The recovery controller records execution/cleanup exits1/1 and keeper
restoration0 at14:33:29. Read-only checks at14:35 find both recovery
owners absent, no validator-marked descendants and the exact restored
eight-GPU keeper healthy. The accepted source and all failure evidence
remain unchanged; this one-shot recovery will not be restarted.
Its103 archived CPU cases and fresh archive/seal/issued-owner checks
passed before launch. It binds160 helpers,529 inventory entries and353
candidate source/config entries. The original failed attempt and both
failed follow-ups remain separately preserved.
See [Distinct Marker-Exit Recovery](#distinct-marker-exit-recovery).

The November 19 13:22:55 receipt retains 44/54 commands and 14/18
decisions: six accepted MuonT cases, six rejected fused cases and
two rejected matched-null cases. Execution stops `needs_attention`;
the controller records failure and keeper restoration at13:23:14.
Fused FP32/ten iterations rejects checkpoint24 continuation with
378 failed tensor comparisons across eight ranks and two of39 failed
loss comparisons. Boundary22 and captured-row/update/work checks pass.
All tested fused BF16/FP32 five/seven/ten-iteration cases now reject
the required post-root and scalar-loss gates. All three trajectories
for `null_leftdiag_fused_ns_tpu_q5_full_lr_lower` are retained:
continuous to update24, interrupted at22, then resumed to24. Its
update24 continuation rejects with409 failed tensor comparisons
and2/39 failed losses; its null-identity outer check passes.
All three BF16/seven-iteration matched-null trajectories are now retained:
continuous24, interrupted22 and resumed24, with zero command and cleanup
exits. Its update24 continuation rejects with448 failed tensors, seven
scalar/structure differences and1/39 failed losses; the null-identity
outer and captured-row/update/work checks pass. The BF16/ten-iteration
matched-null continuous24 and interrupted22 trajectories are retained
with zero command/cleanup exits. Its resumed command exits1 before
training on `Marked process generation changed during handoff`;
no resumed trajectory or numerical decision is retained.
Three FP32 matched-null cases have no retained commands or decisions yet.
Ten planned commands remain without retained successful trajectories;
one has a failed execution sidecar. Independent complete terminal
reconstruction cannot be inferred from this incomplete attempt.
Exact process checks at13:37 find the original controller/validator,
terminal waiter and actual-LR queue absent; the sole Bash document
monitor remains live. The terminal waiter records `needs_attention`
at13:32:52 and issues no canonical terminal. The actual-LR queue
records `needs_attention` at13:36:09 because its complete original
terminal/report is unavailable, with zero of six phases issued.
Their retained13:17:51/13:20:51 live-owner observations predate these
failures and must not be presented as current liveness.
Read-only keeper checks at13:26 and13:28 match all recorded generations,
original devices and eight occupied GPU contexts; a fresh13:40 device
query still shows the same eight keeper-rank contexts.
The accepted-archive read-only localization-handoff observation at
13:26:33 returns `ready: false`: original controller/validator absent,
terminal waiter and actual-LR queue still live. It issues neither a
terminal nor a GPU owner.
Original terminal reconstruction and successor predecessor sealing are
unsatisfied on this failed attempt; GPU issuance remains unissued.
CPU lifecycle regressions reproduce the marker-enumeration exit race.
A local three-attempt complete-rescan correction passes36 ownership
tests, including a real Linux child exit, persistent churn, PID reuse,
changed commands/ancestry and an unauthorized worker discovered during
rescan. The original failing PID remains unidentified. This correction
was initially tested locally, then separately archived and GPU-tested in
the distinct recovery above. That recovery also fails on marked-process
transitions. The original issued archive and failed evidence are unchanged.
See [Failed Follow-Ups and Marker-Exit Correction](#failed-follow-ups-and-marker-exit-correction).
The complete436-test selected caller rerun passes in294.80seconds;
an initial430-pass/two-failure real-process scan attempt remains recorded
as residual lifecycle evidence. Five focused journal/monitor tests pass.
Wave12 remains the latest completed, analyzed and cloud-verified
production wave: its four complete full-resolution histories have a
verified private SwanLab receipt. The compiled-map localization
numerical component's12:00:12 archive remains unchanged.
A distinct file-backed execution/
reconstruction component is accepted at12:27:43 with23 passing staged
CPU tests,155 helpers and523 inventory entries. Fresh archived-entry
verification passes by12:29; all152 parent helpers remain unchanged.
Each complete phase is stored separately, with streaming historical,
source-pair, repeat, phase and cross-process comparisons. Its allocation
owner and CUDA numerical results remain unissued. All90 integrated
localization/execution/replay CPU tests pass in144.10seconds. The separate
predecessor handoff is CPU/source-archive accepted at13:20:07:
25 staged tests pass in258.53seconds, binding191 helpers and559
inventory entries; fresh archived-entry verification passes by13:21.
Its allocation owner remains unimplemented and unissued.
No optimizer is established
as better than independently tuned MuonT in whole-training quality
and efficiency.
The scheduled split-journal refresh completes at14:39:35 without
restart. The same Bash monitor then sleeps900seconds. Five focused
split-document/monitor tests pass. The existing journal-size limitation remains recorded.
The timestamped history below remains preserved.

Read-only process checks on November 19 find the original document monitor
`3776115/919963640` absent and a single externally started replacement
`1588227/933927668` live, holding the same monitor lock.
Its preceding scheduled split-document refresh succeeds November 19 08:24:02, including
a read-only rational-continuation process/device observation captured at
08:19:11 and six authenticated training-path command completions.
An independent read-only check at 08:28:31 verifies the same original
training-path controller `2099685/937961924`, direct validator
`2100106/937966294` and single monitor `1588227/933927668` live.
At that check the growing diagnostic receipt still contains six completed commands and
one accepted case; its latest publication is 08:19:46. The seven-iteration
case decision and one-shot terminal are then unrecorded. The validator
subsequently accepts the seven-iteration case at 08:29:57; the ten-iteration
continuous command exits zero at 08:32:15 and its retained trajectory is
recorded at 08:34:43. The fresh 08:35:36 receipt check finds 7/54 completed
commands, 2/18 accepted cases and the ten-iteration interrupted22 command
selected next. A guarded split-document publication completes by 08:35:22,
preserving both manual-prose hashes and reporting two accepted cases.
No independent terminal or restoration is recorded. Live process
generations are observations, not numerical acceptance or GPU occupancy.
The next scheduled refresh succeeds at 08:44:00. By 08:42:07 all nine
BF16 MuonT commands have complete retained trajectories. The ten-iteration
case is accepted at 08:52:24, bringing the diagnostic to 9/54 commands
and 3/18 accepted cases. A separately CPU/archive-accepted read-only
terminal waiter acknowledges at 08:47:34 as `2189311/938341032` and
records the exact original controller/validator live at 08:47:38.
The waiter checks every 900 seconds and can invoke only the issued
canonical one-shot verifier after original-owner absence and complete
cleanup/restoration evidence. It provides no training or production authority.
FP32 MuonT/five-iteration continuous24 exits/cleans up zero at 08:54:44;
its trajectory is retained at 08:57:11, making 10/54 completed commands.
That fourth case has no numerical decision yet. Guarded publication
completes by 08:59:00, reporting three accepted cases and the authenticated
saved waiter state while preserving both manual-prose hashes.
The FP32/five-iteration interrupted22 command exits/cleans up zero at
08:59:21, and its trajectory enters the aggregate at 09:00:10.
The resumed24 command exits/cleans up zero at 09:02:01; its trajectory
is retained at 09:04:31, bringing the diagnostic to 12/54 commands
and 3/18 validator-accepted cases. The FP32 case decision is pending.
At 09:04:06, exact original controller, validator, waiter and monitor
generations remain live. The waiter's second immutable observation at
09:02:39 records the original controller/validator live; no terminal
claim or independent restoration is recorded. The scheduled document
refresh refuses concurrent edits at 09:03:54 and retains both documents;
the same Bash monitor will retry after its next 900-second sleep.
A read-only source/configuration audit at 09:04:38 checks all six pinned
training files and all 54 existing actual-LR configurations. This supplies
successor preparation evidence only; no distinct actual-LR owner or
binding is issued. See [Training-Path Actual-LR Successor Specification](#training-path-actual-lr-successor-specification).
The guarded split-document refresh completes at 09:12:04 and verifies
both manual-prose hashes unchanged. Its recurring block records all
twelve commands, three accepted cases and the waiter's second observation.
Five focused split-document/monitor tests, scoped Markdown lint and
Bash syntax checks pass. At 09:12:32 all four exact original process
generations remain live; the FP32 numerical decision is still pending.
The validator subsequently accepts FP32 MuonT/five NS iterations at
09:14:48. Checkpoints22/24 have 2,121/2,173 exactly equal tensor
comparisons and no scalar/structure differences; all 39 loss comparisons
pass. Update23 retains sixteen graph-readiness flag differences and
remains diagnostic-only. At 09:26:50 the growing receipt contains
15/54 retained commands and 4/18 validator-accepted cases. All three
FP32/seven-iteration commands have exited/cleaned up zero; that case's
numerical decision is pending. The waiter's third immutable observation
at 09:17:40 finds both original owners live. Fresh `/proc` checks at
09:24:14 verify unchanged controller, validator, waiter and monitor
generations. The existing Bash monitor refreshes both journals at
09:23:48 without a restart.
The distinct actual-LR numerical component now passes all 29 CPU tests
in 75.36 seconds. A read-only preparation audit authenticates both
parent archives, the complete 162-helper import closure and the unchanged
54-case/162-command plan. Separate CPU/source acceptance completes at
09:27:15 with 29 archived passing tests, 162 helpers and 531 inventory
entries; a fresh archived-origin check passes by 09:29:32.
No successor owner, binding, GPU launch or production result is issued.
The guarded split-document refresh completes at 09:35:50 with all
manual prose byte-preserved. The integrated original-training-path/
actual-LR CPU suite passes all 63 tests in 221.51 seconds.
At 09:36:55 the validator accepts FP32 MuonT/seven NS iterations:
checkpoint22/checkpoint24 have 2,121/2,167 exact tensor comparisons,
zero scalar/structure differences, and 39/39 passing loss comparisons.
Update23 retains sixteen graph-readiness flag differences among 2,185
unchanged tensors and remains diagnostic-only.
At 09:37:18 the live diagnostic has 15/54 retained commands and
5/18 accepted cases; the state selects FP32 MuonT/ten-iteration
continuous24 next. All four original process generations remain live.
The waiter's fourth observation at 09:32:40 still records `ready: false`.
All three FP32/ten-iteration commands exit/clean up zero by 09:46:37;
their complete trajectories are retained by 09:49:04, bringing the
diagnostic to 18/54 commands and five accepted cases. The sixth case's
numerical decision remains pending at 09:55:13.
The existing Bash monitor refreshes both documents at 09:43:40.
A distinct actual-LR predecessor component is CPU/source accepted at
09:53:20 with twelve archived passing tests, 169 helpers and 538 inventory
entries; all 114 integrated predecessor/waiter/reporting tests pass.
Fresh archived checking passes, and its read-only observation at
09:54:49 verifies exact original controller, validator and terminal
waiter live, with `ready: false`. No predecessor seal or successor
ownership/launch is issued.
The sixth case, FP32 MuonT/ten NS iterations, is validator-accepted
at 09:59:15. Checkpoints22/24 contain 2,151/2,169 unchanged tensor
comparisons with no scalar/structure differences, and all 39 loss
comparisons pass. Checkpoint23 retains sixteen graph-readiness flag
differences among 2,155 unchanged tensors and remains diagnostic-only.
The first fused output-diagonal/BF16/five-iteration continuous24
command is selected at 09:59:24 and exits/cleans up zero at 10:02:14.
At 10:04:06 its trajectory has not yet entered the aggregate, which
still retains 18/54 commands and 6/18 accepted cases. Command exit
alone supplies no fused-case continuation acceptance.
That trajectory is subsequently retained at 10:04:47, bringing the
aggregate to 19/54 commands and six accepted cases. The 10:06:40
observation finds the original state selecting fused interrupted22;
its retained trajectory and the fused-case decision remain pending.
The interrupted22 sidecar subsequently records zero execution/cleanup
and absent marked descendants at 10:07:21. At 10:08:02 its trajectory
has not entered the aggregate, which remains at nineteen commands
and six accepted cases; resumed24 and the fused decision are pending.
The existing single Bash monitor refreshes both journal blocks at
10:03:39. Fresh archived predecessor observation at 10:01:12 returns
`ready: false`, finding exact original controller, validator and waiter
live. All 149 issued helper files and their inventory are unchanged
at 10:03:33. Original terminal, successor predecessor seal and actual-LR
ownership/binding/output paths remain absent at 10:01:04.
Five focused journal/monitor tests pass; no source, tolerance,
allocation, production or cloud-history change occurs.
All three first fused/BF16/five-iteration trajectories are retained by
10:12:44, bringing the diagnostic to 21/54 commands and six accepted
cases. The fused numerical decision remains pending at 10:23:21.
The original waiter records exact diagnostic owners live at 10:17:42;
fresh process checks at 10:20:16 verify controller, validator, waiter
and single Bash monitor unchanged/live. The monitor refreshes both
journals at 10:23:37. A separate actual-LR adapter candidate is
prepared at 10:19:09 and CPU/source-archive accepted at 10:21:54,
with fourteen passing archived tests, 171 helpers and 542 inventory
entries. Fresh archived-entry verification passes. Only its owner
adapter changes; all original numerical bytes remain unchanged.
Its execution owner/controller, original predecessor seal, actual-LR
binding and GPU launch remain unissued.
The first fused/BF16/five-iteration case is rejected by the original
validator at 10:23:53. Boundary22 is exact across 2,127 tensor
comparisons, and captured-row/update/work checks pass. Post-cached-root24
has 419 failed tensor comparisons across eight ranks, and three of
39 scalar-loss comparisons fail. Update23 has 120 failed tensors and
sixteen graph-readiness differences and remains diagnostic-only.
The original validator continues to fused/BF16/seven iterations:
continuous24 exits/cleans up zero at 10:26:44, and its complete trajectory
is retained at 10:29:06. An archived file-only reporter authenticates
22/54 commands and seven decisions, six accepted and one rejected, at
10:30:12. No independent terminal is recorded. Exact original controller,
validator, waiter and monitor generations remain live at 10:28:39.
The integrated actual-LR numerical/adapter CPU suite passes all
43 tests in 168.93 seconds. Five focused split-document/monitor tests
also pass. These CPU results do not change the fused rejection or
authorize an actual-LR GPU launch.
The seven-iteration interrupted22 command subsequently exits/cleans up
zero at 10:31:37 and is retained at 10:32:24, bringing the receipt
to 23/54 commands and seven decisions. Resumed24 is selected at
10:32:32; the eighth decision is pending. Fresh exact-generation
checks at 10:33:22 find the original controller, validator, waiter
and single Bash monitor live. The scheduled monitor lock still
refuses a second holder. No experiment or monitor is restarted.
The original validator rejects fused/BF16/seven iterations at
10:47:48. Boundary22 is exact across 2,129 tensor comparisons,
and capture/update/work checks pass. Post-cached-root24 has
428 failed tensor comparisons across all eight ranks, two
rank-3 state-counter differences and two failed loss checks out
of 39. Update23 remains diagnostic-only with 114 failed tensors
and twenty scalar/structure differences. The original validator
continues to fused/BF16/ten iterations; its continuous24 trajectory
is retained at 10:53:13. The aggregate contains 25/54 commands
and eight decisions, six accepted and two rejected.
Archived file-only reporting authenticates both rejections at
10:50:35. Exact original controller, validator, waiter and monitor
generations remain live at 10:49:05. The single Bash monitor
refreshes both journals at 10:43:26; its 900-second cycle continues.
The separate actual-LR execution owner/controller is now implemented
without modifying the accepted adapter candidate or any of its
171 inherited helper bytes. Forty-five CPU lifecycle tests and a
separate freeze/visible-identity integration test pass. The complete
53-test suite and independent archived acceptance are running;
archive acceptance is not yet established. No predecessor seal,
actual-LR GPU binding, allocation transfer or launch is issued.
Those CPU runs subsequently pass all53 tests in186.89/188.76seconds,
and the separate execution-owner archive is accepted at10:55:01
with177 helpers/548 inventory entries. Fresh archived-entry checking
passes by10:58. All171 parent helper bytes and the accepted adapter
candidate remain unchanged. Fused/BF16/ten-iteration interrupted22
exits/cleans up zero at10:55:50 and is retained at10:56:39;
resumed24 is selected at10:56:47. The receipt contains26/54 commands
and the same eight decisions, six accepted and two rejected.
At10:56:56, all four original process generations remain exact/live;
the original terminal, successor predecessor seal and actual-LR
binding remain absent. CPU acceptance supplies no new loss,
efficiency or SwanLab result.
At11:01:13 the fused/BF16/ten-iteration resumed trajectory is
retained, bringing progress to27/54 commands. The original validator
rejects that case at11:12:11: checkpoint22 has2,127 exact comparisons;
checkpoint24 has460 failed tensors and three rank3 scalar differences;
four of39 scalar-loss comparisons fail. All three fused BF16
iteration budgets now fail the required post-cached-root continuation
gate. There are nine of18 decisions, six accepted and three rejected.
The same Bash monitor refreshes both journals at11:04:37.
A distinct actual-LR queue passes33 archived CPU cases in46.15seconds
and is accepted at11:17:14 with180 helpers/551 inventory entries.
It preserves the accepted177-helper owner archive and adapter
candidate unchanged. Independent archived-entry checking passes
by11:19. Its hidden-CUDA waiter acknowledges at11:20:12 as
`2541469/939256843` and verifies the exact original owners live at
11:20:42. It checks every900seconds, waits for the existing original
terminal waiter and exact restoration, binds the separately accepted
predecessor seal digest at freeze, and refuses failed or consumed
phases. No successor seal, GPU binding or allocation transfer is issued.
The original fused/FP32/five-iteration continuous24/interrupted22
trajectories are retained at11:17:41/11:21:05, bringing progress to
29/54 commands and nine decisions, six accepted and three rejected.
The resumed24 trajectory is retained at11:25:34, bringing progress
to30/54; its numerical decision remains pending at11:31:53.
The same monitor refreshes both journals at11:24:26 using imports
loaded before the new queue reporter was added. A fresh guarded
split-document render starts at11:26:36 and publishes by11:31:30,
including the authenticated actual-LR queue report; no monitor restart
is needed. Forty-two local queue/reporting tests pass in44.99seconds;
fourteen final selected journal/monitor/reporting tests and scoped
source lint pass. The document-size
lint exclusion remains recorded in `DETAILS.md`. Exact queue liveness
is verified at11:24; production results and optimizer superiority
remain unestablished.
See [Third Fused Training-Path Case Rejection](#third-fused-training-path-case-rejection)
and [Training-Path Actual-LR Queue Preparation](#training-path-actual-lr-queue-preparation).
See [Fourth Training-Path Case Decision](#fourth-training-path-case-decision)
and [First Fused Training-Path Case Rejection](#first-fused-training-path-case-rejection);
successor evidence is in [Training-Path Actual-LR Component CPU Validation](#training-path-actual-lr-component-cpu-validation).
A separate guarded refresh succeeds at 08:19:15, preserving both
manual-prose hashes. The preceding 07:24:08 process loaded an earlier training-path reporter before
its local corrections and preserves a numerical-CPU reporting refusal;
the accepted archive and independent fresh checks remain unchanged.
A separate corrected guarded refresh completes at 07:30:13, accepting
the numerical CPU report and preserving both manual-prose hashes.
Independent nonblocking contention still refuses a second monitor
lock holder. Earlier concurrent-edit
refusals preserve both documents. A separate guarded refresh also
succeeds at 21:24:28 and verifies all manual prose byte-preserved.
No monitor was started or stopped during this journal update.
The precursor times in the following paragraphs are November 18 unless
another date is stated.
A fresh issued-source read-only check completes by 21:42, verifying
exact keeper `1569892/933748533`, all eight original rank generations and
physical A100 UUIDs, with original start-accuracy owners absent and no
active training. This keeper then transfers through the issued controller.
Current source now supports refreshing both split documents from either
entry point; the latest integrated journal/reporting/digest tests pass
297/297 with CUDA hidden in 108.64 seconds. The recurring CLI also saves
immutable observations of the original process generations, marked training
children and physical GPU occupancy. These observations execute no tensors.
The rational full-model successor completes CPU/archive acceptance at
21:18:19: 215 tests pass with complete module coverage, all 112 helpers
and six accepted fixtures authenticated. A fresh reconstruction verifies
the receipt by 21:19. Earlier failed test/import/XML-acceptance attempts
are retained. Source freeze completes at 21:33:10; the issued checker
passes by 21:34. Read-only historical provenance verification completes
at 21:34:51. The GPU controller acknowledges at 21:43:06. Its validator
and first pinned/control eight-rank command are verified live at 21:44.
All twelve commands have exited integer zero by 22:43:51;
all twelve complete retained trajectories authenticate at 22:46:43.
The validator publishes complete analysis November 19 00:35:42:
trace transparency and pinned historical reproduction pass, but rational
all-rank continuation fails. Pinned/rational tensor failures are 97/97 at
update 23 and 460/439 at update 24, pooled across eight ranks; scalar-loss
failures are 4/39 and 5/39 respectively in both control/trace protocols.
The 02:40:35 read-only observation verifies the original controller live,
the direct validator absent and no active training rank or GPU compute
process. The original controller finishes reconstruction and records
`diagnostic_complete` at November 19 02:53:29, with validator/cleanup and
keeper-restoration exits all integer zero. Its result identity and failed
rational-continuation predicate remain unchanged. A fresh read-only
observation at 02:56:55 verifies both original owners absent, no active
training or marked descendants, and eight non-training compute processes
on the original physical GPU UUIDs. The saved restored keeper is
`1837003/936215240`. Independent terminal verification by the original
waiter completes at 05:21:49, accepting exact restoration on the original
eight A100 UUIDs. Its terminal retains the rejected rational intervention.
The separate CUDA-hidden terminal follow-up acknowledges at 22:22:23;
its original generation is verified live. At November 19 03:07:32 it
observes both original owners absent and issues the original archived
read-only terminal verifier as child `1855154/936300767`.
That exact child remains active at 05:16:23 and completes at 05:21:49.
The original waiter records `terminal_reconstructed` at 05:21:51, with
terminal execution and cleanup exits both integer zero and descendants
absent. Read-only process inspection at 05:25:03 finds both waiter and
terminal child absent. The accepted terminal is
`PVC/run-preparation/rational-continuation-terminal-independent-20261118.json`.
Eight archived CPU tests and 28 focused follow-up/reporting tests pass.
It checks original owner generations every 900 seconds and invokes only
the issued terminal verifier after actual owner absence and completed state.
The separate CPU operand-precision study completes at 22:45:29 after
source/archive acceptance at 22:43:10 with 22 passing archived tests.
The first 18-pass/two-failure
test receipt is preserved; both plain FP32 and compensated products fail
the original tolerance on near-rank-deficient fixtures, while full-rank
controls pass. Retained historical-pair comparisons are independently
recomputed at 22:46:12. Six-product emulation reduces output-pair relative
error from `0.0368395` to `3.68270e-5`, but the unchanged tolerance still
fails, including the independent FP64 finite-polynomial pair (`3.66208e-5`,
789,618 mismatches). This rejects higher operand precision alone as a
sufficient correction for this saved pair. No GPU adapter, continuation
gate or production authority is supplied.
The distinct NS step-budget study completes at 23:01:47 after 26 archived
tests; retained pairs/repeats and references are independently recomputed
at 23:02:51. Fewer steps reduce nominal product counts and pair amplification,
but every FP32, BF16-emulated and FP64 pair still fails tolerance.
Five/seven steps are separate quality-cost hypotheses; they are not
validated continuation corrections. The compiled graph-trial component
receives separate CPU/archive acceptance at 23:10:53 with 29 passing tests.
The integrated replay/owner suite passes 81 tests in 42.84 seconds.
At this historical graph-component checkpoint no CUDA owner or launch exists.
The source-bound NS execution/reconstruction component receives separate
CPU/archive acceptance at 23:39:07: all 45 archived tests pass in
9.21 seconds and a fresh archived checker accepts the source, fixtures and
receipt. The affected saved-replay/owner suite passes 97 tests in
45.66 seconds. This component declares six step/mode arms and retains
eager, compiled, graph and timed-final phases; no allocation owner,
GPU timing or launch is issued.
The separate NS diagnostic owner/controller/worker receives CPU/source-archive
acceptance November 19 00:03:08: 75 archived tests pass in 24.71 seconds,
122 helpers and two pinned fixtures are authenticated, and a fresh archived
checker accepts the receipt. The broader saved-replay/owner suite passes
122 tests in 57.97 seconds. The owner requires original rational completion,
independent terminal reconstruction, waiter absence and exact keeper transfer.
At that owner-acceptance checkpoint no NS binding, execution directory,
queue or GPU launch was issued. A separate one-shot queue receives
CPU/archive acceptance at 00:24:32 with 21 passing archived tests,
124 helpers and two fixtures; its original owner acknowledges at 00:25:30.
Fresh archived checking passes, and the 02:40:47 exact-generation check
verifies that queue live. Its saved 02:55:47 state verifies both original
rational owners absent and waits for terminal reconstruction and waiter
absence, with 0/4 phases and no NS GPU binding
or worker. The queue checks original owners every 900 seconds and issues
accepted freeze/check/start/terminal commands once after all gates pass.
After the original rational terminal completes, that same queue issues
freeze/check/start, all with integer-zero execution and cleanup, and
acknowledges controller `1937350/937133300` at 05:26:17. Worker 0
`1937511/937134210` and worker 1 `1940804/937152695` both exit and
clean up integer zero by 05:32:19. The controller publishes retained
analysis at 05:37:38 and records `diagnostic_complete` plus integer-zero
keeper restoration at 05:37:54. At 05:38:37 the original controller and
workers are absent; the new keeper is `1947488/937201461`.
Same-input, graph and cross-process comparisons are exact, but compiled
calls differ from pinned eager and historical controls are not accepted.
At 05:41:28 the original queue issues exact archived independent terminal
verifier `1949828/937224407`. Independent terminal reconstruction and exact
restoration complete at 05:47:02, and the original queue records
`ns_diagnostic_terminal_reconstructed` at 05:47:10, with all 4/4 phases
exiting and cleaning up integer zero. Process inspection at 05:49:38
finds the original queue/verifier absent; both successor CPU queues and
the monitor remain live. The failed historical-control predicate keeps
the full-model successor gate closed.
Journal reporting preserves negative predicates and adds per-arm pair
errors and matched CUDA-event timing ranges when completed receipts exist.
The integrated journal/queue/owner suite passes 286 tests; the final
schema/reporting/split-monitor regressions pass 13 tests.
A distinct NS-budget full-model preparation receives CPU/source-archive
acceptance at 01:05:55 with 16 passing archived tests, 124 authenticated
helpers and three fixtures. A fresh archived-entry check accepts the receipt;
the affected preparation/owner/queue/reporting suite passes 76 tests.
The candidate preserves the failed pinned-root parent and adds 54 equal-LR
configurations across three families, two operand modes and three budgets.
Its owner rejects training before configuration/data/CUDA.
A distinct full-model validator component receives CPU/source-archive
acceptance at 01:35:09: 28 archived tests pass in 135.00 seconds, with
128 authenticated helpers and the complete inherited source/fixture
inventory. A fresh archived check accepts the unchanged receipt; the
integrated validator/configuration/owner/queue/reporting suite passes
104 tests in 185.30 seconds. It reconstructs all 18 lower-LR cases and
54 bounded commands, retains numerical rejection evidence and preserves
the original update-22/update-24 continuation and scalar-loss gates.
Update 23 remains a retained diagnostic, including process-local graph
rewarming. No full-model allocation owner, binding, queue or GPU launch
is issued. The original NS diagnostic terminal and actual-config GPU
gates remain required.
A distinct full-model execution-owner candidate completes preparation by
02:11 under `PVC/candidates/ns-full-model-owner-20261119T0207`.
It changes only `experiment/fused_attribution_owner.py` and retains all
128 accepted validator helpers, adding five separately scoped execution
helpers. Its first complete owner suite passes 28 tests in 78.77 seconds.
The distinct lifecycle-corrected candidate finishes preparation at 02:18.
Ten lifecycle tests and 142 integrated tests pass in 100.00 and
324.12 seconds respectively. CPU/source-archive acceptance completes by
02:34: all 66 archived tests pass in 283.33 seconds with 133 authenticated
helpers. A fresh archived-entry check passes by 02:40:24 with the same
source/archive acceptance status.
Launch-readiness review identifies repeated recursive parent/archive
authentication within one operation. The mutable execution helpers now
reuse that operation's authenticated parent and inventory without caching
or skipping checks at public boundaries. The accepted 66-case archive is
unchanged. Three new real freeze/fresh-process adapter cases require exact
frozen import origins; their first attempt fails before adapter checks
because the temporary fixture lacks `run-preparation`. That fixture is
corrected; all seven focused source/freeze cases pass in 131.04 seconds.
The distinct `ns-full-model-owner-single-pass-20261119T0257` candidate
completes CPU/source-archive acceptance by 03:05: all 69 archived tests
pass in 281.59 seconds, with 133 authenticated helpers and unchanged
accepted validator/numerical bytes. All 145 affected integrated tests
pass in 338.08 seconds. A fresh archived-entry check passes at 03:06:42,
and a guarded split-journal preservation check passes at 03:06:56.
The earlier 66-case receipt/archive hashes remain unchanged. No GPU
binding, owner or full-model command is issued.
A separate full-model queue completes CPU/source-archive acceptance at
03:20:06: all 25 archived tests pass in 27.66 seconds, with 136
authenticated helpers. A fresh archived check exits zero; the affected
queue/owner/validator suite passes 145 tests in 342.76 seconds.
This queue preserves the original NS queue and its exact generations.
It waits for independent NS terminal reconstruction, interpretable
historical controls and original queue/owner/worker absence before issuing
the accepted full-model freeze/check/start once. Its archived waiter
`1875861/936425460` acknowledges at 03:28:19 and saves
`waiting_for_original_ns_queue` at 03:29:07, with 0/4 phases and
the original NS queue generation verified live. No full-model binding,
frozen GPU owner, controller, actual-config run or new SwanLab history exists.
All 216 final reporting/journal tests pass in 25.89 seconds. Independent
guarded split-journal verification completes at 03:38:59, preserving both
manual-prose hashes and all accepted queue/owner/archive/intent identities.
The recurring blocks retain the waiting queue and failed rational predicate.
A distinct full-budget NS study protocol/analysis component completes
CPU/archive acceptance by 03:54:54, preserving the accepted owner and
queue. All 19 archived tests pass in 11.00 seconds with 136 helpers;
61 integrated tests pass in 55.29 seconds. It retains all 54 equal-LR
configuration identities, five 536,870,912-token windows and exact
final-half mixture selection, but provides no production owner or launch.
The current diagnostic gates only lower LR; center/upper and any expanded
LR require independent actual-config gates under eventual production source.
Fresh archived checking and unlaunched analysis both exit zero at
03:57:58: 54 planned trials, zero observed/complete histories, all 18
grids incomplete, no LR choices and no production/cloud/superiority claim.
The shared-filesystem monitor lock reports PID 0 in `fdinfo`; the strict
PID-bound GPU lock verifier refuses that descriptor even though independent
contention confirms the monitor lock is held. A separate read-only probe
at 00:09:55 accepts the actual rational controller's node and manager
descriptors under the unchanged verifier; independent contention refuses
both locks. Worker-time verification and original terminal/restoration
remain required.

At the 05:25:03 process check the original NS diagnostic queue, full-model
queue and single Bash document monitor remain live; the completed terminal
waiter and verifier are absent. Latest saved NS/full-model observations
checked by 05:34 are 05:26:21/05:14:07. The NS queue has completed
3/4 phases and waits for its issued diagnostic owner; the full-model
queue still waits for the original NS queue. A 05:30:55 physical query
observes only worker 1 on GPU 0, consistent with the accepted single-device
replay protocol. Both workers subsequently exit; no post-reconstruction
restoration or current occupancy is inferred from that earlier query.

The 54-configuration actual-LR validator and its read-only original
lower-LR terminal-handoff component are CPU/source-archive accepted,
with fresh archived checks completed by 04:24:35 and 04:41:22 respectively.
The distinct actual-LR execution owner is now CPU/source-archive accepted
by 05:14, with 74 archived tests and 93 integrated tests passing.
Its 153-helper archive preserves all 148 parent helpers and changes only
the separate candidate's ownership adapter; a fresh archived-entry check
passes by 05:16 with unchanged receipt/archive digests and exact imported
helper origins. No binding, controller or GPU command is issued.
A separate actual-LR one-shot queue completes CPU/source-archive acceptance
by 05:32: all 27 archived tests pass in 29.84 seconds and all 156 helpers
authenticate, preserving every accepted owner helper. All 120 affected
queue/owner/validator/study tests pass in 296.87 seconds; a fresh archived-entry
check passes by 05:33. Earlier fixture failures and interruption are retained,
and numerical tolerances remain unchanged. All 233 journal/reporting tests
pass in 27.15 seconds. The single archived CPU waiter
`1945601/937186048` acknowledges at 05:35:04 and saves
`waiting_for_original_full_model_queue` at 05:36:14, with 0/4 phases.
Its exact generation is live at 05:36:18. The original lower-LR terminal
and queue/owner absence still gate GPU issuance. Recurring reporting now
authenticates this exact acknowledgment and preserves rejected decisions.
This is not another wave result. The completed-wave quality/timing conclusions and private
SwanLab evidence remain unchanged; no superior elapsed-efficiency optimizer
relative to MuonT has been established.

At November 19 05:59:09, the original lower-LR CPU queue records
`needs_attention`, with 0/4 phases, after its predecessor check rejects
the completed NS diagnostic's historical-control predicate. Read-only
inspection at 06:00:54 finds that queue absent. The actual-LR queue
then records `needs_attention` at 06:07:02, also with 0/4 phases.
Its nested lower-LR source verifier expects a test helper beside itself
that the actual-LR archive does not contain. At 06:08:51 both original
successor queues are absent. The numerical historical-control rejection
and the separate archive-dependency failure remain retained.
The 05:57 physical GPU query observes only the eight restored keeper
processes; no full-model GPU binding or training command is issued.
A source audit authenticates the same six training/optimizer files
against all three accepted preparation/owner receipts. It finds that
training calls the NS routine outside the model-body and loss compiler
boundaries. The completed NS diagnostic adds a separate full-NS compiled
wrapper. Its failed compiled historical control remains rejected, but
does not establish that training used that wrapper. The follow-up
distinction and planned experiments are recorded in
[NS Compiler Boundary Audit and Follow-Up](#ns-compiler-boundary-audit-and-follow-up).
The distinct training-path predecessor-evidence helper is now locally
implemented and CPU/source-archive accepted. It preserves the original NS
numerical rejection and both zero-phase queue failures. Complete persisted
evidence-chain tests pass, and a fresh archived-entry check succeeds by 06:50.
The original-output digest seal completes at 06:55:39 with exact keeper
restoration and failed numerical predicates retained. No GPU owner is issued.
Fresh archived seal checking succeeds again by 07:05 without rehashing the
large tensor payloads. The local 07:04 physical query observes only the
eight original restored keeper ranks; both failed queues remain absent.
The distinct 18-case numerical protocol receives CPU/source-archive
acceptance at 07:10:02 with all 34 archived tests passing. Fresh archived
checking succeeds by 07:11; execution ownership remains pending. Review
requires the owner to independently pin the complete original source and
configuration inventory. The new owner-only implementation satisfies that
source requirement. Its separately accepted archive passes 55 CPU cases;
fresh archived and issued-source checks pass. The frozen controller
acknowledges at 07:44:23 and its exact generation is live. By 07:45:52
the keeper is released, the validator is acknowledged, and the first
eight-rank MuonT BF16/five-step continuous command is live. No command
or case has completed at that launch observation. The first continuous
command subsequently reaches update 24 and exits with integer-zero
execution and cleanup at 07:47:44. By 07:52, continuous and interrupted
commands have both completed; no case decision is yet recorded.
The resumed command then exits with integer-zero execution and cleanup
at 07:54:52. All three first-case commands are recorded complete by
07:57:23. At 08:07:38, the validator publishes acceptance for
MuonT BF16/five-NS-iteration lower LR: checkpoint22/checkpoint24 and
39 scalar-loss comparisons pass, with zero tensor failures.
Update23 retains only graph-readiness flag differences and remains
diagnostic-only. The exact original controller and validator remain
live at 08:08:14, and the seven-iteration continuous command is running.
That fourth command exits with execution/cleanup codes zero at 08:10:01;
its complete trajectory is added to the aggregate receipt at 08:12:26.
The diagnostic has four completed commands and one validator-accepted case;
the seven-iteration case and independent terminal verification remain pending.
Seven-iteration interrupted22 and resumed24 commands subsequently exit
zero at 08:14:32 and 08:17:16. All six first-two-case commands have
complete retained trajectories by 08:19:46; only the five-iteration
case has a validator decision at the 08:24 observation.
The new milestones are recorded in
[Training-Path Evidence Preparation](#training-path-evidence-preparation).

| Experiment | Current State | Evidence and Next Gate |
| --- | --- | --- |
| Earlier production Waves 0-11 | Waves 0-4 and 7-10 completed, analyzed and privately uploaded; Waves 5/6/11 failed | The earlier Wave-2-running status is historical: Wave 2's SwanLab import was verified November 16; the requested `inner_outer_all_lag` arm completed in Wave 3, whose import is also verified. Full records and large shared-token-window comparisons remain below. Failed waves are retained and are not counted as optimizer results. |
| Wave 12, four selected-LR arms | Completed, analyzed, private SwanLab histories verified | 5,120 updates/2,684,354,560 tokens per arm; five shared 536,870,912-token windows and 22 exact shared evaluations, including 11 in the final half. Canonical receipts: `PVC/run-preparation/analysis-wave12.json` and `PVC/run-preparation/swanlab-upload-wave12.json`; cloud verification November 18 02:04:24. |
| Selected-LR stage profiling | Terminal `needs_attention` | Three MuonT commands exited zero; rank-0 trace-reader rejection stopped trial reconstruction. The other three arms remain unexecuted. Original controller/validator absence and exact keeper restoration were independently observed November 18 03:08:21. |
| Saved MuonT trace-reader correction | CPU/archive-verified; GPU-denied | Receipt at November 18 03:23:38 reconstructs all eight rank traces from MuonT's profiling-on resumed command, updates 201-220. Checkpoint-tensor/trial reconstruction and full profiling acceptance remain absent. Receipt: `PVC/run-preparation/selected-lr-stage-profile-reader-cpu/20261118T0319/receipt.json`. |
| Fused/matched-null full-schedule LR study | Attribution candidate CPU/archive-verified; GPU-denied | Candidate `PVC/candidates/fused-attribution-full-schedule-20261118T0257` contains nine seed-0 full-budget configurations: MuonT, fused output normalization and identity-outer matched null, each at three LRs. Receipt: `PVC/run-preparation/fused-attribution-cpu/full-schedule-20261118T0257/receipt.json`. No production run or new cloud history exists. |
| Attribution validation successor | Terminal `needs_attention`; 6/27 commands exited zero, 1/9 trial check sets passed | Fused's update-24 post-cached-root comparison rejects `rank_0.model.body.emb.weight` at 05:24:21 under the unchanged `rtol=1e-5`, `atol=1e-6` gate. Lower-LR MuonT passes all eight checks with zero observed continuation error; no fused trial acceptance exists despite its three integer-zero commands. Controller failure/restoration is recorded at 05:24:39. Read-only inspection at 05:27:51 verifies original controller/validator and child markers absent, with exact healthy eight-A100 keeper `929078` restored. Remaining null/center/upper trials are unexecuted. Evidence: `PVC/run-preparation/fused-attribution-validation-terminal-observation-20261118T052751.json`. |
| Saved fused checkpoint-continuation diagnosis | Read-only eight-rank CPU diagnostic complete; cause unresolved | Comparing continuous with resumed checkpoints at the same update gives 2,127 exactly equal rank-specific tensor instances at update 22. At update 24, 1,630 of 2,191 rank-specific tensor instances change and 460 fail the original tolerance: 235 model, 72 optimizer and 153 fast-state tensors. Counts are pooled across eight ranks; every rank has failures. Rank 3 also differs in one basis-step counter, fallback count and refinement count; this qualifies the earlier rank-0-only observation. Receipt: `PVC/run-preparation/diagnostic-fused-attribution-all-rank-continuation-20261118T0548.json`. This is not a GPU gate, causal intervention or optimizer result. |
| First-resumed-update fused diagnostic | Diagnostic reconstructed and restored; continuation failure remains | Candidate `PVC/candidates/fused-continuation-owner-fixture-20261118T0634` retains lower LR `0.04525483392`, the full 5,120-update schedule, compilation and cached-root graphs. Receipt `PVC/run-preparation/diagnostic-fused-continuation-gpu-20261118.json` publishes at 09:42:34 with both transparency predicates true. Trace-off/on checkpoints are exactly equal across eight ranks at updates 22/23/24; extra-checkpoint/original comparisons are exact at 22/24, and four loss sets pass 39/39 checks each. Both protocols reproduce 97 rank-0 tensor failures at update 23 and 460 failures across eight ranks at update 24. Original controller terminal reconstruction completes at 12:02:56 and is source/archive-authenticated at 12:04:45. This accepts diagnostic evidence, not numerical continuation or production. |
| Fused continuation diagnostic executor | Terminal `diagnostic_complete`; original controller/validator absent, exact restoration verified | Original controller `1032202/929444503` completes at 12:02:56 with integer-zero execution, cleanup and restoration exits, descendant absence and the unchanged diagnostic receipt hash. Archived successor-helper observation at 12:04:45 verifies original controller/validator PID absence, source/archive/log identities and exact healthy keeper `1264349/930871889`, ranks `1264445` through `1264452`, on all eight original physical A100s. Receipt: `PVC/run-preparation/saved-ns-terminal-predecessor-observation-20261118T1205.json`. This restored keeper subsequently transfers to the separately issued saved-NS replay owner. Original document monitor `3776115/919963640` remains live. |
| Separate stage-aligned offline analyzer | CPU/source-archive verified; saved rank-0/update-23 analysis complete | Focused 34 tests and integrated 49 tests pass; source/archive receipt at 09:08:37. After both saved transparency predicates pass, unchanged helper authenticates one trace pair and completes 662 tensor comparisons at 09:45:47, with complete stable coverage and no nonfinite elements. All 291 gradient tensors are exact; root inputs are exact apart from process-local graph/warm flags. Tiny root/NS-source differences precede large NS-output and update differences. Branch tensors remain unpaired. Evidence: `PVC/run-preparation/continuation-stages-rank0-update23-20261118T0946.json` and its log. This is saved-gate-dependent localization, not independent diagnostic reconstruction, causal proof or production acceptance. |
| Pending-refresh multi-basis CPU round trip | Scalar compiled CPU check exact; default-vector CPU compiler failure preserved | Four nonidentity 4x4 bases and three pending payloads round-trip through the executed candidate's serialization/restoration and masked installation. All 35 tensor comparisons have zero error; eligible, stale and delayed decisions agree at updates 22/23/24. The fixture preloads future work and is not a valid training-checkpoint trajectory. Separate scalar receipt at 09:19:50 uses `cpp.simdlen=1`; the default-vector attempt exits 1 on a CPU code-generation error. This does not validate CUDA ordering, graph replay, the default-vector kernel or the failed training continuation. Evidence: `PVC/run-preparation/pending-refresh-scalar-20261118T0920.json` and `pending-refresh-default-vector-20261118T0920.json`. |
| Actual boundary-checkpoint pending restoration | Eight-rank CPU restoration exact; installation separately fails | At 09:31:42, 172 tensor comparisons pass exactly for executed update-22 checkpoints. All 54 pending source-step-21 payloads are restored exactly and eligible/nonstale at optimizer step 22, training update 23. Separate full-shaped scalar compiled CPU installation fails at 09:35:00. Rank-0 split diagnosis at 09:36:07 localizes the CPU discrepancy to missing low FP16 components; direct eager installation is exact. This qualifies the small probe's shared-constructor reference and does not establish a GPU failure cause. Evidence: `saved-pending-restoration-20261118T0932.json`, `saved-pending-installation-failed-20261118T0936.json` and `diagnose-pending-split-20261118T0938.json` under `PVC/run-preparation/`. |
| Actual GPU-produced post-install split probe | 31 retained payloads exact; 23 later-replaced payloads excluded | CPU read-only comparison at 09:45:13 authenticates trace-off resumed checkpoints 22/23 and directly constructs independent FP16 high/low layouts. All 93 `SB`/`S2`/tolerance comparisons are exact at `rtol=0, atol=0`, including all 13 rank-0 pending payloads. Replaced basis steps are explicitly excluded, not counted as passes. The earlier scalar-CPU low-split discrepancy does not reproduce in these retained GPU-produced states. Evidence: `PVC/run-preparation/check-installed-payload-checkpoint-20261118T0944.json`; no new CUDA execution, continuation acceptance or optimizer result follows. |
| Saved NS operand and precision-sensitivity probes | CPU observations complete; no CUDA reconstruction | At 09:59:58, CPU normalization of authenticated rank-0/update-23 sources passes the original FP32 tolerance, but BF16 operands differ in 3,831/7,997/7,513 elements across three shape groups. At 10:02:38, occurrence-0 ten-step BF16-operand emulation amplifies the source difference to relative Frobenius error `0.0368395`, versus `0.0000373549` in FP32 and `0.0369345` in the saved GPU output pair. Identical-input CPU repeats are exact; neither mode reconstructs either saved CUDA output. CPU reduction/GEMM differences prevent a CUDA causal conclusion. Receipts: `PVC/run-preparation/saved-ns-operands-20261118T1001.json` and `saved-ns-sensitivity-20261118T1004.json`. |
| Source-bound crossed NS replay helper | CPU controls complete; separately owned CUDA replay launched below | New `tracking/replay_saved_ns.py` authenticates the full rank-0/update-23 occurrence-0 pair and retains independent outputs in `C,R,C,R` order. Two fresh CPU-f32 controls finish at 10:18:10/10:19:25; within-process repeats and all four cross-process output comparisons are exact at 10:20:18. All eight saved-CUDA-output comparisons per process fail the original tolerance. Integrated 227-test suite and scoped code lint pass. These historical CPU controls are distinct from the subsequently issued CUDA replay and are not a training result. |
| Saved-NS GPU owner, adapter and worker | Diagnostic-parent successor CPU/archive-verified; historical live-predecessor denial preserved | Distinct receipt at 11:32:40 binds 78 helpers, two pinned-source fixtures and 67 passing tests in 49.15 seconds. Archived freeze/fresh-check/start and retained-output reconstruction have CPU coverage. A missing `diagnostics/` parent is corrected; an isolated execute test verifies restoration after child-spawn failure. Scoped lint passes. The 11:34:47 archived-entry denial remains historical evidence. Receipt: `PVC/run-preparation/saved-ns-gpu-adapter-cpu/diagnostic-parent-20261118T1132/receipt.json`. After original terminal/restoration authentication, this exact archive supplies the separately frozen replay below; mutable checkout helpers are not executed. |
| Saved-input NS CUDA replay | Diagnostic complete; exact same-input/saved-output reproduction and restoration independently verified | Two fresh full-batch `C,R,C,R` processes exit zero with zero cleanup; controller terminal/restoration at 12:08:51. Independent frozen-helper reconstruction at 12:12:17 verifies 4/4 exact finite within-process repeats, 8/8 exact corresponding saved-GPU-output comparisons and 4/4 exact cross-process comparisons. Source pair passes tolerance (relative Frobenius error `1.4513e-6`); its reproduced output pair fails with relative error `0.03693455` and 4,191,329 mismatching elements. Controller/workers are absent; exact healthy keeper `1268038/930907131` is restored on all eight original A100s. Receipt: `PVC/run-preparation/saved-ns-terminal-result-observation-20261118T1212.json`; retained comparison: `PVC/diagnostics/saved-ns-replay-20261118/comparison.json`. This localizes one saved NS occurrence to input sensitivity, not the upstream root cause, full-model continuation or a production gate. |
| Contingent restored-root preparation helper | Historical standalone CPU/archive acceptance; integrated into the separate diagnostic candidate below | `tracking/prepare_restored_roots.py` prepares eligible cached-root kernels using independent `C/tol/SB/S2` scratch, then commits readiness after side-stream success. Its original bytes are retained. Archived 12-test lifecycle suite passes at 11:47:54; integrated 296 tests pass. CPU staging of authenticated rank-0/update-23 continuous/resumed `d/f` inputs passes 16 exact tensor/layout/storage-isolation comparisons at 11:49:05. Receipt: `PVC/run-preparation/restored-roots-cpu/20261118T1153/receipt.json`; scratch result: `PVC/run-preparation/saved-root-scratch-20261118T1154.json`. This standalone receipt issued no GPU execution; the integrated helper is now exercised by the separately frozen diagnostic below. |
| Restored-root scratch-readiness diagnostic | Terminal `needs_attention`; all 12 GPU commands exited zero; independently reconstructed intervention rejected | Validator fails at 15:59:34 on the before-call lifecycle expectation; controller records terminal failure at 15:59:52. The original receipt has no completed `analysis` and is preserved unchanged. Separate prepared-control comparisons retain 97 rank-0 failures at update 23 and 460 all-rank failures at 24, plus 4/39 failed scalar checks. Both trace transparencies are exact; prepared/unprepared rank-0/update-23 stable stages are all 662 exact per mode. Fresh issued-helper observation at 16:10:24 verifies controller `1322053/931372042` and validator `1322447/931376704` absent, no child marker, and exact healthy restored keeper `1423935/932293205`, ranks `1424069` through `1424076`, on the original eight A100s. Failed receipt: `PVC/run-preparation/diagnostic-root-preparation-gpu-20261118.json`; independent failure/restoration observation: `PVC/run-preparation/root-failed-owner-observation-20261118T1610.json`. Separate full reconstruction completes at 18:32:16 and rejects prepared continuation; its result is recorded below. All production gates remain required. |
| Root-kernel stage recorder and crossed replay successor | CUDA diagnostic complete; independent terminal reconstruction complete; historical continuous output reproduction fails | Original controller `1539909/933436525` acknowledges at 19:10:09 and completes at 19:21:10. Two workers exit and clean up with integer zero. Both processes pass trace transparency, eager/capture/replay safety and stage repeats; all 24 cross-process outputs and corresponding stages are exact. Isolated continuous/resumed stages are exact, and historical resumed outputs reproduce exactly. Historical continuous `d` passes tolerance without exact equality; continuous `f` fails all six protocol/phase comparisons per process. Comparison at 19:20:54: `PVC/diagnostics/root-stage-replay-20261118/comparison.json`. This diagnostic does not reproduce the historical full-model divergence or release production. |
| Separate failed-root reconstruction | Completed; intervention rejected; original owner absent and restoration verified | Original `1438221/932406416` completes at 18:32:16 with integer-zero exit. The separate result passes cold capture/repeat, trace transparency, unchanged continuous controls and original-output reproduction; prepared all-rank continuation fails with 97 tensor failures at update 23, 460 at 24 and 4/39 scalar failures. Fresh 18:35:57 source-bound observation verifies original reconstructor/failed owners/markers absent and exact healthy keeper restored. Result: `PVC/run-preparation/root-failed-diagnostic-reconstruction-20261118T1618.json`. Original failed receipt remains unchanged; no production acceptance is issued. |
| Root-stage diagnostic queue | Original one-shot sequence complete; queue absent | All four commands, archived `seal-predecessor`, archived `freeze`, issued `check` and issued `start`, exit zero. Seal completes at 19:09:39; queue records `diagnostic_controller_acknowledged` at 19:10:10. Original queue `1463430/932622453` is absent. Issued binding: `PVC/run-preparation/root-stage-replay-binding-20261118.json`. Original queue and prior failed evidence remain unchanged. |
| Separate root-stage precision analyzer | Completed; all isolated stages and recorded casts agree; historical attribution denied | Original archived consumer completes at 19:34:04. All 12 continuous/resumed group/phase pairs have exact finite stages and no first exact difference or tolerance failure; all 108 recorded FP16 casts match independent rounding. Historical-output reproduction and historical-interpretation predicates are both false. Result: `PVC/run-preparation/root-stage-precision-independent-20261118T1718.json`. This provides isolated precision observations and no causal, continuation or production acceptance. |
| Root-stage read-only terminal/precision follow-up | Both original one-shot phases complete; original runner and children absent | Original runner `1479257/932789135` records `terminal_and_precision_reconstructed` at 19:34:05. Both phases exit and clean up with integer zero and absent marked descendants. Independent terminal reconstruction at 19:30:30 verifies original controller/workers absent and exact healthy restored keeper `1547502/933501005`, ranks `1547638` through `1547645`, on all eight original A100s. Subsequent read-only process inspection finds runner and both original children absent; saved source/archive/log/result identities pass reporting authentication. |
| Cancellation-free quarter-root start hypothesis | Separate CPU/source-archive accepted; 32 saved CPU start comparisons complete; later CUDA trial recorded below | `tracking/root_start_rational.py` replaces subtraction of nearby inverse-quarter powers with an equivalent positive-pair rational coefficient; seven unrelated start outputs and clamped-pair behavior are preserved. Eight focused Decimal-reference tests and 84 archived tests pass. At 17:52:04, original/rational `O` fails/passes the unchanged FP64 gate in 32/32 matrices respectively; worst group errors fall from `6.1450e-5/3.5843e-4` to `7.9939e-9/1.1510e-8`. Both original and rational continuous/resumed CPU starts are exact, so this does not explain the divergence. Result: `PVC/run-preparation/saved-root-start-rational-20261118T1752.json`; CPU receipt: `PVC/run-preparation/root-start-rational-cpu/20261118T1750/receipt.json`. This parent receipt issued no CUDA execution; the separate accepted owner below supplies later CUDA accuracy evidence. Root-iteration, efficiency and production acceptance remain absent. |
| Independent start-accuracy reference and graph-trial component | Separate CPU/archive accepted; independent 32-matrix reference complete; later reused by distinct owner | At 18:20:33, log1p/expm1 FP64 arithmetic confirms original/rational O failures 32/0; FP64-reference relative disagreement is at most `2.67281e-16` and crossed CPU outputs remain exact. A distinct initial-replay-corrected 98-helper/six-fixture archive passes 97 tests at 18:24:11; this historical archive issued no GPU owner or execution. Its accepted components are subsequently preserved in the separately owned CUDA trial below. Result: `PVC/run-preparation/saved-root-start-log-reference-20261118T1822.json`; corrected CPU receipt: `PVC/run-preparation/root-start-accuracy-cpu/first-replay-20261118T1826/receipt.json`. No production release follows. |
| Source-bound start execution and retained-file reconstruction | Separate CPU/archive accepted; unchanged component reused by separately accepted GPU owner | New `tracking/root_start_execution.py` retains independent phase snapshots and complete opposite-variant order, authenticates saved source/configuration/data, and independently reconstructs retained arithmetic and lifecycle evidence. At 19:24:30, 100 helpers/six fixtures pass 109 archived tests in 104.73 seconds. Receipt: `PVC/run-preparation/root-start-execution-cpu/20261118T1922/receipt.json`. This parent issued no CUDA execution; its accepted bytes are preserved in the distinct owner archive below. |
| Separate start-accuracy GPU owner | CUDA diagnostic complete; independently reconstructed terminal and exact restoration verified | Accepted 104-helper/six-fixture archive passes 121 tests at 19:51:31 and preserves all 100 parent helpers exactly. Controller `1566611/933714846` acknowledges at 19:56:32 and completes at 20:02:25; workers `1566736/933715434`, `1567728/933724739` exit and clean up zero. Comparison at 20:02:09 gives 32/32 exact cross-process outputs, exact graph repeats/seven unaffected outputs and all rational `O` checks passing FP64 tolerance. Per process pinned/rational failures are 128/0 out of 128 matrix-by-phase checks per variant. Independent terminal reconstruction at 20:06:05 verifies original owner/worker absence and exact keeper `1569892/933748533`, ranks `1570032` through `1570039`, on the eight original A100s. no full-root, continuation, production or SwanLab acceptance follows. |
| Rational-start full-model continuation successor | Diagnostic and independent terminal complete; correction rejected | All 12 commands exit zero and trajectories authenticate. Analysis publishes November 19 00:35:42: trace transparency and pinned reproduction pass; rational continuation fails, with 97/439 tensor failures at updates 23/24 and 5/39 scalar-loss failures in both protocols. Original controller completes reconstruction at 02:53:29 with integer-zero validator/cleanup/restoration exits. The original archived terminal independently accepts exact restoration at 05:21:49, preserving `intervention_accepted: false` and the original diagnostic result. Terminal: `PVC/run-preparation/rational-continuation-terminal-independent-20261118.json`. No production or cloud import exists. |
| Separate rational diagnostic terminal follow-up | Original read-only terminal reconstruction complete; waiter and verifier absent | Eight archived tests pass November 18 22:21:30; original `1676886/934589879` acknowledges at 22:22:23. It issues exact verifier `1855154/936300767` at November 19 03:07:32, then records `terminal_reconstructed` at 05:21:51 with integer-zero execution/cleanup and descendants absent. Process inspection at 05:25:03 finds both original processes absent. Saved state: `PVC/run-preparation/rational-continuation-terminal-followup-20261118/state.json`. Numerical rejection remains; no production authority follows. |
| Compensated NS operand precision | CPU study complete; operand precision alone insufficient | Archived source accepts 22 tests at 22:43:10. Saved-pair relative error is `0.0368395` with single-product emulation, `6.79126e-5` with three products and `3.68270e-5` with six, versus FP32 `3.73549e-5` and FP64 `3.66208e-5`; every pair fails the unchanged tolerance. Repeats are exact finite. Result and retained verification: `PVC/run-preparation/ns-operand-precision-saved-pair-20261118T2243`. No CUDA adapter, efficiency result or production authority exists. |
| NS step-budget quality-cost hypothesis | CUDA diagnostic and independent terminal complete; historical controls fail | Both opposite-order workers exit/clean up zero. Analysis publishes at 05:37:38; controller records completion/restoration at 05:37:54. All 144 cross-process, 144 same-input and 192 graph-versus-compiled comparisons are exact finite. Ten-step compiled BF16 differs from pinned eager in 40/40 comparisons, relative error 0.036052-0.036098; historical controls fail. All six arms fail 24/24 continuous/resumed pair checks despite faster five/seven-step kernels. Original independent terminal completes at 05:47:02 with exact restoration; queue completes 4/4 phases at 05:47:10 and is absent at 05:49:38. Full-model launch remains gated. Result: `PVC/diagnostics/ns-budget-20261118/comparison.json`; terminal: `PVC/run-preparation/ns-budget-terminal-independent-20261119.json`. |
| NS-budget full-model preparation | Distinct CPU/archive-verified configuration candidate; GPU-denied | At 01:05:55, 16 archived tests pass with 124 authenticated helpers/three fixtures; a fresh archived checker accepts the receipt. Eighteen family/mode/budget cases reserve lower-LR 24/22/24 continuation diagnostics; 54 configurations reserve equal three-point full-budget LR grids. Candidate `PVC/candidates/ns-budget-full-model-alias-corrected-20261119T0104` preserves pinned-root/trainer arithmetic and rejects training before configuration/data/CUDA. Receipt: `PVC/run-preparation/ns-budget-full-model-cpu/20261119T0105/receipt.json`. Original NS diagnostic terminal, distinct full-model owner and actual-config all-rank/captured-update/work gates remain required. No full-model launch, quality result, elapsed-efficiency result or cloud history exists. |
| NS-budget actual-config validator component | Distinct CPU/source-archive accepted; full-model ownership unissued | At 01:35:09, 28 archived tests pass with 128 authenticated helpers; fresh archived checking passes. The integrated suite passes 104 tests in 185.30 seconds. All 18 lower-LR cases retain continuous24/interrupted22/resumed24 and checkpoints22/23/24; acceptance uses unchanged boundary22/post-cached-root24, scalar-loss and fused/null capture/update/work gates. Receipt: `PVC/run-preparation/ns-full-model-validation-component-cpu/single-pass-20261119T0132/receipt.json`. Candidate execution still rejects before configuration/data/CUDA. Original NS diagnostic terminal/restoration, a distinct accepted full-model owner and actual GPU checks remain required; no production or cloud history exists. |
| NS-budget full-model execution owner | Separate corrected CPU/source archive and fresh check accepted; GPU unissued | Candidate `PVC/candidates/ns-full-model-owner-single-pass-20261119T0257` preserves the owner-only transformation and all 128 accepted validator helpers, adding five. All 69 archived tests pass in 281.59 seconds; all 145 integrated tests pass in 338.08 seconds. Receipt: `PVC/run-preparation/ns-full-model-owner-cpu/single-pass-20261119T0300/receipt.json`; fresh archived-entry checking passes at 03:06:42. Earlier 66-case acceptance is preserved unchanged. Original NS terminal, queue/owner absence, exact keeper handoff and unchanged actual-config GPU gates remain required. No frozen binding, controller, full-model run, optimizer result or cloud history exists. |
| NS-budget full-model validation queue | Terminal `needs_attention` at 05:59:09; original queue absent at 06:00:54; GPU unissued | Receipt `PVC/run-preparation/ns-full-model-queue-cpu/20261119T0318/receipt.json` accepts 25 archived tests/136 helpers at 03:20:06. Original waiter `1875861/936425460` refuses the completed NS predecessor's historical-control gate before any freeze/check/start/terminal phase, preserving 0/4 phases. Its failure and original archives are retained; no restart or bypass occurs. Eighteen lower-LR cases/54 eight-rank commands remain unexecuted correctness diagnostics. |
| NS-budget full-budget study protocol and analyzer | Distinct CPU/source archive accepted; production unissued | Receipt `PVC/run-preparation/ns-budget-study-cpu/data-bound-20261119T0352/receipt.json` accepts 19 archived tests/136 helpers; 61 integrated tests pass. All 54 configurations retain equal search allowance, complete shared five-window loss/timing reports, independently tuned matched-null/MuonT comparisons and late-mixture-only provisional LR choices. Lower-LR acceptance cannot authorize center/upper; per-LR source-bound GPU gates, production ownership, worker cleanup/restoration, verified private cloud histories, symmetric boundary expansion and paired seeds3/4/5 remain required. No measured study run or optimizer improvement exists. |
| NS-budget 54-configuration actual-LR validator | Distinct CPU/source archive accepted; separately prepared owner CPU-accepted, GPU-unissued | Receipt `PVC/run-preparation/ns-actual-lr-validator-cpu/single-pass-20261119T0419/receipt.json` accepts 21 archived tests/141 helpers; 68 integrated tests pass. All 54 configurations reserve continuous24/interrupted22/resumed24, 162 eight-rank commands and 1,358,954,496 diagnostic tokens. Boundary22/post-cached-root24, scalar-loss, complete-state and fused/null captured-update/work gates are unchanged. No actual-LR GPU execution, production wave or cloud history exists. |
| NS-budget actual-LR original terminal handoff | Read-only CPU/source archive accepted; original full-model queue still live | Receipt `PVC/run-preparation/ns-actual-lr-predecessor-cpu/20261119T0437/receipt.json` accepts 13 archived tests/148 helpers; 44 integrated tests pass. Fresh archived observation at 04:40:29 reports `ready: false` with exact queue `1875861/936425460` live. It requires the original complete terminal/source/54-command/18-decision/cleanup/restoration evidence, without new tensor reconstruction or production authority. |
| NS-budget actual-LR execution owner | Distinct CPU/source archive and fresh archived check accepted; GPU-unissued | Candidate `PVC/candidates/ns-actual-lr-owner-20261119T0504` changes only the owner adapter. Receipt `PVC/run-preparation/ns-actual-lr-owner-cpu/20261119T0505/receipt.json` accepts all 74 archived tests in 292.92 seconds and 153 helpers, preserving all 148 parent helpers. All 93 integrated tests pass in 300.79 seconds; fresh archived-origin checking exits zero by 05:16. A separate queue is accepted below. Original lower-LR terminal/queue absence and exact restoration remain required before freeze or transfer; no binding, controller, GPU diagnostic, production wave or cloud history exists. |
| NS-budget actual-LR validation queue | Terminal `needs_attention` at 06:07:02; original queue absent at 06:08:51; GPU-unissued | Original waiter `1945601/937186048` stops with 0/4 phases on `ValueError: NS full-model queue source is absent or aliased`. Its nested lower-LR verifier expects `test_ns_full_model_queue.py` beside the imported helper, but the actual-LR archive omits that file; the original lower-LR archive retains it. Earlier 27 archived/120 integrated passing tests and their accepted receipt remain historical. Archive repair alone cannot supply the missing lower-LR terminal or waive the failed historical-control gate. No restart, GPU diagnostic, production wave or cloud history exists. |
| NS compiler-boundary follow-up | Original-training-path attempt stops `needs_attention` at13:23:14; 44/54 commands retained, 14/18 decisions: six accepted, eight rejected | All six MuonT lower-LR BF16/FP32 five/seven/ten-iteration cases pass checkpoints22/24 exactly and 39/39 scalar-loss checks. Fused BF16/five, seven and ten iterations fail checkpoint24 with419/428/460 failed tensors and3/39,2/39,4/39 failed loss checks. Fused FP32/five, seven and ten iterations fail checkpoint24 with377/379/378 failed tensors and1/39,1/39,2/39 failed loss checks. Matched-null BF16/five and seven reject checkpoint24 with409/448 failed tensors and2/39,1/39 failed losses while null-identity outer passes; seven also has seven scalar/structure differences. BF16/ten continuous24/interrupted22 are retained, but resumed exits1 before training on `Marked process generation changed during handoff`; four decisions remain unrecorded. Original controller/validator are absent; nonzero execution/cleanup and zero keeper restoration are recorded, with restoration independently checked. The actual-LR queue stops `needs_attention` at13:36:09 with zero phases because the complete original terminal/report is unavailable. Separate localization numerical/execution archives accept22/23 CPU cases; all90 integrated tests pass. The distinct predecessor archive accepts25 staged CPU tests at13:20:07 with191 helpers/559 inventory entries; its13:26:33 archived observation returns `ready: false`. GPU allocation owner and CUDA localization results are unissued. The local marker-exit correction passes36 CPU tests and is undeployed. Incomplete predecessor coverage supplies no seal or production authority. Compilation, graphs, numerical rejections and tolerances stay unchanged. No new full-budget quality or efficiency result. |
| Training-path independent terminal waiter | Terminal `needs_attention` at13:32:52; absent at13:37; no canonical terminal issued | Receipt `PVC/run-preparation/ns-training-path-followup-cpu/closure-corrected-20261119T0847/receipt.json` retains19 passing archived tests and all149 issued helper bytes. Waiter `2189311/938341032` acknowledges08:47:34 and retains nineteen900-second observations, last successful13:17:51. It fails after original-owner absence because complete execution/cleanup/restoration predicates are unsatisfied; no terminal command is issued. Its original evidence is preserved without restart. No allocation transfer or training/production authority. |
| Training-path actual-LR successor | Distinct numerical component CPU/source accepted at 09:27:15; fresh archived check passes | Receipt `PVC/run-preparation/ns-training-path-actual-lr-component-cpu/20261119T0926/receipt.json` retains 29 passing archived tests, 162 helpers and 531 inventory entries. Read-only audits verify six pinned training files, exact LR/operand/budget transformations for all three families and two complete parent archives. A distinct training-path predecessor/owner/binding and per-configuration GPU validation remain required; the failed lower-LR/actual-LR queues remain unchanged. Proposed allowance: three LRs per family/operand/budget case, 162 bounded commands before any full-budget launch. No successor GPU execution or production result exists. |
| Training-path actual-LR original terminal handoff | Distinct predecessor CPU/source accepted at09:53:20; original attempt and follow-ups have failed, no seal issued | Receipt `PVC/run-preparation/ns-training-path-actual-lr-predecessor-cpu/20261119T0953/receipt.json` retains twelve passing archived tests,169 helpers and538 inventory entries. All114 integrated tests pass. Canonical independent terminal, complete54-command/18-case evidence, retained rejections, original owner/waiter/descendant absence and exact restoration remain required before the one-shot seal. Original-owner absence does not supply missing completion evidence; the actual-LR queue stops13:36:09 before all six phases. No GPU execution or production authority. |
| Training-path actual-LR owner-only adapter preparation | Distinct source candidate and CPU archive accepted; 43 integrated CPU tests pass; execution owner/controller unissued | Candidate `PVC/candidates/ns-training-path-actual-lr-adapter-20261119T1018` changes only `experiment/fused_attribution_owner.py` at 10:19:09. Receipt `PVC/run-preparation/ns-training-path-actual-lr-adapter-cpu/20261119T1020/receipt.json` accepts fourteen archived tests, 171 helpers and 542 inventory entries at 10:21:54; fresh archived checking passes. All 43 integrated numerical/adapter CPU tests pass in 168.93 seconds. All 169 parent helpers, six training files, five numerical readers and 54 actual-LR configurations are preserved. Real unowned trainer/validator invocations refuse before outputs. No predecessor seal, execution owner/controller, GPU binding or launch is issued. |
| Training-path actual-LR separate execution owner | Distinct CPU/source archive accepted at10:55:01; fresh archived check passes; GPU unissued | Receipt `PVC/run-preparation/ns-training-path-actual-lr-execution-owner-cpu/20261119T1051/receipt.json` accepts53 archived tests in188.76seconds,177 helpers and548 inventory entries. All53 local tests pass in186.89seconds. The accepted adapter candidate and all171 parent helper bytes remain unchanged. Freeze requires an explicit original-terminal seal hash; visible ranks use accepted live hidden-parent proof and identity inventory, without hidden-only checks. No seal, binding, GPU transfer, diagnostic launch or production result. |
| Mixed-exit-bracket transition diagnostic | Terminal `needs_attention` at November 19 19:59:22; 23/54 completed commands and 7/18 decisions | Six MuonT BF16/FP32 five/seven/ten-step controls pass. Fused BF16/five-step continuation rejects; seven-step resumed execution fails before training on ownership-marker scan exhaustion, so no seven-step decision exists. Exact restoration is independently verified at 20:06:15. Original owners and both follow-up waiters are absent; the actual-LR queue stopped with 0/6 phases. The distinct scan-exit successor below preserves this failure; complete successor coverage and terminal remain required. |
| Missing-environment scan-exit diagnostic | Failed November19 22:03:52; 8/54 commands and2/18 decisions; exact restoration independently verified | BF16/five and seven-step MuonT controls pass all existing gates. Ten-step resumed execution fails before training in the complete ownership scan; its execution/cleanup exits are1/1, not a numerical decision. The original waiter stops22:12:06. Read-only22:12:53 checking verifies owners/descendants absent and exact keeper restoration. No complete terminal, production or cloud result. |
| Scan-exit actual-LR predecessor reader | Separate CPU34/193-helper archive accepted November 19 21:32:26; archived check passes; waiting for original owners | Pinned file-only reader preserves all four failed coverage summaries and accepted/rejected decisions without tensor rehashing. Read-only 21:33:45 observation finds original controller, validator and waiter live. Numerical/execution archives remain unmerged; no canonical seal, successor execution owner or GPU command is issued. |
| Scan-exit actual-LR adapter and execution owner | Separate CPU16 adapter and CPU53/203-helper execution archives accepted November19 21:53/21:59; archived checks pass; GPU unissued | Numerical candidate/configuration/reader bytes remain unchanged. The declared execution handoff changes to the accepted scan-exit source and preserves its original numerical bytes as a fixture. All54configs/162commands retain the original schedule and tolerances. The distinct47-case/214-helper queue archive accepts22:05; its required complete predecessor terminal is unavailable. No queue intent, successor seal or GPU launch exists. |
| Recheck-exit diagnostic | Failed November19 22:59:49; 1/54 commands and0/18 decisions; exact restoration independently verified | CPU136/source acceptance and separate seal/checks pass before acknowledged launch. Sustained marked-helper churn exhausts the original three scans in the interrupted command. Execution/cleanup exits1/1 remain; waiter stops23:09:30. Fresh23:11:03 checking verifies original owners/waiter absent and exact restored keeper. No complete terminal or numerical case acceptance. |
| Serialized whole-ownership diagnostic | CPU141 accepted; acknowledged launch November19 23:51:10; failed23:54:13 with0/54 commands and0/18 decisions | First continuous command fails before training on the new30-second lock wait. No deadline, scan-attempt or tolerance change follows. Controller records restoration23:54:30; independent23:56:15 checking verifies owner/validator-marker absence and exact keeper3598689 on the original eight A100s. Whole-inspection serialization is insufficient within this bound. No optimizer-quality or cloud result. |
| Serialized read-only terminal waiter | CPU24/191-helper archive accepted November19 23:52:31; stopped `needs_attention` November20 00:09:18 | Waiter3598600/start943780809 stops on its original900-second schedule without a canonical terminal claim. Fresh00:10:16 checking finds its consumed handle and both original owners absent, with the same exact restored keeper. The earlier23:54:17 observation remains historical. No restart or new GPU authority. |
| Query-serialized ownership diagnostic | CPU142/193-helper archive accepted November20 00:11:29; controller/validator acknowledged00:24:51/00:27:05; validating3/54commands0/18decisions | Source authentication and acknowledgment wait remain outside the unchanged30-second lock; hostname/query/complete-scan and final-generation statements retain exact order inside. First MuonT BF16/five-step lower-LR continuous24/interrupted22/resumed24 commands have zero execution/cleanup exits. The first numerical case comparison remains pending. No production wave or cloud result. |
| Query-serialized read-only terminal waiter | CPU24/195-helper archive accepted November20 00:26:45; acknowledged00:27:52 | Waiter3638945/start943982783 verifies both exact original owners live at00:27:57,0/54commands0/18decisions, then retains its900-second schedule. Complete coverage, clean execution/cleanup, actual owner/descendant absence and exact keeper restoration remain required before its canonical terminal. No keeper transfer or new GPU authority. |
| Transition actual-LR successor | Adapter and execution CPU/source archives accepted; independent archived-entry check passes by November 19 18:30; GPU unissued | The distinct candidate retains 54 configurations/162 commands and unchanged numerical gates. Execution receipt `PVC/run-preparation/ns-training-path-transition-actual-lr-execution-cpu/20261119T1822/receipt.json` accepts 53 unique tests in 255.05 seconds, 191 helpers and 565 inventory entries at 18:27:09. All 179 accepted adapter helpers and candidate bytes remain unchanged. Freeze and GPU launch additionally require the complete transition terminal and caller-bound predecessor seal. |
| Precision-placement CUDA component | Distinct 49-case CPU/source archive accepted November 19 20:31:01; CUDA unissued | Receipt `PVC/run-preparation/ns-precision-placement-execution-cpu/closure-reviewed-20261119T2032/receipt.json` retains 163 helpers and 531 inventory entries; fresh archived checking and 113 integrated CPU tests pass. Missing transitive static imports are rejected without relying on sibling existence. An independently accepted exclusive owner, completed predecessor gates and compiled/graph CUDA evidence remain required. |
| Start-accuracy predecessor evidence seal | One-shot seal completed at 19:53:29; exact original restoration verified | Archived sealer hashes 17 canonical evidence files, including both full retained root-stage output files, and verifies original owners/descendants absent and exact healthy keeper `1547502/933501005`, ranks `1547638` through `1547645`, on all eight original A100s. Receipt: `PVC/run-preparation/root-start-predecessor-seal-20261118.json`. Subsequent checks authenticate canonical file identities and small receipt hashes; the seal grants no historical attribution or production acceptance. |
| Proposed Waves 13-15, full-budget attribution LR study | GPU-denied after continuation failure; no production candidate, release or launch | Three LR waves each compare MuonT, fused normalization and identity-outer matched null, with balanced assignments across `.9`/`.80`/`.144` and `.139` untouched. Controller/preparer/Bash-monitor implementation remains mutable and unissued. Saved-state CPU diagnosis, a distinct corrected candidate and complete unchanged-tolerance GPU validation are required before production acceptance. Each successor also waits for preceding complete histories, large-window analysis, cloud verification and exact restoration; settings and gates are in "Full-Budget Attribution Wave Plan" below. |

## Summary

Completed full-budget waves 0-4 and 7-10, including the seed-2 MuonT
companion, are analyzed and full-resolution SwanLab-verified. Wave 0 uses
Toner's Marin PyTorch 130M recipe adapted to A100/FA2; each 0.5M run
processes 2,684,354,560 tokens. Outer-left down improves final C4 by
0.001343 nats at 5.04% extra end-to-end time. Across exploratory paired
seeds 0/1/2, MLP-diagonal output improves C4 by 0.006264/0.003738/0.003651,
mean 0.004551 nats. Selection-seed reuse and independent tuning remain
unresolved. Dense `inner_outer_all_lag` completes in Wave 3 and retains
the loss benefit at 31.3% extra total time. Wave 4's 1M pair improves
fixed-token loss at 14.53% extra main time.

Waves 5 and 6 stop at numerical validation and have no full-budget result.
Wave 5 fails integrated direction parity; Wave 6 fails same-source MuonT
post-cached-root continuation. Their failures and restored keepers remain
recorded. Wave 7's distinct fixed-age installation passes all four-arm
continuous-24 versus interrupted-22/resumed-24 gates. Its completed
inner/outer, outer and fused outer diagonal runs improve final C4 by
0.003831/0.002679/0.002188 nats at 9.98%/9.77%/8.86% extra main time.

Wave 8's outer/fused/combined arms improve final C4 by
0.002679/0.001800/0.002656 nats at 8.64%/7.66%/7.83% extra main time.
Five broad training windows and eleven shared late evaluations retain
small fixed-token gains. Combined fusion takes 9.87 seconds longer than
fused outer diagonal and worsens final/late selection loss. Retrospective
C4 crossing times on completed waves establish no elapsed-time advantage.

The initial twelve-trial LR study is cloud-verified at November 17
03:17:47; its four families choose the upper tested LR, approximately
0.032. The separately gated eight-trial upper expansion is complete and
cloud-verified at 07:42:10. All families choose approximately 0.064,
still the upper boundary of their five-point grid. Candidate final
selection gains of 0.006838-0.017122 nats cost 6.87-9.83% extra total
time. Short cooldown and seed-0 reuse prevent full-budget superiority
claims. Further actual-config LR gates and fresh paired seeds remain
required. The original upstream-product/null diagnostic dtype failure
remains preserved. Its separately identified corrected probe passes all
54 synthetic cases; the corrected full-model validator passes all 13
commands and 24 flags, with cleanup and restoration complete at 09:18:57.
Its original handoff queue independently accepts the evidence and exits
at 09:44:41.
Synthetic update savings do not establish whole-training efficiency, and
the distinct Wave-9 release is verified after all correctness gates pass.
Its full-budget launch is acknowledged at 10:27:56; original workers
complete and restore by 12:16:22, with bound process-absence verification
at 12:17:12. Five-window analysis and private full-resolution import
complete by 12:20:20. Upstream reduces the five window means by
0.001700/0.003395/0.002863/0.001848/0.001378 nats versus MuonT.
It saves 0.95% main time versus legacy, still costs 6.69% over MuonT,
and improves final selection over its matched null by only 0.000383.
The owned-output-factor successor initially fails exact factor equality;
the separately preserved full-trace correction passes all 14 GPU commands
and 26 checks, including zero-error all-arm continuation. Wave 10 then
completes all four full-budget histories and verifies all cloud scalars at
19:43:40. Every recorded same-arm loss matches Wave 9 exactly.
Legacy/upstream/null final C4 gains of 0.001800/0.002743/0.001814 nats
cost 8.40%/7.64%/8.03% extra main-function time versus MuonT.
Mixed sub-percent Wave-10-minus-Wave-9 timing changes establish no
whole-training speed gain from removing unused roots.
The separate seven-point
ceiling-LR expansion starts actual-config validation at12:27:07 but
fails its first command before updates at12:28:38 because the original
queue inherited a CPU-only CUDA mask. Keeper restoration succeeds;
failure evidence remains preserved, with no higher-LR quality result or
tuning-trial launch. The distinct visible-eight recovery passes all24
actual-config commands/eight continuation sets at14:00:51 and restores
at14:01:29. All eight tuning endpoints and five-window analysis complete
14:39:12, workers restore by14:38:48 and all eight cloud histories verify
14:45:56. The seven-point winners are interior:
MuonT/fused0.064, inner-outer/outer0.09051. These 320-update schedules
remain exploratory; they do not locate full-budget optima.

Wave 11's selected-LR actual-config validation passes all 12 GPU commands
and 30 checks, with zero continuation error for all four arms. Its
original production attempt fails before any training update because
the ownership check incorrectly expects Torch ranks to share the parent
session. Failed source, logs, worker acknowledgments and zero-update
receipts remain preserved. Wave 12 is a distinct recovery with the
same scientific settings, separately CPU-tested rank ownership and
serialized keeper allocation. Its 280-file/51-helper release freezes at
23:35:58 after original numerical reconstruction, completed Wave-10
result/cloud verification and exact four-node handoff. The frozen-source
check passes. One launcher starts at23:38:55, completes the required
original-evidence checks and acknowledges all four workers at23:59:43.
The November 18 00:00:10 release-bound SSH snapshot verifies all four
workers running, exact training children and completed keeper transfers.
The 00:25:18 bound SSH snapshot records 1,253/1,068/1,099/1,095
completed updates. The first shared 536,870,912-token window gives
inner-outer/outer/fused training deltas +0.051593/+0.061019/-0.005873
nats; fused's two late selection evaluations average +0.000195 nats.
Those interim results do not establish full-budget quality.
That interim Wave-12 observation is historical. Wave 12 subsequently
completed, its five-window results are recorded below, and the private
full-resolution SwanLab import was verified November 18 02:04:24.
The cross-wave 900-second document monitor remains the recurring updater.
No tested arm establishes independently tuned quality or whole-training
efficiency superior to MuonT.

Outer-left retains input MuonT's down factors and adds output-side work.
Down hybrid removes the large down-input factor and is the efficiency
hypothesis. It used less measured peak memory in wave 0, but was slower
overall and had worse loss than input MuonT.

## Research Questions and Evidence

For a local application $y=Wx$, the weight-gradient contribution is
$\delta x^{\mathsf T}$. Input MuonT uses the input second moment to shape
matrix updates. This endeavor asks whether the second moment of observed
output cotangents adds useful information.

The optimization and efficiency questions are separate:

1. Does output-side shaping improve fixed-token loss over independently
   tuned input MuonT on matched data and runtime?
2. Does replacing the 2048-dimensional down-input factor with a
   512-dimensional output factor improve measured time-to-target and memory
   without materially degrading loss?

The experiment follows the pinned Toner **Marin** implementation. nanoGPT,
the local six-layer ReLU Grug model, and a native Marin/JAX port are outside
this controlled comparison.

### Corrections to the Original Hypothesis

The [original proposal](artifacts/original-proposal.txt)
is preserved as a provenance text artifact. Its curvature, polar-geometry, and cost
claims are hypotheses requiring the following qualifications:

- $B=\mathbb E[\delta\delta^{\mathsf T}]$ is an uncentered observed-cotangent
  moment. It is not automatically a Fisher or generalized Gauss-Newton factor.
  Replacing joint input/output moments with a Kronecker product also requires
  a factorization approximation; cross-token terms can matter.
- An exact polar factor of a full-row-rank wide matrix has
  $PP^{\mathsf T}=I$. Preserving left singular vectors alone does not prove
  that output energy remains anisotropic. Outer input factors and finite
  Newton-Schulz can change the final direction's geometry.
- Toner's inner roots are unnormalized; only outer roots have unit RMS.
  Finite Newton-Schulz cannot be silently replaced with exact polar.
- Attention O and MLP down have different cotangents despite sharing output
  width. They require separate output factors.
- Shaped weight decay is a separate experimental choice. Factor
  normalization does not fix the final direction norm or effective step size.
- The original full hybrid replaces O and down factors. Executed
  `down_hybrid` replaces only down factors.
- C4 is validation data in this recipe. Training uses the pinned
  DCLM/StarCoderData/Proof-Pile-2 mixture.

The [source collection and technical review](artifacts/SOURCES.md) contains
the derivations, retrieved versions, and provenance. It reviewed Toner,
GO-MUON, K-FAC, empirical-Fisher limitations, Shampoo, SOAP, Mousse, Freon,
Spectrally Targeted Muon, Newton-Muon, and batch-size-dependent optimizer
comparisons. No GO-MUON author implementation was identified; a future
GO-MUON-form control must be labeled as a formula control.

## Program Status

| Work | Status | Evidence or Remaining Condition |
| --- | --- | --- |
| Primary-source collection | Complete | Ten parsed PDFs, 17 primary artifacts, pinned Toner snapshot |
| Exact data preparation | Complete | 2876 prepared shards; accepted training/C4 receipt |
| First-stage implementation | Complete | Five implemented arms; 20 seed-0 configurations across four batches |
| CPU and GPU correctness gates | Passed within documented scope | 18 CPU tests; synthetic A100/NCCL checks; real-data three-update continuation |
| Wave 0 | Complete | Four 0.5M-batch runs, 5120 updates each, seed 0 |
| SwanLab histories | Imported and verified | Original JSONL remains authoritative |
| Matched custom-down null control | Superseded by `null_all` | `null_all` ran in wave 1 and isolates numerical execution for the block path |
| Full-model post-cached-root restart | Wave-7 fixed-age control passes at 22:49:38 | Four-arm continuous24 versus22+resume24 at original tolerance; earlier dynamic-installation candidates retain their failures |
| Fair tuning and fresh-seed confirmation | Incomplete | Exploratory paired seeds 0, 1 and 2 completed; independent tuning and fresh confirmation remain |
| Larger-batch training | Wave-4 1M pair complete by 19:28 | MLP-diagonal minus same-wave MuonT: final C4/selection -0.003958/-0.003630; +14.53% main time; 2M/4M remain unrun |
| Wave 1: `inner_outer_all`, `left_all`, `inner_outer_all_diag`, `null_all` | Complete 2026-11-16 | All four reached 5120 updates; see [Wave 1](#wave-1-all-block-arms) |
| `null_left_all` | Defined, not implemented | Matched control for `left_all`; not in wave 1 |
| Wave 2: `inner_outer_hidden`, `inner_outer_mlpdiag`, `inner_outer_leftdiag`, `outer_left_all` | Complete 2026-11-16 | `inner_outer_mlpdiag` has lowest final selection among these four arms: -0.006888 versus wave-0 seed-0 MuonT; see [Wave 2](#wave-2-cheaper-output-geometry) |
| Leading exploratory geometry | `inner_outer_mlpdiag` | C4 and selection improvements in three exploratory paired seeds; roughly 19-23% more end-to-end time; not a tuned superiority result |
| Cheaper exploratory geometry | `inner_outer_leftdiag` | Versus wave-0 seed-0 MuonT, final selection improves by 0.004131 at +10.4% per update; no dense output eigendecomposition; seed-1 confirmation deferred |
| Batched block update | Implemented, first used by wave 2 | `block_update_stack`; wave 1 ran the per-matrix path |
| Wave 3: `inner_outer_mlpdiag` seed 1, `muont` seed 1, `null_mlpdiag`, `inner_outer_all_lag` | Complete; all four uploaded and cloud-verified at 2026-11-16 17:03:57 | Fresh process snapshots, full histories, final checkpoint receipts, and `swanlab-upload-wave3.json` |
| Wave 4 | Complete by 19:40:41; analyzed and cloud-verified 19:43:47 | Four full-resolution histories and final eight-rank checkpoints; seed-2 paired comparison complete at 21:13 after companion completion |
| Seed-2 MuonT companion | Complete 21:12:34; process absent and keeper restored 21:13:00; cloud-verified 21:13:46 | All 5120 updates and five broad windows; seed-2 candidate final C4/selection -0.003651/-0.003609 at +22.53% total time |
| Wave 5 fused-diagonal candidate | Frozen candidate failed integrated A100 direction parity at 19:38:46 | CPU45 and standalone reduction probe pass; failure preserved, keeper restored at 19:39:04; no real-data 24-versus-22+resume execution |
| Wave 5 full-budget launch | Not launched; queue exited 19:48:59 on `needs_attention` | No allocation transfer; corrected candidate requires a new preserved identity and all original gates |
| Wave-5 direction diagnostic | Executed 20:26-20:27; recorded, not a passed launch gate | Both ranks reproduce the old failure and fail production-shaped structured/Gaussian `tpu` updates; FP32 controls pass; keeper restored at 20:27:50 |
| Wave-5 sensitivity follow-up | Executed 20:42-20:43; recorded, not a passed launch gate | Twelve bitwise-equal repeats; inner-only and all-element one-ULP source controls fail, outer-only passes; FP64 moment/factor controls pass; keeper restored at 20:43:47 |
| Wave 6 outer-diagonal geometry | Frozen; synthetic gates pass; same-source MuonT post-cached-root continuation fails; keeper restored 21:46:32; not launched | Failed update-24 down-weight tensor comparison at unchanged tolerance; output-diagonal full-model arms not executed; preserved checkpoints require diagnosis/new validated identity |
| Wave 7 fixed-age host-refresh control | Complete and restored November 17 00:45:46; analyzed and cloud-verified 00:49:03 | Final C4 deltas -0.003831/-0.002679/-0.002188 at +9.98%/+9.77%/+8.86% main time; all four full-resolution histories verified |
| Wave-7 results and refresh diagnostic | Five shared 536,870,912-token windows and eleven exact-shared late evaluations | All logged rank-zero installation ages are one; no measured efficiency win; fused records 0.83% less time versus outer Torch with slightly worse final/late loss |
| Fixed-age LR follow-up | All12 trials analyzed03:16:37 and SwanLab-verified03:17:47; fresh complete/restored handoff passes03:18:10 | Every family chooses upper tested LR0.032; candidate selection gains0.002075/0.000988/0.000047 at+10.35%/+8.84%/+8.48% total time; separate upper-boundary expansion subsequently completes07:42:10; independent full-budget tuning/confirmation remain open |
| Fixed-age actual-LR GPU validator | All 12 trials/36 commands pass 02:27:20; cleanup and `.9` keeper restoration pass 02:27:53 | Bound execution helper rederives final trajectory/log identities at 02:32; controller and marked children absent; Wave-8's then-active `.139` restriction clears at 02:42:19 |
| Fixed-age LR execution helpers | Separate 25-helper execution binding frozen November 17 00:24:49; bound launch passes 02:47:21; campaign completes and monitor exits normally 03:34:33 | All twelve histories analyzed and verified; worker restoration and fresh handoff pass; replacement document monitor3776115 continues across waves |
| Upper-LR boundary expansion | All 24 commands/eight check sets pass; eight trials complete, workers restored by 07:31:32; five-window analyzed and all-eight cloud-verified 07:42:10 | All four families choose upper tested LR approximately 0.064; final selection gains 0.006838-0.017122 at +6.87-9.83% total time; grid-boundary optimum unresolved, independent full-budget tuning remains |
| Combined outer-update cost hypothesis | CPU diagnostic27 and production A100 probe54 cases pass, November 17 01:17:27 | Synthetic eager/graph update savings5.43-15.46%; three rather than four dense right products/matrix; not a full-model efficiency result |
| Wave 8 combined outer-diagonal candidate | All four5120-update histories complete; keepers restored by05:27:07; final five-window analysis complete; original cloud import verified05:38:37 | Final C4 deltas-0.002679/-0.001800/-0.002656 at+8.64%/+7.66%/+7.83% main time; combined costs9.87 seconds more than fused and worsens windows2-5/selection;5120 training/token points and22 evaluations per run verified |
| Wave-8 full-budget release helpers | Bound controller exits zero after four worker acknowledgments03:38:28; monitor3535620 exits normally after analysis/upload verification05:38:37 | Original25 numerical helpers and31-helper binding unchanged; strict monitor exits on complete05:30:31; cross-wave document monitor is restored after its05:33 concurrent-edit refusal |
| Upstream-product/null follow-up | Original controller runs after complete study; fails 07:50:50, `needs_attention`, zero complete parity cases/timings; keeper restoration returns zero | FP32/FP64 `torch.isclose` diagnostic bug; original source/result/intent preserved; distinct corrected diagnostic subsequently passes at unchanged tolerances; original failure establishes no parity or quality/efficiency result |
| Corrected upstream-product/null diagnostic |54 cases/six timings pass08:09:10; controller execution/cleanup/restoration zero08:09:31 | Exact pinned null/input parity; FP64 outer max absolute error2.522152e-8; synthetic update savings9.16-47.41%, not whole-training efficiency; original failure preserved |
| Upstream-product full-model follow-up | Reviewed 288-file validator CPU108/33; frozen43 helpers/24 gates/13 commands; separate45-helper queue stops08:06:35 on original probe failure | Original identity has no controller intent/GPU launch/production release; distinct corrected validator subsequently starts08:33:11, with its own evidence preserved separately |
| Corrected full-model validator | All13 commands/24flags and controller reconstruction pass; restored09:18:57; original queue validates09:44:40/exits09:44:41 | Fresh SSH verifies original queue/reconstruction processes absent; exact all-arm continuation and restored keeper accepted; distinct Wave-9 release subsequently passes09:57:58 |
| Wave 9 upstream-product systems comparison | Complete/restored by12:16:22; original-worker absence verified12:17:12; five-window analyzed/full-resolution SwanLab-verified12:20:20 | Legacy/upstream/null final C4 deltas-0.001800/-0.002743/-0.001814 at+7.71%/+6.69%/+7.77% main time; upstream saves0.95% versus legacy but worsens selection; matched-null final selection gain only0.000383; no tuned or elapsed-time superiority |
| Owned outer-factor follow-up | Distinct numerical candidate CPU137; formatted observation successor CPU140 at12:18:56; all source/archive identities verified | Construct28 locally used outer vectors instead of224 full output projections; retain all pooled moments/reductions and checkpoints; no unused inner roots; GPU ownership/parity/continuation/timing/launch remain open |
| Owned-output full-model validation successor | Distinct formatted candidate CPU316 at13:55:41; separate29-helper controller CPU40 at13:58:24; validation-only296-file/29-helper binding freezes14:40:08 | Read-only prerequisite refuses14:49:42 because restored keepers inherit LR worker markers; all-node14:52:23 diagnosis verifies training handles/child markers absent and exact healthy keepers; original binding preserved, no GPU state/transfer/queue/production release; distinct tested correction required |
| Owned-output exact-keeper handoff successor | Candidate CPU316, controller CPU74/27 helpers; separate14-command GPU validator launches15:27:23; first command passes, second fails15:30:25 on exact same-moment outer factors | Rank1:440/2048 elements differ, max1.19209e-7; no real-data trajectory; terminal needs_attention15:30:44, child/validator exit1, keeper restore0; fresh15:32 original-process/child-marker absence and healthy keeper346192 verified; separate numerical correction/validation identity required |
| Owned-output full-trace successor | Distinct297-file/six-change candidate CPU324/controller CPU79; own binding15:53:21, controller347647 acknowledged15:56:09, validator348068 starts15:57:42; eight-rank synthetic24 passes16:00:56 | Full-shaped trace restores exact factors; width2048 sliced trace differs162/192 comparisons, corrected normalization/outer exact192/192; fresh16:02:59 SSH verifies3/14 zero-exit commands,18/26 saved flags, MuonT interrupted22 running; four-arm continuation/reconstruction/restoration and separate production release remain pending |
| Wave 10 owned-output systems follow-up | Production/ownership helpers CPU130 and scoped cached Ruff/Black pass November17 16:24; no production release or worker launch | Original corrected validator still live with11/14 zero-exit commands, null continuous24 running; separate owner-only candidate CPU verification, all14 commands/26 gates, all-rank reconstruction and exact keeper restoration precede freeze/launch |
| Wave-10 isolated-owner successor | First full candidate CPU444pass/2fail preserved; distinct ownership-module correction and focused production/owner/ledger CPU216 pass16:34 | Saved original GPU14/14 zero-exit commands and26/26 flags; controller reconstruction/restoration and isolated candidate CPU/archive gate remain pending; no launch or measured result |
| Wave-10 formatted isolated-owner preparation | Isolated candidate CPU446 verified16:39:27; generated import lint issue preserved; distinct formatted generator/focused218 tests and scoped Ruff/Black pass16:42 |299 candidate files/exactly four ownership changes; separate formatted CPU/archive receipt and original GPU-controller reconstruction/restoration still required; document monitor scheduled success16:41:59 |
| Wave-10 production readiness | Formatted299-file/four-change candidate CPU446/archive verified16:45:32; original controller passed/restored16:42:49, fresh absence/exact keeper verified16:44 | Generated Ruff/Black and focused218 pass; separate production freeze independently rederives14 commands/26 gates/all-rank histories and restoration before any worker acknowledgment; full-budget results/upload remain absent |
| Wave-10 guarded release acceptance | Pending freeze stopped before publication16:51; no GPU process interrupted; exact node-lock/generation/evidence checks added to mutable controller | Corrected subprocess regressions17 pass and focused production/owner/ledger227 pass16:59; original read-only reconstruction371020 confirmed absent16:58; independent guarded freeze/review still pending, no launch/result/import |
| Wave-10 separate production release | Guarded freeze publishes17:21:07 after independent14-command/26-gate all-rank reconstruction and final exact-keeper/evidence reprobe |299 files/41 frozen helpers,446-test candidate CPU/archive;1197 tracking cases plus20 subtests and12 offline uploaders pass; frozen-source check and one-shot launch still required, no training result/import |
| Wave-10 four-arm production launch | Frozen check passes; all-four workers acknowledged and launch recordslaunched17:42:57; fresh release-bound SSH17:43:27 verifies exact running generations |104/48/13 MuonT/legacy/upstream updates, matched null running/first scalar pending; original document monitor3776115/run monitor132158 retain900-second cadences; no complete shared536870912-token window/final result/import |
| Wave-10 first scheduled running analysis | Release-bound snapshot 17:53:32 verifies all four original workers; 630/512/461/406 updates; frozen analyzer writes its interim artifact | No complete shared 536,870,912-token window, final outcome or verified cloud receipt; original document/run monitors remain live at 17:55 without restart |
| Wave-11 selected-LR full-budget transfer | Preparation hypothesis and equal four-arm budget documented November 17 17:58; original fixed-age source and seven-point selection provenance retained | Distinct candidate CPU/archive verification, actual-config GPU continuation and separate generation-bound release remain required; no GPU job, queue or new measured result |
| Wave-10 first strict shared-window analysis | Original document monitor captures all bytes 18:15:04 and refreshes 18:15:39; first 536,870,912-token train deltas -0.001553/-0.001700/-0.000686 versus MuonT | Prior scheduled analyzer's concurrent JSONL refusal preserved; all four workers running, final five-window/runtime/cloud evidence pending |
| Wave-11 provenance-corrected preparation | Distinct successor issued 18:15:12 after frozen import/archive/all-scalar cloud acceptance passes; generated lint passes | First CPU59 identity preserved; corrected successor full CPU/archive verification pending; GPU execution denied and no queue or production result |
| Wave-11 corrected CPU/archive verification | Corrected successor CPU59 in 21.43 seconds; zero exit and exact archive/provenance recheck verified 18:20:34; latest tracking suite149 passes | 276 candidate files/exactly nine changes match first candidate bytes; both original receipts preserved; actual-config GPU validation and production ownership remain unissued |
| Wave-10 two-window common-prefix milestone | Strict scheduled capture 18:30:44 retains 2,587/2,317/2,267/2,202 updates; all four share two complete 536,870,912-token windows and four exact late evaluations | Bound SSH 18:34:30 verifies original workers running at 2,772/2,496/2,451/2,382; steady training-step time about 2.310/2.305/2.305 versus 2.127 seconds/MTok; full-budget result, final efficiency, cleanup and cloud verification pending |
| Wave-10 three-window common-prefix milestone | Strict manual capture begun 18:59:09 retains 4,081/3,706/3,665/3,593 updates; three complete 536,870,912-token windows and seven exact shared late evaluations | Bound SSH 19:00:59 verifies original workers running at 4,162/3,778/3,748/3,680; output-arm steady step-time overhead 8.45%/8.10%/8.21%; final result, cleanup and cloud verification pending |
| Wave-10 four-window common-prefix milestone | Strict manual capture recorded 19:11:27 retains 4,725/4,294/4,260/4,193 updates; four complete 536,870,912-token windows and eight exact shared late evaluations | Original scheduled SSH19:10:41 all workers running; steady times2.301/2.292/2.296 versus2.123 seconds/MTok; no final outcome, restoration or cloud verification |
| Wave-11 actual-config validation ownership preparation | Distinct278-file/four-change successor prepared19:02:50; CPU68/21.98 seconds and source/archive verification19:06:46 | Both issued LR parents unchanged;12 planned GPU commands/30 numerical flags/100663296 tokens remain unexecuted; original worker/keeper binding and terminal preflight recording corrections under CPU regression verification, no binding/queue/launch |
| Wave-11 guarded controller CPU/archive verification | Corrected34-helper controller CPU46/3.86 seconds, integer-zero exit and exact log/archive verified19:17:37; broader affected CPU162/16.29 seconds | Original worker acknowledgment and restored full-generation checks plus terminal preflight failure state covered; no validation binding/GPU gate/queue/production release; completed Wave10 remains required |
| Wave-10 final systems rerun | All four original workers complete/restored by19:31:12; bound SSH19:33:01 complete; five-window analysis19:32:53 and all-four private cloud verification19:43:40 | Every same-arm recorded loss matches Wave9; mixed sub-percent cross-wave timing, no measured ownership speed gain; final C4 gains0.001800/0.002743/0.001814 cost+8.40%/+7.64%/+8.03% main time; run monitor exits normally, document monitor continues |
| Wave-11 original selected-LR validation launch |278-file/34-helper validation-only binding freezes19:24:33; frozen check passes; one controller246595 acknowledged19:47:16, validator252177 starts19:50:28 after fresh all-node/cloud acceptance |12 commands/30 flags planned; first MuonT continuous24 exits zero by19:53:24; no complete arm continuation gates, restoration, production release or full-budget result |
| Wave-11 terminal validation and production acceptance | Original all12 commands/all30 checks independently reconstructed; terminal state passed20:48:44 with validation/cleanup/restoration0; fresh20:49:12 original-process absence and exact keeper371646 verified | Release candidate CPU98 and48-helper controller CPU52 archives verified; one guarded production freeze is running at20:54, with no release, worker intent or full-budget selected-LR result |
| Wave-11 separate production release | Guarded freeze publishes280-file/48-helper release21:14:28 after repeat12-command/30-check original GPU reconstruction, Wave10 result/import verification and final exact node-specific allocation acceptance | Frozen release check running; no worker acknowledgment, full-budget selected-LR result or import yet; short-study LR transfer remains exploratory |
| Wave-11 preserved pre-update production failure | Frozen check passes and single launch acknowledges four workers21:37:06; all four training children fail early rank-session ownership21:36:45-21:37:25, before any manifest/update | All worker statesneeds_attention/training1/cleanup1/restoration0; fresh all-node SSH21:39 verifies original processes/marked diagnostics absent and exact healthy keepers; issued candidate/release/logs unchanged, distinct corrected owner and source-bound release required |
| Wave-11 rank-session-corrected successor | Distinct candidate prepared 21:47:27; CPU100/21.73 seconds and exact source/archive verification pass 21:53:17 | 280 source files/four changes from validation parent; only owner and owner tests differ from failed production source; original run monitor records all-terminal needs_attention 21:51:34 and exits 21:51:38; separate post-failure allocation acceptance, evidence reconstruction and one-shot release remain required, no new training or upload |
| Wave-12 distinct post-failure recovery | All four original workers complete/restored at5,120updates; independent five-window/22-shared-evaluation reconstruction passes; all four full-resolution private SwanLab histories verified November18 02:04:24 | Fused finalC4/selection deltas-0.002462/-0.002349nats cost9.02%extra main time; inner/outer selection+0.042416/+0.031214; recovery of zero-update Wave11 failure, no measured efficiency or tuned/fresh-seed superiority |
| Selected-LR detached stage-profile validation | Distinct285-file/43-helper validation-only binding freezes02:20:27; candidate CPU112/controller CPU38 accepted; exact eight-A100 off-resume launch verified02:37:40; original controller736872/validator737971 live02:39:58 | Receipts02:31:29/02:38:06 record MuonT continuous/off-resumed exit0;2/12commands complete,220-row histories and200+20 resumed segments/log identities verified; profiling-on result, all-rank parity, CUDA traces and final cleanup/restoration pending |
| Ceiling-LR expansion | Original queue accepts Wave9/starts controller12:27:07; first command fails12:28:38 with CPU-only inherited CUDA mask; controller exits/restores12:28:55 | Zero updates/zero passed check sets; failed original identity preserved, children absent/keeper4086492 healthy; no tuning trial or LR-quality result; mutable guards tested, distinct recovery and all original gates still needed |
| Visible-eight ceiling recovery | All24 actual-config commands/eight check sets pass; all eight trials complete/five-window analyzed14:39:12; workers restored by14:38:48; all-eight cloud verification14:45:56 | Seven-point interior winners MuonT/fused0.064, inner-outer/outer0.09051; higher-LR same-LR selection gains0.032859/0.022354/0.014333 cost+11.46%/+8.66%/+8.19% total time; original campaign monitor independently analyzes/verifies/exits normally, document monitor continues900-second cadence |
| Standalone diagonal probe | Complete on A100 at 19:34:40 | Eight eager reduction workloads, 6.16-6.89x local speedup; communication, EMA, compiled capture and whole training excluded |
| Exploratory elapsed-time crossings | Completed-history diagnostic, November 16 | Actual sparse C4 evaluations at retrospective targets 3.59/3.52/3.46; no measured elapsed-time win; not launch-based confirmation |
| Historical Wave-5-bound matrix-LR study (`lr-0p5m-20261116`) | 12 configs prepared November 16 19:22; not validated or queued | Separate from the active `lr-fixed-age-0p5m-20261116` study; original backend and GPU/configuration gates remain blocked by failed Wave-5 evidence |
| Full hybrid / mechanism campaign | Planned | Separate implementation and validation gates |

The implemented `inner_outer_down` arm was not included in wave 0.
No new GPU job, deployment, restart, or cloud upload is authorized by this
document consolidation.

## Conclusion

The Muon and input MuonT runs match the scientific configuration of the pinned
Toner **Marin PyTorch 130M, 0.5M-token batch, 1x-budget recipe**: architecture,
initialization, data plan, tokenizer, global batch, optimizer hyperparameters,
input-statistic settings, and learning-rate schedule. They are **A100 runtime
adaptations**, not exact H100/FA3 reproductions.

The two output-side runs intentionally change the MLP down-projection update.
They are experiments built on input MuonT, not additional Toner reproductions.
Their down-update numerical implementation also differs from the input-MuonT
control. A small improvement cannot yet be attributed exclusively to output
geometry.

All four jobs are complete. Each processed 5,120 updates and 2,684,354,560
input tokens. SSH process checks on all four assigned nodes found no remaining
`experiment.train` or `torch.distributed.run` process during this audit.
SwanLab contains historical imports, not live training.

The observed output-side benefit is small: outer-left down's final C4 loss
is 0.0013431602340343574 below input MuonT's, with 5.04% longer recorded
end-to-end time. One seed does not establish a repeatable improvement.

## Model

| Setting | Value |
| --- | --- |
| Architecture | Pinned Llama-style, pre-RMSNorm, sequential attention and SwiGLU residual blocks |
| Recipe label | `130m`; not the total parameter count |
| Transformer layers | 32 |
| Hidden width | 512 |
| MLP expansion width | 2,048 |
| Attention heads / head dimension | 8 / 64 |
| Attention projections | Separate Q, K, V, O; no linear biases |
| MLP projections | Gate/up: `(2048, 512)`; down: `(512, 2048)` |
| Context | 4,096 tokens |
| Vocabulary | 128,256 |
| RoPE theta | 10,000 |
| RMSNorm epsilon | `1e-5` |
| Dropout | None; attention dropout is 0 |
| Residual / computation precision | BF16 residuals and autocast; FP32 parameters and norm intermediates |
| Transformer matrices and norms | 134,251,008 parameters |
| Embedding plus output head | 131,334,144 parameters |
| Total trainable parameters | **265,585,152** |
| Embedding/head tying | Separate parameters, initially equal in value |

Initialization uses the pinned `initialize_model`: linear weights draw from
a standard normal truncated to `[-3, 3]`, then divide by `sqrt(fan_in)`;
RMSNorm weights are one. For an unscaled embedding draw `E0`, initialization
is `embedding = E0 / sqrt(512)` and `head = E0 / sqrt(512)`. The head copies
the raw draw before embedding scaling; it is not scaled twice.
The same seed and initialization procedure are used across all arms.

This is not Marin's separate six-layer ReLU Grug model or Toner's nanoGPT
benchmark. Changing either model would change the experiment.

## Data and Tokenization

Training uses DCLM, StarCoderData, and Proof-Pile-2. **C4 is validation data,
not the training corpus.** Source files, their order, document-prefix quotas,
and expected counts come from the pinned `reproduce/marin/data_plan.json`.

| Component | Repository | Prepared Shards | Prepared Tokens | Documents |
| --- | --- | ---: | ---: | ---: |
| DCLM | `mlfoundations/dclm-baseline-1.0` | 1,200 | 23,378,120,924 | 18,701,696 |
| StarCoder | `bigcode/starcoderdata` | 863 | 1,584,860,368 | 1,624,000 |
| Proof-Pile | `EleutherAI/proof-pile-2` | 242 | 423,006,576 | 97,728 |
| C4 validation | `allenai/paloma`, `c4_en` | 6 | 967,614 | 2,091 |

Training data alone has 25,385,987,868 prepared tokens in 2,305 files.
Training plus C4 has 2,311 files. Full preparation produced all 2,876 files
in the upstream plan, including other Paloma subsets, but these runs evaluated
only C4 plus held-out training-source windows.
Prepared counts can exceed each source's nominal budget because the unchanged
preparer finishes fixed document batches/prefixes. The counts above match the
plan's actual per-file expectations; the excess is not extra training budget.

| Tokenizer Setting | Value |
| --- | --- |
| Repository | `meta-llama/Meta-Llama-3.1-8B` |
| BOS / EOS | 128000 / 128001 |
| File format | Little-endian uint32 token streams, `.bin` |
| Encoding | Tokenize `text + " " + eos_token`, with special tokens enabled |
| Prepared directory | `PVC/data/` |
| Accepted receipt | `PVC/data/experiment_receipt.json` |

The prep work used the unchanged Toner tokenizer/preparer; switching worker
counts and local scratch/PVC copying did not change that code. Full verification
checked names, sizes, token-ID bounds, document counts, and generated SHA256s.
This audit rechecked receipt identity and all required file sizes/timestamps;
it did not reread and rehash the whole approximately 102 GB corpus.
Toner did not publish original cache content hashes, so generated receipts
prove local integrity and plan compliance, not byte identity to unavailable
original token caches.

### Mixture and Packing

The pinned mixture weights are DCLM `3.8`, StarCoder `0.25`, Proof-Pile `0.055`.
The loader truncates weighted counts in a block of 2,048 sequences and assigns
the remainder to DCLM. This is the upstream behavior.

| Component | Sequences per 2,048-Sequence Block | Actual Share | Tokens Processed per Run |
| --- | ---: | ---: | ---: |
| DCLM | 1,897 | 92.626953125% | 2,486,435,840 |
| StarCoder | 124 | 6.0546875% | 162,529,280 |
| Proof-Pile | 27 | 1.318359375% | 35,389,440 |

Shards are sorted by filename and concatenated logically. Each source is
partitioned into 4,096-token windows, with a deterministic seeded permutation.
The first 64 permuted windows from each source are excluded from training and
retained for held-out evaluation. Each mixture block has its own seeded
permutation. Global sequence IDs assign the same data to matched updates across
all arms; reducing microbatch size only repartitions those IDs.

Attention is causal and document-separated at EOS boundaries. EOS belongs
to the document preceding it; the next token starts a new attention segment.
Each window's final next-token target is masked with `-100`. Other targets
are the next token in the packed window: an EOS-to-next-document transition
is scored, even though attention resets at that next token. No additional
loss mask is applied at document boundaries.
Windows can span token shards and cut across documents; this is fixed-length
token packing, not one document per sequence.

## Batch and Training Budget

| Setting | Executed Value |
| --- | ---: |
| World size | 8 GPUs per run |
| Global batch | 128 sequences |
| Microbatch | 4 sequences per GPU |
| Gradient accumulation | 4 microbatches per update |
| Local sequences per update | 16 |
| Tokens per GPU microbatch | 16,384 |
| Global tokens per update | **524,288**, the `0p5m` label |
| Optimizer updates | **5,120** |
| Global sequence windows processed | 655,360 |
| Logged input-token budget | **2,684,354,560** |
| Unmasked next-token targets | 2,683,699,200, due to one masked target per window |
| Training-budget multiplier | `chinchilla_multiplier: 1`, the upstream 1x recipe |

Batch identity is `8 GPUs * 4 sequences * 4 accumulation * 4096 tokens`.
Loss is averaged within each microbatch, divided by accumulation, and backward
gradients are averaged by DDP across ranks. DDP synchronization occurs on the
last accumulated backward. There is no mixed-precision GradScaler.

Toner specifies microbatch 8 and accumulation 2 at this same global batch.
Our smaller microbatch preserves data/budget mathematically but changes
BF16 gradient accumulation and kernel shapes; bitwise-identical gradients or
training trajectories are not established.

## Optimizer and Schedule

The seven 2D block matrices per layer, 224 matrices total, use Muon or MuonT.
Embeddings, the head, and RMSNorm parameters use fused AdamW.
Clipping with threshold 1.0 applies to this auxiliary group, **not** the block
matrices, matching the pinned Muon/MuonT routing.

| Setting | `muon` | Other Three Arms |
| --- | ---: | ---: |
| Block-matrix base LR | 0.016 | **0.02262741696** |
| Auxiliary AdamW base LR | 0.0048 | 0.0048 |
| Muon momentum | 0.95 | 0.95 |
| Nesterov | Enabled | Enabled |
| Block-matrix decay coefficient | 0.1 | 0.1, geometry-shaped as described below |
| Head decay | 0.1 | 0.1 |
| Embedding and RMSNorm decay | 0 | 0 |
| Auxiliary AdamW betas | `(0.8, 0.98)` | `(0.8, 0.98)` |
| Auxiliary AdamW epsilon | `1e-15` | `1e-15` |
| Newton-Schulz iterations | 10 | 10 |
| Newton-Schulz epsilon | `1e-5` | `1e-5` |
| Polynomial coefficients | `(3.4445, -4.7750, 2.0315)` | Same |
| NS arithmetic | `ns_mode: "tpu"`: BF16 GEMM operands, FP32 result | Same |

The LR difference between Muon and MuonT is intentional and comes directly
from the pinned recipes. No additional transfer multiplier is applied.
Block direction scaling is `sqrt(max(1, fan_out / fan_in))`; it is 2 for gate/up
and 1 for square/down matrices.

Every optimizer group's LR follows the same factor. For zero-based update `s`:

```text
factor(s) = 1                              for 0 <= s < 1024
factor(s) = 1 - (s - 1024) / 4096          for 1024 <= s <= 5119
lr(s) = base_lr * factor(s)
```

There is no warmup. The first 1,024 updates are stable, followed by linear
decay over 4,096 configured updates, with minimum ratio 0.
At the last applied update (`s=5119`), the factor is `1/4096`, not zero;
zero would be reached at the next, unexecuted index `s=5120`.
The first decay-phase update, `s=1024`, still has factor 1.

### Input Toner

`muont`, `outer_left_down`, and `down_hybrid` retain these input settings:

| Setting | Value |
| --- | --- |
| Input second-moment EMA | 0.938 |
| Trace-normalized damping | 0.001 |
| Inner / outer inverse powers | `(0.5, 0.125)`; not `(0.25, 0.25)` |
| Samples per sequence | 286 |
| Exact sampling positions | `range(32)` plus `range(33, 4096, 16)` |
| Samples per GPU per update | 4,576 |
| Samples across all GPUs per update | 36,608 |
| Capture | Upstream recomputed RMSNorm/attention/SwiGLU sampled rows |
| Root backend / start point | Fast, `begin: "head"` |
| Statistic chunks | `chunk: 0`: one buffer for all accumulated microbatches |
| Initial exact-root updates | 20 |
| Cached-root iterations | 2, with up to 3 repair rounds |
| Eigenbasis refresh cadence | 4 updates, asynchronous and staggered |
| Refresh flagged bases | Enabled |
| Root epsilon / maximum rho | `1e-4` / `0.2` |
| Tolerance kind / fraction / minimum | `floor` / `0.06666666666666667` / `5e-6` |
| Host workers / threads | 2 / 4 |
| Root CUDA graphs | Enabled |
| Overlap | 1: priority side stream overlaps the last backward |
| Ownership | Equal contiguous layer ownership; four layers per rank |

Input statistics update every training update. Cached roots are candidates
for the current statistic; "refresh 4" refers to asynchronous eigenbasis
refresh, not simply holding every input factor fixed for four updates.
Inner roots are unnormalized; outer roots are Frobenius-normalized to
unit RMS. Q/K/V share an input statistic, gate/up share one, attention O has
one, and down has its own 2,048-dimensional input statistic.

The hybrid arm removes **down input** capture, statistics, roots, and host
work; the other input-Toner groups remain. Final input fallback counters
on the logging rank are MuonT 19, outer-left 13, hybrid 0.
These are exact-eigendecomposition fallbacks after repair, not failed updates.
They are rank-0 counters, not totals across eight ranks.

### Arm Definitions

The four executed arms are `muon`, `muont`, `outer_left_down`, and
`down_hybrid`. Muon is the familiar baseline; the definitions below specify
the three non-Muon arms, including their weight decay and numerical factors.
Additional subsections define implemented but unexecuted `inner_outer_down`
and the wave-1 arms `inner_outer_all`, `left_all`, `inner_outer_all_diag`,
and the control `null_all`.

Only down matrices get output factors. Their 32 matrices contain 33,554,432
weights, about 12.63% of all trainable parameters and 25% of the transformer
body. Thus this wave tests a targeted intervention, not full-model two-sided
normalization. Other parameters can still change indirectly through training.

#### Shared Orientation, Momentum, and Update

For each block projection, suppress the layer index and write

$$
y = Wx,\qquad W_t \in \mathbb{R}^{m\times n}.
$$

Rows are output coordinates; columns are input coordinates. Down projections
have $m=512$ and $n=2048$. Let $G_t$ be the globally averaged, accumulated
weight gradient at zero-based update $t$. The executed momentum recurrence
and Nesterov source are

$$
M_t = \mu M_{t-1} + G_t,\qquad
U_t = G_t + \mu M_t,\qquad
\mu=0.95,\quad M_{-1}=0.
$$

This momentum recurrence does not include a $(1-\mu)$ factor or bias correction.
Given gradient-driven direction $D_t$ and decay direction $Q_t$, the update is

$$
\boxed{
W_{t+1} = W_t - \eta_t\left(s_{m,n}D_t+\lambda Q_t\right)
},\qquad
s_{m,n}=\sqrt{\max(1,m/n)},\quad \lambda=0.1.
$$

All three non-Muon arms use the block base LR `0.02262741696` and the schedule
specified above. For down projections, $s_{m,n}=1$. Both $D_t$ and $Q_t$ use
pre-update weights/statistics; the decay term is separate from the loss gradient.
Embeddings, the output head, and RMSNorm parameters remain under auxiliary
AdamW, not these matrix rules.

#### Input and Output Second Moments

Let $x_{t,j}\in\mathbb{R}^{n}$ denote a sampled projection input and
$\delta_{t,j}\in\mathbb{R}^{m}$ its sampled output cotangent. For down
projections, $x$ is the SwiGLU activation and $\delta$ is observed at the
down projection output, before residual addition. The layer-wise EMAs are

$$
\begin{aligned}
C_t &=
\beta C_{t-1}+
(1-\beta)\frac{1}{N_t}\sum_{j=1}^{N_t}x_{t,j}x_{t,j}^{\mathsf T},\\
B_t &=
\beta B_{t-1}+
(1-\beta)\frac{1}{N_t}\sum_{j=1}^{N_t}
\delta_{t,j}\delta_{t,j}^{\mathsf T},
\end{aligned}
\qquad
\beta=0.938,\quad C_{-1}=B_{-1}=0.
$$

Here $j$ indexes samples pooled across all microbatches and ranks, not only
the statistic-owning GPU. The 286 positions per sequence are `range(32)` plus
`range(33,4096,16)`, giving $N_t=128\cdot286=36608$ per update.
FP32 raw sums and counts pool **before** the EMA.

More precisely, if $\ell_{t,r,a}$ is a rank's microbatch-mean loss divided
by accumulation, whose backward is executed by the runner, the collector uses

$$
\delta_{t,j}
=65520\,\frac{\partial\ell_{t,r,a}}{\partial y_{t,j}},
\qquad
65520=4\cdot4095\cdot4.
$$

This removes microbatch-loss averaging and accumulation scaling. It is the
cotangent of the unaveraged sum of valid losses in that microbatch, not just
the derivative of the single loss at sampled position $j$; downstream
attention can couple positions. These are **uncentered second moments**,
not centered covariances. $B_t$ is an observed-gradient moment, not assumed
to equal a Fisher matrix or Hessian.

Both EMAs consume the current update's observations before the parameter
update. They are not bias-corrected before trace normalization; any positive
scalar EMA bias correction cancels from that normalization.

Every rank holds the full model and independent local output-sum/count buffers
for all 32 down projections. Output EMAs and factors are sharded by contiguous
layer ownership: four layers per rank. Counts are all-reduced; sums use
`reduce_scatter_tensor` so each owner receives globally pooled sums for its
layers. Each owner updates its four EMAs and computes their output roots.
It uses those factors for its four owned down-matrix updates, then all-gathers
parameter deltas so every model replica applies all 32 down updates.
Thus capture buffers are replicated while persistent output moments/roots
and down momentum are owner-local.

#### Inner and Outer Factors

For a PSD moment $F\in\mathbb{R}^{d\times d}$ with positive trace, define

$$
A_d(F)=\frac{dF}{\operatorname{tr}(F)}+\epsilon I_d,\qquad
\epsilon=10^{-3},
$$

$$
\mathsf{In}_d(F)=A_d(F)^{-1/2},\qquad
\mathsf{Out}_d(F)=
\frac{\sqrt d\,A_d(F)^{-1/8}}{\left\|A_d(F)^{-1/8}\right\|_F}.
$$

Powers are spectral matrix powers, not elementwise powers. The inner factor
is **unnormalized**; the outer factor has unit RMS:
$\|\mathsf{Out}_d(F)\|_F^2/d=1$.

Write $R_{\mathrm{in},t}$ and $R_{\mathrm{out},t}$ for the **executed** input
factors, targeting $\mathsf{In}_n(C_t)$ and $\mathsf{Out}_n(C_t)$ respectively.
The input backend uses initial exact roots, then cached-eigenbasis
approximations, repair, and exact fallbacks as specified above. It targets
the current $C_t$ every update; input factors are not simply held for four
updates.

Output roots are recomputed by FP32 eigendecomposition on updates
$0,4,8,\ldots$, and held between refreshes:

$$
L_{\mathrm{in},t}=\mathsf{In}_m(B_{\tau(t)}),\qquad
L_{\mathrm{out},t}=\mathsf{Out}_m(B_{\tau(t)}),\qquad
\tau(t)=4\lfloor t/4\rfloor.
$$

For numerical safeguards, the output implementation symmetrizes $B$,
trace-normalizes it, clamps computed negative eigenvalues to zero, and then
adds damping. Zero-trace output statistics return identity factors.
The input exact-root routine instead normalizes clamped eigenvalues by
their mean before damping; for PSD positive-trace moments this has the same
target definition, but it does not explicitly provide the same zero-trace
identity fallback.

#### Finite Newton-Schulz Map

Let $\mathcal N_{10}(Z)$ denote the actual ten-step routine, **not an exact
polar decomposition**. Start with

$$
X_0=\frac{Z}{\|Z\|_F+10^{-5}},
$$

transposing $X_0$ if $m>n$. For $k=0,\ldots,9$, compute

$$
A_k=X_k\star X_k^{\mathsf T},\qquad
X_{k+1}=3.4445X_k+
\left[-4.7750A_k+2.0315(A_k\star A_k)\right]\star X_k.
$$

Here $\star$ denotes GEMM with each operand rounded to BF16 and an FP32
output, matching `ns_mode: "tpu"`; polynomial arithmetic is FP32.
Transpose back afterward if needed. Exact-polar identities must not be
silently substituted for this finite, rounded map.

#### Input MuonT (`muont`)

For all seven block projections $q,k,v,o,\mathrm{gate},\mathrm{up},\mathrm{down}$,

$$
\boxed{
D_t=\mathcal N_{10}(U_tR_{\mathrm{in},t})R_{\mathrm{out},t},
\qquad
Q_t=W_tR_{\mathrm{out},t}^{\,2}.
}
$$

Input factors act on the **right**, both inside and outside Newton-Schulz.
Decay uses the square of the RMS-normalized outer input factor.

#### Outer-Left Down (`outer_left_down`)

Only for MLP down projections,

$$
\boxed{
D_t=L_{\mathrm{out},t}
\mathcal N_{10}(U_tR_{\mathrm{in},t})R_{\mathrm{out},t},
\qquad
Q_t=W_tR_{\mathrm{out},t}^{\,2}.
}
$$

This adds a left output factor **after Newton-Schulz** to input MuonT.
There is no left factor inside Newton-Schulz and no left-shaped decay.
$L_{\mathrm{in},t}$ is computed and stored but unused by this arm's update.
The other six block projections remain input MuonT.

#### Down Hybrid (`down_hybrid`)

Only for MLP down projections,

$$
\boxed{
D_t=L_{\mathrm{out},t}
\mathcal N_{10}(L_{\mathrm{in},t}U_t),
\qquad
Q_t=L_{\mathrm{out},t}^{\,2}W_t.
}
$$

This **replaces** down input normalization with output normalization; it
does not combine both. Down input capture, statistics, and roots are removed.
The other six block projections remain input MuonT, so this is not a
full-model output-only or two-sided optimizer.

#### Inner-and-Outer Down (`inner_outer_down`, Unexecuted)

The fifth implemented arm also modifies only down projections:

$$
\boxed{
D_t=L_{\mathrm{out},t}
\mathcal N_{10}(L_{\mathrm{in},t}U_tR_{\mathrm{in},t})R_{\mathrm{out},t},
\qquad
Q_t=W_tR_{\mathrm{out},t}^{\,2}.
}
$$

It adds both inner-left and outer-left factors to input MuonT while retaining
input-shaped decay. The other six projections remain input MuonT. It has
configurations and correctness checks, but no full-budget training result.

#### Two-Sided All Blocks (`inner_outer_all`, Wave 1)

Defined and implemented on 2026-11-16 in
[`experiment/block_toner.py`](experiment/block_toner.py); scheduled for wave 1.
This is the full Bi-MuonT arm: the `inner_outer_down` rule applied to all
seven block projections of every layer, 224 matrices in total.

For layer $\ell$ and projection
$p\in\{q,k,v,o,\mathrm{gate},\mathrm{up},\mathrm{down}\}$, write
$y_p=W_px_p$ with $W_p\in\mathbb R^{m_p\times n_p}$:

| Projection $p$ | $m_p\times n_p$ | Input Statistic (Shared, Unchanged) | Output Cotangent Observed At | $s_{m,n}$ |
| --- | --- | --- | --- | ---: |
| $q,k$ | $512\times512$ | Q/K/V input, $C^{\mathrm{attn}}$ | Projection output, before head reshape and RoPE | 1 |
| $v$ | $512\times512$ | Q/K/V input, $C^{\mathrm{attn}}$ | Projection output, before head reshape | 1 |
| $o$ | $512\times512$ | Attention output, $C^{o}$ | Projection output, before residual addition | 1 |
| gate, up | $2048\times512$ | MLP input, $C^{\mathrm{mlp}}$ | Projection output, before SwiGLU | 2 |
| down | $512\times2048$ | SwiGLU activation, $C^{\mathrm{down}}$ | Projection output, before residual addition | 1 |

**Input side.** Input statistics, their sharing, the fast root backend,
and $R_{\mathrm{in},p,t},R_{\mathrm{out},p,t}$ are exactly those of input
MuonT. Here $R_{\cdot,p}$ denotes the factors of the statistic containing $p$.

**Output side.** Each projection has its **own** output moment; Q/K/V and
gate/up do not share output statistics because their cotangents differ.
That gives seven output moments per layer: five $512\times512$ and two
$2048\times2048$. For each $p$, define $\delta_{p,t,j}$, $B_{p,t}$, and the
roots exactly as in
[Input and Output Second Moments](#input-and-output-second-moments) and
[Inner and Outer Factors](#inner-and-outer-factors), with $m=m_p$:

$$
L_{\mathrm{in},p,t}\approx\mathsf{In}_{m_p}(B_{p,t}),\qquad
L_{\mathrm{out},p,t}\approx\mathsf{Out}_{m_p}(B_{p,t}).
$$

**Output factors target the current update's moment $B_{p,t}$ at every
update**, exactly as input factors target $C_t$. They are not held between
refreshes. They are executed by Toner's fast root backend, treating the
output moments as additional statistic groups with the **same settings as
the input side** (see [Input Toner](#input-toner)):

- 20 initial exact-root updates, then cached-eigenbasis roots with 2
  iterations and up to 3 repair rounds, and exact fallback after repair.
- Asynchronous, staggered host eigenbasis refresh every 4 updates, with
  flagged-basis refresh.
- Root epsilon `1e-4`, maximum rho `0.2`, `floor` tolerance with fraction
  `1/15` and minimum `5e-6`, and root CUDA graphs.
- Damping $\epsilon=10^{-3}$, which the fast backend fixes, and the split
  inverse powers $(1/2,1/8)$.

"Refresh 4" here refers to the asynchronous eigenbasis, as on the input
side. Every update's accepted roots are candidates for the current
$B_{p,t}$, within the backend's acceptance tolerance.

All seven output groups use the 286 sampled positions per sequence and
$\beta=0.938$. $\delta_p$ is the cotangent of the projection output itself,
so $G_p=\sum_j\delta_{p,j}x_{p,j}^{\mathsf T}$ holds for every $p$. For
$q,k$ this means the pre-RoPE cotangent.

**Output capture.** Each projection's backward stores its sampled
cotangent rows in an independent BF16 row buffer holding all microbatches
of the update, as Toner's input rows do with `chunk: 0`. BF16 is the
projection-output gradient dtype under autocast; the capture op raises on
any other dtype. Rows are multiplied by $2^{16}$, which is exact in BF16.
The 65520 cotangent scale of
[Input and Output Second Moments](#input-and-output-second-moments) is
not applied. Any positive constant scale cancels from the EMA's trace
normalization and leaves every factor unchanged. Diagnostics that report
$B$ in summed-loss units multiply by $(65520/2^{16})^2$. Toner's packed FP32
`syrk`, `reduce_scatter_tensor` to the layer owner, and EMA then run
unchanged on these buffers.

**Same-step timing.** Output statistics and roots begin on their own Toner
side stream once the last microbatch's backward has been issued, and the
optimizer step waits for them. As executed, the output work therefore
starts after the backward rather than per owner chunk during it.
Per-chunk pipelining is an unimplemented execution optimization; it would
not change the target $B_{p,t}$.

**Output engine.** The output statistics run in a second instance of
Toner's fast engine. It has its own CUDA side stream, its own NCCL
communicator, and its own host-refresh pool with the input settings (2
workers, 4 threads). Each rank therefore runs 4 refresh workers in the
two-engine arms.

**Update.** For every one of the seven projections,

$$
\boxed{
D_{p,t}=L_{\mathrm{out},p,t}\,
\mathcal N_{10}\!\left(L_{\mathrm{in},p,t}\,U_{p,t}\,R_{\mathrm{in},p,t}\right)
R_{\mathrm{out},p,t},
\qquad
Q_{p,t}=W_{p,t}R_{\mathrm{out},p,t}^{\,2},
}
$$

$$
W_{p,t+1}=W_{p,t}-\eta_t\left(s_{m_p,n_p}D_{p,t}+\lambda Q_{p,t}\right).
$$

Momentum, Nesterov source, base LR `0.02262741696`, schedule, $\lambda=0.1$,
and $s_{m,n}$ are unchanged from input MuonT. No per-arm LR retuning or
direction-norm grafting is applied before the fair-tuning stage. For tall
gate/up matrices, $\mathcal N_{10}$ transposes internally as defined.
$L_{\mathrm{in}}$ is then $2048\times2048$ inside, and $L_{\mathrm{out}}$
shapes the 512-dimensional column space of the finite-NS output.

Fixed design choices, chosen for comparability with `inner_outer_down`:

- **Decay stays input-shaped**, $Q=WR_{\mathrm{out}}^2$. Left-shaped or
  two-sided decay belongs to the direction/decay factorial, not this arm.
- **Numerical path.** All 224 matrices use the custom down path:
  owner-local momentum, per-matrix Newton-Schulz, direct FP32 root
  products, and separately formed direction/decay. Upstream's batched
  shape-group update is no longer used for any block matrix.
- **Ownership.** Per-layer output moments, roots, and momentum are owned
  by the existing contiguous layer owner, four layers per rank. Parameter
  deltas are all-gathered as in the down-only arms.

This output backend differs from the wave-0 down arms and the
implemented `inner_outer_down`: those use exact FP32 eigendecomposition
every 4 updates, held between refreshes. A down-only versus all-block
comparison must therefore use a `inner_outer_down` rerun with this
backend. Otherwise the comparison must be labeled as also changing the
output root backend.

Because every block matrix now changes numerical path, the arm requires
a matched null control, `null_all`. It runs the identical code path,
output capture, fast roots, host refresh, and scheduling, with
$L_{\mathrm{in},p}=L_{\mathrm{out},p}=I$ forced for all $p$. Its target is
input MuonT, and comparing it with upstream `muont` bounds the execution
difference. Gains of `inner_outer_all` are measured against `null_all`
first and upstream `muont` second.

Planning-level overhead relative to input MuonT, before the measurements
below:

| Item | Per Rank |
| --- | --- |
| Replicated packed output Gram partials, all 32 layers | $32(5\cdot\tfrac{512\cdot513}{2}+2\cdot\tfrac{2048\cdot2049}{2})\cdot4$ B $\approx$ 0.58 GiB |
| `reduce_scatter` payload per update | $\approx$ 0.58 GiB FP32 |
| BF16 sampled-row buffers, all four microbatches | $4576\cdot(5\cdot512+2\cdot2048)\cdot2$ B per layer, $\approx$ 1.8 GiB over 32 layers |
| Owned output state, 4 layers | EMA, basis, staging, and two roots, each $4(5\cdot512^2+2\cdot2048^2)\cdot4$ B $\approx$ 0.15 GiB; $\approx$ 0.75 GiB total |
| GPU root work per update | Cached-basis iterations for 20 width-512 and 8 width-2048 owned statistics |
| Host eigenbasis refresh per 4 updates | 20 width-512 and 8 width-2048 eigendecompositions, versus 12 and 4 for input MuonT |

Output roots add about twice input MuonT's dense root work. Width-2048
statistics dominate: three per layer instead of one. The two host workers
of the input setting may no longer finish refreshes within 4 updates.
Late refreshes leave older cached bases. That changes which roots are
accepted within tolerance and how often repair or fallback runs, not the
target $B_{p,t}$. Record per-group basis age and fallback counts. If
host capacity is raised, apply the same worker and thread settings to
the null control.

#### Left-Only All Blocks (`left_all`, Wave 1)

Defined and implemented on 2026-11-16; scheduled for wave 1.
This is the `down_hybrid` rule applied to all seven block projections of
every layer: output-side inner and outer factors replace the input factors
everywhere. It is the mirror image of input MuonT, with left factors in
place of right factors.

Projections, output cotangent locations, output moments $B_p$, roots
$L_{\mathrm{in},p},L_{\mathrm{out},p}$, output settings, and $s_{m,n}$ are
exactly those of [`inner_outer_all`](#two-sided-all-blocks-inner_outer_all-planned).
For every projection $p$,

$$
\boxed{
D_{p,t}=L_{\mathrm{out},p,t}\,
\mathcal N_{10}\!\left(L_{\mathrm{in},p,t}\,U_{p,t}\right),
\qquad
Q_{p,t}=L_{\mathrm{out},p,t}^{\,2}\,W_{p,t},
}
$$

$$
W_{p,t+1}=W_{p,t}-\eta_t\left(s_{m_p,n_p}D_{p,t}+\lambda Q_{p,t}\right).
$$

There are no input statistics, input capture, input roots, or input host
refresh work for any block matrix. Output capture, the fast root backend
and its settings, and same-step timing are those of `inner_outer_all`.
Toner's root backend serves only the output statistic groups. Momentum, Nesterov source, base LR `0.02262741696`, schedule, and
$\lambda=0.1$ are unchanged from `down_hybrid`, which keeps the input-MuonT
recipe values; this LR is untuned for left-only shaping. Decay is
left-shaped, mirroring input MuonT's $WR_{\mathrm{out}}^2$. All 224 matrices
use the custom per-matrix path and the layer ownership specified for
`inner_outer_all`.

For tall gate/up matrices, the left factors are $2048\times2048$:
$L_{\mathrm{in}}$ acts before the internal NS transpose, and $L_{\mathrm{out}}$
shapes the at most 512-dimensional column space of the finite-NS output.

Forcing $L_{\mathrm{in},p}=L_{\mathrm{out},p}=I$ gives $D=\mathcal N_{10}(U)$
and $Q=W$: a plain Muon-form update at the MuonT LR, not a MuonT null.
The matched control, `null_left_all`, runs that identity-left rule through
the same custom path, output capture, fast roots, and host refresh. Report `left_all`
against `null_left_all`, upstream `muon` at its own LR, and upstream `muont`
as separate baselines.

Wave-0 evidence for this rule is limited to down matrices, where
`down_hybrid` was 0.002358 nats worse than input MuonT on one seed.

Planning-level resources, relative to input MuonT: output-side capture,
communication, root work, and host refresh match `inner_outer_all`. All
four input statistics per layer are removed, three $512\times512$ and one
$2048\times2048$ FP32 moment, with their row buffers, roots, and host work.
The remaining width-2048 statistics are gate/up outputs: two per layer
instead of input MuonT's one.

#### Diagonal Two-Sided All Blocks (`inner_outer_all_diag`, Wave 1)

Defined and implemented on 2026-11-16; scheduled for wave 1.
This is the efficiency variant of
[`inner_outer_all`](#two-sided-all-blocks-inner_outer_all-planned). It keeps
the same update form on all seven block projections, but replaces every
input and output second moment by its diagonal.

For a PSD moment $F\in\mathbb R^{d\times d}$ with positive trace, let
$f=\operatorname{diag}(F)\in\mathbb R^d$. The trace is unchanged,
$\operatorname{tr}F=\mathbf 1^{\mathsf T}f$, so the normalization of
[Inner and Outer Factors](#inner-and-outer-factors) becomes elementwise:

$$
a=\frac{d\,f}{\mathbf 1^{\mathsf T}f}+\epsilon\mathbf 1,\qquad
\mathsf{In}^{\mathrm{diag}}_d(F)=\operatorname{Diag}\!\left(a^{-1/2}\right),\qquad
\mathsf{Out}^{\mathrm{diag}}_d(F)=
\frac{\sqrt d\,\operatorname{Diag}\!\left(a^{-1/8}\right)}{\left\|a^{-1/8}\right\|_2}.
$$

Powers are elementwise, $\epsilon=10^{-3}$, and the outer factor has unit
RMS. Zero-trace statistics return identity factors.

**Statistics.** Each input and output statistic keeps only per-coordinate
second moments, $\beta$-EMAs of $N_t^{-1}\sum_jx_{t,j}\odot x_{t,j}$ and
$N_t^{-1}\sum_j\delta_{t,j}\odot\delta_{t,j}$. Input grouping stays as in
input MuonT: Q/K/V share one input diagonal, gate/up share one, and o and
down each have their own. Output diagonals are per projection, seven per
layer. Sampled positions, BF16 sampled rows with the unapplied constant
cotangent scale, FP32 sums of squares pooled before the EMA, $\beta=0.938$,
and the inverse powers $(1/2,1/8)$ are those of `inner_outer_all`. Using
the same samples isolates the diagonal approximation; using every position
is a separate change.

**Roots.** Diagonal factors are computed exactly from the current update's
EMAs, $C_t$ and $B_{p,t}$, at **every** update, by elementwise powers. There
is no eigendecomposition, cached eigenbasis, repair, or fallback. This
matches the refresh target of `inner_outer_all`, whose input and output
roots both target the current moments every update, so factor age is not
a second difference between the arms.

**Update.** With
$\Lambda_{\cdot,p}$ the diagonal factors defined above in place of
$L_{\cdot,p}$ and $R_{\cdot,p}$, for every projection $p$,

$$
\boxed{
D_{p,t}=\Lambda^{\mathrm{out}}_{\mathrm{out},p,t}\,
\mathcal N_{10}\!\left(\Lambda^{\mathrm{out}}_{\mathrm{in},p,t}\,U_{p,t}\,
\Lambda^{\mathrm{in}}_{\mathrm{in},p,t}\right)
\Lambda^{\mathrm{in}}_{\mathrm{out},p,t},
\qquad
Q_{p,t}=W_{p,t}\left(\Lambda^{\mathrm{in}}_{\mathrm{out},p,t}\right)^{2}.
}
$$

The superscript is the statistic side, input or output; the subscript is
the inner or outer root. Every factor product is a row or column scaling,
done as one FP32 broadcast multiply. Momentum, LR `0.02262741696`,
schedule, $\lambda$, $s_{m,n}$, input-shaped decay, layer ownership, and
the per-matrix finite NS path are unchanged from `inner_outer_all`. This
way a loss difference between the two arms reflects the diagonal
approximation, not a change of NS execution or factor age. One residual
difference remains: full input and output roots come from Toner's
approximate fast backend, while diagonal roots are exact.

The diagonal arm does **not** contain input MuonT as a special case: its
input side is also diagonal. Read `inner_outer_all` versus
`inner_outer_all_diag` as the information cost of the approximation, and
`inner_outer_all_diag` versus `muon` and `muont` as the practical result.
Diagonal output moments are already listed under information controls;
this arm applies the diagonal approximation to both sides of all blocks.

Planning-level resources, not measured:

| Item | `inner_outer_all` | `inner_outer_all_diag` |
| --- | --- | --- |
| Capture per sampled row, width $d$ | $d^2$ outer-product terms | $d$ squares |
| Replicated output partials per rank | $\approx$ 0.58 GiB packed Grams | $32(5\cdot512+2\cdot2048)\cdot4$ B $\approx$ 0.8 MiB |
| Root computation | Cached-basis GPU iterations plus asynchronous host eigendecompositions, widths 512 and 2048 | Elementwise powers |
| Factor application per matrix | Dense products, $O(mn(m+n))$ | Scalings, $O(mn)$ |
| Newton-Schulz, 224 matrices | Unchanged | Unchanged |

Newton-Schulz remains the dominant optimizer cost, so step time cannot fall
below that of a Muon-form update on the same path. In wave 0, input MuonT
was only 1.6% slower end to end than Muon. The expected saving is
therefore mainly relative to `inner_outer_all`, not to input MuonT.
Batching NS by shape group helps every block-path arm equally. It is a
separate systems change, implemented for wave 2 and validated against the
per-matrix target: `block_update_stack` runs one batched update per shape
group per rank instead of one call per matrix, cutting kernel launches on
the update path from roughly $10^4$ to a few hundred per step. Momentum
matches the per-matrix path bitwise; the applied delta differs only by FP32
reassociation inside batched GEMMs, bounded at a relative $10^{-4}$ in
[the equivalence test](experiment/tests/test_block_toner.py). Wave-1 arms
ran on the per-matrix path, so wave-2 step times are not comparable to
wave-1 step times, while the optimizer mathematics is unchanged.

#### Cheaper Output Geometry (Wave 2)

Wave 1 establishes two facts that set up wave 2. Two-sided output factors
on all blocks help against their own matched null control, and the fully
diagonal arm loses most of that advantage while also discarding MuonT's
full input factors. Wave 2 therefore holds the input side at full MuonT
geometry in every arm and cheapens only the output side, so each arm is
`inner_outer_all` with one specific output economy. All four compare
directly against `inner_outer_all` and `null_all` at the same seed and
data order.

| Arm | Output factors | Hypothesis under test |
| --- | --- | --- |
| `inner_outer_hidden` | Full on $q,k,v,o,\text{down}$; none on $\text{gate},\text{up}$ | Width-2048 output geometry carries little of the gain |
| `inner_outer_mlpdiag` | Full at width 512; diagonal at width 2048 | Width-2048 output needs only its scale, not its rotation |
| `inner_outer_leftdiag` | Diagonal at both widths | Output scale alone carries the gain, once input stays full |
| `outer_left_all` | Outer factor $\hat{L}_{\text{out}}$ only, both widths | Wave 0's best arm, `outer_left_down`, generalized to all blocks |

The first three cut output cost directly: `inner_outer_hidden` removes the
two largest Gram matrices and their eigendecompositions, which is where the
output refresh backlog concentrates; `inner_outer_mlpdiag` keeps those rows
but replaces their roots with elementwise powers; `inner_outer_leftdiag`
removes every output eigendecomposition. `outer_left_all` keeps the full
output statistics and tests the algebra instead, since wave 0's best arm
applied only the outer factor. Its decay term becomes
$\hat{L}_{\text{out}}^2 W$ with no inner left product, matching the
left-shaped decay already used by the down-only arms.

#### Lagged Left Factors (`inner_outer_all_lag`, `inner_outer_mlpdiag_lag`)

This arm computes exactly `inner_outer_all`'s factors and applies them one
update late. At update $t$ the left factors come from output cotangent rows
of update $t-1$:

$$
D_t=\hat{L}_{\mathrm{out}}^{(t-1)}\,
\mathcal N_{10}\!\left(L_{\mathrm{in}}^{(t-1)} U_t R_{\mathrm{in}}^{(t)}\right)
R_{\mathrm{out}}^{(t)} .
$$

The input side stays current; only the left side lags. The first update has
no earlier rows and therefore applies input factors only, which is one
update in 5,120.

The motivation is latency, not geometry. Output rows exist only once
backward has finished, so the current implementation launches output
statistics and roots after the backward pass and then waits for them inside
the same update. That wait sits on the critical path, and it is where the
measured 0.19 s of output-engine cost lands, including the roughly 6 exact
eigendecomposition fallbacks per update. Lagging by one update moves the
collection point to just before the *next* launch, so the work has an entire
update plus the following forward and backward to complete instead of only
the update's own kernel-issue window. If the wait is what costs 0.19 s rather
than the arithmetic itself, this arm recovers most of it while applying the
same factors one update later.

Two implementation facts make this correct rather than merely faster. The
cached-root path returns CUDA-graph output buffers that the next replay
rewrites, so factors that outlive their update are copied out; removing that
copy makes [the lag test](experiment/tests/test_block_toner.py) fail.
And an in-flight launch cannot be serialized, so `runtime_state` collects it
before saving: without that, a resumed run would apply factors one update
older than a continuous run and the continuation gate would diverge.

The lag was first implemented only for a dense output engine, and a guard
rejected combining it with a diagonal output side. That excluded exactly the
pairing worth having once wave 2 finished: the lag is a latency change and
`inner_outer_mlpdiag` is the best-performing geometry, so the two are
complementary rather than alternatives. `inner_outer_mlpdiag_lag` now
combines them.

Making that combination meaningful required deciding what lag means for a
diagonal factor. A diagonal has no launch to hide, so lagging it buys no
time; leaving it current would have given the arm two different left-factor
ages. The arm instead applies **one** lag to every left factor: the diagonal
is computed from this update's rows at the same point the dense launch is
issued, then held for the next update.
[The mixed-lag test](experiment/tests/test_block_toner.py) pins that both
sides advance together, and fails if only the dense side is lagged.

Staleness here is one update at statistics momentum 0.938, whose effective
averaging window is about 16 updates, so the lagged moment differs from the
current one by roughly one window-fraction. This is a far smaller lag than
the preconditioner staleness that Shampoo-family methods accept routinely.
It is still a real change to the update, so this arm is not a pure
efficiency control: a loss difference against `inner_outer_all` measures the
cost of the staleness, and the pair together separate latency from geometry.

No implemented output-side candidate applies global direction-norm grafting.
The equations
specify the update algebra using executed factors and the finite NS map,
not bitwise equivalence between implementations: upstream MuonT uses
batched NS, FP16-split FP32-accurate root products, and fused direction/decay;
the custom down path uses per-matrix NS, direct FP32 products, and separately
formed direction/decay.

Implementation references: [down updates](experiment/optimizer.py),
[output factors](experiment/factors.py), [output collection](experiment/capture.py),
[training ordering and loss scaling](experiment/train.py),
[pinned MuonT engine](artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/src/toner/fast/engine.py),
[pinned input roots](artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/src/toner/fast/roots.py),
and [pinned Newton-Schulz routine](artifacts/external-sources/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/src/toner/fast/muon.py).

## Executed Runs Mapped to Their Update Equations

Every executed run, with the exact factors each projection receives. This is
a reference table: the derivations are in [Arm Definitions](#arm-definitions)
above, and nothing here is a new claim.

### Notation

For one matrix $W$ with gradient $G_t$, let $A$ be its trace-normalized,
damped **input** second moment and $B$ its **output** cotangent second
moment, and write

$$
R=A^{-1/2},\quad
\hat R=\operatorname{unitRMS}\!\left(A^{-1/8}\right),\quad
L=B^{-1/2},\quad
\hat L=\operatorname{unitRMS}\!\left(B^{-1/8}\right).
$$

Inner factors are unnormalized; only the outer pair is scaled to unit RMS.
With momentum $M_t=\mu M_{t-1}+G_t$, Nesterov source
$U_t=G_t+\mu M_t$, the 10-step finite Newton-Schulz map
$\mathcal N_{10}$, learning rate $\eta$, decay $\lambda$ and
$s=\max(1,\text{rows}/\text{cols})^{1/2}$, every block arm computes

$$
\Delta W_t=-\eta\left[\,s\,\hat L\;
\mathcal N_{10}\!\left(L\,U_t\,R\right)\hat R
\;+\;\lambda\,W_t\hat R^{2}\right].
$$

Absent factors are the identity. When $\hat R$ is absent the decay term
becomes $\lambda\,\hat L^{2}W_t$, which is the left-shaped decay; when both
outer factors are absent it is plain decoupled decay $\lambda W_t$.

### Per-Projection Moments

The input side has four statistics per layer and the output side seven,
because $q,k,v$ consume the same activations and so share one input moment,
as do $\text{gate},\text{up}$, whereas every projection has its own output
cotangent and can share nothing.

| Projection | Shape, rows $\times$ cols | $s$ | Input moment $A$ | Output moment $B$ |
| --- | --- | ---: | --- | --- |
| $q,k,v$ | $512\times512$ | 1 | $A_{\text{attn}}$, width 512, shared by the three | $B_q,B_k,B_v$, width 512, separate |
| $o$ | $512\times512$ | 1 | $A_o$, width 512 | $B_o$, width 512 |
| $\text{gate},\text{up}$ | $2048\times512$ | 2 | $A_{\text{mlp}}$, width 512, shared by the two | $B_{\text{gate}},B_{\text{up}}$, width 2048, separate |
| $\text{down}$ | $512\times2048$ | 1 | $A_{\text{down}}$, width 2048 | $B_{\text{down}}$, width 512 |

### Factors by Run

All runs are the 0.5M-token batch for 5,120 updates, except the two wave-4
`_1m` runs, which use the 1M batch for 2,560 updates over the same
2,684,354,560 tokens. "dense" means the
matrix root above; "diag" means the same exponent applied elementwise to the
moment's diagonal; "I" means the factor is not applied. A withheld entry is
computed and then discarded, which is what makes an arm a matched control.

**How the loss columns are computed, and over how many updates.** The
comparison columns are averaged over the **11 evaluations in the second half
of the token budget**, updates 2751 through 5120, and every $\Delta$ is
against the seed-matched `muont` over those same updates. For the 1M rows
the same 11 points are updates 1376 through 2560, against the same-wave
`wave4/muont_1m`. The `Selection`
and `C4` columns are the same 11-point means. A single final evaluation was
the first thing recorded here and it was too noisy to rank arms with.

Each of the 11 is one evaluation pass on data never trained on:

- **Selection** is the held-out mixture loss, a token-weighted average over
  three held-out sources with the mixture's own weights, 3.8 DCLM, 0.25
  StarCoder and 0.055 Proof-Pile, divided by their sum 4.105. Each source
  contributes 64 windows of 4,096 tokens, so one evaluation summarizes about
  786,000 tokens and the 11-point window about 8.6M. It is the quantity
  reserved for choosing between arms.
- **C4** is the Paloma `c4_en` validation loss, a separate pass over a fixed
  external set, not part of the selection average.

Each run records 22 evaluations in total, at update 1, then every 250
updates, then 5,120.

**The `Train` columns average the final 536,870,912 tokens.** That is the
last 1,024 updates at 0.5M and the last 512 at 1M, so the two batches cover
the same data rather than the same update count. Training loss is recorded
at **every** update, 5,120 points per 0.5M run against 11 evaluations, which
is why it is the densest statistic available here. `Train` is the
token-weighted mean over that window and `$\Delta$ train` is the mean of the
**per-update paired difference** against the same baseline as the row's other
deltas. Pairing is exact: within one seed and batch every arm consumes the
same tokens in the same order, so the two runs are differenced update by
update before averaging.

This window is dense enough that its own sampling error is not what limits
the comparison. The paired per-update differences have an effective standard
error of 0.000037 to 0.000063 nats for every arm whose difference series is
uncorrelated, against candidate effects near 0.005. Three arms are the
exception, `muon` 0.000160, `left_all` 0.000115 and `inner_outer_all_diag`
0.000306, because their paired differences are strongly autocorrelated,
lag-1 0.64, 0.57 and 0.82. Those three are also the only rows whose delta
moves materially with the window: `muon` runs +0.008852 over the final 128M
tokens against +0.004567 over the final 1.07B, and `left_all` +0.005941
against +0.009971. The leading candidates are flat to within 0.0003 across
the same four windows, for instance `inner_outer_mlpdiag` -0.005251,
-0.005278, -0.005065 and -0.005086. Where an arm's trajectory is genuinely
diverging from the baseline, no single window summarizes it, and the
autocorrelation is the signal that this is happening.

**Training loss is not a held-out quantity and must not be used to select
an arm.** It scores data the optimizer has just consumed, so it rewards
fitting the current batch and cannot see the generalization gap that
`Selection` and `C4` exist to measure. Across the fourteen seed-0 0.5M arms
the three metrics agree on the broad ordering, identically from the fourth
place down apart from `outer_left_down` and `null_mlpdiag` swapping one
place, and all three put the same four arms on top and the same three at the
bottom.

**They disagree about which arm is best, and the disagreement is larger than
the sampling error.** Training loss ranks `inner_outer_all_lag` first,
selection ranks `inner_outer_mlpdiag` first, and C4 ranks `inner_outer_all`
first. Differencing the two leaders directly, `inner_outer_all_lag` beats
`inner_outer_mlpdiag` on final-window training loss by 0.000664 nats with a
standard error of 0.000043, about fifteen times the error and not noise, yet
it is worse on both held-out metrics, by 0.000215 mean late C4 and 0.000691
selection. The dense-left arms fit the consumed tokens slightly better than
the MLP-diagonal arm while generalizing slightly worse. That is the pattern
training loss is structurally unable to report, and it is the reason
`Selection` remains the arm-choosing quantity.

The training deltas are also consistently smaller than the C4 deltas, for
example -0.005065 against -0.006078 for `inner_outer_mlpdiag`. Use the
`Train` columns for resolution, for separating two arms the 11 evaluations
cannot, and for checking that a held-out ranking is not an artifact of
sparse sampling. The separate five-window tables elsewhere in this document
partition the whole run into 536,870,912-token blocks and answer a different
question, how a delta evolves, rather than where it ended.

**Why the second half, and what the narrow window shows.** The
learning-rate schedule decays linearly from update 1,024 to zero at 5,120,
so the second half is past the stable phase and the model is settling
throughout it. The extra column gives the mean over only the **final four**
evaluations, updates 4501 to 5120, where the rate is below 15% of peak.
Comparing the two columns is the point: where they agree the ranking is
insensitive to the window, and where they disagree it is not.

They agree for every seed-0 arm to within 0.0005, except `left_all` and
`inner_outer_all_diag`, whose penalties shrink on the narrow window. They
disagree for the seed-1 candidate, -0.004749 wide against -0.002269 narrow,
and the decline is monotonic across intermediate windows rather than
scattered. On seed 1 the advantage narrows as the rate approaches zero; on
seed 0 it does not. That is a real seed-dependent difference in late
behaviour, not evaluation noise, and it is the main reason the size of the
gain is still uncertain.

**Selection losses are comparable only within a seed.** The held-out window
sample is drawn using `training.seed`, so seed 1 is scored on a different
held-out set than seed 0. The two baselines make this plain: `muont` seed 0
and seed 1 differ by -0.0483 in windowed selection loss while differing by
only +0.00115 in windowed C4, and their held-out components move in opposite
directions, DCLM -0.060 against StarCoder +0.122. That is the signature of a
different sample, not a different model. Each row's $\Delta$ is therefore
taken against its own seed's baseline, and C4, whose set is fixed across
seeds, is the safer metric for any cross-seed statement.

| Run | Seed | Left on $q,k,v,o,\text{down}$ | Left on $\text{gate},\text{up}$ | Right, all | Left age | Selection | $\Delta$ sel | $\Delta$ sel, final 4 | C4 | $\Delta$ C4 | Train | $\Delta$ train |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wave0/muon` | 0 | I | I | I | — | 3.382434 | +0.005505 | +0.007984 | 3.591257 | +0.005345 | 3.265115 | +0.006161 |
| `wave0/muont` | 0 | I | I | $R,\hat R$ dense | — | 3.376929 | 0 | 0 | 3.585912 | 0 | 3.258954 | 0 |
| `wave0/outer_left_down` | 0 | $\hat L$ dense, $\text{down}$ only | I | $R,\hat R$ dense | $t$ | 3.375559 | -0.001370 | -0.001181 | 3.585130 | -0.000782 | 3.257794 | -0.001160 |
| `wave0/down_hybrid` | 0 | $L,\hat L$ dense, $\text{down}$ only | I | dense, $\text{down}$ input replaced | $t$ | 3.379440 | +0.002512 | +0.001462 | 3.589517 | +0.003605 | 3.262130 | +0.003176 |
| `wave1/inner_outer_all` | 0 | $L,\hat L$ dense | $L,\hat L$ dense | $R,\hat R$ dense | $t$ | 3.371147 | -0.005782 | -0.006041 | 3.579533 | -0.006379 | 3.253979 | -0.004975 |
| `wave1/null_all` | 0 | withheld | withheld | $R,\hat R$ dense | — | 3.375643 | -0.001286 | -0.001179 | 3.583614 | -0.002298 | 3.257592 | -0.001362 |
| `wave1/left_all` | 0 | $L,\hat L$ dense | $L,\hat L$ dense | I | $t$ | 3.385635 | +0.008707 | +0.004968 | 3.596808 | +0.010896 | 3.266820 | +0.007866 |
| `wave1/inner_outer_all_diag` | 0 | $L,\hat L$ diag | $L,\hat L$ diag | $R,\hat R$ diag | $t$ | 3.394940 | +0.018011 | +0.010309 | 3.603598 | +0.017686 | 3.272327 | +0.013373 |
| `wave2/inner_outer_hidden` | 0 | $L,\hat L$ dense | I, not observed | $R,\hat R$ dense | $t$ | 3.373861 | -0.003068 | -0.002770 | 3.583287 | -0.002625 | 3.256822 | -0.002132 |
| `wave2/inner_outer_mlpdiag` | 0 | $L,\hat L$ dense | $L,\hat L$ diag | $R,\hat R$ dense | $t$ | 3.370344 | **-0.006584** | -0.006511 | 3.579834 | -0.006078 | 3.253889 | -0.005065 |
| `wave2/inner_outer_leftdiag` | 0 | $L,\hat L$ diag | $L,\hat L$ diag | $R,\hat R$ dense | $t$ | 3.372426 | -0.004503 | -0.003934 | 3.582302 | -0.003610 | 3.255266 | -0.003688 |
| `wave2/outer_left_all` | 0 | $\hat L$ dense, $L=I$ | $\hat L$ dense, $L=I$ | $R,\hat R$ dense | $t$ | 3.372952 | -0.003977 | -0.004344 | 3.581138 | -0.004774 | 3.255076 | -0.003877 |
| `wave3/inner_outer_mlpdiag_seed1` | 1 | $L,\hat L$ dense | $L,\hat L$ diag | $R,\hat R$ dense | $t$ | 3.323852 | **-0.004749** | -0.002269 | 3.582910 | -0.004153 | 3.255143 | -0.003497 |
| `wave3/muont_seed1` | 1 | I | I | $R,\hat R$ dense | — | 3.328601 | 0 | 0 | 3.587063 | 0 | 3.258640 | 0 |
| `wave3/null_mlpdiag` | 0 | withheld dense | withheld diag | $R,\hat R$ dense | — | 3.376279 | -0.000649 | -0.001331 | 3.583869 | -0.002043 | 3.258266 | -0.000688 |
| `wave3/inner_outer_all_lag` | 0 | $L,\hat L$ dense | $L,\hat L$ dense | $R,\hat R$ dense | $t-1$ | 3.371036 | -0.005893 | -0.005841 | 3.580049 | -0.005863 | 3.253225 | -0.005729 |
| `wave4/inner_outer_mlpdiag_lag` | 0 | $L,\hat L$ dense | $L,\hat L$ diag | $R,\hat R$ dense | $t-1$ | 3.371246 | -0.005683 | -0.006282 | 3.580925 | -0.004987 | 3.254002 | -0.004952 |
| `wave4/inner_outer_mlpdiag_1m` | 0, 1M | $L,\hat L$ dense | $L,\hat L$ diag | $R,\hat R$ dense | $t$ | 3.389576 | -0.002029 | -0.003372 | 3.598757 | -0.003522 | 3.271025 | -0.003024 |
| `wave4/muont_1m` | 0, 1M | I | I | $R,\hat R$ dense | — | 3.391605 | 0 | 0 | 3.602279 | 0 | 3.274050 | 0 |
| `wave4/inner_outer_mlpdiag_seed2` | 2 | $L,\hat L$ dense | $L,\hat L$ diag | $R,\hat R$ dense | $t$ | 3.297046 | -0.003697 | -0.003200 | 3.580912 | -0.003967 | 3.254287 | -0.003453 |
| `wave4-baseline/muont_seed2` | 2 | I | I | $R,\hat R$ dense | — | 3.300743 | 0 | 0 | 3.584879 | 0 | 3.257740 | 0 |
| `wave7/muont` | 0, fixed-age | I | I | $R,\hat R$ dense | — | 3.375445 | 0 | 0 | 3.584699 | 0 | 3.257438 | 0 |
| `wave7/inner_outer_leftdiag` | 0, fixed-age | $L,\hat L$ diag | $L,\hat L$ diag | $R,\hat R$ dense | $t$ | 3.372553 | -0.002892 | -0.001997 | 3.581461 | -0.003238 | 3.255576 | -0.001862 |
| `wave7/outer_leftdiag` | 0, fixed-age | $\hat L$ diag, $L=I$ | $\hat L$ diag, $L=I$ | $R,\hat R$ dense | $t$ | 3.373427 | -0.002018 | -0.000729 | 3.582094 | -0.002605 | 3.255886 | -0.001552 |
| `wave7/outer_leftdiag_fused` | 0, fixed-age | $\hat L$ diag, $L=I$ | $\hat L$ diag, $L=I$ | $R,\hat R$ dense | $t$ | 3.374167 | -0.001277 | -0.000656 | 3.582474 | -0.002226 | 3.255920 | -0.001518 |
| `wave8/muont` | 0, fixed-age | I | I | $R,\hat R$ dense | — | 3.375445 | 0 | 0 | 3.584699 | 0 | 3.257438 | 0 |
| `wave8/outer_leftdiag` | 0, fixed-age | $\hat L$ diag, $L=I$ | $\hat L$ diag, $L=I$ | $R,\hat R$ dense | $t$ | 3.373427 | -0.002018 | -0.000729 | 3.582094 | -0.002605 | 3.255886 | -0.001552 |
| `wave8/outer_leftdiag_fused` | 0, fixed-age | $\hat L$ diag, $L=I$ | $\hat L$ diag, $L=I$ | $R,\hat R$ dense | $t$ | 3.372193 | -0.003252 | -0.003024 | 3.582930 | -0.001769 | 3.255782 | -0.001656 |
| `wave8/outer_leftdiag_combined_fused` | 0, fixed-age | $\hat L$ diag, $L=I$ | $\hat L$ diag, $L=I$ | $R,\hat R$ dense | $t$ | 3.372718 | -0.002727 | -0.002027 | 3.582202 | -0.002498 | 3.256185 | -0.001253 |

Wave-7 and Wave-8 rows each use their own completed same-wave fixed-age MuonT reference,
eleven final-half evaluations and the last four exact-shared evaluations
for the narrower selection column. Fixed-age describes input host-basis
installation at the next update boundary; the diagonal output moment
still targets the current update. These package comparisons have no
matched block-path null-output arm and do not isolate output geometry
from update-implementation differences.
Wave-8 rows were read from all 5,120 original JSONL rows per arm after
completion and checked against the canonical five-window analysis.
Repeated Wave-7/8 reference and Torch-outer loss means are identical
recorded trajectories; they are not independent seed replications.

The three wave-0 down-only arms change the $\text{down}$ update alone; their
other six projections take input MuonT's update unchanged.

**Wave 4 is complete by 19:40:41.** The 1M and lagged table rows now use
eleven final-half evaluations; the narrower selection delta uses
the final four shared evaluations. Seed-2 paired rows were updated after
the companion's 21:12:34 completion and also use eleven final-half
evaluations; the narrower selection delta uses the last four shared points.
"running, k/11" counts the second-half evaluations recorded at that time;
complete 11-point columns remain withheld until each run finishes.
Reading the JSONL at 18:45 on November 16 gave lagged 2831/5120,
1M candidate 1543/2560, 1M MuonT 1756/2560, and seed-2 candidate 2753/5120
updates. The interim deltas at exact shared second-half tokens are:

| Candidate minus reference | Shared late evaluations | $\Delta$ C4 | $\Delta$ sel |
| --- | --- | ---: | ---: |
| `wave4/inner_outer_mlpdiag_lag` minus `wave0/muont` | update 2751 | -0.005099 | -0.004548 |
| `wave2/inner_outer_mlpdiag` minus `wave0/muont`, for reference | update 2751 | -0.006892 | -0.005114 |
| `wave4/inner_outer_mlpdiag_lag` minus `wave2/inner_outer_mlpdiag` | update 2751 | +0.001793 | +0.000566 |
| `wave4/inner_outer_mlpdiag_1m` minus `wave4/muont_1m` | updates 1376, 1501 | -0.005079 | -0.002676 |

The lagged arm's first second-half evaluation is worse than current
MLP-diagonal's at the same update. The 18:37 analysis had a
near-zero -0.000037 mean C4 delta over five earlier evaluations, updates
1251-2251. One evaluation does not rank the two arms. The 1M
candidate's mean over two shared points is -0.005079 C4. Per-point C4
deltas were -0.004476 and -0.005681. The seed-2 candidate records C4
3.725899 and selection 3.446057 at update 2751. At that time it had no
same-seed baseline; `wave4-baseline/muont_seed2` has since completed and
its paired row is in the table above. None of the interim values here is a
final-budget result. The completed 1M and lagged rows supersede their
historical interim values; see [1M Pair](#completed-1m-pair-november-16)
and [Completed Lagged Arm](#completed-lagged-arm-november-16).

**What the dense paired statistic settles, and what it does not.** The
interim single point above put the lagged arm 0.001793 C4 behind current
MLP-diagonal and was too thin to rank them. Differencing the two runs update
by update over the final 536,870,912 tokens gives +0.000113 nats with an
effective standard error of 0.000043 and no autocorrelation, so lagging
costs a small but resolvable amount of training loss on this seed. The
held-out columns agree in sign and are larger, +0.001091 mean late C4 and
+0.001513 at the final evaluation. Lag is close to free on loss, not free.
The same comparison applied to `wave7/outer_leftdiag_fused` against
`wave7/outer_leftdiag` gives +0.000033 with a standard error of 0.000049,
indistinguishable from zero, which is the expected result for a reduction
that changes kernels rather than equations and is a useful check on it.

Pairing only works within a seed. Differencing the seed-2 candidate against
the seed-0 candidate by update number gives a standard error of 0.001867,
forty times the within-seed figure, because the two runs consume different
tokens at the same update and the difference series is dominated by batch
composition rather than by the optimizer. Cross-seed statements still need
the fixed C4 set and seed-wise deltas.

### Worked Example: `inner_outer_mlpdiag`

The best-performing arm, written out per projection. Only the output side
differs between the groups, and the scale $s$ differs because the shapes do.

$$
\Delta W_{q}=-\eta\left[\hat L_{q}\,
\mathcal N_{10}\!\left(L_{q}U_t R_{\text{attn}}\right)\hat R_{\text{attn}}
+\lambda W_{q}\hat R_{\text{attn}}^{2}\right]
$$

and identically for $k,v$ with $B_k,B_v$, and for $o$ with
$B_o,A_o$. The MLP expansion pair takes diagonal output factors and
$s=2$,

$$
\Delta W_{\text{gate}}=-\eta\left[2\,
\hat L^{\text{diag}}_{\text{gate}}
\mathcal N_{10}\!\left(L^{\text{diag}}_{\text{gate}}U_t R_{\text{mlp}}\right)
\hat R_{\text{mlp}}
+\lambda W_{\text{gate}}\hat R_{\text{mlp}}^{2}\right],
$$

identically for $\text{up}$, where $L^{\text{diag}}=\operatorname{diag}(B)^{-1/2}$
elementwise. The contraction has a 2048-wide input moment and a 512-wide
dense output moment,

$$
\Delta W_{\text{down}}=-\eta\left[\hat L_{\text{down}}
\mathcal N_{10}\!\left(L_{\text{down}}U_t R_{\text{down}}\right)
\hat R_{\text{down}}
+\lambda W_{\text{down}}\hat R_{\text{down}}^{2}\right].
$$

This is why the arm is cheap. The only dense output roots are at width 512,
which is 3.8% of the output eigendecomposition work, while the width-2048
output moments, 96.2% of that work, are reduced to elementwise powers.
The lagged variant is identical with every $L,\hat L$ built from update
$t-1$'s rows.

## Runtime, Evaluation, and Checkpoints

| Runtime Setting | Value and Evidence Scope |
| --- | --- |
| GPU | NVIDIA A100-SXM4-40GB, SM80 |
| Recorded Python / PyTorch | 3.13.11 / 2.10.0 |
| Recorded CUDA / driver | 12.9 / 580.105.08 |
| Triton | Current training installation: 3.6.0; launch-time package version not captured |
| Attention implementation / package | Executed backend FA2; current `flash-attn` distribution 2.8.3; launch-time package version not captured |
| TF32 | Disabled for CUDA matmul and cuDNN |
| Deterministic algorithms | Enabled |
| Compilation | Static `torch.compile`, body and compiled cross-entropy head |
| Compile options | `max_autotune=True`, `triton.cudagraphs=False`, `max_autotune_gemm_backends="ATEN"`, `emulate_precision_casts=True` |
| DDP | `gradient_as_bucket_view=True`, `broadcast_buffers=False`, `find_unused_parameters=True` |
| Launch environment | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONFAULTHANDLER=1`, `NCCL_INSPECTOR_ENABLE=0` |
| Data/model sharding | DDP replicates model; Toner statistics/root/update ownership is sharded |

`runtime.compile` retains the recipe label `max-autotune-no-cudagraphs`,
but the runner actually supplies explicit compile options, not that mode
argument. Body/head CUDA graphs are disabled; Toner's root CUDA graphs stay
enabled.

Python, PyTorch, CUDA, driver, GPU, and world size are recorded in each
executed manifest. Package-version inspection during this audit additionally
found NumPy 2.5.4, Transformers 4.57.1.post1, Tokenizers 0.22.2,
HuggingFace Hub 0.36.2, and PyArrow 26.0.1 in the training installation.
These current observations are not an immutable launch-time environment
lock; exact executed versions for packages absent from the manifests remain
unknown. The pinned data/tokenizer/source receipts provide separate provenance.

Evaluation uses the training body's compilation/BF16 path and the pinned
Triton `head_ce_forward` for per-token losses. All C4 windows are evaluated:
236 windows, 966,656 input tokens, 966,420 scored next-token targets.
The last partial C4 window is dropped as upstream does.
There is no C4 window cap (`eval_max_windows: 0`).
Distributed padding is excluded from summed losses and target counts.

Every arm has 22 evaluations: updates `1, 251, 501, ..., 5001, 5120`.
The offset follows upstream's zero-based `step % 250 == 0` condition.
Held-out evaluation also uses 64 windows per training source. Selection loss
is `(3.8 * DCLM_loss + 0.25 * StarCoder_loss + 0.055 * ProofPile_loss) / 4.105`.
C4 is separate. These extra held-out evaluations are our addition.
Training loss is the pre-update loss on that update's minibatch; validation
loss is measured after the update.

Checkpoints are saved at updates `250, 500, ..., 5000, 5120`: 21 checkpoints
per arm. Each has eight rank files and a completion receipt; rank 0 also saves
model weights. State binds configuration, source hashes, data receipt, optimizer
moments, roots, output moments, rank ownership, and RNG state.
Final receipts and nonempty rank files were verified, without loading the
approximately 10-14 GB final checkpoint tensors during this audit.

## Differences from Toner

| Area | Toner Reference | Executed Difference | Consequence |
| --- | --- | --- | --- |
| Hardware / attention | H100 / FA3 | A100 40 GB / FA2 | Different kernels, rounding, and throughput; not exact runtime reproduction |
| Software | README integration checks use PyTorch 2.11, Triton 3.6 | Recorded PyTorch 2.10; installed Triton 3.6 | Not the same software stack; original reference run environment is not fully proven by integration-test versions |
| Microbatch | 8, accumulation 2 | 4, accumulation 4 | Same global batch/data; different BF16 accumulation/kernel shapes |
| Body/head compile | Upstream mode argument | Explicit ATEN-only autotuning and precision-cast emulation | Different compiler decisions; trajectory can differ |
| Determinism / DDP | Deterministic flag absent; default bucket rebuilding | Deterministic algorithms and `find_unused_parameters=True` | Chosen for repeatable continuation; different reduction/layout execution |
| Evaluation | C4 only | C4 plus three held-out mixture sets | Additional selection metric/time; training mixture and C4 are retained |
| Checkpoints | Final model save; no resume interface | Periodic distributed state saves and continuation | Additional I/O; host synchronization can affect asynchronous root readiness |
| Optimizer arms | Muon and input MuonT | Two added output-down arms | Intentional algorithm changes, not Toner reproduction |
| Down arithmetic in new arms | Batched NS; FP16-split FP32-accurate root products; fused direction/decay product | Per-matrix NS, direct FP32 products, separately formed direction/decay | Same target equations do not guarantee identical finite-precision updates; attribution needs a matched null control |
| Logging | JSONL | Historical SwanLab import | Cloud timestamps/duration describe uploading, not original training |

The plain `muon` and `muont` arms call the unchanged upstream
`build_optimizers`. The custom down path applies only to the output-side arms.
Output work also changes timing relative to input-root host refreshes.
Same root settings do not guarantee the same cached eigenbasis arrival history
across separate runs.

There is **no configuration drift** in model size, global batch, training
budget, data sources/weights, tokenizer, base LR, schedule, input EMA/damping,
or input root powers. The `experiment.output` dictionary exists in all four
configs, but plain Muon/MuonT do not instantiate or use output factors.
It must not be interpreted as enabling output normalization for those controls.

## Results and the Small Loss Gap

Lower losses are better. Positive `Delta C4` means worse than input MuonT.

| Arm | Final C4 | Delta C4 vs Input MuonT | Final Selection | Final Train | End-to-End Seconds | Peak Allocated GiB, Rank 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Muon | 3.461811238144785 | +0.009112891251 | 3.2460983875967275 | 3.141834259033203 | 5889.713 | 18.971 |
| Input MuonT | 3.4526983468940755 | 0 | 3.2366145097362016 | 3.132052183151245 | 5984.650 | 20.531 |
| Outer-left down | 3.451355186660041 | -0.001343160234 | 3.235340279285975 | 3.1312296390533447 | 6286.405 | 20.574 |
| Down hybrid | 3.4550563843044375 | +0.002358037410 | 3.2376780994071783 | 3.13546085357666 | 6145.105 | 19.230 |

MuonT improves C4 over Muon by 0.0091129, approximately 0.907% lower perplexity.
Outer-left improves C4 over MuonT by 0.0013432, approximately 0.134% lower
perplexity; its loss reduction is approximately 0.0389%.
Hybrid is worse than MuonT by 0.0023580, approximately 0.236% higher perplexity.
Perplexity changes use `exp(loss_difference) - 1`, not loss percentage.

The input-MuonT reproduction is close to, but does not equal, the references:

| Baseline | Pinned Final C4 Reference | Our Final C4 | Our Minus Reference |
| --- | ---: | ---: | ---: |
| Muon | 3.462118 | 3.461811238144785 | -0.000306761855 |
| MuonT `(0.5, 0.125)` | 3.4505794264657865 | 3.4526983468940755 | +0.002118920428 |

The reference Muon-to-MuonT gap is 0.0115386; ours is 0.0091129.
The cause of the reproduction residual has not been isolated. It is larger
than the observed extra output-side gain, which warrants caution but does
not by itself measure within-runtime seed variance.

Reference values above are the `reference_validation_loss` fields in the
pinned `configs/130m/muon_0p5m.json` and `muont_0p5m.json`.
The cross-batch table below reads the corresponding `muon_BATCH.json` and
`muont_BATCH.json` files from that same pinned directory.
These config fields and README do not supply original run IDs, checkpoint
hashes, reference seeds, or a replication/aggregation count. Those reference
identities are unresolved; the numeric fields must not be treated as a
matched-seed baseline, a mean, or an uncertainty estimate.

The chosen batch already has the smallest reference gap among the pinned
130M batch settings:

| Batch Tokens | Pinned Muon C4 | Pinned MuonT C4 | Reference Gain |
| ---: | ---: | ---: | ---: |
| 524,288 | 3.462118 | 3.4505794264657865 | 0.0115386 |
| 1,048,576 | 3.493352 | 3.4688706790277672 | 0.0244813 |
| 2,097,152 | 3.561805 | 3.5159593144768886 | 0.0458457 |
| 4,194,304 | 3.720991 | 3.6258195780209785 | 0.0951714 |

These are upstream reference outcomes, not results of additional local runs.
They help explain why near-overlapping curves at 0.5M are unsurprising; they
do not predict a larger output-factor gain at larger batches.
The common early loss fall from about 12 dominates a full-range chart.
Use unsmoothed C4 at matched tokens and plot deltas against MuonT.
Outer-left is not better at every evaluation: its gap changes sign during
training, though it is better at each of the final seven recorded evaluations.

End-to-end time includes compilation, data loading, evaluation, checkpoints,
and communication. It is not isolated optimizer cost or a direct H100
throughput comparison. Relative to MuonT, outer-left is 5.04% slower and hybrid
2.68% slower in this execution. Peak-memory fields and root fallback counts
are rank-0 observations, not cluster-wide maxima/totals.

The table's seconds are the final rank-0 JSONL `end_to_end_seconds`:
the timer starts at entry to `experiment.train.main`, before argument,
receipt, distributed, model, and optimizer initialization. It ends after the
last evaluation and completed final checkpoint, before writing that final
metric and the final manifest. Python imports, torchrun startup before
`main`, and final process teardown are outside this timer.
`step_seconds` ends after synchronized optimizer updates, before that
update's evaluation/checkpoint; `elapsed_seconds` is sampled at that boundary.
These recorded durations differ from the future launch-based time-to-target
definition below.

CUDA peak counters reset once immediately before the training loop, after
initialization and synchronization. The allocated/reserved fields are read
after each optimizer update and before its evaluation/checkpoint. They are
running peaks, not per-step peaks; earlier evaluations/checkpoints can affect
later reads, but the final evaluation/checkpoint cannot affect the last read.
Persistent allocations already live at reset remain part of the memory
baseline. The table reports the final rank-0 allocated-peak observation.

## Compute Interpretation

Outer-left down has the smallest observed final loss, but it does not
remove any input MuonT factors:

$$
D_{\mathrm{outer\text{-}left}}
=L_{\mathrm{out}}
\underbrace{\mathcal N_{10}(UR_{\mathrm{in}})R_{\mathrm{out}}}
_{D_{\mathrm{input\ MuonT}}}.
$$

It retains the down projection's 2048-dimensional input moment and roots,
then adds output-cotangent capture, a 512-dimensional output moment and
eigendecomposition, and a left matrix product. The current collector also
computes the unused inner output factor. Avoiding that unused factor is
a possible implementation optimization; it does not eliminate the added
moment collection or eigendecomposition.

Down hybrid replaces the down-input moment and roots with output factors.
At expansion ratio four, one FP32 moment shrinks from 16 MiB to 1 MiB:
16x less moment storage and 64x less nominal dense eigendecomposition work
for that replacement. These ratios do not describe the whole optimizer
or training step. Input MuonT uses cached approximate roots, while the
output backend uses periodic exact eigendecomposition; collection,
communication, matrix products, and kernel launches also contribute.

Wave 0 measured outer-left 5.04% slower and hybrid 2.68% slower than input
MuonT end to end. Rank-0 peak allocated memory was 20.531 GiB for MuonT,
20.574 GiB for outer-left, and 19.230 GiB for hybrid. Neither candidate
has established compute savings or a time-to-target improvement. Isolated
optimizer timing and a matched numerical control are still needed.

## Wave 1: All-Block Arms

Launched on 2026-11-16 at 03:21-03:22 Asia/Shanghai. Training is **in
progress**; no wave-1 outcome is recorded yet. The four arms are defined in
[Arm Definitions](#arm-definitions). All four share wave 0's recipe: seed 0,
0.5M-token batch, microbatch 4, 5,120 updates, LR `0.02262741696`, and the
wave-0 MuonT input settings. Compare them with the wave-0 `muon` and `muont`
runs, which used the same data, initialization, and budget.

| Arm | Assigned Node | Torchrun PID | Comparison Role |
| --- | --- | ---: | --- |
| `inner_outer_all` | `28.216.17.9` | 2467745 | Two-sided on all 224 matrices; versus `null_all`, then `muont` |
| `left_all` | `28.216.17.80` | 201976 | Left-only on all matrices; versus `muont` and `muon` (no matched null) |
| `inner_outer_all_diag` | `28.216.17.144` | 200344 | Diagonal speed variant; versus `inner_outer_all`, `muon`, `muont` |
| `null_all` | `28.216.17.139` | 196038 | Same path as `inner_outer_all` with identity left factors |

Outputs are `PVC/runs/wave1/ARM/`, with launch report
`PVC/run-preparation/wave1.json`.

### Wave 1 Outcomes

All four arms completed 5,120 updates. Losses are held-out mixture selection
loss, the quantity reserved for choosing between arms; deltas are against
wave-0 `muont`, negative meaning better.

| Arm | Selection | C4 | Δ selection vs `muont` | Median s/update | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inner_outer_all` | 3.230471 | 3.445799 | **-0.006143** | 1.505 | 24.20 |
| `null_all` | 3.235448 | 3.450540 | -0.001166 | 1.485 | 24.20 |
| `left_all` | 3.239867 | 3.457714 | +0.003252 | 1.456 | 22.20 |
| `inner_outer_all_diag` | 3.243949 | 3.459198 | +0.007335 | 1.275 | 21.37 |
| `outer_left_down` (wave 0) | 3.235340 | — | -0.001274 | 1.143 | 20.57 |
| `muont` (wave 0) | 3.236615 | — | 0.000000 | 1.095 | 20.53 |
| `muon` (wave 0) | 3.246098 | — | +0.009484 | 1.074 | 18.97 |

Four readings follow, in decreasing order of how well the data supports them.

**Two-sided output factors on all blocks are the best result so far, and the
matched control carries about a fifth of it.** `inner_outer_all` beats `muont`
by 0.0061 and beats `null_all`, which shares its entire execution path and
differs only in using identity left factors, by 0.0050. That difference is the
part attributable to output geometry rather than to the block path's own
arithmetic. The remaining 0.0012 that `null_all` gains over `muont` is the
path effect, and it is the same size as wave 0's `outer_left_down` margin.
Wave 0's winner may therefore have been mostly execution, not geometry.
Having run the matched control is what makes this separable at all.

Paired wide-window training loss confirms the same ordering with far more
samples than the 22 sparse evaluations. Every arm sees an identical token
order, so per-update differences are paired; averaging over 2048-update
windows removes per-batch noise. `tracking/compare_runs.py` computes this.

| Arm | Train, updates 1-2048 | Train, updates 2049-4096 |
| --- | ---: | ---: |
| `inner_outer_all` | -0.01188 | -0.00540 |
| `null_all` | +0.00060 | -0.00167 |
| `outer_left_down` (wave 0) | +0.00017 | -0.00034 |
| `left_all` | +0.02600 | +0.01398 |
| `inner_outer_all_diag` | +0.04604 | +0.02550 |
| `muon` (wave 0) | +0.05458 | +0.00466 |

The window deltas are an order of magnitude larger than the final
evaluation deltas and keep their sign across both halves of training, so
the ranking does not rest on the last evaluation point. They also place
wave 0's `outer_left_down` within 0.0004 of `muont` on training loss,
reinforcing that its advantage was small relative to `inner_outer_all`'s.
Window means are not a substitute for held-out loss, since the final
learning-rate decay phase compresses training loss differences; they
measure consistency, not generalization.

**The diagonal approximation does not work here.** `inner_outer_all_diag`
is 0.0135 worse in selection loss than the full arm and worse than `muont`.
It is the only wave-1 arm that also replaces the *input* factors with
diagonals, so its loss reflects two changes at once and does not isolate the
output-side diagonal. Wave 2 separates them.

**Output-only left scaling did not help.** `left_all` is worse than `muont`
by 0.0033 despite applying the same output factors as the winning arm. Input
factors are not merely additive with output factors; the arm that drops them
loses more than the arm that adds them gains. This is one run per arm at one
seed, so the ordering among the three arms within 0.007 of each other is not
yet established.

**The extra step time splits about evenly between the update path and the
output engine.** Arms differ in which components they carry, so medians over
updates 2049-4096 separate them. `null_all` applies no left factors yet costs
the same as the full arm, which rules out factor application as the driver.

| Arm | Median s/update | Input side | Output side | Update path |
| --- | ---: | --- | --- | --- |
| `muon` | 1.074 | none | none | Toner batched |
| `muont` | 1.095 | full | none | Toner batched |
| `inner_outer_all_diag` | 1.275 | diagonal | diagonal | per-matrix, 224 |
| `left_all` | 1.463 | none | full | per-matrix, 224 |
| `null_all` | 1.510 | full | full | per-matrix, 224 |
| `inner_outer_all` | 1.524 | full | full | per-matrix, 224 |

Differences give the component costs. `inner_outer_all_diag` minus `muont`
is +0.180 s, and the diagonal arm carries no dense roots on either side, so
the per-matrix update path costs roughly 0.18-0.20 s. `inner_outer_all`
minus `inner_outer_all_diag` is +0.249 s for adding both dense engines, and
`inner_outer_all` minus `left_all` is +0.061 s for the input engine alone,
leaving about 0.19 s for the dense output engine. Against the 0.429 s total
gap, that is about 42% update path, 44% output engine, 14% input engine on
the block path. Fallbacks sit inside the output engine's share: 31,759 over
the run, about 6 of 224 output statistics per update, against 33 total on
the input side.

**Output cost is concentrated in two of the seven matrices.** The output
statistics split into 160 at width 512 and 64 at width 2048. Dense root work
scales as $d^3$ and Gram accumulation as $d^2$, so the two MLP-width
projections carry 96.2% of the eigendecomposition work and 86.5% of the Gram
work while being 29% of the statistics. They also hold 0.500 GiB of the
0.579 GiB of packed Grams and 1.000 GiB of the 1.156 GiB of cached
eigenbases. This is what makes wave 2 worth running: removing or
diagonalizing the width-2048 output factors should remove most of the output
engine's cost, and batching the update path should remove most of the rest,
while retaining full output geometry exactly where it is cheap.

## Wave 2: Cheaper Output Geometry

Wave 2 completed on November 16 by 14:01; all four histories were analyzed,
imported and cloud-verified. The launch and interim observations below are
historical, not current running status. The GPU allocation was released
after wave 1, then re-taken on owner instruction; all four keepers were
stopped again and keeper-restore watchers reinstalled, one per node.

| Node | Arm | PID | Question |
| --- | --- | --- | --- |
| `28.216.17.9` | `inner_outer_hidden` | 2583793 | Does width-2048 output geometry matter at all? |
| `28.216.17.80` | `inner_outer_mlpdiag` | 219653 | Does width 2048 need only its scale? |
| `28.216.17.144` | `inner_outer_leftdiag` | 214347 | Does output scale alone carry the gain? |
| `28.216.17.139` | `outer_left_all` | 213740 | Is the inner left factor needed? |

| Item | State | Evidence |
| --- | --- | --- |
| Configurations | 16 files, four batch settings each, identical to wave 1 apart from the arm name | [experiment/configs](experiment/configs/) |
| CPU tests | 37 pass at launch | `python -m pytest -c /dev/null experiment/tests -q` |
| GPU validation | All 16 checks pass, including the four block gates | `PVC/run-preparation/validation-wave2.json` |
| Preflight | Clean, no blockers; 32 GPUs idle across four nodes | `PVC/run-preparation/preflight-wave2.json` |
| Launch report | Commands, PIDs, logs and output paths | `PVC/run-preparation/wave2.json` |
| Keeper watchers | One per node, verified | `PVC/runs/wave2/keeper-restore-NODE.log` |

Wave 2 is the first wave to run the batched update path, so its step times
are not comparable with wave 1's except within a matched update window.

### Wave 2 Outcomes

All four arms completed 5,120 updates. Selection loss is the held-out
mixture loss reserved for choosing arms; step-time medians use updates
2049-4096 so they are comparable across waves despite fallbacks
accumulating over training.

| Arm | Selection | Δ vs `muont` | Mean late Δ | Median s/update | vs `muont` | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `inner_outer_mlpdiag` | 3.229726 | **-0.006888** | **-0.006590** | 1.336 | +21.9% | 22.64 |
| `inner_outer_all` (wave 1) | 3.230471 | -0.006143 | -0.005695 | 1.524 | +39.2% | 24.20 |
| `outer_left_all` | 3.232430 | -0.004184 | -0.003820 | 1.419 | +29.6% | 24.20 |
| `inner_outer_leftdiag` | 3.232483 | -0.004131 | -0.004642 | 1.209 | **+10.4%** | 22.36 |
| `inner_outer_hidden` | 3.233505 | -0.003109 | -0.003126 | 1.293 | +18.0% | 21.51 |
| `null_all` (wave 1 control) | 3.235448 | -0.001166 | -0.001479 | 1.510 | +37.8% | 24.20 |
| `outer_left_down` (wave 0) | 3.235340 | -0.001274 | -0.001199 | 1.143 | +4.4% | 20.57 |
| `muont` (wave 0) | 3.236615 | 0 | 0 | 1.095 | 0% | 20.53 |

**The dense width-2048 output roots did not improve this seed's result.**
`inner_outer_mlpdiag` replaces them with elementwise
diagonals and is better than `inner_outer_all` on both the final selection
loss and the mean over late evaluations, while costing 17 percentage points
less per update. Two arms that keep those dense roots, `inner_outer_all` and
`outer_left_all`, are the two slowest arms in the endeavor.

Staleness is a possible explanation, not an established mechanism. Output bases refresh
every 4 updates, and the dense arms record about 32,000 exact-eigendecomposition
fallbacks over a run, roughly 6 of 224 statistics per update, with basis ages
reaching the full refresh interval. The width-2048 dense roots are therefore
often approximations of a moment that has already moved. A diagonal is
recomputed exactly every update and never goes stale. An exact cheap
approximation beat a stale expensive one.

Dropping the width-2048 factors entirely is worse than diagonalizing them:
`inner_outer_hidden` reaches only -0.0031. So that moment carries real
information; it is its rotation, re-estimated too slowly, that was not
paying for itself.

**Two arms are now worth taking forward, for different reasons.**
`inner_outer_mlpdiag` is the best single-seed exploratory result, improving
over `muont` by 0.0069 in selection loss. `inner_outer_leftdiag` is the best
trade, 67% of the gain for 10.4% more step time and no dense output
eigendecomposition at all. Both also carry the three avoidable host stalls
per update fixed after launch, so both step times are upper bounds.

Caveats that bound these claims. One seed per arm: the 0.0007 gap between
`inner_outer_mlpdiag` and `inner_outer_all` is far smaller than the 0.0057
gap from `inner_outer_mlpdiag` to `null_all`, so "both beat the control" is
large within this trajectory while neither replicated significance nor
the ordering between the two leaders is established.
`null_all` is the matched control for the dense arms, not for
`inner_outer_mlpdiag`; `null_mlpdiag` exists for that and is running in wave 3.
Step-time medians exclude compilation and evaluation, and end-to-end
seconds include both.

## Wave 3: Seed Variance, Control, and Lagged Factors

Launched 2026-11-16 14:48 after deploy, GPU revalidation and a clean
preflight. All 16 checks pass in `PVC/run-preparation/validation-wave3.json`,
including real-data continuation for the lagged arm. Launch evidence is
`PVC/run-preparation/wave3.json`; keeper-restore watchers are installed and
verified one per node, logging to `PVC/runs/wave3/keeper-restore-NODE.log`.

Before the deploy the launcher correctly **refused**, which a dry preflight
confirmed: occupied GPUs, configurations missing from the remote nodes, and
`Validation receipt is stale for the current experiment code`. Those three
cleared in order.

| Node | Arm | Seed | Question it answers |
| --- | --- | ---: | --- |
| `28.216.17.9` | `inner_outer_mlpdiag` | 1 | Does the best arm's -0.0069 reproduce on a second seed? |
| `28.216.17.80` | `muont` | 1 | What is the baseline's own seed-to-seed spread? |
| `28.216.17.144` | `null_mlpdiag` | 0 | How much of the best arm's gain is geometry rather than path? |
| `28.216.17.139` | `inner_outer_all_lag` | 0 | Does deferring the left factors by one update recover the root wait? |

Two of the four slots go to seed variance because nothing in this endeavor
has yet established it, and every loss claim depends on it. The gap between
`inner_outer_mlpdiag` and `inner_outer_all` is 0.0007; without a variance
estimate that ordering is not interpretable, whereas the 0.0057 gap from
`inner_outer_mlpdiag` to `null_all` plainly is. Pairing `muont` seed 1 with
`inner_outer_mlpdiag` seed 1 gives both the baseline spread and the
candidate's reproducibility on matched data order.

`null_mlpdiag` exists because without it a wave-3 result on
`inner_outer_mlpdiag` would be unattributable. Wave 1 showed that about a
fifth of `inner_outer_all`'s advantage came from the block path rather than
from output geometry, and only the matched control made that separable.

Waves so far have been seed 0 only, and the launcher hardcoded seed 0 in
its configuration paths, so paired-seed confirmation was not expressible.
Wave entries in `cluster.json` are now `{"arm": ..., "seed": ...}` objects
and the run directory is `ARM` for seed 0 or `ARM_seedN` otherwise, so one
wave can mix a candidate, its control, and a second seed of the same arm.

### Wave 3 Outcomes

All four runs completed 5120 updates and 2,684,354,560 tokens. The Bash
monitor saved fresh per-node process evidence at 16:47:48 and 17:02:55,
then imported and verified every scalar on SwanLab by 17:03:57.
Evidence is `PVC/run-preparation/monitor-wave3/` and
`PVC/run-preparation/swanlab-upload-wave3.json`. Wide-window analysis is
`PVC/run-preparation/analysis-wave3.json` and its `.txt` rendering.

| Arm | Paired MuonT seed | Final C4 delta | Final selection delta | Final-half mean C4 delta | Final-half mean selection delta | End-to-end seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `inner_outer_mlpdiag` seed 1 | 1 | -0.003738 | -0.002534 | -0.004153 | -0.004749 | 7264.62 |
| `null_mlpdiag` seed 0 | 0 | -0.002052 | -0.001332 | -0.002043 | -0.000649 | 7203.32 |
| `inner_outer_all_lag` seed 0 | 0 | -0.005474 | -0.006211 | -0.005863 | -0.005893 | 7857.35 |
| MuonT seed 1 | 1 | 0 | 0 | 0 | 0 | 5899.70 |

The five 536,870,912-token training windows (1024 updates each) give
candidate-minus-same-seed-MuonT deltas:

| Arm | Window 1 | Window 2 | Window 3 | Window 4 | Window 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP-diagonal seed 1 | -0.009496 | -0.005007 | -0.004124 | -0.004290 | -0.003497 |
| MLP-diagonal null seed 0 | +0.001220 | +0.000097 | -0.000488 | -0.000574 | -0.000688 |
| Dense lagged seed 0 | -0.017843 | -0.007789 | -0.005839 | -0.005360 | -0.005729 |

The MLP-diagonal candidate improved final C4 in both exploratory seeds
(seed 0: -0.006264, seed 1: -0.003738), mean -0.005001 nats. Both seeds
also improved within-seed selection loss, but seed 1's final margin is
only 37% of seed 0's. Two seeds, one used for selection, do not establish
replicated significance or a fair tuned-optimizer comparison.
Selection holdouts are seed-dependent: do not infer seed variance from
their absolute losses (MuonT seed 0 3.236615 versus seed 1 3.188087).
The fixed C4 baseline difference is +0.000237 nats.

The seed-0 matched-path null accounts for 0.002052 of the MLP-diagonal
candidate's 0.006264 C4 gain over upstream MuonT. Candidate-minus-null
final deltas are -0.004212 C4 and -0.005556 selection. These isolate a
larger geometry-associated gain than the null path, but the compared
waves also differ in host scheduling and source; a same-source repeat
is needed before treating that as pure causal attribution.

Dense lag retained essentially the current dense arm's final selection
loss (3.230403 versus wave-1 3.230471), but end-to-end time is 31.3%
above MuonT and peak allocated memory is 24.486 GiB versus 20.531.
MLP-diagonal seed 1 is 23.1% slower. Neither is an efficiency win.
Lag and host-stall fixes arrived together, so timing changes from wave 1
do not isolate lag. Proceed with the cheaper lagged MLP-diagonal geometry,
the matched 1M pair, and the seed-2 candidate as defined below.

## Exploratory Elapsed-Time Crossings

On November 16 at 19:06, completed histories were checked against
retrospectively selected C4 targets 3.59, 3.52 and 3.46. This tests whether
the observed fixed-token loss gain has already offset extra runtime; it is
not the predeclared, launch-based efficiency confirmation campaign.
The receipt is `PVC/run-preparation/time-to-target-exploratory-20261116.json`.
Every entry uses the first recorded completed C4 evaluation at or below the
target, with its preceding evaluation/token/time boundary retained.
No loss or time interpolation, training-loss substitute, or final-value
fallback is used. Unreached targets are explicitly censored, including on
completed runs; uninterrupted process histories are required for timing.

| Run | C4 <= 3.59, Seconds | C4 <= 3.52, Seconds | C4 <= 3.46, Seconds |
| --- | ---: | ---: | ---: |
| `wave0/muont`, seed 0 reference | 4721.35 | 5566.31 | 5848.18 |
| `wave2/inner_outer_mlpdiag`, seed 0 | 5624.38 | 6309.14 | 6981.75 |
| `wave2/inner_outer_leftdiag`, seed 0 | 5083.38 | 5702.84 | 6323.60 |
| `wave3/inner_outer_all_lag`, seed 0 | 6226.03 | 6956.95 | 7674.74 |
| `wave3/muont_seed1`, seed 1 reference | 4632.11 | 5480.03 | 5899.70 |
| `wave3/inner_outer_mlpdiag_seed1`, seed 1 | 5723.35 | 6422.69 | 7100.53 |

There is no elapsed-time win at any of these three targets. Fully diagonal
output reaches C4 <= 3.52 at 2,359,820,288 tokens, one evaluation earlier
than seed-0 MuonT at 2,490,892,288 tokens, but still takes 5702.84 versus
5566.31 seconds. MLP-diagonal also reaches that target at the earlier
evaluation in both seeds, but takes 6309.14 versus 5566.31 seconds in
seed 0 and 6422.69 versus 5480.03 seconds in seed 1.

These are the recorded main-function timers, logged after evaluation and
any checkpoint on that update. They include setup, compiled steps,
evaluation and checkpointing, but exclude Python imports and parent launch
overhead; they are not exact launch-to-evaluation times. Cross-wave source
and node timing differences remain confounders. Targets were selected after
inspecting C4, and sparse evaluations bound the observed crossing rather
than locating a continuous-time crossing. This negative diagnostic supports
the systems experiment, not a claim that another target or tuned candidate
cannot improve. Keep wave 5's same-source current/fused/MuonT comparison,
then independently tune surviving families and predeclare fresh confirmation.

The reporting command was:

```bash
root=/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization
/opt/conda/envs/torch-base/bin/python tracking/time_to_target.py \
  --root "$root" --target 3.59 --target 3.52 --target 3.46 \
  --out "$root/run-preparation/time-to-target-exploratory-20261116.json" \
  wave0/muont wave2/inner_outer_mlpdiag wave2/inner_outer_leftdiag \
  wave3/muont_seed1 wave3/inner_outer_mlpdiag_seed1 wave3/inner_outer_all_lag
```

Five focused CPU cases cover sparse crossing, unfinished/completed censoring,
missing evaluations and missing main timers. Together with the two ledger
preservation/completion cases, seven tests pass in a bounded standalone
pytest harness. Scoped repository Ruff/Black, license and syntax checks pass.
The training Python lacks the repository's timeout/xdist plugins; the
standalone harness retained all default safety-marker exclusions and used
an external 120-second timeout. No training environment was changed.
The broader tracking suite subsequently passed 32 tests and 20 subtests.

## Wave 4: Completed

Selected on 2026-11-16 at 16:50, Asia/Shanghai; launched at 17:40.
The launcher and GPU/full-model gates now bind each entry's batch, seed,
source, and configuration. CPU validation passes 42 tests. All 16 gates
passed before the full-budget launch.

| Node | Arm | Batch / Seed | Question it answers |
| --- | --- | --- | --- |
| `28.216.17.9` | `inner_outer_mlpdiag_lag` | 0.5M / 0 | Does one-update-lagged output geometry retain the gain while reducing the root wait? |
| `28.216.17.80` | `inner_outer_mlpdiag` | 1M / 0 | Does the best exploratory geometry help at larger batch? |
| `28.216.17.144` | `muont` | 1M / 0 | Matched-token, same-seed and same-batch baseline for that test |
| `28.216.17.139` | `inner_outer_mlpdiag` | 0.5M / 2 | Third exploratory candidate seed; requires a separately scheduled seed-2 MuonT companion before paired interpretation |

Launch evidence is `PVC/run-preparation/wave4.json`. Training PIDs are
2715673, 255930, 230744, and 228130 in table order. Fresh SSH observations
at 17:42 confirmed all four running. Detached Bash monitor PID 2718105
saves context every 900 seconds under `PVC/run-preparation/monitor-wave4/`,
then analyzes, imports and verifies the completed wave. Keeper-restore
watchers are live on all four nodes and log under
`PVC/runs/wave4/keeper-restore-NODE.log`; restoration refuses unrelated
CUDA occupancy or a reused process identity.

The final gate receipt is `PVC/run-preparation/validation-wave4.json`
(17:37:37). It preserves the initial receipt/hash and extends only the
changed fourth slot. All current continuous-three versus two-plus-resume
comparisons have zero maximum absolute tensor error: lagged MLP-diagonal
3387 comparisons, 1M candidate 2043, 1M MuonT 2003, seed-2 candidate
2043. The exact source snapshot is
`PVC/run-preparation/source-wave4-20261116.tar`. Its experiment hashes
match validation. The earlier archive is named
`source-pre-wave4-20261116.tar` and is not an exact wave-3 source
archive.

The 1M pair uses the pinned 1M LR, beta2, decay and 512-stable/2048-decay
schedule, not an early stop or an unchanged 0.5M schedule. It retains
2,684,354,560 input tokens, microbatch 4, eight ranks, and 2560 updates.
Evaluation/checkpoint/diagnostic intervals are 125 updates, matching the
131,072,000-token interval of the 0.5M runs. The executed evaluation
predicate is based on zero-indexed updates, so cross-batch evaluation
phases differ by 524,288 tokens; compare shared tokens without
interpolation, not nominal equal update numbers. This avoids crediting larger
batch with half as many checkpoints in a runtime comparison.

Predeclared comparisons are lagged MLP-diagonal versus wave-2 current
MLP-diagonal (seed 0) and the wave-4 1M pair. The fourth slot changed in
the worktree during validation, observed at 17:27 on November 16, from
fully diagonal output seed 1 to MLP-diagonal seed 2. Preserve the original
fully diagonal GPU check as extra evidence. Validation was extended to
the actual seed-2 configuration before launch. The seed-2 MuonT companion
must run before any paired seed-2 claim; until then report raw values only.
Training-loss windows span 536,870,912 tokens:
1024 updates at 0.5M or 512 updates at 1M. Report all five full windows,
mean final-half matched-token C4 and selection deltas, final losses,
whole-run and steady-state timings, peak memory, and root/fallback counts.
Do not compare absolute selection loss across seeds: reserved windows
change with seed. Seed-wise deltas and fixed-C4 reporting remain valid.
Lag speed attribution is exploratory because wave 3 also removed host
stalls; a same-source current-versus-lag repeat is needed to isolate lag.
No optimizer superiority or time-to-target improvement is yet established.

### Completed 1M Pair, November 16

Fresh SSH observation at 19:28:19 confirms both 1M parents absent and both
manifests complete at 2560 updates/2,684,354,560 tokens. The candidate
keeper watcher records exit at 19:28:14 and restoration at 19:28:28.
Both histories contain 22 C4/selection evaluations, consecutive unique
updates and final eight-rank checkpoint completion receipts.
`PVC/run-preparation/analysis-wave4.{json,txt}` refreshed at 19:29 preserves
the completed 1M comparison separately from the still-interim lag comparison.

| Metric | MLP-Diagonal 1M | MuonT 1M | Candidate Minus MuonT |
| --- | ---: | ---: | ---: |
| Final C4 | 3.466459535 | 3.470417168 | -0.003957633 |
| Final selection | 3.250569133 | 3.254198730 | -0.003629597 |
| Mean final-half C4, 11 exact shared evaluations | | | -0.003522 |
| Mean final-half selection, 11 exact shared evaluations | | | -0.002029 |
| Recorded main-function time, seconds | 6449.983 | 5631.456 | +818.527, +14.53% |
| Steady seconds/MTok | 2.313 | 2.002 | Higher candidate cost |
| Peak allocated GiB | 25.437 | 21.507 | +3.930 |

All five 536,870,912-token training-window deltas are negative:
-0.021550, -0.007184, -0.004285, -0.003218 and -0.00302.
This supports a small exploratory 1M fixed-token loss benefit under the
matched runtime and untuned recipes, not efficiency or independently tuned
superiority. Mean final-half selection improves by less than its final-point
gap, so report both instead of choosing the more favorable endpoint alone.

The same retrospective C4 targets were checked in
`PVC/run-preparation/time-to-target-wave4-1m-exploratory-20261116.json`.
Both arms first reach <=3.59 at 2,229,272,576 tokens: candidate 5373.60
seconds versus MuonT 4696.10. Both first reach <=3.52 at 2,491,416,576
tokens: 5993.73 versus 5234.38 seconds. Neither reaches <=3.46 by the
full budget; both are explicitly censored. No observed token or elapsed-time
saving exists at those targets. Timing/retrospective-selection limitations
are unchanged from [Elapsed-Time Crossings](#exploratory-elapsed-time-crossings).

The 19:28:19 snapshot still has lagged 4747/5120 and seed-2 candidate
4586/5120, both running. The seed-2 queue recognizes its complete predecessor
at 19:29:13 and is beginning the follow-up handoff; actual validation/launch
requires its new receipts. Wave 4 is not complete and has no complete-wave
verified SwanLab upload yet. Source/history evidence remains private.

### Completed Lagged Arm, November 16

The lagged arm is `wave4/inner_outer_mlpdiag_lag`, configuration
`inner_outer_mlpdiag_lag_0p5m_seed0.json`.
Fresh SSH at 19:37:05 finds its parent absent and its manifest complete
at 5120 updates/2,684,354,560 tokens. All eight final checkpoint files are
nonempty; its watcher records exit at 19:36:59 and keeper restoration at
19:37:15. The refreshed `analysis-wave4.{json,txt}` compares it with wave 2's
current MLP-diagonal geometry using all five 536,870,912-token windows.

| Metric | Lagged MLP-Diagonal | Current MLP-Diagonal, Wave 2 | Lagged Minus Current |
| --- | ---: | ---: | ---: |
| Final C4 | 3.447947714 | 3.446434236 | +0.001513478 |
| Final selection | 3.230059011 | 3.229726368 | +0.000332643 |
| Mean final-half C4, 11 exact shared evaluations | | | +0.001091276 |
| Mean final-half selection, 11 exact shared evaluations | | | +0.000901488 |
| Recorded main-function time, seconds | 6985.658 | 7141.961 | -156.303, -2.19% |
| Steady seconds/MTok | 2.479 | 2.559 | -0.080 |
| Peak allocated GiB | 22.676 | 22.635 | +0.041 |

The five training-window deltas are -0.001554881, -0.000524228,
+0.000070511, -0.000045108 and +0.000112830. Small late training deltas
coexist with slightly worse late held-out losses. Historical source and
runtime differences between waves remain a confound; wave 5's same-source
reference was intended to separate implementation cost.

`comparison-wave4-lag-muont-20261116.json` records the exploratory comparison
with same-seed historical MuonT. Final C4/selection deltas are
-0.004750633/-0.006555499; final-half means are
-0.004987067/-0.005682927. Main time is 16.73% higher and allocated memory
is 22.676 versus 20.531 GiB. The lag therefore retains a loss benefit
without establishing an efficiency win over MuonT.

`time-to-target-wave4-lag-exploratory-20261116.json` retains sparse
retrospective C4 crossings: lagged/current/MuonT reach <=3.59 in
5500.62/5624.38/4721.35 seconds, <=3.52 in
6164.37/6309.14/5566.31 seconds, and <=3.46 in
6824.18/6981.75/5848.18 seconds. Lagged and current first reach <=3.52
one evaluation earlier in tokens than MuonT, but later in recorded time.
These main-function timers exclude imports and parent-launch overhead;
targets were retrospective, so they remain exploratory diagnostics.

### Wave-4 Completion and Unpaired Seed 2

At 19:40:41 fresh SSH finds all four parents absent with complete manifests.
The final seed-2 candidate, `wave4/inner_outer_mlpdiag_seed2` using
`inner_outer_mlpdiag_0p5m_seed2.json`, reached 5120 updates with 22 evaluations and all
eight final checkpoint files nonempty. Its watcher records exit at 19:40:44
and keeper restoration at 19:41:00. Raw final C4 is 3.447452769 and selection
is 3.159488333; recorded main time is 7183.737 seconds, steady cost
2.574 seconds/MTok, and peak allocation 22.635 GiB. Its five broad training
means are 4.331807423, 3.643645142, 3.501389966, 3.385139668 and 3.254286806.
Selection holdouts differ across seeds, preventing a matched selection
comparison with seed-0 or seed-1 MuonT. C4 uses the common validation set,
but a cross-seed C4 difference also mixes optimizer and initialization/data
variation; this candidate still needs its same-seed MuonT companion.

The complete histories are analyzed in `PVC/run-preparation/analysis-wave4.json`.
Monitor PID 2718105 observed completion at 19:42, analyzed the wave, and
imported all four histories. `swanlab-upload-wave4.json` records overall and
per-run `verified` states at 19:43:47; each full-resolution cloud series was
checked against its original scalar history. Both 0.5M arms have 5120 updates
and both 1M arms have 2560, each with 22 evaluations. The monitor wrote
`monitor-wave4/analysis_and_upload_verified_at` and exited normally.
The seed-2 companion monitor continues and will later produce the matched
seed-2 analysis and separate cloud receipt.

### Completed Seed-2 Pair

The companion reached 5120 updates and the final eight-rank checkpoint at
21:12:34 on November 16. SSH at 21:13:06 confirms training parent PID 243798
absent and a complete manifest; its watcher records exit at 21:12:44 and
successful keeper startup at 21:13:00, new keeper PID 249983. The actual
training was uninterrupted at batch 524,288 tokens with the same seed-2
initialization, token stream and held-out mixture as the candidate.

| Quantity | `muont_seed2` | `inner_outer_mlpdiag_seed2` | Candidate Minus Baseline |
| --- | ---: | ---: | ---: |
| Final C4 | 3.451104010 | 3.447452769 | -0.003651241 |
| Final selection | 3.163097359 | 3.159488333 | -0.003609025 |
| Mean final-half C4, 11 matched evaluations | 3.584879011 | 3.580911640 | -0.003967371 |
| Mean final-half selection, 11 matched evaluations | 3.300742873 | 3.297045744 | -0.003697129 |
| Main-function total seconds | 5862.995 | 7183.737 | +1320.742, +22.53% |
| Steady seconds/MTok | 2.099 | 2.574 | +0.475 |
| Peak allocated GiB | 20.531 | 22.635 | +2.104 |

The five complete 536,870,912-token training-window deltas are
-0.008096, -0.004777, -0.003780, -0.003155 and -0.003453. All five favor
the candidate. The final-half comparison uses exactly matching
evaluation tokens, without interpolation. Raw analysis is
`PVC/run-preparation/analysis-wave4-baseline.json` and its paired report
`.txt`. Final fields are now eligible; the old wave-4 analysis preserves
its original unpaired-at-completion state and the companion analysis
supplies the later pair.

The historical import of the companion is full-resolution verified at
21:13:46, run `wave4-baseline-muont_seed2-c13ea3d6f9f9`.
`PVC/run-preparation/swanlab-upload-wave4-baseline.json` binds the complete
manifest, metrics and checkpoint receipt. Verification checks 5120 training
points, 22 C4/held-out/selection evaluations and every runtime scalar, with
exact integers and the same at-most-two-float64-ULP CSV serialization
tolerance used for earlier waves. No training tolerance changed.

Exploratory seeds 0/1/2 now give final C4 deltas -0.006264, -0.003738 and
-0.003651 nats, mean -0.004551 and sample SD 0.001484. Three seeds selected
within an adaptive campaign are not independent tuned confirmation.
Runtime remains roughly 19-23% above MuonT and matched-path attribution
is unresolved. The result supports pursuing cheaper execution/geometry,
not declaring a more efficient optimizer or scaling an untuned winner.
The retrospective C4 thresholds 3.59/3.52/3.46 give candidate main-function
crossing times 5650.44/6342.27/7021.46 seconds versus MuonT
4596.19/5442.95/5725.66. Only 3.52 crosses one sparse evaluation earlier
in tokens; all three are later in elapsed time. Evidence is
`PVC/run-preparation/time-to-target-wave4-seed2-exploratory-20261116.json`.
These retrospective, non-interpolated timings exclude imports and parent
launch overhead and are not confirmatory launch-based time-to-target.

### Next Efficiency Experiment

A code-level constraint limits the lag mechanism: `experiment/train.py`
calls `torch.cuda.synchronize()` after every update. That waits for both
the main training stream and the output-root stream, so output work already
enqueued before it cannot overlap the next forward/backward. Lag can still
move factor refinement and host flag handling, but does not automatically
hide the whole root engine. This is an implementation observation, not a
measured latency attribution.

After wave 4, compare current global synchronization with event-based
main-stream timing for a surviving lagged arm and a same-source MuonT
baseline. Protect observation-buffer lifetimes: output rows must not be
overwritten by the next backward until the previous statistics finish,
and cached graph roots must be copied before graph replay. Preserve
nonfinite checks, exact token order, checkpoint semantics, and the full
GPU and post-transition continuation gates. Use CUDA events plus explicit
dependencies for measurement; removing synchronization without protecting
those lifetimes would change or corrupt the algorithm.

The second efficiency target is fully diagonal output observation:
`DiagonalMoments.update()` currently converts and squares each full sampled
row buffer before summation, then communicates every layer's diagonal.
A fused row-square reduction can remove temporary tensors and kernel
launches without changing the pooled moment equations. Benchmark it
against the existing exact per-coordinate sums, including distributed
pooling and compiled backward capture, before a full run. Neither
optimization is active in wave-4 training or credited as an achieved speedup
here. Fused diagonal reduction is integrated only into the isolated,
unvalidated wave-5 candidate described below.

An isolated two-stage Triton row-square probe is implemented in
`tracking/benchmark_diagonal.py`. It reads the pinned sampler and actual
configurations, yielding 286 sampled positions, 1144 rows per
microbatch, and 4576/9152 rows per rank at 0.5M/1M. Eight cases cover
input hidden/MLP groups with 96/32 buffers and output hidden/MLP groups
with 160/64 buffers. Geometry and source/config hashes are saved in
`PVC/run-preparation/benchmark-diagonal-geometry-20261116.json`; the
remote geometry check matched the local check.

The probe checks six input patterns, tile tails and scratch reuse against
both the production FP32 expression and an independent FP64 reference at
`rtol=1e-5, atol=1e-6`. Timing rotates/reverses the method order and
reports CUDA-event median/p95, synchronized wall intervals and allocator
memory separately, including a Torch baseline with reusable output storage.
GPU execution is queued on the released companion node before its training
launch; receipt/log are
`PVC/run-preparation/benchmark-diagonal-20261116.{json,log}`.
A failed probe preserves evidence and cannot claim a speedup, but does not
prevent a correctly gated MuonT baseline from launching.

The probe completed on `.144` at 19:34:40, A100-SXM4-40GB, Torch 2.10.0,
Triton 3.6.0, seed 26. All 48 six-pattern/tail/scratch-reuse checks and
eight derived production-shape comparisons passed at unchanged tolerances.
Across these checks, fused-versus-production maximum relative error is
3.19603e-7; fused-versus-FP64 is 1.92565e-7. The largest absolute discrepancy,
2048 against production, occurs in intentionally large anisotropic sums
and has relative error 2.21987e-7; report relative scale with that absolute
value. Production-shaped Gaussian cases have at most 0.002929688 absolute
fused-versus-production error. All zero-overwrite checks passed.

| Batch | Group, Buffers x Width | Torch Median ms | Fused Median ms | Torch / Fused | Peak Working MiB, Torch / Fused |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5M | Input hidden, 96 x 512 | 3.877819 | 0.576096 | 6.73x | 27.126 / 6.939 |
| 0.5M | Input MLP, 32 x 2048 | 5.086576 | 0.763387 | 6.66x | 108.251 / 9.251 |
| 0.5M | Output hidden, 160 x 512 | 6.503114 | 0.944517 | 6.89x | 27.251 / 12.314 |
| 0.5M | Output MLP, 64 x 2048 | 10.151600 | 1.500635 | 6.76x | 108.501 / 18.626 |
| 1M | Input hidden, 96 x 512 | 6.923696 | 1.123413 | 6.16x | 54.188 / 13.689 |
| 1M | Input MLP, 32 x 2048 | 9.675098 | 1.484187 | 6.52x | 180.251 / 18.251 |
| 1M | Output hidden, 160 x 512 | 11.555733 | 1.855531 | 6.23x | 54.313 / 22.814 |
| 1M | Output MLP, 64 x 2048 | 19.325653 | 2.950912 | 6.55x | 180.501 / 36.751 |

Timing uses 30 rotated/reversed-order samples, three calls per sample,
with setup/compilation excluded and repeated fixed synthetic buffers.
Torch reusable-output timing differs by less than 0.04% in median;
its speedups are 6.16-6.89x as well. Working memory is allocator-rounded,
excludes input buffers, and includes fused persistent scratch/output.
Fused reductions allocate no additional temporary memory during measurement;
this does not remove their persistent storage.

This evidence supports investigating the fused integration. It excludes
communication, EMA, compiled backward capture and complete training.
The separate integrated direction-parity failure below prevents treating
the probe as candidate GPU validation or whole-training efficiency evidence.

## Wave 5: Isolated Fused-Diagonal Candidate

Implemented on November 16 in
`PVC/candidates/fused-diagonal-20261116`, extracted from the exact wave-4
source archive. The active wave-4 source and queued seed-2 helper identities
were not replaced. This candidate was reviewed and frozen at 18:36:39.
The queued A100 attempt ran at 19:37-19:39 and failed integrated direction
parity. Its immutable failed identity and logs are preserved; no full-budget
training launched.

The question is whether reusable, two-stage BF16 row-square reductions
with FP32 accumulation can reduce output-observation overhead while retaining
the existing pooled moment and parameter-update equations. The candidate
adds `experiment/diagonal_sums.py`, reusable scratch/output storage and
pointer tables, and an explicit `torch`/`triton` diagonal backend in
`block_toner.py`. It packs those sums into the existing statistics
all-reduce and executes on the same stream; it does not claim root-stream
overlap or change the checkpointed diagonal EMA equations.

| Planned Node | Arm | Batch / Seed | Purpose |
| --- | --- | --- | --- |
| `.9` | `inner_outer_mlpdiag` | 0.5M / 0 | Same-source current-backend numerical and throughput reference |
| `.80` | `inner_outer_mlpdiag_fused` | 0.5M / 0 | Fused MLP-diagonal reduction with otherwise matched geometry |
| `.144` | `inner_outer_leftdiag_fused` | 0.5M / 0 | Fully diagonal output without dense output roots |
| `.139` | `muont` | 0.5M / 0 | Same-source optimizer loss and systems baseline |

These are planned allocations from the isolated `cluster.json`, not acquired
GPUs. The seed-2 companion on `.144` takes precedence over a wave-5 launch.
All runs retain 2,684,354,560 tokens, 5120 updates, microbatch 4, eight ranks,
and the 0.5M recipe. Eight additional fused-arm configurations cover the
existing four batch sizes; only the 0.5M wave is selected.

At 18:18:16, `experiment.validate` CPU validation passed 45 tests;
`PVC/run-preparation/validation-wave5-cpu.json` records the exact command,
source hashes and stdout. The new coverage includes mixed-width pooled
diagonal EMA continuation and saved cached-root state restoration.
`PVC/run-preparation/fused-diagonal-sm80-compile.json` records successful
ahead-of-time compilation of both kernels at small/tail and production
shapes. Compilation is not accelerator numerical or latency evidence.

The candidate requires a new `fused_diagonal_pipeline` GPU gate covering
FP32-expression/independent-FP64 parity, distributed pooling, EMA roots,
update directions, scratch reuse and 24-step restored diagonal state.
Its full-real-data continuation gate is strengthened to 24 continuous
updates versus 22 plus resume and requires saved cached-root use on every
rank. Existing tolerance is unchanged. Process-local graphs are rebuilt
after restore; their activation state is saved in `toner_state.py`.
The isolated launcher requires these gates before full-budget training.
After review fixes and final formatting, the CPU receipt was refreshed at
18:36:22 with all 45 tests passing and final source hashes matching.
All candidate experiment files pass Ruff and Black. The 112 pinned Toner
files were verified after materializing them within the isolated tree;
the archived source symlink is relative and destination-local.

Read-only review at 18:27 identified two release-gate gaps: unmatched
checkpoint/evaluation quiescence at update 22 and missing parity on actual
compiled-backward rows. The isolated candidate now evaluates and checkpoints
both continuation branches at update 22, compares their boundary tensors
before restart, and checks update 24 afterward. A bounded execution-only
captured-row check compares fused sums with FP32 and independent FP64
references, and pooled EMAs/roots with Torch on the same full-model buffers
each update, including both widths of the fully diagonal output arm.
`real_data_fused_diagonal_capture` is now a required launch gate.
Follow-up review confirmed this placement but caught a metric-dispatch
regression, corrected at 18:33 before candidate binding. None of these
checks has executed on an A100 yet.

Review closed with no remaining confirmed source blockers. The source
archive is
`PVC/run-preparation/source-wave5-fused-diagonal-20261116.tar`.
`PVC/run-preparation/wave5-candidate.json` binds 270
candidate/configuration/ pinned-source files, the CPU receipt, data
receipt, validation controller, queue script and predecessor PID. All
bound files and the archive were verified before starting queue PID
2791058 at 18:37.

`tracking/queue_wave5_validation.sh` waits for `.9`'s wave-4 lagged run to
complete and its keeper watcher to restore the allocation before executing
`tracking/validate_wave5.py`. Its first snapshot at 18:37:20 found the
predecessor still running; no GPU validation has executed.
Queue snapshots/PID/lock are under `PVC/run-preparation/queue-wave5-validation/`,
with log `PVC/run-preparation/queue-wave5-validation.log`.
The controller refuses identity changes,
unrelated GPU occupancy and an incomplete predecessor, restores the keeper
after validation, and does not launch wave 5. Only measured passing A100
gates permit a later full-budget launch decision.

### Failed A100 Integration Gate, November 16

The queue observed `.9`'s predecessor complete/restored at 19:37:46,
released identified keeper PID 2865533, and ran the frozen candidate's
two-rank `experiment.gpu_checks`. Eight preceding checks printed pass:
collector, microbatch partition, distributed statistics, FA2, down updates,
synthetic compiled body, output rows and block updates. The
`fused_diagonal_pipeline` test passed its sum/EMA/root assertions before
failing the final parameter-direction comparison at `gpu_checks.py:451`.
Its tolerance remains `rtol=0.003, atol=5e-5`.

Rank 0 mismatched 606/1551 entries (39.1%), maximum absolute difference
0.000319293; rank 1 mismatched 573/1551 (36.9%), maximum 0.000297012.
Before BF16 rounding, the synthetic weight/gradient construction has rank
at most two. Finite BF16 Newton-Schulz sensitivity is a possible explanation,
not an established A100 cause. A passing sum/root tolerance does not authorize
dropping update parity.

Validation exited 1; `wave5-validation-state.json` records `needs_attention`
and successful keeper restoration at 19:39:04 (new keeper PID 2870194).
The GPU subprocess did not finish, so its aggregate checks remain false
in `validation-wave5-gpu.json` despite earlier printed passes. Full-real-data
captured-row checks and 24-continuous versus 22-plus-resume checks have not
run. Evidence is `PVC/validation/wave5_fused_diagonal/gpu_checks.log`,
`PVC/run-preparation/validation-wave5-gpu.{json,log}`, and the validation
state. The validation queue exited; the launch queue must refuse this state.
Preserve these artifacts and the frozen archive. Any correction requires
a separate source identity, fresh CPU/accelerator receipts and deliberate
queue rebinding; no silent retry or tolerance relaxation is permitted.

Read-only review found a second coverage gap: this gate uses the per-matrix
`block_update` with only left factors, while training uses
`block_update_stack` with full input/right factors. The later captured-row
gate checks sums, moments and roots, but does not compare the resulting
Torch/fused production updates. The next isolated diagnostic must preserve
the failed structured fixture and unchanged tolerances, record factor and
preconditioned-source errors plus per-iteration Newton-Schulz divergence,
and add full-rank and structured cases through the batched production path
with right factors. FP32 is a diagnostic control; production-mode parity
must still pass. No scratch/packed-buffer defect was confirmed by review.
The diagnostic implementing this plan is described below; the original
failed candidate evidence continues to block full-budget launch.

### Direction Diagnostic Results, November 16

The bounded diagnostic ran at 20:26-20:27 on two A100-SXM4-40GB ranks on
`.9`, using the unchanged frozen candidate and the command above. Controller
PID 2900337 launched Torchrun PID 2900355 and ranks 2900439/2900440. Both
ranks finished three labeled cases and wrote the gathered result at 20:27:25.
The original small fixture's first Q case reproduces the prior failure
exactly: 606/1551 mismatches with maximum 0.000319293 on rank 0,
573/1551 with maximum 0.000297012 on rank 1.
All eight small per-matrix production-mode comparisons fail; their eight
FP32 controls pass.

The twelve batched comparisons use two matrices per width per rank:
small widths 33/65, or production widths 512/2048 with 512 input columns.
Production-shaped cases include fixed dense SPD right factors and diagonal
left factors; these are synthetic batched mathematics, not compiled backward,
actual training gradients, full input-engine state or throughput measurements.
All observed moment, inner/outer factor and preconditioned-source comparisons
pass their unchanged tolerance. Maximum observed moment/factor/source
absolute errors are 1.78814e-7/1.78814e-7/1.49012e-8. Every batched `tpu`
parameter change fails `rtol=0.003, atol=5e-5`, while all twelve FP32
parameter-change controls pass.

| Case | Width | `tpu` Mismatches, Rank 0 / Rank 1 | Entries per Rank | Maximum Update Error, Rank 0 / Rank 1 | Relative Frobenius Update Error, Rank 0 / Rank 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original structured fixture | 33 | 729 / 908 | 3,102 | 3.29599e-4 / 3.42726e-4 | 1.7741% / 2.0403% |
| Original structured fixture | 65 | 1,877 / 1,690 | 6,110 | 3.60307e-4 / 3.56633e-4 | 1.9626% / 1.8204% |
| Structured production shape | 512 | 210 / 173 | 524,288 | 1.02472e-4 / 8.96943e-5 | 0.6115% / 0.5927% |
| Structured production shape | 2048 | 851 / 1,070 | 2,097,152 | 1.03587e-4 / 1.15895e-4 | 0.5422% / 0.5770% |
| Gaussian production shape | 512 | 36,976 / 38,027 | 524,288 | 2.50768e-4 / 1.88601e-4 | 1.3821% / 1.3894% |
| Gaussian production shape | 2048 | 71,570 / 69,012 | 2,097,152 | 2.32786e-4 / 2.25955e-4 | 1.1510% / 1.1414% |

The Gaussian width-2048 preconditioned matrix has measured condition number
8.10/8.56 for the first matrix on ranks 0/1. It fails despite not sharing
the original near-low-rank conditioning. In Gaussian cases divergence becomes
visible by Newton-Schulz iteration 3 and grows through iteration 10.
Zero BF16 mismatches in an unnormalized source do not guarantee parity:
Gaussian width 512 records zero such mismatches on both ranks but still
fails. Production `tpu` math normalizes in FP32 and repeatedly casts GEMM
operands during the iteration. This evidence is consistent with small
reduction/factor round-off being amplified by mixed-precision finite
Newton-Schulz; it does not establish one faulty kernel operation or a training
loss effect. The hypothesis that only the old structured fixture caused
the gate failure is rejected by these synthetic cases.

`state.json` records child/cleanup exit codes zero and exact integer
`keeper_restore=0` at 20:27:50; new keeper PID 2900664 has eight live
CUDA workers. Fresh `ps` found no diagnostic parent or ranks afterward.
The original gate receipt, frozen candidate and its 270-file binding
remain unchanged. Evidence is under
`PVC/validation/diagonal_diagnostic_20261116_2024/`: `diagnostic.log`,
`result.json`, `state.json`, and `diagnostic-source.tar`. Full tracking
verification passes 56 tests plus 20 subtests; the isolated logging
environment passes all five offline uploader tests. Scoped lint and Bash
syntax checks pass. No new training scalar history exists to upload for
this diagnostic; completed wave histories remain separately verified.

The next useful experiment is a same-source sensitivity control on a new,
preserved identity: repeat identical preconditioned-source `tpu` updates,
then apply the observed or one-ULP FP32 perturbation while holding roots
and all other inputs fixed. Measure normalization, every NS iteration,
unscaled direction and final parameter change, with unchanged tolerances.
This can distinguish deterministic repeat parity from finite-precision
perturbation amplification before changing a reducer or an optimizer.
Swap only the inner versus outer left factors to localize where divergence
enters, and compare sums/EMA/roots against independent FP64 references.
Retain the original fixture and Gaussian width-2048 case; Torch comparison
alone does not establish which reducer is closer to exact arithmetic.
If a precision-preserving reducer is pursued, it must pass the original
fixture, full-rank batched/right-factor controls, actual captured rows and
24-update restart gates. FP32 NS or source quantization would be separately
labeled numerical arms with matched controls and measured cost, not silent
fixes. This was the proposed next diagnostic at 20:27; its later
implementation and measured outcome are recorded below. The existing
failed wave-5 candidate remains ineligible.

### Sensitivity Follow-Up Results

The new two-A100 execution completed on November 16 at 20:43:22 with the
unchanged frozen wave-5 candidate, production `ns_mode: "tpu"`, ten NS
iterations, moment tolerance `rtol=1e-5, atol=1e-6`, and parameter-change
tolerance `rtol=0.003, atol=5e-5`. It retains the small structured fixture
and structured/Gaussian production shapes with fixed dense SPD right
factors. These are synthetic controls, not actual training gradients,
compiled capture, optimizer trajectories or measured loss.

| Batched Control, Both Ranks | Production `tpu` Pass / Total | Maximum Parameter-Change Error | FP32 Control Pass / Total |
| --- | ---: | ---: | ---: |
| Identical repeat | 12 / 12, bitwise equal | 0 | 12 / 12 |
| Swap inner factor only | 0 / 12 | 3.60307e-4 | 12 / 12 |
| Swap outer factor only | 12 / 12 | 1.86265e-9 | 12 / 12 |
| Swap both factors | 0 / 12 | 3.60307e-4 | 12 / 12 |
| Move every FP32 source element one ULP toward positive infinity | 0 / 12 | 5.23166e-4 | 12 / 12 |

The eight original per-matrix production checks also remain failures.
Repeat determinism makes the separate upstream-prefix traces useful for
localizing perturbation growth, but each prefix is a separate execution,
not a single instrumented trajectory. The one-ULP control moves every
element; it does not claim that changing one matrix element suffices.
Independent FP64 pooled EMA and diagonal-root references pass for both
reducers at the original moment tolerance. Neither reducer is uniformly
closer to FP64, so this evidence does not identify a faulty reduction.

Together, the controls support deterministic mixed-precision finite-NS
sensitivity to tiny inner-factor/source perturbations in the tested cases.
They motivate testing an output factor applied only after NS, where output
reduction round-off does not change the NS input. They do not establish
its loss retention, production captured-update parity, efficiency, or
general numerical robustness. FP32 is only a diagnostic control and is
not substituted for the production arm.
The outer-only swap control retains the Torch inner factor inside NS; it
isolates outer-factor round-off, not the new geometry with no inner-left
factor. Wave 6 therefore requires direct checks with `left_inner=None`.

Controller PID 2904738, Torchrun PID 2904747 and ranks 2904815/2904816
exited with child and cleanup codes zero. `state.json` records exact
integer `keeper_restore=0` at 20:43:47; keeper PID 2905040 was freshly
found live afterward. Evidence is
`PVC/validation/diagonal_diagnostic_20261116_2040/`, including
`result.json`, `state.json`, `diagnostic.log`, and
`diagnostic-source.tar`. The failed gate, wave-5 binding and its source
archives remain unchanged. No new training scalar history exists to
upload for these diagnostics.

At 20:31:11, a fresh SSH snapshot found the seed-2 MuonT companion still
running on `.144` at 2928/5120 updates. The 20:31:41 analysis reads 2955
updates and two complete 536,870,912-token windows, candidate minus baseline
-0.008096 and -0.004777. Six exact-shared late C4/selection evaluations
give means -0.003120/-0.003157; final outcomes remain withheld. Candidate
steady seconds/MTok are 2.566 versus 2.101, still no measured efficiency
advantage at the observed common token range. Neither its full-budget
paired result nor its new SwanLab import is complete. Monitor PID 2862826
and ledger PID 2784987 are live and continue every 900 seconds.

The document itself now has a detached 900-second ledger refresh:
`tracking/monitor_endeavor.sh`, PID 2784987, first successful refresh
18:32:36 on November 16. It reads existing receipts without changing
training or queued helper identities; observations retain their original
times and incomplete analyses cannot publish final loss fields. Its log is
`PVC/run-preparation/monitor-endeavor.log`, with PID/lock under
`PVC/run-preparation/monitor-endeavor/`. The loop continues across waves
until explicitly stopped. Fifteen tracking tests passed, including two
document-preservation/interim-completion tests; scoped lint and Bash syntax
also passed.

At 19:31 the unbound ledger renderer was extended to display prepared tuning
campaigns and their identity-matched saved analysis. It retains preparation
and analysis timestamps, reports complete/observed trial counts and tokens,
and withholds stale campaign results; it does not claim live training or
cloud verification from prepared configs. The existing 900-second Bash
monitor loads this renderer on each refresh. The five tuning and three
ledger cases pass together.

At 20:32, the unbound renderer also reads the launch queue's last parsed
execution event and exit receipt, separately from its immutable binding.
It displays each bounded direction diagnostic's state, exact restoration
outcome and hash-matched result receipt without treating it as a launch
gate. Five ledger tests cover preservation, incomplete final fields, stale
result identities and the failed/stopped queue. The active Bash loop loads
this renderer each refresh; no training-bound helper or frozen source changed.

## Wave 6: Outer-Diagonal Geometry

Decision recorded on November 16 after the 20:43 sensitivity result.
This is a distinct geometry/backend experiment, not a correction or
successful rerun of the failed inner-fused wave 5. The isolated mutable
candidate is `PVC/candidates/outer-diagonal-20261116/`; controller,
comparison, uploader and 900-second Bash-monitor helpers are being reviewed
under `tracking/`. At this observation it is not frozen, GPU-validated,
queued for full training, or launched.

The planned four independent eight-A100 runs share seed 0, batch 524,288
tokens, microbatch 4, 5120 updates and 2,684,354,560 input tokens:

| Node | Arm | Purpose |
| --- | --- | --- |
| `.9` | `muont` | Same-source input-MuonT reference |
| `.80` | `inner_outer_leftdiag` | Torch diagonal output inside and outside NS |
| `.144` | `outer_leftdiag` | Torch output diagonal only outside NS |
| `.139` | `outer_leftdiag_fused` | Same outer-only geometry with fused reduction |

All seven block projections retain full input factors and input-shaped
decay. The new arms use diagonal output moments at both widths, EMA 0.938,
damping 0.001, outer exponent 1/8 and unit-RMS outer factors:

$$
D_{p,t}=L_{\mathrm{out},p,t}\,
\mathcal N_{10}(U_{p,t}R_{\mathrm{in},p,t})R_{\mathrm{out},p,t},
\qquad Q_{p,t}=W_{p,t}R_{\mathrm{out},p,t}^{\,2}.
$$

`apply_left_inner=False` is the intended geometry change; output reduction
differences cannot enter this arm's NS source. No lag, precision, NS,
LR, exponent, source quantization or synchronization-policy change is
intended. The outer-only Torch versus inner/outer Torch comparison measures
loss retention and runtime after removing inner-left shaping. Outer-only
fused versus outer-only Torch tests the reducer's numerical and measured
systems effect. With four slots there is no matched block-path null;
comparison with MuonT is package performance, not pure geometry attribution.
The prior standalone reducer savings were only about 14 ms and excluded
the rest of training; no large throughput improvement is assumed.

Release requires an immutable new source/config/helper binding, final CPU
evidence, the unchanged general GPU gates, an actual outer-only integrated
pipeline gate, captured-row/update parity for every owned production shape,
and real-data continuous-24 versus 22-plus-resume coverage of the exact
four configurations. Captured-update checks must hold actual weights,
gradients, momentum and full right factors fixed while shadowing Torch and
FP64 output statistics; production NS source/output equality and original
update tolerance are required. The original failing inner-fused test remains
available and must never be reported as passing from an outer-only check.

GPU validation may use `.9` only after verified keeper release and bounded
child cleanup/restoration. Full-budget launch must additionally wait for the
seed-2 companion to complete, five-window analysis, fresh history-bound
SwanLab verification, and verified allocation ownership on every node.
Lost launch acknowledgments, partial inventories and failed restoration
must be reconciled before another release; no timeout alone authorizes a
restart. New preparation, gates, launch IDs, monitoring observations and
results will be appended here and included in the automated ledger below.
At 20:56, read-only controller review holds freezing while post-spawn
receipt failures, unconditional idle-node recovery, watcher identity
preservation, interruption-safe remote release and exact predecessor
analysis coverage are being corrected. These are unbound helper changes;
no frozen wave-5 helper or running experiment is modified.

### Failed Reference Continuation

The actual bound validation exits 1 after MuonT's continuous-24,
interrupted-22 and resumed-to-24 commands all exit zero. The update-22
boundary comparison and cached-root checkpoint inspection had returned
success, but the update-24 comparison fails at
`rank_0.model.body.blocks.3.down.weight` with the unchanged continuation
tolerance `rtol=1e-5, atol=1e-6`. Matching logged update-23/24 training losses
do not establish tensor parity; final C4 also differs by about 1.78e-6.
No inner/outer, outer-only Torch or outer-only fused full-model gate was
executed. The synthetic outer-only pass remains valid within its scope,
but no captured-row production gate, four-arm continuation pass or
full-training efficiency claim follows.

The controller finishes marker-bound cleanup with no surviving marked
process and restores integer-zero keeper status at 21:46:32. The new
keeper PID is 2999703; a subsequent status check confirms eight CUDA
contexts and health. All frozen wave-5/wave-6 sources and helper/archive
bindings remain exact. Do not retry this GPU-failed identity, reduce the
continuation scope, or relax tolerance. Next quantify the preserved
22/24 checkpoints on CPU and isolate restore/graph/root-state divergence
before preparing any new release identity.

## Wave 7: Fixed-Age Refresh Control

At 22:02:54 on November 16, fresh local process observation confirms
document monitor PID 2784987 and keeper PID 2999703 alive. The monitor's
22:02:38 cycle completed normally. No full-budget training is running.
The completed histories and Wave-6 failure remain the recorded outcomes.

A new candidate began preparation at 22:03 at
`PVC/candidates/fixed-age-outer-diagonal-20261116`, copied from the frozen
Wave-6 tree without modifying it. Its only intended scientific/runtime
configuration change is explicit `experiment.refresh_install_age=1` in
all four selected configurations. The inherited model, data, 5120-update
schedule, precision, NS, factor settings and root CUDA graphs remain
unchanged. This is a scheduling-control experiment, not an efficiency claim.

The candidate-local `RefreshTonerCore` retains the pinned numerical
engine but joins and installs pending host decompositions at the next
optimizer-update boundary, independent of `Future.done()` timing.
Early-completed refreshes wait until their fixed age; late workers are
joined at that same boundary. Pending snapshots older than an exact
fallback are discarded. Checkpoints retain the pending source update
and bind the installation-age policy. Each observed installation records
group, matrix index, source update, installation update and whether it
replaced the basis.

Both MuonT and all block input/full-output engines use that same explicit
policy. The planned four-arm allocation remains `.9` MuonT, `.80`
`inner_outer_leftdiag`, `.144` `outer_leftdiag`, `.139`
`outer_leftdiag_fused`, seed 0 and 524288 tokens/update.
CPU tests, fresh binding and all original synthetic plus actual four-arm
continuous-24/interrupted-22/resumed-24 gates must pass before launch.
Graph rewarming may still cause divergence; fixed-age scheduling is a
testable hypothesis, not a diagnosed cure. Added host joins require
measured throughput and a same-source MuonT reference.

The first focused CPU probe passes five scheduling cases and fails one
attempt to exercise the upstream compiled FP16 masked basis installer on
CPU: Inductor reports `CppVecOverrides.to_dtype()` rejecting
`use_compute_types`. Production code is unchanged. CPU scheduling coverage
therefore uses an in-memory basis store; it is not compiled-kernel evidence.
The original A100 installation and continuation gates remain mandatory.

At the initial CPU probe, this candidate was implemented but not yet
CPU-approved, frozen, GPU-validated, queued, launched or uploaded.
Frozen Wave-5/Wave-6
candidates, helpers, archives, failed checks and original tolerances
are preserved.

At 22:11:11, the first complete candidate CPU execution passes 58 tests and records
all three CPU gates in
`PVC/run-preparation/validation-wave7-cpu-pre-lint.json`.
This provisional receipt predates final formatting and binding; it is not
the launch-bound receipt. Fresh allocation status confirms `.9` PID
2999703, `.80` PID 262975, `.144` PID 249983 and `.139` PID 235248 healthy,
each with eight CUDA contexts. Controller contract tests and scoped lint
are in progress; no GPU allocation has been transferred.
By 22:13, fifty focused controller/ledger tests pass. The final validator
also requires complete 24-update histories with actual rank-0 installation
age exactly one, and checks the saved policy on every cached-root checkpoint
rank. These checks strengthen scheduling evidence without replacing any
of the original GPU or continuation gates.

At 22:14, final scoped Ruff/Black/license/AST/whitespace/EOF checks pass
for the new controller/ledger files and nine edited candidate Python files.
All 165 tracking tests plus 20 subtests and five offline uploader tests
pass. The final candidate receipt
`PVC/run-preparation/validation-wave7-cpu.json` passes all three CPU gates,
including 59 tests and the expected 265585152 parameters. A structured
configuration comparison confirms all four selected configs differ from
Wave 6 only by `refresh_install_age=1`; the inherited synthetic
`gpu_checks.py` is byte-for-byte unchanged. Immutable binding and actual
GPU validation are now authorized by these gates, not full-budget launch.

The candidate freezes at 22:15:34 with 275 source/config/pinned files
and 14 controller helpers at `PVC/controllers/wave7-20261116/tracking`.
Binding `PVC/run-preparation/wave7-candidate.json`. Archive
`PVC/run-preparation/source-wave7-fixed-age-outer-diagonal-20261116.tar`.
Independent inventory/archive checks still match both frozen Waves 5/6.
Their original failed identities are not retried.

At 22:15:59 the bound controller has released the exact `.9` keeper
generation and is running validation child PID 3053697. State:
`PVC/run-preparation/wave7-validation-state.json`. Command: `python
PVC/controllers/wave7-20261116/tracking/wave7.py --root PVC --phase
validate`. The controller owns node/controller locks, bounded child
cleanup and keeper restoration. All twenty required gates remain the
release contract; this execution has not yet established GPU parity or
training approval.

The two-A100 synthetic receipt completes at 22:17:24 with all twelve
checks true, including cached-root continuation. Structured/Gaussian
production-shaped outer-only updates preserve exact NS source, output
and momentum; maximum reported update error is `1.862645149230957e-9`,
at the unchanged tolerance. Synthetic receipt
`PVC/validation/wave7_fixed_age_outer_diagonal/gpu_checks.json`. Actual
real-data four-arm continuation remains pending, and the document
monitor refresh succeeds again at 22:17:39.

By 22:21 both MuonT continuous-24 and interrupted-22 commands have exited
zero with saved checkpoints. Actual update-22 histories record the same
rank-0 fixed-age installation sequence, including all four down-input
statistics. The original-tolerance checkpoint comparison at update 22
passes before the resume command starts by 22:23.
Update-24 continuation has not yet passed. All short-run losses are
execution evidence only, not optimizer-quality comparisons.
Fifteen additional Wave-7 watcher tests pass with scoped lint.
All 21 completed Wave-0-through-4/companion manifests and scalar histories
are reread against their verified SwanLab receipt hashes and remain
unchanged. This is source-file integrity confirmation, not a new cloud
readback or upload.

At 22:25 the validator advances to `inner_outer_leftdiag_continuous`
only after MuonT's original-tolerance all-rank update-24 comparison and
fixed-age history checks pass. MuonT continuous-24/interrupted-22/resumed-24
commands all exit zero; cached-root checkpoint evidence on every rank
and update-22 boundary comparison also pass. Fixed-age scheduling thus
addresses the observed reference continuation failure in this execution.
This is one scheduling-control gate, not a measured efficiency gain or
general causal proof. Three output-arm full-model continuation gates
are still pending, so full-budget launch remains withheld.
The expanded tracking suite passes 180 tests plus 20 subtests.

At 22:27 a prelaunch logging audit identifies that the new structured
`input_refresh_installations` list is intentionally kept in raw JSONL,
but the original historical uploader only accepts scalar metrics.
The original frozen candidate/controller remains unchanged and keeps
running numerical validation. Full-budget launch must additionally wait
for a separately identified logging-release helper binding.

The unbound `refresh_history.py` projects those retained records into
per-update/group installation counts, installed/discarded counts, age
extrema and source-update extrema; it preserves every original scalar
unchanged and rejects malformed event records. A separate historical
uploader uses that projection and the original cloud verification rules.
Raw event rows remain local and bound by the complete JSONL hash; they
are not scalar uploads. The logging-release controller must bind the
exact original Wave-7 candidate, successful GPU receipt/state and original
source/helper/archive identity before release. No numerical source/config,
tolerance, gate scope or GPU-failed identity is changed by this work.
The logging follow-up is not yet tested or frozen.

By 22:35, the separately identified logging follow-up passes 60 focused
controller/watcher/projection tests, six offline uploader tests, scoped
lint and Bash syntax. Copy-derived fixture identity failures in the first
test invocation are corrected before binding. The raw JSONL is never
rewritten; original scalar metrics use the existing exact/full-resolution
readback contract. Structured events are projected into explicitly named
scalar summaries and remain available in the hash-bound local history.
The release binding is still pending successful completion/restoration
of the original four-arm GPU validation.

At 22:33, `inner_outer_leftdiag` has passed both all-rank update-22 and
update-24 comparisons and complete fixed-age history checks; the validator
advances to `outer_leftdiag`. All three commands for each of MuonT and
inner/outer diagonal exit zero. No output-geometry loss or efficiency
claim follows from these short correctness runs.

At 22:36, the full tracking suite passes 243 tests plus 20 subtests,
including changed-original-GPU/helper/archive refusal tests for the
logging-release binding. Both historical uploader suites pass eleven
offline cases in total. Original frozen Wave-7 source, helpers and archive
are reread unchanged during validation. The third arm's continuous-24
command exits zero; its interrupted-22 trajectory is approaching the
cached-root checkpoint. No full-budget launch or new cloud import has
occurred.

At 22:41 the third arm's interrupted-22 and resumed-24 commands have both
exited zero. Its saved rank-0 resumed history records age-one installations
at both resumed updates. The parent is still executing the all-rank
update-24 checkpoint comparison; that comparison is not yet recorded as
passed. The fourth, fused arm has not started. Original controller PID
3053376 and validation child PID 3053697 remain active, with the same
marker and held allocation. No validation restart or second controller
was launched.

The latest focused tracking execution passes 71 tests, covering the
logging-release controller, watcher, structured-event projection and
document renderer. The pending scoped lint execution collected at
22:39 exits zero. The document monitor's latest successful scheduled
cycle remains 22:32:39, with its existing 900-second cadence unchanged.
Final logging-release freeze and launch still require all twenty
original GPU gates and verified integer-zero keeper restoration.

At 22:42 the validator advances to `outer_leftdiag_fused_continuous`,
establishing that `outer_leftdiag` also passes both checkpoint comparisons
and complete fixed-age history checks. The fused trajectory starts with
its production-shaped capture and update-parity verifiers enabled. Those
per-update checks pass through update 11 at the 22:42:51 observation;
continuous-24, interrupted-22 and resumed-24 are still required.
The complete tracking suite is rerun after the logging-release integrity
guard and passes 243 tests plus 20 subtests; all eleven offline uploader
tests and scoped required lint also pass. Full-budget launch remains
pending the fourth arm and controller cleanup/restoration.

At 22:45 the fused continuous-24 command has exited zero, with all 24
capture/update checks passing. The interrupted-22 trajectory also reaches
its complete cached-root checkpoint, with passing recorded checks and
the same age-one installation sequence at update 22. Its all-rank
checkpoint comparison and resumed-24 endpoint remain pending.
A read-only logging projection audit accepts the observed production
numeric fields and 241 retained refresh events without changing any
original scalar value. The audit excludes the two execution-verifier
boolean fields, which are emitted only under validation flags and are
not full-budget training metrics. It does not upload these smoke histories.
Fresh allocation status finds `.80`, `.144` and `.139` keepers healthy;
`.9` remains intentionally released for the marked validation process.

At 22:49 all thirteen GPU command receipts have integer-zero exit codes.
The fused resumed trajectory reaches update 24 with passing capture and
update-parity checks; the final all-rank checkpoint comparison is still
running. All eight observed rank-0 histories contain 24 consecutive rows.
MuonT and inner/outer diagonal each record 36 age-one installations per
trajectory, outer diagonal records 32 and fused outer diagonal records 33.
Maximum fused update-parity error across either history is
`1.3969838619232178e-9`. These observations do not substitute for the
remaining aggregate gate receipt or keeper restoration.
The existing ledger monitor completes its next scheduled cycle at
22:47:39. Independent Wave-5/Wave-6 source/archive inventories and
Wave-6/Wave-7 bound helper inventories remain unchanged.

### Completed Wave 7, November 17

All four runs finish the original 5,120-update, 2,684,354,560-token
seed-0 budget. The fused arm completes at 00:43:57, outer Torch at
00:44:42 and inner/outer at 00:44:44. Their generation-bound watchers
restore the keepers at 00:44:30, 00:45:14 and 00:45:10, respectively.
Fresh four-node SSH at 00:45:46 finds every original training parent
absent and all four watcher receipts `restored`:
`PVC/run-preparation/monitor-wave7/manual-20261117T0048.json`.
Each history has 5,120 consecutive unique training updates, 22 evaluations
and nonempty final checkpoint files for all eight ranks.

The unchanged frozen analyzer verifies the source, configuration and data
identities before writing `PVC/run-preparation/analysis-wave7.json`
and `.txt`. It compares five complete, nonoverlapping 536,870,912-token
windows and all eleven exact-shared evaluations in the final half.
Exact-shared means identical training-token positions; the late means
are arithmetic means over those eleven unsmoothed evaluation deltas.
The held-out-mixture dataset weights remain those in
[Mixture and Packing](#mixture-and-packing).
Loss deltas below are candidate minus same-source fixed-age MuonT.
These are exploratory seed-0 comparisons; the selected geometry and LR
have not undergone independent tuning and confirmation.

The four arms use separate original eight-A100-SXM4-40GB allocations,
with identical model, data, schedule and fixed-age input-factor policy.
`inner_outer_leftdiag` applies diagonal output-cotangent factors both
before and after NS; `outer_leftdiag` applies them after NS only.
Both retain dense input factors. The fused arm changes the outer
observation/reduction implementation under the same geometry.
The detailed equations and numerical bindings are retained in the
[Wave-6 definition](#wave-6-outer-diagonal-geometry) and
[Wave-7 source binding](#wave-7-fixed-age-refresh-control).

| Arm | Final C4 | Final Mixture | C4 Delta | Mixture Delta | Main Seconds | Time Delta | Peak Allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `muont` | 3.451658143 | 3.234892453 | 0 | 0 | 5979.03 | 0 | 20.531 |
| `inner_outer_leftdiag` | 3.447827534 | 3.232378417 | -0.003830609 | -0.002514036 | 6575.89 | +9.98% | 22.359 |
| `outer_leftdiag` | 3.448979533 | 3.234034251 | -0.002678610 | -0.000858202 | 6563.18 | +9.77% | 22.359 |
| `outer_leftdiag_fused` | 3.449470441 | 3.234777478 | -0.002187702 | -0.000114975 | 6508.51 | +8.86% | 22.390 |

Main-function time includes setup, compilation, evaluations and
checkpoints; it excludes Python imports and parent launch overhead.
Steady training seconds/MTok are 2.122/2.334/2.331/2.311 in table order.
This is the sum of recorded training `step_seconds` after the first
52,428,800 warmup tokens, divided by the corresponding tokens; separate
evaluation/checkpoint timers are excluded. Compilation or host waits
inside a training step remain included. The candidate costs remain higher.

| Arm Minus MuonT | Window 1 | Window 2 | Window 3 | Window 4 | Window 5 | Late C4 Mean | Late Mixture Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `inner_outer_leftdiag` | -0.005154 | -0.004800 | -0.003438 | -0.002271 | -0.001862 | -0.003238 | -0.002892 |
| `outer_leftdiag` | -0.001751 | -0.003540 | -0.002878 | -0.002110 | -0.001552 | -0.002605 | -0.002018 |
| `outer_leftdiag_fused` | -0.001187 | -0.003955 | -0.002701 | -0.001507 | -0.001518 | -0.002226 | -0.001277 |

Outer-only records 12.72 seconds less main time, or 0.19%, than
inner/outer diagonal, while final C4/mixture worsen by
0.001152/0.001656. Fused outer records 54.67 seconds less main time,
or 0.83%, than outer Torch, while final C4/mixture worsen by
0.000491/0.000743. These single-run timing differences do not estimate
run-to-run variation or isolate the causal cost of either change.
Fused outer's late-half deltas against outer Torch are also
positive, 0.000379/0.000740. This run does not establish long-horizon
floating-point equivalence or an overall efficiency win.

The existing retrospective C4 thresholds are reused unchanged:
`PVC/run-preparation/time-to-target-wave7-exploratory-20261117.json`.
No sparse evaluation or time is interpolated.

| Arm | C4 <= 3.59, Seconds | C4 <= 3.52, Seconds | C4 <= 3.46, Seconds |
| --- | ---: | ---: | ---: |
| `muont` | 4698.96 | 5551.77 | 5835.35 |
| `inner_outer_leftdiag` | 5171.57 | 5798.56 | 6423.71 |
| `outer_leftdiag` | 5161.53 | 5786.57 | 6411.86 |
| `outer_leftdiag_fused` | 5123.19 | 6049.54 | 6359.32 |

The two Torch candidates reach 3.52 one evaluation earlier in tokens,
at 2,359,820,288 versus MuonT's 2,490,892,288 tokens, but both take
longer than MuonT. All three candidates take longer at
every listed threshold. These thresholds were chosen retrospectively
in the earlier study and the timer excludes launch overhead; they do
not replace a predeclared confirmatory time-to-target experiment.

The final refresh diagnostic, recorded at 00:47:35 in
`PVC/run-preparation/analysis-wave7-refresh-final-20261117.json`,
verifies all logged rank-zero installation ages are one. It records
23712/23747/23765/23729 installed events and zero discarded events in
table order. One event is a recorded factor-basis installation for one
engine/group/matrix; age one means installation at the update boundary
immediately after its source snapshot. It is distinct from basis age
on subsequent updates. Final fallback counts are 13/25/8/15; candidate extra-root
counts are 4899/5032/4919, while MuonT does not log that field.
These counters do not isolate host wait, CUDA stage costs or causes
of the Torch/fused loss differences.

The bound historical uploader verifies all four private SwanLab histories
at 00:49:03: `PVC/run-preparation/swanlab-upload-wave7.json`.
Every arm has 5,120 `train/loss` and token-axis points, 22 C4/mixture
evaluations, verified configuration and original training provenance.
All scalar columns are checked at full resolution; integer values and
steps match exactly, with the existing two-float64-ULP CSV serialization
bound for exported floats. This bound does not change training tolerances.
Raw refresh-event lists remain in the hash-bound local JSONL; their
documented scalar projections are verified in the cloud.
The original 900-second Wave-7 monitor subsequently observes completion
at 00:57:16, reruns the same five-window analyzer and read-only cloud
verification, then exits successfully at 00:57:45:
`PVC/run-preparation/monitor-wave7/analysis_and_upload_verified_at`.
No cloud run is duplicated and this completed-wave monitor is not
restarted. The document monitor and distinct actual-LR validation
process monitor remain active.

The next experiment remains the already frozen equal-budget, four-family
LR study. Every actual short-schedule LR must first pass its own all-rank
cached-root continuation and fused captured-update gates after complete
Wave-7 cloud verification. Trial ranking uses final held-out-mixture
loss only. The combined outer-update arithmetic was initially a separate
CPU-only cost hypothesis and now has a separately prepared production GPU diagnostic;
the running and frozen numerical sources are unchanged. See the
[actual LR launch record](#actual-lr-validation-started-november-17)
for the separately identified validation execution.

### Combined Outer-Update Efficiency Hypothesis

Source inspection finds four dense right-factor products per matrix in
the frozen candidate's `block_update_stack`: one before NS, one on the
direction after NS and two on input-shaped decay. The pinned MuonT
`_muont_out` combines post-NS direction and decay into one final right
product, for three total. Both paths currently stack factors per owned
projection, including projections sharing input statistics; this is
not unique to the candidate and does not isolate the measured slowdown.
Candidate phase telemetry cannot currently attribute the full penalty.

For diagonal outer-left factor `L`, unit-RMS right factor `R`, unchanged
NS output `X` and shape scale `s`, the proposed post-NS combination is
`-(lr * decay * W @ R + lr * s * L @ X) @ R`.
The reference evaluates direction and the two decay products separately.
This is the same real-arithmetic formula but a different floating-point
evaluation order. It must not be called bitwise equivalent or promoted
without unchanged-tolerance production and full-model checks.

`tracking/diagnose_combined_outer.py` implements this isolated diagnostic,
loading only the exact frozen `block_update_stack`, normalization and NS
definitions with the existing AST loader. It performs CPU FP32 cases
with NS mode `f32`, deliberately not claiming production `tpu` validation.
It holds the reference NS output unchanged and checks only post-NS
arithmetic for square/tall/wide matrices, Gaussian/zero/anisotropic
inputs and zero/base/small learning rates.

At 00:35:31 the post-lint receipt saves 27 passing fixtures:
`PVC/run-preparation/diagnostic-combined-outer-cpu-post-lint-20261117.json`.
Maximum absolute error is 1.490116119385e-8 and relative Frobenius error
1.437206549326e-7 at unchanged `rtol=1e-5, atol=1e-6`. all recorded
candidate/config/NS source hashes are rechecked. The pre-lint receipt is
preserved separately. Six independent FP64 expression tests pass; the
full tracking suite passes 298 tests plus 20 subtests, with scoped lint
passing. This probe uses no GPU allocation.

Executed owned stacks per rank are 16x512x512, 8x2048x512 and 4x512x2048.
Removing one right product saves a theoretical 30,064,771,072 multiply-add
FLOPs per rank/update using two FLOPs per multiply-add. This does not
predict memory, latency, total optimizer speed or whole-training gains.
The follow-up after the LR study would compare unchanged reference
and separately identified combined-arithmetic paths, first with
production `tpu` NS source/output and outer-update parity, then
all-rank continuous24 versus22+resume24 with cached root graphs,
fresh distributed capture/statistics gates, and phase/total timing.
Any parity or continuation failure blocks training. No candidate,
frozen helper, graph setting, tolerance, configuration or launch is
changed by this CPU diagnostic.

### Wave 8: Combined Outer-Diagonal Candidate

The passing production-shape probe motivates a separately copied
candidate at `PVC/candidates/combined-outer-fixed-age-20261117`.
The executed/frozen Wave-7 trainer, its source archives and the active
actual-LR campaign remain untouched. Wave 8 is implemented but not
frozen, GPU-validated or full-budget launched at the 01:50 observation.

The four proposed same-source seed-0, 0.5M arms are `muont`,
`outer_leftdiag`, `outer_leftdiag_fused` and
`outer_leftdiag_combined_fused`. They retain 5,120 updates,
2,684,354,560 tokens, original LR schedules, data, microbatch4/world8,
fixed refresh-installation age1 and root CUDA graphs. The combined
configuration differs from fused outer-diagonal only in its arm identity.
MuonT and both uncombined controls retain their existing update equations.
AST checks confirm unchanged `block_update`, `block_update_stack`
and `stack_unit_rms`. The candidate keeps the exact original pinned
source inventory; the complete inventory now has 280 files, including
the new pre-CUDA execution-ownership module.

`BlockArm.combine_outer_products` selects a distinct
`combined_outer_update_stack` only for the new arm. The parameter
delta follows the production-tested three-right-product formula,
retaining exact momentum and pre-NS arithmetic. No inner output factor
is applied. The existing real-data outer verifier still checks fused
captured statistics against Torch/FP64 and outer-update parity at its
original tolerance. An additional same-factor combined-versus-reference
check requires exact momentum/NS source/output and parameter-delta
`rtol=1e-5, atol=1e-6` before every owned shape is applied.
This tighter arithmetic check does not replace the original gates.

Validation covers the original synthetic NCCL/statistics, FA2,
compiled-body, cached-root and microbatch checks, plus actual candidate
production-shaped combined updates. Each of all four full-model configs
must pass continuous24 versus interrupted22/resumed24 across all eight
ranks at the original continuation tolerance. Both fused trajectories
must log captured-row/outer-update parity for all 24 updates; the
combined arm must additionally log strict combined-update parity for
every update. Short gate losses are not optimizer-quality measurements.

The pre-review CPU receipt at 01:38:10 records 72 passing tests:
`PVC/run-preparation/validation-wave8-cpu-pre-review.json`.
Twelve FP64 reference cases cover square/tall/wide, zero/nonzero LR
and both momentum conventions. An actual `BlockToner.step` test checks
combined dispatch for every owned shape. Four controller regressions
pass for wrong-node refusal, changed uncombined equations, unrelated
commands with success flags, and an altered rank7 checkpoint.
Scoped Ruff/Black checks pass for the unbound controller. Final review,
canonical source-bound CPU receipt and immutable freeze remain pending.

At 01:49:32 the canonical CPU receipt records 75 passing tests after
the first independent ownership review. Direct CUDA validation/training
must be descended from the exact generation-bound marked validator;
the validator waits up to30 seconds for its controller's spawn
acknowledgment while continuously requiring matching PID/start ticks
and argv. Training rejects changed config/data/output/resume paths,
unbounded steps and altered capture flags before CUDA initialization.
`launch --launch` and `start_local` explicitly refuse production.
Three new regressions cover missing ownership, the acknowledgment
race and changed/unbounded training trajectories.

The second review identifies a forward-contract blocker before freeze:
a validation-only guard would also reject a future authorized
full-budget run. An explicit production-ownership branch is now being
added to this still-unbound candidate. It requires a separate immutable
launch release tied to the exact passing candidate/GPU/restoration
receipts, source/helper/archive hashes, node-to-arm assignment and
marked worker/launcher generations and ancestry. Failed validation
authorization must never fall back to production. The prior CPU
receipt is therefore not approval of this newer edit; final tests,
canonical receipt and independent review must precede freeze.

The production-ownership follow-up is implemented and CPU-tested by
01:54:55. It is activated only by explicit `--launch-release` pointing
to canonical `PVC/run-preparation/wave8-launch-binding.json`; that
release does not yet exist. It additionally requires the exact
full-budget config/data/output command, environment, passed GPU
receipt and integer-zero cleanup/keeper restoration, immutable
candidate/helper/archive hashes, actual node assignment, and live
generation-bound marked worker/torchrun ancestry. No resume or
execution-check flag is released. A failed validation ownership check
has no production fallback. `launch.py` remains disabled.

The final review adds one further data-identity check: the launch
release must retain the frozen GPU-validated data-receipt hash, not
merely match a currently replaced file. A regression substitutes the
data receipt and updates the launch binding, and is still refused. The
final canonical CPU receipt is
`PVC/run-preparation/validation-wave8-cpu.json`, dated01:58:47: 84
passing tests. The earlier focused ownership suite passes15; nine new
cases cover valid production authorization and missing/altered
release/helper/node/ flags/markers/generation/data plus refused
fallback. The established tracking suite passes315 tests plus20
subtests; offline uploader tests pass12. The first broad command failed
collection because the experiment import path and SwanLab environment
were omitted; the corrected split-environment reruns pass. Scoped
Ruff/Black, Markdown/file checks and `git diff --check` pass.
The280-file inventory, unchanged reference ASTs and completed54-case
production probe binding are rechecked. Fresh `.139` SSH at01:56 finds
only restored keeper248199 and workers248313-248320 on the original
eight A100-SXM4-40GBs; no training or new validation is running there.
Independent read-only review confirms the last blocker is resolved
at01:59; immutable freeze and exact controller-owned validation are
next, not full-budget training approval.

`tracking/wave8.py` reuses the 22 byte-identical actual-LR allocation
helpers, with new source checks and explicit `.139` validation ownership.
It requires the passing 54-case probe, complete/cloud-bound Wave 7,
a fresh predecessor process snapshot, original eight-A100 hardware
and the existing per-node lock before exact keeper-generation release.
The numerical worker checks live controller PID/start ticks, argv,
parent/marked-child identity and source binding before GPU validation.
The controller reconstructs all-rank checkpoint comparisons and exact
command/history coverage, joins marked children and restores the keeper.
No automatic training handoff is installed at this preparation stage.
Full-budget launch must also respect the active LR campaign's allocation
and cannot preempt its workers.

After passing all actual gates, compare every full 536,870,912-token
training window and all eleven exact-token final-half evaluations.
Report combined-minus-uncombined-fused loss and runtime separately
from every candidate-minus-MuonT comparison, including phase/main/
end-to-end time, memory and retrospective loss-target crossings.
The primary hypothesis is reduced update cost without losing the
outer-diagonal fixed-token improvement; existing probe percentages
do not predict an achieved full-model saving. New training results
will be imported and verified in the same private SwanLab project
only after completion, then recorded here.

## Remaining Experimental Program

The initial wave is complete. The following controls, tuning campaign,
diagnostics, and confirmation runs remain planned. Existing code supports
the five first-stage arms; later controls require implementation and validation.

### Batch-Specific Configuration

Each full run retains the 2,684,354,560-input-token budget, 4096-token context,
and eight ranks. Active A100 configurations use microbatch 4 for every arm.
The pinned author microbatch is 8; its accumulation counts are half the
active A100 counts shown below.

| Field | 0.5M | 1M | 2M | 4M Extension |
| --- | ---: | ---: | ---: | ---: |
| Input tokens/update | 524288 | 1048576 | 2097152 | 4194304 |
| Global sequences | 128 | 256 | 512 | 1024 |
| Microbatch/rank | 4 | 4 | 4 | 4 |
| Accumulation | 4 | 8 | 16 | 32 |
| Full updates | 5120 | 2560 | 1280 | 640 |
| Muon matrix LR | 0.016 | 0.016 | 0.032 | 0.0452548 |
| MuonT/candidate initial matrix LR | 0.02262741696 | 0.02262741696 | 0.04525483392 | 0.063999951815088 |
| Auxiliary AdamW LR | 0.0048 | 0.006788224 | 0.0096 | 0.0095999912336 |
| Matrix decay coefficient | 0.1 | 0.2 | 0.141421 | 0.141421 |
| Auxiliary beta1 | 0.8 | 0.8 | 0.8 | 0.8 |
| Auxiliary beta2 | 0.98 | 0.9604 | 0.92236816 | 0.85076302 |
| Stable updates | 1024 | 512 | 256 | 128 |
| Linear-decay updates | 4096 | 2048 | 1024 | 512 |

These settings come from the pinned batch recipes and generated
[executed-family configurations](experiment/configs/). They do not imply
that the larger-batch runs have been executed. Common momentum, NS arithmetic,
auxiliary routing, and input-factor settings remain those documented above.

### Survivor Controls

1. **Matched additive custom-down null-output control.** For outer-left,
   force $L_{\mathrm{out}}=I$ while retaining its input factors and custom
   per-matrix NS/direct-product path. For inner-and-outer down, also force
   $L_{\mathrm{in}}=I$. Compare with upstream input MuonT. Retain collection
   and root work when isolating output geometry; separately measure a
   no-collection version for systems attribution. An identity-left hybrid
   instead yields $D=\mathcal N_{10}(U), Q=W$ on down matrices; compare it
   with a matched plain-down update at the same LR and with input MuonT
   as separate replacement baselines. It is not a MuonT-equivalent null.
2. **Direction/decay factorial.** For additive two-sided candidates, cross
   presence/absence of outer-left scaling with input-only versus two-sided
   decay. For replacement candidates, cross presence/absence of outer-left
   scaling with isotropic versus left-shaped decay.
3. **Norm and LR controls.** Compare independently LR-tuned original
   directions with per-matrix grafting
   $D_{\mathrm{graft}}=D\|D_{\mathrm{input}}\|_F/\|D\|_F$.
   Keep decay unchanged and report the extra reference-direction work.
   Define zero-direction handling before implementation.
4. **Information controls.** Use diagonal output moments, observed
   eigenvalues with a seeded randomized basis, and shuffled eigenvalues in
   the observed basis. Also test
   $B_{\mathrm{grad},t}=G_tG_t^{\mathsf T}/n$ with the same raw EMA,
   damping, powers, and refresh. This gradient-Gram control is inspired by
   Mousse; it is not a reproduction.
5. **GO-MUON-form control.** Define $R_{1/4}=\mathsf{Out}_n(C)$ and
   $L_{1/4}=\mathsf{Out}_m(B)$ with their spectral exponent changed from
   $1/8$ to $1/4$. Use these same quarter-power factors inside and outside:
   $D_0=L_{1/4}\mathcal N_{10}(L_{1/4}UR_{1/4})R_{1/4}$.
   Graft $D_0$ per matrix to the input-MuonT direction's Frobenius norm,
   $D=D_0\|D_{\mathrm{input}}\|_F/\|D_0\|_F$, and use
   $Q=WR_{\mathrm{out}}^2$ with the usual input outer power $1/8$.
   This explicitly chosen decay/common-factor recipe is a formula control;
   author implementation and decay parity remain unresolved.
6. **Dimension-capped input control.** Use right-only block-diagonal down
   factors capped at width 512 to separate smaller-factor benefits from
   output-information benefits.
7. **Exact-polar diagnostics.** On small matrices, compare exact SVD-polar
   with the actual finite NS map, specifying rank-deficient conventions.

Full Bi-MuonT applies both inner and outer left/right factors to every
block matrix; it is now defined as
[`inner_outer_all`](#two-sided-all-blocks-inner_outer_all-planned) with its
`null_all` control. Left-only shaping of every block matrix is defined as
[`left_all`](#left-only-all-blocks-left_all-planned) with its `null_left_all`
control. Full hybrid would retain input factors on Q/K/V/gate/up
and use separate output factors on O/down. None of these is an executed arm.

### Fair Tuning

Each family and batch gets the same token allowance: three matrix-LR
multipliers $\{1/\sqrt2,1,\sqrt2\}$ and at most six additional refinements.
Families are Muon, input MuonT, and each surviving candidate.
Use 167,772,160 tokens per short-budget trial: 320/160/80 updates at
0.5M/1M/2M, with a completed 20%-stable/80%-linear-decay schedule.
These are separate short-budget studies, not early stops of the full schedule.

Candidate refinements are bounded one-factor changes motivated by diagnostics:
output inner power $\{0,1/4,1/2\}$, outer power $\{1/16,1/8\}$,
damping $\{10^{-3},10^{-2}\}$, EMA $\{0.938,0.97,0.99\}$, and refresh
$\{1,4,8\}$. Do not run their full Cartesian product.
Muon may refine LR/decay; input MuonT may refine input strengths, damping,
EMA, and refresh within the same allowance. Changing input damping from
0.001 requires an exact-root reference backend because the pinned fast
backend fixes it.

Initially keep each batch's auxiliary LR and matrix decay fixed while
varying matrix LR. On survivors, separately compare matched `LR * decay`
settings; LR changes otherwise change regularization strength.
Count failed-run tokens in each family's allowance.

Tuning uses seed 0's fixed 64 reserved windows per training source, excluded
from every tuning run. Aggregate token-weighted source losses with mixture
weights 3.8/0.25/0.055. Freeze settings before confirmation.
C4 has already been inspected during exploration; disclose that use and
retain it as the reporting benchmark.

#### Completed Down-LR Round, November 17

All four down-LR trials complete by 02:57:00, with update-320 receipts
and nonempty checkpoints on all eight ranks. The bound analyzer saves
their complete results at 02:57:40 to the same campaign `analysis.json`.
Each uses 167,772,160 tokens and matrix LR 0.015999999973151494. The
remaining base/up trials keep the study status incomplete.

| Down-LR Family | Final C4 | Final Selection | Late C4 Delta | Late Selection Delta | Total Seconds | Extra Total Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `muont` | 4.724832 | 4.566138 | 0 | 0 | 528.68 | 0 |
| `inner_outer_leftdiag` | 4.733083 | 4.578702 | +0.002536 | +0.004936 | 565.34 | +6.93% |
| `outer_leftdiag` | 4.740426 | 4.585450 | +0.010404 | +0.012569 | 567.87 | +7.41% |
| `outer_leftdiag_fused` | 4.740958 | 4.588685 | +0.011279 | +0.014082 | 548.74 | +3.79% |

Late columns average five actual evaluations in the final half of the
trial's tokens; deltas are candidate minus down-LR MuonT. Final
selection deltas are +0.012565/+0.019312/+0.022548. Sparse held-out
points are not interpolated. Total seconds use the original manifest
timer, not cloud-import duration.

The five complete 33,554,432-token training-window deltas are:

| Down-LR Candidate | Window 1 | Window 2 | Window 3 | Window 4 | Window 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inner_outer_leftdiag` | +0.009780 | -0.005582 | -0.002560 | +0.002086 | +0.009196 |
| `outer_leftdiag` | +0.005017 | +0.000010 | +0.005727 | +0.009847 | +0.017135 |
| `outer_leftdiag_fused` | +0.005168 | +0.002360 | +0.007338 | +0.011239 | +0.017615 |

The inner/outer advantage in the middle windows does not persist at
the short-schedule endpoint. All three candidates are worse in final
and late held-out loss at this down LR and cost more total time.
This is a completed matched-LR round, not independent family tuning.
The 320-update schedule has its own cooldown and cannot establish the
ranking under the 5120-update schedule. Finish base/up with the same
token allowance before selecting each family's short-study LR.

MuonT's history is full-resolution SwanLab-verified at 02:56:01;
inner/outer and outer diagonal follow at 02:57. Each imported history
has 320 training/token points and nine held-out evaluations, plus
original configuration, runtime, refresh scalars and checkpoint
provenance. The existing integer-exact/two-float64-ULP CSV export
bound remains unchanged. The campaign cloud receipt is
`PVC/campaigns/lr-fixed-age-0p5m-20261116/swanlab-upload.json` and
remains partial until all 12 histories are verified. Fused down-LR
verification completes at 02:58:24, and that invocation also rechecks
the other three original cloud histories without duplicating runs.
The saved receipt reports `partial_verified`, 4 of 12 histories,
each with 320 training/token points and nine evaluations.

Fresh bound SSH snapshot at 02:58:36 saves all four workers running
with their down-LR child cleanup completed and base-LR children active.
It is recorded in `monitor/down-complete-20261117T0258.json` and
becomes `monitor/latest.json`. The sequential read records base
MuonT/inner-outer/outer at 77/31/10 updates, while fused base startup
has a running manifest but no scalar row yet. No family has all
three LRs complete, and no LR choice is recorded. Workers retain their
nodes through base/up; final keeper restoration is still pending
completion of each family.

#### Completed Base-LR Round, November 17

All four base-LR trials have complete update-320 histories and final
eight-rank checkpoint receipts by the bound analysis at 03:06:46. The
preserved analysis is
`PVC/campaigns/lr-fixed-age-0p5m-20261116/analysis-base-complete-20261117T0307.json`.
It also becomes the campaign's current `analysis.json`; subsequent
monitor cycles may advance that current file. Every base trial uses
matrix LR 0.02262741696 and the same 167,772,160-token short schedule.
Eight of twelve trials are complete, with no complete family sweep or
selected LR.

| Base-LR Family | Final C4 | Final Selection | Late C4 Delta | Late Selection Delta | Total Seconds | Extra Total Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `muont` | 4.597693 | 4.422130 | 0 | 0 | 497.99 | 0 |
| `inner_outer_leftdiag` | 4.602934 | 4.431168 | -0.000636 | +0.002315 | 545.14 | +9.47% |
| `outer_leftdiag` | 4.612755 | 4.441723 | +0.009435 | +0.013340 | 539.57 | +8.35% |
| `outer_leftdiag_fused` | 4.612543 | 4.441234 | +0.011058 | +0.012785 | 540.13 | +8.46% |

Late deltas use the five actual evaluations in the final half, matched
to base-LR MuonT. A direct history recheck at 03:10 confirms the exact
same late C4/selection tokens in both down and base rounds:
84,410,368; 105,381,888; 126,353,408; 147,324,928; 167,772,160.
No missing evaluation is inferred. Final selection deltas are
+0.009038/+0.019593/+0.019104 nats. Original manifest timers supply
total seconds; cloud-import timing is not training runtime.

| Base-LR Candidate | Window 1 | Window 2 | Window 3 | Window 4 | Window 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inner_outer_leftdiag` | +0.004380 | -0.004524 | -0.007407 | +0.002513 | +0.008653 |
| `outer_leftdiag` | -0.004893 | -0.000724 | +0.001887 | +0.012536 | +0.019368 |
| `outer_leftdiag_fused` | -0.004123 | +0.001259 | +0.002097 | +0.012461 | +0.018053 |

Each window has 33,554,432 tokens and 64 paired updates. The middle
inner/outer training-loss advantage does not persist at the endpoint.
Its small late C4 advantage also does not carry to final C4 or final
held-out-mixture selection. All candidates cost more total time.
The base LR improves final selection over down LR within each family,
but up LR must finish before any choice is made. Neither this result
nor the down round establishes the 5,120-update ranking.

Fresh bound SSH at 03:07:04 records all four family workers running,
with all down/base histories complete. MuonT's up-LR trial has 60
updates and inner/outer up has a running manifest but no scalar yet;
outer/fused up have not yet written manifests at the sequential read.
This observation is preserved in
`monitor/base-complete-20261117T0307.json` and copied to
`monitor/latest.json`. Workers retain their node locks until all
three trials finish, marked-child cleanup succeeds and keepers are
restored. Wave-8 full-budget launch remains pending complete-study
analysis, private cloud verification and fresh restoration checks.

The separate frozen uploader completes all four base imports and
reverifies the four original down histories at 03:07:44. Each has
320 original training/token points and nine C4/selection/held-out
evaluations. All original scalar columns, trial configurations and
checkpoint provenance are verified under the unchanged integer-exact/
two-float64-ULP serialization bound. The current receipt is
`swanlab-upload.json`, status `partial_verified`, 8 of 12 histories.
This is not complete-study cloud verification or an LR choice.

The mutable `tracking/update_endeavor.py` now renders saved final
selection, late held-out deltas, final total timers and same-LR
complete shared training-window deltas at each 900-second refresh.
Running or failed endpoints cannot publish final values; missing
windows and unmatched LRs remain unpaired. Each saved trial's unique
name, family and LR tuple must match the campaign plan. Because the
frozen analyzer does not retain late evaluation tokens/counts, the
recurring aggregate late-mean deltas are explicitly coverage-unverified,
not paired held-out estimates. The manual down/base tables above use
the separate direct history coverage check.
It preserves the scientific
prose outside its markers and changes no frozen numerical, launch,
analysis or uploader helper. Verification passes 345 tracking tests,
12 isolated offline uploader tests and scoped Ruff/Black checks.
The document monitor's existing 03:02:59 cycle succeeds without a
duplicate monitor.

#### Completed Initial LR Study, November 17

All twelve trials are complete in the scheduled bound analysis at
03:16:37. The preserved result is
`analysis-complete-20261117T0316.json`. Each family consumes exactly
503,316,480 tokens; the full initial study consumes 2,013,265,920
tokens. No failed trial or extra search allowance is excluded. The
three-LR selection losses are:

| Family | Down LR Selection | Base LR Selection | Up LR Selection | Selected Tested LR |
| --- | ---: | ---: | ---: | ---: |
| `muont` | 4.566138 | 4.422130 | 4.334664 | 0.03199999994630299 |
| `inner_outer_leftdiag` | 4.578702 | 4.431168 | 4.332589 | 0.03199999994630299 |
| `outer_leftdiag` | 4.585450 | 4.441723 | 4.333676 | 0.03199999994630299 |
| `outer_leftdiag_fused` | 4.588685 | 4.441234 | 4.334616 | 0.03199999994630299 |

Each selection uses only completed final held-out-mixture loss.
C4 does not select any LR. Every winner lies at the upper boundary
and every family improves monotonically across the tested grid.
This study therefore does not locate any family's optimum. Selection
and comparison reuse the exploratory seed-0 holdout, so the
best-tested comparison is not independent confirmation.

| Up-LR Family | Final C4 | Final Selection Delta | Late C4 Delta | Late Selection Delta | Total Seconds | Extra Total Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `muont` | 4.519382 | 0 | 0 | 0 | 494.08 | 0 |
| `inner_outer_leftdiag` | 4.517309 | -0.002075 | -0.002908 | -0.002772 | 545.24 | +10.35% |
| `outer_leftdiag` | 4.517034 | -0.000988 | -0.003381 | -0.001444 | 537.75 | +8.84% |
| `outer_leftdiag_fused` | 4.518962 | -0.000047 | -0.001200 | -0.000588 | 535.96 | +8.48% |

These deltas compare each candidate with up-LR MuonT. Direct
history reads confirm the same five late C4/selection token positions
as down/base: 84,410,368; 105,381,888; 126,353,408; 147,324,928;
167,772,160. Original full-trial manifest timers supply total seconds.

| Up-LR Candidate | Window 1 | Window 2 | Window 3 | Window 4 | Window 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inner_outer_leftdiag` | -0.011642 | -0.013921 | -0.009562 | +0.001325 | +0.001246 |
| `outer_leftdiag` | -0.009354 | -0.007643 | -0.004334 | +0.000988 | -0.000021 |
| `outer_leftdiag_fused` | -0.008732 | -0.008339 | -0.003423 | +0.002589 | +0.001450 |

All windows have 33,554,432 tokens and 64 paired training points.
The early/middle training gains mostly diminish in the last two
windows. Final/late held-out means retain small gains at this LR,
unlike down/base endpoints, but no candidate is faster overall.
Fused outer versus Torch outer remains a numerically distinct
long trajectory; its approximately 1.79-second total-time saving
does not establish equivalence or meaningful optimizer efficiency.
The short-schedule cooldown is not the 5,120-update cooldown.

All twelve cloud histories are full-resolution verified at 03:17:47.
`swanlab-upload.json` now reports `verified`, 12 of 12 histories;
each has 320 original training/token points and nine evaluations,
with unchanged scalar/configuration/checkpoint provenance checks.
The scheduled monitor performs the final three candidate imports
and reverifies the original nine without duplicating runs.

Worker restoration receipts finish at 03:12:51, 03:15:38,
03:16:18 and 03:16:44 in family-table order. The scheduled
03:16:34 snapshot crossed the fused cleanup boundary and still
reports `active`; it is not sufficient for the full-budget handoff.
A separate fresh all-node observation at 03:18:10 verifies every
worker complete, all workers terminal and no final-trial marked
descendants; it is preserved in
`monitor/complete-restored-20261117T0317.json` and becomes the current
`monitor/latest.json`. Additional SSH at 03:20-03:21 verifies the
restored keeper generations and exact eight-worker CUDA occupancy:

| Original Node | Restored Keeper PID | Healthy Owned CUDA Workers |
| --- | ---: | ---: |
| `28.216.17.9` | 3502759 | 8 |
| `28.216.17.80` | 284406 | 8 |
| `28.216.17.144` | 269149 | 8 |
| `28.216.17.139` | 278523 | 8 |

The bound keeper-generation helper verifies launcher/child command
and session identities; CUDA PIDs equal the eight owned children.
The idle-training check passes on every node. The document monitor
successfully refreshes at 03:18:01 on its unchanged 900-second cadence.
Its next scheduled refresh succeeds at 03:33:02. The LR monitor's
final scheduled snapshot is complete at 03:33:29; complete analysis
is rederived at 03:33:33 and all twelve cloud histories are reverified
at 03:34:33. Monitor 3454921 records
`monitor/analysis_and_upload_verified_at`, exits normally and records
`monitor/stopped_at` at 03:34:33. It is not restarted; the document
monitor remains active across the next wave.

**Next decision.** Proceed with the separately frozen Wave-8
combined-product systems comparison only after the complete-study
handoff passes. It retains the original full-budget LR and all four
controls to measure whether production-tested arithmetic savings
survive whole-model training without compromising held-out loss.
No short-study selected LR is substituted into that release.
After Wave 8 completes, expand the upper LR boundary under a newly
predeclared equal-budget search for every compared family, with
actual-config continuation/capture gates before execution.
Then use full-budget, fresh-seed confirmation for any survivor.
The present evidence is small exploratory loss improvement with
extra work, not a better-performing and more efficient optimizer
than independently optimized MuonT.

At 03:21 the unchanged bound full-budget controller starts:

```bash
/opt/conda/envs/torch-base/bin/python \
  "$PVC/controllers/wave8-launch-20261117/tracking/wave8_release.py" \
  --root "$PVC" --phase launch
```

Its complete LR-study source/config/evidence checks, original Wave-8
all-rank numerical rederivation and fresh allocation preflights run
before training intents are created. At 03:21:50 controller 3515308 is
active but no `wave8-launch-state.json` or `runs/wave8` exists. This is
controller execution, not acknowledgment of any full-budget GPU training
run. By 03:32:44 the original all-rank numerical rederivation has passed
without a timeout or tolerance change. The controller advances through
fresh allocation inspections: MuonT `.9` completes and outer diagonal
`.80` is in progress. No launch intent exists at this intermediate
stage. By 03:35:03, all four inspections pass and
`PVC/run-preparation/wave8-launch-state.json` records `launching` under
the unchanged release. Monitor 3535620 is positively acknowledged and
repeats every 900 seconds under `PVC/run-preparation/monitor-wave8/`.
Its first snapshot/analysis cycle runs before any training manifest
exists and reports `analysis_incomplete_or_refused` at 03:35:11, as
required. No cloud import is attempted. The MuonT start request is in
progress; the four-arm launch has not yet been acknowledged. The
reviewed mutable document/results helpers and this completed-LR record
are archived at
`PVC/run-preparation/source-endeavor-lr-results-20261117-0327.tar`. The
archive contains only `ENDEAVOR.md`, `tracking/update_endeavor.py`, its
regression tests and `tracking/README.md`; it is not a numerical
candidate or training release.

#### Completed Wave 8, November 17

All four arms complete the original seed-0, eight-A100, 524,288-token
batch schedule at matrix LR 0.02262741696: 5,120 updates and
2,684,354,560 tokens per arm. All final eight-rank checkpoint receipts
are present. The original worker receipts record exit, marked-child
cleanup and keeper restoration codes zero. The bound four-node SSH
snapshot at 05:27:44 verifies terminal worker generations, no marked
children, and restored keepers:
`PVC/run-preparation/monitor-wave8/continuation-20261117T0528.json`.

| Arm | Training Completion | Worker Completion / Keeper Restoration |
| --- | --- | --- |
| `muont` | 05:16:13 | 05:16:38 |
| `outer_leftdiag` | 05:25:40 | 05:26:04 |
| `outer_leftdiag_fused` | 05:25:35 | 05:26:02 |
| `outer_leftdiag_combined_fused` | 05:26:36 | 05:27:07 |

The unchanged frozen analyzer completes successfully:

```bash
/opt/conda/envs/torch-base/bin/python \
  "$PVC/controllers/wave8-launch-20261117/tracking/analyze_wave8.py" \
  --root "$PVC"
```

Canonical results are `PVC/run-preparation/analysis-wave8.json`. Its
source release is unchanged. Every comparison is eligible at the full
budget and retains all five 536,870,912-token windows, 1,024 original
training losses each. Negative deltas mean candidate minus same-wave
MuonT:

| Candidate | Window 1 | Window 2 | Window 3 | Window 4 | Window 5 | Final C4 Delta | Final Selection Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `outer_leftdiag` | -0.001751 | -0.003540 | -0.002878 | -0.002110 | -0.001552 | -0.002679 | -0.000858 |
| `outer_leftdiag_fused` | -0.001553 | -0.004791 | -0.003128 | -0.001820 | -0.001656 | -0.001800 | -0.003100 |
| `outer_leftdiag_combined_fused` | -0.002291 | -0.002820 | -0.002117 | -0.001377 | -0.001253 | -0.002656 | -0.001884 |

Late C4/selection means use eleven exact shared evaluations after
1,342,177,280 tokens through the full budget. Their deltas are
-0.002605/-0.002018 for Torch outer, -0.001769/-0.003252 for fused
outer, and -0.002498/-0.002727 for combined fused. The per-run table
above records the absolute late means, final-four selection deltas
and final-window training means.
Selection is the token-weighted held-out DCLM/StarCoder/Proof-Pile
mixture loss defined in [Factors by Run](#factors-by-run); C4 is the
separate Paloma `c4_en` loss. "Final" is the last evaluation at update
5,120. "Late" averages the eleven second-half evaluations, while
"final-four" averages only the last four shared evaluations.

| Arm | Final C4 | Final Selection | Main-Function Seconds | Extra Main Time | Steady Seconds / MTok | Peak Allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `muont` | 3.451658 | 3.234892 | 6004.798592 | reference | 2.122850 | 20.530577 |
| `outer_leftdiag` | 3.448980 | 3.234034 | 6523.498468 | +8.64% | 2.326477 | 22.359492 |
| `outer_leftdiag_fused` | 3.449858 | 3.231792 | 6464.938750 | +7.66% | 2.302727 | 22.389584 |
| `outer_leftdiag_combined_fused` | 3.449002 | 3.233009 | 6474.806915 | +7.83% | 2.309260 | 22.389584 |

The main-function timer includes setup, evaluation and checkpoints;
it excludes Python imports and parent-launch overhead. Every candidate
costs more per token and allocates more peak device memory than MuonT.
Steady seconds/MTok sums recorded training-step time after the first
52,428,800 tokens (100 updates), divided by the corresponding tokens.
It excludes separately timed evaluation and checkpoints.

Combined-minus-fused training-window deltas are
-0.000737/+0.001971/+0.001011/+0.000443/+0.000403.
Combined improves final C4 by 0.000856 and late C4 by 0.000729, but
worsens final selection by 0.001216 and late selection by 0.000525.
It takes 9.868165 more main-function seconds and 16.079130 more summed
training-step seconds. The [synthetic combined-product diagnostic](#wave-8-combined-outer-diagonal-candidate)
measured kernel savings; this seed's whole-training comparison has no
measured saving and does not isolate the reason for that difference.
The differing C4/selection rankings and
single selected seed do not support a general superiority claim.

The preserved retrospective C4 thresholds also show no elapsed-time win:

| Arm | Seconds to C4 <= 3.59 | Seconds to C4 <= 3.52 | Seconds to C4 <= 3.46 |
| --- | ---: | ---: | ---: |
| `muont` | 4729.482 | 5583.089 | 5867.286 |
| `outer_leftdiag` | 5124.882 | 5748.782 | 6371.964 |
| `outer_leftdiag_fused` | 5080.802 | 5697.168 | 6315.316 |
| `outer_leftdiag_combined_fused` | 5089.831 | 6016.751 | 6326.023 |

These are first completed sparse evaluations at or below each target,
timed cumulatively from entry to the recorded training main function,
with no loss/time interpolation. The retrospective thresholds and
main-function timing are exploratory, not launch-based confirmatory
time-to-target measurements.

At 05:29, original Bash monitor 3535620 remains the upload owner.
Its latest scheduled observation is mixed completion at 05:21:34;
the manually saved terminal snapshot above does not replace its input.
No complete-wave SwanLab receipt is recorded yet, and no duplicate
uploader is started. Queue 3641454 last records `waiting_for_wave8`
at 05:22:22. It must receive the original full-resolution cloud
verification, rederive the five-window analysis and capture fresh
terminal/restoration evidence before its one-shot actual-LR validator.
The existing document and strict-observation monitors remain on their
900-second cadence.

The original completion monitor subsequently saves a fresh terminal
snapshot at 05:36:49. All four private SwanLab histories are verified at
05:38:37; monitor 3535620 records `analysis_and_upload_verified_at`,
records its normal stop time and exits. The supplemental strict observer
had exited normally on `complete` at 05:30:31. The final cloud receipt
is `PVC/run-preparation/swanlab-upload-wave8.json`.

Every run's full-resolution export has 5,120 training/token points,
22 C4 and selection evaluations, and the exact local count for every
other scalar column. Configuration and token-axis checks pass.
Integer values are exact; floating export values satisfy the existing
two-ULP serialization bound, so float64 bitwise identity is not claimed.
Original manifest/history/checkpoint hashes and all eight checkpoint
rank-file sizes match each imported run's provenance.

| Arm | Verified SwanLab Run ID |
| --- | --- |
| `muont` | `wave8-muont-0c21c4ecaa85` |
| `outer_leftdiag` | `wave8-outer_leftdiag-4fc92a2125d4` |
| `outer_leftdiag_fused` | `wave8-outer_leftdiag_fused-35c081158926` |
| `outer_leftdiag_combined_fused` | `wave8-outer_leftdiag_combined_fused-bc2426a46b62` |

Run URLs and per-column verification errors/counts remain in the cloud
receipt. The original uploader alone performed the import. The boundary
queue's 05:37:38 `waiting_for_wave8_upload` observation predates this
receipt; its next cycle must validate the completed provenance and fresh
restoration before starting the unchanged actual-LR validator.

The cross-wave document monitor's 05:33 concurrent-edit refusal preserved
manual edits. Its mutable shell now logs a failed refresh and retries
after 900 seconds. The ambiguity and concurrent-edit checks in the Python
renderer remain fail-closed. A sequential Bash regression reproduces a
failed first refresh, observes preserved manual prose, and verifies a
successful second refresh with both delays exactly 900 seconds.
The monitor is restarted as detached PID 3776115, freshly observed alive
after its 05:44:45 successful refresh. No frozen training/controller
source or GPU allocation is changed.

The next experiments remain the [eight equal-budget upper-LR trials](#upper-lr-boundary-expansion-prepared-november-17),
followed by the separately frozen upstream-product/null diagnostic.
The four original fixed-age families each receive matrix LRs
0.04525483392 and 0.06399999989260598 for 320 updates,
167,772,160 tokens and nine evaluations per trial. This is short-budget
screening, separate from independent full-budget LR selection.
Combined fusion is not promoted as an efficiency improvement.
Independent full-budget LR selection and fresh paired seeds are still
required to test improvement over tuned MuonT.

##### Completed Upper-LR Sweep, November 17

All eight trials complete 320 updates/167,772,160 tokens each. Worker
restoration finishes at 07:27:45/07:30:41/07:31:13/07:31:32 in
MuonT/inner-outer/outer/fused order. Fresh bound four-node snapshot
`monitor/continuation-20261117T0735.json` verifies every worker terminal
at 07:35:14. The original frozen analyzer completes at 07:36:33:
`monitor/continuation-analysis-20261117T0736.json`. Its CLI is the
preceding analyzer command with this output filename. The original
monitor's canonical analysis is independently refreshed at 07:41:08.

The higher matrix LR is 0.06399999989260598; source/data, seed, batch,
schedule, original eight-A100 assignments and age-one policy match the
first round. Each full training window is 33,554,432 tokens/64 losses.

| Candidate Minus MuonT | Window 1 | Window 2 | Window 3 | Window 4 | Window 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inner_outer_leftdiag` | -0.001846 | -0.022072 | -0.022287 | -0.018418 | -0.014565 |
| `outer_leftdiag` | +0.005516 | -0.004500 | -0.006107 | -0.005494 | -0.005527 |
| `outer_leftdiag_fused` | +0.009130 | -0.005973 | -0.006817 | -0.006504 | -0.006478 |

| Family | Final Selection | Final Delta | Late Selection Delta | Late C4 Delta | Total Seconds | Extra Time | Steady Seconds / MTok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `muont` | 4.251045721 | reference | reference | reference | 505.213643 | reference | 2.110255 |
| `inner_outer_leftdiag` | 4.233923618 | -0.017122102 | -0.018931928 | -0.016672037 | 554.876969 | +9.83% | 2.406080 |
| `outer_leftdiag` | 4.244207478 | -0.006838243 | -0.005018445 | -0.006507168 | 543.400540 | +7.56% | 2.336730 |
| `outer_leftdiag_fused` | 4.241099412 | -0.009946309 | -0.008303572 | -0.006765770 | 539.934095 | +6.87% | 2.329600 |

Late means retain the preceding identical five-token evaluation coverage.
Every family selects this upper point among its five completed LRs,
each consuming 838,860,800 cumulative screening tokens. The optimum is
still unresolved at the boundary. Short cooldown, seed-0 selection reuse
and higher time/memory cost preclude an independent optimizer win.
Another LR expansion needs fresh actual-config gates; independently
selected full-budget schedules and fresh paired seeds remain required.

The original monitor imports/verifies all eight full-resolution
histories and exits normally at 07:42:10. `swanlab-upload.json` is
`verified`. Each has 320 training/token points, nine C4/selection
evaluations and all original scalar columns; local read-only
provenance/count checks pass, including exact integers,
configuration/token axes, original manifests/
histories/configs/eight-rank checkpoints and recorded two-ULP export
bounds. The first-round IDs remain above; second-round IDs are in this
receipt. No duplicate uploader or replacement study monitor is launched.

##### Failed Synthetic Diagnostic and Withheld Handoff, November 17

The original synthetic queue records `running_probe` at 07:49:02 after
rederiving the completed upper-LR study, cloud receipt and restoration.
Controller 295416/generation1178012035 owns marked child295606. At
07:50:50 the controller records `needs_attention`, exit1/cleanup1,
keeper_restore0. `diagnose_upstream_outer_gpu.py:275` compares the FP32
actual delta with the independent FP64 equation through
`diagnose_fused_diagonal.py:41`; `torch.isclose` raises `RuntimeError:
Float did not match Double`. The incomplete `result.json` contains zero
completed cases and zero timings. This dtype failure gives no optimizer
parity verdict. Preserve the original diagnostic binding and result. Do
not retry its persisted intent or mutate its frozen helpers. Here
`cleanup_exit_code` is the joined diagnostic child's exit code after
marked-child cleanup, rather than a separate cleanup operation's return
code. No `cleanup_error` is recorded; fresh inspection confirms no
children.

The separate full-model handoff queue freezes at 07:45:35 in
`PVC/run-preparation/upstream-full-model-queue/`, with 45 helpers: the
validator's 43 unchanged helpers plus its Python/Bash queue.
Installation is acknowledged on `.139` at 07:51:14, PID296334/
generation1178025263. Its state is `waiting_for_probe` at07:51:24. The
original probe queue consumes the failure and exits at08:04:08; the
full-model queue records `needs_attention` and exits at08:06:35 without
a controller intent. No full-model controller intent exists. CPU
queue/ledger tests pass; independent queue safety review remains
unperformed.

The first installation check refuses before intent because 12 original
CPU-receipt helper paths are absent on `.139`. Only those absent files
are materialized from checksum-verified CPU archives at 07:50:26;
existing files, receipts and releases remain unchanged. Evidence:
`PVC/run-preparation/upstream-cpu-helper-materialization-20261117T0751.json`.
The frozen binding check then passes and installation succeeds.

Correction requires a distinct synthetic diagnostic with FP64 comparison
operands, the independent FP64 equation and original rtol1e-5/atol1e-6.
The current full-model binding hard-binds the failed original probe;
successful corrected evidence must feed a separate validator identity.
Actual capture, distributed statistics, scalar transitions, all-rank
24-versus22+resume24 gates and production denial remain required.

### Wave 9: Upstream-Product Systems Comparison

Question: do pinned upstream right-product numerics remove enough
output-diagonal implementation overhead to improve whole-training
efficiency, while retaining the exploratory fixed-token loss benefit?
The matched null separates execution-path overhead/numerical differences
from the additional output-side geometry.

| Arm | Allocation | Purpose |
| --- | --- | --- |
| `muont` | `28.216.17.9`, eight A100s | Same-source input baseline |
| `outer_leftdiag_fused` | `28.216.17.80`, eight A100s | Legacy direct-product fused output diagonal |
| `outer_leftdiag_upstream_fused` | `28.216.17.144`, eight A100s | Pinned upstream right products with fused output diagonal |
| `null_leftdiag_upstream_fused` | `28.216.17.139`, eight A100s | Matched null with pinned input/MuonT update |

The executed four-arm wave retains the corrected validator's exact seed-0
configuration, matrix LR 0.02262741696, 524,288 tokens per update,
5,120 updates and 2,684,354,560 tokens. The matched null disables output
factor application while retaining its collector and statistics; it uses
the same pinned input/NS/outer products as the upstream output arm.
Data, schedule, FA2/compiler, fixed installation age and eight-rank execution are
unchanged. No continuation/retry or shortened production schedule is
planned. Compare five complete 536,870,912-token training windows,
eleven exact-shared late evaluations, final C4/held-out selection, main
and recorded total time, and peak allocated/reserved memory. Report the
upstream-minus-legacy and candidate-minus-null pairs alongside each
arm versus MuonT. Earlier synthetic speedups do not substitute for these
measurements. Seed-0 reuse and no independent full-budget LR selection
make this a systems/exploratory quality wave, not tuned superiority.

Preparation completes at 09:00:10 in the distinct
`PVC/candidates/upstream-full-model-production-20261117` candidate.
Its 289-file inventory differs from the frozen corrected validator
only at `experiment/execution_owner.py`, new
`experiment/production_owner.py`, `experiment/cluster.json` and
`experiment/tests/test_upstream_release.py`. Training, validation,
capture, optimizer, pinned sources and every configuration are unchanged.
The original validation owner and tests remain present.

CPU verification completes at 09:01:02: all 118 candidate tests pass
with `CUDA_VISIBLE_DEVICES=""`. The exact command is:

```bash
/opt/conda/envs/torch-base/bin/python -m pytest -c /dev/null \
  -p no:cacheprovider -q \
  -m 'not slow and not integration and not data_integration and not cluster and not requires_cluster and not docker and not manual' \
  experiment/tests
```

Its working directory is the production candidate. Preserved evidence:

- Preparation: candidate `production-preparation.json`.
- CPU receipt:
  `PVC/run-preparation/upstream-production-cpu/20261117T0900/receipt.json`.

The production controller must call the exact frozen original queue's
`completed()` reconstruction on `.139`, after its `validated` state,
controller/queue exit and healthy restored eight-GPU keeper are verified.
Only then may it freeze a separate Wave-9 source/helper release and
preflight all four allocations. Canonical release/output paths are
`PVC/run-preparation/wave9-launch-binding.json` and
`PVC/runs/wave9/ARM`. Persisted per-node launch intents prohibit retries;
unverified child cleanup prohibits keeper restoration.

The 900-second Bash monitor snapshots original worker generations,
compares large windows, and uploads completed full-resolution histories
to private `output-side-normalization-marin`, verifying every scalar series.
It starts with acknowledged launch; the document monitor continues throughout.

At 09:10:11, bound SSH records all 13 zero-exit commands and 24 true
flags. The final child receipt time is 09:08:44. At 09:14:35 the
controller/queue remain live, child cleanup is zero, and restoration is
not yet recorded. All four saved update-22/update-24 comparisons are
exact: MuonT2185/2219, legacy fused2221/2253, upstream fused2237/2243
and null2175/2241 tensor comparisons, respectively. Final reconstruction
still precedes release. Tracking613+20subtests, focused71, uploader12,
source/document lint and Bash syntax pass.

At09:18:57, the original controller finalizes `passed` with execution0,
cleanup0 and keeper_restore0. Fresh09:19:35 SSH verifies the controller
exited and original queue live. Its independent acceptance starts on the
unchanged 900-second cadence; no manual queue step or restart occurs.

Launch-control review identifies and fixes three controller-only risks:
mutable source drift during long reconstruction, partial keeper occupancy,
and startup intents stuck indefinitely after verified process absence.
The controller now rechecks CPU/candidate/helper hashes before committing
staged freeze artifacts; every node requires eight exact keeper workers
on eight distinct GPU UUIDs under its node lock, including an immediate
pre-release generation recheck; spawn failures retain their original intent
and marker/generation-bound absence is terminal without retry. No prepared
candidate, owner/test hash, numerical source or tolerance changes.
Final controller11/full tracking620+20subtests and scoped lint pass.
An earlier invocation fails on an incomplete fixture; its repaired rerun
passes. Read-only occupancy checks pass on `.9` at09:21 and
`.80`/`.144`/`.139` at09:23:17 with eight distinct GPU UUIDs. No process
is stopped. The original production CPU receipt/log recheck passes09:24.

Startup remains unknown until the original starter is quiescent and its
intent/state/marker descendants are rechecked. Interrupted freeze resumes
only identical saved publication; it creates no training intent.
Focused112/tracking625+20subtests, lint/Bash syntax pass09:32;
document monitor3776115 refreshes09:32:22 without restarting.

At 09:39:41, original-binding SSH verifies queue300791, step325828 and
reconstruction child326022 live, controller301264 exited, 13/13 commands
and24/24 flags, execution/cleanup/restoration0. No release exists. Receipt:
`PVC/run-preparation/upstream-full-model-observation-20261117T0940.json`.
The current ledger/production/owner suite passes 83 CPU tests.

The original queue independently reconstructs checkpoint/history
evidence, records `validated`09:44:40 and exits09:44:41. Fresh SSH finds
all three processes absent; original source/logs,24 gates,exact tensors
and restored eight-GPU keeper remain bound. At09:45 production freeze
starts with the original CPU receipt and rederives `.139` queue
`completed()` evidence. It creates no training intent.

Wave-9 generic ledger progress requires the release-artifact hashes and
snapshot `binding_sha256`; analysis requires `source_release_sha256`.
Helper drift or missing release withholds both; a mismatched snapshot
leaves separately bound analysis intact. Four document regressions pass.
Focused87/tracking600+20subtests pass with CUDA hidden and explicit
imports; isolated uploader12 and scoped lint pass09:47:35.
No frozen candidate, helper, numerical source or queue changes.

Wave-9 freeze completes at 09:57:27 after the original frozen queue's
`completed()` reconstruction passes again. The separate release is
`PVC/run-preparation/wave9-launch-binding.json`. It binds 289 candidate
files and 52 helpers: the unchanged 43 original helpers plus nine
production/analysis/upload helpers. Source archive:
`PVC/run-preparation/source-wave9-launch-20261117.tar`. The exact frozen
controller's `--phase check` exits zero at 09:57:58; receipt
`PVC/run-preparation/wave9-frozen-check-20261117T0958.json`. At 09:58,
that controller starts `--phase launch`. It first reconstructs the
original acceptance, then inspects all four allocations before
persisting any worker intent. There is no worker acknowledgment at this
observation; training and results remain pending.

The separately preserved document/ledger snapshot is
`PVC/run-preparation/source-endeavor-wave9-ledger-20261117T0952.tar`. It
contains the preceding document and the source-bound ledger regressions;
it is separate from every numerical release.

The first launch attempt passes original acceptance reconstruction,
then fails the `.80` inspection around 10:09 before creating
`wave9-launch-state.json`, a worker intent, run directory or monitor.
The preserved predecessor CPU receipt refers to helper copies under
the node-local checkout; twelve older exact copies had been
materialized only on `.139`. `.80` lacks
`tracking/prepare_upstream_full_model.py`. The failure is provenance
availability, not a failed numerical gate or training result.

At 10:10-10:11, all twelve absent helper copies are materialized on each
of `.80` and `.144` from their original CPU archives, after verifying
every expected checksum and refusing any existing mismatch. No file is
overwritten; no candidate, receipt, source archive, kernel or GPU
process changes. The exact frozen controller's read-only allocation
inspections then pass on both nodes. Repair receipt:
`PVC/run-preparation/wave9-cpu-provenance-materialization-20261117T1012.json`.
The pre-intent failure and verified absence of every launch artifact are
preserved in
`PVC/run-preparation/wave9-preintent-preflight-failure-20261117T1012.json`
at 10:11:46. Repeating preflight can use the unchanged release;
persisted worker intents still prohibit any training retry. At10:14 the
unchanged launch entry point repeats preflight; original acceptance
reconstruction remains active at10:21 with no worker intent.

#### First Ceiling-LR Results, November 17 14:29

All four first trials at matrix LR0.09050966784 complete320 updates/
167,772,160 tokens with successful exact trial commands and bounded
child cleanup. Their family workers retain their allocations for the
second LR0.12799999978521195 trial. No family restoration or complete
seven-point LR choice is claimed at this entry.

The original frozen analyzer produces a separate preserved analysis
at14:29:31. It covers five complete33,554,432-token windows and the same
five late evaluation-token points84,410,368/105,381,888/126,353,408/
147,324,928/167,772,160. Deltas below compare the same LR and seed with
MuonT; negative loss deltas are better.

| Arm | Final Selection | Final Selection Delta | Late Selection Delta | Late C4 Delta | Final Total Seconds | Extra Total Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `muont` | 4.253260209 | 0 | 0 | 0 | 496.783 | 0% |
| `inner_outer_leftdiag` | 4.231412757 | -0.021847452 | -0.026857115 | -0.023871684 | 577.202 | +16.19% |
| `outer_leftdiag` | 4.236746536 | -0.016513672 | -0.019512393 | -0.016708322 | 564.717 | +13.67% |
| `outer_leftdiag_fused` | 4.243626408 | -0.009633801 | -0.010590405 | -0.008023013 | 568.101 | +14.36% |

| Candidate | Window1 Delta | Window2 Delta | Window3 Delta | Window4 Delta | Window5 Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inner_outer_leftdiag` | -0.028833911 | -0.025607839 | -0.026253007 | -0.024708562 | -0.021714181 |
| `outer_leftdiag` | -0.009142861 | -0.025399774 | -0.020349383 | -0.017234899 | -0.014242359 |
| `outer_leftdiag_fused` | -0.019875735 | -0.015398890 | -0.008363225 | -0.007601485 | -0.008018270 |

The broad windows and shared late evaluations retain fixed-token gains.
All candidates cost more total time than matched-LR MuonT, and their
peak allocation is22.359/22.359/22.390GiB versus20.531GiB.
Fused is3.384 seconds slower than outer at this short budget and has
worse selection loss. These short schedules include startup/checkpoints;
they do not establish whole-training efficiency or full-budget ranking.
Independent full-budget LR selection and fresh paired-seed confirmation
remain required.

The original campaign monitor28910 captures fresh workers14:28:07,
analyzes14:28:18 and full-resolution verifies three completed histories
at14:28:54. Each verified history retains320 token/training points and
nine C4/selection evaluations, plus all original scalar columns and
checkpoint/history provenance. Fused completes after the monitor's
selection of importable trials and remains pending for the next cycle.
No duplicate uploader or monitor is installed. The original queue
records `launched` and exits normally14:26:29; fresh process inspection
finds it absent while the campaign and document monitors remain live.

At14:31:46 the higher-LR histories retain319/152/92/42 updates for
MuonT/inner-outer/outer/fused. All observed training losses are finite;
incomplete trajectories receive no final-loss or cumulative-LR choice.
Fresh acceptance of the formatted owned-output CPU316 and separate
controllerCPU40 receipts rechecks296 candidate files,29 helper files,
their archives and the original ceiling binding. The owned-output
14-command GPU validation is still unlaunched and waits for all eight
LR endpoints, restored/absent workers, full five-window analysis and
verified cloud histories.

#### Final Ceiling-LR Results, November 17 14:39

All eight exact-config trials finish320 updates/167,772,160 tokens.
MuonT, inner-outer, outer and fused workers record successful
keeper restoration at14:32:18/14:35:59/14:37:14/14:38:48 respectively.
The original frozen helper's fresh SSH snapshot14:39:37 reports
`complete`, four terminal workers and all eight consecutive320-point
histories. Each trial retains its original command, integer-zero
training/cleanup exits, final eight-rank checkpoint and source identity.
The initial14:37:24 snapshot and14:38:14 incomplete analysis remain
preserved; fused's unfinished endpoint was not treated as a final loss.

The original frozen analyzer produces a separate complete result at
14:39:12. Both LR groups have five complete33,554,432-token training
windows and exact late evaluation tokens84,410,368/105,381,888/
126,353,408/147,324,928/167,772,160. The lower-LR table above remains
unchanged. The higher-LR0.12799999978521195 comparison is:

| Arm | Final Selection | Same-LR Selection Delta | Late Selection Delta | Late C4 Delta | Final Total Seconds | Extra Total Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `muont` | 4.302122859 | 0 | 0 | 0 | 500.146 | 0% |
| `inner_outer_leftdiag` | 4.269263712 | -0.032859147 | -0.039263516 | -0.035554433 | 557.453 | +11.46% |
| `outer_leftdiag` | 4.279768733 | -0.022354126 | -0.027851216 | -0.022444850 | 543.442 | +8.66% |
| `outer_leftdiag_fused` | 4.287789485 | -0.014333374 | -0.012568874 | -0.009648385 | 541.126 | +8.19% |

| Candidate | Window1 Delta | Window2 Delta | Window3 Delta | Window4 Delta | Window5 Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inner_outer_leftdiag` | -0.038863078 | -0.038758144 | -0.045986459 | -0.041163541 | -0.031813398 |
| `outer_leftdiag` | -0.042845234 | -0.025849752 | -0.032428347 | -0.028486729 | -0.020750172 |
| `outer_leftdiag_fused` | +0.000626571 | -0.011922143 | -0.016894467 | -0.016724609 | -0.013995208 |

All candidates improve the final and five-point late losses against
same-LR MuonT, but fused's first broad training window is worse.
Every family's0.128 endpoint is worse than its0.09051 endpoint.
The larger same-LR gaps at0.128 are not evidence to select that LR.
Peak allocation remains20.531GiB for MuonT and22.359/22.359/22.390GiB
for the candidates. Fused saves2.316 seconds versus outer at this LR
but worsens final selection by0.008021; all three cost more recorded
total time than MuonT.

The cumulative analyzer ranks all seven completed equal-budget points
per family using final reserved-mixture selection loss only:

| Family | Selected Matrix LR | Selected Final Selection | Delta vs Independently Selected MuonT | Upper Boundary Selected |
| --- | ---: | ---: | ---: | --- |
| `muont` | 0.06399999989260598 | 4.251045721 | 0 | no |
| `inner_outer_leftdiag` | 0.09050966784 | 4.231412757 | -0.019632964 | no |
| `outer_leftdiag` | 0.09050966784 | 4.236746536 | -0.014299184 | no |
| `outer_leftdiag_fused` | 0.06399999989260598 | 4.241099412 | -0.009946309 | no |

Comparisons of these independently selected short-budget endpoints use
the same five full token windows and exact late evaluation tokens.
Their reserved-mixture late deltas and recorded runtime are:

| Candidate | Late Selection Delta | Late C4 Delta | Selected Total Seconds | Extra Total Time vs Selected MuonT |
| --- | ---: | ---: | ---: | ---: |
| `inner_outer_leftdiag` | -0.017198898 | -0.010291590 | 577.202 | +14.25% |
| `outer_leftdiag` | -0.009854176 | -0.003128228 | 564.717 | +11.78% |
| `outer_leftdiag_fused` | -0.008303572 | -0.006765770 | 539.934 | +6.87% |

Selected MuonT records505.214 seconds. Unlike the same-LR comparisons,
the independently selected inner-outer/outer LRs have worse first two
training windows, then better last three; fused is worse in window1.

| Candidate | Window1 Delta | Window2 Delta | Window3 Delta | Window4 Delta | Window5 Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inner_outer_leftdiag` | +0.028214671 | +0.003487252 | -0.009813152 | -0.013968080 | -0.015054032 |
| `outer_leftdiag` | +0.047905721 | +0.003695317 | -0.003909528 | -0.006494418 | -0.007582210 |
| `outer_leftdiag_fused` | +0.009129919 | -0.005973093 | -0.006816566 | -0.006504051 | -0.006478213 |

The original frozen comparison helper preserves this matched-seed,
five-window selected-endpoint result at
`PVC/run-preparation/lr-ceiling-seven-point-selected-comparison-20261117T1454.json`.
This cross-campaign runtime includes startup/checkpoints and is
exploratory; the lower final loss does not establish elapsed-time
superiority or independently validated full-budget performance.

This resolves the upper-edge selection issue for the observed short
grid, not the full-budget optimum. Each family has1,174,405,120 observed
tuning tokens across seven endpoints and retains two permitted
refinements; none is silently consumed or launched. C4 remains reporting
only. These reused seed-0,320-update schedules do not establish a
quality/efficiency win over independently tuned full-budget MuonT.
The next quality decision requires separately validated full-budget LR
settings and frozen fresh paired-seed comparisons, not another
unconditional upper expansion.

At this milestone the cloud receipt is still `partial_verified` from
14:28:54, covering only the three earlier lower-LR histories. The
original campaign monitor28910/start ticks923007652 holds its lock and
retains the900-second cycle; no second uploader is started. Its next
scheduled cycle is approximately14:43:54. Keep final analysis distinct
from all-eight cloud verification.

Document monitor3776115/start ticks919963640 successfully refreshes at
14:38:14 after its earlier safe concurrent-edit refusal. Its original
lock, command and900-second cadence remain verified. The focused
ledger suite passes77 tests with the complete safe marker exclusions.

#### Wave 10 Final Five-Window Result, November 17 19:31-19:35

Wave10 completes and retains small fixed-token loss gains at higher
recorded time and peak allocated memory than MuonT. The rank-owned
outer factors produce the same recorded losses as Wave9; the rerun
establishes no whole-training speed gain. Independent LR selection and
fresh paired-seed confirmation remain required.

All four original workers complete 5,120 updates, 2,684,354,560 tokens
and the bound final eight-rank checkpoint. Terminal receipts record
integer-zero training, cleanup and keeper-restoration exits. Original
release-bound SSH at 19:33:01 reports status `complete` and
`all_workers_terminal=true`, with all four histories complete:

| Arm / Node | Terminal Receipt Time | Restored Keeper Parent PID |
| --- | --- | ---: |
| `muont` / `.9` | 19:19:33 | 210585 |
| `outer_leftdiag_fused` / `.80` | 19:29:03 | 325833 |
| `outer_leftdiag_upstream_fused` / `.144` | 19:29:35 | 310662 |
| `null_leftdiag_upstream_fused` / `.139` | 19:31:12 | 379530 |

The frozen analyzer succeeds at 19:32:53. It verifies original
source/configuration/data identity, uninterrupted consecutive histories,
all eight final checkpoint files per arm and 22 exact C4/selection
evaluation positions. No history is trimmed, filled or repaired. The
executed candidate is
`PVC/candidates/owned-output-production-isolated-owner-formatted-20261117`.
Each arm retains base matrix LR 0.02262741696, seed 0, 524,288 tokens
per update, fixed refresh-install age 1 and the original
1,024-stable/4,096-decay schedule. Each node runs eight A100-SXM4-40GB
GPUs. Node suffixes in the table refer to `28.216.17.9`, `28.216.17.80`,
`28.216.17.144` and `28.216.17.139`. Refresh-install age is counted in
optimizer updates from the statistics snapshot to installation of its
refreshed factor. Age 1 installs at the prescribed next-update boundary
independently of asynchronous completion timing. The three
candidate-minus-MuonT train-loss deltas use equal complete
536,870,912-token windows:

| Arm | Window 1 | Window 2 | Window 3 | Window 4 | Window 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `outer_leftdiag_fused` | -0.001553388 | -0.004791398 | -0.003128387 | -0.001819739 | -0.001655799 |
| `outer_leftdiag_upstream_fused` | -0.001699864 | -0.003395067 | -0.002863237 | -0.001847870 | -0.001378032 |
| `null_leftdiag_upstream_fused` | -0.000685884 | -0.000891889 | -0.001710066 | -0.000950178 | -0.001138804 |

Window endpoints are 536,870,912; 1,073,741,824; 1,610,612,736;
2,147,483,648; and 2,684,354,560 tokens. Training means are token
weighted. Late evaluation deltas are arithmetic means over the same
11 exact positions strictly above 1,342,177,280 tokens:
1,442,316,288; 1,573,388,288; 1,704,460,288; 1,835,532,288;
1,966,604,288; 2,097,676,288; 2,228,748,288; 2,359,820,288;
2,490,892,288; 2,621,964,288; and 2,684,354,560.
C4 is separate Paloma `c4_en` loss. Selection is the reserved
DCLM/StarCoder/Proof-Pile normalized held-out mean:
`(3.8 * DCLM + 0.25 * StarCoder + 0.055 * Proof-Pile) / 4.105`.

| Arm | Final C4 | Final Selection | Late C4 Delta | Late Selection Delta | Recorded Main Seconds | Steady Seconds / MTok | Peak Allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `muont` | 3.451658143 | 3.234892453 | 0.000000000 | 0.000000000 | 5973.203 | 2.121130 | 20.530577 |
| `outer_leftdiag_fused` | 3.449858208 | 3.231792290 | -0.001768693 | -0.003251625 | 6474.957 | 2.300039 | 22.389584 |
| `outer_leftdiag_upstream_fused` | 3.448915529 | 3.233105483 | -0.002597064 | -0.002453269 | 6429.315 | 2.289435 | 22.389585 |
| `null_leftdiag_upstream_fused` | 3.449843709 | 3.233488820 | -0.001222194 | -0.002009508 | 6452.678 | 2.294055 | 22.389585 |

Final C4 deltas are -0.001799935/-0.002742614/-0.001814434 nats;
final selection deltas are -0.003100164/-0.001786970/-0.001403634.
Recorded main-function time includes setup, training, evaluation and
checkpointing, but excludes Python imports and parent launch overhead.
Steady training-step time excludes the first 52,428,800 tokens and
separate setup/evaluation/checkpoint/shutdown work. Legacy/upstream/null
cost 8.40%/7.64%/8.03% extra main time and 8.43%/7.93%/8.15% extra
steady step time versus MuonT. Peak allocated memory is about 1.859 GiB
higher for all three. No lower-memory or whole-training efficiency
claim is supported.

Upstream saves 45.642 seconds, 0.705%, versus legacy, improves final
C4 by 0.000942679, but worsens final selection by 0.001313194 nats.
Against the matched null it improves final C4/selection by
0.000928180/0.000383336 and late C4/selection by
0.001374870/0.000443761. Its fifth-window train gain over null is
only 0.000239229 nats. The null retains output collection, pooled
moments, refresh/checkpoint state and upstream arithmetic, with
strength zero and identity outer factor. Its own baseline gains prevent
attributing the entire baseline gap to output geometry.

Retrospective C4 targets 3.59/3.52/3.46 establish no measured elapsed-time
win for upstream over MuonT. MuonT crosses at recorded times
4690.203/5547.118/5830.801 seconds;
upstream crosses at 5052.605/5665.638/6280.281. Legacy/upstream reach
the 3.52 evaluation one token interval earlier than MuonT, but still
take longer. Crossings are exact observed evaluations without
interpolation; their preceding evaluation bounds remain in the analysis.
Targets were exploratory and the timer excludes parent launch overhead,
so these are not confirmatory launch-based time-to-target measurements.

The frozen same-config Wave9 comparison records zero loss delta in
every window and all 22 C4/selection evaluations. An independent
row-by-row check finds zero differences in `train_loss`, `c4_loss`,
`selection_loss` and `heldout_losses` across all 5,120 positions
for every same-arm pair:

| Arm | Wave10 Minus Wave9 Main Seconds | Main-Time Change | Steady-Step-Time Change |
| --- | ---: | ---: | ---: |
| `muont` | -23.386 | -0.390% | +0.070% |
| `outer_leftdiag_fused` | +15.864 | +0.246% | -0.100% |
| `outer_leftdiag_upstream_fused` | +31.754 | +0.496% | +0.421% |
| `null_leftdiag_upstream_fused` | -9.969 | -0.154% | -0.258% |

These mixed, sub-percent timings are a single observational rerun on
different clock intervals. Exact repeated losses verify the measured
trajectory after the ownership migration; they are not an independent
seed replication or proof of every unlogged tensor's equality.
Avoided root work needs a stage profile before an efficiency claim.
The next quality wave remains the separately selected-LR full-budget
transfer. The stage profiler and fresh paired seeds remain separate
unexecuted experiments.

Both JSON artifacts are preserved together in
`PVC/run-preparation/wave10-final-analysis-observation-20261117T1935.tar`.
The prior document is archived at
`PVC/run-preparation/endeavor-before-wave10-final-result-20261117T1935.tar`.
The frozen analysis command is:

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/torch-base/bin/python \
  "$PVC/controllers/wave10-launch-20261117/tracking/analyze_owned_output_run.py" \
  --root "$PVC"
```

At 19:35, original monitors 3776115/132158 remain live on their
900-second cadence. The manual complete snapshot does not replace the
run monitor's canonical `latest.json`. Wave10 SwanLab verification
remains pending that monitor's next complete-worker cycle. The mutable
ledger successfully renders the saved
`wave11-selected-lr-validation-binding.json` with source-bound planned
command/check counts and no execution claim; the affected
149-test suite passes and scoped repository Ruff/Black/AST/license
checks pass. No optimizer, configuration, issued source or GPU
allocation changes during final analysis.

At 19:37-19:38, separate frozen read-only handoff checks pass on all
four original nodes. They verify original acknowledged worker and
training handles absent, no marked training descendants, and exact
healthy restored keeper generations. Inherited worker markers occur
only on those recorded keeper generations. These checks do not release
an allocation or replace cloud verification.
The document monitor's next scheduled refresh succeeds 19:34:51 on
the unchanged 900-second cadence. The ledger review's numerical-failure
and malformed-receipt gaps are corrected in mutable reporting code.
The broader affected suite passes 153 tests in 14.39 seconds, including
saved GPU failures after zero-exit commands and null receipt structures.
Scoped Ruff/Black/AST/license and document whitespace/EOF/conflict/
invocation checks pass. The existing document file-size rule remains
exceeded; no experimental evidence is removed.

#### Wave 12 Final Five-Window Result, November 18 01:54-01:59

The distinct frozen-controller SSH snapshot at 01:54:48.536615 verifies
all four original workers terminal and complete. Each retains 5,120
updates and 2,684,354,560 tokens, its final eight-rank checkpoint identity,
and integer-zero training/cleanup/keeper-restoration exits.
Inner-outer's terminal state is recorded at 01:53:43.649953; its restored
keeper is `338333/1620947177` on `28.216.17.80`. MuonT/outer/fused retain
the exact previously verified restored generations on their original nodes.
This recovery of Wave 11's zero-update ownership failure is not an
independent scientific replication.

Independent analysis runs the issued Wave-12 analyzer with its own frozen
imports against the completed histories, saving a distinct final observation.
All comparisons are paired seed-0 updates with identical data identity
and batch tokens. Five complete windows of 1,024 updates/536,870,912 tokens
cover the entire budget. No partial window or interpolated evaluation is used.

| Arm | Window 1 Train Delta | Window 2 | Window 3 | Window 4 | Window 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `muont` | 0 | 0 | 0 | 0 | 0 |
| `inner_outer_leftdiag` | +0.051593185868 | +0.057541904505 | +0.052360697184 | +0.050557769137 | +0.046857859939 |
| `outer_leftdiag` | +0.061018768465 | +0.069954498904 | +0.063745131483 | +0.056893423665 | +0.042437023483 |
| `outer_leftdiag_fused` | -0.005872853100 | -0.006467067171 | -0.003875716589 | -0.003242881037 | -0.002814263105 |

The corresponding MuonT window means are
4.402290010592/3.787353420397/3.639884402277/3.505965305259/3.334844573168.
Deltas are candidate minus same-wave MuonT, in nats.

All 22 evaluations share exact token positions. The final-half mean uses
the 11 evaluations at updates 2,751/3,001/3,251/3,501/3,751/4,001/4,251/
4,501/4,751/5,001/5,120, after 1,342,177,280 tokens.

| Arm | Final C4 | Final Selection | Final C4 Delta | Final Selection Delta | 11 Late C4 Delta | 11 Late Selection Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `muont` | 3.491818353478 | 3.277536405521 | 0 | 0 | 0 | 0 |
| `inner_outer_leftdiag` | 3.533942883087 | 3.319952527934 | +0.042124529609 | +0.042416122412 | +0.049206183053 | +0.049293741403 |
| `outer_leftdiag` | 3.519598842994 | 3.308750795602 | +0.027780489516 | +0.031214390080 | +0.048761128717 | +0.050255834608 |
| `outer_leftdiag_fused` | 3.489356775095 | 3.275187087218 | -0.002461578383 | -0.002349318304 | -0.001805780194 | -0.002908530792 |

| Arm | Recorded Main Seconds | Extra Time vs MuonT | Steady Step Seconds/MTok | Peak Allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| `muont` | 5953.993978 | 0% | 2.111585168109 | 20.530576705933 |
| `inner_outer_leftdiag` | 6833.002884 | +14.763349% | 2.420791945459 | 22.359491825104 |
| `outer_leftdiag` | 6541.880753 | +9.873822% | 2.322088509387 | 22.359491825104 |
| `outer_leftdiag_fused` | 6490.861581 | +9.016932% | 2.303545064997 | 22.389584064484 |

Recorded main-function time includes setup, evaluation and checkpointing,
but excludes Python import and parent launch overhead. Steady step time
excludes the first 100 updates; it is not launch-based time-to-target.
Fused reaches retrospective C4 thresholds 3.59/3.52 at the same evaluations
as MuonT, but later in recorded time: 6035.672202/6342.525118 seconds versus
5534.253878/5817.317861. Neither reaches 3.46. No measured elapsed-time
win is present. High-LR inner-outer/outer remain worse throughout all
five training windows and final/late selection comparisons.

Actual matrix LRs are `0.06399999989260598` for MuonT/fused and
`0.09050966784` for inner-outer/outer. These transferred short-study
choices are not full-budget LR optima. Fused retains a small descriptive
fixed-token gain, but slower steps and extra memory; seed 0 remains
exploratory. A same-source/same-LR matched-null comparison is needed to
attribute the gain, followed by equal full-schedule LR neighborhoods and
fresh paired seeds before a superiority claim.

The original 900-second wave monitor still owns canonical final analysis,
full-resolution private SwanLab upload and independent cloud verification.
No competing uploader is started. At this milestone its canonical analysis/
cloud acceptance remains pending; the distinct final observation is not a
cloud receipt. The original document monitor remains live and is not
duplicated. All earlier immutable interim captures and read refusals remain
preserved.

### Post-Wave-12 Experiment Decisions

The bounded profiling attempt is terminal `needs_attention` at
November 18 03:01:45. Three MuonT commands exit zero; all eight saved
traces expose the CPU/GPU reader-schema mismatch described above.
The original controller/validator are absent and exact keeper restoration
is independently verified at 03:08:21. The separate CPU/archive-accepted
reader now reconstructs all eight saved MuonT traces at 03:23:38.
Checkpoint-tensor/trial reconstruction, remaining-arm execution and
a separate profiling release remain required.
The matched-null/full-schedule LR neighborhood is implemented and
CPU/archive-accepted at 03:00:45. Its distinct actual-config validation
candidate/controller receives CPU/archive acceptance at 04:22:05/04:38:44,
freezes at 04:41:56 and launches owned validation at 04:48.
The attempt is subsequently terminal `needs_attention` at 05:24:39:
six of 27 commands exit zero and lower-LR MuonT passes all eight checks,
but fused checkpoint-24 continuation rejects the rank-0 embedding tensor
under unchanged tolerances. Original-worker absence and exact keeper
restoration are independently verified at 05:27:51. Saved-state CPU
diagnosis and a distinct fully validated correction remain required;
this attempt cannot supply numerical acceptance or production ownership.
The nine production runs and fresh-seed confirmation remain unlaunched.
Wave 12's four equal-budget runs, original-worker cleanup/checkpoints,
five shared 536,870,912-token windows and 22 exact evaluations now pass
independent final reconstruction. Original-monitor canonical analysis
and full-resolution SwanLab verification now pass at 02:04:24;
the independent exact-allocation handoff also passes before GPU transfer.
Use reserved-mixture selection loss to rank the transferred LR choices;
report C4 separately. Check the last two training windows and all 11
shared final-half evaluations, as well as full-budget endpoints.
Compare against same-wave MuonT and label any Wave-7 base-LR comparison
as a cross-wave observation. A transfer result does not establish
full-budget tuning or fresh-seed superiority.

The systems follow-up retains the CPU-verified Wave-9 stage profiler
described in "Actual-Training Stage Profiler Prepared" as an instrumentation
template. A distinct CPU-only preparation is implemented against
Wave 12's frozen selected-LR source, because the original profiler also wraps
combined/upstream functions absent from these four arms. Preserve Wave 12's
actual configurations and numerical helpers, add only bounded instrumentation
and an early execution-denial owner, and capture updates 201-220 in fresh
successor-owned runs. First CPU/archive acceptance completes at 00:31:08;
the distinct full-closure successor receives acceptance at 00:51:22.
The failed bounded attempt establishes actual eight-rank execution and
independently verified terminal restoration. Numerical/trace acceptance
for the full protocol is absent. The distinct CPU reader now reconstructs
saved MuonT traces. Complete checkpoint-tensor/trial reconstruction and
the full 200/220 resume matrix under successor ownership.
Profile output collection, communication, root work, packing and update
arithmetic before choosing another efficiency implementation.
Wave 10's exact repeated recorded losses and mixed sub-percent runtime
changes show that removing unused output roots alone has not delivered
a measured whole-training saving.

The prepared fused/null/MuonT grid tests attribution and full-budget LR
selection before the paired-seed campaign. Complete its actual-config
GPU numerical/continuation gates and separate execution release before
launch. Preserve the current source, settings and full budgets.
Use the profiler's accepted stage evidence, when available, to choose
the next efficiency implementation. A small seed-0 fixed-token gain with
slower execution does not meet the optimizer goal.

### Cancellation-Free Start Accuracy Result, November 18

The accepted rational quarter-root start improves the numerical accuracy
of `O` on all saved matrices under the unchanged `rtol=1e-5`, `atol=1e-6`
FP64-reference gate. Its derived `Ob` and `dn0` change with `O`; the other
seven outputs remain exactly equal. This diagnostic has no training-loss
endpoint or measured optimizer-efficiency result.

Two fresh opposite-order CUDA processes use the preserved rank-0/update-23
continuous/resumed matrices, with full shapes `d=(12,512,512)` and
`f=(4,2048,2048)`. Both compiled variants execute eagerly and under explicit
CUDA-graph capture, first replay and second replay. The table reports each
process separately; four phases over the same matrices are repeated
numerical checks, not independent training replications.

| Process | Group | Matrix-by-Phase Checks per Variant | Pinned Failures | Rational Failures |
| --- | --- | ---: | ---: | ---: |
| 0 | `d` | 96 | 96 | 0 |
| 0 | `f` | 32 | 32 | 0 |
| 1 | `d` | 96 | 96 | 0 |
| 1 | `f` | 32 | 32 | 0 |

Worst absolute `O` errors fall from `1.6792206647e-4` to
`7.9939396665e-9` for `d`, and from `3.5843348586e-4` to
`1.1509636182e-8` for `f`. All 32 cross-process outputs are exact.
Each process also passes 24 repeat comparisons, 16 seven-output
preservation checks, 64 FP16-cast checks and 16 continuous/resumed
comparisons exactly. The independent reference uses FP64 log1p/expm1
arithmetic, with the separate Decimal-based CPU reference retained.

Source/archive acceptance has 121 passing CPU tests and preserves all
100 inherited numerical/execution helpers. The comparison completes at
20:02:09 and the controller records completion/restoration at 20:02:25.
Independent retained-file and terminal reconstruction completes at
20:06:05. Canonical comparison:
`PVC/diagnostics/root-start-accuracy-20261118/comparison.json`.
Independent result:
`PVC/run-preparation/root-start-accuracy-terminal-independent-20261118T2001.json`.
Execution identities and the full receipt chain remain in `DETAILS.md`.

Both variants' isolated continuous/resumed starts are exact. The earlier
isolated root-stage replay still fails to reproduce historical continuous
`f`. Better start accuracy therefore does not identify the cause of the
full-model continuation failure, validate subsequent root iterations or
Newton-Schulz, or release the attribution waves.

### Rational-Start Full-Model Continuation Proposal

Proposed November 18 20:33; complete CPU/archive acceptance at
21:18:19; GPU controller acknowledged at 21:43:06, Asia/Shanghai.
Six helpers exist:
`tracking/rational_continuation_contract.py`, `rational_root_start.py`,
`rational_continuation_predecessor.py`, `prepare_rational_continuation.py`,
`rational_continuation_validation.py` and
`rational_continuation_controller.py`, with behavior tests in
`test_rational_root_start.py` and `test_rational_continuation.py`.
The initial focused CPU suite passes 24 tests in 48.31 seconds with CUDA hidden,
offline dependencies, all seven safety-marker exclusions and the
60-second per-test/600-second session guards.
Evidence: `PVC/run-preparation/test-rational-continuation-complete-focused-20261118T2053.xml`.
The earlier 22-test result is retained separately. At that initial
20:54:59 checkpoint, no distinct candidate, complete candidate CPU/archive
receipt, GPU controller binding or acknowledged launch exists. Later
candidate attempts and accepted CPU evidence are recorded below.

Covered behavior includes compiled start accuracy against the independent
FP64 reference, unchanged inputs/RNG and seven unrelated outputs,
already-replaced-callable refusal, per-rank resume source/variant identity,
scoped candidate transformation and unowned-training refusal. CPU
acceptance rejects incomplete/skipped XML, changed source and noninteger
zero exits while retaining failed attempts. Real subprocess tests verify
detached-child and timeout cleanup and allocation restoration under locks
after validator spawn/nonzero failures. Helper discovery retains all 104
accepted parent helpers within the new 112-helper closure; six accepted
CPU fixtures are copied and source-archived separately.
These focused tests do not establish complete candidate acceptance or
exercise the twelve GPU commands. At the initial focused checkpoint,
retained-trajectory and terminal reconstruction coverage remains pending;
the expanded suite and complete acceptance below cover those behaviors.

At 21:03, the expanded suite records 42 passes and one failed test in
84.16 seconds. The failure is a `NameError` introduced by an unused-variable
edit in the candidate-preservation test; it occurs before that test's
unowned-training check. After correction, that test passes independently
in 10.53 seconds and all eight new files plus this journal pass scoped
lint. The failed run remains preserved:
`PVC/run-preparation/test-rational-continuation-reconstruction-20261118T2102.xml`;
correction evidence is
`PVC/run-preparation/test-rational-continuation-name-correction-20261118T2103.xml`.
The added tests cover complete per-rank interrupted/resumed start
receipts, hard-link/source/variant rejection, all six historical command
coordinates and all checkpoint ranks, and terminal rejection for live
owners, noninteger-zero exits, changed logs, invalid chronology or a
replacement keeper. Historical reconstruction now authenticates
interrupted evidence as well as complete trajectories.
A fresh read-only predecessor check completed by 21:02:38 verifies exact
keeper `1569892/933748533`, its eight original rank generations and all
eight physical A100 UUIDs, with original start-accuracy owners absent.
Complete archived candidate acceptance remains the next gate; no GPU
transfer or training command has been issued.

At 21:05, distinct candidate
`PVC/candidates/rational-start-full-model-20261118T2104` is prepared
with exactly the six declared start/trainer/owner file changes.
Preparation: `rational-continuation-preparation.json`. The complete
archived CPU suite starts with output
`PVC/run-preparation/rational-continuation-cpu/20261118T2105`. At
21:06:14 it is still running and has reported two test failures. Their
complete failure reports, XML, log, source archive and receipt must be
retained and reviewed before any correction or acceptance. No binding
freeze, allocation transfer or GPU command exists.

The first complete candidate suite finishes at 21:07:37 with 216 passes
and two failures in 122.09 seconds. Both distributed moment tests fail
in child-process import before the distributed calculation:
`ModuleNotFoundError: No module named 'apdcephfs_tj32'`. The external
evidence directory used as pytest's root causes absolute-path
test-module names that multiprocessing cannot import. Using the
unchanged candidate as pytest's root makes both tests pass in 14.08
seconds; evidence is
`PVC/run-preparation/test-rational-candidate-distributed-import-20261118T2108.xml`.
The corrected runner retains importlib mode, hidden CUDA, safety
exclusions and timeout guards. The failed candidate, complete source
archive, XML, log and receipt remain unchanged. Failed receipt:
`PVC/run-preparation/rational-continuation-cpu/20261118T2105/receipt.json`
A distinct corrected candidate and complete rerun remain required.

The candidate-root rerun finishes at 21:12:19 with all 218 tests passing
in 127.12 seconds. Acceptance remains rejected because JUnit omits
module names for 69 tracking cases outside its root, preventing
complete-module authentication. This zero-exit attempt is preserved as
`needs_attention`:
`PVC/run-preparation/rational-continuation-cpu/candidate-root-20261118T2110/receipt.json`.
A common experiment-root/prepend-import probe then passes both
distributed tests and seven terminal cases in 15.34 seconds, with all
nine JUnit module names present:
`PVC/run-preparation/test-rational-common-root-cpu-runner-20261118T2114.xml`.
The final runner uses that common root and import mode. The duplicate
tracking invocation of `test_rational_root_start.py` is removed; the
same three installer tests still execute from the prepared candidate. No
skipped tests or numerical tolerance changes are introduced.

The 21:06 scheduled document refresh refuses publication at 21:08:42
because manual journal edits occur during rendering. It preserves both
documents and records `ledger_refresh_refused_retry_next_cycle`.
Replacement monitor `1588227/933927668` remains live with one `sleep 900`
child. A stable-document refresh will be verified after the next
acceptance milestone; no replacement monitor is started.

At 21:16, the distinct common-root candidate is prepared:
`PVC/candidates/rational-start-full-model-common-root-20261118T2115`.
Its complete CPU suite starts under
`PVC/run-preparation/rational-continuation-cpu/common-root-20261118T2116`.
No failures are reported at 21:16:23; acceptance requires the final
integer-zero exit, complete non-skipped XML/module coverage, unchanged
source, all six fixtures and authenticated archive. The earlier 218-pass
suite remains rejected and is not relabeled.

At 21:18:19, complete candidate CPU/archive acceptance succeeds:
215 tests pass in 127.27 seconds with zero failures, errors or skips,
all 13 expected modules present and no duplicate test IDs.
All 112 helper files, six accepted CPU fixtures, scoped candidate changes,
data identity and complete source archive authenticate. A fresh
`require_cpu` reconstruction confirms the same accepted receipt by 21:19:
`PVC/run-preparation/rational-continuation-cpu/common-root-20261118T2116/receipt.json`.
The three installer tests execute inside the candidate; the reduced total
reflects removal of their duplicate tracking invocation.
Scoped code/journal lint passes. This accepts CPU/source/archive
preparation only. No twelve-command GPU result, continuation pass,
optimizer-quality result, efficiency result or SwanLab import follows.
Production Waves 13-15 remain denied by the unresolved original
continuation failure.

The stable-document refresh attempt begun at 21:16 also refuses publication
after an external writer edits both documents at 21:17:26. The refusal
preserves those edits. A later guarded refresh succeeds at 21:24:28,
verifying all manual prose outside both automated blocks byte-preserved.
The single monitor's scheduled cycle then succeeds at 21:26:40; both
blocks contain the authenticated report rendered at 21:23:43.
No monitor is started, stopped or duplicated during this continuation.
Subsequent manual milestone edits use the same update lock.

At November 18 21:28-21:31, a fresh archived-helper reconstruction
revalidates complete CPU/archive acceptance and the exact restored
keeper `1569892/933748533`, all eight original rank generations and
physical A100 UUIDs. The candidate bootstrap independently verifies
all 112 pinned Toner files without CUDA execution.
The existing 900-second document monitor remains live and sleeping.
The archived validator's read-only six-command historical provenance
preflight completes at 21:34:51, authenticating original commands,
manifests, source, data, histories and every retained rank checkpoint at
22/23/24, including interrupted manifests. Receipt:
`PVC/run-preparation/rational-continuation-historical-preflight-20261118T2130.json`.
This verifies retained provenance and supplies no numerical continuation pass.

Freeze completes at 21:33:10 from the accepted CPU archive. A fresh
check from the separately issued helpers passes by 21:34. Binding:
`PVC/run-preparation/rational-continuation-binding-20261118.json`;
archive: `PVC/run-preparation/source-rational-continuation-20261118.tar`;
helpers: `PVC/controllers/rational-continuation-20261118/tracking`.
Saved reporting independently reconstructs CPU/source evidence and confirms
no controller intent, state or result at 21:41. Fresh issued-source,
idle-training and exact-keeper/owner-absence checks under the existing
allocation locks precede GPU start. No production wave is released.

GPU launch acknowledges at 21:43:06 through the issued `--phase start`
entry point. Original controller `1643665/934354200` and direct validator
`1643862/934355655` pass exact generation/command/environment/parent
checks. Keeper release is complete. At 21:44, the first pinned/control
continuous command's torchrun `1644120/934357603` and ranks `1644202`
through `1644209` are live; CUDA occupancy matches all eight original
physical A100 UUIDs. Compiler workers remain enabled. The first command
has no retained outcome yet, so loss and continuation results are unknown.
Canonical saved execution:
`PVC/run-preparation/rational-continuation-state-20261118.json`;
result: `PVC/run-preparation/diagnostic-rational-continuation-gpu-20261118.json`.
The single 900-second monitor refreshes both documents at 21:44:29.

Eight new receipt-backed reporting tests pass in 1.43 seconds with CUDA
hidden; scoped code/journal lint passes. The integrated reporting/ledger/
digest suite records 279 passes and one concurrent-edit test failure in
the separate digest-consolidation helper. This journal-safety failure
is preserved, then corrected by checking both document contents before
consolidation. The strengthened regression preserves timestamps while
changing either document. The complete rerun passes 281 tests in 102.36
seconds; source/journal lint and `git diff --check` pass. `DETAILS.md`
passes all checks except its preserved large-file limit (1,141.1 KiB).
No issued GPU helper is changed.

By 21:49:13, pinned/control continuous24 has exited integer zero and
the validator has authenticated its complete retained trajectory:
24 metric rows, 12,582,912 tokens and all eight rank checkpoints at
22/23/24. Its endpoint C4/selection losses are
`6.902892915655013 / 6.828252455671001`; these bounded diagnostic
measurements establish no full-budget quality or efficiency comparison.
The next command is pinned/control interrupted22. Original controller
and validator remain generation-bound; the remaining eleven commands
and complete continuation/reproduction/transparency/terminal gates are pending.

At 21:54:35, the saved receipt contains three integer-zero pinned/control
commands. Continuous24 and interrupted22 trajectories authenticate;
resumed24 has reached its endpoint and exited zero, with checkpoint/
provenance reconstruction pending. Pinned/trace and all rational commands
remain unexecuted at this observation; full diagnostic interpretation,
exact terminal restoration and production release remain pending.
A guarded refresh completes by 21:54 using the report rendered at
21:51:40 and verifies all manual prose in both documents byte-preserved.
The single monitor `1588227/933927668` remains live, sleeping 900 seconds
after its 21:44:29 scheduled success. Fresh issued checking by 21:54
authenticates the unchanged binding and all 112 helper files.

At 22:04:51, the validator records four integer-zero commands with retained
trajectory authentication: pinned/control continuous24, interrupted22,
resumed24 and pinned/trace continuous24. No `analysis` or failure is recorded
yet. The fresh read-only observation at 22:06:02 verifies original controller
`1643665/934354200`, direct validator `1643862/934355655` and eight marked
training ranks on the expected physical A100s during pinned/trace
interrupted22. Its source and saved owner receipts authenticate; full tensor,
historical-reproduction and trace-transparency reconstruction remain pending.
No rational trajectory or terminal restoration exists at this checkpoint.

The existing Bash monitor now saves immutable process/device observations
before refreshing both documents. Generation substitution or a different
physical allocation refuses capture; a capture refusal is reported separately
and does not prevent the receipt-backed journal refresh. The scheduled cycle
captures original owners and active ranks at 21:59:31 and publishes both
documents at 22:02:35. The same monitor remains live and holds its original
lock; no duplicate monitor or competing GPU task is launched.
The focused observation suite passes 12 tests in 1.56 seconds, after a
preserved 10-pass/two-failure attempt with missing fixture markers. The complete
reporting/journal/digest suite passes 285 tests in 102.43 seconds with CUDA
hidden. All mutable reporting changes remain outside the frozen numerical
helper closure. Receipts and source identities are in `DETAILS.md`.

The next observation at 22:10:16 verifies the same original owners and
eight active ranks during pinned/trace resumed24. The saved result at
22:08:50 contains five integer-zero commands and five retained trajectories;
pinned/trace interrupted22 is now authenticated. Complete analysis is absent.
A guarded split-document refresh rendered at 22:08:51 completes by 22:12.
Scoped code lint, `ENDEAVOR.md` lint and `git diff --check` pass.
`DETAILS.md` fails only the preserved large-file limit; no history is removed.

At 22:12:09, pinned/trace resumed24 exits integer zero. All six pinned
commands have now completed execution, with five retained trajectories
authenticated and the sixth audit pending. No rational command or complete
analysis is recorded. The current-state table uses this saved outcome time;
the latest explicit live process/device observation remains 22:10:16.

By 22:15:03, all six pinned retained trajectories authenticate. The first
rational/control command is verified live on the original A100s at 22:16:07
and exits integer zero at 22:17:56; its trajectory authenticates at 22:20:52.
At 22:24:20, the rational/control interrupted22 trajectory also authenticates,
giving eight integer-zero commands and eight retained trajectories.
Read-only observation at 22:24:58 verifies the original controller/validator
and all eight training ranks during rational/control resumed24.
No complete numerical analysis is recorded.

The separate `tracking/rational_continuation_followup.py` is archived and
CPU-tested before issuance. Its eight real-owner/readiness/child tests pass
in 2.97 seconds; source/archive acceptance completes at 22:21:30. The issued
archived helper acknowledges original owner `1676886/934589879` at 22:22:23,
with CUDA hidden. Fresh exact command/cwd/environment/generation verification
confirms it live by 22:23. The first saved state is
`waiting_for_original_owners`; it authenticates the original GPU controller
and validator, checks every 900 seconds and issues zero commands while they
remain present. Generation substitution or incomplete terminal state refuses
the follow-up. After actual original-owner absence, it will run only the
frozen issued `rational_continuation_controller.py --phase terminal`, retaining
original output and child cleanup evidence. It has no allocation, training
or production-release authority.

The recurring reporter now authenticates that separate CPU archive,
acknowledged owner and saved follow-up state. Pending/failed states remain
distinct from a completed independently reconstructed terminal receipt.
Focused follow-up/reporting tests pass 28 cases in 4.42 seconds; the complete
integrated suite passes 297 tests in 108.64 seconds. Required code lint passes.
The existing Bash monitor captures at 22:17:37 and publishes both documents
at 22:20:03. Its original process and lock remain unchanged.
Detailed source identities, archived commands and receipts are in `DETAILS.md`.

Rational/control resumed24 exits zero at 22:26:22; its trajectory
authenticates by 22:29:13, giving nine integer-zero commands and nine
retained trajectories. The 22:29:03 live observation verifies the original
controller/validator while no training ranks are present during the audit.
At 22:31:01, the same exact owners and eight ranks on the expected A100s
verify live during rational/trace continuous24. Complete analysis remains
pending. The guarded split-document refresh rendered at 22:28:34 completes
by 22:31 and includes the read-only follow-up status.
The 297-test integrated receipt, code/document lint and `git diff --check`
pass; `DETAILS.md` retains only its existing size-limit exception.

At 22:40:33, all eleven completed commands have authenticated trajectories.
The 22:42:30 read-only capture verifies the exact original controller and
validator plus ranks `1691751` through `1691758` on the eight expected A100s
during the final rational/trace resumed24 command. Complete analysis and
terminal restoration remain pending. The single Bash monitor captures at
22:35:05 and publishes both documents at 22:37:29. The separate terminal
follow-up's 22:37:25 observation remains `waiting_for_original_owners`;
no competing verifier or GPU task is issued.

The final rational/trace resumed24 command exits integer zero at 22:43:51,
giving all 12 planned execution exits. At the 22:46:10 capture, eleven
trajectories are authenticated, the original controller/validator remain
live, and no training rank or CUDA context is observed while final retained
files are audited. No `analysis`, numerical continuation acceptance or
restoration is yet recorded. Execution exits alone do not accept the
intervention.

All twelve retained trajectories authenticate at 22:46:43. The 22:49:23
read-only observation verifies the same exact original owners with no active
training rank or CUDA context during analysis. Complete comparison gates,
numerical acceptance and exact terminal restoration remain pending.

At 23:03:15, fresh read-only observation still verifies the original
controller/validator live with all twelve execution outcomes retained.
The validator remains in analysis; no completed gate result is published.
The single Bash monitor captures at 22:52:30 and refreshes both documents at
22:55:53. The terminal follow-up's 22:52:25 original-owner observation
remains waiting. A slow retained-file analysis is not terminal failure and
does not permit restarting the diagnostic or competing for its allocation.

At November 19 00:35:42, the original validator publishes
`completed_rational_start_full_model_diagnostic_not_production`.
Trace-off/on transparency and pinned historical reproduction both pass;
`rational_all_rank_continuation` and `diagnostic_intervention_accepted`
are false. This is the validator's saved result, pending separate
original-controller and terminal reconstruction.

| Variant, Both Protocols | Update-22 Tensor Failures | Update-23 Tensor Failures | Update-24 Tensor Failures | Scalar-Loss Failures |
| --- | --- | --- | --- | --- |
| Pinned | 0 | 97 | 460 | 4/39 |
| Rational | 0 | 97 | 439 | 5/39 |

Tensor counts pool eight rank-specific checkpoints. All 97 update-23
failures are on rank 0; update-24 failures occur on every rank. Rational's
update-24 counts are 254/28/28/27/26/26/26/24 by rank 0-7, versus pinned's
260/30/30/30/28/28/28/26. Update-23 scalar/structure differences remain
16 for either variant; update-24 has three for pinned and zero for rational.
Neither smaller failure counts nor isolated start accuracy accepts the
failed continuation. All four protocols' scalar-loss comparison sets
fail the original tolerance, so these bounded losses are not evidence
of improved training quality.

At 00:41:37, read-only authentication finds original controller
`1643665/934354200` live and direct validator `1643862/934355655` absent,
with no active training ranks or GPU compute processes. The controller
is rederiving the complete retained result under its original allocation;
saved state remains `validating`, with zero cleanup and absent marked
descendants. The original CUDA-hidden terminal follow-up remains live;
its saved 00:37:28 state waits for original owner absence.
Independent terminal/restoration remains required. No original owner is
restarted and no numerical tolerance is weakened.

Question: does installing the accepted rational start before any original
model/root warm-up change the failed full-model continuation at the
unchanged tolerance, while reproducing the pinned controls and preserving
trace transparency?

| Setting | Proposed Protocol |
| --- | --- |
| Parent | Preserved `fused-continuation-owner-fixture-20261118T0634`, bound by `PVC/run-preparation/fused-continuation-binding-20261118.json` |
| Arms | Pinned and rational start, each with trace-off/control and trace-on execution |
| Trajectories | Continuous to 24; interrupted at 22 and resumed to 24; retain checkpoints and traces at 22/23/24 |
| Commands | Twelve distinct eight-rank commands; one-shot execution, no original-run restart |
| Training settings | Seed 0, LR `0.04525483392`, 524,288 tokens/update, unchanged 5,120-update schedule with 1,024 stable and 4,096 decay updates |
| Planned diagnostic tokens | 100,663,296, excluding warm-up; this is a bounded diagnostic, not a completed production budget |
| Runtime | Compilation and cached-root CUDA graphs enabled; no scratch-readiness intervention |
| Numerical gates | All-rank checkpoint/loss continuation, pinned historical reproduction and trace transparency; `rtol=1e-5`, `atol=1e-6` unchanged |

The candidate transformation is scoped to installing the source-authenticated
start, adding explicit per-rank variant receipts and resume checks, and
retargeting the existing execution owner. Pinned author code, other
optimizer arithmetic, configuration and data remain byte-preserved.
The rational callable is installed before warm-up, so this experiment
retains the original full-model capture context omitted by isolated replay.

CPU/XML/archive acceptance, authenticated freeze, issued checking and
fresh exact-keeper handoff are complete; the GPU diagnostic is running.
Process-local graph-descriptor or scalar
differences must remain reported and must not be silently excluded to
obtain a pass. Use the completed start-accuracy terminal as the allocation
predecessor; freshly verify original owner absence and the exact restored
eight-A100 keeper under the existing locks. The existing monitor continues
at 900-second intervals; no competing GPU work or duplicate monitor is
authorized by this proposal.

Stopping condition: retain all original command outcomes and independently
reconstruct the complete checkpoints, losses, trace lifecycle and exact
restoration. Interpret the rational intervention only after pinned
historical reproduction and both trace transparencies pass. Failed
reproduction withholds causal attribution. Failed rational continuation
rejects this proposed correction. Even an accepted bounded intervention
requires the separate complete production gates before Waves 13-15,
followed by their full histories, large shared-token-window comparisons
and verified private SwanLab uploads.

### Compensated NS Operand-Precision Study

Proposed and CPU/source-archive accepted November 18 22:43:10. Question:
does retaining low-order BF16 operand components reduce the historical
continuous/resumed NS-output difference enough to justify a separately
owned GPU diagnostic? The pinned normalization, coefficients
`(3.4445, -4.7750, 2.0315)`, ten polynomial steps and `eps=1e-5`
are unchanged. Only its batched-product primitive is replaced in this
CPU study; no training source or issued GPU helper changes.

For an FP32 operand `x`, define `x0 = FP32(BF16(x))`,
`x1 = FP32(BF16(x-x0))` and `x2 = FP32(BF16(x-x0-x1))`, and
similarly for `y`. `bf16_1` retains `x0@y0`; `bf16_3` adds
`x0@y1 + x1@y0`; `bf16_6` also adds
`x0@y2 + x1@y1 + x2@y0`. Corrections are summed before the leading
product. Each NS step has three batched products, so the nominal product
counts are 3, 9 and 18 per step, respectively; FP32 uses three.
All products and accumulations execute on CPU in FP32 after operand
rounding. These counts are an arithmetic cost description, with no measured
CUDA or elapsed-training speedup.

The initial CPU test receipt records 18 passes and two failures at the
unchanged `rtol=1e-5`, `atol=1e-6` FP64 comparison. Plain FP32 also
fails on those near-rank-deficient fixtures, whose smallest singular values
are approximately `1.18e-8` to `3.69e-8`; the six-product maximum absolute
errors are `0.00406064` and `0.00212469`. The accepted tests preserve these
two negative regimes and add independent full-rank wide/tall controls.
Their FP32 and six-product errors satisfy the original tolerance; single-
and three-product modes do not. Noncontiguous product accuracy, exact
residual cancellation and exact pinned-FP32 parity also pass.
The final mutable suite passes 22 tests in 4.19 seconds; the same 22 tests
pass from the archived source in 6.72 seconds. Required scoped code lint passes.

Source/test receipt:
`PVC/run-preparation/ns-operand-precision-cpu/20261118T2243/receipt.json`.
The failed receipt remains
`PVC/run-preparation/test-ns-operand-precision-20261118T2240.xml`.
The archived runner executes with `CUDA_VISIBLE_DEVICES=` and eight CPU
threads against the authenticated historical rank-0/update-23/occurrence-0
`[16,512,512]` pair, in full-batch `C,R,C,R` order. An independent per-matrix
FP64 finite-polynomial reference is retained; it is not an exact polar
reference. Retained outputs, hashes and all original-tolerance comparisons
are written to
`PVC/run-preparation/ns-operand-precision-saved-pair-20261118T2243`.
The archived study completes at 22:45:29. Retained-output hashes and all
pair/repeat/FP64 comparison metrics are independently recomputed at 22:46:12:

| Arithmetic | Products / NS Step | C/R Relative Frobenius Error | Pointwise Tolerance Failures |
| --- | ---: | ---: | ---: |
| CPU FP32 | 3 | `3.7354873e-5` | 809,599 |
| CPU `bf16_1` | 3 | `0.0368394852` | 4,191,356 |
| CPU `bf16_3` | 9 | `6.7912612e-5` | 2,546,005 |
| CPU `bf16_6` | 18 | `3.6827009e-5` | 792,534 |
| Per-matrix FP64 reference | 3 | `3.6620813e-5` | 789,618 |

All modes have exact finite same-input repeats. Each comparison covers
4,194,304 elements at `rtol=1e-5`, `atol=1e-6`, and every pair fails.
The single/six-product maximum pair errors are `0.01455230 / 3.74168e-5`;
their mean absolute errors are `0.00117900 / 7.82976e-7`.
The six-product continuous/resumed outputs also fail comparison with their
own FP64 references in 3,078/3,117 elements, versus FP32 4,524/5,063.
None of the eight mode/input FP64 comparisons passes.

Retained verification:
`PVC/run-preparation/ns-operand-precision-saved-pair-20261118T2243/retained-verification.json`.
These CPU measurements show that compensated operands reduce the BF16
amplification but leave a finite-polynomial input-sensitivity difference
above the original tolerance. Increasing operand precision alone is
insufficient for this saved pair. CPU accumulation differs from CUDA;
no new historical CUDA reconstruction or causal upstream conclusion follows.

Stopping condition: preserve every pair/reference mismatch and same-input
repeat, then choose the next correction from both this study and the complete
rational full-model diagnostic. This CPU study has met its stopping condition.
Do not issue compensated operands alone as a claimed continuation fix.
The next useful diagnostic must test the upstream source discrepancy or
the polynomial sensitivity while preserving the original gates and controls.
A CPU improvement alone does not pass CUDA continuation or permit a
production wave. Full saved-pair GPU reproduction,
original full-model gates, quality and elapsed-time measurements remain
required. No SwanLab optimizer history or efficiency claim follows.
Henry, Tang and Heinecke, arXiv:1904.06376v1 (2019), is recorded as related
literature metadata only; full text was unavailable for inspection.

### NS Precision-Placement Screen

Proposed and implemented November 19 19:30, Asia/Shanghai. Question:
can higher-precision iterations concentrated at the beginning or end
of the pinned finite map reduce the historical input-pair difference
without requiring FP32 throughout? This tests endpoint placement,
not interior placement or general alternation. It changes numerical
arithmetic and requires separate future CUDA and training acceptance.

For five and seven steps, compare one, two and three FP32 iterations
at the prefix or suffix, plus homogeneous BF16-operand and FP32 controls.
Add homogeneous ten-step controls, giving eighteen schedules.
Each iteration's three products use one explicit immutable mode.
The recurrence state, normalization and accumulation remain FP32;
BF16 denotes CPU operand rounding followed by FP32 products.
Normalization is performed once. Orientation, expression ordering,
coefficients `(3.4445, -4.7750, 2.0315)`, `eps=1e-5` and
`rtol=1e-5`, `atol=1e-6` remain unchanged.

For each batch matrix, initialize `X = FP32(U)/(||FP32(U)||_F+eps)`,
then transpose if rows exceed columns. With scheduled product `P_i`,
each iteration evaluates `A = P_i(X, X^T)`, then
`X = a*X + P_i(b*A + c*P_i(A,A), X)`. Return the original
orientation in FP32. `P_i` is FP32 batched multiplication, or
batched multiplication of operands rounded to BF16 and restored to
FP32. No extra renormalization occurs between iterations.

Before execution, fix the comparison rule: compare prefix and suffix
at the same step count and FP32-product count, reporting C/R relative
Frobenius error and worst-matrix error alongside each output's own
FP64 finite-polynomial accuracy. All full-batch `C,R,C,R` outputs,
same-input repeats and geometric metrics must be retained and
independently reconstructed. The earlier authenticated SVD-polynomial
references are reused; their arithmetic is not rerun. Exact reproduction
of all six homogeneous saved controls is required before interpreting
placements. Partial-polar geometry remains descriptive, with no output
grafting or training-quality claim. Counts of `3k` FP32 and `3(q-k)`
BF16-operand products are an arithmetic inventory, not measured speed.

The first focused test run retains nine passes and two failures:
per-matrix prefix references differ from the pinned batched map in
10/30 and 15/30 elements at the original tolerance, with maximum
absolute differences `1.97887e-5` and `1.56164e-5`.
The corrected tests compare an independently expressed batched
recurrence at exact tolerance and retain the per-matrix failures as
negative reduction evidence. All thirteen tests pass in 1.58 seconds,
including exact homogeneous-map parity, mixed-schedule repeat/input
preservation and changed-retained-output detection.

The separate source/test archive inherits the accepted localization
execution inventory. It changes only the numerical source loader and
its tests, then adds the precision runner, tests and preparer.
The earlier loader authenticated an open descriptor but reread the
pathname for executable controls; those bytes could differ from the
traced AST. The correction executes the three selected definitions
from the same verified buffer after the existing file checks finish.
Two deterministic pathname-replacement regressions verify rejection
during verification and unchanged numerical/canonical-filename behavior
after verification, without executing an unverified side effect.
The component's 24 and execution's 23 CPU cases pass separately.
Accepted parent archives remain preserved; owner implementation still
needs a fresh binding to the reviewed transition actual-LR queue.

At this checkpoint, separate complete CPU/source acceptance is running,
and no mixed saved-pair result, CUDA execution, continuation correction,
optimizer loss, elapsed-training efficiency or SwanLab import is claimed.
The CPU screen stops after retained numerical comparisons and controls
are independently verified. A useful placement must still justify its
own GPU protocol and actual-configuration continuation checks before
fair full-budget LR tuning, five large shared-token windows, verified
private histories and fresh paired seeds.

At 19:31:36 the first complete archive passes60cases in144.38seconds
and records158helpers/526inventory entries. Independent review
withholds measurement: loaded helper origins were not checked by
both public entry points, verification omitted the published replay
report, and subprocess/XML failures could omit a structured receipt.
This archive remains preserved and unlaunched. The correction adds
entry/helper-origin checks before accepted-evidence reads, reconstructs
the complete replay report including saved-output comparisons and
checks scope/tolerance, and retains timeout/start/XML failures.
Four boundary regressions bring focused coverage to17passing cases
in10.36seconds.

The distinct reviewed archive passes64cases in153.41seconds and is
accepted at19:37:53:
`PVC/run-preparation/ns-precision-placement-cpu/reviewed-20261119T1935/receipt.json`.
It retains158helpers/526inventory entries, with only the two declared
parent corrections and three new helpers. Fresh archived-entry
checking passes by19:39. Independent code re-review clears all three
findings; full public-verifier tampering, separate subprocess-start
failure and malformed-XML regressions remain coverage gaps.
The CPU saved-pair measurement is issued from this archive at19:39;
its complete eighteen-schedule result and independent verification
remain pending. No CUDA or optimizer acceptance follows.
The original single Bash monitor publishes both journals at19:34:05.
At19:34:40 GPU coverage is20/54commands and6/18decisions; the first
fused BF16/five-step interrupted22 command completes, while its
resumed24 and comparison remain pending.

#### Precision-Placement CPU Result

The archived measurement completes with integer-zero exit at
19:41:16. Independent verification completes at19:42:28, rederiving
all eighteen trials' output hashes, full replay reports, saved-output
comparisons, C/R and repeated C/R comparisons, own-reference errors,
per-matrix metrics and geometry from retained tensors. All six
homogeneous controls reproduce their prior full-batch outputs exactly;
all36same-input repeat comparisons are exact finite.
Evidence:
`PVC/run-preparation/ns-precision-placement-saved-pair-20261119T1940/result.json`
and `retained-verification.json`.

| Steps | FP32 Iterations | Prefix C/R Error | Suffix C/R Error | Worst-Matrix Suffix Error |
| --- | ---: | ---: | ---: | ---: |
| 5 | 1 | `0.00912089` | `0.00406587` | `0.00652437` |
| 5 | 2 | `0.00844020` | `0.000458246` | `0.000761564` |
| 5 | 3 | `0.00614314` | `0.0000349660` | `0.0000897796` |
| 7 | 1 | `0.0215829` | `0.0180140` | `0.0215898` |
| 7 | 2 | `0.0209555` | `0.0123973` | `0.0162282` |
| 7 | 3 | `0.0188395` | `0.00548877` | `0.00897795` |

Errors are relative Frobenius differences between continuous and
resumed inputs' outputs. All eighteen full-batch pairs fail
`rtol=1e-5`, `atol=1e-6`; the homogeneous five/seven-step FP32
pair errors remain `1.00113e-5` and `1.95859e-5`.
Suffix placement reduces pair error relative to its matched prefix
by2.24/18.42/175.69times at five steps and1.20/1.69/3.43times at
seven steps. This result is limited to one historical occurrence
and CPU operand emulation.

The five-step two-BF16/three-FP32 suffix uses six BF16-operand and nine
FP32 products. Its771,245/4,194,304 pair elements fail the original
pointwise tolerance. Its continuous/resumed own-reference comparisons
also fail in3,530,584/3,529,756elements, with relative errors
`0.000258459 / 0.000256545`. By contrast, homogeneous five/seven-step
FP32 outputs pass their own references. Every mixed schedule fails
both own-reference comparisons. The suffix's continuous mean cosine
to the partial polar is0.9513715, close to homogeneous five-step FP32,
but cosine alone hides its pointwise errors and does not measure
optimizer loss. Seven-step three-FP32 suffix retains0.9803958mean
cosine yet has much larger pair/reference errors.

Each cosine is the whole-matrix Frobenius inner product of the
output with its source's FP64 SVD partial polar, divided by the
product of their Frobenius norms; the reported mean weights all
sixteen batch matrices equally. The partial polar replaces non-null
singular values by one, with null directions set to zero at
`sigma <= eps_fp64*max(rows,columns)*sigma_max`.

Decision: the screen meets its CPU stopping condition. Precision at
the suffix is worth retaining as a bounded numerical hypothesis;
none of these schedules is a continuation correction. Do not enlarge
the current running matrix or alter accepted controls. After the
current original and actual-LR diagnostics complete with canonical
terminals and exact restoration, a separately accepted CUDA protocol
may compare the five-step two-BF16/three-FP32 suffix with its reverse
placement and homogeneous five-step BF16/FP32 and ten-step BF16 controls.
Require two opposite-order fresh processes, full original source
layouts, separate training-eager and diagnostic-compiled controls,
call/capture-executed/replay/repeat evidence, raw event-timing samples,
independent retained reconstruction and exact cleanup/restoration.
Compilation and graphs remain enabled. Preserve the original saved
ten-step BF16 C/R rejection and the completed fully compiled NS
diagnostic's failed historical-control and pair predicates; source
authentication or new owner acceptance cannot change those outcomes.
Original tolerances stay unchanged. This proposed mixed-schedule
CUDA successor is not implemented, accepted, queued or launched.
The corrected compiler-localization numerical archive is preserved;
its exclusive owner and new transition-queue binding remain required.
Whole-training efficiency, fair LR-tuned quality, fresh paired seeds,
five large shared-token windows and verified private SwanLab histories
are still required for an optimizer improvement claim.

#### Fused Five-Step Continuation Rejection

November19 19:50:23, Asia/Shanghai: the original transition aggregate
publishes its first fused result, `outer_leftdiag_fused_ns_tpu_q5_full_lr_lower`,
at matrix LR0.04525483392. Continuous24/interrupted22/resumed24
complete with successful execution and cleanup, but the numerical
case is rejected. The full5,120-update LR schedule, compiled
surrounding training helpers, cached-root graphs and original
`rtol=1e-5`, `atol=1e-6` remain unchanged.

Boundary22passes2,127tensor comparisons with zero failures and zero
scalar/structure differences. Post-cached-root24fails419/2,199tensor
comparisons, spanning all eight ranks, with zero scalar/structure
differences. Rank failure counts are248/27/27/24/25/23/23/22.
Update23is diagnostic only:120/2,167tensor comparisons fail and
sixteen graph-state flag differences remain recorded. Captured
output rows, update equations and matched work pass their gate.
Of39scalar-loss comparisons,36pass and three update24evaluations
fail:

| Metric | Resumed | Continuous | Resumed Minus Continuous |
| --- | ---: | ---: | ---: |
| Selection Loss | `6.820275175` | `6.820408647` | `-0.000133472` |
| Held-Out DCLM | `6.720488783` | `6.720626453` | `-0.000137670` |
| Held-Out StarCoder | `8.254213303` | `8.254301800` | `-0.0000884964` |

These signs are resumed-versus-continuous reproducibility differences;
they do not compare optimizer quality or establish a loss improvement.
The five-step BF16 budget has not corrected this fused continuation
case. MuonT's same-mode/same-budget control passes, but this partial
matrix does not locate the fused discrepancy's upstream source.
Retain the rejection and finish all18decisions before a canonical
terminal. The separately reviewed actual-LR queue must preserve
rejected cases and wait for exact restoration; its readiness is not
granted by successful process exits. Full-budget attribution Waves13-15
remain withheld. No production history or SwanLab upload follows.

#### Precision-Placement CUDA Component Draft

November19 19:54: the proposed five-schedule CUDA diagnostic now has
local numerical-map and file-backed execution/reconstruction code.
The maps retain the unmodified pinned routine for homogeneous
five-step BF16/FP32 and ten-step BF16 controls. Two changed five-step
maps place three FP32 iterations after or before two BF16-operand
iterations. Each iteration's three products share its explicit
mode. Default CUDA products use pinned BF16 operands with FP32
accumulation; CPU product injection is restricted to synthetic
test emulation.

Each of two fresh opposite-order processes declares120phase files:
five schedules, four repeated-source calls and six phases
(`pinned_eager`, `compiled_call`, `capture_executed`, `replay`,
`replay_repeat`, `timed_final`). Fifty raw CUDA-event replay samples
are retained per compiled call. Eager training-path controls remain
separate from fully compiled diagnostic controls. The snapshot
called capture-executed follows the first explicit graph replay.
Reconstruction authenticates files and rederives source pairs,
historical outputs, own-FP64 finite-polynomial errors, same-input
repeats, graph drift and event statistics, preserving every failure.
Kernel-event timing excludes compile, capture, transfers and training.

Focused tests and independent review are running; this draft has
no accepted source archive, CUDA/Inductor evidence, exclusive owner,
new predecessor binding or launch. The current transition and
actual-LR queue continue unchanged. The draft cannot authorize CUDA
initialization without a separately accepted caller that verifies
physical-device ownership before initialization, between calls and
after reconstruction. CPU/source acceptance and exclusive restoration
must precede execution; later training integration still needs original
continuation gates and full-budget quality-cost comparisons.

At19:56:23, a distinct CPU/source archive accepts the numerical
maps and execution/reconstruction component:
`PVC/run-preparation/ns-precision-placement-execution-cpu/20261119T1957/receipt.json`.
All43staged cases pass13.27seconds:22map cases and21file-backed
execution cases. Fresh archived-entry checking passes by19:58.
The archive retains all158reviewed parent helpers byte-for-byte
and adds five helpers, giving163helpers/531inventory entries.
The preparer derives the authenticated parent preparation AST,
changing only successor parent, helper/test lists, counts, status
and scope import; no inherited helper is overwritten.
Required source lint passes. Independent review and the broader
numerical regression suite remain pending at this checkpoint.
No CUDA/Inductor execution, exclusive owner, new predecessor seal,
GPU handoff or training acceptance is inferred.

CPU tests cover exact pinned FP32 controls on square/tall/wide
FP32/BF16 inputs, explicit mixed recurrence parity with CPU-only
operand emulation and real Dynamo compilation using the CPU eager
backend. File-backed tests cover120-phase round trips, opposite
process pairing, partial/changed/aliased inventories, source/storage
mutation, timed-final nonfinite drift, raw-event arithmetic, owner
denial before CUDA and public reconstruction of rehashed report
substitutions. These tests do not exercise CUDA BF16 accumulation
or Inductor lowering.

The original transition's19:55:35aggregate has22/54commands and
7/18decisions:6accepted and1rejected. The fused BF16/seven-step
continuous24command completes; its remaining trajectories and
decision remain pending. Fresh19:56:59process inspection finds both
original owners, both waiters and the sole Bash monitor live.
Queue19:50:25remains `waiting_for_transition_owners`,0/6phases.
The existing monitor publishes both journals19:52:49 and continues
its900-second sleep. The completed CPU result and partial
continuation rejection supply no full-budget loss window or new
SwanLab history. Wave12remains the latest analyzed/verified
production wave; the full efficiency/performance objective is open.

#### Precision CUDA Review Checkpoint

November19 20:13, Asia/Shanghai: a distinct corrected CPU/source
archive is retained at
`PVC/run-preparation/ns-precision-placement-execution-cpu/reviewed-20261119T2001/receipt.json`,
accepted20:02:22.675480. It passes46staged cases in32.67seconds,
retains163helpers/531inventory entries and changes none of the
158accepted parent helpers. The original43-case archive remains
preserved. A fresh hidden-CUDA archived-origin `--phase check`
exits zero by20:13; this checks that exact46-case receipt and archive.

The preceding integrated CPU suite passes107cases in159.51seconds,
and the corrected focused execution suite passes24cases in14.60seconds.
Two independently identified defects are corrected: phase loads now
clone output inside the bound-file context rather than escaping
mmap-backed storage, and the imported release-owner hasher is mandatory
and origin-checked. Regression cases cover backing-file overwrite and
omitted/rebound hasher sources. Source identities, commands and test
XMLs are retained in DETAILS.

Independent review still withholds CUDA execution: closure discovery
in `ns_precision_placement_execution.py` queues an imported helper
only if its sibling already exists. A transitive local dependency
missing from both the helper directory and inventory can therefore
escape that origin check. The synthetic test-closure builder repeats
the same omission. Next correction must distinguish declared external
imports from required local imports and reject missing local helpers,
with a real missing-transitive regression before a new acceptance
archive and review. The current46-case acceptance is retained, not
rewritten or promoted into complete source-origin coverage.

No exclusive owner, predecessor seal, CUDA/Inductor execution,
training continuation, full-budget quality-cost result or SwanLab
history is issued. The two original continuation waiters have stopped
after the transition failure described below; the precision successor
cannot bypass those incomplete predecessor gates.

#### Precision Import-Closure Correction

November 19 20:35, Asia/Shanghai: the missing-transitive-import
review finding is corrected in a distinct CPU/source archive:
`PVC/run-preparation/ns-precision-placement-execution-cpu/closure-reviewed-20261119T2032/receipt.json`.
Acceptance is recorded at 20:31:01.747039; 49 staged tests pass
in 32.68 seconds, with 163 helpers and 531 inventory entries.
The 22 map tests and 27 execution/reconstruction tests retain the
original numerical tolerances. A fresh hidden-CUDA invocation of
the archived preparer with `--phase check` passes by 20:32.
The integrated compiler-localization, precision-study, map and
execution suite passes 113 cases in 165.57 seconds.

The execution binder walks authenticated static imports throughout
the required transitive helper closure. Declared standard-library,
Torch, Triton and NumPy imports are external; every other imported
module requires an authenticated local helper regardless of whether
its file exists. Same-package relative imports bind their local
sources as well; parent-relative imports are refused. Fresh-process
regressions append a deferred import to an authenticated transitive
helper, rehash that helper, and verify refusal for an absent file/hash,
a present file without its hash, and a missing relative dependency.
Existing valid package/flat import branches remain accepted.
This is static-import coverage, not a claim that arbitrary dynamic
imports are authenticated by that walk.

Compared with the preceding 46-case archive, only execution binding,
its regression tests and the preparer's expected test count change.
All 158 reviewed CPU-study parent helpers remain byte-for-byte exact,
as do the precision maps and their tests. The earlier 43/46-case
archives and the first failed 49-case test attempt remain preserved.
Receipt, archive, source and XML hashes are retained in DETAILS.
Scoped source lint and `git diff --check` pass.

The next experiment remains the separately owned five-schedule,
two-opposite-order CUDA diagnostic already specified above, retaining
eager/compiled controls, graph phases and full saved batches. It is
not launched: the failed training-path predecessor lacks complete
54-command/18-decision terminal evidence, and no precision execution
owner has been accepted. First prepare a distinct ownership-only
successor that resolves missing-environment reads using coherent
directory absence or identity-bound whole-process exit followed by
a complete rescan. Preserve the failed 23/54 diagnostic, its fused
rejection, stopped waiter/queue identities and exact restoration.
Do not restart consumed one-shots, increase retries as a substitute
for exit proof, relax tolerances or alter training mathematics.

The original Bash journal monitor publishes both documents at
20:30:47 and resumes `sleep 900` without a restart. Wave 12 remains
the latest full-budget, large-window analyzed, privately verified
SwanLab wave. No new training result, whole-training efficiency
claim, cloud import or optimizer superiority follows from CPU/source
acceptance; the research objective remains active.

#### Transition Partial Failure And Restoration

November19 20:06, Asia/Shanghai: the distinct original-training-path
mixed-exit-bracket diagnostic has stopped, preserving its frozen
candidate `PVC/candidates/ns-training-path-exit-transition-closure-corrected-20261119T1700`
and binding `PVC/run-preparation/ns-training-path-exit-transition-20261119-binding.json`.
Executed matrix LR remains0.04525483392, with the full5,120-update
schedule, continuous24/interrupted22/resumed24, eight ranks,
original numerical readers, compiled surrounding helpers, CUDA
graphs and `rtol=1e-5`, `atol=1e-6` unchanged.

Aggregate `PVC/run-preparation/validation-ns-training-path-exit-transition-20261119-gpu.json`
stops at19:59:22 with23/54 authenticated complete commands,
7/18 decisions, six accepted MuonT controls and one rejected fused
BF16/five-step case. The fused BF16/seven-step continuous24 and
interrupted22 trajectories both complete with execution/cleanup0/0.
Its resumed24 command, PID3313401, fails at19:59:21 before model
construction or training. No seven-step continuation decision,
remaining fused/null result or complete matrix is claimed.

The rank0 traceback identifies `complete_scan_exhausted` in the
issued ownership helper after three marker-scan attempts, retaining
unstable PID3313593. Its failure-attribution snapshot says the
process was not previously confirmed marked; a later read finds
its `/proc` files absent. Those separate reads cannot establish
the process's coherent marker identity or the failure's cause.
The resumed command records execution/cleanup1/1 and marked
descendants absent. The validator's generic
`NS full-model command, exact process generation, cleanup or log identity differs`
therefore reflects a nonzero-exit receipt here, not demonstrated
command/source/log tampering or an optimizer numerical rejection.
The full failure log and execution receipt remain preserved at
`PVC/validation/ns_training_path_transition_20261119/outer_leftdiag_fused_ns_tpu_q7_full_lr_lower_resumed.log`
and its `.execution.json` companion.

The controller records validator/cleanup1/1 and keeper restoration0
at19:59:40. Independent fresh checking at20:06:15 uses the exact
issued, binding-authenticated helper and verifies both original
owners, all23 completed-command handles and the failed command
absent. All validator/command markers are absent. Restored
keeper3313700 and ranks3313868-3313875 retain their recorded
generations and exclusively occupy the same eight A100 UUIDs.
Controller-marker matches belong only to that exact restored keeper.
Observation:
`PVC/run-preparation/endeavor-transition-failure-restoration-observation-20261119T2006.json`.
This verifies restoration, not full numerical reconstruction.

The original read-only terminal waiter records `needs_attention`
at20:04:58 because the owners exited without complete accepted
terminal prerequisites. Its error wording does not overturn the
separately verified physical restoration. The reviewed actual-LR
queue records `needs_attention` at20:05:59 with0/6 phases: its
external reader refuses the absent canonical predecessor terminal.
Both processes are absent at the fresh check, and no terminal,
predecessor seal, actual-LR freeze or successor GPU launch is issued.
The original Bash journal monitor remains live and keeps its
exclusive900-second loop.

Next experiment remains gated: preserve the consumed failure and
the six accepted/one rejected numerical decisions, then prepare a
distinct ownership-only correction if justified by coherent process
evidence and CPU regressions. Do not relax numerical tolerances,
reinterpret failed cases as accepted, restart this one-shot identity,
or allow the old actual-LR queue to proceed on restoration alone.
Full18-case coverage and independent terminal reconstruction are
still required before a new actual-LR handoff. The precision-placement
CUDA component remains unlaunched. Wave12's five536,870,912-token
windows and exact shared evaluations remain the latest full-budget
analysis; no new full-resolution SwanLab history exists from this
failed diagnostic. Better quality and efficiency than fairly tuned
MuonT remain unestablished.

#### Missing-Environment Scan-Exit Successor

November 19 21:00, Asia/Shanghai: the distinct namespace
`ns-training-path-scan-exit-20261119` passes complete CPU/source
acceptance at 20:56:19. Candidate:
`PVC/candidates/ns-training-path-scan-exit-reviewed-20261119T2058`.
Receipt:
`PVC/run-preparation/ns-training-path-scan-exit-20261119-cpu/reviewed-20261119T2058/receipt.json`.
All 129 cases pass in 18.54 seconds: 45 owner lifecycle, 36 inherited
handoff, 33 missing-environment lifecycle and 15 source/preservation
cases. The archive contains 178 helpers and 547 inventory entries;
fresh archived-origin checking passes at 20:58.

Hypothesis: the three-rescan failure can be resolved by waiting for
coherent whole-process teardown when a scanned process's environment
file disappears. The new handler pins a direct process directory with
`O_DIRECTORY|O_NOFOLLOW`, brackets its exact ownership through
descriptor-relative kernel-stat reads, and requires empty command,
zero resident memory and kernel-exit evidence. It accepts verified
directory disappearance or the unchanged inherited whole-process
drain when an empty environment becomes readable, then repeats the
complete scan. Missing live files, directory aliases/replacement,
PID reuse, identity drift and surviving threads remain refusals.
The earlier transient PID's marker ownership is still unknown.

The original mixed-bracket empty-command sampling, pidfd identity,
whole-process drain and timeout remain unchanged. The fresh diagnostic
retains eighteen cases and 54 eight-rank commands at matrix LR
0.04525483392, using continuous24/interrupted22/resumed24 on the
original 5,120-update schedule. Training/configuration bytes,
numerical readers, `rtol=1e-5`, `atol=1e-6`, compilation and CUDA
graphs are unchanged. Prior accepted archives and consumed one-shots
are preserved.

The first staging archive at
`PVC/run-preparation/ns-training-path-scan-exit-20261119-cpu/20261119T2100`
retains 129 passing cases but is `needs_attention` because source
changed during staging. It supplies no accepted source or launch.
The reviewed archive above is separate. The new preservation seal at
`PVC/run-preparation/ns-training-path-scan-exit-predecessor-seal-20261119.json`
authenticates the failed transition's 23/54 commands, 7/18 decisions,
six accepted controls, fused five-step rejection, failed seven-step
resumed command, stopped eleven-observation terminal waiter and
zero-phase actual-LR queue. All older failures and checkpoint identities
remain included. Seal creation passes at 20:59; independent seal
checking, freeze, keeper transfer and GPU launch remain pending.

This is an ownership-execution gate. Complete diagnostic coverage and
an independently reconstructed terminal remain required before a
distinct actual-LR handoff. Precision CUDA placement, full-budget
waves, shared large-window comparisons and verified private SwanLab
uploads remain separate next steps. Wave 12 is still the latest
complete analyzed/uploaded wave, and better quality and efficiency
than fairly tuned MuonT remain unestablished.

November 19 21:09 launch milestone: independent archived seal checking
passes before freeze. The new binding at
`PVC/run-preparation/ns-training-path-scan-exit-20261119-binding.json`
records freeze at 21:02:50; the issued controller's `--phase check`
passes. Controller acknowledgment at 21:07:10 and direct validator
acknowledgment at 21:08:36 are recorded in
`PVC/controllers/ns-training-path-scan-exit-20261119/controller.json`
and `PVC/run-preparation/ns-training-path-scan-exit-20261119-state.json`.
The controller holds the original node/allocation-manager locks and
records exact predecessor keeper release complete. The first of
54 commands runs continuous24 for
`muont_ns_tpu_q5_full_lr_lower_0p5m_seed0.json`, retaining the original
schedule and numerical gates. Fresh process inspection at 21:09
finds both original owners live and the old keeper absent.

`tracking/ns_training_path_scan_exit_reporting.py` binds the accepted
CPU receipt, preservation seal, frozen binding and acknowledged
controller by digest. The original Bash monitor publishes both journals
at 21:07:43 and keeps its 900-second sleep. Its next cycle will
include the distinct report. Five focused report/split/concurrent-edit
regressions pass in 23.32 seconds; scoped source/content lint passes,
with the inherited journal-size exception unchanged. The first local
test command fails before collection on a missing guard module path;
the corrected guarded run above is separate. A new read-only terminal
waiter is being CPU-archived against this launch; it is not yet accepted
or started. Complete diagnostic results, independent terminal,
actual-LR handoff, precision CUDA work, full-budget experiments and
verified SwanLab uploads remain pending.

November 19 21:19 waiter and monitoring milestone: the separate archive
`PVC/run-preparation/ns-training-path-scan-exit-followup-cpu/20261119T2110/receipt.json`
accepts 24 cases in 4.51 seconds at 21:09:11 and retains all 178
issued helpers unchanged, adding only the waiter and its test.
Fresh archived checking passes before acknowledgment at 21:11:58
as `3412036/942807401`. Its saved 21:12:03 observation finds both
original owners exact/live and waits; no terminal child is issued.
The waiter checks every 900 seconds, invokes only the original
canonical verifier after complete coverage and clean restoration,
and retains rejected cases and failed one-shot evidence.

The aggregate at 21:17:00 records 2/54 commands and 0/18 decisions.
The first continuous24 receipt records execution/cleanup 0/0 at
21:11:08; the interrupted22 command is also admitted. Resumed24
is active. The eight training ranks occupy the original A100 UUIDs
at 21:16. These are bounded correctness trajectories; no shared
large training window, full-budget loss comparison or cloud history
is available from this diagnostic.

The new saved-waiter report is included in both journals. Manual
refresh completes at 21:18 with report time 21:14:07. It preserves
an explicit diagnostic-evidence refusal because mutable receipts
change during inspection; coherent progress is retried by the sole
existing Bash monitor. The waiter projection authenticates its
archive, acknowledged generation and saved observation. The original
monitor remains live, holds its exclusive lock and sleeps 900 seconds.
The 328-case integrated journal suite passes in 151.63 seconds;
six final scan-exit/waiter/split/concurrent regressions pass in
29.77 seconds. The final integrated rerun passes 329 cases in
161.70 seconds by 21:20. Scoped source/content lint and Bash syntax pass.
Full journal lint reports only the inherited size limit, detailed
in DETAILS; evidence and historical results are not removed.

Next gates remain full eighteen-case coverage and independent
terminal reconstruction, then a distinct actual-LR handoff retaining
all rejections. Precision CUDA placement and full-budget studies
still require their own accepted execution gates. Completed waves
will continue to receive shared large-window analyses and separately
verified full-resolution private SwanLab uploads. No optimizer
quality/efficiency improvement over fairly tuned MuonT is established.

November 19 21:21 final publication verification: both journals retain
the accepted diagnostic, distinct acknowledged waiter and dated
two-command/zero-decision observation. The final 329-case suite is
recorded at
`PVC/run-preparation/test-endeavor-scan-exit-final-integrated-20261119T2117.xml`.
Source/content lint and `git diff --check` pass. Controller, direct
validator, terminal waiter and sole Bash monitor remain live; the
monitor's original sleep child remains in its 900-second interval.
No required foreground check/test session remains. Complete
diagnostic decisions, successor experiments, large-window outcomes
and verified new SwanLab histories remain open.

November 19 21:35 diagnostic and handoff milestone: the aggregate at
21:31:55 records 3/54 completed commands and 1/18 case decisions.
`muont_ns_tpu_q5_full_lr_lower` accepts at matrix LR `0.04525483392`
on continuous24/interrupted22/resumed24 with the original 5,120-update
schedule. All five protocol gates pass, as do all 39 scalar-loss
comparisons. Tensor comparison failures are zero across all eight
ranks at updates 22/23/24. The first-resumed update23 checkpoint
retains its recorded `cached_root_graph_active` scalar differences
for groups `d` and `f`; this endpoint remains diagnostic-only under
the unchanged acceptance protocol. No numerical tolerance, compilation
or CUDA-graph requirement is relaxed. Supporting aggregate:
`PVC/run-preparation/validation-ns-training-path-scan-exit-20261119-gpu.json`.
This is a bounded correctness result, not a full-budget loss comparison
or a quality/efficiency outcome.

The distinct reader namespace
`ns-training-path-scan-exit-actual-lr-predecessor-20261119`
is independently CPU/archive accepted at 21:32:26:
`PVC/run-preparation/ns-training-path-scan-exit-actual-lr-predecessor-cpu/staged-20261119T2132/receipt.json`.
All 34 archived cases pass, with 193 authenticated helpers. The
closure retains the 180-helper issued waiter archive, pinned reporter
parents, both prior predecessor implementations/preparers/tests and
three new reader/preparer/test files. Staging and acceptance are
separate phases; fresh archived-origin checking passes. Tests exercise
live/zombie/incomplete owners, real reaped CPU children, incomplete or
rebound terminal evidence, unchanged failed checkpoint identities,
one-shot publication and changed-after-test source refusal. All four
failed coverage summaries are authenticated: original44/14, recovery2/0,
exit-drain6/2 and transition23/7. Rejected decisions remain retained.
No tensor payload is freshly read or hashed by the predecessor.

The numerical and execution archives have distinct
`owned_output_handoff.py` identities. The source plan records this
conflict rather than merging or replacing either accepted archive.
The read-only observation at 21:33:45 publishes
`PVC/run-preparation/ns-training-path-scan-exit-actual-lr-readiness-20261119T2134.json`
with `ready: false` and `waiting_for_scan_exit_owners`. Original
controller, validator and terminal waiter are exact/live; all three
completed command parents are absent. No canonical handoff seal,
execution-owner/controller binding, allocation transfer or successor
GPU command is issued. Full54-command/18-decision coverage, exact
restoration and the existing independent waiter terminal remain
required before sealing.

The original Bash monitor successfully publishes a coherent
three-command/zero-decision projection at 21:26:38 and resumes
`sleep 900` with its exclusive lock. The waiter records its second
observation at 21:27:04. Fresh 21:35 inspection keeps all three
original owners and the sole monitor live. Both journals are archived
under the common update lock before this milestone. Required scoped
source lint and `git diff --check` pass. The broader inherited handoff
and journal suite finishes with 422 passes in 191.77 seconds:
`PVC/run-preparation/test-ns-scan-exit-lr-predecessor-integrated-20261119T2134.xml`.
All source/content lint checks pass; full document lint reports only
the retained 500KiB journal-size rule. Final verification is published
November 19 21:38 under the shared lock, preserving the accepted source
and all prior evidence. Wave 12 remains the latest
full-budget analyzed and privately SwanLab-verified wave. Actual-LR
gates, precision CUDA placement, new full-budget experiments, shared
large-window analysis and verified full-resolution uploads remain open.

November 19 21:39 progress update: the 21:36:56 aggregate admits
4/54 commands and retains the one accepted five-step MuonT control.
The fourth command is `muont_ns_tpu_q7_full_lr_lower` continuous24;
its execution receipt at 21:34:29 records integer-zero execution
and cleanup with marked descendants absent. Interrupted/resumed
seven-step trajectories and their case decision remain pending.
Original controller, validator, independent waiter and sole Bash
monitor are live at 21:38; the monitor retains its 900-second sleep.
No handoff seal, successor GPU launch, new full-budget result or
cloud upload is inferred. Both journals are archived under the
shared lock before this update; accepted sources remain unchanged.

November 19 21:55 actual-LR adapter milestone: the distinct candidate
`PVC/candidates/ns-training-path-scan-exit-actual-lr-adapter-20261119T2144`
is prepared from the accepted numerical adapter. Its CPU archive at
`PVC/run-preparation/ns-training-path-scan-exit-actual-lr-adapter-cpu/20261119T2146/receipt.json`
accepts all16tests at21:53:00, with185helpers and559inventory entries.
The local corrected suite passes16cases in354.68seconds; the archived
suite passes16cases in359.15seconds. Fresh archived-origin checking
passes. The original trainer, NS numerical readers, all54actual-LR
configurations,162continuous/interrupted/resumed commands, original
5120-update schedule, tolerances, compilation and cached-root graphs
are unchanged.

The sole changed inherited helper is `owned_output_handoff.py`, now
the separately accepted scan-exit version. The original numerical
version is retained as a fixture. The source closure includes the
accepted missing-environment handler and transitive pinned derivation
parents; the conflicting numerical and canonical-terminal archives
are not merged. Real unowned trainer/validator invocations refuse
before outputs. Rehashed training/configuration/metadata changes,
changed parent sources and incomplete archive evidence remain
refusals. The first local test run retains16fixture errors from an
incorrect copied handler pin; it supplies no acceptance. The corrected
source uses the independently checked accepted handler digest.

Distinct execution-owner/controller/test sources are now implemented
and bound to this accepted adapter receipt. Their CPU archive is
running and supplies no accepted execution or GPU authority yet.
The new queue sources are distinct from all consumed predecessors;
87selected real lifecycle tests pass after correcting an inherited
sealer-filename reference. Five complete-archive tests remain deferred
until the required execution receipt exists, and are not counted as
covered. Earlier failures remain retained. The queue is unbound to an
execution acceptance, unarchived and unlaunched. No terminal seal or
successor GPU binding is created.

The running diagnostic's21:44:38aggregate retains6/54commands,
1/18decisions: one accepted five-step MuonT control and pending
seven-step comparison. Controller, validator and independent waiter
are live at21:54. The waiter records its third observation21:42:05;
the original Bash monitor publishes21:45:38 and resumes `sleep900`.
Both journals are archived under the common lock before this milestone.
All new source lint passes. Full coverage/independent terminal,
accepted actual-LR execution and queue, precision CUDA placement,
new full-budget waves, shared large-window comparisons and verified
private full-resolution SwanLab uploads remain required. No optimizer
superiority or goal completion is established.

November 19 22:05 execution and control milestone: the distinct actual-LR
execution owner is CPU/archive accepted at 21:59:22, with 53 passing tests,
203 helpers and 577 inventory entries. Fresh checking from its archived
tracking directory passes at 22:01. Accepted receipt:
`PVC/run-preparation/ns-training-path-scan-exit-actual-lr-execution-cpu/20261119T2154/receipt.json`.
The accepted adapter candidate, all 54 configurations and 162 bounded
commands, original 5,120-update schedule, numerical readers, tolerances,
compilation and cached-root graphs remain unchanged. The new queue pins
this execution receipt. All 47 local queue cases pass in 38.06 seconds,
including the five complete-archive tests previously deferred. Its separate
queue CPU/archive suite is running; no acceptance or launch is inferred.

The scan-exit aggregate at 22:03:52 records 8/54 completed commands and
2/18 decisions. Both `muont_ns_tpu_q5_full_lr_lower` and
`muont_ns_tpu_q7_full_lr_lower` pass the five existing gates at matrix LR
`0.04525483392`. Each has 39 passing scalar-loss comparisons. The seven-step
control has 2,123/2,161/2,163 tensor comparisons at updates 22/23/24, with
zero failed tensors. Its 16 first-resumed graph-state scalar differences
retain the unchanged diagnostic-only policy. The ten-step control has
completed continuous24 and interrupted22 commands; no case decision yet.
These bounded continuation results establish no full-budget loss or
whole-training efficiency result.

At 22:04 the original controller, direct validator and independent waiter
remain live. The sole Bash journal monitor publishes both documents at
22:04:32 and resumes its 900-second sleep. The new queue reporter is
included in both automated blocks; the missing import found by required
lint is fixed. Both journals are archived under the common update lock
before this milestone. Complete diagnostic coverage and independent
terminal, accepted queue launch, actual-LR gates, precision CUDA placement,
new full-budget waves, shared large-window comparisons and verified private
full-history SwanLab uploads remain required. Wave 12 remains the latest
full-budget analyzed and privately cloud-verified wave.

November 19 22:15 failure and queue-acceptance milestone: the scan-exit
diagnostic stops at 8/54 completed commands and 2/18 accepted control
decisions. The next `muont_ns_tpu_q10_full_lr_lower` resumed command fails
at 22:03:52 before training in `complete_scan_exhausted`. Its saved
execution/cleanup exits are 1/1, with marked descendants absent.
The ownership failure records previously confirmed marked child3517243,
parent3516734, exact start tick943118652 and a retained live command
digest. Later reads find its files absent. Those later reads cannot
establish the cause or coherent identity at the original failure.
The validator's generic command/generation/cleanup/log error supplies no
evidence of source or log substitution. No ten-step numerical decision
or full-budget result follows.

The controller stops `needs_attention` at 22:04:12 and records zero-exit
keeper restoration. The independent terminal waiter stops
`needs_attention` at 22:12:06; neither one-shot is restarted.
Read-only verification at 22:12:53 confirms controller, validator, waiter,
failed launcher/ranks and retained marked child absent, no validator
markers, and exact keeper3517296 with eight ranks exclusively occupying
the original eight A100 UUIDs. Receipt:
`PVC/run-preparation/endeavor-scan-exit-failure-restoration-observation-20261119T2213.json`.
This verifies current restoration without changing the retained failed
cleanup exits or incomplete diagnostic coverage.

The separate actual-LR queue is CPU/archive accepted at 22:05:07:
47 passing cases, 214 helpers and 588 inventory entries. Fresh archived
checking passes. Receipt:
`PVC/run-preparation/ns-training-path-scan-exit-actual-lr-queue-cpu/20261119T2202/receipt.json`.
It preserves the accepted execution/adapter archives and 54 configurations/
162 commands. Its reporter now binds the receipt; the final integrated
journal/reporting suite passes 267 cases in 93.94 seconds, including a
both-document publication/refusal regression. The queue remains unlaunched
because its original complete-terminal requirement cannot be met.
The accepted source, failed queues and all consumed claims remain untouched.

Next ownership hypothesis, planned and unlaunched: the scan's recheck
branches for `current is None` and terminal state currently only mark the
scan unstable. Test a distinct successor that requires process-directory
absence or the existing identity-bound whole-process exit proof there,
then repeats the complete scan. Keep PID reuse, surviving threads,
live drift and incomplete absence as refusals. First reproduce the saved
failure in CPU fixtures; then require a new source archive, preservation
seal and all original distributed/numerical gates before GPU launch.
The actual-LR and precision-placement numerical experiments remain pending
behind those ownership gates.

Both journals are archived under the common lock before this milestone.
An extra manual renderer refresh is stopped after a shared-filesystem open
stalls while holding the update lock; it publishes no new report. The sole
original Bash monitor remains live in its 900-second sleep and will retry.
Large-window results and verified full-history SwanLab uploads will be added
here when a new full-budget wave completes. Wave 12 remains the latest such
wave, and no optimizer superiority or goal completion is established.

### NS Step-Budget Quality-Cost Study

CPU/source-archive acceptance November 18 23:00:03; measurement completes
at 23:01:47. Question: can fewer iterations of Toner's pinned quintic map
reduce optimizer work without losing the useful output-normalized direction?
This is a new numerical-arm hypothesis, motivated by the observed ten-step
amplification and the preserved upstream Muon source's five-step default.
The latter has different BF16 normalization/accumulation and `eps=1e-7`;
this study retains Toner's FP32 normalization, `eps=1e-5` and coefficients
`(3.4445, -4.7750, 2.0315)`. It is not an upstream Muon reproduction.

The archived CPU runner `tracking/ns_step_budget.py` evaluates
`q = 3,5,7,10` with FP32 and single-BF16-operand emulation, eight CPU threads,
and the same authenticated full historical rank0/update23/occurrence0
`[16,512,512]` continuous/resumed pair in `C,R,C,R` order.
Every output is retained. Independent FP64 SVD gives scalar singular-value
polynomials at each budget, then reconstructs the finite matrix map.
The partial polar sets numerically null singular directions to zero using
`sigma > eps_fp64 * max(rows,columns) * sigma_max`. All 32 saved matrices
are rank 512 under this convention; the batch-wide smallest singular value
is `2.25745e-10` for continuous and `1.91641e-10` for resumed.
This is a diagnostic geometric reference, with no
assumption that exact polar is the best training update.

| Steps | Nominal Products | FP32 C/R Error | BF16-Emulated C/R Error | BF16 Pair Tolerance Failures | Mean BF16 Continuous Cosine to Partial Polar | Mean BF16 Raw Relative Polar Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 9 | `3.06917e-6` | `0.000468647` | 3,501,364 | `0.774263` | `0.652564` |
| 5 | 15 | `1.00113e-5` | `0.00926950` | 4,178,320 | `0.951345` | `0.309694` |
| 7 | 21 | `1.95859e-5` | `0.0216559` | 4,188,756 | `0.980342` | `0.211339` |
| 10 | 30 | `3.73549e-5` | `0.0368395` | 4,191,356 | `0.979295` | `0.211032` |

Every C/R pair, including all four independent FP64 finite-map pairs,
fails `rtol=1e-5`, `atol=1e-6`. FP64 relative errors are
`3.04642e-6 / 9.88145e-6 / 1.93002e-5 / 3.66208e-5`, with
583/80,437/380,135/789,618 pointwise failures. All identical-input repeats
are exact finite. FP32 three/five/seven-step outputs pass their own
SVD-polynomial reference; ten-step FP32 and every BF16-emulated output fail.
Five/seven-step BF16 directions differ from their ten-step controls by
relative errors `0.411688 / 0.375523`; the change cannot be labeled an
implementation-equivalent speedup. Norm-adjusted geometric metrics are
reported separately and do not graft the executed output.

Twenty-six tests pass in 4.17 seconds; the same archived tests pass in
5.32 seconds. The integrated saved-replay/owner suite passes 78 cases in
44.94 seconds. Source/archive receipt:
`PVC/run-preparation/ns-step-budget-cpu/20261118T2300/receipt.json`.
Saved study and separate retained-output verification:
`PVC/run-preparation/ns-step-budget-saved-pair-20261118T2300/result.json`
and `retained-verification.json`. At 23:02:51 the latter rederives all
output hashes, pair/repeat/own-FP64/ten-step comparisons; it does not rerun
the SVD arithmetic or copied geometry summary.

Conclusion: five/seven steps remain worth testing as quality-cost arms,
while three steps discard more direction geometry. No step budget passes
the saved continuation tolerance; fewer steps alone are not accepted as a
continuation correction. Nominal product reductions of 50%/30% for five/
seven steps say nothing yet about full-training speed or loss.

The next proposed CUDA diagnostic compares unchanged ten-step control
with five/seven-step maps and FP32 controls on full saved sources, in two
fresh opposite-order processes. Require compiled call, capture, replay and
repeat output evidence, exact input/RNG preservation, original saved-output
reproduction for ten-step controls, independent reference comparisons,
complete raw timing samples and exact cleanup/restoration.
`tracking/ns_budget_trial.py` implements the graph-retention/timing component;
its report/drift/CPU-denial coverage brings the focused CPU suite to 28
passes in 4.30 seconds. It has no standalone CUDA entry point or allocation
owner. Original rational diagnostic completion and its independent terminal
restoration precede any separate owner acceptance, binding or GPU issuance.
Compilation and graphs stay enabled; no original gate or tolerance changes.
After a GPU diagnostic, each retained numerical arm needs its own full-model
continuation gates and fair LR-tuned quality-cost comparison against MuonT,
with large shared-token windows and verified private SwanLab histories.

At 23:10:53, the revised graph component receives separate six-file
CPU/source-archive acceptance with 29 tests in 5.84 seconds; its mutable
focused run passes in 4.14 seconds. A final retained snapshot after all
50 timed graph replays exposes post-timing drift and nonfinite output
without hiding those failures. The complete compiled-call/capture-executed/
replay/repeat/timed-final reports retain every comparison and timing sample.
The integrated replay/owner suite passes 81 cases in 42.84 seconds;
scoped code/journal lint and `git diff --check` pass.
Receipt:
`PVC/run-preparation/ns-budget-trial-cpu/20261118T2310/receipt.json`.
This archive contains a numerical trial component and CPU tests; no
source-bound CUDA worker/controller, accepted owner, freeze or execution
receipt is issued. The measured CPU study archive remains unchanged.
The next launch work is a separate accepted owner and worker after original
rational analysis and exact terminal restoration, followed by the proposed
GPU controls and full-model gates. No optimizer quality or cloud history
is inferred from this acceptance.

At 23:11:26, the exact original rational controller/validator remain live,
without active training ranks or CUDA occupancy; complete analysis is still
absent. The scheduled Bash monitor captures at 23:10:54 and refreshes both
documents at 23:13:16. Its original monitor lock is preserved; no duplicate
monitor, competing verifier or GPU task is issued. Separate terminal
follow-up remains `waiting_for_original_owners` at 23:07:26.

At 23:21:44, a fresh source-bound read-only observation again verifies the
original controller and validator generations, all twelve retained commands,
and the original eight physical A100 UUIDs. No active training launcher,
rank or GPU compute process is observed. The result remains `diagnosing`
with no published `analysis` or error. The validator continues reading
retained pinned/trace evidence; this observation supplies no completion
estimate or numerical verdict. Receipt:
`PVC/run-preparation/monitor-rational-continuation/20261118T152143761291.json`.
The single Bash monitor remains live with its original monitor lock, and
the separate terminal waiter remains live. Their earlier saved receipts
are historical observations. Five/seven-step CUDA diagnostics and proposed
production waves remain unlaunched. All future wave entries continue to
distinguish launch, large shared-token-window analysis and verified
full-resolution private SwanLab import.

The one-shot split-document refresh publishes at 23:25:57 from observations
beginning 23:23:39. Bash hash comparison confirms the manual scientific
record outside the recurring block is byte-preserved. Four focused
split-document/900-second-monitor regression tests pass with CUDA hidden;
the journal's scoped lint and `git diff --check` pass. The original
terminal waiter records exact diagnostic owners still live at 23:22:26.
No completed rational analysis or new GPU launch follows.

At 23:39:07, `tracking/ns_budget_execution.py` and
`tracking/prepare_ns_budget_execution.py` receive separate source/archive
acceptance. The numerical component declares pinned `tpu` BF16-operand
and `f32` arms at five/seven/ten steps. Each fresh process uses all six
arms and four complete saved-source calls; process 1 reverses both arm
and input order. The 24 calls retain 144 independent snapshots across
pinned-eager/compiled-call/capture-executed/replay/repeat/timed-final phases.
Each graph call records all 50 CUDA-event intervals, input values/layout
and RNG preservation. Reconstruction rederives all historical ten-step,
compiled/eager, same-input, cross-process, FP64 finite-map and timing
comparisons; failed comparisons are retained. Changed step counts are
separate numerical arms, and graph timing excludes compilation,
capture, transfers and full training.

Mutable tests pass 45/45 in 6.88 seconds; archived tests pass 45/45 in
9.21 seconds. A fresh archived checker verifies the receipt, complete
helper bytes and two pinned fixtures. The integrated saved-replay/owner
suite passes 97 cases in 45.66 seconds. An initial 44-pass/one-failure
test exposes timing-list aliasing; independent lifecycle snapshots correct
that issue. Preparation denials for an inherited `experiment` test import
and attempted Python parsing of a preserved Bash helper occur before
output-directory creation and are recorded in `DETAILS.md`.
All earlier study, graph and issued rational archives remain unchanged.
Receipt:
`PVC/run-preparation/ns-budget-execution-cpu/20261118T2340/receipt.json`.
This accepts numerical execution/reconstruction code only; no source-bound
CUDA worker/controller, accepted allocation owner, binding, GPU execution
or SwanLab optimizer history exists.

At 23:41:03, read-only observation verifies the original rational
controller/validator and eight physical A100 UUIDs; no active training
rank or GPU compute process is observed. Complete analysis is still absent.
The separate original terminal waiter remains waiting at 23:37:26, and
the single Bash monitor's latest scheduled publication is 23:30:39.
The next step remains separate NS owner/worker acceptance after original
rational analysis and independently verified terminal restoration.
Full-model gates, fair LR comparisons, large shared-token-window loss
analysis and verified private imports are still required for production.

### NS Budget Diagnostic Owner Preparation

CPU/source-archive acceptance November 19 00:03:08; no GPU issuance.
The separate `ns_budget_gpu_owner.py`, `ns_budget_gpu_controller.py` and
`ns_budget_gpu_worker.py` preserve every accepted numerical execution helper.
The archive contains 122 helpers and two pinned fixtures.
Seventy-five archived tests pass in 24.71 seconds; a fresh archived process
verifies the complete source, fixture, XML and receipt identities.
The mutable suite passes 75 tests in 21.30 seconds; the broader saved-replay/
owner suite passes 122 tests in 57.97 seconds. Receipt:
`PVC/run-preparation/ns-budget-gpu-owner-cpu/20261119T0003/receipt.json`.
Earlier archives remain unchanged.

The proposed execution uses two serial fresh processes on one UUID-bound
A100 while the controller owns the exact eight-device allocation.
Each process executes all six five/seven/ten-step BF16-operand/FP32 arms
and retains all eager, compiled, capture, replay, repeat and timed-final
outputs. Process 1 reverses arm and source order. Fifty CUDA-event intervals
per call remain raw evidence; timing excludes compilation, capture,
transfers and full training. Historical interpretation requires reproduced
ten-step saved outputs, compiled/eager and graph parity, exact finite
same-input repeats and cross-process agreement. Failed numerical comparisons
are retained, including a false historical-interpretation predicate.

Before freeze or keeper transfer, the owner requires the original rational
controller/validator absent, all 12 integer-zero outcomes and bound logs,
completed analysis, independent terminal reconstruction, completed/absent
original waiter, and the exact restored keeper generation and physical
mapping. A negative rational intervention result can support a separately
owned diagnostic; it supplies no continuation or production acceptance.
Workers require acknowledged direct-child generations, exclusive node/
manager descriptors, exact UUID mapping and no competing occupancy.
Real subprocess tests cover nonzero/spawn failures, detached descendants,
cleanup and restoration. Terminal tests reconstruct retained outputs and
reject live owners, altered ancestry, reports, comparisons or keeper identity.

CPU tests exercise lock ownership on local temporary files. A separate
read-only probe of the live shared-filesystem monitor descriptor finds
`FLOCK ADVISORY WRITE 0` in `fdinfo`; the inherited strict PID-bound
GPU verifier rejects that descriptor. Independent nonblocking contention
confirms the monitor lock remains held. This observation establishes a
filesystem compatibility question for actual node/manager locks, not GPU
owner acceptance. The accepted archive is preserved; no PID ownership
check or numerical tolerance is relaxed.
At 00:09:55, a separate read-only probe authenticates original controller
`1643665/934354200` and accepts its actual node descriptor 3 and manager
descriptor 4 with the unchanged archived verifier. Both `fdinfo` rows name
PID 1643665; independent contention exits 1 for each lock.
The monitor-descriptor anomaly therefore does not reproduce on these
actual allocation descriptors. Receipt:
`PVC/run-preparation/ns-budget-lock-compatibility-20261119T0010.json`.
This verifies current predecessor locks only; a future NS worker must
independently verify its own controller's exact descriptors.

At November 19 00:09:55, the original rational controller and validator
still authenticate live. All 12 outcomes and trajectories are retained,
with no published `analysis` or error and no active training rank or GPU
compute process. The single Bash monitor publishes both documents at
00:05:31. The original CUDA-hidden terminal waiter remains live and waiting.
The NS binding, execution state/directory, queue and launch remain absent.
The waiter saves another exact-owner waiting observation at 00:07:27.
After original terminal/restoration and successor-time ownership are verified,
the separate diagnostic can proceed. Any later numerical successor still
requires full-model gates, fair LR comparisons, large shared-token windows
and verified private SwanLab histories before an optimizer claim.

### NS Budget Diagnostic Queue And Reporting

The separate one-shot queue receives CPU/source-archive acceptance
November 19 00:24:32. All 122 owner helpers and two pinned fixtures are
byte-preserved; `ns_budget_queue.py` and its tests bring the inventory
to 124 helpers. Twenty-one archived tests pass in 11.00 seconds,
and fresh archived checking passes with the fixed CUDA-hidden environment.
The integrated queue/owner/replay/rational-follow-up suite passes
104 cases in 28.78 seconds. Receipt:
`PVC/run-preparation/ns-budget-queue-cpu/20261119T0024/receipt.json`.

Original queue `1745559/935328583` acknowledges at 00:25:30.
A fresh 00:41:37 command/generation/marker/environment check finds it live.
Its saved 00:40:44 state is `waiting_for_original_rational_terminal`,
with zero completed phases. Work and original intent/state/log receipts
are under `PVC/run-preparation/ns-budget-queue-20261119`.
The queue polls every 900 seconds without performing keeper operations.
It requires exact original rational controller/validator/waiter absence,
completed rational analysis and independently reconstructed terminal
restoration before invoking accepted owner `freeze`, issued `check`
and issued `start` once. After actual issued controller/worker absence,
it invokes issued `terminal` once, with a 14,400-second reconstruction
timeout. Failed or partial launches are preserved and never retried.

The diagnostic, if those gates pass, remains six BF16-operand/
FP32 five/seven/ten-step arms in two serial fresh opposite-order processes.
It is not a production wave. Independent terminal evidence will be written
to `PVC/run-preparation/ns-budget-terminal-independent-20261119.json`.
No NS execution binding or GPU worker exists at this checkpoint.
The negative rational intervention does not waive original terminal,
ownership or NS numerical-reconstruction requirements.

The existing split-document reporter now authenticates queue/source/archive/
intent/phase-log and eventual diagnostic/terminal identities, preserving
failures and false predicates. Completed records add per-arm source-pair
gate failures, relative-error ranges, CUDA-event replay median ranges and
same-process/source/call-index q10-to-arm timing ratios. These are kernel
measurements excluding compilation, capture, transfers and training;
they do not establish whole-training efficiency. Reports do not execute
tensors or infer current process liveness from saved receipts.

The broader journal/queue/owner suite passes 286 cases in 53.03 seconds.
Final schema/reporting/split-document/900-second-monitor regressions pass
13 cases in 3.15 seconds. Required scoped code lint and `git diff --check`
pass. The single original Bash monitor captures at 00:38:18 and publishes
both documents at 00:40:43, including the rational rejection and queued
NS status. No duplicate monitor or competing GPU work is started.
Full-model gates, fair LR tuning, full-budget large shared-token-window
comparisons and verified private SwanLab histories remain required;
no new optimizer quality, efficiency or cloud history is claimed.

An independent guarded refresh completes at November 19 00:46:54 from the
ENDEAVOR entry point. It verifies both manual records byte-preserved and
the same NS queue/negative rational report present in both automated
blocks. Receipt:
`PVC/run-preparation/journal-ns-budget-refresh-independent-20261119T0044.json`.
Final transition-independent reporting tests pass 13/13 in 3.12 seconds;
the accepted GPU/queue archives remain unchanged. ENDEAVOR/code lint
passes. DETAILS alone exceeds the preserved file-size limit; its evidence
history is retained. The original rational reconstruction and waiting queue
remain live, so no terminal or production acceptance is inferred.

### NS Budget CUDA Diagnostic Results

November 19 05:50, Asia/Shanghai: the original queued six-arm diagnostic
has completed two fresh opposite-order CUDA processes on one UUID-bound
A100. Both worker execution and cleanup exits are integer zero.
The original controller independently reconstructs their retained outputs
at 05:37:38, records `diagnostic_complete` and keeper restoration at
05:37:54, then exits. At 05:38:37 the original controller and both workers
are absent. A physical query by 05:39 observes eight non-training keeper
processes on the original A100 UUIDs. Independent terminal verification
by the original queue starts at 05:41:28 and completes at 05:47:02,
accepting exact restoration while retaining the failed historical controls.
The queue completes all 4/4 phases at 05:47:10 with integer-zero execution
and cleanup; process inspection at 05:49:38 finds both queue and terminal
verifier absent. Both successor CPU waiters and the monitor remain live.

Result:
`PVC/diagnostics/ns-budget-20261118/comparison.json`.
Execution binding and state:
`PVC/run-preparation/ns-budget-binding-20261118.json` and
`PVC/run-preparation/ns-budget-state-20261118.json`.
Independent terminal:
`PVC/run-preparation/ns-budget-terminal-independent-20261119.json`.
The unchanged source, coefficients, original saved rank-0/update-23 input
pair, full `(16,512,512)` batches, exact commands, hardware mapping and
receipt identities are retained in `DETAILS.md`.
The two inputs pass the original tolerance with relative Frobenius error
`1.4512877e-6`.

All 144 corresponding cross-process outputs, 144 same-input repeats and
192 graph/capture/replay comparisons against the compiled call are exact
finite. The ten-step BF16 pinned-eager phase reproduces historical outputs
exactly in all eight source/call/process comparisons. Its compiled call
and graph phases fail all 40 corresponding comparisons against pinned
eager, with relative error `0.0360516-0.0360976`. Thus the saved aggregate
`historical_controls_interpretable` remains false. Every arm's compiled
output differs from eager exactly; FP32 five/seven-step differences pass
the original tolerance, while ten-step FP32 fails it in all 40 comparisons.
These outputs do not locate the first differing operation.

| Operand Mode | Steps | Pair Failures / Checks | Pair Relative Error | Replay Median ms | Matched Ten-Step / Arm Time |
| --- | ---: | ---: | --- | --- | --- |
| BF16 operands (`tpu`) | 10 | 24/24 | 0.0369231-0.0369345 | 2.10763-2.40267 | 1 |
| BF16 operands (`tpu`) | 7 | 24/24 | 0.0217187-0.0217451 | 1.49274-1.70755 | 1.23477-1.52866 |
| BF16 operands (`tpu`) | 5 | 24/24 | 0.00927204-0.00927247 | 1.05003-1.19506 | 1.77258-2.28819 |
| FP32 | 10 | 24/24 | 3.71149e-5-3.73519e-5 | 9.35557-9.36211 | 1 |
| FP32 | 7 | 24/24 | 1.98406e-5-1.98426e-5 | 6.56158-6.56678 | 1.42494-1.42681 |
| FP32 | 5 | 24/24 | 1.01491e-5-1.01541e-5 | 4.69971-5.95754 | 1.57054-1.99127 |

Pair checks compare continuous and resumed sources across all retained
phases/repeats in both processes. Timing ranges cover per-call medians
of 50 CUDA-event replays, not confidence intervals. Ratios match operand
mode, process, source and call index. Compilation, graph capture,
transfers and full training are excluded. Fewer steps lower measured
kernel cost and pair sensitivity, but none passes the original pair gate.
No whole-training efficiency or optimizer-quality result follows.

The existing lower-LR full-model waiter requires interpretable historical
controls. Independent terminal completion cannot turn that false predicate
true. No lower-LR, actual-LR or production GPU command is authorized by this
result, and no new SwanLab history is uploaded.

The source audit below separates a training-path follow-up from compiled
normalization/recurrence isolation. At the completed diagnostic checkpoint,
both are proposed and have no accepted source archive, allocation owner or
launch. Later evidence-helper and numerical-component progress is recorded
below. A corrected execution path still needs source-bound full-model and
per-LR gates, then fair full-budget large-window/SwanLab comparisons.

### NS Compiler Boundary Audit and Follow-Up

November 19 05:59, Asia/Shanghai: a read-only audit of the accepted NS
configuration candidate and both execution-owner candidates finds the same
training compiler boundaries in all three. Six files per candidate match
their accepted receipt hashes: `experiment/train.py`, `optimizer.py`,
`block_toner.py`, `refresh_toner.py`, and pinned `toner/fast/muon.py` and
`engine.py`. The candidate and receipt paths and file identities are in
`DETAILS.md`.

`train.py:183-196` compiles `model.body` and the loss with
`runtime["compile_options"]`; `train.py:338` calls each optimizer's
`step()` separately. The MuonT path calls undecorated
`MU.newton_schulz_batched` from pinned `engine.py:833`, between compiled
input/output product helpers. The output-normalization and matched-null
paths call the same undecorated routine from `block_toner.py:356`.
`RefreshTonerCore` changes host-basis installation and does not wrap NS
compilation. Here a root is an inverse fractional power of a moment matrix
used as a preconditioner; its cached CUDA graph executes before NS.
Root compilation and cached-root CUDA graphs remain enabled.
The optional continuation-trace wrapper calls the original matrix update
without compiling it. This is an audit of source-defined paths in
unlaunched candidates, not an observation of running training.

The completed diagnostic instead constructs
`torch.compile(partial(newton_schulz_batched, ...), dynamic=False)` in
`tracking/ns_budget_execution.py:365-368`. Compiled-call and graph outputs
are exact with each other but disagree with pinned eager outputs. This
measures an additional NS execution boundary. The saved eager
ten-step BF16 phase reproduces historical outputs in 8/8 comparisons.
The historical continuous/resumed input pair still produces a rejected
output pair; this audit does not correct that numerical sensitivity or
the failed full-model continuation. It also does not explain the first
differing operation inside compiled NS.

Two distinct experiments could change the next decision:

1. Training-path fidelity and step budget. Preserve the accepted training
   compiler boundaries, coefficients, operand modes, original inputs and
   full-budget LR schedule. A distinct source-bound diagnostic should
   authenticate the completed original terminal, owner absence and exact
   keeper transfer while retaining its failed compiled-control predicate.
   Establish historical eager-path reproduction separately. Then run
   source-bound continuous-24/interrupted-22/resumed-24 checks for the
   proposed five/seven/ten-step, BF16/FP32 and MuonT/fused/null cases.
   The fused arm applies output diagonal normalization; its matched null
   computes the same output statistics but applies an identity outer factor.
   Retain all eight ranks, original boundary-22 and post-cached-root
   update-24 tolerances and scalar losses. Check that fused/null backward
   capture preserves the output rows, parameter updates match their
   reference equations, and the null retains the same output-statistic
   work. Historical
   input-pair sensitivity must be reported for every case. This requires
   a separately accepted protocol and owner; the existing failed
   predecessor contract cannot be edited or bypassed.
2. Compiled-map localization. Treat full-NS compilation as a separate
   optimization hypothesis. Preserve its unmodified compiled control,
   then isolate normalization-only and recurrence-only compiler boundaries
   in two fresh opposite-order processes. Retain intermediates and first
   differing BF16 operands. Instrumented outputs must independently match
   the corresponding uninstrumented control before causal interpretation.
   Keep compiled calls, CUDA graphs and original tolerances. Kernel timing
   must remain separate from whole-training cost, and any eventual training
   integration needs new actual-config continuation gates.

Neither numerical experiment is implemented, accepted or launched at this
05:59 checkpoint. Later local evidence-helper work is recorded below.
No existing queue, archive, result, compiler setting or tolerance is changed.
The current journal/reporting regression suite passes 233/233 with CUDA
hidden in 27.98 seconds. Complete wave histories, large shared-token-window
analysis, private SwanLab verification, fair LR tuning and fresh paired
seeds remain required before an optimizer superiority claim.

At 05:59:09 the original lower-LR queue records `needs_attention` before
issuing any phase. Its exact saved error is
`RuntimeError: NS full-model predecessor terminal reconstruction/control/restoration is incomplete`.
The archived predecessor check requires
`historical_controls_interpretable: true`; the independently reconstructed
NS terminal retains `false`. Read-only process inspection at 06:00:54
finds the lower-LR queue absent and the actual-LR queue still live.
The lower-LR terminal is absent, so queue absence does not authorize
actual-LR execution. Both proposed experiments must preserve these failures
and use separately accepted ownership. The existing monitor continues.

The actual-LR queue subsequently records `needs_attention` at 06:07:02:
`ValueError: NS full-model queue source is absent or aliased`, with 0/4
phases. Read-only inspection finds its nested
`ns_full_model_queue.helper_sources` expects
`test_ns_full_model_queue.py` beside the imported queue helper.
That file exists in the original lower-LR archive and is declared in its
receipt, but is absent from both the actual-LR archive and its inventory.
This is a separate frozen-import dependency failure; fixing it would
require a distinct source/archive acceptance. At 06:08:51 both original
queues are absent. No lower-LR terminal, GPU binding or training exists,
and the historical-control gate remains false.

#### Compiled-Map Localization Component CPU Acceptance

November 19 12:00:12, Asia/Shanghai: the normalization-only versus
recurrence-only compiler experiment now has an implemented numerical
component and source-archive acceptance. Twenty-two staged CPU tests
pass in13.11seconds; fresh archived-entry checking passes by12:01.
Accepted receipt:
`PVC/run-preparation/ns-compiler-localization-component-cpu/20261119T1200/receipt.json`.
It inherits the accepted149-helper original-training-path owner archive
unchanged and adds three numerical/preparation/test helpers, giving152
helpers and520 inventory entries. The running candidate and all
accepted archives remain unchanged. Hashes and exact commands are in
`DETAILS.md`.

The component derives its functions from the authenticated pinned NS
abstract syntax tree. The unmodified whole-map eager/compiled controls
remain separate from normalization-only and recurrence-only compilation.
The traced maps retain the normalization input/norm/denominator/output,
orientation, every iteration input/output, and raw/cast operands and
outputs for all three batched products per iteration. Tracing retains
references until the caller snapshots completed outputs; it changes
tensor lifetimes and compiler outputs. Stage interpretation requires
exact finite traced/control output parity for both sides of every
comparison. Nonfinite intermediates or incomplete stage coverage
withhold interpretation. The first changed BF16 operand is reported
separately from the first changed intermediate.

The protocol declares BF16-operand/FP32-accumulation and FP32 ten-step
arms, four compiler boundaries, continuous/resumed repeated inputs,
and call/capture/replay/replay-repeat phases. Each of two opposite-order
fresh processes must retain128 complete phases. CPU retention checks
source values, shape, stride, storage offset and identity without
normalizing the saved layouts. Repeated-input and phase comparisons
are reconstructed from independent CPU snapshots.

CPU tests establish exact source-split/traced parity against the
unmodified FP32 finite map across square, tall and wide synthetic
batches, plus a real TorchDynamo full-graph test using the CPU eager
backend. This backend test does not exercise CUDA/Inductor lowering.
Synthetic BF16 operand records test comparison localization only;
they do not establish actual BF16/FP32-accumulation CUDA transparency.
The first13-pass/three-failure CPU attempt remains retained: the
finite-tree guard imported the wrong helper, and the compiler tried
to read a closed `/proc/self/fd` source name. Corrected maps retain
the authenticated canonical source filename and original numerics.

The CUDA execution/reconstruction component and allocation owner remain
unimplemented and unissued. Next, inherit this accepted component
unchanged, retain both original saved batches and historical eager/
compiled results, and reconstruct all source-pair/repeat/cross-process
comparisons. A separately accepted owner must wait for the current
original and actual-LR diagnostics, canonical terminals and exact
restoration. It must retain compilation and CUDA graphs, verify
trace/control transparency before interpreting operands, and preserve
every rejected predecessor predicate. The component supplies no
GPU launch, correction, optimizer quality, timing or SwanLab result.

The final integrated localization/saved-replay suite passes all67
CPU tests in17.28seconds, with CUDA hidden and original safety/time
guards. Required scoped source/content lint and `git diff --check`
pass; only the previously documented journal-size check is excluded.
The accepted component archive is unchanged. The existing scheduled
publisher refreshes both recurring blocks at12:08:22 without restart.

#### Compiled-Map Localization Execution CPU Acceptance

November 19 12:27:43, Asia/Shanghai: a separate source-bound CUDA
execution and CPU reconstruction component is CPU/source-archive
accepted. All23 staged tests pass in137.92seconds; a fresh process
importing only the accepted archive passes verification by12:29.
Receipt:
`PVC/run-preparation/ns-compiler-localization-execution-cpu/20261119T1225/receipt.json`.
The three new execution/preparation/test helpers inherit all152
accepted localization helpers unchanged, giving155 helpers and523
inventory entries. Neither the current training candidate nor an
accepted predecessor archive is modified.

The executor retains the original full saved continuous/resumed
`[16,512,512]` source batches at rank0/update23/occurrence0 from
`PVC/run-preparation/diagnostic-fused-continuation-gpu-20261118.json`
and its bound trace inventories. Two operand modes, four compiler
boundaries, four repeated-source calls and four graph/call phases
give128 phases per process. Each phase is saved in its own
authenticated tensor file, with at most the current comparison's
traces resident. Reconstruction loads individual files to compare
eager/compiler boundaries, both historical saved outputs, source
pairs, graph phases, same-input repeats and corresponding occurrences
in two opposite-order processes. Missing or reordered phases, aliased
paths, changed files, source/storage mutation and substituted report
arithmetic are rejected. Failed numerical comparisons remain in the
report. Exact finite traced/control parity and finite intermediates
still precede stage interpretation.

Control and instrumented maps use separate CUDA graphs. The intended
call/capture-executed/replay/replay-repeat protocol preserves original
compilation, three executions of each captured graph, UUID-bound
device checks, source layouts and CPU/CUDA RNG. "Capture-executed"
means the snapshot after the first explicit execution of the captured
graph, not an unexecuted capture buffer. CPU tests exercise
file-backed reconstruction and synthetic BF16 operand schemas; no
CUDA/Inductor transparency, physical-device execution or cleanup has
been tested by this acceptance.

The exclusive allocation owner remains unimplemented and unissued.
This is the process responsible for exclusive GPU use and restoration
of the eight-device allocation's keeper processes after cleanup.
It must wait for the
[original training-path diagnostic](#training-path-evidence-preparation)
and [actual-LR queue](#training-path-actual-lr-queue-preparation),
their canonical terminal receipts, exact controller/worker absence
and restored keeper generations on the original physical GPU UUIDs.
No standalone CUDA entry, production run, timing or cloud import is
issued. Future training integration still needs actual-configuration
continuation gates, fair full-budget LR neighborhoods, fresh paired
seeds, large shared-token windows and verified SwanLab histories.

The integrated localization/execution/saved-replay suite passes90/90
CPU tests in144.10seconds, with zero failures/errors/skips and the
original safety/time guards. Its XML and archive hashes are in
`DETAILS.md`. Required source/content lint passes; the unchanged
500KiB journal-size policy remains the sole excluded check.

#### Compiler Localization Handoff Draft

November 19 12:49, Asia/Shanghai: a local
`tracking/ns_compiler_localization_predecessor.py` draft prepares the
handoff for the compiler-boundary experiment described above. It has
no CPU test result, accepted source archive, predecessor seal,
allocation-owner acknowledgement or GPU execution.
The accepted numerical and execution components remain unchanged.
Their CPU acceptance does not establish CUDA/Inductor transparency.

November 19 13:13 follow-up: the complete local handoff suite finishes
with25 passing tests in230.096seconds, zero failures/errors/skips.
Evidence:
`PVC/run-preparation/test-ns-compiler-localization-predecessor-draft-20261119T1309.xml`.
This supersedes the earlier untested-draft state, not the unchanged
accepted parent archives. Separate staged CPU/source-archive acceptance
is still pending. A CUDA-hidden read-only call to the local helper's
`readiness(PVC)` at13:12 returns
`waiting_for_original_and_actual_lr_owners`, `ready: false`, with
the exact original controller, validator and terminal waiter, actual-LR
queue and marked descendants still present. No seal, canonical
terminal, allocation-owner acknowledgement or CUDA command is issued
by this check.

The planned handoff must authenticate the original terminal owned by
the existing training-path waiter and the later actual-LR terminal
owned by the existing queue. It must preserve all original and actual-LR
rejections, require complete54/162-command and18/54-decision coverage,
verify exact absence of their controllers, validators, waiters and
marked descendants, and bind the restored eight-A100 keeper generations
on the original physical GPU UUIDs. These conditions remain unsatisfied
while the original diagnostic and actual-LR queue are live.
The helper must never invoke another predecessor terminal or deserialize
checkpoint tensors.

Next, implement and accept
the localization allocation owner/controller/worker before queuing
GPU execution. The two fresh reverse-order processes must each retain
128 file-backed phases, with exact finite traced/control transparency
required before interpreting compiler-boundary differences. No timing,
production, full-budget optimizer quality or SwanLab result follows
from this preparation. Full-budget LR neighborhoods, fresh paired
seeds, five large shared-token windows and independently verified
private histories remain required for subsequent production waves.

#### Compiler Localization Handoff Archive Acceptance

November 19 13:20:07.898623, Asia/Shanghai: the distinct file-only
predecessor handoff is CPU/source-archive accepted. Its staged suite
passes25 tests in258.53seconds with zero failures/errors/skips and CUDA
hidden. Receipt:
`PVC/run-preparation/ns-compiler-localization-predecessor-cpu/20261119T1316/receipt.json`.
The accepted155-helper localization execution parent and180-helper
actual-LR queue parent remain unchanged. The merged closure adds the
separately pinned file-only reporter and four handoff/preparation/test
helpers, giving191 authenticated helpers and559 inventory entries.
Fresh `--phase check` from this archive passes by13:21.

The accepted `--phase observe` records
`waiting_for_original_and_actual_lr_owners`, `ready: false`,
at13:21:00.483544. Artifact:
`PVC/run-preparation/ns-compiler-localization-predecessor-observation-20261119T1321.json`.
Exact original controller, validator, terminal waiter, actual-LR queue
and marked descendants remain live. Both canonical completed terminals,
all original/actual-LR decisions, exact owner absence and restored
keeper generations are still required before a predecessor seal.
No seal, terminal invocation, GPU release or allocation owner is issued.

This acceptance exercises refusal for live owners and incomplete,
changed or rebound terminal/source/checkpoint evidence, preserving
rejected cases without deserializing tensor payloads. It does not
validate CUDA/Inductor transparency or supply production authority.
The next unimplemented component is the exclusive localization
allocation owner/controller/worker; its separate acceptance precedes
any queued CUDA command.

Required source lint passes from cached offline tools. The earlier
online lint attempt fails fetching Ruff/Black from PyPI, not source
checks; the successful retry changes no files. Five focused journal/
900-second-monitor tests pass in1.46seconds, Bash syntax and scoped
content/whitespace checks pass. The documented500KiB journal-size
limit remains unchanged. Receipt/source/archive hashes and exact
preparation command are recorded in `DETAILS.md`.
No new full-budget quality, whole-training efficiency or SwanLab result.

### Training-Path Evidence Preparation

November 19 06:40, Asia/Shanghai: the training-path fidelity/step-budget
follow-up now has a locally implemented evidence helper,
`tracking/ns_training_path_predecessor.py`, and focused CPU tests.
The original six-arm NS result and the two failed successor queues remain
unchanged. The helper checks their completed receipts and process absence
without rerunning numerical reconstruction or launching training.
This work adds no optimizer-quality or whole-training timing result.

The historical result remains the one reported in
[NS Budget CUDA Diagnostic Results](#ns-budget-cuda-diagnostic-results):
ten-step BF16 eager outputs reproduce saved historical outputs exactly
in 8/8 comparisons, while all 40 compiled/graph comparisons against those
outputs fail tolerance. All 144 continuous/resumed input-pair checks fail
across six arms and two processes. Tests exercise complete saved JSON
coverage and preserve these rejected predicates; they do not recompute
tensor arithmetic.

A CUDA-hidden read-only component check accepts the original NS queue
archive, lower-LR queue archive and actual-LR queue archive against their
pinned receipts, retained source inventories and completed CPU test XML.
It also checks acknowledged queue ancestry, actual process absence and
six training/numerical files against their original accepted hashes.
The two successor queues retain `needs_attention` and zero issued
freeze/check/start/terminal phases. Their numerical-control rejection and
missing archived dependency are preserved separately.

Review identifies missing dependency coverage and concurrent-seal risks
in the unaccepted helper. Local corrections retain archive/source
dependencies under canonical file identities, reject incomplete identity
coverage, require hidden CUDA, stage and fsync complete seal bytes before
atomic non-overwriting publication, and parse the same bytes that were
digest-checked. Final owner/keeper checks recheck retained file identities.
Focused tests pass after those changes;
earlier failed fixtures and passing receipts remain in `DETAILS.md`.
The final affected suite passes 268/268 in 30.22 seconds with CUDA hidden,
including all 26 focused predecessor cases. Required source lint passes.
Full `evidence()` integration has not run, and no original retained tensor
file is rehashed or deserialized during these component checks.

Before any resource handoff, the helper still needs complete evidence-path
testing, a distinct CPU/source archive and a fresh process importing only
that accepted archive. Its prospective one-shot seal binds original
retained files and exact keeper restoration; it supplies no numerical
continuation or production approval. The training-path numerical protocol
and execution owner are not implemented or issued. The separately proposed
compiled normalization/recurrence experiment is also unimplemented.
Both experiments retain the original full-model/per-LR continuation,
capture/update/work and scalar-loss gates.

The existing Bash monitor publishes both journals at 06:26:12 from a
06:21:57 read-only process/device observation. A separate local physical
query at 06:27 sees only keeper ranks on the eight original A100 UUIDs,
and a process check finds both failed queues absent. No monitor, accepted
queue or keeper is restarted. No lower-LR or actual-LR GPU command,
new production wave or SwanLab training history is issued by this work.
Full-budget shared-token-window analysis, verified private cloud histories,
equal LR search allowance and fresh paired seeds remain required.

#### Evidence-Helper Archive Acceptance

November 19 06:53, Asia/Shanghai: separate CPU/source-archive acceptance
publishes at 06:49:05 under
`PVC/run-preparation/ns-training-path-predecessor-cpu/20261119T0649/receipt.json`.
All 37 archived tests pass in 10.60 seconds. Its 138 helpers preserve all
133 accepted lower-LR execution-owner helpers, adding the new predecessor,
persisted support, archive preparer and two test files. The 504-file inventory
also retains the original complete saved comparison JSON as a CPU fixture.
All 279 affected journal/reporting/helper tests pass in 34.50 seconds.
A fresh process using only the accepted archive authenticates the receipt
by 06:50; required source lint passes.

Six new cases exercise the complete persisted terminal chain through the
real `evidence()` path, including queue commands, worker cleanup, complete
retained-output identities, trainer source and a seal round trip. Five
archive cases check source/test/archive changes and fresh-process import
origins. These CPU fixtures do not execute training or reconstruct the
original numerical outputs. Original-file component checks accept the
three queue archives, the separately accepted 75-case NS diagnostic-owner
archive and all six trainer/numerical hashes. Both failed queues remain
absent and retain zero issued phases.

The live one-shot seal remains absent. Its later operation must digest-check
the original retained output files and reobserve the exact keeper under
the manager lock; it does not repeat tensor reconstruction or authorize
training. A separately identified 18-case training-path numerical protocol
is now in implementation, reusing accepted readers without changing their
coefficients, tolerances, compilation boundaries or failed predecessors.
Source/archive acceptance and exclusive execution ownership remain required.
The 06:45:30 monitor publication and separate 06:50 local GPU query show
the restored keeper ranks and no new local training. No optimizer-quality,
whole-training timing or new SwanLab result follows from this acceptance.

At 06:55:39 the accepted archived sealer completes against the original
retained files. It hashes 1,444 canonical evidence files, including both
original 2.3 GB output payloads and 1,419 source/archive dependencies,
without deserializing tensors or repeating numerical reconstruction.
It verifies the exact restored keeper under the manager lock and preserves
both failed zero-phase queue decisions, historical eager reproduction and
the rejected compiled control. The one-shot seal is
`PVC/run-preparation/ns-training-path-predecessor-seal-20261119.json`.
It explicitly denies GPU execution and production authority.
The successor must jointly bind this seal's identity, the issuing accepted
CPU receipt and verifier source identities. Fresh archived seal checking
succeeds by 07:05, authenticating retained identities and small receipts
without rehashing the large payloads. No numerical command or keeper
transfer has been issued.

#### Training-Path Numerical Protocol CPU Check

November 19 07:05, Asia/Shanghai: the distinct
`tracking/ns_training_path_contract.py` and
`tracking/ns_training_path_validation.py` are implemented and held for
source/archive acceptance. All 34 focused CPU tests pass in 146.91 seconds
with CUDA hidden; the complete XML records zero failures, errors or skips.
Required source lint passes. These tests use persisted CPU fixtures and
CPU child processes. They do not run the proposed GPU diagnostic.

The protocol retains 18 cases at matrix LR `0.04525483392`: MuonT, fused
output diagonal and identity-outer matched null, each at five/seven/ten NS
steps and BF16/FP32 NS operand modes. The null computes the fused arm's
output statistics but applies an identity outer factor. Each case compares
continuous with interrupted/resumed training within the same arm.
Its 54 eight-rank commands stop at update 24, or at update 22 before
resuming to 24, and retain checkpoints 22/23/24. Planned diagnostic work
is 452,984,832 tokens. The LR schedule still uses the original
5,120-update horizon; these commands do not execute that full budget.
Numerical readers, continuation tolerances, scalar-loss and fused/null
capture/update/work gates are unchanged. It uses a distinct training-path
binding and output namespace and requires a separately issued owner;
the current parent adapter denies execution.

Read-only review finds that six explicit trainer/numerical pins and
candidate-local configuration comparisons do not independently pin every
original numerical dependency. Before execution, the owner must authenticate
all original configuration bytes and the complete inherited source
inventory against accepted predecessor evidence, permitting only its
separately reviewed ownership-adapter change. This includes checkpoint
restoration, capture and compilation settings. The owner must also bind
the accepted evidence-helper receipt, verifier sources and original-output
seal. CPU/archive acceptance of the protocol alone grants no GPU authority.

The single Bash monitor publishes both documents at 07:04:48 from the
07:00:32 read-only observation. A separate local query at 07:04 sees only
the original restored keeper ranks on all eight A100 UUIDs; both failed
queues are absent. No new production wave, loss/timing result or SwanLab
history is available. Full-budget comparisons, equal LR search allowance
and fresh paired seeds remain required.

At 07:10:02 separate CPU/source-archive acceptance publishes
`PVC/run-preparation/ns-training-path-validation-component-cpu/20261119T0707/receipt.json`.
All 34 archived tests pass in 148.83 seconds, with 143 authenticated helpers
and 509 inventory entries. Every one of the 138 accepted evidence-parent
helpers is preserved; five new numerical/support/test/preparation files
are added. A fresh process using only the accepted archive checks its
source, test XML, archive and exact import origins successfully by 07:11.
The source-owner and lifecycle components remain in implementation.
No binding, allocation transfer or GPU command is issued.

#### Execution-Owner Review and Recurring Reporting

November 19 07:24, Asia/Shanghai: the distinct owner-only source and
lifecycle implementation passes 47 local CPU tests in 21.00 seconds,
including 37 lifecycle and 10 source-preservation cases. These local
results have no persisted test XML and do not establish archived owner
acceptance. Review confirms independent pinning of all inherited
configuration/source bytes, the exact single ownership-adapter change,
the live original controller's acceptance proof, and separation of
CUDA-hidden predecessor checks from CUDA-visible rank checks.

Two review findings remain launch gates. Terminal verification currently
permits a second output pathname for the same controller generation,
which could repeat tensor reconstruction; it needs a canonical one-shot
identity and exclusive publication. Physical occupancy checks can be
empty before training, leaving the calling rank's actual CUDA ordinal
mapping unchecked before work; the original physical GPU UUID mapping
must be verified at that boundary. No live preparation, execution-owner
archive, binding, keeper transfer or GPU launch is issued.

The read-only training-path reporter is now connected to both recurring
document blocks. It authenticates the numerical CPU archive, source
inventory, test XML and logs, and explicitly withholds unbound owner and
GPU results. All 22 focused cases pass; all 311 integrated
journal/reporting/digest cases pass in 105.08 seconds with CUDA hidden.
Two real-archive checks independently recover 34 passing CPU cases,
143 helpers and 509 entries. Earlier local environment/helper-type
reporting refusals remain recorded in `DETAILS.md`; the corrected
reporter accepts the inherited shell helper under its original hash.

The scheduled monitor publishes at 07:24:08 using its earlier loaded
reporter and preserves the refusal. Both documents' manual-prose hashes
are unchanged across that refresh. A separate corrected guarded refresh
is pending. The original monitor and all eight restored keeper ranks
are live at 07:24:26; no optimizer-quality, whole-training efficiency or
new SwanLab result follows from these component tests.

At November 19 07:31, the two review corrections are implemented and
55 local CPU cases pass in 21.38 seconds: 45 lifecycle cases and 10
source-preservation cases. The retained XML is
`PVC/run-preparation/test-ns-training-path-owner-review-fixes-20261119T0730.xml`.
Terminal intent, outcome and receipt publication are exclusive and tied
to the original controller generation. Identity-only CUDA driver
inspection checks all eight ordinal-to-UUID mappings before work,
including empty occupancy. Independent review and separately archived
owner acceptance remain required. The numerical-component archive,
trainer source, tolerances and failed predecessors are unchanged.

The corrected guarded document refresh completes at 07:30:13 using a
07:25:55 report timestamp. Both recurring blocks now authenticate the
34-case numerical CPU acceptance and explicitly withhold owner/GPU
acceptance. Both manual-prose hashes are byte-preserved. The single
900-second Bash monitor remains live and will use the corrected reporter
on its next invocation; it was not restarted. No new full-budget wave,
large-window loss result, whole-training timing or private SwanLab
history is issued.

At November 19 07:38, final targeted review accepts the two local
one-shot-terminal and pre-work device-identity corrections within their
CPU component scope. Downstream terminal acceptance must require the
complete outcome and matching receipt hash; a failed or premature
reconstruction attempt consumes its claim.
The first real preparation fails before candidate creation because its
CPU environment import is outside the accepted helper closure. A distinct
correction obtains the identical fixed environment from the accepted
predecessor helper; all 55 CPU cases pass again in 20.99 seconds.
The corrected owner-only candidate is prepared at 07:36:33:
`PVC/candidates/ns-training-path-owner-closure-corrected-20261119T0736`.
All 353 candidate files retain their parent bytes except the reviewed
ownership adapter; 149 helpers preserve every accepted parent helper.
Separate CPU/archive acceptance is running. No binding, keeper transfer,
GPU command or numerical training result is issued.

At November 19 07:43, separate owner CPU/archive acceptance publishes
at 07:38:31 under
`PVC/run-preparation/ns-training-path-owner-cpu/closure-corrected-20261119T0738/receipt.json`.
All 55 archived cases pass in 23.83 seconds, with 149 helpers and
517 inventory entries. A fresh archived-entry check passes by 07:40.
Actual frozen-source materialization completes at 07:41:12, binding all
18 cases and 54 commands to the reviewed candidate, accepted data and
original evidence seal. A fresh issued-source checker passes by 07:42:43.
The accepted controller start is now in progress; acknowledgment and
allocation transfer are not yet observed. Training compilation and
cached-root graphs remain enabled; launch-parent compilation-disable
overrides are absent. These correctness-gate preparations add no
full-budget optimizer result or cloud history.

At November 19 07:46, the distinct diagnostic is launched. The exact
controller acknowledges at 07:44:23; read-only checks verify that
generation and its direct validator live. The controller completes
keeper release under the node/manager locks and acknowledges its
validator at 07:45:07. The first planned command,
`muont_ns_tpu_q5_full_lr_lower` continuous through update 24, is live by
07:45:52. Its receipt contains all 18 planned cases but zero completed
commands/results at that observation. The other cases remain pending.
This launch preserves the 5,120-update LR horizon, original model/loss
compilation, compiled matrix products, cached-root graphs and original
continuation tolerances. No historical failed queue is restarted.
Complete all-rank evidence, original-owner cleanup/restoration and
independent one-shot terminal acceptance remain required.

#### Training-Path Running Progress

November 19 08:04, Asia/Shanghai: all three commands for
`muont_ns_tpu_q5_full_lr_lower` have exited with execution and cleanup
codes `0`. This case uses BF16 NS operands and five NS iterations,
matrix LR `0.04525483392`, seed 0 and the original 5,120-update schedule.
Its continuous run completes updates 1-24; its interrupted run stops
after update 22; its resumed run loads checkpoint 22 and completes
updates 23-24. The continuous history contains 24 persisted update
records and checkpoints 22/23/24. Its execution receipt records both
zero exits and absence of child processes carrying the command's
unique marker at 07:47:44:
`PVC/validation/ns_training_path_20261119/muont_ns_tpu_q5_full_lr_lower_continuous.execution.json`.
The interrupted and resumed execution receipts in that directory end
in `_interrupted.execution.json` and `_resumed.execution.json`;
their exit times are 07:52:12 and 07:54:52. The validator records all
three completed commands by 07:57:23. It has published no acceptance
or rejection for this case at 08:04; retained checkpoint/loss comparison
is still required. The original controller `2099685/937961924` and
validator `2100106/937966294` are live in the 08:03:39 read-only process
check. Empty GPU compute-process queries between commands do not
establish completion of the 18-case diagnostic or restoration of the
eight non-training allocation-keeper ranks.

The recurring reporter checks file hashes and linked receipt consistency
for the source archive whose 55 CPU execution-owner tests passed, the
original evidence seal, frozen source/data binding and acknowledgment
of this controller's launch. It checks command execution sidecars and
log hashes before counting completed commands. It retains numerical
rejections and requires a complete, hash-linked one-shot terminal outcome
before reporting independent numerical reconstruction and restoration.
It does not execute tensors or process/device verifiers. All 51 focused
reporting tests pass in 24.75 seconds; all 364 integrated journal,
reporting and digest tests pass in 136.17 seconds with CUDA hidden.
The full guarded split-document refresh succeeds at 07:59:22, reporting
3/54 authenticated completed commands and zero numerical case decisions.
Both manual-prose hashes are unchanged across that publication.
Test receipts are `PVC/run-preparation/test-ns-training-path-reporting-owner-fixed-20261119T0750.xml`
and `PVC/run-preparation/test-ns-training-path-journal-integration-20261119T0751.xml`.
The first failed reporting fixture receipt remains separately retained.
These checks produce no new full-budget loss, end-to-end efficiency
measurement or SwanLab training history.

Before final reporting acceptance, review identifies three additional
receipt-consistency requirements: compare gate flags with retained
checkpoint/loss predicates, refuse extra issued executable files, and
retain premature terminal attempts even when controller state/result
is absent. Corrections and focused regressions are in progress.
The accepted diagnostic source and original numerical tolerances are
unchanged. At that review checkpoint no numerical case acceptance had
been reported.

#### First Training-Path Case Decision

November 19 08:07:38, Asia/Shanghai: the original validator records
`muont_ns_tpu_q5_full_lr_lower` accepted, with all five gates true.
The saved all-rank comparisons contain 2,135 tensor comparisons at
update22 and 2,147 at update24; every tensor is unchanged and no scalar
or structure difference is reported at those acceptance boundaries.
All 39 scalar-loss comparisons pass the original tolerance.
The retained update23 comparison has 2,167 unchanged tensors and no
tensor failures, but 16 process-local graph-readiness flag differences
across eight ranks. Its `within_tolerance: false` remains recorded.
Update23 is diagnostic-only under the original protocol and is not
silently counted as a pass.

Canonical growing receipt:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[0]`. The first-case observation is also retained at
`PVC/run-preparation/validation-ns-training-path-first-case-observation-20261119T0810.json`.
This is the validator's saved numerical reconstruction;
the separate one-shot terminal reconstruction and exact keeper restoration
are still pending. MuonT has no output-capture ablation in this case
(`capture: {}`), so its true capture/work gate supplies no fused/null
capture evidence. The result checks checkpoint continuation within the
five-iteration MuonT arm; it does not establish parity with ten-iteration
MuonT, repair the rejected standalone compiled-NS controls, or measure
full-budget quality or end-to-end efficiency.

At 08:08:14, read-only process checks find the exact original controller
and validator live and command 4/54, MuonT BF16/seven-NS-iteration
continuous24, running. The remaining 17 cases must receive their own
decisions under the unchanged source/configuration-specific gates.
Later full-budget experiments still require independently accepted
terminal evidence, actual-LR gates, equal tuning allowance, five shared
536,870,912-token windows, private SwanLab verification and fresh
paired seeds. No new production wave or cloud history is issued.

At 08:10:01, the seven-iteration continuous24 command exits with
execution/cleanup codes zero and its marked children absent. Its complete
trajectory enters the aggregate receipt at 08:12:26, making 4/54 completed
commands. This does not complete that numerical case; interrupted22 and
resumed24 evidence and its own checkpoint/loss comparisons remain required.

The reporting corrections now pass 72 focused tests in 44.88 seconds.
Gate flags must agree with retained checkpoint22/checkpoint24 and loss
predicates; update23 remains diagnostic-only. Extra issued executable
files refuse progress reporting. Claimed or failed one-shot terminal
attempts remain visible before state/result publication, and a purported
complete outcome without that evidence is refused. Keeper comparison
uses process generation and physical device identity rather than dynamic
scheduler-status or observation ordering. All 385 final integrated tests
pass in 155.73 seconds with CUDA hidden; required scoped lint and the
unchanged Bash monitor's syntax check pass. Independent scoped source
review confirms the three reporting gaps are closed.
Test receipts:
`PVC/run-preparation/test-ns-training-path-reporting-review-20261119T0808.xml`
and `PVC/run-preparation/test-ns-training-path-journal-review-final-20261119T0813.xml`.
A fresh read-only report accepts the issued owner/binding and first
numerical decision while retaining independent terminal acceptance as
pending. Guarded publication completes at 08:19:15; both manual-prose
hashes are unchanged. The existing scheduled monitor publishes again
at 08:24:02, showing 6/54 authenticated commands and one saved accepted
case. No accepted diagnostic source is edited.

The seven-iteration interrupted22 and resumed24 command receipts record
execution and cleanup codes zero at 08:14:32 and 08:17:16. All six
commands' retained trajectories are present in the growing receipt at
08:19:46. Its seven-iteration numerical case analysis remains pending
at that checkpoint.
The original controller/validator and single document monitor remain live;
no independent terminal/restoration receipt or new SwanLab history exists.

#### Second Training-Path Case Decision

November 19 08:29:57, Asia/Shanghai: the original validator accepts
`muont_ns_tpu_q7_full_lr_lower`, with all five recorded gates true.
The executed configuration is
`experiment/configs/muont_ns_tpu_q7_full_lr_lower_0p5m_seed0.json`
under `PVC/candidates/ns-training-path-owner-closure-corrected-20261119T0736`.
It retains BF16 NS operands, seven NS iterations, matrix LR
`0.04525483392`, seed 0, eight ranks and the original 5,120-update LR
horizon. Its continuous24/interrupted22/resumed24 commands and their
zero execution/cleanup outcomes were recorded above.

| Retained Comparison | Tensor Comparisons Across Eight Ranks | Changed or Failed Tensors | Scalar or Structure Differences | Saved Tolerance Predicate |
| --- | ---: | ---: | ---: | --- |
| Boundary checkpoint22 | 2,123 | 0 | 0 | `true` |
| First-resumed checkpoint23, diagnostic only | 2,161 | 0 | 16 graph-readiness flags | `false` |
| Post-cached-root checkpoint24 | 2,163 | 0 | 0 | `true` |

All 39 scalar-loss comparisons pass; `capture: {}` again supplies no
fused/null capture evidence. The 16 update23 differences are the
`cached_root_graph_active` flags for groups `d` and `f` on all eight
ranks. Their false tolerance predicate remains retained rather than
being silently promoted. Canonical numerical evidence:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[1]`, published at 08:29:57. These are the original validator's
case results, not independent terminal reconstruction.
There are now two accepted lower-LR MuonT continuation cases and no
rejected case decisions yet. This does not establish equality between
five/seven/ten-iteration finite maps, historical compiled-control repair,
full-training quality or end-to-end efficiency.

The ten-iteration MuonT continuous24 command records integer-zero
execution/cleanup and marked-child absence at 08:32:15. Its complete
retained trajectory is added at 08:34:43, making 7/54 completed commands.
At 08:35:36, the controller state selects its interrupted22 command;
that case still needs interrupted/resumed completion and its own
numerical decision. No new production wave or SwanLab history exists.

#### Next Training-Path Wave Transition

November 19 08:35, Asia/Shanghai: the read-only generation check at
08:28:31 verifies
the original controller, direct validator and sole document monitor by
PID and start ticks; repeated process checks still find them live.
The latest growing result remains `validating`, with seven complete
commands and five/seven-iteration MuonT accepted by the validator.
The ten-iteration case awaits its remaining commands and retained
checkpoint/loss analysis; zero command exits do not count as acceptance.
No terminal claim, outcome or receipt is recorded. The existing monitor
continues its 900-second sleep between refreshes and needs no restart.

| Stage | State at This Observation | Required Evidence Before Advancing |
| --- | --- | --- |
| Original-training-path correctness diagnostic | Running; 7/54 commands complete, 2/18 cases accepted by the validator | Finish all 18 case decisions, retain rejected cases and update23 diagnostics, authenticate original-owner cleanup/restoration, then invoke the frozen controller's canonical one-shot terminal only after both original owners are absent and state is complete. |
| Full-budget NS quality/cost study | Planned only; the existing 54-trial protocol has no measured histories | Prepare a separately identified training-path successor, bind its exact source and actual LR configurations, and obtain unchanged-tolerance per-config continuation/capture/update/work acceptance. Neither this lower-LR diagnostic nor the failed earlier queues authorize center/upper LR or production. |
| Paired confirmation | Fresh seeds 3/4/5 reserved | Complete equal LR allowance and any symmetric boundary expansion, then compare selected candidates, independently tuned MuonT and matched null on fresh paired seeds. |

The next quality experiment remains the documented three-family,
two-operand, five/seven/ten-iteration study at equal full-budget LR
allowance. Five/seven iterations change the finite NS map; a 24-update
continuation result cannot select them for quality or establish a speed
benefit. Do not rank or prune arms using these diagnostic losses.
For each eligible production wave, keep launch, completion, analysis
and upload milestones separate. Require all five shared
536,870,912-token windows, 22 exact shared evaluations and independently
verified full-resolution histories in the private SwanLab project
`output-side-normalization-marin` before the next production wave.
No new production release, wave launch, cloud import or optimizer
improvement over MuonT is reported at this observation.

#### Third Training-Path Case Decision

November 19 08:52:24, Asia/Shanghai: the original validator accepts
`muont_ns_tpu_q10_full_lr_lower`, using
`experiment/configs/muont_ns_tpu_q10_full_lr_lower_0p5m_seed0.json`
under the same `ns-training-path-owner-closure-corrected-20261119T0736`
candidate. This is BF16, ten NS iterations, lower matrix LR
`0.04525483392`, seed 0 and the unchanged full-budget LR horizon.
Its continuous24, interrupted22 and resumed24 commands exit and clean
up integer zero at 08:32:15, 08:36:50 and 08:39:34 respectively;
all nine first-three-case trajectories are retained by 08:42:07.

Checkpoint22 has 2,145 exact tensor comparisons and checkpoint24 has
2,179, with no changed tensors, failures or scalar/structure differences
at either acceptance boundary. All 39 scalar-loss comparisons pass.
Checkpoint23 has 2,127 unchanged tensors but sixteen cached-root
graph-readiness flag differences across eight ranks; its false tolerance
predicate remains diagnostic-only and retained. MuonT's `capture: {}`
does not establish fused/null backward capture or matched output work.
Canonical result:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[2]`. Three BF16 MuonT cases now have validator acceptance;
the other fifteen case decisions and independent terminal remain pending.
These within-arm continuation checks do not compare finite NS maps or
establish a full-training quality/efficiency improvement.

#### Issued Read-Only Training-Path Terminal Waiter

November 19 08:47, Asia/Shanghai: a separate CPU-only waiter is archived
and launched without changing the issued diagnostic or transferring
allocation. New helpers are `tracking/ns_training_path_followup.py` and
`tracking/test_ns_training_path_followup.py`. Its first real preparation
stops before output creation because the flat import closure does not
contain `ns_budget_queue.py`. The corrected helper uses the already issued
process observation APIs directly; the failed attempt is not reused.
The separate accepted archive preserves all 149 issued helpers unchanged
and adds those two helpers. Nineteen archived tests pass in 3.40 seconds;
the archived XML records 3.307 seconds. All 74 integrated waiter/owner
tests pass in 24.12 seconds. Required scoped lint passes, and a fresh
archived-entry check accepts the unchanged receipt.

CPU/archive receipt:
`PVC/run-preparation/ns-training-path-followup-cpu/closure-corrected-20261119T0847/receipt.json`.
The single waiter acknowledges at 08:47:34; its original generation is
`2189311/938341032`, verified live after the launcher exits.
Its first immutable observation at 08:47:38 checks exact controller
`2099685/937961924` and direct validator `2100106/937966294` live,
then saves `waiting_for_original_owners`. At that observation nine
commands and two case decisions are retained. Evidence directory:
`PVC/run-preparation/ns-training-path-terminal-followup-20261119`,
with `intent.json`, `state.json` and `observation-000000.json`.

Every 900 seconds it checks the same owner generations. Zombie handles,
substituted owners, incomplete original cleanup/restoration and existing
terminal claims refuse issuance. After actual owner/marked-descendant
absence and complete original state/results, it rechecks readiness and
invokes only the original frozen controller's generation-bound terminal
path. Any original failed or premature terminal claim remains consumed;
it never repeats reconstruction under another pathname. Child execution,
detached-descendant cleanup and terminal-outcome hashes are retained.
Numerically rejected cases remain rejected. This waiter does not own
GPU allocation, launch a quality study, alter tolerances, provide a new
tensor reader or authorize production/cloud upload.
The document monitor remains the single existing Bash process.

At 08:59, the read-only waiter reporter is integrated with both recurring
document blocks. It authenticates the accepted nineteen-test source archive,
exact waiter acknowledgment, saved original generations and immutable
observation identity, without running process/device or tensor verifiers.
Reported terminal completion additionally requires the original accepted
one-shot terminal, retained rejections, exact child ancestry and zero
execution/cleanup codes; changed logs and boolean cleanup codes are refused.
Failures remain visible without changing original numerical acceptance.
All 102 focused tests pass in 56.32 seconds, and all 320 integrated
journal/reporting/digest/waiter tests pass in 83.24 seconds with CUDA
hidden. Scoped code/Markdown lint passes. Test receipts:
`PVC/run-preparation/test-ns-training-path-followup-reporting-20261119T0852.xml`
and `PVC/run-preparation/test-ns-training-path-followup-journal-integrated-20261119T0854.xml`.
Guarded publication completes by 08:59:00, preserving both manual-prose
hashes. The original numerical/owner/follow-up archives are unchanged.

FP32 MuonT with five NS iterations completes its continuous24 command
at 08:54:44 with integer-zero execution/cleanup and marked children absent.
Its complete trajectory is retained at 08:57:11, making 10/54 commands
and three accepted cases. The state selects interrupted22 at 08:57:20.
This case still requires its own interrupted/resumed trajectories and
all-rank/loss decision. Exact controller, validator, waiter and monitor
generations remain live in the fresh process checks. No independent
terminal/restoration, full-budget successor or new cloud history exists.

#### FP32 Five-Iteration Command Completion

November 19 09:05, Asia/Shanghai: all three
`muont_ns_f32_q5_full_lr_lower` commands now have integer-zero execution
and cleanup receipts, with marked descendants absent. The executed
configuration is
`experiment/configs/muont_ns_f32_q5_full_lr_lower_0p5m_seed0.json`
under `PVC/candidates/ns-training-path-owner-closure-corrected-20261119T0736`;
it retains FP32 NS operands, five iterations, LR `0.04525483392`,
seed 0, eight ranks and the original 5,120-update schedule.

| Command | Execution/Cleanup Completed | Complete Trajectory Retained |
| --- | --- | --- |
| continuous24 | 08:54:44 | 08:57:11 |
| interrupted22 | 08:59:21 | 09:00:10 |
| resumed22-to-24 | 09:02:01 | 09:04:31 |

Canonical evidence is the growing
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`gpu_commands[9:12]`, and the matching continuous/interrupted/resumed
`.execution.json` sidecars in `PVC/validation/ns_training_path_20261119`.
At the 09:05 checkpoint there are 12/54 retained commands and three
accepted BF16 MuonT cases. The FP32 case then has no checkpoint/loss
decision yet; its later acceptance is recorded below. Its short-run losses
are not used to select NS budgets or rank optimizer quality.
The waiter's second immutable observation, `observation-000001.json`,
records both original owners live at 09:02:39. Independent terminal
reconstruction and exact keeper restoration remain pending.

#### Fourth Training-Path Case Decision

November 19 09:14:48, Asia/Shanghai: the original validator accepts
`muont_ns_f32_q5_full_lr_lower`, the fourth of eighteen cases. It uses
the unchanged issued candidate and configuration from the preceding
command-completion entry: FP32 NS operands, five iterations, matrix LR
`0.04525483392`, seed 0, eight ranks, model compilation and cached-root
CUDA graphs, with the original 5,120-update LR horizon.

| Comparison | Tensor Comparisons | Changed/Failed Tensors | Scalar/Structure Differences | Predicate |
| --- | ---: | --- | ---: | --- |
| interrupted/continuous checkpoint22 | 2,121 | 0/0 | 0 | accepted boundary |
| resumed/continuous checkpoint23 | 2,173 | 0/0 | 16 | false; retained diagnostic only |
| resumed/continuous checkpoint24 | 2,173 | 0/0 | 0 | accepted post-cached-root boundary |
| scalar loss | 39 | zero failed comparisons | not applicable | accepted |

The sixteen update23 differences are cached-root graph-readiness flags
across eight ranks. The original false tolerance predicate is preserved;
it is not weakened or counted as an accepted checkpoint.
MuonT's empty `capture` evidence does not establish fused/null captured
rows, update parity or matched work.
Canonical result: `PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[3]`; configuration SHA256 and command/log evidence are in
`DETAILS.md`. This is within-arm continuation evidence, not equality
between finite NS maps, repaired historical standalone controls,
full-budget optimizer quality or whole-training efficiency.

At 09:26:50 the aggregate contains 15/54 commands and four accepted
cases. FP32 MuonT/seven iterations exits/cleans up zero at 09:17:05,
09:21:39 and 09:24:21 for continuous24, interrupted22 and resumed24,
respectively; the complete trajectories enter the aggregate at
09:19:32, 09:22:29 and 09:26:48. Its numerical decision remains pending.
The waiter's third immutable observation at 09:17:40 records both
original owners live and `ready: false`. The sole Bash monitor refreshes
both journal blocks at 09:23:48. No independent terminal, exact final
restoration, successor GPU launch or new private cloud history is inferred.

#### Fifth Training-Path Case Decision

November 19 09:36:55.670664, Asia/Shanghai: the original validator
accepts `muont_ns_f32_q7_full_lr_lower`. Configuration
`experiment/configs/muont_ns_f32_q7_full_lr_lower_0p5m_seed0.json`
under the unchanged issued candidate retains FP32 NS operands,
seven iterations, LR `0.04525483392`, seed 0, eight ranks,
the original 5,120-update LR horizon, compilation and cached-root graphs.
The source/configuration identity and all three command/log sidecars
are recorded in `DETAILS.md`.

| Checkpoint | Tensor Comparisons | Changed/Failed Tensors | Scalar/Structure Differences | Predicate |
| --- | ---: | --- | ---: | --- |
| 22 | 2,121 | 0/0 | 0 | accepted boundary |
| 23 | 2,185 | 0/0 | 16 | false; retained diagnostic only |
| 24 | 2,167 | 0/0 | 0 | accepted post-cached-root boundary |

All 39 scalar-loss comparisons pass. Update23's sixteen cached-root
graph-readiness differences remain retained and diagnostic-only.
Empty MuonT capture evidence supplies no fused/null capture or matched-work
acceptance. Canonical growing result:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[4]`.
At 09:37:18 there are 15/54 retained commands and five accepted cases.
The state selects FP32/ten-iteration continuous24 at 09:37:04;
its result is pending. Original controller, validator, waiter and monitor
generations remain live. The waiter's fourth immutable observation,
`observation-000003.json`, records original owners live and `ready: false`
at 09:32:40. No independent terminal or exact final restoration follows.
These within-arm continuation results do not select an NS budget,
repair historical controls or establish a better optimizer.

#### Training-Path Actual-LR Component CPU Validation

November 19 09:25, Asia/Shanghai: a distinct actual-LR numerical
component is implemented in five mutable tracking files:
`ns_training_path_lr_contract.py`, `ns_training_path_lr_validation.py`,
`ns_training_path_lr_test_support.py`, `test_ns_training_path_lr_validation.py`
and `prepare_ns_training_path_lr_validation.py`.
It directly reuses the accepted actual-LR trial reader; all six original
training files and five accepted numerical-reader hashes stay pinned.
It reserves the separate `ns-training-path-actual-lr-20261119`
binding/controller/state/output/result namespace described by its
`locations` contract. No owner adapter or ownership release is supplied.
Attempted execution against the current issued adapter fails before
creating output.

All 29 corrected CPU tests pass in 75.36 seconds with CUDA hidden.
They exercise complete 162-command retained reconstruction, an
independent rank-7 rejection, erased/forged decision refusal, common-RNG
and capture failures, retained graph-rewarming diagnostics, current-owner
rejection and bounded child/detached-descendant cleanup.
The first draft run's six fixture-path failures are retained, not treated
as GPU failures or accepted evidence. The correction uses the existing
interrupted-manifest contract; numerical tolerances are unchanged.
Required scoped Ruff/Black/AST/license/whitespace/conflict lint passes.

A CUDA-hidden read-only preparer audit authenticates the complete
149-helper training-owner archive and 141-helper actual-LR numerical
archive. Their overlap agrees byte-for-byte; their union plus these
five new files forms a complete 162-helper flat import closure.
The plan still has 54 configurations, 162 commands and 1,358,954,496
diagnostic tokens. Separate source/CPU preparation completes at 09:27:15 at
`PVC/run-preparation/ns-training-path-actual-lr-component-cpu/20261119T0926`.
Its receipt status is
`ns_training_path_actual_lr_component_cpu_archive_verified_owner_unissued`.
All 29 archived tests pass in 74.07 seconds; the XML records
73.975 seconds with zero failed/error/skipped cases.
The archive contains 162 helpers and 531 authenticated inventory entries.
A fresh process using the archived preparer and exact archived import
origins accepts the unchanged receipt by 09:29:32.
Commands, source/archive hashes and retained test evidence are in
`DETAILS.md`. Five focused split-document, ambiguous-boundary and
900-second monitor-retry tests also pass in 1.52 seconds.
The integrated original-training-path and actual-LR numerical suite
passes all 63 cases in 221.51 seconds, with the 600-second session
guard and full safe marker exclusions. XML/hash evidence is in
`DETAILS.md`. Guarded publication of both recurring blocks completes
at 09:35:50 and verifies all manual prose byte-preserved.

This component does not authenticate or issue a future predecessor,
owner, binding, queue or GPU launch. The original complete diagnostic,
canonical independent terminal, owner/waiter/descendant absence and
exact restoration remain required. The next experiment's scientific
scope and fair full-budget allowance are unchanged below.

#### FP32 Ten-Iteration Command Completion

November 19 09:55, Asia/Shanghai: the original validator has retained
all three `muont_ns_f32_q10_full_lr_lower` trajectories. The issued
configuration retains FP32 NS operands, ten iterations, lower matrix LR
`0.04525483392`, seed 0, eight ranks, original 5,120-update schedule,
model compilation and cached-root CUDA graphs.

| Command | Execution/Cleanup Zero, Marked Children Absent | Complete Trajectory Retained |
| --- | --- | --- |
| continuous24 | 09:39:18 | 09:41:48 |
| interrupted22 | 09:43:58 | 09:44:48 |
| resumed22-to-24 | 09:46:37 | 09:49:04 |

Canonical growing aggregate:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`gpu_commands[15:18]`; configuration SHA256 and log/sidecar identities
are in `DETAILS.md`. At 09:55:13 there are eighteen retained commands,
five accepted cases and no sixth numerical decision.
Short diagnostic losses are not used to choose iteration budgets
or claim optimizer quality/efficiency. The next original case decision,
full 18-case result and canonical independent terminal remain required.

#### Sixth Training-Path Case Decision

November 19 09:59:15.980807, Asia/Shanghai: the original validator
accepts `muont_ns_f32_q10_full_lr_lower`. The unchanged issued
configuration uses FP32 NS operands, ten iterations, matrix LR
`0.04525483392`, seed 0 and eight ranks. The original 5,120-update
schedule, compilation and cached-root CUDA graphs remain enabled.
Configuration and command/log identities are retained in `DETAILS.md`.

| Checkpoint | Tensor Comparisons | Changed/Failed Tensors | Scalar/Structure Differences | Predicate |
| --- | ---: | --- | ---: | --- |
| 22 | 2,151 | 0/0 | 0 | accepted boundary |
| 23 | 2,155 | 0/0 | 16 | false; retained diagnostic only |
| 24 | 2,169 | 0/0 | 0 | accepted post-cached-root boundary |

All 39 scalar-loss comparisons pass. The sixteen checkpoint23
differences are cached-root graph-readiness flags across eight ranks.
Their false predicate remains unchanged. MuonT's `capture: {}` does
not establish fused/null captured-row, update or matched-work evidence.
Canonical growing aggregate:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[5]`. All six MuonT lower-LR cases now have validator-accepted
within-arm continuation, but the original eighteen-case diagnostic
and independent terminal are incomplete.
No iteration budget is selected from these short runs, no historical
control rejection is repaired, and no optimizer superiority follows.

#### First Fused Training-Path Command

November 19 10:02:14, Asia/Shanghai: the first
`outer_leftdiag_fused_ns_tpu_q5_full_lr_lower` continuous24 command
exits and cleans up with integer zero and absent marked descendants.
The state selected it at 09:59:24, using BF16 NS operands, five
iterations, matrix LR `0.04525483392`, seed 0, eight ranks, the original
5,120-update schedule and unchanged compiled/cached-root paths.
It requests diagonal-capture and outer-update verification.
Its private execution sidecar and log identities are in `DETAILS.md`.

At 10:04:06 the canonical aggregate still contains eighteen retained
commands and six accepted cases: this command's complete retained
trajectory has not yet been published. Interrupted22/resumed24 and
the fused-case numerical decision remain pending. Zero command exit
does not establish three-trajectory continuation or independently
reconstructed captured-row/update/work acceptance.
The trajectory enters the aggregate at 10:04:47.201321 as
`gpu_commands[18]`, bringing it to nineteen retained commands and
six accepted cases. State publication at 10:04:55 selects fused
interrupted22. A fresh 10:06:40 observation still finds that selection,
with no interrupted/resumed trajectory or fused-case decision published.
The interrupted22 sidecar subsequently records integer-zero execution/
cleanup and absent marked descendants at 10:07:21.310719. At 10:08:02
the aggregate still has nineteen retained commands and six accepted
cases; interrupted22's retained trajectory, resumed24 and the fused
numerical decision remain pending.
The interrupted trajectory is retained at 10:08:10.935276.
Resumed22-to-24 exits/cleans up zero at 10:10:16.850552 and its
complete trajectory enters the aggregate at 10:12:44.218931.
All three trajectories are now retained as `gpu_commands[18:21]`;
the aggregate contains twenty-one commands and six accepted cases.
The seventh, fused-case numerical decision remains pending at 10:23:21.
Original controller and validator remain live; neither zero command
exit nor complete trajectory coverage replaces the all-rank decision.
The next scientific decision remains whether fused and matched-null
cases pass the unchanged all-rank gates, followed by actual-LR gates
and equal-budget production tuning with large shared-token windows.
The diagnostic supplies no full-budget quality, whole-training
efficiency or new SwanLab result.

#### First Fused Training-Path Case Rejection

November 19 10:23:53.027730, Asia/Shanghai: the original validator
rejects `outer_leftdiag_fused_ns_tpu_q5_full_lr_lower`. The configuration
uses BF16 NS operands, five iterations, matrix LR `0.04525483392`,
seed 0 and eight ranks, with the original 5,120-update schedule,
compilation and cached-root CUDA graphs enabled. All three commands
exit and clean up with integer zero; complete retained trajectories
do not imply continuation acceptance.

| Checkpoint | Tensor Comparisons | Changed Tensors | Failed Tensors | Scalar/Structure Differences | Predicate |
| --- | ---: | ---: | ---: | ---: | --- |
| 22 | 2,127 | 0 | 0 | 0 | accepted boundary |
| 23 | 2,167 | 120 | 120 | 16 | false; retained diagnostic only |
| 24 | 2,199 | 1,635 | 419 | 0 | rejected post-cached-root boundary |

Counts pool rank-specific comparisons across all eight ranks.
Checkpoint23's tensor failures are on rank 0; checkpoint24 has
failures on every rank. Three of 39 scalar-loss comparisons fail,
all at update24: selection, held-out DCLM and held-out StarCoder.
Both continuous and resumed trajectories pass the retained
diagonal-capture, outer-update-parity and output-work checks.
Boundary continuation and capture evidence therefore do not resolve
the post-cached-root training-state failure.

Canonical growing aggregate:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[6]`. This is the original validator's saved decision;
independent complete terminal reconstruction remains pending.
No tolerance, numerical source or graph policy is changed, and the
original case is not retried.

The same validator continues the planned diagnostic. Its fused
BF16/seven-iteration continuous24 command exits/cleans up zero at
10:26:44.356019 and is retained at 10:29:06.876894 as
`gpu_commands[21]`; interrupted22 is selected at 10:29:15.
At 10:30:12, the unchanged archived file-only reporter authenticates
22/54 commands, seven decisions, six accepted cases and this one
rejection. The remaining fused iteration/precision cases and
matched-null controls can test whether the failure is shared across
those configurations. Their decisions, exact restoration and the
independent original terminal must complete before successor handoff.
The seven-iteration interrupted22 command then exits/cleans up
integer zero at 10:31:37.504302 and is retained at
10:32:24.875591 as `gpu_commands[22]`. Resumed24 is selected at
10:32:32; the eighth case decision remains pending at 10:33:22.
The aggregate now retains 23/54 commands with the same seven
decisions, six accepted and one rejected.

The separately specified equal-LR successor still requires a distinct
accepted execution owner and per-configuration GPU gates. A rejected
configuration cannot enter full-budget tuning on these results.
Completed production Wave12 remains the latest quality/efficiency
evidence; no better whole-training optimizer or new SwanLab history
is established by this diagnostic.

#### Second Fused Training-Path Case Rejection

November 19 10:47:48.899487, Asia/Shanghai: the original validator
rejects `outer_leftdiag_fused_ns_tpu_q7_full_lr_lower`. This is the
seven-iteration BF16 configuration at matrix LR `0.04525483392`,
seed 0 and eight ranks, retaining the full 5,120-update schedule,
compilation and cached-root CUDA graphs. All three commands exit
and clean up with integer zero, and their complete trajectories
are retained by 10:36:55.

| Checkpoint | Tensor Comparisons | Changed Tensors | Failed Tensors | Scalar/Structure Differences | Predicate |
| --- | ---: | ---: | ---: | ---: | --- |
| 22 | 2,129 | 0 | 0 | 0 | accepted boundary |
| 23 | 2,155 | 114 | 114 | 20 | false; retained diagnostic only |
| 24 | 2,167 | 1,609 | 428 | 2 | rejected post-cached-root boundary |

Counts pool rank-specific comparisons across eight ranks.
Checkpoint23 has 111 tensor failures on rank0 and three on rank3.
Its twenty scalar/structure differences include sixteen graph-readiness
flags plus rank3 basis-step, pending-map, fallback and refinement-work
differences. At checkpoint24 every rank has failed tensors; rank3's
fallback/refinement counters remain different. Two of 39 scalar-loss
checks fail at24: C4 and held-out ProofPile. Both continuous/resumed
trajectories pass diagonal-capture, outer-update-parity and retained-work
checks. Canonical growing result is
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[7]`; complete independent terminal reconstruction is pending.

Five and seven BF16 NS iterations both fail the required post-root
continuation gate under the original training boundary. The changed
iteration count alone therefore has not supplied a passing fused
configuration. These short trajectories do not rank optimizer quality
or efficiency. The remaining ten-iteration, FP32 and matched-null
cases retain their planned commands and original tolerances.
No failure is erased, retried or converted into production authority.

The original validator next selects fused/BF16/ten-iteration
continuous24 at 10:47:57. That command exits/cleans up zero at
10:50:42.574680 and is retained at 10:53:13.437169 as
`gpu_commands[24]`. The diagnostic then has 25/54 retained commands,
eight decisions, six MuonT acceptances and two fused rejections.
All original diagnostic/waiter processes remain live; no successor
seal, transfer or production launch follows.

#### Third Fused Training-Path Case Rejection

November 19 11:12:11.838939, Asia/Shanghai: the original validator
rejects `outer_leftdiag_fused_ns_tpu_q10_full_lr_lower`, seed0,
matrix LR `0.04525483392`, BF16 NS operands and ten iterations.
The executed candidate retains the original trainer, eight ranks,
5,120-update schedule, compilation and cached-root CUDA graphs.
Continuous24/interrupted22/resumed24 commands exit and clean up with
integer zero at10:50:42,10:55:50 and10:58:44; their complete trajectories
are retained at10:53:13,10:56:39 and11:01:13.

| Checkpoint | Tensor Comparisons | Changed Tensors | Failed Tensors | Scalar/Structure Differences | Predicate |
| --- | ---: | ---: | ---: | ---: | --- |
| 22 | 2,127 | 0 | 0 | 0 | accepted boundary |
| 23 | 2,141 | 97 | 97 | 16 | false; retained diagnostic only |
| 24 | 2,191 | 1,630 | 460 | 3 | rejected post-cached-root boundary |

Counts pool rank-specific comparisons across eight ranks.
All97 failures at23 occur on rank0; the sixteen scalar differences
are graph-readiness flags. At24 every rank has failed tensors:
260,30,30,30,28,28,28,26 by rank0-7. Rank3 additionally retains
a group-d basis-step difference, fallback-counter difference and
refinement-work difference. Four of39 scalar-loss comparisons fail
at24: C4, selection loss, held-out DCLM and ProofPile.
Both trajectories pass captured-row, outer-update and retained-work
gates. Supporting artifact:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[8]`; configuration and execution digests are in `DETAILS.md`.
This is the original validator's retained decision. Complete independent
canonical-terminal reconstruction remains pending.

Five, seven and ten BF16 iteration budgets all fail the required
post-root continuation gate under the original training boundary.
Changing the iteration count within this grid has not supplied a
passing fused BF16 configuration. The FP32 and matched-null cases
continue under unchanged tolerances. These bounded losses do not
establish optimizer quality, efficiency or production authority.

#### First FP32 Fused Training-Path Rejection

November 19 11:36:42.508156, Asia/Shanghai: the original validator
rejects `outer_leftdiag_fused_ns_f32_q5_full_lr_lower`. It uses FP32
NS operands, five iterations, matrix LR `0.04525483392`, seed0 and
eight ranks, retaining the original 5,120-update schedule, compiled
model/root paths and cached-root CUDA graphs. All three commands
exit and clean up with integer zero. Their complete continuous24,
interrupted22 and resumed24 trajectories are retained at11:17:41,
11:21:05 and11:25:34.

| Checkpoint | Tensor Comparisons | Changed Tensors | Failed Tensors | Scalar/Structure Differences | Predicate |
| --- | ---: | ---: | ---: | ---: | --- |
| 22 | 2,123 | 0 | 0 | 0 | accepted boundary |
| 23 | 2,177 | 142 | 0 | 16 | false; retained diagnostic only |
| 24 | 2,199 | 1,638 | 377 | 0 | rejected post-cached-root boundary |

Counts pool comparisons across all eight ranks. The checkpoint24
failure counts by rank0-7 are244,23,23,19,18,18,18,14.
Checkpoint23's changed tensors all pass the unchanged tensor tolerance;
its sixteen differences are cached-root graph-readiness flags.
One of39 scalar-loss checks fails: update24 C4 is
`6.895737120522038` resumed versus `6.895666985019262` continuous.
Both trajectories pass diagonal-capture, outer-update-parity and
retained-output-work checks.

Canonical growing aggregate:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[9]` and `gpu_commands[27:30]`. Source/configuration,
command/log identities and receipt hash at observation are in
`DETAILS.md`. This is the original validator's saved decision;
complete independent canonical-terminal reconstruction is pending.

FP32 operands with five NS iterations do not supply a passing fused
continuation configuration in this lower-LR case. Compared with the
BF16/five-iteration case, fewer failed tensors or scalar checks do not
override the binary continuation gate or establish a causal explanation.
FP32/seven and ten iterations and matched-null controls can still
distinguish iteration/precision effects from effects shared by the
output-statistic path. The validator selects FP32/seven-iteration
continuous24 at11:36:51; those later decisions remain pending.

The FP32/seven continuous24 command exits/cleans up zero at
11:39:30.966585 and its trajectory is retained at11:41:55.700010.
Interrupted22 exits/cleans up zero at11:44:29.316465 and is retained
at11:45:18.720802. The aggregate now retains32/54 commands and
the same ten decisions, six accepted and four rejected. Neither
command completion nor its short losses establish case acceptance.
The existing Bash publisher completes its scheduled split-document
refresh at11:46:13, reporting these32 commands and the accepted
actual-LR queue's saved zero-phase waiting state. It is not restarted.

The separately accepted actual-LR queue still waits for the original
complete result, independent terminal, owner/waiter absence and exact
restoration. Compiled-map localization remains planned and unimplemented
under [NS Compiler Boundary Audit and Follow-Up](#ns-compiler-boundary-audit-and-follow-up).
No rejected case enters full-budget tuning on these results.
Production comparisons still require five shared536,870,912-token
windows, all22 exact evaluations, fair full-budget LR neighborhoods,
fresh paired seeds and independently verified full-resolution private
SwanLab histories. No new production wave, cloud import, quality or
whole-training efficiency claim follows from this diagnostic.

#### Second FP32 Fused Training-Path Rejection

November 19 12:00:40.225960, Asia/Shanghai: the original validator
rejects `outer_leftdiag_fused_ns_f32_q7_full_lr_lower`, seed0,
matrix LR `0.04525483392`, FP32 NS operands and seven iterations.
The original5,120-update schedule, eight-rank trainer, compilation and
cached-root CUDA graphs remain unchanged. Complete continuous24/
interrupted22/resumed24 trajectories are retained11:41:55/11:45:18/
11:49:50; all commands execute and clean up with integer zero.

| Checkpoint | Tensor Comparisons | Changed Tensors | Failed Tensors | Scalar/Structure Differences | Predicate |
| --- | ---: | ---: | ---: | ---: | --- |
| 22 | 2,137 | 0 | 0 | 0 | accepted boundary |
| 23 | 2,161 | 117 | 0 | 16 | false; retained diagnostic only |
| 24 | 2,197 | 1,633 | 379 | 0 | rejected post-cached-root boundary |

Counts pool comparisons across all eight ranks. Checkpoint24 failure
counts by rank0-7 are245,22,20,20,19,18,19,16. Checkpoint23's changed
tensors remain within tolerance; its sixteen scalar differences are
cached-root graph-readiness flags. One of39 loss checks fails at24:
held-out ProofPile is `7.218519595305416` resumed versus
`7.2186112514200085` continuous. Capture/update/retained-work checks
pass in both trajectories.

Canonical growing aggregate:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[10]` and `gpu_commands[30:33]`. Configuration and log digests
are in `DETAILS.md`. Independent complete terminal reconstruction is
pending; the original decision is preserved without retry.
The original validator selects fused FP32/ten continuous24 at12:00:48.
Both tested lower-LR fused FP32 budgets now fail the required
post-root gate. This rejects those configurations' continuation
acceptance; it does not rank full-budget optimizer quality or explain
the first differing operation. FP32/ten and all matched-null controls
retain their planned commands and original tolerances. No production
or cloud-history result follows.

The next fused FP32/ten continuous24 command exits and cleans up with
integer zero at12:03:34.132192 and is retained at12:06:03.850729 as
`gpu_commands[33]`. This brings progress to34/54 commands with the
same eleven decisions. Interrupted/resumed trajectories and its
numerical decision remain pending. The12:08:22 scheduled journal
refresh reports this progress and the actual-LR queue's12:05:43
zero-phase waiting state; no competing refresher is started.

#### Third FP32 Fused Training-Path Rejection

November 19 12:25:05.925653, Asia/Shanghai: the original validator
rejects `outer_leftdiag_fused_ns_f32_q10_full_lr_lower`, seed0,
matrix LR `0.04525483392`, FP32 NS operands and ten iterations.
The original5,120-update schedule, eight-rank trainer, compilation
and cached-root CUDA graphs remain unchanged. All three continuous24/
interrupted22/resumed24 commands exit and clean up with integer zero;
the complete resumed trajectory is retained by12:14:01.

| Checkpoint | Tensor Comparisons | Changed Tensors | Failed Tensors | Scalar/Structure Differences | Predicate |
| --- | ---: | ---: | ---: | ---: | --- |
| 22 | 2,139 | 0 | 0 | 0 | accepted boundary |
| 23 | 2,185 | 160 | 18 | 16 | false; retained diagnostic only |
| 24 | 2,177 | 1,619 | 378 | 0 | rejected post-cached-root boundary |

Changed tensors contain any bitwise inequality. Failed tensors exceed
the unchanged elementwise `rtol=1e-5`, `atol=1e-6` gate; scalar losses
use the same values with `math.isclose`. Checkpoint23 is the first
resumed update and retains process-local graph rewarming as diagnostic
evidence. Checkpoint24 is the required post-cached-root continuation
boundary. These roles are fixed by the accepted protocol.

Counts pool comparisons across all eight ranks. Checkpoint24 failed
counts by rank0-7 are245,23,21,19,19,19,18,14. Checkpoint23's
eighteen failed tensors are on rank0; sixteen graph-readiness flag
differences remain diagnostic-only. Two of39 scalar-loss comparisons
fail at24: StarCoder is `8.187337121240095` resumed versus
`8.187177337889851` continuous; ProofPile is `7.210719971917342`
versus `7.210626144658078`. Capture/update/retained-work checks pass
in both trajectories.

Canonical aggregate:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[11]` and `gpu_commands[33:36]`. Configuration and log
identities are recorded in `DETAILS.md`. Progress reaches36/54
retained commands and12/18 decisions, six accepted/six rejected.
All six lower-LR fused BF16/FP32 five/seven/ten-iteration cases now
reject the required post-root and loss gates; all six MuonT cases
pass. Matched-null decisions and independent complete terminal
reconstruction are still pending. These bounded checks do not rank
full-budget quality or explain the first upstream difference.

At12:25:31 all original controller, validator, terminal waiter,
actual-LR queue and single document-monitor generations remain live.
The actual-LR queue records zero phases/noerror at12:20:44.
The existing monitor refreshes both journal blocks at12:28:20.
No diagnostic is restarted, and no rejected case is admitted to
production on this evidence.

At12:30:29 the first matched-null BF16/five-iteration continuous24
trajectory enters the aggregate as command37/54. Decisions remain
12/18, six accepted/six rejected; matched-null interrupted/resumed
trajectories and its case decision are pending.
The interrupted22 trajectory is subsequently retained12:33:56,
bringing progress to38/54 commands with unchanged decisions.
Fresh12:34:14 process checks verify all five original generations
live; the actual-LR queue still waits with zero phases/noerror.

#### First Matched-Null Trajectory Completion

November 19 12:38:29.384163, Asia/Shanghai: the original aggregate
retains the resumed-to-update24 trajectory for
`null_leftdiag_fused_ns_tpu_q5_full_lr_lower`, seed0, matrix LR
`0.04525483392`, BF16 NS operands and five iterations. This is the
matched-null control for the corresponding fused output-diagonal
case. Its continuous24, interrupted22 and resumed24 commands retain
integer-zero execution and cleanup, with marked descendants absent.
The resumed command exits at12:36:01.708281.
The original5,120-update schedule, compiled trainer, fixed-age root
installation and cached-root CUDA graphs remain unchanged.

Progress is39/54 retained commands and12/18 numerical decisions,
six accepted MuonT and six rejected fused cases. The first matched-null
decision is pending; its completed trajectories supply no continuation
acceptance or full-budget loss comparison.
Evidence is
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`gpu_commands[36:39]`; configuration and retained-history identities
are in `DETAILS.md`.
The file/archive-authenticating reporting helper confirms this progress
with no saved error and no terminal outcome. It does not reconstruct
the checkpoint tensors.

At12:48:56 the original controller, validator, terminal waiter,
actual-LR queue and sole Bash monitor remain live with unchanged
generations. The queue's12:35:46 saved state has zero of six phases
and no error; its54-configuration/162-command diagnostic has not
launched. The scheduled publisher refreshes both recurring blocks
at12:48:15, then retains its900-second sleep cycle.
Five focused split-document preservation, ambiguous-boundary refusal,
monitor-retry and digest-summary CPU tests pass with CUDA hidden.
No original process is restarted or competing terminal invoked.
Wave12 remains the latest analyzed production result with independently
verified private full-resolution SwanLab histories. The overall research
goal remains active; no superior whole-training optimizer is established.

#### First Matched-Null Training-Path Rejection

November 19 12:49:29.566776, Asia/Shanghai: the original validator
rejects `null_leftdiag_fused_ns_tpu_q5_full_lr_lower` at the required
post-cached-root update24 boundary. The executed seed0,
matrix LR `0.04525483392`, BF16 operands, five NS iterations,
compiled trainer and original5,120-update schedule are unchanged.

| Checkpoint | Tensor Comparisons | Changed Tensors | Failed Tensors | Scalar/Structure Differences | Predicate |
| --- | ---: | ---: | ---: | ---: | --- |
| 22 | 2,141 | 0 | 0 | 0 | accepted boundary |
| 23 | 2,173 | 140 | 140 | 16 | false; retained diagnostic only |
| 24 | 2,187 | 1,620 | 409 | 0 | rejected post-cached-root boundary |

Counts pool all eight ranks. Update24 failed counts by rank0-7 are
249,26,26,26,23,23,19,17. All140 failed update23 comparisons are
on rank0; its sixteen graph-readiness flag differences remain
diagnostic-only. The unchanged continuation tolerances apply.
Two of39 scalar-loss comparisons fail at24: C4 is
`6.896858772653531` resumed versus `6.896973391828133`
continuous; StarCoder is `8.18233799331063` versus
`8.18245009134759`.
Both trajectories pass diagonal-capture parity, outer-update parity,
retained-work checks and `null_identity_outer_passed`.

The matched-null case fails even when its outer update is identity.
This rules out treating continuation rejection as unique to applying
the nonidentity fused output-diagonal update. It does not identify
the first differing operation or establish a compiler cause; root
work, capture and other executed paths still require localization.
The other five matched-null cases and independent complete terminal
reconstruction remain pending.

Evidence:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[12]` and `gpu_commands[36:39]`; hashes are in `DETAILS.md`.
Progress is39/54 commands and13/18 decisions, six accepted/seven
rejected. At12:52:35 all five original process generations remain
live. The actual-LR queue's12:50:48 receipt still has zero phases
and no error. No case is retried, no terminal is duplicated, and
no new production or SwanLab acceptance follows.

#### Seven-Iteration Matched-Null Trajectories Retained

November 19 13:02:40.023400, Asia/Shanghai: the original diagnostic
aggregate retains commands40-42 for
`null_leftdiag_fused_ns_tpu_q7_full_lr_lower`. Executed source:
`PVC/candidates/ns-training-path-owner-closure-corrected-20261119T0736`;
configuration:
`experiment/configs/null_leftdiag_fused_ns_tpu_q7_full_lr_lower_0p5m_seed0.json`.
Seed0, matrix LR `0.04525483392`, BF16 operands, seven NS iterations,
eight ranks, compiled trainer, CUDA graphs and the original5,120-update
schedule are unchanged. These are bounded correctness trajectories,
not a production wave.

| Command | Trajectory | Retained Endpoint | Command/Cleanup Exit | Command Receipt Time |
| --- | --- | ---: | --- | --- |
| 40 | continuous | 24 updates; 12,582,912 tokens | 0/0 | November 19 12:52:14.529962 |
| 41 | interrupted | 22 updates; 11,534,336 tokens | 0/0 | November 19 12:57:12.989378 |
| 42 | resumed from22 | 24 updates; 12,582,912 tokens | 0/0 | November 19 13:00:10.907020 |

Each command records marked-descendant cleanup and retained checkpoint/
history identities. The growing receipt has42/54 commands but only13/18
case decisions: six accepted MuonT cases and seven rejected fused/null
cases. The seven-iteration matched-null numerical decision is pending;
command exits do not establish continuation, capture/update/work or
scalar-loss acceptance. Twelve commands and five decisions remain.
The prior five-iteration matched-null rejection and all fused
rejections remain unchanged.

Evidence:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`gpu_commands[39:42]`, with the exact distributed commands and retained
trajectory identities; recurring receipt-backed reports are below
and in `DETAILS.md`.
At13:10:43 exact process-generation checks find the original controller,
validator, terminal waiter, actual-LR queue and sole Bash document
monitor live. The saved actual-LR queue observation at13:05:49 remains
`waiting_for_original_training_path_owners`, zero of six phases and
no error. Complete original terminal reconstruction and exact
restoration remain pending; no duplicate terminal or GPU successor
is launched.

The existing900-second Bash monitor refreshes both journals successfully
at13:08:09 without restart. Continue recording new hypotheses,
acknowledged experiments, numerical decisions, complete-wave large-window
comparisons and verified imports in this file. Wave12 remains the latest
complete, analyzed and privately cloud-verified production wave.
No new full-budget result or SwanLab import follows from these probes;
the goal of outperforming independently tuned MuonT in quality and
whole-training efficiency remains open.

#### Second Matched-Null Training-Path Rejection

November 19 13:13:47.113010, Asia/Shanghai: the original validator
rejects `null_leftdiag_fused_ns_tpu_q7_full_lr_lower` under the same
seed0, LR `0.04525483392`, BF16/seven-iteration configuration and
executed source identified above. This is the numerical decision
following the earlier pending three-trajectory observation.

| Checkpoint | Tensor Comparisons | Changed Tensors | Failed Tensors | Scalar/Structure Differences | Predicate |
| --- | ---: | ---: | ---: | ---: | --- |
| 22 | 2,143 | 0 | 0 | 0 | accepted boundary |
| 23 | 2,195 | 172 | 172 | 16 | false; retained diagnostic only |
| 24 | 2,197 | 1,636 | 448 | 7 | rejected post-cached-root boundary |

Counts pool all eight ranks. Update24 failed tensor counts by rank0-7
are259,29,29,27,27,26,26,25. The seven scalar/structure differences
are on ranks3/4: basis-step, fallback and extra-work counters differ,
with one pending-basis structure difference on rank4. All172 failed
update23 comparisons are on rank0; its sixteen graph-readiness flag
differences remain diagnostic-only. Original tolerances are unchanged.
One of39 scalar-loss comparisons fails at24: ProofPile is
`7.305592693306587` resumed versus `7.3054980981299` continuous.
Both trajectories pass diagonal-capture parity, outer-diagonal
update parity, retained-work and `null_identity_outer_passed`.

The second identity-outer rejection confirms that these two tested
matched-null configurations share the required continuation failure
without applying nonidentity output normalization. It does not isolate
a compiler cause, explain the first differing operation, or establish
optimizer quality or efficiency. Complete original reconstruction,
the other four matched-null decisions, actual-LR gates and the
separately owned localization experiment remain required.

Evidence:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`results[13]` and `gpu_commands[39:42]`.
The receipt records42/54 commands and14/18 decisions, six accepted/
eight rejected. This changes neither the preserved prior rejections
nor the pending terminal/restoration and production gates.
No new production run or SwanLab import is claimed.

November 19 13:19:03.434189 progress follow-up: command43 retains
`null_leftdiag_fused_ns_tpu_q10_full_lr_lower` continuous24 on the
same candidate, seed0 and LR, with ten BF16 NS iterations. Its command
receipt time is13:16:38.401748; command/cleanup exits are0/0 and marked
descendants are absent. The trajectory has24 updates and12,582,912
tokens. Interrupted22/resumed24 and the numerical decision remain
pending; this is not an accepted case. Evidence:
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
`gpu_commands[42]`. Progress is43/54 commands,14/18 decisions;
eleven commands and four decisions remain unretained.

#### Training-Path Execution Failure and Restoration

November 19 13:23:14.245028, Asia/Shanghai: the original controller
records `needs_attention`, validator exit1/cleanup1 and
`keeper_restore: 0`. The diagnostic retains44/54 successful command
trajectories and14/18 numerical decisions, six accepted/eight rejected.
Command44 retained the ten-step BF16 matched-null interrupted22
trajectory, command/cleanup0/0, at13:22:55.246533; its command receipt
time is13:21:39.632714. The subsequent resumed24 command fails before
training and is preserved only in its failed execution sidecar/log.
It must not be counted as a successful trajectory or case decision.

Rank0 fails the entry ownership check in `experiment.train.main`
through `require_rank`/`require_controller`/`marker_processes`:
`RuntimeError: Marked process generation changed during handoff`.
The distributed launcher terminates the other ranks. The sidecar at
13:22:55.219043 records integer command/cleanup exits1/1 with marked
descendants absent. The validator then rejects the nonzero command
under its unchanged execution-evidence gate:
`ValueError: NS full-model command, exact process generation, cleanup or log identity differs`.
This is an execution/ownership failure, not a numerical acceptance
or a new optimizer-loss result. The precise changing process has not
been independently isolated.

Artifacts:
`PVC/validation/ns_training_path_20261119/null_leftdiag_fused_ns_tpu_q10_full_lr_lower_resumed.execution.json`,
the matching `.log`,
`PVC/run-preparation/validation-ns-training-path-gpu-20261119.json`,
and `PVC/run-preparation/ns-training-path-state-20261119.json`.
All executed source/configuration identities and previous rejections
are preserved. No original run is restarted or receipt overwritten.

Read-only process checks at13:25:10 find original controller2099685
and validator2100106 absent, with the existing terminal waiter,
actual-LR queue and single document monitor still live. Fresh
`nvidia-smi` at13:26:23 matches the eight occupied contexts in the
restoration receipt. The accepted helper's keeper observation at13:28
also matches every recorded process generation, physical device and
occupied context for restored keeper2768623 and ranks2768771-2768778.
These observations do not supply missing command/decision coverage.

Archived handoff observation at13:26:33.933279:
`PVC/run-preparation/ns-compiler-localization-predecessor-failed-observation-20261119T1326.json`;
`ready: false`, original controller/validator absent, original terminal
waiter and actual-LR queue live. The existing waiter owns any canonical
terminal decision; no duplicate terminal, seal, GPU successor or
production run is issued. The actual-LR queue's saved13:20:51
observation predates failure and remains zero of six phases.

The original one-shot failure cannot provide complete54-command/
18-decision predecessor evidence. Next, retain the canonical failed
follow-up outcome, identify the ownership-observation failure with CPU
process-lifecycle evidence, and prepare a separately accepted corrected
execution identity without weakening process, continuation or
numerical gates. Existing source archives and failed attempts remain
unchanged. This does not waive the fused/null continuation rejections,
actual-LR validation, full-budget comparisons or verified cloud histories.

The scheduled journal refresh completes at13:27:37 with both manual
prose hashes unchanged and all prior milestones preserved. The same
900-second Bash monitor continues without restart. No new SwanLab
import, full-budget quality or whole-training efficiency claim follows.

#### Failed Follow-Ups and Marker-Exit Correction

November 19 13:41, Asia/Shanghai. The original read-only terminal waiter
records `needs_attention` at13:32:52.417306:
`RuntimeError: Training-path original owners exited without complete cleanup/restoration`.
Its last successful live-owner observation remains13:17:51; it has
nineteen retained observations and issues no terminal command.
The keeper is the restored eight-GPU holding workload; its restoration
command exits0. Completion also requires validator execution/cleanup
exit codes0/0, complete54-command/18-case evidence, absent descendants
and verified restoration. The validator's execution/cleanup exit
codes are1/1 and its diagnostic coverage is incomplete.
The actual-LR queue subsequently records `needs_attention`
at13:36:09.702630:
`RuntimeError: Training-path actual-LR predecessor complete original terminal/report is unavailable`.
It retains zero of six phases. Fresh13:37 process checks find both
follow-ups and the original controller/validator absent. Their saved
older live-owner observations are historical. No one-shot process is
restarted and no canonical terminal, successor seal or GPU owner is issued.

Evidence:
`PVC/run-preparation/ns-training-path-terminal-followup-20261119/state.json`
and `followup.log`;
`PVC/run-preparation/ns-training-path-actual-lr-queue-20261119/state.json`
and `queue.log`. The original aggregate remains44/54 commands and14/18
decisions, six accepted/eight rejected, with its failure evidence
unchanged. The sole900-second Bash document monitor remains live.

The original resumed-training command failed its rank-entry ownership
check at13:22:55 on `Marked process generation changed during handoff`.
That failure stopped the validator and left incomplete results, causing
the waiter and actual-LR queue failures above.
CPU experiment: can a legitimate marked helper exit between reading its
environment and rechecking its identity produce that error?
Four deterministic lifecycle regressions fail on the previous
helper: removed and zombie processes raise the same
`Marked process generation changed during handoff`; disappearance while
reading the environment can also omit a newly discovered live child.
The regression XML is
`PVC/run-preparation/test-marker-exit-regression-before-20261119T1337.xml`.
This reproduces a sufficient failure mechanism, but the original GPU
log does not identify the changing PID or establish its exact cause.

The local `tracking/owned_output_handoff.py` now rescans the complete
process set after transient disappearance or unreadable environments,
up to three attempts. It discards partial results, retains identity
checks across attempts and rechecks all marked results before returning.
Persistent churn/unreadability, live marker removal, PID substitution,
changed commands, ancestry, groups or sessions still raise.
A new unauthorized marked worker found during rescan still prevents
keeper handoff. Errors include the relevant PID when available.
All36 focused ownership cases pass, including a real Linux subprocess
exiting after its environment read. Evidence:
`PVC/run-preparation/test-marker-exit-regression-final-20261119T1340.xml`.
Required scoped source lint passes with offline tools.
This is a local CPU-tested correction. The issued controller helper
and all accepted deployment archives remain unchanged; deployment and
GPU validation require a separate accepted execution identity.

Final verification, November19 13:51: the complete selected caller/
ownership/reporting suite passes436tests in294.80seconds, with no
failures/errors/skips:
`PVC/run-preparation/test-marker-exit-callers-final-20261119T1346.xml`.
An earlier attempt passes430tests and fails two real-`/proc` caller
scans on repeated unreadable environment and process reparenting.
The focused three-case recheck and the complete rerun pass; the earlier
failures remain evidence of residual process-lifecycle risk.
Five split-journal/monitor tests also pass. Required source and journal
content lint passes; the existing500KiB document-size policy still
fails and is unchanged.
The local corrected source and tests are preserved separately in
`PVC/run-preparation/source-marker-exit-local-correction-20261119T1349.tar`;
this snapshot is not an accepted deployment archive.
The original single Bash monitor publishes both documents at13:45:46,
including both failed follow-up states, then sleeps900seconds.

#### Distinct Marker-Exit Recovery

November19 14:22, Asia/Shanghai. A separately named diagnostic is prepared
to test the bounded marker-rescan correction through the complete
training-path protocol. The original44-command/14-decision failure,
its failed resumed command and both zero-launch follow-up failures
remain unchanged. None supplies complete diagnostic acceptance.

The fresh candidate is
`PVC/candidates/ns-training-path-marker-exit-recovery-20261119T1415`.
All353 candidate source/configuration entries match the accepted parent
except `experiment/fused_attribution_owner.py`, which selects the distinct
recovery owner. The source inventory retains149 parent helpers; only
`owned_output_handoff.py` is replaced by the locally tested three-attempt
complete-rescan correction. Namespaced orchestration is derived from the
accepted parent using Python AST transformations. Numerical reader bytes,
all eighteen family/operand/NS-budget configurations, actual lower learning
rates, compilation, cached-root CUDA graphs and tolerances are preserved.
Controller, command, output and canonical terminal identities are separate
from the consumed original attempt.

The focused local recovery suite passes22cases in63.71seconds.
The separately staged archive passes103cases in66.67seconds:
45 inherited lifecycle/physical-allocation/terminal cases,
36 measured handoff cases and22 recovery cases. These cover retained
rejections, incomplete coverage, owner/descendant absence, candidate-only
changes, complete source closure and frozen evidence-inventory binding.
Source lint passes. Fresh archived-entry verification succeeds.
The failed-attempt preservation seal authenticates307 retained evidence
files and704 checkpoint-file identities, plus exact restored-keeper
identity. Checkpoint digests are retained from the byte-pinned original
aggregate; large tensor payloads are not freshly hashed or rederived.
The seal explicitly grants no completion, numerical waiver, production
or GPU execution authority. An independent archived seal check passes.

The new source is frozen and the issued-owner check passes by14:22.
GPU launch is unissued at this observation. The next measured result
must come from all54 fresh commands and18 retained numerical decisions,
followed by the recovery's independent canonical terminal. Existing
fused/null continuation rejections remain recorded even if recovery
execution completes. Actual-LR and compiler-localization successors
require separately bound predecessor evidence; their failed original
queues will not be restarted.

Wave12 remains the latest completed production comparison. No new
full-budget loss history or SwanLab import is produced by these CPU,
source or preservation checks. No optimizer is established as better
than independently tuned MuonT in both quality and whole-training time.
The existing900-second Bash journal monitor is retained without restart;
the recovery reporter is added to its receipt-backed recurring block below.
Exact commands, hashes and process receipts are recorded in `DETAILS.md`.

#### Recovery Launch and Failed Resumed Entry

November19 14:36, Asia/Shanghai. The recovery diagnostic launched at
14:24:21.955035. The launcher acknowledged controller2855659 with Linux
process-start counter940361750; its direct child validator2855940 has
start counter940364559. Read-only process observations at14:25 and14:31
find those exact generations live. The controller records transfer of
the eight-GPU allocation from the holding workload to the diagnostic.

The first case uses MuonT with five Newton-Schulz iterations in BF16,
the original lower learning rate and the5120-update schedule truncated
only at the diagnostic endpoint. Its continuous run finishes24updates,
12,582,912tokens. Execution and descendant cleanup both exit0 at
14:27:22.187086; the validator authenticates its manifest, scalar history
and eight-rank checkpoints, publishing the retained trajectory at
14:29:47.835807. The interrupted run finishes22updates,11,534,336tokens;
its retained trajectory is published14:32:47.066663. These are two
separate successful commands in the new recovery cohort. Each numerical
case requires continuous24, interrupted22 and resumed24, so no case
decision follows from those two trajectories alone.

The third command, resumed from checkpoint22 to update24, fails its
rank-entry ownership check before training. The log now identifies
`Marked process generation changed during handoff: pid=2873161`,
the same error forPID2873160 and
`Marked process environment changed during handoff: pid=2873160`.
The failed launcher2872833/940413116 exits1; descendant cleanup also
returns1, with marked descendants absent, at14:33:11.287920.
The aggregate records `needs_attention`,2/54successful commands and
0/18case decisions at14:33:11.310617. It excludes the failed resumed
command from retained trajectories. The controller records the failure
and keeper restoration at14:33:29.528017.

Read-only archived-helper checks at14:35 find controller/validator
handles absent and no validator-marked descendants. Restored keeper
2873208/940414920 and ranks2873348-2873355 match the restoration receipt
and all eight original GPU contexts. The keeper is the holding workload,
not a training run. No canonical complete terminal is issued, and this
consumed recovery identity will not be restarted.

This GPU attempt shows that the CPU-tested complete-rescan correction
does not resolve every live process-observation transition. The named
PIDs are absent after cleanup; their exact commands and ancestry were
not retained in the exception. It remains unknown whether the failures
were transient fork/exec, reparenting, marker disappearance or another
generation change. The next experiment is a CPU process-lifecycle
reproducer and read-only structured failure attribution that records
the relevant before/after identities, without accepting PID reuse,
changed ancestry or partial process absence. Any deployment requires
a new accepted source archive and unused execution namespace. Original
numerical rejection tolerances and continuation gates remain unchanged.

All215 journal/reporting tests pass in42.59seconds, and the final
failure-reporting rerun passes215cases in39.63seconds; nine focused
monitor/recovery/terminal-report tests pass. Required source and document
content checks pass; the existing500KiB document-size policy still fails.
The receipt-backed recovery section is now included in both recurring
journals, with saved process observations explicitly dated. The original
single Bash monitor remains live and sleeps900seconds between updates.
Wave12 remains the latest full-budget analyzed/SwanLab-verified production
wave; this failed short diagnostic supplies no new optimizer-quality
comparison, whole-training efficiency result or cloud import.

The scheduled monitor publishes the failed recovery evidence to both
journals at14:39:35 and then returns to its900-second sleep. A separate
guarded manual refresh also completes. Both updates preserve the manual
acceptance, launch and failure records. Only the monitor and restored
holding workload remain live; no diagnostic owner or foreground test
session remains at this checkpoint.

The [numerical protocol](#training-path-numerical-protocol-cpu-check)
requires continuous24/interrupted22/resumed24 trajectories and eight-rank
checkpoint22/24 plus scalar-loss comparisons at the original tolerances;
checkpoint23 remains diagnostic. A canonical independent terminal is a
source-bound reconstruction receipt for a complete attempt. A successor
seal authenticates that receipt, retained decisions, absent owners and
exact restoration before any allocation transfer. Neither exists for
this failed attempt. The next experiments remain separately gated:

1. Prepare a distinct corrected training-path execution from the
   unchanged numerical protocol. Authenticate the failed original
   attempt, both failed follow-ups, owner/descendant absence and exact
   keeper restoration. Bind the corrected helper to a new immutable
   source archive and unused candidate/controller/output identities.
   Never fill missing original trajectories or reinterpret its rejections.
2. Complete a separately issued18-case/54-command training-path
   diagnostic at the existing tolerances, with compilation and CUDA
   graphs enabled. Verify complete numerical decisions and a distinct
   independent terminal before any successor seal. Original fused/null
   rejections remain evidence even if a later attempt differs.
3. Rebind compiler-boundary localization and actual-LR diagnostics to
   valid complete predecessor evidence under distinct successor
   identities. Compare historical continuous/resumed inputs, compiled
   repeats and cross-process outputs before selecting a numerical change.
4. Launch only candidates that pass those gates for full5,120-update
   independently tuned MuonT comparisons. Analyze complete shared
   536,870,912-token windows and exact shared evaluations, include
   whole-training time, then upload and verify every full-resolution
   history in SwanLab before choosing the next wave.

Wave12 remains the latest completed, large-window analyzed and privately
cloud-verified production wave. Compared with its independently tuned
`muont` reference, `outer_leftdiag_fused` has late shared-evaluation C4/
selection-loss deltas-0.001806/-0.002909 over11 evaluations and
adds536.9seconds to recorded whole-training time. See the
[five-window result](#wave-12-final-five-window-result-november-18-0154-0159).
None of the evaluated candidates has established superiority in both
quality and whole-training efficiency. No new production wave or
SwanLab upload is claimed at this checkpoint.

#### CPU Ownership Failure Attribution

November19 15:08, Asia/Shanghai. A separate CPU diagnostic now retains
structured evidence at the six rejecting boundaries in the consumed
recovery ownership scan. `tracking/owned_output_attribution.py` derives
the scan from the exact recovery helper, replacing only exception
construction. The complete-rescan logic, identity comparisons, marker
checks and rejection branches remain unchanged. This experiment adds
attribution; it does not correct the failed recovery or authorize
another GPU owner.

Deterministic fake-process tests retain both sides of live command
emptying/replacement, parent/group/session changes and PID substitution.
Live empty-environment and marker-removal cases still reject. Exhausted
environment scans still reject without returning partial absence.
Stat-generation drift retains its two bracketing observations. The
evidence contains PID, state, parent, process group, session, start time,
command hash/argument count and marker-presence boolean. It emits no
raw command arguments, marker values or environment contents. Delayed
process samples are explicitly separate reads after the failure; their
flags or eventual disappearance cannot establish the earlier cause.

The focused ownership/attribution/recovery suite first passes71 cases
in69.33seconds. Review identifies and corrects three attribution gaps:
the materialized handler dependency is copied and its location/bytes
are checked; environment byte counts include NUL separators; each
retained PID records whether it was previously confirmed marked.
Standalone import and tampered-handler rejection are exercised.
The reviewed source passes71 cases in68.87seconds. All36 unchanged
inherited ownership tests also pass against the corrected separately
materialized scan in1.77seconds; the earlier version passed in1.74seconds;
four split-journal/monitor cases pass in1.68seconds. Test XML retains
redacted structured failure snapshots. Source formatting, lint and AST
checks pass. The first48-case attempt retains46passes/twofailures:
the new marker-removal fixture failed to force a rescan and was corrected;
the unchanged real-Linux handoff case exhausted its scans on unrelated
process churn. The later runs pass, but that lifecycle failure remains
evidence. Numerical tolerances and test deadlines are unchanged.
Receipts, exact commands and source identities are in `DETAILS.md`.

A final review corrects handler authentication order: local source bytes
are hashed before execution, and only those verified bytes are compiled.
A tampered handler with a top-level file-writing side effect rejects
without creating the file. The final source passes71 cases in67.44seconds
and all36 inherited materialized-scan cases in1.69seconds. Review finds
no further defect in this delta. Earlier snapshots remain retained CPU
evidence; none is an accepted GPU deployment.

Exploratory CPU lifecycle controls on this host also observe a live
exit-teardown interval with empty command/environment files, successful
exec with marker loss, and reparenting with ancestry drift. All probe
children are reaped. These observations are retained only in the
execution transcript; no runnable source, raw-output file or independent
receipt was preserved. They are preliminary mechanism checks and
cannot supply predecessor acceptance or attribute the GPU failure.

The GPU recovery remains2/54successful commands and0/18decisions.
The failure PIDs2873160/2873161 had no contemporaneous structured
snapshots, so their transitions remain unknown. Current absence is
insufficient to distinguish exit teardown, exec, reparenting or PID
reuse. Another deployment requires separately archived/accepted source,
preserved failure evidence, exact keeper handoff and a fresh unused
execution identity. No consumed controller, validator or queue restarts.

The sole scheduled monitor publishes both journals at14:57:38, then
sleeps900seconds as child2915231. The manual journal addition holds the
shared publication lock after that refresh. Wave12 remains the latest
full-budget analyzed and privately SwanLab-verified production wave.
No new production loss window, optimizer-superiority result or SwanLab
upload follows from these CPU attribution tests.

#### Whole-Process Exit-Drain Successor

November19 15:35, Asia/Shanghai. A fresh successor tests whether
kernel-confirmed terminal exit can resolve the ownership-observation
failure while keeping live exec, PID reuse and ancestry changes rejected.
The numerical protocol remains eighteen lower-LR cases and54 commands:
continuous24/interrupted22/resumed24, unchanged5120-update schedule,
eight-rank checkpoints, original continuation/scalar tolerances,
compilation and cached-root CUDA graphs. Both consumed training-path
attempts, their failures and existing rejected decisions remain intact.

The separate `owned_output_exit_drain.py` derives the attributed scan.
Only empty-command transitions with unchanged ownership, or an empty
environment for a previously confirmed marked process, enter the drain.
Live teardown requires bracketing PF_EXITING flags, zero virtual memory,
empty command/environment files and exact PID/start/parent/group/session.
The drain waits at most one second for explicit directory disappearance
or identity-bound process-wide pidfd exit. A zombie leader alone is
insufficient. The scan discards partial results, rescans the complete
process set with retained identities and rechecks drained PIDs before
returning. PID substitution, changed ancestry, live memory/nonempty
replacement commands and incomplete process files still reject.

A retained CPU lifecycle probe runs15:26:56.033431-15:26:56.442899 on
Linux5.4.241-1-tlinux4-0017.7,x86_64. The process pidfd forPID2959175
is unreadable while its leader isZ/PF_EXITING and worker2959176 remains
live. It stays unreadable after nonleader exec preserves PID/start time;
two complete private-group rescans recover the live replacement.
After replacement exit, the same pidfd becomes readable and
waitid(WNOWAIT) confirms exit0. The probe also retains exit teardown,
leader exec, reparenting and complete cleanup evidence; all five children
are reaped. Four focused lifecycle-probe tests pass.
The runnable source, receipt and redacted raw evidence live in
`lifecycle-probe-pidfd-20261119T152700/`. This establishes synthetic
mechanisms, not the changing recovery PIDs' cause or production readiness.

The installed Python lacks `os.pidfd_open`; glibc is2.28. The host
supports the explicit x86_64pidfd_open syscall434. The helper checks the
procfs PID namespace and exact ownership before and after polling,
and fails closed on unsupported syscall/architecture. A real CPU
subprocess test confirms terminal pidfd readability before reaping.
Review corrects leader-Z acceptance, ambiguous missing-file absence,
post-poll identity binding and parent-source authentication order.
No accepted original/recovery source is edited.

Final local successor/exit-drain tests pass43cases in67.95seconds.
The new predecessor verifier preserves original44/54commands14/18decisions
and recovery2/54commands0/18decisions, without claiming completion or
fresh hashing of large checkpoint payloads. CPU archive validation for
`ns-training-path-exit-drain-reviewed-20261119T1536` passes124 cases
in72.96seconds, binding167 helpers and536 inventory entries.
Fresh archived-entry verification passes. The final descriptor-binding
revision adds a procfs `fdinfo` PID check before and after pidfd polling.
Fifteen focused exit-drain cases pass, including the real kernel path.
A fresh125-case candidate/archive is preparing; neither version is
GPU-issued at this observation. Earlier candidate/staging failures are retained: missing
ownership-test closure and attempting to derive from the already-derived
scan. The reviewed staging uses the exact byte-pinned recovery parent.
No new GPU owner, canonical terminal, full-budget production result or
SwanLab upload is issued.

November19 15:52 update, Asia/Shanghai. The final descriptor-bound
candidate `ns-training-path-exit-drain-pidfd-bound-20261119T1542`
passes125 staged cases in70.94seconds. Its CPU receipt at15:41:56 binds
167 helpers and536 inventory entries. These are CPU source/ownership
tests, separate from the eighteen numerical GPU cases:
`PVC/run-preparation/ns-training-path-exit-drain-20261119-cpu/pidfd-bound-20261119T1542/receipt.json`.
The separate15:42:41 predecessor seal byte-checks330 evidence files
and retains736 checkpoint-file identities with their saved payload
digests. The seal separately verifies the restored allocation-keeper
processes by PID/start time and physical GPU UUID. It does not reconstruct
large tensors or grant a numerical waiver. Both failed attempts and
their consumed follow-ups remain unchanged.

The issued source, archive and167 helper hashes bind the accepted CPU
receipt and separate seal at freeze15:44:16:
`PVC/run-preparation/ns-training-path-exit-drain-20261119-binding.json`.
Fresh issued-owner archive/check verification passes before the one-shot
start. Acknowledgment at15:47:40 identifies controller
`2985173/940861596`; direct validator `2985494/940864815`
acknowledges15:48:35. The controller records complete release of the
previous restored keeper. Fresh15:51 process checks confirm the exact
controller and validator PID/start-time pairs live and keeper2873208 absent.
The work directory is `PVC/validation/ns_training_path_exit_drain_20261119`.

The first `muont_ns_tpu_q5_full_lr_lower/continuous` log reaches
update24. At this checkpoint the authenticated aggregate still reports
`validating`,0/54 completed commands and0/18 decisions; a logged update
alone supplies no accepted execution, cleanup or case result.
The15:51 GPU query returns no compute contexts at that instant; it does
not prove persistent training occupancy. The `tpu` operand mode denotes
the original BF16 Newton-Schulz operand policy; this execution uses GPUs.
Each numerical case requires three commands, then one gate decision.
The18-case/54-command diagnostic and independent completion/restoration
verification report remain unfinished. Actual-LR/compiler successors require separately bound
complete evidence; no consumed queue is restarted.
Wave12 remains the latest full-budget analyzed and privately
SwanLab-verified wave, with all four5120-point histories checked after
upload. No new production comparison or upload is recorded,
and no candidate in this program has established both better quality and better
whole-training efficiency than independently tuned MuonT.

November19 15:55 follow-up. The receipt-only reporter authenticates
1/54 completed commands and0/18 decisions in the aggregate saved15:53:18.
The MuonT BF16/five-iteration continuous command has integer-zero
execution/cleanup exits and reaches24updates; the interrupted command is
starting. The command receipt time is15:50:46.919033. No eighteen-case
numerical decision or independent terminal is complete.
The same single Bash monitor publishes both recurring blocks15:52:07,
then sleeps900seconds as child2999359. The full journal suite passes216
cases in52.44seconds. Source lint and document content checks pass;
the pre-existing500KiB journal-size lint failure remains.

November19 15:58 follow-up. The corrected exit-drain reporter authenticates
2/54 completed commands and0/18 decisions. The interrupted22 receipt at
15:55:21.749080 has zero execution/cleanup exits. The resumed24 trajectory
now logs update24 with the same displayed training/C4/selection losses as
continuous24; this is a scalar-log observation. All-rank continuation
comparison, resumed execution/cleanup receipt and numerical decision
remain pending. No failed attempt is retrospectively completed.
Five focused report/split/monitor tests pass17.39seconds after the
report-label correction; source lint and journal content checks pass.

November19 16:00 successor decision, planned only. The existing failed
actual-LR queue cannot consume this fresh attempt's receipts. After
54/54 commands,18/18 decisions, independent canonical terminal and exact
owner/descendant absence plus keeper restoration, prepare a distinct
exit-drain-bound actual-LR predecessor, owner archive and queue. Preserve
the actual-LR plan of54 cases/162 commands and original numerical readers,
training schedule, compilation and CUDA graphs. Its candidate adapter
and measured handoff need separate CPU/archive acceptance and fresh
freeze/check before launch. Complete numerical coverage can retain
rejections; it does not authorize rejected configurations for production.
Only after that actual-LR terminal may a distinct compiler-localization
predecessor bind its evidence. Neither successor is implemented or launched
by this journal update; all old queues and one-shot claims remain consumed.

November19 16:06 checkpoint. All three MuonT BF16/five-iteration
trajectories are authenticated, including resumed24 with execution/
cleanup exits0/0 at15:57:58.312684. The aggregate saved16:00:24.710156
records3/54 commands and0/18 decisions. The validator remains live and
is reconstructing the retained eight-rank continuation comparisons.
Logged scalar equality supplies no accepted case before those checks.

Read-only review identifies three receipt-reporting defects: preparer
execution before pinned-byte authentication, overly narrow failed-command
reporting that can hide completed progress, and omitted controller-start/
validator-generation consistency checks. The new reporter now executes
the authenticated retained byte buffer, authenticates exact owner
generations/direct ancestry, and separates next-command failure details
from strictly accepted completed commands. The accepted parent reporter
and all issued workload sources remain unchanged. Sixteen focused
report/split/monitor regressions pass13.58seconds; source lint passes.
A full updated journal-suite rerun is underway. These fixes change
reporting, not optimizer arithmetic or the running diagnostic.

November19 16:11 verification. The full updated journal suite passes227
cases in52.74seconds; a final sixteen-case report/split/monitor rerun
passes13.34seconds. The same original monitor publishes both journals
16:10:23, authenticating3/54 completed commands and0/18 decisions with
the reviewed reporter. Source and content checks pass; the pre-existing
500KiB document-size lint failure remains. Exact controller/validator
PID/start pairs and the sole monitor remain live. No numerical case,
complete diagnostic terminal, new production wave or SwanLab upload
is reported. The active goal remains open.

November19 16:12 first numerical result. The aggregate saved
16:10:36.061890 retains accepted `muont_ns_tpu_q5_full_lr_lower`;
the reviewed receipt-only reporter independently authenticates its
three commands and gate predicates. Checkpoint22 has2,135 unchanged
tensor comparisons and checkpoint24 has2,147, across all eight ranks;
both have zero failed tensors or scalar/structure differences.
Checkpoint23 has2,167 unchanged tensor comparisons, but sixteen
`cached_root_graph_active` flags differ; it remains first-resumed
diagnostic-only evidence under the unchanged protocol. All39 scalar
loss comparisons pass. This fresh baseline reproduces required
continuation; it does not clear fused/null candidates, establish better
optimizer quality/efficiency, or complete the18-case diagnostic.
Current status is3/54commands,1/18decisions; no canonical terminal or
SwanLab import is recorded.

November19 16:24 continuation. The second MuonT BF16/seven-iteration
case now has all three continuous24/interrupted22/resumed24 commands
retained with zero exits; its checkpoint comparison is underway.
The diagnostic remains6/54 completed commands and1/18 decisions.
Fresh controller2985173/start940861596, direct validator
2985494/start940864815 and sole journal monitor1588227/start933927668
are live. No completion or new production window is inferred.

A distinct read-only terminal waiter is staged from the byte-pinned
original waiter, retaining all167 issued helpers unchanged and adding
only a new namespaced runner/test pair. Its accepted CPU archive has
24 passing cases in4.74seconds at16:19:15:
`PVC/run-preparation/ns-training-path-exit-drain-followup-cpu/20261119T1620/receipt.json`.
It waits on exact original owner PID/start/group/session and direct
validator ancestry every900seconds. Incomplete process files, substituted
generations, live changed commands and zombie leaders cannot authorize
absence. After full54-command/18-decision coverage, actual owner/
descendant absence and clean recorded restoration, it invokes only
the issued canonical independent terminal verifier once. Rejections
remain retained; no training, transfer or production is authorized.
Archived-entry checking and read-only launch review pass. Its distinct
launch is planned immediately after this entry; no waiter acknowledgment
is yet recorded here. The prior failed terminal waiter remains consumed.
The separate actual-LR predecessor is being implemented for this new
terminal; its seal cannot be issued from partial results.

#### Exit-Drain Waiter Launch Milestone

November 19 16:33, Asia/Shanghai. The distinct read-only terminal waiter
acknowledges at 16:25:28.123736 as `3071163/941088367`:
`PVC/run-preparation/ns-training-path-exit-drain-terminal-followup-20261119/intent.json`.
The receipt reporter checks the acknowledgment, 24-case CPU archive,
issued diagnostic binding and retained observation against their pinned
source and receipt digests. These checks establish saved-record provenance;
they do not reconstruct tensors or prove current liveness.
Its saved state is `waiting_for_original_owners`.
The first immutable observation at 16:25:32.104564 finds the exact
controller and direct validator live:
`PVC/run-preparation/ns-training-path-exit-drain-terminal-followup-20261119/observation-000000.json`.
It retains 6/54 completed commands and 1/18 numerical decisions from the
16:22:30 diagnostic aggregate. Both BF16 MuonT five/seven-iteration
cases have all three trajectories retained; only the five-iteration case
has an accepted gate decision. The seven-iteration comparison is pending.

A separate read-only `/proc` check at 16:32:24 verifies controller
`2985173/940861596`, direct validator `2985494/940864815`, waiter
`3071163/941088367` and sole Bash journal monitor
`1588227/933927668` live with their expected identities. This observation
does not establish persistent GPU occupancy or a new numerical result.
The waiter checks every 900 seconds and may invoke only the already-issued
independent terminal verifier after all 54 commands and 18 decisions,
controller/validator and marked-descendant absence, and exact restoration
of the recorded eight-GPU allocation keeper.
No terminal claim, keeper transfer, new GPU launch, numerical waiver or
SwanLab import follows from its acknowledgment. Failed predecessors and
consumed queues remain preserved.

Seventeen focused reporting/split-document/monitor cases pass in
15.649 seconds; the full journal suite then passes 228 cases in
53.753 seconds, with zero failures, errors or skips. Test receipts:
`PVC/run-preparation/test-endeavor-exit-waiter-20261119T1627.xml` and
`PVC/run-preparation/test-endeavor-exit-waiter-full-20261119T1628.xml`.
The sole Bash monitor finishes its scheduled refresh at 16:28:45 and
sleeps 900 seconds. That render loaded the waiter reporter before its
acknowledgment was pinned, so it retains a withheld-evidence notice.
The current reporter authenticates the saved waiter evidence independently.
The existing `update_endeavor.update` entry point refreshes the recurring
blocks in `ENDEAVOR.md` and `DETAILS.md` under the shared update lock;
its concurrent-edit guard preserves all prose outside those blocks.
New hypotheses, acknowledged launches, failures,
large-window comparisons and verified imports continue in this file.
The existing document-size lint limitation remains recorded.

#### Exit-Drain Partial-Failure Milestone

November 19 16:39, Asia/Shanghai. The exit-drain attempt stops
`needs_attention` at 16:33:03.101480. Its retained aggregate has
6/54 successful commands and 2/18 accepted baseline decisions:
`PVC/run-preparation/validation-ns-training-path-exit-drain-20261119-gpu.json`.
`muont_ns_tpu_q7_full_lr_lower` now passes all five required gates.
Across eight ranks, checkpoints 22/24 have 2,123/2,163 tensor
comparisons, zero failed tensors and zero scalar/structure differences.
All 39 scalar-loss comparisons pass. Checkpoint 23 has 2,161 tensor
comparisons and sixteen graph-state flag differences; it remains
first-resumed diagnostic-only under the unchanged protocol.
These two accepted baseline cases do not accept the other sixteen cases
or establish optimizer quality or whole-training efficiency.

The seventh command, `muont_ns_tpu_q10_full_lr_lower/continuous`,
fails before training. Its execution receipt at 16:33:03.073076 records
execution/cleanup exits 1/1 and marked-descendant absence:
`PVC/validation/ns_training_path_exit_drain_20261119/muont_ns_tpu_q10_full_lr_lower_continuous.execution.json`.
The retained log identifies rank 1's ownership check rejecting
`Exit drain lacks bracketed kernel teardown` for PID 3084811:
the bracketing samples change from `S` to `Z`, while both retain the
same PID/start/parent/group/session, PF_EXITING, zero virtual memory
and empty command/environment files. The executed helper rejects
mixed live/terminal samples before pidfd polling. This records the
specific refusal; it does not prove process-wide terminal exit or
justify changing the numerical tolerances. The failed command supplies
no retained training trajectory or ten-iteration numerical decision.

The controller records failure and keeper restoration at
16:33:20.078143:
`PVC/run-preparation/ns-training-path-exit-drain-20261119-state.json`.
Its execution/cleanup exits are 1/1, marked descendants are absent, and
keeper restoration exits 0. A fresh read-only check at 16:38:01 verifies
controller/validator absence and the restored keeper
`3084882/941134166`, all nine expected process generations and eight
exclusive contexts on the original GPU UUIDs. A separate 16:38:28 scan
finds no validator-marked processes and confirms the original waiter and
sole Bash monitor live. The waiter's saved 16:25:32 live-owner observation
remains historical; its next scheduled check has not yet been recorded.
The incomplete 6-command/2-decision attempt cannot produce the required
complete independent terminal or authorize the actual-LR successor.
No consumed owner or queue is restarted.

The guarded split-journal refresh finishes by 16:38:29 and preserves
the manual scientific and execution records byte-for-byte. It records
the authenticated partial failure and waiter acknowledgment in both
recurring blocks. The existing Bash monitor continues without restart.
Next experiment, planned only: separately test the observed mixed
live/terminal bracket against exact-identity process-wide pidfd evidence.
Keep zombie-leader/live-thread, nonleader exec, PID reuse, incomplete
files and changed ancestry rejected unless whole-process exit is proved.
Any correction needs a distinct accepted CPU/source archive, preserved
failure seal and fresh bounded GPU attempt. The issued failed source
remains unchanged. No new GPU work or SwanLab import is launched here.

November 19 16:42 follow-up. The terminal waiter's next 900-second
check stops `needs_attention` at 16:40:32.808225:
`PVC/run-preparation/ns-training-path-exit-drain-terminal-followup-20261119/state.json`.
Its retained error is
`RuntimeError: Training-path original owners exited without complete cleanup/restoration`.
The receipt reporter authenticates this failure while retaining the sole
16:25:32 immutable observation. Its saved live-owner states predate
the diagnostic failure. No second successful observation, canonical
terminal claim or independent complete report is issued.
Fresh 16:41:24 process checking finds the waiter absent and the exact
original journal monitor live. Keeper restoration exit 0 remains separate
from the failed diagnostic's execution/cleanup exits 1/1 and incomplete
6/54-command, 2/18-decision coverage. The waiter and failed diagnostic
are consumed; neither is restarted.

#### Mixed-Exit-Bracket Successor

November 19 17:05, Asia/Shanghai. The observed rank-1 `S`-to-`Z`
bracket from the consumed exit-drain failure motivates a distinct
bounded handoff correction. A mixed live/terminal pair now permits
another teardown observation only when both samples retain exact
ownership, PF_EXITING, zero virtual memory and empty command/environment
files. It does not establish exit. The scan returns only on process
directory disappearance or an identity-bound process-wide readable
pidfd with a terminal sample, then repeats the full ownership scan.
Zombie-leader/live-thread, nonleader exec, changed identity, live memory,
missing flags and incomplete files remain rejected.

The unchanged parent drain source stays byte-pinned. Six new helper/test
files implement the derived handoff, retained-failure predecessor,
fresh execution namespace, archived preparation and regressions.
Local successor/lifecycle tests pass 48 cases in 70.07 seconds.
The final closure-corrected archived acceptance passes 129 unique cases
in 76.40 seconds, with no failures, errors or skips: 45 owner/GPU-source,
36 inherited ownership, 21 lifecycle and 27 predecessor/source cases.
Required source lint and the independent archived-entry check pass.
Earlier incomplete staging remains retained and is not acceptance.

Accepted CPU receipt, timestamp 17:02:50.255340:
`PVC/run-preparation/ns-training-path-exit-transition-20261119-cpu/closure-corrected-20261119T1700/receipt.json`.
Candidate:
`PVC/candidates/ns-training-path-exit-transition-closure-corrected-20261119T1700`.
The receipt binds 173 helpers/542 inventory entries and source archive
`56173aac10f73158aadadef4003cbf8c626c4c026b951d87fcfb5693ac2eecae`.
CPU receipt digest:
`ad18edf9b4f230e78bd7eef2a97f44bde307a1c524192d37f23491257b6fc392`.
Tests XML digest:
`12fa743be4b8350c76ef3dccff9cd0d210bee5293f0b2568fc4de1c67b4fd10a`.

Executed acceptance uses hidden CUDA, the archived helpers, offline
pytest-timeout, 60-second per-test/600-second session limits and all
seven repository safety-marker exclusions. Numerical trainer/configuration
bytes, original lower LR, full 5,120-update schedule, 18 cases/54 commands,
readers, tolerances, compilation and cached-root CUDA graphs are retained.
Only the measured handoff adapter and fresh execution identities differ.
The separate predecessor seal retains both older consumed failures,
the 6/54-command, 2/18-decision exit-drain failure, its failed waiter,
source/archive/log identities and saved checkpoint digests. Current
checkpoint file identities are checked without rehashing large tensors.

Next gates are the archived preservation-seal check, distinct issued
freeze/check, controller-owned keeper transfer and fresh bounded GPU
launch. No direct keeper operation, consumed-owner restart, numerical
waiver, production launch or SwanLab import is authorized by CPU
acceptance. The sole existing Bash journal monitor remains active;
manual milestones are appended under the shared document lock.

November 19 17:14 launch milestone. The archived predecessor seal is
written at 17:05:15.594303 and independently checked before freeze:
`PVC/run-preparation/ns-training-path-exit-transition-predecessor-seal-20261119.json`.
Its digest is
`070227bc90c040368f7ce2a28ef897425a1f24639d870377954827dbe20f9b49`;
it retains 554 evidence records and 832 checkpoint file identities.
The issued binding freezes at 17:08:05.754835:
`PVC/run-preparation/ns-training-path-exit-transition-20261119-binding.json`.
It binds 54 commands/18 cases and 2,814 authenticated file identities;
only `experiment/fused_attribution_owner.py` differs from the accepted
numerical parent. Binding digest:
`18c9b21d16ff817059177e6746c3ae2246d39232396220c2b1deec805a33ca3d`.
Issued source archive digest:
`5defe28731900a9c1e9a9d7fa08abd298e932475a0520c58345f0067718bb32b`.

The issued controller check passes and hidden-CUDA `--phase start`
acknowledges at 17:10:57.209628:
`PVC/controllers/ns-training-path-exit-transition-20261119/controller.json`.
Controller PID/start is `3118453/941361276`; acknowledgment digest:
`42a901376f00df7b1672ab5d9dad610f669097ca0e48c16690ea4e48b2fb54e1`.
The controller owns keeper release/restoration; no direct manager
operation is performed. Its saved state at 17:11:58.310730 records
`validating`, keeper release attempted/complete and direct validator
`3118822/941365079` acknowledged. Fresh 17:14:11 process checks verify
both exact owners and sole monitor `1588227/933927668` live. Eight rank
PIDs `3119133` through `3119140` occupy the eight original A100 UUIDs.
These live observations do not establish completed trajectories.

A separate read-only transition reporter authenticates the accepted
CPU receipt, failed-predecessor seal, issued binding/acknowledgment,
saved generations, complete-command receipts and case decisions.
It preserves incomplete command failures and does not run a readiness
check, keeper action, GPU command or tensor reconstruction. The current
projection records 0/54 commands and 0/18 decisions; the full diagnostic
and independent canonical terminal remain pending. Nine focused
report/split-document/900-second-monitor regressions pass in 23.63
seconds and required source lint passes. The original single Bash
monitor refreshed both journals at 17:05:10 and continues without
restart; its next updater process loads the new report. No production
wave, full-budget quality/efficiency result or SwanLab import is added.

November 19 17:24 follow-up. The aggregate receipt at 17:19:42.152654
retains two completed commands, MuonT BF16/five-iteration continuous24
and interrupted22, both with execution/cleanup exits 0/0:
`PVC/run-preparation/validation-ns-training-path-exit-transition-20261119-gpu.json`.
There are zero case decisions; neither partial loss history nor
continuous update24 establishes the required continuation acceptance.
The guarded journal refresh publishes 1/54 commands and 0/18 decisions
in both recurring blocks using its 17:15:23 observation timestamp.
The later aggregate is separately dated here.

The fresh read-only waiter is CPU/source-archive accepted at
17:18:42.342132, with 24 unique passing cases in 4.58 seconds,
175 helpers and all 173 issued diagnostic helpers retained unchanged:
`PVC/run-preparation/ns-training-path-exit-transition-followup-cpu/20261119T1718/receipt.json`.
Receipt digest:
`6a0994915727bf06c606ff0a2bfcece8e4070045ac1f0dcb5f2e79d6b3041906`.
Archive digest:
`c07fe2326bf68d609f6813925b64de0c0dc985a18da252132be67f829b2595d6`.
The archived runner check passes before `--phase start` acknowledges
at 17:19:45.237525 as `3127027/941414079`:
`PVC/run-preparation/ns-training-path-exit-transition-terminal-followup-20261119/intent.json`.
Acknowledgment digest:
`1641a7cfd73b11d71a91220c7aab6b9aa6e9601d3804237ce4d89abaef0aae0d`.

Its first immutable observation at 17:19:49.345081 records original
controller `3118453/941361276` and direct validator
`3118822/941365079` exact/live, with 2/54 commands and zero decisions:
`PVC/run-preparation/ns-training-path-exit-transition-terminal-followup-20261119/observation-000000.json`.
The waiter checks every900seconds. It invokes only this generation's
issued canonical terminal after complete54-command18-decision coverage,
actual owner/descendant absence and zero-exit exact keeper restoration.
All rejected cases and consumed terminal outcomes remain retained;
it transfers no keeper and launches no new GPU experiment.

The separate saved-waiter reporter binds CPU/source acceptance, runner,
acknowledgment, immutable observations and canonical terminal outcome.
Ten focused reporting/split-document/monitor regressions pass in20.38
seconds, followed by all233 journal cases in70.26seconds. Required
source lint, Markdown content and whitespace checks pass. The journal
still exceeds the existing500KiB repository limit; results and failed
evidence are preserved. The existing single Bash monitor is retained
without restart. Full diagnostic completion, independently accepted
actual-LR/compiler successors, complete production large-window
comparisons and verified private SwanLab histories remain outstanding.
No optimizer is established as both better-performing and more efficient
than independently tuned MuonT.

November 19 17:25 progress checkpoint. The aggregate advances at
17:23:56.734615 to 3/54 completed commands and zero case decisions.
MuonT BF16/five-iteration continuous24, interrupted22 and resumed24
all retain execution/cleanup exits 0/0 and marked-descendant absence;
the scalar-loss/checkpoint comparison is still running. Saved per-command
receipt times are 17:14:18.722761, 17:18:53.209582 and 17:21:27.928689.
The 17:24 exact-handle probe verifies controller `3118453/941361276`,
direct validator `3118822/941365079`, waiter `3127027/941414079` and
sole monitor `1588227/933927668` live. The original scheduled Bash
monitor completes its refresh at17:23:46, with both journal blocks
dated17:20:13 and showing the earlier2-command/0-decision aggregate.
It loaded reporter imports before the waiter projection was added;
the next updater process loads that projection without restart.
No duplicate monitor, completed diagnostic, canonical terminal,
production wave or SwanLab import is recorded.

#### Transition Actual-LR Handoff Acceptance

November19 17:48, Asia/Shanghai. The next experiment remains actual-LR
continuation under the original compiled training boundary: three
optimizer families, BF16/FP32 NS operands, five/seven/ten iterations
and three independently specified LRs, totaling54 configurations and
162 bounded continuous24/interrupted22/resumed24 commands. Every
configuration retains the full5120-update schedule, original numerical
readers, tolerances, compilation and cached-root CUDA graphs. The
question is whether continuation gates survive at each actual LR,
not whether a bounded probe establishes full-budget quality or
whole-training efficiency. Rejected lower-LR controls remain rejected.

The transition terminal requires a distinct handoff because the
accepted numerical and execution archives contain different
`owned_output_handoff.py` bytes: numerical
`b359c02fd184f1afa7c32f3a4fbab8693ea41f7c4a7225f38cd17d7ab52b43f6`,
execution
`ec894019aa1af6ceadd81da2a4045a868a0fa34dff7c5c6da08a6230868defcb`.
Neither original archive is changed or merged. Three new helper/test
files derive the file-only predecessor from its unchanged byte-pinned
parent and run the accepted waiter/reporter closures in isolated
subprocesses. All175 issued waiter helpers are retained; reporter,
parent and successor sources bring the separate closure to184 helpers.
The old consumed actual-LR queue and deferred exit-drain predecessor
are unchanged and will not be restarted or repurposed.

The original inherited suite passes27 cases. Two additional real
subprocess/archive-publication regressions cover clean staging and
changed source bytes despite a successful pytest exit, plus one-shot
acceptance refusal. The final local suite passes29 cases in13.92seconds;
the archived suite passes29 unique cases in15.49seconds with zero
failures/errors/skips. Required source lint passes.

Two earlier staging attempts retain passing29-case XML but fail strict
archive identity authentication immediately after writing `source.tar`.
Flushing alone does not resolve the observed identity change; its
cause is not established. Their original helpers, tar files and logs
are preserved without acceptance:
`PVC/run-preparation/ns-training-path-transition-actual-lr-predecessor-cpu/20261119T1745/staging-failure.json`
and
`PVC/run-preparation/ns-training-path-transition-actual-lr-predecessor-cpu/fsync-20261119T1746/receipt.json`.
The latter receipt is explicitly `needs_attention`, not acceptance.
Their digests are
`0e2522b11ac3170f2f242872fa34f33bdbbecf69f0694c23cf860530e2a7c3b5`
and
`6dd4b28cfcff0293324c7c5492c19167593f26014ba213d28ab180b92a57a794`.

The corrected protocol separates staging from acceptance. `--phase cpu`
publishes only `staged.json` with acceptance pending and no execution
authority; `--phase accept` must run from that staged archived entry
and authenticate complete tests, all source origins, the full archive
and stable file identities before publishing one accepted receipt.
Failure is preserved separately and cannot authorize a retry in the
same namespace. Strict identity checks and numerical tolerances are
unchanged. Executed CPU acceptance uses hidden CUDA, offline uv/
pytest-timeout,60-second per-test/600-second session limits and all
seven repository safety-marker exclusions.

Accepted receipt at17:47:35.485728:
`PVC/run-preparation/ns-training-path-transition-actual-lr-predecessor-cpu/staged-20261119T1748/receipt.json`.
Receipt digest:
`c119d1b53146cbfec8065b496535e1b9c8ebd73b78687f9d589fbf6ba0777762`.
Source archive digest:
`cbf807bb3583bcf497eba4049e8c936c0b15a019df474a8981dad2413a9dc232`.
Tests XML digest:
`2288621ba250546cc94133006b841fb0f1542925bed965611f766bbe1e6e28ad`.
The independent archived `--phase check` passes. The archived
`--phase observe` preserves readiness false at
`PVC/run-preparation/ns-training-path-transition-actual-lr-predecessor-observation-20261119T1748.json`.
No seal is attempted while the exact controller/validator/waiter
or their marked descendants remain.

The live diagnostic aggregate at17:46:13.207094 has6/54 complete
commands and1/18 accepted decision:
`PVC/run-preparation/validation-ns-training-path-exit-transition-20261119-gpu.json`.
MuonT BF16/five-iteration, matrix LR0.04525483392, passes all five
required gates and39/39 scalar-loss comparisons. Checkpoints22/24
pass unchanged tolerances; checkpoint23 remains diagnostic only.
MuonT BF16/seven-iteration now has continuous/interrupted/resumed
execution/cleanup exits0/0 but its numerical comparison is pending.
The17:47:40 read-only probe verifies all three exact original owners
live and every completed command owner absent. The independent
waiter and original Bash document monitor continue their900-second
cycles; the scheduled journal refresh finishes17:42:20.

Next gates are full transition54-command18-decision coverage, its
existing independent canonical terminal, exact owner/descendant
absence and keeper restoration, followed by one caller-bound seal.
The actual-LR adapter and execution owner need distinct reviewed
CPU/source acceptance against these separate archives before any
freeze or GPU launch. Compiler localization remains a later separately
gated successor. Complete production waves still require large shared
token-window comparisons and verified private full-resolution SwanLab
histories. No optimizer is yet established as both better-performing
and more efficient than independently tuned MuonT.

#### Transition Actual-LR Adapter Acceptance

November19 18:20, Asia/Shanghai. The distinct actual-LR successor now
has a separately accepted candidate adapter. It preserves all54
family/operand/iteration/LR configurations and162 bounded eight-rank
continuous24/interrupted22/resumed24 commands, the original5120-update
schedule, training sources, byte-pinned numerical readers, unchanged
tolerances, compilation and cached-root CUDA graphs. This is an
execution-boundary preparation, not a new mathematical optimizer or
full-budget training result.

The accepted numerical adapter archive is authenticated directly
without executing its old ownership gate. The new terminal-reader
archive remains external and is authenticated in an isolated hidden-CUDA
subprocess. The reviewed execution closure explicitly changes only
`tracking/owned_output_handoff.py` from
`b359c02fd184f1afa7c32f3a4fbab8693ea41f7c4a7225f38cd17d7ab52b43f6`
to the measured transition implementation
`ec894019aa1af6ceadd81da2a4045a868a0fa34dff7c5c6da08a6230868defcb`.
All other inherited numerical helper bytes are unchanged. The original
helper remains in the new inventory as
`fixtures/transition-lr-original-owned_output_handoff.py`; the accepted
original numerical/execution archives are not modified or silently
merged. The candidate itself changes only
`experiment/fused_attribution_owner.py`, whose parent bytes are retained.
Preparation metadata binds the explicit helper transformation, both
CPU parent receipts and the external required terminal seal.

An earlier prepared candidate
`PVC/candidates/ns-training-path-transition-actual-lr-adapter-20261119T1805`
is preserved and unaccepted after the derived validator's owner-check
name was corrected. The original source and consumed queue are unchanged.
Four initial archive tests exposed a test-fixture subprocess mock
intercepting the independent terminal reader; that mock now handles only
the expensive CPU-run boundary and the terminal reader executes for real.
The corrected unissued-owner regression passes, as do all15 archived
cases; passing refusal tests create no training output or GPU binding.

Corrected candidate:
`PVC/candidates/ns-training-path-transition-actual-lr-adapter-corrected-20261119T1807`.
Accepted receipt at18:15:56.273180:
`PVC/run-preparation/ns-training-path-transition-actual-lr-adapter-cpu/corrected-20261119T1807/receipt.json`.
The archived suite passes15 unique cases in320.09seconds with zero
failures/errors/skips;179 helpers and553 inventory entries are
authenticated. Hidden CUDA, offline uv/pytest-timeout,60-second
per-test/600-second session limits and all seven repository safety-marker
exclusions are retained. Required source lint and independent archived
`--phase check` pass.
Receipt digest:
`8c9ea7cf47c3fcb437bc591cc90a82341651767ea19305082963363ebb05b9f7`.
Source archive digest:
`1073c50980bc39b946dcfdf083fb1e50dcb5b316ba8f461b9fabf5258f2ceed4`.
Tests XML digest:
`79bf04cba3d9c6fd157e62bc22931a6fad1b73563eb68fcc2fab74f5e707efee`.

The distinct execution-owner draft binds that exact adapter receipt
and resolves to191 helpers, including retained byte-pinned parent
sources. All179 accepted adapter helpers/candidate bytes remain
unchanged. It retains original physical UUID checks, real node/manager
locks, exact controller/validator/launcher/rank ancestry, visible
parent proof, one-shot terminal, cleanup and keeper restoration. The
changed predecessor boundary authenticates the separate accepted
terminal archive by subprocess and retains failed-checkpoint file
identities without fresh tensor reads. All45 lifecycle cases pass in
2.71seconds after the explicit bridge import correction; required
source lint passes. Its additional complete source/archive/ownership
suite is still running at this checkpoint. No accepted execution-owner
receipt, GPU binding, keeper transfer or launch is issued.

The live transition aggregate advances at18:18:59.438260:
`PVC/run-preparation/validation-ns-training-path-exit-transition-20261119-gpu.json`.
It retains9/54 complete commands and3/18 accepted decisions. MuonT
BF16/five-, seven- and ten-iteration cases each pass all five gates and
39/39 scalar-loss comparisons at the unchanged lower matrix LR.
All required checkpoint22/24 comparisons pass; checkpoint23 remains
diagnostic only. The seven-iteration decision is dated17:56:22.860152
and the ten-iteration decision18:18:59.438260. Fresh18:15 process
checks verify controller`3118453/941361276`, direct validator
`3118822/941365079`, waiter`3127027/941414079` and sole Bash journal
monitor`1588227/933927668` live. The waiter's fourth observation is
dated18:04:51; checks remain every900seconds. The original scheduled
monitor completes its next refresh18:19:32 without restart.

The next required action is full execution-owner CPU/source acceptance,
then complete transition54-command18-decision coverage and its existing
independent canonical terminal, exact absence/restoration and the
one-shot predecessor seal before any distinct actual-LR freeze or GPU
launch. Rejected controls and all failures remain retained. No new
production wave or SwanLab upload is recorded; Wave12 remains latest
complete, large-window analyzed and privately cloud-verified. The
search for an optimizer better-performing and more efficient than
independently tuned MuonT remains open.

#### Transition Execution Archive Milestone

November 19 18:33, Asia/Shanghai. All 53 local execution-source and
lifecycle cases pass. The complete archived execution suite started
at 18:22:49 and passes all 53 unique cases in 255.05 seconds with zero
failures/errors/skips. Receipt publication at 18:27:09 accepts the
distinct execution source:
`PVC/run-preparation/ns-training-path-transition-actual-lr-execution-cpu/20261119T1822/receipt.json`.
Independent archived-entry checking passes by 18:30 using the exact
fixed hidden-CUDA environment. The archive retains the accepted
actual-LR adapter candidate, all 179 adapter helpers and 12 separately
named execution/retained-parent helpers, for 191 helpers and 565
inventory entries. Numerical settings remain 54 configurations and
162 bounded commands: three
optimizer families (MuonT, fused normalization and identity-outer matched
null), two NS operand modes (BF16/FP32), three NS iteration budgets
(five/seven/ten) and three LRs per case. The currently running transition
diagnostic covers the 18 family/operand/budget cases at lower LR only,
with three trajectories per case and 54 commands. Both programs retain
the full 5,120-update learning-rate schedule, compilation, cached-root
CUDA graphs and original tolerances. Detailed CPU commands and receipt
identities are in `DETAILS.md`. CPU/source acceptance does not authorize freeze
or GPU launch and does not establish training quality.

The authenticated transition reporter reads aggregate 18:26:44 and
controller receipt 18:26:53. It accepts eleven completed command records
and three case decisions. The three MuonT BF16 cases each retain zero
failed tensors or scalar/structure differences at the two required
checkpoints:

| NS Iterations | Checkpoint 22 Tensor Comparisons | Checkpoint 24 Tensor Comparisons | Scalar-Loss Comparisons | Required Gates |
| --- | ---: | ---: | ---: | --- |
| 5 | 2,135 | 2,147 | 39/39 pass | All five pass |
| 7 | 2,123 | 2,163 | 39/39 pass | All five pass |
| 10 | 2,145 | 2,179 | 39/39 pass | All five pass |

Each case has sixteen first-resumed checkpoint-23 graph-state flag
differences and zero failed tensors. Checkpoint 23 remains diagnostic
only under the unchanged protocol. The five required gates are complete
actual-config trajectories, boundary all-rank continuation, all-rank
continuation after the cached-root update, scalar-loss continuation
and captured-row/update/work. Continuation compares uninterrupted and interrupted/resumed
trajectories at the same update, using `rtol=1e-5`, `atol=1e-6`.
The table's zero scalar/structure differences denotes no differences
reported by that reader; it does not claim exact bitwise tensor equality.
The fourth case, MuonT FP32 with five iterations, retains continuous24
with 12,582,912 tokens and
execution/cleanup exits 0/0 at 18:21:17. Its interrupted22 command is
also complete with execution/cleanup exits 0/0 at 18:25:54. The resumed
command completes with execution/cleanup exits 0/0 and no marked
descendants at 18:28:31. The later 18:30:58 aggregate therefore retains
12/54 commands and the same 3/18 decisions. Its independent case
comparison remains pending. No FP32 case decision or fused/matched-null
acceptance is recorded by this
checkpoint.

Fresh read-only checks at 18:26:22 verify controller
`3118453/941361276`, its direct validator `3118822/941365079`, waiter
`3127027/941414079` and document monitor `1588227/933927668` live.
The waiter's saved 18:19:52 observation remains
`waiting_for_original_owners`. The existing Bash monitor last published
both journal blocks at 18:19:32 and sleeps 900 seconds before its next
refresh; its descriptor 9 retains the original monitor lock.
Five focused split-document/concurrent-edit/monitor-retry tests pass
in 2.05 seconds with CUDA hidden. No monitor is restarted or duplicated.

The next experiment remains the separately prepared actual-LR
continuation study. Execution archive acceptance and archived-origin
checking now pass. Before freeze or launch, require complete transition
54-command/18-decision coverage, the existing independent canonical
terminal, original owner/waiter/descendant absence, exact keeper
restoration and a caller-bound one-shot predecessor seal. Compiler
localization and production Waves 13-15 retain their separate gates.
The canonical terminal must record complete successful execution,
cleanup and restoration and agree with all 18 accepted/rejected case
decisions. Rejected numerical cases remain rejected; complete diagnostic
coverage does not grant numerical acceptance. Keeper checking requires
the terminal-recorded process generations on the same eight original
A100 GPU UUIDs, with the exact current keeper live. The separate seal
is `PVC/run-preparation/ns-training-path-transition-actual-lr-predecessor-20261119-seal.json`;
its accepted reader binds the original terminal and evidence identities
to the digest supplied by the successor caller.
For every completed production wave, record five complete shared
536,870,912-token windows, exact shared evaluations and full-resolution
private SwanLab verification before choosing and launching its successor.
No new production loss window or cloud import is added here; Wave 12
remains the latest analyzed and verified production wave.

#### Transition Queue Review Milestone

November 19 18:58, Asia/Shanghai. The current diagnostic compares
uninterrupted training through update 24 with training interrupted at
update 22 and resumed through 24, on eight ranks at matrix LR
0.04525483392. It retains the original 5,120-update schedule,
compilation, cached-root CUDA graphs and `rtol=1e-5`, `atol=1e-6`.
The 18:53:21 aggregate records 15/54 completed commands and four
accepted cases out of the required 18 family/operand/iteration cases:
MuonT BF16 with five/seven/ten Newton-Schulz iterations, plus MuonT
FP32 with five iterations. Complete coverage requires decisions for
all 18 cases; rejected cases remain rejected.

The new FP32/five-iteration result passes all five gates and 39/39
scalar-loss comparisons. Across all eight ranks, update 22 has
2,121 tensor comparisons and update 24 has 2,173; both have zero
failed tensors and zero scalar/structure differences. Update 23 has
2,173 tensor comparisons, zero failed tensors and sixteen graph-state
flag differences. This first-resumed checkpoint remains diagnostic
only. FP32/seven-iteration continuous24/interrupted22/resumed24
commands now complete; their independent case comparison remains
pending. The source-bound aggregate is
`PVC/run-preparation/validation-ns-training-path-exit-transition-20261119-gpu.json`.

The successor actual-LR queue preserves the accepted 191-helper
execution archive and candidate byte-for-byte. It adds seven
separately named queue/retained-parent helpers, giving 198 helpers.
Its first corrected archive passes 42 unique CPU cases in 35.18
seconds, is accepted at 18:47:34 and passes independent archived-entry
checking by 18:50:
`PVC/run-preparation/ns-training-path-transition-actual-lr-queue-cpu/corrected-20261119T1847/receipt.json`.
The preceding 41-pass/one-failure archive is retained separately.

Independent review withholds launch of this accepted queue archive.
Its terminal-admission check can consume the one-shot reconstruction
attempt while a non-keeper controller-marked descendant is live.
The terminal checker subsequently rejects acceptance, but the
premature attempt cannot be retried. The observer also bounds only
its immediate child, leaving nested readers outside timeout cleanup.
No waiter is launched from those bytes.
Queue-only corrections now wait for controller-marked descendants,
exempt only exact recorded restored-keeper generations, and clean
up independently marked nested observers on success or timeout.
Five new regression cases and all 42 inherited cases pass locally:
47/47 in 36.33 seconds. The separate complete archive passes 47
unique cases in 36.51 seconds and publishes acceptance at 18:58:03:
`PVC/run-preparation/ns-training-path-transition-actual-lr-queue-cpu/reviewed-20261119T1855/receipt.json`.
It retains 198 helpers and 572 inventory entries. Required source lint,
independent archived-entry checking and source/archive re-review pass.
The reviewer verifies unchanged inherited 191-helper bytes and
inventory, matching generated runtime, all archive bindings and the
separate external-source symlink. A hidden-CPU waiter start is
issued from these archived bytes and acknowledges at 19:04:03 as
`3263097/942039903`. Its first immutable observation at 19:04:54
finds the original controller, validator and independent terminal
waiter exact/live and records `waiting_for_transition_owners`.
It retains zero of six handoff phases; no seal, freeze, keeper
transfer or successor GPU work is issued.

Once accepted and reviewed, the hidden-CPU waiter may observe the
original transition every 900 seconds. GPU validation still requires
complete 54-command/18-decision coverage, the original independent
waiter's canonical terminal, exact controller/validator/waiter and
descendant absence, restoration of the recorded keeper generations
on the same eight A100 GPU UUIDs, and a caller-bound predecessor seal.
Only then may the distinct queue run
`seal/check-seal/freeze/check/start/terminal` for 54 actual-LR
configurations and 162 bounded commands. These cover three optimizer
families, two operand modes, three iteration budgets and three LRs.
The numerical archive and external terminal archive remain separate.

The full journal/reporting suite passes 243 cases in 71.38 seconds
before the review corrections and 243 cases in 70.42 seconds after
them. Nineteen pinned reporting/split/monitor cases pass in 16.66
seconds. The existing Bash monitor publishes
both documents at 18:56:46 without restart. No production wave
completes and no new SwanLab import is due from these partial
continuation diagnostics. Wave 12 retains all four full-resolution
verified histories, five complete shared 536,870,912-token training
windows and exact shared evaluations. A later production wave must
finish its large-window analysis and cloud verification before its
successor is selected. Better quality and efficiency than independently
tuned MuonT remain unestablished.

At 19:03:42 the aggregate also accepts MuonT FP32 with seven
Newton-Schulz iterations, raising coverage to 5/18 case decisions
with the same 15/54 complete commands. All five gates and 39/39
scalar-loss comparisons pass. Updates 22/24 retain 2,121/2,167
tensor comparisons with zero failed tensors or scalar/structure
differences. Update 23 has 2,185 tensor comparisons, zero failed
tensors and sixteen graph-state flag differences; it remains
diagnostic only. This extends continuation evidence for MuonT;
it supplies no full-budget loss or efficiency comparison.

The 19:08:46 aggregate retains 16/54 completed commands and the
same five accepted decisions. MuonT FP32/ten-iteration continuous24
completes with 12,582,912 tokens, execution/cleanup exits 0/0 and
marked descendants absent; its interrupted/resumed trajectories
and numerical decision remain pending. At 19:10 the receipt-backed
refresh publishes both journal blocks with the reviewed queue's
acknowledgment and waiting state. Ten acknowledgment-bound reporting
cases pass in 1.74 seconds. The same sole Bash monitor and both
independent waiters remain live; no duplicate monitor is created.

#### Training-Path Actual-LR Queue Preparation

November 19 11:16, Asia/Shanghai: a distinct actual-LR successor
queue is implemented with three new helper/test files. It inherits
all177 accepted execution-owner helpers and the accepted adapter
candidate unchanged. Its complete CPU/source-archive acceptance
subsequently passes33 archived cases in46.15seconds and is accepted
at11:17:14.908054 with180 helpers/551 inventory entries:
`PVC/run-preparation/ns-training-path-actual-lr-queue-cpu/20261119T1116/receipt.json`.
Independent archived-entry verification and unchanged-parent comparisons
pass by11:19. The accepted hidden-CUDA waiter acknowledges
at11:20:12.882536 as `2541469/939256843`. Its first immutable
observation at11:20:42 finds the original controller, validator and
terminal waiter exact/live; it retains zero of six execution phases.
Fresh live checking at11:24 verifies the same queue generation,
with original terminal, successor seal and actual-LR binding unissued.

The queue checks the original controller, validator, terminal waiter,
terminal child and marked descendants every900seconds. It invokes
no original terminal. Once the existing waiter records a complete
canonical terminal and exact restoration, the queue runs the separately
accepted predecessor sealer/checker, passes that seal's exact digest
to the accepted owner's freeze, checks the issued archive and starts
the bounded actual-LR diagnostic. After exact issued-owner absence and
clean restoration, it invokes only the issued canonical actual-LR
terminal. Original rejections, failures and consumed one-shot claims
are retained without retry. CPU acceptance, a waiter acknowledgement
and this planned diagnostic supply no production or SwanLab result.

The recurring journal report now authenticates this accepted queue
archive, original acknowledgement, immutable owner observations,
completed phase ancestry/logs, retained failures, predecessor decisions
and canonical actual-LR terminal outcome. It uses saved receipts only
and executes no readiness, keeper, launch or tensor-reconstruction action.
The11:24:26 scheduled refresh loaded imports before this reporter
was added. A fresh guarded split-journal render, observed at11:26:36,
publishes the accepted queue block in both files by11:31:30; future
900-second monitor processes load the reporter without a restart.
Forty-two local queue/reporting tests pass in44.99seconds, and fourteen
selected split-document/monitor/reporting cases pass after a saved
observation-timestamp regression is added. The reporter authenticates
owner evidence while allowing the queue's later read-only observation
to omit its earlier publication timestamp. Full source lint
passes; journal lint passes with the previously documented large-file
check excluded.

#### Training-Path Actual-LR Predecessor CPU Acceptance

November 19 09:53:20, Asia/Shanghai: the separate read-only terminal
handoff component is accepted with twelve archived CPU tests in
14.10 seconds, 169 helpers and 538 authenticated inventory entries.
Receipt:
`PVC/run-preparation/ns-training-path-actual-lr-predecessor-cpu/20261119T0953/receipt.json`.
It preserves all 162 accepted actual-LR numerical-component helpers,
adds both original waiter helpers unchanged, and pins the accepted
file-only training-path reporter. Four new files implement the
predecessor, preparation and CPU fixtures/tests; no issued file changes.
All 114 integrated predecessor/waiter/reporting tests pass in
67.92 seconds. Required scoped source lint passes.

The handoff authenticates the original generation-bound one-shot
terminal and full 54-command/18-case report in an isolated CUDA-hidden
process. It preserves every rejection and reads no checkpoint tensors.
The proposed distinct seal binds original source/data/log/receipt
identities and checkpoint digests already reconstructed by the original
terminal, requiring private checkpoint change times preceding the
original reconstruction claim. Original controller, validator, waiter,
terminal child and marked descendants must be absent; exact healthy
eight-A100 keeper restoration is checked under the manager lock.
Changed evidence, substituted owners, incomplete claims, erased
rejections and rebound restoration refuse handoff. No new terminal
claim is issued, and an existing failed terminal is never retried.

A fresh archived-entry check passes. Its original-owner observation,
`PVC/run-preparation/ns-training-path-actual-lr-predecessor-observation-20261119T0955.json`,
captured at 09:54:49, records controller `2099685/937961924`,
validator `2100106/937966294` and waiter `2189311/938341032` live
and returns `ready: false`. The distinct predecessor seal is absent.
The earlier draft fixture failures and lock/source isolation corrections
are retained in `DETAILS.md`; they do not change GPU evidence or
numerical tolerances. No actual-LR owner, binding, queue, allocation
transfer, production launch or new SwanLab history is issued.

Fresh archived observation at November 19 10:01:12,
`PVC/run-preparation/ns-training-path-actual-lr-predecessor-observation-20261119T1002.json`,
again returns `ready: false`: the exact original controller, validator
and waiter are live, with marked descendants present. The observation
filename is not its capture time. Read-only pathname checks at
10:01:04 find the original terminal claim/outcome/receipt, distinct
predecessor seal and all actual-LR binding/controller/state/output/
result paths absent. The actual-LR execution owner/controller/adapter
is not yet implemented; CPU predecessor acceptance is not launch
readiness. Original terminal completion and exact restoration must
precede its one-shot seal and successor source-bound handoff.
All 149 issued helper hashes and the exact inventory authenticate at
10:03:33 without bytecode pollution. The existing 900-second Bash
monitor refreshes both journals at 10:03:39 without restart.
Five focused split-document, ambiguous-boundary and monitor-retry tests
pass in 1.53 seconds. Execution evidence and hashes are in `DETAILS.md`.
The required lint entry passes Markdown, whitespace, newline and
conflict checks but flags both journals above its 500-KiB file limit.
Both files were already above that limit before these additions.
The history is retained as requested; this is a document-size lint
limitation, not a numerical or experiment failure.

#### Training-Path Actual-LR Adapter CPU Acceptance

November 19 10:21:54, Asia/Shanghai: an owner-only adapter candidate
is separately CPU/source-archive accepted. Candidate:
`PVC/candidates/ns-training-path-actual-lr-adapter-20261119T1018`;
receipt:
`PVC/run-preparation/ns-training-path-actual-lr-adapter-cpu/20261119T1020/receipt.json`.
Preparation completes at 10:19:09, changing only
`experiment/fused_attribution_owner.py`. The original adapter is
retained separately in the archive. Every original trainer, NS
compiler-boundary, configuration, pinned-source and data byte is preserved.

Fourteen archived tests pass in 98.16 seconds. The complete flat
archive has 171 helpers and 542 inventory entries, preserving all
169 accepted predecessor/numerical helpers unchanged. A read-only
preparation audit also verifies all five accepted numerical readers,
all 353 original source/config inventory entries and the unchanged
54-case/162-command plan. Fresh archived-entry checking passes.
Source lint and `git diff --check` pass. Intermediate fixture setup
errors are retained in `DETAILS.md`; their correction creates the
missing temporary data directory, without changing numerical gates.

The integrated actual-LR numerical/adapter CPU suite subsequently
passes all 43 tests in 168.93 seconds, with zero failures, errors or
skips. Its retained XML is
`PVC/run-preparation/test-ns-training-path-actual-lr-adapter-integrated-20261119T1023.xml`;
the hash and execution scope are recorded in `DETAILS.md`.
This separate integrated run does not replace or modify the accepted
fourteen-test source archive.
The journal update passes all five focused split-document/monitor
tests and Bash syntax checking. Required scoped lint passes Markdown,
whitespace, newline and conflict checks; its full invocation fails
only because both retained journals exceed the existing 500-KiB
limit. A rerun with only `--skip large_files` passes the remaining
checks. Historical evidence and the lint limit are unchanged.

The tests run the real trainer and actual-LR validator with hidden
CUDA; both refuse the unissued execution owner before creating
training or result output. They also reject rehashed trainer/engine/
configuration changes, modifications to both parent and successor,
extra/removed source, adapter/metadata drift, imports outside the
accepted closure, truncated passing XML and altered source/parent
archives. This is source and refusal coverage, not CUDA correctness
or a loss/efficiency experiment.

This preparer does not acquire GPU ownership, invoke terminal
reconstruction or seal the original predecessor. The distinct
execution owner/controller is implemented separately below; its complete
CPU/archive acceptance subsequently passes at10:55:01. No actual-LR GPU binding or
launch is issued. Independently accept that owner/controller using the prepared adapter,
requiring original canonical terminal completion, full retained
18-case/54-command evidence, preserved rejections, exact original
owner/waiter/descendant absence and healthy keeper restoration before
freeze or allocation transfer. Only its separately owned actual-LR
GPU gates can precede equal-budget full-training tuning and verified
SwanLab histories. No better whole-training optimizer is established.

#### Training-Path Actual-LR Execution Owner Preparation

November 19 10:54, Asia/Shanghai: a separate owner/controller and
CPU acceptance component is implemented in six new helper/test files.
It preserves the already accepted adapter candidate, preparation,
all original training/configuration bytes and all 171 inherited helper
bytes. Six new helpers form a 177-helper closure with 548 inventory
entries; no original accepted archive is edited.

Forty-five CPU lifecycle tests pass in 2.73 seconds. The separately
corrected freeze/visible-identity integration test passes in31.25seconds.
It retains original terminal rejections and checkpoint identities,
requires the original seal hash, and confirms the visible-rank
identity path calls no hidden-only acceptance API. Real CPU child
tests cover detached descendants, timeout/nonzero/spawn failure
cleanup, restoration under both locks, exact physical CUDA UUID
mapping, launcher/rank ancestry and canonical one-shot terminals.
The complete 53-test suite and independently staged archived suite
subsequently pass in186.89/188.76seconds. Independent source/archive
acceptance completes at10:55:01; a fresh archived-entry check passes
by10:58. Receipt:
`PVC/run-preparation/ns-training-path-actual-lr-execution-owner-cpu/20261119T1051/receipt.json`.
All177 helpers and548 inventory entries authenticate, preserving
every171-helper parent byte and the unchanged accepted candidate.
Draft fixture/archive-layout failures and XMLs remain in `DETAILS.md`.
Scoped source lint and `git diff --check` pass.

The future freeze requires the caller-supplied hash of the separately
sealed original terminal, all18 original decisions/54 commands,
retained rejections, original controller/validator/waiter/descendant
absence and exact healthy keeper restoration. Hidden-CUDA acceptance
independently verifies the full source/archive chain. Visible validators
and ranks require the live original parent's accepted proof and
operation inode inventory; they do not reconstruct tensors or call
hidden-only predecessor APIs. Actual-LR commands retain the complete
54-configuration/162-command protocol and original numerical readers.

No predecessor seal, actual-LR binding/controller, GPU transfer,
diagnostic launch, full-budget tuning or new SwanLab history is issued.
The original lower-LR diagnostic and canonical terminal must finish
before any successor handoff. A production configuration must pass its
own actual-LR correctness gates before equal-budget, large-window
quality/efficiency comparisons.
The original ten-iteration fused interrupted22 trajectory is retained
at10:56:39 after integer-zero execution/cleanup10:55:50;
resumed24 is selected at10:56:47. At10:56:56, the receipt retains
26/54 commands and eight decisions, while all original handles remain
exact/live. No original terminal or successor seal is recorded.

#### Training-Path Actual-LR Successor Specification

November 19 09:04:38, Asia/Shanghai: a CUDA-hidden read-only audit
checks all six `ns_training_path_contract.TRAINING_FILES` digests in the
issued candidate and validates its 54 existing configurations using
`ns_budget_study.actual_config`. Every configuration differs from its
preserved family reference only by matrix LR, NS operand mode and NS
iteration count. Each family has eighteen configurations. This audit
does not issue an owner, update a frozen archive or authorize GPU execution.
The exact audit command and aggregate configuration-map digest are recorded in
`DETAILS.md` under "Training-Path Actual-LR Source Audit, November 19".

The proposed successor preserves the original trainer/NS compiler
boundary and the accepted numerical readers. It evaluates MuonT, fused
output diagonal and identity-outer matched null, each with BF16/FP32
NS operands, five/seven/ten iterations and the same LR allowance:

| LR Label | Exact Matrix LR |
| --- | ---: |
| lower | 0.04525483392 |
| center | 0.06399999989260598 |
| upper | 0.09050966784 |

The 54 configurations require 162 continuous24/interrupted22/resumed24
commands, totaling 1,358,954,496 diagnostic tokens. Every configuration
must independently pass checkpoint22/checkpoint24 and scalar-loss gates;
fused/null configurations also require captured-row, update-parity,
retained-work and null-identity evidence as applicable. Update23 remains
a retained diagnostic. Tolerances, model compilation, cached-root CUDA
graphs, seed 0 and the original 5,120-update LR horizon stay unchanged.
No center/upper configuration is accepted by a lower-LR case decision.

Implementation must use a distinct training-path successor namespace,
authenticate the original complete diagnostic and canonical independent
terminal, require original controller/validator/waiter and marked-child
absence, and verify exact restored allocation before any handoff.
The failed `ns-full-model-queue-20261119` and
`ns-actual-lr-queue-20261119` identities and archives remain consumed.
No successor owner, binding, GPU command or production wave is issued
at this checkpoint.

If the correctness gates pass, retain the equal 54-trial full-budget
grid, totaling 144,955,146,240 training tokens before any expansion or
fresh-seed confirmation. Report all five shared 536,870,912-token
windows, 22 exact evaluations, end-to-end time and independently verified
private SwanLab histories before each production-wave transition.
Boundary winners require symmetric LR expansion; selected cases then
use fresh paired seeds 3/4/5 against independently tuned MuonT and matched
null. No better whole-training efficiency optimizer has been established.

The guarded research/details refresh completes at 09:12:04 with both
manual-prose hashes unchanged. All five focused document/monitor tests
pass in 1.53 seconds with CUDA hidden; scoped Markdown lint, Bash syntax
and `git diff --check` pass. Test receipt:
`PVC/run-preparation/test-endeavor-request-split-monitor-20261119T0907.xml`.
Read-only checks at 09:12:32 verify the exact controller, validator,
terminal waiter and sole Bash monitor live. The issued 149-helper
inventory and every helper digest are exact; no source or numerical
configuration has changed. The FP32 five-iteration case decision and
independent terminal remain pending.

### NS Budget Actual-LR Queue Acceptance And Launch

The separate queue passes 27 archived CPU cases in 29.84 seconds with
156 authenticated helpers, preserving all 153 accepted execution-owner
helpers. All 120 integrated queue/owner/validator/predecessor/study tests
pass in 296.87 seconds; fresh archived checking exits zero by 05:33.
Earlier fixture failures and test-clock interruption remain in `DETAILS.md`.
No numerical tolerance or accepted parent archive is changed.

Receipt:
`PVC/run-preparation/ns-actual-lr-queue-cpu/20261119T0534/receipt.json`.
One archived CPU waiter acknowledges at 05:35:04 and saves
`waiting_for_original_full_model_queue` at 05:36:14, with 0/4 phases.
It checks original owners every 900 seconds and can issue one accepted
freeze/check/start/terminal sequence only after original lower-LR terminal,
queue/owner absence and exact keeper handoff. Per-case rejections remain
in evidence; it performs no tensor reconstruction or keeper operation.

The recurring split-journal report authenticates the accepted queue and
its original acknowledgment, complete phase logs and terminal decisions.
All 233 journal/reporting tests and 13 final authenticated split-monitor
regressions pass. A guarded refresh completes by 05:43 and verifies both
manual records byte-preserved; the original Bash monitor publishes at
05:46:46 without restart. No actual-LR binding, GPU diagnostic, full-budget
wave result or new private cloud history is issued.

### NS Budget Full-Model Configuration Preparation

Prepared November 19; CPU/source-archive acceptance completes at 01:05:55.
The hypothesis is separate from a continuation correction: test whether
five/seven-step finite maps can retain useful output-normalization quality
at less optimizer work, against equally budgeted MuonT and matched-null
controls. Ten steps remains the reference. The saved-pair study above still
rejects continuation tolerance for every budget/mode, so no full-model
correctness result is inferred from preparation.

The distinct candidate
`PVC/candidates/ns-budget-full-model-alias-corrected-20261119T0104`
derives from the authenticated failed pinned-root continuation parent,
not from the rejected rational-start intervention. Trainer, root-start,
compiled batched update and all existing parent source/configuration bytes
are preserved. Only 54 new configurations and the denial-only execution
owner change. Training fails before reading configuration, data or CUDA
unless a separately accepted full-model owner is implemented and issued.

| Dimension | Prepared Cases |
| --- | --- |
| Families | `muont`, `outer_leftdiag_fused`, `null_leftdiag_fused` |
| NS operand modes | Actual trainer `tpu` and `f32` paths, not a replacement CPU emulation |
| Iteration budgets | `q=5,7,10`; pinned coefficients, FP32 normalization and `eps=1e-5` unchanged |
| First diagnostic LR | Lower matrix LR `0.04525483392`; auxiliary LR and decay unchanged |
| Diagnostic trajectories | Continuous 24, interrupted 22 and resumed 22-to-24; original 5,120-update schedule retained |
| Reserved LR grid | Lower `0.04525483392`, center `0.06399999989260598`, upper `0.09050966784`, equal for all 18 cases |
| Full configuration budget | 5,120 updates, 524,288 tokens/update, eight ranks, seed 0; 2,684,354,560 tokens/configuration |

The 18 diagnostic cases plan 452,984,832 tokens including independent
continuous/interrupted/resumed execution. The 54 full configurations reserve
144,955,146,240 tokens if all cases eventually pass their gates; this is
an inventory and budget, not an approved campaign or launch.
All failures must remain recorded. The unchanged original NS diagnostic
must first complete, with independent terminal/restoration and original
owner absence. A distinct source-bound full-model executor must then pass
actual-config compiled/cached-root all-rank checkpoint and scalar-loss
continuation checks at the original tolerances. Fused and matched-null
cases also require captured-row, update-parity and work-count gates.
Every configuration actually launched needs its own applicable gates.

For later LR selection, use exact shared mixture-selection evaluations in
the final half; C4 is reporting-only. Compare five complete shared
536,870,912-token training windows, all exact shared evaluations, endpoint
and the final two windows. A boundary winner requires symmetric LR
expansion across compared families; selected arms need fresh paired
seeds 3/4/5. Retain complete timing and full-resolution histories, analyze
each completed production wave and verify its private SwanLab import before
releasing successors. Nominal NS-product savings or isolated CUDA-event
timing are not whole-training speedups or proof of superiority to MuonT.

The receipt
`PVC/run-preparation/ns-budget-full-model-cpu/20261119T0105/receipt.json`
authenticates 124 helpers and three fixtures; 16 archived tests pass in
40.78 seconds. A fresh archived-entry check accepts source, log, XML,
inventory and alias identity. Tests execute the real trainer's early
denial and preserved batched update for square/tall/wide matrices at all
three budgets against an independent recurrence. They reject unrelated
schedule, auxiliary-LR, numerical-source and owner tampering.
Alias/overlapping-fixture regressions preserve source bytes and reject
conflicting inherited fixtures. The integrated preparation/owner/queue/
reporting suite passes 76 tests in 59.02 seconds; four split-journal/
900-second-monitor regressions pass. Scoped code lint and
`git diff --check` pass.

The initial fixture-test failure, partial preparation and failed
01:00 archive staging are preserved under distinct identities in DETAILS.
Corrections use new candidate/output directories, not rewritten accepted
archives. No full-model executor, binding, queue, GPU run, new optimizer
quality/elapsed-efficiency result or SwanLab history is issued.
At 01:03:36 the original rational controller is still reconstructing,
with no active rank or GPU compute process; the original terminal waiter
and NS queue remain live at 01:07:03. The single Bash document monitor's
latest scheduled refresh is 00:58:17 and its 900-second sleep is unchanged.

At 01:12:38 an independent guarded refresh verifies all manual prose in
both journals byte-preserved, including this preparation and its denied
launch status. Receipt:
`PVC/run-preparation/journal-ns-full-model-refresh-independent-20261119T0110.json`.
The original terminal waiter saves another waiting observation at 01:07:29;
the NS queue remains waiting at 01:10:45 with 0/4 phases. No acceptance
or GPU launch is inferred from those saved waits. ENDEAVOR and edited
source lint pass; DETAILS passes content checks but remains above the
existing large-file limit, with its evidence history retained.

### NS Budget Actual-Config Validator Component

CPU/source-archive acceptance completes November 19 01:35:09.
The separately archived validator can retain and independently reconstruct
the prepared 18 lower-LR cases. It supplies no allocation ownership; the
configuration candidate's existing denial-only owner still rejects execution
before configuration, data or CUDA access.

The protocol is 54 eight-rank commands: continuous24, independent
interrupted22 and resumed24 for each MuonT/fused/matched-null, actual
`tpu/f32`, `q=5,7,10` case. Each actual configuration must equal its original
lower-LR reference except NS operand mode and iteration count.
The original 5,120-update schedule, auxiliary learning rate, root arithmetic,
compilation and cached-root graphs are preserved. The planned validation
budget remains 452,984,832 tokens.

The validator authenticates complete private all-rank checkpoints at
updates 22/23/24, source/configuration/data identities, numerical fast state,
AdamW state, model state and RNG state. It retains every tensor and scalar
mismatch. Acceptance requires the original boundary22 and post-cached-root24
comparisons at `rtol=1e-5`, `atol=1e-6`, plus complete scalar-loss continuation.
Fused and matched-null also require captured-row, update-parity and retained
output-work gates; matched-null requires identity outer application.
Update23 is diagnostic only, with process-local cached-graph rewarming
recorded explicitly. It never substitutes for the update24 gate.

Numerically rejected cases remain in the result and do not prevent recording
later cases. Execution or identity failures stop the one-shot sequence.
Independent reconstruction checks every command's original launcher
ancestry, log, cleanup and complete trajectory, then rederives accepted
and rejected case lists. Bounded subprocesses retain their
`.execution.json` receipts even when spawn, execution or cleanup fails.
No rejected numerical result is promoted to a passed production gate.

The accepted receipt is
`PVC/run-preparation/ns-full-model-validation-component-cpu/single-pass-20261119T0132/receipt.json`.
It authenticates 128 helpers, complete inherited candidate/source/fixture
inventory and the exact source alias. All 28 archived tests pass in
135.00 seconds; a fresh archived-entry checker accepts the unchanged
source, receipt, archive, XML and log. The integrated validator/
configuration/NS-owner/queue/reporting suite passes 104 tests in
185.30 seconds. CPU fixtures cover complete 18-case/54-command retained
reconstruction and real hidden-CUDA subprocess cleanup.

Earlier missing-directory/configuration-fixture failures and monolithic
54-command test timeouts are preserved in DETAILS. The correction validates
each checkpoint tree in one deserialization pass, retaining before/after
hash checks, and separates full-fixture creation from full reconstruction.
Each test keeps the original 60-second timeout and the suite keeps its
600-second budget. No numerical tolerance or GPU gate is relaxed.

The November 19 01:40:12 read-only observation still finds the original
rational controller reconstructing, its validator absent, no marked
descendants, no active training rank and no GPU compute occupancy.
The original terminal waiter and NS queue verify live; their saved states
remain waiting, with zero queue phases. The single Bash document monitor's
latest successful scheduled refresh is 01:33:20, after capture at 01:30:59.
No owner or monitor is restarted or duplicated.

The next execution step requires the original NS diagnostic's independently
reconstructed terminal/restoration and original owner absence, followed by
a distinct source-bound full-model owner/controller and actual GPU gates.
There is no full-model binding, GPU launch, new full-budget quality result,
elapsed-efficiency result or SwanLab history. The later equal-LR, large
shared-token-window and fresh-seed program above remains required.
No optimizer with proven superior elapsed efficiency to MuonT has been found.

An independent guarded split-journal refresh completes at 01:47:04 and
verifies both manual records byte-preserved, including this milestone.
Receipt:
`PVC/run-preparation/journal-ns-validator-refresh-independent-20261119T0144.json`.
All 13 journal/reporting/900-second-monitor regressions pass in 3.25 seconds.
ENDEAVOR and the four validator sources pass required lint;
`git diff --check` passes. DETAILS passes content checks but exceeds the
existing file-size limit at 1,204.8 KB; its evidence history is retained.

### NS Budget Full-Model Execution Owner Preparation

Prepared November 19; CPU/source-archive acceptance completes by 02:34.
This is a distinct execution layer for the accepted 18-case/54-command
actual-config validator, not another change to the optimizer.
The candidate changes only `experiment/fused_attribution_owner.py` into
an adapter that requires the separately issued frozen tracking owner.
Trainer, configurations, arithmetic, inherited metadata and all
128 accepted validator helpers are preserved. Five new helpers supply
source preparation, predecessor verification, rank ownership, controller
lifecycle and tests. The original candidate and accepted archives remain
unchanged.

The predecessor is the original NS budget diagnostic, not an idle-GPU
observation. It requires independently reconstructed terminal evidence,
original queue/controller/worker absence, exact command ancestry and
cleanup receipts, interpretable historical controls and the exact restored
keeper generation. A changed finite-map budget is a separate hypothesis;
saved pair-tolerance failures are not converted into continuation passes.
The issued owner then requires exclusive node/manager lock descriptors,
completed keeper release, original controller/validator ancestry, exact
per-command launcher/rank identity, eight-rank topology and physical GPU
UUID ownership with no foreign compute process.

The controller is one-shot and retains failures. Safe descendant cleanup
precedes keeper restoration; all 54 retained commands and 18 case decisions
must reconstruct before terminal completion. Numerical rejections remain
rejections. The original update22/update24 and scalar-loss tolerances,
fused/null capture/update/work checks, compilation and CUDA graphs remain
required. This supplies no production authority or whole-training result.

Candidate:
`PVC/candidates/ns-full-model-owner-20261119T0207`.
Its first complete owner suite passes 28 tests in 78.77 seconds, with
real detached-subprocess cleanup and positive/negative predecessor,
rank-ancestry and physical-occupancy fixtures. The initial 18-test suite
passes in 229.34 seconds; the expanded boundary subset passes 21 tests in
2.06 seconds. Scoped source lint and `git diff --check` pass.
Subsequent test-helper changes require the distinct candidate
`PVC/candidates/ns-full-model-owner-lifecycle-20261119T0214`; preparation
completes at 02:18. Its ten focused lifecycle tests pass in 100.00 seconds,
covering archive/inventory roundtrips, rehashed incomplete XML, retained
spawn failure, real detached start acknowledgment, numerical rejection
preservation and exact terminal restoration. Complete CPU acceptance now
requires both test modules and all 66 cases, not module presence alone.
The integrated validator/owner/configuration/NS-owner/queue/reporting suite
passes 142 tests in 324.12 seconds with no failures, errors or skips.
CPU/source-archive acceptance completes under
`PVC/run-preparation/ns-full-model-owner-cpu/lifecycle-20261119T0219`.
All 66 archived tests pass in 283.33 seconds, with 133 authenticated helpers
and complete inherited source/fixture inventory. The accepted receipt status
is `ns_full_model_execution_owner_cpu_archive_verified_gpu_unissued`.
It is source-bound to the distinct corrected candidate and preserves all
128 parent helper bytes. A fresh archived-entry `--phase check` passes by
02:40:24 with the same acceptance status and unchanged receipt/archive hashes;
no GPU freeze or launch follows from CPU acceptance. Exact receipts and
subsequent failures remain in DETAILS.

The scheduled observation captured at 02:05:42 finds the original rational
controller live, its validator absent, no marked training descendants and
no GPU compute occupancy. The scheduled split-journal refresh succeeds
at 02:08:10. Direct process inspection at 02:09 finds the original controller,
terminal waiter, NS queue and single Bash monitor live. The saved waiter
and queue remain waiting at 02:07:30 and 02:10:46; the queue has 0/4 phases.
No monitor or owner is restarted, no GPU work is launched, and no new
full-model binding or SwanLab history is issued.

An independent guarded refresh completes at 02:21:55 and verifies both
manual journals byte-preserved, including this preparation milestone.
All 13 split-journal/reporting/900-second-monitor regressions pass in
3.11 seconds. Receipt:
`PVC/run-preparation/journal-ns-owner-refresh-independent-20261119T0220.json`.
The next scheduled capture at 02:23:12 still finds the original rational
controller live, its validator absent and no training/compute occupancy;
the split-document refresh succeeds at 02:25:33. Saved waiter/queue
observations remain waiting at 02:22:31 and 02:25:47 with zero queue phases.
ENDEAVOR and edited sources pass required lint and `git diff --check`;
DETAILS passes content checks but remains above the existing large-file
limit, with evidence history retained.

The next scheduled split-journal refresh succeeds at 02:43:17 after its
02:40:35 physical observation. The original rational controller remains
live, its direct validator is absent, and active ranks/compute occupancy
are empty. Saved waiter and NS queue observations remain waiting at
02:37:31 and 02:40:47; zero queue phases have executed.
Fresh archived-entry verification is preserved in
`PVC/run-preparation/ns-full-model-owner-archived-entry-check-20261119T0240.json`.

November 19 launch-readiness follow-up: the accepted 66-case archive remains
immutable. Mutable execution-source/owner/controller helpers now reuse the
authenticated parent, helper inventory and preparation within each
operation; every public entry still authenticates its parent and all
source/archive evidence, and freeze still rechecks preparation and
predecessor before issuing a binding. There is no cross-call cache or
numerical change. Three additional cases exercise real CPU source staging,
freeze, and a fresh-process candidate-adapter import, including changed
frozen bytes and a byte-identical helper imported from a mutable origin.
The first focused run retains nine passes and three failures in
173.75 seconds because its temporary root lacks `run-preparation`;
no adapter acceptance follows from that attempt. Only the fixture
directory is corrected. All seven focused source/archive/freeze cases pass
in 131.04 seconds, including the fresh-process positive adapter roundtrip
and both negative origin/byte checks. The distinct candidate
`PVC/candidates/ns-full-model-owner-single-pass-20261119T0257`
is prepared; its 69-case archived CPU suite and affected integrated tests
are running at the 03:01 checkpoint. CPU/source-archive acceptance then
completes by 03:05 under
`PVC/run-preparation/ns-full-model-owner-cpu/single-pass-20261119T0300`.
All 69 archived tests pass in 281.59 seconds and all 145 integrated tests
pass in 338.08 seconds, with no failures/errors/skips.
All 133 helpers and the complete inherited source/fixture inventory
authenticate. The accepted status remains
`ns_full_model_execution_owner_cpu_archive_verified_gpu_unissued`.
A fresh process executes the archived check and passes at 03:06:42;
its receipt/archive hashes match before and after.
Receipt:
`PVC/run-preparation/ns-full-model-single-pass-archived-entry-check-20261119T0304.json`.
The accepted 66-case candidate/archive remains unchanged; these corrected
helpers do not retroactively replace its evidence or authorize GPU work.
Supporting failure:
`PVC/run-preparation/test-ns-full-model-freeze-first-20261119T0251.xml`.
Passing focused receipt:
`PVC/run-preparation/test-ns-full-model-freeze-directory-20261119T0258.xml`.
All 19 journal/reporting/split-monitor regressions pass in 1.71 seconds.

The original rational controller records completed independent result
reconstruction and keeper restoration at 02:53:29. Its numerical rejection
is unchanged. Read-only observation
`PVC/run-preparation/monitor-rational-continuation/20261118T185654708080.json`
verifies both original owners absent at 02:56:55, zero marked training
descendants and eight non-training compute processes on the original
devices. This does not independently accept the restored generation.
Original terminal waiter and NS queue remain live; their recorded
900-second sequence is preserved. No new GPU diagnostic, production wave
or SwanLab upload is launched manually.
The single Bash monitor refreshes both journals at 03:00:56 after its
02:58:18 physical observation; no monitor restart occurs.
The independent guarded split-journal refresh passes at 03:06:56,
preserving all manual prose and the failed rational-continuation predicate:
`PVC/run-preparation/journal-ns-single-pass-refresh-independent-20261119T0304.json`.
At 03:07:32 the original terminal waiter starts the exact issued read-only
verifier; child `1855154/936300767` is live at 03:08:11. This is ongoing
terminal reconstruction, not exact-restoration or numerical acceptance.
The original NS queue still has 0/4 phases; the 900-second sequence remains
unchanged. Required ENDEAVOR/source lint and `git diff --check` pass;
DETAILS passes content checks but exceeds its existing size limit, with
evidence history retained.

Next: retain the original NS
diagnostic terminal/restoration gates, then issue actual-config GPU
validation only after original queue/owner absence and exact handoff.
Surviving cases still require fair full-budget LR waves, large common-token
windows, fresh paired seeds and verified private SwanLab histories.
No optimizer with proven superior elapsed efficiency to MuonT has been found.

### NS Budget Full-Model Validation Queue

November 19 03:25, Asia/Shanghai: CPU/source-archive acceptance completed;
no queue launch or GPU command has been issued at this observation.
The separate queue extends the accepted single-pass full-model owner
with `tracking/ns_full_model_queue.py` and its 25-case test helper, plus
the explicitly pinned original NS process observer. All 133 inherited
helpers, candidate arithmetic, validator tolerances and source/fixture
inventory are preserved; the complete queue archive has 136 helpers.
Receipt:
`PVC/run-preparation/ns-full-model-queue-cpu/20261119T0318/receipt.json`.
Status: `ns_full_model_queue_cpu_archive_verified_gpu_unissued`.
All 25 archived tests pass in 27.66 seconds; fresh archived checking
exits zero. The affected queue/full-model-owner/validator/original-NS
suite passes 145 tests in 342.76 seconds with no failures, errors or skips.

The original NS queue is not replaced or restarted. The new CPU waiter
checks its exact original generation and retained phase generations every
900 seconds. It cannot freeze before the original NS queue, diagnostic
owner and workers are absent, the original terminal is independently
reconstructed, historical controls are interpretable and exact keeper
handoff is accepted. It then issues one accepted full-model
freeze/check/start sequence, waits for its issued owners to exit, and
invokes only the issued terminal verifier. Accepted and rejected cases
remain distinct; a false all-case predicate is not rewritten as success.
The queue itself runs no tensor arithmetic or keeper operation.

The planned diagnostic retains 18 lower-LR family/mode/budget cases,
54 eight-rank commands, continuous24/interrupted22/resumed24, the
original 5,120-update schedule and checkpoints22/23/24. Original
boundary22/post-cached-root24, scalar-loss, fused/null capture/update/work
and null-identity gates remain unchanged at `rtol=1e-5`, `atol=1e-6`.
Update23 remains diagnostic only. A real detached-subprocess test
ensures phase cleanup does not terminate the separately acknowledged
GPU controller, which replaces the inherited phase marker.
Fresh-process tests refuse changed archived bytes and byte-identical
helpers imported from mutable origins. Initial fixture/import-closure
failures remain in DETAILS; neither closure nor tolerances were weakened.

The split-document renderer includes the new queue in both recurring
status blocks and authenticates its accepted receipt/source bytes.
Before receipt pinning, all 32 focused reporting/split-monitor
regressions pass in 3.39 seconds; the receipt-pinned subset passes
nine cases in 1.63 seconds. At 03:28:18, the original rational
terminal verifier `1855154/936300767` is actively reconstructing;
the original NS queue still waits with 0/4 phases. No competing
reconstruction, monitor restart, GPU launch or duplicate cloud upload
is performed during this preparation.

The archived CPU waiter acknowledges at 03:28:19 as
`1875861/936425460`. Its first saved observation at 03:29:07 verifies
original NS queue `1745559/935328583` live and records
`waiting_for_original_ns_queue`, 0/4 completed phases.
Receipt: `PVC/run-preparation/ns-full-model-queue-20261119/intent.json`;
state: `PVC/run-preparation/ns-full-model-queue-20261119/state.json`.
Only this new archived CPU waiter is started; no GPU binding or
full-model command is issued. The reporting acknowledgment digest is
pinned to this original generation, with no automatic restart authority.

Final reporting/split-journal coverage passes all 216 tests in
25.89 seconds, without failures, errors or skips. Independent guarded
verification completes at 03:38:59, authenticating the live original
CPU waiter and preserving both manual-prose hashes plus accepted
queue/owner/intent/archive identities. Both recurring blocks retain
one full-model queue section, its 0/4 waiting state and the false
rational-continuation predicate. Receipt:
`PVC/run-preparation/journal-ns-full-model-queue-independent-20261119T0332.json`.
The original Bash monitor independently publishes the new section at
03:36:12 after its 03:33:17 read-only observation; no restart occurs.
At 03:39:06 the rational terminal child remains active and both NS queues
remain live. No full-model binding, production result or new cloud import
is inferred. Required source/ENDEAVOR lint and whitespace checks pass;
DETAILS retains its pre-existing large-file-limit failure without
deleting research evidence.

Next: retain all original predecessor gates and record
the actual NS diagnostic and full-model case decisions when available.
Surviving cases still require fair full-budget LR waves, complete large
shared-token windows, fresh paired seeds and verified private SwanLab
histories. CPU acceptance and bounded continuation loss do not establish
better optimizer quality or elapsed efficiency than MuonT.

### NS Budget Full-Budget Study Protocol And Analysis

November 19 03:56, Asia/Shanghai: a distinct CPU/source-archive component
prepares the next quality-cost study without changing the accepted
training candidate, running original queue, full-model diagnostic or
its tolerances. Helpers are `tracking/ns_budget_study.py`,
`tracking/prepare_ns_budget_study.py` and
`tracking/test_ns_budget_study.py`.
The accepted 133-helper execution-owner parent is preserved exactly;
three study helpers give a complete 136-helper archive.
Receipt:
`PVC/run-preparation/ns-budget-study-cpu/data-bound-20261119T0352/receipt.json`.
Status: `ns_budget_study_protocol_cpu_archive_verified_execution_unissued`.
All 19 archived tests pass in 11.00 seconds; 61 affected comparison/
configuration/history tests and 17 subtests pass in 55.29 seconds.
The initial 17-case tests and 59-case integrated results are retained.
An initial archive attempt stops before testing on a missing top-level
data-receipt field; the corrected helper obtains that identity through
the authenticated parent preparation. Failed staging is preserved,
not reused or rewritten.

Question: can five/seven-step NS reduce input-orthogonalization work
while preserving or improving full-training loss, and does fused output
normalization add benefit beyond a same-path identity-outer null?
Five/seven/ten steps nominally use 15/21/30 matrix products per NS
application. These are distinct finite-map arms; ratios 0.5/0.7/1.0
are arithmetic counts, not whole-training speed or equivalent updates.
The study retains all 18 family/operand/budget cases at the same
three matrix LRs: `0.04525483392`, `0.06399999989260598`,
`0.09050966784`. Families are MuonT, fused output diagonal and
identity-outer matched null; operand modes are pinned TPU-style BF16
and FP32. Each of 54 trials reserves 5,120 updates and
2,684,354,560 tokens, 144,955,146,240 tokens for the full initial grid
if every case becomes eligible. No changed schedule, auxiliary LR,
data, output work, training arithmetic or source identity is accepted.

The currently queued full-model diagnostic tests only lower LR.
Its result cannot authorize center/upper LR, boundary expansions or
training under a new production owner. Every actual LR configuration
must pass separately owned compiled/cached-root all-rank boundary22/
post-cached-root24 continuation, scalar-loss and complete fused/null
captured-row/update/work and null-identity gates, unchanged
`rtol=1e-5`, `atol=1e-6`, under the eventual production source.
Update23 remains diagnostic. Failed cases are retained, not promoted
by source or LR extrapolation. The study component supplies no GPU
release, allocation manager, production rank owner or automatic upload.

The analyzer verifies exact configuration/source/data provenance,
uninterrupted consecutive histories and complete mixture evaluations.
It emits complete shared 536,870,912-token training windows only;
full-budget history requires all five windows and 22 exact evaluations,
including 11 in the final half. It retains final-two-window reports,
endpoints, steady seconds/MTok and total measured process time separately.
Same-LR cohorts compare against canonical ten-step BF16 MuonT,
same-finite-map MuonT and matched null. Independently selected case LRs
are compared separately against selected canonical MuonT and selected
matched null.

Provisional LR selection is the mean reserved-mixture loss over all
exact evaluations in `(1,342,177,280, 2,684,354,560]`, not final-point
loss or C4. No case receives a choice until its equal three-point grid
completes; source mismatch, missing mixture/evaluation, changed seed,
execution-check histories, shared metric files and incomplete final
checkpoints are refused. History completion is not owner/restoration,
checkpoint-tensor or cloud acceptance.
Any lower/upper winner requires symmetric expansion to both
`0.032` and `0.128` for every compared case, with new actual-config
gates; further boundary winners require continued balanced expansion.
Selected cases then require fresh paired seeds3/4/5, complete histories,
large common-token comparisons and private full-resolution SwanLab
verification before a replicated quality/efficiency claim.

At 03:56:22 the original rational terminal verifier remains active;
both NS queues remain live and unadvanced. The 900-second Bash monitor
refreshes both journals at 03:54:14 after its 03:51:13 physical capture.
No keeper transfer, GPU diagnostic, production wave or cloud import
is manually launched by this study preparation. Next: finish fresh
archived-entry checks, retain original terminal/NS/full-model gates,
then use accepted/rejected cases to prepare separately gated LR validation
and production ownership. No superior elapsed-efficiency optimizer
relative to MuonT is established.

Fresh archived-entry checking and analysis complete at 03:57:58.
Receipt:
`PVC/run-preparation/ns-budget-study-archived-entry-independent-20261119T0355.json`.
Analysis:
`PVC/run-preparation/ns-budget-study-unlaunched-analysis-20261119T0355.json`.
Both exact archived commands exit zero, accepted receipt/archive/plan
hashes are unchanged, and analysis reports 54 planned trials,
zero observed/complete histories, 18 incomplete grids and no LR winners.
No measured result, full-model GPU binding, production acceptance or
SwanLab verification is fabricated. At 03:58:40 the original terminal
verifier remains active, both NS queues remain live, and independent
monitor-lock contention refuses another holder. The accepted owner and
queue receipt identities remain unchanged. Required study/ENDEAVOR lint
passes; DETAILS content checks pass with its existing large-file-limit
failure retained at 1,238.0 KB.

### NS Budget Actual-LR Validator Preparation

November 19 04:14, Asia/Shanghai: a separate validator component is
implemented for all 54 family/operand/budget/LR configurations in the
accepted full-budget study. It reserves 162 bounded eight-rank commands:
continuous24, interrupted22 and resumed24 per configuration,
1,358,954,496 diagnostic tokens. This is preparation, not a launched
wave, GPU acceptance or full-training result.

The component reuses the accepted complete fast/AdamW/model/RNG state
readers, checkpoint comparisons, scalar-loss comparisons and fused/null
captured-row/update/work checks without changing their tolerances.
Boundary22 and post-cached-root24 determine continuation acceptance;
update23 remains diagnostic. Every actual LR is checked independently,
and a numerical rejection stays recorded while later trials proceed.
Incomplete source, command or trajectory evidence stops execution.
The distinct actual-LR owner boundary deliberately refuses the current
lower-LR execution adapter before output creation. No actual-LR owner,
adapter, binding, controller, queue or GPU command has been issued.

Initial focused CPU coverage passes 16 tests in 43.09 seconds, including
all-family center/upper-LR reader round trips, actual-config/source
refusals and real zero/nonzero subprocess receipts. Expanded coverage
passes 19 cases but records two fixture errors when constructing all
54 numerical checkpoint trees exceeds the original 60-second test
budget. That receipt is retained. The compact-source fixture then builds
within the limit, but full reconstruction exceeds the same test budget;
its one-pass/one-failure receipt is also retained. The new validator
now reconstructs each trajectory once and compares that exact result
against the command receipt, instead of repeating the same full-state
reader. No checkpoint, loss, capture, source, ancestry or decision gate
is removed. Full-source per-family tests remain separate. Real
detached-child, timeout and spawn-failure cleanup tests pass.

The corrected focused suite passes all 21 tests in 114.71 seconds.
The complete 162-command CPU fixture retains one numerical rejection
and 53 accepted fixtures, and refuses consistently forged acceptance;
these are synthetic test outcomes, not GPU results. Four split-journal/
900-second-monitor regressions pass in 1.68 seconds. The affected
actual-LR/lower-LR/study suite passes all 68 tests in 250.36 seconds.
Distinct CPU/source-archive acceptance completes by 04:23:11:
`ns_actual_lr_validator_component_cpu_archive_verified_owner_unissued`.
Receipt:
`PVC/run-preparation/ns-actual-lr-validator-cpu/single-pass-20261119T0419/receipt.json`.
All 21 archived tests pass in 116.23 seconds; the 136 accepted parent
helpers are unchanged in the complete 141-helper archive.
The accepted parent study plan, candidate, configuration and data
identities remain preserved. The exact archived entry passes a fresh
CUDA-hidden `--phase check` by 04:24:35 with integer-zero exit;
receipt/archive/plan digests remain unchanged.

At 04:23:53 original terminal child `1855154/936300767` is active;
the original NS diagnostic queue and full-model waiter remain live,
0/4 phases. Latest saved full-model waiter check04:14:07 authenticates
the original NS queue live. The original 900-second Bash monitor
publishes both journals at04:12:21 after its04:09:15 physical capture.
No competing reconstruction, keeper transfer, production run or cloud
upload is started. Required source/ENDEAVOR lint and whitespace checks
pass; DETAILS retains its pre-existing large-file-limit failure.
Next: prepare distinct actual-LR ownership only after the
original diagnostic/full-model terminal gates become interpretable.
Survivors still require fair full-budget waves, large shared-token
windows, exact restoration, private SwanLab verification, balanced LR
expansion and fresh paired seeds. No better elapsed-efficiency
optimizer than MuonT is established.

### NS Budget Actual-LR Terminal Handoff Preparation

November 19 04:40, Asia/Shanghai: a separate read-only predecessor
component now authenticates the eventual handoff from the original
lower-LR full-model queue. It does not change the running queue,
training adapter, candidate, numerical readers or accepted validator.
No actual-LR execution owner is prepared or issued by this component.

The observer checks the original full-model queue generation and its
acknowledged phases before requiring completed receipts. A live exact
generation or marked descendant remains a wait, not a restart trigger.
The handoff requires complete original freeze/check/start/terminal phase
receipts, original queue/controller/validator/command absence, exact
source/configuration/data/archive identities, all 54 lower-LR command
receipts and 18 case decisions, independently reconstructed terminal
origin, integer-zero cleanup/restoration, completion chronology and
the same restored physical keeper. Existing reporting and queue bytes
are preserved. This component checks receipt identities and retained
decision agreement; it does not perform another tensor reconstruction.
The original independent terminal remains the numerical authority.

Every lower-LR rejection remains recorded. Synthetic terminal fixtures
include 17 accepted cases and one rejected case only to test preservation;
these are not GPU outcomes. Lower-LR acceptance grants no permission
for center/upper LR, a changed production owner, full-budget training
or SwanLab upload. The accepted 54-configuration actual-LR validator
still requires distinct ownership and actual-config gates.

Initial focused coverage passes eight tests in 2.99 seconds.
Complete pipeline coverage passes 13 tests in 6.09 seconds, including
live real CPU-process waiting, changed phase ancestry/cleanup/terminal
origin, refreshed but inconsistent decisions, source substitution and
keeper substitution. The affected queue/reporting suite passes 44
tests in 29.72 seconds. Distinct CPU/archive acceptance completes by
04:38:54, with all 13 archived tests passing in 15.09 seconds:
`ns_actual_lr_predecessor_component_cpu_archive_verified_owner_unissued`.
Receipt:
`PVC/run-preparation/ns-actual-lr-predecessor-cpu/20261119T0437/receipt.json`.
The complete 148-helper archive preserves all 141 parent helper bytes
and inventory, adding the original queue/reporting closure and four
new component helpers. Fresh archived checking and a separate
read-only original-queue observation both exit integer zero by 04:41:22.
Receipt/archive digests stay unchanged; archived-only import origins
authenticate. Observation at 04:40:29 reports
`waiting_for_original_full_model_queue`, `ready: false`, and verifies
exact original generation `1875861/936425460` live. Receipt:
`PVC/run-preparation/ns-actual-lr-predecessor-observation-20261119T0439.json`.
No terminal reconstruction or launch is performed by that observation.

At 04:41:54 terminal child `1855154/936300767` is live, with the
original terminal waiter, NS diagnostic queue and full-model queue.
Both NS queues retain 0/4 phases. The original 900-second Bash monitor
publishes both journals at 04:30:27 after its 04:27:23 capture.
No competing reconstructor, GPU command, keeper transfer, production
wave or duplicate cloud upload is started. Required source/ENDEAVOR
lint and whitespace checks pass; four split-journal/900-second-monitor
regressions pass in 1.59 seconds. Next: consume the original NS/full-model outcomes when
available and implement a distinct actual-LR owner only where those
outcomes justify it. Full-budget large-window analysis, private
SwanLab verification, balanced LR search and fresh paired seeds remain
required. No superior elapsed-efficiency optimizer is established.

### Rolling Journal Checkpoint And Actual-LR Owner Draft

November 19 04:50, Asia/Shanghai: the single original Bash document monitor
publishes both journals at 04:48:27 using physical/process capture
`PVC/run-preparation/monitor-rational-continuation/20261118T204528360300.json`
from 04:45:29. It sleeps 900 seconds after each refresh; authentication
adds about three minutes to the observed completion-to-completion interval.
No monitor, original waiter, queue or terminal reconstruction is restarted
or duplicated by this update.

Read-only process inspection finds the exact original rational terminal
child still active under its original waiter, both NS queues live and
the restored keeper process live. The saved rational result still rejects
continuation; slow terminal reconstruction does not reverse that decision.
Both queues retain 0/4 phases and no full-model GPU binding or actual-LR
command has been issued. The latest physical capture contains no active
training ranks, no marked diagnostic descendants and eight keeper compute
processes on the original physical A100s. That observation is not an
independent terminal-restoration acceptance.

Two mutable helpers now draft a separately owned actual-LR execution
boundary: `tracking/ns_lr_full_model_gpu_owner.py` and
`tracking/ns_lr_full_model_execution_source.py`. Their intended scope is
the accepted 54 actual-LR configurations and 162 bounded commands, not
production training. They have not been formatted, tested, prepared into
a candidate or accepted into a CPU archive. The expected controller,
GPU test support and GPU tests are absent. Accepted parent/archive bytes
are unchanged. No claim of executable ownership or launch readiness follows.

Next milestones remain separate: finish source-bound owner/controller and
behavioral CPU/archive acceptance; consume the original independently
reconstructed rational and NS terminals; obtain interpretable lower-LR
full-model outcomes and exact restoration; then run independently gated
actual-LR diagnostics only when eligible. Surviving cases still require fair
full-budget wave ownership, five complete shared-token windows, exact
late-mixture comparisons, full-resolution verified private SwanLab histories,
balanced LR expansion and fresh paired seeds. Wave 12 remains the latest
completed/analyzed/cloud-verified production wave. Its fused loss advantage
does not establish better elapsed efficiency than MuonT, and this checkpoint
adds no new optimizer-quality result or duplicate cloud upload.

### NS Budget Actual-LR Execution Owner Acceptance

November 19 05:14, Asia/Shanghai: the distinct actual-LR controller,
rank owner and source/archive preparer are now implemented and CPU-accepted.
Candidate:
`PVC/candidates/ns-actual-lr-owner-20261119T0504`.
Receipt:
`PVC/run-preparation/ns-actual-lr-owner-cpu/20261119T0505/receipt.json`,
published at 05:12:13 and fully authenticated by 05:14.
Status: `ns_actual_lr_execution_owner_cpu_archive_verified_gpu_unissued`.
This supersedes the 04:50 two-file-draft status, without changing any
accepted parent archive or adding a measured optimizer result.

The source transformation changes only `experiment/fused_attribution_owner.py`.
All 148 accepted parent helpers and inherited numerical/configuration/data
inventory entries are preserved; five new helpers make a 153-helper bundle.
The separate adapter requires its frozen actual-LR owner before trainer
configuration/data/CUDA and before validator output creation. Ownership
requires the original acknowledged controller/direct validator generations,
independent torch launcher/rank sessions, complete eight-rank topology,
per-command markers, held node/manager locks and owned physical GPU UUIDs.
The one-shot controller restores the exact accepted physical allocation
after cleanup, including validator nonzero/timeout/spawn failures.
Its independent terminal reconstructs retained case decisions and exact
restoration; numerical rejection is not promoted to acceptance.

Initial real-process/allocation coverage passes 28 cases in 2.57 seconds.
The source/archive subset retains five passes and seven failures in
99.07 seconds: staging incorrectly resolved inherited fixture files under
the temporary test root instead of the authenticated parent archive.
The new preparer now resolves those fixtures from the already-accepted
parent helper directory, retaining exact inventory hash checks.
All seven corrected archive/fresh-import cases pass in 106.31 seconds.
The complete integrated owner/validator/predecessor/study suite passes
93 tests in 300.79 seconds. The distinct source archive passes all
74 cases in 292.92 seconds: 21 unchanged validator, 13 unchanged
predecessor and 40 new lifecycle/source tests, with no failures/errors/skips.
The old stderr expectation remains unchanged in the inherited validator
suite; separate new-owner tests verify real denial and absent output.
Original numerical tolerances, compilation and CUDA graphs are unchanged.

A fresh CPU-only archived `--phase check` starts by 05:14 and is still
pending at this checkpoint. No actual-LR freeze/start is attempted:
the original terminal child remains active at 05:13:46, and both original
NS queues are live with 0/4 phases. Saved NS/full-model observations
are 05:10:51/04:59:07, both waiting for their original predecessors.
The single Bash monitor publishes both journals at 05:07:19 using
`PVC/run-preparation/monitor-rational-continuation/20261118T210328807669.json`
from 05:03:28. No original owner is restarted, no competing reconstruction
is created, and no GPU allocation or private SwanLab history is issued.

Fresh archived-entry checking completes by 05:16 with integer-zero exit,
the same acceptance status, exact accepted import origins and unchanged
receipt/archive digests. Final split-journal/900-second-monitor coverage
passes four cases in 1.67 seconds, and required source/ENDEAVOR lint plus
whitespace checks pass. At 05:16:23 the original terminal child remains
active under its original waiter; both original queues, the keeper and
the single monitor remain live. No new GPU phase or cloud upload starts.

Next: consume the unchanged original terminal/NS/lower-LR outcomes,
and require exact restoration plus all
per-LR numerical gates before any successor production release.
Full-budget large-window comparisons, fair tuning, balanced expansion,
fresh paired seeds and independently verified full-resolution private
SwanLab imports remain outstanding. No better elapsed-efficiency optimizer
than MuonT has been demonstrated.

### Full-Budget Attribution Wave Plan

Proposed on November 18 05:03; these are planned assignments, with no
production candidate, issued release or acknowledged worker yet.
At 05:20, the separate production controller/preparer/Bash monitor
implementation is mutable and unissued. Helper-closure review and
CPU/archive acceptance do not replace the pending GPU-validation and
terminal-restoration gates.
The 05:24 fused checkpoint-continuation failure withholds the entire
production campaign. Exact restoration is verified at 05:27:51, but
a distinct correction must pass all numerical/continuation gates at the
original tolerances before these planned assignments can be released.
The 05:31 rank-0 CPU diagnosis confirms an exact update-22 boundary and
260 tolerance failures by update 24; it is not causal or launch evidence.
At 05:36 the owner/preparer/handoff/reporting implementation has 74
passing local CPU tests, but controller behavior/archive acceptance
remains pending. No production candidate or release is issued.
At 05:50, controller inventory/failure-preservation corrections and the
missing behavior tests pass in a 257-test combined CPU/ledger suite.
The real helper closure has 69 files with preserved candidate-helper
bytes. Complete production candidate/controller archive acceptance
remains unissued, and the failed numerical validator still blocks launch.
At 05:54 the all-rank CPU diagnosis is complete: exact update-22
boundaries but 460 update-24 tolerance failures and rank-3 root
decision/counter divergence. The 261-test combined CPU suite passes.
A distinct first-resumed-update diagnostic successor is proposed,
not prepared or GPU-launched.
Use three sequential LR waves from the accepted nine-config study.
Rotate the three families so each occupies each active node once.
Node `28.216.17.139` remains outside the campaign's allocation and
launch operations.

| Proposed Wave | Matrix LR | `28.216.17.9` | `28.216.17.80` | `28.216.17.144` |
| --- | ---: | --- | --- | --- |
| Wave 13 | `0.04525483392` | `muont` | `outer_leftdiag_fused` | `null_leftdiag_fused` |
| Wave 14 | `0.06399999989260598` | `null_leftdiag_fused` | `muont` | `outer_leftdiag_fused` |
| Wave 15 | `0.09050966784` | `outer_leftdiag_fused` | `null_leftdiag_fused` | `muont` |

Each run retains the parent configuration bytes: seed 0, batch 524,288
tokens, eight ranks, 5,120 updates/2,684,354,560 tokens and the full
1,024-stable/4,096-decay schedule. Total planned training is
24,159,191,040 tokens. Use a distinct ownership-only successor and
nine separately acknowledged one-shot workers with no validation-boundary,
profiling or resume flags. Keep numerical modules, all nine configs
and the original validation evidence unchanged.

Before the first production release, independently reconstruct the
frozen validator's complete 27 commands and 93 check flags on CPU,
verify original controller/validator absence, integer-zero execution,
cleanup/restoration and the exact current keeper under the shared
manager lock. On `.9`, the first predecessor is the attribution
validator's newly restored keeper. On `.80`/`.144`, it is each verified
Wave-12 restoration. Later waves accept the immediately preceding
production restoration, including the recorded rank-to-physical-GPU
mapping. Keep inherited predecessor markers confined to those exact
keeper generations. Hold node/manager locks through release, execution,
cleanup and restoration; refuse partial or unverified transfers.

After each wave, require all three full-budget histories and final
eight-rank checkpoints, exact original-worker cleanup/restoration,
independently reproduced five-window analysis and verified full-resolution
private SwanLab histories before the next wave. Analyze each family
against same-wave MuonT and fused against same-LR matched null using
five complete 536,870,912-token windows and 22 exact evaluations.
Report the last two training windows, all 11 final-half selection
evaluations and endpoints separately. Choose full-budget family LRs
only after all nine runs complete. If any winner is an initial endpoint,
expand both log-LR endpoints equally for all three families before
tuned-superiority claims, as required by the accepted study policy.
Fresh paired seeds 3/4/5 remain reserved.

### Paired-Seed Confirmation

Confirm Muon, independently tuned input MuonT, and the selected
candidate with proposed fresh paired seeds $\{3,4,5\}$ at 0.5M/1M/2M.
Reserve these seeds from exploratory use before issuing confirmation
configurations. Pair initialization and data order across arms within
each seed. Report seed-wise fixed-token deltas and uncertainty; nearby
evaluations of one run are not replications. Confirmation uses
seed-dependent holdouts without retuning. Add remaining Paloma subsets
only under a predeclared secondary aggregation. The historical plans
proposed $\{0,1,2\}$ and later $\{1,2,3\}$. Seed 0 was used for
exploration and tuning; paired seeds 1 and 2 already contributed to
exploratory Waves 3/4. Reusing any of them is a replication of
previously inspected evidence and must remain separate from fresh
confirmation uncertainty. At November 17 23:47:05, a read-only census of
all 37 full-budget run manifests observes only seeds 0/1/2. Receipt
`PVC/run-preparation/confirmation-seed-usage-observation-20261117T2348.json`,
binds every manifest and the proposed unlaunched fresh-seed set. Recheck
usage before freezing the confirmation campaign.

The original practical margins were at least 0.003-nat mean C4 improvement
for optimization, or at most 0.002-nat degradation with at least 10%
time-to-target reduction and lower peak allocated memory for efficiency.
These are decision targets, not achieved outcomes or established statistical
thresholds. Freeze or revise them after estimating baseline variability,
before running the confirmation campaign. Increase seeds if uncertainty
overlaps the practical margin.

### Mechanistic Diagnostics

At baseline checkpoint fractions 0%, 10%, 50%, and 100%, use two disjoint
reserved pools, initially 32 sequences each. Expand unstable estimates.
Record input/output top-to-mean eigenvalue ratios, leading energy shares,
moment drift, direction cosines, direction/decay norms, effective step
ratios, and factor age.

For a diagnostic pool with $N$ uniformly sampled positions, define fresh
empirical moments $\widehat C=N^{-1}\sum_jx_jx_j^{\mathsf T}$ and
$\widehat B=N^{-1}\sum_j\delta_j\delta_j^{\mathsf T}$ using summed-loss
cotangents with the same scale as training capture. For fixed
unit-Frobenius directions, compare
$N^{-1}\sum_j(\delta_j^{\mathsf T}Dx_j)^2$ with
$\operatorname{tr}(\widehat B D\widehat C D^{\mathsf T})$ on that same pool.
Use fresh empirical moments, not training EMAs or stale roots, for this
factorization comparison. Label the latter a
Kronecker-factored cotangent-moment quadratic. Inspect cross-token terms
and compare rankings with fresh-batch Hessian/GGN contractions.
A correlation interval containing zero provides no support for the
curvature explanation; anisotropy alone does not establish it.

Measure stable rank and entropy rank separately for weights, activations,
gradients, momentum, gradient-only direction, and total parameter step.
Define zero-matrix conventions and whether entropy uses singular values
or squared singular values. Off-diagonal energy in a weight singular basis
does not prove finite-step spectrum preservation.
If $\Delta W$ denotes the actual parameter change, effective step ratios
are $\|\Delta W\|/\|W\|$; do not multiply by LR again.

### Systems Measurements

Profile steady-state median/p95 step latency; input/output collection,
communication, refresh/refinement, fallback, and matrix-application costs;
device allocated/reserved memory; host memory; and launch-to-completion time.
Keep compiler startup separate from steady-state timing and included in
end-to-end timing. Use the same production checkpoints and runtime settings
for every compared arm.

For time-to-target, predeclare a target per batch using baseline-only pilots.
Use common token-based evaluations, initially every 64M tokens with the
actual rounded interval recorded. Report crossing intervals and censoring
when targets are unreached. Time begins at launch and ends after the first
completed evaluation below target; include compilation, evaluation, and
checkpointing. Report training-only time as a secondary measure.
Do not infer token savings from the current final loss gap.

### Campaign Budget and Allocation

Keep four independent eight-GPU jobs per wave when all 32 A100s are allocated.
This preserves each run's global batch and avoids inter-node DDP traffic.
It reduces elapsed campaign time through concurrency, not total GPU-hours.
GPU occupancy and ownership must be checked anew before another launch.

| Campaign | Full Runs | Input Tokens, Excluding Tuning |
| --- | ---: | ---: |
| Completed wave 0 | 4 | 10,737,418,240 |
| Three arms, two batches, three seeds | 18 | 48,318,382,080 |
| Three arms, three batches, three seeds | 27 | 72,477,573,120 |
| Five arms, three batches, three seeds | 45 | 120,795,955,200 |

Only scale surviving candidates. A 4M-batch extension or Toner 300M/520M
model requires its own configuration and validation. No GPU-hour forecast
is established from the current end-to-end times alone.

## Recheck-Exit Successor Preparation

November 19, 2026 22:32, Asia/Shanghai. The next diagnostic tests whether
requiring whole-process exit proof in the marked-process final recheck
eliminates the scan-exit execution failure without weakening ownership
checks. The accepted scan-exit attempt exhausted three rechecks before
the ten-step MuonT resumed command entered training. The new helper
`tracking/owned_output_recheck_exit.py` applies the existing exit proof
to both unavailable and terminal recheck observations and requires a
fresh complete scan afterward. PID reuse, partial process files,
surviving threads and live ownership drift remain refusals.

All 55 local ownership/preservation cases pass in 15.52 seconds:
`PVC/run-preparation/test-recheck-exit-full-local-20261119T2235.xml`.
The filename is a preparation label; execution completed before this
22:32 checkpoint. Required scoped source lint passes. The planned
distinct candidate is
`PVC/candidates/ns-training-path-recheck-exit-reviewed-20261119T2232`,
with CPU receipt target
`PVC/run-preparation/ns-training-path-recheck-exit-20261119-cpu/reviewed-20261119T2232/receipt.json`.
Preparation and the 136-case archived suite remain separate from
acceptance and launch.

The successor retains the failed scan-exit 8/54-command, 2/18-decision
evidence, stopped four-observation waiter, accepted unissued actual-LR
queue and all older failures. Its 18 cases and 54 commands preserve the
original lower LR and 5,120-update schedule, training/configuration
bytes, numerical readers and tolerances. Only the owner adapter and
ownership helper change. After complete archive verification, the
separate predecessor seal must prove consumed-owner/marker absence
and exact healthy eight-A100 keeper restoration before freeze/check
and a new acknowledged GPU diagnostic. No failed owner is restarted.
Wave 12 remains the latest full-budget, large-window analyzed and
SwanLab-verified wave. No optimizer superiority is established.

### Recheck-Exit CPU Acceptance

November 19, 2026 22:43, Asia/Shanghai. The corrected candidate
`PVC/candidates/ns-training-path-recheck-exit-final-reviewed-20261119T2241`
accepts its separate CPU/source archive at 22:42:24. All 136 unique
cases pass in 21.68 seconds: 45 GPU-owner lifecycle, 36 handoff,
41 exit/recheck and 14 failure-preservation cases. The accepted receipt
contains 183 helpers and 552 inventory entries:
`PVC/run-preparation/ns-training-path-recheck-exit-20261119-cpu/closure-reviewed-20261119T2241/receipt.json`.
Fresh checking from that archived preparer and archived working
directory passes at 22:43. Scoped source and journal-content lint pass;
the inherited journal-size rule still fails separately.

The first archive retains 135 passes and one zombie-fixture failure.
The inherited fixture's 20-field stat record could not establish the
stricter whole-process proof. Its successor-specific derivation supplies
the complete terminal stat, empty process files and fake-clock directory
disappearance, with an assertion that disappearance occurred. The
second archive retains 135 passes and one source-contract assertion
failure: the assertion incorrectly required a test file in the older
numerical archive's helper map. The corrected test retains the explicit
adapted-test hash and unchanged numerical-reader checks. Both failed
archives remain `needs_attention` and unissued.

Separate predecessor sealing, freeze/check, exact allocation transfer
and GPU launch remain pending. The original single Bash monitor
publishes both journals at 22:41:49 and returns to `sleep 900`.
No complete diagnostic, production result, new SwanLab import or
optimizer-superiority claim follows from CPU acceptance.

### Recheck-Exit Seal And Freeze

November 19, 2026 22:50, Asia/Shanghai. The separate predecessor seal
passes at 22:44:56 and independent archived `check-seal` passes before
freeze. The seal retains 2,321 evidence files, 1,328 checkpoint payload
identities, the failed 8/54-command and 2/18-decision scan-exit result,
the stopped waiter and the accepted unissued actual-LR queue. It
verifies consumed-owner/marker absence and the exact restored keeper
`3517296` with its eight original ranks on all eight physical A100s.
These checks supply no missing numerical decision or complete terminal.
Seal:
`PVC/run-preparation/ns-training-path-recheck-exit-predecessor-seal-20261119.json`.

Freeze publishes the distinct recheck-exit diagnostic binding and
source archive; the fresh issued-controller `check` passes by 22:49.
Binding:
`PVC/run-preparation/ns-training-path-recheck-exit-20261119-binding.json`.
The one-shot issued-controller `start` is underway at 22:50. Allocation
transfer and validator acknowledgment remain unconfirmed at this entry.
No consumed owner is restarted. The receipt-authenticated reporting
projection is now included in both automated journal blocks, with the
accepted CPU receipt, seal and frozen binding pinned. Six focused
reporting/split-journal tests pass, followed by a fresh frozen-binding
report test. The original 900-second Bash monitor loads this reporter
on its next cycle without restart. Numerical readers, tolerances,
compilation and CUDA graphs remain unchanged.

### Recheck-Exit Launch And Terminal Waiter

November 19, 2026 22:55, Asia/Shanghai. The fresh issued controller
acknowledges at 22:51:58 as `3557212/943407368`. Its direct validator
`3558375/943415476` acknowledges at 22:53:19 after exact recorded
keeper release; state records the original node/manager locks and
`validating`. Fresh 22:53:55 process inspection finds both generations
live and old keeper3517296 absent. The first MuonT BF16/five-step
lower-LR continuous command is underway. No completed trajectory,
case decision or full-budget result is established at this entry.

The distinct read-only terminal waiter accepts its 24-case CPU/source
archive in 4.59 seconds, with 185 helpers and no GPU execution.
Fresh archived `check` passes before `start`. The waiter acknowledges
at 22:54:23 as `3559219/943421918`; its first 22:54:29 observation
finds the exact controller and validator live and retains 0/54-command,
0/18-decision progress. It checks those original generations every
900 seconds and invokes only the issued canonical terminal after
complete coverage, actual owner/descendant absence and clean exact
restoration. Rejected cases remain evidence. No consumed waiter or
queue is restarted.

Launch receipt:
`PVC/controllers/ns-training-path-recheck-exit-20261119/controller.json`.
Waiter acceptance:
`PVC/run-preparation/ns-training-path-recheck-exit-followup-cpu/20261119T2253/receipt.json`.
Waiter acknowledgment:
`PVC/run-preparation/ns-training-path-recheck-exit-terminal-followup-20261119/intent.json`.
Both authenticated projections are included in the existing journal
monitor's next cycle. Wave 12 remains latest full-budget, large-window
analyzed and SwanLab-verified. Actual-LR/precision CUDA gates, production
waves, large-window results and verified full histories remain open;
no optimizer superiority is established.

### Recheck-Exit Failure And Interpretation

November 19, 2026 23:02, Asia/Shanghai. The launched recheck-exit
diagnostic stops `needs_attention` at 22:59:49 after 1/54 completed
commands and 0/18 case decisions. The BF16/five-step MuonT lower-LR
continuous24 command completes; the interrupted22 command fails
before training. Rank5's ownership check reports
`complete_scan_exhausted` after three scans, with previously marked
helpers3564679 and3564689 unstable. Both retained commands have the
same helper-command digest as the earlier scan-exit failure. Delayed
post-failure reads find their stat/cmdline files absent; those separate
reads cannot establish their state during the failed scan.

The failed interrupted command's execution and cleanup exits are1/1.
It contributes no completed trajectory or numerical decision. The
controller records terminal failure and keeper restoration at23:00:07.
Independent read-only inspection at23:00:38 finds original controller
3557212 and validator3558375 absent, no validator-marked descendants,
and the exact restored keeper3564728 with ranks3564875-3564882 on
all eight original physical A100s. Restoration does not waive failed
cleanup or supply the missing54-command/18-decision terminal. The
original waiter remains live; its22:54:29 owner observation is
historical and its next900-second observation will evaluate failure.

This result does not support the hypothesis that correcting final
recheck exit handling alone resolves the live ownership failure.
The next planned investigation must reproduce new marked helpers
appearing across all three complete scans and test whether ownership
attestation can be stabilized before concurrent compiler-helper churn.
Any successor needs its own reviewed source, regression coverage,
failed-evidence seal and ownership gates. Keep three scan attempts,
the existing exit deadline, numerical tolerances, compilation and
CUDA graphs unchanged. No successor is prepared or launched from
this interpretation, and no failed owner or consumed waiter is restarted.

Failure evidence:
`PVC/run-preparation/validation-ns-training-path-recheck-exit-20261119-gpu.json`
and
`PVC/validation/ns_training_path_recheck_exit_20261119/muont_ns_tpu_q5_full_lr_lower_interrupted.{log,execution.json}`.
The sole original Bash monitor publishes both new authenticated
status blocks at23:00:58 and returns to `sleep900`. Seven focused
reporting/split-journal tests and284 integrated journal/reporting test
executions pass. Wave12 remains latest full-budget, large-window
analyzed and SwanLab-verified. No new cloud import or optimizer
superiority is established; the research goal remains active.

### Helper-Churn Experiment And Next Diagnostic

November 19, 2026 23:13, Asia/Shanghai. A CPU-only experiment using
`tracking/owned_output_helper_churn.py` runs the exact issued recheck-exit
ownership scan against an isolated fake process tree. Each exit occurs
after its marker is read, proves directory absence, and creates a new
marked helper for the next complete scan. The stable marked process
remains present throughout. Accepted and issued helper bytes are unchanged.

| Requested Successive Exits | Observed Exits | Result |
| --- | ---: | --- |
| 0 | 0 | Complete scan returns both original marked processes |
| 1 | 1 | Complete scan returns the survivor and first replacement |
| 2 | 2 | Complete scan returns the survivor and second replacement |
| 3 | 3 | `complete_scan_exhausted`; no partial ownership returned |
| 4 | 3 | Same refusal at the original three-scan limit |

All five behavior tests pass in 0.69 seconds with hidden CUDA and the
existing 60-second test/600-second session timeout guard. The fourth
requested exit is never reached because the scan already refuses after
three. Result:
`PVC/run-preparation/owned-output-helper-churn-20261119T2309.json`.
This controlled result reproduces exhaustion despite proven exits and
preservation of the stable process. The live failed scan retained only
its final unstable identities, so no per-attempt churn history or causal
attribution is established.

The next proposed diagnostic tests rank startup ordering. The issued
trainer imports numerical and compiled modules before entering
`require_fused_attribution_owner`. A distinct successor could attest all
eight rank identities before those imports and hold a pre-import barrier
until every rank passes. The later full ownership and physical-GPU checks
must remain in place. First reproduce staggered helper startup in CPU
fixtures, including a rank that fails attestation; then independently
review the barrier and its failure behavior. No source for this startup
barrier has been prepared, accepted or launched. Preserve the original
training equations, configuration, three scan attempts, exit deadline,
numerical tolerances, compilation and CUDA graphs.

The original terminal waiter independently stops `needs_attention` at
23:09:30 after its scheduled 900-second interval, reporting that the
original owners exited without complete cleanup/restoration. It retains
its first 22:54:29 observation and issues no canonical terminal.
Fresh 23:11:03 read-only checking finds controller3557212, validator3558375
and waiter3559219 absent, no validator-marked descendants, and the exact
restored keeper3564728 with its eight ranks exclusively on the original
A100s. This restoration does not waive failed execution/cleanup or
complete the missing 54-command/18-decision protocol. No consumed
controller, validator or waiter is restarted.

The existing Bash monitor1588227 retains its original generation,
exclusive lock and 900-second interval. Both automated journal blocks
continue to use the same authenticated reports; the README now describes
that split-journal behavior. Scientific milestones stay in this file,
while dense receipts stay in DETAILS. Wave12 remains the latest
full-budget, large-window analyzed and privately SwanLab-verified wave.
The actual-LR queue remains unissued, and improvement over independently
tuned MuonT remains unestablished.

November 19, 2026 23:22 verification. The five CPU cases also pass in
0.60 seconds under Torch's Python3.13 environment. A fresh run using the
accepted archived ownership helpers reproduces all five issued-helper
trial records exactly. Seven focused journal/monitor regressions pass
in29.36seconds. The authenticated manual refresh publishes both
status blocks by23:18; byte comparisons verify all manual prose remains
unchanged. The original scheduled monitor then completes its refresh
and resumes its900-second interval without restart. Scoped source and
document-content lint, Bash syntax and `git diff --check` pass. Full
journal lint still fails only the inherited500KiB file-size rule.

### Serialized Ownership Diagnostic

November 19, 2026 23:46, Asia/Shanghai. The retained command signatures
for both unstable helpers in the failed recheck-exit interrupted run
exactly match
`["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]`:
three arguments and 64 argument bytes. Their recorded parents are
training ranks. The unchanged rank ownership reader executes this
query before scanning validator-marked processes. Concurrent readers
are therefore a concrete candidate source of the marked transient
helpers. A digest and parent match are evidence for this experiment,
not proof of the failed scan's per-attempt history or live causality.

The startup-barrier proposal above is superseded. Its exploratory
12-case CPU result is preserved at
`PVC/run-preparation/test-ns-training-startup-barrier-20261119T2331.xml`;
the two exploratory source files were removed before acceptance or
launch. No import barrier, trainer change or GPU authority followed.

The distinct candidate is
`PVC/candidates/ns-training-path-serialized-reviewed-20261119T2340`.
`tracking/ns_training_owner_serialization.py` adds one canonical direct
regular-file lock around the entire unchanged `require_controller`
body, with `O_NOFOLLOW`, inode checks, nonblocking `flock` and a bounded
30-second wait. A waiting rank cannot enter the complete ownership
inspection until the previous reader exits. Every existing physical-GPU,
process-generation, ancestry, lock, source and environment check remains.
The trainer and numerical configurations are unchanged; all 18 cases
and 54 lower-LR/full-5120-schedule diagnostic commands remain. Three
complete scan attempts, the one-second exit proof, numerical readers
and tolerances, compilation and cached-root CUDA graphs are preserved.

The CPU fake-process study reproduces marked NVIDIA-query churn
through all three scans and the original `complete_scan_exhausted`
refusal. Once competing helper churn is quiescent, the same scan
under the inspection lock returns all eight stable fake ranks.
Contention, timeout, exceptional cleanup and aliased/replaced-lock
tests pass. Together with source-preservation tests, all 19 local cases
pass in 10.62 seconds. The first local run had 17 passes and two failures
from missing inherited queue aliases; its XML is retained, and explicit
parent aliases fix that preparation issue without changing numerical
or ownership gates.

The separate CPU/source archive accepts at 23:35:02 with 141 unique
passing cases in 22.54 seconds, 189 helpers and 558 inventory entries:
`PVC/run-preparation/ns-training-path-serialized-20261119-cpu/reviewed-20261119T2340/receipt.json`.
Fresh archived-entry checking passes. Its separate failed-predecessor
seal accepts at 23:37:10, preserving 2,535 evidence-file identities
and 1,352 checkpoint identities:
`PVC/run-preparation/ns-training-path-serialized-predecessor-seal-20261119.json`.
A fresh independent archived seal check passes again before freeze.
The failed recheck's 1/54 commands and 0/18 decisions, stopped waiter,
all older failures and accepted unissued actual-LR queue stay intact.

At this checkpoint freeze is in progress. No issued-controller check,
acknowledged controller/validator, keeper transfer or new terminal waiter
is yet claimed. The receipt-pinned reporter is integrated into both
automated journal blocks; six focused reporting, split-document and
monitor regressions pass in 13.23 seconds, and required scoped source
lint passes. The sole original monitor publishes at 23:40:59 and keeps
its original generation and 900-second interval. No duplicate monitor
or uploader is started.

This is an execution-correctness diagnostic, not a full-budget optimizer
comparison. Wave 12 remains the latest completed, large-window analyzed
and privately SwanLab-verified wave. No new loss result, optimizer
superiority, whole-training efficiency improvement or cloud import is
established. The next decision remains whether this unchanged numerical
protocol can complete, followed by the separately gated actual-LR and
precision-placement experiments and new full-budget wave comparisons.
Dense receipt identities and verification hashes are in DETAILS.

### Serialized Launch, Failure And Next Decision

November 20, 2026 00:00, Asia/Shanghai. The preceding preparation
checkpoint is historical. Freeze publishes November19 23:46:29:
`PVC/run-preparation/ns-training-path-serialized-20261119-binding.json`.
Fresh issued-controller checking passes before the distinct one-shot
start. Controller3595881/start943762646 acknowledges23:51:10.
Validator3597661/start943773234 acknowledges23:52:56 as its original
direct child, after exact keeper release. Both launch commands use
the separately accepted issued helper archive, not mutable checkout
helpers. Trainer/configuration and all existing numerical gates remain.

The separately accepted read-only terminal waiter has24unique passing
cases in4.67seconds and191helpers. Fresh archived-entry checking passes:
`PVC/run-preparation/ns-training-path-serialized-followup-cpu/20261119T2353/receipt.json`.
It acknowledges at23:54:12 as3598600/start943780809. Its first
23:54:17 observation reports `waiting_for_original_owners`, saved state
`validating`,0/54 completed commands and0/18 decisions. This is a
historical observation, not current diagnostic liveness. Its original
900-second schedule is unchanged; no restart or duplicate waiter follows.

The first MuonT BF16/five-step lower-LR continuous command fails before
training at23:54:13. Waiting ranks raise
`RuntimeError: Owner inspection lock deadline exhausted`.
The command's execution and cleanup exits are1/1, with recorded
marked-descendant absence. It is excluded from completed trajectories.
The result remains0/54 completed commands and0/18 numerical decisions:
`PVC/run-preparation/validation-ns-training-path-serialized-20261119-gpu.json`.
The validator's generic exact-command/generation/cleanup/log rejection
does not establish source substitution; the retained command log
identifies the new lock-wait failure.

The controller records `needs_attention` and keeper restoration at
23:54:30. Independent23:56:15 read-only checking finds controller and
validator absent, no validator-marked processes, and `same_keeper=True`
for restored keeper3598689/start943781037 and ranks3598829-3598836.
Those exact eight ranks exclusively occupy the same original A100s:
`PVC/run-preparation/serialized-failure-restored-observation-20261119T2356.json`.
Restoration does not waive failed execution/cleanup or supply missing
54-command/18-decision coverage. No canonical complete terminal is
issued, and the consumed diagnostic is not restarted.

The result rejects whole-inspection serialization within its30-second
wait bound. The critical section includes source/binding authentication
as well as transient GPU queries and marked-process scans. The retained
logs do not measure individual lock-holder durations or prove deadlock.
The next proposed experiment isolates only the transient query/marker
inspection under the same lock, keeping the source, ancestry, physical
GPU and generation checks and their ordering intact outside or inside
the appropriate boundary. First reproduce contention with a slow
authenticated-source phase in CPU fixtures and verify identical
acceptance/refusal behavior. A new GPU successor additionally requires
its own accepted source archive, preserved failed evidence and original
waiter/owner absence. Do not extend the wait, scan attempts or exit
deadline, relax numerical tolerances, or disable compilation/graphs.
This narrower candidate is proposed only; no source acceptance or
launch is claimed.

Seven focused journal/waiter/monitor regressions pass18.35seconds.
All259 affected CPU cases pass140.90seconds. Required scoped source
lint and journal-content lint pass; full journal lint reports only
the existing500KiB size rule. The sole original Bash monitor publishes
both files November20 at00:00:10 from its November19 23:56:01 report
and resumes `sleep900`. Both automated blocks include the pinned
serialized failure and waiter reports; manual milestones remain outside
those blocks. Journals are archived under the same shared update lock
before this final publication. No uploader or monitor is duplicated.

Wave12 remains the latest full-budget wave with complete large-window
analysis and privately verified SwanLab histories. This failed startup
probe adds no loss history or optimizer-efficiency result. The accepted
actual-LR queue remains unissued. Continue to record each new accepted
candidate, acknowledged wave, shared large-window comparison, failure
and verified cloud import separately in this file; the objective of
improving on independently tuned MuonT remains open.

### Query-Serialization CPU Acceptance

November20 00:13, Asia/Shanghai. The next diagnostic now has a distinct
prepared source and accepted CPU archive; the proposed status above is
historical. The original whole-inspection waiter3598600 stops
`needs_attention` at00:09:18.673294 on its scheduled900-second loop,
without invoking a canonical terminal. Its process handle is absent.
Fresh00:10:16 read-only checking also finds the original controller and
validator absent, no validator-marked processes, and the same restored
keeper3598689 on the original eight A100s:
`PVC/run-preparation/query-serialized-stopped-predecessor-observation-20261120T0010.json`.

The new implementation is
`tracking/ns_training_path_query_serialized_source.py`, with separate
preparer, predecessor reader and behavior tests. It derives the original
18cases/54commands, actual lower-LR/full5120 schedule and unchanged
trainer/configuration bytes. The existing `require_controller` statements
retain their exact AST order. Binding/source authentication and the
original acknowledgment wait move outside the lock. All statements from
the hostname query through the final original-owner generation recheck
remain inside it, including ancestry, held-node/manager-lock checks,
both NVIDIA queries, physical ordinal/UUID checks, complete marked-process
scan and occupied-rank verification. The same lock helper bytes retain
the30-second wait, canonical/direct regular-file and inode checks.
Three complete scans, one-second exit proof, numerical readers and
tolerances, compilation and cached-root CUDA graphs remain unchanged.

A CPU boundary regression holds one authenticated-source read open while
another caller completes the actual derived ownership reader. The blocked
caller remains unfinished until explicitly released. Both return the same
verified state/binding. Separate-descriptor flock checks verify that all
queries and marked-process scans occur under exclusive serialization.
This reproduces the relevant concurrency distinction without sleeping,
CUDA work, deadline extension or acceptance waiver. Other CPU cases
preserve zero-progress failed evidence, refuse substituted failed commands/
decisions/cleanup/generations, and deny any still-present original owner
or descendant. It is mechanism/correctness evidence, not GPU efficiency
or optimizer quality.

Final candidate:
`PVC/candidates/ns-training-path-query-serialized-final-reviewed-20261120T0012`.
Separate acceptance at00:11:29.251949 has142unique passing cases in
22.45seconds,193helpers and562inventory entries:
`PVC/run-preparation/ns-training-path-query-serialized-20261120-cpu/reviewed-20261120T0012/receipt.json`.
Fresh archived-entry checking passes. Initial preparation refused a
duplicate helper before creating a candidate; helper-name deduplication
fixes packaging without changing an ownership or numerical gate.
The first successfully prepared candidate remains preserved; a distinct
final candidate pins the now-terminal original waiter receipts.

At this checkpoint the separate failed-predecessor seal is in progress.
No issued successor binding, acknowledged GPU controller/validator or
new waiter is claimed. Source lint and `git diff --check` pass. Journals
are archived under the original shared update lock before this milestone;
the sole15-minute Bash monitor is not restarted or duplicated. Dense
receipt hashes remain in DETAILS. Wave12 is still latest complete,
large-window analyzed and privately SwanLab-verified; the new CPU
acceptance supplies no training loss, full-budget result or superiority
over independently tuned MuonT.

### Query-Serialization Seal And Issuance

November20 00:23, Asia/Shanghai. The distinct failed-predecessor seal
passes at00:13:48.340691, authenticating2746evidence files and
1352checkpoint identities. Fresh archived `--phase check-seal` passes.
This preserves the failed whole-inspection diagnostic, stopped original
waiter, earlier accepted/rejected cases and still-unissued actual-LR
queue; it does not turn partial coverage into a complete terminal.
Seal:
`PVC/run-preparation/ns-training-path-query-serialized-predecessor-seal-20261120.json`.

Archived freeze completes at00:18:47.938866. The fresh issued controller's
`--phase check` passes before one distinct `--phase start` invocation.
Binding:
`PVC/run-preparation/ns-training-path-query-serialized-20261120-binding.json`.
Source archive:
`PVC/run-preparation/source-ns-training-path-query-serialized-20261120.tar`.
The executed candidate remains the final reviewed00:12 candidate above.
At this observation start preflight is still running; no acknowledged
controller/validator, keeper transfer or new training command is claimed.
The original stopped waiter and all failed one-shots are not restarted.

The journal reporter now pins the verified binding. Its first full
regression run retains240passes and one reporting refusal because freeze
published while the binding pin was unset. Pinning the independently
verified binding resolves that refusal; the two focused cases pass in
14.42seconds, and the complete rerun is in progress. The reporter's scope
accurately distinguishes query/marker serialization from whole-inspection
serialization. No accepted candidate or issued GPU source is edited.
Required scoped source lint passes.

The original Bash monitor1588227 remains the sole publisher, completing
both journals at00:19:23 from its00:15:13 report and resuming `sleep900`.
Both journals are archived under `.endeavor-update.lock` before this
milestone. New experiment hypotheses, accepted releases, acknowledged
launches, failures, large shared-token-window results and independently
verified full SwanLab imports continue to receive distinct dated entries
here. Wave12 remains latest full-budget analyzed/cloud-verified; this
issuance supplies no new optimizer loss, efficiency or superiority result.

### Query-Serialized Launch And Read-Only Waiter

November20 00:30, Asia/Shanghai. The distinct controller acknowledges
at00:24:51.867094 as3633332/start943964742. The original direct validator
acknowledges at00:27:05.736020 as3637693/start943978129, after complete
exact keeper release. The first bounded eight-rank command executes
MuonT BF16/five-step lower-LR with the retained full5120 schedule,
continuous24-update trajectory and checkpoint boundary22. Its log
reaches update22 by00:30. The aggregate still admits0/54completed
commands and0/18case decisions at this observation; running metrics
alone do not supply clean execution/cleanup or continuation acceptance.

The executed source/configuration remains the accepted final00:12
candidate and00:18issued archive above. Its retained work directory is
`PVC/validation/ns_training_path_query_serialized_20261119`, despite
the November20 namespace date; this literal is part of the issued
identity and is not rewritten. First-command log:
`PVC/validation/ns_training_path_query_serialized_20261119/muont_ns_tpu_q5_full_lr_lower_continuous.log`.
No source, numerical tolerance, scan budget, lock deadline, compilation
or CUDA-graph gate is changed after issuance.

The distinct read-only waiter archive accepts all24unique cases in
5.02seconds at00:26:45.437056, retaining all193issued helper bytes plus
its two new files. Fresh archived `--phase check --cpu` passes before
the separate one-shot start. Waiter3638945/start943982783 acknowledges
at00:27:52.277328; its first00:27:57.918722 observation verifies exact
original controller and direct validator generations live, with
0/54commands0/18decisions. It retains the900-second interval and may
invoke only the issued independent canonical terminal after complete
coverage, clean exits, actual owner/descendant absence and exact
restoration. Failed old waiters/controllers remain consumed and unchanged.
Waiter acceptance:
`PVC/run-preparation/ns-training-path-query-serialized-followup-cpu/20261120T0026/receipt.json`.

The reporter pins the verified controller acknowledgment and separately
accepted waiter receipts. Both automated journal blocks include its
saved observations; the original sole monitor is not restarted.
The first waiter regression catches an omitted research-summary
allowlist entry; that reporting omission is corrected without editing
GPU sources. The earlier complete241-case journal suite passes in
147.82seconds. New waiter/summary regression verification is in progress.
Wave12 remains latest full-budget large-window analyzed and privately
SwanLab-verified. This is an acknowledged correctness diagnostic, not
a full-budget successor wave or evidence of optimizer superiority.

### Query-Serialized First Commands And Journal Verification

November20 00:39, Asia/Shanghai. The first MuonT BF16/five-step lower-LR
continuous command finishes at00:30:14.275132, with24updates, integer
execution/cleanup exits0/0 and marked-descendant absence. The interrupted
22-update command finishes at00:35:15.292437 with the same clean exits.
The aggregate at00:36:07.601060 admits2/54completed trajectories and
0/18case decisions. The resumed log reaches24updates by00:39; neither
that log nor matching displayed losses replaces its trajectory validation
or the independently reconstructed case decision.
Receipts are the `.execution.json` files beside the first-command log
above and `muont_ns_tpu_q5_full_lr_lower_interrupted.log`.

These two clean executions are progress beyond the previous startup
failures, not complete diagnostic acceptance. Complete54-command/
18-decision evidence, original-owner/descendant absence, clean exact
keeper restoration and independent terminal remain required. Actual-LR
and precision gates, full-budget successor experiments, large shared-token
comparisons and independently verified SwanLab histories remain open.
No optimizer-quality or whole-training efficiency improvement is claimed.

All242integrated journal cases pass in164.70seconds. XML independently
contains242unique testcase identities, zero failures/errors/skips:
`PVC/run-preparation/test-endeavor-query-waiter-integrated-20261120T0032.xml`.
The seven focused waiter/split-document/monitor cases pass20.53seconds.
Required source lint, journal-content checks, Bash syntax and
`git diff --check` pass. Full journal lint fails only the inherited
500KiB size rule; no historical evidence is removed to satisfy it.

The original sole Bash monitor publishes both documents at00:39:11
from its00:34:25 report, retaining its original generation, exclusive
lock and900-second schedule. Both automated blocks now contain the
query diagnostic and its separately authenticated read-only waiter.
The shared update lock initially defers the manual append while that
publisher authenticates its report; no duplicate refresh is launched.
Both journals are archived under that lock before this verification.
Wave12 remains latest full-budget analyzed and privately cloud-verified.

November20 00:41 progress receipt: the first resumed command finishes
at00:38:20.014263 with integer execution/cleanup exits0/0 and marked
descendants absent. The aggregate at00:40:46.417751 now admits
3/54commands and0/18decisions; all three first-control trajectories
are retained, while the numerical case comparison remains pending.
Receipt:
`PVC/validation/ns_training_path_query_serialized_20261119/muont_ns_tpu_q5_full_lr_lower_resumed.execution.json`.
The00:39automated publication remains its dated2/54observation;
the sole monitor will refresh from newer authenticated evidence on its
next900-second cycle. No extra publisher or complete-terminal claim is
created. Both journals are archived under the shared lock before this
increment; the scientific objective and full-budget successors remain open.

### Query-Serialized First Acceptance And Actual-LR Reader

November20 00:59, Asia/Shanghai. The original validator records the
first accepted MuonT BF16/five-step lower-LR case at00:50:57.442848.
Executed configuration remains
`experiment/configs/muont_ns_tpu_q5_full_lr_lower_0p5m_seed0.json`
in the accepted final00:12candidate, matrixLR0.04525483392, eight
A100 ranks, continuous24/interrupted22/resumed24 and the original
full5120schedule. All five existing gates pass. Checkpoints22 and24
have zero failed tensor or scalar comparisons, and all39scalar-loss
comparisons pass. Checkpoint23 has zero failed tensor comparisons but
16graph-state scalar differences; its original diagnostic-only treatment
remains unchanged. This is within-arm continuation evidence, not a loss
comparison against a candidate or evidence that five iterations are best.

The next seven-step continuous command finishes24updates at
00:53:40.787797 with integer execution/cleanup exits0/0 and marked
descendants absent. Aggregate00:56:09.173383 admits4/54commands and
1/18decisions. Receipt:
`PVC/validation/ns_training_path_query_serialized_20261119/muont_ns_tpu_q7_full_lr_lower_continuous.execution.json`.
The original interrupted command is running; no seven-step case decision
or complete diagnostic terminal follows from its continuous history.
The sole original read-only waiter retains its900-second interval and
records its third waiting observation at00:57:59.476307.

A distinct actual-LR predecessor reader is implemented in
`tracking/ns_training_path_query_serialized_lr_predecessor.py`, with
separate preparer and external-boundary tests. It derives the original
isolated terminal reader, retaining complete54-command/18-decision
coverage, the original waiter's sole canonical independent terminal,
exact owner/terminal-child/marked-descendant absence, clean restoration
and exact current keeper. It reuses checkpoint digests and checks payload
identities without reading tensor payloads. Numerical and execution
archives remain separate, including conflicting handoff-helper identities.
The latest failed seal does not flatten every older summary: the reader
authenticates its retained ancestor seals directly, preserving original
44/14, recovery2/0, exit-drain6/2, transition23/7, scan-exit8/2,
recheck1/0 and whole-inspection0/0, all rejections and consumed one-shots.

Separate archived acceptance at00:55:08.098908 has39unique passing
cases in18.76seconds and211authenticated helpers:
`PVC/run-preparation/ns-training-path-query-serialized-actual-lr-predecessor-cpu/staged-20261120T0053/receipt.json`.
Fresh archived-entry checking passes. Its00:57:24.601748 read-only
observation returns `waiting_for_query_serialized_owners`, `ready: false`,
with exact controller3633332, validator3637693 and waiter3638945 live,
plus active marked descendants. No seal, GPU transfer or launch is issued.
The integrated predecessor suites pass132executions in54.32seconds
(46distinct inherited testcase identities); the acceptance archive
separately verifies all39unique identities. Seven focused journal/
split-publication/monitor regressions pass18.90seconds. Earlier fixture
derivation failures remain archived. Scoped source lint and diff-check pass.

Next experiment remains a distinct query-serialized actual-LR adapter,
execution owner and queue for the preserved54configurations/162bounded
commands: three families, three actual LRs, BF16/FP32 and5/7/10iterations.
This reader acceptance does not prepare or launch those components.
No old failed queue may consume these receipts. Only after full original
coverage, clean restoration and independent terminal can its own
unchanged-tolerance actual-LR gates proceed. Full-budget successor waves
still require separate production ownership, complete536,870,912-token
shared windows and exact shared evaluations, followed by independently
verified complete private SwanLab histories before the next wave.
Wave12 remains the latest completed/analyzed/cloud-verified wave; improving
on independently tuned MuonT remains an open objective.

The original Bash monitor1588227/start933927668 publishes both journals
at00:58:51 from its00:54:14 report and resumes `sleep900`. Its source,
generation and exclusive fd9 lock are unchanged. Both journals are
archived under `.endeavor-update.lock` before this manual milestone.
New hypotheses, accepted sources, acknowledged launches, failures,
large-window results and verified cloud imports will continue to receive
separate dated entries here, with dense receipts in DETAILS.

November20 01:02 progress increment: the seven-step interrupted22-update
command finishes00:58:38.852132 with integer execution/cleanup0/0 and
marked descendants absent. Aggregate00:59:29.342790 admits5/54commands,
1/18decisions, `validating`, no error. Its receipt is
`PVC/validation/ns_training_path_query_serialized_20261119/muont_ns_tpu_q7_full_lr_lower_interrupted.execution.json`.
Resumed validation and the complete seven-step decision remain pending.
The00:58automated4/54observation remains historical until the sole
monitor's next original900-second cycle. Source/content lint, Bash syntax
and diff-check pass; full document lint reports only inherited500KiB
size limits, ENDEAVOR791.4KB/DETAILS1608.7KB before this increment.
Both journals are archived under the shared lock before this append;
no history, failed evidence or accepted source is removed.

### Query Actual-LR Adapter Preparation

November 20 01:15, Asia/Shanghai. A distinct candidate is prepared at
01:12:32.873523:
`PVC/candidates/ns-training-path-query-serialized-actual-lr-adapter-20261120T0113`.
Its source modules are `tracking/ns_training_path_query_serialized_lr_`
with `parent.py`, `contract.py`, `validation.py` and `source.py`, plus
the source-bound external-boundary tests. Preparation changes only
`experiment/fused_attribution_owner.py` relative to the accepted numerical
candidate. The plan preserves 54 configurations and 162 bounded eight-rank
commands: MuonT, fused output normalization and matched identity-outer null;
matrix LRs 0.04525483392, 0.06399999989260598 and 0.09050966784; BF16 and
FP32 operands; five, seven and ten Newton-Schulz iterations. Continuous24
versus interrupted22/resumed24 retains each original full5120 schedule.
The planned bounded budget is1,358,954,496tokens, not a production wave.

The adapter pins the independently accepted query actual-LR predecessor
reader and187helper identities. Numerical and predecessor archives remain
separate. Its explicit handoff transformation retains the original
numerical helper as
`fixtures/query-serialized-lr-original-owned_output_handoff.py`; all other
inherited numerical helpers, trainer/configuration bytes, tolerances,
compilation and CUDA graphs are unchanged. Source lint passes. The local
16-case CPU suite completion is observed shortly after this checkpoint.
Its XML records start01:06:52 and393.475seconds elapsed, placing completion
near01:13:26; earlier tool-session polling prose lagged that artifact.
CPU/archive acceptance and fresh archived-entry checking are not yet
established at01:15. Preparation is not a GPU binding, execution-owner
acceptance, terminal seal or launch.

The original seven-step lower-LR resumed command finishes01:01:43.026252
with integer execution/cleanup0/0 and marked descendants absent.
Aggregate01:04:16.598908 retains6/54commands and1/18decisions, `validating`,
no error. The first five-step MuonT case remains accepted; the seven-step
numerical decision is pending. The original controller, validator and
read-only waiter remain live in their exact generations. The sole Bash
monitor retains its exclusive lock and900-second sleep after its00:58:51
publication; its next cycle may publish newer evidence independently.

Next: finish separate adapter CPU/archive acceptance, then implement and
accept the distinct execution owner and queue with the reviewed query-only
inspection-lock boundary. GPU gates require the original full54-command/
18-decision coverage, clean restoration and independent canonical terminal.
Full-budget successors still need separate production ownership, complete
536,870,912-token shared windows and exact shared evaluations, followed by
independently verified complete private SwanLab histories. Wave12 remains
the latest full-budget analyzed and cloud-verified wave. No optimizer
superiority is established. Both journals are archived under the shared
lock before this milestone; dense identities and receipts are in DETAILS.

### Query Scan Failure And Restored Allocation

November 20 01:19, Asia/Shanghai. The01:14:48.607372 aggregate preserves
6/54completed commands and2/18case decisions, then stops `needs_attention`.
Both MuonT BF16/five-step and BF16/seven-step lower-LR controls pass all five
original continuation gates at matrixLR0.04525483392. Seven-step
configurationSHA is
`138d37889cc168cc83347b8d36f2cd02ef22eae461c6a7810e23ecc07e510e3b`;
checkpoints22/24 have zero failed tensor/scalar comparisons and all39loss
checks pass. Checkpoint23 retains16diagnostic-only graph-state scalar
differences and zero failed tensors, under its existing acceptance policy.
These are within-arm continuation results, not optimizer-quality comparisons.

The next ten-step continuous command is not launched. The failure occurs
in the validator's pre-command `require_ns_training_path_gpu_owner` call,
whose marker scan exhausts its three complete attempts. Retained unstable
PIDs are3730921,3730943,3730947,3730948,3730949,3730952, all explicitly
`previously_confirmed_marked: false`. Later missing process files are only
delayed observations and do not establish coherent identity or live causality.
The controller and validator stop; recorded execution/cleanup exits remain1/1.
Restoration records0 at01:15:06.674160. Independent01:16:42.046271 checking
finds both original owners absent and no validator-marked descendants.
Keeper3731474 and its exact eight ranks3731705-3731712 exclusively occupy
the original eight physical A100s. This does not waive either failed exit,
complete missing coverage, or create the original independent terminal.
Observation:
`PVC/run-preparation/query-failure-restored-observation-20261120T0118.json`.

The sole original waiter retains its fourth01:13:00waiting observation and
900-second interval; it is not restarted or replaced. The sole Bash monitor
publishes both journals at01:18:58 from its01:13:54report and returns to
`sleep900`, retaining generation and exclusive lock. That automated report
time and the later failed aggregate are distinct dated observations.
The local adapter suite passes16cases in393.58seconds and five journal
split-publication/monitor tests pass3.07seconds. Separate archived adapter
acceptance remains running. Full journal lint reports only the inherited
500KiB size rule; document-content lint and diff-check pass.

The prepared query actual-LR candidate remains unissued. Its predecessor
reader requires the complete original terminal, so this failed6/54,2/18
diagnostic cannot supply its seal or launch authority. Next investigate
the unrelated-process missing-environment branch with CPU-only reproduction
against exact issued bytes, preserving all scan attempts, deadlines,
identity/ownership rules and numerical gates. Any successor must have a
separate reviewed source/archive and failed-predecessor record; do not reuse
the failed controller or weaken its acceptance predicates. Wave12 remains
the latest complete large-window analyzed/private-SwanLab-verified wave,
and the scientific improvement objective remains open. Both journals are
archived under the shared lock before this failure milestone.

### Query Adapter Acceptance And Unmarked Churn Study

November 20 01:24, Asia/Shanghai. Separate actual-LR adapter archive
accepts at01:21:54.065972:16unique passing cases in381.99seconds,
187authenticated helpers and561inventory entries.
Receipt:
`PVC/run-preparation/ns-training-path-query-serialized-actual-lr-adapter-cpu/20261120T0116/receipt.json`.
Fresh checking from the archived entry and its own flat helper path
passes at01:23. Numerical/predecessor archives stay separate; the full
54configuration/162command plan and original full5120schedules are
unchanged. This is CPU/source acceptance only. The prepared candidate
has no accepted execution owner, queue, canonical predecessor seal or
GPU binding. Its complete-original-terminal requirement remains unsatisfied.

The new CPU-only study in `tracking/owned_output_unmarked_churn.py`
uses the exact issued query handoff and exit-helper bytes. It removes
an unrelated fake process directory after the process observation but
before its environment read, then exposes a new unrelated process to
the next scan. With zero, one or two exits, the original scanner returns
both marked survivors. Churn continuing through all three complete
scan attempts returns no partial ownership and produces
`complete_scan_exhausted`, with `previously_confirmed_marked: false`.
A present live fake process whose environment file is missing still
refuses; unavailable environment is not treated as empty or as absence.
Six new regressions and the five earlier marked-churn cases pass11cases
in1.12seconds. Source lint passes after an unused import/format correction.
The retained01:23:06study uses the exact issued query source directory:
`PVC/run-preparation/unmarked-churn-study-20261120T0124.json`.

This reproduces a refusal mechanism compatible with the failed snapshot,
not the live processes' identity or causality. No scanner implementation,
identity proof, ownership policy, three-attempt limit, one-second exit
deadline, numerical tolerance, compilation or CUDA graph is changed.
Next investigate why unrelated short-lived helpers are included in the
complete scan and whether their execution can be isolated without dropping
any ownership evidence. Any correction requires its own source-bound
CPU/archive review and a preserved failed-predecessor record before GPU
use. The original failed controller and validator remain consumed; the
original read-only waiter stays on its own900-second schedule. The sole
Bash monitor remains live in `sleep900` after01:18:58publication.
Wave12 remains the latest complete large-window analyzed and independently
private-SwanLab-verified wave; no superiority or efficiency claim follows.
Both journals are archived under the shared lock before this milestone.

November20 01:26 verification increment:16combined journal and marked/
unmarked-churn regressions pass4.06seconds. Required scoped source and
document-content lint and diff-check pass; the inherited500KiB journal
size exception remains explicit. The sole monitor stays live, sleeping900
seconds after01:18:58publication, and the original waiter retains its
scheduled observation interval. No new full-budget wave, complete terminal,
GPU successor or SwanLab import is claimed.

### Stopped Query Waiter And Preserved Failure

November 20 01:41, Asia/Shanghai. The original read-only waiter stops
`needs_attention` at01:28:01.329405 on its own900-second schedule. Its
four retained observations remain historical; the final error reports
original owners exiting without complete cleanup/restoration. It issues
no canonical terminal. The independent01:30:59.919608 observation verifies
controller3633332, validator3637693, waiter3638945 and all six completed
launcher handles absent, no validator-marked descendants, and the same
restored keeper3731474 with ranks3731705-3731712 exclusively occupying
the original eight A100s.
Artifact:
`PVC/run-preparation/query-and-waiter-stopped-observation-20261120T0132.json`.

The failed query retains6/54commands and2/18decisions, both accepted MuonT
BF16 lower-LR five/seven-step controls, and recorded execution/cleanup1/1.
The ten-step continuous command was never spawned: failure occurred in
the validator's pre-command ownership check. No failed-command execution
sidecar, training log or ten-step trajectory may be invented to complete
a predecessor template. The stopped waiter and controller remain consumed.
The accepted actual-LR adapter still cannot consume a complete original
terminal, because none exists; it stays unissued.

### Streaming Observation-Order Study

November 20 01:41, Asia/Shanghai. Hypothesis: first materializing every
process identity delays earlier processes' environment reads enough for
unrelated short-lived processes to disappear and exhaust the complete scan.
A separate candidate in `tracking/owned_output_streaming_scan.py` derives
only the marker scanner's iteration from the exact issued query handoff.
It interleaves each process identity and environment read. The original
process enumeration and filters, every complete-scan body statement,
final marked-process recheck, three attempts and one-second identity-bound
whole-process exit proof remain unchanged. All issued/accepted scanner
bytes and numerical/trainer/configuration bytes remain untouched.

In the controlled fake-process-tree comparison, observing a later process
retires an earlier unrelated process and exposes its replacement. The
materialized scanner exhausts all three attempts without partial ownership;
the streaming candidate returns both marked survivors. This tests a
specific stale-read mechanism, not the live failure's cause or a guarantee
against all concurrent namespace changes. All36pinned original handoff
cases plus10ordering/lifecycle cases pass46tests. Marked continuous churn,
live marker removal, reused PIDs, partial files, persistent unavailable
environments and surviving threads still refuse. Accepted whole-process
exit retains newly discovered marked children after a complete rescan.
The combined scanner/churn suite passes57cases in5.98seconds; five focused
split-journal/monitor tests pass3.13seconds. Scoped source lint passes.
Source snapshot:
`PVC/run-preparation/streaming-scan-study-source-20261120T0141.tar`.
Test evidence:
`PVC/run-preparation/test-streaming-and-churn-final-20261120T0140.xml`.
This is a retained working-source study, not a source-archive acceptance
receipt, GPU gate, production result, efficiency result or SwanLab import.

Next experiment: prepare a distinct streaming/query diagnostic namespace,
authenticate its complete source closure and CPU archive, and preserve
the failed6/54,2/18predecessor with an explicit pre-command-failure reader.
That reader must require the next command's absence rather than a
nonexistent spawned-failure sidecar. Fresh independent predecessor and
issued-source checks are required before any GPU launch. Preserve all
earlier rejections, accepted controls, failed seals, consumed one-shots,
unissued queues, numerical gates, compilation and CUDA graphs.

The sole original Bash monitor1588227/start933927668 publishes both
journals at01:38:23 from the01:34:01report, then returns to `sleep900`.
No monitor, controller, validator, waiter, uploader or GPU experiment is
duplicated or restarted. Both journals are archived under the shared lock
before this milestone. Wave12 remains the latest full-budget wave with
large shared-token-window analysis and independently verified complete
private SwanLab histories. Further full-budget experiments, comparison
and verified uploads remain required; no superiority over MuonT is claimed.

November20 01:48 final verification: all299unique journal/scanner/churn
cases pass in159.25seconds, with zero failures, errors or skips.
XML:
`PVC/run-preparation/test-endeavor-streaming-and-churn-clean-20261120T0145.xml`.
The first broader run's298passes/1failure remains retained. A generated
pytest bytecode cache from the initial test invocation made the historical
scan-exit waiter's exact flat-helper inventory refuse. All180pinned
helpers and its archive were unchanged. The generated cache is preserved
outside the accepted helper directory, and the unchanged strict reporter
authenticates again; no acceptance rule is relaxed. The source snapshot
matches all six current study/test files and the57-case XML contains57
unique passing cases. Source and document-content lint, Bash syntax and
diff-check pass. Full journal lint still reports only the inherited500KiB
size limit, separately recorded in DETAILS. Both journals are archived
under the shared lock before this final increment. No new GPU experiment,
full-budget result, complete terminal or SwanLab import is claimed.

### Streaming Diagnostic Prepared

November 20 02:01, Asia/Shanghai. The distinct
`ns-training-path-streaming-20261120` candidate was prepared at01:55:47:
`PVC/candidates/ns-training-path-streaming-20261120T0156/ns-training-path-streaming-preparation.json`.
It records353source files and210declared helpers. Fresh manifest checking
matches all353candidate files without drift. The only candidate training
source change is `experiment/fused_attribution_owner.py`; the handoff
replacement changes marker-scan identity/environment observation order.
Trainer/configuration bytes,18cases/54bounded commands, lower LR and the
original5120-update schedule, numerical readers/tolerances, three scans,
one-second whole-process exit proof, query-lock boundary, compilation and
CUDA graphs remain unchanged.

The separate pre-command-failure reader retains6completed commands and
2accepted MuonT controls without inventing a failed training execution.
It requires every remaining output directory, log and execution sidecar
to be absent, including symlinks, and preserves consumed-owner/marker
checks and older seal checking. All17local predecessor/source cases pass;
the earlier16pass/1failure fixture-closure result remains retained.
Test evidence:
`PVC/run-preparation/test-streaming-diagnostic-closure-20261120T0156.xml`.
Required lint passes for all four new working files after formatting.
The earlier299-case journal/scanner/churn result remains separate evidence.

This is preparation and local CPU evidence only. The generated
archive-local study fixtures have not yet passed the planned201-test
archived acceptance. Fresh archived-entry checking, an independently
checked failed-precommand predecessor seal, frozen binding/checking and
all unchanged GPU gates remain required. Neither this candidate nor the
accepted actual-LR adapter is launched. Subsequent full-budget waves,
large shared-token comparisons and independently verified private
full-history uploads remain pending.

Read-only checking near01:58 finds the original failed controller,
validator and waiter absent, the exact restored keeper3731474 and
ranks3731705-3731712 occupying all eight original GPUs, and sole monitor
1588227/start933927668 live. The monitor publishes both journals at01:57:54
from its01:53:27report and returns to `sleep900`; no duplicate or restart
is made. The task controller reports the broader goal paused at this
status check, so no new CPU acceptance or GPU experiment is started here.
Both journals are archived under the shared lock before this milestone.

The earlier Wave2-running message is historical: its private SwanLab
receipt verifies all four histories on November16 at16:47:37, and
`inner_outer_all_lag` completed in Wave3, whose latest verification receipt
is November16 at17:50:36. Wave12 remains the latest full-budget analyzed
and cloud-verified wave, with all four histories verified November18
at02:04:24. Its fused arm has final C4 delta-0.002462 and late shared-C4
delta-0.001806 versus MuonT, but steady seconds/MTok are2.304 versus2.112
and recorded total time is536.9seconds longer. These exploratory seed0
results do not establish a more efficient, better-performing optimizer.

<!-- BEGIN AUTOMATED RESEARCH STATUS -->
## Recurring Experiment Status

Refreshed 2026-11-20 16:28:11, Asia/Shanghai, from the same authenticated
receipt-backed reports as the detailed ledger in [`DETAILS.md`](DETAILS.md).
Receipt times below are historical observations, not fresh process or GPU checks.
Only the two latest numbered wave reports and current diagnostic gates appear here.
Earlier results, large-window comparisons and manual experiment decisions remain above.
Candidate-minus-reference loss deltas are in nats; negative is better. Late means
use the final half of common observed tokens. Interim windows and correctness probes
do not establish full-budget quality, efficiency, production approval or cloud verification.

### Fused Matched-Null Full-Schedule Neighborhood

`PVC/candidates/fused-attribution-full-schedule-20261118T0257/fused-attribution-preparation.json`.
Nine exploratory seed-0 full-budget runs: MuonT/fused/identity-outer null at 0.04525483392, 0.06399999989260598 and 0.09050966784.
Each retains 5,120 updates/2,684,354,560 tokens; total 24,159,191,040 tokens planned.
Five large training windows and all exact evaluations; eleven final-half selection evaluations rank each family. Fresh seeds 3,4,5 remain reserved.
Identity-outer null retains output collection/moments/roots and factor application; timing remains diagnostic.
Source/archive-bound CPU suite exited zero with CUDA hidden at 2026-11-18 03:00:45.
The CPU archive verifies all 41 issued helpers; later working-helper edits do not replace that evidence.
GPU numerical/continuation gates and separate execution ownership remain required; no new wave results or cloud imports.

### Fused Attribution Actual-Config Validation

Saved preparation: `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/candidates/fused-attribution-validation-strict-20261118T0418/fused-attribution-validation-preparation.json`.
Source-bound nine-config validation: 27 eight-rank commands; 226,492,416 validation tokens planned.
Actual 5,120-update schedules remain fixed; bounded trajectories are independent continuous 24 versus interrupted 22 and resumed to 24.
The allocation predecessor remains the preserved failed profiler, not a passed profiler or a new production result.
Source/archive-bound validation CPU suite exited zero with CUDA hidden at 2026-11-18 04:22:05; `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/run-preparation/fused-attribution-validation-cpu/strict-20261118T0420/receipt.json`.
CPU acceptance uses its issued archive, not later mutable working helpers.
Saved validation-only binding: `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/run-preparation/fused-attribution-validation-binding-20261118.json`.
Saved attribution controller intent `acknowledged`; owner `881447/928247224`; not fresh process liveness.
Saved attribution state `needs_attention` at 2026-11-18 05:24:39; spawn status `acknowledged`.
Saved validator owner `884726/928265274`; log `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/controllers/fused-attribution-validation-20261118/validator.log`.
Preserved attribution error: `RuntimeError: Attribution bounded GPU validator failed`.
Preserved attribution keeper_restore: `0`.
Saved command `muont_full_lr_lower/continuous`.
Saved command `muont_full_lr_lower/interrupted`.
Saved command `muont_full_lr_lower/resumed`.
Saved command `outer_leftdiag_fused_full_lr_lower/continuous`.
Saved command `outer_leftdiag_fused_full_lr_lower/interrupted`.
Saved command `outer_leftdiag_fused_full_lr_lower/resumed`.
Source-bound saved attribution GPU commands: 6 / 27 exited zero; 0 failed at 2026-11-18 05:24:21; `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/run-preparation/validation-fused-attribution-gpu-20261118.json`.
Saved attribution GPU receipt status `needs_attention`.
Partial progress counts exited-zero commands only. Check flags below are saved reports, not independent numerical acceptance.

| Actual Config Trial | Saved Check Flags |
| --- | --- |
| `muont_full_lr_lower` | actual_full_budget_config=true, compiled_model=true, cached_root_checkpoint=true, boundary_continuation=true, post_cached_root_continuation=true, fixed_age_history=true, all_rank_checkpoint_identity=true, loss_continuation=true |
| `outer_leftdiag_fused_full_lr_lower` | actual_full_budget_config=false, compiled_model=false, cached_root_checkpoint=false, boundary_continuation=false, post_cached_root_continuation=false, fixed_age_history=false, all_rank_checkpoint_identity=false, loss_continuation=false, captured_diagonal_rows=false, outer_diagonal_update_parity=false, output_work_retained=false |
| `null_leftdiag_fused_full_lr_lower` | actual_full_budget_config=false, compiled_model=false, cached_root_checkpoint=false, boundary_continuation=false, post_cached_root_continuation=false, fixed_age_history=false, all_rank_checkpoint_identity=false, loss_continuation=false, captured_diagonal_rows=false, outer_diagonal_update_parity=false, output_work_retained=false, null_identity_outer=false |
| `muont_full_lr_center` | actual_full_budget_config=false, compiled_model=false, cached_root_checkpoint=false, boundary_continuation=false, post_cached_root_continuation=false, fixed_age_history=false, all_rank_checkpoint_identity=false, loss_continuation=false |
| `outer_leftdiag_fused_full_lr_center` | actual_full_budget_config=false, compiled_model=false, cached_root_checkpoint=false, boundary_continuation=false, post_cached_root_continuation=false, fixed_age_history=false, all_rank_checkpoint_identity=false, loss_continuation=false, captured_diagonal_rows=false, outer_diagonal_update_parity=false, output_work_retained=false |
| `null_leftdiag_fused_full_lr_center` | actual_full_budget_config=false, compiled_model=false, cached_root_checkpoint=false, boundary_continuation=false, post_cached_root_continuation=false, fixed_age_history=false, all_rank_checkpoint_identity=false, loss_continuation=false, captured_diagonal_rows=false, outer_diagonal_update_parity=false, output_work_retained=false, null_identity_outer=false |
| `muont_full_lr_upper` | actual_full_budget_config=false, compiled_model=false, cached_root_checkpoint=false, boundary_continuation=false, post_cached_root_continuation=false, fixed_age_history=false, all_rank_checkpoint_identity=false, loss_continuation=false |
| `outer_leftdiag_fused_full_lr_upper` | actual_full_budget_config=false, compiled_model=false, cached_root_checkpoint=false, boundary_continuation=false, post_cached_root_continuation=false, fixed_age_history=false, all_rank_checkpoint_identity=false, loss_continuation=false, captured_diagonal_rows=false, outer_diagonal_update_parity=false, output_work_retained=false |
| `null_leftdiag_fused_full_lr_upper` | actual_full_budget_config=false, compiled_model=false, cached_root_checkpoint=false, boundary_continuation=false, post_cached_root_continuation=false, fixed_age_history=false, all_rank_checkpoint_identity=false, loss_continuation=false, captured_diagonal_rows=false, outer_diagonal_update_parity=false, output_work_retained=false, null_identity_outer=false |
Preserved attribution GPU error: `AssertionError: rank_0.model.body.emb.weight`.
Saved evidence does not inspect live ranks, authorize production or establish full-budget optimizer quality, timing or cloud results.

### Restored-Root GPU Diagnostic

`PVC/run-preparation/root-preparation-binding-20261118.json`.
Saved diagnostic freeze at 2026-11-18 13:24:37; twelve bounded commands.
Saved controller intent `acknowledged` at 2026-11-18 13:26:04.
Saved executor state `needs_attention` at 2026-11-18 15:59:52.
Recorded keeper restoration: `0`.
Preserved executor error: `RuntimeError: Root diagnostic validator failed`.
Saved GPU receipt `needs_attention` at 2026-11-18 15:59:34: 2/2 cold-probe exits and 12/12 full-model exits are integer zero.
Preserved GPU error: `ValueError: Root trace first-capture/next-replay lifecycle differs`.
Saved receipts do not establish live occupancy. No production acceptance, full-budget result or SwanLab import is inferred; all original gates remain required.

### Independent Failed-Root Reconstruction

`PVC/run-preparation/root-failure-reconstruction-20261118T1618`.
Saved one-shot acknowledgment at 2026-11-18 16:18:28: PID `1438221/932406416`.
Saved reconstruction state `reconstructed` at 2026-11-18 18:32:16.
Separate reconstructed receipt: `PVC/run-preparation/root-failed-diagnostic-reconstruction-20261118T1618.json`.
Saved diagnostic predicate `cold_capture_repeat_safety`: `True`.
Saved diagnostic predicate `trace_transparency`: `True`.
Saved diagnostic predicate `continuous_control_unchanged`: `True`.
Saved diagnostic predicate `unprepared_original_reproduced`: `True`.
Saved diagnostic predicate `prepared_all_rank_continuation`: `False`.
The original GPU receipt remains `needs_attention`. Saved acknowledgment does not establish process liveness. No CUDA launch, continuation acceptance, optimizer result or SwanLab import is inferred.

### Root-Stage Terminal And Precision Follow-Up

`PVC/run-preparation/root-stage-terminal-followup-20261118T1718`.
Saved precision CPU acceptance at 2026-11-18 17:14:26: 76 tests, 91 helpers and 6 fixtures.
Saved read-only follow-up CPU acceptance at 2026-11-18 17:21:31: six tests.
Saved follow-up acknowledgment at 2026-11-18 17:22:15: PID `1479257/932789135`.
Saved follow-up state `terminal_and_precision_reconstructed` at 2026-11-18 19:34:05; 900-second observations.
Read-only phase `terminal`: `complete`, recorded exit `0`.
Read-only phase `precision`: `complete`, recorded exit `0`.
Saved terminal reconstruction: `PVC/run-preparation/root-stage-terminal-independently-reconstructed-20261118T1718.json`; retained-output/restoration receipt bound.
Saved precision inspection: `PVC/run-preparation/root-stage-precision-independent-20261118T1718.json`; historical-output reproduction predicate `False`.
Saved acknowledgment is not current liveness. This renderer does not independently rederive numerical comparison, cast semantics or restoration. No optimizer result, efficiency or production acceptance is inferred.

### Separately Owned CUDA Start-Accuracy Trial

Saved CPU acceptance: `PVC/run-preparation/root-start-gpu-owner-cpu/20261118T1952/receipt.json` at 2026-11-18 19:51:31; 121 passing archived tests.
Saved freeze at 2026-11-18 19:55:26; two full-batch opposite-order processes.
Saved controller intent `acknowledged` at 2026-11-18 19:56:32.
Saved controller state `diagnostic_complete` at 2026-11-18 20:02:25; 2/2 workers with integer-zero execution and cleanup exits.
Recorded keeper restoration: `0`.
Saved predicate `all_cross_process_exact_finite`: `True`.
Saved predicate `all_repeat_outputs_exact_finite`: `True`.
Saved predicate `all_unaffected_outputs_exact_finite`: `True`.
Saved predicate `all_rational_O_within_fp64_tolerance`: `True`.
Saved process 0 `d` O/FP64 failures across both modes/four phases: pinned 96/96, rational 0/96.
Saved process 0 `f` O/FP64 failures across both modes/four phases: pinned 32/32, rational 0/32.
Saved process 1 `d` O/FP64 failures across both modes/four phases: pinned 96/96, rational 0/96.
Saved process 1 `f` O/FP64 failures across both modes/four phases: pinned 32/32, rational 0/32.
Saved independent retained-output/restoration reconstruction: `PVC/run-preparation/root-start-accuracy-terminal-independent-20261118T2001.json`.
Saved state is not current liveness. The renderer authenticates small receipt/source bytes, not retained tensor arithmetic or current CUDA occupancy. Full-root accuracy, original capture-context reproduction, full-model continuation, runtime, quality and production gates remain required. No production wave, efficiency improvement or SwanLab import is inferred.

### Rational-Start Full-Model Continuation Diagnostic

Accepted CPU/source archive: `PVC/run-preparation/rational-continuation-cpu/common-root-20261118T2116/receipt.json` at 2026-11-18 21:18:19; 215 passing tests, 112 helpers and six fixtures.
Saved freeze at 2026-11-18 21:33:10; pinned/rational x control/trace, twelve eight-rank commands.
Seed 0, LR 0.04525483392, 524,288 tokens/update and original 5,120-update schedule; bounded 22/23/24 trajectories.
Saved controller intent `acknowledged` at 2026-11-18 21:43:06.
Saved controller state `diagnostic_complete` at 2026-11-19 02:53:29.
Recorded keeper restoration: `0`.
Retained command outcomes: 12/12; 12 integer-zero exits.
Saved diagnostic predicate `trace_transparency`: `True`.
Saved diagnostic predicate `pinned_original_reproduced`: `True`.
Saved diagnostic predicate `rational_all_rank_continuation`: `False`.
Saved separate read-only follow-up `terminal_reconstructed` at 2026-11-19 05:21:51; original-owner checks every 900 seconds.
Saved independently reconstructed terminal: `PVC/run-preparation/rational-continuation-terminal-independent-20261118.json`; intervention accepted `False`.
Receipt-backed state does not establish current liveness or tensor arithmetic. Independent retained-result and exact-restoration reconstruction remain required. Compilation, CUDA graphs and original tolerances stay enabled. This bounded diagnostic supplies no production authorization, full-budget quality, efficiency or SwanLab import.

### NS Budget Diagnostic And Queue

Saved queue CPU acceptance at 2026-11-19 00:24:32; 21 passing tests, 124 helpers and two fixtures.
Saved owner CPU acceptance at 2026-11-19 00:03:08; 75 passing tests.
Six pinned BF16-operand/FP32 arms at five/seven/ten steps; two full-batch opposite-order processes.
Saved queue acknowledgment at 2026-11-19 00:25:30; 900-second original-owner checks.
Saved queue `ns_diagnostic_terminal_reconstructed` at 2026-11-19 05:47:10; 4/4 phases.
Saved original-owner observation: controller `absent`, validator `absent`, terminal_waiter `absent`.
Saved diagnostic freeze at 2026-11-19 05:26:03; separate execution binding.
Saved diagnostic controller intent `acknowledged` at 2026-11-19 05:26:17.
Saved diagnostic `diagnostic_complete` at 2026-11-19 05:37:54; 2/2 completed workers.
Recorded keeper restoration: `0`.
Saved diagnostic predicate `all_ten_step_tpu_phases_reproduce_saved_outputs`: `False`.
Saved diagnostic predicate `all_compiled_phases_match_pinned_eager`: `False`.
Saved diagnostic predicate `all_graph_phases_match_compiled_call`: `True`.
Saved diagnostic predicate `all_same_input_repeats_exact_finite`: `True`.
Saved diagnostic predicate `historical_controls_interpretable`: `False`.
Saved cross-process exact finite predicate: `True`.

| Process | Operand Mode | Steps | Pair Gate Failures | Pair Relative Error | Replay Median ms | Matched q10/Arm Time Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `tpu` | 10 | 12/12 | 0.0369231 to 0.0369345 | 2.11834 to 2.40267 | 1 to 1 |
| 0 | `tpu` | 7 | 12/12 | 0.0217187 to 0.0217451 | 1.50349 to 1.70755 | 1.37703 to 1.52866 |
| 0 | `tpu` | 5 | 12/12 | 0.00927204 to 0.00927247 | 1.05003 to 1.19506 | 1.77258 to 2.28819 |
| 0 | `f32` | 10 | 12/12 | 3.71149e-05 to 3.73519e-05 | 9.36093 to 9.36211 | 1 to 1 |
| 0 | `f32` | 7 | 12/12 | 1.98406e-05 to 1.98426e-05 | 6.56158 to 6.56678 | 1.42554 to 1.42681 |
| 0 | `f32` | 5 | 12/12 | 1.01491e-05 to 1.01541e-05 | 4.70112 to 4.70202 | 1.99083 to 1.99127 |
| 1 | `tpu` | 10 | 12/12 | 0.0369231 to 0.0369345 | 2.10763 to 2.12782 | 1 to 1 |
| 1 | `tpu` | 7 | 12/12 | 0.0217187 to 0.0217451 | 1.49274 to 1.7069 | 1.23477 to 1.41629 |
| 1 | `tpu` | 5 | 12/12 | 0.00927204 to 0.00927247 | 1.05197 to 1.05338 | 2.00254 to 2.02271 |
| 1 | `f32` | 10 | 12/12 | 3.71149e-05 to 3.73519e-05 | 9.35557 to 9.362 | 1 to 1 |
| 1 | `f32` | 7 | 12/12 | 1.98406e-05 to 1.98426e-05 | 6.56234 to 6.56557 | 1.42494 to 1.42645 |
| 1 | `f32` | 5 | 12/12 | 1.01491e-05 to 1.01541e-05 | 4.69971 to 5.95754 | 1.57054 to 1.99094 |

Ranges span saved phases/repeats or matched calls, not confidence intervals.

Saved independent retained-output/restoration reconstruction: `PVC/run-preparation/ns-budget-terminal-independent-20261119.json`.
Saved receipts do not prove current liveness or rederive retained tensors. CUDA-event timing excludes compilation, capture, transfers and training. Changed budgets are separate numerical arms; full-model gates, fair LR tuning, large shared-token windows and verified private histories remain required. No optimizer quality, elapsed-efficiency, continuation, production or SwanLab acceptance follows.

### NS Full-Model Validation Queue

Saved full-model queue evidence refused: `RuntimeError: NS owner imported different accepted helper bytes: owned_output_handoff.py`.

### NS Actual-LR Validation Queue

Saved actual-LR queue evidence refused: `RuntimeError: NS owner imported different accepted helper bytes: owned_output_handoff.py`.

### NS Original-Training-Path Diagnostic

Saved separate numerical CPU/source-archive acceptance at 2026-11-19 07:10:02: 34 passing CPU cases in 148.833 seconds; 143 authenticated helpers and 509 inventory entries.
Accepted numerical receipt: `PVC/run-preparation/ns-training-path-validation-component-cpu/20261119T0707/receipt.json`; inherited numerical candidate, not an accepted execution owner.
The distinct plan has eighteen lower-LR cases and 54 bounded eight-rank commands (continuous24/interrupted22/resumed24); these are planned commands, not GPU executions or accepted GPU cases.
Saved separately accepted execution-owner CPU archive at 2026-11-19 07:38:31: 55 passing CPU cases in 23.735 seconds; 149 authenticated helpers and 517 inventory entries.
Owner candidate: `/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/candidates/ns-training-path-owner-closure-corrected-20261119T0736`.
Saved execution phase `needs_attention` at 2026-11-19 13:23:14.
44/54 authenticated zero-exit commands; command exits alone are not numerical acceptance.
Saved controller PID/start `2099685/937961924`; validator spawn `acknowledged`.
Saved validator PID/start `2100106/937966294`.
Preserved error: `RuntimeError: NS training-path bounded validator failed`.
Preserved result_error: `ValueError: NS full-model command, exact process generation, cleanup or log identity differs`.
Saved case predicates: 6 accepted, 8 rejected; result status `needs_attention`. These are receipt predicates, not freshly reconstructed tensors.
Case `muont_ns_tpu_q5_full_lr_lower`: saved accepted `True`; false gates `none`.
Case `muont_ns_tpu_q7_full_lr_lower`: saved accepted `True`; false gates `none`.
Case `muont_ns_tpu_q10_full_lr_lower`: saved accepted `True`; false gates `none`.
Case `muont_ns_f32_q5_full_lr_lower`: saved accepted `True`; false gates `none`.
Case `muont_ns_f32_q7_full_lr_lower`: saved accepted `True`; false gates `none`.
Case `muont_ns_f32_q10_full_lr_lower`: saved accepted `True`; false gates `none`.
Case `outer_leftdiag_fused_ns_tpu_q5_full_lr_lower`: saved accepted `False`; false gates `post_cached_root_all_rank_continuation,scalar_loss_continuation`.
Case `outer_leftdiag_fused_ns_tpu_q7_full_lr_lower`: saved accepted `False`; false gates `post_cached_root_all_rank_continuation,scalar_loss_continuation`.
Case `outer_leftdiag_fused_ns_tpu_q10_full_lr_lower`: saved accepted `False`; false gates `post_cached_root_all_rank_continuation,scalar_loss_continuation`.
Case `outer_leftdiag_fused_ns_f32_q5_full_lr_lower`: saved accepted `False`; false gates `post_cached_root_all_rank_continuation,scalar_loss_continuation`.
Case `outer_leftdiag_fused_ns_f32_q7_full_lr_lower`: saved accepted `False`; false gates `post_cached_root_all_rank_continuation,scalar_loss_continuation`.
Case `outer_leftdiag_fused_ns_f32_q10_full_lr_lower`: saved accepted `False`; false gates `post_cached_root_all_rank_continuation,scalar_loss_continuation`.
Case `null_leftdiag_fused_ns_tpu_q5_full_lr_lower`: saved accepted `False`; false gates `post_cached_root_all_rank_continuation,scalar_loss_continuation`.
Case `null_leftdiag_fused_ns_tpu_q7_full_lr_lower`: saved accepted `False`; false gates `post_cached_root_all_rank_continuation,scalar_loss_continuation`.
One-shot terminal outcome `not_recorded`; only a bound complete outcome records independent reconstruction and exact restoration.
Reporting checks issued file/archive evidence only; it does not reconstruct tensors, inspect current liveness or grant launch authority. Rejected historical controls remain rejected.
CPU tests are not GPU results, optimizer superiority, causal proof, full-budget quality, whole-training efficiency, production authority or SwanLab acceptance.
Saved read-only terminal waiter `needs_attention` at 2026-11-19 13:32:52; not a fresh process check.
Waiter PID/start `2189311/938341032`; 19 retained original-owner observations at 900-second intervals.
Saved original owner states: `{"controller": "live_exact_generation", "validator": "live_exact_generation"}`.
Preserved read-only terminal waiter error: `RuntimeError: Training-path original owners exited without complete cleanup/restoration`.

### NS Marker-Exit Recovery Diagnostic

CPU/source acceptance at 2026-11-19 14:15:49: 103 passing cases, 160 helpers and 529 inventory entries.
The failed original remains 44/54 commands, 14/18 decisions, six accepted/eight rejected; its separate preservation seal is dated 2026-11-19 14:18:25.
Recovery saved status: `needs_attention`; 2/54 completed commands and 0/18 case decisions.
Source/configuration changes are restricted to the owner adapter and the measured handoff helper. Original numerical readers, tolerances, compilation and CUDA graphs are retained.
Controller receipt at 2026-11-19 14:33:29: `2855659/940361750`. This is a saved observation and does not establish current liveness or GPU occupancy.
Retained execution error: `RuntimeError: NS training-path bounded validator failed`.
Failed command: `muont_ns_tpu_q5_full_lr_lower/resumed` at 2026-11-19 14:33:11; execution/cleanup exits1/1, marked descendants absent. It is excluded from completed trajectory counts.
Independent complete terminal and successor validation remain separately gated. This diagnostic supplies no full-budget optimizer-quality, whole-training efficiency or SwanLab import result.

### NS Whole-Process Exit-Drain Diagnostic

CPU/source acceptance at 2026-11-19 15:41:56: 125 passing cases, 167 helpers and 536 inventory entries.
Both failures remain original44/54commands14/18decisions and recovery2/54commands0/18decisions; the separate preservation seal is dated 2026-11-19 15:42:41.
Exit-drain saved status: `needs_attention`; 6/54 completed commands and 2/18 case decisions.
Source/configuration changes are restricted to the owner adapter and the measured handoff helper. Original numerical readers, tolerances, compilation and CUDA graphs are retained.
Controller receipt at 2026-11-19 16:33:20: `2985173/940861596`. This is a saved observation and does not establish current liveness or GPU occupancy.
Retained execution error: `RuntimeError: NS training-path bounded validator failed`.
Next-command receipt: `muont_ns_tpu_q10_full_lr_lower/continuous` at 2026-11-19 16:33:03; execution exit `1`, cleanup exit `1`, marked-descendant absence `True`. Excluded from completed trajectory counts.
- `muont_ns_tpu_q5_full_lr_lower`: accepted; failed gates: none.
- `muont_ns_tpu_q7_full_lr_lower`: accepted; failed gates: none.
Independent complete terminal and successor validation remain separately gated. This diagnostic supplies no full-budget optimizer-quality, whole-training efficiency or SwanLab import result.

### NS Mixed-Exit-Bracket Diagnostic

CPU/source acceptance at 2026-11-19 17:02:50: 129 passing cases, 173 helpers and 542 inventory entries.
Preserved failures remain original44/54commands14/18decisions, recovery2/54commands0/18decisions and exit-drain6/54commands2/18decisions; the separate preservation seal is dated 2026-11-19 17:05:15.
Transition saved status: `needs_attention`; 23/54 completed commands and 7/18 case decisions.
Source/configuration changes are restricted to the owner adapter and the measured handoff helper. Original numerical readers, tolerances, compilation and CUDA graphs are retained.
Controller receipt at 2026-11-19 19:59:40: `3118453/941361276`. This is a saved observation and does not establish current liveness or GPU occupancy.
Retained execution error: `RuntimeError: NS training-path bounded validator failed`.
Next-command receipt: `outer_leftdiag_fused_ns_tpu_q7_full_lr_lower/resumed` at 2026-11-19 19:59:21; execution exit `1`, cleanup exit `1`, marked-descendant absence `True`. Excluded from completed trajectory counts.
- `muont_ns_tpu_q5_full_lr_lower`: accepted; failed gates: none.
- `muont_ns_tpu_q7_full_lr_lower`: accepted; failed gates: none.
- `muont_ns_tpu_q10_full_lr_lower`: accepted; failed gates: none.
- `muont_ns_f32_q5_full_lr_lower`: accepted; failed gates: none.
- `muont_ns_f32_q7_full_lr_lower`: accepted; failed gates: none.
- `muont_ns_f32_q10_full_lr_lower`: accepted; failed gates: none.
- `outer_leftdiag_fused_ns_tpu_q5_full_lr_lower`: rejected; failed gates: post_cached_root_all_rank_continuation, scalar_loss_continuation.
Independent complete terminal and successor validation remain separately gated. This diagnostic supplies no full-budget optimizer-quality, whole-training efficiency or SwanLab import result.

### NS Transition Independent Terminal Waiter

Saved waiter status `needs_attention`; original-owner checks every900seconds.
CPU/source archive:24 passing cases,175 helpers; `run-preparation/ns-training-path-exit-transition-followup-cpu/20261119T1718/receipt.json`.
Saved waiter PID/start `3127027/941414079`; observation time 2026-11-19 20:04:58.
Saved original owner states: `{'controller': 'live_exact_generation', 'validator': 'live_exact_generation'}`.
Retained waiter error: `RuntimeError: Training-path original owners exited without complete cleanup/restoration`.
No keeper transfer, new GPU command, numerical waiver, production or SwanLab authority.

### NS Transition Actual-LR Queue

Saved queue CPU/source-archive acceptance at 2026-11-19 18:58:03; 47 passing cases and 198 authenticated helpers.
Accepted execution-owner parent: `PVC/run-preparation/ns-training-path-transition-actual-lr-execution-cpu/20261119T1822/receipt.json`.
54 actual-LR configurations and 162 bounded eight-rank commands retain the original 5,120-update schedule.
The existing original waiter owns its sole canonical terminal; this queue invokes no original terminal.
Saved queue intent `acknowledged` at 2026-11-19 19:04:03; original-owner checks every 900 seconds.
Saved queue `needs_attention` at 2026-11-19 20:05:59; 0/6 completed phases.
Saved original-owner states: controller `live_exact_generation`, validator `live_exact_generation`, terminal_waiter `live_exact_generation`, command_3119049 `absent`, command_3123996 `absent`, command_3127140 `absent`, command_3136338 `absent`, command_3141070 `absent`, command_3147037 `absent`, command_3157594 `absent`, command_3167494 `absent`, command_3176064 `absent`, command_3199666 `absent`, command_3206297 `absent`, command_3209606 `absent`, command_3219349 `absent`, command_3236010 `absent`, command_3243312 `absent`, command_3262803 `absent`, command_3267603 `absent`, command_3270631 `absent`, command_3282129 `absent`, command_3287673 `absent`, command_3292073 `absent`.
Preserved queue failure: `RuntimeError: Transition-LR external evidence reader refused evidence (exit 1): rmalization/run-preparation/ns-training-path-transition-actual-lr-predecessor-cpu/staged-20261119T1748/tracking/ns_training_path_transition_lr_predecessor.py", line 200, in evidence
  File "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/run-preparation/ns-training-path-transition-actual-lr-predecessor-cpu/staged-20261119T1748/tracking/ns_training_path_transition_lr_predecessor.py", line 200, in <genexpr>
  File "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/run-preparation/ns-training-path-transition-actual-lr-predecessor-cpu/staged-20261119T1748/tracking/ns_training_path_transition_lr_predecessor.py", line 62, in saved
  File "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/run-preparation/ns-training-path-transition-actual-lr-predecessor-cpu/staged-20261119T1748/tracking/ns_training_path_transition_lr_predecessor.py", line 54, in checked
    .replace("ns-training-path-exit-drain", "ns-training-path-exit-transition")
           ^^^^^^^^^^^^^^^^^^^
  File "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/run-preparation/ns-training-path-transition-actual-lr-predecessor-cpu/staged-20261119T1748/tracking/ns_training_path_transition_lr_predecessor.py", line 46, in file_identity
    ),

  File "/opt/conda/envs/torch-base/lib/python3.13/pathlib/_abc.py", line 437, in lstat
    return self.stat(follow_symlinks=False)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/torch-base/lib/python3.13/pathlib/_local.py", line 515, in stat
    return os.stat(self, follow_symlinks=follow_symlinks)
           ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/run-preparation/ns-training-path-exit-transition-20261119-terminal-f607ce7b8fa5bbfb332a3ad9e6084041670f1685a1674db44cbf8230f717ef48.json'
`.
Saved receipts do not establish current liveness or rederive tensor arithmetic. Failed phases and consumed claims remain retained.
Bounded continuation gates supply no production, full-budget quality, efficiency or SwanLab acceptance. Large shared-token comparisons and independently verified private full histories remain required.

### NS Missing-Environment Scan-Exit Diagnostic

CPU/source acceptance at 2026-11-19 20:56:19: 129 passing cases, 178 helpers and 547 inventory entries.
Preserved failures remain original44/54commands14/18decisions, recovery2/54commands0/18decisions exit-drain6/54commands2/18decisions and transition23/54commands7/18decisions; the separate preservation seal is dated 2026-11-19 20:59:40.
Scan-exit saved status: `needs_attention`; 8/54 completed commands and 2/18 case decisions.
Source/configuration changes are restricted to the owner adapter and the measured handoff helper. Original numerical readers, tolerances, compilation and CUDA graphs are retained.
Controller receipt at 2026-11-19 22:04:12: `3403676/942778625`. This is a saved observation and does not establish current liveness or GPU occupancy.
Retained execution error: `RuntimeError: NS training-path bounded validator failed`.
Next-command receipt: `muont_ns_tpu_q10_full_lr_lower/resumed` at 2026-11-19 22:03:52; execution exit `1`, cleanup exit `1`, marked-descendant absence `True`. Excluded from completed trajectory counts.
- `muont_ns_tpu_q5_full_lr_lower`: accepted; failed gates: none.
- `muont_ns_tpu_q7_full_lr_lower`: accepted; failed gates: none.
Independent complete terminal and successor validation remain separately gated. This diagnostic supplies no full-budget optimizer-quality, whole-training efficiency or SwanLab import result.

### NS Whole-Process Recheck-Exit Diagnostic

CPU/source acceptance at 2026-11-19 22:42:24: 136 passing cases, 183 helpers and 552 inventory entries.
Preserved failures remain original44/54commands14/18decisions, recovery2/54commands0/18decisions exit-drain6/54commands2/18decisions and transition23/54commands7/18decisions and scan-exit8/54commands2/18decisions; the separate preservation seal is dated 2026-11-19 22:44:56.
Recheck-exit saved status: `needs_attention`; 1/54 completed commands and 0/18 case decisions.
Source/configuration changes are restricted to the owner adapter and the measured handoff helper. Original numerical readers, tolerances, compilation and CUDA graphs are retained.
Controller receipt at 2026-11-19 23:00:07: `3557212/943407368`. This is a saved observation and does not establish current liveness or GPU occupancy.
Retained execution error: `RuntimeError: NS training-path bounded validator failed`.
Next-command receipt: `muont_ns_tpu_q5_full_lr_lower/interrupted` at 2026-11-19 22:59:49; execution exit `1`, cleanup exit `1`, marked-descendant absence `True`. Excluded from completed trajectory counts.
Independent complete terminal and successor validation remain separately gated. This diagnostic supplies no full-budget optimizer-quality, whole-training efficiency or SwanLab import result.

### NS Serialized Ownership Diagnostic

CPU/source acceptance at 2026-11-19 23:35:02: 141 passing cases, 189 helpers and 558 inventory entries.
Preserved failures remain original44/54commands14/18decisions, recovery2/54commands0/18decisions exit-drain6/54commands2/18decisions and transition23/54commands7/18decisions and scan-exit8/54commands2/18decisions and recheck1/54commands0/18decisions; the separate preservation seal is dated 2026-11-19 23:37:10.
Serialized saved status: `needs_attention`; 0/54 completed commands and 0/18 case decisions.
Source/configuration changes are restricted to the owner adapter and full ownership-inspection serialization; the accepted handoff helper and trainer/configuration bytes are retained. Original numerical readers, tolerances, compilation and CUDA graphs are retained.
Controller receipt at 2026-11-19 23:54:30: `3595881/943762646`. This is a saved observation and does not establish current liveness or GPU occupancy.
Retained execution error: `RuntimeError: NS training-path bounded validator failed`.
Next-command receipt: `muont_ns_tpu_q5_full_lr_lower/continuous` at 2026-11-19 23:54:13; execution exit `1`, cleanup exit `1`, marked-descendant absence `True`. Excluded from completed trajectory counts.
Independent complete terminal and successor validation remain separately gated. This diagnostic supplies no full-budget optimizer-quality, whole-training efficiency or SwanLab import result.

### NS Query-Serialized Ownership Diagnostic

CPU/source acceptance at 2026-11-20 00:11:29: 142 passing cases, 193 helpers and 562 inventory entries.
Preserved failures remain original44/54commands14/18decisions, recovery2/54commands0/18decisions exit-drain6/54commands2/18decisions and transition23/54commands7/18decisions and scan-exit8/54commands2/18decisions and recheck1/54commands0/18decisions and whole-inspection0/54commands0/18decisions; the separate preservation seal is dated 2026-11-20 00:13:48.
Query-serialized saved status: `needs_attention`; 6/54 completed commands and 2/18 case decisions.
The owner adapter and query/marker-scan lock boundary change; source authentication remains outside the unchanged 30-second lock. The handoff helper, trainer/configuration bytes and every ownership statement/order are retained. Original numerical readers, tolerances, compilation and CUDA graphs are retained.
Controller receipt at 2026-11-20 01:15:06: `3633332/943964742`. This is a saved observation and does not establish current liveness or GPU occupancy.
Retained execution error: `RuntimeError: NS training-path bounded validator failed`.
Next-command receipt: `muont_ns_tpu_q10_full_lr_lower/continuous` at not recorded; execution exit `not recorded`, cleanup exit `not recorded`, marked-descendant absence `not recorded`. No next-command execution receipt is recorded.
- `muont_ns_tpu_q5_full_lr_lower`: accepted; failed gates: none.
- `muont_ns_tpu_q7_full_lr_lower`: accepted; failed gates: none.
Independent complete terminal and successor validation remain separately gated. This diagnostic supplies no full-budget optimizer-quality, whole-training efficiency or SwanLab import result.

### NS Query-Serialized Independent Terminal Waiter

Saved waiter status `needs_attention`; original-owner checks every900seconds.
CPU/source archive:24 passing cases,195 helpers; `run-preparation/ns-training-path-query-serialized-followup-cpu/20261120T0026/receipt.json`.
Saved waiter PID/start `3638945/943982783`; observation time 2026-11-20 01:28:01.
Saved original owner states: `{'controller': 'live_exact_generation', 'validator': 'live_exact_generation'}`.
Retained waiter error: `RuntimeError: Training-path original owners exited without complete cleanup/restoration`.
No keeper transfer, new GPU command, numerical waiver, production or SwanLab authority.

### NS Serialized Independent Terminal Waiter

Saved waiter status `needs_attention`; original-owner checks every900seconds.
CPU/source archive:24 passing cases,191 helpers; `run-preparation/ns-training-path-serialized-followup-cpu/20261119T2353/receipt.json`.
Saved waiter PID/start `3598600/943780809`; observation time 2026-11-20 00:09:18.
Saved original owner states: `{'controller': 'live_exact_generation', 'validator': 'absent'}`.
Retained waiter error: `RuntimeError: Training-path original owners exited without complete cleanup/restoration`.
No keeper transfer, new GPU command, numerical waiver, production or SwanLab authority.

### NS Recheck-Exit Independent Terminal Waiter

Saved waiter status `needs_attention`; original-owner checks every900seconds.
CPU/source archive:24 passing cases,185 helpers; `run-preparation/ns-training-path-recheck-exit-followup-cpu/20261119T2253/receipt.json`.
Saved waiter PID/start `3559219/943421918`; observation time 2026-11-19 23:09:30.
Saved original owner states: `{'controller': 'live_exact_generation', 'validator': 'live_exact_generation'}`.
Retained waiter error: `RuntimeError: Training-path original owners exited without complete cleanup/restoration`.
No keeper transfer, new GPU command, numerical waiver, production or SwanLab authority.

### NS Scan-Exit Independent Terminal Waiter

Saved waiter status `needs_attention`; original-owner checks every900seconds.
CPU/source archive:24 passing cases,180 helpers; `run-preparation/ns-training-path-scan-exit-followup-cpu/20261119T2110/receipt.json`.
Saved waiter PID/start `3412036/942807401`; observation time 2026-11-19 22:12:06.
Saved original owner states: `{'controller': 'live_exact_generation', 'validator': 'live_exact_generation'}`.
Retained waiter error: `RuntimeError: Training-path original owners exited without complete cleanup/restoration`.
No keeper transfer, new GPU command, numerical waiver, production or SwanLab authority.

### NS Scan-Exit Actual-LR Queue

Saved queue CPU/source-archive acceptance at 2026-11-19 22:05:07; 47 passing cases and 214 authenticated helpers.
Accepted execution-owner parent: `PVC/run-preparation/ns-training-path-scan-exit-actual-lr-execution-cpu/20261119T2154/receipt.json`.
54 actual-LR configurations and 162 bounded eight-rank commands retain the original 5,120-update schedule.
The existing original waiter owns its sole canonical terminal; this queue invokes no original terminal.
Saved receipts do not establish current liveness or rederive tensor arithmetic. Failed phases and consumed claims remain retained.
Bounded continuation gates supply no production, full-budget quality, efficiency or SwanLab acceptance. Large shared-token comparisons and independently verified private full histories remain required.

### NS Exit-Drain Independent Terminal Waiter

Saved waiter status `needs_attention`; original-owner checks every900seconds.
CPU/source archive:24 passing cases,169 helpers; `run-preparation/ns-training-path-exit-drain-followup-cpu/20261119T1620/receipt.json`.
Saved waiter PID/start `3071163/941088367`; observation time 2026-11-19 16:40:32.
Saved original owner states: `{'controller': 'live_exact_generation', 'validator': 'live_exact_generation'}`.
Retained waiter error: `RuntimeError: Training-path original owners exited without complete cleanup/restoration`.
No keeper transfer, new GPU command, numerical waiver, production or SwanLab authority.

### NS Actual-LR Training-Path Queue

Saved queue CPU/source-archive acceptance at 2026-11-19 11:17:14; 33 passing cases and 180 authenticated helpers.
Accepted execution-owner parent: `PVC/run-preparation/ns-training-path-actual-lr-execution-owner-cpu/20261119T1051/receipt.json`.
54 actual-LR configurations and 162 bounded eight-rank commands retain the original 5,120-update schedule.
The existing original waiter owns its sole canonical terminal; this queue invokes no original terminal.
Saved queue intent `acknowledged` at 2026-11-19 11:20:12; original-owner checks every 900 seconds.
Saved queue `needs_attention` at 2026-11-19 13:36:09; 0/6 completed phases.
Saved original-owner states: controller `live_exact_generation`, validator `live_exact_generation`, terminal_waiter `live_exact_generation`, marked_descendants `marked_descendants_present`.
Preserved queue failure: `RuntimeError: Training-path actual-LR predecessor complete original terminal/report is unavailable`.
Saved receipts do not establish current liveness or rederive tensor arithmetic. Failed phases and consumed claims remain retained.
Bounded continuation gates supply no production, full-budget quality, efficiency or SwanLab acceptance. Large shared-token comparisons and independently verified private full histories remain required.

### wave11

Launch receipt: `PVC/run-preparation/wave11.json`; recorded 2026-11-17 21:51:34.
Saved monitor state: `needs_attention` at 2026-11-17 21:51:34.

| Run | Node | State | Updates / Budget |
| --- | --- | --- | ---: |
| `muont` | `28.216.17.9` | needs_attention | unknown / 5120 |
| `inner_outer_leftdiag` | `28.216.17.80` | needs_attention | unknown / 5120 |
| `outer_leftdiag` | `28.216.17.144` | needs_attention | unknown / 5120 |
| `outer_leftdiag_fused` | `28.216.17.139` | needs_attention | unknown / 5120 |

No complete-wave verified SwanLab receipt is recorded.

### wave12

Launch receipt: `PVC/run-preparation/wave12.json`; recorded 2026-11-18 02:02:22.
Saved monitor state: `complete` at 2026-11-18 02:02:22.

| Run | Node | State | Updates / Budget |
| --- | --- | --- | ---: |
| `muont` | `28.216.17.9` | complete | 5120 / 5120 |
| `inner_outer_leftdiag` | `28.216.17.80` | complete | 5120 / 5120 |
| `outer_leftdiag` | `28.216.17.144` | complete | 5120 / 5120 |
| `outer_leftdiag_fused` | `28.216.17.139` | complete | 5120 / 5120 |

SwanLab receipt: all 4 histories verified at 2026-11-18 02:04:24;
`PVC/run-preparation/swanlab-upload-wave12.json`.

Analysis artifact modified 2026-11-18 02:02:34:
`PVC/run-preparation/analysis-wave12.json`.

Comparison `wave12/inner_outer_leftdiag` minus `wave12/muont`: full-budget eligible.
Complete 536,870,912-token training-window deltas: +0.051593, +0.057542, +0.052361, +0.050558, +0.046858.
Exact-shared late C4 delta +0.049206 (11 evaluations); selection delta +0.049294 (11 evaluations).
Final C4 delta +0.042125; final selection delta +0.042416.
Steady seconds/MTok: candidate 2.421, reference 2.112.
Full-budget recorded total-time delta: +879.0 seconds.

Comparison `wave12/outer_leftdiag` minus `wave12/muont`: full-budget eligible.
Complete 536,870,912-token training-window deltas: +0.061019, +0.069954, +0.063745, +0.056893, +0.042437.
Exact-shared late C4 delta +0.048761 (11 evaluations); selection delta +0.050256 (11 evaluations).
Final C4 delta +0.027780; final selection delta +0.031214.
Steady seconds/MTok: candidate 2.322, reference 2.112.
Full-budget recorded total-time delta: +587.9 seconds.

Comparison `wave12/outer_leftdiag_fused` minus `wave12/muont`: full-budget eligible.
Complete 536,870,912-token training-window deltas: -0.005873, -0.006467, -0.003876, -0.003243, -0.002814.
Exact-shared late C4 delta -0.001806 (11 evaluations); selection delta -0.002909 (11 evaluations).
Final C4 delta -0.002462; final selection delta -0.002349.
Steady seconds/MTok: candidate 2.304, reference 2.112.
Full-budget recorded total-time delta: +536.9 seconds.


<!-- END AUTOMATED RESEARCH STATUS -->

## Document Management

Edit this document for changes to scientific scope, status, executed setup,
equations, results, interpretation, or next experiments. Record the date,
run identity, and supporting artifact when adding a new outcome.
Keep planned settings separate from executed settings.
Keep routine repeated monitoring prose compact; preserve raw receipts and
archive the preceding document before consolidating historical summaries.
Do not remove numerical results, failed evidence, hypotheses or source
identities to make room for another wave.

| Document or Directory | Role |
| --- | --- |
| `ENDEAVOR.md` | Authoritative research record: question, method, results, conclusions |
| `DETAILS.md` | Execution details: process receipts, source identities, checksums, automated wave ledger |
| `experiment/README.md` | Operational commands; retained with the training package |
| `tracking/README.md` | Historical-upload and verification commands |
| `artifacts/SOURCES.md` | Source provenance and pre-experiment technical review |
| `artifacts/original-proposal.txt` | Original scientific input, preserved byte-for-byte after the earlier whitespace formatting |

Superseded plan, status, audit pointer files, and historical Markdown copies
were removed after consolidation. The original proposal is a provenance
artifact, not an active configuration. The local source-manifest path now
points to its text snapshot while retaining the original captured checksum.
Do not edit downloaded papers, pinned author code, or past manifests to
match a later interpretation.

Per the November 17-19 continuation requests, keep new wave experiments and
results in this file. After the November 18 document split, the existing
900-second Bash document monitor updates the detailed receipt-backed
ledger in `DETAILS.md` and a bounded "Recurring Experiment Status" block
here from the same authenticated report. Either document entry point
refreshes both files; the active monitor does not need a restart.
All prose outside those two automated blocks is preserved. Add manual
milestone entries for a new hypothesis, validated release, acknowledged
launch, complete shared large-window comparison, failure, final result
or verified SwanLab import. Each entry records the observation date,
executed source/configuration and artifact identity. Keep prepared,
running, completed, analyzed and cloud-verified states separate.
Running-wave comparisons use complete shared token windows and exact
shared evaluations. Completion requires full-budget history and worker/
cleanup receipts; upload completion requires the full-resolution cloud
verification receipt. Neither a correctness probe nor a synthetic timing
establishes optimizer quality or whole-training efficiency.

November19 20:18 monitor verification: the original Bash monitor
publishes both documents20:12:22 and returns to `sleep900`, keeping
its original generation and exclusive lock. Multiline saved-error
text exposed two trailing-whitespace lines in those generated blocks.
`tracking/update_endeavor.py` now strips trailing whitespace only in
the replacement payload; raw receipts and all surrounding manual
prose remain unchanged. No running or accepted GPU source is edited.
The split-document regression passes with11focused cases; the full
journal/queue-reporting suite passes244cases in66.78seconds.
Supporting XML:
`PVC/run-preparation/test-endeavor-multiline-full-journal-20261119T2017.xml`.
Required scoped source/content lint and `git diff --check` pass,
with the pre-existing500KiB journal-size exception reported separately
in DETAILS. The existing monitor loads this renderer on its next cycle;
no duplicate monitor, uploader or experiment is launched.

November 19 20:39 publication verification: the precision import-closure
milestone and terminal-status table correction above are manually
published under the same shared lock. Six split-document, concurrent-edit,
monitor-retry and multiline regressions pass; fresh archived execution
binding authenticates all 163 helpers. Source and document-content lint
pass, with only the existing journal-size rule reported separately in
DETAILS. The sole original Bash monitor remains in its 900-second sleep;
new launches, large-window results and verified cloud imports will keep
their separate dated milestones in this document.

`experiment/deploy.py` now includes `ENDEAVOR.md` in future deployment
payloads. No remote deployment was run during cleanup; old worker-side files
were not deleted. Training mathematics, configurations, data, and existing
checkpoint files were unchanged.

Checkpoint identity hashes every `experiment/*.py`, including deployment
helpers. This document-path change therefore changes the current source
identity, even though it does not change training numerics. Old checkpoints
retain their executed identities and cannot be resumed from the modified
checkout without restoring the executed source or a separately reviewed
identity migration. Existing launch-validation receipts likewise refer to
their recorded source version; rerun required gates before a new launch.
