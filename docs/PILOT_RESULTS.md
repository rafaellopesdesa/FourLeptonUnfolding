# Production pilots: measured results and attempt history

Snapshot: 2026-11-13 18:21 UTC; includes signal production recovery and ZZ pilot.
Each configuration contains 1,000 generated events. This is a software and
production-sizing pilot, not an unfolding closure study.

| Configuration | Selected reco | Reco 115–130 GeV | Fiducial | Seconds | Peak RSS (MiB) | Compact bytes |
|---|---:|---:|---:|---:|---:|---:|
| Pythia nominal | 172 | 167 | 234 | 93 | 790 | 255194 |
| Pythia scale up | 172 | 167 | 234 | 94 | 789 | 255181 |
| Pythia scale down | 170 | 167 | 234 | 93 | 789 | 255104 |
| Herwig nominal | 140 | 134 | 201 | 96 | 650 | 253074 |

Selected counts refer to the extended analysis region; the narrower Higgs
window is shown separately. Correlated Pythia variants use identical hard
events; Herwig uses an independent seed. All selected rows have valid
certified Born kinematics. The small independent shower samples do not
establish the size/significance of an acceptance difference.

The complete generation → shower → detector → analysis chain ran on one
CPU per task. Slurm reports 80.5/84.2 CPU seconds for nominal Pythia/Herwig.
Peak RSS uses Slurm's reported KiB divided by 1024. Intermediate LHE/HepMC
and Delphes files were removed after compact-output validation.

The compact files and SHA256 values are in
[the full-chain summary](evidence/gg_H-summary-1634275.json). Persisted samples
are under `/project/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/samples/pilot_20261113/`.
Active I/O remains on /work. The normalized/decorated diagnostic files are
also persisted under `diagnostics/1634795/` within that pilot directory.

## Normalization and c_HB controls

The existing merger was used separately for each sample. Its checks give
sum(weight_shape)=N and sum(weight)=sigma × 312000 pb^-1. Raw Pythia and
Herwig weights have different conventions; comparing their unmerged sums
would be invalid. Normalized inclusive yields are 2931.56 and 2937.44 in
this small pilot; corresponding selected yields are 505.24 and 411.24.

The native scan uses c_HB = −0.1, 0, +0.1 at Lambda=1 TeV, retaining the
repository's squared dimension-six convention and fixed widths. All ratios
are finite and nonnegative; the c_HB=0 ratio is exactly one. At +0.1,
selected Herwig ratios range from 0.70765 to 2.42841, with median 0.99713.
The selected weighted yield becomes 410.56. These integrals fluctuate with
the finite template and are not coefficient estimates.

Tau-origin rows use the explicit `unsupported='unity'` diagnostic policy:

| Sample | Successfully evaluated | Unsupported tau, inclusive | Unsupported tau, selected |
|---|---:|---:|---:|
| Pythia nominal | 471/1000 | 529/1000 | 1/172 |
| Herwig nominal | 443/1000 | 557/1000 | 1/140 |

No unsupported row is marked as a successful matrix-element evaluation.
The final closure definition must specify this component's treatment.
See [native scan evidence](evidence/signal-checks-1634795.json) for status
counts, quantiles, signed yields and file hashes.

Pythia has one negative-weight event inclusively and none selected; Herwig
has none in these 1,000 rows. The selected signed-Poisson variance/net ratio
is one for this pilot. Larger samples can reveal negative selected weights;
this result does not validate positive data-like sampling at production scale.
No pseudo-data have been drawn or deformed after sampling.

## Attempt history

| Jobs | Outcome | Resolution |
|---|---|---|
| 1631376, 1631669, 1631672, 1631674 | Rejected LHE after generation/simulation | Accept only the standard trailing POWHEG RNG footer; preserve bytes/hash |
| 1633039–1633042 | Analysis tree produced; standalone validation import failed | Add repository import bootstrap and subprocess regression |
| 1634269, 1634272, 1634273, 1634274 | All four complete | Compact validation and persistence pass in 1634275 |
| 1634795 | Normalization, identity and native HB checks pass | Diagnostic ROOT files/checksums persisted |

Earlier shard diagnostics were copied into `runs/pilot/retries/<launcher-job>/`
before retry. Summary history includes Slurm IDs, so failures are not erased.
The reusable fixes are on main; tests and run inputs are on codex branches.

## Larger pilot and first production wave

All four 10,000-event pilots completed and passed validation/persistence in
summary 1635020. Their compact files are under the corresponding /project
`samples/pilot10k_20261113/` directory. Full evidence:
[10k summary](evidence/gg_H-summary-1635020.json).

| Configuration | Selected reco | Reco 115–130 GeV | Fiducial | Walltime | Compact bytes |
|---|---:|---:|---:|---:|---:|
| Pythia nominal | 1528 | 1487 | 2138 | 11 min 44 s | 1789348 |
| Pythia scale up | 1544 | 1494 | 2138 | 10 min 49 s | 1789444 |
| Pythia scale down | 1526 | 1480 | 2138 | 11 min | 1789425 |
| Herwig nominal | 1477 | 1425 | 1971 | 9 min 40 s | 1756573 |

Slurm's recorded peak RSS is approximately 2 GiB. All selected Born rows
are valid. Selected tau-origin counts are 2/3/1/2 respectively. Pythia has
two negative-weight rows inclusively and Herwig five, but none are selected;
the selected signed-Poisson variance/net ratio remains one in this pilot.
The positivity caveat for production therefore remains unresolved.

The measured acceptance now suggests about 33–34 million showered events
per signal configuration to reach five million selected events. Roughly
550–650 CPU hours per configuration is a naive throughput extrapolation,
before retries, grid/setup cost and production variability. Compact signal
storage is roughly 6 GB per configuration before merging/diagnostic copies.
Actual validated counts determine completion, not this estimate.

Launcher 1636246 completed after successful 10k summary and independently verified
persisted hashes/selected counts and storage headroom. It submitted exactly
100 × 10,000 events for each of the four signal configurations, with at most
four concurrent workers per array (16 total). Each requests one CPU, 3 GB,
one hour; 100 GB of /work and 10 GB of /project headroom are required below
the soft targets. A dependent job validates/persists every compact shard and
appends selected totals to the working status. No further wave is automatic.
Arrays 1636684 (nominal), 1636696 (up), 1636697 (down), and 1636700 (Herwig)
are running; summary 1636701 now waits for the recovery controller and its
dependent recovery jobs. This first wave requests one
million showered events per configuration and does
not satisfy the five-million-selected target by itself.

At 18:21 UTC, original-array successes were 28/28/28/27. Two completed Herwig
retries bring the total to 113/400 shards. A Slurm success is followed by the
independent summary's validation and checksum-checked persistence. The raw
[scheduler snapshot](evidence/production-status-20261113.json) retains failures;
Slurm combines not-yet-started pending tasks into a single record.

### Finite Herwig input and runtime deployment failure

Original Herwig tasks 4, 10, 17 and 31 exhausted their finite hard-event input.
Diagnostic 1642288 reproduced seed 40000005: one remnant-extraction failure
and one meson-decay retry limit discarded two events, leaving 9,998 accepted
events after 10,000 attempts. See [native statistics](evidence/herwig-eof-native-statistics.out)
and [log excerpts with original SHA256](evidence/herwig-eof-native-excerpts.json).

Check 1643289 supplied 10,100 hard events for 10,000 showered outputs. The
first 10,000 hard-event blocks were byte-identical to the failed source.
All 10,000 output source identities are unique, with zero repeats in this
check. The implementation requests a 1% hard-event reserve only for Herwig;
it never reopens the source or suppresses generator exceptions. Source and
accepted counts remain explicit in [provenance](evidence/herwig-buffer-1643289.json).
Targeted software checks pass (23 Generation + 8 BatchSubmit in 1644270).
Recovery array 1644720 has completed tasks 4 and 10; task 17 is running.

During deployment, I replaced the shared shell runner while workers were
executing it on NFS. Eight tasks then failed with stale file handles (one
nominal, four scale-up, two scale-down, one Herwig). Partial outputs were
rejected. Runtime paths are now frozen; future jobs use the immutable
[source/environment snapshot](evidence/runtime-snapshot.json). Diagnostics
are preserved before retry. Controller 1645888 waits for the original arrays
and initial retries, then retries known EOF/NFS failures once and updates the
summary dependency. Unknown failures stop it for investigation. The original
queued jobs retain the old Slurm-spooled worker, so additional EOF failures
may still need that final recovery round.

Summary 1646583 now distinguishes supplied hard events from showered rows and
records repeated source occurrences. The original 10k Herwig pilot has 9,999
unique source identities and one repeated occurrence. This is not removed;
shared hard identities must remain in the same training/test partition.

## Background pilot

ZZ integration 1631379 completed in 1 h 35 min. Summary 1643989 validated and
persisted all four 1,000-event configurations:

| Configuration | Selected reco | Reco 115–130 GeV | Fiducial | Inclusive negative rows | Selected tau rows |
|---|---:|---:|---:|---:|---:|
| Pythia nominal | 12 | 5 | 19 | 7 | 4 |
| Pythia scale up | 12 | 5 | 19 | 7 | 4 |
| Pythia scale down | 12 | 5 | 19 | 7 | 4 |
| Herwig nominal | 5 | 2 | 12 | 8 | 0 |

All selected Born kinematics are valid and no selected negative rows occur
in this pilot. The selected counts are too small for a reliable shower
comparison or final production estimate. Full [summary](evidence/ZZ-summary-1643989.json)
records signed yields, hashes and persistent paths.

Launcher 1646582 checks the pilot hashes and storage headroom, then submits
four 10k background shards with fresh seeds and the immutable runtime, one
CPU / 3 GB / one hour each. The launcher completed and submitted jobs
1646727–1646730, with dependent summary 1646731. The summary measures acceptance and
validates/persists the compact outputs; no larger ZZ wave is automatic.

## 19:44 UTC checkpoint: wave one and larger ZZ pilot

Signal completion is 225/400 shards (56.25%): 55 Pythia nominal, 56 up,
56 down, and 58 Herwig, including all three successful initial retries.
Sixteen workers remain active. The wave summary and /project publication wait
for completion and recovery. [Snapshot](evidence/production-status-20261113-checkpoint2.json)
records each configuration and its remaining failed task indices.

Thirteen shards await recovery: eight original NFS failures, four additional
Herwig EOF failures, and Pythia nominal task 37. The latter failed its existing
HepMC integrity guard after producing 10,000 showered rows; its cause is not
established. One same-seed retry retains the guard and preserves intermediates
for investigation if it recurs. No failed shard is accepted or silently omitted.

All four 10k ZZ jobs completed in 10–11 minutes. Summary 1646731 initially
missed scale-suffixed campaign names. Correcting filename discovery, with the
same expected counts and physics checks, gives a passing rerun in 1653805.
All compact outputs are now checksum-verified and persisted on /project.

| Configuration | Selected reco | Reco 115–130 GeV | Fiducial | Inclusive negative rows | Selected tau rows |
|---|---:|---:|---:|---:|---:|
| Pythia nominal | 77 | 24 | 107 | 97 | 12 |
| Pythia scale up | 77 | 25 | 107 | 97 | 13 |
| Pythia scale down | 71 | 23 | 107 | 97 | 9 |
| Herwig nominal | 81 | 28 | 102 | 84 | 12 |

All selected Born kinematics are valid. Selected net weight is now smaller
than selected absolute weight in every configuration: negative selected MC
is present. This confirms the need to validate the positive data-like intensity
before constructing Poisson pseudo-experiments. Approximate acceptance is
0.7–0.8%; this remains a timing pilot, not the five-million-selected background
production. [Full signed sums, source counts and hashes](evidence/ZZ-summary-1653805.json).

## 21:56 UTC checkpoint: original arrays finished, recovery running

Wave one has 384/400 successful shards: 98 nominal, 96 scale-up, 98 scale-down,
and 92 Herwig. These include the three initial successful Herwig retries.
Controller 1645888 completed at 21:51 UTC and submitted 16 remaining retries
on immutable code: arrays 1672510, 1672513, 1672518 and 1672519. Twelve tasks
are running and four are queued. The seven additional EOF failures are from
the original Slurm-spooled worker; recovery workers use the hard-event reserve.
The other causes are eight NFS failures and one Pythia provenance rejection.

Summary 1636701 is confirmed to wait for all four recovery arrays. The final
selected-event totals and wave-wide persistence have not yet been produced.
[Checkpoint and recovery task list](evidence/production-status-20261113-checkpoint3.json).
The original wave is complete only after all required shards pass validation.

## First signal wave complete

Summary 1636701 passed at 22:18 UTC on 13 November 2026. All 400 compact shards
were validated and persisted on /project, with no missing or rejected output.
The final recovery also resolved the isolated Pythia provenance rejection;
its original root cause remains undetermined and no guard was disabled.

| Configuration | Showered rows | Selected reco | Reco 115–130 GeV | Fiducial | Selected tau |
|---|---:|---:|---:|---:|---:|
| Pythia nominal | 1,000,000 | 160,171 | 154,700 | 220,453 | 355 |
| Pythia scale up | 1,000,000 | 161,104 | 154,883 | 220,453 | 367 |
| Pythia scale down | 1,000,000 | 158,946 | 153,646 | 220,453 | 340 |
| Herwig nominal | 1,000,000 | 148,747 | 143,773 | 204,914 | 343 |

All selected Born kinematics are valid. The Pythia samples have nonzero
negative selected weight; Herwig has none selected in this wave. Inclusive
negative counts are 429/429/429/391. The Herwig output includes 33 repeated
hard-source occurrences; these remain recorded and must share training splits
with their source events. Its supplied hard-event count is 1,001,100 because
the eleven recovery jobs used the reserve. Pythia supplied 1,000,000 each.

The four compact outputs total approximately 714 MB before diagnostic copies.
These counts reach approximately 3% of the five-million-selected target per
signal configuration. [Final aggregate evidence](evidence/signal-wave1-final-1636701.json)
includes signed sums and the checksum/location of the complete per-file manifest.
Raw Pythia/Herwig sums have different conventions and must not be compared as
physical yields before normalization.

## Scaling with node-local temporary storage

The user authorized up to 100 simultaneous jobs per physics process. The
bounded next wave has 200 shards × 10,000 showered events in each of eight
configurations: 16 million rows total. Four configurations × 25 workers give
100 gg_H and 100 ZZ workers. Each task requests one CPU, 4 GiB and one hour;
actual concurrency remains subject to the scheduler. There is no automatic
continuation beyond this bounded wave.

Live normal-QOS limits allow 1,000 running and 2,000 submitted jobs per user;
the new campaign fits within those limits. Inspected peak memory in earlier
3 GiB workers approached their request, motivating 4 GiB for this wave.
No CPU threads beyond the allocated core are requested by numerical libraries.

Inventory 1678427 measured 26.283 GB in /work and 0.732 GB in /project. Retained
10k jobs used 3.544–3.548 GB of intermediate files each. Running 200 of those
under /work would exceed the project limit. Unity documents the
[`localscratch` node feature](https://docs.unity.rc.umass.edu/documentation/cluster_specs/features/).
Probe 1678602 found `/localscratch` mounted from local ext4 storage on cpu045,
with 1.12 TB free. This differs from shared HPC Workspace scratch.

The local wrapper verifies the mount, checks at least 60 GB free initially,
samples task size every ten seconds, and stops on more than 8 GB per task or
less than 20 GB filesystem headroom. The eight real-input local pilots must
finish with validated outputs, no leftover task workspace, and sampled peaks
below 5 GB before the large arrays are released. These checks are not Slurm
disk reservations; other users can change local free space.

Preparation checks prerequisite /project hashes and reserves 50 GB below the
300 GB /work target and 10 GB below the 2 TB /project target. Successful final
summary is followed by checksum-protected removal of redundant compact /work
copies; persistent samples and diagnostics remain. All runtime and controller
files used by active workers are frozen snapshots.

Evidence: [project usage](evidence/storage-1678427.txt),
[retained task sizes and job-private /tmp](evidence/node-storage-1678572.txt),
[node-local mount](evidence/local-storage-1678602.txt), and
[seven campaign/cleanup checks](evidence/parallel-checks-1679082.txt).

### Local pilot and submission outcome

All eight local-storage pilots (1679053–1679060) passed with zero leftover
workspaces and no guard firing. Sampled peaks were 3.531 GB for each signal
Pythia setting, 3.035 GB for signal Herwig, 3.633–3.634 GB for ZZ Pythia, and
3.130 GB for ZZ Herwig. Walltimes ranged from 6 min 32 s to 8 min 32 s.
Nodes and event seeds differ from earlier pilots; this is not a controlled
filesystem performance comparison. [Per-task measurements](evidence/parallel-local-pilots.json).

Gate 1679061 also passed its renewed storage checks and submitted arrays
1679616–1679619 for gg_H and 1679620–1679623 for ZZ. Each has 199 remaining
10k tasks and a 25-worker throttle. The eight completed pilots are the first
shards of the same configurations, so the total remains 200 per configuration.
Recovery controller 1679624 waits for all arrays; final summary 1679625 waits
for recovery. Archive scheduler 1679187 attaches duplicate cleanup only after
successful final publication. [Immutable controls and submission manifest](evidence/parallel-wave2-submission.json).

## Large-wave completion and calorimeter investigation

Snapshot: 2026-11-14 11:12 UTC. Wave two saved 1,598 of 1,600 shards, totaling
15.98M showered rows. All four ZZ configurations passed; gg_H nominal Pythia
and Herwig each lack one 10k shard. Full counts and manifest checksums are in
[the checkpoint](evidence/parallel-wave2-checkpoint.json). The selection
efficiency is approximately 0.74% for ZZ and 14.6–16.1% for gg_H. These
counts do not constitute an unfolding result.

Pythia seed 50000044 and Herwig seed 60000046 both crashed at the same
`SimpleCalorimeter::Process` instruction. Their reported peak RSS was below
4 GiB, local free space remained above 194 GB, and the scratch guards did not
fire. The original logs and failed task directories are retained. The recovery
controller correctly refused this previously unrecognized error; summary
1679625 still persisted the 1,598 valid compact files and reported the two
holes. Cleanup 1679659 never ran.

The pinned Delphes 3.5.1 particle-hit branch indexes `fTrackFractions[number]`
with a particle index. This is outside the vector when there are more particles
than tracks and can dereference a null pointer for zero tracks. The crash
instruction is the indexed double load at `SimpleCalorimeter::Process+0xd23`
in the HepMC3 executable. Upstream fixed the neutral-energy calculation in
[7f2b717](https://github.com/delphes/delphes/commit/7f2b717c96671e78b6724b27aceea2580c9ce73a).
We backported that change into a separate installation without touching the
production runtime. A neutral-only executable regression and ten retained-input
cases are queued: both failed seeds and a successful control from each of the
eight configurations. Fixed outputs will be validated and compared branch by
branch. Until those checks pass, saved production samples are held from
training and no further bulk wave is submitted.

Diagnostics, configs and job records live under
`runs/calorimeter_diagnosis_20261114`; build 1706140 and comparison jobs
1706181/1706183–1706192. No original seed is replaced or event silently dropped.

Update 2026-11-14 11:25 UTC: the isolated build and neutral-only regression passed.
The old executable returns 129 with `SimpleCalorimeter::Process` in its crash
trace; the corrected executable returns zero and gives 16.56550407 GeV for
16.56550469 GeV of neutral input energy. Both original failing production
seeds also reproduced their crashes and completed all 10,000 events with the
fix. The eight successful-control comparisons are in progress. Gate 1706261
requires all checks and original LHE/HepMC hash equality before submitting
two repairs, validation and checksum-protected cleanup. Main now includes
the small upstream backport at e966bd6.

[Slurm accounting](evidence/parallel-wave2-scheduler.json) measures a peak of
200 workers, 100 per process; 1 h 41 min from the first pilot start to the
last production task; median task time 465 s; and 218.58 allocated CPU-hours.
The [full scratch audit](evidence/parallel-wave2-storage.json) covers all
1,600 tasks: largest sampled footprint 3.659 GB, least sampled free space
92.879 GB, zero guard trips and zero uncleaned task workspaces. A sequential
inventory during the retained-input diagnostics measured 64.111 GB /work and
3.700 GB /project. The temporary increase in /work is bounded diagnostic
input retention, not bulk production intermediates.

[Current diagnostic evidence](evidence/calorimeter-diagnosis-checkpoint.json)
records which comparisons have completed; two repaired diagnostic shards
are not yet counted as certified production outputs.
