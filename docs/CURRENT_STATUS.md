# FourLeptonStudy current status

<!-- DAY_PRODUCTION_BEGIN -->
## Production scheduled for 14 November

Updated: 2026-11-14T12:30:36.924633+00:00

State: **waiting_for_repair**. Queued after two repaired shards, full-wave validation and verified duplicate cleanup.
New waves stop at 2026-11-15T00:00:00+00:00; running jobs finish. Maximum eight waves, 16M showered rows each.
Concurrency cap: 100 gg_H + 100 ZZ. One CPU and 4 GiB per task.
Validated waves completed: 0.
Control record: `/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/production_day_20261114/state.json`.

<!-- DAY_PRODUCTION_END -->



Checkpoint updated 2026-11-14 12:36 UTC. All eight detector controls are now certified:
seven match across nodes exactly, and the remaining control matches exactly
when both outputs are analyzed on the same node. Its cross-node difference
is restricted to six Born angles, at most 8.88e-16. The neutral-only crash
regression also passes. No physics NN training or unfolding has run yet.

The two production repairs (1707700/1707701) completed. Full-wave validation
1707702 is running, followed by archive 1707703. Day launcher **1708008** waits
for that archive to succeed, then automatically runs further bounded waves.
Monitor **1708012** starts the 30-minute status updates. The active fixed runtime is
`runs/calorimeter_diagnosis_20261114/runtime`, with detector installation
`install/delphes-neutral-energy-20261114`; both remain frozen throughout this run. The automated block
above is authoritative for subsequent wave status and cumulative counts.

## Saved checkpoint before the day run

Counts below exclude pilots and combine signal waves one and two. These are
selected reconstructed events in the extended analysis region, not showered
input rows. The goal is at least five million selected events **per row**.

| Process / configuration | Saved selected events | Fraction of 5M target |
|---|---:|---:|
| gg_H Pythia nominal | 478742 | 9.57% |
| gg_H Pythia scale up | 483395 | 9.67% |
| gg_H Pythia scale down | 476806 | 9.54% |
| gg_H Herwig nominal | 440177 | 8.80% |
| ZZ Pythia nominal | 14830 | 0.297% |
| ZZ Pythia scale up | 14792 | 0.296% |
| ZZ Pythia scale down | 14843 | 0.297% |
| ZZ Herwig nominal | 14837 | 0.297% |

Wave two contains 15.98M showered rows and 2.968 GB of compact output. All
successful shards passed structural/provenance checks and were copied to
/project. Total project usage was 3.700 GB at 11:07 UTC. Reaching the selected
ZZ target takes much more generation: its measured selection efficiency is
about 0.74%, compared with 14.6–16.1% for gg_H in this wave.

The failed seeds are 50000044 (Pythia nominal, task 43) and 60000046
(Herwig nominal, task 45). Neither failure was an OOM or scratch-space guard.
Inspection identifies a particle index used on `fTrackFractions` in Delphes
3.5.1; the upstream correction is commit
[7f2b717](https://github.com/delphes/delphes/commit/7f2b717c96671e78b6724b27aceea2580c9ce73a).
The neutral-only regression reproduces the original crash and passes with
the fix (expected/measured energy 16.56550469/16.56550407 GeV). Both failed
production seeds complete with the fixed build. All eight controls are certified. The one non-identical cross-node comparison
was resolved by same-node analysis (1707521), exact across all 78 branches.

Diagnostic run: `runs/calorimeter_diagnosis_20261114`. Build 1706140; ten
original-code reproductions 1706142/1706144–1706152; fixed comparisons
1706181/1706183–1706191; neutral-only regression 1706192. The ten cases cover
the two failures and one successful control from each configuration. They
retain bounded intermediate inputs on /work for reuse, with 50 GB headroom
checked before the build. The original production runtime is untouched. The original strict comparison stopped gate 1706261. The same-node certificate
and explicit eight-epsilon bound allowed gate 1707522 to verify all ten input
LHE/HepMC hashes and submit the two repairs. Validation and cleanup gate the
day-long continuation described above.


Slurm accounting confirms a peak of **200 simultaneous tasks**, exactly 100
per process. The wave ran from 23:50:04 to 01:31:10 UTC (1 h 41 min), with a
median task duration of 465 s. These timings include the pilots and queue
ramp, and exclude the final validation job.

The full scratch audit found 1,600 usage records, maximum sampled per-task
usage 3.659 GB, minimum sampled free local disk 92.879 GB, zero guard trips,
and zero leftover workspaces. The current sequential inventory during the
retained-input diagnostics measured 64.111 GB /work and 3.700 GB /project;
these remain below the 300 GB / 2 TB targets.

## Confirmed configuration and authorization

- c_HB = +0.1 at Lambda = 1 TeV; controls 0 and −0.1. Repository LO amplitude
  convention, including the squared dimension-six term and fixed Higgs width.
- Seven NN coordinates: cos(theta1), cos(theta2), cos(theta*), Psi, Phi, mZ1,
  mZ2. Angles stay in radians. No m4l, y4l or pT4l NN inputs.
- 13.6 TeV and the repository luminosity convention of 312 fb^-1.
- No regularization or fitted score calibration. Default ensemble: four
  networks, four hidden layers with 1024 neurons. Tune reco_pull separately.
- The user authorized up to **100 concurrent jobs per physics process** and
  suggested worker scratch. The new wave has 25 slots per configuration:
  100 gg_H plus 100 ZZ. This is a limit; Slurm determines actual concurrency.
- Node-local intermediates are temporary. Code, controls and diagnostics stay
  in this /work project; final samples/models go to the corresponding /project.

## First signal wave: complete

Summary 1636701 passed at 2026-11-13 22:18 UTC. All 400 shards were validated,
checksum-checked and persisted under /project `samples/production_wave1_20261113/`.
Each configuration has one million showered rows.

| Configuration | Selected reconstructed | Reco 115–130 GeV | Fiducial |
|---|---:|---:|---:|
| Pythia nominal | 160171 | 154700 | 220453 |
| Pythia scale up | 161104 | 154883 | 220453 |
| Pythia scale down | 158946 | 153646 | 220453 |
| Herwig nominal | 148747 | 143773 | 204914 |

All selected Born kinematics are valid. Pythia has nonzero negative selected
weight; Herwig has none selected in this wave. Herwig has 33 repeated hard
source occurrences, retained in metadata and subject to shared-source training
splits. It supplied 1,001,100 hard events because eleven recovery shards used
the extra-event reserve. The original recovery failures are documented in
`report/docs/PILOT_RESULTS.md`; all required replacements passed.

These samples reach about 3% of the five-million-selected target per signal
configuration. Evidence: `runs/production_signal_wave1/gg_H-summary-1636701.json`.

## Background pilot: complete

ZZ integration and all four 1k/10k pilots passed. Final 10k summary 1653805
validated/persisted the outputs after correcting scale-suffix filename discovery.

| Configuration | Showered rows | Selected reco | Reco 115–130 GeV | Fiducial |
|---|---:|---:|---:|---:|
| Pythia nominal | 10000 | 77 | 24 | 107 |
| Pythia scale up | 10000 | 77 | 25 | 107 |
| Pythia scale down | 10000 | 71 | 23 | 107 |
| Herwig nominal | 10000 | 81 | 28 | 102 |

Negative selected weight occurs in every configuration. Signed MC remains
intact; positive data-like intensity construction needs validation before
Poisson pseudo-experiments. Selected tau rows are 12/13/9/12. All selected
Born kinematics are valid. The first background production wave is summarized above.

## Larger wave and live controls

Run area: `runs/production_parallel_wave2`; tag `production_parallel_wave2_20261113`.
Two million showered events per configuration, eight configurations: 1,600
shards and 16 million rows total. Each task requests one CPU, 4 GiB and one
hour on `cpu` with `--constraint=localscratch`.

| Job | Purpose | Recorded state |
|---|---|---|
| 1678427 | Project storage inventory | Complete |
| 1678572 / 1678602 | Temporary/local storage probes | Complete |
| 1678870 / 1679082 | Five gate checks / seven checks including cleanup | Passed |
| 1678905 | Verify prerequisites and prepare immutable controls | Complete |
| 1679053–1679060 | Eight 10k local-disk pilots | All passed; local cleanup verified |
| 1679061 | Gate and submit remaining 1,592 tasks | Passed and submitted all eight arrays |
| 1679616–1679619 | Four gg_H bulk arrays | Terminal; two Delphes failures |
| 1679620–1679623 | Four ZZ bulk arrays | All completed |
| 1679624 | Bounded recovery controller | Stopped on unknown calorimeter crash |
| 1679625 | Validate/persist the entire wave | Saved 1598 shards; failed overall with two missing |
| 1679187 | Attach verified duplicate cleanup | Completed |
| 1679659 | Verify saved hashes and remove duplicate work shards | Cancelled after summary failure; no deletion |

The eight local pilots finished in 6 min 32 s to 8 min 32 s, with sampled
peaks of 3.035–3.634 GB and no remaining task files. All guards stayed enabled.
These timings use different nodes/seeds from the earlier pilots and do not
isolate a filesystem-only speedup.

The gate checks output checksums, successful cleanup and sampled temporary
size below 5 GB. It repeats project storage checks before releasing bulk arrays.
Eight arrays have a 25-task limit each: at most 200 production jobs in total.
Normal QOS currently allows 1,000 running / 2,000 submitted jobs per user;
this bounded wave fits within those limits. Its original stop after one wave
is superseded by the user-authorized day controller above.

The completion controller permits one bounded same-seed recovery for recognized
finite-LHE or HepMC-integrity failures, up to 32 tasks; unknown failures stop
for investigation. Final summary validates/persists both processes. A dependent
archive job verifies every saved checksum before deleting redundant compact
/work copies. It retains persistent outputs, diagnostics and all controls.

## Storage measurements and safeguards

Inventory 1678427 measured **26.283 GB /work** and **0.732 GB /project** at
23:38–23:41 UTC. Soft limits remain 300 GB /work and 2 TB /project; hard limits
remain 400 GB and 3 TB. Each launch/gate reserves 50 GB and 10 GB respectively
below the soft limits. A retained 10k task uses about 3.55 GB of intermediate
files, so 200 simultaneous workers must not place them all on /work.

Unity documents the `localscratch` node feature:
https://docs.unity.rc.umass.edu/documentation/cluster_specs/features/
Probe 1678602 found a writable local ext4 mount on cpu045 with about 1.12 TB
free. Inspected /tmp mounts are job-private local storage; shared HPC Workspace
scratch is distinct. The worker explicitly verifies /localscratch and samples
its task footprint every ten seconds. It requires 60 GB free at startup and
stops on more than 8 GB task usage or less than 20 GB free. These checks are
not Slurm disk reservations; other users can affect free space.

## Code, validation and reporting

- Published main: `e966bd655a5a5e38d79f8bc331083dd17a5dc88b`.
- Physics tests: `codex_validation`, `100417897f52e1e369b1f66ae611d284de6de952`.
- Published campaign: `codex_bootstrap_20261113`, `2e7cb02f777e6f31e6f355c831b4e5f695e64ea4`.
- Unique report: `codex_report`, 23 reveal.js slides under `/docs`, ready for
  GitHub Pages. Live hosting has not been configured.
- Runtime snapshot: `runs/runtime_snapshots/e07de718c4d3967c8efd8f752f5ed1f282be9305`,
  tree equal to main 29a5e30. New wave controls are frozen in the run's `control/`.
  Do not replace executing scripts or installed dependencies while jobs use them.
- 428 passing component checks, seven campaign/cleanup checks, and the new
  executable neutral-only calorimeter regression. Eight detector controls are certified, with an exact same-node check for
  one cross-node rounding difference. Eight additional day-controller checks
  pass; no additional physics training has occurred.
  Real-input normalization, native −0.1/0/+0.1 SMEFT ratios and NN readers passed
  earlier pilot checks; these are not HPO or unfolding closure results.
- Main includes systematic endpoint shape/yield training through NSBI,
  provenance checks/shared hard-event partitions, zero weight decay for new
  fits, stage-specific OmniFold settings/resume validation, and finite-LHE fixes.
- All repository Markdown documentation was reviewed and fourteen cited paper
  full texts cached with checksums. Core methods/relevant sections were studied;
  long reference works were read selectively. Register: `report/docs/READING_NOTES.md`.

## Remaining research

Reach at least five million selected reconstructed events per sample. Construct
and validate the c_HB-deformed positive data-like intensity, keeping signed
MC accounting and an explicit tau policy. Deform before drawing at least 100
pseudo-experiments. Freeze the nominal background-removal pair across toys,
refit ordinary OmniFold for each statistical replica, tune reco_pull and
iteration count on development data, then demonstrate independent closure and
unbinned coefficient inference. Only then implement and validate nuisance
profiling and the 100-point auxiliary ensemble. Marginal systematic endpoints
still need conditional-response transport tests; weighted likelihoods and the
uniform auxiliary box require explicit inference/coverage interpretation.

Detailed attempt history, measured pilots and training diagnostics live in
`report/docs/PILOT_RESULTS.md` and `report/docs/TRAINING_LEDGER.md`.

2026-11-14T00:01:05.110873+00:00: All eight local pilots passed; 1,592 remaining 10k tasks submitted with a 200-job total cap (100 per physics process). Summary 1679625.

Scheduler snapshot 2026-11-14T00:03:47.829518+00:00: gg_H {'PENDING': 713, 'RUNNING': 83}; ZZ {'PENDING': 796}.

Pilot update 2026-11-14T01:43:10.468486+00:00: gg_H; 2 configurations with errors. Measured details: `runs/production_parallel_wave2/gg_H-summary.json`.
- pythia nominal: 318571 selected / 1990000 showered rows from 1990000 supplied hard events; 0 repeated source occurrences.
- pythia up: 322291 selected / 2000000 showered rows from 2000000 supplied hard events; 0 repeated source occurrences.
- pythia down: 317860 selected / 2000000 showered rows from 2000000 supplied hard events; 0 repeated source occurrences.
- herwig nominal: 291430 selected / 1990000 showered rows from 2009900 supplied hard events; 0 repeated source occurrences.

Pilot update 2026-11-14T01:51:12.025783+00:00: ZZ; 0 configurations with errors. Measured details: `runs/production_parallel_wave2/ZZ-summary.json`.
- pythia nominal: 14830 selected / 2000000 showered rows from 2000000 supplied hard events; 0 repeated source occurrences.
- pythia up: 14792 selected / 2000000 showered rows from 2000000 supplied hard events; 0 repeated source occurrences.
- pythia down: 14843 selected / 2000000 showered rows from 2000000 supplied hard events; 0 repeated source occurrences.
- herwig nominal: 14837 selected / 2000000 showered rows from 2020000 supplied hard events; 0 repeated source occurrences.

2026-11-14T12:21:06.326774+00:00: Calorimeter fix passed a neutral-only regression, two exact-input crash reproductions and eight identical-output controls. Two same-seed repairs submitted, followed by summary 1707702 and archive 1707703.

Pilot update 2026-11-14T12:40:31.890013+00:00: gg_H; 0 configurations with errors. Measured details: `runs/production_parallel_wave2/gg_H-summary.json`.
- pythia nominal: 320211 selected / 2000000 showered rows from 2000000 supplied hard events; 0 repeated source occurrences.
- pythia up: 322291 selected / 2000000 showered rows from 2000000 supplied hard events; 0 repeated source occurrences.
- pythia down: 317860 selected / 2000000 showered rows from 2000000 supplied hard events; 0 repeated source occurrences.
- herwig nominal: 292915 selected / 2000000 showered rows from 2020000 supplied hard events; 0 repeated source occurrences.
