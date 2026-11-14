# FourLeptonStudy current status

Updated: 2026-11-14 00:02 UTC. The first signal wave is complete and persisted. All eight
node-local pilots passed, and the next 1,592 tasks have been submitted with a
limit of 100 simultaneous gg_H jobs plus 100 ZZ jobs. Slurm determines the
actual running count. No physics NN training, HPO, unfolding or coefficient
measurement has run yet.

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
Born kinematics are valid. Full background production begins with the next wave.

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
| 1679616–1679619 | Four gg_H arrays, 199 remaining tasks each | Submitted; max 25 workers each |
| 1679620–1679623 | Four ZZ arrays, 199 remaining tasks each | Submitted; max 25 workers each |
| 1679624 | Bounded recovery controller | Waiting for all eight arrays |
| 1679625 | Validate/persist the entire wave | Waiting for recovery controller |
| 1679187 | Attach verified duplicate cleanup | Completed |
| 1679659 | Verify saved hashes and remove duplicate work shards | Waiting for successful summary |

The eight local pilots finished in 6 min 32 s to 8 min 32 s, with sampled
peaks of 3.035–3.634 GB and no remaining task files. All guards stayed enabled.
These timings use different nodes/seeds from the earlier pilots and do not
isolate a filesystem-only speedup.

The gate checks output checksums, successful cleanup and sampled temporary
size below 5 GB. It repeats project storage checks before releasing bulk arrays.
Eight arrays have a 25-task limit each: at most 200 production jobs in total.
Normal QOS currently allows 1,000 running / 2,000 submitted jobs per user;
this bounded wave fits within those limits. No further wave is automatic.

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

- Published main: `29a5e301674821421a72a759ec7288cb9a38095b`.
- Physics tests: `codex_validation`, `b57c0f1562ff87f2791e2983e0f389d19263d602`.
- Published campaign: `codex_bootstrap_20261113`, `6dfd9aad31281032c0decf41bba4408de11c606d`.
- Unique report: `codex_report`, 21 reveal.js slides under `/docs`, ready for
  GitHub Pages. Live hosting has not been configured.
- Runtime snapshot: `runs/runtime_snapshots/e07de718c4d3967c8efd8f752f5ed1f282be9305`,
  tree equal to main 29a5e30. New wave controls are frozen in the run's `control/`.
  Do not replace executing scripts or installed dependencies while jobs use them.
- 428 passing component checks plus seven passing campaign/cleanup checks.
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
