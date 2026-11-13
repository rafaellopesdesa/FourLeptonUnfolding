# FourLeptonStudy current status

Updated: 2026-11-13 21:56 UTC. The original signal arrays have finished. Wave one has
384/400 successful shards (96%); the remaining 16 have submitted recovery jobs
(12 running, four queued at this snapshot). Final validation/persistence waits
for those jobs. All four 10k ZZ pilots remain validated and persisted. No
unfolding, HPO, or coefficient measurement has yet been performed.

## Confirmed study

- **c_HB = +0.1 at Lambda = 1 TeV**, confirmed by the user; controls 0 and −0.1.
  Repository LO SM + dimension-six amplitude-squared convention; no extra
  Higgs width/branching-ratio shift.
- Seven inputs: cos theta1, cos theta2, cos theta*, Psi, Phi, mZ1, mZ2.
  Raw-radian angles. m4l is selection-only; no y4l/pT4l NN inputs.
- 13.6 TeV; repository luminosity convention 312 fb^-1.
- No regularization or fitted calibration. Large-sample baseline: ensemble
  of four networks, four hidden layers of 1024 neurons; tune reco_pull separately.

## Completed implementation and reading

1. Reviewed AGENTS.md and all repository Markdown documentation at initial
   main `48a960af91bee00d948dcf5108daa784d4131d4c`; inspected the corresponding
   implementation. Fourteen paper full texts are cached with SHA256 checksums
   under `references/`. Core methods and relevant detector/model sections
   were studied; long references were read selectively, not every appendix.
   Detailed register: `report/docs/READING_NOTES.md`.
2. Installed locked Analysis and NSBI/PyTorch environments; built patched
   Delphes and POWHEG/Pythia/Herwig through Slurm. Pinned compatible POWHEG
   core `9c3cd9e`; recorded ROOT prerequisites for clean workers.
3. Added NSBI systematic shape/yield endpoint training, undeformed-MC
   provenance checks and shared hard-event train/test partitions.
4. Enforced zero weight decay for new fits, added independent OmniFold stage
   settings with resume validation, and fixed NFS training-cache cleanup.
5. Fixed full-chain failures: the standard POWHEG RNG footer after the LHE
   closing tag, and imports when the output validator runs from a worker
   directory. Original LHE bytes and hashes remain intact.
6. **428 software checks pass across the final component runs:** 5 endpoint,
   86 background-removal, 82 OmniFold, 132 Analysis, 11 Tools, 36 Simulation,
   23 Generation, 8 BatchSubmit, 13 Plotting, 32 SMEFT. Jobs: 1628168;
   1632409 (six component passes; native teardown failed); 1633186 (native
   suite rerun passes); 1633505 (132 Analysis checks pass). These are software
   checks, not physics HPO or unfolding closure. Latest targeted run 1644270
   passes 23 Generation + 8 BatchSubmit checks, including finite-LHE recovery.

## Measured signal pilot

Each configuration generated 1,000 events through the complete chain.
Compact outputs passed LHE/Born validation and checksum-checked publication
under `/project/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/samples/pilot_20261113/`.

| Configuration | Selected reco | Reco, 115–130 GeV | Fiducial | Walltime | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| gg_H Pythia nominal | 172 | 167 | 234 | 93 s | 790 |
| gg_H Pythia muon scale up | 172 | 167 | 234 | 94 s | 789 |
| gg_H Pythia muon scale down | 170 | 167 | 234 | 93 s | 789 |
| gg_H Herwig nominal | 140 | 134 | 201 | 96 s | 650 |

Jobs 1634269/1634272/1634273/1634274 and summary 1634275 completed successfully.
Files are approximately 0.25 MB each, including unselected events. This is
not a precise shower comparison. All selected rows have valid certified Born
kinematics. Pythia has one negative event inclusively and none selected;
Herwig has none in this pilot. Production-scale positivity is not established.

Normalization/deformation job 1634795 completed successfully:

- Existing merger gives sum(weight)=sigma*312000 and sum(weight_shape)=N.
  Raw shower weight conventions differ; compare only normalized physical yields.
- Pythia nominal/up/down retained LHE source-event identities are identical.
- Native −0.1/0/+0.1 ratios are finite/nonnegative; c_HB=0 is exactly unity.
- Tau rows are explicitly counted and retained unchanged for this diagnostic:
  1/172 selected Pythia, 1/140 selected Herwig; 529/1000 and 557/1000 inclusively.
  Unsupported rows are never labeled successful ME evaluations.
- At +0.1, selected Herwig ratios span 0.708–2.428; its weighted selected
  yield changes from 411.24 to 410.56 at 312 fb^-1 in this small template.
  These are template diagnostics, not measured data or fitted coefficients.

Evidence: `runs/pilot/gg_H-summary-1634275.json` and
`runs/pilot/diagnostics/1634795/signal-checks.json`. Normalized/decorated
compact templates are persisted in the pilot's `diagnostics/1634795/`.

## Completed 10k pilot and production sizing

All four 10,000-event signal jobs completed; summary 1635020 validated and
persisted them under /project `samples/pilot10k_20261113/`.

| Configuration | Selected reco | Reco 115–130 GeV | Fiducial | Walltime |
|---|---:|---:|---:|---:|
| Pythia nominal | 1528 | 1487 | 2138 | 11 min 44 s |
| Pythia up | 1544 | 1494 | 2138 | 10 min 49 s |
| Pythia down | 1526 | 1480 | 2138 | 11 min |
| Herwig nominal | 1477 | 1425 | 1971 | 9 min 40 s |

Peak recorded RSS is approximately 2 GiB; compact output is 1.76–1.79 MB
per shard. All selected Born rows are valid. Selected tau counts: 2/3/1/2.
Pythia has two negative rows inclusively and Herwig five, but none selected.
Production-scale positive-intensity validation is still needed.

Real training input checks also passed (1635491): systematic readers accept
all four normalized 1k configurations in signal/extended regions with and
without fiducial selection; OmniFold reads 234 fiducial Pythia MC and 201
fiducial Herwig truth rows. No network was fitted in these checks.

Measured acceptance suggests roughly 33–34 million showered events per
signal configuration for the five-million-selected target, and approximately
550–650 CPU hours / 6 GB compact output per configuration by naive scaling.
Actual validated counts determine completion.

## Active jobs and next checkpoint

| Job | Purpose | Last observed state |
|---|---|---|
| 1631379 | ZZ integration, 1 CPU / 4 GB / 48 h | Completed in 1 h 35 min |
| 1631380 | ZZ Pythia nominal, 1,000 events | Completed |
| 1631381 / 1643989 | Submit ZZ variants / validate all four 1k pilots | Completed; all compact outputs persisted |
| 1635016–1635019 | Four signal configurations, 10,000 events each | Completed |
| 1635020 | Validate/persist larger signal pilot | Completed |
| 1636246 | Guarded first signal production-wave launcher | Completed; checks passed |
| 1636684 | Pythia nominal, 100 × 10k events | Original array finished; remaining shards in recovery |
| 1636696 / 1636697 | Pythia up/down, 100 × 10k each | Original arrays finished |
| 1636700 | Herwig nominal, 100 × 10k events | Original array finished |
| 1644720 | Retry Herwig tasks 4, 10, 17 with reserve | All three completed successfully |
| 1645888 | One final recovery round for known failures | Completed; submitted 16 retries |
| 1672510 / 1672513 / 1672518 / 1672519 | Nominal / up / down / Herwig recovery | 12 tasks running, four queued |
| 1636701 | Validate/persist production wave | Waiting for all four recovery arrays |
| 1646582 | Launch four bounded 10k ZZ checks | Completed |
| 1646727–1646730 | Four 10k ZZ configurations | Completed; immutable runtime, fresh seeds |
| 1646731 / 1653805 | Validate/persist larger ZZ pilot | Filename discovery corrected; rerun passed for all four |
| 1646583 | Verify revised count summary on 10k signal pilot | Completed |

The first production wave requests exactly 100 × 10,000 showered events per
signal configuration, fresh seeds and the validated grid. It checks all
four persisted pilot hashes and storage headroom before submission. At most
four workers per array (16 total), each 1 CPU / 3 GB / 1 h; require 100 GB
/work and 10 GB /project headroom below the soft targets. Its dependent
summary validates/persists compact shards and appends actual selected totals.
No next wave is automatic. One million showered events per configuration
is only the first installment toward five million selected events. New Herwig
workers supply 1% extra hard events to cover rare shower discards. Original
queued tasks retain their old batch script, so remaining EOF failures are
handled by the completion recovery controller. Current successful shard counts, including recovered tasks,
are 98/96/98/92 across the four configurations, totaling 384/400.
Slurm completion is followed by independent summary validation and persistence.

## Production failures and recovery

Herwig can discard a supplied hard event. Reproducing seed 40000005 in
1642288 produced 9,998 accepted events from 10,000 attempts: remnant extraction
and a meson-decay retry limit caused two discards. The finite LHE then exhausted.
Buffer check 1643289 supplied 10,100 hard events and obtained 10,000 unique
showered source identities with zero repeats; the first 10,000 hard-event
blocks exactly matched the original failed input. The source is never reopened.
Main now separates requested hard/showered counts and retains native logs.
Full-chain retries 1644720_4, _10 and _17 have completed successfully.

I also replaced an executing shell script on NFS during deployment. This caused
eight separate stale-file-handle failures. Incomplete outputs were rejected;
original diagnostics are preserved for retry. Active runtime files are now
frozen. New jobs and the final recovery use an immutable snapshot of code
commit e07de718 (tree equal to published main 29a5e30) plus hashed environment
scripts. Do not edit the runtime or its installed dependencies while jobs use
it. Unknown failures stop the recovery controller for investigation.

Counts now distinguish supplied hard events, showered rows and repeated source
occurrences. The earlier 10k Herwig pilot has 9,999 unique source identities and
one repeated occurrence; this is recorded, not discarded. Shared-source split
rules remain essential. Updated summary check 1646583 passes.

## Completed background pilot

ZZ integration completed in 1 h 35 min. Summary 1643989 validated and persisted
all four 1k configurations. Pythia nominal/up/down each select 12 reconstructed
rows (5 in the Higgs window), with 19 fiducial rows; Herwig selects 5 (2 in the
window), with 12 fiducial rows. All selected Born kinematics are valid. There
are 7/7/7/8 negative rows inclusively, none selected. Selected tau counts are
4/4/4/0. This is too small to estimate the shower difference or final production
size precisely. Launcher 1646582 checks persisted checksums/storage, then runs
four 10k ZZ shards (1646727–1646730) using the immutable runtime, one CPU /
3 GB / one hour each. All four completed in approximately 10–11 minutes. Summary 1646731 missed the
scale-variation filenames; after correcting discovery, 1653805 validated and
persisted all four files. Selected reco counts are 77/77/71/81 (Higgs window:
24/25/23/28). All selected Born kinematics are valid. Each configuration now
has negative selected weight, so positive data-like intensity construction
requires explicit validation before pseudo-experiments.

## Earlier recovery checkpoint (19:44 UTC)

At 19:44 UTC, 13 signal shards remain failed: eight original NFS failures,
four further finite-LHE Herwig failures, and one Pythia provenance failure.
Nominal Pythia task 37 was rejected because the HepMC file identity changed
during validation; the root cause is unresolved. The pending completion
controller now allows one same-seed retry with all integrity checks enabled
and intermediates retained. A repeated failure remains an error.

The existing controller waits for the four original arrays, runs the bounded
recovery, and then releases final validation/persistence. At current throughput,
roughly 2–3 hours remain for this first wave including recovery and summary;
this is an estimate, not a scheduler guarantee. The five-million-selected-event
requirements need substantial additional production beyond this wave.

## Recovery now running (21:56 UTC)

The original arrays are terminal. Controller 1645888 completed at 21:51 UTC
and submitted 16 same-seed retries using the immutable runtime: two nominal,
four scale-up, two scale-down, and eight Herwig. Their recorded causes are
eight NFS failures, seven Herwig finite-input failures, and one unresolved
Pythia provenance failure. No new failure category was encountered.

Twelve retries are running and four Herwig tasks are queued. Summary 1636701
now depends on all four recovery arrays, as verified in Slurm. It will validate
all 400 compact shards, count selected reconstructed events, and copy verified
files to /project. No complete-wave selected total is available yet.

About 20–40 minutes of remaining runtime is plausible at the measured rate,
plus queue delays and any further failure investigation. This is only the
first signal wave; the five-million-selected targets require further waves.
ZZ bulk production and neural unfolding have not started.

## Branches and directories

Reference: https://github.com/rafaellopesdesa/FourLeptonUnfolding

- Published main: `29a5e301674821421a72a759ec7288cb9a38095b`.
- Published tests: `codex_validation`, `b57c0f1562ff87f2791e2983e0f389d19263d602`.
- Published campaign inputs: `codex_bootstrap_20261113`, `fc106375876ad512cea799e040e51abf84e6155f`.
- Unique report: `codex_report`, 20 reveal.js slides under `/docs`, with
  measured pilots, training ledger and failure history. GitHub Pages-ready; live hosting is not configured.
- `codebase/`: implementation; `validation/`: tests; `repository/`: runtime
  and campaign inputs; `report/`: report worktree; builds in `install/`.
- Active I/O and logs in `runs/`; final compact samples in /project.

## Checks required before interpretation

- Preserve signed MC sums and sum(w^2). Signed Poisson sampling has variance
  mu_plus+mu_minus, not physical net expectation; validate a positive selected
  intensity before data-like ensembles. Zero selected negative events in a
  small pilot does not establish positivity at production scale.
- Deform the intensity before sampling. Multiplying already sampled toys by
  SMEFT weights changes their statistical variance.
- Fix and document unsupported tau treatment before final closure.
- Validate sideband contamination/transport. Freeze the nominal background
  pair across toys; refit OmniFold for each statistical replica.
- Marginal systematic endpoints require conditional-response transport and
  nuisance interpolation validation before profiling.
- Weighted background-subtracted likelihoods need statistical validation.
  Uniform auxiliary draws define a sensitivity ensemble, not Gaussian coverage.

## Remaining research and storage

Produce ≥5M selected reconstructed events per required sample and ≥100 valid
pseudo-experiments; optimize ordinary OmniFold and demonstrate independent
O_HB closure/c_HB inference; then implement and validate Profile OmniFold and
the auxiliary ensemble. No physics convergence or coverage claim is made.

Soft targets: /work 300 GB, /project 2 TB; hard limits 400 GB and 3 TB.
Last full inventory (1635989, 16:48–16:50 UTC): /work 14.16 GB, /project
2.53 MB, before the 10k pilot added another 7.12 MB of compact outputs.
Active production intermediates will grow within the reserved headroom.
All substantial computation/I/O uses Slurm. Interactive allocation 1638161
runs until 2026-11-14 01:04 UTC; independent batch jobs retain their allocations.

## Automatic pilot updates

Earlier signal attempts failed before publication on LHE footer parsing,
then standalone validator imports. Both were fixed and regression-tested.
Successful replacement: 1634269/1634272/1634273/1634274; summary 1634275.

Pilot update 2026-11-13T16:51:12.891805+00:00: gg_H; 0 configurations with errors. Measured details: `runs/pilot10k/gg_H-summary.json`.
- pythia nominal: 1528 selected / 10000 generated in validated persisted shards.
- pythia up: 1544 selected / 10000 generated in validated persisted shards.
- pythia down: 1526 selected / 10000 generated in validated persisted shards.
- herwig nominal: 1477 selected / 10000 generated in validated persisted shards.

Pilot update 2026-11-13T17:51:24.963393+00:00: ZZ; 0 configurations with errors. Measured details: `runs/pilot/ZZ-summary.json`.
- pythia nominal: 12 selected / 1000 generated in validated persisted shards.
- pythia up: 12 selected / 1000 generated in validated persisted shards.
- pythia down: 12 selected / 1000 generated in validated persisted shards.
- herwig nominal: 5 selected / 1000 generated in validated persisted shards.

Pilot update 2026-11-13T18:20:28.754587+00:00: gg_H; 0 configurations with errors. Measured details: `runs/pilot10k/gg_H-summary.json`.
- pythia nominal: 1528 selected / 10000 showered rows from 10000 supplied hard events; 0 repeated source occurrences.
- pythia up: 1544 selected / 10000 showered rows from 10000 supplied hard events; 0 repeated source occurrences.
- pythia down: 1526 selected / 10000 showered rows from 10000 supplied hard events; 0 repeated source occurrences.
- herwig nominal: 1477 selected / 10000 showered rows from 10000 supplied hard events; 1 repeated source occurrences.

Pilot update 2026-11-13T18:34:36.495196+00:00: ZZ; 2 configurations with errors. Measured details: `runs/pilot10k/ZZ-summary.json`.
- pythia nominal: 77 selected / 10000 showered rows from 10000 supplied hard events; 0 repeated source occurrences.
- pythia up: 0 selected / 0 showered rows from 0 supplied hard events; 0 repeated source occurrences.
- pythia down: 0 selected / 0 showered rows from 0 supplied hard events; 0 repeated source occurrences.
- herwig nominal: 81 selected / 10000 showered rows from 10100 supplied hard events; 0 repeated source occurrences.

Pilot update 2026-11-13T19:38:45.519584+00:00: ZZ; 0 configurations with errors. Measured details: `runs/pilot10k/ZZ-summary.json`.
- pythia nominal: 77 selected / 10000 showered rows from 10000 supplied hard events; 0 repeated source occurrences.
- pythia up: 77 selected / 10000 showered rows from 10000 supplied hard events; 0 repeated source occurrences.
- pythia down: 71 selected / 10000 showered rows from 10000 supplied hard events; 0 repeated source occurrences.
- herwig nominal: 81 selected / 10000 showered rows from 10100 supplied hard events; 0 repeated source occurrences.
