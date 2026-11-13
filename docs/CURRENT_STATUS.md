# FourLeptonStudy current status

Updated: 2026-11-13 16:55 UTC. Signal 1k/10k pilots validated; first
signal production wave running and ZZ integration active. No unfolding, HPO, or coefficient
measurement has yet been performed.

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
6. **422 software checks pass across the final component runs:** 5 endpoint,
   86 background-removal, 82 OmniFold, 132 Analysis, 11 Tools, 36 Simulation,
   18 Generation, 7 BatchSubmit, 13 Plotting, 32 SMEFT. Jobs: 1628168;
   1632409 (six component passes; native teardown failed); 1633186 (native
   suite rerun passes); 1633505 (132 Analysis checks pass). These are software
   checks, not physics HPO or unfolding closure.

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

Measured acceptance suggests roughly 33–34 million generated events per
signal configuration for the five-million-selected target, and approximately
550–650 CPU hours / 6 GB compact output per configuration by naive scaling.
Actual validated counts determine completion.

## Active jobs and next checkpoint

| Job | Purpose | Last observed state |
|---|---|---|
| 1631379 | ZZ integration, 1 CPU / 4 GB / 48 h | Running; active CPU use verified |
| 1631380 | ZZ Pythia nominal, 1,000 events | Waiting for grid |
| 1631381 | Submit ZZ scale variants and Herwig | Waiting for grid |
| 1635016–1635019 | Four signal configurations, 10,000 events each | Completed |
| 1635020 | Validate/persist larger signal pilot | Completed |
| 1636246 | Guarded first signal production-wave launcher | Completed; checks passed |
| 1636684 | Pythia nominal, 100 × 10k events | Running; at most four workers |
| 1636696 / 1636697 | Pythia up/down, 100 × 10k each | Running; four workers per array |
| 1636700 | Herwig nominal, 100 × 10k events | Running; at most four workers |
| 1636701 | Validate/persist production wave | Waiting for all four arrays |

The first production wave has exactly 100 × 10,000 generated events per
signal configuration, fresh seeds and the validated grid. It checks all
four persisted pilot hashes and storage headroom before submission. At most
four workers per array (16 total), each 1 CPU / 3 GB / 1 h; require 100 GB
/work and 10 GB /project headroom below the soft targets. Its dependent
summary validates/persists compact shards and appends actual selected totals.
No next wave is automatic. One million generated events per configuration
is only the first installment toward five million selected events.
ZZ integration and its dependent four-configuration pilot remain independent.

## Branches and directories

Reference: https://github.com/rafaellopesdesa/FourLeptonUnfolding

- Published main: `60adda8451b5bf07e46b7c4cf3f29e7756c07671`.
- Published tests: `codex_validation`, `a3007e9b143a261a45259edf0b9e1b2f29f5044d`.
- Published campaign inputs: `codex_bootstrap_20261113`, `88f13164d8a5be8c6b9a0b42dbaf9c7a1ea21ca1`.
- Unique report: `codex_report`, 18 reveal.js slides under `/docs`, with
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
All substantial computation/I/O uses Slurm. Interactive allocation 1625339
ends at 17:03 UTC; independent batch jobs retain their allocations.

## Automatic pilot updates

Earlier signal attempts failed before publication on LHE footer parsing,
then standalone validator imports. Both were fixed and regression-tested.
Successful replacement: 1634269/1634272/1634273/1634274; summary 1634275.

Pilot update 2026-11-13T16:51:12.891805+00:00: gg_H; 0 configurations with errors. Measured details: `runs/pilot10k/gg_H-summary.json`.
- pythia nominal: 1528 selected / 10000 generated in validated persisted shards.
- pythia up: 1544 selected / 10000 generated in validated persisted shards.
- pythia down: 1526 selected / 10000 generated in validated persisted shards.
- herwig nominal: 1477 selected / 10000 generated in validated persisted shards.
