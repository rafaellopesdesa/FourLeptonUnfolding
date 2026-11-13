# FourLeptonStudy current status

Updated: 2026-11-13 16:13 UTC. Active bootstrap and validation checkpoint.
No physics sample, optimized unfolding, or coefficient result has been
produced yet.

## Confirmed setup

- **c_HB = +0.1 at Lambda = 1 TeV**, confirmed by the user; controls 0 and −0.1.
  Retain the repository's LO SM + dimension-six amplitude-squared convention;
  no additional width/branching-ratio shift.
- Seven NN coordinates only: cos theta1, cos theta2, cos theta*, Psi, Phi,
  mZ1, mZ2. Phi/Psi are raw radians. m4l is selection-only; no pT4l/y4l inputs.
- 13.6 TeV collisions and repository normalization convention 312 fb^-1.
- No NN regularization or fitted calibration. Large-sample baseline: four
  networks, four hidden layers, 1024 neurons. Tune reco_pull separately.

## Completed

1. Reviewed AGENTS.md and all repository Markdown documentation at initial
   main `48a960af91bee00d948dcf5108daa784d4131d4c`. Inspected implementation
   and studied relevant methods in linked papers. Fourteen full texts cached
   under `references/` with checksums. Long references were studied selectively;
   not every appendix or bibliography has been read. See the reading register.
2. Installed locked Analysis and NSBI/PyTorch environments; built patched
   Delphes 3.5.1 on Slurm. Fixed ROOT prerequisites for clean batch workers.
3. Implemented systematic density and yield endpoint training through the
   existing NSBI estimator, including undeformed-MC provenance checks and
   common partition assignment for repeated hard events.
4. Removed nonzero weight-decay defaults and reject penalties in new fits.
   Added independent OmniFold stage settings with persisted resume semantics
   and the requested large-sample defaults.
5. Fixed NFS training-cache cleanup. **173 tests pass**: 5 endpoint + 86
   background-removal + 82 OmniFold (Slurm 1628168), including small real NSBI
   fits. These are software checks, not physics HPO or closure.
6. Published validated code on main. Preserved regression tests on
   `codex_validation`, then removed test files from main as instructed.
7. Created reveal.js slides, reading notes, a training ledger and validation
   evidence under `report/docs/` on the unique `codex_report` branch. Static
   files are GitHub Pages-ready; live Pages hosting is not configured.

## Branches and directories

Reference: https://github.com/rafaellopesdesa/FourLeptonUnfolding

- Published main: `4a0e3078a0d5e7d418f179a9f3ea2255f66ab16f`.
- Published validation snapshot: `09d33b8d13a6ca01fba09a9f6cbe55055d052d99`;
  local validation also includes the tested ROOT prerequisite fix.
- `codebase/`: implementation worktree, `codex_training_policy`.
- `validation/`: regression worktree, `codex_validation`.
- `repository/`: runtime checkout, `codex_bootstrap_20261113`; inputs in
  `Campaigns/bootstrap/`, builds under `install/`, logs/results under `runs/`.
- `report/`: unique `codex_report` worktree; publication being finalized.
- Working status: `/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/docs/CURRENT_STATUS.md`.

## Jobs and next checkpoint

| Job | Purpose | Last observed state |
|---|---|---|
| 1626358 | Original generator build | Failed in POWHEG gg_H; dependencies installed |
| 1630665 | Declaration-patch retry | gg_H built; ZZ failed on new core flags |
| 1631091 | Pinned-core retry, 2 CPUs / 8 GB | Running; core 9c3cd9e |
| 1626360 | Analysis environment | Completed |
| 1626425 | Patched Delphes build | Completed |
| 1626671 | NSBI/PyTorch environment | Completed |
| 1628168 | 173 regression checks | Completed, exit 0 |
| 1629782 | Clean-worker ROOT/Delphes verification | Completed, exit 0 |
| 1629903 | Original pilot launcher | Cancelled after failed dependency |
| 1631123 | Pilot launcher | Pending after pinned-core build |
| 1630302 | Storage inventory | Completed: /work 13.93 GB; /project 0 bytes |

The pilot submits one 1,000-event shard for each of gg_H and ZZ with Pythia
nominal/up/down and Herwig nominal. It prepares one integration grid per
process, then reuses it across showers/scale settings. Workers run the complete
generation → simulation → analysis chain and remove validated intermediates.
Initial limits: integration 1 CPU / 4 GB / 48 h; full-chain task 1 CPU / 4 GB /
6 h. These limits are not measured consumption.

Dependent summary jobs record selected/fiducial counts, signed sums,
negative-weight fractions, Born/tau coverage and bytes per shard. Validated
compact outputs are copied with SHA256 checks to
`/project/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/samples/pilot_20261113/`.
They append results to the working status file. Active I/O remains on /work.
No automatic large-production escalation is enabled; its sizing awaits pilot
acceptance, runtime and storage measurements.

## Checks before physics interpretation

- Signed Poisson resampling has variance mu_plus + mu_minus, not physical net
  expectation mu_plus − mu_minus. Quantify this and validate a positive
  sampling intensity before data-like pseudo-experiments.
- Deform the sampling intensity before drawing Poisson toys; multiplying
  already sampled events by SMEFT weights changes variance.
- Count unsupported tau decays and undefined Born angles. Signal Born leptons
  come from certified shower records, not the undecayed LHE Higgs.
- Validate sideband signal contamination and transport into the Higgs window.
  Freeze the background-removal pair across toys, but refit OmniFold for each
  statistical toy; applying frozen unfolding models does not refit it.
- Marginal systematic endpoints are implemented. Conditional response
  transport, nuisance interpolation and profiling are not yet validated.
- Background-subtracted weighted likelihoods need explicit statistical
  validation. The requested uniform auxiliary square is a sensitivity
  ensemble and does not automatically define Gaussian coverage.

## Remaining research

Measure the pilot, produce ≥5M selected reconstructed events per required
sample and ≥100 valid pseudo-experiments, optimize ordinary OmniFold on
development ensembles, demonstrate independent O_HB closure and c_HB inference,
then implement/validate Profile OmniFold and the auxiliary ensemble. No
physics or convergence claim is made at this checkpoint.

## Storage and execution

Soft targets: /work 300 GB and /project 2 TB; hard limits 400 GB and 3 TB.
Slurm is used for substantial computation and I/O. Interactive allocation
1625339 has 2 CPUs / 8 GB and ends at 17:03 UTC; independent batch jobs keep
their allocations. The default tool sandbox fails during bwrap initialization;
approved escalated execution works.

## Automatic pilot updates
