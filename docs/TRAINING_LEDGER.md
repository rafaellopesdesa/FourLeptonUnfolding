# Training and validation ledger

Updated 2026-11-13. No physics training or hyperparameter optimization has yet
been performed. Tiny test fixtures are not simulated Higgs results.

## Endpoint artifact integration check

- Slurm attempts: 1627364, 1627826, 1628168. Five endpoint tests passed in each.
- Input: synthetic ROOT fixtures with seven coordinates, metadata, repeated
  source identities and a negative event weight. Varied weights are twice the
  nominal weights. These fixtures exercise contracts, not realistic dynamics.
- Hyperparameters: four members, one hidden layer, four neurons, batch 16,
  one optimizer step per member, CPU; zero weight decay, no post-hoc calibration.
- HPO: none. A relaxed baseline tolerance of 0.5 is used only to make the
  artifact mechanics test deterministic; it is not a production criterion.
- Ensemble selection: all four raw members retained; selected checkpoints may
  include step zero. Near-null discrimination warnings are expected.
- Reweighting: held-out closure report generated; physics goodness of fit is
  not asserted by the fixture.
- Calibration tests: the existing raw reliability diagnostic is emitted;
  no fitted correction is applied.
- Normalization: physical yield ratio is exactly 2, full ratio is 2 × raw
  shape ratio, and saved/reloaded predictions agree. No empirical factor is
  fitted to force the raw shape mean to one.
- Provenance: mislabeled showers/scales, deformation metadata and mismatched
  luminosity are rejected. Shared hard events receive identical partitions.

## Background-removal regression training

The pre-existing tiny neural smoke tests and analytic signed-BCE checks are
run on `codex_validation`. They test correction-to-subtraction artifact
compatibility, signed training, raw odds, ensemble selection and diagnostics.
Their architectures are fixture-specific; no HPO is performed.

Attempt 1627364 encountered two NFS cleanup errors after the fits. Explicit
closure of memory maps before temporary-directory deletion fixed these;
86 background-removal tests pass in 1627826 and 1628168. The regression for
cleanup also verifies the exception path. No production model was lost.

## OmniFold regression training

The suite combines real small NSBI ratio fits with deterministic histogram
ratios that test cumulative updates, yields, checkpointing and resume behavior.
These synthetic tests supply no information about the correct physics
iteration count or architecture.

Attempt 1627826 had one failing mutation test: multiplying the newly zero
weight decay by two did not modify it. The test now changes its value and
again detects inconsistent serialized training settings.

Attempt 1628168 passes **82 OmniFold tests**, including independent stage
overrides, fixed seeds, resume preservation and invalid-configuration rejection.
Together with **86 background-removal and 5 endpoint tests: 173 tests pass**.

## Production fits: pending

For every future correction, subtraction, endpoint, OmniFold stage and
coefficient ratio, record:

1. Input checksums, event partitions, signed yields, sum(w²), selected counts
   and luminosity; code/toolkit/environment versions and Slurm allocation.
2. All candidate hyperparameters and budgets, selection criterion, failed
   candidates, retries, chosen ensemble members and independent test sets.
3. Learning curves and selected steps; ensemble versus member validation BCE.
4. Seven marginal and multivariate reweighting closure, including tails and
   correlations; use the appropriate covariance for shared events.
5. Raw-score reliability and saturation diagnostics without calibration.
6. Shape normalization expectation, actual raw ratio mean, separate yield
   ratio and propagated uncertainty; no empirical renormalization.
7. Runtime, peak memory, artifact size, publication checksum and failure state.

The nominal HPO study will optimize reco_pull on development toys, then freeze
its rule before the independent closure/statistical ensemble. The report will
include unsuccessful choices as well as the selected ones.
