# Sideband data-to-MC Correction

This stage trains a target-to-reference density ratio from raw NN outputs in the
strict upper sideband

```text
reconstructed && 130 < reco_m_ZZ < 160 GeV.
```

Label 1 is the reconstructed, stat-limited Herwig pseudo-data in nominal
`data.root`. Label 0 is the physically weighted union of reconstructed events
in `ZZ_pythia.root` and `gg_H_pythia.root`. The Pythia files are combined
before class balancing, so their physical relative normalization is
preserved.

The classifier uses the eight decay variables listed in
[`BackgroundRemoval.common`](../common.py). `reco_m_ZZ` is never a network
input; it defines the strict sideband and contributes to the exact-duplicate
fingerprint described below. The resulting correction is later used in the
strict `115 < reco_m_ZZ < 130` GeV analysis region and on all reconstructed
`ZZ` rows. This is an extrapolation in reconstructed four-lepton mass: it
assumes that the eight-dimensional ratio learned in the sideband transfers
outside it. Sideband closure alone does not test that assumption.

Train from the repository root:

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml correction-train \
  --data-root /path/to/merged/data.root \
  --gg-h-root /path/to/merged/gg_H_pythia.root \
  --zz-root /path/to/merged/ZZ_pythia.root \
  --output-dir /path/to/models/correction
```

The default training branch is `weight`. At the common 312 fb$^{-1}$, this is
the signed unit observation weight for the stat-limited pseudo-data and the
luminosity-scaled expected-event weight for Pythia. Use `--weight-branch` only
when an intentional alternative normalization is required.

The later Training stage defaults to `weight_nominal_pb`. The pair is
explicitly supported because `weight` differs from `weight_nominal_pb` only by
the common `312000 pb^-1` luminosity factor, which cancels from the ratios.
A custom Correction branch makes the measure unknown; Training then requires
its explicit `--allow-weight-measure-mismatch` acknowledgement.

For the raw ensemble mean balanced-class score `c_C(x)`, the artifact represents

```text
C_shape(x) = c_C(x) / (1 - c_C(x))  # estimates p_data(x) / p_Pythia_MC(x)
C(x)       = [sum(w_data) / sum(w_Pythia_MC)] * C_shape(x).
```

`C(x)` is a full yield-times-shape ratio, not a bounded purity. Training later
multiplies it into every selected `ZZ_pythia` event. Application evaluates it
for every reconstructed `ZZ_pythia` row, including rows outside the analysis
region. As prescribed for this analysis, `gg_H_pythia` is left at unity even
though it is included in the Correction reference mixture. The manifest
records its fraction of that mixture. Applying the mixture-derived ratio only
to `ZZ` is exact only to the extent that the sideband `ggH` fraction is
negligible.

Nominal pseudo-data are sampled with replacement and receive new sequential
`event_id` values. Correction therefore assigns their split from a
deterministic fingerprint of the eight inputs plus `reco_m_ZZ`, ensuring that
exact bootstrap replicas stay together in fit, validation, or closure.
Pythia splits use the merged `event_id`.

Signed data and MC weights are accepted in the sideband. The balanced BCE
uses each signed weight directly, normalized by its class's net signed yield.
Negative entries therefore subtract their loss contribution; they are not
made positive or silently dropped. Class yields and the factor multiplying
`C_shape` use net sums `sum(w)`, while statistical variances and histogram
uncertainties use `sum(w^2)`. Each class and deterministic split must retain a
finite positive net sum so that its normalization is defined. This supports
the negative unit-weight pseudo-data produced by the merger without changing
their signed-measure convention.

Minibatches are proposed with probabilities proportional to `abs(weight)` and
use a signed importance coefficient, so their expected loss remains exactly
the net-normalized signed BCE. Absolute weights are also used to construct a
stable feature scaler; they are not used for any physical yield or learned
density ratio. Network logits are bounded by `--logit-clip`, which makes the
signed finite-sample BCE bounded without altering event weights. The ensemble
mean probability is converted directly to odds. No logistic calibration,
intercept adjustment, or empirical ratio-normalization multiplier is fitted.
The physical target/reference yield ratio remains a separate factor.

Correction retains four hidden layers of 1024 neurons and batch size 8192.
The shared training controls now apply matrix-only NAdam weight decay,
validation by optimizer update, checkpoint selection with validation-check
patience, and an absolute `ln(2)` baseline check in addition to the relative
ensemble filter. See
[training controls](../README.md#validation-and-optimization-controls) for
defaults and CLI options. New artifacts record training policy version 1 and
per-member update/checkpoint histories; earlier raw-odds artifacts remain
usable for inference.

The artifact contains one checksummed state file per ensemble member (four by
default, configurable with `--ensemble-size`, with a minimum of four),
`manifest.json`, and `diagnostics.pdf`. The manifest stores the strict bounds
under `selections.mass_window_gev`, the target/reference and component yields,
the sideband `ggH` fraction, feature order, the raw-odds inference contract,
and checksummed input provenance. It also records the weight branch,
normalization family, units, and conversion to pb when known; custom units are recorded as arbitrary. The
PDF includes learning curves, raw-score reliability, observed
ratio-normalization checks, ensemble spread, and two data-versus-MC yield closures
for all eight inputs: the learned `(ZZ+ggH)*C` ratio and the requested
downstream `ZZ*C+ggH` prescription. Their difference makes the `ZZ`-dominance
approximation visible rather than hiding it. The network is fit on 60%;
the 15% validation split controls early stopping and ensemble retries. Raw
reliability and closure are evaluated on the untouched 25% closure split;
they do not fit a correction or force the empirical ratio mean to one. The
scalar yield factor intentionally uses the full selected totals, so absolute-yield closure shares
that scalar information.

Train this artifact only from nominal `data.root`. It is frozen when Training
weights `ZZ` and when Application decorates the `ZZ` sample. Numbered
`data_XXXX.root` files are evaluated by the resulting single frozen
background-removal model, so their results are indirectly conditional on this
one nominal-data Correction fit; they do not include uncertainty from
retraining the Correction for each toy.

Input SHA-256 checksums are recorded by default and are used by Training and
directory Application to enforce a single campaign. Use
`--skip-input-checksums` only when byte-level provenance is intentionally
unavailable; doing so disables the corresponding downstream match checks.

New artifacts use format 4. Older calibrated models are rejected. Retrain
Correction in a new model directory, then retrain background removal and
redecorate the original merged files; retain earlier outputs separately.

An existing output directory is rejected unless `--overwrite` is supplied.
For safety, overwrite is limited to a recognized Correction artifact of the
same model kind.
