# Sideband data-to-MC Correction

This stage trains a calibrated target-to-reference density ratio in the
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

For a calibrated balanced-class score `c_C(x)`, the artifact represents

```text
C_shape(x) = p_data(x) / p_Pythia_MC(x) = c_C(x) / (1 - c_C(x))
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

Ordinary BCE cannot represent a signed density. A selected negative data or
MC weight in the sideband therefore causes training to fail explicitly; the
code does not take absolute values or silently drop negative entries. This is
an important precondition because the merger can emit negative unit-weight
pseudo-data: Correction will hard-fail if one lies in the selected sideband.

The artifact contains one checksummed state file per ensemble member (four by
default, configurable with `--ensemble-size`, with a minimum of four),
`manifest.json`, and `diagnostics.pdf`. The manifest stores the strict bounds
under `selections.mass_window_gev`, the target/reference and component yields,
the sideband `ggH` fraction, feature order, calibration, and checksummed input
provenance. It also records the weight branch, normalization family, units,
and conversion to pb when known; custom units are recorded as arbitrary. The
PDF includes learning curves, raw and calibrated reliability,
ratio normalization, ensemble spread, and two data-versus-MC yield closures
for all eight inputs: the learned `(ZZ+ggH)*C` ratio and the requested
downstream `ZZ*C+ggH` prescription. Their difference makes the `ZZ`-dominance
approximation visible rather than hiding it. The network is fit on 60%;
calibration and shape normalization use the 15% validation split, and both are
evaluated on the untouched 25% closure split. The scalar yield factor
intentionally uses the full selected totals, so absolute-yield closure shares
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

An existing output directory is rejected unless `--overwrite` is supplied.
For safety, overwrite is limited to a recognized Correction artifact of the
same model kind.
