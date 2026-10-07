# Background-removal Training

This stage learns the bounded signal purity used for data in the strict
analysis region

```text
reconstructed && 115 < reco_m_ZZ < 130 GeV.
```

`reco_m_ZZ` only defines this selection. The model uses the eight inputs
listed in the [main BackgroundRemoval documentation](../README.md#variables-and-regions).

The positive class is reconstructed-and-fiducial `gg_H_pythia`. The negative
class is reconstructed nonfiducial `gg_H_pythia` plus reconstructed
`ZZ_pythia`. Before the classes are balanced, every selected `ZZ` event is
multiplied by the full yield-times-shape factor from the sideband Correction
artifact. The `ggH` components are not correction-weighted.

Run this stage only after Correction, from the repository root:

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml train \
  --gg-h-root /path/to/merged/gg_H_pythia.root \
  --zz-root /path/to/merged/ZZ_pythia.root \
  --correction-model-dir /path/to/models/correction \
  --output-dir /path/to/models/background
```

Training defaults to `weight_nominal_pb`. Because the inputs share the same
luminosity, its yield ratio is equivalent to using the luminosity-scaled
`weight` branch. Any selected negative base weight or nonfinite/nonpositive
Correction prediction causes an explicit failure.

More precisely, the default Correction branch `weight` and Training branch
`weight_nominal_pb` differ only by the common `312000 pb^-1` factor written by
the merger. That factor cancels in both calibrated ratios, so this pair is
known-compatible and needs no override. Any custom or unknown branch pairing,
including two same-named custom branches, is accepted only with
`--allow-weight-measure-mismatch` after its normalization semantics have been
verified. The Training manifest records the override and both contracts;
unknown yield units are stored as `arbitrary`, not mislabeled as pb or events.

With the default checksum behavior, `gg_H_pythia.root` and `ZZ_pythia.root`
must be byte-identical to the Pythia inputs recorded by the Correction
artifact. This catches accidental mixing of merged campaigns before an
expensive training run. `--skip-input-checksums` bypasses that byte-level
comparison and omits hashes from the new artifact; reserve it for an
intentional workflow whose files cannot remain byte-stable.

The output contains one checksummed state file per ensemble member (four by
default, configurable with `--ensemble-size`, with a minimum of four),
`manifest.json`, and `diagnostics.pdf`. The manifest records the strict
analysis window, class and component yields, feature order, calibration, input
provenance, and the exact Correction-manifest SHA-256. The PDF uses the
untouched closure split for learned-shape and calibration tests in every model
variable. The scalar yield factor is intentionally computed from the full
selected totals, so the absolute-yield closure shares that one scalar input.

An existing output directory is rejected unless `--overwrite` is supplied.
For safety, overwrite is limited to a recognized background-removal artifact
of the same model kind.

The Correction is frozen while this model is trained. Application then uses
this one frozen background-removal model for nominal data and every numbered
pseudo-experiment; the associated Correction is applied directly only to the
`ZZ` sample. The resulting toy spread is conditional on the pair and does not
include model-refit uncertainty.

For the ratio equations, architecture defaults, and application recipe, see
the [main documentation](../README.md).
