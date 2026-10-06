# Background removal

This directory implements the background-removal stage that precedes the
iterative detector correction.  It follows the classifier density-ratio
convention used by the [OmniFold paper](https://arxiv.org/abs/1911.09107) and
uses the neural-network implementation from the
[IRIS-HEP NSBI toolkit](https://github.com/iris-hep/nsbi-lhc-toolkit), pinned to
commit `fc09848fc6540fd32310faebbe9db6eea7ecd17b`.

This is **not yet the iterative OmniFold step**.  Its output estimates the
component that is simultaneously reconstructed and fiducial, still expressed
in reconstructed variables.  Resolution and migration corrections will be a
separate stage.

## Statistical definition

Only these nine decay observables enter the classifier, in this exact order:

```text
reco_Phi, reco_Phi1, reco_Psi,
reco_cos_theta1, reco_cos_theta2, reco_cos_theta_star,
reco_m_Z1, reco_m_Z2, reco_m_ZZ
```

The three periodic angles are represented internally by sine and cosine.  This
avoids an artificial discontinuity at `-pi/+pi` while adding no production
information and no information beyond the requested variables.

The classes are:

- label 1, signal `S`: `gg_H_pythia.root` with
  `reconstructed && fiducial`;
- label 0, contamination `B`: all reconstructed events in `ZZ_pythia.root`,
  plus `gg_H_pythia.root` with `reconstructed && !fiducial`.

The default physical weight is `weight_nominal_pb`.  Using `weight` would give
the same ratios because both merged files have the same luminosity, but the
cross-section weight makes the luminosity cancellation explicit.  The `ZZ`
and nonfiducial-`gg_H` pieces are combined with their physical relative yields
*before* the negative class is normalized.  They are never normalized as two
separate samples.

Let

```text
S = sum(weight_nominal_pb) for ggH reconstructed and fiducial
B = sum(weight_nominal_pb) for reconstructed ZZ
  + sum(weight_nominal_pb) for ggH reconstructed and not fiducial.
```

The BCE sees independently normalized classes, so a calibrated score `c(x)`
learns

```text
r_shape(x) = p_S(x) / p_B(x) = c(x) / (1 - c(x)).
```

The yield and physical differential ratios are

```text
r_yield = S / B
r_physical(x) = r_yield * r_shape(x).
```

The factor that removes background from a data mixture is the bounded signal
purity

```text
background_removal_weight(x) = r_physical(x) / (1 + r_physical(x)).
```

It is important not to multiply data by the unbounded physical odds.  If the
input data weight is `weight`, the signal contribution is
`weight * background_removal_weight`.

## Environment

The ML dependencies are intentionally isolated from the lightweight analysis
environment:

```bash
pixi install --locked --manifest-path BackgroundRemoval/pixi.toml
```

The environment pins the NSBI toolkit revision and explicitly installs its
PyTorch/Lightning runtime dependencies.

## Train

Run from the repository root, using the merged Pythia files made by
`Analysis/merge_analysis_outputs.py`:

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml train \
  --gg-h-root /path/to/merged/gg_H_pythia.root \
  --zz-root /path/to/merged/ZZ_pythia.root \
  --output-dir /path/to/background_model
```

Defaults follow the agreed training prescription:

- four independent ensemble members;
- four hidden layers with 1024 SiLU cells per layer;
- BCE with logits, NAdam, and gradual exponential learning-rate decay;
- no dropout, no weight decay, and no explicit regularization;
- deterministic 60% fit, 15% validation/calibration, and 25% untouched
  closure split;
- arithmetic-mean probability ensemble;
- bounded affine-logit calibration (preventing divergent slopes for separated
  validation samples) and `E_B[r_shape] = 1` normalization using only the
  validation split;
- member retry when the common validation loss is a robust outlier;
- strict validation of the 312 fb$^{-1}$ input metadata.

The script reads ROOT files in chunks and writes a temporary memory-mapped
cache, so it does not materialize multi-million-row pandas data frames.  Use
`--cache-directory` to select node-local scratch storage.  The cache is removed
when training finishes.

Ordinary BCE density-ratio estimation requires nonnegative measures.  If a
selected Pythia training event has a negative weight, training stops with an
explicit error.  The code deliberately does not take absolute values, discard
negative events, or pass signed weights to an unbounded weighted BCE.

The model directory contains `manifest.json`, four checked member state files,
and `diagnostics.pdf`.  The PDF uses the untouched closure split and contains:

- member learning curves;
- score stability and weighted reliability/calibration curves;
- the density-ratio normalization test;
- ensemble-spread diagnostics;
- `B * r_shape -> S` shape closure for every one of the nine variables;
- `(S + B) * background_removal_weight -> S` yield closure for every variable.

The PDF reports both `E_B[r_shape]` and the inverse `E_S[1/r_shape]` check,
and shows raw-versus-calibrated reliability on closure.  Its binned chi-square
numbers are discrepancy summaries, not p-values: their finite-template term
accounts for the signal events shared by the purity estimate and its target,
but they do not include learned-model uncertainty.
The 25% closure bank is untouched by network fitting and calibration.  The
scalar `S/B` factor follows the requested full-sample total-yield definition,
so its feature-independent normalization uses all selected events.

The manifest records the selections, input checksums, exact feature order,
split seed, scaler, network configuration, member seeds/retries, calibration,
component yields, and toolkit revision.

## Apply to `data.root`

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml apply \
  --input /path/to/merged/data.root \
  --model-dir /path/to/background_model \
  --output /path/to/merged/data_background_removed.root
```

Inference is chunked.  It preserves the input entry order, all original tree
branches, the original signed `weight`, and existing `TObjString` metadata.  It
writes a new file atomically and adds:

| Branch | Meaning |
|---|---|
| `background_removal_weight` | Bounded `float32` factor to multiply each data event by |

Keeping the default to one four-byte branch avoids recreating the intermediate
file-size problem that motivated this workflow.  Pass
`--write-diagnostic-branches` only when event-level debugging is needed; it
also writes `signal_score_balanced`, `signal_score_ensemble_std`,
`background_shape_ratio`, `signal_to_background_ratio`, and the `float64`
convenience branch `weight_background_removed = weight *
background_removal_weight`.  The same information is already summarized by
the model PDF and manifest.

A `background_removal_metadata` object records the model-manifest checksum,
the exact written branch list, and the application summary.  Reconstructed
rows with nonfinite inputs are rejected, and the script refuses to silently
process non-reconstructed rows.

To replace an existing output file, add `--overwrite`.  To intentionally
redecorate an already decorated tree, also add
`--replace-existing-branches`.

## Interpretation and assumptions

This signal-versus-contamination construction assumes that the observed data
are described by the simulated mixture `S + B`, including the relative
normalization and support of its `ZZ` and nonfiducial-`gg_H` components.  It is
a statistical purity weight, not a truth tag for an individual event.  It also
imports the Pythia signal model into this preprocessing step.  The included
Pythia closure tests are therefore mandatory evidence; generator-dependence
stress tests should be added when independent suitable samples are available.

The original OmniFold paper introduces classifier-based density-ratio
reweighting.  Background/noise treatments are extensions of that core method;
the exact class construction here is the analysis prescription documented
above.

## Tests

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml test
```
