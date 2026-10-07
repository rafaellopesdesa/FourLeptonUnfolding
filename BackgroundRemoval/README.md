# Background removal

This directory implements the two classifier stages that precede the
iterative detector correction:

1. **Correction** learns a data-to-Pythia density ratio in the upper-mass
   sideband.
2. **Training** uses that ratio on the Pythia `ZZ` background and learns the
   signal purity in the analysis region.
3. **Application** decorates nominal data, all pseudo-experiments, and the two
   Pythia samples with one consistent pair of frozen models.

The classifiers follow the density-ratio convention used in the
[OmniFold paper](https://arxiv.org/abs/1911.09107) and use the neural-network
implementation from the
[IRIS-HEP NSBI toolkit](https://github.com/iris-hep/nsbi-lhc-toolkit), pinned to
commit `fc09848fc6540fd32310faebbe9db6eea7ecd17b`.

This is **not yet the iterative OmniFold resolution correction**. The output
is an event-level preprocessing factor in reconstructed phase space.

## Variables and regions

Both classifiers use the same eight decay observables, in this exact order:

```text
reco_Phi, reco_Phi1, reco_Psi,
reco_cos_theta1, reco_cos_theta2, reco_cos_theta_star,
reco_m_Z1, reco_m_Z2
```

The three periodic angles are represented internally by sine and cosine.
`reco_m_ZZ` is **not** a neural-network input. It defines the following
strict, open intervals and, in Correction, contributes to the exact-duplicate
fingerprint used to keep bootstrap replicas in the same data split:

| Purpose | Selection |
|---|---|
| Correction sideband | `reconstructed && 130 < reco_m_ZZ < 160` GeV |
| Analysis region | `reconstructed && 115 < reco_m_ZZ < 130` GeV |

Events exactly on a boundary are outside the corresponding region.

The Correction is therefore transported in reconstructed four-lepton mass:
it is learned in the upper sideband but used in the analysis region, and
Application also evaluates it on reconstructed `ZZ` events outside both
windows. Because `reco_m_ZZ` is not an input, this assumes that the learned
eight-dimensional data/MC ratio is transferable in `reco_m_ZZ`. The current
sideband closure report validates the learned domain; it does not by itself
validate that mass extrapolation.

## 1. Data-to-MC Correction

Correction is trained only in the strict `130 < reco_m_ZZ < 160` GeV
sideband. Its classes are:

- target, label 1: reconstructed events from the nominal stat-limited Herwig
  `data.root`;
- reference, label 0: the physically weighted union of reconstructed events
  from `ZZ_pythia.root` and `gg_H_pythia.root`.

The two Pythia components retain their physical relative normalization before
the target and reference classes are balanced for BCE training. If `c_C(x)`
is the calibrated balanced-class score, then

```text
C_shape(x) = p_data(x) / p_Pythia(x) = c_C(x) / (1 - c_C(x))
k_C         = sum(w_data) / [sum(w_ZZ) + sum(w_ggH)]
C(x)        = k_C * C_shape(x)
```

`C(x)` is the full, unbounded yield-times-shape correction. Although its
reference class is the combined `ZZ + ggH` Pythia mixture, the downstream
analysis deliberately applies it to `ZZ_pythia` only. `gg_H_pythia` remains
uncorrected. Interpreting this mixture-derived factor as a `ZZ`-only
correction is exact only when the `ggH` contamination of the sideband is
negligible. The Correction manifest records the sideband `ggH` fraction so
this modeling assumption can be checked explicitly.

Only nominal `data.root` is used to derive `C(x)`. Never retrain it on a
numbered `data_XXXX.root` pseudo-experiment.

The merger samples pseudo-data with replacement and assigns new sequential
`event_id` values. To keep bootstrap copies of the same nominal-data event
from leaking across fit, validation, and closure, Correction groups exact
duplicates using a deterministic fingerprint of the eight model inputs plus
`reco_m_ZZ`. Pythia events continue to split by their merged `event_id`.

Train the Correction artifact from the repository root:

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml correction-train \
  --data-root /path/to/merged/data.root \
  --gg-h-root /path/to/merged/gg_H_pythia.root \
  --zz-root /path/to/merged/ZZ_pythia.root \
  --output-dir /path/to/models/correction
```

Correction defaults to the luminosity-scaled `weight` branch: pseudo-data
then have unit-magnitude observation weights, while Pythia has its expected
312 fb$^{-1}$ yield. A different branch can be selected with
`--weight-branch`.

Training intentionally defaults to `weight_nominal_pb`. This is a
known-compatible pairing: in the merged files,
`weight = weight_nominal_pb * 312000 pb^-1`, so the two measures differ only
by one common constant that cancels from each density and yield ratio. If
either stage uses a custom or unknown `--weight-branch`, Training requires
`--allow-weight-measure-mismatch` after its semantics have been checked. The
override and both branch contracts are stored in the artifact; unknown units
are recorded as arbitrary rather than being labeled as pb or expected events.

See [`Correction/README.md`](Correction/README.md) for the artifact and
diagnostic details.

## 2. Background-removal Training

Training is restricted to the strict `115 < reco_m_ZZ < 130` GeV analysis
region. Its classes are:

- signal, label 1: `gg_H_pythia.root` events satisfying
  `reconstructed && fiducial`;
- background, label 0: `gg_H_pythia.root` events satisfying
  `reconstructed && !fiducial`, together with reconstructed
  `ZZ_pythia.root` events.

Every selected `ZZ_pythia` event is first weighted by the full Correction
factor `C(x)`. The nonfiducial `ggH` component is not correction-weighted.
Their relative yields are preserved before BCE class balancing.

Let `S` be the weighted signal yield and `B` the combined weighted background
yield after the `ZZ` correction. For the calibrated balanced-class score
`c_T(x)`, Training constructs

```text
r_shape(x) = p_S(x) / p_B(x) = c_T(x) / (1 - c_T(x))
k_T        = S / B
rho(x)     = k_T * r_shape(x)
P_S(x)     = rho(x) / (1 + rho(x))
```

The bounded signal purity `P_S(x)`, not the unbounded odds `rho(x)`, is the
`background_removal_weight` applied to data in the analysis region.

From the repository root, train with the exact Correction artifact from step 1:

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml train \
  --gg-h-root /path/to/merged/gg_H_pythia.root \
  --zz-root /path/to/merged/ZZ_pythia.root \
  --correction-model-dir /path/to/models/correction \
  --output-dir /path/to/models/background
```

The Training manifest stores the SHA-256 checksum of the Correction manifest.
Application refuses a mismatched artifact pair. By default, Training also
checks that its two Pythia ROOT files are byte-identical to the Pythia inputs
recorded by Correction. Keep the merged files unchanged between the two
commands. `--skip-input-checksums` disables this byte-level link for an
intentional workflow that cannot retain stable files, but weakens provenance
and should not be used for routine production.

See [`Training/README.md`](Training/README.md) for training and validation
details.

## Training configuration and signed weights

Both stages use the same defaults:

- four independent ensemble members by default (configurable with
  `--ensemble-size`, with a minimum of four);
- four hidden layers with 1024 SiLU cells per layer;
- BCE with logits, NAdam, and exponential learning-rate decay;
- no dropout, weight decay, or other explicit regularization;
- deterministic 60% fit, 15% validation/calibration, and 25% untouched
  closure splits;
- arithmetic-mean probability aggregation;
- affine-logit calibration followed by validation-split normalization of the
  shape ratio, `E_reference[r_shape] = 1`;
- robust retry of a member whose common validation loss is an outlier;
- validation of the 312 fb$^{-1}$ input metadata.

Training is chunked through a temporary memory-mapped cache. Use
`--cache-directory` to place it on node-local scratch; it is removed after the
run.

Ordinary BCE density-ratio estimation requires a nonnegative measure. Either
training stage therefore **stops if any selected event has a negative
training weight**. It never takes absolute values, drops negative events, or
passes signed weights to BCE. Zero-weight events make no contribution. This
training restriction does not alter application: signed event weights in an
input ROOT file are preserved exactly. In particular, the merger can produce
negative unit-weight pseudo-data, so Correction training has an explicit
positive-weight-sideband precondition and will hard-fail if such an event is
selected.

Each model directory contains `manifest.json`, one checksummed state file per
ensemble member (four by default), and `diagnostics.pdf`. The PDF reports
learning curves, raw and calibrated reliability, ratio normalization,
ensemble spread, and held-out
closure tests for all eight model variables. The learned shape and calibration
are evaluated on the untouched 25% closure split. The scalar yield ratio is,
intentionally, computed from the full selected sample, so absolute-yield
closure is not independent of that one scalar. The binned chi-square values
are discrepancy summaries, not p-values.

For Correction, the report shows both `(ZZ+ggH)*C` (the density ratio that was
learned) and `ZZ*C+ggH` (the prescription that is actually applied). This
directly quantifies the documented `ZZ`-dominance approximation in every input
variable.

## 3. Application

For every sample, `analysis_region` is one exactly when
`reconstructed && 115 < reco_m_ZZ < 130` GeV and zero otherwise. The complete
application truth table is:

| Sample | Event condition | `analysis_region` | `background_removal_weight` |
|---|---|---:|---:|
| `data.root`, `data_XXXX.root` | reconstructed and `115 < reco_m_ZZ < 130` | 1 | Training purity `P_S(x)` |
| `data.root`, `data_XXXX.root` | otherwise | 0 | 1 |
| `gg_H_pythia.root` | reconstructed and `115 < reco_m_ZZ < 130` | 1 | 1 |
| `gg_H_pythia.root` | otherwise | 0 | 1 |
| `ZZ_pythia.root` | reconstructed and `115 < reco_m_ZZ < 130` | 1 | Correction factor `C(x)` |
| `ZZ_pythia.root` | reconstructed and outside that window | 0 | Correction factor `C(x)` |
| `ZZ_pythia.root` | not reconstructed | 0 | 1 |

Thus the Correction factor is evaluated for every reconstructed `ZZ` row,
not only rows in either mass window. Non-reconstructed rows may contain
nonfinite reconstructed observables because no model is evaluated for them.

From the repository root, the recommended directory command loads one
verified, frozen artifact pair and uses the role-specific rule consistently:
the background model for nominal and numbered data, the Correction for `ZZ`,
and unity for `ggH`:

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml apply-directory \
  --merged-dir /path/to/merged \
  --output-dir /path/to/decorated \
  --background-model-dir /path/to/models/background \
  --correction-model-dir /path/to/models/correction
```

When the merged directory contains the format-v2
`pseudo_data_manifest.json` produced by the merger, that manifest is
authoritative: Application validates its contiguous ensemble indices and
exact file names and processes only the listed pseudo-data files. It copies
the manifest unchanged into the decorated directory and records its checksum.
If the manifest is absent for a legacy campaign, Application falls back to
numeric discovery of `data_0001.root`, `data_0002.root`, and so on; the index
is zero-padded to at least four digits, so names such as `data_10000.root` are
also supported. Herwig component files and similarly named lookalikes are
ignored.

Run Application only after the merger has finished and while the merged
directory is quiescent. The command snapshots and rechecks source provenance,
but it is not a synchronization mechanism for files being rewritten
concurrently.

By default the directory command hashes nominal `data.root` and both Pythia
files and requires them to match the inputs recorded in the two training
artifacts. For a deliberate application to a different compatible campaign,
pass `--allow-input-mismatch`; every mismatch remains explicit in
`background_removal_application_manifest.json`. This override does not relax
the artifact-pair, feature, mass-window, or luminosity checks.

The input and output directories must differ and the output must not overlap
the input or model directories. The command builds the complete campaign in a
staging directory before publication, so an inference or file-processing
failure leaves an existing managed output intact; publication also attempts
to restore the previous directory if its final rename fails. The summary
`background_removal_application_manifest.json` records both model checksums,
the upstream pseudo-data manifest provenance, inference and branch-writing
options, checksum checks, and per-file source/output provenance.

All nominal and numbered data files use the same frozen background-removal
ensemble, which was itself trained once using the frozen Correction.
Consequently, comparisons among the decorated pseudo-experiments measure
statistical fluctuations conditional on that model pair. They do not
propagate uncertainty from refitting Correction or Training, and the
pseudo-experiments remain conditional on their shared finite Herwig template.

From the repository root, decorate one file by specifying its sample kind
explicitly:

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml apply \
  --input /path/to/merged/data.root \
  --output /path/to/decorated/data.root \
  --sample-kind data \
  --background-model-dir /path/to/models/background \
  --correction-model-dir /path/to/models/correction
```

The other accepted sample kinds are `gg-h` and `zz`. See
[`Application/README.md`](Application/README.md) for branch definitions and
overwrite options.

The data factor is a statistical purity weight, not a truth tag for an
individual event. Its interpretation assumes that the corrected simulated
signal-plus-background mixture describes the data in the analysis region.

## Environment and tests

From the repository root, install the isolated, locked environment once:

```bash
pixi install --locked --manifest-path BackgroundRemoval/pixi.toml
```

From the repository root, run the regression tests with:

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml test
```
