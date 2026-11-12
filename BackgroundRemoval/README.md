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

Both classifiers use exactly seven NN inputs, in this order:

```text
reco_Phi, reco_Psi,
reco_cos_theta1, reco_cos_theta2, reco_cos_theta_star,
reco_m_Z1, reco_m_Z2
```

`reco_Phi` and `reco_Psi` are the angles themselves, in radians. Each supplies
one input, with no sine or cosine encoding; `Phi1` is excluded. Preprocessing
preserves the seven variables and then standardizes them using the fit split.
OmniFold uses the same seven inputs at reconstructed or dressed-truth level.
The other observables remain available in the analysis ntuple, so this change
does not require event generation, Delphes, or ntuple production to be rerun.

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
seven-dimensional data/MC ratio is transferable in `reco_m_ZZ`. The current
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
is the raw mean balanced-class score of the ensemble, then

```text
C_shape(x) = c_C(x) / (1 - c_C(x))  # estimates p_data(x) / p_Pythia(x)
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
duplicates using a deterministic fingerprint of the seven source branches plus
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
yield after the `ZZ` correction. For the raw mean balanced-class score
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
- four hidden layers with 1024 SiLU cells per layer and batch size 8192;
- signed BCE with logits bounded to `[-logit-clip, +logit-clip]`, NAdam, and
  learning-rate decay by optimizer update;
- weight decay `1e-4` on matrix weights only, implemented as coupled L2 in
  NAdam; biases are not penalized and there is no dropout;
- deterministic 60% fit, 15% validation, and 25% untouched
  closure splits;
- arithmetic-mean probability aggregation;
- direct odds of the mean probability for the shape ratio, without a fitted
  score calibration or empirical ratio normalization;
- robust retry of a member whose common validation loss is an outlier in
  either direction;
- validation of the 312 fb$^{-1}$ input metadata.

These background stages retain their larger architecture. OmniFold uses
smaller defaults for its three signal-to-signal ratio fits: two hidden layers
of 64 neurons and batch size 512. The features, signed objective, raw-odds
inference, and training controls are shared; equal architecture is no longer
assumed for these different tasks.

### Validation and optimization controls

Validation runs before training, after the first optimizer update, and then
at most every `--validation-interval-steps 25` updates. The interval is reduced
for a small fit class so validation does not wait for many repeated draws of
that class: it is the smaller of the requested interval and
`max(1, ceil(min(N_target_fit, N_reference_fit) / (batch_size // 2)))`, using
nonzero-weight fit row counts. The final update is also checked.
`--patience 12` counts validation checks without a loss improvement of at
least `--validation-min-delta 1e-5`;
it does not count epochs. The lowest-loss checked NN state is restored even
if its improvement is smaller than this patience threshold. The initial
untrained NN is eligible and is explicitly flagged if selected; it is not
replaced by a synthetic constant classifier.

The initial learning rate is `3e-4`. `--learning-rate-decay 0.98` and
`--learning-rate-decay-steps 100` express the decay per 100 optimizer updates,
independent of epoch boundaries. `--maximum-epochs` remains a budget control:
the update cap is `maximum_epochs * steps_per_epoch`. By default, an epoch
contains enough balanced minibatches to match the larger fit class's row
count; `--steps-per-epoch` overrides that number. Changing that budget knob
does not change the learning rate at a given optimizer update.
After `t` updates the learning rate is `eta_t = eta_0 * 0.98**(t / 100)`;
the first update uses `eta_0`.

Each balanced classifier has constant-score BCE `ln(2)`. A member whose best
unpenalized validation BCE exceeds that value by more than
`--baseline-tolerance 0.01` is retrained or rejected, in addition to the
relative ensemble outlier check. Before publication, the actual ensemble mean
score must also pass this absolute validation BCE threshold. Each member's
improvement over the constant classifier is recorded; the ensemble emits a
warning if its mean score does not improve on that baseline. This does not
force artificial discrimination or substitute a constant model.
A near-unity ratio can be the correct result. Inspect the validation and
closure diagnostics to distinguish that case from an unsuccessful fit.

New artifacts record `training_policy_version: 1`, the effective update and
validation budgets, and each member's validation history and selected
checkpoint. Reports include stopping reasons, the constant baseline, fit
counts, and parameter counts when recorded. Earlier epoch-only raw artifacts
retain their original learning-curve axis. Existing raw-odds artifacts remain
valid for inference; changing
these defaults does not alter their stored models or explicit configurations.

Training is chunked through a temporary memory-mapped cache. Use
`--cache-directory` to place it on node-local scratch; it is removed after the
run.

Both stages support signed event weights. For class `c`, define its net
selected yield and squared-weight sum as

```text
W_c = sum_i(w_i)
Q_c = sum_i(w_i^2)
```

Class balancing uses the **net** yield `W_c`, and the optimized signed BCE is

```text
L = 0.5 * [sum_signal(w_i * BCE_i) / W_signal
         + sum_background(w_i * BCE_i) / W_background].
```

Thus a negative-weight event subtracts its BCE contribution. Its sign is
never replaced by an absolute value, and the event is not discarded. The
network logit—not the event weight—is bounded by `--logit-clip`; this is
necessary because a finite-sample signed BCE would otherwise be unbounded
below at a negative-weight atom. Zero-weight events make no contribution and
may be omitted from the training cache. A finite positive net yield is still
required for each class and split because it is the normalization denominator;
that is a condition on the aggregate measure, not a veto on individual
negative events.

For a numerically efficient unbiased minibatch estimate, fit events are drawn
with probability proportional to `abs(w_i)` and receive the coefficient
`sign(w_i) * sum(abs(w)) / sum(w)` in the loss. The expectation of that
estimator is exactly `sum(w_i * BCE_i) / sum(w_i)`; the absolute value controls
only the proposal distribution and never changes the signed objective.
Feature centering and scaling similarly use class-balanced `abs(w)` moments,
because preprocessing variances must be nonnegative. The physical loss,
yields, and closure sums continue to use signed weights.

For ensemble mean probability `s(x)`, inference uses `r_shape(x) = s(x)/(1-s(x))`
and `R(x) = K * r_shape(x)`, where `K` is the selected target/reference net
yield ratio. Numerical logit bounds remain part of the inference contract.
There is no fitted logistic or temperature calibration, intercept adjustment,
or multiplier enforcing `E_reference[r_shape] = 1`. That empirical mean is
measured as a diagnostic and can differ from one. Such a discrepancy, like
poor raw-score reliability, must be addressed by training and validation; it
is not corrected after training. The physical yield factor `K` is retained.

All weighted yields and histogram contents in the artifacts and diagnostics
are signed sums. Statistical variances use `Q_c = sum(w_i^2)`, so negative
weights increase the uncertainty even when they cancel the central value;
the corresponding effective count is `W_c^2 / Q_c`. Application preserves
the original signed event weights exactly. This convention is particularly
important for the signed unit-weight pseudo-data emitted by the merger.
The expected-event yield `W_c` is not the number of independent training
examples. Row counts, unique source events, and the weight-based effective
count describe different things; bootstrap replicas can increase rows without
adding independent information. Correction keeps exact replicas in the same
partition. Assess sample size using these quantities rather than interpreting
a luminosity-scaled yield as a training count.

Each model directory contains `manifest.json`, one checksummed state file per
ensemble member (four by default), and `diagnostics.pdf`. The PDF reports
learning curves, raw-score reliability, observed ratio-normalization checks,
ensemble spread, and held-out closure tests for all seven model inputs,
including `Phi` and `Psi` in radians over `[-pi, pi]`.
Reliability is a diagnostic of the NN output, with no correction fitted to
its curve. The learned ratio and its reliability are evaluated on the
untouched 25% closure split. The validation split controls early stopping and
ensemble retries. The scalar yield ratio is intentionally computed from the
full selected sample, so absolute-yield
closure is not independent of that one scalar. The binned chi-square values
are discrepancy summaries, not p-values.

Model artifacts use format version 5 and explicitly record the seven-input
feature order and raw-ensemble-odds rule. Compatibility depends on the exact
feature contract, not only the format number: previous format-5 models using
`cos(Phi)` and `cos(Psi)` are incompatible, as are format-4 models with eleven
encoded inputs. Train a new Correction, then a new background-removal model
using that Correction, and redecorate the original merged inputs into a new
output directory. Start a fresh OmniFold campaign from these newly decorated
files. Keep old artifacts and decorated outputs separately; changing a manifest cannot
convert the trained networks or their existing weights to the new inputs.
For its standard background-removed data weight, OmniFold checks the embedded
`background_removal_metadata` object when present. It requires decoration
format 3, the exact seven-input contract with unencoded `Phi` and `Psi`, and
the current raw estimator. Metadata-free inputs and custom weight branches
remain accepted, so retain
the model/output provenance as well.

Removing post-training corrections does not establish that a network is
well calibrated. Use the raw reliability and held-out closure diagnostics to
assess each training under the validation controls above; no output correction
is fitted when a diagnostic fails.

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
