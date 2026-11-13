# Systematic endpoint ratios

This module fits the missing experimental and theory endpoints using the
existing NSBI-backed `omnifold.ratios.RatioEstimator`. There is no additional
network implementation, regularization, score calibration, or empirical ratio
renormalization. The default ensemble has four members, each with four hidden
layers of 1024 neurons. Training artifacts retain validation histories,
ensemble selection, raw reliability, reweighting closure and normalization
diagnostics. A default architecture is a starting configuration, not an HPO
result.

For nominal weights `w0` and varied weights `wv`, the artifact defines

```text
shape_ratio(x) = p_varied(x) / p_nominal(x)
yield_ratio    = sum(wv[selected]) / sum(w0[selected])
full_ratio(x)  = yield_ratio * shape_ratio(x)
```

All sums retain signed weights. Variances use `sum(w^2)`. Both inputs must
carry physical expected-event weights at the same luminosity and the same
process content. Train separate signal and background ratios; a combined
input is accepted only if it is a merged MC sample with full source metadata,
not a pseudo-data file. Mixture weights must retain physical relative yields.

| Endpoint | Reference | Target | NP anchor |
|---|---|---|---:|
| `theory` | POWHEG+Pythia nominal | Undeformed POWHEG+HERWIG nominal | +1 |
| `muon_up` | POWHEG+Pythia nominal | Pythia muon momentum multiplied by 1.01 | +1 |
| `muon_down` | POWHEG+Pythia nominal | Pythia muon momentum multiplied by 0.99 | −1 |

The theory endpoint supplies one alternate generator. A negative theory
anchor and interpolation/extrapolation convention are not implied by this
training and must be specified and validated before profiling.

The seven inputs are `Phi`, `Psi`, `cos_theta1`, `cos_theta2`,
`cos_theta_star`, `m_Z1`, `m_Z2`, all at reconstruction level. `m_ZZ` only
selects the open mass window. `--region signal` (default), `sideband`, and
`extended` select (115,130), (130,160), and (105,160) GeV respectively.
`--fiducial-signal` additionally requires truth fiducial selection and is
allowed only for a pure gg_H input.

Run on a Slurm compute allocation, using the installed BackgroundRemoval
environment, with inputs staged on active research storage:

```bash
python -m Systematics.train \
  --nominal /path/to/nominal/gg_H_pythia.root \
  --varied /path/to/nominal/gg_H_herwig.root \
  --variation theory --fiducial-signal \
  --output-dir /path/to/staged-models/theory_ggH --device cuda

python -m Systematics.train \
  --nominal /path/to/nominal/gg_H_pythia.root \
  --varied /path/to/muon_up/gg_H_pythia.root \
  --variation muon_up --fiducial-signal \
  --output-dir /path/to/staged-models/muon_up_ggH --device cuda
```

Repeat for the down variation and ZZ without `--fiducial-signal`. The source
records certify process and shower; simulation metadata certifies the scale
choice. SMEFT-decorated, background-corrected, legacy metadata-free and
pseudo-data inputs are rejected. A checksum and LHE event ordinal determine
the split, so repeated or shared hard events remain together across nominal
and varied files regardless of reordered merged `event_id` values. Partial
overlap between samples can correlate closure histograms; the generic PDF
errors omit that covariance and must not be interpreted as calibrated tests.

Evaluate an endpoint through the same inference API as OmniFold:

```python
from omnifold.ratios import RatioEstimator
model = RatioEstimator.load('/path/to/staged-models/theory_ggH', device='cpu')
ratios = model.predict(reco_features)  # columns in the seven-input order
shape, full, yield_ratio = ratios['shape_ratio'], ratios['full_ratio'], model.yield_ratio
```

A new output directory is required. Models are committed before PDF creation;
if plotting fails, retain the model and regenerate plots with
`omnifold.diagnostics.plot_ratio_diagnostics`. Copy completed models and reports
to project storage after staged I/O finishes.

These are reconstructed marginal ratios for the supplied nominal truth
mixture. They do not by themselves implement the conditional response ratio
`p(reco|truth,theta)/p(reco|truth,0)` required by Profile OmniFold. Their transport
under evolving truth weights must be validated, or conditional response ratios
must be learned. The likelihood, NP interpolation, profiling, and convergence
study remain subsequent work after ordinary unfolding closure.

References: [ATLAS NSBI systematic factorization, Section 5](https://arxiv.org/html/2412.01600v2#S5)
and [Profile OmniFold, Section 3](https://arxiv.org/html/2512.07074v1#S3).
