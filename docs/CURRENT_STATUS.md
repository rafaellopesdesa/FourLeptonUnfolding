# FourLeptonStudy current status

Updated: 2026-11-19T15:34:36+00:00

The accepted samples are merged, normalized and validated. The full Herwig SMEFT template at c_HB = -0.1, 0 and +0.1, Lambda = 1 TeV, has passed its checks. Production is closed.

| Configuration | ggF entries | ggF selected reco | ZZ entries | ZZ selected reco |
|---|---:|---:|---:|---:|
| Pythia nominal | 35,000,000 | 5,607,187 | 67,990,000 | 504,772 |
| Pythia muon scale +1% | 35,000,000 | 5,642,232 | 67,990,000 | 504,985 |
| Pythia muon scale −1% | 35,000,000 | 5,569,862 | 67,990,000 | 505,436 |
| Herwig nominal | 35,000,000 | 5,150,310 | 68,000,000 | 500,780 |

Reco counts use 105 < m4l < 160 GeV; Pythia scale variants are correlated.

The report now leads with current samples and full-sample physics distributions. Generator comparisons and before/after SMEFT plots are available as SVG and PDF, with yield and unit-area shape views. The detailed production history is archived.

The pseudo-data ensemble is validated. Background removal and ordinary OmniFold follow. SMEFT normalization is evaluated within selected reco/fiducial regions; inclusive low-mass tails are not used to normalize the target.

Report: https://rafaellopesdesa.github.io/FourLeptonUnfolding/

Plot gallery: https://rafaellopesdesa.github.io/FourLeptonUnfolding/distributions.html

## Physical pseudo-experiments

One nominal, 20 development and 100 held-out pseudo-experiments are available at +0.1, with nominal controls at 0 and −0.1. The held-out mean is 1,474.04 events for an expectation of 1,472.47 at 312 fb⁻¹. The variance is 1,400.0 events². For pseudo-data draws, a negative contribution cancels against nearby positive weights within each process, mass region, channel, fiducial category and source partition. Simulation training and validation use the original signed weights in the BCE. Five focused regression tests passed against the frozen training code. The largest tested shape changes from this procedure are 0.026 and 0.026 Run 3 statistical errors for marginal and joint projections, respectively.
