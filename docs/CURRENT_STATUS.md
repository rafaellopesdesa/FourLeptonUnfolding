# FourLeptonStudy current status

Updated: 19 November 2026 · 22:30 UTC

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

Twenty development and 100 ensemble pseudo-experiments are available, including working nominal member 47, at +0.1, with nominal controls at 0 and −0.1. The held-out mean is 1,474.04 events for an expectation of 1,472.47 at 312 fb⁻¹. The variance is 1,400.0 events². For pseudo-data draws, a negative contribution cancels against nearby positive weights within each process, mass region, channel, fiducial category and source partition. Simulation training and validation use the original signed weights in the BCE. Five focused regression tests passed against the frozen training code. The largest tested shape changes from this procedure are 0.026 and 0.026 Run 3 statistical errors for marginal and joint projections, respectively.

## Working nominal

Member 47 supplies both the sideband fit (637 observed events) and the fixed Higgs data (726 observed events). At 312 fb⁻¹, Pythia ZZ predicts 301.36 Higgs-window events before correction and 296.86 with nominal fit 47. Herwig predicts 299.86. The 100 corrections give a median of 296.57 events and a central 68% interval of 286.10–310.85. All 100 draws enter the bands. Hyperparameter selection uses development draws reweighted with the new nominal pair. [Plots](zz-transport.html?nominal=47).

## Ordinary OmniFold results

The Run 3–sized pilot compared one and two iterations. Its yield-sensitive score favored 1 iteration with two hidden layers of 32 neurons for the reconstructed-data classifier. The Pythia prior predicts 647.55 fiducial events. Member 47 unfolds to 611.69, compared with 592.87 in the deformed Herwig reference. [Study and figures](ordinary-omnifold.html). [Network validation by iteration](distributions.html#omnifold-validation). Varying Higgs-window data in 100 unfoldings gives a median fiducial yield of 612.67 events and a 16th–84th percentile interval of 589.37–638.52. Varying sideband subtraction in 100 unfoldings gives a median fiducial yield of 612.40 events and a 16th–84th percentile interval of 603.38–621.21.
