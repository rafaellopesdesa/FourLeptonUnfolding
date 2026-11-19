# FourLeptonStudy current status

Updated: 2026-11-19T14:39:24+00:00

The accepted samples are merged, normalized and validated. The full Herwig SMEFT template at c_HB = -0.1, 0 and +0.1, Lambda = 1 TeV, has passed its checks. Production is closed.

| Configuration | ggF entries | ggF selected reco | ZZ entries | ZZ selected reco |
|---|---:|---:|---:|---:|
| Pythia nominal | 35,000,000 | 5,607,187 | 67,990,000 | 504,772 |
| Pythia muon scale +1% | 35,000,000 | 5,642,232 | 67,990,000 | 504,985 |
| Pythia muon scale −1% | 35,000,000 | 5,569,862 | 67,990,000 | 505,436 |
| Herwig nominal | 35,000,000 | 5,150,310 | 68,000,000 | 500,780 |

Reco counts use 105 < m4l < 160 GeV; Pythia scale variants are correlated.

The report now leads with current samples and full-sample physics distributions. Generator comparisons and before/after SMEFT plots are available as SVG and PDF, with yield and unit-area shape views. The detailed production history is archived.

Next: validate physical Poisson pseudo-experiments from the signed generator templates, derive nominal background removal, then optimize and validate ordinary OmniFold. No unfolding or NN training has yet been performed. SMEFT normalization is evaluated within selected reco/fiducial regions; inclusive low-mass tails are not used to normalize the target.

Report: https://rafaellopesdesa.github.io/FourLeptonUnfolding/

Plot gallery: https://rafaellopesdesa.github.io/FourLeptonUnfolding/distributions.html
