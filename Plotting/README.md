# Herwig pseudo-data versus Pythia validation plots

`plot_data_mc.py` reads one merged output directory and writes a multipage PDF
covering every physics variable in the compact Analysis tree:

- the three polar-angle cosines;
- `Phi`, `Phi1`, and `Psi`;
- `m_Z1`, `m_Z2`, `m_ZZ`, `y_ZZ`, and `pT_ZZ`;
- the four-lepton channel category.

Every page has reconstruction-level and fiducial-level panels with ratio
subpanels. The style is `mplhep.style.ATLAS`, without an ATLAS label. The
Pythia `ZZ` and `gg_H` components are filled stacked histograms, with a hatched
finite-MC statistical band.

## Statistically correct references

At reconstruction level, black markers come from `data.root`, the signed
Poisson-bootstrap mixture of the two Herwig samples. A bin is

```text
content = sum(weight) = N_positive - N_negative
error   = sqrt(sum(weight^2)) = sqrt(N_positive + N_negative)
```

The error is not `sqrt(abs(content))` when negative POWHEG events exist.

`data.root` contains only reconstructed-and-selected rows. Its copied truth
columns are therefore conditional on reconstruction and must not be presented
as an unbiased fiducial distribution. The fiducial black markers instead use
the luminosity-scaled, fiducial-selected `ZZ_herwig.root` and
`gg_H_herwig.root` samples. They are labeled **Herwig pseudo-truth**; their
error bars are finite-template `sqrt(sum(weight^2))` uncertainties rather than
a second detector-data Poisson draw.

## Run

After the merger has created the five standard ROOT files:

```bash
pixi run --manifest-path Analysis/pixi.toml plot \
  /work/pi_rclsa_umass_edu/$USER/FourLeptonAnalysis/merged \
  --output /work/pi_rclsa_umass_edu/$USER/FourLeptonAnalysis/merged/four_lepton_comparison.pdf
```

The default is 312 fb$^{-1}$ and the script validates the `lumi=312000`
branch in every input. Use `--data-file data_0001.root` to plot another
pseudo-experiment. `--overwrite` is required to replace an existing PDF.

The plotter uses the merged `weight` branch directly. For MC this is
`weight_nominal_pb*lumi`; for pseudo-data it is the signed unit observation
weight. Underflow and overflow are folded into the first and last displayed
bins, respectively. Bookkeeping IDs, constant normalization branches, and
selection booleans are deliberately not plotted.
