# Validation and unfolding-input plots

## Merged Herwig pseudo-data versus Pythia

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

## Background-removed data versus signal MC

`plot_background_removed.py` is the reconstruction-level starting-point check
for the later OmniFold iterations. It reads the output directory produced by
`BackgroundRemoval` Application and makes one page for each of the same 12
observables and bin definitions used by `plot_data_mc.py`. Every page has a
filled Pythia signal histogram, black purity-weighted data markers, a
finite-MC statistical band, and a data-to-signal ratio.

The samples are defined to match the background-removal Training classes:

| Plotted sample | Selection | Histogram weight | Reconstructed values |
|---|---|---|---|
| Background-removed data | `analysis_region == 1` | `weight * background_removal_weight` | `reco_*` |
| Pythia signal | `analysis_region == 1 && fiducial` | `weight` | `reco_*` |

`analysis_region == 1` means reconstructed with the strict open interval
`115 < reco_m_ZZ < 130` GeV. The extra `fiducial` requirement on Pythia ggH is
intentional: reconstructed-but-nonfiducial ggH was part of the negative class
in Training, so including it here would compare the purity-weighted data to a
different target. The plotted values are nevertheless reconstruction-level
variables in both samples. No `ZZ` sample enters this report.

The eight decay variables used by the purity model are `Phi`, `Phi1`, `Psi`,
the three polar-angle cosines, `m_Z1`, and `m_Z2`. The other four shared plots
(`m_ZZ`, `y_ZZ`, `pT_ZZ`, and channel) are useful out-of-model validation
projections, but the eight-dimensional density-ratio construction does not by
itself guarantee exact background subtraction in those four observables.

Run it after the directory-level Application command has produced the
decorated directory:

```bash
pixi run --locked --manifest-path Analysis/pixi.toml plot-background-removed \
  /work/pi_rclsa_umass_edu/$USER/FourLeptonAnalysis/decorated \
  --output /work/pi_rclsa_umass_edu/$USER/FourLeptonAnalysis/plots/background_removed_vs_ggH.pdf
```

Use, for example, `--data-file data_0001.root` to inspect a numbered
pseudo-experiment. The script requires the directory-level Application
manifest and verifies its file checksums, common frozen-model provenance,
sample roles, strict signal-region definition, ggH unity correction, and
312 fb$^{-1}$ normalization. It checks the inputs again after making all
pages, so a concurrent campaign replacement is rejected instead of publishing
a mixed report. Keep the decorated directory quiescent while plotting.

The comparison uses absolute yields; it does not normalize either sample to
unit area. `--overwrite` is required to replace an existing PDF. Keeping plots
outside the decorated directory, as above, also prevents a previous PDF from
being carried into a later `apply-directory --overwrite` campaign snapshot.

The data error bars use
`sqrt(sum((weight * background_removal_weight)^2))`; the ggH band uses
`sqrt(sum(weight^2))`. These are finite-sample uncertainties conditional on
the frozen background-removal model. They do not include model, calibration,
or Correction uncertainty, and this report is not yet an unfolded result.
