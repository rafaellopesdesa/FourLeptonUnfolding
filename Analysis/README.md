# Compact four-lepton analysis tree

`build_analysis_tree.py` reads one or more current Delphes outputs and writes
one compact `Analysis` tree entry for every input event. Events are never
dropped when they fail the truth-fiducial or reconstruction-level selection.

## Installation

From the repository root, install the Python dependencies in your environment:

```bash
python3 -m pip install -r Analysis/requirements.txt
```

On Unity, first load/source the same ROOT and Delphes environment used for the
simulation if needed. The reducer itself reads ROOT with uproot and does not
require PyROOT.

## Selection

Both truth and reconstruction levels use the supplied selection:

- electrons and muons: `pT > 5 GeV`, `|eta| < 2.7`;
- leading SFOS pair: smallest `|mZ-mll|`;
- sub-leading SFOS pair: remaining pair with smallest `|mZ-mll|`;
- ordered leading-lepton thresholds: 20, 15, and 10 GeV;
- `50 < mZ1 < 106 GeV`, `12 < mZ2 < 115 GeV`;
- `deltaR > 0.1` between every pair of selected leptons;
- every selected-lepton SFOS mass must exceed 5 GeV;
- default extended region: `105 < m4l < 160 GeV`.

Use `--mass-region signal` for `115 < m4l < 130 GeV`. Jets within
`deltaR <= 0.1` of a selected lepton are conceptually removed by the quoted
definition; because this analysis has no jet requirement or jet output, that
cleaning cannot change event selection and no jet branches are read.

Truth uses `DressedElectron` and `DressedMuon`. Leptons with a hadron anywhere
in their ancestry are excluded, while leptons from `Z -> tau tau` remain
eligible. Reconstruction uses the loose pre-isolation `RecoElectron` and
`RecoMuon` branches; no additional isolation requirement appears in the
provided table.

## Running

One simulation output:

```bash
python3 Analysis/build_analysis_tree.py \
  /path/to/job_000000_seed1001/delphes_ATLAS/delphes.root \
  -o /path/to/analysis/job_000000.root
```

A complete campaign directory is discovered recursively:

```bash
python3 Analysis/build_analysis_tree.py \
  /work/pi_rclsa_umass_edu/FourLeptonUnfoldingGeneration/CAMPAIGN \
  -o /work/pi_rclsa_umass_edu/FourLeptonAnalysis/CAMPAIGN.root
```

The default uproot chunk size is 50 MB and can be changed with
`--step-size "100 MB"`.

## Output branches

The tree contains:

- `event_id`, the original `event_number`, and the nominal `weight`;
- `fiducial` and `reconstructed` booleans;
- `type`: 0=`4mu`, 1=`2mu2e`, 2=`2e2mu`, 3=`4e`, or -1 if no candidate;
- truth and reconstructed copies of all Tools observables, prefixed by
  `truth_` and `reco_` respectively.

`2mu2e` means that the leading, closest-to-mZ pair is the muon pair; `2e2mu`
means that it is the electron pair. `type` comes from the truth candidate when
one exists and otherwise falls back to the reconstructed candidate. Kinematic
variables are filled whenever a pairable four-lepton candidate exists, even if
it fails the full selection;
they are `NaN` only when no candidate exists or an angle is mathematically
undefined. The booleans must therefore be used as masks in unfolding.
