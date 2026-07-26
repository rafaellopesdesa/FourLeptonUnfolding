# Compact four-lepton analysis tree

`build_analysis_tree.py` reads one or more current Delphes outputs and writes
one compact `Analysis` tree entry for every input event. Events are never
dropped when they fail the truth-fiducial or reconstruction-level selection.

## Installation

Pixi is the recommended environment manager. It installs Python and all
analysis dependencies without root privileges, records the resolved package
versions in `pixi.lock`, and runs commands in the environment without a
separate activation step.

If Pixi is not already available, install its single user-level executable:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.bashrc
```

Then create/update the environment from the repository root:

```bash
pixi install --manifest-path Analysis/pixi.toml
```

The committed `Analysis/pixi.lock` makes every installation use the same
complete dependency solution. When dependencies are intentionally changed,
run `pixi update --manifest-path Analysis/pixi.toml` and commit both the
manifest and refreshed lockfile. The local environment is stored under
`Analysis/.pixi/` and should not be committed.

On Unity, clone the repository under `/work/pi_rclsa_umass_edu/` before running
`pixi install`; the resulting environment is then visible from the worker
nodes. The reducer reads ROOT files with uproot, so neither the ROOT module nor
PyROOT is required for this analysis step.

As a compatibility fallback, the existing pip requirements can still be used
from the repository root:

```bash
python3 -m pip install -r Analysis/requirements.txt
```

## Selection

Both truth and reconstruction levels use the supplied selection:

- electrons and muons: `pT > 5 GeV`, `|eta| < 2.5`;
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

Truth uses `DressedElectron` and `DressedMuon`. Leptons with a hadron-decay
origin are excluded and a `W`, `Z`, or hard virtual-photon ancestor is
required. A photon ancestor must have mass above 5 GeV, retaining the
continuum `gamma* -> ll` contribution while rejecting conversion electrons.
Status copies and intermediate taus are traversed, so leptons from
`Z/W -> tau -> e/mu` remain eligible. The ancestry traversal terminates at the
decaying electroweak boson for the allowed-origin test. An independent hadron
veto examines the complete ancestry until an incoming parton, matching the
Delphes implementation and rejecting bosons produced in a hadron decay.
Delphes `M1` and `M2` are followed as two individual mother indices, never as
an inclusive interval containing unrelated event-record particles.

Reconstruction uses the final `RecoElectron` and `RecoMuon` collections from
the current Simulation card. These branches are post-smearing and include the
simplified loose reconstruction/identification and isolation efficiencies.
The reducer also requires the diagnostic `RecoElectronNoIso` and
`RecoMuonNoIso` branches as a schema-version check, so stale files made with
the former pre-isolation response cannot silently enter the analysis.
The Python reducer then applies exactly the same kinematic, pairing, and event
cuts to those reconstructed objects as it applies to truth. In particular,
there is no second isolation cut in Analysis and no hidden electron-only
7 GeV threshold.

## Running

One simulation output:

```bash
pixi run --manifest-path Analysis/pixi.toml analyze \
  /path/to/job_000000_seed1001/delphes_ATLAS/delphes.root \
  -o /path/to/analysis/job_000000.root
```

A complete campaign directory is discovered recursively:

```bash
pixi run --manifest-path Analysis/pixi.toml analyze \
  /work/pi_rclsa_umass_edu/FourLeptonUnfoldingGeneration/CAMPAIGN \
  -o /work/pi_rclsa_umass_edu/FourLeptonAnalysis/CAMPAIGN.root
```

The default uproot chunk size is 50 MB and can be changed with
`--step-size "100 MB"`.

## Merging and Herwig pseudo-data

`merge_analysis_outputs.py` looks in one directory for compact Analysis files
matching:

- `ZZ_pythia_*.root`;
- `ZZ_herwig_*.root`;
- `gg_H_pythia_*.root`;
- `gg_H_herwig_*.root`.

It writes `ZZ_pythia.root`, `ZZ_herwig.root`, `gg_H_pythia.root`, and
`gg_H_herwig.root`. All events are retained and assigned new sequential
`event_id` values. Within each merged file the common weight scale is chosen
so that `sum(weight)` equals the number of entries. This removes the artificial
normalization increase from concatenating independent jobs while preserving
their weighted distributions and efficiencies.

The same command creates `data.root` from the reconstructed-and-selected
events in the two merged Herwig samples. For each process it calculates

```text
expected events = cross_section_pb * luminosity_fb * 1000
                  * sum(weight[reconstructed]) / sum(weight[all])
```

and draws an independent Poisson event count. Events are then downselected
without replacement according to their positive nominal weights, the two
components are mixed and shuffled, and every output data event receives
`weight = 1`. The command stops with a request for more Herwig events if the
300 fb^-1 draw exceeds the available reconstructed sample.
The default luminosity is 300 fb^-1 and the default random seed is 12345.
The Pythia merged files can therefore be used as simulation while `data.root`
acts as statistically independent Herwig pseudo-data.

From the repository root:

```bash
pixi run --manifest-path Analysis/pixi.toml merge \
  /work/pi_rclsa_umass_edu/rclsa/FourLeptonUnfolding/Output \
  --luminosity-fb 300 \
  --seed 12345 \
  --overwrite
```

By default outputs are written into the input directory. Use
`--output-directory DIR` to separate them. The current reducer stores the
physical POWHEG normalization in `cross_section_pb`; the Higgs value already
includes the `2.771E-04` branching-fraction correction applied by Simulation.
If a shower converter records a running cross-section estimate, the merger uses
the final positive value from each job and averages those job estimates with
their generated-event counts.
Compact files made with an older reducer can still be used by supplying both
`--zz-cross-section-pb VALUE` and `--gg-h-cross-section-pb VALUE`, although
rerunning the inexpensive Analysis reduction is preferred.

Signed NLO event weights cannot define a unit-weight probability sample. The
merger permits them in the four simulation outputs but deliberately refuses to
construct `data.root` if either Herwig component contains a negative weight.
This prevents silently biased pseudo-data; the present positive-weight POWHEG
baseline is expected to satisfy this requirement.

Run the Analysis and shared Tools unit tests in the same environment with:

```bash
pixi run --manifest-path Analysis/pixi.toml test
```

Alternatively, after `cd Analysis`, Pixi discovers `pixi.toml` automatically,
so the shorter forms `pixi install`, `pixi run analyze ...`, and
`pixi run test` are equivalent.

## Output branches

The tree contains:

- `event_id`, the original `event_number`, the nominal `weight`, and the
  physical `cross_section_pb` normalization;
- `fiducial` and `reconstructed` booleans;
- `truth_type` and `reco_type`: 0=`4mu`, 1=`2mu2e`, 2=`2e2mu`,
  3=`4e`, or -1 when that level has no candidate;
- `type`: a compatibility alias equal to `reco_type` when a reconstructed
  candidate exists, otherwise `truth_type`;
- truth and reconstructed copies of all Tools observables, prefixed by
  `truth_` and `reco_` respectively.

`2mu2e` means that the leading, closest-to-mZ pair is the muon pair; `2e2mu`
means that it is the electron pair. Separate truth and reconstructed types
preserve rare pairing migrations and prevent reconstructed pseudo-data from
using a truth-biased category. Kinematic variables are filled whenever a
pairable four-lepton candidate exists, even if it fails the full selection;
they are `NaN` only when no candidate exists or an angle is mathematically
undefined. The booleans must therefore be used as masks in unfolding.
The reducer also prints the full truth/reconstruction overlap (`both`,
`fiducial-only`, `reconstructed-only`, and `neither`) after each run. It
reports both raw event counts and sums of nominal POWHEG weights, inclusively
and per four-lepton channel:

- the correction factor `C = reco / fiducial`;
- the conditional selection efficiency `both / fiducial`;
- the nonfiducial leakage fraction `reco-only / reco`.

The `_count` columns are useful debugging ratios. The `_weight` columns are
the physical normalization diagnostics when event weights are nonuniform or
signed.

For channel rows, truth-selected counts and `Nboth` use the truth pairing,
while reconstructed and reconstructed-only counts use the reconstructed
pairing. This makes a rare `2mu2e`/`2e2mu` pairing migration visible rather
than assigning both levels a shared channel by construction. These diagnostics
are printed only; the compact ROOT schema is unchanged.
