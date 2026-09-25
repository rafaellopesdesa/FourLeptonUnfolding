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
pixi install --locked --manifest-path Analysis/pixi.toml
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
`event_id` values. The default luminosity is 312 fb$^{-1}$. For input generator
weights $g_i$, aggregate sum $S=\sum_i g_i$, process cross section $\sigma$ in
pb, and $L=312000$ pb$^{-1}$, the merged branches are

```text
weight_shape      = g_i * N / S
weight_nominal_pb = g_i * cross_section_pb / S
lumi              = 1000 * luminosity_fb
weight            = weight_nominal_pb * lumi
```

Thus `sum(weight_shape)=N`, `sum(weight_nominal_pb)=cross_section_pb`, and
`sum(weight)=cross_section_pb*lumi`. Signs and differential distributions are
preserved. `luminosity_fb` is also stored explicitly as 312.0. The
`cross_section_pb` branch is the sample-level process cross section repeated
on each row; multiplying that branch by luminosity separately for every event
would overcount by the sample size. This assumes compatible POWHEG weight
conventions across jobs; the merger cannot repair inconsistently normalized
inputs.

The same command creates an ensemble of pseudo-data files from the
reconstructed-and-selected events in the two merged Herwig samples. Signed
POWHEG weights are handled without discarding negative events or changing
their sign. For each process, define

```text
S  = sum(weight[all])
W+ = sum(weight[reconstructed and weight > 0])
W- = sum(abs(weight[reconstructed and weight < 0]))
a  = 1000 * luminosity_fb * cross_section_pb / S
```

The tool independently draws `N+ ~ Poisson(a*W+)` and
`N- ~ Poisson(a*W-)`, then samples the corresponding sign pool with
replacement and probabilities proportional to `abs(weight)`. Selected events
receive unit-magnitude weights `+1` or `-1`. This is the exact signed Poisson
bootstrap: its expected signed yield is `a*(W+ - W-)`, and it reduces to
ordinary positive unit-weight pseudo-data when the input has no reconstructed
negative weights.

The first file is named `data.root`; additional files are
`data_0001.root`, `data_0002.root`, and so on. Independent random streams are
derived reproducibly from `--seed`, conditional on the shared Herwig template.
Each ROOT file contains detailed
positive, negative, net, and effective-statistics metadata. The campaign
summary is written to `pseudo_data_manifest.json`.

By default, the number of files is selected automatically from a conservative
finite-Monte-Carlo information budget. With

```text
Q+ = sum(weight^2[reconstructed and weight > 0])
Q- = sum(weight^2[reconstructed and weight < 0])
```

the tool evaluates the positive, negative, and net effective sample sizes and
equivalent luminosities:

```text
N_eff,+   = W+^2 / Q+
N_eff,-   = W-^2 / Q-
N_eff,net = (W+ - W-)^2 / (Q+ + Q-)

L_eff,+   = S*W+       / (1000*cross_section_pb*Q+)
L_eff,-   = S*W-       / (1000*cross_section_pb*Q-)
L_eff,net = S*(W+-W-)  / (1000*cross_section_pb*(Q++Q-))
```

An absent sign component is non-limiting. The limiting luminosity is the
smallest applicable value over the positive, negative, and net components of
both `ZZ` and `gg_H`; the automatic ensemble count is
`floor(L_eff,limiting / luminosity_fb)`.

From the repository root:

```bash
pixi run --manifest-path Analysis/pixi.toml merge \
  /work/pi_rclsa_umass_edu/$USER/FourLeptonAnalysis/shards \
  --output-directory /work/pi_rclsa_umass_edu/$USER/FourLeptonAnalysis/merged \
  --luminosity-fb 312 \
  --seed 12345 \
  --pseudo-data-ensembles auto \
  --overwrite
```

The full-chain batch worker writes all four filename patterns directly into
the shared flat shard directory, so no manual renaming or movement is needed.
By default merged outputs are written into the input directory; use
`--output-directory DIR` as above to separate them. The current reducer stores the
physical POWHEG normalization in `cross_section_pb`; the Higgs value already
includes the `2.771E-04` branching-fraction correction applied by Simulation.
If a shower converter records a running cross-section estimate, the merger uses
the final positive value from each job and averages those job estimates with
their generated-event counts.
Compact files made with an older reducer can still be used by supplying both
`--zz-cross-section-pb VALUE` and `--gg-h-cross-section-pb VALUE`, although
rerunning the inexpensive Analysis reduction is preferred.

To request a fixed number of files, use `--pseudo-data-ensembles N`. A request
above the recommendation is refused unless
`--allow-ensemble-oversubscription` is also given. The recommendation is a
quality criterion, not a mathematical maximum: sampling with replacement can
make arbitrarily many random toys conditional on the same empirical Herwig
template, but those toys share its finite-Monte-Carlo modeling uncertainty.

Important: when negative events are present, these files are signed
pseudo-observations. Their signed bin counts follow a difference of Poisson
variables, not an ordinary Poisson distribution, and they are not literal
all-positive detector data. A classifier or OmniFold loss must explicitly
support signed sample weights. Dropping negative events, taking their absolute
weights, or resetting every output weight to `+1` would bias differential
shapes. A genuinely positive fake-data sample requires a separate local
positive-resampling model; see the
[Positive Resampler](https://arxiv.org/abs/2005.09375) and
[unbiased cell-resampling](https://arxiv.org/abs/2109.07851) approaches.

Run the complete Analysis, Tools, Simulation, Generation, batch-worker, and
plotting regression suite in the same environment with:

```bash
pixi run --manifest-path Analysis/pixi.toml test
```

Alternatively, after `cd Analysis`, Pixi discovers `pixi.toml` automatically,
so the shorter forms `pixi install --locked`, `pixi run analyze ...`, and
`pixi run test` are equivalent.

## Output branches

An unmerged compact tree produced by `build_analysis_tree.py` contains:

- `event_id`, the original `event_number`, the nominal `weight`, and the
  physical `cross_section_pb` normalization;
- `fiducial` and `reconstructed` booleans;
- `truth_type` and `reco_type`: 0=`4mu`, 1=`2mu2e`, 2=`2e2mu`,
  3=`4e`, or -1 when that level has no candidate;
- `type`: a compatibility alias equal to `reco_type` when a reconstructed
  candidate exists, otherwise `truth_type`;
- truth and reconstructed copies of all Tools observables, prefixed by
  `truth_` and `reco_` respectively.

Merged MC trees add/replace the normalization branches described above:
`weight_shape`, `weight_nominal_pb`, `lumi`, `luminosity_fb`, and the
luminosity-scaled `weight`. In a pseudo-data tree, `weight` is instead the
signed unit observation weight (`+1` or `-1`), `lumi` remains the selected
luminosity in pb$^{-1}$, and `weight_nominal_pb=weight/lumi`, so the same
row-wise formula remains true. Mixed-process pseudo-data has
`cross_section_pb=NaN` by design.

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
