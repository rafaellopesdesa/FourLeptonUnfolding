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

### Joining the original LHE event

New generation/simulation runs carry the original **1-based LHE source-event
index** through HepMC and into `Delphes.Event.Number`. The reducer joins on
that identifier, using the source and output checksums recorded in the
simulation-bound `lhe-provenance.json` beside the Delphes file. It never
assumes that the nth accepted shower event came from the nth LHE event.
Rejected shower attempts can therefore leave gaps without shifting later
associations.

The default reducer discovers this sidecar for each Delphes input and reads
the original LHE path recorded there. Use strict mode for a production shard:

```bash
pixi run --manifest-path Analysis/pixi.toml analyze \
  /path/to/run/delphes_ATLAS/delphes.root \
  --require-lhe \
  -o /path/to/analysis/job_000000.root
```

If a standalone run has been moved, the original LHE file and sidecar can be
supplied explicitly for a **single** Delphes input:

```bash
pixi run --manifest-path Analysis/pixi.toml analyze \
  /new/location/delphes.root \
  --lhe /new/location/pwgevents.lhe \
  --lhe-map /new/location/lhe-provenance.json \
  --require-lhe \
  -o /new/location/analysis.root
```

`--lhe-map` names the provenance manifest, not an array of guessed ordinal
matches. `--lhe` overrides the source path; it does not bypass the recorded
source-content or Delphes-content checks. A bare LHE file and a matching
event count are insufficient evidence of the association. The output ROOT
object `lhe_metadata` records full source hashes, paths, and the per-input
provenance needed to audit the join.

Legacy Delphes files without LHE provenance remain readable without
`--require-lhe`; the new LHE branches are then present but marked unavailable.
Strict mode requires a matched hard event for every output row. It does
**not** require that every hard event contain four decay leptons, so the
undecayed `gg_H` signal remains valid. To recover hard observables from an old
sample, use a trustworthy recorded event association or rerun the shower and
simulation with the updated chain. Equal numbers of events do not establish
that association.

The standalone validator supports the same production requirement:

```bash
python -m Analysis.validate_analysis_output \
  /path/to/analysis/job_000000.root --require-lhe
```

Without this option it also accepts legacy compact files with no LHE
branches. When any LHE branch is present, it checks the complete LHE schema
and consistency of the availability masks. Its summary counts matched hard
events, four-lepton candidates, and Higgs records separately. A zero hard
four-lepton count for the current `gg_H` sample is expected, even when every
event has a valid LHE association.

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

The shard validator requires individual event weights and their accumulated
statistics to be finite, but it does not require every shard to have a
positive net `sum(weight)`: a small signed-weight shard may legitimately have
zero or negative net weight. The physical normalization check is made only
after all shards for a sample have been aggregated, where the total `S` in the
formula above must be finite and positive.

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
summary is written to `pseudo_data_manifest.json`. Because merging and
pseudo-data resampling require multiple streaming passes, the tool records and
rechecks each source's device, inode, size, nanosecond timestamps, and SHA-256
digest before finalizing an output. It aborts if an input is replaced or
modified between passes. The campaign manifest also stores the SHA-256 digest
of every completed `data*.root` file so downstream stages can verify that they
are using the exact pseudo-experiments described by the manifest. A complete
campaign is built in a private staging directory and published only after all
files and the manifest pass these checks; a failed rebuild therefore leaves no
partial new campaign and does not replace an existing campaign. With
`--overwrite`, keep enough free space for the existing campaign and one newly
staged campaign until publication succeeds. Publication and rollback use
no-clobber installs and verify the exact inode and digest before removing a
published file. If another process replaces an output during rollback, that
replacement is preserved and the command reports a conflict. When it cannot
restore a displaced pre-campaign file at its original name without clobbering
the concurrent file, it retains the old object in a type-safe recovery
directory beside the outputs as `.NAME.pre-campaign-*.recovery/NAME` and
reports that recovery path. Recovery copies are also retained after a failed
overwrite even when restoration succeeds, so a replacement arriving directly
after the no-clobber restore cannot destroy the last copy. If even that durable
recovery move fails (for example because of an I/O or permission error), the
tool disables automatic cleanup of the private `.pseudo_data_campaign.*`
staging directory and reports its path for manual recovery.

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

The repository's `BackgroundRemoval` trainers consume these files directly
with a signed weighted BCE on a finite, configured logit interval: central
values use net sums `sum(w)` and uncertainties use `sum(w^2)`. Absolute
weights appear only in the minibatch proposal and feature preprocessing, with
signed importance coefficients that leave the intended net-normalized loss
unchanged. The logit bound prevents the signed finite-sample objective from
running to minus infinity; it does not clip or replace event weights.

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
  `truth_` and `reco_` respectively;
- the LHE hard-record observables, availability masks, and source identifiers
  described below.

### LHE branches

LHE lepton kinematics use outgoing (`ISTUP=1`) electrons and muons from the
hard record. Two disjoint SFOS pairs are chosen with the same leading-pair
and subleading-pair convention as Analysis, but **no lepton pT/eta cuts or
fiducial event cuts** are imposed. The kinematic definitions are shared with
`Tools/four_lepton_kinematics.py`. These quantities therefore exist
independently of the dressed-truth and reconstructed selections.

| Branches | Meaning |
|---|---|
| `lhe_available` | A source LHE event was matched and read successfully |
| `lhe_event_index`, `lhe_source_id` | One-based source-event index and source-content identifier; together identify the source event across merged shards |
| `lhe_process_id`, `lhe_weight`, `lhe_scale` | Raw `IDPRUP`, signed `XWGTUP`, and `SCALUP` from the LHE event header |
| `lhe_n_final_leptons` | Number of direct outgoing electrons and muons in the hard record |
| `lhe_has_taus` | The hard record contains at least one outgoing tau |
| `lhe_has_four_leptons`, `lhe_type` | A candidate containing two disjoint SFOS electron/muon pairs exists; channel codes use the same 0–3 convention as truth/reco |
| `lhe_kinematics_valid` | All eleven four-lepton observables are finite |
| `lhe_cos_theta_star`, `lhe_cos_theta1`, `lhe_cos_theta2` | Hard four-lepton polar-angle observables |
| `lhe_Phi`, `lhe_Phi1`, `lhe_Psi` | Hard four-lepton azimuthal-angle observables |
| `lhe_m_Z1`, `lhe_m_Z2`, `lhe_m_ZZ` | Hard dilepton and four-lepton masses |
| `lhe_y_ZZ`, `lhe_pT_ZZ` | Hard four-lepton rapidity and transverse momentum |
| `lhe_has_higgs`, `lhe_higgs_kinematics_valid` | A status-1/2 Higgs is present; a unique terminal Higgs has finite production kinematics |
| `lhe_higgs_m`, `lhe_higgs_y`, `lhe_higgs_pT` | Mass, rapidity, and transverse momentum of that hard Higgs |

The source ID is the first 64 bits of the original LHE SHA-256. Full digests
are kept in `lhe_metadata`, and combining sources with colliding short IDs
but different full digests is rejected. Merging can renumber `event_id`
without changing the LHE source/index pair. Multiple accepted entries or
pseudo-data resamples of the same hard event intentionally share that pair.

Absent quantities are `NaN`, absent event indices and channel codes are `-1`,
and availability masks are false. A four-lepton candidate can exist while
its angles are mathematically undefined; in that case finite masses or
production kinematics can still be retained and `lhe_kinematics_valid` is
false. Check the relevant masks and finite values before applying a
deformation.

The current `gg_H` LHE contains an undecayed Higgs. Such events can have
`lhe_available=True` and valid `lhe_higgs_*` while
`lhe_has_four_leptons=False` and all four-lepton `lhe_*` observables are
unavailable. An LHE tau final state is not replaced by its later shower
electron/muon descendants. Neither case is repaired with dressed `truth_`
values. A decayed hard-process signal sample is required to define genuine
LHE Higgs-decay angular deformations.

`lhe_weight` remains the raw source-event weight for provenance; it is not
the normalized MC training weight. Use the merged `weight_nominal_pb` or
luminosity-scaled `weight` for physical yields, and multiply a chosen
deformation factor into that weight once. Merging and pseudo-data sampling
preserve the per-event LHE columns along with the other compact observables.

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
are printed only; the LHE availability columns are separate from these
truth/reconstruction overlap diagnostics.
