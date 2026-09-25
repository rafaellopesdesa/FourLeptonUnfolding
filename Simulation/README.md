# Delphes detector simulation

This directory turns the showered HepMC events from `Generation/` into a
Delphes ROOT tree based on Delphes's bundled ATLAS card. The resolved card is
adapted for a four-lepton fiducial cross-section and unfolding study. It
performs no event selection or physics analysis.

## Version and detector card

The installer pins Delphes 3.5.1, the latest stable release when this setup was
created. It uses the release's own `cards/delphes_card_ATLAS.tcl` and verifies
its SHA-256 checksum before compiling. Delphes is a generic fast-simulation
framework and this is an ATLAS-like public parameterization distributed by
Delphes; it is not an official ATLAS detector simulation or centrally
validated Run-3 configuration. Pile-up is not enabled in the standard card.

### Event retention and truth/reconstruction content

Delphes itself does not discard events merely because no detector object was
reconstructed: its HepMC readers fill one `Delphes` tree entry after every
input event. The runner now counts the `E` records in each HepMC file and
requires the output entry count to match exactly (or to match `--max-events`
for a smoke test). A mismatch is a failed simulation and cannot receive a
`SUCCESS` marker.

The resolved card keeps both truth and reconstructed views:

| Branch | Meaning |
|---|---|
| `Particle` | All HepMC particles, including status and ancestry |
| `StableParticle` | Explicit HepMC status-1, post-shower particles |
| `DressedElectron`, `DressedMuon` | Prompt W/Z/gamma*-origin truth leptons dressed with eligible photons in `deltaR < 0.1` |
| `RecoElectronNoIso`, `RecoMuonNoIso` | Dressed H4l leptons after momentum response and loose reconstruction+ID efficiency |
| `RecoElectron`, `RecoMuon` | Final H4l response objects after the separate loose isolation efficiency |
| `Electron`, `Muon` | Unmodified generic-card Delphes objects, retained only as a diagnostic |
| `HasFourRecoLeptons` | Technical marker: at least four final H4l reconstructed electrons plus muons |

`HasFourRecoLeptons` is not the analysis selection: it imposes no charge,
flavor, pairing, mass, or ordered-pT requirement. The final reconstructed
branches have already undergone the probabilistic isolation response, but all
kinematic and event choices remain downstream so every generated event can
participate in response, efficiency, and out-of-fiducial migrations.

The fiducial-truth branches apply photon dressing to stable electrons and
muons, following Section 4.1.2 of `ATL-COM-PHYS-2019-930`:

- the bare electron or muon is stable (`status = 1`);
- it must descend from a W, Z, or hard virtual photon and have no
  hadron-decay ancestor;
- dressing photons are stable, have no hadron-decay ancestor, and satisfy
  `deltaR(photon, bare lepton) < 0.1`;
- no photon-pT threshold is imposed;
- each eligible photon is assigned once, to the closest eligible lepton.

The dressed four-momentum is the bare four-momentum plus all assigned photon
four-momenta. Delphes `M1` and `M2` are treated as two mother indices, not as
the endpoints of an array interval. Cycles are guarded explicitly, and hadron
ancestry traversal stops at an incoming quark or gluon so that the beam proton
does not make every hard-process lepton look nonprompt. A photon ancestor is
accepted only when its invariant mass exceeds 5 GeV. This retains the
`gamma* -> lepton lepton` part of continuum four-lepton production while
rejecting low-mass conversion electrons.

The note removes tau decay modes from its quoted fiducial signal. This project
previously made the explicit choice to retain them because generation and the
normalization use a branching fraction including taus. The card therefore
accepts `W/Z -> tau -> electron/muon` chains and records that choice in
`simulation-metadata.txt`. Exact note-level tau removal would require changing
that project setting and regenerating the resolved cards; it is not a
per-analysis cut in the current workflow.

The original bare leptons and photons remain available in `StableParticle`,
and `Particle` retains the complete ancestry used to reject photons from
hadron decays. No fiducial pT, eta, pairing, or mass requirement is applied
while constructing the dressed branches.

### H4l reconstruction and efficiency model

The generic Delphes ATLAS card is not suitable for the soft four-lepton
response by itself: its final electron and muon efficiencies are zero below
10 GeV, and its generic cone isolation is not the Run-2 particle-flow working
point. More importantly, using its pre-isolation collections admits
nonprompt leptons and creates an artificial reconstructed-only population.

The resolved card now builds dedicated `RecoElectron` and `RecoMuon`
collections from the prompt dressed leptons. It applies:

1. the Run-2 ECAL energy-resolution proxy used in
   `OffshellAngularProduction` for electrons, and the existing momentum proxy
   for muons;
2. a loose reconstruction+identification efficiency;
3. a separate loose prompt-lepton isolation efficiency.

The final two stages are independent Bernoulli efficiencies. They are not also
passed through the generic Delphes cone-isolation modules, so isolation is
counted exactly once. The response depends on pT and broad `abs(eta)` regions
only; there is no phi dependence. A technical response buffer starts at
4 GeV so an object smeared upward can pass the downstream 5 GeV selection.
The runner sets a nonzero deterministic Delphes random seed from each
generation job's `run-metadata.txt`, so the stochastic response is
reproducible. For a direct HepMC file without metadata it uses a deterministic
fallback; `--random-seed N` supplies an explicit base seed and increments it
across multiple discovered inputs.

The central pT-dependent response is continuous and piecewise linear between
the following anchors. This avoids giving an unbinned classifier artificial
features at hard efficiency-bin edges:

| electron pT anchor (GeV) | reco+Loose-ID | Loose_VarRad isolation proxy |
|---:|---:|---:|
| 5 | 0.85 | 0.68 |
| 7 | 0.90 | 0.77 |
| 10 | 0.92 | 0.84 |
| 15 | 0.95 | 0.91 |
| 20 | 0.953 | 0.95 |
| 25 | 0.957 | 0.97 |
| 30 | 0.96 | 0.985 |

| muon pT anchor (GeV) | reco+Loose-ID | PflowLoose isolation proxy |
|---:|---:|---:|
| 5 | 0.96 | 0.72 |
| 6 | 0.98 | 0.80 |
| 8 | 0.985 | 0.88 |
| 10 | 0.99 | 0.92 |
| 15 | 0.99 | 0.96 |
| 20 | 0.99 | 0.985 |
| 30 | 0.99 | 0.995 |

Both efficiencies ramp continuously from zero at the 4 GeV technical buffer
to the 5 GeV anchor and plateau above the last anchor.

Small electron eta modifiers model the barrel/endcap transition and forward
losses; muon eta modifiers are at most 0.5%. The 5–7 GeV electron response and
the 2.47–2.5 edge are explicit extrapolations because the public measurement
does not validate that full region. These values are a documented
phenomenology-level proxy, not official Run-3 calibration constants.
The electron isolation anchors use the public `Loose_VarRad` curve, conditional
on Medium electron identification, as a proxy rather than the exact H4l
`FixedCutPflowLoose` working point. The muon isolation anchors are likewise
conditional performance measurements (Medium identification and vertex
association) applied here after a Loose-ID proxy. Treating these factors as
independent Bernoulli stages is a deliberate phenomenology approximation.

### Electron ECAL resolution

`RecoElectron` and `RecoElectronNoIso` now use
`H4lElectronECalSmearing`, model `atlas_run2_ecal_snc_v1`, ported unchanged
from `OffshellAngularProduction`. It represents the total calibrated ECAL
energy response without a track-energy combination. The former tracking-like
formula was not appropriate as the final electron energy resolution.

For dressed energy $E$ and pseudorapidity $\eta$, define
$T=\max(E/\cosh\eta,1\;\mathrm{GeV})$. The relative response is

$$
\left(\frac{\sigma_E}{E}\right)^2 =
\frac{S_T^2}{T} +
\frac{N_{0,T}^2+N_{\mathrm{PU},T}^2}{T^2} +
C_{\mathrm{MC}}^2+c_{\mathrm{data}}^2.
$$

The effective coefficients are:

| $\lvert\eta\rvert$ interval | $S_T$ | $N_{0,T}$ | $N_{\mathrm{PU},T}$ | $C_{\mathrm{MC}}$ | $c_{\mathrm{data}}$ |
|---|---:|---:|---:|---:|---:|
| $[0,0.8]$ | 0.09 | 0.30 | 0.55 | 0.004 | 0.007 |
| $(0.8,1.37]$ | 0.12 | 0.84 | 0.55 | 0.004 | 0.009 |
| $(1.37,1.52]$ | 0.15 | 1.20 | 0.70 | 0.010 | 0.025 |
| $(1.52,2.0]$ | 0.10 | 0.55 | 0.60 | 0.004 | 0.015 |
| $(2.0,2.5)$ | 0.08 | 0.50 | 0.60 | 0.004 | 0.017 |

These are source-informed phenomenological coefficients, not an official
ATLAS calibration table. Their basis is the Run-2 electron-response behavior
in [arXiv:1908.00005](https://arxiv.org/abs/1908.00005) and the additional
data smearing discussed in
[arXiv:2309.05471](https://arxiv.org/abs/2309.05471). The fixed pile-up-noise
term is an effective Run-2 contribution; this setup still has no overlaid
pile-up events. The Delphes `MomentumSmearing` module is used only as the
positive response engine, evaluating the dressed energy and eta. The response
is applied once, before reconstruction/ID and isolation efficiencies.

This is a generated-card change. Existing Delphes and compact Analysis files
must be regenerated, but Delphes does not need to be recompiled and POWHEG
events do not need to be regenerated. New full-chain batch tasks pick up the
model automatically.

Both truth and reconstruction are finally selected in `Analysis/` with the
user-chosen common acceptance `pT > 5 GeV`, `|eta| < 2.5`. The attached note
uses `|eta| < 2.7` at fiducial level, while the real Run-2 reconstruction uses
electrons above 7 GeV within `|eta| < 2.47` and loose muons above 5 GeV within
`|eta| < 2.7`. The common acceptance is therefore a deliberate matched
simplification for the unfolding study.

Jets are clustered with anti-kt `R = 0.4`. The finder retains a 20 GeV
technical buffer, while a deterministic acceptance filter keeps all upstream
jets with `pT > 30 GeV` and `|eta| < 4.5`; it adds no further stochastic
inefficiency. Reconstructed jets have nevertheless already passed the generic
Delphes calorimeter/JES response and object-overlap processing, so the full
stored collection should not be described as globally unit-efficiency. No jet
enters the current four-lepton event selection. The publication-level truth
definition instead uses `|y| < 4.4` and excludes leptonic vector-boson decay
products; that distinction must be implemented before jet observables are
added to the unfolding.

## Install

On Ubuntu 24.04 with sudo access:

```bash
cd FourLeptonUnfolding/Simulation
./install_delphes.sh --jobs 8
source env.sh
```

The truth-dressing implementation extends Delphes's `LeptonDressing` module.
After pulling this change, rerun `install_delphes.sh` once so the new ancestry
patch is applied and incremental `make` recompiles that module. A full clean
installation is not required.
The installer recognizes previously applied dressing features even after a
later incremental patch modifies the same source region, so existing source
trees can be upgraded in place.

Without sudo, the script can bootstrap ROOT locally with micromamba:

```bash
./install_delphes.sh --skip-apt --jobs 8
source env.sh
```

This route expects `c++`, `make`, `git`, `curl`, `patch`, `tar`, `bzip2`, and
the usual Ubuntu 24.04 runtime libraries.

### Unity installation

On Unity, `mpich/4.2.1` is the module that makes ROOT available. Install below
`/work/pi_rclsa_umass_edu/` and always use `--skip-apt`:

```bash
cd /work/pi_rclsa_umass_edu/$USER/FourLeptonStudy/FourLeptonUnfolding/Simulation
module load mpich/4.2.1
module load root/6.30.06
./install_delphes.sh \
  --skip-apt \
  --jobs 8 \
  --root-module mpich/4.2.1
source env.sh
```

The `--root-module mpich/4.2.1` option performs the equivalent of
`module load mpich/4.2.1`, verifies that `root-config` becomes available, and
records the module in the generated `env.sh`. Sourcing `env.sh` therefore
reloads the same module for later interactive or worker-node runs. Keeping the
repository and installation under `/work` ensures that worker nodes can access
the code, ROOT environment, and Delphes libraries.

## Run

The runner accepts all layouts produced by the current generation scripts.

Standalone run directory:

```bash
./run_simulation.sh ../Generation/runs/gg_H_pythia_seed101
```

One direct HepMC file (the process must be supplied if no adjacent
`run-metadata.txt` exists):

```bash
./run_simulation.sh /path/to/events.hepmc3 --process gg_H
```

Complete Unity campaign:

```bash
./run_simulation.sh \
  /work/pi_rclsa_umass_edu/FourLeptonUnfoldingGeneration/ggH_pythia_run3_10M
```

For a campaign, only `jobs/job_*` directories carrying the generation
`SUCCESS` marker are processed. To parallelize detector simulation in a later
Slurm layer, the same runner can be called on one `jobs/job_*` directory per
array task. It reads the HepMC version header and selects `DelphesHepMC2` or
`DelphesHepMC3` accordingly, using the filename extension only as a fallback.
More precisely, it uses the serialization marker: Herwig's default output is
written by HepMC3's `WriterAsciiHepMC2`, so its header reports a HepMC3 library
version while its event records use HepMC2 `IO_GenEvent` syntax. Such files
must be processed with `DelphesHepMC2`.

After Delphes exits, the runner checks both the required branch names and their
GenParticle/Electron/Muon leaf schemas, annotates each event with
`HasFourRecoLeptons`, and validates exact event retention. A malformed or
incomplete ROOT tree is treated as a failed simulation rather than receiving a
`SUCCESS` marker. Existing Delphes outputs must be regenerated because the
meaning of `RecoElectron` and `RecoMuon` has intentionally changed.

By default each result is written next to its HepMC input:

```text
job_000000_seed1001/
  events.hepmc3
  run-metadata.txt
  delphes_ATLAS/
    delphes.root
    delphes.log
    delphes_card_ATLAS_resolved.tcl
    simulation-metadata.txt
    SUCCESS
```

Use `--output-root DIR` to place outputs elsewhere. Run `--help` for smoke-test
limits and overwrite behavior.

## Higgs normalization

The `gg_H` POWHEG sample carries the inclusive gluon-fusion normalization even
though the shower forces `H -> ZZ(*) -> 4l`. For `gg_H`, the runner therefore
sets

```text
WeightScale = BR(H -> ZZ -> 4l, including taus) = 2.771E-04.
```

The small version-pinned Delphes patch in `patches/` applies this factor while
reading either HepMC format. The standard `Event.Weight` value and every value
in the standard `Weight` branch are multiplied by the factor. For consistency,
`Event.CrossSection` and `Event.CrossSectionError` are scaled too. For `ZZ`,
the factor is exactly 1. The resolved card and `simulation-metadata.txt` record
the applied value for every output.

Official references:

- [Delphes repository and usage](https://github.com/delphes/delphes)
- [Delphes 3.5.1 release](https://github.com/delphes/delphes/releases/tag/3.5.1)
- [bundled ATLAS card](https://github.com/delphes/delphes/blob/3.5.1/cards/delphes_card_ATLAS.tcl)
- [ATLAS Run-2 muon reconstruction, identification, and isolation performance](https://arxiv.org/abs/2012.00578)
- [ATLAS Run-2 electron reconstruction, identification, and isolation performance](https://arxiv.org/abs/2308.13362)
