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
| `DressedElectron`, `DressedMuon` | Fiducial-truth leptons dressed with eligible photons in `deltaR < 0.1` |
| `RecoElectron`, `RecoMuon` | Loose reconstructed leptons before isolation |
| `Electron`, `Muon` | Standard-card reconstructed leptons after isolation and overlap removal |
| `HasFourRecoLeptons` | Technical marker: at least four loose reconstructed electrons plus muons |

`HasFourRecoLeptons` is not the analysis selection: it imposes no charge,
flavor, pairing, mass, ordered-pT, or isolation requirement. Those choices
must remain downstream so every generated event can participate in response,
efficiency, and out-of-fiducial migrations.

The bundled card normally sets both electron and muon object efficiencies to
zero below 10 GeV and its tracking parameterization stops at `|eta|=2.5`.
In the per-job resolved copy only, the runner changes the reconstructed-lepton
threshold to `pT > 5 GeV` and extends the final endcap tracking-efficiency and
momentum-resolution bins to `|eta| < 2.7` for both flavors. This extrapolation
is intentionally analysis-specific and is not a standard ATLAS electron
acceptance. The original endcap efficiencies/resolutions are held constant
from 2.5 to 2.7. Analysis-level ordered-pT and event cuts remain downstream.

The fiducial-truth branches apply photon dressing to stable electrons and
muons. A photon is eligible only when it is status 1, has no hadron-decay
ancestor before the ancestry chain reaches an incoming quark or gluon, and
satisfies `deltaR < 0.1` relative to a bare lepton. Traversal stops at that
parton so that the beam proton is not mistaken for a hadron-decay ancestor.
The Delphes `M1` and `M2` fields are treated as two individual mother indices,
not as the endpoints of an inclusive particle-index range.
There is no photon-pT threshold. If one photon lies inside more than one lepton
cone, it is assigned only to the nearest bare lepton, preventing double
counting. The dressed four-momentum is the bare lepton four-momentum plus all
photons assigned to it.

The original bare leptons and photons remain available in `StableParticle`,
and `Particle` retains the complete ancestry used to reject photons from
hadron decays. No fiducial pT, eta, isolation, pairing, or mass requirement is
applied while constructing the dressed branches.

## Install

On Ubuntu 24.04 with sudo access:

```bash
cd FourLeptonUnfolding/Simulation
./install_delphes.sh --jobs 8
source env.sh
```

The truth-dressing implementation extends Delphes's `LeptonDressing` module.
After pulling this change, rerun `install_delphes.sh` once so the incremental
`make` recompiles that module. A full clean installation is not required.
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

After Delphes exits, the runner also checks the required truth and reconstructed
branches, annotates each event with `HasFourRecoLeptons`, and validates exact
event retention. A malformed or incomplete ROOT tree is treated as a failed
simulation rather than receiving a `SUCCESS` marker.

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
