# Delphes detector simulation

This directory turns the showered HepMC events from `Generation/` into the
standard Delphes ROOT tree using Delphes's bundled ATLAS card. It performs no
event selection or analysis.

## Version and detector card

The installer pins Delphes 3.5.1, the latest stable release when this setup was
created. It uses the release's own `cards/delphes_card_ATLAS.tcl` and verifies
its SHA-256 checksum before compiling. Delphes is a generic fast-simulation
framework and this is an ATLAS-like public parameterization distributed by
Delphes; it is not an official ATLAS detector simulation or centrally
validated Run-3 configuration. Pile-up is not enabled in the standard card.

## Install

On Ubuntu 24.04 with sudo access:

```bash
cd FourLeptonUnfolding/Simulation
./install_delphes.sh --jobs 8
source env.sh
```

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
