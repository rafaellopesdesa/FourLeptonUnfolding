# Event generation

This directory provides a first, reproducible Ubuntu 24.04 workflow for

- POWHEG-BOX-V2 `gg_H` and `ZZ`;
- PYTHIA 8.317;
- Herwig 7.3.0/ThePEG 2.3.0;
- GSL 2.8, Boost 1.83.0, LHAPDF, FastJet and HepMC3.

POWHEG is cloned from its current official GitLab repository. The two process
repositories are initialized as pinned submodules, and the resolved commits are
recorded in `software/versions.txt`. The installer also creates the process
object/module directories expected by the upstream make-only build system;
POWHEG does not use a separate `configure` step. Its build is explicitly bound
to the locally installed `lhapdf-config` and `fastjet-config`. The LHAPDF
include flags and library directory are passed to the POWHEG compiler and
linker instead of relying on system search paths.

## Install

The full source build is substantial. On a clean machine, allow tens of GB of
free disk space and expect it to take from tens of minutes to several hours.

```bash
cd FourLeptonUnfolding/Generation
./install_generators.sh
source env.sh
```

To install elsewhere or limit compilation parallelism:

```bash
./install_generators.sh --prefix /data/generators/four-lepton --jobs 8
```

The script installs Ubuntu build packages with `apt`. Use `--skip-apt` only if
the compiler and base build tools are already available. GSL is built inside
the local installation prefix, so root privileges are not needed for it:

```bash
./install_generators.sh --skip-apt
```

This no-`sudo` route still expects `c++`, `gfortran`, `make`, `cmake`, `curl`,
`git`, and `tar` to be available on the host system. Boost headers and the
compiled Boost.Test library are built inside the local installation prefix.
The required `CT14lo`, `CT14nlo`, and `NNPDF31_nlo_as_0118` PDF data sets are
downloaded directly into that prefix before Herwig is built. This does not
depend on LHAPDF's optional Python bindings or its `lhapdf install` command.

On systems providing GSL through Environment Modules or Lmod, the local GSL
build can be skipped:

```bash
./install_generators.sh --skip-apt --gsl-module gsl/2.8
```

Use `--gsl-module auto` to try `gsl/2.8`, or set the `GSL_MODULE` environment
variable. The installer validates the loaded module with `gsl-config`; if the
module command, requested module, or GSL headers are unavailable, it falls back
to building GSL 2.8 in the local installation prefix. The resolved GSL prefix
is also written to `env.sh`, so subsequent generation jobs do not need to run
`module load` themselves.

## Unity specific recommendations

On Unity, clone the repository and compile the complete generator stack below
`/work/pi_rclsa_umass_edu/`, not in `/home`. `/work` is mounted on the worker
nodes and is intended for job I/O; compiling in `/home` can leave batch jobs
without access to the expected code, executables, or libraries.

```bash
mkdir -p /work/pi_rclsa_umass_edu/$USER/FourLeptonStudy
cd /work/pi_rclsa_umass_edu/$USER/FourLeptonStudy
git clone https://github.com/rafaellopesdesa/FourLeptonUnfolding
cd FourLeptonUnfolding/Generation
./install_generators.sh \
  --prefix /work/pi_rclsa_umass_edu/$USER/FourLeptonStudy/install \
  --jobs 8 --skip-apt --gsl-module gsl/2.8
source env.sh
```

After installation, use `submit_generation.sh` to create Unity Slurm campaigns
with 1000--10000 events per job. The full workflow and output layout are in
[`../BatchSubmit/README.md`](../BatchSubmit/README.md).

## Run the four combinations

By default, the runner uses the validated Run-3 cards in `../PowhegCards`.
These set 6800 GeV per beam and otherwise keep the generation phase space as
inclusive as the matrix element permits. `--run-card` remains available for
explicit alternatives.

The forced `gg_H` shower decay includes `Z -> e+e-`, `mu+mu-` and
`tau+tau-`. After pulling a change to `powheg_pythia8.cc`, rerun the installer
once to rebuild the small PYTHIA bridge executable; PYTHIA and Herwig themselves
do not need to be recompiled. The Herwig decay selection is generated at run
time.

```bash
./run_generation.sh gg_H pythia --events 1000 --seed 101
./run_generation.sh gg_H herwig --events 1000 --seed 102
./run_generation.sh ZZ   pythia --events 1000 --seed 201
./run_generation.sh ZZ   herwig --events 1000 --seed 202
```

For a controlled shower comparison, generate the POWHEG LHE sample once and
reuse it. For example:

```bash
./run_generation.sh gg_H pythia --events 1000 --seed 301
./run_generation.sh gg_H herwig \
  --lhe runs/gg_H_pythia_seed301/pwgevents.lhe \
  --events 0 --seed 301
```

Without `--run-card`, the runner uses the committed Run-3 card in
`../PowhegCards`, changing only the event count, seed, PDF ID and grid-reuse
switches in the private run copy.

## Outputs

Each run gets its own directory under `runs/` and contains the resolved
`powheg.input`, logs, LHE input, showered HepMC output, and `run-metadata.txt`.
Generated samples and locally installed software are excluded from Git.

## Physics scope of this first baseline

The `ZZ` process contains the leptonic diboson matrix element supplied by its
POWHEG implementation and is configured for charged electrons, muons and taus.
Its generation-level cuts are `mll > 4 GeV` and `m4l > 95 GeV`. They regulate
the low-mass virtual-photon region and avoid generating below the planned
fiducial measurement while retaining margins below the all-SFOS-pair
requirement `mll > 5 GeV` and the `m4l > 105 GeV` extended region. The
four-lepton threshold is supplied by a version-pinned
patch because upstream POWHEG-BOX-V2/ZZ has no native `m4lmin` keyword. After
pulling this change, rerun `install_generators.sh`; its incremental build will
apply the patch and rebuild only the affected POWHEG process. The `gg_H`
process produces the Higgs, and this baseline forces `H -> ZZ(*)` with each Z
decaying to `e`, `mu` or
`tau` pairs in the shower. Before using these events for full phase-space
unfolding, we must validate decay angles, off-shell behavior, identical-lepton
effects, tau feed-down and the precise POWHEG shower-veto prescription in both
showers. A later stage can compare this factorized baseline to a dedicated
four-lepton matrix element such as POWHEG-BOX-RES `gg4l`.

Do not reuse a `ZZ` integration or upper-bound grid produced before the
`m4l > 95 GeV` patch, including grids made with either the original 4 GeV card
or the temporary 20 GeV card. The revised phase space requires a fresh
standalone run or a new Unity campaign/grid stage.
