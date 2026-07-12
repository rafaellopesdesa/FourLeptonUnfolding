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
POWHEG does not use a separate `configure` step.

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

In Unity, use the /work/ space for compilation

```bash
cd /work/rclsa_umass_edu/puser]
mkdir Generation; cd Generation
mkdir install
git clone https://github.com/rafaellopesdesa/FourLeptonUnfolding
cd FourLeptonUnfolding/Generation
./install_generators.sh --prefix ../../install --jobs 8 --skip-apt --gsl-module gsl/2.8
source env.sh
```

## Run the four combinations

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

Without `--run-card`, the runner uses the process's upstream `testrun-lhc`
card, changing only the event count, seed, PDF ID and grid-reuse switches. This
is intended as an installation smoke test. Production cards for the 10-million
event samples should be committed separately and validated before submission.

## Outputs

Each run gets its own directory under `runs/` and contains the resolved
`powheg.input`, logs, LHE input, showered HepMC output, and `run-metadata.txt`.
Generated samples and locally installed software are excluded from Git.

## Physics scope of this first baseline

The `ZZ` process contains the leptonic diboson matrix element supplied by its
POWHEG implementation. The `gg_H` process produces the Higgs and this baseline
forces `H -> ZZ(*) -> 4e/4mu/2e2mu` in the shower. Before using these events for
full phase-space unfolding, we must validate decay angles, off-shell behavior,
identical-lepton effects, and the precise POWHEG shower-veto prescription in
both showers. A later stage can compare this factorized baseline to a dedicated
four-lepton matrix element such as POWHEG-BOX-RES `gg4l`.
