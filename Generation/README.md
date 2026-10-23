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

After installing Generation, Delphes, and the Analysis Pixi environment, use
`submit_generation.sh` to create Unity Slurm campaigns with 1000--10000 events
per job. Each task now runs the complete generation, simulation, and compact
analysis chain in transient worker scratch. The full workflow and output
layout are in
[`../BatchSubmit/README.md`](../BatchSubmit/README.md).

## Run the four combinations

By default, the runner uses the validated Run-3 cards in `../PowhegCards`.
These set 6800 GeV per beam ($\sqrt{s}=13.6$ TeV) and otherwise keep the
generation phase space as inclusive as the matrix element permits.
`--run-card` remains available for explicit alternatives, but every run card
must retain that beam energy. Before either shower starts, the runner parses
the authoritative LHE `<init>` record and refuses any file whose two beam
energies are not 6800 GeV. This check also covers external `--lhe` inputs.

The forced `gg_H` shower decay includes `Z -> e+e-`, `mu+mu-` and
`tau+tau-`. After pulling a change to `powheg_pythia8.cc`, rerun the installer
once to rebuild the small PYTHIA bridge executable; PYTHIA and Herwig themselves
do not need to be recompiled. The Herwig decay selection is generated at run
time.

For PYTHIA matching, `POWHEG:nFinal` is set to 1 for `gg_H` and 2 for `ZZ`.
The latter may look surprising because the LHE record contains four final-state
leptons. POWHEG also records their two parent Z bosons as status-2 resonances;
when `PowhegHooks` is initialized, PYTHIA's hard shower system contains the two
Z bosons (and an optional POWHEG emission), while the decay leptons are attached
afterwards. Counting the leptons instead would make `PowhegHooks` abort with
`wrong number of final state particles in event`.

Herwig does not use PYTHIA's `nFinal` logic. The generated Herwig input follows
the upstream `LHE-POWHEG.in` prescription: it reads `SCALUP` from each LHE event
as the shower-veto scale through `MaxPtIsMuF Yes` and
`RestrictPhasespace Yes`. Consequently the resonance multiplicity issue does
not occur in the Herwig path and no Herwig configuration change is required.

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
`powheg.input`, logs, LHE input, showered HepMC output, `run-metadata.txt`, and
`lhe-provenance.json`.
Generated samples and locally installed software are excluded from Git.
The metadata records both beam energies and `sqrt_s_gev=13600`.

The provenance sidecar records the original LHE path, SHA-256 and event
count, the HepMC path and SHA-256, and the source-event numbering convention.
Source identity is retained automatically:

- the PYTHIA bridge sets each output event number to the one-based index of
  the actual LHE event that was read, including gaps from failed attempts;
- HERWIG receives a private copy of the LHE with native source-event numbers
  and uses `LesHouchesHandler:EventNumbering LHE`. The original LHE is not
  changed.

These identifiers follow the accepted hard event through the shower; they
are not a new consecutive numbering of accepted events. Repeated source
events, when present, retain their source identity and are recorded in the
provenance summary. The Simulation and Analysis stages validate this
association before copying hard-record observables into the compact tree.

After pulling this feature, rerun `install_generators.sh` with the same
installation options to rebuild the small `powheg-pythia8` bridge. HERWIG
uses the updated generated input and does not require recompilation for
source numbering. Existing HepMC outputs without the provenance sidecar
cannot be safely joined by event position. If the original LHE was retained,
reshower it with `--lhe` using the updated runner, then rerun Simulation and
Analysis. The bookkeeping change itself does not alter the POWHEG matrix
element or require new integration grids.

Standalone runs retain their intermediates. In a submitted Unity campaign,
the same files live only in worker scratch and are deleted after the compact
Analysis ROOT shard has been validated and published. Its hard-event
observables and source identifiers have already been copied into the compact
tree before that cleanup.

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

### What is present at LHE level

The hard record and the shower decay record are different objects in this
baseline. For `gg_H`, POWHEG writes an **undecayed Higgs**. Its LHE mass,
rapidity, and transverse momentum can be retained for production-level
reweighting, but the LHE file contains no Higgs-decay lepton angles or
dilepton masses. PYTHIA and HERWIG subsequently generate those decays.
For `ZZ`, the POWHEG LHE record already contains the leptonic final state.

The compact analysis therefore keeps LHE information separate from its
existing dressed-particle `truth_` quantities. Missing LHE decay variables
remain unavailable; they are never filled from showered or dressed leptons.
To deform Higgs **decay** distributions at the hard-matrix-element level,
the signal generator must supply a decayed four-lepton LHE record. Retaining
the current undecayed record does not create that information. Smooth
dressed-particle closure tests can still use the existing `truth_` variables.

When the LHE final state contains taus, their later electron or muon decay
products belong to the shower/decay record. They must not be relabeled as
direct LHE electrons or muons. The analysis records availability and tau
content explicitly so these cases can be selected correctly.

Do not reuse a `ZZ` integration or upper-bound grid produced before the
`m4l > 95 GeV` patch, including grids made with either the original 4 GeV card
or the temporary 20 GeV card. The revised phase space requires a fresh
standalone run or a new Unity campaign/grid stage.
