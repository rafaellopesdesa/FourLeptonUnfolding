# Unity full-chain batch production

`Generation/submit_generation.sh` submits a storage-efficient Unity Slurm
campaign. Each array task runs

```text
POWHEG + shower -> Delphes -> compact Analysis ROOT tree
```

in worker-local scratch and atomically publishes only the compact Analysis
file. The private POWHEG-grid copy, LHE, HepMC, and Delphes ROOT file are
transient. Small cards, logs, checksums, and stage status remain in the
campaign directory. Use `--keep-intermediates` only for a small debugging
campaign when the complete per-task work area is genuinely needed.

Unity permits at most 1900 tasks in one array. Larger campaigns are split into
dependent waves of at most 1900 tasks, with at most 100 running concurrently by
default.

## One-time setup

Clone and install below `/work/pi_rclsa_umass_edu/`, which is visible from
worker nodes. Complete all three setup steps before submission:

```bash
cd /work/pi_rclsa_umass_edu/$USER/FourLeptonUnfolding

cd Generation
./install_generators.sh \
  --prefix /work/pi_rclsa_umass_edu/$USER/FourLeptonInstall \
  --jobs 8 --skip-apt --gsl-module gsl/2.8

cd ../Simulation
./install_delphes.sh --skip-apt --jobs 8 --root-module mpich/4.2.1

cd ..
pixi install --manifest-path Analysis/pixi.toml --locked
```

The submitter verifies both generated environment files and resolves the
Analysis Python interpreter once on the login node. Workers invoke that
interpreter directly, avoiding concurrent Pixi environment solves or locks.

## Submit the four samples

All four campaigns must use the same flat `--analysis-output-root`; their
collision-safe filenames then match the merger patterns automatically.

```bash
cd /work/pi_rclsa_umass_edu/$USER/FourLeptonUnfolding/Generation

ANALYSIS_SHARDS=/work/pi_rclsa_umass_edu/$USER/FourLeptonAnalysis/shards

./submit_generation.sh ZZ pythia \
  --jobs 1000 --events-per-job 10000 \
  --campaign ZZ_pythia_run3 \
  --analysis-output-root "$ANALYSIS_SHARDS"

./submit_generation.sh ZZ herwig \
  --jobs 1000 --events-per-job 10000 \
  --campaign ZZ_herwig_run3 \
  --analysis-output-root "$ANALYSIS_SHARDS"

./submit_generation.sh gg_H pythia \
  --jobs 1000 --events-per-job 10000 \
  --campaign gg_H_pythia_run3 \
  --analysis-output-root "$ANALYSIS_SHARDS"

./submit_generation.sh gg_H herwig \
  --jobs 1000 --events-per-job 10000 \
  --campaign gg_H_herwig_run3 \
  --analysis-output-root "$ANALYSIS_SHARDS"
```

The default scratch preference is `SLURM_TMPDIR`, then `TMPDIR`, then `/tmp`.
Pass `--scratch-root DIR` if the site provides a different node-local area.
`--time` covers the complete chain, not generation alone. Run `--help` for
memory, seed, concurrency, card, mass-region, and debugging options.

## POWHEG grid stage

Each campaign first runs one grid-preparation job. Event arrays start through
an `afterok` dependency and copy the grid into their private scratch area,
because POWHEG may update grid statistics while generating events. The
one-event LHE and HepMC used during grid setup are removed after the grid has
been validated.

`--reuse-grid DIR` accepts an existing compatible grid. A new grid records its
beam energy, process, PDF ID, and run-card checksum. Reuse is rejected unless
all four values are present and match, so metadata-less legacy grids must be
rebuilt. Continue to reuse only grids made with a compatible executable.

## Persistent layout

```text
CAMPAIGN/
  campaign.env
  submission.txt
  inputs/powheg.input
  grid/                         # one shared reusable POWHEG grid
  logs/grid-JOBID.out
  logs/events-ARRAYID_TASKID.out
  jobs/job_000000_seed1001/
    SUCCESS                     # only after compact ROOT publication
    slurm-status.txt            # stage and final path
    output-metadata.txt         # checksum and event count
    diagnostics/                # small generation/Delphes logs and cards
```

Compact shards are published separately, for example:

```text
ZZ_pythia_ZZ_pythia_run3_job_000000_seed1001.root
gg_H_herwig_gg_H_herwig_run3_job_000137_seed1138.root
```

Publication uses a temporary file in the destination directory, validates its
complete tree schema and entry count, and creates the final name atomically. A
requeued task accepts a final file only when its checksum and campaign-config
fingerprint match that task's recorded metadata; it never adopts a colliding
file from another campaign. `FAILED` and `slurm-status.txt` identify the
failing stage.

## Merge immediately after all four campaigns finish

```bash
cd /work/pi_rclsa_umass_edu/$USER/FourLeptonUnfolding

MERGED=/work/pi_rclsa_umass_edu/$USER/FourLeptonAnalysis/merged

pixi run --manifest-path Analysis/pixi.toml merge \
  "$ANALYSIS_SHARDS" \
  --output-directory "$MERGED" \
  --luminosity-fb 312 \
  --pseudo-data-ensembles auto \
  --seed 12345
```

Then make the complete PDF comparison:

```bash
pixi run --manifest-path Analysis/pixi.toml plot \
  "$MERGED" \
  --output "$MERGED/four_lepton_comparison.pdf"
```

The merger and plotter details, including signed Herwig weights, are in
[`../Analysis/README.md`](../Analysis/README.md) and
[`../Plotting/README.md`](../Plotting/README.md).

## Monitoring

```bash
squeue --me
sacct -j ARRAY_JOB_ID
tail -F /work/pi_rclsa_umass_edu/FourLeptonUnfoldingGeneration/CAMPAIGN/logs/events-ARRAYID_TASKID.out
```

A grid failure prevents its arrays from starting; a failed wave prevents the
next wave from starting. Inspect the Slurm log and the job's
`slurm-status.txt`/`diagnostics/` directory before resubmitting.

Unity references:

- [batch jobs](https://unityhpc.org/documentation/jobs/sbatch/)
- [array jobs](https://unityhpc.org/documentation/jobs/sbatch/arrays/)
- [large job counts](https://unityhpc.org/documentation/jobs/sbatch/large-count/)
- [storage](https://unityhpc.org/documentation/cluster_specs/storage/)
