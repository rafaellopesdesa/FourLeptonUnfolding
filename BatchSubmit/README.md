# Unity batch generation

This directory contains the Slurm worker used by
`Generation/submit_generation.sh`. Unity uses Slurm for non-interactive work,
and repeated simulations are a natural fit for an array job. Unity currently
allows up to 1900 tasks in one array, so the submission script splits larger
campaigns into dependent waves of at most 1900 tasks. Each wave is throttled
with Slurm's `%N` array syntax; the default is 100 simultaneously running
tasks.

## Before submitting

Both the repository and the locally compiled generator installation must be on
Unity's shared high-performance storage, below
`/work/pi_rclsa_umass_edu/`. Do not compile in `/home`: worker nodes must see
the same executables, libraries, cards and scripts. See `Generation/README.md`
for an example installation layout.

Run the submission script from the repository copied to `/work`:

```bash
cd /work/pi_rclsa_umass_edu/$USER/FourLeptonUnfolding/Generation
source env.sh

# 10 million events = 1000 array tasks x 10000 events/task
./submit_generation.sh gg_H pythia \
  --jobs 1000 \
  --events-per-job 10000 \
  --campaign ggH_pythia_run3_10M
```

Equivalent `gg_H herwig`, `ZZ pythia`, and `ZZ herwig` campaigns use the same
command. Use `--account NAME` only if your Unity setup requires an explicit
Slurm account. `--partition cpu` is the default. Run `--help` for the memory,
time, seed, concurrency, run-card and output options.

The script intentionally accepts 1000--10000 events per task. For a 10-million
event sample, 10000 per task produces 1000 tasks and fits in one Unity array.
Smaller chunks are supported; campaigns over 1900 tasks are automatically
chained into multiple arrays, so a later wave starts only after the preceding
wave succeeds.

## Grid preparation

POWHEG's adaptive integration and upper-bound construction are expensive. The
submission therefore has two stages:

1. a single batch job prepares a compatible integration grid;
2. the event array starts with an `afterok` dependency, copies the grid into
   each private job directory, and runs with `--keep-grids`.

Private copies are important because POWHEG can update statistics and bound
files during event generation. If a previously validated grid was made with
the same process, card, PDF and executable, pass it with `--reuse-grid DIR`.
Compatibility is the user's responsibility; the script checks only that the
expected grid and upper-bound files exist.

## Output layout

The default campaign root is
`/work/pi_rclsa_umass_edu/FourLeptonUnfoldingGeneration/`:

```text
CAMPAIGN/
  campaign.env
  submission.txt
  inputs/powheg.input
  grid/
  logs/grid-JOBID.out
  logs/events-ARRAYID_TASKID.out
  jobs/job_000000_seed1001/
  jobs/job_000001_seed1002/
  ...
```

Each job directory contains its resolved POWHEG card, LHE file, showered HepMC
file, generator logs, metadata, and either `SUCCESS` or `FAILED`. The campaign
configuration and card are made read-only after submission to preserve the
resolved inputs.

## Monitoring

```bash
squeue --me
sacct -j ARRAY_JOB_ID
tail -F /work/pi_rclsa_umass_edu/FourLeptonUnfoldingGeneration/CAMPAIGN/logs/events-ARRAYID_TASKID.out
```

Use `scancel JOB_ID` to cancel a grid job or an array. A grid failure prevents
the dependent event arrays from starting; similarly, a failed array wave keeps
later waves from running. Inspect the corresponding Slurm log and the
per-directory `slurm-status.txt` before resubmitting.

Unity references used for this setup:

- [batch jobs](https://unityhpc.org/documentation/jobs/sbatch/)
- [array jobs](https://unityhpc.org/documentation/jobs/sbatch/arrays/)
- [large job counts](https://unityhpc.org/documentation/jobs/sbatch/large-count/)
- [storage and mount points](https://unityhpc.org/documentation/cluster_specs/storage/)
