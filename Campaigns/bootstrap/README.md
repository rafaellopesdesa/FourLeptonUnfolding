# Unity bootstrap and pilot

Run inputs belong to `codex_bootstrap_20261113`, not main. All active I/O is
inside the authorized /work project. Compact validated samples and trained
artifacts are later copied with checksums to the corresponding /project.

The install jobs build the generator stack and patched Delphes, and install
the locked Analysis and BackgroundRemoval Python environments. The validation
job uses the dedicated `codex_validation` worktree. Dependency builds and
tests run through Slurm; no training or simulation is run on a login node.

`launch_pilot.sbatch` is submitted with an `afterok` dependency on successful
generator and clean-environment Delphes setup. Its initial phase submits one
fresh POWHEG integration and one 1,000-event Pythia nominal shard per process.
After each grid succeeds, a dependent launcher submits Pythia up/down and
Herwig nominal using that grid. Eight sample configurations result.

Resource requests: setup 1 CPU / 2 GB / 30 min; each integration or event job
1 CPU / 4 GB, at most 48 h for integration and 6 h for 1,000 full-chain events.
These are initial limits, not measured consumption. Check `sacct`, selections,
signed yields, Born coverage and bytes/event before sizing the 5M-reco campaign.
There is no automatic large-production escalation.

`retry_pilot.sbatch PROCESS` preserves the previous per-shard diagnostics,
then retries the four configurations with their existing cards, seeds and
validated grid. Use it only after addressing the recorded failure. Summary
JSON files retain the Slurm job identifier so failed attempts are not erased.

`validate_physics.sbatch [COMPONENT ...]` runs the Analysis, Tools, Simulation,
Generation, BatchSubmit, Plotting and SMEFT regression suites by default.
Native shared-library test builds are removed after Python exits because
removing an open library on NFS leaves temporary `.nfs` files.

Nominal/up/down Pythia request the same hard-process seed. Actual retained
LHE identities must establish correlation; identical seeds alone do not
guarantee matched detector random decisions. Herwig uses a separate seed.

The confirmed deformation is C_HB = +0.1 at Lambda = 1 TeV, with 0 and −0.1
controls. It is not applied in this undecorated pilot. Positive data-like
sampling variance must be validated before the pseudo-experiment ensemble.

After all signal shards validate, `check_signal_pilot.sbatch` runs the existing
MC merger separately for each configuration and checks physical normalization,
shared Pythia hard-event identities, and the native −0.1/0/+0.1 coefficient
scan. Tau-origin rows use an explicitly recorded unity policy for this
diagnostic. These weighted templates are not pseudo-experiments.

`launch_signal_10k.sbatch` is a second, bounded timing check: one fresh
10,000-event shard per signal configuration, using the validated grid and
new seeds. Each worker requests 1 CPU / 2 GB / 1 h, informed by the successful
1,000-event jobs (93–96 seconds; peak RSS 650–790 MiB). Its dependent summary
validates and persists the compact outputs. This does not launch the 5M-reco
production campaign.

`check_training_inputs.sbatch` exercises systematic and OmniFold readers on
the normalized 1k pilot in `runs/pilot/diagnostics/1634795`. It checks all
four configurations in the signal/extended regions, with/without fiducial
selection, plus full fiducial MC/truth input. It does not train a network.

`launch_signal_wave1.sbatch` is submitted after successful 10k pilot summary.
It independently checks all four persisted checksums and selected counts,
and requires 100 GB of /work headroom below the 300 GB soft target plus
10 GB of /project headroom below 2 TB. It launches exactly 100 × 10,000 events
per signal configuration, four concurrent workers per array (16 total),
with fresh seeds, one CPU / 3 GB / one hour per worker. The 3 GB request
provides headroom above the approximately 2 GB recorded by the 10k pilots.
The dependent summary validates every shard, persists compact files, records
actual selected totals and reports missing/failed shards. No subsequent
production wave is launched automatically. One million generated events is
not the five-million-selected-event target.
