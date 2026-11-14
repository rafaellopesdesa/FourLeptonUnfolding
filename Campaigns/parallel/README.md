# Bounded production with node-local intermediates

The user authorized up to 100 simultaneous jobs **per physics process**.
This wave uses four configurations per process and 25 workers per configuration:
100 gg_H plus 100 ZZ, at most 200 production workers in total. Each task requests
one CPU, 4 GiB memory and one hour on `cpu`, with `--constraint=localscratch`.
Slurm decides how many run at once. No GPUs are requested.

The wave contains 200 × 10,000 showered events per configuration: 16 million
rows across eight configurations. Fresh seed ranges begin at 50/60 million
for gg_H Pythia/Herwig and 70/80 million for ZZ. The three Pythia detector
settings share hard-event seeds. This is an installment toward the required
five million selected reconstructed events per sample.

Unity's [node features](https://docs.unity.rc.umass.edu/documentation/cluster_specs/features/)
identify `/localscratch` as local NVMe storage. The live probe 1678602 found
a writable ext4 local mount on cpu045 with about 1.12 TB available. `/tmp`
was also job-private local storage in the inspected allocations; a shared
`/scratch` mount is distinct and requires an allocated HPC Workspace.
This campaign explicitly requests the `localscratch` feature and checks the
mount type at runtime. It creates only disposable task directories there.
Code, controls, diagnostics and compact active outputs stay in the project's
/work directory; validated samples are copied to /project.

Storage inventory 1678427 measured 26.28 GB in /work and 0.732 GB in /project.
Retained 10k signal attempts use 3.54–3.55 GB of intermediate files each;
200 such tasks would exceed the /work limit. The node-local wrapper samples
temporary size every ten seconds, requires 60 GB initial free space, and
terminates a task if its workspace exceeds 8 GB or local free space drops
below 20 GB. These are observed filesystem checks, not Slurm disk reservations
(`TmpDisk=0` on inspected nodes). Unrelated jobs can change free space.

Submit `prepare.sbatch` only once for this wave. It checks storage headroom,
verifies all prerequisite persisted sample hashes, copies immutable control
files, and submits eight 10k local-storage pilots. The gate verifies their
output checksums, successful cleanup and sampled peak below 5 GB before
submitting the remaining 1,592 tasks. It repeats the project storage checks,
reserving 50 GB below the 300 GB /work target and 10 GB below the 2 TB /project
target. New stages are held until their job identifiers and dependencies
are recorded, and a partial submission cancels only its own held jobs.

The unchanged physics runtime is snapshot e07de718, matching main tree
20efbbe8. Herwig supplies 1% extra hard events while preserving the requested
shower count. All provenance guards remain enabled. New production controls
are copied under the wave's `control/` directory and made read-only before use;
never replace files underneath active workers.

The finish job permits one same-seed retry for recognized finite-LHE or
HepMC-integrity failures, at most 32 failed tasks. Unknown failures stop for
investigation. The final summary validates/persists both processes and records
actual selected totals. No next wave is automatic. `archive_work_copies.py`
is submitted after successful final summary: it verifies every persistent
checksum before removing any redundant compact /work copy. It retains all
diagnostics, controls and summary metadata. Failed waves are never archived.

Campaign checks live here on the `codex_bootstrap_20261113` branch, never main.
They cover gates, caps, dependency order, partial-submission rollback, failure
classification, and checksum-protected duplicate cleanup. `validate.sbatch`
runs them through Slurm. Physics-input validation is the separate eight-pilot
gate; these software checks do not establish unfolding closure.

## Calorimeter failure diagnosis (14 November)

The bulk arrays completed 1,598/1,600 tasks. Two signal seeds crashed in
Delphes 3.5.1 `SimpleCalorimeter::Process`: its particle-hit loop indexed a
track-fraction vector. The upstream correction is commit
`7f2b717c96671e78b6724b27aceea2580c9ce73a`.

`calorimeter_build.sbatch` builds that backport in a separate installation.
`calorimeter_diagnosis.py` submits the two original failure seeds and eight
successful controls, retaining bounded inputs under /work. It changes only
diagnostic storage settings and leaves the production runtime untouched.
`calorimeter_compare.py` reuses each identical HepMC input with the fixed
detector, validates the resulting compact tree, and compares every analysis
branch against the original when it completed. A separate executable
regression on `codex_validation` checks a neutral-only event with zero tracks:
the original crashes and the fixed version conserves the input energy.

`calorimeter_recover.py` may run only after all eleven checks succeed. It also
verifies that all ten diagnostic LHE/HepMC hashes equal their original
production hashes, and checks the fixed binary/environment checksums and
storage headroom. It archives both failed task records, submits exactly two
same-seed repairs with explicit fixed-runtime configs, then a full summary
and checksum-protected duplicate cleanup. It never substitutes seeds,
silently drops events, or launches the next bulk wave. Record job IDs and
dependencies in the diagnostic run before releasing work.
