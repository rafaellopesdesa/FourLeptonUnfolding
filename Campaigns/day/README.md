# Continuous production on 14 November 2026

The user requested production throughout the day. This controller submits
successive bounded waves, each 200 × 10,000 showered events for eight
configurations. Four arrays per physics process have a 25-task cap each:
at most 100 gg_H plus 100 ZZ workers. Each requests one CPU, 4 GiB, one hour
and the `localscratch` feature. Worker intermediates remain on local disk,
with the existing 8 GB per-task / 20 GB free-space guards and cleanup checks.

Only one wave is active at a time. No new wave is submitted at or after
2026-11-15 00:00 UTC, and there can be no more than eight waves (128M
showered rows). Already submitted tasks finish. A `STOP` file in the day
directory prevents the next wave; it does not discard running work.
The clock is UTC, as used in the project status files.

The first launch depends on successful repaired-wave validation and archival.
The fixed detector is a separate installation; the runtime and controls are
snapshots with recorded checksums. Each wave receives fresh non-overlapping
seed ranges, with correlated nominal/up/down Pythia hard-event seeds.
Control code and scalar campaign inputs are made read-only before submission.
All controller/summary processes explicitly limit BLAS/OpenMP threads to one.

Every launch inventories storage and requires 50 GB headroom below the 300 GB
/work target and 10 GB below the 2 TB /project target. Array and dependent
job IDs are saved while held, then released together. Partial submission
cancels only those held jobs. After all arrays finish, the controller checks
worker status, output/config hashes and local cleanup. Recognized finite-LHE
or HepMC-integrity failures get at most one same-seed retry, up to 32 tasks.
Unknown failures, failed guards or a second failed attempt stop advancement.

Two process summaries validate and persist compact shards. Their storage
accounting inventories /project once each and counts new copies, avoiding
a full directory walk for every shard. The preceding 10 GB reservation
comfortably covers the approximately 3 GB combined wave. No training or other
large writes are scheduled concurrently in this project. Both summaries must
pass before the controller verifies all saved hashes and removes duplicate
/work shards. Only then can another wave launch. Partial waves retain all
successful compact output and do not authorize another wave.

A small Slurm monitor runs every 30 minutes and records active worker counts
and SUCCESS markers. It updates `/docs/CURRENT_STATUS.md`, the local reveal.js
slide, and the report evidence JSON; GitHub remains a separately published
snapshot. Monitoring stops when the run finishes/stops, or at 04:00 UTC on
15 November. All job records remain under `runs/production_day_20261114` and
the individual `runs/production_day_20261114_wNN` directories.

The eight checks in `tests/test_day.py` cover caps, dependency ordering,
record-before-release, cancellation after partial submission, deadlines,
fresh/correlated seeds, prerequisite validation, and unknown-failure stops.
Run them with `validate.sbatch` on Unity. The code and tests belong on the
campaign `codex_` branch, never main.

The actual frozen configuration and initial scheduler receipt are mirrored in
`production_20261114.json` and `submission_20261114.json`. The live versions
and subsequent wave/job records remain under the run directory above.
