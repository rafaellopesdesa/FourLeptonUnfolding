# Signal pilot: measured results and attempt history

Snapshot: 2026-11-13, after successful summary 1634275 and native scan 1634795.
Each configuration contains 1,000 generated events. This is a software and
production-sizing pilot, not an unfolding closure study.

| Configuration | Selected reco | Reco 115–130 GeV | Fiducial | Seconds | Peak RSS (MiB) | Compact bytes |
|---|---:|---:|---:|---:|---:|---:|
| Pythia nominal | 172 | 167 | 234 | 93 | 790 | 255194 |
| Pythia scale up | 172 | 167 | 234 | 94 | 789 | 255181 |
| Pythia scale down | 170 | 167 | 234 | 93 | 789 | 255104 |
| Herwig nominal | 140 | 134 | 201 | 96 | 650 | 253074 |

Selected counts refer to the extended analysis region; the narrower Higgs
window is shown separately. Correlated Pythia variants use identical hard
events; Herwig uses an independent seed. All selected rows have valid
certified Born kinematics. The small independent shower samples do not
establish the size/significance of an acceptance difference.

The complete generation → shower → detector → analysis chain ran on one
CPU per task. Slurm reports 80.5/84.2 CPU seconds for nominal Pythia/Herwig.
Peak RSS uses Slurm's reported KiB divided by 1024. Intermediate LHE/HepMC
and Delphes files were removed after compact-output validation.

The compact files and SHA256 values are in
[the full-chain summary](evidence/gg_H-summary-1634275.json). Persisted samples
are under `/project/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/samples/pilot_20261113/`.
Active I/O remains on /work. The normalized/decorated diagnostic files are
also persisted under `diagnostics/1634795/` within that pilot directory.

## Normalization and c_HB controls

The existing merger was used separately for each sample. Its checks give
sum(weight_shape)=N and sum(weight)=sigma × 312000 pb^-1. Raw Pythia and
Herwig weights have different conventions; comparing their unmerged sums
would be invalid. Normalized inclusive yields are 2931.56 and 2937.44 in
this small pilot; corresponding selected yields are 505.24 and 411.24.

The native scan uses c_HB = −0.1, 0, +0.1 at Lambda=1 TeV, retaining the
repository's squared dimension-six convention and fixed widths. All ratios
are finite and nonnegative; the c_HB=0 ratio is exactly one. At +0.1,
selected Herwig ratios range from 0.70765 to 2.42841, with median 0.99713.
The selected weighted yield becomes 410.56. These integrals fluctuate with
the finite template and are not coefficient estimates.

Tau-origin rows use the explicit `unsupported='unity'` diagnostic policy:

| Sample | Successfully evaluated | Unsupported tau, inclusive | Unsupported tau, selected |
|---|---:|---:|---:|
| Pythia nominal | 471/1000 | 529/1000 | 1/172 |
| Herwig nominal | 443/1000 | 557/1000 | 1/140 |

No unsupported row is marked as a successful matrix-element evaluation.
The final closure definition must specify this component's treatment.
See [native scan evidence](evidence/signal-checks-1634795.json) for status
counts, quantiles, signed yields and file hashes.

Pythia has one negative-weight event inclusively and none selected; Herwig
has none in these 1,000 rows. The selected signed-Poisson variance/net ratio
is one for this pilot. Larger samples can reveal negative selected weights;
this result does not validate positive data-like sampling at production scale.
No pseudo-data have been drawn or deformed after sampling.

## Attempt history

| Jobs | Outcome | Resolution |
|---|---|---|
| 1631376, 1631669, 1631672, 1631674 | Rejected LHE after generation/simulation | Accept only the standard trailing POWHEG RNG footer; preserve bytes/hash |
| 1633039–1633042 | Analysis tree produced; standalone validation import failed | Add repository import bootstrap and subprocess regression |
| 1634269, 1634272, 1634273, 1634274 | All four complete | Compact validation and persistence pass in 1634275 |
| 1634795 | Normalization, identity and native HB checks pass | Diagnostic ROOT files/checksums persisted |

Earlier shard diagnostics were copied into `runs/pilot/retries/<launcher-job>/`
before retry. Summary history includes Slurm IDs, so failures are not erased.
The reusable fixes are on main; tests and run inputs are on codex branches.

## Larger pilot and first production wave

All four 10,000-event pilots completed and passed validation/persistence in
summary 1635020. Their compact files are under the corresponding /project
`samples/pilot10k_20261113/` directory. Full evidence:
[10k summary](evidence/gg_H-summary-1635020.json).

| Configuration | Selected reco | Reco 115–130 GeV | Fiducial | Walltime | Compact bytes |
|---|---:|---:|---:|---:|---:|
| Pythia nominal | 1528 | 1487 | 2138 | 11 min 44 s | 1789348 |
| Pythia scale up | 1544 | 1494 | 2138 | 10 min 49 s | 1789444 |
| Pythia scale down | 1526 | 1480 | 2138 | 11 min | 1789425 |
| Herwig nominal | 1477 | 1425 | 1971 | 9 min 40 s | 1756573 |

Slurm's recorded peak RSS is approximately 2 GiB. All selected Born rows
are valid. Selected tau-origin counts are 2/3/1/2 respectively. Pythia has
two negative-weight rows inclusively and Herwig five, but none are selected;
the selected signed-Poisson variance/net ratio remains one in this pilot.
The positivity caveat for production therefore remains unresolved.

The measured acceptance now suggests about 33–34 million generated events
per signal configuration to reach five million selected events. Roughly
550–650 CPU hours per configuration is a naive throughput extrapolation,
before retries, grid/setup cost and production variability. Compact signal
storage is roughly 6 GB per configuration before merging/diagnostic copies.
Actual validated counts determine completion, not this estimate.

Launcher 1636246 completed after successful 10k summary and independently verified
persisted hashes/selected counts and storage headroom. It submitted exactly
100 × 10,000 events for each of the four signal configurations, with at most
four concurrent workers per array (16 total). Each requests one CPU, 3 GB,
one hour; 100 GB of /work and 10 GB of /project headroom are required below
the soft targets. A dependent job validates/persists every compact shard and
appends selected totals to the working status. No further wave is automatic.
Arrays 1636684 (nominal), 1636696 (up), 1636697 (down), and 1636700 (Herwig)
are running; summary 1636701 waits for all four. This first wave has one
million generated events per configuration and does
not satisfy the five-million-selected target by itself.

## Background pilot

ZZ integration 1631379 remains active, with CPU use verified; dependent
jobs submit its four configurations. Its acceptance, signs and timing have
not yet been measured. This background work is independent of the validated
signal production wave.
