# Application

Application writes two compact branches while preserving the input tree,
entry order, original signed weights, fixed-size arrays, and string metadata:

| Branch | Type | Meaning |
|---|---:|---|
| `analysis_region` | `uint8` | 1 exactly for `reconstructed && 115 < reco_m_ZZ < 130` GeV; 0 otherwise |
| `background_removal_weight` | `float32` | Factor prescribed below; multiply it into the preserved nominal `weight` |

The mass boundaries are excluded. `reco_m_ZZ` is not a model input. The
complete recipe is:

| Sample kind | Event condition | `analysis_region` | `background_removal_weight` |
|---|---|---:|---:|
| `data` | reconstructed and `115 < reco_m_ZZ < 130` | 1 | bounded purity from Training |
| `data` | otherwise | 0 | 1 |
| `gg-h` | reconstructed and `115 < reco_m_ZZ < 130` | 1 | 1 |
| `gg-h` | otherwise | 0 | 1 |
| `zz` | reconstructed and `115 < reco_m_ZZ < 130` | 1 | full data/MC factor from Correction |
| `zz` | reconstructed and outside that window | 0 | full data/MC factor from Correction |
| `zz` | not reconstructed | 0 | 1 |

Only rows that require a prediction are passed to a network. Consequently,
non-reconstructed rows may contain nonfinite reconstructed variables. The
application verifies that the Training artifact records the exact Correction
manifest supplied on the command line. It also verifies the standard
`weight`/`weight_nominal_pb` measure pairing, or requires the Training artifact
to contain the explicit custom-measure override used when that pair was
created.

## Complete merged directory

This is the recommended interface. Run it from the repository root:

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml apply-directory \
  --merged-dir /path/to/merged \
  --output-dir /path/to/decorated \
  --background-model-dir /path/to/models/background \
  --correction-model-dir /path/to/models/correction
```

It loads the artifact pair once and applies the role-specific rule shown
above: the frozen background model to `data.root` and every selected numbered
pseudo-experiment, the frozen Correction to `ZZ_pythia.root`, and unity to
`gg_H_pythia.root`. Numbered pseudo-experiments never receive separately
trained background or Correction models.

If a format-v2 `pseudo_data_manifest.json` is present, it is authoritative:
the command validates its contiguous indices and exact file names, uses that
file list, and copies the manifest unchanged to the decorated directory. If
it is absent for a legacy campaign, the command falls back to numerically
sorted names such as `data_0001.root`; indices are padded to at least four
digits, so `data_10000.root` is also valid. Herwig component files and
similarly named lookalikes are neither copied nor modified.

Run this command only after the merger has finished and while the input
directory is quiescent. Source files are snapshotted and rechecked, but this is
not a lock against concurrent rewrites.

When the artifacts retain their default input checksums, nominal data and both
Pythia files are SHA-256 checked against the recorded training inputs. A
deliberate cross-campaign application can use `--allow-input-mismatch`;
mismatches are still recorded in the campaign manifest, and all model-pair,
schema, mass-window, and luminosity checks remain mandatory.

The input and output directories must differ and the output must not overlap
the input or either model directory. The complete campaign is staged before
it replaces a prior managed output, so inference and file-processing failures
leave the old campaign intact; publication also attempts rollback if its
final rename fails. A successful `--overwrite` removes stale managed
pseudo-output files. The output directory includes
`background_removal_application_manifest.json`, which records both artifact
checksums, matching source and copied upstream-manifest checksums, all run
options, input match results, and per-file source/output hashes and counts.

Every data file is evaluated with the same frozen background-removal model,
which was trained using one frozen nominal-data Correction. Thus the spread
among numbered outputs is conditional on that pair; it does not propagate
either model's refit uncertainty and remains conditional on the shared finite
Herwig template.

Correction was learned only in `130 < reco_m_ZZ < 160` GeV but is evaluated
on all reconstructed `ZZ` rows, including the strict
`115 < reco_m_ZZ < 130` GeV analysis region. Since `reco_m_ZZ` is not a model
input, this is an explicit mass-transfer assumption rather than an
interpolation learned by the network.

## One file

Run the single-file interface from the repository root as well:

```bash
pixi run --locked --manifest-path BackgroundRemoval/pixi.toml apply \
  --input /path/to/merged/data.root \
  --output /path/to/decorated/data.root \
  --sample-kind data \
  --background-model-dir /path/to/models/background \
  --correction-model-dir /path/to/models/correction
```

Use `--sample-kind gg-h` for `gg_H_pythia.root` and `--sample-kind zz` for
`ZZ_pythia.root`.

Inference is chunked and each output file is written atomically. Existing
outputs are rejected unless `--overwrite` is supplied. Decorating a tree that
already has any primary or diagnostic output branch additionally requires
`--replace-existing-branches`. Replacement removes all prior output branches;
diagnostic branches are written back only when `--write-diagnostic-branches`
is also supplied.

The embedded `background_removal_metadata` object records the pre-decoration
source path, size, modification time, and SHA-256. This also preserves useful
provenance when the explicitly requested `--overwrite` operation is in-place.

Add `--write-diagnostic-branches` only for event-level debugging. It writes
scores, ratios, ensemble spread, and the convenience branch
`weight_background_removed = weight * background_removal_weight`; the
original signed `weight` is always retained unchanged.
