# LO Higgs-decay deformation with O_HB

This directory adds an event-by-event matrix-element ratio for the CP-even
Warsaw-basis operator

$$
\mathcal L = \mathcal L_{\mathrm{SM}} +
\frac{C_{HB}}{\Lambda^2}(H^\dagger H)B_{\mu\nu}B^{\mu\nu}.
$$

There is no extra gauge-coupling factor in this operator definition.
`C_HB` is dimensionless; `--lambda-tev` supplies the scale in TeV. All other
Wilson coefficients are zero. The tool reads the compact ntuple's certified
`born_*` variables and writes additional weights. It preserves the original
event rows, nominal weights, and signed-weight convention.

The intended use is a controlled LO deformation of the current showered
POWHEG signal, with the QCD production distribution unchanged. It does not
turn that sample into a complete NLO SMEFT prediction.

## Matrix element and coefficient convention

The native backend is generated with **MadGraph5_aMC@NLO 3.5.17 LTS** and
**SMEFTsim 3.0.2**, using `SMEFTsim_U35_MwScheme_UFO` restricted to `cHB`.
The electroweak input scheme fixes `mW`, `mZ`, and `GF`. Propagator-shift
insertions are disabled, and widths are held fixed at their reference
values. `SMHLOOP=0`: loop-induced SM Higgs vertices are excluded from this
tree-level definition.

The generated decay matrix elements include the `HZZ`, `HZgamma`, and
`Hgammagamma` contributions allowed by that model. For identical final
leptons, all exchange diagrams and their interference are included. A
selected Z1/Z2 pairing supplies coordinates; it does not remove crossed
diagrams from the amplitude.

At amplitude level the restriction is `NP <= 1`. The stored ratio includes
the square of the dimension-six amplitude:

$$
r(x) = \frac{|\mathcal M_{\mathrm{SM}} + x\mathcal M_{HB}|^2}
{|\mathcal M_{\mathrm{SM}}|^2}
= 1 + x\,a + x^2 b,
\qquad
x = \frac{C_{HB}}{(\Lambda/\mathrm{TeV})^2}.
$$

The reference amplitude `M_HB` uses a coefficient of one at 1 TeV. Thus
`a` contains SM–dimension-six interference and `b` contains dimension-six
squared. This is not a prediction complete through order `1/Lambda^4`:
dimension-eight interference is absent. No Higgs total-width or branching
fraction rescaling is applied. The coefficient-zero ratio is exactly one.
Coefficient values in the examples below illustrate a scan; they are not
phenomenological bounds or a statement of EFT validity.

## Input scope

Supported signal channels are direct, massless `4e`, `4mu`, and `2e2mu`
decays, including both mixed-flavor Z1/Z2 orderings. The compact masses
`born_m_ZZ`, `born_m_Z1`, `born_m_Z2`, angles `born_cos_theta1`,
`born_cos_theta2`, `born_Phi`, and `born_type` determine a canonical
four-lepton rest-frame configuration. For a scalar Higgs decay, an overall
rotation or boost does not change the squared amplitude, so the production
angles, system rapidity, and pT are unnecessary for this decay-only ratio.
The reconstruction preserves the invariants needed for identical-lepton
exchange interference within this massless model. The native Born record
can contain the physical electron/muon masses: retaining its pair masses
and decay angles while setting the daughter masses to zero is an
approximation, particularly near a dilepton threshold.

Inputs must carry the Born source metadata from the current Analysis chain.
Certified signal rows have `born_source=2` or `3`, identifying PYTHIA or
HERWIG hard-decay input respectively. Certified continuum `ZZ` rows with
`born_source=1` are passed through with a unity ratio; this module does not
apply an `O_HB` deformation to the background. Process identification comes
from provenance, not from truth/reco channel labels or filenames alone.

Tau channels are unsupported by this massless direct-e/mu backend. A tau
that later produces an electron or muon is still a tau in the hard Born
record and is not relabeled as a direct supported signal decay. The default
policy is to raise on unsupported target rows. `--unsupported unity`
explicitly leaves them unchanged and records their status; inspect this
fraction when comparing targets. Invalid or unavailable Born kinematics
must likewise never be interpreted as a successful matrix-element weight.

The current PYTHIA and HERWIG baselines decay assigned Z pairs and omit the
identical-lepton exchange interference included in the full `4e`/`4mu`
denominator matrix element. Their decay density therefore generally differs
from that denominator. The weighted target is
`q_deformed = q_generated * r` before normalization; it is a controlled LO
matrix-element deformation, not exact SMEFT event generation. An exact
density change would require a matching SM baseline or a separate correction
from the generated density to it. Exact unity at `C_HB=0` checks the ratio's
normalization convention, not equivalence of those SM decay models.

## Run on Unity or locally

Use the existing Analysis Python environment and a C++17 compiler. The
generated matrix-element source is vendored in this directory. Running the
reweighter does not require a MadGraph installation, Fortran, or a model
download. A separate Python environment can instead install
`python -m pip install -r SMEFT/requirements.txt`. First use compiles it with
`CXX` (default `c++`) and caches the
result under `SMEFT/.build`. The source hash, compiler command/flags, and
library hash are checked before a cached build is reused. To build before
an interactive session or choose another cache location:

```bash
pixi run --manifest-path Analysis/pixi.toml python SMEFT/build_backend.py \
  --build-dir /path/to/smeft_native_cache
```

`--force` rebuilds the native library. Python callers can select that location
with `NativeHBBackend(build_dir=...)`; `reweight.py --build-dir PATH` selects
the same cache for command-line use. The default first-use path needs no
separate build command. If `c++` is unavailable on Unity, load an available
GCC module or set `CXX` to the compiler executable before running.

From the repository root, a coefficient scan on a merged HERWIG signal is:

```bash
pixi run --manifest-path Analysis/pixi.toml python SMEFT/reweight.py \
  /path/to/merged/gg_H_herwig.root \
  --output-dir /path/to/smeft_scan \
  --cHB -0.1 0 0.1 \
  --lambda-tev 1 \
  --unsupported unity
```

The explicit unity policy is useful for the baseline generation, which
includes tau decays. Omit it to require every signal row to be supported.
The same input selection accepts files, shell-style globs, and directories.
For fresh mixed signal-plus-background pseudo-data files:

```bash
pixi run --manifest-path Analysis/pixi.toml python SMEFT/reweight.py \
  '/path/to/merged/data*.root' \
  --output-dir /path/to/smeft_pseudo_data \
  --cHB -0.1 0 0.1 \
  --lambda-tev 1 \
  --unsupported unity \
  --weight-branch weight
```

Use `-o /path/to/decorated.root` for one input if an explicit output filename
is preferred. `--weight-branch` selects the scalar nominal weight multiplied
by the ratios; its default is `weight`. Signed nominal weights retain their
sign under the squared-amplitude ratio.
Directories select their immediate `*.root` files. Existing output files
require `--overwrite`, and outputs may never alias inputs. An already
SMEFT-decorated input is rejected to prevent accidental repeated decoration.

## Output and downstream use

For `N` input rows and `K` requested coefficients, the added branches include
`smeft_hb_ratio` and `smeft_hb_weight`, each with shape `(N, K)`. Coefficient
order follows `--cHB`. The first stores the decay ratio and the second the
selected nominal weight multiplied by that ratio. Per-event linear and
quadratic coefficients, validity, and status branches accompany the scan:

| Branch | Meaning |
|---|---|
| `smeft_hb_ratio` | `K` ratios in requested coefficient order |
| `smeft_hb_weight` | `K` signed nominal weights multiplied by those ratios |
| `smeft_hb_linear`, `smeft_hb_quadratic` | The per-event `a` and `b` in the polynomial above; zero for unity pass-through rows |
| `smeft_hb_valid` | True for evaluated signal and certified unchanged `ZZ`; false for unsupported signal retained at unity |
| `smeft_hb_status` | Explicit evaluation or pass-through reason |

Status codes are 0=`evaluated`, 1=`background_unity`,
2=`unavailable_born`, 3=`tau_unsupported`, 4=`unsupported_channel`, and
5=`invalid_born_kinematics`. The ROOT string object `smeft_hb_metadata`
records the coefficient values and scale, formulas, status counts,
per-coefficient weight sums, backend/source/build identity, and input SHA-256.
Existing LHE/Born provenance is retained. Missing or inconsistent provenance
fails even with `--unsupported unity`; that option is not permission to
guess a process or event identity.

For example, inspect a completed scan in a notebook:

```python
import json
import uproot

with uproot.open("/path/to/smeft_scan/gg_H_herwig.root") as source:
    metadata = json.loads(str(source["smeft_hb_metadata"]))
    columns = source["Analysis"].arrays(
        ["smeft_hb_weight", "smeft_hb_valid", "smeft_hb_status"],
        library="np",
    )

point = metadata["coefficients"].index(0.1)
target_weights = columns["smeft_hb_weight"][:, point]
print(metadata["status_counts"])
```

The original `weight` is not replaced. Select the intended coefficient
column explicitly when plotting or supplying target weights downstream;
adding these branches alone does not change an existing training run.
Use fresh HERWIG signal or mixed pseudo-data before background removal,
then rerun background-removal training for the selected deformed target.
Existing subtraction weights were fitted to the old target and should not
silently be carried over as if that fit had been repeated.

Decoration does not regenerate pseudo-experiment counts. Reweighted
pseudo-data provide a weighted closure target. To obtain new Poisson
pseudo-experiments at a changed signal intensity, the resampling step must
use the deformed signal template and its new yield. The ratio is not
automatically normalized to preserve the old signal yield; its integral can
change as well as its shape.

## Validation

From the repository root:

```bash
python -m unittest discover -s SMEFT/tests -v
```

These tests compile and call the actual native amplitudes. They compare
mixed-flavor amplitudes with independent Dirac traces, test arbitrary
coefficient values against the quadratic representation, and check Lorentz
invariance, particle permutations, identical-flavor channels, and helicity
call-order independence. ROOT/CLI tests cover signed weights, mixed sample
identities, unsupported channels, metadata retention, and atomic output.
The massless momentum reconstruction is also checked against the existing
analysis angle conventions.

## Primary references

The exact generator/model commits, generation commands, restricted cards,
upstream hashes, and local source modifications are recorded in
[`generated/provenance.json`](generated/provenance.json). The frozen input
card is [`cards/param_card.dat`](cards/param_card.dat). Upstream license
notices are retained in [`licenses/`](licenses/).

- [SMEFTsim source and versioned releases](https://github.com/SMEFTsim/SMEFTsim),
  including the `U35` flavor assumption and `MwScheme` electroweak inputs.
- [SMEFTsim 3.0 — a practical guide](https://arxiv.org/abs/2012.11343),
  I. Brivio, JHEP 04 (2021) 073.
- [MadGraph5_aMC@NLO source](https://github.com/mg5amcnlo/mg5amcnlo) and
  [official release downloads](https://launchpad.net/mg5amcnlo).
- [The automated computation of tree-level and next-to-leading order
  differential cross sections, and their matching to parton shower
  simulations](https://arxiv.org/abs/1405.0301), J. Alwall et al.
