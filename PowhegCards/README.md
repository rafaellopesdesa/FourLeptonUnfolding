# POWHEG Run-3 cards

These are the default `powheg.input` cards used by
`Generation/run_generation.sh`:

- `gg_H.powheg.input`: inclusive gluon-fusion Higgs production;
- `ZZ.powheg.input`: off-shell `ZZ/Zgamma*/gamma*gamma* -> 4l` production,
  with `l = e, mu, tau`.

Both cards use 6800 GeV per proton beam (`sqrt(s) = 13.6 TeV`) and the central
member of `NNPDF31_nlo_as_0118`. The runner replaces the event count, random
seed, PDF member IDs and grid-reuse settings in its private copy of the card.

No Born-level generation cut is needed for inclusive `gg_H`, whose Born final
state is a massive Higgs boson. For continuum four-lepton production, however,
a literally cut-free cross section is not finite: the matrix element includes
off-shell photons and has a low-mass `gamma* -> l+l-` singularity when leptons
are treated as massless. The original setup with only `m(l+l-) > 4 GeV`
proved impractically inefficient: POWHEG spent excessive time sampling low
four-lepton masses and repeatedly violated the inclusive upper bound. The
final `ZZ` card retains `m(l+l-) > 4 GeV` but adds `m(4l) > 95 GeV`. The low dilepton
threshold is necessary because the fiducial selection only requires every
SFOS pairing to exceed 5 GeV (and the selected sub-leading pair to exceed
12 GeV). The four-lepton threshold is provided by the small
version-pinned source patch installed from
`Generation/patches/powheg-zz-m4lmin.patch` because the upstream process has no
native four-lepton-mass input keyword.

POWHEG applies `mllmin` to crossed opposite-sign pairings in same-flavor `4e`
and `4mu` events, so the earlier 20 GeV value could remove events that satisfy
the intended fiducial definition. The 4 GeV cut leaves a 1 GeV buffer below
the all-SFOS-pair veto, while `m(4l)>95 GeV` leaves 10 GeV below the analysis
region. No generator-level lepton-pT or eta cut is imposed. The generation
cuts remain part of the generated background phase-space definition.

All grids made before the `m4l > 95 GeV` requirement—including both the
original 4 GeV card and the temporary 20 GeV card—are incompatible with this
phase-space definition. Generate a fresh grid/new campaign; do not pass an old
grid through `--keep-grids` or `--reuse-grid`.

The `gg_H` LHE weights correspond to inclusive Higgs production. Since the
shower decay is forced to `H -> ZZ(*) -> 4l`, physical yield predictions must
apply the appropriate Higgs and Z branching fractions; they are not inserted
automatically into the LHE weights. They cancel in an acceptance computed
within the forced decay channel.
