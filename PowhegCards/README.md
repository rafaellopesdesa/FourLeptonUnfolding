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
are treated as massless. The original `m(l+l-) > 4 GeV` setup proved
impractically inefficient: POWHEG spent excessive time sampling the
virtual-photon region and repeatedly violated the inclusive upper bound. The
`ZZ` card now uses `m(l+l-) > 20 GeV`, matching the value in the upstream
reference card, and `m(4l) > 95 GeV`. The latter is provided by the small
version-pinned source patch installed from
`Generation/patches/powheg-zz-m4lmin.patch` because the upstream process has no
native four-lepton-mass input keyword.

These cuts retain margins below the intended `m(l+l-) >= 50 GeV` and
`m(4l) >= 105 GeV` analysis region. The dilepton margin is intentionally wider
because POWHEG applies `mllmin` to crossed opposite-sign pairings in
same-flavor `4e` and `4mu` events. No generator-level lepton-pT or eta cut is
imposed; all such requirements remain analysis-level choices. The generation
cuts are part of the generated background phase-space definition and must be
included when computing acceptance.

Integration grids made with the earlier 4 GeV card are incompatible with this
phase-space definition. Generate a fresh grid/new campaign; do not pass an old
grid through `--keep-grids` or `--reuse-grid`.

The `gg_H` LHE weights correspond to inclusive Higgs production. Since the
shower decay is forced to `H -> ZZ(*) -> 4l`, physical yield predictions must
apply the appropriate Higgs and Z branching fractions; they are not inserted
automatically into the LHE weights. They cancel in an acceptance computed
within the forced decay channel.
