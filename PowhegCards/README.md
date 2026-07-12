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
are treated as massless. The `ZZ` card therefore imposes only
`m(l+l-) > 4 GeV`. This technical phase-space definition must be retained when
computing the total-to-fiducial acceptance. All later lepton-pT, eta and Z-mass
requirements should be applied at analysis level.

The `gg_H` LHE weights correspond to inclusive Higgs production. Since the
shower decay is forced to `H -> ZZ(*) -> 4l`, physical yield predictions must
apply the appropriate Higgs and Z branching fractions; they are not inserted
automatically into the LHE weights. They cancel in an acceptance computed
within the forced decay channel.
