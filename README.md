# FourLeptonUnfolding

Calvin Independent Study

Phenomenology study of full-phase-space unfolding in
\(H\rightarrow ZZ^{(*)}\rightarrow 4\ell\), starting from POWHEG-BOX-V2 event
generation and alternative parton-shower models.

The workflow is split into:

- [`Generation/`](Generation/) for POWHEG, Pythia, Herwig, and the Unity
  campaign submitter;
- [`BatchSubmit/`](BatchSubmit/) for the storage-efficient full-chain Slurm
  worker (`generation -> Delphes -> compact Analysis`);
- [`Simulation/`](Simulation/) for the H4l Delphes response;
- [`Analysis/`](Analysis/) for matched LHE, dressed-truth, and reconstructed
  observables, merging, and 312 fb$^{-1}$ Herwig pseudo-data ensembles;
- [`Plotting/`](Plotting/) for the multipage Herwig-versus-Pythia validation
  report and the reconstruction-level background-removed-data versus ggH
  starting-point comparison;
- [`BackgroundRemoval/`](BackgroundRemoval/) for the decay-only, strict
  130--160 GeV sideband data-to-MC correction, strict 115--130 GeV
  signal-purity training, and consistent frozen-model decoration of nominal
  data, pseudo-experiments, and Pythia samples;
- [`omnifold/`](omnifold/) for interactive notebook-driven unfolding,
  calibrated reco/truth/reco ratios per iteration, fiducial closure,
  fixed-reference likelihood objectives, and pseudo-experiment decoration.

The built-in samples use 6800 GeV per beam, $\sqrt{s}=13.6$ TeV. The runtime
also validates the LHE beam record so custom cards and external LHE inputs
cannot silently mix collision energies.

New full-chain productions retain source-matched LHE observables before
intermediate files are deleted. The current `gg_H` hard record contains an
undecayed Higgs: its LHE production variables are available, while its decay
angles first exist in the shower. Availability masks keep that distinction
explicit; dressed truth is never substituted for missing LHE information.

Signed POWHEG weights are preserved throughout reduction, merging,
pseudo-data generation, plotting, Background Correction, Background
Removal, and OmniFold. Physics yields use net sums `sum(w)` and statistical variances use
`sum(w^2)`; negative events are not dropped or converted to positive weights.
