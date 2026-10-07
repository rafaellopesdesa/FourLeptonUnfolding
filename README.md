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
- [`Analysis/`](Analysis/) for reduction, merging, and 312 fb$^{-1}$ Herwig
  pseudo-data ensembles;
- [`Plotting/`](Plotting/) for the multipage Herwig-versus-Pythia validation
  report and the reconstruction-level background-removed-data versus ggH
  starting-point comparison;
- [`BackgroundRemoval/`](BackgroundRemoval/) for the decay-only, strict
  130--160 GeV sideband data-to-MC correction, strict 115--130 GeV
  signal-purity training, and consistent frozen-model decoration of nominal
  data, pseudo-experiments, and Pythia samples.

The built-in samples use 6800 GeV per beam, $\sqrt{s}=13.6$ TeV. The runtime
also validates the LHE beam record so custom cards and external LHE inputs
cannot silently mix collision energies.
