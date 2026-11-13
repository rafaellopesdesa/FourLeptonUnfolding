# Source register and implementation notes

Updated 2026-11-13. This register distinguishes review of a source from a
complete independent reproduction. Local full-text HTML and extracted text
are under `FourLeptonStudy/references/`, with URLs and SHA256 checksums in
`cache_manifest.json`. These cached papers are not republished in the report.
Long reference papers were read selectively; their unrelated physics examples,
complete appendices, and bibliographies have not all been read.

## Repository and operational documentation

Reviewed the repository at `48a960af91bee00d948dcf5108daa784d4131d4c`, including
the root README and all Markdown documentation in Generation, PowhegCards,
BatchSubmit, Simulation, Analysis, Tools, SMEFT (including generated-code
regeneration), BackgroundRemoval (Correction, Training, Application), Plotting
and OmniFold. Inspected the corresponding generation/worker scripts, event
metadata and merging, training/ratio/workflow code, and diagnostic contracts.

The [NSBI toolkit](https://github.com/iris-hep/nsbi-lhc-toolkit) is pinned by the
project to `fc09848fc6540fd32310faebbe9db6eea7ecd17b`. The new endpoint trainer
reuses the existing project wrapper around its DensityRatioLightning model,
rather than introducing another training implementation.

The [Unity documentation](https://docs.unity.rc.umass.edu/documentation/),
storage, quotas, Slurm batch/array guidance, partitions and GPU pages were
consulted. Live `scontrol`, `sinfo` and accounting results determine resources.
Use /work for active I/O and /project for persisted compact results. This
project's two-directory restriction overrides generic scratch examples.

## Physics and inference literature

| Source | Study scope | Consequence for this project |
|---|---|---|
| [OmniFold, 1911.09107v2](https://arxiv.org/abs/1911.09107v2) | Iterative classifier-ratio construction, closure demonstration, iteration and uncertainty discussion | Preserve paired reco/truth identities. Track the distinction between incremental reco pull and cumulative truth weights. Low-statistics early stopping and inference need an explicit study; the paper's high-statistics illustration does not select our hyperparameters. |
| [Resonance spin/parity, 1208.4018v3](https://arxiv.org/abs/1208.4018v3) | Kinematic definitions and spin-zero decay structure, especially §II | Psi is derived from Phi1 and Phi/2. Preserve orientation, lepton-charge and Z-ordering conventions; seven raw coordinates enter every project ratio. Production kinematics still affect acceptance even when excluded as NN inputs. |
| [Profile OmniFold, 2512.07074v1](https://arxiv.org/abs/2512.07074v1) | Conditional-response formulation, update equations, Gaussian/CMS demonstrations and convergence discussion | Validate conditional response transport; vary nuisance initializations and inspect objective/GOF histories. Population convergence statements do not establish finite-network global convergence. The published accuracy-based V needs an independently capable discriminator. |
| [ATLAS SOFT-2023-01, 2412.01600v2](https://arxiv.org/abs/2412.01600v2) | Density-ratio diagnostics, likelihood systematic factors, pseudo-experiments/Neyman construction, interpolation appendix | Separate shape and yield factors. Test raw-score reliability and reweighting without fitting calibrations. Positive sampling intensities and coverage checks are needed for data-like toys and nonlinear coefficient inference. |
| [ATLAS HIGP-2024-14, 2412.01548v2](https://arxiv.org/abs/2412.01548v2) | Analysis and likelihood construction, validation, nuisance treatment | Use extended yields and unbinned density ratios with auxiliary constraints. Its off-shell analysis is a methodological reference, not the likelihood/data model for this on-shell weighted unfolding study. |
| [NLO likelihood / Born projection, 2606.11083v2](https://arxiv.org/abs/2606.11083v2) | §2.3, kinematic projection equations | The longitudinal/transverse boost construction retains mass and rapidity while projecting away transverse recoil. The repository implements a kinematic map, not the full NLO matrix-element likelihood. |
| [Positive Resampler, 2005.09375v1](https://arxiv.org/abs/2005.09375v1) | Motivation and local signed-to-positive resampling construction | Global replacement by absolute weights changes the target. Any positive reference used for toys requires independent shape and yield closure in the selected phase space. |
| [Cell resampling, 2109.07851v2](https://arxiv.org/abs/2109.07851v2) | §2 cell construction, finite-radius limitations and conclusions | Finite cells may smear structures; the asymptotic statement does not remove finite-sample bias. Check distances, radii and resolution before adopting this alternative. |
| [Muon performance, 2012.00578v2](https://arxiv.org/abs/2012.00578v2) | Efficiency methodology, acceptance and conclusions | The card is an approximation informed by Run-2 performance. The uniform ±1% scale proxy is the project's model, not the published ATLAS uncertainty decomposition. |
| [Electron/photon efficiencies, 2308.13362v2](https://arxiv.org/abs/2308.13362v2) | Reconstruction/ID/isolation methodology and scope | Efficiencies depend on kinematics and working points. Validate the fast-simulation acceptance; do not claim an official Run-3 ATLAS response. |
| [Electron/photon performance, 1908.00005v2](https://arxiv.org/abs/1908.00005v2) | Reconstruction and calibration discussion, conclusions | Clustering, identification and energy response have different roles. Preserve the detector card's explicit assumptions and test resulting low-pT four-lepton acceptance. |
| [Electron energy calibration, 2309.05471v2](https://arxiv.org/abs/2309.05471v2) | Calibration chain, precision/scope and cross-check overview | The electron resolution approximation is not an additional fitted calibration of NN outputs. This paper does not justify the chosen muon uncertainty proxy. |
| [SMEFTsim, 2012.11343v3](https://arxiv.org/abs/2012.11343v3) | Basis/input conventions, parameter restriction and interaction-order guidance (§8), summary | Record U35/Mw input scheme, cutoff and operator restriction. Squared dimension-six terms do not form a complete order-Λ⁻⁴ prediction. Keep width and parameter-shift choices explicit. |
| [MG5_aMC@NLO, 1405.0301v2](https://arxiv.org/abs/1405.0301v2) | Event-by-event reweighting (§2.3.3), standalone outputs and reweighting restrictions (B.5–B.6), conclusions | Check support and denominator conventions. A LO matrix-element decay ratio applied to a showered POWHEG sample defines a deformation; it is not by itself an exact NLO SMEFT simulation. |

The original ATLAS landing pages linked in AGENTS.md were also queried.
Their corresponding arXiv papers supplied the full methodological text when
the landing pages could not be retrieved.

## Decisions and unresolved checks

- **Injection confirmed:** C_HB = +0.1, Lambda = 1 TeV. Keep 0 and −0.1 controls.
- **Seven-input contract:** m4l remains a selection variable; production
  variables are excluded as requested. Their effect on response is a closure
  check, not a reason to silently add them to the networks.
- **No regularization or fitted calibration:** upstream nonzero weight-decay
  defaults were found and removed for new fits. Preserve raw-score and
  normalization diagnostics, validation stopping and recorded numerical bounds.
- **Source provenance:** shower Born decay fields and LHE fields are distinct.
  Require a certified source and measure supported direct-lepton coverage.
- **Signed weights:** preserve signs in MC fits and sums. The existing signed
  bootstrap has a different variance from ordinary data. Establish the toy
  construction before producing the 100-run statistical ensemble.
- **Deformation timing:** applying r after Poisson sampling changes the toy
  variance. Deform the intensity/template first for data-like sampling.
- **Truth independence:** reserve independent pseudo-experiments and truth
  events for final closure; freeze the background pair across toys, but refit
  OmniFold for statistical replicas.
- **Endpoint scope:** marginal systematic models are implemented. Conditional
  response interpolation and nuisance minimization are not yet validated.
  Only a +1 Herwig theory endpoint exists; a negative endpoint or extrapolation
  convention must be documented and tested rather than silently invented.
- **Inference:** background-subtracted weights do not automatically define an
  ordinary extended Poisson likelihood. Validate the statistical construction
  and coefficient coverage before reporting intervals.
- **Uniform auxiliary ensemble:** the requested square is a sensitivity design;
  it is not the Gaussian auxiliary sampling used in nominal coverage studies.

## Next literature checks

Revisit precise response-ratio and nuisance-update equations while implementing
profiling, and the ATLAS interpolation and coverage procedures while building
the coefficient likelihood. Complete any remaining detector subsection needed
to interpret discrepancies observed in the pilot. These are implementation
checks, not claims of completed physics reproduction.
