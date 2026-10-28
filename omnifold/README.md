# OmniFold in a notebook or from the command line

After BackgroundRemoval, use either
[`interactive_omnifold.ipynb`](interactive_omnifold.ipynb) or
`python -m omnifold` from the repository root. Both interfaces use the same
session, three density-ratio fits, and saved iteration checkpoints. A campaign
created in the notebook can continue from the CLI, and a CLI campaign can
continue in the notebook.

The notebook fits **one iteration each time you run its `session.step()`
cell**, then displays diagnostics and closure plots. The CLI runs a requested
number of iterations and saves PDF reports after each completed iteration by
default. Neither interface automatically chooses the stopping iteration.

## Inputs

| Input | Purpose |
|---|---|
| Background-removal-decorated `data.root` | Nominal reconstructed target; only this file trains the iterative models |
| `gg_H_pythia.root` | Original paired truth/reconstructed simulation and the fixed reference |
| Full `gg_H_herwig.root` (optional) | Independent fiducial truth closure, including events that fail reconstruction |
| Decorated `data_*.root` files | Evaluation of the same frozen sequence on pseudo-experiments |

Data use `weight_background_removed`, which already contains the original
observation weight multiplied by the background-removal factor. Do not
multiply it by `weight` a second time. With the default branch setting, the
session can instead construct it from `weight * background_removal_weight`.
MC use the luminosity-scaled `weight` branch by default. Data and MC must
represent the same luminosity and event-yield convention.

The detector selection is `reconstructed && 115 < reco_m_ZZ < 130` GeV.
The simulated signal additionally requires `fiducial`, because the preceding
background-removal step treats nonfiducial ggH as background. Fiducial events
that fail the detector selection are retained for the truth result.

The three ratio models use the same eight decay inputs as BackgroundRemoval:
`Phi`, `Phi1`, `Psi`, `cos_theta1`, `cos_theta2`, `cos_theta_star`, `m_Z1`, and
`m_Z2`, with `reco_` or `truth_` prefixes. Periodic encoding turns the three
azimuthal angles into sine/cosine pairs. `m_ZZ` defines the detector region;
it is not an input to these networks. Truth means dressed particle level,
not LHE level.

**Do not use the mixed, reconstruction-selected `data.root` as the fiducial
truth reference.** It contains backgrounds and omits reconstruction losses.
The optional Herwig signal file is used only for diagnostic comparisons; its
truth variables never enter data-to-MC training or choose an iteration.

## The three ratios

Let $w_i^0$ be the original MC event weight, $T_0(z)=1$, and $A$ denote MC
events passing both the fiducial and detector selections. Every fitted ratio
has a shape part $r=p_{\mathrm{target}}/p_{\mathrm{reference}}$ and a yield
part $K=\sum w_{\mathrm{target}}/\sum w_{\mathrm{reference}}$; its full
intensity ratio is $R=Kr$. Both sums retain negative event weights.

For iteration $n$:

1. **Reco pull:** train data against MC weighted by $w^0T_{n-1}(z)$ on $A$.
   Its full ratio $R_n^{\mathrm{pull}}(x)$ is the incremental detector-level
   correction.
2. **Truth push:** on the same paired events $A$, train the target with
   weights $w^0T_{n-1}(z)R_n^{\mathrm{pull}}(x)$ against the original
   weights $w^0$, using truth inputs. The resulting full ratio $T_n(z)$ is
   **cumulative**. Evaluate it on all fiducial MC, including detector misses.
3. **Reco likelihood ratio:** train MC weighted by $w^0T_n(z)$ against
   original MC weighted by $w^0$, on $A$ and using reco inputs. The reference
   is always the original PYTHIA sample. This cumulative ratio can be
   evaluated on nominal data and all other pseudo-experiments.

The next iteration starts from $w^0T_n(z)$. Do not multiply $T_n$ by the
previous truth ratio again. The stage-2 training yield ratio is defined on
the paired selection $A$; the total fiducial yield after extending $T_n$ to
missed events need not equal that paired yield.

Training the second ratio on the selected paired events cancels their
common selection efficiency at fixed truth inputs. This requires the
response to be adequately described by those inputs and simulation. An
unobserved truth region with zero selection efficiency cannot be constrained
by these data. Dependence on omitted response variables, including production
kinematics, should be checked with closure tests.

## Network, calibration, and diagnostics

`RatioConfig` defaults match BackgroundRemoval: four hidden layers of 1024
neurons with swish activation, the pinned NSBI toolkit
`DensityRatioLightning` model, four ensemble members, NAdam, batch size 8192,
initial learning rate $3\times10^{-4}$ with exponential factor 0.98,
100 maximum epochs, and patience 12. Logits are bounded at 30. The same
signed BCE, ensemble checks, affine-logit calibration, and shape-preserving
normalization procedure are reused.

Negative weights remain signed in the loss, calibration, physical yields,
and diagnostics. Absolute weights are used only for the stable input scaler
and importance-sampling proposal. A finite positive net yield is required
for each normalized class/split. Statistical variances use $\sum w^2$.

Fit, validation/calibration, and closure partitions are kept fixed across
iterations. Copies of a source event in the target and reference must share
their partition. The validation partition calibrates each ratio; the closure
partition supplies its reweighting and calibration diagnostics. This local
closure split is a diagnostic of that ratio fit, not an independent dataset
for estimating the statistical uncertainty of the entire adaptive unfolding.

The notebook and CLI reports expose learning curves, calibration, ensemble
spread, and per-variable weighted closure for each stage. Reconstructed and
truth comparison figures show absolute expected-event yields by default;
set notebook `SHAPE_ONLY=True` or CLI `--shape-only` for normalized
comparisons. This plotting option does not change training or fitted yield
ratios. The Herwig truth curve is an independent diagnostic target, not a
training input.

## Unity OnDemand setup

Keep the checkout, environment, ROOT inputs, and run outputs under a suitable
`/work/...` location. No system installation or `sudo` is needed. Use one of
the following environment options, from the repository root.

With Pixi:

```bash
pixi install --manifest-path omnifold/pixi.toml
pixi run --manifest-path omnifold/pixi.toml install-kernel
```

This directory adds Jupyter packages to the pinned BackgroundRemoval model
stack. Its first Pixi install resolves the environment and writes a local
`pixi.lock`; retain that lock with the campaign for exact environment reuse.

Alternatively, with an available Python 3.12 interpreter:

```bash
python3.12 -m venv /work/YOUR_GROUP/YOUR_USER/envs/fourlepton-omnifold
source /work/YOUR_GROUP/YOUR_USER/envs/fourlepton-omnifold/bin/activate
python -m pip install --upgrade pip
python -m pip install -r omnifold/requirements.txt
python -m ipykernel install --user --name fourlepton-omnifold \
  --display-name "FourLepton OmniFold"
```

In Unity OnDemand, launch **Interactive Apps → JupyterLab**, request a
GPU-enabled partition and one GPU, and allocate enough memory for the
selected arrays. Start with 32–64 GB for the large production samples and
adjust after observing actual use. Select the **FourLepton OmniFold** kernel
in the notebook. Installing the kernel does not request a GPU: the OnDemand
job must allocate it. `device="auto"` uses CUDA when available and otherwise
runs on CPU; the notebook prints the detected device. Use `device="cuda"`
to require a GPU rather than allowing CPU fallback.

`RatioConfig(inference_batch_size=...)` controls prediction batches as well
as the training-time validation batches. Reduce it if GPU memory is limited.
Committed ensemble models reside on CPU between prediction calls and move
to the requested device for evaluation, so retaining earlier iterations
does not keep every ensemble resident on the GPU.

Unity documents custom kernels and the current interactive-session limits
in its [JupyterLab OnDemand guide](https://docs.unity.rc.umass.edu/documentation/software/ondemand/jupyterlab-ondemand/).
Do not place Python package names in the OnDemand **Modules** field. Request
sufficient wall time for an iteration and resume completed iterations after
an allocation ends; an interrupted fit may need to be repeated.

The command-line interface uses this same environment and adds no
dependencies. Run it in an existing allocated Unity session or job with the
appropriate resources; invoking a training command does not request an
allocation. CLI plots use a noninteractive backend, so report generation
does not need a notebook or display server.

## Command-line workflow

The following commands assume the existing OmniFold environment is active
and the working directory is the repository root. To use Pixi instead,
replace `python -m omnifold COMMAND` with
`pixi run --manifest-path omnifold/pixi.toml COMMAND`; tasks are provided for
`train`, `resume`, `report`, `apply`, `history`, and `likelihood`.

Create a campaign and fit one complete three-stage iteration:

```bash
RUN_DIR=/work/YOUR_GROUP/YOUR_USER/omnifold/run_001

python -m omnifold train \
  --run-dir "$RUN_DIR" \
  --data-root /path/to/background_removed/data.root \
  --gg-h-root /path/to/merged/gg_H_pythia.root \
  --truth-root /path/to/merged/gg_H_herwig.root \
  --iterations 1
```

`--truth-root` is optional and is used only for fiducial closure. `train`
requires a new run directory. `--iterations` defaults to 1. To inspect that
result and then fit two more iterations:

```bash
python -m omnifold history --run-dir "$RUN_DIR"
python -m omnifold resume --run-dir "$RUN_DIR" --iterations 2
```

On `resume`, `--iterations 2` means **two additional iterations**, not a total
of two. For example, a run with iteration 1 already committed will finish
with iterations 1–3. Neural-network patience and epoch limits still operate
within each ratio fit; they do not select an OmniFold stopping iteration.

Training options are the hyphenated `RatioConfig` names, except that model
overwrite is not exposed: `--ensemble-size`, `--hidden-layers`, `--neurons`,
`--batch-size`, `--maximum-epochs`, `--patience`, `--learning-rate`,
`--learning-rate-decay`, `--max-retries`, `--seed`, `--logit-clip`,
`--calibration-scale-min`, `--calibration-scale-max`,
`--inference-batch-size`, `--steps-per-epoch`, and `--device`.
`--seed` controls the ratio-model randomness; `--split-seed` separately
controls the fixed data partitions. `--data-weight-branch`,
`--mc-weight-branch`, and `--truth-weight-branch` select the input weights
described above. Use `python -m omnifold train --help` to inspect defaults.

All commands accept `--cpu-threads`; its default is `SLURM_CPUS_PER_TASK`
when set, otherwise 4. `--device auto` uses CUDA when available and CPU
otherwise; use `--device cuda` to require the allocated GPU. Commands that
load an existing campaign can override the execution device, for example
`resume --run-dir "$RUN_DIR" --device cpu --iterations 1`. The saved inputs,
features, split, weight branches, and training configuration remain fixed;
use a new campaign to change them.

### Reports without retraining

`train` and `resume` save PDF diagnostics after every completed iteration.
Use `--no-reports` to postpone plotting, then generate reports separately:

```bash
python -m omnifold report --run-dir "$RUN_DIR"
python -m omnifold report --run-dir "$RUN_DIR" --iteration 1
python -m omnifold report --run-dir "$RUN_DIR" --all-iterations --shape-only
```

With no iteration option, `report` uses the latest completed iteration.
`--iteration` is one-based and is mutually exclusive with `--all-iterations`.
Reports are written below `RUN_DIR/reports/iteration_NNNN/`; ratio diagnostics
go in `ratios/*.pdf` and comparisons in either `comparison_yields/*.pdf` or
`comparison_shapes/*.pdf`. The accompanying `history.json` is a snapshot of
the history through the selected iteration. `--shape-only` changes only the
comparison plots; the ratio diagnostics remain the same. It is also
available on `train` and `resume` for their automatic comparison reports.

Each training iteration is committed before its reports are generated. If
plotting fails, the command exits with an error but the completed checkpoint
is retained. Run `history` to see what completed, repair the reporting issue,
and rerun `report`. Resuming trains additional iterations; it does not refit
that committed iteration merely to regenerate its plots.

### Export and evaluate the frozen sequence

Apply the saved sequence to all requested pseudo-experiments:

```bash
python -m omnifold apply \
  --run-dir "$RUN_DIR" \
  --inputs '/path/to/background_removed/data*.root' \
  --output-dir /path/to/omnifold_export
```

`--inputs` accepts one or more files, quoted globs, or directories. Directory
inputs select `data*.root`. The nominal PYTHIA MC is included by default;
`--no-mc` omits it. `--overwrite` permits replacement of a recognized prior
export; original inputs are never overwritten. The vector branches and
their interpretation are described below.

History and frozen-model likelihood commands print JSON; add `--output` to
save the same result to a new file. Existing JSON output files are not
replaced:

```bash
python -m omnifold history --run-dir "$RUN_DIR" \
  --output /path/to/omnifold_history.json
python -m omnifold likelihood --run-dir "$RUN_DIR"
python -m omnifold likelihood --run-dir "$RUN_DIR" \
  --data-root /path/to/background_removed/data_0001.root \
  --output /path/to/toy_0001_likelihood.json
```

Omitting `--data-root` evaluates the nominal data. Supplying another toy
evaluates the same frozen models and does not retrain that pseudo-experiment.
The statistical limitations of these weighted objectives are discussed below.

## Notebook workflow and resuming

1. Set the ROOT paths, a new `RUN_DIR`, `MODE="create"`, and `RatioConfig` in
   the configuration cell.
2. Run the setup cell to create the session.
3. Run `result = session.step()` once. It fits the three ratio ensembles and
   records the completed iteration.
4. Run the diagnostics, comparisons, and history cells. Rerun the step cell
   only when you want another iteration. The notebook does not choose a
   stopping iteration automatically.
5. After restarting Jupyter, set `MODE="resume"` and use the same `RUN_DIR`.
   `OmniFoldSession.load(RUN_DIR)` restores the saved sequence. A device
   override can be passed as `device="cpu"` or `device="cuda"`.
6. Run the export cell when the desired iteration sequence is complete.

The same resume instructions apply to a campaign created with the CLI.
Conversely, `python -m omnifold resume --run-dir RUN_DIR --iterations N`
continues a notebook-created campaign. No conversion or copied model files
are needed.

The original ROOT inputs must remain unchanged during a campaign. Use a new
run directory for different input files, selections, or model settings.
For exact reruns, preserve the run manifest, model artifacts, input files,
and the resolved environment.

Each training call creates `RUN_DIR/.training.lock` and removes it when the
call finishes or handles an interruption. If the kernel or process is
killed, that lock can remain. **Confirm that the old job and training
process have stopped before removing a stale lock.** Then load the campaign
and rerun `session.step()`. Loading recovers the last fully committed
iteration; the interrupted iteration is fitted again. Abandoned staging
directories named `RUN_DIR/iterations/.iteration_*` are ignored by loading
and may be removed after confirming that no training is using them. Do not
remove the committed `iteration_0001`, `iteration_0002`, etc. directories.
Do not train the same run concurrently from a notebook, CLI process, or
second batch job. The lock protects one campaign across both interfaces.

## Decoration and likelihood objectives

`session.decorate(inputs=[...], output_dir=..., include_mc=True)` evaluates
the frozen model sequence on each requested data file and on the nominal MC.
It writes new decorated files, preserving their original entries and
branches. The iteration vectors are:

| Branch stem | Meaning |
|---|---|
| `omnifold_reco_pull` | Incremental full reco ratio for each iteration |
| `omnifold_truth_push` | Cumulative full truth ratio for each iteration |
| `omnifold_reco_likelihood` | Cumulative full reco ratio relative to original MC |

Each stem has corresponding `_shape` and `_yield_ratio` vectors and a scalar
`_valid` selection mask. Vector position zero is iteration 1. Invalid ratios
are marked rather than treated as unit weights; global yield factors are
repeated even outside the evaluation domain. The `omnifold_weight`
vectors contain the original background-subtracted observation weights for
data and $w^0T_n(z)$ for MC. The data weights are not multiplied by the
likelihood ratio.

Truth-ratio values on pseudo-data are optional diagnostic evaluations when
their truth branches exist. They do not identify the signal component of a
mixed sample. Real data have no truth inputs, so that ratio is unavailable
there; it never enters the data likelihood objective.

The stage-3 ratio provides the weighted shape objective

$$
\Delta\ell_n^{\mathrm{shape}}=\sum_{j\in\mathrm{data}}w_j\log r_n(x_j)
$$

and the corresponding extended objective

$$
\Delta\ell_n^{\mathrm{extended}}=
\sum_{j\in\mathrm{data}}w_j\log R_n(x_j)
-(N_n-N_0),
$$

where $N_n$ and $N_0$ are the selected reconstructed yields of unfolded and
original MC. They are compared to the same reference at every iteration.
With background-removal weights, these are **weighted objectives**, not
automatically calibrated ordinary Poisson log-likelihoods. The nominal data
also trained the unfolding sequence; its objective curve is descriptive.
Inference about parameters, stopping rules, and confidence intervals needs
an explicit statistical construction and validation. Frozen-model toy
evaluation is conditional on the one nominal-data training and does not
include retraining uncertainty.

The notebook names exports by their final iteration. To repeat an export to
the same directory, set `OVERWRITE_EXPORT=True`; replacement is limited to a
recognized prior OmniFold export. Input files are never overwritten.

Run the regression suite with:

```bash
pixi run --manifest-path omnifold/pixi.toml test
```
