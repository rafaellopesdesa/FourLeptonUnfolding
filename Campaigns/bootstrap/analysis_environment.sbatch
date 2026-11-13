#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-analysis-env
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/analysis-env-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/repository"
export TMPDIR="$STUDY/runs/bootstrap/tmp/analysis-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$STUDY/runs/bootstrap/cache"
export PIXI_CACHE_DIR="$XDG_CACHE_HOME/pixi"
export RATTLER_CACHE_DIR="$XDG_CACHE_HOME/rattler"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$STUDY/runs/bootstrap/analysis-${SLURM_JOB_ID}.status"' EXIT
pixi install --manifest-path Analysis/pixi.toml --locked
Analysis/.pixi/envs/default/bin/python -c 'import numpy, uproot, vector, awkward, pyhepmc; print("Analysis imports passed")'
