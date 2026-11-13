#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-training-env
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/training-env-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/codebase"
export TMPDIR="$STUDY/runs/bootstrap/tmp/training-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$STUDY/runs/bootstrap/cache"
export PIXI_CACHE_DIR="$XDG_CACHE_HOME/pixi"
export RATTLER_CACHE_DIR="$XDG_CACHE_HOME/rattler"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$STUDY/runs/bootstrap/training-${SLURM_JOB_ID}.status"' EXIT
pixi install --manifest-path BackgroundRemoval/pixi.toml --locked
BackgroundRemoval/.pixi/envs/default/bin/python -c 'import torch, nsbi_common_utils; print("Training imports passed", torch.__version__)'
