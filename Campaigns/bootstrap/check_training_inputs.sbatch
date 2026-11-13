#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-input-checks
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/input-checks-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/repository"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$STUDY/runs/bootstrap/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
"$STUDY/codebase/BackgroundRemoval/.pixi/envs/default/bin/python" Campaigns/bootstrap/check_training_inputs.py
