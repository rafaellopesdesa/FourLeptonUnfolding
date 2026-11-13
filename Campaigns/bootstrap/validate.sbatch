#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-validation
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/validation-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/validation"
export TMPDIR="$STUDY/runs/bootstrap/tmp/validation-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$STUDY/runs/bootstrap/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$TMPDIR"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$STUDY/runs/bootstrap/validation-${SLURM_JOB_ID}.status"' EXIT
TRAINING_PYTHON="$STUDY/codebase/BackgroundRemoval/.pixi/envs/default/bin/python"
"$TRAINING_PYTHON" -m unittest discover -s Systematics/tests -v
"$TRAINING_PYTHON" -m unittest discover -s BackgroundRemoval/tests -v
"$TRAINING_PYTHON" -m unittest discover -s omnifold/tests -v
