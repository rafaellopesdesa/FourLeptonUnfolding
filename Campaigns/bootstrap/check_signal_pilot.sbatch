#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-signal-checks
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/signal-checks-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/repository"
export PYTHONPATH="$PWD"
export TMPDIR="$STUDY/runs/bootstrap/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$STUDY/repository/Analysis/.pixi/envs/default/bin/python" Campaigns/bootstrap/check_signal_pilot.py
