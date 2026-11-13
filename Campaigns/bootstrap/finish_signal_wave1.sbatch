#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-wave1-completion
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/wave1-completion-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
"$STUDY/repository/Analysis/.pixi/envs/default/bin/python" \
  "$STUDY/repository/Campaigns/bootstrap/finish_signal_wave1.py"
