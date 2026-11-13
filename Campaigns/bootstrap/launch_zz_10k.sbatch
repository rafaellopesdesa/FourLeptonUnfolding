#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-ZZ-10k-setup
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/ZZ-10k-setup-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
"$STUDY/repository/Analysis/.pixi/envs/default/bin/python" \
  "$STUDY/repository/Campaigns/bootstrap/launch_zz_10k.py"
