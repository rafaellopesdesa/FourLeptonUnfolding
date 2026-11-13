#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-archive-queue
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:03:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/archive-queue-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
"$STUDY/repository/Analysis/.pixi/envs/default/bin/python" \
  "$STUDY/repository/Campaigns/parallel/queue_archive.py"
