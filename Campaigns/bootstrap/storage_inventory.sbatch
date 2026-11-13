#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-storage
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:10:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/storage-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
date -u +%FT%TZ
du -sb "$STUDY" /project/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
