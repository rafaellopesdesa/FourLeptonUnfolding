#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-parallel-checks
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/parallel-checks-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/repository"
export TMPDIR="$STUDY/runs/bootstrap/tmp"
"$STUDY/repository/Analysis/.pixi/envs/default/bin/python" \
  -m unittest discover -s Campaigns/parallel/tests -v
