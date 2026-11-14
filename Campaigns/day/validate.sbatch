#!/usr/bin/env bash
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --job-name=fourlepton_day_checks
#SBATCH --export=NONE
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/day-checks-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/repository"
export TMPDIR="$STUDY/runs/bootstrap/tmp"
"$STUDY/repository/Analysis/.pixi/envs/default/bin/python" -m unittest discover -s Campaigns/day/tests -v
