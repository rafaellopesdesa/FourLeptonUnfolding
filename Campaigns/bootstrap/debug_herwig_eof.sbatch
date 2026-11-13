#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-herwig-eof
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/herwig-eof-setup-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/repository"
export PATH="/home/rclsa_umass_edu/.pixi/bin:$PATH"
export XDG_CACHE_HOME="$STUDY/runs/bootstrap/cache"
export PIXI_CACHE_DIR="$XDG_CACHE_HOME/pixi"
export RATTLER_CACHE_DIR="$XDG_CACHE_HOME/rattler"
export TMPDIR="$STUDY/runs/bootstrap/tmp"
export SBATCH_QOS=normal
./Generation/submit_generation.sh gg_H herwig \
  --jobs 1 --events-per-job 10000 --base-seed 40000005 \
  --campaign herwig_eof_diagnostic_20261113 \
  --output-root "$STUDY/runs/herwig_eof/campaigns" \
  --analysis-output-root "$STUDY/runs/herwig_eof/shards" \
  --scratch-root "$STUDY/runs/herwig_eof/scratch" \
  --reuse-grid "$STUDY/runs/pilot/campaigns/gg_H_pythia_pilot_20261113/grid" \
  --muon-scale nominal --mass-region extended --keep-intermediates \
  --partition cpu --account pi_rclsa_umass_edu \
  --time 00:30:00 --mem 3G --max-concurrent 1
