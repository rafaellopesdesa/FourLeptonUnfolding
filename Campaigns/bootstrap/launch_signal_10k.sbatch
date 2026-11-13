#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-signal-10k
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/signal-10k-setup-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/repository"
export PATH="/home/rclsa_umass_edu/.pixi/bin:$PATH"
export XDG_CACHE_HOME="$STUDY/runs/bootstrap/cache"
export PIXI_CACHE_DIR="$XDG_CACHE_HOME/pixi"
export RATTLER_CACHE_DIR="$XDG_CACHE_HOME/rattler"
export TMPDIR="$STUDY/runs/bootstrap/tmp"
export SBATCH_QOS=normal
base="$STUDY/runs/pilot10k"
grid="$STUDY/runs/pilot/campaigns/gg_H_pythia_pilot_20261113/grid"
mkdir -p "$base/scratch"
dependencies=""
for configuration in pythia:nominal pythia:up pythia:down herwig:nominal; do
  shower="${configuration%:*}"
  scale="${configuration#*:}"
  seed=1002001
  [[ "$shower" != herwig ]] || seed=11002001
  campaign="gg_H_${shower}_pilot10k_20261113"
  ./Generation/submit_generation.sh gg_H "$shower" \
    --jobs 1 --events-per-job 10000 --base-seed "$seed" --campaign "$campaign" \
    --output-root "$base/campaigns" --analysis-output-root "$base/shards/$scale" \
    --scratch-root "$base/scratch" --reuse-grid "$grid" \
    --muon-scale "$scale" --mass-region extended --partition cpu \
    --account pi_rclsa_umass_edu --time 01:00:00 --mem 2G --max-concurrent 1
  [[ "$scale" == nominal ]] || campaign+="_muon_scale_$scale"
  job_id="$(awk -F= '$1 == "array_job_ids" {print $2}' "$base/campaigns/$campaign/submission.txt")"
  [[ "$job_id" =~ ^[0-9]+$ ]]
  dependencies+=":$job_id"
done
sbatch --parsable --dependency="afterany$dependencies" \
  "$STUDY/repository/Campaigns/bootstrap/pilot_summary.sbatch" gg_H pilot10k pilot10k_20261113 10000
