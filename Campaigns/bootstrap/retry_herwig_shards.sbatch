#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-herwig-recovery
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/herwig-recovery-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
campaign="$STUDY/runs/production_signal_wave1/campaigns/gg_H_herwig_production_wave1_20261113"
record="$STUDY/runs/production_signal_wave1/retries/$SLURM_JOB_ID"
(($# > 0 && $# <= 10))
mkdir -p "$record"
for index in "$@"; do
  [[ "$index" =~ ^[0-9]+$ ]] && ((index < 100))
  state="$(sacct -j "1636700_$index" --format=State -n -X | awk 'NF {print $1; exit}')"
  [[ "$state" == FAILED ]]
  printf -v label '%06d' "$index"
  job="$campaign/jobs/job_${label}_seed$((40000001 + index))"
  [[ -f "$job/FAILED" && ! -e "$job/SUCCESS" ]]
  cp -a "$job" "$record/"
done
printf -v indices '%s,' "$@"
indices="${indices%,}"
retry_id="$(sbatch --parsable --account=pi_rclsa_umass_edu --partition=cpu --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=3G --time=01:00:00 \
  --array="$indices%1" --export=NONE --job-name=herwig_recovery \
  --output="$campaign/logs/recovery-%A_%a.out" \
  "$STUDY/repository/BatchSubmit/unity_generation_job.sh" events "$campaign/campaign.env" 0)"
retry_id="${retry_id%%;*}"
[[ "$retry_id" =~ ^[0-9]+$ ]]
printf '%s\n' "$retry_id" > "$record/retry_job_id.txt"
dependencies=afterany:1636684:1636696:1636697:1636700
for prior in "$STUDY/runs/production_signal_wave1/retries"/*/retry_job_id.txt; do
  read -r prior_id < "$prior"
  [[ "$prior_id" =~ ^[0-9]+$ ]]
  dependencies+=":$prior_id"
done
scontrol update JobId=1636701 Dependency="$dependencies"
printf 'Recovery array: %s; shards: %s; one additional worker at a time.\n' "$retry_id" "$indices"
