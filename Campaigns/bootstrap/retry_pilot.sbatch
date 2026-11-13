#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-pilot-retry
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/pilot-retry-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
process="${1:?process required}"
[[ "$process" == gg_H || "$process" == ZZ ]]
record="$STUDY/runs/pilot/retries/${SLURM_JOB_ID}"
mkdir -p "$record"
dependencies=""
for sample in "${process}_pythia_pilot_20261113" \
              "${process}_pythia_pilot_20261113_muon_scale_up" \
              "${process}_pythia_pilot_20261113_muon_scale_down" \
              "${process}_herwig_pilot_20261113"; do
  campaign="$STUDY/runs/pilot/campaigns/$sample"
  [[ -r "$campaign/campaign.env" ]]
  cp -a "$campaign/jobs" "$record/$sample"
  job_id="$(sbatch --parsable --account=pi_rclsa_umass_edu --partition=cpu --qos=normal \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=06:00:00 \
    --array=0-0%1 --export=NONE --job-name="retry_$sample" \
    --output="$campaign/logs/events-%A_%a.out" \
    "$STUDY/repository/BatchSubmit/unity_generation_job.sh" events "$campaign/campaign.env" 0)"
  job_id="${job_id%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]]
  printf '%s %s\n' "$sample" "$job_id" >> "$record/jobs.txt"
  dependencies+=":$job_id"
done
sbatch --parsable --dependency="afterany$dependencies" \
  "$STUDY/repository/Campaigns/bootstrap/pilot_summary.sbatch" "$process"
