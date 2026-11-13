#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-pilot-setup
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/pilot-setup-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/repository"
export PATH="/home/rclsa_umass_edu/.pixi/bin:$PATH"
export XDG_CACHE_HOME="$STUDY/runs/bootstrap/cache"
export PIXI_CACHE_DIR="$XDG_CACHE_HOME/pixi"
export RATTLER_CACHE_DIR="$XDG_CACHE_HOME/rattler"
export TMPDIR="$STUDY/runs/pilot/tmp"
export SBATCH_QOS=normal
mkdir -p "$TMPDIR" "$STUDY/runs/pilot/scratch"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$STUDY/runs/bootstrap/pilot-setup-${SLURM_JOB_ID}.status"' EXIT

submit_sample() {
  local process="$1" shower="$2" scale="$3" seed="$4"
  shift 4
  ./Generation/submit_generation.sh "$process" "$shower" \
    --jobs 1 --events-per-job 1000 --base-seed "$seed" \
    --campaign "${process}_${shower}_pilot_20261113" \
    --output-root "$STUDY/runs/pilot/campaigns" \
    --analysis-output-root "$STUDY/runs/pilot/shards/$scale" \
    --scratch-root "$STUDY/runs/pilot/scratch" \
    --muon-scale "$scale" --mass-region extended \
    --partition cpu --account pi_rclsa_umass_edu \
    --time 06:00:00 --grid-time 48:00:00 --mem 4G --max-concurrent 1 "$@"
}

case "${1:-initial}" in
  initial)
    # One fresh integration per process; variants reuse it after validation.
    for process in gg_H ZZ; do
      seed=1001001
      [[ "$process" != ZZ ]] || seed=2001001
      submit_sample "$process" pythia nominal "$seed"
      manifest="$STUDY/runs/pilot/campaigns/${process}_pythia_pilot_20261113/submission.txt"
      grid_id="$(awk -F= '$1 == "grid_job_id" {print $2}' "$manifest")"
      [[ "$grid_id" =~ ^[0-9]+$ ]]
      sbatch --parsable --dependency="afterok:$grid_id" \
        "$STUDY/repository/Campaigns/bootstrap/launch_pilot.sbatch" variants "$process"
    done
    ;;
  variants)
    process="${2:?process required}"
    [[ "$process" == gg_H || "$process" == ZZ ]]
    seed=1001001
    [[ "$process" != ZZ ]] || seed=2001001
    grid="$STUDY/runs/pilot/campaigns/${process}_pythia_pilot_20261113/grid"
    for scale in up down; do
      # Same seed requests matching hard events; verify hashes rather than
      # assuming identical stochastic detector histories.
      submit_sample "$process" pythia "$scale" "$seed" --reuse-grid "$grid"
    done
    submit_sample "$process" herwig nominal "$((seed + 10000000))" --reuse-grid "$grid"
    dependencies=""
    for sample in "${process}_pythia_pilot_20261113" \
                  "${process}_pythia_pilot_20261113_muon_scale_up" \
                  "${process}_pythia_pilot_20261113_muon_scale_down" \
                  "${process}_herwig_pilot_20261113"; do
      job_id="$(awk -F= '$1 == "array_job_ids" {print $2}' "$STUDY/runs/pilot/campaigns/$sample/submission.txt")"
      [[ "$job_id" =~ ^[0-9]+$ ]]
      dependencies+=":$job_id"
    done
    sbatch --parsable --dependency="afterany$dependencies" \
      "$STUDY/repository/Campaigns/bootstrap/pilot_summary.sbatch" "$process"
    ;;
  *) echo 'Expected initial or variants PROCESS' >&2; exit 2 ;;
esac
