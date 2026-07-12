#!/bin/bash
# Worker script for Unity Slurm. Resource requests are supplied by
# Generation/submit_generation.sh so one file can serve every campaign.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -Eeuo pipefail

usage() {
  echo "Usage: unity_generation_job.sh grid|events CAMPAIGN_CONFIG [TASK_OFFSET]" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
MODE="$1"
CONFIG_FILE="$(realpath "$2")"
TASK_OFFSET="${3:-0}"

[[ "$MODE" == "grid" || "$MODE" == "events" ]] || usage
[[ -r "$CONFIG_FILE" ]] || {
  echo "Campaign configuration is not readable: $CONFIG_FILE" >&2
  exit 1
}
[[ "$TASK_OFFSET" =~ ^[0-9]+$ ]] || {
  echo "TASK_OFFSET must be a non-negative integer" >&2
  exit 2
}

# This file is created by submit_generation.sh and contains only shell-quoted
# scalar assignments.
# shellcheck source=/dev/null
source "$CONFIG_FILE"

required=(REPO_ROOT CAMPAIGN_DIR PROCESS SHOWER EVENTS_PER_JOB BASE_SEED GRID_DIR RUN_CARD PDF_ID)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || {
    echo "Missing $name in $CONFIG_FILE" >&2
    exit 1
  }
done

RUNNER="$REPO_ROOT/Generation/run_generation.sh"
[[ -x "$RUNNER" ]] || {
  echo "Generation runner is missing or not executable: $RUNNER" >&2
  exit 1
}

write_status() {
  local file="$1"
  local state="$2"
  local exit_code="$3"
  {
    printf 'status=%s\n' "$state"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-not-set}"
    printf 'slurm_array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID:-not-set}"
    printf 'slurm_array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID:-not-set}"
    printf 'task_id=%s\n' "${task_id:-not-set}"
    printf 'seed=%s\n' "${seed:-not-set}"
    printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$file"
}

run_grid_stage() {
  mkdir -p "$GRID_DIR"
  local status_file="$GRID_DIR/slurm-status.txt"
  local rc=0

  {
    printf 'status=running\n'
    printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-not-set}"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$status_file"

  "$RUNNER" "$PROCESS" "$SHOWER" \
    --events 1 \
    --seed "$BASE_SEED" \
    --run-card "$RUN_CARD" \
    --output-dir "$GRID_DIR" \
    --pdf-id "$PDF_ID" || rc=$?

  if ((rc == 0)); then
    compgen -G "$GRID_DIR/pwg*grid*.dat" >/dev/null || {
      echo "POWHEG grid preparation produced no pwg*grid*.dat file" >&2
      rc=1
    }
    compgen -G "$GRID_DIR/pwg*ubound*.dat" >/dev/null || {
      echo "POWHEG grid preparation produced no pwg*ubound*.dat file" >&2
      rc=1
    }
  fi

  if ((rc == 0)); then
    touch "$GRID_DIR/GRID_READY"
    write_status "$status_file" complete 0
  else
    touch "$GRID_DIR/GRID_FAILED"
    write_status "$status_file" failed "$rc"
  fi
  return "$rc"
}

run_event_stage() {
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
    echo "events mode must run as a Slurm array task" >&2
    return 1
  }
  [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || {
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID" >&2
    return 1
  }

  local task_id=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))
  local seed=$((BASE_SEED + task_id))
  local task_label
  printf -v task_label '%06d' "$task_id"
  local job_dir="$CAMPAIGN_DIR/jobs/job_${task_label}_seed${seed}"
  local status_file="$job_dir/slurm-status.txt"
  local rc=0
  mkdir -p "$job_dir"

  {
    printf 'status=running\n'
    printf 'task_id=%s\n' "$task_id"
    printf 'seed=%s\n' "$seed"
    printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-not-set}"
    printf 'slurm_array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID:-not-set}"
    printf 'slurm_array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$status_file"

  [[ -f "$GRID_DIR/GRID_READY" || "${REUSED_GRID:-0}" == 1 ]] || {
    echo "Grid directory is not ready: $GRID_DIR" >&2
    write_status "$status_file" failed 1
    return 1
  }

  # Each task gets private copies: POWHEG may update statistics and upper-bound
  # files while generating events, so workers must not write a shared grid.
  while IFS= read -r -d '' grid_file; do
    cp -p "$grid_file" "$job_dir/"
  done < <(find "$GRID_DIR" -maxdepth 1 -type f \
    \( -name 'pwg*.dat' -o -name 'pwg*.top' \) -print0)

  compgen -G "$job_dir/pwg*grid*.dat" >/dev/null || {
    echo "No integration grid was copied from $GRID_DIR" >&2
    write_status "$status_file" failed 1
    return 1
  }
  compgen -G "$job_dir/pwg*ubound*.dat" >/dev/null || {
    echo "No upper-bound grid was copied from $GRID_DIR" >&2
    write_status "$status_file" failed 1
    return 1
  }

  "$RUNNER" "$PROCESS" "$SHOWER" \
    --events "$EVENTS_PER_JOB" \
    --seed "$seed" \
    --run-card "$RUN_CARD" \
    --output-dir "$job_dir" \
    --pdf-id "$PDF_ID" \
    --keep-grids || rc=$?

  if ((rc == 0)); then
    touch "$job_dir/SUCCESS"
    write_status "$status_file" complete 0
  else
    touch "$job_dir/FAILED"
    write_status "$status_file" failed "$rc"
  fi
  return "$rc"
}

case "$MODE" in
  grid) run_grid_stage ;;
  events) run_event_stage ;;
esac
