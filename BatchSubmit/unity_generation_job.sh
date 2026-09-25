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

required=(
  REPO_ROOT CAMPAIGN_NAME CAMPAIGN_DIR PROCESS SHOWER EVENTS_PER_JOB BASE_SEED
  GRID_DIR RUN_CARD PDF_ID ANALYSIS_OUTPUT_ROOT ANALYSIS_PYTHON MASS_REGION
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || {
    echo "Missing $name in $CONFIG_FILE" >&2
    exit 1
  }
done

GENERATION_RUNNER="$REPO_ROOT/Generation/run_generation.sh"
SIMULATION_RUNNER="$REPO_ROOT/Simulation/run_simulation.sh"
ANALYSIS_RUNNER="$REPO_ROOT/Analysis/build_analysis_tree.py"
ANALYSIS_VALIDATOR="$REPO_ROOT/Analysis/validate_analysis_output.py"
for runner in "$GENERATION_RUNNER" "$SIMULATION_RUNNER"; do
  [[ -x "$runner" ]] || {
    echo "Pipeline shell runner is missing or not executable: $runner" >&2
    exit 1
  }
done
for runner in "$ANALYSIS_RUNNER" "$ANALYSIS_VALIDATOR"; do
  [[ -r "$runner" ]] || {
    echo "Pipeline Python runner is missing or not readable: $runner" >&2
    exit 1
  }
done
[[ -x "$ANALYSIS_PYTHON" ]] || {
  echo "Analysis Python is missing or not executable: $ANALYSIS_PYTHON" >&2
  exit 1
}

ACTIVE_WORK_DIR=""
ACTIVE_JOB_DIR=""
ACTIVE_STATUS_FILE=""
ACTIVE_STAGE="not-started"
ACTIVE_TASK_ID="not-set"
ACTIVE_SEED="not-set"
ACTIVE_ANALYSIS_OUTPUT=""
ACTIVE_PARTIAL_OUTPUT=""
ACTIVE_KEEP_INTERMEDIATES="${KEEP_INTERMEDIATES:-0}"

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
    printf 'task_id=%s\n' "$ACTIVE_TASK_ID"
    printf 'seed=%s\n' "$ACTIVE_SEED"
    printf 'stage=%s\n' "$ACTIVE_STAGE"
    printf 'analysis_output=%s\n' "${ACTIVE_ANALYSIS_OUTPUT:-not-set}"
    printf 'work_directory=%s\n' "${ACTIVE_WORK_DIR:-not-set}"
    printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$file"
}

copy_if_present() {
  local source="$1"
  local destination="$2"
  [[ -f "$source" ]] || return 0
  cp -p -- "$source" "$destination"
}

collect_diagnostics() {
  [[ -n "$ACTIVE_WORK_DIR" && -n "$ACTIVE_JOB_DIR" ]] || return 0
  local diagnostic_dir="$ACTIVE_JOB_DIR/diagnostics"
  mkdir -p "$diagnostic_dir"
  copy_if_present "$ACTIVE_WORK_DIR/generation/powheg.log" \
    "$diagnostic_dir/generation-powheg.log"
  copy_if_present "$ACTIVE_WORK_DIR/generation/pythia.log" \
    "$diagnostic_dir/generation-pythia.log"
  copy_if_present "$ACTIVE_WORK_DIR/generation/herwig-read.log" \
    "$diagnostic_dir/generation-herwig-read.log"
  copy_if_present "$ACTIVE_WORK_DIR/generation/herwig.log" \
    "$diagnostic_dir/generation-herwig.log"
  copy_if_present "$ACTIVE_WORK_DIR/generation/powheg.input" \
    "$diagnostic_dir/powheg.input"
  copy_if_present "$ACTIVE_WORK_DIR/generation/herwig.in" \
    "$diagnostic_dir/herwig.in"
  copy_if_present "$ACTIVE_WORK_DIR/generation/run-metadata.txt" \
    "$diagnostic_dir/generation-metadata.txt"
  copy_if_present "$ACTIVE_WORK_DIR/simulation/generation/delphes.log" \
    "$diagnostic_dir/delphes.log"
  copy_if_present \
    "$ACTIVE_WORK_DIR/simulation/generation/simulation-metadata.txt" \
    "$diagnostic_dir/simulation-metadata.txt"
  copy_if_present \
    "$ACTIVE_WORK_DIR/simulation/generation/delphes_card_ATLAS_resolved.tcl" \
    "$diagnostic_dir/delphes_card_ATLAS_resolved.tcl"
}

cleanup_work_directory() {
  ((ACTIVE_KEEP_INTERMEDIATES == 0)) || return 0
  [[ -n "$ACTIVE_WORK_DIR" ]] || return 0
  [[ -f "$ACTIVE_WORK_DIR/.four-lepton-batch-workdir" ]] || {
    echo "Refusing to clean unmarked work directory: $ACTIVE_WORK_DIR" >&2
    return 1
  }
  rm -rf -- "$ACTIVE_WORK_DIR"
  ACTIVE_WORK_DIR=""
}

fail_event_stage() {
  local rc="$1"
  trap - ERR INT TERM
  set +e
  echo "Pipeline failed during stage '$ACTIVE_STAGE' (exit $rc)" >&2
  collect_diagnostics
  [[ -z "$ACTIVE_PARTIAL_OUTPUT" ]] || rm -f -- "$ACTIVE_PARTIAL_OUTPUT"
  if [[ -n "$ACTIVE_JOB_DIR" ]]; then
    rm -f -- "$ACTIVE_JOB_DIR/SUCCESS"
    touch "$ACTIVE_JOB_DIR/FAILED"
  fi
  [[ -z "$ACTIVE_STATUS_FILE" ]] || write_status "$ACTIVE_STATUS_FILE" failed "$rc"
  cleanup_work_directory
  exit "$rc"
}

run_grid_stage() {
  mkdir -p "$GRID_DIR"
  local status_file="$GRID_DIR/slurm-status.txt"
  local rc=0
  rm -f -- "$GRID_DIR/GRID_READY" "$GRID_DIR/GRID_FAILED"

  {
    printf 'status=running\n'
    printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-not-set}"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$status_file"

  "$GENERATION_RUNNER" "$PROCESS" "$SHOWER" \
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
    # The grid and upper-bound files are reusable; the one-event LHE and HepMC
    # produced while preparing them are not.
    for transient in \
      "$GRID_DIR"/pwgevents*.lhe \
      "$GRID_DIR"/events.hepmc \
      "$GRID_DIR"/events.hepmc3; do
      [[ ! -e "$transient" ]] || rm -f -- "$transient"
    done
    {
      printf 'process=%s\n' "$PROCESS"
      printf 'pdf_id=%s\n' "$PDF_ID"
      printf 'beam_energy_gev=6800\n'
      printf 'sqrt_s_gev=13600\n'
      printf 'run_card_sha256=%s\n' "$(sha256sum "$RUN_CARD" | awk '{print $1}')"
    } >"$GRID_DIR/grid-metadata.txt"
    rm -f -- "$GRID_DIR/GRID_FAILED"
    touch "$GRID_DIR/GRID_READY"
    write_status "$status_file" complete 0
  else
    rm -f -- "$GRID_DIR/GRID_READY"
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
  local output_name="${PROCESS}_${SHOWER}_${CAMPAIGN_NAME}_job_${task_label}_seed${seed}.root"
  local analysis_output="$ANALYSIS_OUTPUT_ROOT/$output_name"
  mkdir -p "$job_dir"

  ACTIVE_JOB_DIR="$job_dir"
  ACTIVE_STATUS_FILE="$status_file"
  ACTIVE_STAGE=preflight
  ACTIVE_TASK_ID="$task_id"
  ACTIVE_SEED="$seed"
  ACTIVE_ANALYSIS_OUTPUT="$analysis_output"
  trap 'fail_event_stage "$?"' ERR
  trap 'fail_event_stage 130' INT
  trap 'fail_event_stage 143' TERM

  {
    printf 'status=running\n'
    printf 'task_id=%s\n' "$task_id"
    printf 'seed=%s\n' "$seed"
    printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-not-set}"
    printf 'slurm_array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID:-not-set}"
    printf 'slurm_array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
    printf 'stage=%s\n' "$ACTIVE_STAGE"
    printf 'analysis_output=%s\n' "$analysis_output"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$status_file"

  [[ -f "$GRID_DIR/GRID_READY" || "${REUSED_GRID:-0}" == 1 ]] || {
    echo "Grid directory is not ready: $GRID_DIR" >&2
    fail_event_stage 1
  }

  mkdir -p "$ANALYSIS_OUTPUT_ROOT"
  if [[ -e "$analysis_output" ]]; then
    local prior_metadata="$job_dir/output-metadata.txt"
    [[ -r "$prior_metadata" ]] || {
      echo "Existing output has no matching task metadata: $analysis_output" >&2
      echo "Refusing to accept a potentially colliding campaign output." >&2
      fail_event_stage 1
    }
    local stored_output stored_output_sha stored_config_sha current_output_sha current_config_sha
    stored_output="$(awk -F= '$1 == "analysis_output" {print substr($0, index($0, "=") + 1); exit}' "$prior_metadata")"
    stored_output_sha="$(awk -F= '$1 == "analysis_sha256" {print $2; exit}' "$prior_metadata")"
    stored_config_sha="$(awk -F= '$1 == "campaign_config_sha256" {print $2; exit}' "$prior_metadata")"
    current_output_sha="$(sha256sum "$analysis_output" | awk '{print $1}')"
    current_config_sha="$(sha256sum "$CONFIG_FILE" | awk '{print $1}')"
    [[ "$stored_output" == "$analysis_output" && \
       -n "$stored_output_sha" && "$stored_output_sha" == "$current_output_sha" && \
       -n "$stored_config_sha" && "$stored_config_sha" == "$current_config_sha" ]] || {
      echo "Existing output does not match this task's recorded campaign provenance: $analysis_output" >&2
      fail_event_stage 1
    }
    "$ANALYSIS_PYTHON" "$ANALYSIS_VALIDATOR" "$analysis_output" \
      --expected-entries "$EVENTS_PER_JOB"
    ACTIVE_STAGE=complete
    rm -f -- "$job_dir/FAILED"
    touch "$job_dir/SUCCESS"
    write_status "$status_file" complete 0
    trap - ERR INT TERM
    echo "Already complete: $analysis_output"
    return 0
  fi

  local scratch_parent="${SCRATCH_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}"
  if ((ACTIVE_KEEP_INTERMEDIATES)); then
    scratch_parent="$job_dir/intermediates"
  fi
  mkdir -p "$scratch_parent"
  ACTIVE_WORK_DIR="$(
    mktemp -d "$scratch_parent/four-lepton-${SLURM_JOB_ID:-local}-${task_label}.XXXXXX"
  )"
  touch "$ACTIVE_WORK_DIR/.four-lepton-batch-workdir"
  local generation_dir="$ACTIVE_WORK_DIR/generation"
  mkdir -p "$generation_dir"

  # Each task gets private copies: POWHEG may update statistics and upper-bound
  # files while generating events, so workers must not write a shared grid.
  # The copies live in transient worker scratch, not persistent campaign space.
  ACTIVE_STAGE=grid-copy
  while IFS= read -r -d '' grid_file; do
    cp -p -- "$grid_file" "$generation_dir/"
  done < <(find "$GRID_DIR" -maxdepth 1 -type f \
    \( -name 'pwg*.dat' -o -name 'pwg*.top' \) -print0)

  compgen -G "$generation_dir/pwg*grid*.dat" >/dev/null || {
    echo "No integration grid was copied from $GRID_DIR" >&2
    fail_event_stage 1
  }
  compgen -G "$generation_dir/pwg*ubound*.dat" >/dev/null || {
    echo "No upper-bound grid was copied from $GRID_DIR" >&2
    fail_event_stage 1
  }

  ACTIVE_STAGE=generation
  "$GENERATION_RUNNER" "$PROCESS" "$SHOWER" \
    --events "$EVENTS_PER_JOB" \
    --seed "$seed" \
    --run-card "$RUN_CARD" \
    --output-dir "$generation_dir" \
    --pdf-id "$PDF_ID" \
    --keep-grids

  ACTIVE_STAGE=simulation
  local -a simulation_command=(
    "$SIMULATION_RUNNER" "$generation_dir"
    --process "$PROCESS"
    --output-root "$ACTIVE_WORK_DIR/simulation"
    --higgs-br "${HIGGS_BR:-2.771E-04}"
    --overwrite
  )
  if [[ -n "${DELPHES_CARD:-}" ]]; then
    simulation_command+=(--card "$DELPHES_CARD")
  fi
  "${simulation_command[@]}"

  local delphes_output="$ACTIVE_WORK_DIR/simulation/generation/delphes.root"
  [[ -s "$delphes_output" ]] || {
    echo "Simulation did not produce $delphes_output" >&2
    fail_event_stage 1
  }

  ACTIVE_STAGE=analysis
  local scratch_analysis="$ACTIVE_WORK_DIR/analysis.root"
  "$ANALYSIS_PYTHON" "$ANALYSIS_RUNNER" "$delphes_output" \
    --output "$scratch_analysis" \
    --mass-region "$MASS_REGION"
  "$ANALYSIS_PYTHON" "$ANALYSIS_VALIDATOR" "$scratch_analysis" \
    --expected-entries "$EVENTS_PER_JOB"

  ACTIVE_STAGE=publish
  ACTIVE_PARTIAL_OUTPUT="$ANALYSIS_OUTPUT_ROOT/.${output_name}.partial.${SLURM_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID}"
  rm -f -- "$ACTIVE_PARTIAL_OUTPUT"
  cp --reflink=auto -- "$scratch_analysis" "$ACTIVE_PARTIAL_OUTPUT"
  "$ANALYSIS_PYTHON" "$ANALYSIS_VALIDATOR" "$ACTIVE_PARTIAL_OUTPUT" \
    --expected-entries "$EVENTS_PER_JOB"
  if ln -- "$ACTIVE_PARTIAL_OUTPUT" "$analysis_output"; then
    rm -f -- "$ACTIVE_PARTIAL_OUTPUT"
  else
    echo "Final output name was claimed while this task was running: $analysis_output" >&2
    echo "Refusing to adopt a file whose campaign provenance cannot be proven." >&2
    fail_event_stage 1
  fi
  ACTIVE_PARTIAL_OUTPUT=""

  ACTIVE_STAGE=complete
  collect_diagnostics
  {
    printf 'analysis_output=%s\n' "$analysis_output"
    printf 'analysis_sha256=%s\n' "$(sha256sum "$analysis_output" | awk '{print $1}')"
    printf 'campaign_config_sha256=%s\n' "$(sha256sum "$CONFIG_FILE" | awk '{print $1}')"
    printf 'events=%s\n' "$EVENTS_PER_JOB"
    printf 'mass_region=%s\n' "$MASS_REGION"
  } >"$job_dir/output-metadata.txt"
  rm -f -- "$job_dir/FAILED"
  touch "$job_dir/SUCCESS"
  write_status "$status_file" complete 0
  cleanup_work_directory
  trap - ERR INT TERM
  echo "Published compact Analysis output: $analysis_output"
}

case "$MODE" in
  grid) run_grid_stage ;;
  events) run_event_stage ;;
esac
