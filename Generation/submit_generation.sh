#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"
WORKER="$REPO_ROOT/BatchSubmit/unity_generation_job.sh"
UNITY_WORK_ROOT="/work/pi_rclsa_umass_edu"

usage() {
  cat <<'EOF'
Submit a POWHEG -> shower -> Delphes -> Analysis campaign to Unity Slurm.

Usage:
  ./submit_generation.sh PROCESS SHOWER [options]

Required positional arguments:
  PROCESS                 gg_H or ZZ
  SHOWER                  pythia or herwig

Campaign options:
  --jobs N                Number of event-generation tasks (default: 1)
  --events-per-job N      Events in each task, 1000--10000 (default: 5000)
  --campaign NAME         Unique campaign name (default: timestamped name)
  --output-root DIR       Campaign parent directory
                           (default: /work/pi_rclsa_umass_edu/FourLeptonUnfoldingGeneration)
  --analysis-output-root DIR
                           Shared flat directory for compact Analysis ROOT shards
                           (default: OUTPUT_ROOT/analysis)
  --scratch-root DIR      Worker-local scratch parent (default: SLURM_TMPDIR,
                           then TMPDIR, then /tmp)
  --base-seed N           Seed for task zero (default: 1001)
  --run-card FILE         Override PowhegCards/PROCESS.powheg.input
  --pdf-id ID             LHAPDF member ID (default: 303400)
  --reuse-grid DIR        Reuse a compatible existing POWHEG grid and skip grid job
  --mass-region NAME      extended or signal (default: extended)
  --delphes-card FILE     Override the installed ATLAS base card
  --higgs-br VALUE        H->ZZ->4l weight factor (default: 2.771E-04)
  --keep-intermediates    Persist the complete per-task work directory for debugging

Slurm options:
  --partition NAME        Partition (default: cpu)
  --account NAME          Slurm account; omitted by default
  --time HH:MM:SS         Full-chain task limit (default: 24:00:00)
  --grid-time HH:MM:SS    Grid-job limit (default: 48:00:00)
  --mem SIZE              Memory per task (default: 8G)
  --max-concurrent N      Maximum tasks running in one array (default: 100)
  --dry-run               Build and print the campaign without calling sbatch
  -h, --help              Show this help

Examples:
  # 10 million events in 1000 jobs, after preparing one shared grid
  ./submit_generation.sh gg_H pythia --jobs 1000 --events-per-job 10000 \
    --campaign ggH_pythia_run3_10M \
    --analysis-output-root /work/pi_rclsa_umass_edu/FourLeptonAnalysis

  # Inspect a small campaign without submitting it
  ./submit_generation.sh ZZ herwig --jobs 2 --events-per-job 1000 \
    --output-root /tmp/four-lepton-dry-run --dry-run
EOF
}

if (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; then
  usage
  exit 0
fi
if (($# < 2)); then
  usage >&2
  exit 2
fi

PROCESS="$1"
SHOWER="$2"
shift 2

JOBS=1
EVENTS_PER_JOB=5000
CAMPAIGN=""
OUTPUT_ROOT="$UNITY_WORK_ROOT/FourLeptonUnfoldingGeneration"
ANALYSIS_OUTPUT_ROOT=""
SCRATCH_ROOT=""
BASE_SEED=1001
RUN_CARD=""
PDF_ID="${PDF_ID:-303400}"
REUSE_GRID=""
MASS_REGION=extended
DELPHES_CARD=""
HIGGS_BR="2.771E-04"
KEEP_INTERMEDIATES=0
PARTITION=cpu
ACCOUNT=""
TIME_LIMIT=24:00:00
GRID_TIME_LIMIT=48:00:00
MEMORY=8G
MAX_CONCURRENT=100
DRY_RUN=0
MAX_ARRAY_SIZE=1900

need_value() {
  (($# >= 2)) || {
    echo "Option $1 requires a value" >&2
    exit 2
  }
}

while (($#)); do
  case "$1" in
    --jobs) need_value "$@"; JOBS="$2"; shift 2 ;;
    --events-per-job) need_value "$@"; EVENTS_PER_JOB="$2"; shift 2 ;;
    --campaign) need_value "$@"; CAMPAIGN="$2"; shift 2 ;;
    --output-root) need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --analysis-output-root) need_value "$@"; ANALYSIS_OUTPUT_ROOT="$2"; shift 2 ;;
    --scratch-root) need_value "$@"; SCRATCH_ROOT="$2"; shift 2 ;;
    --base-seed) need_value "$@"; BASE_SEED="$2"; shift 2 ;;
    --run-card) need_value "$@"; RUN_CARD="$2"; shift 2 ;;
    --pdf-id) need_value "$@"; PDF_ID="$2"; shift 2 ;;
    --reuse-grid) need_value "$@"; REUSE_GRID="$2"; shift 2 ;;
    --mass-region) need_value "$@"; MASS_REGION="$2"; shift 2 ;;
    --delphes-card) need_value "$@"; DELPHES_CARD="$2"; shift 2 ;;
    --higgs-br) need_value "$@"; HIGGS_BR="$2"; shift 2 ;;
    --keep-intermediates) KEEP_INTERMEDIATES=1; shift ;;
    --partition) need_value "$@"; PARTITION="$2"; shift 2 ;;
    --account) need_value "$@"; ACCOUNT="$2"; shift 2 ;;
    --time) need_value "$@"; TIME_LIMIT="$2"; shift 2 ;;
    --grid-time) need_value "$@"; GRID_TIME_LIMIT="$2"; shift 2 ;;
    --mem) need_value "$@"; MEMORY="$2"; shift 2 ;;
    --max-concurrent) need_value "$@"; MAX_CONCURRENT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$PROCESS" == gg_H || "$PROCESS" == ZZ ]] || {
  echo "PROCESS must be gg_H or ZZ" >&2
  exit 2
}
[[ "$SHOWER" == pythia || "$SHOWER" == herwig ]] || {
  echo "SHOWER must be pythia or herwig" >&2
  exit 2
}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--jobs must be a positive integer" >&2
  exit 2
}
[[ "$EVENTS_PER_JOB" =~ ^[0-9]+$ ]] && ((EVENTS_PER_JOB >= 1000 && EVENTS_PER_JOB <= 10000)) || {
  echo "--events-per-job must be between 1000 and 10000" >&2
  exit 2
}
[[ "$BASE_SEED" =~ ^[1-9][0-9]*$ ]] || {
  echo "--base-seed must be a positive integer" >&2
  exit 2
}
((BASE_SEED + JOBS - 1 <= 900000000)) || {
  echo "The final seed must not exceed 900000000" >&2
  exit 2
}
[[ "$PDF_ID" =~ ^[0-9]+$ ]] || {
  echo "--pdf-id must be a non-negative integer" >&2
  exit 2
}
[[ "$MASS_REGION" == extended || "$MASS_REGION" == signal ]] || {
  echo "--mass-region must be extended or signal" >&2
  exit 2
}
[[ "$HIGGS_BR" =~ ^[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$ ]] && \
    awk -v value="$HIGGS_BR" 'BEGIN { exit !(value > 0) }' || {
  echo "--higgs-br must be a positive number" >&2
  exit 2
}
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max-concurrent must be a positive integer" >&2
  exit 2
}
((MAX_CONCURRENT <= 1000)) || {
  echo "--max-concurrent must not exceed Unity's 1000-running-jobs user limit" >&2
  exit 2
}

if [[ -z "$CAMPAIGN" ]]; then
  CAMPAIGN="$(date -u +%Y%m%dT%H%M%SZ)_${PROCESS}_${SHOWER}"
fi
[[ "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "--campaign may contain only letters, digits, dots, underscores, and hyphens" >&2
  exit 2
}

OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
if [[ -z "$ANALYSIS_OUTPUT_ROOT" ]]; then
  ANALYSIS_OUTPUT_ROOT="$OUTPUT_ROOT/analysis"
fi
ANALYSIS_OUTPUT_ROOT="$(realpath -m "$ANALYSIS_OUTPUT_ROOT")"
[[ -z "$SCRATCH_ROOT" ]] || SCRATCH_ROOT="$(realpath -m "$SCRATCH_ROOT")"
if ((DRY_RUN == 0)); then
  command -v sbatch >/dev/null || {
    echo "sbatch is unavailable; run this script on a Unity login node." >&2
    exit 1
  }
  [[ "$REPO_ROOT" == "$UNITY_WORK_ROOT"/* ]] || {
    echo "The repository must be cloned and compiled below $UNITY_WORK_ROOT for Unity jobs." >&2
    echo "Current repository: $REPO_ROOT" >&2
    exit 1
  }
  [[ "$OUTPUT_ROOT" == "$UNITY_WORK_ROOT"/* ]] || {
    echo "Campaign output must be below $UNITY_WORK_ROOT" >&2
    exit 1
  }
  [[ "$ANALYSIS_OUTPUT_ROOT" == "$UNITY_WORK_ROOT"/* ]] || {
    echo "Compact Analysis output must be below $UNITY_WORK_ROOT" >&2
    exit 1
  }
fi

[[ -x "$WORKER" ]] || {
  echo "Missing executable worker: $WORKER" >&2
  exit 1
}
[[ -r "$SCRIPT_DIR/env.sh" ]] || {
  echo "Missing $SCRIPT_DIR/env.sh; compile the generator stack under /work first." >&2
  exit 1
}
[[ -r "$REPO_ROOT/Simulation/env.sh" ]] || {
  echo "Missing $REPO_ROOT/Simulation/env.sh; run Simulation/install_delphes.sh first." >&2
  exit 1
}
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"
if ((DRY_RUN == 0)); then
  [[ "${GENERATION_PREFIX:-}" == "$UNITY_WORK_ROOT"/* ]] || {
    echo "GENERATION_PREFIX must be below $UNITY_WORK_ROOT; reinstall from /work." >&2
    echo "Current prefix: ${GENERATION_PREFIX:-not-set}" >&2
    exit 1
  }
fi

if [[ -n "$DELPHES_CARD" ]]; then
  DELPHES_CARD="$(realpath "$DELPHES_CARD")"
  [[ -r "$DELPHES_CARD" ]] || {
    echo "Delphes base card is not readable: $DELPHES_CARD" >&2
    exit 1
  }
fi

command -v pixi >/dev/null || {
  echo "pixi is required; install it and initialize Analysis/pixi.toml first." >&2
  exit 1
}
# Resolve the environment once on the login node. Array workers call this
# interpreter directly, avoiding hundreds of concurrent Pixi lock operations.
pixi install --locked --manifest-path "$REPO_ROOT/Analysis/pixi.toml"
ANALYSIS_PYTHON="$(
  pixi run --manifest-path "$REPO_ROOT/Analysis/pixi.toml" \
    python -c 'import sys; print(sys.executable)'
)"
[[ -x "$ANALYSIS_PYTHON" ]] || {
  echo "Could not resolve the Analysis Python interpreter" >&2
  exit 1
}
"$ANALYSIS_PYTHON" -c 'import awkward, numpy, uproot, vector'

if [[ -z "$RUN_CARD" ]]; then
  RUN_CARD="$REPO_ROOT/PowhegCards/${PROCESS}.powheg.input"
fi
RUN_CARD="$(realpath "$RUN_CARD")"
[[ -r "$RUN_CARD" ]] || {
  echo "POWHEG run card is not readable: $RUN_CARD" >&2
  exit 1
}

CAMPAIGN_DIR="$OUTPUT_ROOT/$CAMPAIGN"
[[ ! -e "$CAMPAIGN_DIR" ]] || {
  echo "Campaign already exists: $CAMPAIGN_DIR" >&2
  exit 1
}
mkdir -p "$CAMPAIGN_DIR"/{inputs,jobs,logs,grid} "$ANALYSIS_OUTPUT_ROOT"
cp "$RUN_CARD" "$CAMPAIGN_DIR/inputs/powheg.input"
RUN_CARD="$CAMPAIGN_DIR/inputs/powheg.input"

REUSED_GRID=0
if [[ -n "$REUSE_GRID" ]]; then
  GRID_DIR="$(realpath "$REUSE_GRID")"
  [[ -d "$GRID_DIR" ]] || {
    echo "Grid directory does not exist: $GRID_DIR" >&2
    exit 1
  }
  compgen -G "$GRID_DIR/pwg*grid*.dat" >/dev/null || {
    echo "No pwg*grid*.dat file found in $GRID_DIR" >&2
    exit 1
  }
  compgen -G "$GRID_DIR/pwg*ubound*.dat" >/dev/null || {
    echo "No pwg*ubound*.dat file found in $GRID_DIR" >&2
    exit 1
  }
  GRID_METADATA="$GRID_DIR/grid-metadata.txt"
  [[ -r "$GRID_METADATA" ]] || {
    echo "Reused grid has no readable grid-metadata.txt, so its beam energy cannot be verified." >&2
    echo "Prepare a new grid with this version before using --reuse-grid." >&2
    exit 1
  }
  grid_metadata_value() {
    local key="$1"
    awk -F= -v wanted="$key" '$1 == wanted {print substr($0, index($0, "=") + 1); exit}' \
      "$GRID_METADATA"
  }
  reused_process="$(grid_metadata_value process)"
  reused_pdf_id="$(grid_metadata_value pdf_id)"
  reused_sqrt_s="$(grid_metadata_value sqrt_s_gev)"
  reused_card_sha256="$(grid_metadata_value run_card_sha256)"
  expected_card_sha256="$(sha256sum "$RUN_CARD" | awk '{print $1}')"
  [[ "$reused_sqrt_s" == 13600 ]] || {
    echo "Reused grid has sqrt_s_gev=${reused_sqrt_s:-missing}; this project requires 13600." >&2
    exit 1
  }
  [[ "$reused_process" == "$PROCESS" ]] || {
    echo "Reused grid process is ${reused_process:-missing}; expected $PROCESS." >&2
    exit 1
  }
  [[ "$reused_pdf_id" == "$PDF_ID" ]] || {
    echo "Reused grid PDF ID is ${reused_pdf_id:-missing}; expected $PDF_ID." >&2
    exit 1
  }
  [[ "$reused_card_sha256" == "$expected_card_sha256" ]] || {
    echo "Reused grid run-card checksum does not match the selected card." >&2
    exit 1
  }
  if ((DRY_RUN == 0)); then
    [[ "$GRID_DIR" == "$UNITY_WORK_ROOT"/* ]] || {
      echo "A reused grid must be below $UNITY_WORK_ROOT" >&2
      exit 1
    }
  fi
  REUSED_GRID=1
else
  GRID_DIR="$CAMPAIGN_DIR/grid"
fi

CONFIG_FILE="$CAMPAIGN_DIR/campaign.env"
write_config() {
  local name="$1"
  local value="$2"
  printf '%s=%q\n' "$name" "$value"
}
{
  write_config REPO_ROOT "$REPO_ROOT"
  write_config CAMPAIGN_NAME "$CAMPAIGN"
  write_config CAMPAIGN_DIR "$CAMPAIGN_DIR"
  write_config PROCESS "$PROCESS"
  write_config SHOWER "$SHOWER"
  write_config JOBS "$JOBS"
  write_config EVENTS_PER_JOB "$EVENTS_PER_JOB"
  write_config TOTAL_EVENTS "$((JOBS * EVENTS_PER_JOB))"
  write_config BASE_SEED "$BASE_SEED"
  write_config GRID_DIR "$GRID_DIR"
  write_config REUSED_GRID "$REUSED_GRID"
  write_config RUN_CARD "$RUN_CARD"
  write_config PDF_ID "$PDF_ID"
  write_config ANALYSIS_OUTPUT_ROOT "$ANALYSIS_OUTPUT_ROOT"
  write_config ANALYSIS_PYTHON "$ANALYSIS_PYTHON"
  write_config MASS_REGION "$MASS_REGION"
  write_config SCRATCH_ROOT "$SCRATCH_ROOT"
  write_config DELPHES_CARD "$DELPHES_CARD"
  write_config HIGGS_BR "$HIGGS_BR"
  write_config KEEP_INTERMEDIATES "$KEEP_INTERMEDIATES"
  write_config BEAM_ENERGY_GEV 6800
  write_config SQRT_S_GEV 13600
  write_config CREATED_UTC "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$CONFIG_FILE"
chmod 0444 "$CONFIG_FILE" "$RUN_CARD"

common_sbatch=(
  --parsable
  --partition="$PARTITION"
  --nodes=1
  --ntasks=1
  --cpus-per-task=1
  --mem="$MEMORY"
  --export=NONE
)
if [[ -n "$ACCOUNT" ]]; then
  common_sbatch+=(--account="$ACCOUNT")
fi

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

GRID_JOB_ID=""
if ((REUSED_GRID == 0)); then
  grid_command=(
    sbatch "${common_sbatch[@]}"
    --job-name="grid_${PROCESS}"
    --time="$GRID_TIME_LIMIT"
    --output="$CAMPAIGN_DIR/logs/grid-%j.out"
    "$WORKER" grid "$CONFIG_FILE"
  )
  if ((DRY_RUN)); then
    echo "Grid command:"
    print_command "${grid_command[@]}"
    GRID_JOB_ID=DRY_RUN_GRID
  else
    GRID_SUBMISSION="$("${grid_command[@]}")"
    GRID_JOB_ID="${GRID_SUBMISSION%%;*}"
  fi
fi

ARRAY_JOB_IDS=()
offset=0
previous_job_id="$GRID_JOB_ID"
wave=0
while ((offset < JOBS)); do
  remaining=$((JOBS - offset))
  wave_size=$((remaining < MAX_ARRAY_SIZE ? remaining : MAX_ARRAY_SIZE))
  array_end=$((wave_size - 1))
  array_command=(
    sbatch "${common_sbatch[@]}"
    --job-name="${PROCESS}_${SHOWER}_${wave}"
    --time="$TIME_LIMIT"
    --array="0-${array_end}%${MAX_CONCURRENT}"
    --output="$CAMPAIGN_DIR/logs/events-%A_%a.out"
  )
  if [[ -n "$previous_job_id" ]]; then
    array_command+=(--dependency="afterok:${previous_job_id}")
  fi
  array_command+=("$WORKER" events "$CONFIG_FILE" "$offset")

  if ((DRY_RUN)); then
    echo "Event-array wave $wave (global tasks $offset-$((offset + wave_size - 1))):"
    print_command "${array_command[@]}"
    array_job_id="DRY_RUN_ARRAY_${wave}"
  else
    array_submission="$("${array_command[@]}")"
    array_job_id="${array_submission%%;*}"
  fi
  ARRAY_JOB_IDS+=("$array_job_id")
  previous_job_id="$array_job_id"
  offset=$((offset + wave_size))
  wave=$((wave + 1))
done

{
  printf 'campaign_dir=%s\n' "$CAMPAIGN_DIR"
  printf 'grid_job_id=%s\n' "${GRID_JOB_ID:-reused}"
  printf 'array_job_ids=%s\n' "${ARRAY_JOB_IDS[*]}"
  printf 'jobs=%s\n' "$JOBS"
  printf 'events_per_job=%s\n' "$EVENTS_PER_JOB"
  printf 'total_events=%s\n' "$((JOBS * EVENTS_PER_JOB))"
  printf 'max_concurrent_per_wave=%s\n' "$MAX_CONCURRENT"
  printf 'analysis_output_root=%s\n' "$ANALYSIS_OUTPUT_ROOT"
  printf 'mass_region=%s\n' "$MASS_REGION"
  printf 'keep_intermediates=%s\n' "$KEEP_INTERMEDIATES"
  printf 'beam_energy_gev=6800\n'
  printf 'sqrt_s_gev=13600\n'
  printf 'submitted_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$CAMPAIGN_DIR/submission.txt"

echo "Campaign: $CAMPAIGN_DIR"
echo "Grid job: ${GRID_JOB_ID:-reused grid}"
echo "Event arrays: ${ARRAY_JOB_IDS[*]}"
echo "Planned events: $((JOBS * EVENTS_PER_JOB))"
echo "Compact outputs: $ANALYSIS_OUTPUT_ROOT/${PROCESS}_${SHOWER}_${CAMPAIGN}_job_*.root"
if ((DRY_RUN)); then
  echo "Dry run only: no jobs were submitted."
else
  echo "Monitor with: squeue --me"
  echo "Accounting:  sacct -j ${ARRAY_JOB_IDS[0]}"
fi
