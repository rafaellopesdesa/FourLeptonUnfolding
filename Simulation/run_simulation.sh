#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HIGGS_BR_DEFAULT="2.771E-04"

usage() {
  cat <<'EOF'
Run Delphes with the bundled ATLAS card on generated HepMC events.

Usage:
  ./run_simulation.sh INPUT [options]

INPUT may be:
  - an events.hepmc3 or events.hepmc file;
  - a standalone Generation run directory containing one such file;
  - a Unity generation campaign containing jobs/job_*/ directories;
  - one individual Unity jobs/job_* directory.

Options:
  --process NAME       auto, gg_H, or ZZ (default: auto from run-metadata.txt)
  --output-root DIR    Put outputs below DIR/JOB_LABEL instead of beside input
  --card FILE          Override Delphes's bundled delphes_card_ATLAS.tcl
  --higgs-br VALUE     gg_H weight factor (default: 2.771E-04)
  --max-events N       Process at most N events per input; 0 means all (default)
  --max-files N        Process at most N discovered files; 0 means all (default)
  --overwrite          Replace an existing output from this script
  -h, --help           Show this help

Examples:
  ./run_simulation.sh ../Generation/runs/gg_H_pythia_seed101
  ./run_simulation.sh /work/pi_rclsa_umass_edu/FourLeptonUnfoldingGeneration/CAMPAIGN
  ./run_simulation.sh events.hepmc3 --process gg_H --output-root /work/output
EOF
}

if (($# == 1)) && [[ "$1" == -h || "$1" == --help ]]; then
  usage
  exit 0
fi
if (($# < 1)); then
  usage >&2
  exit 2
fi

INPUT="$1"
shift
PROCESS=auto
OUTPUT_ROOT=""
CARD=""
HIGGS_BR="$HIGGS_BR_DEFAULT"
MAX_EVENTS=0
MAX_FILES=0
OVERWRITE=0

need_value() {
  (($# >= 2)) || {
    echo "Option $1 requires a value" >&2
    exit 2
  }
}

while (($#)); do
  case "$1" in
    --process) need_value "$@"; PROCESS="$2"; shift 2 ;;
    --output-root) need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --card) need_value "$@"; CARD="$2"; shift 2 ;;
    --higgs-br) need_value "$@"; HIGGS_BR="$2"; shift 2 ;;
    --max-events) need_value "$@"; MAX_EVENTS="$2"; shift 2 ;;
    --max-files) need_value "$@"; MAX_FILES="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$PROCESS" == auto || "$PROCESS" == gg_H || "$PROCESS" == ZZ ]] || {
  echo "--process must be auto, gg_H, or ZZ" >&2
  exit 2
}
[[ "$HIGGS_BR" =~ ^[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$ ]] || {
  echo "--higgs-br must be a positive number" >&2
  exit 2
}
awk -v value="$HIGGS_BR" 'BEGIN { exit !(value > 0) }' || {
  echo "--higgs-br must be positive" >&2
  exit 2
}
[[ "$MAX_EVENTS" =~ ^[0-9]+$ ]] || {
  echo "--max-events must be a non-negative integer" >&2
  exit 2
}
[[ "$MAX_FILES" =~ ^[0-9]+$ ]] || {
  echo "--max-files must be a non-negative integer" >&2
  exit 2
}

[[ -r "$SCRIPT_DIR/env.sh" ]] || {
  echo "Missing $SCRIPT_DIR/env.sh; run install_delphes.sh first." >&2
  exit 1
}
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

command -v python3 >/dev/null || {
  echo "python3 is required to prepare the fiducial-study Delphes card" >&2
  exit 1
}
[[ -x "$DELPHES_ROOT/DelphesHepMC2" && -x "$DELPHES_ROOT/DelphesHepMC3" ]] || {
  echo "Delphes HepMC readers are unavailable below $DELPHES_ROOT" >&2
  exit 1
}

INPUT="$(realpath -m "$INPUT")"
if [[ -z "$CARD" ]]; then
  CARD="${DELPHES_ATLAS_CARD:-$DELPHES_ROOT/cards/delphes_card_ATLAS.tcl}"
fi
CARD="$(realpath "$CARD")"
[[ -r "$CARD" ]] || {
  echo "Delphes card is not readable: $CARD" >&2
  exit 1
}
[[ -z "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"

declare -a INPUT_FILES=()
discover_single_directory() {
  local directory="$1"
  local -a matches=()
  [[ -s "$directory/events.hepmc3" ]] && matches+=("$directory/events.hepmc3")
  [[ -s "$directory/events.hepmc" ]] && matches+=("$directory/events.hepmc")
  ((${#matches[@]} == 1)) || {
    echo "Expected exactly one events.hepmc3 or events.hepmc in $directory" >&2
    return 1
  }
  INPUT_FILES+=("${matches[0]}")
}

if [[ -f "$INPUT" ]]; then
  [[ -s "$INPUT" ]] || { echo "Input file is empty: $INPUT" >&2; exit 1; }
  INPUT_FILES+=("$INPUT")
elif [[ -d "$INPUT/jobs" ]]; then
  while IFS= read -r -d '' job_dir; do
    [[ -f "$job_dir/SUCCESS" ]] || continue
    discover_single_directory "$job_dir"
    if ((MAX_FILES > 0 && ${#INPUT_FILES[@]} >= MAX_FILES)); then
      break
    fi
  done < <(find "$INPUT/jobs" -mindepth 1 -maxdepth 1 -type d -name 'job_*' -print0 | sort -z)
elif [[ -d "$INPUT" ]]; then
  discover_single_directory "$INPUT"
else
  echo "Input does not exist: $INPUT" >&2
  exit 1
fi

if ((MAX_FILES > 0 && ${#INPUT_FILES[@]} > MAX_FILES)); then
  INPUT_FILES=("${INPUT_FILES[@]:0:MAX_FILES}")
fi
((${#INPUT_FILES[@]} > 0)) || {
  echo "No completed generation inputs were found below $INPUT" >&2
  exit 1
}

metadata_value() {
  local key="$1"
  local file="$2"
  [[ -r "$file" ]] || return 0
  awk -F= -v wanted="$key" '$1 == wanted {print substr($0, index($0, "=") + 1); exit}' "$file"
}

detect_hepmc_format() {
  local file="$1"
  local header
  header="$(head -n 20 "$file")"
  # The serialization marker, not the library version, determines the reader.
  # In particular, Herwig linked against HepMC3 writes its default GenEvent
  # output with HepMC3::WriterAsciiHepMC2: the Version line starts with 3, but
  # the event record itself uses HepMC2 IO_GenEvent syntax.
  if grep -q 'HepMC::Asciiv3-START_EVENT_LISTING' <<<"$header"; then
    echo 3
  elif grep -q 'HepMC::IO_GenEvent-START_EVENT_LISTING' <<<"$header"; then
    echo 2
  elif grep -q 'HepMC::Version 3' <<<"$header"; then
    echo 3
  elif grep -q 'HepMC::Version 2' <<<"$header"; then
    echo 2
  elif [[ "$file" == *.hepmc3 ]]; then
    echo 3
  elif [[ "$file" == *.hepmc ]]; then
    echo 2
  else
    echo "Cannot identify HepMC2 or HepMC3 format for $file" >&2
    return 1
  fi
}

validate_delphes_output() {
  local output_file="$1"
  local log_file="$2"
  local expected_events="$3"
  DELPHES_OUTPUT_FILE="$output_file" \
    DELPHES_EXPECTED_EVENTS="$expected_events" \
    root -l -b -q "$SCRIPT_DIR/check_delphes_output.C" >>"$log_file" 2>&1
}

count_hepmc_events() {
  local input_file="$1"
  awk '$1 == "E" {count++} END {print count + 0}' "$input_file"
}

failures=0
completed=0
for input_file in "${INPUT_FILES[@]}"; do
  input_dir="$(dirname "$input_file")"
  metadata_file="$input_dir/run-metadata.txt"
  metadata_process="$(metadata_value process "$metadata_file")"
  resolved_process="$PROCESS"
  if [[ "$resolved_process" == auto ]]; then
    resolved_process="$metadata_process"
    [[ "$resolved_process" == gg_H || "$resolved_process" == ZZ ]] || {
      echo "Cannot infer process for $input_file; pass --process gg_H or --process ZZ." >&2
      failures=$((failures + 1))
      continue
    }
  elif [[ -n "$metadata_process" && "$metadata_process" != "$resolved_process" ]]; then
    echo "Process mismatch for $input_file: metadata=$metadata_process, option=$resolved_process" >&2
    failures=$((failures + 1))
    continue
  fi

  weight_scale=1.0
  [[ "$resolved_process" != gg_H ]] || weight_scale="$HIGGS_BR"
  format="$(detect_hepmc_format "$input_file")" || {
    failures=$((failures + 1))
    continue
  }
  reader="$DELPHES_ROOT/DelphesHepMC${format}"

  label="$(basename "$input_dir")"
  if [[ -n "$OUTPUT_ROOT" ]]; then
    output_dir="$OUTPUT_ROOT/$label"
  else
    output_dir="$input_dir/delphes_ATLAS"
  fi
  output_file="$output_dir/delphes.root"
  resolved_card="$output_dir/delphes_card_ATLAS_resolved.tcl"
  log_file="$output_dir/delphes.log"
  status_file="$output_dir/simulation-metadata.txt"

  input_events="$(count_hepmc_events "$input_file")"
  if ((input_events <= 0)); then
    echo "No HepMC event records were found in $input_file" >&2
    failures=$((failures + 1))
    continue
  fi
  expected_events="$input_events"
  if ((MAX_EVENTS > 0 && MAX_EVENTS < expected_events)); then
    expected_events="$MAX_EVENTS"
  fi

  if [[ -f "$output_dir/SUCCESS" && -s "$output_file" && $OVERWRITE -eq 0 ]]; then
    if validate_delphes_output "$output_file" "$log_file" "$expected_events"; then
      printf '[simulation] Already complete: %s\n' "$output_file"
    else
      echo "Existing SUCCESS output has an empty or invalid Delphes tree: $output_file" >&2
      echo "Rerun with --overwrite to produce the fiducial-study branches." >&2
      failures=$((failures + 1))
    fi
    continue
  fi
  mkdir -p "$output_dir"
  if ((OVERWRITE)); then
    rm -f "$output_file" "$resolved_card" "$log_file" "$status_file" \
      "$output_dir/SUCCESS" "$output_dir/FAILED"
  elif [[ -e "$output_file" || -e "$output_dir/FAILED" ]]; then
    echo "Existing incomplete output blocks $output_dir; inspect it or use --overwrite." >&2
    failures=$((failures + 1))
    continue
  fi

  python3 "$SCRIPT_DIR/prepare_fiducial_card.py" "$CARD" "$resolved_card"
  {
    printf '\n# Added by FourLeptonUnfolding/Simulation/run_simulation.sh\n'
    printf 'set WeightScale %.17g\n' "$weight_scale"
    ((MAX_EVENTS == 0)) || printf 'set MaxEvents %d\n' "$MAX_EVENTS"
  } >>"$resolved_card"

  printf '[simulation] %s -> %s (HepMC%s, process=%s, weight scale=%s)\n' \
    "$input_file" "$output_file" "$format" "$resolved_process" "$weight_scale"

  if "$reader" "$resolved_card" "$output_file" "$input_file" >"$log_file" 2>&1 && \
      [[ -s "$output_file" ]] && \
      validate_delphes_output "$output_file" "$log_file" "$expected_events"; then
    {
      printf 'input_file=%s\n' "$input_file"
      printf 'output_file=%s\n' "$output_file"
      printf 'process=%s\n' "$resolved_process"
      printf 'hepmc_format=%s\n' "$format"
      printf 'weight_scale=%s\n' "$weight_scale"
      printf 'higgs_br_h_to_zz_to_4l_including_taus=%s\n' "$HIGGS_BR"
      printf 'weight_branches_scaled=Event.Weight,Weight.Weight\n'
      printf 'cross_section_fields_scaled=Event.CrossSection,Event.CrossSectionError\n'
      printf 'input_events=%s\n' "$input_events"
      printf 'output_events=%s\n' "$expected_events"
      printf 'event_retention_validated=true\n'
      printf 'truth_particles=StableParticle(status_1,bare,no_photon_dressing)\n'
      printf 'loose_reco_leptons=RecoElectron,RecoMuon(pre_isolation)\n'
      printf 'reconstruction_marker=HasFourRecoLeptons\n'
      printf 'delphes_version=%s\n' "${DELPHES_VERSION:-unknown}"
      printf 'delphes_commit=%s\n' "$(git -C "$DELPHES_ROOT" rev-parse HEAD)"
      printf 'card=%s\n' "$CARD"
      printf 'card_sha256=%s\n' "$(sha256sum "$CARD" | awk '{print $1}')"
      printf 'max_events=%s\n' "$MAX_EVENTS"
      printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >"$status_file"
    touch "$output_dir/SUCCESS"
    completed=$((completed + 1))
  else
    touch "$output_dir/FAILED"
    echo "Delphes failed for $input_file; inspect $log_file" >&2
    failures=$((failures + 1))
  fi
done

printf '[simulation] Complete: %d new output(s), %d failure(s)\n' "$completed" "$failures"
((failures == 0))
