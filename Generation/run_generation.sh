#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"
POWHEG_CARD_DIR="$REPO_ROOT/PowhegCards"

usage() {
  cat <<'EOF'
Run POWHEG-BOX-V2 and shower the resulting LHE file.

Usage:
  ./run_generation.sh PROCESS SHOWER [options]

Required positional arguments:
  PROCESS             gg_H or ZZ
  SHOWER              pythia or herwig

Options:
  --events N          Number of events (default: 1000; 0 means all input LHE events)
  --seed N            Positive random seed (default: 1)
  --run-card FILE     Override the default PowhegCards/PROCESS.powheg.input card
  --lhe FILE          Skip POWHEG and shower this existing LHE file
  --output-dir DIR    Run directory (default: Generation/runs/PROCESS_SHOWER_seedSEED)
  --pdf-id ID         LHAPDF member ID written to lhans1/lhans2 (default: 303400)
  --keep-grids        Reuse POWHEG integration grids found in the output directory
  -h, --help          Show this help

Examples:
  ./run_generation.sh gg_H pythia --events 10000 --seed 101
  ./run_generation.sh gg_H herwig --lhe runs/gg_H_pythia_seed101/pwgevents.lhe
  ./run_generation.sh ZZ pythia --events 10000 --seed 201
  ./run_generation.sh ZZ herwig --events 10000 --seed 301

For a shower comparison, generate an LHE sample once and pass that same file to
both shower choices with --lhe.
EOF
}

if (($# < 2)); then
  usage >&2
  exit 2
fi

PROCESS="$1"
SHOWER="$2"
shift 2

EVENTS=1000
SEED=1
RUN_CARD=""
INPUT_LHE=""
OUTPUT_DIR=""
PDF_ID="${PDF_ID:-303400}"
KEEP_GRIDS=0
SHOWER_EVENTS=0

while (($#)); do
  case "$1" in
    --events)
      EVENTS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --run-card)
      RUN_CARD="$(realpath "$2")"
      shift 2
      ;;
    --lhe)
      INPUT_LHE="$(realpath "$2")"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$(realpath -m "$2")"
      shift 2
      ;;
    --pdf-id)
      PDF_ID="$2"
      shift 2
      ;;
    --keep-grids)
      KEEP_GRIDS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$PROCESS" == "gg_H" || "$PROCESS" == "ZZ" ]] || {
  echo "PROCESS must be gg_H or ZZ" >&2
  exit 2
}
[[ "$SHOWER" == "pythia" || "$SHOWER" == "herwig" ]] || {
  echo "SHOWER must be pythia or herwig" >&2
  exit 2
}
[[ "$EVENTS" =~ ^[0-9]+$ ]] || {
  echo "--events must be a non-negative integer" >&2
  exit 2
}
[[ "$SEED" =~ ^[1-9][0-9]*$ ]] || {
  echo "--seed must be a positive integer" >&2
  exit 2
}

if [[ ! -f "$SCRIPT_DIR/env.sh" ]]; then
  echo "Missing $SCRIPT_DIR/env.sh; run install_generators.sh first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh"

PROCESS_DIR="$POWHEG_ROOT/$PROCESS"
HERWIG_PDFSET="${GENERATION_PDFSET:-NNPDF31_nlo_as_0118}"
[[ -x "$PROCESS_DIR/pwhg_main" ]] || {
  echo "Missing $PROCESS_DIR/pwhg_main; rerun install_generators.sh." >&2
  exit 1
}

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$SCRIPT_DIR/runs/${PROCESS}_${SHOWER}_seed${SEED}"
fi
mkdir -p "$OUTPUT_DIR"

log() {
  printf '[run] %s\n' "$*"
}

set_powheg_value() {
  local key="$1"
  local value="$2"
  local file="$3"
  local tmp="${file}.tmp"
  awk -v wanted="$key" -v replacement="$value" '
    BEGIN { found = 0 }
    {
      first = $1
      sub(/^#+/, "", first)
      if (!found && first == wanted) {
        comment = ""
        bang = index($0, "!")
        if (bang > 0) comment = " " substr($0, bang)
        print wanted " " replacement comment
        found = 1
      } else {
        print
      }
    }
    END {
      if (!found) print wanted " " replacement
    }
  ' "$file" >"$tmp"
  mv "$tmp" "$file"
}

choose_default_run_card() {
  printf '%s/%s.powheg.input\n' "$POWHEG_CARD_DIR" "$PROCESS"
}

run_powheg() {
  if [[ -z "$RUN_CARD" ]]; then
    RUN_CARD="$(choose_default_run_card)"
  fi
  [[ -n "$RUN_CARD" && -f "$RUN_CARD" ]] || {
    echo "No POWHEG run card found; pass one with --run-card." >&2
    exit 1
  }
  if [[ "$PROCESS" == ZZ ]] && \
      awk '$1 == "m4lmin" && $2 + 0 > 0 {found=1} END {exit !found}' "$RUN_CARD"; then
    grep -Fq 'powheginput("#m4lmin")' "$PROCESS_DIR/Born_phsp.f" || {
      echo "The ZZ card requests m4lmin, but the installed process lacks its source patch." >&2
      echo "Rerun Generation/install_generators.sh before generating this sample." >&2
      exit 1
    }
  fi

  cp "$RUN_CARD" "$OUTPUT_DIR/powheg.input"
  set_powheg_value numevts "$EVENTS" "$OUTPUT_DIR/powheg.input"
  set_powheg_value iseed "$SEED" "$OUTPUT_DIR/powheg.input"
  set_powheg_value lhans1 "$PDF_ID" "$OUTPUT_DIR/powheg.input"
  set_powheg_value lhans2 "$PDF_ID" "$OUTPUT_DIR/powheg.input"
  if ((KEEP_GRIDS)); then
    set_powheg_value use-old-grid 1 "$OUTPUT_DIR/powheg.input"
    set_powheg_value use-old-ubound 1 "$OUTPUT_DIR/powheg.input"
  else
    set_powheg_value use-old-grid 0 "$OUTPUT_DIR/powheg.input"
    set_powheg_value use-old-ubound 0 "$OUTPUT_DIR/powheg.input"
  fi

  log "Running POWHEG-BOX-V2/$PROCESS in $OUTPUT_DIR"
  (
    cd "$OUTPUT_DIR"
    "$PROCESS_DIR/pwhg_main" >powheg.log 2>&1
  )

  INPUT_LHE="$OUTPUT_DIR/pwgevents.lhe"
  if [[ ! -s "$INPUT_LHE" ]]; then
    INPUT_LHE="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'pwgevents*.lhe' -print -quit)"
  fi
  [[ -n "$INPUT_LHE" && -s "$INPUT_LHE" ]] || {
    echo "POWHEG did not produce an LHE file; inspect $OUTPUT_DIR/powheg.log." >&2
    exit 1
  }
}

run_pythia() {
  local output="$OUTPUT_DIR/events.hepmc3"
  log "Showering $(basename "$INPUT_LHE") with PYTHIA 8"
  "$GENERATION_PREFIX/bin/powheg-pythia8" \
    "$INPUT_LHE" "$output" "$SHOWER_EVENTS" "$SEED" "$PROCESS" \
    >"$OUTPUT_DIR/pythia.log" 2>&1
  [[ -s "$output" ]] || {
    echo "PYTHIA did not produce $output; inspect $OUTPUT_DIR/pythia.log." >&2
    exit 1
  }
  log "Wrote $output"
}

write_herwig_input() {
  local output="$OUTPUT_DIR/events.hepmc"
  cat >"$OUTPUT_DIR/herwig.in" <<EOF
cd /Herwig/Generators
set EventGenerator:NumberOfEvents $SHOWER_EVENTS
set EventGenerator:RandomNumberGenerator:Seed $SEED
set EventGenerator:DebugLevel 0
set EventGenerator:PrintEvent 0
set EventGenerator:MaxErrors 10000

cd /Herwig/EventHandlers
library LesHouches.so
create ThePEG::LesHouchesEventHandler LesHouchesHandler
set LesHouchesHandler:PartonExtractor /Herwig/Partons/PPExtractor
set LesHouchesHandler:CascadeHandler /Herwig/Shower/ShowerHandler
set LesHouchesHandler:DecayHandler /Herwig/Decays/DecayHandler
set LesHouchesHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler
set LesHouchesHandler:WeightOption VarNegWeight
set /Herwig/Generators/EventGenerator:EventHandler /Herwig/EventHandlers/LesHouchesHandler

# Do not add cuts after the POWHEG hard-event generation.
create ThePEG::Cuts /Herwig/Cuts/NoCuts

# Use the same NLO PDF set for the LHE hard process. Herwig retains its tuned
# default shower/MPI PDFs unless a later physics configuration changes them.
create ThePEG::LHAPDF /Herwig/Partons/LHAPDF ThePEGLHAPDF.so
set /Herwig/Partons/LHAPDF:PDFName $HERWIG_PDFSET
set /Herwig/Partons/RemnantDecayer:AllowTop Yes
set /Herwig/Partons/LHAPDF:RemnantHandler /Herwig/Partons/HadronRemnants
set /Herwig/Particles/p+:PDF /Herwig/Partons/LHAPDF
set /Herwig/Particles/pbar-:PDF /Herwig/Partons/LHAPDF
set /Herwig/Partons/PPExtractor:FirstPDF /Herwig/Partons/LHAPDF
set /Herwig/Partons/PPExtractor:SecondPDF /Herwig/Partons/LHAPDF

create ThePEG::LesHouchesFileReader LesHouchesReader
set LesHouchesReader:FileName $INPUT_LHE
set LesHouchesReader:AllowedToReOpen No
set LesHouchesReader:InitPDFs 0
set LesHouchesReader:Cuts /Herwig/Cuts/NoCuts
set LesHouchesReader:MomentumTreatment RescaleEnergy
set LesHouchesReader:PDFA /Herwig/Partons/LHAPDF
set LesHouchesReader:PDFB /Herwig/Partons/LHAPDF
insert LesHouchesHandler:LesHouchesReaders 0 LesHouchesReader

# POWHEG's hardest emission is already present in the LHE file. These settings
# restrict the Herwig shower to the phase space below the LHE hard scale.
set /Herwig/Shower/ShowerHandler:MaxPtIsMuF Yes
set /Herwig/Shower/ShowerHandler:RestrictPhasespace Yes
set /Herwig/Shower/PartnerFinder:PartnerMethod Random
set /Herwig/Shower/PartnerFinder:ScaleChoice Partner

read snippets/HepMC.in
set /Herwig/Analysis/HepMC:Filename $output
set /Herwig/Analysis/HepMC:PrintEvent 10000000
EOF

  if [[ "$PROCESS" == "gg_H" ]]; then
    cat >>"$OUTPUT_DIR/herwig.in" <<'EOF'

# Baseline forced decay: H -> ZZ(*) and Z -> e/mu/tau.
do /Herwig/Particles/h0:SelectDecayModes h0->Z0,Z0;
do /Herwig/Particles/Z0:SelectDecayModes /Herwig/Particles/Z0/Z0->e-,e+; /Herwig/Particles/Z0/Z0->mu-,mu+; /Herwig/Particles/Z0/Z0->tau-,tau+;
EOF
  fi

  cat >>"$OUTPUT_DIR/herwig.in" <<'EOF'

saverun FourLeptonLHERun /Herwig/Generators/EventGenerator
EOF
}

run_herwig() {
  local output="$OUTPUT_DIR/events.hepmc"
  write_herwig_input
  log "Preparing Herwig run"
  (
    cd "$OUTPUT_DIR"
    "$GENERATION_PREFIX/bin/Herwig" read herwig.in >herwig-read.log 2>&1
    "$GENERATION_PREFIX/bin/Herwig" run FourLeptonLHERun.run \
      -N "$SHOWER_EVENTS" --seed="$SEED" >herwig.log 2>&1
  )
  [[ -s "$output" ]] || {
    echo "Herwig did not produce $output; inspect herwig-read.log and herwig.log." >&2
    exit 1
  }
  log "Wrote $output"
}

if [[ -n "$INPUT_LHE" ]]; then
  [[ -s "$INPUT_LHE" ]] || {
    echo "Input LHE file is missing or empty: $INPUT_LHE" >&2
    exit 1
  }
  log "Using existing LHE file $INPUT_LHE"
else
  ((EVENTS > 0)) || {
    echo "--events 0 is only valid together with --lhe." >&2
    exit 2
  }
  run_powheg
fi

if ((EVENTS == 0)); then
  SHOWER_EVENTS="$(awk '/<event([[:space:]>])/{count++} END{print count+0}' "$INPUT_LHE")"
  ((SHOWER_EVENTS > 0)) || {
    echo "Could not find any events in $INPUT_LHE" >&2
    exit 1
  }
else
  SHOWER_EVENTS="$EVENTS"
fi

case "$SHOWER" in
  pythia) run_pythia ;;
  herwig) run_herwig ;;
esac

cat >"$OUTPUT_DIR/run-metadata.txt" <<EOF
process=$PROCESS
shower=$SHOWER
events=$EVENTS
shower_events=$SHOWER_EVENTS
seed=$SEED
input_lhe=$INPUT_LHE
powheg_card=${RUN_CARD:-external-LHE}
powheg_commit=$(git -C "$POWHEG_ROOT" rev-parse HEAD)
process_commit=$(git -C "$PROCESS_DIR" rev-parse HEAD)
EOF

log "Run complete; metadata: $OUTPUT_DIR/run-metadata.txt"
