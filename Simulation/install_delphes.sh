#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$SCRIPT_DIR/software}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$SCRIPT_DIR/downloads}"
JOBS="${JOBS:-$(nproc)}"
SKIP_APT=0
ROOT_MODULE="${ROOT_MODULE:-}"

DELPHES_VERSION="${DELPHES_VERSION:-3.5.1}"
DELPHES_REF="${DELPHES_REF:-3.5.1}"
DELPHES_COMMIT="${DELPHES_COMMIT:-28658365abeb71ee36dfc739f9670c1514c0cb10}"
DELPHES_URL="${DELPHES_URL:-https://github.com/delphes/delphes.git}"
ROOT_VERSION="${ROOT_VERSION:-6.38.04}"
ATLAS_CARD_SHA256="${ATLAS_CARD_SHA256:-b6a97bcd6e2b4f19e218eaa07b3afe1937d66452f0519de0b8c367eff46fe69d}"
PATCH_FILES=(
  "$SCRIPT_DIR/patches/delphes-weight-scale.patch"
  "$SCRIPT_DIR/patches/delphes-truth-lepton-dressing.patch"
  "$SCRIPT_DIR/patches/delphes-ancestry-parton-stop.patch"
  "$SCRIPT_DIR/patches/delphes-mother-indices.patch"
)

usage() {
  cat <<'EOF'
Install Delphes and ROOT locally.

Usage: ./install_delphes.sh [options]

Options:
  --prefix DIR       Installation prefix (default: Simulation/software)
  --jobs N           Parallel build jobs (default: number of CPU cores)
  --skip-apt         Do not install Ubuntu build prerequisites
  --root-module M    Load ROOT environment module M before building;
                     use "auto" to try the unversioned module "root"
  -h, --help         Show this help

If root-config is already available, that ROOT installation is used. Otherwise
the script bootstraps a local ROOT environment with micromamba. Version values
near the top of the script can be overridden through environment variables.
EOF
}

while (($#)); do
  case "$1" in
    --prefix) (($# >= 2)) || { echo "--prefix requires a directory" >&2; exit 2; }; PREFIX="$2"; shift 2 ;;
    --jobs) (($# >= 2)) || { echo "--jobs requires an integer" >&2; exit 2; }; JOBS="$2"; shift 2 ;;
    --skip-apt) SKIP_APT=1; shift ;;
    --root-module) (($# >= 2)) || { echo "--root-module requires a module name" >&2; exit 2; }; ROOT_MODULE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--jobs must be a positive integer" >&2
  exit 2
}

PREFIX="$(realpath -m "$PREFIX")"
DOWNLOAD_DIR="$(realpath -m "$DOWNLOAD_DIR")"
DELPHES_ROOT="$PREFIX/src/Delphes-$DELPHES_VERSION"
ROOT_ENV="$PREFIX/root-$ROOT_VERSION"
mkdir -p "$PREFIX/bin" "$PREFIX/src" "$DOWNLOAD_DIR"

log() {
  printf '[install] %s\n' "$*" >&2
}

install_ubuntu_dependencies() {
  ((SKIP_APT)) && return
  local apt=(apt-get)
  if ((EUID != 0)); then
    command -v sudo >/dev/null || {
      echo "sudo is unavailable; use --skip-apt when build tools are already installed." >&2
      exit 1
    }
    apt=(sudo apt-get)
  fi
  log "Installing Ubuntu 24.04 build prerequisites"
  "${apt[@]}" update
  DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y --no-install-recommends \
    bzip2 build-essential ca-certificates curl git patch tar
}

initialize_module_command() {
  type module >/dev/null 2>&1 && return 0
  local init_script
  for init_script in \
    "${MODULESHOME:+$MODULESHOME/init/bash}" \
    /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh \
    /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
    [[ -n "$init_script" && -r "$init_script" ]] || continue
    # shellcheck disable=SC1090
    source "$init_script"
    type module >/dev/null 2>&1 && return 0
  done
  return 1
}

ROOT_MODULE_ACTIVE=0
try_root_module() {
  [[ -n "$ROOT_MODULE" ]] || return 1
  initialize_module_command || {
    log "The module command is unavailable"
    return 1
  }
  [[ "$ROOT_MODULE" != auto ]] || ROOT_MODULE=root
  log "Trying ROOT environment module $ROOT_MODULE"
  module load "$ROOT_MODULE" || return 1
  command -v root-config >/dev/null || {
    module unload "$ROOT_MODULE" >/dev/null 2>&1 || true
    return 1
  }
  ROOT_MODULE_ACTIVE=1
}

install_local_root() {
  local micromamba="$PREFIX/bin/micromamba"
  local archive="$DOWNLOAD_DIR/micromamba-linux-64.tar.bz2"
  if [[ ! -x "$micromamba" ]]; then
    if [[ ! -s "$archive" ]]; then
      log "Downloading micromamba"
      curl --fail --location --retry 4 --retry-delay 3 \
        --output "$archive.part" \
        https://micro.mamba.pm/api/micromamba/linux-64/latest
      mv "$archive.part" "$archive"
    fi
    tar -xjf "$archive" -C "$PREFIX" bin/micromamba
  fi

  if [[ ! -x "$ROOT_ENV/bin/root-config" ]]; then
    log "Installing ROOT $ROOT_VERSION without root privileges"
    MAMBA_ROOT_PREFIX="$PREFIX/micromamba" "$micromamba" create -y \
      --prefix "$ROOT_ENV" --channel conda-forge "root=$ROOT_VERSION"
  fi
  # shellcheck source=/dev/null
  source "$ROOT_ENV/bin/thisroot.sh"
}

install_ubuntu_dependencies
try_root_module || true
if ! command -v root-config >/dev/null; then
  install_local_root
fi

for command_name in c++ git make patch root root-config sha256sum; do
  command -v "$command_name" >/dev/null || {
    echo "Required command is unavailable: $command_name" >&2
    exit 1
  }
done

ROOT_PREFIX="$(realpath -m "$(root-config --prefix)")"
ROOT_BINDIR="$(realpath -m "$(root-config --bindir)")"
ROOT_LIBDIR="$(realpath -m "$(root-config --libdir)")"
ROOT_RESOLVED_VERSION="$(root-config --version)"
log "Using ROOT $ROOT_RESOLVED_VERSION from $ROOT_PREFIX"

if [[ ! -d "$DELPHES_ROOT/.git" ]]; then
  [[ ! -e "$DELPHES_ROOT" ]] || {
    echo "Existing non-git directory blocks installation: $DELPHES_ROOT" >&2
    exit 1
  }
  log "Cloning Delphes $DELPHES_VERSION"
  git clone --depth 1 --branch "$DELPHES_REF" "$DELPHES_URL" "$DELPHES_ROOT"
fi

RESOLVED_COMMIT="$(git -C "$DELPHES_ROOT" rev-parse HEAD)"
[[ "$RESOLVED_COMMIT" == "$DELPHES_COMMIT" ]] || {
  echo "Delphes source is at $RESOLVED_COMMIT, expected $DELPHES_COMMIT" >&2
  echo "Remove or move $DELPHES_ROOT before changing versions." >&2
  exit 1
}

patch_features_present() {
  local patch_name="$1"
  case "$patch_name" in
    delphes-weight-scale.patch)
      grep -Fq 'void DelphesHepMC2Reader::SetWeightScale(double weightScale)' \
        "$DELPHES_ROOT/classes/DelphesHepMC2Reader.cc" &&
        grep -Fq 'void DelphesHepMC3Reader::SetWeightScale(double weightScale)' \
          "$DELPHES_ROOT/classes/DelphesHepMC3Reader.cc"
      ;;
    delphes-truth-lepton-dressing.patch)
      grep -Fq 'fRequireNoHadronAncestor = GetBool(' \
        "$DELPHES_ROOT/modules/LeptonDressing.cc" &&
        grep -Fq 'Bool_t HasHadronAncestor(const Candidate *candidate) const;' \
          "$DELPHES_ROOT/modules/LeptonDressing.h"
      ;;
    delphes-ancestry-parton-stop.patch)
      grep -Fq 'Bool_t LeptonDressing::IsParton(Int_t pid) const' \
        "$DELPHES_ROOT/modules/LeptonDressing.cc" &&
        grep -Fq 'if(IsParton(ancestor->PID)) continue;' \
          "$DELPHES_ROOT/modules/LeptonDressing.cc"
      ;;
    delphes-mother-indices.patch)
      grep -Fq 'const Int_t second = candidate->M2;' \
        "$DELPHES_ROOT/modules/LeptonDressing.cc" &&
        grep -Fq 'if(second >= 0 && second < size && second != first)' \
          "$DELPHES_ROOT/modules/LeptonDressing.cc"
      ;;
    *)
      return 1
      ;;
  esac
}

for patch_file in "${PATCH_FILES[@]}"; do
  [[ -r "$patch_file" ]] || {
    echo "Missing Delphes patch: $patch_file" >&2
    exit 1
  }
  patch_name="$(basename "$patch_file")"
  if git -C "$DELPHES_ROOT" apply --reverse --check "$patch_file" >/dev/null 2>&1 ||
      patch_features_present "$patch_name"; then
    log "$patch_name is already applied"
  elif git -C "$DELPHES_ROOT" apply --check "$patch_file"; then
    log "Applying $patch_name"
    git -C "$DELPHES_ROOT" apply "$patch_file"
  else
    echo "$patch_name does not apply cleanly to $DELPHES_ROOT" >&2
    exit 1
  fi
done

CARD="$DELPHES_ROOT/cards/delphes_card_ATLAS.tcl"
CARD_CHECKSUM="$(sha256sum "$CARD" | awk '{print $1}')"
[[ "$CARD_CHECKSUM" == "$ATLAS_CARD_SHA256" ]] || {
  echo "Unexpected ATLAS card checksum: $CARD_CHECKSUM" >&2
  exit 1
}

log "Building Delphes $DELPHES_VERSION"
make -C "$DELPHES_ROOT" -j "$JOBS"
[[ -x "$DELPHES_ROOT/DelphesHepMC2" && -x "$DELPHES_ROOT/DelphesHepMC3" ]] || {
  echo "Delphes build did not produce both HepMC readers" >&2
  exit 1
}

ROOT_INIT_MODE=paths
if ((ROOT_MODULE_ACTIVE)); then
  ROOT_INIT_MODE=module
elif [[ -r "$ROOT_PREFIX/bin/thisroot.sh" ]]; then
  ROOT_INIT_MODE=thisroot
fi

cat >"$SCRIPT_DIR/env.sh" <<EOF
#!/usr/bin/env bash
# Generated by install_delphes.sh
export DELPHES_ROOT="$DELPHES_ROOT"
export DELPHES_VERSION="$DELPHES_VERSION"
export DELPHES_ATLAS_CARD="$CARD"
export DELPHES_ROOT_INIT_MODE="$ROOT_INIT_MODE"
export DELPHES_ROOT_MODULE="$ROOT_MODULE"
export DELPHES_ROOT_PREFIX="$ROOT_PREFIX"

if [[ "\$DELPHES_ROOT_INIT_MODE" == module ]]; then
  if ! type module >/dev/null 2>&1; then
    for init_script in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
      /etc/profile.d/z00_lmod.sh /usr/share/Modules/init/bash \
      /usr/share/lmod/lmod/init/bash; do
      if [[ -r "\$init_script" ]]; then
        source "\$init_script"
        type module >/dev/null 2>&1 && break
      fi
    done
  fi
  module load "\$DELPHES_ROOT_MODULE"
elif [[ "\$DELPHES_ROOT_INIT_MODE" == thisroot ]]; then
  source "$ROOT_PREFIX/bin/thisroot.sh"
else
  export PATH="$ROOT_BINDIR:\$PATH"
  export LD_LIBRARY_PATH="$ROOT_LIBDIR:\${LD_LIBRARY_PATH:-}"
fi

export PATH="\$DELPHES_ROOT:\$PATH"
export LD_LIBRARY_PATH="\$DELPHES_ROOT:\${LD_LIBRARY_PATH:-}"
EOF
chmod +x "$SCRIPT_DIR/env.sh"

cat >"$PREFIX/versions.txt" <<EOF
delphes_version=$DELPHES_VERSION
delphes_commit=$DELPHES_COMMIT
root_version=$ROOT_RESOLVED_VERSION
root_prefix=$ROOT_PREFIX
root_init_mode=$ROOT_INIT_MODE
atlas_card=$CARD
atlas_card_sha256=$CARD_CHECKSUM
weight_patch_sha256=$(sha256sum "${PATCH_FILES[0]}" | awk '{print $1}')
truth_lepton_dressing_patch_sha256=$(sha256sum "${PATCH_FILES[1]}" | awk '{print $1}')
ancestry_parton_stop_patch_sha256=$(sha256sum "${PATCH_FILES[2]}" | awk '{print $1}')
mother_indices_patch_sha256=$(sha256sum "${PATCH_FILES[3]}" | awk '{print $1}')
EOF

log "Installation complete"
log "Environment: source $SCRIPT_DIR/env.sh"
log "ATLAS card: $CARD"
log "Version manifest: $PREFIX/versions.txt"
