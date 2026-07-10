#!/usr/bin/env bash
set -Eeuo pipefail

# Install a self-contained POWHEG-BOX-V2 + shower stack on Ubuntu 24.04.
# Everything except apt build prerequisites is installed below PREFIX.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-${SCRIPT_DIR}/software}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${SCRIPT_DIR}/downloads}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
JOBS="${JOBS:-$(nproc)}"
SKIP_APT=0

LHAPDF_VERSION="${LHAPDF_VERSION:-6.5.6}"
FASTJET_VERSION="${FASTJET_VERSION:-3.4.3}"
HEPMC3_VERSION="${HEPMC3_VERSION:-3.3.0}"
PYTHIA_VERSION="${PYTHIA_VERSION:-8.317}"
PYTHIA_REF="${PYTHIA_REF:-83a5c9986f30746cfb390435a9b10ac55a758fca}"
THEPEG_VERSION="${THEPEG_VERSION:-2.3.0}"
HERWIG_VERSION="${HERWIG_VERSION:-7.3.0}"
PDFSET="${PDFSET:-NNPDF31_nlo_as_0118}"

POWHEG_URL="${POWHEG_URL:-https://gitlab.com/POWHEG-BOX/V2/POWHEG-BOX-V2.git}"
POWHEG_REF="${POWHEG_REF:-master}"

usage() {
  cat <<'EOF'
Usage: ./install_generators.sh [options]

Options:
  --prefix DIR    Installation prefix (default: Generation/software)
  --jobs N        Parallel build jobs (default: number of CPU cores)
  --skip-apt      Do not install Ubuntu build prerequisites
  -h, --help      Show this help

Environment variables can override all version variables near the top of the
script. Re-running the script is safe: completed components are skipped.
EOF
}

while (($#)); do
  case "$1" in
    --prefix)
      PREFIX="$(realpath -m "$2")"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --skip-apt)
      SKIP_APT=1
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

mkdir -p "$PREFIX" "$DOWNLOAD_DIR" "$BUILD_DIR"
PREFIX="$(realpath "$PREFIX")"
DOWNLOAD_DIR="$(realpath "$DOWNLOAD_DIR")"
BUILD_DIR="$(realpath "$BUILD_DIR")"

export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"
export LHAPDF_DATA_PATH="$PREFIX/share/LHAPDF"

log() {
  printf '[install] %s\n' "$*" >&2
}

install_ubuntu_dependencies() {
  ((SKIP_APT)) && return

  local apt=(apt-get)
  if ((EUID != 0)); then
    command -v sudo >/dev/null || {
      echo "sudo is required for apt installation; use --skip-apt if dependencies are already present." >&2
      exit 1
    }
    apt=(sudo apt-get)
  fi

  log "Installing Ubuntu 24.04 build prerequisites"
  "${apt[@]}" update
  DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y --no-install-recommends \
    autoconf automake bison build-essential ca-certificates cmake curl flex \
    gfortran git libboost-all-dev libbz2-dev libgsl-dev liblzma-dev \
    libreadline-dev libsqlite3-dev libssl-dev libtool libxml2-dev \
    patch pkg-config python3 python3-dev rsync tar uuid-dev xz-utils zlib1g-dev
}

download() {
  local url="$1"
  local output="$DOWNLOAD_DIR/$2"
  if [[ ! -s "$output" ]]; then
    log "Downloading $url"
    curl --fail --location --retry 4 --retry-delay 3 --output "$output.part" "$url"
    mv "$output.part" "$output"
  fi
  printf '%s\n' "$output"
}

unpack_clean() {
  local archive="$1"
  local source_dir="$2"
  rm -rf "$source_dir"
  mkdir -p "$source_dir"
  tar --no-same-owner -xf "$archive" --strip-components=1 -C "$source_dir"
}

configure_optional_prefix() {
  local configure_help="$1"
  local option="$2"
  local value="$3"
  if grep -q -- "$option" <<<"$configure_help"; then
    printf '%s=%s\n' "$option" "$value"
  fi
}

install_lhapdf() {
  local stamp="$PREFIX/.installed-lhapdf-$LHAPDF_VERSION"
  [[ -e "$stamp" ]] && return
  local archive
  archive="$(download \
    "https://lhapdf.hepforge.org/downloads/LHAPDF-${LHAPDF_VERSION}.tar.gz" \
    "LHAPDF-${LHAPDF_VERSION}.tar.gz")"
  local src="$BUILD_DIR/LHAPDF-$LHAPDF_VERSION"
  unpack_clean "$archive" "$src"
  log "Building LHAPDF $LHAPDF_VERSION"
  (
    cd "$src"
    ./configure --prefix="$PREFIX" --disable-pyext
    make -j"$JOBS"
    make install
  )
  touch "$stamp"
}

install_fastjet() {
  local stamp="$PREFIX/.installed-fastjet-$FASTJET_VERSION"
  [[ -e "$stamp" ]] && return
  local archive
  archive="$(download \
    "https://fastjet.fr/repo/fastjet-${FASTJET_VERSION}.tar.gz" \
    "fastjet-${FASTJET_VERSION}.tar.gz")"
  local src="$BUILD_DIR/fastjet-$FASTJET_VERSION"
  unpack_clean "$archive" "$src"
  log "Building FastJet $FASTJET_VERSION"
  (
    cd "$src"
    ./configure --prefix="$PREFIX" --enable-shared
    make -j"$JOBS"
    make install
  )
  touch "$stamp"
}

install_hepmc3() {
  local stamp="$PREFIX/.installed-hepmc3-$HEPMC3_VERSION"
  [[ -e "$stamp" ]] && return
  local archive
  archive="$(download \
    "https://gitlab.cern.ch/hepmc/HepMC3/-/archive/${HEPMC3_VERSION}/HepMC3-${HEPMC3_VERSION}.tar.gz" \
    "HepMC3-${HEPMC3_VERSION}.tar.gz")"
  local src="$BUILD_DIR/HepMC3-$HEPMC3_VERSION"
  unpack_clean "$archive" "$src"
  log "Building HepMC3 $HEPMC3_VERSION"
  cmake -S "$src" -B "$src/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DHEPMC3_ENABLE_ROOTIO=OFF \
    -DHEPMC3_ENABLE_PYTHON=OFF
  cmake --build "$src/build" --parallel "$JOBS"
  cmake --install "$src/build"
  touch "$stamp"
}

install_pythia8() {
  local stamp="$PREFIX/.installed-pythia-$PYTHIA_VERSION"
  [[ -e "$stamp" ]] && return
  local src="$BUILD_DIR/pythia-${PYTHIA_VERSION}"
  if [[ ! -d "$src/.git" ]]; then
    log "Cloning PYTHIA $PYTHIA_VERSION"
    git clone https://gitlab.com/Pythia8/releases.git "$src"
  fi
  git -C "$src" checkout "$PYTHIA_REF"
  log "Building PYTHIA release $PYTHIA_VERSION"
  (
    cd "$src"
    ./configure --prefix="$PREFIX" \
      --with-lhapdf6="$PREFIX" --with-hepmc3="$PREFIX"
    make -j"$JOBS"
    make install
  )
  touch "$stamp"
}

install_thepeg() {
  local stamp="$PREFIX/.installed-thepeg-$THEPEG_VERSION"
  [[ -e "$stamp" ]] && return
  local archive
  archive="$(download \
    "https://thepeg.hepforge.org/downloads/ThePEG-${THEPEG_VERSION}.tar.bz2" \
    "ThePEG-${THEPEG_VERSION}.tar.bz2")"
  local src="$BUILD_DIR/ThePEG-$THEPEG_VERSION"
  unpack_clean "$archive" "$src"
  log "Building ThePEG $THEPEG_VERSION"
  (
    cd "$src"
    local help_text
    local -a options
    help_text="$(./configure --help)"
    options=(--prefix="$PREFIX")
    while IFS= read -r option; do
      [[ -n "$option" ]] && options+=("$option")
    done < <(
      configure_optional_prefix "$help_text" --with-lhapdf "$PREFIX"
      configure_optional_prefix "$help_text" --with-hepmc "$PREFIX"
      configure_optional_prefix "$help_text" --with-hepmc3 "$PREFIX"
    )
    if grep -q -- '--with-hepmcversion' <<<"$help_text"; then
      options+=(--with-hepmcversion=3)
    fi
    ./configure "${options[@]}"
    make -j"$JOBS"
    make install
  )
  touch "$stamp"
}

install_herwig() {
  local stamp="$PREFIX/.installed-herwig-$HERWIG_VERSION"
  [[ -e "$stamp" ]] && return
  local archive
  archive="$(download \
    "https://herwig.hepforge.org/downloads/Herwig-${HERWIG_VERSION}.tar.bz2" \
    "Herwig-${HERWIG_VERSION}.tar.bz2")"
  local src="$BUILD_DIR/Herwig-$HERWIG_VERSION"
  unpack_clean "$archive" "$src"
  log "Building Herwig $HERWIG_VERSION"
  (
    cd "$src"
    local help_text
    local -a options
    help_text="$(./configure --help)"
    options=(--prefix="$PREFIX" --with-thepeg="$PREFIX")
    while IFS= read -r option; do
      [[ -n "$option" ]] && options+=("$option")
    done < <(
      configure_optional_prefix "$help_text" --with-lhapdf "$PREFIX"
      configure_optional_prefix "$help_text" --with-fastjet "$PREFIX"
      configure_optional_prefix "$help_text" --with-hepmc "$PREFIX"
      configure_optional_prefix "$help_text" --with-hepmc3 "$PREFIX"
    )
    ./configure "${options[@]}"
    make -j"$JOBS"
    make install
  )
  "$PREFIX/bin/Herwig" build
  touch "$stamp"
}

install_pdfset() {
  if "$PREFIX/bin/lhapdf" list --installed 2>/dev/null | grep -qx "$PDFSET"; then
    return
  fi
  log "Installing LHAPDF set $PDFSET"
  "$PREFIX/bin/lhapdf" update
  "$PREFIX/bin/lhapdf" install "$PDFSET"
}

install_powheg() {
  local root="$PREFIX/src/POWHEG-BOX-V2"
  mkdir -p "$PREFIX/src"
  if [[ ! -d "$root/.git" ]]; then
    log "Cloning POWHEG-BOX-V2"
    git clone "$POWHEG_URL" "$root"
  fi
  git -C "$root" checkout "$POWHEG_REF"
  git -C "$root" submodule update --init gg_H ZZ

  for process in gg_H ZZ; do
    log "Building POWHEG-BOX-V2/$process"
    make -C "$root/$process" -j1 pwhg_main
  done
}

install_pythia_driver() {
  local source="$SCRIPT_DIR/powheg_pythia8.cc"

  log "Building the POWHEG-to-PYTHIA8 HepMC3 driver"
  # Word splitting is intentional for the flags printed by the config helper.
  # shellcheck disable=SC2046
  c++ -O2 -std=c++17 \
    $("$PREFIX/bin/pythia8-config" --cxxflags) \
    "$source" -o "$PREFIX/bin/powheg-pythia8" \
    $("$PREFIX/bin/pythia8-config" --ldflags) \
    $("$PREFIX/bin/pythia8-config" --libs) \
    -L"$PREFIX/lib" -L"$PREFIX/lib64" -lHepMC3 \
    -Wl,-rpath,"$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib64"
}

write_environment() {
  cat >"$SCRIPT_DIR/env.sh" <<EOF
#!/usr/bin/env bash
export GENERATION_PREFIX="$PREFIX"
export POWHEG_ROOT="$PREFIX/src/POWHEG-BOX-V2"
export GENERATION_PDFSET="$PDFSET"
export PATH="$PREFIX/bin:\${PATH}"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:\${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:\${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$PREFIX:\${CMAKE_PREFIX_PATH:-}"
export LHAPDF_DATA_PATH="$PREFIX/share/LHAPDF"
EOF
  chmod +x "$SCRIPT_DIR/env.sh"
}

write_versions() {
  local powheg_root="$PREFIX/src/POWHEG-BOX-V2"
  {
    printf 'LHAPDF %s\n' "$LHAPDF_VERSION"
    printf 'FastJet %s\n' "$FASTJET_VERSION"
    printf 'HepMC3 %s\n' "$HEPMC3_VERSION"
    printf 'PYTHIA %s\n' "$PYTHIA_VERSION"
    printf 'ThePEG %s\n' "$THEPEG_VERSION"
    printf 'Herwig %s\n' "$HERWIG_VERSION"
    printf 'PDF set %s\n' "$PDFSET"
    printf 'POWHEG-BOX-V2 %s\n' "$(git -C "$powheg_root" rev-parse HEAD)"
    printf 'POWHEG gg_H %s\n' "$(git -C "$powheg_root/gg_H" rev-parse HEAD)"
    printf 'POWHEG ZZ %s\n' "$(git -C "$powheg_root/ZZ" rev-parse HEAD)"
  } >"$PREFIX/versions.txt"
}

install_ubuntu_dependencies
install_lhapdf
install_fastjet
install_hepmc3
install_pythia8
install_thepeg
install_herwig
install_pdfset
install_powheg
install_pythia_driver
write_environment
write_versions

log "Installation complete"
log "Environment: source $SCRIPT_DIR/env.sh"
log "Version manifest: $PREFIX/versions.txt"
