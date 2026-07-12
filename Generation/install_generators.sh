#!/usr/bin/env bash
set -Eeuo pipefail

# Install a self-contained POWHEG-BOX-V2 + shower stack on Ubuntu 24.04.
# Dependencies are installed below PREFIX unless explicitly supplied by an
# environment module.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-${SCRIPT_DIR}/software}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${SCRIPT_DIR}/downloads}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
JOBS="${JOBS:-$(nproc)}"
SKIP_APT=0

GSL_VERSION="${GSL_VERSION:-2.8}"
GSL_MODULE="${GSL_MODULE:-}"
BOOST_VERSION="${BOOST_VERSION:-1.83.0}"
LHAPDF_VERSION="${LHAPDF_VERSION:-6.5.6}"
FASTJET_VERSION="${FASTJET_VERSION:-3.4.3}"
HEPMC3_VERSION="${HEPMC3_VERSION:-3.3.0}"
PYTHIA_VERSION="${PYTHIA_VERSION:-8.317}"
PYTHIA_REF="${PYTHIA_REF:-83a5c9986f30746cfb390435a9b10ac55a758fca}"
THEPEG_VERSION="${THEPEG_VERSION:-2.3.0}"
HERWIG_VERSION="${HERWIG_VERSION:-7.3.0}"
PDFSET="${PDFSET:-NNPDF31_nlo_as_0118}"
LHAPDF_SET_BASE_URL="${LHAPDF_SET_BASE_URL:-https://lhapdfsets.web.cern.ch/current}"

POWHEG_URL="${POWHEG_URL:-https://gitlab.com/POWHEG-BOX/V2/POWHEG-BOX-V2.git}"
POWHEG_REF="${POWHEG_REF:-master}"

usage() {
  cat <<'EOF'
Usage: ./install_generators.sh [options]

Options:
  --prefix DIR    Installation prefix (default: Generation/software)
  --jobs N        Parallel build jobs (default: number of CPU cores)
  --skip-apt      Do not install Ubuntu build prerequisites
  --gsl-module M  Try GSL environment module M; use "auto" for gsl/VERSION
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
    --gsl-module)
      (($# >= 2)) || {
        echo "--gsl-module requires a module name (for example gsl/2.8)." >&2
        exit 2
      }
      GSL_MODULE="$2"
      shift 2
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
GSL_PREFIX="$PREFIX"
GSL_RESOLVED_VERSION="$GSL_VERSION"
GSL_MODULE_ACTIVE=0

export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"
export LHAPDF_DATA_PATH="$PREFIX/share/LHAPDF"
export BOOST_ROOT="$PREFIX"

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
    gfortran git libbz2-dev liblzma-dev \
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

validate_lhapdf_install() {
  local config="$PREFIX/bin/lhapdf-config"
  [[ -x "$config" ]] || return 1

  local config_prefix include_dir lib_dir
  config_prefix="$(realpath -m "$("$config" --prefix)")"
  include_dir="$(realpath -m "$("$config" --includedir)")"
  lib_dir="$(realpath -m "$("$config" --libdir)")"

  [[ "$config_prefix" == "$PREFIX" ]] || return 1
  [[ -f "$include_dir/LHAPDF/LHAPDF.h" ]] || return 1
  [[ -e "$lib_dir/libLHAPDF.so" || \
     -f "$lib_dir/libLHAPDF.a" || \
     -e "$lib_dir/libLHAPDF.dylib" ]] || return 1
}

initialize_module_command() {
  type module >/dev/null 2>&1 && return 0

  local -a init_scripts=()
  if [[ -n "${MODULESHOME:-}" ]]; then
    init_scripts+=("$MODULESHOME/init/bash")
  fi
  init_scripts+=(
    /etc/profile.d/modules.sh
    /etc/profile.d/lmod.sh
    /etc/profile.d/z00_lmod.sh
    /usr/share/Modules/init/bash
    /usr/share/lmod/lmod/init/bash
  )

  local init_script
  for init_script in "${init_scripts[@]}"; do
    if [[ -r "$init_script" ]]; then
      # Environment Modules and Lmod intentionally modify this shell.
      # shellcheck disable=SC1090
      source "$init_script"
      type module >/dev/null 2>&1 && return 0
    fi
  done
  return 1
}

try_gsl_module() {
  [[ -n "$GSL_MODULE" ]] || return 1
  [[ "$GSL_MODULE" != auto ]] || GSL_MODULE="gsl/$GSL_VERSION"

  if ! initialize_module_command; then
    log "The module command is unavailable; building GSL locally instead"
    return 1
  fi

  log "Trying GSL environment module $GSL_MODULE"
  if ! module load "$GSL_MODULE"; then
    log "Could not load $GSL_MODULE; building GSL locally instead"
    return 1
  fi

  local gsl_config resolved_prefix resolved_version
  gsl_config="$(command -v gsl-config || true)"
  resolved_prefix="${gsl_config:+$("$gsl_config" --prefix 2>/dev/null || true)}"
  resolved_version="${gsl_config:+$("$gsl_config" --version 2>/dev/null || true)}"

  if [[ -z "$gsl_config" || -z "$resolved_prefix" || -z "$resolved_version" || \
        ! -d "$resolved_prefix/include/gsl" ]]; then
    log "Module $GSL_MODULE did not expose a usable gsl-config and headers; building GSL locally instead"
    module unload "$GSL_MODULE" >/dev/null 2>&1 || true
    return 1
  fi

  GSL_PREFIX="$(realpath -m "$resolved_prefix")"
  GSL_RESOLVED_VERSION="$resolved_version"
  GSL_MODULE_ACTIVE=1
  log "Using GSL $GSL_RESOLVED_VERSION from $GSL_MODULE ($GSL_PREFIX)"
}

install_gsl() {
  try_gsl_module && return

  GSL_PREFIX="$PREFIX"
  GSL_RESOLVED_VERSION="$GSL_VERSION"
  GSL_MODULE_ACTIVE=0
  local stamp="$PREFIX/.installed-gsl-$GSL_VERSION"
  if [[ -e "$stamp" ]]; then
    if [[ -x "$PREFIX/bin/gsl-config" ]]; then
      GSL_RESOLVED_VERSION="$("$PREFIX/bin/gsl-config" --version)"
    fi
    return
  fi
  local archive
  archive="$(download \
    "https://ftp.gnu.org/gnu/gsl/gsl-${GSL_VERSION}.tar.gz" \
    "gsl-${GSL_VERSION}.tar.gz")"
  local src="$BUILD_DIR/gsl-$GSL_VERSION"
  unpack_clean "$archive" "$src"
  log "Building GSL $GSL_VERSION"
  (
    cd "$src"
    ./configure --prefix="$PREFIX"
    make -j"$JOBS"
    make install
  )
  GSL_RESOLVED_VERSION="$("$PREFIX/bin/gsl-config" --version)"
  touch "$stamp"
}

install_boost() {
  local stamp="$PREFIX/.installed-boost-$BOOST_VERSION"
  [[ -e "$stamp" ]] && return

  local boost_version_underscore="${BOOST_VERSION//./_}"
  local archive
  archive="$(download \
    "https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${boost_version_underscore}.tar.bz2" \
    "boost_${boost_version_underscore}.tar.bz2")"
  local src="$BUILD_DIR/boost_$boost_version_underscore"
  unpack_clean "$archive" "$src"
  log "Building Boost $BOOST_VERSION headers and Boost.Test"
  (
    cd "$src"
    ./bootstrap.sh --prefix="$PREFIX" --with-libraries=test
    ./b2 -j"$JOBS" \
      --prefix="$PREFIX" \
      --layout=system \
      variant=release \
      threading=multi \
      link=shared,static \
      install
  )
  [[ -f "$PREFIX/include/boost/test/unit_test.hpp" ]] || {
    echo "Boost.Test headers were not installed below $PREFIX." >&2
    exit 1
  }
  [[ -e "$PREFIX/lib/libboost_unit_test_framework.so" || \
     -f "$PREFIX/lib/libboost_unit_test_framework.a" ]] || {
    echo "Boost unit_test_framework was not installed below $PREFIX/lib." >&2
    exit 1
  }
  touch "$stamp"
}

install_lhapdf() {
  local stamp="$PREFIX/.installed-lhapdf-$LHAPDF_VERSION"
  if [[ -e "$stamp" ]] && validate_lhapdf_install; then
    return
  fi
  if [[ -e "$stamp" ]]; then
    log "Existing LHAPDF stamp is invalid; rebuilding LHAPDF $LHAPDF_VERSION"
  fi
  local archive
  archive="$(download \
    "https://lhapdf.hepforge.org/downloads/LHAPDF-${LHAPDF_VERSION}.tar.gz" \
    "LHAPDF-${LHAPDF_VERSION}.tar.gz")"
  local src="$BUILD_DIR/LHAPDF-$LHAPDF_VERSION"
  unpack_clean "$archive" "$src"
  log "Building LHAPDF $LHAPDF_VERSION"
  (
    cd "$src"
    ./configure --prefix="$PREFIX" --disable-python
    make -j"$JOBS"
    make install
  )
  validate_lhapdf_install || {
    echo "LHAPDF headers or libraries were not installed correctly below $PREFIX." >&2
    exit 1
  }
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
      configure_optional_prefix "$help_text" --with-gsl "$GSL_PREFIX"
      configure_optional_prefix "$help_text" --with-boost "$PREFIX"
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
      configure_optional_prefix "$help_text" --with-gsl "$GSL_PREFIX"
      configure_optional_prefix "$help_text" --with-boost "$PREFIX"
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
  local pdfset="$1"
  [[ "$pdfset" =~ ^[A-Za-z0-9_.+-]+$ ]] || {
    echo "Invalid LHAPDF set name: $pdfset" >&2
    exit 1
  }

  local pdfdir="$PREFIX/share/LHAPDF"
  [[ -f "$pdfdir/$pdfset/$pdfset.info" ]] && return

  local archive
  archive="$(download \
    "$LHAPDF_SET_BASE_URL/${pdfset}.tar.gz" \
    "${pdfset}.tar.gz")"
  local stage="$BUILD_DIR/lhapdf-set-$pdfset"
  rm -rf "$stage"
  mkdir -p "$stage" "$pdfdir"
  log "Installing LHAPDF set $pdfset"
  tar --no-same-owner -xzf "$archive" -C "$stage"
  [[ -f "$stage/$pdfset/$pdfset.info" ]] || {
    echo "Downloaded archive for $pdfset does not contain $pdfset/$pdfset.info." >&2
    exit 1
  }
  rm -rf "$pdfdir/$pdfset"
  mv "$stage/$pdfset" "$pdfdir/"
}

install_pdfsets() {
  # Herwig 7.3 builds its default repository during `make install` and needs
  # both CT14 sets at that point. The analysis set is installed here as well.
  local pdfset
  for pdfset in CT14lo CT14nlo "$PDFSET"; do
    install_pdfset "$pdfset"
  done
}

install_powheg() {
  local root="$PREFIX/src/POWHEG-BOX-V2"
  local lhapdf_config="$PREFIX/bin/lhapdf-config"
  local fastjet_config="$PREFIX/bin/fastjet-config"
  validate_lhapdf_install || {
    echo "A complete local LHAPDF installation is required before building POWHEG." >&2
    exit 1
  }
  [[ -x "$fastjet_config" ]] || {
    echo "Local fastjet-config not found at $fastjet_config." >&2
    exit 1
  }

  local lhapdf_cppflags lhapdf_libdir powheg_cxx
  lhapdf_cppflags="$("$lhapdf_config" --cppflags)"
  lhapdf_libdir="$("$lhapdf_config" --libdir)"
  powheg_cxx="${CXX:-g++}"

  mkdir -p "$PREFIX/src"
  if [[ ! -d "$root/.git" ]]; then
    log "Cloning POWHEG-BOX-V2"
    git clone "$POWHEG_URL" "$root"
  fi
  git -C "$root" checkout "$POWHEG_REF"
  git -C "$root" submodule update --init gg_H ZZ

  for process in gg_H ZZ; do
    local process_dir="$root/$process"
    # The upstream process Makefiles write object and Fortran module files to
    # pre-existing directories, but do not create those directories themselves.
    # gg_H uses obj/ and mod/; ZZ with gfortran uses obj-gfortran/.
    mkdir -p \
      "$process_dir/obj" \
      "$process_dir/obj-gfortran" \
      "$process_dir/mod"
    log "Building POWHEG-BOX-V2/$process"
    LIBRARY_PATH="$lhapdf_libdir:${LIBRARY_PATH:-}" \
      make -C "$process_dir" -j1 \
        LHAPDF_CONFIG="$lhapdf_config" \
        FASTJET_CONFIG="$fastjet_config" \
        CXX="$powheg_cxx $lhapdf_cppflags" \
        pwhg_main
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
export GENERATION_GSL_PREFIX="$GSL_PREFIX"
export GENERATION_GSL_MODULE="$GSL_MODULE"
export GENERATION_BOOST_PREFIX="$PREFIX"
export BOOST_ROOT="$PREFIX"
export POWHEG_ROOT="$PREFIX/src/POWHEG-BOX-V2"
export GENERATION_PDFSET="$PDFSET"
export PATH="$GSL_PREFIX/bin:$PREFIX/bin:\${PATH}"
export LD_LIBRARY_PATH="$GSL_PREFIX/lib:$GSL_PREFIX/lib64:$PREFIX/lib:$PREFIX/lib64:\${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$GSL_PREFIX/lib/pkgconfig:$GSL_PREFIX/lib64/pkgconfig:$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:\${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$GSL_PREFIX:$PREFIX:\${CMAKE_PREFIX_PATH:-}"
export LHAPDF_DATA_PATH="$PREFIX/share/LHAPDF"
EOF
  chmod +x "$SCRIPT_DIR/env.sh"
}

write_versions() {
  local powheg_root="$PREFIX/src/POWHEG-BOX-V2"
  {
    printf 'GSL %s\n' "$GSL_RESOLVED_VERSION"
    if ((GSL_MODULE_ACTIVE)); then
      printf 'GSL module %s\n' "$GSL_MODULE"
    else
      printf 'GSL source local build\n'
    fi
    printf 'Boost %s (headers and Boost.Test)\n' "$BOOST_VERSION"
    printf 'LHAPDF %s\n' "$LHAPDF_VERSION"
    printf 'FastJet %s\n' "$FASTJET_VERSION"
    printf 'HepMC3 %s\n' "$HEPMC3_VERSION"
    printf 'PYTHIA %s\n' "$PYTHIA_VERSION"
    printf 'ThePEG %s\n' "$THEPEG_VERSION"
    printf 'Herwig %s\n' "$HERWIG_VERSION"
    printf 'Herwig default PDF sets CT14lo CT14nlo\n'
    printf 'PDF set %s\n' "$PDFSET"
    printf 'POWHEG-BOX-V2 %s\n' "$(git -C "$powheg_root" rev-parse HEAD)"
    printf 'POWHEG gg_H %s\n' "$(git -C "$powheg_root/gg_H" rev-parse HEAD)"
    printf 'POWHEG ZZ %s\n' "$(git -C "$powheg_root/ZZ" rev-parse HEAD)"
  } >"$PREFIX/versions.txt"
}

install_ubuntu_dependencies
install_gsl
install_boost
install_lhapdf
install_pdfsets
install_fastjet
install_hepmc3
install_pythia8
install_thepeg
install_herwig
install_powheg
install_pythia_driver
write_environment
write_versions

log "Installation complete"
log "Environment: source $SCRIPT_DIR/env.sh"
log "Version manifest: $PREFIX/versions.txt"
