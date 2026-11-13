#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-build
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/generators-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/repository/Generation"
export TMPDIR="$STUDY/runs/bootstrap/tmp/generators-${SLURM_JOB_ID}"
export DOWNLOAD_DIR="$STUDY/install/downloads/generators"
export BUILD_DIR="$STUDY/install/build/generators"
mkdir -p "$TMPDIR"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$STUDY/runs/bootstrap/generators-${SLURM_JOB_ID}.status"' EXIT
./install_generators.sh --skip-apt --gsl-module gsl/2.8 --jobs "$SLURM_CPUS_PER_TASK" --prefix "$STUDY/install/generators"
