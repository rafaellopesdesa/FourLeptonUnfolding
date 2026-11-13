#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-physics-checks
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/physics-checks-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/validation"
export TMPDIR="$STUDY/runs/bootstrap/tmp/physics-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$STUDY/runs/bootstrap/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$TMPDIR"
export FOURLEPTON_NATIVE_TEST_BUILDS="$TMPDIR/native-builds"
trap 'code=$?; rm -rf -- "$FOURLEPTON_NATIVE_TEST_BUILDS"; printf "exit_code=%s\n" "$code" > "$STUDY/runs/bootstrap/physics-${SLURM_JOB_ID}.status"' EXIT
ANALYSIS_PYTHON="$STUDY/repository/Analysis/.pixi/envs/default/bin/python"
components=(Analysis Tools Simulation Generation BatchSubmit Plotting SMEFT)
if (($#)); then components=("$@"); fi
for component in "${components[@]}"; do
  "$ANALYSIS_PYTHON" -m unittest discover -s "$component/tests" -v
done
