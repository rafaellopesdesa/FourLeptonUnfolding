#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-delphes
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/delphes-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/repository/Simulation"
export TMPDIR="$STUDY/runs/bootstrap/tmp/delphes-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
source /usr/share/lmod/lmod/init/bash
module purge
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$STUDY/runs/bootstrap/delphes-${SLURM_JOB_ID}.status"' EXIT
./install_delphes.sh --skip-apt --root-prerequisite mpich/4.2.1 --root-module root/6.30.06 --jobs "$SLURM_CPUS_PER_TASK" --prefix "$STUDY/install/delphes"
module purge
source ./env.sh
root-config --version
ldd "$DELPHES_ROOT/DelphesHepMC3" > "$STUDY/runs/bootstrap/delphes-${SLURM_JOB_ID}-ldd.txt"
if rg -q 'not found' "$STUDY/runs/bootstrap/delphes-${SLURM_JOB_ID}-ldd.txt"; then
  echo 'Unresolved detector runtime dependency' >&2
  exit 1
fi
