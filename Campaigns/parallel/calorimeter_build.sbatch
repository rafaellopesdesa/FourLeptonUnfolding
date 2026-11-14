#!/usr/bin/env bash
#SBATCH --job-name=fourlepton_calo_build
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/calorimeter_diagnosis_20261114/logs/build-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
DIAG="$STUDY/runs/calorimeter_diagnosis_20261114"
PREFIX="$STUDY/install/delphes-neutral-energy-20261114"
test ! -e "$PREFIX"
used=$(du -sb "$STUDY" | cut -f1)
test "$((used + 50000000000))" -lt 300000000000
cp -a "$STUDY/install/delphes" "$PREFIX"
"$DIAG/runtime/Simulation/install_delphes.sh" --skip-apt --jobs 1 \
  --root-module root/6.30.06 --root-prerequisite mpich/4.2.1 --prefix "$PREFIX"
cp "$STUDY/runs/runtime_snapshots/e07de718c4d3967c8efd8f752f5ed1f282be9305/Generation/env.sh" \
  "$DIAG/runtime/Generation/env.sh"
find "$DIAG/runtime" -type f -exec chmod a-w {} +
sha256sum "$PREFIX/src/Delphes-3.5.1/DelphesHepMC2" "$PREFIX/src/Delphes-3.5.1/DelphesHepMC3"
touch "$DIAG/BUILD_SUCCESS"
