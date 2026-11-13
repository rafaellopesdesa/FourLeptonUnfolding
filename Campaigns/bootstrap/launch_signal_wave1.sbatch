#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-signal-wave1
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/signal-wave1-setup-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
PROJECT=/project/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
cd "$STUDY/repository"
export PATH="/home/rclsa_umass_edu/.pixi/bin:$PATH"
export XDG_CACHE_HOME="$STUDY/runs/bootstrap/cache"
export PIXI_CACHE_DIR="$XDG_CACHE_HOME/pixi"
export RATTLER_CACHE_DIR="$XDG_CACHE_HOME/rattler"
export TMPDIR="$STUDY/runs/bootstrap/tmp"
export SBATCH_QOS=normal
# Fail closed unless all four 10k pilots have validated and been persisted.
"$STUDY/repository/Analysis/.pixi/envs/default/bin/python" - <<'PY'
from pathlib import Path
import hashlib, json
base = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
data = json.loads((base/'runs/pilot10k/gg_H-summary.json').read_text())
assert len(data['samples']) == 4
for record in data['samples']:
    assert not record['errors'] and len(record['files']) == 1
    item = record['files'][0]
    assert item['validation']['entries'] == 10000 and item['reconstructed'] >= 1000
    with Path(item['persisted_path']).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
print('Four 10k pilot configurations validated; production precondition satisfied.')
PY
work_used="$(du -sb "$STUDY" | awk '{print $1}')"
project_used="$(du -sb "$PROJECT" | awk '{print $1}')"
# Reserve 100 GB for at most 16 simultaneous full-chain workers plus this wave.
(( work_used + 100000000000 < 300000000000 ))
(( project_used + 10000000000 < 2000000000000 ))
base="$STUDY/runs/production_signal_wave1"
grid="$STUDY/runs/pilot/campaigns/gg_H_pythia_pilot_20261113/grid"
mkdir -p "$base/scratch"
dependencies=""
for configuration in pythia:nominal pythia:up pythia:down herwig:nominal; do
  shower="${configuration%:*}"; scale="${configuration#*:}"
  seed=30000001
  [[ "$shower" != herwig ]] || seed=40000001
  campaign="gg_H_${shower}_production_wave1_20261113"
  ./Generation/submit_generation.sh gg_H "$shower" \
    --jobs 100 --events-per-job 10000 --base-seed "$seed" --campaign "$campaign" \
    --output-root "$base/campaigns" --analysis-output-root "$base/shards/$scale" \
    --scratch-root "$base/scratch" --reuse-grid "$grid" \
    --muon-scale "$scale" --mass-region extended --partition cpu \
    --account pi_rclsa_umass_edu --time 01:00:00 --mem 3G --max-concurrent 4
  [[ "$scale" == nominal ]] || campaign+="_muon_scale_$scale"
  job_id="$(awk -F= '$1 == "array_job_ids" {print $2}' "$base/campaigns/$campaign/submission.txt")"
  [[ "$job_id" =~ ^[0-9]+$ ]]
  dependencies+=":$job_id"
done
sbatch --parsable --dependency="afterany$dependencies" --time=02:00:00 \
  "$STUDY/repository/Campaigns/bootstrap/pilot_summary.sbatch" gg_H \
  production_signal_wave1 production_wave1_20261113 10000 100
