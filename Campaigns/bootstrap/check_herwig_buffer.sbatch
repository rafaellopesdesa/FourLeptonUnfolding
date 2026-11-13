#!/usr/bin/env bash
#SBATCH --job-name=fourlepton-herwig-buffer
#SBATCH --account=pi_rclsa_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy/runs/bootstrap/logs/herwig-buffer-%j.log
set -Eeuo pipefail
STUDY=/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy
source "$STUDY/repository/Generation/env.sh"
export TMPDIR="$STUDY/runs/bootstrap/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
buffer_dir="$STUDY/runs/herwig_eof/buffer-${SLURM_JOB_ID}"
mkdir "$buffer_dir"
cp "$STUDY/runs/pilot/campaigns/gg_H_pythia_pilot_20261113/grid"/pwg*.dat "$buffer_dir/"
cp "$STUDY/runs/production_signal_wave1/campaigns/gg_H_herwig_production_wave1_20261113/jobs/job_000004_seed40000005/diagnostics/powheg.input" "$buffer_dir/"
"$STUDY/repository/Analysis/.pixi/envs/default/bin/python" - "$buffer_dir/powheg.input" <<'PY'
from pathlib import Path
import re, sys
p=Path(sys.argv[1]); s,n=re.subn(r'(?m)^(\s*numevts\s+)\S+', r'\g<1>10100',p.read_text())
assert n==1
p.write_text(s)
PY
cd "$buffer_dir"
"$POWHEG_ROOT/gg_H/pwhg_main" > powheg.log 2>&1
# Confirm that the extra hard events do not change the original source prefix.
"$STUDY/repository/Analysis/.pixi/envs/default/bin/python" - "$STUDY" "$buffer_dir" <<'PY'
from pathlib import Path
import hashlib, sys
study, out=map(Path,sys.argv[1:])
original=list((study/'runs/herwig_eof/campaigns/herwig_eof_diagnostic_20261113/jobs/job_000000_seed40000005/intermediates').glob('*/generation/pwgevents.lhe'))
assert len(original)==1
def prefix_hash(path):
 h=hashlib.sha256(); active=False; count=0
 with path.open('rb') as f:
  for line in f:
   if line.strip()==b'<event>': active=True
   if active: h.update(line)
   if line.strip()==b'</event>':
    count+=1; active=False
    if count==10000: return h.hexdigest()
 raise ValueError('Incomplete hard-event prefix')
assert prefix_hash(original[0])==prefix_hash(out/'pwgevents.lhe')
print('First 10000 hard events match the failed seed exactly.',flush=True)
PY
"$STUDY/repository/Generation/run_generation.sh" gg_H herwig \
  --lhe "$buffer_dir/pwgevents.lhe" --events 10000 --seed 40000005 \
  --output-dir "$buffer_dir" --pdf-id 303400
"$STUDY/repository/Analysis/.pixi/envs/default/bin/python" - "$buffer_dir" <<'PY'
from pathlib import Path
import json, sys
p=Path(sys.argv[1]); d=json.loads((p/'lhe-provenance.json').read_text())
assert d['lhe_event_count']==10100 and d['hepmc_event_count']==10000
assert d['hepmc_unique_source_events']==10000 and d['hepmc_repeated_source_events']==0
print(json.dumps(d,indent=2))
PY
