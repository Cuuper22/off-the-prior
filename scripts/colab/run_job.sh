#!/usr/bin/env bash
# Run one Off-the-Prior job on a Colab GPU runtime and bring its results back.
#   scripts/colab/run_job.sh <job_name> <job.sh> <gpu> [extra files to upload...]
# The job script runs inside the unpacked repo on Colab and must write
# everything it produces under results/colab_out/; that tree is downloaded into
# results/colab/<job_name>/ (unpacked), together with usage before/after and the log.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
JOB=$1; JOBSH=$2; GPU=$3; shift 3
ROOT=$(git rev-parse --show-toplevel)
OUT=$ROOT/results/colab/$JOB
mkdir -p "$OUT"
S=otp-$JOB
BUNDLE=$(mktemp -d)/otp_bundle.tar.gz
git -C "$ROOT" archive --format=tar.gz -o "$BUNDLE" HEAD
colab usage > "$OUT/usage_before.txt" 2>&1 || true
date -u +%FT%TZ > "$OUT/started_utc.txt"
cleanup() { colab stop -s "$S" >/dev/null 2>&1 || true; colab usage > "$OUT/usage_after.txt" 2>&1 || true; date -u +%FT%TZ > "$OUT/ended_utc.txt"; }
trap cleanup EXIT
colab new -s "$S" --gpu "$GPU"
colab upload -s "$S" "$BUNDLE" /content/otp_bundle.tar.gz
colab exec -s "$S" -f "$ROOT/scripts/colab/remote_setup.py" | tee "$OUT/setup.txt"
colab upload -s "$S" "$JOBSH" /content/otp/job.sh
for f in "$@"; do  # extra files: LOCAL=REMOTE (REMOTE relative to the repo root on Colab)
  rdir=$(dirname "/content/otp/${f#*=}")
  echo "import os; os.makedirs('$rdir', exist_ok=True)" | colab exec -s "$S"
  colab upload -s "$S" "${f%%=*}" "/content/otp/${f#*=}"
done
colab exec -s "$S" -f "$ROOT/scripts/colab/remote_start.py"
while true; do
  st=$(colab exec -s "$S" -f "$ROOT/scripts/colab/remote_status.py" 2>&1 || true)
  echo "$(date -u +%T) $(echo "$st" | tail -1)"
  echo "$st" | grep -q "STATE done" && break
  python3 -c "import time; time.sleep(120)"
done
echo "$st" > "$OUT/final_status.txt"
echo 'import subprocess; subprocess.run(["bash","-c","cd /content/otp && mkdir -p results/colab_out && tar -czf /content/results.tar.gz results/colab_out && cp /content/job.log /content/otp_job.log"])' | colab exec -s "$S"
colab download -s "$S" /content/results.tar.gz "$OUT/results.tar.gz"
colab download -s "$S" /content/otp_job.log "$OUT/job.log"
tar -xzf "$OUT/results.tar.gz" -C "$OUT" && rm "$OUT/results.tar.gz"
echo "results in $OUT"
