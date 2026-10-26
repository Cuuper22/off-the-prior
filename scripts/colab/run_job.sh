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
# compute-unit balances are logged outside the repo (SPEND_DIR), one JSON line per job
SPEND_DIR=${OTP_SPEND_DIR:-/home/claude/off-the-prior-tasks/colab_spend}
mkdir -p "$SPEND_DIR"
bal() { colab usage 2>/dev/null | sed -n 's/.*balance: \([0-9.]*\).*/\1/p'; }
B0=$(bal || true)
date -u +%FT%TZ > "$OUT/started_utc.txt"
cleanup() {
  colab stop -s "$S" >/dev/null 2>&1 || true
  date -u +%FT%TZ > "$OUT/ended_utc.txt"
  B1=$(bal || true)
  echo "{\"job\": \"$JOB\", \"gpu\": \"$GPU\", \"start\": \"$(cat "$OUT/started_utc.txt")\", \"end\": \"$(cat "$OUT/ended_utc.txt")\", \"balance_before\": \"$B0\", \"balance_after\": \"$B1\"}" >> "$SPEND_DIR/colab_spend.jsonl"
}
trap cleanup EXIT
colab new -s "$S" --gpu "$GPU"
colab upload -s "$S" "$BUNDLE" /content/otp_bundle.tar.gz
colab exec -s "$S" -f "$ROOT/scripts/colab/remote_setup.py" | tee "$OUT/setup.txt"
colab upload -s "$S" "$JOBSH" /content/otp/job.sh
for f in "$@"; do  # extra files: LOCAL=REMOTE (REMOTE relative to the repo root on Colab)
  rdir=$(dirname "/content/otp/${f#*=}")
  echo "import os; os.makedirs('$rdir', exist_ok=True)" | colab exec -s "$S"
  if [ "$(stat -c %s "${f%%=*}")" -gt 41943040 ]; then  # the upload endpoint rejects large files: send 40 MB parts
    pd=$(mktemp -d); split -b 40M -d -a 3 "${f%%=*}" "$pd/part."
    for p in "$pd"/part.*; do colab upload -s "$S" "$p" "$rdir/$(basename "$p")"; done
    echo "import subprocess; subprocess.run('cat $rdir/part.* > /content/otp/${f#*=} && rm $rdir/part.*', shell=True, check=True)" | colab exec -s "$S"
    rm -rf "$pd"
  else
    colab upload -s "$S" "${f%%=*}" "/content/otp/${f#*=}"
  fi
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
