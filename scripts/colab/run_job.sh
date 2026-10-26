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
# poll through file listing (the kernel exec channel can hang on an idle runtime); every third
# poll, pull any results/colab_out/*/scores_partial.csv (sweep.py checkpoints) into $OUT/partial/,
# so a lost runtime costs at most one chunk: relaunch with the partial file as an extra upload
# (LOCAL=results/colab_out/<dir>/scores_partial.csv) and --resume in the job script.
n=0
while true; do
  n=$((n+1))
  if [ $((n % 3)) -eq 0 ]; then
    for d in $(timeout 90 colab ls -s "$S" /content/otp/results/colab_out 2>/dev/null | grep '/$'); do
      if timeout 90 colab ls -s "$S" "/content/otp/results/colab_out/$d" 2>/dev/null | grep -q scores_partial.csv; then
        mkdir -p "$OUT/partial/$d"
        timeout 600 colab download -s "$S" "/content/otp/results/colab_out/${d}scores_partial.csv" "$OUT/partial/${d}scores_partial.csv" >/dev/null 2>&1 || true
      fi
    done
  fi
  if timeout 90 colab ls -s "$S" /content 2>/dev/null | grep -q "job.exit"; then break; fi
  if ! timeout 90 colab sessions 2>/dev/null | grep -q "\[$S\]"; then echo "session $S is gone"; exit 3; fi
  echo "$(date -u +%T) running"
  python3 -c "import time; time.sleep(120)"
done
mkdir -p "$OUT/results/colab_out"
timeout 120 colab download -s "$S" /content/job.exit "$OUT/job.exit" || true
timeout 300 colab download -s "$S" /content/job.log "$OUT/job.log" || true
# fetch results/colab_out/<dir>/<file> by listing (two levels), no kernel exec needed
for d in $(timeout 90 colab ls -s "$S" /content/otp/results/colab_out 2>/dev/null | grep '/$'); do
  mkdir -p "$OUT/results/colab_out/$d"
  for f in $(timeout 90 colab ls -s "$S" "/content/otp/results/colab_out/$d" 2>/dev/null | grep -v '/$'); do
    timeout 1800 colab download -s "$S" "/content/otp/results/colab_out/$d$f" "$OUT/results/colab_out/$d$f"
  done
done
echo "results in $OUT"
