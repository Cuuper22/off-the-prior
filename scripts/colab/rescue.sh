#!/usr/bin/env bash
# Fetch a finished Colab job's results through file operations and stop its runtime.
#   scripts/colab/rescue.sh <job_name>     (session otp-<job_name>)
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
JOB=$1; S=otp-$JOB
ROOT=$(git rev-parse --show-toplevel); OUT=$ROOT/results/colab/$JOB
mkdir -p "$OUT/results/colab_out"
timeout 120 colab download -s "$S" /content/job.exit "$OUT/job.exit"
timeout 300 colab download -s "$S" /content/job.log "$OUT/job.log"
for d in $(timeout 90 colab ls -s "$S" /content/otp/results/colab_out 2>/dev/null | grep '/$'); do
  mkdir -p "$OUT/results/colab_out/$d"
  for f in $(timeout 90 colab ls -s "$S" "/content/otp/results/colab_out/$d" 2>/dev/null | grep -v '/$'); do
    timeout 1800 colab download -s "$S" "/content/otp/results/colab_out/$d$f" "$OUT/results/colab_out/$d$f"
  done
done
colab stop -s "$S"
date -u +%FT%TZ > "$OUT/ended_utc.txt"
SPEND_DIR=${OTP_SPEND_DIR:-/home/claude/off-the-prior-tasks/colab_spend}
echo "{\"job\": \"$JOB (rescued by file download)\", \"start\": \"$(cat "$OUT/started_utc.txt" 2>/dev/null)\", \"end\": \"$(cat "$OUT/ended_utc.txt")\", \"balance_after\": \"$(colab usage 2>/dev/null | sed -n 's/.*balance: \([0-9.]*\).*/\1/p')\"}" >> "$SPEND_DIR/colab_spend.jsonl"
echo "rescued $JOB into $OUT"
