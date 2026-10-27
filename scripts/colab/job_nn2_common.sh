# Neuron-atlas follow-up on a Colab GPU for the size in $NN2_SIZE:
#   A: direction steering (positive control) and group steering (scripts/neurons_followup_a.py)
#   C: capture + labeling runs on the main corpus (full and subsampled) whose exemplars feed the motif clusters
# Inputs uploaded as extra files: results/neurons/corpus/main.npz and results/neurons/$NN2_SIZE/poles.csv and
# results/neurons/followup/plan_c.csv. Every step is resumable (--resume); outputs sit one level below results/colab_out.
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
S=$NN2_SIZE
OUT=results/colab_out
python3 scripts/neurons_followup_a.py --size $S --poles results/neurons/$S/poles.csv --corpus results/neurons/corpus/main.npz \
  --out $OUT/nn2_a_$S --batch 128 --checkpoint-every 10 --resume
# C runs: one line per row of plan_c.csv for this size (tag, subsample, seed, batch)
python3 - <<'PY' > $OUT/nn2c_commands_$S.txt
import csv, os
S = os.environ["NN2_SIZE"]
for r in csv.DictReader(open("results/neurons/followup/plan_c.csv")):
    if r["size"] != S:
        continue
    sub = f"--subsample {r['subsample']} --subsample-seed {r['seed']}" if int(r["subsample"]) else ""
    print(f"python3 scripts/neurons_run.py --size {S} --corpus results/neurons/corpus/{r['corpus_file']} --out results/colab_out/nn2c_{r['tag']} "
          f"--layers-per-chunk 8 --batch {r['batch']} --resume {sub}")
PY
while read -r cmd; do
  echo "$cmd"
  bash -c "$cmd" || echo STEP_FAILED "$cmd"
done < $OUT/nn2c_commands_$S.txt
