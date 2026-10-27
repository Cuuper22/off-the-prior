# Neuron audit for the sizes in $NN_SIZES on a Colab GPU: capture + labeling on every context variant, steering,
# failure-map link, attention heads. The corpora are uploaded as extra files (results/neurons/corpus/*.npz,
# results/neurons/real/real.npz and real_events.npy). Every run is resumable (--resume), so a lost runtime costs one chunk.
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CORP=results/neurons/corpus
OUT=results/colab_out
for S in $NN_SIZES; do
  python3 scripts/neurons_run.py --size $S --corpus $CORP/main.npz,$CORP/mv.npz --real results/neurons/real/real.npz \
    --out $OUT/nn_${S}_main --acts-sub 2048 --acts-dir $OUT/acts --layers-per-chunk 4 --batch 64 --resume
  python3 scripts/neurons_run.py --size $S --corpus $CORP/short.npz --out $OUT/nn_${S}_short --layers-per-chunk 4 --batch 64 --resume
  python3 scripts/neurons_run.py --size $S --corpus $CORP/long.npz --out $OUT/nn_${S}_long --layers-per-chunk 4 --batch 16 --resume
  python3 scripts/neurons_run.py --size $S --corpus $CORP/xlong.npz --out $OUT/nn_${S}_xlong --layers-per-chunk 4 --batch 4 --resume
  python3 scripts/neurons_steer.py --size $S --runs $OUT --corpus $CORP/main.npz --out $OUT/nn_steer
  python3 scripts/neurons_failure.py --size $S --runs $OUT --out $OUT/nn_failure
  python3 scripts/neurons_heads.py --size $S --corpus $CORP/main.npz --mv $CORP/mv.npz --out $OUT/nn_heads_$S
done
