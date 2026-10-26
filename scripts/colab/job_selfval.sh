# Colab: the zero-training self-validation fix (Timestral vs. method of analogues, chosen by backtest)
# on every family's samples, all four sizes.
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=4
for fam in in_prior chaos tails regime; do
  python3 scripts/selfval_fix.py --samples samples/$fam/samples.npz --sizes 5M 28M 123M 209M --T 2048 --H 128 --windows 2 \
    --out results/colab_out/selfval_$fam
done
