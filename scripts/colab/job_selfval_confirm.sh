# Colab: confirmation of the frozen self-validation rule (K=2 windows, margin 0.7) on FRESH seeds
# (seed offset 100; none of these samples were used to choose the rule).
set -e
export OTP_DEVICE=cuda OTP_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for fam in in_prior chaos tails regime; do
  python3 scripts/sweep.py --grid otp.grids.$fam --seed-offset 100 --seeds 8 --sizes --classical \
    --contexts 2048 --horizons 128 --workers 12 --no-references --out results/colab_out/fresh_$fam
  OTP_THREADS=4 OMP_NUM_THREADS=4 python3 scripts/selfval_fix.py --samples results/colab_out/fresh_$fam/samples.npz \
    --sizes 5M 28M 123M 209M --T 2048 --H 128 --windows 2 --out results/colab_out/confirm_$fam
  rm -f results/colab_out/fresh_$fam/samples.npz
done
