# Failure-link step for 123M and 209M with the final neurons_failure.py (extras: label tables, long corpus, cached inputs).
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for S in 123M 209M; do
  python3 scripts/neurons_failure.py --size $S --runs results/colab_out --out results/colab_out/nn_failure || echo STEP_FAILED failure $S
done
