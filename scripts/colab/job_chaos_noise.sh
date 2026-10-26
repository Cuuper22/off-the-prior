# Colab: noise-level axis for chaotic generators (all four sizes, references included), on the VM's samples.
set -e
export OTP_DEVICE=cuda OTP_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 scripts/sweep.py --grid otp.grids.chaos_noise --samples-from samples/chaos_noise --sizes 5M 28M 123M 209M \
  --contexts 2048 --horizons 16 128 1024 --workers 12 --batch-size 32 --out results/colab_out/chaos_noise
