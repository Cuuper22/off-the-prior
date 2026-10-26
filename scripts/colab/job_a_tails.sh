# Colab job A (pilot on the tails grid): 123M and 209M on the VM sweep's exact samples.
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/sweep.py --grid otp.grids.tails --samples-from samples/tails --no-references \
  --sizes 123M 209M --contexts 512 2048 8192 --horizons 16 128 1024 --batch-size 32 \
  --out results/colab_out/tails_big
