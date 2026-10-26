# Colab job A (regime grid): all sizes on the VM sweep's exact samples (5M/28M double as a GPU-vs-CPU check).
set -e
export OTP_DEVICE=cuda OTP_THREADS=4
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/sweep.py --grid otp.grids.regime --samples-from samples/regime --no-references \
  --sizes 5M 28M 123M 209M --contexts 512 2048 8192 --horizons 16 128 1024 --batch-size 32 \
  --out results/colab_out/regime_big
