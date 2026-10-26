# Colab job B: all four sizes with contexts up to 32,768 on the long-context grid.
# Samples are generated here (T = 32768), with the true-dynamics oracle and seasonal naive as references
# (ETS/ARIMA fit only the last 1024 steps, so they would match the T = 8192 sweeps; skipped for time).
set -e
export OTP_DEVICE=cuda OTP_THREADS=4
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/sweep.py --grid otp.grids.long_ctx --sizes 5M 28M 123M 209M --contexts 2048 8192 32768 \
  --horizons 16 128 1024 2048 --seeds 8 --classical seasonal_naive --workers 10 --batch-size 8 \
  --out results/colab_out/long_ctx
