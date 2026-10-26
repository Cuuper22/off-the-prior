# Colab: the full chaos family sweep (sample generation, oracle, ETS/ARIMA/seasonal naive, all four sizes).
set -e
export OTP_DEVICE=cuda OTP_THREADS=4
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/sweep.py --grid otp.grids.chaos --sizes 5M 28M 123M 209M --contexts 512 2048 8192 \
  --horizons 16 128 1024 2048 --seeds 8 --workers 11 --batch-size 32 --out results/colab_out/chaos
