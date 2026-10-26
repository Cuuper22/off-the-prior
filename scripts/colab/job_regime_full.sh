# Colab: the full regime family sweep on the worker's samples (oracle, ETS/ARIMA/seasonal naive, all four sizes).
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/sweep.py --grid otp.grids.regime --samples-from samples/regime --sizes 5M 28M 123M 209M \
  --contexts 512 2048 8192 --horizons 16 128 1024 --workers 11 --batch-size 32 --out results/colab_out/regime
