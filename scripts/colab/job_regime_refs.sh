# Colab: oracle + ETS/ARIMA/seasonal naive references for the regime grid on the worker's samples
# (the Timestral rows for all four sizes come from job A, results/colab/regime_big).
set -e
export OTP_DEVICE=cuda OTP_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nproc
python3 scripts/sweep.py --grid otp.grids.regime --samples-from samples/regime --sizes \
  --contexts 512 2048 8192 --horizons 16 128 1024 --workers 12 --out results/colab_out/regime_refs
