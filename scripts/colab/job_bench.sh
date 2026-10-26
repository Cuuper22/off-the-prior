# Colab: build both benchmark sets and score the baselines on them.
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 bench/build.py --set diag --workers 12 --out results/colab_out/bench_data
python3 bench/build.py --set lb --workers 12 --out results/colab_out/bench_data
for set in diag lb; do
  for m in oracle seasonal_naive ets arima timestral-5M timestral-28M timestral-123M timestral-209M; do
    python3 bench/eval.py --set $set --split all --model $m --data results/colab_out/bench_data --workers 12 \
      --out results/colab_out/bench_lb_tmp
  done
done
# flatten to two levels for the driver: bench_lb/<model>_<set>_all__<file>
mkdir -p results/colab_out/bench_lb
for d in results/colab_out/bench_lb_tmp/*/; do
  b=$(basename "$d"); for f in "$d"*; do cp "$f" "results/colab_out/bench_lb/${b}__$(basename "$f")"; done
done
rm -rf results/colab_out/bench_lb_tmp
