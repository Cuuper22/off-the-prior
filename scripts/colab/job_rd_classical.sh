# Colab CPU job: the classical models (seasonal naive, ETS, ARIMA) on the chunks not yet scored (regression baseline).
# One worker process per core; each takes the chunks whose number is its index modulo the worker count.
# Inputs uploaded with the job: parts_have.tar.gz (chunks already scored elsewhere, skipped here) and the two
# *_restricted.npz window files (used for scoring only, not stored in the repo).
set -e
export OTP_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pip install -q pyarrow >/dev/null 2>&1 || true
N=$(nproc)
echo "cores $N"; python3 -c "import statsmodels, numpy, scipy; print('statsmodels', statsmodels.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
mkdir -p results/colab_out/rd_scores/parts
tar -xzf parts_have.tar.gz -C results/colab_out/rd_scores/parts
pids=""
for i in $(seq 0 $((N-1))); do
  python3 scripts/score_regression.py --sets analogues real_extra in_prior real --only classical --parts-only \
    --worker $i/$N --out results/colab_out/rd_scores > results/colab_out/rd_scores/worker_$i.log 2>&1 &
  pids="$pids $!"
done
rc=0; for p in $pids; do wait $p || rc=1; done
find results/colab_out/rd_scores/parts -name "*.csv" -newer parts_have.tar.gz -exec cp {} results/colab_out/rd_scores/ ;
ls results/colab_out/rd_scores | wc -l
exit $rc
