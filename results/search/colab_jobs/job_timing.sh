# Colab timing job for the search code: CPU count, then one short run of each method (pool and model already warm
# after the first), then the transfer script on a handful of elites to time scoring on each size.
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 -c "import os; print('cpu_count', os.cpu_count())"
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/search.py --methods cmaes map_elites gradient --seeds 0 --budget 360 --workers 11 --out results/colab_out --flat-prefix timing_
python3 scripts/search_transfer.py --runs "results/colab_out/timing_*_seed0/elites.json" --top 5 --workers 11 --out results/colab_out/timing_transfer
