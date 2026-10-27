# Transfer: re-score the top 20 elites of each method (pooled over its 3 search seeds) on 209M, 123M, 28M and 5M with fresh seeds.
# The elites.json files of all runs are uploaded to results/search/runs/<run>/elites.json by the launch command.
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/search_transfer.py --runs results/search/runs --top 20 --workers 11 --out results/colab_out/srcht_transfer
