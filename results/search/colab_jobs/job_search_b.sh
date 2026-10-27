# Equal-budget runs on 5M: MAP-Elites, 3 search seeds, 3000 Timestral calls per run (same budget as job_search_a.sh).
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/search.py --methods map_elites --seeds 0 1 2 --budget 3000 --size 5M --workers 11 --out results/colab_out --flat-prefix srch_ --resume
