# Colab smoke job for the search code: environment, timing, one short run of two methods and the input-space study.
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nproc; free -g | head -2; nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/search.py --methods cmaes gradient --seeds 0 --budget 120 --workers 11 --out results/colab_out --flat-prefix smoke_
python3 scripts/search.py --methods input_space --sizes 5M 209M --n-series 2 --out results/colab_out --flat-prefix smoke_
