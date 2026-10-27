# Equal-budget runs on 5M: CMA-ES and gradient search (a), 3 search seeds each, 3000 Timestral calls per run;
# the input-space study (b) runs beside them (it only uses the GPU). Runs already finished are skipped, partial runs resume.
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/search.py --methods input_space --sizes 5M 28M 123M 209M --n-series 20 --out results/colab_out --flat-prefix srchis_ --resume > results/colab_out/input_space.log 2>&1 &
IS=$!
python3 scripts/search.py --methods cmaes gradient --seeds 0 1 2 --budget 3000 --size 5M --workers 11 --out results/colab_out --flat-prefix srch_ --resume
wait $IS
cat results/colab_out/input_space.log
cp results/colab_out/input_space.log results/colab_out/srchis_input_space/input_space.log
