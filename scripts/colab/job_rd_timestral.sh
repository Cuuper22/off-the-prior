# Colab job: the four Timestral sizes on the extra real set and the analogue subset (regression baseline).
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
pip install -q pyarrow >/dev/null 2>&1 || true
python3 scripts/score_regression.py --sets real_extra analogues --only timestral --parts-only --out results/colab_out/rd_scores
# the driver downloads files two levels deep, so flatten the cached chunks
cp results/colab_out/rd_scores/parts/*.csv results/colab_out/rd_scores/
ls results/colab_out/rd_scores | head -50
