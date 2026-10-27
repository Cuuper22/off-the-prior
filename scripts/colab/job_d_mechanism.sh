# Colab job D: mechanism measurements (readouts, attention, template match) at all four sizes.
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/mechanism_run.py --studies all --sizes 5M 28M 123M 209M --n 300 --n-paths 300 --workers 10 \
  --out results/colab_out/mechanism
