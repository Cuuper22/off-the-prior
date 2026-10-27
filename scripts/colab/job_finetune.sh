# Colab (one A100): full-budget fine-tuning grid. Four sizes x protocols {lofo, knob} x held-out families x
# methods {full, lora} x replay shares {0, 0.3, 0.5} = 192 runs, each trained and scored; budgets are the ones in
# the config cell of notebooks/finetune_colab.ipynb.
# scripts/finetune_job.py writes results/colab_out/ft/scores_partial.csv after every run; run_job.sh pulls it down.
# To resume a lost session: relaunch with  LOCAL=results/colab_out/ft/scores_partial.csv  as an extra upload
# (the file is restored and finished runs are skipped).
set -e
export OTP_DEVICE=cuda OTP_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /content | tail -1
RESUME=""
if [ -f results/colab_out/ft/scores_partial.csv ]; then RESUME="--resume"; echo "resuming from scores_partial.csv"; fi
python3 scripts/finetune_job.py --out results/colab_out/ft --ckpt-root /content/ckpt_ft --pool-dir /content/pools_ft \
  --sizes 5M 28M 123M 209M --protocols lofo knob --holdouts chaos regime tails mixture \
  --methods full lora --replay-fracs 0 0.3 0.5 \
  --steps 5M=500 28M=500 123M=300 209M=300 --batch 5M=32 28M=32 123M=16 209M=16 \
  --lr-full 5M=5e-5 28M=3e-5 123M=2e-5 209M=1e-5 --lr-lora 5M=1e-3 28M=1e-3 123M=5e-4 209M=5e-4 \
  --rank 8 --alpha 16 --pool-batches 500 --pool-workers 11 --contexts 256 512 1024 2048 --horizons 32 128 512 \
  --n-eval 32 --n-paths 500 --real-npz bench/regression/real.npz --in-prior-npz bench/regression/in_prior.npz --threads 4 --checkpoint-every 1 $RESUME \
  --note "Colab run on one A100 through scripts/colab/run_job.sh (job script scripts/colab/job_finetune.sh). Budgets per size are in job.json." \
  --note "Regression sets: bench/regression/in_prior.npz (500 series with oracle) and bench/regression/real.npz (the windows committed in that file; real windows have no oracle)." \
  --note "Checkpoints were deleted after scoring (--keep-checkpoints not given); only the score tables, costs and loss curves came back."
