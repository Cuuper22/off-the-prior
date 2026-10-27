# Neuron audit, 123M. Run: scripts/colab/run_job.sh nn-123M scripts/colab/job_nn_123M.sh A100 \
#   results/neurons/corpus/main.npz=results/neurons/corpus/main.npz (and mv, short, long, xlong; real/real.npz, real/real_events.npy)
set -e
NN_SIZES="123M" bash scripts/colab/job_nn_common.sh
