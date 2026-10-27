# Neuron-atlas follow-up, 123M. Run (from the repo root, master or the follow-up branch):
#   CODE_ONLY=1 scripts/colab/run_job.sh nn2-123M scripts/colab/job_nn2_123M.sh G4,H100,A100 \
#     results/neurons/corpus/main.npz=results/neurons/corpus/main.npz \
#     results/neurons/123M/poles.csv=results/neurons/123M/poles.csv \
#     results/neurons/followup/plan_c.csv=results/neurons/followup/plan_c.csv
set -e
NN2_SIZE=123M bash scripts/colab/job_nn2_common.sh
