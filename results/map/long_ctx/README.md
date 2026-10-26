# results/map/long_ctx: context lengths up to 32,768 (Colab job B)

28 configs (two per generator across chaos, regime, tails, plus in-prior controls; see
`otp/grids/long_ctx.py`), 8 seeds each, all four Timestral sizes, contexts 2048 / 8192 / 32768,
horizons 16 / 128 / 1024 / 2048. References: the true-dynamics oracle and seasonal naive (ETS and
ARIMA fit only the last 1024 steps, so they are the same as in the 8192-step family sweeps).
Run on a Colab A100 by `scripts/colab/run_job.sh long_ctx scripts/colab/job_b_long.sh A100`
(job script `scripts/colab/job_b_long.sh`; log and timing in `results/colab/long_ctx/`).

`samples.npz` (122 MB) is not committed; the job regenerates it deterministically.
Oracle rows were rebuilt per context length by `scripts/repair_oracle_rows.py`
(`provenance_oracle_repair.json`). Two samples (`rossler/c5.7/s2`, `rossler/c5.7/s6`) have
oracle paths that diverged at long horizons (non-finite oracle quantiles from step 2035 and
step 195); their oracle-ratio rows are non-finite and are left out of every aggregate.
