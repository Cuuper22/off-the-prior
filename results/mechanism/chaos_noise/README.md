# results/mechanism/chaos_noise: how the chaos ratios depend on the noise level

Mackey-Glass (tau = 17), Lorenz (rho = 28) and Rossler (c = 5.7) with dynamic and observation
noise both set to 0.1%, 0.3%, 1%, 3% and 10% of the signal's reference scale (Rossler only up
to 1%: above that its noisy trajectories leave the attractor). 16 seeds, context 2048,
horizons 16 / 128 / 1024, all four Timestral sizes, oracle and classical references.
Grid: `otp/grids/chaos_noise.py`. Samples generated on the VM, models and references run on a
Colab A100 by `scripts/colab/job_chaos_noise.sh` (log in `results/colab/chaos_noise/`).
`summary.csv` from `python scripts/summarize.py results/mechanism/chaos_noise --out results/mechanism/chaos_noise/summary.csv`.
