# results/map/chaos

Chaos and dynamics family, grid module `otp.grids.chaos`. Files in this folder:

- `scores.csv`: one row per sample x forecaster x context length T x horizon H (`oracle` is the true-dynamics oracle, a Monte Carlo continuation of the true process from its true state).
- `provenance.json`: command line, git commit, host, threads, wall time.
- `summary.csv`: per `[config_id, generator, knob_value, forecaster, T, H]`: `n` samples; for `mase_ratio_oracle`, `scrps_ratio_oracle` and `mase_ratio_classical` the geometric mean over seeds (`_gmean`), mean, std and 95% bootstrap interval; `cov80_mean` (share of horizon points inside the 10%-90% interval, 0.8 is calibrated), `mase_mean_raw` and `n_finite_ratio` (samples with a finite oracle ratio).
- `samples.npz`: the generated samples (see the regeneration note below).

Sweep: 344 samples, 33024 score rows, wall time 938.6 s (from `provenance.json`; threads 4, cuda True, git field `empty`).

## Regenerate

```
python scripts/sweep.py --grid otp.grids.chaos --sizes 5M 28M 123M 209M --contexts 512 2048 8192 --horizons 16 128 1024 2048 --seeds 8 --workers 11 --batch-size 32 --out results/map/chaos
python -m otp.grids.chaos summarize --dir results/map/chaos
python -m otp.generators.chaos calibrate --out results/map/chaos
python -m otp.generators.chaos lyapunov --out results/map/chaos
```

This README is written by `python -m otp.grids.chaos readme` from `summary.csv`, `scores.csv` and `provenance.json` (the Notes section is passed in with `--note`).

## Samples with non-finite oracle quantiles

In these samples at least one of the oracle's 500 noisy continuations diverged (the dynamic noise pushed the state off the attractor), so the oracle quantiles are NaN from some horizon step on. Their oracle `mase`/`scrps` are NaN; the ratios for them are dropped from the geometric means (`n_finite_ratio` in `summary.csv` counts the samples kept) and the oracle `cov80` is blanked. Timestral and classical rows on these samples are finite.

| sample_id | T | H | oracle rows with NaN mase |
|---|---|---|---|
| rossler/c6.5/s2 | 512 | 2048 | 1 |
| rossler/c6.5/s2 | 2048 | 2048 | 1 |
| rossler/c6.5/s2 | 8192 | 2048 | 1 |
| rossler/c6.5/s4 | 512 | 2048 | 1 |
| rossler/c6.5/s4 | 2048 | 2048 | 1 |
| rossler/c6.5/s4 | 8192 | 2048 | 1 |
| rossler/c9/s0 | 512 | 1024 | 1 |
| rossler/c9/s0 | 512 | 2048 | 1 |
| rossler/c9/s0 | 2048 | 1024 | 1 |
| rossler/c9/s0 | 2048 | 2048 | 1 |
| rossler/c9/s0 | 8192 | 1024 | 1 |
| rossler/c9/s0 | 8192 | 2048 | 1 |
| rossler/c9/s4 | 512 | 1024 | 1 |
| rossler/c9/s4 | 512 | 2048 | 1 |
| rossler/c9/s4 | 2048 | 1024 | 1 |
| rossler/c9/s4 | 2048 | 2048 | 1 |
| rossler/c9/s4 | 8192 | 1024 | 1 |
| rossler/c9/s4 | 8192 | 2048 | 1 |

## Notes

- The scores in `scores.csv` were produced on a Colab A100 (host and package versions in `provenance.json`; its `git` field is empty because the job ran outside a git checkout), with the generators, grid and `scripts/sweep.py` from this branch. The job log and timings are in `colab_job/`. All four Timestral sizes (5M, 28M, 123M, 209M) are in this one file; there is no separate big-model run.
- Sample identity: `colab_job/vm_reference_hashes.json` holds SHA-256 hashes of 3 samples (`mackey_glass/tau17/s1`, `kuramoto/K1/s2`, `logistic_map/r3.9/s0`) generated on the VM with this worktree's code; the lead verified that they match the Colab samples byte for byte.
- `samples.npz` (138 MB) is not committed (`.gitignore` in this folder). Sample generation is seeded and deterministic: the first command above re-creates it, together with the scores.
- Oracle rows are stored per context length T (1376 rows each for T = 512, 2048, 8192), so `scripts/repair_oracle_rows.py` was not needed.
- A first sweep of the 5M/28M grid on the VM CPU was stopped by the out-of-memory handler after sample generation reached `kuramoto/K5`; it saved no files and none of its output is used.
- Classical models (`seasonal_naive`, `ets`, `arima`) get seasonal period 1 for every config (the generators set no integer seasonal period), so `seasonal_naive` here is the plain naive forecast. They fit the last 1024 steps of the context at every T.
- Contexts T = 32768 are not in this folder. The long-context chaos runs (the 10 config ids in `otp/grids/chaos_long.py`) were done in Colab job B at all sizes; the results are on master in `results/map/long_ctx/` (filter `config_id` to those ids). `chaos_long` was not run separately here.
- Largest Lyapunov exponents per config (two-trajectory separation, no noise) are in `lyapunov.csv`; `calibration.csv` holds the observed-signal means and standard deviations behind each generator's `sd_ref`.
- The Colab job log (`colab_job/job.log`) contains numpy overflow warnings from the Rossler right-hand side; the log does not say which samples they came from.

## Main curves

Ratio columns are `mase_ratio_oracle_gmean` (geometric mean over seeds of the forecaster's MASE divided by the oracle's MASE on the same sample; 1 means as good as the oracle) copied from `summary.csv`. `cov80` columns are `cov80_mean`.

### T = 2048, H = 128

| config_id | knob | timestral-5M ratio | timestral-28M ratio | timestral-123M ratio | timestral-209M ratio | seasonal_naive ratio | ets ratio | arima ratio | timestral-5M cov80 | timestral-28M cov80 | timestral-123M cov80 | timestral-209M cov80 | seasonal_naive cov80 | ets cov80 | arima cov80 | oracle cov80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kuramoto/K0 | 0 | 17.224 | 12.585 | 9.897 | 8.403 | 33.619 | 77.805 | 32.704 | 0.774 | 0.842 | 0.851 | 0.825 | 0.666 | 0.894 | 0.782 | 0.805 |
| kuramoto/K0.5 | 0.5 | 14.316 | 9.274 | 8.731 | 8.084 | 35.140 | 49.314 | 26.488 | 0.823 | 0.902 | 0.857 | 0.832 | 0.719 | 0.920 | 0.760 | 0.756 |
| kuramoto/K1 | 1 | 12.742 | 10.584 | 9.892 | 9.447 | 37.368 | 47.929 | 28.293 | 0.771 | 0.792 | 0.794 | 0.778 | 0.708 | 0.964 | 0.781 | 0.776 |
| kuramoto/K1.5 | 1.5 | 9.891 | 8.359 | 5.979 | 5.607 | 61.854 | 151.643 | 23.404 | 0.837 | 0.787 | 0.872 | 0.861 | 0.621 | 0.854 | 0.771 | 0.733 |
| kuramoto/K2 | 2 | 5.848 | 3.779 | 2.857 | 2.331 | 83.254 | 182.573 | 21.588 | 0.832 | 0.840 | 0.819 | 0.880 | 0.622 | 0.858 | 0.830 | 0.775 |
| kuramoto/K3 | 3 | 4.183 | 2.435 | 1.711 | 1.423 | 102.777 | 559.583 | 11.051 | 0.932 | 0.856 | 0.796 | 0.851 | 0.702 | 0.674 | 0.864 | 0.785 |
| kuramoto/K5 | 5 | 3.769 | 2.287 | 1.596 | 1.475 | 128.930 | 393.773 | 9.457 | 0.923 | 0.835 | 0.789 | 0.809 | 0.612 | 0.806 | 0.751 | 0.781 |
| kuramoto/K8 | 8 | 5.184 | 2.278 | 1.859 | 1.429 | 132.496 | 487.382 | 13.546 | 0.850 | 0.877 | 0.750 | 0.853 | 0.644 | 0.749 | 0.702 | 0.814 |
| logistic_map/r2.8 | 2.8 | 1.004 | 1.008 | 1.007 | 1.008 | 1.231 | 1.010 | 1.012 | 0.823 | 0.820 | 0.819 | 0.824 | 0.999 | 0.828 | 0.824 | 0.821 |
| logistic_map/r3.2 | 3.2 | 1.071 | 1.152 | 1.032 | 1.021 | 33.687 | 32.912 | 1.096 | 0.726 | 0.721 | 0.793 | 0.821 | 1.000 | 1.000 | 0.927 | 0.817 |
| logistic_map/r3.5 | 3.5 | 1.356 | 1.320 | 1.166 | 1.165 | 30.104 | 27.136 | 24.875 | 0.727 | 0.719 | 0.817 | 0.839 | 1.000 | 0.968 | 0.810 | 0.854 |
| logistic_map/r3.56 | 3.56 | 0.942 | 1.112 | 1.121 | 1.128 | 8.156 | 7.264 | 6.939 | 0.651 | 0.555 | 0.701 | 0.721 | 1.000 | 0.829 | 0.836 | 0.806 |
| logistic_map/r3.6 | 3.6 | 1.072 | 1.088 | 1.064 | 1.073 | 4.862 | 3.977 | 3.807 | 0.521 | 0.562 | 0.727 | 0.763 | 1.000 | 0.848 | 0.861 | 0.804 |
| logistic_map/r3.7 | 3.7 | 1.030 | 1.025 | 1.024 | 1.021 | 1.417 | 1.104 | 1.089 | 0.746 | 0.778 | 0.769 | 0.763 | 0.999 | 0.827 | 0.829 | 0.792 |
| logistic_map/r3.83 | 3.83 | 1.014 | 1.008 | 0.994 | 0.996 | 1.399 | 1.035 | 1.026 | 0.774 | 0.833 | 0.801 | 0.784 | 0.999 | 0.833 | 0.840 | 0.787 |
| logistic_map/r3.9 | 3.9 | 1.049 | 1.036 | 1.036 | 1.035 | 1.613 | 1.055 | 1.051 | 0.757 | 0.786 | 0.776 | 0.777 | 0.999 | 0.840 | 0.837 | 0.807 |
| logistic_map/r4 | 4 | 1.031 | 1.034 | 1.027 | 1.024 | 1.310 | 1.036 | 1.035 | 0.757 | 0.808 | 0.793 | 0.802 | 0.999 | 0.758 | 0.762 | 0.818 |
| lorenz63/rho10 | 10 | 1.004 | 1.010 | 1.007 | 1.005 | 1.610 | 1.011 | 1.011 | 0.794 | 0.792 | 0.785 | 0.786 | 0.991 | 0.780 | 0.780 | 0.790 |
| lorenz63/rho15 | 15 | 1.016 | 1.010 | 1.007 | 1.007 | 1.389 | 1.016 | 1.016 | 0.817 | 0.819 | 0.816 | 0.820 | 1.000 | 0.811 | 0.812 | 0.813 |
| lorenz63/rho20 | 20 | 1.067 | 1.038 | 1.024 | 1.005 | 1.481 | 1.086 | 1.078 | 0.805 | 0.809 | 0.801 | 0.818 | 0.995 | 0.810 | 0.814 | 0.814 |
| lorenz63/rho24 | 24 | 1.510 | 1.481 | 1.413 | 1.435 | 1.697 | 1.697 | 1.442 | 0.750 | 0.780 | 0.776 | 0.784 | 0.982 | 0.982 | 0.746 | 0.824 |
| lorenz63/rho26 | 26 | 1.563 | 1.537 | 1.533 | 1.512 | 2.046 | 2.046 | 1.613 | 0.773 | 0.788 | 0.762 | 0.790 | 0.977 | 0.977 | 0.739 | 0.860 |
| lorenz63/rho28 | 28 | 1.310 | 1.275 | 1.269 | 1.241 | 1.861 | 1.861 | 1.331 | 0.766 | 0.794 | 0.782 | 0.792 | 0.984 | 0.984 | 0.741 | 0.772 |
| lorenz63/rho28_oe2 | 28 | 29.891 | 30.894 | 27.489 | 27.906 | 48.378 | 79.197 | 39.673 | 0.802 | 0.812 | 0.810 | 0.808 | 0.546 | 0.901 | 0.725 | 0.855 |
| lorenz63/rho28_oe20 | 28 | 1.226 | 1.207 | 1.184 | 1.183 | 1.477 | 1.453 | 1.230 | 0.762 | 0.791 | 0.783 | 0.775 | 0.996 | 0.993 | 0.743 | 0.818 |
| lorenz63/rho28_oe5 | 28 | 2.996 | 2.767 | 2.672 | 2.648 | 3.950 | 6.776 | 3.862 | 0.775 | 0.825 | 0.820 | 0.812 | 0.881 | 0.976 | 0.955 | 0.820 |
| lorenz63/rho35 | 35 | 1.401 | 1.341 | 1.325 | 1.305 | 1.811 | 1.811 | 1.387 | 0.786 | 0.807 | 0.803 | 0.809 | 0.980 | 0.980 | 0.757 | 0.809 |
| lorenz63/rho45 | 45 | 1.263 | 1.238 | 1.201 | 1.183 | 1.688 | 1.688 | 1.260 | 0.779 | 0.799 | 0.800 | 0.795 | 0.973 | 0.972 | 0.753 | 0.752 |
| mackey_glass/tau10 | 10 | 1.471 | 1.322 | 1.225 | 1.159 | 24.983 | 85.980 | 10.843 | 0.912 | 0.797 | 0.797 | 0.783 | 0.813 | 0.921 | 0.979 | 0.801 |
| mackey_glass/tau14 | 14 | 2.445 | 1.740 | 1.762 | 1.665 | 29.198 | 72.828 | 13.746 | 0.641 | 0.762 | 0.709 | 0.707 | 0.593 | 0.843 | 0.839 | 0.832 |
| mackey_glass/tau17 | 17 | 9.574 | 6.314 | 6.797 | 6.666 | 26.706 | 43.078 | 19.448 | 0.635 | 0.795 | 0.792 | 0.746 | 0.697 | 0.923 | 0.780 | 0.819 |
| mackey_glass/tau20 | 20 | 9.541 | 8.264 | 5.180 | 4.393 | 27.527 | 54.326 | 18.527 | 0.595 | 0.750 | 0.884 | 0.882 | 0.558 | 0.854 | 0.743 | 0.775 |
| mackey_glass/tau25 | 25 | 8.157 | 9.947 | 7.495 | 8.431 | 22.239 | 33.451 | 12.445 | 0.734 | 0.752 | 0.784 | 0.748 | 0.527 | 0.893 | 0.792 | 0.703 |
| mackey_glass/tau30 | 30 | 21.871 | 21.382 | 12.877 | 13.677 | 42.722 | 49.685 | 29.192 | 0.638 | 0.682 | 0.744 | 0.764 | 0.501 | 0.924 | 0.739 | 0.830 |
| mackey_glass/tau6 | 6 | 1.899 | 1.345 | 1.120 | 1.093 | 16.740 | 37.357 | 4.772 | 0.512 | 0.620 | 0.694 | 0.717 | 0.897 | 0.892 | 0.698 | 0.739 |
| rossler/c2 | 2 | 1.685 | 1.361 | 1.306 | 1.267 | 13.221 | 16.687 | 3.597 | 0.727 | 0.785 | 0.865 | 0.873 | 0.780 | 0.918 | 0.951 | 0.878 |
| rossler/c3 | 3 | 1.528 | 1.257 | 1.150 | 1.172 | 11.132 | 29.900 | 5.256 | 0.738 | 0.797 | 0.861 | 0.827 | 0.820 | 0.906 | 0.889 | 0.845 |
| rossler/c4 | 4 | 2.342 | 2.249 | 2.151 | 2.002 | 12.332 | 21.130 | 5.419 | 0.604 | 0.584 | 0.603 | 0.680 | 0.768 | 0.913 | 0.865 | 0.756 |
| rossler/c4.5 | 4.5 | 2.125 | 1.827 | 1.763 | 1.489 | 10.070 | 19.624 | 5.860 | 0.722 | 0.771 | 0.808 | 0.838 | 0.801 | 0.910 | 0.852 | 0.751 |
| rossler/c5 | 5 | 2.486 | 2.638 | 2.593 | 2.365 | 8.995 | 38.703 | 5.888 | 0.651 | 0.630 | 0.620 | 0.643 | 0.866 | 0.884 | 0.824 | 0.805 |
| rossler/c5.7 | 5.7 | 3.314 | 2.614 | 2.417 | 2.045 | 12.504 | 24.504 | 8.159 | 0.736 | 0.802 | 0.883 | 0.888 | 0.831 | 0.925 | 0.809 | 0.848 |
| rossler/c6.5 | 6.5 | 2.173 | 1.789 | 1.696 | 1.730 | 9.328 | 28.004 | 6.029 | 0.793 | 0.861 | 0.823 | 0.778 | 0.781 | 0.858 | 0.753 | 0.695 |
| rossler/c9 | 9 | 7.734 | 6.302 | 5.450 | 4.305 | 15.188 | 23.280 | 9.605 | 0.615 | 0.725 | 0.863 | 0.771 | 0.786 | 0.952 | 0.799 | 0.829 |

### T = 2048, H = 2048

| config_id | knob | timestral-5M ratio | timestral-28M ratio | timestral-123M ratio | timestral-209M ratio | seasonal_naive ratio | ets ratio | arima ratio | timestral-5M cov80 | timestral-28M cov80 | timestral-123M cov80 | timestral-209M cov80 | seasonal_naive cov80 | ets cov80 | arima cov80 | oracle cov80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kuramoto/K0 | 0 | 12.983 | 12.775 | 12.477 | 12.507 | 20.242 | 48.483 | 18.745 | 0.525 | 0.537 | 0.607 | 0.508 | 0.975 | 0.993 | 0.868 | 0.826 |
| kuramoto/K0.5 | 0.5 | 8.129 | 8.031 | 7.417 | 7.427 | 12.221 | 17.825 | 9.278 | 0.547 | 0.569 | 0.602 | 0.523 | 0.979 | 0.995 | 0.840 | 0.755 |
| kuramoto/K1 | 1 | 4.177 | 3.941 | 3.917 | 3.585 | 5.895 | 8.408 | 4.912 | 0.512 | 0.528 | 0.534 | 0.504 | 0.979 | 0.998 | 0.837 | 0.825 |
| kuramoto/K1.5 | 1.5 | 8.106 | 6.180 | 7.085 | 5.425 | 12.427 | 34.858 | 8.341 | 0.536 | 0.566 | 0.592 | 0.629 | 0.971 | 0.991 | 0.686 | 0.829 |
| kuramoto/K2 | 2 | 27.202 | 22.457 | 19.043 | 12.147 | 41.113 | 113.060 | 26.870 | 0.540 | 0.576 | 0.658 | 0.717 | 0.972 | 0.991 | 0.757 | 0.846 |
| kuramoto/K3 | 3 | 42.404 | 34.799 | 32.196 | 21.316 | 59.317 | 457.778 | 32.672 | 0.603 | 0.625 | 0.746 | 0.828 | 0.980 | 0.980 | 0.558 | 0.740 |
| kuramoto/K5 | 5 | 55.140 | 39.895 | 43.146 | 26.365 | 80.035 | 394.974 | 38.016 | 0.600 | 0.666 | 0.706 | 0.821 | 0.973 | 0.988 | 0.366 | 0.794 |
| kuramoto/K8 | 8 | 58.713 | 42.882 | 45.780 | 28.669 | 84.737 | 430.336 | 36.738 | 0.591 | 0.667 | 0.706 | 0.805 | 0.974 | 0.984 | 0.428 | 0.833 |
| logistic_map/r2.8 | 2.8 | 0.998 | 0.999 | 0.999 | 0.999 | 1.200 | 0.999 | 0.999 | 0.439 | 0.403 | 0.413 | 0.401 | 1.000 | 0.799 | 0.805 | 0.794 |
| logistic_map/r3.2 | 3.2 | 2.503 | 12.758 | 5.321 | 1.947 | 32.315 | 31.548 | 2.135 | 0.593 | 0.458 | 0.796 | 0.600 | 1.000 | 1.000 | 0.896 | 0.802 |
| logistic_map/r3.5 | 3.5 | 2.196 | 7.812 | 3.651 | 3.272 | 17.447 | 15.693 | 15.612 | 0.566 | 0.471 | 0.753 | 0.679 | 1.000 | 0.957 | 0.767 | 0.836 |
| logistic_map/r3.56 | 3.56 | 1.268 | 3.856 | 1.122 | 1.996 | 4.773 | 4.232 | 4.220 | 0.487 | 0.420 | 0.580 | 0.539 | 1.000 | 0.842 | 0.833 | 0.793 |
| logistic_map/r3.6 | 3.6 | 1.208 | 3.057 | 1.099 | 2.394 | 4.508 | 3.704 | 3.694 | 0.479 | 0.446 | 0.627 | 0.440 | 1.000 | 0.852 | 0.854 | 0.793 |
| logistic_map/r3.7 | 3.7 | 1.004 | 1.003 | 1.003 | 1.003 | 1.320 | 1.049 | 1.048 | 0.409 | 0.400 | 0.401 | 0.396 | 1.000 | 0.825 | 0.825 | 0.795 |
| logistic_map/r3.83 | 3.83 | 1.003 | 1.005 | 1.002 | 1.004 | 1.321 | 1.001 | 1.003 | 0.396 | 0.357 | 0.378 | 0.365 | 1.000 | 0.813 | 0.939 | 0.787 |
| logistic_map/r3.9 | 3.9 | 1.002 | 1.002 | 1.002 | 1.005 | 1.509 | 1.003 | 1.002 | 0.396 | 0.371 | 0.385 | 0.380 | 1.000 | 0.834 | 0.833 | 0.798 |
| logistic_map/r4 | 4 | 1.001 | 1.001 | 1.001 | 1.002 | 1.269 | 1.001 | 1.001 | 0.397 | 0.393 | 0.402 | 0.400 | 1.000 | 0.724 | 0.726 | 0.797 |
| lorenz63/rho10 | 10 | 0.999 | 0.999 | 0.999 | 0.999 | 1.642 | 0.999 | 0.999 | 0.472 | 0.414 | 0.429 | 0.411 | 0.999 | 0.803 | 0.802 | 0.801 |
| lorenz63/rho15 | 15 | 1.000 | 0.999 | 0.999 | 0.999 | 1.322 | 1.000 | 1.000 | 0.450 | 0.397 | 0.423 | 0.407 | 1.000 | 0.781 | 0.782 | 0.790 |
| lorenz63/rho20 | 20 | 1.003 | 1.000 | 0.998 | 0.998 | 1.417 | 1.002 | 1.002 | 0.488 | 0.401 | 0.416 | 0.407 | 1.000 | 0.808 | 0.809 | 0.806 |
| lorenz63/rho24 | 24 | 1.042 | 1.036 | 1.032 | 1.041 | 1.176 | 1.176 | 1.029 | 0.460 | 0.481 | 0.455 | 0.435 | 0.999 | 0.999 | 0.738 | 0.798 |
| lorenz63/rho26 | 26 | 1.023 | 1.019 | 1.019 | 1.019 | 1.268 | 1.268 | 1.021 | 0.432 | 0.451 | 0.428 | 0.425 | 0.999 | 0.999 | 0.743 | 0.800 |
| lorenz63/rho28 | 28 | 1.016 | 1.014 | 1.013 | 1.011 | 1.399 | 1.399 | 1.017 | 0.423 | 0.452 | 0.431 | 0.403 | 0.999 | 0.999 | 0.740 | 0.795 |
| lorenz63/rho28_oe2 | 28 | 1.117 | 1.119 | 1.100 | 1.100 | 1.612 | 2.773 | 1.186 | 0.455 | 0.469 | 0.450 | 0.454 | 0.955 | 0.994 | 0.769 | 0.799 |
| lorenz63/rho28_oe20 | 28 | 1.010 | 1.010 | 1.008 | 1.008 | 1.338 | 1.341 | 1.013 | 0.431 | 0.440 | 0.439 | 0.407 | 1.000 | 1.000 | 0.752 | 0.798 |
| lorenz63/rho28_oe5 | 28 | 1.068 | 1.052 | 1.051 | 1.056 | 1.551 | 2.199 | 1.537 | 0.449 | 0.483 | 0.465 | 0.453 | 0.993 | 0.998 | 0.997 | 0.796 |
| lorenz63/rho35 | 35 | 1.027 | 1.023 | 1.019 | 1.018 | 1.326 | 1.326 | 1.024 | 0.423 | 0.455 | 0.443 | 0.412 | 0.999 | 0.999 | 0.747 | 0.798 |
| lorenz63/rho45 | 45 | 1.010 | 1.008 | 1.006 | 1.005 | 1.388 | 1.388 | 1.009 | 0.415 | 0.438 | 0.441 | 0.400 | 0.998 | 0.998 | 0.764 | 0.797 |
| mackey_glass/tau10 | 10 | 5.020 | 5.481 | 5.001 | 4.287 | 7.848 | 31.316 | 6.782 | 0.674 | 0.503 | 0.590 | 0.524 | 0.988 | 0.995 | 0.749 | 0.815 |
| mackey_glass/tau14 | 14 | 3.175 | 2.379 | 2.176 | 2.642 | 5.043 | 14.640 | 3.218 | 0.489 | 0.556 | 0.679 | 0.522 | 0.970 | 0.990 | 0.800 | 0.826 |
| mackey_glass/tau17 | 17 | 1.294 | 1.365 | 1.337 | 1.310 | 1.594 | 2.796 | 1.377 | 0.450 | 0.436 | 0.490 | 0.422 | 0.979 | 0.995 | 0.786 | 0.795 |
| mackey_glass/tau20 | 20 | 1.127 | 1.122 | 1.094 | 1.106 | 1.657 | 3.554 | 1.171 | 0.462 | 0.463 | 0.541 | 0.456 | 0.962 | 0.991 | 0.767 | 0.778 |
| mackey_glass/tau25 | 25 | 1.129 | 1.172 | 1.129 | 1.136 | 1.786 | 3.109 | 1.171 | 0.454 | 0.464 | 0.501 | 0.426 | 0.948 | 0.993 | 0.777 | 0.799 |
| mackey_glass/tau30 | 30 | 1.293 | 1.296 | 1.249 | 1.255 | 1.689 | 2.099 | 1.302 | 0.448 | 0.444 | 0.509 | 0.463 | 0.945 | 0.995 | 0.748 | 0.794 |
| mackey_glass/tau6 | 6 | 2.935 | 2.311 | 2.544 | 2.521 | 4.154 | 9.858 | 2.694 | 0.521 | 0.528 | 0.485 | 0.425 | 0.994 | 0.964 | 0.691 | 0.629 |
| rossler/c2 | 2 | 1.939 | 1.921 | 2.053 | 2.007 | 3.238 | 4.267 | 2.377 | 0.562 | 0.472 | 0.482 | 0.414 | 0.986 | 0.995 | 0.845 | 0.845 |
| rossler/c3 | 3 | 1.900 | 2.030 | 1.875 | 1.987 | 2.981 | 9.421 | 2.374 | 0.581 | 0.449 | 0.512 | 0.421 | 0.989 | 0.994 | 0.779 | 0.742 |
| rossler/c4 | 4 | 1.994 | 1.778 | 1.765 | 1.789 | 3.179 | 6.008 | 2.115 | 0.521 | 0.456 | 0.499 | 0.426 | 0.985 | 0.995 | 0.780 | 0.720 |
| rossler/c4.5 | 4.5 | 1.879 | 1.585 | 1.677 | 1.655 | 2.632 | 5.609 | 1.952 | 0.496 | 0.477 | 0.498 | 0.437 | 0.988 | 0.994 | 0.772 | 0.743 |
| rossler/c5 | 5 | 2.114 | 1.880 | 1.942 | 1.912 | 2.600 | 12.349 | 2.185 | 0.498 | 0.476 | 0.487 | 0.425 | 0.992 | 0.993 | 0.772 | 0.818 |
| rossler/c5.7 | 5.7 | 1.876 | 1.731 | 1.765 | 1.742 | 2.645 | 5.572 | 1.981 | 0.511 | 0.486 | 0.496 | 0.437 | 0.989 | 0.995 | 0.768 | 0.835 |
| rossler/c6.5 | 6.5 | 1.489 | 1.427 | 1.441 | 1.441 | 2.256 | 7.922 | 1.599 | 0.520 | 0.489 | 0.488 | 0.423 | 0.986 | 0.991 | 0.764 | 0.822 |
| rossler/c9 | 9 | 1.046 | 1.021 | 1.029 | 1.027 | 1.584 | 2.747 | 1.050 | 0.535 | 0.493 | 0.495 | 0.433 | 0.985 | 0.997 | 0.784 | 0.777 |

### Context length: ratio to oracle by T

Forecaster timestral-5M, H = 128.

| config_id | knob | timestral-5M T=512 | timestral-5M T=2048 | timestral-5M T=8192 |
|---|---|---|---|---|
| kuramoto/K0 | 0 | 21.376 | 17.224 | 15.770 |
| kuramoto/K0.5 | 0.5 | 18.501 | 14.316 | 11.953 |
| kuramoto/K1 | 1 | 16.240 | 12.742 | 12.023 |
| kuramoto/K1.5 | 1.5 | 13.910 | 9.891 | 9.160 |
| kuramoto/K2 | 2 | 8.522 | 5.848 | 5.173 |
| kuramoto/K3 | 3 | 6.487 | 4.183 | 3.700 |
| kuramoto/K5 | 5 | 6.151 | 3.769 | 3.163 |
| kuramoto/K8 | 8 | 7.370 | 5.184 | 4.278 |
| logistic_map/r2.8 | 2.8 | 1.007 | 1.004 | 1.006 |
| logistic_map/r3.2 | 3.2 | 1.097 | 1.071 | 1.075 |
| logistic_map/r3.5 | 3.5 | 1.333 | 1.356 | 1.443 |
| logistic_map/r3.56 | 3.56 | 0.929 | 0.942 | 0.968 |
| logistic_map/r3.6 | 3.6 | 1.084 | 1.072 | 1.059 |
| logistic_map/r3.7 | 3.7 | 1.040 | 1.030 | 1.028 |
| logistic_map/r3.83 | 3.83 | 1.020 | 1.014 | 1.004 |
| logistic_map/r3.9 | 3.9 | 1.050 | 1.049 | 1.047 |
| logistic_map/r4 | 4 | 1.032 | 1.031 | 1.032 |
| lorenz63/rho10 | 10 | 1.004 | 1.004 | 1.004 |
| lorenz63/rho15 | 15 | 1.011 | 1.016 | 1.015 |
| lorenz63/rho20 | 20 | 1.090 | 1.067 | 1.061 |
| lorenz63/rho24 | 24 | 1.522 | 1.510 | 1.413 |
| lorenz63/rho26 | 26 | 1.603 | 1.563 | 1.504 |
| lorenz63/rho28 | 28 | 1.321 | 1.310 | 1.302 |
| lorenz63/rho28_oe2 | 28 | 34.182 | 29.891 | 29.663 |
| lorenz63/rho28_oe20 | 28 | 1.253 | 1.226 | 1.224 |
| lorenz63/rho28_oe5 | 28 | 3.031 | 2.996 | 3.011 |
| lorenz63/rho35 | 35 | 1.447 | 1.401 | 1.387 |
| lorenz63/rho45 | 45 | 1.280 | 1.263 | 1.262 |
| mackey_glass/tau10 | 10 | 2.701 | 1.471 | 1.381 |
| mackey_glass/tau14 | 14 | 3.078 | 2.445 | 2.418 |
| mackey_glass/tau17 | 17 | 10.766 | 9.574 | 9.068 |
| mackey_glass/tau20 | 20 | 11.107 | 9.541 | 8.579 |
| mackey_glass/tau25 | 25 | 9.352 | 8.157 | 6.636 |
| mackey_glass/tau30 | 30 | 24.928 | 21.871 | 19.308 |
| mackey_glass/tau6 | 6 | 2.048 | 1.899 | 1.662 |
| rossler/c2 | 2 | 1.876 | 1.685 | 1.606 |
| rossler/c3 | 3 | 1.878 | 1.528 | 1.482 |
| rossler/c4 | 4 | 2.703 | 2.342 | 2.253 |
| rossler/c4.5 | 4.5 | 2.014 | 2.125 | 1.980 |
| rossler/c5 | 5 | 2.790 | 2.486 | 2.310 |
| rossler/c5.7 | 5.7 | 3.596 | 3.314 | 3.271 |
| rossler/c6.5 | 6.5 | 2.806 | 2.173 | 2.019 |
| rossler/c9 | 9 | 10.613 | 7.734 | 6.952 |

