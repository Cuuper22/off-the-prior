# results/map/tails: heavy-tail and fractal family, Timestral failure map

Facts only. Every number below is copied from `summary.csv` in this folder.

## Contents

- `scores.csv`: one row per sample x forecaster x context length T x horizon H (written by `scripts/sweep.py`).
- `provenance.json`: command line, git commit, host, threads, wall time of the sweep.
- `summary.csv`: built from `scores.csv` by `results/map/tails/summarize.py` with `otp.analysis.ratios` and
  `otp.analysis.curve`, grouped by `[config_id, generator, knob_value, forecaster, T, H]`.
- `samples.npz`: the generated samples (contexts, realized futures, oracle quantiles). Not committed if over 50 MB;
  see the regeneration commands.
- `README.md`: this file (written by `summarize.py`).

Sweep: 35 configs, 8 seeds each (280 samples), Timestral 28M, 5M,
context lengths [512, 2048, 8192], horizons [16, 128, 1024], classical models: arima, ets, seasonal_naive.
Wall time of the sweep (`provenance.json`): 4537.5 s on 1 thread(s), git `3193683`.

## Regeneration

```
export PYTHONPATH=$PWD OTP_THREADS=1
python scripts/sweep.py --grid otp.grids.tails --sizes 5M 28M --contexts 512 2048 8192 --horizons 16 128 1024 --seeds 8 --n-paths 2000 --out results/map/tails
python results/map/tails/summarize.py --dir results/map/tails
```

Sample generation is seeded and deterministic (`samples.npz` is rebuilt by the first command if absent).
`--n-paths 2000` is used instead of the harness default of 500: the oracle quantiles at levels 0.01 and 0.99
are Monte Carlo estimates from 2000 simulated continuations (1% of 500 is 5 paths; heavy tails make that noisy).

## Terms

- Sample: one simulated series of length 8192 (context) plus 1024 (realized future) for one config and seed.
- T: context length used by the forecaster (the last T steps of the 8192). H: horizon; horizons 16 and 128 are
  prefixes of the 1024-step forecast.
- Oracle: Monte Carlo continuation of the true generating process from its true state with known parameters
  (2000 paths). It is computed once per sample; its rows are copied to every T in `summary.csv`.
- MASE: mean absolute error of the median forecast divided by the in-sample one-step naive error on the context.
  sCRPS: scaled CRPS (average pinball loss over the 21 quantile levels, times 2, divided by the same scale).
  Ratio to oracle = metric of the forecaster / metric of the oracle on the same sample, T and H.
  Ratio to classical = metric / metric of the best classical model on that sample (arima, ets, seasonal_naive).
- gmean: geometric mean over seeds of the per-sample ratio (the mean of the logs, exponentiated). median: median over
  seeds of the per-sample ratio. Heavy-tailed series can make a few samples dominate a mean; the median is reported
  next to the geometric mean for that reason.
- cov80, cov98: fraction of realized future points inside the forecaster's [0.1, 0.9] and [0.01, 0.99] quantile
  intervals, averaged over seeds. A calibrated forecast has 0.80 and 0.98.

## Generators (`otp/generators/tails.py`)

- `stable_ar`: x_t = rho x_(t-1) + e_t with e_t symmetric alpha-stable (scale 1, sampled by the Chambers-Mallows-Stuck
  formula); rho = 1.0 is a random walk. alpha = 2.0 is Gaussian (standard deviation sqrt(2)). Oracle state: last value.
- `fbm`: fractional Brownian motion, unit time step, increments are fractional Gaussian noise. Context increments are
  exact (Davies-Harte); future increments are drawn from the exact Gaussian conditional given the last 2048 increments
  (the oracle conditions on that window, not on the whole 8192-step history).
- `hawkes`: discrete-time approximation of a Hawkes process with exponential kernel: counts N_t ~ Poisson(mu + s_t),
  s_(t+1) = d s_t + eta (1 - d) N_t, d = exp(-1/4); mean count per bin is 5.0 for every eta (mu = 5 (1 - eta)).
- `msm_cascade`: Markov-switching multifractal, multipliers in {1.4, 0.6}, renewal probability of the fastest multiplier
  0.5, frequency ratio 3 between neighbouring multipliers, returns = sqrt(product of multipliers) x standard normal.
  Config id suffix `ret` = observed series is the returns, `cum` = the cumulative sum of the returns.

## Tables: T = 2048, H = 128

Columns: knob value, forecaster, MASE and sCRPS ratios to the oracle (gmean and median over seeds), MASE ratio to
the best classical model (gmean), cov80, cov98. The `oracle` row shows the oracle's own coverage on the same samples.

**tails/fbm** (knob `H_hurst`)

| H_hurst | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |
|---|---|---|---|---|---|---|---|---|
| 0.1 | oracle | - | - | - | - | - | 0.810 | 0.979 |
| 0.1 | timestral-5M | 1.034 | 1.026 | 1.041 | 1.033 | 0.9975 | 0.820 | 0.976 |
| 0.1 | timestral-28M | 1.025 | 1.007 | 1.028 | 1.016 | 0.9886 | 0.812 | 0.979 |
| 0.1 | arima | 1.102 | 1.087 | 1.107 | 1.085 | 1.063 | 0.765 | 0.965 |
| 0.1 | ets | 1.164 | 1.03 | 1.295 | 1.281 | 1.123 | 0.894 | 0.998 |
| 0.1 | seasonal_naive | 1.313 | 1.32 | 3.03 | 3.232 | 1.266 | 0.995 | 1.000 |
| 0.2 | oracle | - | - | - | - | - | 0.813 | 0.985 |
| 0.2 | timestral-5M | 0.9921 | 0.9932 | 0.984 | 0.9903 | 1.001 | 0.813 | 0.992 |
| 0.2 | timestral-28M | 0.9954 | 0.994 | 0.9983 | 1.004 | 1.005 | 0.816 | 0.991 |
| 0.2 | arima | 1.111 | 1.025 | 1.149 | 1.087 | 1.122 | 0.758 | 0.946 |
| 0.2 | ets | 1.076 | 1.067 | 1.27 | 1.355 | 1.086 | 0.976 | 1.000 |
| 0.2 | seasonal_naive | 1.125 | 1.05 | 2.048 | 2.349 | 1.136 | 0.999 | 1.000 |
| 0.3 | oracle | - | - | - | - | - | 0.787 | 0.992 |
| 0.3 | timestral-5M | 1.008 | 1.002 | 1.025 | 1.02 | 1.004 | 0.780 | 0.994 |
| 0.3 | timestral-28M | 1 | 1.007 | 1.016 | 1.001 | 0.997 | 0.781 | 0.991 |
| 0.3 | arima | 1.11 | 1.066 | 1.172 | 1.139 | 1.107 | 0.746 | 0.998 |
| 0.3 | ets | 1.047 | 0.9723 | 1.19 | 1.211 | 1.043 | 0.938 | 1.000 |
| 0.3 | seasonal_naive | 1.061 | 0.9921 | 1.422 | 1.545 | 1.058 | 0.990 | 1.000 |
| 0.5 | oracle | - | - | - | - | - | 0.689 | 0.973 |
| 0.5 | timestral-5M | 0.9726 | 0.9973 | 1.024 | 1.031 | 0.9723 | 0.677 | 0.982 |
| 0.5 | timestral-28M | 0.9559 | 0.9322 | 0.9922 | 0.9559 | 0.9556 | 0.712 | 0.971 |
| 0.5 | arima | 1.002 | 0.9974 | 0.999 | 0.9988 | 1.002 | 0.682 | 0.968 |
| 0.5 | ets | 1.007 | 0.9968 | 1.01 | 1.006 | 1.007 | 0.704 | 0.976 |
| 0.5 | seasonal_naive | 1.004 | 0.997 | 1.002 | 1.001 | 1.004 | 0.694 | 0.962 |
| 0.7 | oracle | - | - | - | - | - | 0.927 | 1.000 |
| 0.7 | timestral-5M | 1.102 | 1.074 | 1.09 | 1.059 | 0.9886 | 0.777 | 1.000 |
| 0.7 | timestral-28M | 1.105 | 1.129 | 1.104 | 1.068 | 0.9913 | 0.757 | 1.000 |
| 0.7 | arima | 1.173 | 1.41 | 1.093 | 1.225 | 1.053 | 0.774 | 0.990 |
| 0.7 | ets | 1.214 | 1.39 | 1.065 | 1.247 | 1.089 | 0.781 | 0.977 |
| 0.7 | seasonal_naive | 1.288 | 1.444 | 1.187 | 1.318 | 1.156 | 0.430 | 0.805 |
| 0.8 | oracle | - | - | - | - | - | 0.912 | 0.996 |
| 0.8 | timestral-5M | 1.155 | 1.108 | 1.003 | 0.9745 | 0.7783 | 0.899 | 0.996 |
| 0.8 | timestral-28M | 1.186 | 1.099 | 1.049 | 1.019 | 0.7992 | 0.877 | 0.999 |
| 0.8 | arima | 1.562 | 1.732 | 1.208 | 1.219 | 1.053 | 0.877 | 0.992 |
| 0.8 | ets | 1.973 | 1.843 | 1.535 | 1.462 | 1.329 | 0.611 | 0.983 |
| 0.8 | seasonal_naive | 2.158 | 2.282 | 1.969 | 2.069 | 1.454 | 0.179 | 0.405 |
| 0.9 | oracle | - | - | - | - | - | 0.656 | 0.980 |
| 0.9 | timestral-5M | 1.057 | 1.13 | 1.085 | 1.168 | 1.335 | 0.657 | 1.000 |
| 0.9 | timestral-28M | 1.074 | 1.066 | 1.102 | 1.149 | 1.356 | 0.596 | 1.000 |
| 0.9 | arima | 1.109 | 1.22 | 1.118 | 1.198 | 1.401 | 0.604 | 0.986 |
| 0.9 | ets | 1.111 | 1.486 | 1.193 | 1.517 | 1.403 | 0.622 | 0.905 |
| 0.9 | seasonal_naive | 1.254 | 1.466 | 1.653 | 1.977 | 1.583 | 0.204 | 0.376 |

**tails/hawkes** (knob `eta`)

| eta | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |
|---|---|---|---|---|---|---|---|---|
| 0 | oracle | - | - | - | - | - | 0.889 | 0.985 |
| 0 | timestral-5M | 1.013 | 1.009 | 1.011 | 1.013 | 1.009 | 0.784 | 0.975 |
| 0 | timestral-28M | 1.011 | 1.012 | 1.01 | 1.013 | 1.008 | 0.773 | 0.973 |
| 0 | arima | 1.004 | 1.005 | 1.012 | 1.016 | 1.001 | 0.751 | 0.976 |
| 0 | ets | 1.004 | 1.006 | 1.012 | 1.017 | 1 | 0.751 | 0.976 |
| 0 | seasonal_naive | 1.494 | 1.289 | 4.699 | 4.651 | 1.489 | 0.997 | 1.000 |
| 0.3 | oracle | - | - | - | - | - | 0.883 | 0.989 |
| 0.3 | timestral-5M | 1.002 | 0.9979 | 1.005 | 1 | 1 | 0.792 | 0.979 |
| 0.3 | timestral-28M | 1.003 | 1 | 1.005 | 1 | 1.001 | 0.774 | 0.984 |
| 0.3 | arima | 1.005 | 1.003 | 1.008 | 1.005 | 1.003 | 0.778 | 0.983 |
| 0.3 | ets | 1.014 | 1.006 | 1.011 | 1.009 | 1.012 | 0.812 | 0.982 |
| 0.3 | seasonal_naive | 1.543 | 1.479 | 4.562 | 4.461 | 1.54 | 0.994 | 1.000 |
| 0.6 | oracle | - | - | - | - | - | 0.853 | 0.982 |
| 0.6 | timestral-5M | 1.021 | 1.006 | 1.015 | 1.001 | 1.021 | 0.795 | 0.978 |
| 0.6 | timestral-28M | 1.014 | 1.006 | 1.01 | 0.9983 | 1.015 | 0.768 | 0.980 |
| 0.6 | arima | 1.017 | 1.004 | 1.017 | 1.016 | 1.018 | 0.784 | 0.977 |
| 0.6 | ets | 1.082 | 1.063 | 1.098 | 1.084 | 1.082 | 0.872 | 0.988 |
| 0.6 | seasonal_naive | 1.351 | 1.137 | 4.056 | 4.18 | 1.352 | 0.998 | 1.000 |
| 0.8 | oracle | - | - | - | - | - | 0.828 | 0.990 |
| 0.8 | timestral-5M | 1.002 | 1.033 | 1.021 | 1.029 | 1.031 | 0.780 | 0.975 |
| 0.8 | timestral-28M | 0.9971 | 1.017 | 1.016 | 1.017 | 1.026 | 0.780 | 0.972 |
| 0.8 | arima | 0.9779 | 0.9846 | 1 | 1.009 | 1.006 | 0.778 | 0.973 |
| 0.8 | ets | 1.119 | 1.066 | 1.151 | 1.103 | 1.151 | 0.851 | 0.990 |
| 0.8 | seasonal_naive | 1.288 | 1.231 | 3.709 | 3.603 | 1.325 | 0.996 | 1.000 |
| 0.9 | oracle | - | - | - | - | - | 0.896 | 0.997 |
| 0.9 | timestral-5M | 1.055 | 1.067 | 1.088 | 1.07 | 1.021 | 0.840 | 0.987 |
| 0.9 | timestral-28M | 1.088 | 1.089 | 1.118 | 1.104 | 1.054 | 0.812 | 0.986 |
| 0.9 | arima | 1.071 | 1.057 | 1.088 | 1.072 | 1.037 | 0.831 | 0.982 |
| 0.9 | ets | 1.39 | 1.315 | 1.423 | 1.419 | 1.346 | 0.831 | 0.989 |
| 0.9 | seasonal_naive | 1.687 | 1.62 | 3.702 | 3.977 | 1.633 | 0.996 | 1.000 |
| 0.95 | oracle | - | - | - | - | - | 0.901 | 0.993 |
| 0.95 | timestral-5M | 1.063 | 1.04 | 1.06 | 1.067 | 1.109 | 0.806 | 0.980 |
| 0.95 | timestral-28M | 1.036 | 1.047 | 1.037 | 1.056 | 1.08 | 0.802 | 0.986 |
| 0.95 | arima | 1.053 | 0.9892 | 1.091 | 1.089 | 1.098 | 0.890 | 0.980 |
| 0.95 | ets | 1.174 | 1.213 | 1.22 | 1.262 | 1.225 | 0.905 | 0.979 |
| 0.95 | seasonal_naive | 1.026 | 1.032 | 3.289 | 3.337 | 1.07 | 0.996 | 1.000 |

**tails/msm, observed series = cumulative sum** (knob `k`)

| k | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |
|---|---|---|---|---|---|---|---|---|
| 1 | oracle | - | - | - | - | - | 0.856 | 0.997 |
| 1 | timestral-5M | 1.036 | 1.033 | 1.029 | 1.046 | 1.057 | 0.852 | 0.963 |
| 1 | timestral-28M | 1.037 | 1.049 | 1.026 | 1.049 | 1.059 | 0.854 | 0.963 |
| 1 | arima | 1.006 | 1.012 | 1.006 | 1.019 | 1.027 | 0.862 | 0.996 |
| 1 | ets | 0.9807 | 1.01 | 0.9756 | 1.006 | 1.001 | 0.883 | 0.997 |
| 1 | seasonal_naive | 1.007 | 1.012 | 1.007 | 1.015 | 1.027 | 0.862 | 0.997 |
| 2 | oracle | - | - | - | - | - | 0.747 | 0.946 |
| 2 | timestral-5M | 0.9417 | 0.9562 | 0.9604 | 0.9709 | 0.9433 | 0.751 | 0.979 |
| 2 | timestral-28M | 0.9365 | 0.9589 | 0.9416 | 0.9595 | 0.9381 | 0.755 | 0.966 |
| 2 | arima | 1.004 | 1.007 | 1.009 | 1.014 | 1.006 | 0.730 | 0.952 |
| 2 | ets | 1.004 | 1.007 | 1.035 | 1.005 | 1.006 | 0.743 | 0.938 |
| 2 | seasonal_naive | 1.002 | 1.007 | 1.003 | 1.01 | 1.004 | 0.749 | 0.945 |
| 4 | oracle | - | - | - | - | - | 0.877 | 0.999 |
| 4 | timestral-5M | 1.019 | 1 | 0.983 | 0.9492 | 1.038 | 0.887 | 0.999 |
| 4 | timestral-28M | 0.9952 | 0.9535 | 0.9512 | 0.9699 | 1.014 | 0.891 | 0.999 |
| 4 | arima | 0.9889 | 0.9904 | 0.9808 | 0.9774 | 1.007 | 0.902 | 0.999 |
| 4 | ets | 1.001 | 1.005 | 0.9919 | 0.9678 | 1.019 | 0.909 | 0.999 |
| 4 | seasonal_naive | 0.9883 | 0.9923 | 0.9733 | 0.9788 | 1.007 | 0.902 | 0.999 |
| 6 | oracle | - | - | - | - | - | 0.933 | 1.000 |
| 6 | timestral-5M | 1.014 | 0.9769 | 0.9825 | 0.9431 | 1.016 | 0.883 | 1.000 |
| 6 | timestral-28M | 1.061 | 1.094 | 1.014 | 1.02 | 1.063 | 0.875 | 1.000 |
| 6 | arima | 1.147 | 1.02 | 1.1 | 1.08 | 1.149 | 0.817 | 0.980 |
| 6 | ets | 1.077 | 0.9961 | 1.038 | 0.9937 | 1.079 | 0.894 | 0.995 |
| 6 | seasonal_naive | 1.002 | 0.9928 | 0.9946 | 0.9816 | 1.004 | 0.886 | 0.995 |
| 8 | oracle | - | - | - | - | - | 0.900 | 0.995 |
| 8 | timestral-5M | 1.06 | 1.025 | 1.088 | 1.048 | 1.068 | 0.805 | 0.969 |
| 8 | timestral-28M | 1.102 | 1.12 | 1.126 | 1.116 | 1.109 | 0.794 | 0.949 |
| 8 | arima | 1.012 | 1.005 | 1.216 | 1.244 | 1.019 | 0.835 | 0.965 |
| 8 | ets | 0.9953 | 1.007 | 1.262 | 1.294 | 1.002 | 0.841 | 0.979 |
| 8 | seasonal_naive | 1.013 | 1.006 | 1.299 | 1.34 | 1.02 | 0.859 | 0.950 |

**tails/msm, observed series = returns** (knob `k`)

| k | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |
|---|---|---|---|---|---|---|---|---|
| 1 | oracle | - | - | - | - | - | 0.797 | 0.977 |
| 1 | timestral-5M | 0.9925 | 0.9947 | 0.9921 | 0.9974 | 0.9992 | 0.815 | 0.984 |
| 1 | timestral-28M | 0.9929 | 0.9959 | 0.9923 | 0.9971 | 0.9996 | 0.814 | 0.974 |
| 1 | arima | 0.9936 | 0.9956 | 0.9933 | 0.9971 | 1 | 0.818 | 0.970 |
| 1 | ets | 0.9939 | 0.9953 | 0.9937 | 0.9969 | 1.001 | 0.817 | 0.970 |
| 1 | seasonal_naive | 1.249 | 1.193 | 4.57 | 4.528 | 1.257 | 0.999 | 1.000 |
| 2 | oracle | - | - | - | - | - | 0.806 | 0.985 |
| 2 | timestral-5M | 1.006 | 0.998 | 1.006 | 0.9971 | 1.002 | 0.809 | 0.983 |
| 2 | timestral-28M | 1.006 | 0.9972 | 1.006 | 0.9956 | 1.002 | 0.807 | 0.980 |
| 2 | arima | 1.006 | 0.9987 | 1.008 | 0.9997 | 1.002 | 0.825 | 0.968 |
| 2 | ets | 1.006 | 0.9987 | 1.008 | 1 | 1.003 | 0.824 | 0.967 |
| 2 | seasonal_naive | 1.326 | 1.317 | 4.761 | 4.857 | 1.322 | 0.998 | 1.000 |
| 4 | oracle | - | - | - | - | - | 0.781 | 0.974 |
| 4 | timestral-5M | 1.002 | 0.9958 | 1.007 | 1.001 | 1.001 | 0.773 | 0.970 |
| 4 | timestral-28M | 0.9996 | 0.9959 | 1.005 | 0.9975 | 0.999 | 0.764 | 0.969 |
| 4 | arima | 1.001 | 0.9972 | 1.005 | 1.008 | 1 | 0.802 | 0.952 |
| 4 | ets | 1.001 | 0.9969 | 1.004 | 1.008 | 1 | 0.801 | 0.954 |
| 4 | seasonal_naive | 1.598 | 1.28 | 4.554 | 4.631 | 1.597 | 0.994 | 0.998 |
| 6 | oracle | - | - | - | - | - | 0.827 | 0.975 |
| 6 | timestral-5M | 0.9739 | 0.9492 | 0.9901 | 0.9614 | 1 | 0.855 | 0.974 |
| 6 | timestral-28M | 0.9745 | 0.9492 | 0.9924 | 0.9632 | 1.001 | 0.853 | 0.974 |
| 6 | arima | 0.9744 | 0.9498 | 1.003 | 0.9782 | 1 | 0.872 | 0.966 |
| 6 | ets | 0.9744 | 0.9497 | 1.002 | 0.9779 | 1.001 | 0.873 | 0.965 |
| 6 | seasonal_naive | 1.406 | 1.266 | 5.313 | 4.903 | 1.443 | 0.999 | 1.000 |
| 8 | oracle | - | - | - | - | - | 0.779 | 0.974 |
| 8 | timestral-5M | 1.146 | 1.134 | 1.157 | 1.162 | 1.001 | 0.775 | 0.967 |
| 8 | timestral-28M | 1.145 | 1.131 | 1.152 | 1.142 | 0.9998 | 0.765 | 0.968 |
| 8 | arima | 1.145 | 1.136 | 1.209 | 1.191 | 1 | 0.869 | 0.966 |
| 8 | ets | 1.146 | 1.136 | 1.209 | 1.191 | 1 | 0.869 | 0.966 |
| 8 | seasonal_naive | 1.525 | 1.353 | 7.648 | 8.435 | 1.332 | 0.999 | 1.000 |

**tails/stable_ar, rho = 0.9** (knob `alpha`)

| alpha | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |
|---|---|---|---|---|---|---|---|---|
| 1.1 | oracle | - | - | - | - | - | 0.888 | 0.992 |
| 1.1 | timestral-5M | 1.025 | 1.011 | 1.111 | 1.033 | 1.007 | 0.902 | 0.984 |
| 1.1 | timestral-28M | 0.9669 | 1.006 | 1.056 | 1.02 | 0.9501 | 0.880 | 0.984 |
| 1.1 | arima | 1.56 | 1.26 | 2.599 | 2.1 | 1.533 | 0.965 | 0.979 |
| 1.1 | ets | 1.418 | 1.351 | 6.92 | 5.971 | 1.393 | 0.906 | 0.969 |
| 1.1 | seasonal_naive | 1.418 | 1.348 | 13.89 | 14.86 | 1.394 | 0.975 | 0.994 |
| 1.3 | oracle | - | - | - | - | - | 0.785 | 0.982 |
| 1.3 | timestral-5M | 1.194 | 1.06 | 1.225 | 1.08 | 0.9536 | 0.828 | 0.979 |
| 1.3 | timestral-28M | 1.202 | 1.081 | 1.217 | 1.067 | 0.96 | 0.804 | 0.981 |
| 1.3 | arima | 1.464 | 1.365 | 1.949 | 1.403 | 1.169 | 0.916 | 0.966 |
| 1.3 | ets | 1.507 | 1.302 | 4.277 | 3.518 | 1.203 | 0.968 | 0.991 |
| 1.3 | seasonal_naive | 1.511 | 1.312 | 12.04 | 5.892 | 1.207 | 0.997 | 0.999 |
| 1.5 | oracle | - | - | - | - | - | 0.744 | 0.957 |
| 1.5 | timestral-5M | 1.086 | 1.068 | 1.099 | 1.096 | 1.024 | 0.764 | 0.950 |
| 1.5 | timestral-28M | 1.073 | 1.08 | 1.083 | 1.078 | 1.012 | 0.744 | 0.951 |
| 1.5 | arima | 1.069 | 1.062 | 1.167 | 1.184 | 1.008 | 0.852 | 0.919 |
| 1.5 | ets | 1.664 | 1.394 | 2.237 | 2.54 | 1.569 | 0.958 | 0.987 |
| 1.5 | seasonal_naive | 1.663 | 1.384 | 2.835 | 3.057 | 1.569 | 0.989 | 1.000 |
| 1.7 | oracle | - | - | - | - | - | 0.844 | 0.978 |
| 1.7 | timestral-5M | 0.9617 | 0.9594 | 0.9788 | 0.974 | 0.9658 | 0.867 | 0.979 |
| 1.7 | timestral-28M | 0.9681 | 0.9718 | 0.9829 | 0.9842 | 0.9722 | 0.856 | 0.982 |
| 1.7 | arima | 0.997 | 0.9962 | 1.052 | 1.074 | 1.001 | 0.900 | 0.977 |
| 1.7 | ets | 1.377 | 1.445 | 2.071 | 1.921 | 1.383 | 0.994 | 1.000 |
| 1.7 | seasonal_naive | 1.388 | 1.443 | 2.564 | 2.394 | 1.394 | 0.998 | 1.000 |
| 1.9 | oracle | - | - | - | - | - | 0.773 | 0.983 |
| 1.9 | timestral-5M | 1.07 | 1.053 | 1.074 | 1.063 | 1.042 | 0.744 | 0.977 |
| 1.9 | timestral-28M | 1.053 | 1.042 | 1.056 | 1.039 | 1.026 | 0.761 | 0.977 |
| 1.9 | arima | 1.027 | 1.019 | 1.033 | 1.024 | 1 | 0.754 | 0.968 |
| 1.9 | ets | 1.351 | 1.175 | 1.752 | 1.752 | 1.315 | 0.967 | 0.999 |
| 1.9 | seasonal_naive | 1.344 | 1.177 | 2.049 | 2.062 | 1.308 | 0.977 | 0.999 |
| 2 | oracle | - | - | - | - | - | 0.775 | 0.963 |
| 2 | timestral-5M | 0.9971 | 0.9958 | 0.9973 | 0.9981 | 1.034 | 0.790 | 0.971 |
| 2 | timestral-28M | 1.005 | 0.9969 | 1.006 | 1 | 1.043 | 0.796 | 0.970 |
| 2 | arima | 1.031 | 1.03 | 1.027 | 1.043 | 1.069 | 0.787 | 0.968 |
| 2 | ets | 1.201 | 1.2 | 1.7 | 1.743 | 1.246 | 0.974 | 0.995 |
| 2 | seasonal_naive | 1.194 | 1.18 | 1.742 | 1.755 | 1.239 | 0.976 | 0.997 |

**tails/stable_ar, rho = 1.0** (knob `alpha`)

| alpha | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |
|---|---|---|---|---|---|---|---|---|
| 1.1 | oracle | - | - | - | - | - | 0.695 | 0.980 |
| 1.1 | timestral-5M | 1.184 | 1.036 | 1.07 | 1.039 | 1.302 | 0.683 | 0.821 |
| 1.1 | timestral-28M | 0.9759 | 0.9863 | 1.011 | 0.9863 | 1.073 | 0.688 | 0.832 |
| 1.1 | arima | 0.9676 | 0.9685 | 1.325 | 1.197 | 1.064 | 0.774 | 0.863 |
| 1.1 | ets | 0.9107 | 0.949 | 1.381 | 1.243 | 1.002 | 0.774 | 0.863 |
| 1.1 | seasonal_naive | 0.9674 | 0.9679 | 1.454 | 1.258 | 1.064 | 0.766 | 0.927 |
| 1.3 | oracle | - | - | - | - | - | 0.618 | 0.985 |
| 1.3 | timestral-5M | 1.085 | 1.082 | 1.164 | 1.157 | 1.088 | 0.631 | 0.964 |
| 1.3 | timestral-28M | 1.345 | 1.125 | 1.267 | 1.138 | 1.35 | 0.639 | 0.966 |
| 1.3 | arima | 1.003 | 1.033 | 1.603 | 1.16 | 1.006 | 0.674 | 0.898 |
| 1.3 | ets | 0.997 | 1.033 | 1.577 | 1.101 | 1 | 0.700 | 0.916 |
| 1.3 | seasonal_naive | 1.003 | 1.033 | 2.189 | 1.747 | 1.006 | 0.882 | 0.949 |
| 1.5 | oracle | - | - | - | - | - | 0.832 | 0.999 |
| 1.5 | timestral-5M | 1.085 | 1.115 | 1.035 | 1.055 | 1.042 | 0.819 | 0.982 |
| 1.5 | timestral-28M | 1.028 | 1.007 | 1.009 | 0.9899 | 0.987 | 0.812 | 0.986 |
| 1.5 | arima | 1.043 | 1.075 | 1.106 | 1.058 | 1.001 | 0.843 | 0.969 |
| 1.5 | ets | 1.047 | 1.072 | 1.113 | 1.058 | 1.005 | 0.843 | 0.969 |
| 1.5 | seasonal_naive | 1.049 | 1.074 | 1.368 | 1.211 | 1.007 | 0.945 | 0.979 |
| 1.7 | oracle | - | - | - | - | - | 0.905 | 1.000 |
| 1.7 | timestral-5M | 1.088 | 1.026 | 1.11 | 1.088 | 1.089 | 0.814 | 0.887 |
| 1.7 | timestral-28M | 1.009 | 1.016 | 1.077 | 1.032 | 1.01 | 0.828 | 0.887 |
| 1.7 | arima | 1.021 | 1.027 | 1.041 | 1.053 | 1.021 | 0.957 | 1.000 |
| 1.7 | ets | 1.009 | 0.987 | 1.061 | 1.073 | 1.009 | 0.957 | 1.000 |
| 1.7 | seasonal_naive | 1.02 | 1.027 | 1.096 | 1.046 | 1.021 | 0.986 | 1.000 |
| 1.9 | oracle | - | - | - | - | - | 0.799 | 0.981 |
| 1.9 | timestral-5M | 0.9668 | 0.9783 | 1.053 | 1.022 | 0.9855 | 0.692 | 0.893 |
| 1.9 | timestral-28M | 0.9497 | 0.9736 | 1.019 | 1.01 | 0.9681 | 0.671 | 0.892 |
| 1.9 | arima | 0.9867 | 0.9787 | 0.9807 | 0.9864 | 1.006 | 0.831 | 0.986 |
| 1.9 | ets | 0.9867 | 0.9851 | 0.9841 | 0.991 | 1.006 | 0.839 | 0.987 |
| 1.9 | seasonal_naive | 0.9859 | 0.9787 | 1.013 | 0.9997 | 1.005 | 0.897 | 0.980 |
| 2 | oracle | - | - | - | - | - | 0.917 | 1.000 |
| 2 | timestral-5M | 0.9875 | 1.02 | 0.9911 | 1.009 | 1.004 | 0.930 | 1.000 |
| 2 | timestral-28M | 0.9997 | 0.984 | 0.9931 | 0.9811 | 1.016 | 0.931 | 1.000 |
| 2 | arima | 0.9923 | 0.9833 | 0.9798 | 0.9729 | 1.009 | 0.922 | 1.000 |
| 2 | ets | 1.004 | 0.9759 | 1.006 | 1.001 | 1.02 | 0.926 | 1.000 |
| 2 | seasonal_naive | 0.9942 | 0.9846 | 0.9891 | 0.9921 | 1.011 | 0.922 | 1.000 |

## 123M and 209M models

The 123M and 209M runs are not in this folder: they moved to Colab job A (full grid). This folder holds the
5M and 28M main sweep only.

## Oracle calibration check

`oracle_calibration.py` builds 100 independent samples per config (context 1024, horizon 128,
500 oracle paths) and measures how often the realized future falls inside the oracle's own 80% and 98% intervals
(`oracle_calibration.csv`; `se` = standard error over samples). A calibrated oracle gives 0.80 and 0.98.
The oracle cov80 and cov98 in the tables above come from 8 samples per config at 128 correlated steps, so they
move more than these. Hawkes counts are integers, so an interval with inclusive end points covers more than its nominal level.

| config | oracle cov80 | se | oracle cov98 | se |
|---|---|---|---|---|
| tails/stable_ar/alpha2_rho0.9 | 0.802 | 0.009 | 0.976 | 0.003 |
| tails/stable_ar/alpha1.9_rho0.9 | 0.786 | 0.010 | 0.975 | 0.004 |
| tails/stable_ar/alpha1.7_rho0.9 | 0.813 | 0.011 | 0.985 | 0.003 |
| tails/stable_ar/alpha1.5_rho0.9 | 0.801 | 0.011 | 0.978 | 0.004 |
| tails/stable_ar/alpha1.3_rho0.9 | 0.783 | 0.014 | 0.975 | 0.006 |
| tails/stable_ar/alpha1.1_rho0.9 | 0.796 | 0.012 | 0.969 | 0.006 |
| tails/stable_ar/alpha2_rho1.0 | 0.755 | 0.026 | 0.968 | 0.009 |
| tails/stable_ar/alpha1.9_rho1.0 | 0.778 | 0.027 | 0.978 | 0.007 |
| tails/stable_ar/alpha1.7_rho1.0 | 0.739 | 0.028 | 0.959 | 0.015 |
| tails/stable_ar/alpha1.5_rho1.0 | 0.800 | 0.027 | 0.982 | 0.007 |
| tails/stable_ar/alpha1.3_rho1.0 | 0.765 | 0.030 | 0.953 | 0.016 |
| tails/stable_ar/alpha1.1_rho1.0 | 0.833 | 0.025 | 0.981 | 0.008 |
| tails/fbm/H0.1 | 0.769 | 0.015 | 0.965 | 0.006 |
| tails/fbm/H0.2 | 0.779 | 0.018 | 0.971 | 0.006 |
| tails/fbm/H0.3 | 0.772 | 0.022 | 0.980 | 0.005 |
| tails/fbm/H0.5 | 0.815 | 0.024 | 0.971 | 0.010 |
| tails/fbm/H0.7 | 0.800 | 0.025 | 0.983 | 0.007 |
| tails/fbm/H0.8 | 0.807 | 0.028 | 0.978 | 0.008 |
| tails/fbm/H0.9 | 0.830 | 0.024 | 0.988 | 0.006 |
| tails/hawkes/eta0 | 0.886 | 0.003 | 0.986 | 0.001 |
| tails/hawkes/eta0.3 | 0.877 | 0.003 | 0.987 | 0.001 |
| tails/hawkes/eta0.6 | 0.871 | 0.004 | 0.986 | 0.001 |
| tails/hawkes/eta0.8 | 0.865 | 0.004 | 0.988 | 0.001 |
| tails/hawkes/eta0.9 | 0.868 | 0.008 | 0.992 | 0.002 |
| tails/hawkes/eta0.95 | 0.833 | 0.014 | 0.983 | 0.006 |
| tails/msm/k1_ret | 0.789 | 0.004 | 0.973 | 0.001 |
| tails/msm/k2_ret | 0.802 | 0.004 | 0.979 | 0.001 |
| tails/msm/k4_ret | 0.796 | 0.007 | 0.976 | 0.002 |
| tails/msm/k6_ret | 0.796 | 0.009 | 0.975 | 0.003 |
| tails/msm/k8_ret | 0.792 | 0.009 | 0.975 | 0.003 |
| tails/msm/k1_cum | 0.808 | 0.025 | 0.966 | 0.012 |
| tails/msm/k2_cum | 0.767 | 0.024 | 0.981 | 0.006 |
| tails/msm/k4_cum | 0.767 | 0.027 | 0.959 | 0.013 |
| tails/msm/k6_cum | 0.851 | 0.020 | 0.978 | 0.009 |
| tails/msm/k8_cum | 0.816 | 0.026 | 0.984 | 0.005 |
