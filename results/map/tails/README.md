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

Sweep: 35 configs, 24 seeds each (840 samples), Timestral 28M, 5M,
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
| 0.1 | timestral-5M | 1.034 | 1.033 | 1.04 | 1.04 | 0.9975 | 0.820 | 0.976 |
| 0.1 | timestral-28M | 1.025 | 1.017 | 1.028 | 1.025 | 0.9886 | 0.812 | 0.979 |
| 0.1 | arima | 1.102 | 1.076 | 1.107 | 1.092 | 1.063 | 0.765 | 0.965 |
| 0.1 | ets | 1.164 | 1.027 | 1.294 | 1.286 | 1.123 | 0.894 | 0.998 |
| 0.1 | seasonal_naive | 1.312 | 1.318 | 3.029 | 3.221 | 1.266 | 0.995 | 1.000 |
| 0.2 | oracle | - | - | - | - | - | 0.813 | 0.985 |
| 0.2 | timestral-5M | 0.9872 | 0.9884 | 0.9791 | 0.982 | 1.001 | 0.813 | 0.992 |
| 0.2 | timestral-28M | 0.9904 | 0.9989 | 0.9933 | 0.9983 | 1.005 | 0.816 | 0.991 |
| 0.2 | arima | 1.106 | 1.031 | 1.144 | 1.076 | 1.122 | 0.758 | 0.946 |
| 0.2 | ets | 1.071 | 1.073 | 1.263 | 1.36 | 1.086 | 0.976 | 1.000 |
| 0.2 | seasonal_naive | 1.12 | 1.043 | 2.038 | 2.339 | 1.136 | 0.999 | 1.000 |
| 0.3 | oracle | - | - | - | - | - | 0.787 | 0.992 |
| 0.3 | timestral-5M | 1.008 | 1.014 | 1.025 | 1.026 | 1.004 | 0.780 | 0.994 |
| 0.3 | timestral-28M | 1 | 1.009 | 1.016 | 1.002 | 0.997 | 0.781 | 0.991 |
| 0.3 | arima | 1.11 | 1.077 | 1.172 | 1.137 | 1.107 | 0.746 | 0.998 |
| 0.3 | ets | 1.047 | 0.9627 | 1.19 | 1.198 | 1.043 | 0.938 | 1.000 |
| 0.3 | seasonal_naive | 1.061 | 0.9823 | 1.422 | 1.531 | 1.058 | 0.990 | 1.000 |
| 0.5 | oracle | - | - | - | - | - | 0.689 | 0.973 |
| 0.5 | timestral-5M | 0.9659 | 0.9879 | 1.017 | 1.03 | 0.9723 | 0.677 | 0.982 |
| 0.5 | timestral-28M | 0.9493 | 0.9248 | 0.9854 | 0.9506 | 0.9556 | 0.712 | 0.971 |
| 0.5 | arima | 0.9956 | 0.9927 | 0.9922 | 0.9906 | 1.002 | 0.682 | 0.968 |
| 0.5 | ets | 1.001 | 0.9943 | 1.003 | 0.9983 | 1.007 | 0.704 | 0.976 |
| 0.5 | seasonal_naive | 0.9975 | 0.9954 | 0.9954 | 0.9921 | 1.004 | 0.694 | 0.962 |
| 0.7 | oracle | - | - | - | - | - | 0.927 | 1.000 |
| 0.7 | timestral-5M | 1.097 | 1.084 | 1.086 | 1.069 | 0.9886 | 0.777 | 1.000 |
| 0.7 | timestral-28M | 1.1 | 1.128 | 1.099 | 1.064 | 0.9913 | 0.757 | 1.000 |
| 0.7 | arima | 1.168 | 1.393 | 1.089 | 1.217 | 1.053 | 0.774 | 0.990 |
| 0.7 | ets | 1.209 | 1.364 | 1.06 | 1.231 | 1.089 | 0.781 | 0.977 |
| 0.7 | seasonal_naive | 1.283 | 1.436 | 1.183 | 1.29 | 1.156 | 0.430 | 0.805 |
| 0.8 | oracle | - | - | - | - | - | 0.912 | 0.996 |
| 0.8 | timestral-5M | 1.167 | 1.093 | 1.013 | 1.005 | 0.7783 | 0.899 | 0.996 |
| 0.8 | timestral-28M | 1.198 | 1.09 | 1.06 | 1.029 | 0.7992 | 0.877 | 0.999 |
| 0.8 | arima | 1.578 | 1.752 | 1.22 | 1.238 | 1.053 | 0.877 | 0.992 |
| 0.8 | ets | 1.993 | 1.912 | 1.551 | 1.492 | 1.329 | 0.611 | 0.983 |
| 0.8 | seasonal_naive | 2.18 | 2.392 | 1.989 | 2.076 | 1.454 | 0.179 | 0.405 |
| 0.9 | oracle | - | - | - | - | - | 0.656 | 0.980 |
| 0.9 | timestral-5M | 1.003 | 1.005 | 1.03 | 1.054 | 1.335 | 0.657 | 1.000 |
| 0.9 | timestral-28M | 1.019 | 1.026 | 1.045 | 1.043 | 1.356 | 0.596 | 1.000 |
| 0.9 | arima | 1.053 | 1.068 | 1.061 | 1.074 | 1.401 | 0.604 | 0.986 |
| 0.9 | ets | 1.054 | 1.224 | 1.133 | 1.248 | 1.403 | 0.622 | 0.905 |
| 0.9 | seasonal_naive | 1.19 | 1.207 | 1.568 | 1.623 | 1.583 | 0.204 | 0.376 |

**tails/hawkes** (knob `eta`)

| eta | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |
|---|---|---|---|---|---|---|---|---|
| 0 | oracle | - | - | - | - | - | 0.889 | 0.985 |
| 0 | timestral-5M | 1.012 | 1.012 | 1.01 | 1.01 | 1.009 | 0.784 | 0.975 |
| 0 | timestral-28M | 1.011 | 1.012 | 1.01 | 1.009 | 1.008 | 0.773 | 0.973 |
| 0 | arima | 1.003 | 1.002 | 1.012 | 1.012 | 1.001 | 0.751 | 0.976 |
| 0 | ets | 1.003 | 1.003 | 1.012 | 1.012 | 1 | 0.751 | 0.976 |
| 0 | seasonal_naive | 1.493 | 1.29 | 4.697 | 4.71 | 1.489 | 0.997 | 1.000 |
| 0.3 | oracle | - | - | - | - | - | 0.883 | 0.989 |
| 0.3 | timestral-5M | 1.005 | 1.004 | 1.008 | 1.009 | 1 | 0.792 | 0.979 |
| 0.3 | timestral-28M | 1.005 | 1.006 | 1.007 | 1.008 | 1.001 | 0.774 | 0.984 |
| 0.3 | arima | 1.008 | 1.008 | 1.011 | 1.013 | 1.003 | 0.778 | 0.983 |
| 0.3 | ets | 1.017 | 1.01 | 1.014 | 1.013 | 1.012 | 0.812 | 0.982 |
| 0.3 | seasonal_naive | 1.547 | 1.503 | 4.575 | 4.515 | 1.54 | 0.994 | 1.000 |
| 0.6 | oracle | - | - | - | - | - | 0.853 | 0.982 |
| 0.6 | timestral-5M | 1.025 | 1.015 | 1.019 | 1.011 | 1.021 | 0.795 | 0.978 |
| 0.6 | timestral-28M | 1.019 | 1.015 | 1.014 | 1.01 | 1.015 | 0.768 | 0.980 |
| 0.6 | arima | 1.021 | 1.016 | 1.021 | 1.019 | 1.018 | 0.784 | 0.977 |
| 0.6 | ets | 1.086 | 1.075 | 1.102 | 1.097 | 1.082 | 0.872 | 0.988 |
| 0.6 | seasonal_naive | 1.357 | 1.154 | 4.074 | 4.219 | 1.352 | 0.998 | 1.000 |
| 0.8 | oracle | - | - | - | - | - | 0.828 | 0.990 |
| 0.8 | timestral-5M | 1.001 | 1.021 | 1.02 | 1.023 | 1.031 | 0.780 | 0.975 |
| 0.8 | timestral-28M | 0.9959 | 1.016 | 1.015 | 1.035 | 1.026 | 0.780 | 0.972 |
| 0.8 | arima | 0.9768 | 0.9832 | 0.9993 | 1.006 | 1.006 | 0.778 | 0.973 |
| 0.8 | ets | 1.117 | 1.077 | 1.15 | 1.108 | 1.151 | 0.851 | 0.990 |
| 0.8 | seasonal_naive | 1.287 | 1.251 | 3.705 | 3.683 | 1.325 | 0.996 | 1.000 |
| 0.9 | oracle | - | - | - | - | - | 0.896 | 0.997 |
| 0.9 | timestral-5M | 1.045 | 1.042 | 1.078 | 1.057 | 1.021 | 0.840 | 0.987 |
| 0.9 | timestral-28M | 1.079 | 1.093 | 1.108 | 1.097 | 1.054 | 0.812 | 0.986 |
| 0.9 | arima | 1.062 | 1.033 | 1.078 | 1.063 | 1.037 | 0.831 | 0.982 |
| 0.9 | ets | 1.377 | 1.293 | 1.41 | 1.352 | 1.346 | 0.831 | 0.989 |
| 0.9 | seasonal_naive | 1.672 | 1.645 | 3.668 | 4.042 | 1.633 | 0.996 | 1.000 |
| 0.95 | oracle | - | - | - | - | - | 0.901 | 0.993 |
| 0.95 | timestral-5M | 1.061 | 1.042 | 1.058 | 1.069 | 1.109 | 0.806 | 0.980 |
| 0.95 | timestral-28M | 1.034 | 1.019 | 1.035 | 1.028 | 1.08 | 0.802 | 0.986 |
| 0.95 | arima | 1.051 | 0.9804 | 1.089 | 1.079 | 1.098 | 0.890 | 0.980 |
| 0.95 | ets | 1.172 | 1.203 | 1.217 | 1.192 | 1.225 | 0.905 | 0.979 |
| 0.95 | seasonal_naive | 1.024 | 1.09 | 3.283 | 3.464 | 1.07 | 0.996 | 1.000 |

**tails/msm, observed series = cumulative sum** (knob `k`)

| k | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |
|---|---|---|---|---|---|---|---|---|
| 1 | oracle | - | - | - | - | - | 0.856 | 0.997 |
| 1 | timestral-5M | 1.03 | 1.022 | 1.023 | 1.035 | 1.057 | 0.852 | 0.963 |
| 1 | timestral-28M | 1.032 | 1.036 | 1.02 | 1.038 | 1.059 | 0.854 | 0.963 |
| 1 | arima | 1.001 | 1.009 | 1 | 1.005 | 1.027 | 0.862 | 0.996 |
| 1 | ets | 0.9752 | 1.007 | 0.9701 | 0.9943 | 1.001 | 0.883 | 0.997 |
| 1 | seasonal_naive | 1.001 | 1.009 | 1.002 | 1.008 | 1.027 | 0.862 | 0.997 |
| 2 | oracle | - | - | - | - | - | 0.747 | 0.946 |
| 2 | timestral-5M | 0.9427 | 0.9395 | 0.9615 | 0.961 | 0.9433 | 0.751 | 0.979 |
| 2 | timestral-28M | 0.9375 | 0.95 | 0.9426 | 0.9644 | 0.9381 | 0.755 | 0.966 |
| 2 | arima | 1.005 | 1.002 | 1.01 | 1.017 | 1.006 | 0.730 | 0.952 |
| 2 | ets | 1.006 | 1.01 | 1.036 | 1.017 | 1.006 | 0.743 | 0.938 |
| 2 | seasonal_naive | 1.003 | 1.007 | 1.004 | 1.006 | 1.004 | 0.749 | 0.945 |
| 4 | oracle | - | - | - | - | - | 0.877 | 0.999 |
| 4 | timestral-5M | 1.039 | 1.014 | 1.003 | 0.9895 | 1.038 | 0.887 | 0.999 |
| 4 | timestral-28M | 1.015 | 1.023 | 0.9702 | 0.9886 | 1.014 | 0.891 | 0.999 |
| 4 | arima | 1.009 | 0.9963 | 1 | 0.9926 | 1.007 | 0.902 | 0.999 |
| 4 | ets | 1.021 | 1.006 | 1.012 | 0.983 | 1.019 | 0.909 | 0.999 |
| 4 | seasonal_naive | 1.008 | 0.9966 | 0.9927 | 0.9799 | 1.007 | 0.902 | 0.999 |
| 6 | oracle | - | - | - | - | - | 0.933 | 1.000 |
| 6 | timestral-5M | 1.029 | 0.9889 | 0.9968 | 0.9812 | 1.016 | 0.883 | 1.000 |
| 6 | timestral-28M | 1.076 | 1.011 | 1.028 | 0.9976 | 1.063 | 0.875 | 1.000 |
| 6 | arima | 1.163 | 1.016 | 1.116 | 1.027 | 1.149 | 0.817 | 0.980 |
| 6 | ets | 1.093 | 1.032 | 1.053 | 1.033 | 1.079 | 0.894 | 0.995 |
| 6 | seasonal_naive | 1.016 | 1.009 | 1.009 | 1.019 | 1.004 | 0.886 | 0.995 |
| 8 | oracle | - | - | - | - | - | 0.900 | 0.995 |
| 8 | timestral-5M | 1.03 | 1.013 | 1.057 | 1.068 | 1.068 | 0.805 | 0.969 |
| 8 | timestral-28M | 1.07 | 0.9982 | 1.094 | 1.115 | 1.109 | 0.794 | 0.949 |
| 8 | arima | 0.9831 | 0.9906 | 1.181 | 1.161 | 1.019 | 0.835 | 0.965 |
| 8 | ets | 0.9668 | 0.9739 | 1.226 | 1.203 | 1.002 | 0.841 | 0.979 |
| 8 | seasonal_naive | 0.9836 | 0.9936 | 1.261 | 1.231 | 1.02 | 0.859 | 0.950 |

**tails/msm, observed series = returns** (knob `k`)

| k | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |
|---|---|---|---|---|---|---|---|---|
| 1 | oracle | - | - | - | - | - | 0.797 | 0.977 |
| 1 | timestral-5M | 0.9984 | 0.9987 | 0.998 | 0.9985 | 0.9992 | 0.815 | 0.984 |
| 1 | timestral-28M | 0.9988 | 0.9998 | 0.9982 | 0.9981 | 0.9996 | 0.814 | 0.974 |
| 1 | arima | 0.9995 | 1 | 0.9992 | 0.9992 | 1 | 0.818 | 0.970 |
| 1 | ets | 0.9998 | 0.9998 | 0.9996 | 0.9989 | 1.001 | 0.817 | 0.970 |
| 1 | seasonal_naive | 1.256 | 1.192 | 4.597 | 4.557 | 1.257 | 0.999 | 1.000 |
| 2 | oracle | - | - | - | - | - | 0.806 | 0.985 |
| 2 | timestral-5M | 0.9993 | 0.9993 | 0.9999 | 1.001 | 1.002 | 0.809 | 0.983 |
| 2 | timestral-28M | 0.9992 | 0.9988 | 0.9993 | 1 | 1.002 | 0.807 | 0.980 |
| 2 | arima | 0.9993 | 0.9994 | 1.001 | 1.003 | 1.002 | 0.825 | 0.968 |
| 2 | ets | 0.9997 | 0.9991 | 1.002 | 1.003 | 1.003 | 0.824 | 0.967 |
| 2 | seasonal_naive | 1.318 | 1.288 | 4.73 | 4.857 | 1.322 | 0.998 | 1.000 |
| 4 | oracle | - | - | - | - | - | 0.781 | 0.974 |
| 4 | timestral-5M | 1.005 | 1.003 | 1.01 | 1.011 | 1.001 | 0.773 | 0.970 |
| 4 | timestral-28M | 1.002 | 1.003 | 1.008 | 1.009 | 0.999 | 0.764 | 0.969 |
| 4 | arima | 1.004 | 1.002 | 1.007 | 1.01 | 1 | 0.802 | 0.952 |
| 4 | ets | 1.003 | 1.001 | 1.007 | 1.009 | 1 | 0.801 | 0.954 |
| 4 | seasonal_naive | 1.602 | 1.275 | 4.566 | 4.542 | 1.597 | 0.994 | 0.998 |
| 6 | oracle | - | - | - | - | - | 0.827 | 0.975 |
| 6 | timestral-5M | 0.9974 | 0.9957 | 1.014 | 1.01 | 1 | 0.855 | 0.974 |
| 6 | timestral-28M | 0.998 | 0.9979 | 1.016 | 1.014 | 1.001 | 0.853 | 0.974 |
| 6 | arima | 0.9979 | 0.9982 | 1.027 | 1.017 | 1 | 0.872 | 0.966 |
| 6 | ets | 0.998 | 0.9974 | 1.027 | 1.017 | 1.001 | 0.873 | 0.965 |
| 6 | seasonal_naive | 1.44 | 1.171 | 5.442 | 5.452 | 1.443 | 0.999 | 1.000 |
| 8 | oracle | - | - | - | - | - | 0.779 | 0.974 |
| 8 | timestral-5M | 1 | 1 | 1.01 | 1.013 | 1.001 | 0.775 | 0.967 |
| 8 | timestral-28M | 0.9994 | 0.9999 | 1.006 | 1.006 | 0.9998 | 0.765 | 0.968 |
| 8 | arima | 0.9998 | 1.001 | 1.055 | 1.032 | 1 | 0.869 | 0.966 |
| 8 | ets | 1 | 1.001 | 1.055 | 1.031 | 1 | 0.869 | 0.966 |
| 8 | seasonal_naive | 1.331 | 1.236 | 6.677 | 7.356 | 1.332 | 0.999 | 1.000 |

**tails/stable_ar, rho = 0.9** (knob `alpha`)

| alpha | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |
|---|---|---|---|---|---|---|---|---|
| 1.1 | oracle | - | - | - | - | - | 0.888 | 0.992 |
| 1.1 | timestral-5M | 1.111 | 1.011 | 1.204 | 1.051 | 1.007 | 0.902 | 0.984 |
| 1.1 | timestral-28M | 1.048 | 1.02 | 1.145 | 1.02 | 0.9501 | 0.880 | 0.984 |
| 1.1 | arima | 1.691 | 1.029 | 2.818 | 1.983 | 1.533 | 0.965 | 0.979 |
| 1.1 | ets | 1.537 | 1.454 | 7.502 | 5.717 | 1.393 | 0.906 | 0.969 |
| 1.1 | seasonal_naive | 1.538 | 1.464 | 15.05 | 13.52 | 1.394 | 0.975 | 0.994 |
| 1.3 | oracle | - | - | - | - | - | 0.785 | 0.982 |
| 1.3 | timestral-5M | 0.9869 | 0.9987 | 1.012 | 0.9989 | 0.9536 | 0.828 | 0.979 |
| 1.3 | timestral-28M | 0.9935 | 0.9908 | 1.006 | 1.005 | 0.96 | 0.804 | 0.981 |
| 1.3 | arima | 1.21 | 1.065 | 1.61 | 1.201 | 1.169 | 0.916 | 0.966 |
| 1.3 | ets | 1.245 | 1.155 | 3.534 | 2.977 | 1.203 | 0.968 | 0.991 |
| 1.3 | seasonal_naive | 1.249 | 1.168 | 9.945 | 5.238 | 1.207 | 0.997 | 0.999 |
| 1.5 | oracle | - | - | - | - | - | 0.744 | 0.957 |
| 1.5 | timestral-5M | 1.018 | 1.02 | 1.03 | 1.026 | 1.024 | 0.764 | 0.950 |
| 1.5 | timestral-28M | 1.005 | 1.012 | 1.015 | 1.016 | 1.012 | 0.744 | 0.951 |
| 1.5 | arima | 1.002 | 1.019 | 1.094 | 1.05 | 1.008 | 0.852 | 0.919 |
| 1.5 | ets | 1.56 | 1.37 | 2.097 | 2.411 | 1.569 | 0.958 | 0.987 |
| 1.5 | seasonal_naive | 1.559 | 1.359 | 2.657 | 2.665 | 1.569 | 0.989 | 1.000 |
| 1.7 | oracle | - | - | - | - | - | 0.844 | 0.978 |
| 1.7 | timestral-5M | 0.9819 | 0.9927 | 0.9993 | 1.003 | 0.9658 | 0.867 | 0.979 |
| 1.7 | timestral-28M | 0.9883 | 1.01 | 1.003 | 1.017 | 0.9722 | 0.856 | 0.982 |
| 1.7 | arima | 1.018 | 1.01 | 1.074 | 1.059 | 1.001 | 0.900 | 0.977 |
| 1.7 | ets | 1.406 | 1.445 | 2.114 | 1.955 | 1.383 | 0.994 | 1.000 |
| 1.7 | seasonal_naive | 1.417 | 1.457 | 2.618 | 2.386 | 1.394 | 0.998 | 1.000 |
| 1.9 | oracle | - | - | - | - | - | 0.773 | 0.983 |
| 1.9 | timestral-5M | 1.055 | 1.043 | 1.06 | 1.047 | 1.042 | 0.744 | 0.977 |
| 1.9 | timestral-28M | 1.039 | 1.02 | 1.042 | 1.025 | 1.026 | 0.761 | 0.977 |
| 1.9 | arima | 1.013 | 0.9984 | 1.019 | 1.003 | 1 | 0.754 | 0.968 |
| 1.9 | ets | 1.332 | 1.134 | 1.728 | 1.721 | 1.315 | 0.967 | 0.999 |
| 1.9 | seasonal_naive | 1.325 | 1.136 | 2.021 | 2.084 | 1.308 | 0.977 | 0.999 |
| 2 | oracle | - | - | - | - | - | 0.775 | 0.963 |
| 2 | timestral-5M | 0.9989 | 0.9907 | 0.9991 | 0.9867 | 1.034 | 0.790 | 0.971 |
| 2 | timestral-28M | 1.007 | 0.9962 | 1.008 | 1.004 | 1.043 | 0.796 | 0.970 |
| 2 | arima | 1.033 | 1.019 | 1.029 | 1.032 | 1.069 | 0.787 | 0.968 |
| 2 | ets | 1.203 | 1.203 | 1.703 | 1.755 | 1.246 | 0.974 | 0.995 |
| 2 | seasonal_naive | 1.196 | 1.183 | 1.745 | 1.773 | 1.239 | 0.976 | 0.997 |

**tails/stable_ar, rho = 1.0** (knob `alpha`)

| alpha | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |
|---|---|---|---|---|---|---|---|---|
| 1.1 | oracle | - | - | - | - | - | 0.695 | 0.980 |
| 1.1 | timestral-5M | 1.217 | 1.011 | 1.1 | 1.041 | 1.302 | 0.683 | 0.821 |
| 1.1 | timestral-28M | 1.003 | 1.017 | 1.039 | 1.036 | 1.073 | 0.688 | 0.832 |
| 1.1 | arima | 0.9945 | 0.9942 | 1.362 | 1.115 | 1.064 | 0.774 | 0.863 |
| 1.1 | ets | 0.936 | 0.9931 | 1.419 | 1.163 | 1.002 | 0.774 | 0.863 |
| 1.1 | seasonal_naive | 0.9943 | 0.9942 | 1.494 | 1.264 | 1.064 | 0.766 | 0.927 |
| 1.3 | oracle | - | - | - | - | - | 0.618 | 0.985 |
| 1.3 | timestral-5M | 1.079 | 1.038 | 1.158 | 1.072 | 1.088 | 0.631 | 0.964 |
| 1.3 | timestral-28M | 1.339 | 1.016 | 1.261 | 1.054 | 1.35 | 0.639 | 0.966 |
| 1.3 | arima | 0.9978 | 1.001 | 1.595 | 1.059 | 1.006 | 0.674 | 0.898 |
| 1.3 | ets | 0.9921 | 0.9966 | 1.57 | 1.057 | 1 | 0.700 | 0.916 |
| 1.3 | seasonal_naive | 0.9978 | 1.001 | 2.178 | 1.731 | 1.006 | 0.882 | 0.949 |
| 1.5 | oracle | - | - | - | - | - | 0.832 | 0.999 |
| 1.5 | timestral-5M | 1.047 | 0.9946 | 0.9984 | 0.9744 | 1.042 | 0.819 | 0.982 |
| 1.5 | timestral-28M | 0.9921 | 0.9927 | 0.974 | 0.9506 | 0.987 | 0.812 | 0.986 |
| 1.5 | arima | 1.006 | 1.009 | 1.068 | 1.011 | 1.001 | 0.843 | 0.969 |
| 1.5 | ets | 1.011 | 1.01 | 1.074 | 1.02 | 1.005 | 0.843 | 0.969 |
| 1.5 | seasonal_naive | 1.012 | 1.01 | 1.32 | 1.186 | 1.007 | 0.945 | 0.979 |
| 1.7 | oracle | - | - | - | - | - | 0.905 | 1.000 |
| 1.7 | timestral-5M | 1.072 | 0.9902 | 1.094 | 1.089 | 1.089 | 0.814 | 0.887 |
| 1.7 | timestral-28M | 0.9945 | 1.013 | 1.061 | 1.013 | 1.01 | 0.828 | 0.887 |
| 1.7 | arima | 1.006 | 0.9998 | 1.026 | 1.014 | 1.021 | 0.957 | 1.000 |
| 1.7 | ets | 0.9942 | 0.9876 | 1.046 | 1.048 | 1.009 | 0.957 | 1.000 |
| 1.7 | seasonal_naive | 1.006 | 0.9998 | 1.08 | 1.052 | 1.021 | 0.986 | 1.000 |
| 1.9 | oracle | - | - | - | - | - | 0.799 | 0.981 |
| 1.9 | timestral-5M | 0.9916 | 0.9902 | 1.08 | 1.045 | 0.9855 | 0.692 | 0.893 |
| 1.9 | timestral-28M | 0.9741 | 1.003 | 1.045 | 1.021 | 0.9681 | 0.671 | 0.892 |
| 1.9 | arima | 1.012 | 1.007 | 1.006 | 1.004 | 1.006 | 0.831 | 0.986 |
| 1.9 | ets | 1.012 | 1.015 | 1.009 | 1.017 | 1.006 | 0.839 | 0.987 |
| 1.9 | seasonal_naive | 1.011 | 1.008 | 1.039 | 1.002 | 1.005 | 0.897 | 0.980 |
| 2 | oracle | - | - | - | - | - | 0.917 | 1.000 |
| 2 | timestral-5M | 0.9927 | 1.032 | 0.9964 | 1.021 | 1.004 | 0.930 | 1.000 |
| 2 | timestral-28M | 1.005 | 0.9953 | 0.9983 | 0.9884 | 1.016 | 0.931 | 1.000 |
| 2 | arima | 0.9975 | 0.9924 | 0.985 | 0.9853 | 1.009 | 0.922 | 1.000 |
| 2 | ets | 1.009 | 0.9902 | 1.011 | 1.011 | 1.02 | 0.926 | 1.000 |
| 2 | seasonal_naive | 0.9995 | 0.9941 | 0.9944 | 0.9987 | 1.011 | 0.922 | 1.000 |

## 123M and 209M models

The 123M and 209M runs are not in this folder: they moved to Colab job A (full grid). This folder holds the
5M and 28M main sweep only.

## Oracle calibration check

`oracle_calibration.py` builds 100 independent samples per config (context 1024, horizon 128,
500 oracle paths) and measures how often the realized future falls inside the oracle's own 80% and 98% intervals
(`oracle_calibration.csv`; `se` = standard error over samples). A calibrated oracle gives 0.80 and 0.98.
The oracle cov80 and cov98 in the tables above come from 24 samples per config at 128 correlated steps, so they
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
