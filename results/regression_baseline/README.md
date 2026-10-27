# Regression baseline: Timestral sizes and classical models

Baseline scores of the four released Timestral sizes (5M, 28M, 123M, 209M), seasonal naive, ETS and ARIMA
on the regression sets in `bench/regression/` (see the README there for how the sets are built). A fine-tuned
model is checked against these numbers to see whether normal forecasting got worse. Real-data numbers are not
a leaderboard claim; the data may overlap Timestral's pretraining data in unknown ways.

Files: `scores.csv` (one row per window and forecaster), `summary.csv` (the tables below),
`provenance.json` (command, commit, host, compute seconds, subsets). Produced by `scripts/score_regression.py`;
`parts/` holds the cached chunks and is not committed.

Metrics (definitions in `otp/metrics.py`): MASE is the mean absolute error of the median forecast divided by the
in-sample seasonal-naive error of the context. sCRPS is the average pinball loss over the 21 quantile levels, times 2,
divided by the same scale. WQL is the GIFT-Eval weighted quantile loss on levels 0.1 to 0.9. cov80 is the fraction of
horizon points inside the 0.1 to 0.9 interval (0.8 is perfectly calibrated). calib is the mean over the 21 levels of
the absolute gap between empirical coverage and the level (0 is perfect).

Aggregation, as GIFT-Eval does it: for each group (dataset/term pair in the real sets, family/dataset in the
analogue set, seasonal period in the in-prior set) a forecaster's score is the mean over windows of its MASE or
sCRPS; the relative score is that value divided by seasonal naive's value for the same group; the overall relative
score is the geometric mean of the group relative scores over groups. The `ALL` raw metrics are means over windows.
Windows with a non-finite metric value are left out of that metric (WQL is undefined for windows whose horizon
values are all zero). The oracle exists only for the in-prior set, where `*_rel_oracle` is computed the same way
against it; it is never used for the real sets.

`summary.csv` has two bases. `own`: each forecaster on the windows it scored, with seasonal naive (and the oracle)
restricted to the same windows. `common`: every forecaster on the windows that all forecasters of the set scored.
The tables show the `common` basis, then the `own` basis.

WQL note, real set: the raw WQL columns are means over windows. For seasonal naive the median window WQL is 0.2765 and the largest is 1.229e+17 (window `bitbrains_fast_storage/H/fastStorage_19`, MASE 0), so the mean WQL is set by a few windows with a near-zero or zero in-sample scale; MASE, sCRPS and the relative scores are the columns to compare.

WQL note, real_extra set: the raw WQL columns are means over windows. For seasonal naive the median window WQL is 0.5372 and the largest is 3.23e+18 (window `real_extra/owid_covid_new_cases/Japan`, MASE 0), so the mean WQL is set by a few windows with a near-zero or zero in-sample scale; MASE, sCRPS and the relative scores are the columns to compare.

WQL note, analogues set: the raw WQL columns are means over windows. For seasonal naive the median window WQL is 0.8163 and the largest is 220 (window `analogues/monash_sunspot_daily/T1`, MASE 0.8155), so the mean WQL is set by a few windows with a near-zero or zero in-sample scale; MASE, sCRPS and the relative scores are the columns to compare.

Platforms. Chunks of 40 windows were scored on different machines (`chunk_platforms.csv` lists set, forecaster, chunk numbers and platform). Seasonal naive and the oracle are deterministic. ETS is fitted by a numerical optimizer and is not bit-identical across platforms: a cross-check of one ETS chunk between the laptop and the VM found small differences in per-window scores, while ARIMA agreed more closely. So the ETS rows of the real set mix 3 laptop chunks (Windows) with 8 VM chunks (Linux), and those of the in-prior set mix 2 laptop chunks with 11 VM chunks. Timestral was run on CPU (VM) for the real and in-prior sets and on a Colab T4 GPU for the extra real and analogue sets.

| set | forecaster | chunks | platform |
|---|---|---|---|
| analogues | seasonal_naive | 000-001 | laptop (Windows; Python 3.12.10, numpy 2.4.6, scipy 1.18.0, statsmodels 0.15.0) |
| analogues | ets | 000-001 | laptop (Windows; Python 3.12.10, numpy 2.4.6, scipy 1.18.0, statsmodels 0.15.0) |
| analogues | arima | 000-001 | laptop (Windows; Python 3.12.10, numpy 2.4.6, scipy 1.18.0, statsmodels 0.15.0) |
| analogues | timestral-5M | 000-001 | Colab T4 GPU (job log in results/colab/rd-timestral/) |
| analogues | timestral-28M | 000-001 | Colab T4 GPU (job log in results/colab/rd-timestral/) |
| analogues | timestral-123M | 000-001 | Colab T4 GPU (job log in results/colab/rd-timestral/) |
| analogues | timestral-209M | 000-001 | Colab T4 GPU (job log in results/colab/rd-timestral/) |
| real_extra | seasonal_naive | 000-001 | laptop (Windows; Python 3.12.10, numpy 2.4.6, scipy 1.18.0, statsmodels 0.15.0) |
| real_extra | ets | 000-001 | laptop (Windows; Python 3.12.10, numpy 2.4.6, scipy 1.18.0, statsmodels 0.15.0) |
| real_extra | arima | 000-001 | laptop (Windows; Python 3.12.10, numpy 2.4.6, scipy 1.18.0, statsmodels 0.15.0) |
| real_extra | timestral-5M | 000-001 | Colab T4 GPU (job log in results/colab/rd-timestral/) |
| real_extra | timestral-28M | 000-001 | Colab T4 GPU (job log in results/colab/rd-timestral/) |
| real_extra | timestral-123M | 000-001 | Colab T4 GPU (job log in results/colab/rd-timestral/) |
| real_extra | timestral-209M | 000-001 | Colab T4 GPU (job log in results/colab/rd-timestral/) |
| in_prior | arima | 000-012 | laptop (Windows; Python 3.12.10, numpy 2.4.6, scipy 1.18.0, statsmodels 0.15.0) |
| in_prior | ets | 011-012 | laptop (Windows; Python 3.12.10, numpy 2.4.6, scipy 1.18.0, statsmodels 0.15.0) |
| in_prior | oracle | 000-012 | laptop (Windows; Python 3.12.10, numpy 2.4.6, scipy 1.18.0, statsmodels 0.15.0) |
| real | arima | 000-010 | laptop (Windows; Python 3.12.10, numpy 2.4.6, scipy 1.18.0, statsmodels 0.15.0) |
| real | ets | 000-002 | laptop (Windows; Python 3.12.10, numpy 2.4.6, scipy 1.18.0, statsmodels 0.15.0) |
| in_prior | ets | 000-010 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |
| in_prior | seasonal_naive | 000-012 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |
| real | ets | 003-010 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |
| real | seasonal_naive | 000-010 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |
| in_prior | timestral-5M | 000-012 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |
| real | timestral-5M | 000-010 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |
| in_prior | timestral-28M | 000-012 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |
| real | timestral-28M | 000-010 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |
| in_prior | timestral-123M | 000-012 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |
| real | timestral-123M | 000-010 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |
| in_prior | timestral-209M | 000-012 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |
| real | timestral-209M | 000-010 | lab VM (Linux; Python 3.13.16, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, torch 2.14.1 CPU) |

## Real set (GIFT-Eval and fev-bench), common windows (412 windows, 31 groups)

| forecaster | n | mase | scrps | wql | cov80 | calib | mase_rel_snaive | scrps_rel_snaive |
|---|---|---|---|---|---|---|---|---|
| timestral-5M | 412 | 0.8897 | 0.6287 | 2.918e+11 | 0.8156 | 0.1802 | 0.6957 | 0.49 |
| timestral-28M | 412 | 0.8957 | 0.6308 | 8.32e+10 | 0.8114 | 0.1838 | 0.6746 | 0.4783 |
| timestral-123M | 412 | 0.9205 | 0.6438 | 2.539e+11 | 0.8053 | 0.1851 | 0.6621 | 0.4717 |
| timestral-209M | 412 | 0.9079 | 0.6317 | 1.571e+11 | 0.8134 | 0.1825 | 0.6513 | 0.4603 |
| seasonal_naive | 412 | 1.317 | 1.142 | 3.184e+14 | 0.8481 | 0.1774 | 1 | 1 |
| ets | 412 | 1.323 | 1.406 | 7.233e+14 | 0.8902 | 0.1738 | 1.021 | 1.076 |
| arima | 412 | 2.067 | 1.491 | 1.989e+14 | 0.8485 | 0.1508 | 1.269 | 0.9406 |

## Real set (GIFT-Eval and fev-bench), each forecaster on its own windows

| forecaster | n | mase | scrps | wql | cov80 | calib | mase_rel_snaive | scrps_rel_snaive |
|---|---|---|---|---|---|---|---|---|
| timestral-5M | 412 | 0.8897 | 0.6287 | 2.918e+11 | 0.8156 | 0.1802 | 0.6957 | 0.49 |
| timestral-28M | 412 | 0.8957 | 0.6308 | 8.32e+10 | 0.8114 | 0.1838 | 0.6746 | 0.4783 |
| timestral-123M | 412 | 0.9205 | 0.6438 | 2.539e+11 | 0.8053 | 0.1851 | 0.6621 | 0.4717 |
| timestral-209M | 412 | 0.9079 | 0.6317 | 1.571e+11 | 0.8134 | 0.1825 | 0.6513 | 0.4603 |
| seasonal_naive | 412 | 1.317 | 1.142 | 3.184e+14 | 0.8481 | 0.1774 | 1 | 1 |
| ets | 412 | 1.323 | 1.406 | 7.233e+14 | 0.8902 | 0.1738 | 1.021 | 1.076 |
| arima | 412 | 2.067 | 1.491 | 1.989e+14 | 0.8485 | 0.1508 | 1.269 | 0.9406 |

### Real set (GIFT-Eval and fev-bench): relative MASE to seasonal naive per group (own windows)

| group | timestral-5M | timestral-28M | timestral-123M | timestral-209M | seasonal_naive | ets | arima |
|---|---|---|---|---|---|---|---|
| LOOP_SEATTLE/H | 0.9232 | 0.8881 | 0.8776 | 0.8452 | 1 | 1.39 | 1.672 |
| LOOP_SEATTLE/H@medium | 0.7147 | 0.7175 | 0.7011 | 0.6951 | 1 | 0.8725 | 1.181 |
| M_DENSE/H | 0.7095 | 0.8016 | 0.8357 | 0.7819 | 1 | 1.072 | 1.388 |
| M_DENSE/H@medium | 0.6271 | 0.6101 | 0.6137 | 0.6113 | 1 | 0.8876 | 1.366 |
| SZ_TAXI/15T | 0.754 | 0.7561 | 0.7544 | 0.7537 | 1 | 1.033 | 0.7508 |
| SZ_TAXI/15T@medium | 0.7154 | 0.7164 | 0.7104 | 0.7117 | 1 | 1.131 | 0.9004 |
| bitbrains_fast_storage/H | 0.6097 | 0.591 | 0.6045 | 0.6144 | 1 | 1.174 | 0.9575 |
| electricity/D | 0.9675 | 1.02 | 0.9939 | 1.014 | 1 | 0.9692 | 1.165 |
| electricity/H | 0.9507 | 0.9561 | 0.9661 | 0.912 | 1 | 1.691 | 2.314 |
| electricity/H@medium | 0.8865 | 0.8782 | 0.854 | 0.8464 | 1 | 1.037 | 2.391 |
| entsoe/1H | 1.037 | 2.015 | 2.061 | 1.961 | 1 | 1.009 | 1.756 |
| entsoe/1H@medium | 0.7276 | 0.726 | 0.7079 | 0.6924 | 1 | 0.9408 | 0.7916 |
| ercot/1H | 0.623 | 0.6438 | 0.6073 | 0.5393 | 1 | 0.9023 | 1.133 |
| favorita_stores/1D | 0.8341 | 0.8382 | 0.8278 | 0.816 | 1 | 0.9232 | 1.153 |
| fred_md_2025 | 0.4169 | 0.4169 | 0.3768 | 0.3717 | 1 | 0.3756 | 0.4524 |
| hospital_admissions/1D | 0.6284 | 0.6274 | 0.6282 | 0.6277 | 1 | 0.6338 | 0.6333 |
| kdd_cup_2018_with_missing/H | 0.6613 | 0.6596 | 0.6344 | 0.6379 | 1 | 1.171 | 0.6689 |
| m4_daily | 0.8082 | 0.815 | 0.8372 | 0.8537 | 1 | 0.8282 | 0.8065 |
| m4_hourly | 0.6431 | 0.5609 | 0.5462 | 0.5368 | 1 | 1.634 | 12.86 |
| m4_monthly | 0.716 | 0.6765 | 0.6841 | 0.6945 | 1 | 0.6964 | 0.923 |
| m4_weekly | 0.1508 | 0.1397 | 0.1398 | 0.1248 | 1 | 0.2053 | 0.1965 |
| m5/1W | 0.5874 | 0.5771 | 0.5671 | 0.5672 | 1 | 0.6873 | 0.5693 |
| redset/1H | 0.8557 | 0.8154 | 0.767 | 0.7585 | 1 | 1.23 | 2.04 |
| restaurant | 0.7144 | 0.7153 | 0.7096 | 0.7021 | 1 | 0.7233 | 0.8704 |
| rohlik_sales/1D | 0.8275 | 0.8181 | 0.8145 | 0.8047 | 1 | 0.855 | 0.9514 |
| rossmann/1D | 0.3993 | 0.3524 | 0.3433 | 0.3277 | 1 | 0.6257 | 1.24 |
| solar/10T | 0.8805 | 0.2928 | 0.163 | 0.1666 | 1 | 16.11 | 32.35 |
| solar/10T@medium | 0.8453 | 0.7703 | 0.8447 | 0.8868 | 1 | 1.272 | 1.505 |
| solar/H | 1.001 | 0.8955 | 1.046 | 1.011 | 1 | 1.924 | 1.021 |
| solar/H@medium | 0.9492 | 0.9176 | 0.8956 | 0.889 | 1 | 1.41 | 2.529 |
| uk_covid_utla/1D | 0.5186 | 0.5537 | 0.6237 | 0.6247 | 1 | 0.9262 | 1.042 |

### Real set (GIFT-Eval and fev-bench): relative sCRPS to seasonal naive per group (own windows)

| group | timestral-5M | timestral-28M | timestral-123M | timestral-209M | seasonal_naive | ets | arima |
|---|---|---|---|---|---|---|---|
| LOOP_SEATTLE/H | 0.6036 | 0.5856 | 0.5786 | 0.5666 | 1 | 1.627 | 1.193 |
| LOOP_SEATTLE/H@medium | 0.3749 | 0.3717 | 0.3651 | 0.3626 | 1 | 1.99 | 0.6 |
| M_DENSE/H | 0.6822 | 0.7719 | 0.8117 | 0.7358 | 1 | 1.379 | 1.236 |
| M_DENSE/H@medium | 0.4395 | 0.4275 | 0.4298 | 0.4252 | 1 | 2.009 | 0.8565 |
| SZ_TAXI/15T | 0.7224 | 0.7244 | 0.7203 | 0.7215 | 1 | 1.003 | 0.7351 |
| SZ_TAXI/15T@medium | 0.456 | 0.4561 | 0.4477 | 0.4476 | 1 | 1.04 | 0.5383 |
| bitbrains_fast_storage/H | 0.2739 | 0.2697 | 0.2761 | 0.2767 | 1 | 1.143 | 0.6822 |
| electricity/D | 0.8387 | 0.886 | 0.8449 | 0.8667 | 1 | 0.9579 | 1.024 |
| electricity/H | 0.8817 | 0.8867 | 0.8974 | 0.8546 | 1 | 1.6 | 1.938 |
| electricity/H@medium | 0.6737 | 0.6739 | 0.6536 | 0.6473 | 1 | 1.93 | 1.685 |
| entsoe/1H | 0.9215 | 1.803 | 1.881 | 1.738 | 1 | 1.048 | 1.581 |
| entsoe/1H@medium | 0.8448 | 0.8448 | 0.8396 | 0.8073 | 1 | 1.184 | 0.8003 |
| ercot/1H | 0.6224 | 0.6575 | 0.6336 | 0.5553 | 1 | 0.9429 | 1.168 |
| favorita_stores/1D | 0.6456 | 0.6433 | 0.6344 | 0.6246 | 1 | 0.7792 | 0.8702 |
| fred_md_2025 | 0.4056 | 0.415 | 0.3929 | 0.3907 | 1 | 0.4029 | 0.452 |
| hospital_admissions/1D | 0.5882 | 0.5882 | 0.5883 | 0.5886 | 1 | 0.5953 | 0.5945 |
| kdd_cup_2018_with_missing/H | 0.5801 | 0.5742 | 0.5596 | 0.5623 | 1 | 1.201 | 0.6779 |
| m4_daily | 0.7834 | 0.7928 | 0.8069 | 0.8173 | 1 | 0.8641 | 0.8987 |
| m4_hourly | 0.5871 | 0.5169 | 0.5071 | 0.4904 | 1 | 1.594 | 12.13 |
| m4_monthly | 0.703 | 0.6743 | 0.6795 | 0.6903 | 1 | 0.7009 | 0.9055 |
| m4_weekly | 0.1437 | 0.1338 | 0.1363 | 0.1251 | 1 | 0.2299 | 0.2179 |
| m5/1W | 0.5486 | 0.5391 | 0.5294 | 0.5295 | 1 | 0.6781 | 0.5502 |
| redset/1H | 0.4572 | 0.4314 | 0.4149 | 0.4082 | 1 | 0.919 | 1.187 |
| restaurant | 0.6768 | 0.6775 | 0.6763 | 0.6693 | 1 | 0.6896 | 0.8255 |
| rohlik_sales/1D | 0.7136 | 0.7081 | 0.7091 | 0.6988 | 1 | 0.8084 | 0.8219 |
| rossmann/1D | 0.3015 | 0.2743 | 0.2787 | 0.2632 | 1 | 0.5103 | 0.9596 |
| solar/10T | 0.01679 | 0.007159 | 0.004092 | 0.003742 | 1 | 1.672 | 1.141 |
| solar/10T@medium | 0.4997 | 0.4388 | 0.4854 | 0.4976 | 1 | 2.614 | 1.225 |
| solar/H | 0.6869 | 0.576 | 0.6532 | 0.6258 | 1 | 1.963 | 0.7125 |
| solar/H@medium | 0.3697 | 0.3428 | 0.3329 | 0.3258 | 1 | 3.683 | 0.9486 |
| uk_covid_utla/1D | 0.52 | 0.5496 | 0.5943 | 0.5927 | 1 | 0.9332 | 1.083 |

## Extra real set (further open-license sources), common windows (80 windows, 11 groups)

| forecaster | n | mase | scrps | wql | cov80 | calib | mase_rel_snaive | scrps_rel_snaive |
|---|---|---|---|---|---|---|---|---|
| timestral-5M | 80 | 0.7767 | 0.566 | 1.577e+14 | 0.8147 | 0.2066 | 0.5428 | 0.2487 |
| timestral-28M | 80 | 0.7772 | 0.5586 | 7.173e+13 | 0.8217 | 0.2062 | 0.4573 | 0.2337 |
| timestral-123M | 80 | 0.7617 | 0.5482 | 2.34e+14 | 0.8155 | 0.2047 | 0.4425 | 0.2264 |
| timestral-209M | 80 | 0.7801 | 0.5565 | 7.113e+13 | 0.8179 | 0.207 | 0.4325 | 0.2197 |
| seasonal_naive | 80 | 0.931 | 1.419 | 7.161e+16 | 0.9289 | 0.1976 | 1 | 1 |
| ets | 80 | 0.8879 | 1.291 | 9.255e+16 | 0.9377 | 0.1907 | 0.9042 | 0.8012 |
| arima | 80 | 0.9986 | 1.634 | 2.361e+17 | 0.9262 | 0.1922 | 1.232 | 0.7663 |

## Extra real set (further open-license sources), each forecaster on its own windows

| forecaster | n | mase | scrps | wql | cov80 | calib | mase_rel_snaive | scrps_rel_snaive |
|---|---|---|---|---|---|---|---|---|
| timestral-5M | 80 | 0.7767 | 0.566 | 1.577e+14 | 0.8147 | 0.2066 | 0.5428 | 0.2487 |
| timestral-28M | 80 | 0.7772 | 0.5586 | 7.173e+13 | 0.8217 | 0.2062 | 0.4573 | 0.2337 |
| timestral-123M | 80 | 0.7617 | 0.5482 | 2.34e+14 | 0.8155 | 0.2047 | 0.4425 | 0.2264 |
| timestral-209M | 80 | 0.7801 | 0.5565 | 7.113e+13 | 0.8179 | 0.207 | 0.4325 | 0.2197 |
| seasonal_naive | 80 | 0.931 | 1.419 | 7.161e+16 | 0.9289 | 0.1976 | 1 | 1 |
| ets | 80 | 0.8879 | 1.291 | 9.255e+16 | 0.9377 | 0.1907 | 0.9042 | 0.8012 |
| arima | 80 | 0.9986 | 1.634 | 2.361e+17 | 0.9262 | 0.1922 | 1.232 | 0.7663 |

### Extra real set (further open-license sources): relative MASE to seasonal naive per group (own windows)

| group | timestral-5M | timestral-28M | timestral-123M | timestral-209M | seasonal_naive | ets | arima |
|---|---|---|---|---|---|---|---|
| fred_daily | 0.9653 | 0.991 | 0.9632 | 0.9575 | 1 | 0.9415 | 0.9487 |
| ghcn_daily_PRCP | 0.6604 | 0.6601 | 0.6568 | 0.6566 | 1 | 1.265 | 1.553 |
| ghcn_daily_TMAX | 0.5981 | 0.6066 | 0.6066 | 0.6105 | 1 | 0.9054 | 0.8824 |
| monash_bitcoin | 0.943 | 0.9274 | 0.9092 | 1.006 | 1 | 0.8849 | 0.9204 |
| monash_pedestrian_counts | 0.8552 | 0.8839 | 0.8687 | 0.8488 | 1 | 0.9903 | 2.287 |
| monash_sunspot_daily | 0.03489 | 0.00571 | 0.005884 | 0.004292 | 1 | 0.2555 | 0.3767 |
| monash_web_traffic_daily | 0.7781 | 0.7619 | 0.7734 | 0.7482 | 1 | 0.9227 | 0.875 |
| obf_control_activity_15min | 0.369 | 0.3445 | 0.3293 | 0.3446 | 1 | 1.589 | 0.8257 |
| owid_covid_new_cases | 1.257 | 1.153 | 1.055 | 0.9662 | 1 | 1.176 | 4.525 |
| seattle_fremont_bridge | 0.5662 | 0.6265 | 0.5842 | 0.551 | 1 | 1.024 | 1.461 |
| usgs_daily_discharge | 0.5499 | 0.519 | 0.4551 | 0.5133 | 1 | 0.7749 | 2.013 |

### Extra real set (further open-license sources): relative sCRPS to seasonal naive per group (own windows)

| group | timestral-5M | timestral-28M | timestral-123M | timestral-209M | seasonal_naive | ets | arima |
|---|---|---|---|---|---|---|---|
| fred_daily | 0.8658 | 0.8878 | 0.8644 | 0.851 | 1 | 0.9521 | 0.9021 |
| ghcn_daily_PRCP | 0.1924 | 0.1901 | 0.1926 | 0.1924 | 1 | 0.5437 | 0.5258 |
| ghcn_daily_TMAX | 0.5124 | 0.52 | 0.5197 | 0.523 | 1 | 0.9842 | 0.766 |
| monash_bitcoin | 0.935 | 0.9045 | 0.8929 | 0.9804 | 1 | 0.8661 | 0.9126 |
| monash_pedestrian_counts | 0.3149 | 0.3114 | 0.3043 | 0.2928 | 1 | 0.682 | 0.9037 |
| monash_sunspot_daily | 0.0488 | 0.03054 | 0.03203 | 0.02428 | 1 | 0.3911 | 0.3014 |
| monash_web_traffic_daily | 0.2779 | 0.2681 | 0.2707 | 0.2616 | 1 | 0.742 | 0.5462 |
| obf_control_activity_15min | 0.219 | 0.2003 | 0.1891 | 0.1849 | 1 | 0.8145 | 0.4866 |
| owid_covid_new_cases | 0.03329 | 0.0294 | 0.02633 | 0.02542 | 1 | 1.257 | 2.753 |
| seattle_fremont_bridge | 0.465 | 0.5438 | 0.4952 | 0.4748 | 1 | 1.027 | 1.188 |
| usgs_daily_discharge | 0.1952 | 0.1755 | 0.1598 | 0.1652 | 1 | 0.9509 | 0.682 |

## Analogue subset (real series standing in for the synthetic failure families), common windows (69 windows, 17 groups)

| forecaster | n | mase | scrps | wql | cov80 | calib | mase_rel_snaive | scrps_rel_snaive |
|---|---|---|---|---|---|---|---|---|
| timestral-5M | 69 | 3.051 | 2.403 | 0.8132 | 0.6645 | 0.2268 | 0.7652 | 0.5674 |
| timestral-28M | 69 | 3.056 | 2.377 | 0.7352 | 0.6682 | 0.2237 | 0.7335 | 0.5517 |
| timestral-123M | 69 | 3.08 | 2.406 | 0.7146 | 0.6591 | 0.2203 | 0.7385 | 0.5491 |
| timestral-209M | 69 | 3.13 | 2.434 | 0.6848 | 0.6699 | 0.22 | 0.7345 | 0.5419 |
| seasonal_naive | 69 | 3.568 | 3.169 | 5.101 | 0.7745 | 0.2005 | 1 | 1 |
| ets | 69 | 4.077 | 3.716 | 3.116 | 0.7842 | 0.2063 | 1.149 | 1.049 |
| arima | 69 | 4.716 | 3.651 | 2.224 | 0.7551 | 0.2092 | 1.112 | 0.8458 |

## Analogue subset (real series standing in for the synthetic failure families), each forecaster on its own windows

| forecaster | n | mase | scrps | wql | cov80 | calib | mase_rel_snaive | scrps_rel_snaive |
|---|---|---|---|---|---|---|---|---|
| timestral-5M | 69 | 3.051 | 2.403 | 0.8132 | 0.6645 | 0.2268 | 0.7652 | 0.5674 |
| timestral-28M | 69 | 3.056 | 2.377 | 0.7352 | 0.6682 | 0.2237 | 0.7335 | 0.5517 |
| timestral-123M | 69 | 3.08 | 2.406 | 0.7146 | 0.6591 | 0.2203 | 0.7385 | 0.5491 |
| timestral-209M | 69 | 3.13 | 2.434 | 0.6848 | 0.6699 | 0.22 | 0.7345 | 0.5419 |
| seasonal_naive | 69 | 3.568 | 3.169 | 5.101 | 0.7745 | 0.2005 | 1 | 1 |
| ets | 69 | 4.077 | 3.716 | 3.116 | 0.7842 | 0.2063 | 1.149 | 1.049 |
| arima | 69 | 4.716 | 3.651 | 2.224 | 0.7551 | 0.2092 | 1.112 | 0.8458 |

### Analogue subset (real series standing in for the synthetic failure families): relative MASE to seasonal naive per group (own windows)

| group | timestral-5M | timestral-28M | timestral-123M | timestral-209M | seasonal_naive | ets | arima |
|---|---|---|---|---|---|---|---|
| coupled_oscillators/fingrid_frequency_1s | 0.792 | 0.8 | 0.8077 | 0.8184 | 1 | 1.125 | 1.303 |
| coupled_oscillators/monash_pedestrian_counts | 0.852 | 0.8823 | 0.8444 | 0.8646 | 1 | 1.363 | 1.566 |
| coupled_oscillators/obf_control_activity_15min | 0.369 | 0.3445 | 0.3293 | 0.3446 | 1 | 1.589 | 0.8257 |
| delay_feedback/lh | 0.6764 | 0.6736 | 0.6912 | 0.6867 | 1 | 0.6623 | 0.621 |
| delay_feedback/lynx | 0.6253 | 0.4076 | 0.5606 | 0.5514 | 1 | 2.824 | 0.5808 |
| delay_feedback/monash_sunspot_daily | 0.6611 | 0.661 | 0.6594 | 0.666 | 1 | 0.7493 | 0.8548 |
| delay_feedback/nino34_monthly | 0.9475 | 0.9212 | 0.9175 | 0.8903 | 1 | 0.9327 | 0.9306 |
| delay_feedback/silso_monthly_sunspots | 0.8041 | 0.7711 | 0.7448 | 0.6971 | 1 | 1.211 | 0.9052 |
| heavy_tails/fred_daily | 0.8497 | 0.8509 | 0.8523 | 0.8519 | 1 | 0.8544 | 0.8576 |
| heavy_tails/monash_bitcoin | 0.9794 | 0.9674 | 0.9521 | 0.9596 | 1 | 1.013 | 1.188 |
| heavy_tails/usgs_daily_discharge | 0.4125 | 0.3659 | 0.3796 | 0.3485 | 1 | 1.01 | 2.955 |
| heavy_tails/usgs_quakes_daily | 0.8592 | 0.8583 | 0.8565 | 0.8551 | 1 | 0.9155 | 1.006 |
| regime_switches/fred_energy_breaks | 0.9038 | 0.9028 | 0.8975 | 0.9029 | 1 | 0.9331 | 0.9489 |
| regime_switches/fred_monthly_breaks | 0.8044 | 0.7958 | 0.8012 | 0.8017 | 1 | 0.7841 | 0.7914 |
| regime_switches/monash_pedestrian_counts | 1.076 | 0.9973 | 0.9033 | 0.8958 | 1 | 4.103 | 2.098 |
| regime_switches/owid_covid_new_cases | 1.006 | 1.025 | 1.056 | 1.105 | 1 | 1.304 | 2.063 |
| regime_switches/seattle_fremont_bridge | 0.8585 | 0.8655 | 0.8135 | 0.7905 | 1 | 0.876 | 1.408 |

### Analogue subset (real series standing in for the synthetic failure families): relative sCRPS to seasonal naive per group (own windows)

| group | timestral-5M | timestral-28M | timestral-123M | timestral-209M | seasonal_naive | ets | arima |
|---|---|---|---|---|---|---|---|
| coupled_oscillators/fingrid_frequency_1s | 0.7689 | 0.7667 | 0.7731 | 0.7835 | 1 | 1.456 | 1.195 |
| coupled_oscillators/monash_pedestrian_counts | 0.7611 | 0.7936 | 0.7507 | 0.7672 | 1 | 1.379 | 1.207 |
| coupled_oscillators/obf_control_activity_15min | 0.219 | 0.2003 | 0.1891 | 0.1849 | 1 | 0.8145 | 0.4866 |
| delay_feedback/lh | 0.739 | 0.7323 | 0.7357 | 0.7365 | 1 | 0.7368 | 0.6976 |
| delay_feedback/lynx | 0.496 | 0.4059 | 0.4904 | 0.4457 | 1 | 2.451 | 0.5806 |
| delay_feedback/monash_sunspot_daily | 0.4822 | 0.4781 | 0.4786 | 0.4815 | 1 | 0.6822 | 0.6326 |
| delay_feedback/nino34_monthly | 1.041 | 1.002 | 0.9876 | 0.9518 | 1 | 0.9633 | 1.029 |
| delay_feedback/silso_monthly_sunspots | 0.5243 | 0.5086 | 0.4961 | 0.4729 | 1 | 0.9641 | 0.8078 |
| heavy_tails/fred_daily | 0.5863 | 0.5867 | 0.5878 | 0.5888 | 1 | 0.6299 | 0.6318 |
| heavy_tails/monash_bitcoin | 0.92 | 0.913 | 0.9139 | 0.9152 | 1 | 0.9592 | 1.133 |
| heavy_tails/usgs_daily_discharge | 0.09963 | 0.09396 | 0.09341 | 0.08756 | 1 | 1.003 | 0.8051 |
| heavy_tails/usgs_quakes_daily | 0.5886 | 0.5897 | 0.5885 | 0.5882 | 1 | 0.7184 | 0.6976 |
| regime_switches/fred_energy_breaks | 0.9616 | 0.9636 | 0.9624 | 0.9667 | 1 | 0.9536 | 0.968 |
| regime_switches/fred_monthly_breaks | 0.8207 | 0.8092 | 0.8155 | 0.8179 | 1 | 0.804 | 0.8175 |
| regime_switches/monash_pedestrian_counts | 0.5251 | 0.4949 | 0.4437 | 0.4419 | 1 | 2.575 | 0.9333 |
| regime_switches/owid_covid_new_cases | 0.7929 | 0.7788 | 0.818 | 0.8431 | 1 | 1.15 | 1.67 |
| regime_switches/seattle_fremont_bridge | 0.5092 | 0.5159 | 0.4807 | 0.4632 | 1 | 1.225 | 0.7821 |

## In-prior set, common windows (500 windows, 6 groups)

| forecaster | n | mase | scrps | wql | cov80 | calib | mase_rel_snaive | scrps_rel_snaive | mase_rel_oracle | scrps_rel_oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| timestral-5M | 500 | 0.6795 | 0.4613 | 0.08796 | 0.8104 | 0.06562 | 0.6107 | 0.5552 | 1.066 | 1.068 |
| timestral-28M | 500 | 0.6639 | 0.4502 | 0.08663 | 0.8034 | 0.06259 | 0.597 | 0.542 | 1.042 | 1.043 |
| timestral-123M | 500 | 0.6569 | 0.4451 | 0.08635 | 0.803 | 0.06039 | 0.5907 | 0.5359 | 1.031 | 1.031 |
| timestral-209M | 500 | 0.6557 | 0.4442 | 0.08595 | 0.808 | 0.05948 | 0.5896 | 0.5348 | 1.029 | 1.029 |
| seasonal_naive | 500 | 1.11 | 0.8364 | 0.1394 | 0.8255 | 0.1438 | 1 | 1 | 1.746 | 1.924 |
| ets | 500 | 1.084 | 0.8878 | 0.1648 | 0.8952 | 0.1493 | 0.9543 | 1.06 | 1.666 | 2.04 |
| arima | 500 | 5.628 | 4.861 | 0.5452 | 0.7586 | 0.1461 | 3.31 | 3.313 | 5.78 | 6.374 |
| oracle | 500 | 0.6376 | 0.4322 | 0.08305 | 0.7952 | 0.04433 | 0.5728 | 0.5198 | 1 | 1 |

## In-prior set, each forecaster on its own windows

| forecaster | n | mase | scrps | wql | cov80 | calib | mase_rel_snaive | scrps_rel_snaive | mase_rel_oracle | scrps_rel_oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| timestral-5M | 500 | 0.6795 | 0.4613 | 0.08796 | 0.8104 | 0.06562 | 0.6107 | 0.5552 | 1.066 | 1.068 |
| timestral-28M | 500 | 0.6639 | 0.4502 | 0.08663 | 0.8034 | 0.06259 | 0.597 | 0.542 | 1.042 | 1.043 |
| timestral-123M | 500 | 0.6569 | 0.4451 | 0.08635 | 0.803 | 0.06039 | 0.5907 | 0.5359 | 1.031 | 1.031 |
| timestral-209M | 500 | 0.6557 | 0.4442 | 0.08595 | 0.808 | 0.05948 | 0.5896 | 0.5348 | 1.029 | 1.029 |
| seasonal_naive | 500 | 1.11 | 0.8364 | 0.1394 | 0.8255 | 0.1438 | 1 | 1 | 1.746 | 1.924 |
| ets | 500 | 1.084 | 0.8878 | 0.1648 | 0.8952 | 0.1493 | 0.9543 | 1.06 | 1.666 | 2.04 |
| arima | 500 | 5.628 | 4.861 | 0.5452 | 0.7586 | 0.1461 | 3.31 | 3.313 | 5.78 | 6.374 |
| oracle | 500 | 0.6376 | 0.4322 | 0.08305 | 0.7952 | 0.04433 | 0.5728 | 0.5198 | 1 | 1 |

<!-- analogue-report:begin -->
## Analogues versus synthetic families

Real analogue windows (the `analogues` set; no oracle exists for them) against the synthetic failure family each stands in for. For every window, the ratio is a Timestral size's MASE (or sCRPS) divided by the lowest of seasonal naive, ETS and ARIMA on that window; the table gives the geometric mean of the ratio over the family's windows (below 1: Timestral is lower than the best classical model). The synthetic columns are the same quantity (`*_ratio_classical_gmean`, geometric mean over the generators listed and their knob values) read from `results/map/<chaos|tails|regime>/summary_all_sizes.csv` at context 2048 and horizon 16 or 128. The pairing of families to generators is: delay_feedback to mackey_glass, coupled_oscillators to kuramoto, heavy_tails to fbm, hawkes, msm_cascade and stable_ar, regime_switches to frequency_jump, markov_ar, mean_jump and variance_jump. The real windows have their own horizons and periods (see `bench/regression/manifest.csv`), so the columns are placed side by side, not matched window to window. Per-dataset rows are in `analogue_vs_synthetic.csv`.

### MASE ratio to the best classical model

| family_analogue | forecaster | real windows | real MASE ratio | synthetic MASE ratio, H16 | synthetic MASE ratio, H128 |
|---|---|---|---|---|---|
| delay_feedback | timestral-5M | 12 | 0.922 | 0.458 | 0.385 |
| delay_feedback | timestral-28M | 12 | 0.756 | 0.324 | 0.326 |
| delay_feedback | timestral-123M | 12 | 0.771 | 0.257 | 0.266 |
| delay_feedback | timestral-209M | 12 | 0.732 | 0.234 | 0.260 |
| coupled_oscillators | timestral-5M | 19 | 0.788 | 0.542 | 0.437 |
| coupled_oscillators | timestral-28M | 19 | 0.779 | 0.328 | 0.286 |
| coupled_oscillators | timestral-123M | 19 | 0.762 | 0.288 | 0.225 |
| coupled_oscillators | timestral-209M | 19 | 0.777 | 0.298 | 0.197 |
| heavy_tails | timestral-5M | 15 | 0.899 | 1.064 | 1.023 |
| heavy_tails | timestral-28M | 15 | 0.876 | 1.058 | 1.020 |
| heavy_tails | timestral-123M | 15 | 0.881 | 1.054 | 1.012 |
| heavy_tails | timestral-209M | 15 | 0.867 | 1.054 | 1.012 |
| regime_switches | timestral-5M | 23 | 1.029 | 1.041 | 1.018 |
| regime_switches | timestral-28M | 23 | 0.991 | 1.019 | 0.977 |
| regime_switches | timestral-123M | 23 | 0.965 | 1.010 | 0.965 |
| regime_switches | timestral-209M | 23 | 0.977 | 1.008 | 0.965 |

### sCRPS ratio to the best classical model

| family_analogue | forecaster | real windows | real sCRPS ratio | synthetic sCRPS ratio, H16 | synthetic sCRPS ratio, H128 |
|---|---|---|---|---|---|
| delay_feedback | timestral-5M | 12 | 0.693 | 0.410 | 0.390 |
| delay_feedback | timestral-28M | 12 | 0.646 | 0.317 | 0.329 |
| delay_feedback | timestral-123M | 12 | 0.647 | 0.235 | 0.268 |
| delay_feedback | timestral-209M | 12 | 0.619 | 0.221 | 0.266 |
| coupled_oscillators | timestral-5M | 19 | 0.615 | 0.530 | 0.417 |
| coupled_oscillators | timestral-28M | 19 | 0.610 | 0.319 | 0.273 |
| coupled_oscillators | timestral-123M | 19 | 0.588 | 0.284 | 0.215 |
| coupled_oscillators | timestral-209M | 19 | 0.581 | 0.285 | 0.189 |
| heavy_tails | timestral-5M | 15 | 0.595 | 0.951 | 0.952 |
| heavy_tails | timestral-28M | 15 | 0.585 | 0.944 | 0.950 |
| heavy_tails | timestral-123M | 15 | 0.585 | 0.942 | 0.945 |
| heavy_tails | timestral-209M | 15 | 0.579 | 0.941 | 0.945 |
| regime_switches | timestral-5M | 23 | 0.836 | 0.948 | 0.829 |
| regime_switches | timestral-28M | 23 | 0.806 | 0.932 | 0.798 |
| regime_switches | timestral-123M | 23 | 0.794 | 0.921 | 0.788 |
| regime_switches | timestral-209M | 23 | 0.786 | 0.919 | 0.788 |

<!-- analogue-report:end -->
