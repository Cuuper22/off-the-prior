# Off-the-Prior benchmarks

Two benchmark sets built from this project's simulators. Every series comes with the **true-dynamics
oracle's** 21 quantiles (the best possible forecast given the simulator's true state), so a model can be
scored not only against seasonal naive (as on GIFT-Eval) but against the best achievable forecast.

## OffPrior-Diag (diagnostic)

372 series (372 public, 0 held-out). Each config was chosen to cover one failure type found in the failure map (delay oscillators,
smooth chaos, coupled oscillators, chaos within the first Lyapunov times, recent regime switches, heavy-tailed
random walks, long memory, bursts) plus controls inside the prior and controls that nobody can forecast. Every
series carries `failure_type`, its generator tags and a one-line `reason` (see `data/diag_manifest.csv`).
Context 2,048 steps, horizon 128.

## OffPrior-LB (leaderboard)

3776 series (1888 public, 1888 held-out): every config of the chaos, regime, tails and in-prior grids plus 40 fixed random
mixture-of-primitives genomes, 16 seeds per config spread over contexts 512 / 2,048 / 8,192 and horizons
16 / 128 / 1,024. Seed indices alternate between a public and a held-out split. The seeds and the code are
public, so the held-out split is a convenience (futures left out of the default Hugging Face config), not
a secret test set.

## Run it

```bash
# the diagnostic files are in git; the leaderboard files are on the Hugging Face dataset
huggingface-cli download cuuper22/off-the-prior-bench --repo-type dataset --include "data/lb_*" --local-dir bench
python bench/eval.py --set diag --model timestral-5M
python bench/eval.py --set lb --split public --predictions my_preds/ --name mymodel
```

The files can also be rebuilt from the configs (`python bench/build.py --set diag`, `python bench/build.py --set lb
--workers 8`), but use the published files for scores you report. The published files were built on Windows 11, Python 3.12, numpy 2.4.6, scipy 1.18.0.
A rebuild on Linux, Python 3.13, numpy 2.5.3, scipy 1.18.1 matched it array-for-array for 25 of 31 diagnostic configs; in the other
6 (kuramoto/K1, kuramoto/K1.5, mackey_glass/tau17, mackey_glass/tau20, mackey_glass/tau30, tails/fbm/H0.8) floating-point differences between platforms change the simulated paths; in some of them the
paths then diverge (largest difference 3.9 context standard deviations). Details:
`data/cross_platform_diag.json`. `*_SHA256SUMS` hold hashes of the array contents of the published files.

`--predictions DIR`: for every data file `bench/data/<set>_<split>_T<T>_H<H>.npz` put a file with the same
name in DIR holding `quantiles`, an array (N, H, 21) in the data file's series order, at levels
0.01, 0.05, 0.10, ..., 0.95, 0.99.

## Scores

Per series: MASE, scaled CRPS (sCRPS), WQL, 80% coverage, calibration error (definitions in `otp/metrics.py`).
Summary, GIFT-Eval style: geometric mean over series of MASE and sCRPS divided by seasonal naive's; plus the
geometric mean of MASE divided by the oracle's (1.0 = as good as the true dynamics).

### Baselines, OffPrior-Diag (all splits)

| model | series | MASE / seasonal naive | sCRPS / seasonal naive | MASE / oracle | 80% coverage |
|---|---|---|---|---|---|
| oracle | 372 | 0.272 | 0.209 | 1.000 | 0.789 |
| timestral-209M | 372 | 0.543 | 0.427 | 2.002 | 0.781 |
| timestral-123M | 372 | 0.550 | 0.435 | 2.028 | 0.771 |
| timestral-28M | 372 | 0.607 | 0.478 | 2.239 | 0.751 |
| timestral-5M | 372 | 0.656 | 0.515 | 2.417 | 0.705 |
| ets | 372 | 1.213 | 1.190 | 4.468 | 0.875 |
| arima | 372 | 1.123 | 0.872 | 4.150 | 0.738 |
| seasonal_naive | 372 | 1.000 | 1.000 | 3.679 | 0.814 |

By failure type (Timestral-209M):

| failure type | series | MASE / seasonal naive | MASE / oracle |
|---|---|---|---|
| bursty | 12 | 0.938 | 0.991 |
| chaos_short_reach | 24 | 0.699 | 4.455 |
| control_in_prior | 60 | 0.699 | 1.007 |
| control_old_switch | 12 | 0.745 | 1.016 |
| control_unknowable | 12 | 0.804 | 0.978 |
| control_unpredictable | 12 | 0.764 | 1.033 |
| coupled_oscillators | 36 | 0.177 | 7.942 |
| delay_oscillator | 36 | 0.223 | 6.515 |
| heavy_tail_walk | 12 | 1.011 | 1.006 |
| long_memory | 12 | 0.643 | 1.123 |
| multifractal | 12 | 1.037 | 1.033 |
| periodic_map | 12 | 0.046 | 1.187 |
| recent_switch | 84 | 1.074 | 1.808 |
| smooth_chaos | 24 | 0.273 | 2.875 |
| switching | 12 | 0.753 | 1.090 |

### Baselines, OffPrior-LB (public + held-out)

| model | series | MASE / seasonal naive | sCRPS / seasonal naive | MASE / oracle | 80% coverage |
|---|---|---|---|---|---|
| oracle | 3776 | 0.463 | 0.286 | 1.000 | 0.798 |
| timestral-209M | 3776 | 0.608 | 0.385 | 1.314 | 0.793 |
| timestral-123M | 3776 | 0.615 | 0.389 | 1.329 | 0.791 |
| timestral-28M | 3776 | 0.637 | 0.402 | 1.375 | 0.783 |
| timestral-5M | 3776 | 0.669 | 0.425 | 1.446 | 0.779 |
| ets | 3776 | 1.006 | 0.926 | 2.172 | 0.889 |
| arima | 3776 | 1.039 | 0.663 | 2.245 | 0.770 |
| seasonal_naive | 3776 | 1.000 | 1.000 | 2.159 | 0.882 |

By family (Timestral-209M):

| family | series | MASE / seasonal naive | MASE / oracle |
|---|---|---|---|
| chaos | 688 | 0.244 | 2.162 |
| in_prior | 128 | 0.511 | 1.051 |
| mixture | 640 | 0.619 | 1.471 |
| regime | 1760 | 0.782 | 1.147 |
| tails | 560 | 0.866 | 1.011 |

Numbers above are read from `bench/leaderboard/*/summary.json` by `bench/make_docs.py`.
