# Covariates on coupled systems: does Timestral's one-way covariate attention cost accuracy?

Timestral reads a covariate through "conditioning rows": the main-series rows read the covariate rows, the covariate
rows never read the main series (one-way attention). Several main rows given as one multivariate context read each
other in both directions. This folder holds six coupled systems with known equations, Timestral scored on the main
series `y` under four input conditions, and the true-dynamics oracle and classical models as references.

Everything below is produced by the committed scripts. All numbers in this file are copied from
`summary.csv` (the tables are written by `scripts/covariate_sweep.py merge` into `tables.md` and pasted at the end of
this file). Branch: `task/t4_covariates`.

## Terms

- **MASE**: mean absolute error of the median forecast, divided by the in-sample seasonal-naive error (`otp.metrics`).
- **MASE ratio to the oracle**: MASE(forecaster) / MASE(oracle) per sample, geometric mean over samples, with a 95%
  bootstrap interval. The oracle is a Monte Carlo continuation of the true process from its true latent state with known
  parameters (500 paths). 1.00 means the forecaster matches the oracle.
- **Matched oracle**: the oracle whose information equals the condition's. For `known_cov` it conditions on the realized
  future of x (for LeakyKnown the exact Gaussian posterior given the noisy copy of y, computed per horizon). For `uni`,
  `past_cov` and `joint` it averages over the future of x (columns `*_oracle_marg`, `*_ratio_oracle_matched` in
  `scores.csv`).
- **Covariate gain**: MASE(uni) / MASE(condition) per sample, geometric mean. Above 1: the covariate helped.
- **Oracle gain**: MASE(uni) / MASE(matched oracle): the gain a forecaster using the covariate perfectly could reach.
- **joint/past_cov**: MASE(joint) / MASE(past_cov) per sample. Same information in both; `past_cov` routes it one way,
  `joint` routes it both ways (x is a second main row, only row 0 is scored). Below 1: two-way is more accurate.

## Systems (`otp/generators/coupled.py`, family `coupled`)

| system | knob (5 values) | structure |
|---|---|---|
| LaggedDriver | `lag` L in 0, 8, 32, 128, 512 | x exogenous (AR + season); y_t = a x_{t-L} + small AR noise; x offered as past or known-future covariate |
| NonlinearDriver | `nonlin` in 0, 0.25, 0.5, 0.75, 1 | y = a sd_x ((1-s) u + s sin(omega u)) + noise, u = x_{t-32} / sd_x; x past or known-future |
| FeedbackLinear | `coupling` k in 0, 0.05, 0.15, 0.4, 1 | y and x in a coupled AR(1) loop; k = 0 is independent; x past-only |
| FeedbackPredPrey | `coupling` k in 0, 0.25, 0.5, 1, 2 | noisy predator-prey in log space (y prey, x predator), linearized around the equilibrium; x past-only |
| HiddenDriver | `x_noise` in 0.05, 0.25, 0.5, 1, 2 | latent z (AR + season) drives y with lag 32; x = z + noise; z is never shown; x past-only |
| LeakyKnown | `noise_ratio` in 0.05, 0.25, 1, 4, 16 | x_t = y_{t-1} + noise, given as a known-future covariate; oracle is the exact posterior given x's future |

For the exogenous drivers the exogenous noise is shared across oracle paths (`shared_normal`), so the matched oracle
conditions on the same x future as the realized one. Tests: `tests/test_coupled.py`.

## What ran

- Samples: 6 systems x 5 knob values x 8 seeds = 240 samples, 500 oracle paths each, longest horizon 1024
  (`provenance.json`, run `samples`). `samples/` (107 MB) is git-ignored; regenerate with the first command below
  (seeded, deterministic).
- Timestral, sizes 5M and 28M, full grid: 5 knob values x 8 seeds, T = 2048, H in 16, 128, 1024 (horizons are prefixes
  of one 1024-step forecast).
- Timestral, sizes 123M and 209M, reduced grid: knob values 1, 3 and 5 of the 5 (first, middle, last), seeds 0 to 3,
  T = 2048, H in 16, 128.
- Context-length check: size 5M, LaggedDriver, FeedbackLinear and FeedbackPredPrey, full grid, T = 512 and 8192
  (T = 2048 comes from the full run), H in 16, 128, 1024. The same samples are used at every T; the oracle is scored on
  the same context window as the model.
- References: oracle, marginal oracle and seasonal naive, ETS and ARIMA (fitted on the last 1024 context steps) on y
  alone, at every (sample, T) cell that has Timestral rows.
- Nothing was cut. The 209M runs were timed with the others: mean `sec_per_sample` at H = 16 per condition (from
  `scores.csv`, T = 2048): 209M uni 1.99, past_cov 7.72, known_cov 14.25, joint 3.46; 123M uni 0.92, past_cov 3.51,
  known_cov 7.13, joint 1.59; 28M uni 0.39, past_cov 1.88, known_cov 4.23, joint 0.80; 5M uni 0.13, past_cov 0.53,
  known_cov 1.19, joint 0.24.
- Rows: `scores.csv` has 14208 rows (one per sample x forecaster x condition x T x H); `summary.csv` has 1362 rows.
- Wall time from `provenance.json`: samples 267.7 s; models `full` 1615.8 s; models `ctx` 457.2 s; refs
  (final invocation, HiddenDriver and LeakyKnown) 597.6 s. The `models:reduced` entry in `provenance.json` is from a
  final invocation that found every part file present and ran nothing (0 rows, 0.0 s); the reduced-grid part files
  were produced by earlier invocations of the same command, whose provenance entries were overwritten. The references for the other four
  systems were produced by earlier invocations of the `refs` command, whose timings were also overwritten.
- Host: 2-core CPU VM, one thread per process (`OTP_THREADS=1`), `nice -n 10`, shared with other jobs; the run was
  interrupted once by a machine restart and resumed (part files are skipped when present).

## Regeneration

```
git checkout task/t4_covariates && export PYTHONPATH=$PWD OTP_THREADS=1
nice -n 10 python3 scripts/covariate_sweep.py samples
nice -n 10 python3 scripts/covariate_sweep.py models --label full --sizes 5M 28M --contexts 2048 --grid full --seeds 8
nice -n 10 python3 scripts/covariate_sweep.py models --label ctx --sizes 5M --contexts 512 8192 \
    --systems LaggedDriver FeedbackLinear FeedbackPredPrey --grid full --seeds 8 --batch-size 4
nice -n 10 python3 scripts/covariate_sweep.py models --label reduced --sizes 123M 209M --contexts 2048 \
    --grid reduced --seeds 4 --horizons 16 128 --batch-size 4
nice -n 10 python3 scripts/covariate_sweep.py refs --contexts 512 2048 8192
nice -n 10 python3 scripts/covariate_sweep.py merge
python3 -m pytest -q tests/test_coupled.py
```

`parts/` (3.2 MB, committed) holds the per-system score files; `merge` alone rebuilds `scores.csv`, `summary.csv` and
`tables.md` from them without rerunning any model.

## Files

- `scores.csv`: long table, one row per sample x forecaster x condition x T x H x variate; columns include the metrics
  (`mase`, `scrps`, `wql`, `cov80`, `cov98`, `calib`), `mase_ratio_oracle` (oracle conditioned on x's future),
  `mase_ratio_oracle_marg`, `mase_ratio_oracle_matched`, `mase_ratio_classical`, and the scrps counterparts.
- `summary.csv`: per system x knob x condition x size x T x H: geometric-mean ratios to the oracle, covariate gain,
  oracle gain, joint/past_cov, each with a 95% bootstrap interval (`*_lo95`, `*_hi95`), and `n`.
  Rows with condition `classical_best` give the best of seasonal naive, ETS and ARIMA per sample over the marginal oracle.
- `tables.md`: the tables pasted below. `provenance.json`: run records. `parts/`: per-system score files.

## Selected values (copied from `summary.csv`)

Largest and smallest joint/past_cov per system over all knob values, sizes and horizons (geometric mean [95% CI]):

| system | smallest | largest |
|---|---|---|
| FeedbackLinear | 0.982 [0.973, 0.991] (coupling=1, 209M, H=16) | 1.011 [1.002, 1.021] (coupling=0.4, 5M, H=16) |
| FeedbackPredPrey | 0.812 [0.687, 0.959] (coupling=2, 209M, H=16) | 1.035 [0.904, 1.176] (coupling=0.5, 209M, H=16) |
| LaggedDriver | 0.832 [0.717, 0.921] (lag=32, 209M, H=16) | 1.013 [0.999, 1.027] (lag=0, 209M, H=16) |
| NonlinearDriver | 0.771 [0.481, 1.022] (nonlin=0, 123M, H=16) | 1.005 [0.997, 1.014] (nonlin=0.75, 28M, H=128) |
| HiddenDriver | 0.870 [0.771, 0.955] (x_noise=0.05, 28M, H=16) | 1.027 [0.998, 1.063] (x_noise=0.5, 123M, H=16) |
| LeakyKnown | 0.989 [0.972, 1.006] (noise_ratio=1, 28M, H=16) | 1.018 [0.997, 1.040] (noise_ratio=0.05, 123M, H=16) |

LaggedDriver with `known_cov`, T = 2048, covariate gain / oracle gain (the oracle gain is MASE(uni) / MASE(matched oracle)):

| size | lag | H=16 | H=128 | H=1024 |
|---|---|---|---|---|
| 5M | 0 | 2.55 / 5.50 | 1.95 / 7.22 | 1.04 / 8.11 |
| 5M | 8 | 1.59 / 5.97 | 1.73 / 7.59 | 1.07 / 8.42 |
| 5M | 32 | 1.15 / 8.31 | 1.14 / 7.19 | 1.01 / 7.79 |
| 5M | 128 | 0.96 / 7.19 | 1.01 / 7.88 | 0.99 / 8.36 |
| 5M | 512 | 1.00 / 6.32 | 1.03 / 8.27 | 1.00 / 8.38 |
| 28M | 0 | 4.39 / 5.62 | 5.31 / 7.24 | 1.24 / 7.94 |
| 28M | 8 | 2.59 / 5.98 | 3.00 / 7.59 | 1.17 / 8.30 |
| 28M | 32 | 1.55 / 7.91 | 1.40 / 7.15 | 1.08 / 7.53 |
| 28M | 128 | 1.01 / 7.75 | 1.01 / 7.89 | 1.00 / 8.19 |
| 28M | 512 | 0.95 / 6.13 | 1.01 / 8.16 | 1.02 / 8.31 |

LeakyKnown `known_cov`, MASE ratio to the matched oracle at noise_ratio=0.05: 5M 8.47 / 20.92 / 42.71 and 28M 3.81 /
9.94 / 38.41 at H = 16 / 128 / 1024 (Table 3 below has the intervals and all other noise levels).

The three tables the task asks for are Table 1 (covariate gain), Table 2 (joint vs past_cov on the Feedback systems) and
Table 3 (LeakyKnown); Table 1b (ratios to the oracle for all conditions) and Table 4 (context-length check) follow.

## Table 1. Covariate gain per system and knob (T = 2048)

Covariate gain = MASE(uni) / MASE(condition), geometric mean over samples, with 95% bootstrap CI in brackets. Above 1 means the covariate helped. `oracle` is the same ratio for the oracle that matches the condition's information (MASE(uni) / MASE(oracle)): the largest gain available.

### Size 5M, H = 16

| system | knob | past_cov | known_cov | joint | oracle (past_cov, or known_cov where offered) |
|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 0.98 [0.96, 1.00] | 2.55 [1.89, 3.20] | 0.98 [0.95, 1.00] | 5.50 [3.43, 8.41] |
| LaggedDriver | lag=8 | 1.18 [0.94, 1.54] | 1.59 [1.26, 2.10] | 1.20 [0.93, 1.62] | 5.97 [4.22, 9.39] |
| LaggedDriver | lag=32 | 1.07 [1.00, 1.16] | 1.15 [1.03, 1.31] | 1.08 [1.00, 1.17] | 8.31 [5.77, 11.69] |
| LaggedDriver | lag=128 | 0.98 [0.95, 1.01] | 0.96 [0.88, 1.05] | 0.98 [0.94, 1.02] | 7.19 [5.85, 8.97] |
| LaggedDriver | lag=512 | 1.01 [0.96, 1.07] | 1.00 [0.94, 1.06] | 1.01 [0.95, 1.07] | 6.32 [5.06, 7.94] |
| NonlinearDriver | nonlin=0 | 1.01 [0.97, 1.06] | 1.03 [0.96, 1.11] | 1.01 [0.97, 1.06] | 7.03 [5.74, 8.53] |
| NonlinearDriver | nonlin=0.25 | 1.12 [1.02, 1.25] | 1.15 [0.99, 1.37] | 1.13 [1.02, 1.25] | 5.47 [4.43, 6.54] |
| NonlinearDriver | nonlin=0.5 | 1.09 [1.01, 1.18] | 1.03 [0.88, 1.21] | 1.09 [1.01, 1.17] | 6.70 [4.56, 9.70] |
| NonlinearDriver | nonlin=0.75 | 1.11 [1.03, 1.21] | 1.07 [1.01, 1.13] | 1.12 [1.04, 1.22] | 5.88 [5.02, 6.97] |
| NonlinearDriver | nonlin=1 | 1.10 [0.98, 1.21] | 1.07 [0.97, 1.16] | 1.10 [0.98, 1.22] | 6.22 [5.21, 7.43] |
| FeedbackLinear | coupling=0 | 1.01 [1.00, 1.02] | - | 1.01 [1.00, 1.02] | 1.03 [1.00, 1.07] |
| FeedbackLinear | coupling=0.05 | 1.01 [0.99, 1.02] | - | 1.01 [0.99, 1.02] | 1.02 [1.00, 1.04] |
| FeedbackLinear | coupling=0.15 | 1.03 [0.99, 1.11] | - | 1.03 [0.98, 1.11] | 1.04 [1.01, 1.08] |
| FeedbackLinear | coupling=0.4 | 1.00 [0.96, 1.03] | - | 0.99 [0.96, 1.03] | 1.00 [0.96, 1.04] |
| FeedbackLinear | coupling=1 | 1.06 [1.02, 1.12] | - | 1.06 [1.01, 1.11] | 1.17 [0.87, 1.55] |
| FeedbackPredPrey | coupling=0 | 1.03 [1.00, 1.07] | - | 1.03 [1.00, 1.06] | 1.15 [1.03, 1.32] |
| FeedbackPredPrey | coupling=0.25 | 0.98 [0.95, 1.01] | - | 0.97 [0.93, 1.02] | 0.91 [0.81, 1.03] |
| FeedbackPredPrey | coupling=0.5 | 0.96 [0.87, 1.06] | - | 0.95 [0.84, 1.09] | 0.91 [0.74, 1.16] |
| FeedbackPredPrey | coupling=1 | 1.01 [0.76, 1.28] | - | 1.00 [0.71, 1.35] | 1.15 [0.92, 1.41] |
| FeedbackPredPrey | coupling=2 | 1.02 [0.88, 1.22] | - | 1.02 [0.90, 1.15] | 1.14 [0.97, 1.31] |
| HiddenDriver | x_noise=0.05 | 1.03 [0.95, 1.10] | - | 1.03 [0.95, 1.10] | 5.53 [3.68, 8.38] |
| HiddenDriver | x_noise=0.25 | 0.97 [0.93, 1.02] | - | 0.97 [0.93, 1.02] | 6.31 [4.90, 8.08] |
| HiddenDriver | x_noise=0.5 | 1.00 [0.98, 1.03] | - | 1.00 [0.98, 1.02] | 4.47 [3.18, 5.77] |
| HiddenDriver | x_noise=1 | 1.00 [0.98, 1.03] | - | 1.00 [0.98, 1.03] | 3.90 [2.88, 5.23] |
| HiddenDriver | x_noise=2 | 1.01 [0.99, 1.04] | - | 1.02 [0.99, 1.04] | 5.30 [3.68, 7.53] |
| LeakyKnown | noise_ratio=0.05 | 1.02 [1.00, 1.04] | 1.73 [1.40, 2.19] | 1.01 [1.00, 1.02] | 14.61 [10.10, 20.51] |
| LeakyKnown | noise_ratio=0.25 | 1.00 [0.99, 1.01] | 1.68 [1.30, 2.14] | 1.00 [0.99, 1.01] | 5.23 [4.11, 6.95] |
| LeakyKnown | noise_ratio=1 | 1.00 [0.98, 1.02] | 1.92 [1.44, 2.43] | 1.00 [0.99, 1.01] | 2.82 [2.04, 3.66] |
| LeakyKnown | noise_ratio=4 | 1.01 [0.99, 1.03] | 1.66 [1.13, 2.47] | 1.00 [0.99, 1.01] | 1.85 [1.31, 2.57] |
| LeakyKnown | noise_ratio=16 | 1.00 [0.99, 1.01] | 1.03 [0.96, 1.10] | 1.00 [0.99, 1.02] | 1.07 [0.94, 1.24] |

### Size 5M, H = 128

| system | knob | past_cov | known_cov | joint | oracle (past_cov, or known_cov where offered) |
|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.00 [1.00, 1.00] | 1.95 [1.63, 2.23] | 1.00 [1.00, 1.00] | 7.22 [6.26, 8.28] |
| LaggedDriver | lag=8 | 1.04 [1.00, 1.08] | 1.73 [1.59, 1.87] | 1.04 [1.00, 1.08] | 7.59 [6.92, 8.34] |
| LaggedDriver | lag=32 | 1.03 [1.00, 1.07] | 1.14 [1.06, 1.22] | 1.03 [0.99, 1.07] | 7.19 [6.56, 7.86] |
| LaggedDriver | lag=128 | 1.01 [0.99, 1.02] | 1.01 [0.97, 1.05] | 1.01 [0.99, 1.02] | 7.88 [7.25, 8.67] |
| LaggedDriver | lag=512 | 1.01 [0.99, 1.03] | 1.03 [0.99, 1.07] | 1.01 [0.99, 1.03] | 8.27 [7.41, 9.48] |
| NonlinearDriver | nonlin=0 | 1.02 [1.00, 1.03] | 1.16 [1.12, 1.20] | 1.02 [1.00, 1.03] | 6.90 [5.86, 8.09] |
| NonlinearDriver | nonlin=0.25 | 1.08 [1.00, 1.18] | 1.24 [1.15, 1.33] | 1.08 [1.00, 1.19] | 6.62 [5.54, 7.91] |
| NonlinearDriver | nonlin=0.5 | 1.02 [0.98, 1.05] | 1.10 [1.04, 1.16] | 1.01 [0.98, 1.05] | 5.73 [4.82, 6.80] |
| NonlinearDriver | nonlin=0.75 | 1.03 [1.00, 1.05] | 1.04 [1.01, 1.07] | 1.03 [1.00, 1.05] | 5.64 [5.21, 6.10] |
| NonlinearDriver | nonlin=1 | 1.05 [1.02, 1.08] | 1.03 [1.01, 1.06] | 1.05 [1.02, 1.08] | 6.97 [6.56, 7.35] |
| FeedbackLinear | coupling=0 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.01] |
| FeedbackLinear | coupling=0.05 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.01] |
| FeedbackLinear | coupling=0.15 | 1.00 [0.99, 1.01] | - | 1.00 [0.99, 1.01] | 1.00 [1.00, 1.01] |
| FeedbackLinear | coupling=0.4 | 1.00 [1.00, 1.01] | - | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.00] |
| FeedbackLinear | coupling=1 | 1.00 [0.98, 1.02] | - | 1.00 [0.98, 1.02] | 1.08 [0.99, 1.20] |
| FeedbackPredPrey | coupling=0 | 1.03 [0.98, 1.07] | - | 1.02 [0.98, 1.06] | 1.28 [1.08, 1.53] |
| FeedbackPredPrey | coupling=0.25 | 1.02 [0.99, 1.04] | - | 1.02 [0.99, 1.05] | 1.17 [1.05, 1.34] |
| FeedbackPredPrey | coupling=0.5 | 1.03 [0.98, 1.12] | - | 1.02 [0.97, 1.11] | 1.00 [0.90, 1.11] |
| FeedbackPredPrey | coupling=1 | 1.01 [0.96, 1.06] | - | 1.01 [0.96, 1.06] | 1.01 [0.93, 1.10] |
| FeedbackPredPrey | coupling=2 | 1.01 [0.99, 1.04] | - | 1.01 [0.98, 1.03] | 1.07 [1.00, 1.16] |
| HiddenDriver | x_noise=0.05 | 1.04 [1.01, 1.07] | - | 1.04 [1.01, 1.08] | 1.33 [1.17, 1.52] |
| HiddenDriver | x_noise=0.25 | 1.01 [1.00, 1.03] | - | 1.01 [1.00, 1.03] | 1.33 [1.22, 1.48] |
| HiddenDriver | x_noise=0.5 | 1.02 [1.00, 1.03] | - | 1.02 [1.00, 1.04] | 1.27 [1.14, 1.41] |
| HiddenDriver | x_noise=1 | 1.00 [0.99, 1.02] | - | 1.00 [0.99, 1.01] | 1.27 [1.16, 1.38] |
| HiddenDriver | x_noise=2 | 1.00 [0.99, 1.01] | - | 1.00 [0.99, 1.02] | 1.50 [1.30, 1.76] |
| LeakyKnown | noise_ratio=0.05 | 1.00 [1.00, 1.01] | 1.81 [1.48, 2.13] | 1.00 [1.00, 1.00] | 37.78 [34.36, 41.32] |
| LeakyKnown | noise_ratio=0.25 | 1.00 [1.00, 1.00] | 1.98 [1.67, 2.34] | 1.00 [1.00, 1.01] | 8.86 [7.56, 10.39] |
| LeakyKnown | noise_ratio=1 | 1.00 [1.00, 1.00] | 1.97 [1.80, 2.16] | 1.00 [1.00, 1.00] | 3.20 [2.87, 3.54] |
| LeakyKnown | noise_ratio=4 | 1.01 [1.00, 1.01] | 1.46 [1.26, 1.78] | 1.00 [1.00, 1.00] | 1.75 [1.49, 2.14] |
| LeakyKnown | noise_ratio=16 | 1.00 [1.00, 1.01] | 1.02 [0.98, 1.07] | 1.00 [1.00, 1.01] | 1.04 [0.92, 1.18] |

### Size 5M, H = 1024

| system | knob | past_cov | known_cov | joint | oracle (past_cov, or known_cov where offered) |
|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.00 [1.00, 1.00] | 1.04 [0.98, 1.08] | 1.00 [1.00, 1.00] | 8.11 [7.59, 8.72] |
| LaggedDriver | lag=8 | 1.01 [1.00, 1.02] | 1.07 [1.04, 1.09] | 1.01 [1.00, 1.02] | 8.42 [7.97, 8.90] |
| LaggedDriver | lag=32 | 1.00 [1.00, 1.01] | 1.01 [1.00, 1.02] | 1.00 [1.00, 1.01] | 7.79 [7.21, 8.45] |
| LaggedDriver | lag=128 | 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] | 1.00 [1.00, 1.01] | 8.36 [7.90, 8.89] |
| LaggedDriver | lag=512 | 1.01 [1.00, 1.01] | 1.00 [0.98, 1.01] | 1.01 [1.00, 1.01] | 8.38 [7.91, 8.99] |
| NonlinearDriver | nonlin=0 | 1.01 [1.00, 1.03] | 1.03 [1.01, 1.05] | 1.01 [1.00, 1.03] | 8.13 [7.76, 8.51] |
| NonlinearDriver | nonlin=0.25 | 1.01 [1.00, 1.03] | 1.03 [1.01, 1.05] | 1.02 [1.00, 1.03] | 6.74 [6.35, 7.11] |
| NonlinearDriver | nonlin=0.5 | 1.00 [0.99, 1.01] | 0.99 [0.97, 1.01] | 1.00 [0.99, 1.01] | 5.86 [5.55, 6.23] |
| NonlinearDriver | nonlin=0.75 | 1.01 [1.00, 1.02] | 0.99 [0.97, 1.00] | 1.01 [1.00, 1.02] | 6.35 [6.11, 6.57] |
| NonlinearDriver | nonlin=1 | 1.02 [1.01, 1.03] | 1.00 [0.99, 1.01] | 1.02 [1.01, 1.03] | 7.49 [7.25, 7.78] |
| FeedbackLinear | coupling=0 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=0.05 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=0.15 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=0.4 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=1 | 1.00 [0.99, 1.00] | - | 1.00 [0.99, 1.00] | 0.99 [0.98, 1.01] |
| FeedbackPredPrey | coupling=0 | 1.02 [1.01, 1.04] | - | 1.02 [1.01, 1.03] | 1.10 [1.00, 1.20] |
| FeedbackPredPrey | coupling=0.25 | 1.00 [1.00, 1.01] | - | 1.00 [1.00, 1.01] | 1.04 [1.00, 1.08] |
| FeedbackPredPrey | coupling=0.5 | 1.01 [1.00, 1.02] | - | 1.01 [1.00, 1.02] | 1.03 [1.01, 1.06] |
| FeedbackPredPrey | coupling=1 | 1.00 [0.99, 1.00] | - | 1.00 [0.99, 1.01] | 1.01 [1.00, 1.02] |
| FeedbackPredPrey | coupling=2 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.01 [1.00, 1.02] |
| HiddenDriver | x_noise=0.05 | 1.00 [0.99, 1.01] | - | 1.00 [0.99, 1.01] | 1.10 [1.07, 1.12] |
| HiddenDriver | x_noise=0.25 | 0.99 [0.99, 1.00] | - | 1.00 [0.99, 1.00] | 1.08 [1.04, 1.13] |
| HiddenDriver | x_noise=0.5 | 1.00 [0.99, 1.01] | - | 1.00 [0.99, 1.01] | 1.09 [1.06, 1.14] |
| HiddenDriver | x_noise=1 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.01] | 1.11 [1.08, 1.16] |
| HiddenDriver | x_noise=2 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.01] | 1.15 [1.07, 1.24] |
| LeakyKnown | noise_ratio=0.05 | 1.00 [1.00, 1.01] | 1.08 [1.05, 1.10] | 1.00 [1.00, 1.01] | 45.93 [43.79, 48.49] |
| LeakyKnown | noise_ratio=0.25 | 1.00 [1.00, 1.00] | 1.07 [1.04, 1.11] | 1.00 [1.00, 1.00] | 10.34 [10.08, 10.58] |
| LeakyKnown | noise_ratio=1 | 1.00 [1.00, 1.00] | 1.09 [1.07, 1.10] | 1.00 [1.00, 1.00] | 3.52 [3.42, 3.64] |
| LeakyKnown | noise_ratio=4 | 1.01 [1.00, 1.01] | 1.07 [1.04, 1.09] | 1.01 [1.00, 1.01] | 1.74 [1.64, 1.87] |
| LeakyKnown | noise_ratio=16 | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.01] | 1.00 [1.00, 1.00] | 1.15 [1.10, 1.19] |

### Size 28M, H = 16

| system | knob | past_cov | known_cov | joint | oracle (past_cov, or known_cov where offered) |
|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 0.99 [0.98, 1.01] | 4.39 [3.06, 5.81] | 0.99 [0.97, 1.01] | 5.62 [3.59, 8.52] |
| LaggedDriver | lag=8 | 1.50 [1.25, 1.90] | 2.59 [2.07, 3.34] | 1.54 [1.28, 1.95] | 5.98 [4.30, 9.16] |
| LaggedDriver | lag=32 | 1.66 [1.28, 2.20] | 1.55 [1.22, 2.04] | 1.85 [1.37, 2.45] | 7.91 [5.61, 10.88] |
| LaggedDriver | lag=128 | 1.01 [0.97, 1.05] | 1.01 [0.97, 1.05] | 1.01 [0.97, 1.05] | 7.75 [6.41, 9.63] |
| LaggedDriver | lag=512 | 0.98 [0.94, 1.02] | 0.95 [0.91, 1.00] | 0.98 [0.94, 1.02] | 6.13 [5.12, 7.39] |
| NonlinearDriver | nonlin=0 | 1.53 [1.28, 1.84] | 1.36 [1.14, 1.65] | 1.65 [1.33, 2.10] | 7.26 [5.92, 8.84] |
| NonlinearDriver | nonlin=0.25 | 1.83 [1.53, 2.17] | 1.66 [1.32, 2.06] | 1.92 [1.60, 2.28] | 6.12 [5.04, 7.30] |
| NonlinearDriver | nonlin=0.5 | 1.62 [1.19, 2.24] | 1.46 [1.04, 2.08] | 1.64 [1.22, 2.27] | 6.45 [4.38, 9.40] |
| NonlinearDriver | nonlin=0.75 | 1.28 [1.05, 1.60] | 1.17 [1.00, 1.42] | 1.31 [1.07, 1.64] | 5.87 [5.11, 6.80] |
| NonlinearDriver | nonlin=1 | 1.20 [1.12, 1.27] | 1.12 [1.02, 1.24] | 1.22 [1.14, 1.30] | 6.34 [5.42, 7.42] |
| FeedbackLinear | coupling=0 | 1.00 [0.99, 1.00] | - | 1.00 [0.99, 1.00] | 1.02 [1.00, 1.04] |
| FeedbackLinear | coupling=0.05 | 1.01 [1.00, 1.01] | - | 1.01 [1.00, 1.01] | 1.00 [0.98, 1.02] |
| FeedbackLinear | coupling=0.15 | 1.04 [1.00, 1.12] | - | 1.04 [1.00, 1.13] | 1.04 [1.01, 1.09] |
| FeedbackLinear | coupling=0.4 | 1.02 [0.99, 1.06] | - | 1.02 [0.98, 1.06] | 1.00 [0.96, 1.04] |
| FeedbackLinear | coupling=1 | 1.03 [0.96, 1.11] | - | 1.02 [0.95, 1.12] | 1.24 [1.02, 1.53] |
| FeedbackPredPrey | coupling=0 | 1.01 [0.98, 1.05] | - | 1.02 [0.99, 1.05] | 1.16 [1.05, 1.32] |
| FeedbackPredPrey | coupling=0.25 | 1.00 [0.94, 1.03] | - | 0.99 [0.92, 1.04] | 0.94 [0.83, 1.06] |
| FeedbackPredPrey | coupling=0.5 | 0.95 [0.90, 1.00] | - | 0.95 [0.87, 1.02] | 0.90 [0.70, 1.19] |
| FeedbackPredPrey | coupling=1 | 1.12 [0.95, 1.32] | - | 1.08 [0.80, 1.42] | 1.19 [0.88, 1.53] |
| FeedbackPredPrey | coupling=2 | 1.06 [0.91, 1.25] | - | 1.11 [0.91, 1.35] | 1.04 [0.84, 1.31] |
| HiddenDriver | x_noise=0.05 | 1.20 [1.05, 1.40] | - | 1.38 [1.13, 1.74] | 5.31 [3.55, 7.89] |
| HiddenDriver | x_noise=0.25 | 1.29 [1.06, 1.69] | - | 1.35 [1.11, 1.73] | 6.02 [4.65, 7.67] |
| HiddenDriver | x_noise=0.5 | 1.05 [0.99, 1.14] | - | 1.08 [0.99, 1.19] | 4.35 [2.97, 5.79] |
| HiddenDriver | x_noise=1 | 1.02 [0.99, 1.04] | - | 1.03 [0.98, 1.10] | 3.87 [2.84, 5.22] |
| HiddenDriver | x_noise=2 | 1.02 [1.00, 1.04] | - | 1.03 [1.00, 1.05] | 5.28 [3.44, 7.87] |
| LeakyKnown | noise_ratio=0.05 | 1.00 [0.99, 1.02] | 3.90 [3.11, 4.90] | 0.99 [0.98, 1.00] | 14.86 [10.07, 21.45] |
| LeakyKnown | noise_ratio=0.25 | 1.00 [0.97, 1.03] | 3.03 [2.17, 4.24] | 1.00 [0.99, 1.02] | 5.22 [4.10, 7.08] |
| LeakyKnown | noise_ratio=1 | 1.00 [0.97, 1.02] | 2.73 [2.16, 3.43] | 1.01 [1.00, 1.03] | 2.91 [2.10, 3.77] |
| LeakyKnown | noise_ratio=4 | 1.01 [0.99, 1.02] | 1.92 [1.43, 2.67] | 1.00 [0.99, 1.01] | 1.86 [1.33, 2.66] |
| LeakyKnown | noise_ratio=16 | 1.01 [0.99, 1.02] | 1.03 [0.96, 1.09] | 1.01 [0.99, 1.02] | 0.99 [0.86, 1.17] |

### Size 28M, H = 128

| system | knob | past_cov | known_cov | joint | oracle (past_cov, or known_cov where offered) |
|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.00 [1.00, 1.00] | 5.31 [4.57, 6.11] | 1.00 [1.00, 1.00] | 7.24 [6.36, 8.23] |
| LaggedDriver | lag=8 | 1.05 [1.02, 1.07] | 3.00 [2.71, 3.28] | 1.05 [1.03, 1.07] | 7.59 [7.03, 8.23] |
| LaggedDriver | lag=32 | 1.04 [0.96, 1.13] | 1.40 [1.18, 1.67] | 1.06 [0.98, 1.16] | 7.15 [6.56, 7.79] |
| LaggedDriver | lag=128 | 1.00 [0.98, 1.02] | 1.01 [0.98, 1.03] | 1.00 [0.99, 1.02] | 7.89 [7.31, 8.60] |
| LaggedDriver | lag=512 | 1.00 [0.98, 1.02] | 1.01 [0.99, 1.03] | 1.00 [0.98, 1.02] | 8.16 [7.19, 9.52] |
| NonlinearDriver | nonlin=0 | 1.07 [1.01, 1.14] | 1.44 [1.25, 1.67] | 1.08 [1.02, 1.15] | 7.04 [6.01, 8.17] |
| NonlinearDriver | nonlin=0.25 | 1.19 [1.13, 1.27] | 1.83 [1.56, 2.20] | 1.22 [1.16, 1.31] | 6.62 [5.51, 7.94] |
| NonlinearDriver | nonlin=0.5 | 1.06 [0.98, 1.16] | 1.55 [1.34, 1.82] | 1.08 [1.00, 1.17] | 5.69 [4.78, 6.82] |
| NonlinearDriver | nonlin=0.75 | 1.02 [0.97, 1.07] | 1.17 [1.09, 1.28] | 1.02 [0.96, 1.07] | 5.64 [5.21, 6.12] |
| NonlinearDriver | nonlin=1 | 1.09 [1.04, 1.16] | 1.08 [1.03, 1.14] | 1.10 [1.04, 1.18] | 6.82 [6.32, 7.23] |
| FeedbackLinear | coupling=0 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.01] |
| FeedbackLinear | coupling=0.05 | 1.00 [1.00, 1.01] | - | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] |
| FeedbackLinear | coupling=0.15 | 1.00 [1.00, 1.01] | - | 1.00 [1.00, 1.01] | 1.01 [1.00, 1.02] |
| FeedbackLinear | coupling=0.4 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] |
| FeedbackLinear | coupling=1 | 0.98 [0.94, 1.03] | - | 0.98 [0.94, 1.02] | 1.03 [0.90, 1.15] |
| FeedbackPredPrey | coupling=0 | 1.01 [0.97, 1.04] | - | 1.01 [0.98, 1.04] | 1.19 [0.97, 1.50] |
| FeedbackPredPrey | coupling=0.25 | 1.04 [1.00, 1.08] | - | 1.04 [1.00, 1.10] | 1.15 [1.06, 1.31] |
| FeedbackPredPrey | coupling=0.5 | 1.00 [0.98, 1.02] | - | 1.01 [0.97, 1.05] | 1.02 [0.89, 1.17] |
| FeedbackPredPrey | coupling=1 | 1.01 [0.98, 1.04] | - | 1.01 [0.96, 1.06] | 1.02 [0.94, 1.12] |
| FeedbackPredPrey | coupling=2 | 1.02 [0.98, 1.06] | - | 1.05 [1.00, 1.09] | 1.08 [1.02, 1.15] |
| HiddenDriver | x_noise=0.05 | 1.15 [1.05, 1.29] | - | 1.19 [1.07, 1.32] | 1.34 [1.16, 1.54] |
| HiddenDriver | x_noise=0.25 | 1.06 [1.01, 1.11] | - | 1.09 [1.02, 1.18] | 1.31 [1.21, 1.42] |
| HiddenDriver | x_noise=0.5 | 1.07 [1.04, 1.11] | - | 1.12 [1.06, 1.19] | 1.31 [1.19, 1.45] |
| HiddenDriver | x_noise=1 | 1.01 [0.99, 1.02] | - | 1.02 [0.99, 1.04] | 1.26 [1.13, 1.40] |
| HiddenDriver | x_noise=2 | 1.00 [0.99, 1.01] | - | 1.01 [1.00, 1.02] | 1.49 [1.26, 1.79] |
| LeakyKnown | noise_ratio=0.05 | 1.00 [1.00, 1.01] | 3.74 [3.42, 4.11] | 1.00 [1.00, 1.01] | 37.16 [34.22, 40.45] |
| LeakyKnown | noise_ratio=0.25 | 1.00 [1.00, 1.01] | 3.54 [2.98, 4.15] | 1.00 [1.00, 1.01] | 8.70 [7.39, 10.37] |
| LeakyKnown | noise_ratio=1 | 1.01 [1.00, 1.03] | 2.72 [2.34, 3.11] | 1.01 [1.00, 1.01] | 3.21 [2.89, 3.55] |
| LeakyKnown | noise_ratio=4 | 1.00 [1.00, 1.01] | 1.55 [1.34, 1.87] | 1.00 [1.00, 1.00] | 1.67 [1.42, 2.04] |
| LeakyKnown | noise_ratio=16 | 1.00 [1.00, 1.00] | 1.02 [0.98, 1.06] | 1.00 [1.00, 1.00] | 1.02 [0.91, 1.15] |

### Size 28M, H = 1024

| system | knob | past_cov | known_cov | joint | oracle (past_cov, or known_cov where offered) |
|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.00 [1.00, 1.00] | 1.24 [1.15, 1.32] | 1.00 [1.00, 1.00] | 7.94 [7.35, 8.64] |
| LaggedDriver | lag=8 | 1.01 [1.00, 1.02] | 1.17 [1.12, 1.24] | 1.01 [1.00, 1.03] | 8.30 [7.93, 8.67] |
| LaggedDriver | lag=32 | 1.00 [0.99, 1.02] | 1.08 [1.05, 1.13] | 1.00 [0.98, 1.02] | 7.53 [7.00, 8.15] |
| LaggedDriver | lag=128 | 1.00 [0.99, 1.01] | 1.00 [1.00, 1.01] | 1.00 [0.99, 1.01] | 8.19 [7.72, 8.70] |
| LaggedDriver | lag=512 | 1.01 [1.00, 1.03] | 1.02 [1.00, 1.03] | 1.01 [1.00, 1.03] | 8.31 [7.79, 8.96] |
| NonlinearDriver | nonlin=0 | 1.01 [0.99, 1.03] | 1.12 [1.07, 1.19] | 1.01 [0.99, 1.05] | 8.05 [7.63, 8.55] |
| NonlinearDriver | nonlin=0.25 | 1.01 [0.99, 1.03] | 1.12 [1.09, 1.15] | 1.01 [1.00, 1.03] | 6.55 [6.10, 6.97] |
| NonlinearDriver | nonlin=0.5 | 0.99 [0.97, 1.02] | 1.11 [1.09, 1.14] | 0.99 [0.97, 1.02] | 5.69 [5.39, 6.06] |
| NonlinearDriver | nonlin=0.75 | 1.00 [1.00, 1.01] | 1.02 [0.98, 1.04] | 1.00 [0.99, 1.02] | 6.20 [5.99, 6.42] |
| NonlinearDriver | nonlin=1 | 1.01 [1.00, 1.02] | 1.00 [1.00, 1.01] | 1.01 [1.00, 1.02] | 7.32 [7.07, 7.60] |
| FeedbackLinear | coupling=0 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.01] |
| FeedbackLinear | coupling=0.05 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=0.15 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=0.4 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=1 | 1.00 [0.99, 1.01] | - | 1.00 [0.99, 1.01] | 0.99 [0.98, 1.01] |
| FeedbackPredPrey | coupling=0 | 1.01 [0.99, 1.02] | - | 1.01 [0.99, 1.03] | 1.05 [0.98, 1.12] |
| FeedbackPredPrey | coupling=0.25 | 1.02 [1.01, 1.03] | - | 1.02 [1.01, 1.03] | 1.05 [1.02, 1.08] |
| FeedbackPredPrey | coupling=0.5 | 1.00 [1.00, 1.01] | - | 1.01 [1.00, 1.01] | 1.03 [1.00, 1.06] |
| FeedbackPredPrey | coupling=1 | 1.01 [1.00, 1.03] | - | 1.01 [1.00, 1.02] | 1.01 [0.99, 1.04] |
| FeedbackPredPrey | coupling=2 | 1.01 [1.00, 1.01] | - | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] |
| HiddenDriver | x_noise=0.05 | 1.01 [1.00, 1.03] | - | 1.02 [1.00, 1.04] | 1.08 [1.06, 1.11] |
| HiddenDriver | x_noise=0.25 | 1.01 [0.98, 1.04] | - | 1.02 [0.99, 1.04] | 1.06 [1.02, 1.10] |
| HiddenDriver | x_noise=0.5 | 1.01 [0.99, 1.02] | - | 1.02 [1.00, 1.04] | 1.09 [1.04, 1.18] |
| HiddenDriver | x_noise=1 | 1.01 [1.00, 1.01] | - | 1.01 [1.00, 1.02] | 1.10 [1.07, 1.14] |
| HiddenDriver | x_noise=2 | 1.00 [1.00, 1.00] | - | 1.00 [0.99, 1.00] | 1.13 [1.04, 1.25] |
| LeakyKnown | noise_ratio=0.05 | 1.00 [1.00, 1.01] | 1.16 [1.10, 1.22] | 1.01 [1.00, 1.01] | 44.48 [42.86, 46.27] |
| LeakyKnown | noise_ratio=0.25 | 1.00 [1.00, 1.01] | 1.14 [1.04, 1.24] | 1.00 [1.00, 1.01] | 10.19 [9.77, 10.66] |
| LeakyKnown | noise_ratio=1 | 1.00 [1.00, 1.01] | 1.15 [1.09, 1.19] | 1.00 [1.00, 1.01] | 3.39 [3.31, 3.47] |
| LeakyKnown | noise_ratio=4 | 1.00 [1.00, 1.01] | 1.13 [1.10, 1.17] | 1.01 [1.00, 1.01] | 1.69 [1.59, 1.82] |
| LeakyKnown | noise_ratio=16 | 1.00 [1.00, 1.01] | 1.02 [1.00, 1.03] | 1.00 [1.00, 1.01] | 1.13 [1.09, 1.17] |

### Size 123M, H = 16

| system | knob | past_cov | known_cov | joint | oracle (past_cov, or known_cov where offered) |
|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.01 [1.00, 1.01] | 3.73 [1.77, 5.84] | 1.00 [0.99, 1.00] | 4.36 [2.01, 7.15] |
| LaggedDriver | lag=32 | 2.10 [1.48, 2.99] | 1.45 [0.98, 2.55] | 2.47 [1.54, 3.41] | 7.88 [4.59, 11.81] |
| LaggedDriver | lag=512 | 0.96 [0.93, 0.99] | 0.96 [0.91, 1.00] | 0.96 [0.93, 0.99] | 5.74 [4.32, 7.62] |
| NonlinearDriver | nonlin=0 | 1.75 [1.22, 2.11] | 1.09 [0.91, 1.43] | 2.27 [1.22, 4.39] | 6.40 [4.69, 8.32] |
| NonlinearDriver | nonlin=0.5 | 1.81 [1.33, 2.26] | 1.39 [1.11, 1.93] | 2.13 [1.57, 2.76] | 5.28 [3.21, 10.41] |
| NonlinearDriver | nonlin=1 | 1.70 [1.31, 2.28] | 1.52 [1.21, 1.74] | 1.72 [1.29, 2.34] | 5.10 [4.31, 6.28] |
| FeedbackLinear | coupling=0 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [0.98, 1.02] |
| FeedbackLinear | coupling=0.15 | 1.08 [0.97, 1.26] | - | 1.08 [0.98, 1.25] | 1.05 [1.00, 1.14] |
| FeedbackLinear | coupling=1 | 1.03 [0.95, 1.20] | - | 1.04 [0.94, 1.22] | 1.20 [1.03, 1.55] |
| FeedbackPredPrey | coupling=0 | 0.99 [0.99, 1.00] | - | 0.98 [0.98, 0.99] | 1.17 [1.06, 1.33] |
| FeedbackPredPrey | coupling=0.5 | 0.99 [0.81, 1.15] | - | 1.02 [0.80, 1.25] | 0.96 [0.60, 1.60] |
| FeedbackPredPrey | coupling=2 | 1.15 [0.94, 1.39] | - | 1.32 [0.96, 1.80] | 1.26 [0.95, 1.69] |
| HiddenDriver | x_noise=0.05 | 1.78 [1.17, 2.70] | - | 1.88 [1.17, 3.02] | 4.35 [2.40, 7.90] |
| HiddenDriver | x_noise=0.5 | 1.00 [0.98, 1.03] | - | 0.98 [0.97, 0.99] | 3.50 [1.95, 5.04] |
| HiddenDriver | x_noise=2 | 0.99 [0.97, 1.01] | - | 1.00 [0.96, 1.03] | 5.74 [3.66, 11.12] |
| LeakyKnown | noise_ratio=0.05 | 1.00 [0.96, 1.04] | 6.28 [4.28, 8.64] | 0.98 [0.96, 0.99] | 19.10 [14.57, 25.04] |
| LeakyKnown | noise_ratio=1 | 1.03 [0.99, 1.06] | 3.43 [2.10, 5.61] | 1.04 [0.98, 1.10] | 3.48 [2.54, 4.87] |
| LeakyKnown | noise_ratio=16 | 1.00 [0.99, 1.02] | 1.00 [0.98, 1.04] | 1.00 [0.99, 1.02] | 1.02 [0.94, 1.14] |

### Size 123M, H = 128

| system | knob | past_cov | known_cov | joint | oracle (past_cov, or known_cov where offered) |
|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.00 [1.00, 1.00] | 6.65 [5.26, 8.40] | 1.00 [1.00, 1.00] | 7.18 [5.55, 9.34] |
| LaggedDriver | lag=32 | 1.14 [1.07, 1.28] | 1.59 [1.15, 2.41] | 1.16 [1.07, 1.29] | 7.23 [6.87, 7.78] |
| LaggedDriver | lag=512 | 0.99 [0.97, 1.01] | 1.01 [0.99, 1.04] | 0.99 [0.97, 1.01] | 8.08 [7.36, 9.56] |
| NonlinearDriver | nonlin=0 | 1.14 [1.05, 1.23] | 1.30 [1.09, 1.74] | 1.18 [1.08, 1.37] | 7.10 [5.30, 9.42] |
| NonlinearDriver | nonlin=0.5 | 1.04 [1.00, 1.08] | 1.84 [1.59, 2.18] | 1.05 [1.01, 1.10] | 5.18 [3.86, 7.53] |
| NonlinearDriver | nonlin=1 | 1.17 [1.04, 1.33] | 1.39 [1.22, 1.58] | 1.18 [1.03, 1.34] | 6.41 [5.83, 6.90] |
| FeedbackLinear | coupling=0 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [0.98, 1.01] |
| FeedbackLinear | coupling=0.15 | 1.01 [1.00, 1.03] | - | 1.01 [1.00, 1.03] | 1.00 [0.99, 1.02] |
| FeedbackLinear | coupling=1 | 1.03 [1.00, 1.06] | - | 1.03 [1.00, 1.07] | 1.01 [0.88, 1.17] |
| FeedbackPredPrey | coupling=0 | 1.01 [0.99, 1.03] | - | 1.00 [0.97, 1.03] | 1.29 [1.01, 1.63] |
| FeedbackPredPrey | coupling=0.5 | 1.00 [0.89, 1.11] | - | 1.04 [0.88, 1.20] | 1.00 [0.76, 1.17] |
| FeedbackPredPrey | coupling=2 | 1.03 [0.98, 1.07] | - | 1.05 [1.01, 1.09] | 1.08 [1.01, 1.16] |
| HiddenDriver | x_noise=0.05 | 1.17 [1.00, 1.43] | - | 1.15 [0.94, 1.45] | 1.32 [1.12, 1.61] |
| HiddenDriver | x_noise=0.5 | 1.00 [1.00, 1.01] | - | 1.02 [1.00, 1.04] | 1.16 [1.05, 1.27] |
| HiddenDriver | x_noise=2 | 1.00 [0.99, 1.01] | - | 1.00 [0.99, 1.02] | 1.52 [1.21, 2.17] |
| LeakyKnown | noise_ratio=0.05 | 1.01 [1.01, 1.02] | 9.60 [8.89, 10.26] | 1.01 [1.00, 1.02] | 35.51 [32.14, 38.25] |
| LeakyKnown | noise_ratio=1 | 1.03 [1.00, 1.06] | 3.24 [2.55, 3.76] | 1.03 [1.00, 1.06] | 3.45 [2.70, 4.02] |
| LeakyKnown | noise_ratio=16 | 1.00 [1.00, 1.00] | 1.00 [0.98, 1.02] | 1.00 [1.00, 1.00] | 0.90 [0.83, 0.95] |

### Size 209M, H = 16

| system | knob | past_cov | known_cov | joint | oracle (past_cov, or known_cov where offered) |
|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.00 [0.99, 1.01] | 3.73 [2.06, 5.19] | 0.98 [0.96, 1.00] | 4.47 [2.23, 7.00] |
| LaggedDriver | lag=32 | 2.41 [1.48, 3.56] | 2.64 [1.64, 3.83] | 2.90 [1.69, 3.98] | 7.92 [4.59, 11.70] |
| LaggedDriver | lag=512 | 1.04 [1.02, 1.06] | 1.03 [1.00, 1.05] | 1.03 [1.01, 1.05] | 5.86 [4.36, 7.87] |
| NonlinearDriver | nonlin=0 | 1.47 [1.09, 2.17] | 1.88 [1.38, 2.55] | 1.90 [1.50, 2.48] | 6.36 [4.63, 8.57] |
| NonlinearDriver | nonlin=0.5 | 2.04 [1.45, 2.87] | 1.91 [1.61, 2.20] | 2.25 [1.71, 2.97] | 4.97 [2.96, 9.15] |
| NonlinearDriver | nonlin=1 | 1.64 [1.36, 1.96] | 1.73 [1.40, 2.06] | 1.81 [1.66, 2.05] | 4.86 [4.27, 5.53] |
| FeedbackLinear | coupling=0 | 1.00 [1.00, 1.01] | - | 1.00 [1.00, 1.01] | 1.00 [0.98, 1.03] |
| FeedbackLinear | coupling=0.15 | 1.08 [1.00, 1.25] | - | 1.09 [1.01, 1.24] | 1.05 [1.00, 1.15] |
| FeedbackLinear | coupling=1 | 0.97 [0.95, 1.00] | - | 0.99 [0.96, 1.03] | 1.19 [1.09, 1.35] |
| FeedbackPredPrey | coupling=0 | 0.95 [0.90, 1.01] | - | 0.94 [0.89, 1.01] | 1.16 [1.03, 1.33] |
| FeedbackPredPrey | coupling=0.5 | 0.94 [0.84, 0.99] | - | 0.91 [0.72, 1.09] | 0.94 [0.53, 1.67] |
| FeedbackPredPrey | coupling=2 | 1.10 [1.00, 1.21] | - | 1.35 [1.04, 1.76] | 1.28 [0.98, 1.68] |
| HiddenDriver | x_noise=0.05 | 1.77 [0.98, 3.21] | - | 1.85 [0.99, 3.46] | 4.35 [2.43, 7.78] |
| HiddenDriver | x_noise=0.5 | 0.98 [0.96, 1.00] | - | 0.96 [0.84, 1.07] | 3.49 [1.93, 5.10] |
| HiddenDriver | x_noise=2 | 1.02 [1.00, 1.04] | - | 1.01 [0.96, 1.06] | 5.93 [3.64, 11.41] |
| LeakyKnown | noise_ratio=0.05 | 1.01 [1.00, 1.02] | 5.96 [4.78, 7.46] | 1.00 [0.98, 1.02] | 19.39 [14.28, 26.32] |
| LeakyKnown | noise_ratio=1 | 1.02 [1.01, 1.03] | 3.24 [2.11, 4.98] | 1.02 [1.00, 1.06] | 3.32 [2.58, 4.72] |
| LeakyKnown | noise_ratio=16 | 1.00 [1.00, 1.01] | 1.04 [0.99, 1.09] | 1.00 [0.99, 1.01] | 0.99 [0.90, 1.14] |

### Size 209M, H = 128

| system | knob | past_cov | known_cov | joint | oracle (past_cov, or known_cov where offered) |
|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.00 [1.00, 1.00] | 6.68 [5.17, 8.63] | 1.00 [1.00, 1.00] | 7.11 [5.54, 9.24] |
| LaggedDriver | lag=32 | 1.21 [1.12, 1.33] | 2.58 [2.18, 3.04] | 1.20 [1.13, 1.31] | 6.87 [6.40, 7.38] |
| LaggedDriver | lag=512 | 0.97 [0.89, 1.01] | 0.98 [0.91, 1.03] | 0.97 [0.89, 1.01] | 7.60 [7.25, 8.27] |
| NonlinearDriver | nonlin=0 | 1.10 [1.01, 1.19] | 1.95 [1.38, 2.64] | 1.13 [1.07, 1.19] | 6.92 [5.20, 9.22] |
| NonlinearDriver | nonlin=0.5 | 1.07 [1.03, 1.11] | 2.07 [1.64, 2.74] | 1.09 [1.06, 1.12] | 5.14 [3.84, 7.40] |
| NonlinearDriver | nonlin=1 | 1.14 [1.04, 1.25] | 1.52 [1.27, 1.74] | 1.15 [1.06, 1.26] | 6.25 [5.38, 7.00] |
| FeedbackLinear | coupling=0 | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [0.98, 1.01] |
| FeedbackLinear | coupling=0.15 | 1.01 [1.00, 1.03] | - | 1.01 [1.00, 1.03] | 1.01 [0.99, 1.02] |
| FeedbackLinear | coupling=1 | 0.94 [0.90, 1.02] | - | 0.95 [0.90, 1.02] | 1.04 [0.96, 1.13] |
| FeedbackPredPrey | coupling=0 | 0.96 [0.88, 1.04] | - | 0.96 [0.87, 1.05] | 1.24 [0.99, 1.54] |
| FeedbackPredPrey | coupling=0.5 | 0.98 [0.93, 1.03] | - | 1.01 [0.84, 1.18] | 1.03 [0.74, 1.38] |
| FeedbackPredPrey | coupling=2 | 1.01 [0.98, 1.03] | - | 1.04 [0.99, 1.09] | 1.07 [1.01, 1.14] |
| HiddenDriver | x_noise=0.05 | 1.22 [1.00, 1.54] | - | 1.19 [0.95, 1.57] | 1.30 [1.11, 1.59] |
| HiddenDriver | x_noise=0.5 | 1.02 [1.01, 1.03] | - | 1.05 [1.02, 1.07] | 1.11 [1.03, 1.18] |
| HiddenDriver | x_noise=2 | 0.99 [0.99, 1.00] | - | 1.01 [0.98, 1.04] | 1.54 [1.19, 2.25] |
| LeakyKnown | noise_ratio=0.05 | 1.00 [1.00, 1.01] | 12.23 [11.54, 13.35] | 1.01 [1.00, 1.01] | 35.36 [32.06, 38.24] |
| LeakyKnown | noise_ratio=1 | 1.02 [1.01, 1.03] | 3.01 [2.42, 3.38] | 1.02 [1.00, 1.04] | 3.19 [2.64, 3.59] |
| LeakyKnown | noise_ratio=16 | 1.01 [1.00, 1.01] | 1.01 [0.99, 1.02] | 1.00 [1.00, 1.01] | 0.92 [0.84, 0.98] |

## Table 1b. MASE ratio to the oracle per condition (T = 2048)

Geometric mean over samples [95% bootstrap CI] of MASE(Timestral) / MASE(oracle matched to the condition's information). `classical` is the best of seasonal naive, ETS, ARIMA per sample over the same oracle (marginal oracle).

### Size 5M, H = 16

| system | knob | uni | past_cov | known_cov | joint | classical |
|---|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.00 [0.89, 1.12] | 1.02 [0.90, 1.13] | 2.16 [1.72, 2.91] | 1.02 [0.90, 1.14] | 1.03 [0.80, 1.32] |
| LaggedDriver | lag=8 | 1.76 [1.39, 2.33] | 1.50 [1.22, 1.87] | 3.75 [3.10, 4.57] | 1.47 [1.19, 1.87] | 1.45 [0.98, 2.04] |
| LaggedDriver | lag=32 | 8.26 [5.66, 11.74] | 7.69 [5.48, 10.15] | 7.19 [5.49, 9.18] | 7.67 [5.46, 10.12] | 7.97 [5.82, 10.56] |
| LaggedDriver | lag=128 | 7.23 [5.87, 8.97] | 7.35 [5.92, 9.19] | 7.48 [5.96, 9.40] | 7.36 [5.93, 9.18] | 7.11 [5.71, 8.88] |
| LaggedDriver | lag=512 | 6.37 [5.10, 8.01] | 6.30 [4.92, 8.12] | 6.30 [5.24, 7.64] | 6.33 [4.93, 8.19] | 5.31 [4.15, 7.06] |
| NonlinearDriver | nonlin=0 | 7.12 [5.83, 8.63] | 7.05 [5.59, 8.79] | 6.81 [5.30, 8.50] | 7.05 [5.57, 8.79] | 7.18 [5.59, 9.16] |
| NonlinearDriver | nonlin=0.25 | 5.50 [4.46, 6.66] | 4.90 [4.05, 5.86] | 4.75 [4.11, 5.61] | 4.89 [4.04, 5.84] | 5.60 [4.17, 7.34] |
| NonlinearDriver | nonlin=0.5 | 6.73 [4.62, 9.68] | 6.18 [4.47, 8.54] | 6.49 [4.98, 8.24] | 6.20 [4.48, 8.57] | 5.75 [4.10, 7.87] |
| NonlinearDriver | nonlin=0.75 | 5.82 [4.99, 6.96] | 5.23 [4.35, 6.35] | 5.50 [4.79, 6.39] | 5.22 [4.35, 6.34] | 5.28 [4.70, 5.96] |
| NonlinearDriver | nonlin=1 | 6.16 [5.14, 7.39] | 5.62 [4.39, 7.10] | 5.82 [4.60, 7.30] | 5.61 [4.36, 7.11] | 6.48 [5.22, 7.91] |
| FeedbackLinear | coupling=0 | 1.03 [1.00, 1.07] | 1.03 [1.00, 1.06] | - | 1.03 [1.00, 1.06] | 1.00 [0.98, 1.03] |
| FeedbackLinear | coupling=0.05 | 1.02 [1.00, 1.04] | 1.01 [0.99, 1.03] | - | 1.01 [0.99, 1.03] | 1.00 [0.98, 1.02] |
| FeedbackLinear | coupling=0.15 | 1.04 [1.01, 1.08] | 1.01 [0.97, 1.04] | - | 1.01 [0.97, 1.04] | 0.98 [0.92, 1.03] |
| FeedbackLinear | coupling=0.4 | 1.00 [0.96, 1.04] | 1.00 [0.97, 1.03] | - | 1.00 [0.97, 1.04] | 0.96 [0.92, 1.00] |
| FeedbackLinear | coupling=1 | 1.17 [0.87, 1.55] | 1.10 [0.85, 1.42] | - | 1.11 [0.86, 1.42] | 1.14 [1.00, 1.31] |
| FeedbackPredPrey | coupling=0 | 1.15 [1.03, 1.32] | 1.12 [1.01, 1.25] | - | 1.12 [1.01, 1.25] | 1.05 [0.96, 1.16] |
| FeedbackPredPrey | coupling=0.25 | 0.91 [0.81, 1.03] | 0.93 [0.85, 1.04] | - | 0.94 [0.86, 1.04] | 0.90 [0.77, 1.05] |
| FeedbackPredPrey | coupling=0.5 | 0.91 [0.74, 1.16] | 0.95 [0.77, 1.15] | - | 0.95 [0.78, 1.14] | 0.69 [0.51, 0.93] |
| FeedbackPredPrey | coupling=1 | 1.15 [0.92, 1.41] | 1.13 [0.94, 1.38] | - | 1.15 [0.96, 1.39] | 0.96 [0.69, 1.25] |
| FeedbackPredPrey | coupling=2 | 1.14 [0.97, 1.31] | 1.11 [0.98, 1.26] | - | 1.11 [0.99, 1.27] | 1.18 [0.68, 1.76] |
| HiddenDriver | x_noise=0.05 | 5.53 [3.68, 8.38] | 5.36 [3.47, 8.27] | - | 5.35 [3.46, 8.28] | 6.05 [4.13, 8.64] |
| HiddenDriver | x_noise=0.25 | 6.31 [4.90, 8.08] | 6.48 [5.11, 8.06] | - | 6.50 [5.12, 8.10] | 5.64 [4.51, 7.35] |
| HiddenDriver | x_noise=0.5 | 4.47 [3.18, 5.77] | 4.45 [3.17, 5.68] | - | 4.46 [3.17, 5.68] | 3.66 [2.71, 4.62] |
| HiddenDriver | x_noise=1 | 3.90 [2.88, 5.23] | 3.89 [2.89, 5.20] | - | 3.89 [2.89, 5.22] | 4.36 [3.27, 5.69] |
| HiddenDriver | x_noise=2 | 5.30 [3.68, 7.53] | 5.23 [3.59, 7.40] | - | 5.21 [3.57, 7.39] | 5.44 [4.02, 7.75] |
| LeakyKnown | noise_ratio=0.05 | 0.99 [0.88, 1.11] | 0.98 [0.87, 1.08] | 8.47 [6.64, 10.75] | 0.99 [0.88, 1.10] | 0.90 [0.78, 1.03] |
| LeakyKnown | noise_ratio=0.25 | 1.04 [0.95, 1.13] | 1.04 [0.96, 1.12] | 3.11 [2.58, 3.67] | 1.04 [0.95, 1.13] | 0.98 [0.79, 1.27] |
| LeakyKnown | noise_ratio=1 | 1.10 [0.97, 1.28] | 1.10 [0.97, 1.27] | 1.47 [1.22, 1.77] | 1.10 [0.97, 1.27] | 1.05 [0.83, 1.36] |
| LeakyKnown | noise_ratio=4 | 1.01 [0.90, 1.13] | 1.00 [0.90, 1.12] | 1.11 [0.99, 1.23] | 1.02 [0.90, 1.13] | 0.77 [0.62, 0.91] |
| LeakyKnown | noise_ratio=16 | 1.05 [0.93, 1.17] | 1.05 [0.93, 1.17] | 1.04 [0.93, 1.16] | 1.04 [0.93, 1.16] | 0.90 [0.64, 1.27] |

### Size 5M, H = 128

| system | knob | uni | past_cov | known_cov | joint | classical |
|---|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.00 [0.95, 1.04] | 1.00 [0.95, 1.04] | 3.69 [3.16, 4.25] | 1.00 [0.95, 1.04] | 1.06 [0.96, 1.23] |
| LaggedDriver | lag=8 | 1.02 [0.96, 1.07] | 0.98 [0.91, 1.04] | 4.40 [4.14, 4.63] | 0.98 [0.91, 1.04] | 1.05 [0.89, 1.22] |
| LaggedDriver | lag=32 | 1.27 [1.15, 1.40] | 1.24 [1.13, 1.34] | 6.32 [5.68, 7.17] | 1.24 [1.13, 1.34] | 1.44 [1.27, 1.64] |
| LaggedDriver | lag=128 | 7.91 [7.31, 8.66] | 7.85 [7.32, 8.56] | 7.81 [7.19, 8.68] | 7.85 [7.32, 8.56] | 8.49 [7.49, 9.78] |
| LaggedDriver | lag=512 | 8.34 [7.49, 9.56] | 8.24 [7.41, 9.42] | 8.02 [7.30, 9.03] | 8.25 [7.42, 9.43] | 8.57 [7.22, 10.46] |
| NonlinearDriver | nonlin=0 | 1.28 [1.17, 1.41] | 1.26 [1.15, 1.38] | 5.96 [5.07, 7.03] | 1.26 [1.15, 1.38] | 1.51 [1.34, 1.71] |
| NonlinearDriver | nonlin=0.25 | 1.34 [1.26, 1.42] | 1.24 [1.14, 1.33] | 5.35 [4.53, 6.14] | 1.23 [1.14, 1.33] | 1.36 [1.12, 1.59] |
| NonlinearDriver | nonlin=0.5 | 1.32 [1.22, 1.44] | 1.30 [1.21, 1.41] | 5.21 [4.51, 5.97] | 1.30 [1.21, 1.41] | 1.38 [1.25, 1.53] |
| NonlinearDriver | nonlin=0.75 | 1.22 [1.14, 1.30] | 1.19 [1.13, 1.26] | 5.42 [5.06, 5.76] | 1.19 [1.12, 1.26] | 1.34 [1.26, 1.45] |
| NonlinearDriver | nonlin=1 | 1.35 [1.27, 1.45] | 1.29 [1.23, 1.37] | 6.73 [6.32, 7.18] | 1.29 [1.23, 1.37] | 1.46 [1.37, 1.58] |
| FeedbackLinear | coupling=0 | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.00] | - | 1.00 [0.99, 1.00] | 0.99 [0.98, 1.00] |
| FeedbackLinear | coupling=0.05 | 1.00 [1.00, 1.01] | 1.00 [1.00, 1.01] | - | 1.00 [1.00, 1.01] | 1.00 [0.99, 1.00] |
| FeedbackLinear | coupling=0.15 | 1.00 [1.00, 1.01] | 1.00 [0.99, 1.01] | - | 1.00 [0.99, 1.01] | 0.99 [0.98, 1.01] |
| FeedbackLinear | coupling=0.4 | 1.00 [0.99, 1.00] | 0.99 [0.99, 1.00] | - | 1.00 [0.99, 1.00] | 0.99 [0.98, 1.00] |
| FeedbackLinear | coupling=1 | 1.08 [0.99, 1.20] | 1.08 [0.98, 1.20] | - | 1.08 [0.98, 1.20] | 1.02 [0.96, 1.07] |
| FeedbackPredPrey | coupling=0 | 1.28 [1.08, 1.53] | 1.24 [1.08, 1.44] | - | 1.25 [1.08, 1.45] | 0.99 [0.81, 1.21] |
| FeedbackPredPrey | coupling=0.25 | 1.17 [1.05, 1.34] | 1.15 [1.04, 1.31] | - | 1.15 [1.04, 1.30] | 1.15 [0.98, 1.37] |
| FeedbackPredPrey | coupling=0.5 | 1.00 [0.90, 1.11] | 0.97 [0.90, 1.05] | - | 0.97 [0.90, 1.05] | 1.11 [0.95, 1.28] |
| FeedbackPredPrey | coupling=1 | 1.01 [0.93, 1.10] | 1.00 [0.92, 1.08] | - | 1.00 [0.92, 1.09] | 1.03 [1.01, 1.06] |
| FeedbackPredPrey | coupling=2 | 1.07 [1.00, 1.16] | 1.06 [1.00, 1.12] | - | 1.07 [0.99, 1.14] | 1.12 [0.98, 1.32] |
| HiddenDriver | x_noise=0.05 | 1.33 [1.17, 1.52] | 1.29 [1.14, 1.47] | - | 1.28 [1.14, 1.47] | 1.37 [1.13, 1.62] |
| HiddenDriver | x_noise=0.25 | 1.33 [1.22, 1.48] | 1.32 [1.20, 1.47] | - | 1.31 [1.20, 1.47] | 1.39 [1.22, 1.61] |
| HiddenDriver | x_noise=0.5 | 1.27 [1.14, 1.41] | 1.26 [1.14, 1.38] | - | 1.25 [1.14, 1.37] | 1.36 [1.13, 1.60] |
| HiddenDriver | x_noise=1 | 1.27 [1.16, 1.38] | 1.26 [1.16, 1.38] | - | 1.26 [1.16, 1.38] | 1.29 [1.08, 1.53] |
| HiddenDriver | x_noise=2 | 1.50 [1.30, 1.76] | 1.51 [1.31, 1.75] | - | 1.49 [1.30, 1.74] | 1.64 [1.39, 1.96] |
| LeakyKnown | noise_ratio=0.05 | 1.05 [1.02, 1.08] | 1.05 [1.02, 1.08] | 20.92 [17.57, 26.16] | 1.05 [1.02, 1.08] | 1.08 [1.00, 1.21] |
| LeakyKnown | noise_ratio=0.25 | 1.03 [1.00, 1.06] | 1.03 [1.00, 1.06] | 4.47 [4.17, 4.77] | 1.03 [0.99, 1.06] | 1.10 [1.02, 1.19] |
| LeakyKnown | noise_ratio=1 | 1.03 [0.98, 1.10] | 1.03 [0.98, 1.10] | 1.63 [1.56, 1.70] | 1.03 [0.98, 1.10] | 1.17 [1.06, 1.31] |
| LeakyKnown | noise_ratio=4 | 1.05 [0.98, 1.13] | 1.05 [0.98, 1.13] | 1.20 [1.15, 1.25] | 1.05 [0.98, 1.14] | 1.03 [0.95, 1.12] |
| LeakyKnown | noise_ratio=16 | 1.05 [1.01, 1.09] | 1.04 [1.01, 1.09] | 1.02 [0.92, 1.12] | 1.04 [1.01, 1.09] | 1.19 [1.05, 1.33] |

### Size 5M, H = 1024

| system | knob | uni | past_cov | known_cov | joint | classical |
|---|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.08 [1.05, 1.10] | 1.07 [1.05, 1.10] | 7.80 [7.14, 8.57] | 1.07 [1.05, 1.10] | 1.17 [1.07, 1.27] |
| LaggedDriver | lag=8 | 1.06 [1.05, 1.06] | 1.04 [1.03, 1.05] | 7.88 [7.49, 8.25] | 1.04 [1.03, 1.05] | 1.12 [1.02, 1.24] |
| LaggedDriver | lag=32 | 1.08 [1.07, 1.09] | 1.07 [1.06, 1.09] | 7.74 [7.14, 8.39] | 1.07 [1.06, 1.08] | 1.20 [1.11, 1.31] |
| LaggedDriver | lag=128 | 1.18 [1.15, 1.23] | 1.18 [1.14, 1.23] | 8.46 [7.95, 9.03] | 1.18 [1.14, 1.23] | 1.29 [1.22, 1.37] |
| LaggedDriver | lag=512 | 1.92 [1.81, 2.04] | 1.91 [1.81, 2.02] | 8.41 [7.98, 8.99] | 1.91 [1.80, 2.02] | 2.08 [1.90, 2.27] |
| NonlinearDriver | nonlin=0 | 1.09 [1.07, 1.11] | 1.08 [1.06, 1.09] | 7.93 [7.62, 8.22] | 1.08 [1.06, 1.09] | 1.37 [1.21, 1.60] |
| NonlinearDriver | nonlin=0.25 | 1.11 [1.08, 1.13] | 1.09 [1.07, 1.11] | 6.55 [6.15, 6.90] | 1.09 [1.07, 1.11] | 1.22 [1.15, 1.30] |
| NonlinearDriver | nonlin=0.5 | 1.09 [1.06, 1.13] | 1.09 [1.06, 1.12] | 5.92 [5.64, 6.23] | 1.09 [1.06, 1.12] | 1.28 [1.15, 1.44] |
| NonlinearDriver | nonlin=0.75 | 1.05 [1.02, 1.09] | 1.04 [1.01, 1.07] | 6.43 [6.18, 6.65] | 1.04 [1.01, 1.07] | 1.15 [1.11, 1.19] |
| NonlinearDriver | nonlin=1 | 1.07 [1.04, 1.11] | 1.05 [1.03, 1.08] | 7.48 [7.28, 7.75] | 1.05 [1.03, 1.08] | 1.09 [1.06, 1.13] |
| FeedbackLinear | coupling=0 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=0.05 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=0.15 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=0.4 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=1 | 0.99 [0.98, 1.01] | 1.00 [0.98, 1.01] | - | 0.99 [0.98, 1.01] | 0.99 [0.98, 1.01] |
| FeedbackPredPrey | coupling=0 | 1.10 [1.00, 1.20] | 1.07 [0.99, 1.16] | - | 1.08 [0.99, 1.17] | 1.04 [0.96, 1.18] |
| FeedbackPredPrey | coupling=0.25 | 1.04 [1.00, 1.08] | 1.03 [1.00, 1.07] | - | 1.03 [1.00, 1.07] | 1.00 [0.98, 1.02] |
| FeedbackPredPrey | coupling=0.5 | 1.03 [1.01, 1.06] | 1.02 [1.00, 1.05] | - | 1.02 [1.00, 1.05] | 1.03 [1.00, 1.07] |
| FeedbackPredPrey | coupling=1 | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] | - | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] |
| FeedbackPredPrey | coupling=2 | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] | - | 1.01 [1.00, 1.02] | 1.02 [1.00, 1.04] |
| HiddenDriver | x_noise=0.05 | 1.10 [1.07, 1.12] | 1.09 [1.07, 1.12] | - | 1.09 [1.07, 1.12] | 1.16 [1.08, 1.25] |
| HiddenDriver | x_noise=0.25 | 1.08 [1.04, 1.13] | 1.09 [1.04, 1.14] | - | 1.09 [1.04, 1.14] | 1.16 [1.06, 1.28] |
| HiddenDriver | x_noise=0.5 | 1.09 [1.06, 1.14] | 1.09 [1.05, 1.15] | - | 1.09 [1.05, 1.15] | 1.16 [1.07, 1.25] |
| HiddenDriver | x_noise=1 | 1.11 [1.08, 1.16] | 1.11 [1.08, 1.16] | - | 1.11 [1.08, 1.16] | 1.22 [1.10, 1.36] |
| HiddenDriver | x_noise=2 | 1.15 [1.07, 1.24] | 1.15 [1.07, 1.24] | - | 1.14 [1.07, 1.24] | 1.25 [1.15, 1.37] |
| LeakyKnown | noise_ratio=0.05 | 1.07 [1.04, 1.10] | 1.06 [1.04, 1.09] | 42.71 [41.39, 44.38] | 1.06 [1.04, 1.10] | 1.15 [1.07, 1.23] |
| LeakyKnown | noise_ratio=0.25 | 1.06 [1.04, 1.09] | 1.06 [1.04, 1.08] | 9.66 [9.38, 9.97] | 1.06 [1.04, 1.08] | 1.16 [1.06, 1.28] |
| LeakyKnown | noise_ratio=1 | 1.07 [1.04, 1.09] | 1.07 [1.04, 1.09] | 3.24 [3.13, 3.36] | 1.07 [1.04, 1.09] | 1.27 [1.17, 1.37] |
| LeakyKnown | noise_ratio=4 | 1.06 [1.03, 1.09] | 1.06 [1.03, 1.09] | 1.63 [1.56, 1.73] | 1.06 [1.03, 1.09] | 1.13 [1.05, 1.23] |
| LeakyKnown | noise_ratio=16 | 1.05 [1.04, 1.07] | 1.05 [1.03, 1.07] | 1.14 [1.11, 1.18] | 1.05 [1.03, 1.07] | 1.24 [1.16, 1.33] |

### Size 28M, H = 16

| system | knob | uni | past_cov | known_cov | joint | classical |
|---|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.02 [0.92, 1.13] | 1.03 [0.93, 1.14] | 1.28 [1.09, 1.59] | 1.03 [0.93, 1.14] | 1.03 [0.80, 1.32] |
| LaggedDriver | lag=8 | 1.77 [1.43, 2.25] | 1.18 [1.03, 1.37] | 2.31 [1.98, 2.76] | 1.14 [0.99, 1.33] | 1.45 [0.98, 2.04] |
| LaggedDriver | lag=32 | 7.86 [5.54, 10.84] | 4.72 [4.00, 5.43] | 5.10 [4.41, 5.79] | 4.24 [3.67, 4.79] | 7.97 [5.82, 10.56] |
| LaggedDriver | lag=128 | 7.79 [6.45, 9.61] | 7.74 [6.41, 9.58] | 7.69 [6.32, 9.60] | 7.69 [6.38, 9.55] | 7.11 [5.71, 8.88] |
| LaggedDriver | lag=512 | 6.18 [5.13, 7.46] | 6.32 [5.26, 7.55] | 6.44 [5.36, 7.72] | 6.31 [5.25, 7.53] | 5.31 [4.15, 7.06] |
| NonlinearDriver | nonlin=0 | 7.36 [5.99, 8.99] | 4.82 [3.92, 5.89] | 5.32 [4.41, 6.35] | 4.45 [3.72, 5.43] | 7.18 [5.59, 9.16] |
| NonlinearDriver | nonlin=0.25 | 6.16 [5.03, 7.41] | 3.37 [2.66, 4.16] | 3.68 [2.97, 4.59] | 3.22 [2.54, 4.00] | 5.60 [4.17, 7.34] |
| NonlinearDriver | nonlin=0.5 | 6.48 [4.44, 9.43] | 4.00 [3.03, 5.66] | 4.41 [3.29, 6.33] | 3.95 [3.02, 5.54] | 5.75 [4.10, 7.87] |
| NonlinearDriver | nonlin=0.75 | 5.81 [5.10, 6.79] | 4.53 [3.52, 5.69] | 5.00 [4.01, 6.13] | 4.45 [3.46, 5.58] | 5.28 [4.70, 5.96] |
| NonlinearDriver | nonlin=1 | 6.28 [5.34, 7.36] | 5.25 [4.28, 6.48] | 5.65 [4.57, 7.05] | 5.15 [4.20, 6.36] | 6.48 [5.22, 7.91] |
| FeedbackLinear | coupling=0 | 1.02 [1.00, 1.04] | 1.02 [1.00, 1.05] | - | 1.02 [1.00, 1.05] | 1.00 [0.98, 1.03] |
| FeedbackLinear | coupling=0.05 | 1.00 [0.98, 1.02] | 1.00 [0.97, 1.02] | - | 0.99 [0.97, 1.02] | 1.00 [0.98, 1.02] |
| FeedbackLinear | coupling=0.15 | 1.04 [1.01, 1.09] | 1.00 [0.97, 1.02] | - | 1.00 [0.97, 1.02] | 0.98 [0.92, 1.03] |
| FeedbackLinear | coupling=0.4 | 1.00 [0.96, 1.04] | 0.98 [0.96, 1.01] | - | 0.98 [0.97, 1.00] | 0.96 [0.92, 1.00] |
| FeedbackLinear | coupling=1 | 1.24 [1.02, 1.53] | 1.21 [1.04, 1.45] | - | 1.22 [1.04, 1.46] | 1.14 [1.00, 1.31] |
| FeedbackPredPrey | coupling=0 | 1.16 [1.05, 1.32] | 1.14 [1.05, 1.31] | - | 1.14 [1.03, 1.30] | 1.05 [0.96, 1.16] |
| FeedbackPredPrey | coupling=0.25 | 0.94 [0.83, 1.06] | 0.95 [0.85, 1.07] | - | 0.95 [0.85, 1.07] | 0.90 [0.77, 1.05] |
| FeedbackPredPrey | coupling=0.5 | 0.90 [0.70, 1.19] | 0.94 [0.73, 1.24] | - | 0.95 [0.75, 1.20] | 0.69 [0.51, 0.93] |
| FeedbackPredPrey | coupling=1 | 1.19 [0.88, 1.53] | 1.06 [0.87, 1.30] | - | 1.10 [1.00, 1.20] | 0.96 [0.69, 1.25] |
| FeedbackPredPrey | coupling=2 | 1.04 [0.84, 1.31] | 0.99 [0.90, 1.13] | - | 0.94 [0.88, 1.02] | 1.18 [0.68, 1.76] |
| HiddenDriver | x_noise=0.05 | 5.31 [3.55, 7.89] | 4.42 [3.04, 6.53] | - | 3.85 [2.72, 5.46] | 6.05 [4.13, 8.64] |
| HiddenDriver | x_noise=0.25 | 6.02 [4.65, 7.67] | 4.66 [3.36, 6.39] | - | 4.48 [3.18, 6.21] | 5.64 [4.51, 7.35] |
| HiddenDriver | x_noise=0.5 | 4.35 [2.97, 5.79] | 4.14 [2.90, 5.39] | - | 4.02 [2.87, 5.16] | 3.66 [2.71, 4.62] |
| HiddenDriver | x_noise=1 | 3.87 [2.84, 5.22] | 3.80 [2.78, 5.23] | - | 3.76 [2.74, 5.25] | 4.36 [3.27, 5.69] |
| HiddenDriver | x_noise=2 | 5.28 [3.44, 7.87] | 5.20 [3.39, 7.75] | - | 5.15 [3.40, 7.63] | 5.44 [4.02, 7.75] |
| LeakyKnown | noise_ratio=0.05 | 1.01 [0.86, 1.14] | 1.01 [0.86, 1.13] | 3.81 [3.06, 4.63] | 1.02 [0.88, 1.14] | 0.90 [0.78, 1.03] |
| LeakyKnown | noise_ratio=0.25 | 1.04 [0.93, 1.15] | 1.03 [0.94, 1.14] | 1.72 [1.41, 2.16] | 1.03 [0.93, 1.14] | 0.98 [0.79, 1.27] |
| LeakyKnown | noise_ratio=1 | 1.13 [1.01, 1.30] | 1.14 [1.01, 1.28] | 1.06 [0.94, 1.19] | 1.12 [1.00, 1.27] | 1.05 [0.83, 1.36] |
| LeakyKnown | noise_ratio=4 | 1.02 [0.92, 1.13] | 1.02 [0.93, 1.11] | 0.97 [0.84, 1.10] | 1.02 [0.93, 1.12] | 0.77 [0.62, 0.91] |
| LeakyKnown | noise_ratio=16 | 0.97 [0.85, 1.10] | 0.96 [0.84, 1.10] | 0.96 [0.82, 1.11] | 0.96 [0.84, 1.09] | 0.90 [0.64, 1.27] |

### Size 28M, H = 128

| system | knob | uni | past_cov | known_cov | joint | classical |
|---|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.00 [0.96, 1.04] | 1.00 [0.96, 1.04] | 1.36 [1.30, 1.43] | 1.00 [0.96, 1.04] | 1.06 [0.96, 1.23] |
| LaggedDriver | lag=8 | 1.02 [0.95, 1.07] | 0.97 [0.91, 1.03] | 2.53 [2.30, 2.76] | 0.97 [0.91, 1.03] | 1.05 [0.89, 1.22] |
| LaggedDriver | lag=32 | 1.26 [1.15, 1.39] | 1.21 [1.14, 1.29] | 5.11 [4.28, 6.22] | 1.19 [1.13, 1.26] | 1.44 [1.27, 1.64] |
| LaggedDriver | lag=128 | 7.91 [7.35, 8.62] | 7.91 [7.38, 8.58] | 7.83 [7.31, 8.50] | 7.88 [7.35, 8.56] | 8.49 [7.49, 9.78] |
| LaggedDriver | lag=512 | 8.23 [7.24, 9.59] | 8.23 [7.27, 9.40] | 8.10 [7.13, 9.28] | 8.23 [7.26, 9.40] | 8.57 [7.22, 10.46] |
| NonlinearDriver | nonlin=0 | 1.31 [1.20, 1.42] | 1.23 [1.12, 1.35] | 4.90 [4.13, 5.85] | 1.22 [1.11, 1.34] | 1.51 [1.34, 1.71] |
| NonlinearDriver | nonlin=0.25 | 1.34 [1.27, 1.41] | 1.12 [1.05, 1.21] | 3.62 [3.21, 4.10] | 1.10 [1.01, 1.18] | 1.36 [1.12, 1.59] |
| NonlinearDriver | nonlin=0.5 | 1.31 [1.23, 1.42] | 1.23 [1.17, 1.32] | 3.68 [3.29, 4.15] | 1.22 [1.16, 1.31] | 1.38 [1.25, 1.53] |
| NonlinearDriver | nonlin=0.75 | 1.22 [1.15, 1.30] | 1.19 [1.13, 1.26] | 4.82 [4.50, 5.16] | 1.19 [1.14, 1.27] | 1.34 [1.26, 1.45] |
| NonlinearDriver | nonlin=1 | 1.32 [1.25, 1.40] | 1.21 [1.16, 1.27] | 6.33 [5.66, 6.97] | 1.20 [1.15, 1.25] | 1.46 [1.37, 1.58] |
| FeedbackLinear | coupling=0 | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] | - | 1.00 [0.99, 1.01] | 0.99 [0.98, 1.00] |
| FeedbackLinear | coupling=0.05 | 1.00 [0.99, 1.00] | 1.00 [0.99, 1.00] | - | 1.00 [0.99, 1.00] | 1.00 [0.99, 1.00] |
| FeedbackLinear | coupling=0.15 | 1.01 [1.00, 1.02] | 1.00 [1.00, 1.01] | - | 1.00 [1.00, 1.01] | 0.99 [0.98, 1.01] |
| FeedbackLinear | coupling=0.4 | 1.00 [0.99, 1.00] | 1.00 [0.99, 1.00] | - | 1.00 [0.99, 1.00] | 0.99 [0.98, 1.00] |
| FeedbackLinear | coupling=1 | 1.03 [0.90, 1.15] | 1.04 [0.92, 1.18] | - | 1.05 [0.92, 1.18] | 1.02 [0.96, 1.07] |
| FeedbackPredPrey | coupling=0 | 1.19 [0.97, 1.50] | 1.18 [0.98, 1.46] | - | 1.17 [0.98, 1.45] | 0.99 [0.81, 1.21] |
| FeedbackPredPrey | coupling=0.25 | 1.15 [1.06, 1.31] | 1.11 [1.02, 1.26] | - | 1.10 [1.02, 1.25] | 1.15 [0.98, 1.37] |
| FeedbackPredPrey | coupling=0.5 | 1.02 [0.89, 1.17] | 1.02 [0.90, 1.15] | - | 1.01 [0.90, 1.14] | 1.11 [0.95, 1.28] |
| FeedbackPredPrey | coupling=1 | 1.02 [0.94, 1.12] | 1.01 [0.95, 1.08] | - | 1.01 [0.96, 1.07] | 1.03 [1.01, 1.06] |
| FeedbackPredPrey | coupling=2 | 1.08 [1.02, 1.15] | 1.06 [0.98, 1.15] | - | 1.03 [0.97, 1.10] | 1.12 [0.98, 1.32] |
| HiddenDriver | x_noise=0.05 | 1.34 [1.16, 1.54] | 1.16 [1.05, 1.30] | - | 1.13 [1.02, 1.27] | 1.37 [1.13, 1.62] |
| HiddenDriver | x_noise=0.25 | 1.31 [1.21, 1.42] | 1.24 [1.14, 1.36] | - | 1.20 [1.09, 1.34] | 1.39 [1.22, 1.61] |
| HiddenDriver | x_noise=0.5 | 1.31 [1.19, 1.45] | 1.23 [1.12, 1.33] | - | 1.17 [1.09, 1.27] | 1.36 [1.13, 1.60] |
| HiddenDriver | x_noise=1 | 1.26 [1.13, 1.40] | 1.25 [1.12, 1.38] | - | 1.24 [1.11, 1.38] | 1.29 [1.08, 1.53] |
| HiddenDriver | x_noise=2 | 1.49 [1.26, 1.79] | 1.49 [1.27, 1.79] | - | 1.48 [1.26, 1.78] | 1.64 [1.39, 1.96] |
| LeakyKnown | noise_ratio=0.05 | 1.03 [1.00, 1.07] | 1.03 [0.99, 1.07] | 9.94 [8.93, 11.01] | 1.03 [1.00, 1.07] | 1.08 [1.00, 1.21] |
| LeakyKnown | noise_ratio=0.25 | 1.01 [0.99, 1.04] | 1.01 [0.98, 1.04] | 2.46 [2.25, 2.67] | 1.01 [0.98, 1.04] | 1.10 [1.02, 1.19] |
| LeakyKnown | noise_ratio=1 | 1.03 [0.96, 1.12] | 1.02 [0.95, 1.09] | 1.18 [1.13, 1.24] | 1.03 [0.96, 1.10] | 1.17 [1.06, 1.31] |
| LeakyKnown | noise_ratio=4 | 1.01 [0.95, 1.08] | 1.01 [0.95, 1.07] | 1.08 [1.05, 1.12] | 1.01 [0.95, 1.08] | 1.03 [0.95, 1.12] |
| LeakyKnown | noise_ratio=16 | 1.03 [0.99, 1.07] | 1.03 [0.99, 1.07] | 1.00 [0.91, 1.09] | 1.03 [0.99, 1.07] | 1.19 [1.05, 1.33] |

### Size 28M, H = 1024

| system | knob | uni | past_cov | known_cov | joint | classical |
|---|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.05 [1.02, 1.09] | 1.05 [1.02, 1.09] | 6.40 [5.81, 7.13] | 1.05 [1.02, 1.09] | 1.17 [1.07, 1.27] |
| LaggedDriver | lag=8 | 1.04 [1.02, 1.06] | 1.03 [1.01, 1.05] | 7.08 [6.67, 7.53] | 1.03 [1.01, 1.05] | 1.12 [1.02, 1.24] |
| LaggedDriver | lag=32 | 1.04 [1.03, 1.06] | 1.04 [1.02, 1.05] | 6.95 [6.44, 7.52] | 1.04 [1.02, 1.06] | 1.20 [1.11, 1.31] |
| LaggedDriver | lag=128 | 1.16 [1.13, 1.20] | 1.16 [1.13, 1.19] | 8.16 [7.70, 8.63] | 1.16 [1.13, 1.19] | 1.29 [1.22, 1.37] |
| LaggedDriver | lag=512 | 1.90 [1.79, 2.03] | 1.88 [1.78, 2.00] | 8.18 [7.70, 8.78] | 1.88 [1.78, 2.00] | 2.08 [1.90, 2.27] |
| NonlinearDriver | nonlin=0 | 1.08 [1.05, 1.12] | 1.07 [1.05, 1.09] | 7.18 [6.84, 7.49] | 1.06 [1.05, 1.07] | 1.37 [1.21, 1.60] |
| NonlinearDriver | nonlin=0.25 | 1.07 [1.06, 1.09] | 1.06 [1.04, 1.08] | 5.84 [5.57, 6.10] | 1.06 [1.04, 1.08] | 1.22 [1.15, 1.30] |
| NonlinearDriver | nonlin=0.5 | 1.06 [1.04, 1.08] | 1.06 [1.04, 1.09] | 5.13 [4.89, 5.41] | 1.06 [1.04, 1.10] | 1.28 [1.15, 1.44] |
| NonlinearDriver | nonlin=0.75 | 1.03 [1.01, 1.05] | 1.02 [1.01, 1.04] | 6.10 [5.91, 6.30] | 1.02 [1.01, 1.04] | 1.15 [1.11, 1.19] |
| NonlinearDriver | nonlin=1 | 1.04 [1.02, 1.07] | 1.03 [1.01, 1.05] | 7.29 [7.04, 7.58] | 1.03 [1.01, 1.05] | 1.09 [1.06, 1.13] |
| FeedbackLinear | coupling=0 | 1.00 [1.00, 1.01] | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=0.05 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=0.15 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=0.4 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| FeedbackLinear | coupling=1 | 0.99 [0.98, 1.01] | 0.99 [0.98, 1.01] | - | 0.99 [0.98, 1.01] | 0.99 [0.98, 1.01] |
| FeedbackPredPrey | coupling=0 | 1.05 [0.98, 1.12] | 1.04 [0.97, 1.11] | - | 1.04 [0.97, 1.10] | 1.04 [0.96, 1.18] |
| FeedbackPredPrey | coupling=0.25 | 1.05 [1.02, 1.08] | 1.03 [1.00, 1.06] | - | 1.03 [1.00, 1.06] | 1.00 [0.98, 1.02] |
| FeedbackPredPrey | coupling=0.5 | 1.03 [1.00, 1.06] | 1.02 [1.00, 1.05] | - | 1.02 [1.00, 1.05] | 1.03 [1.00, 1.07] |
| FeedbackPredPrey | coupling=1 | 1.01 [0.99, 1.04] | 1.00 [0.99, 1.02] | - | 1.00 [0.99, 1.02] | 1.01 [1.00, 1.02] |
| FeedbackPredPrey | coupling=2 | 1.01 [1.00, 1.02] | 1.00 [0.99, 1.02] | - | 1.00 [0.99, 1.01] | 1.02 [1.00, 1.04] |
| HiddenDriver | x_noise=0.05 | 1.08 [1.06, 1.11] | 1.07 [1.04, 1.10] | - | 1.06 [1.02, 1.09] | 1.16 [1.08, 1.25] |
| HiddenDriver | x_noise=0.25 | 1.06 [1.02, 1.10] | 1.05 [1.00, 1.10] | - | 1.04 [0.99, 1.09] | 1.16 [1.06, 1.28] |
| HiddenDriver | x_noise=0.5 | 1.09 [1.04, 1.18] | 1.09 [1.03, 1.17] | - | 1.07 [1.03, 1.14] | 1.16 [1.07, 1.25] |
| HiddenDriver | x_noise=1 | 1.10 [1.07, 1.14] | 1.10 [1.06, 1.13] | - | 1.09 [1.06, 1.13] | 1.22 [1.10, 1.36] |
| HiddenDriver | x_noise=2 | 1.13 [1.04, 1.25] | 1.13 [1.05, 1.25] | - | 1.13 [1.04, 1.26] | 1.25 [1.15, 1.37] |
| LeakyKnown | noise_ratio=0.05 | 1.03 [1.01, 1.06] | 1.03 [1.01, 1.05] | 38.41 [36.26, 40.77] | 1.03 [1.01, 1.05] | 1.15 [1.07, 1.23] |
| LeakyKnown | noise_ratio=0.25 | 1.05 [1.02, 1.09] | 1.04 [1.01, 1.08] | 8.95 [8.39, 9.51] | 1.04 [1.01, 1.08] | 1.16 [1.06, 1.28] |
| LeakyKnown | noise_ratio=1 | 1.03 [1.01, 1.05] | 1.03 [1.01, 1.05] | 2.95 [2.82, 3.12] | 1.03 [1.01, 1.05] | 1.27 [1.17, 1.37] |
| LeakyKnown | noise_ratio=4 | 1.03 [1.00, 1.07] | 1.03 [1.00, 1.07] | 1.49 [1.41, 1.58] | 1.03 [1.00, 1.07] | 1.13 [1.05, 1.23] |
| LeakyKnown | noise_ratio=16 | 1.03 [1.02, 1.05] | 1.03 [1.02, 1.05] | 1.11 [1.07, 1.15] | 1.03 [1.01, 1.05] | 1.24 [1.16, 1.33] |

### Size 123M, H = 16

| system | knob | uni | past_cov | known_cov | joint | classical |
|---|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 0.92 [0.87, 0.95] | 0.92 [0.87, 0.95] | 1.17 [1.11, 1.23] | 0.92 [0.87, 0.96] | 1.03 [0.80, 1.32] |
| LaggedDriver | lag=32 | 7.92 [4.49, 11.86] | 3.77 [3.09, 4.45] | 5.42 [3.94, 7.46] | 3.21 [2.80, 3.68] | 7.97 [5.82, 10.56] |
| LaggedDriver | lag=512 | 5.76 [4.36, 7.61] | 6.00 [4.57, 7.86] | 5.97 [4.68, 7.62] | 5.98 [4.57, 7.83] | 5.31 [4.15, 7.06] |
| NonlinearDriver | nonlin=0 | 6.44 [4.71, 8.41] | 3.67 [3.21, 4.17] | 5.89 [3.88, 8.94] | 2.83 [1.94, 3.81] | 7.18 [5.59, 9.16] |
| NonlinearDriver | nonlin=0.5 | 5.23 [3.25, 10.01] | 2.88 [1.85, 5.09] | 3.80 [2.34, 8.12] | 2.46 [1.71, 3.87] | 5.75 [4.10, 7.87] |
| NonlinearDriver | nonlin=1 | 5.02 [4.28, 6.28] | 2.95 [2.40, 3.96] | 3.35 [2.55, 4.39] | 2.92 [2.35, 3.98] | 6.48 [5.22, 7.91] |
| FeedbackLinear | coupling=0 | 1.00 [0.98, 1.02] | 1.00 [0.98, 1.02] | - | 1.00 [0.98, 1.02] | 1.00 [0.98, 1.03] |
| FeedbackLinear | coupling=0.15 | 1.05 [1.00, 1.14] | 0.97 [0.91, 1.03] | - | 0.97 [0.91, 1.03] | 0.98 [0.92, 1.03] |
| FeedbackLinear | coupling=1 | 1.20 [1.03, 1.55] | 1.16 [1.04, 1.30] | - | 1.16 [1.05, 1.27] | 1.14 [1.00, 1.31] |
| FeedbackPredPrey | coupling=0 | 1.17 [1.06, 1.33] | 1.18 [1.07, 1.34] | - | 1.19 [1.08, 1.37] | 1.05 [0.96, 1.16] |
| FeedbackPredPrey | coupling=0.5 | 0.96 [0.60, 1.60] | 0.97 [0.62, 1.61] | - | 0.94 [0.63, 1.41] | 0.69 [0.51, 0.93] |
| FeedbackPredPrey | coupling=2 | 1.26 [0.95, 1.69] | 1.10 [0.99, 1.21] | - | 0.96 [0.90, 1.05] | 1.18 [0.68, 1.76] |
| HiddenDriver | x_noise=0.05 | 4.35 [2.40, 7.90] | 2.45 [1.99, 2.93] | - | 2.31 [2.00, 2.62] | 6.05 [4.13, 8.64] |
| HiddenDriver | x_noise=0.5 | 3.50 [1.95, 5.04] | 3.50 [1.96, 5.04] | - | 3.59 [2.00, 5.15] | 3.66 [2.71, 4.62] |
| HiddenDriver | x_noise=2 | 5.74 [3.66, 11.12] | 5.78 [3.67, 11.08] | - | 5.76 [3.57, 11.10] | 5.44 [4.02, 7.75] |
| LeakyKnown | noise_ratio=0.05 | 1.05 [0.96, 1.14] | 1.05 [0.96, 1.15] | 3.04 [2.29, 4.04] | 1.07 [0.98, 1.18] | 0.90 [0.78, 1.03] |
| LeakyKnown | noise_ratio=1 | 1.15 [0.94, 1.42] | 1.12 [0.94, 1.34] | 1.02 [0.82, 1.24] | 1.11 [0.93, 1.29] | 1.05 [0.83, 1.36] |
| LeakyKnown | noise_ratio=16 | 1.00 [0.94, 1.06] | 1.00 [0.94, 1.06] | 1.01 [0.96, 1.10] | 1.00 [0.95, 1.05] | 0.90 [0.64, 1.27] |

### Size 123M, H = 128

| system | knob | uni | past_cov | known_cov | joint | classical |
|---|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.03 [1.02, 1.05] | 1.03 [1.02, 1.05] | 1.08 [1.00, 1.15] | 1.03 [1.02, 1.05] | 1.06 [0.96, 1.23] |
| LaggedDriver | lag=32 | 1.38 [1.24, 1.54] | 1.20 [1.15, 1.26] | 4.55 [3.19, 6.23] | 1.19 [1.14, 1.24] | 1.44 [1.27, 1.64] |
| LaggedDriver | lag=512 | 8.16 [7.43, 9.64] | 8.24 [7.48, 9.91] | 8.02 [7.15, 9.67] | 8.26 [7.50, 9.90] | 8.57 [7.22, 10.46] |
| NonlinearDriver | nonlin=0 | 1.39 [1.31, 1.52] | 1.22 [1.17, 1.27] | 5.47 [3.51, 8.53] | 1.18 [1.11, 1.23] | 1.51 [1.34, 1.71] |
| NonlinearDriver | nonlin=0.5 | 1.20 [1.15, 1.27] | 1.15 [1.08, 1.27] | 2.82 [2.39, 3.34] | 1.14 [1.08, 1.25] | 1.38 [1.25, 1.53] |
| NonlinearDriver | nonlin=1 | 1.35 [1.24, 1.49] | 1.15 [1.06, 1.25] | 4.62 [3.81, 5.59] | 1.15 [1.05, 1.27] | 1.46 [1.37, 1.58] |
| FeedbackLinear | coupling=0 | 1.00 [0.98, 1.01] | 1.00 [0.98, 1.01] | - | 1.00 [0.98, 1.01] | 0.99 [0.98, 1.00] |
| FeedbackLinear | coupling=0.15 | 1.00 [0.99, 1.02] | 0.99 [0.98, 1.01] | - | 0.99 [0.98, 1.01] | 0.99 [0.98, 1.01] |
| FeedbackLinear | coupling=1 | 1.01 [0.88, 1.17] | 0.98 [0.86, 1.12] | - | 0.99 [0.85, 1.14] | 1.02 [0.96, 1.07] |
| FeedbackPredPrey | coupling=0 | 1.29 [1.01, 1.63] | 1.28 [0.98, 1.65] | - | 1.29 [0.97, 1.68] | 0.99 [0.81, 1.21] |
| FeedbackPredPrey | coupling=0.5 | 1.00 [0.76, 1.17] | 1.00 [0.85, 1.14] | - | 0.96 [0.86, 1.08] | 1.11 [0.95, 1.28] |
| FeedbackPredPrey | coupling=2 | 1.08 [1.01, 1.16] | 1.05 [0.98, 1.13] | - | 1.03 [0.98, 1.09] | 1.12 [0.98, 1.32] |
| HiddenDriver | x_noise=0.05 | 1.32 [1.12, 1.61] | 1.13 [1.09, 1.15] | - | 1.14 [1.10, 1.22] | 1.37 [1.13, 1.62] |
| HiddenDriver | x_noise=0.5 | 1.16 [1.05, 1.27] | 1.16 [1.05, 1.27] | - | 1.14 [1.02, 1.25] | 1.36 [1.13, 1.60] |
| HiddenDriver | x_noise=2 | 1.52 [1.21, 2.17] | 1.53 [1.22, 2.15] | - | 1.52 [1.21, 2.14] | 1.64 [1.39, 1.96] |
| LeakyKnown | noise_ratio=0.05 | 1.03 [1.01, 1.06] | 1.02 [1.00, 1.05] | 3.70 [3.60, 3.78] | 1.02 [1.00, 1.05] | 1.08 [1.00, 1.21] |
| LeakyKnown | noise_ratio=1 | 1.12 [0.98, 1.27] | 1.08 [0.97, 1.20] | 1.06 [1.05, 1.08] | 1.09 [0.98, 1.20] | 1.17 [1.06, 1.31] |
| LeakyKnown | noise_ratio=16 | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.00] | 0.90 [0.83, 0.95] | 0.99 [0.98, 1.00] | 1.19 [1.05, 1.33] |

### Size 209M, H = 16

| system | knob | uni | past_cov | known_cov | joint | classical |
|---|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 0.94 [0.86, 1.03] | 0.95 [0.86, 1.04] | 1.20 [1.08, 1.37] | 0.96 [0.87, 1.07] | 1.03 [0.80, 1.32] |
| LaggedDriver | lag=32 | 7.95 [4.49, 11.64] | 3.30 [3.05, 3.65] | 3.00 [2.42, 3.78] | 2.74 [2.54, 2.97] | 7.97 [5.82, 10.56] |
| LaggedDriver | lag=512 | 5.88 [4.40, 7.85] | 5.68 [4.27, 7.55] | 5.70 [4.26, 7.65] | 5.73 [4.34, 7.57] | 5.31 [4.15, 7.06] |
| NonlinearDriver | nonlin=0 | 6.39 [4.64, 8.66] | 4.34 [2.49, 7.55] | 3.38 [2.38, 6.12] | 3.37 [2.34, 5.36] | 7.18 [5.59, 9.16] |
| NonlinearDriver | nonlin=0.5 | 4.92 [2.99, 8.91] | 2.41 [1.93, 3.53] | 2.60 [1.78, 4.64] | 2.19 [1.75, 3.10] | 5.75 [4.10, 7.87] |
| NonlinearDriver | nonlin=1 | 4.79 [4.24, 5.47] | 2.91 [2.29, 3.76] | 2.80 [2.23, 3.78] | 2.64 [2.38, 2.93] | 6.48 [5.22, 7.91] |
| FeedbackLinear | coupling=0 | 1.00 [0.98, 1.03] | 1.00 [0.98, 1.02] | - | 1.00 [0.98, 1.02] | 1.00 [0.98, 1.03] |
| FeedbackLinear | coupling=0.15 | 1.05 [1.00, 1.15] | 0.97 [0.92, 1.00] | - | 0.96 [0.93, 1.00] | 0.98 [0.92, 1.03] |
| FeedbackLinear | coupling=1 | 1.19 [1.09, 1.35] | 1.23 [1.13, 1.41] | - | 1.21 [1.11, 1.37] | 1.14 [1.00, 1.31] |
| FeedbackPredPrey | coupling=0 | 1.16 [1.03, 1.33] | 1.22 [1.09, 1.45] | - | 1.23 [1.09, 1.46] | 1.05 [0.96, 1.16] |
| FeedbackPredPrey | coupling=0.5 | 0.94 [0.53, 1.67] | 1.01 [0.60, 1.77] | - | 1.04 [0.68, 1.66] | 0.69 [0.51, 0.93] |
| FeedbackPredPrey | coupling=2 | 1.28 [0.98, 1.68] | 1.17 [0.98, 1.39] | - | 0.95 [0.89, 1.02] | 1.18 [0.68, 1.76] |
| HiddenDriver | x_noise=0.05 | 4.35 [2.43, 7.78] | 2.45 [2.36, 2.51] | - | 2.35 [2.11, 2.51] | 6.05 [4.13, 8.64] |
| HiddenDriver | x_noise=0.5 | 3.49 [1.93, 5.10] | 3.55 [1.94, 5.14] | - | 3.64 [1.98, 5.25] | 3.66 [2.71, 4.62] |
| HiddenDriver | x_noise=2 | 5.93 [3.64, 11.41] | 5.83 [3.49, 11.36] | - | 5.89 [3.60, 11.01] | 5.44 [4.02, 7.75] |
| LeakyKnown | noise_ratio=0.05 | 1.06 [0.97, 1.18] | 1.06 [0.97, 1.16] | 3.25 [2.76, 4.01] | 1.07 [0.99, 1.16] | 0.90 [0.78, 1.03] |
| LeakyKnown | noise_ratio=1 | 1.10 [0.92, 1.31] | 1.08 [0.91, 1.29] | 1.03 [0.84, 1.27] | 1.07 [0.93, 1.24] | 1.05 [0.83, 1.36] |
| LeakyKnown | noise_ratio=16 | 0.97 [0.91, 1.05] | 0.97 [0.91, 1.04] | 0.95 [0.85, 1.08] | 0.97 [0.92, 1.04] | 0.90 [0.64, 1.27] |

### Size 209M, H = 128

| system | knob | uni | past_cov | known_cov | joint | classical |
|---|---|---|---|---|---|---|
| LaggedDriver | lag=0 | 1.02 [1.00, 1.04] | 1.02 [1.00, 1.04] | 1.06 [1.05, 1.09] | 1.02 [1.00, 1.04] | 1.06 [0.96, 1.23] |
| LaggedDriver | lag=32 | 1.31 [1.20, 1.47] | 1.09 [1.03, 1.12] | 2.67 [2.43, 2.93] | 1.09 [1.05, 1.13] | 1.44 [1.27, 1.64] |
| LaggedDriver | lag=512 | 7.68 [7.31, 8.34] | 7.95 [7.26, 9.33] | 7.74 [7.09, 9.02] | 7.95 [7.25, 9.35] | 8.57 [7.22, 10.46] |
| NonlinearDriver | nonlin=0 | 1.36 [1.25, 1.49] | 1.23 [1.10, 1.41] | 3.55 [2.48, 5.99] | 1.21 [1.10, 1.37] | 1.51 [1.34, 1.71] |
| NonlinearDriver | nonlin=0.5 | 1.20 [1.14, 1.28] | 1.12 [1.09, 1.16] | 2.48 [2.32, 2.65] | 1.10 [1.05, 1.18] | 1.38 [1.25, 1.53] |
| NonlinearDriver | nonlin=1 | 1.32 [1.23, 1.38] | 1.16 [1.10, 1.23] | 4.12 [3.25, 5.19] | 1.14 [1.09, 1.20] | 1.46 [1.37, 1.58] |
| FeedbackLinear | coupling=0 | 1.00 [0.98, 1.01] | 1.00 [0.98, 1.01] | - | 1.00 [0.98, 1.01] | 0.99 [0.98, 1.00] |
| FeedbackLinear | coupling=0.15 | 1.01 [0.99, 1.02] | 0.99 [0.99, 1.00] | - | 0.99 [0.99, 1.00] | 0.99 [0.98, 1.01] |
| FeedbackLinear | coupling=1 | 1.04 [0.96, 1.13] | 1.10 [0.98, 1.24] | - | 1.09 [0.98, 1.22] | 1.02 [0.96, 1.07] |
| FeedbackPredPrey | coupling=0 | 1.24 [0.99, 1.54] | 1.30 [0.94, 1.79] | - | 1.30 [0.92, 1.79] | 0.99 [0.81, 1.21] |
| FeedbackPredPrey | coupling=0.5 | 1.03 [0.74, 1.38] | 1.05 [0.79, 1.39] | - | 1.02 [0.86, 1.32] | 1.11 [0.95, 1.28] |
| FeedbackPredPrey | coupling=2 | 1.07 [1.01, 1.14] | 1.07 [1.01, 1.13] | - | 1.03 [0.98, 1.11] | 1.12 [0.98, 1.32] |
| HiddenDriver | x_noise=0.05 | 1.30 [1.11, 1.59] | 1.07 [1.03, 1.12] | - | 1.09 [1.02, 1.19] | 1.37 [1.13, 1.62] |
| HiddenDriver | x_noise=0.5 | 1.11 [1.03, 1.18] | 1.09 [1.01, 1.16] | - | 1.06 [0.96, 1.15] | 1.36 [1.13, 1.60] |
| HiddenDriver | x_noise=2 | 1.54 [1.19, 2.25] | 1.54 [1.20, 2.25] | - | 1.53 [1.20, 2.16] | 1.64 [1.39, 1.96] |
| LeakyKnown | noise_ratio=0.05 | 1.03 [1.00, 1.06] | 1.02 [1.00, 1.06] | 2.89 [2.66, 3.13] | 1.02 [0.99, 1.05] | 1.08 [1.00, 1.21] |
| LeakyKnown | noise_ratio=1 | 1.03 [0.95, 1.10] | 1.02 [0.94, 1.08] | 1.06 [1.03, 1.09] | 1.02 [0.94, 1.07] | 1.17 [1.06, 1.31] |
| LeakyKnown | noise_ratio=16 | 1.01 [0.98, 1.04] | 1.00 [0.98, 1.03] | 0.91 [0.82, 0.98] | 1.00 [0.98, 1.04] | 1.19 [1.05, 1.33] |

## Table 2. Joint versus past_cov on the Feedback systems (T = 2048)

`joint/past_cov` = MASE(joint) / MASE(past_cov) per sample, geometric mean [95% CI]; below 1 means two-way attention is more accurate. The two ratio columns are MASE ratios to the oracle.

### FeedbackLinear, size 5M

| knob | H | past_cov / oracle | joint / oracle | joint/past_cov | n |
|---|---|---|---|---|---|
| coupling=0 | 16 | 1.03 [1.00, 1.06] | 1.03 [1.00, 1.06] | 1.000 [1.000, 1.000] | 8 |
| coupling=0 | 128 | 1.00 [0.99, 1.00] | 1.00 [0.99, 1.00] | 1.000 [1.000, 1.000] | 8 |
| coupling=0 | 1024 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.000 [1.000, 1.000] | 8 |
| coupling=0.05 | 16 | 1.01 [0.99, 1.03] | 1.01 [0.99, 1.03] | 1.000 [0.999, 1.000] | 8 |
| coupling=0.05 | 128 | 1.00 [1.00, 1.01] | 1.00 [1.00, 1.01] | 1.000 [1.000, 1.000] | 8 |
| coupling=0.05 | 1024 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.000 [1.000, 1.000] | 8 |
| coupling=0.15 | 16 | 1.01 [0.97, 1.04] | 1.01 [0.97, 1.04] | 1.000 [0.998, 1.004] | 8 |
| coupling=0.15 | 128 | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] | 1.000 [1.000, 1.001] | 8 |
| coupling=0.15 | 1024 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.000 [1.000, 1.000] | 8 |
| coupling=0.4 | 16 | 1.00 [0.97, 1.03] | 1.00 [0.97, 1.04] | 1.006 [1.000, 1.011] | 8 |
| coupling=0.4 | 128 | 0.99 [0.99, 1.00] | 1.00 [0.99, 1.00] | 1.001 [1.000, 1.004] | 8 |
| coupling=0.4 | 1024 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.000 [1.000, 1.001] | 8 |
| coupling=1 | 16 | 1.10 [0.85, 1.42] | 1.11 [0.86, 1.42] | 1.005 [1.000, 1.010] | 8 |
| coupling=1 | 128 | 1.08 [0.98, 1.20] | 1.08 [0.98, 1.20] | 0.999 [0.997, 1.001] | 8 |
| coupling=1 | 1024 | 1.00 [0.98, 1.01] | 0.99 [0.98, 1.01] | 1.000 [0.999, 1.000] | 8 |

### FeedbackLinear, size 28M

| knob | H | past_cov / oracle | joint / oracle | joint/past_cov | n |
|---|---|---|---|---|---|
| coupling=0 | 16 | 1.02 [1.00, 1.05] | 1.02 [1.00, 1.05] | 1.000 [0.999, 1.000] | 8 |
| coupling=0 | 128 | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] | 1.000 [1.000, 1.000] | 8 |
| coupling=0 | 1024 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.000 [1.000, 1.000] | 8 |
| coupling=0.05 | 16 | 1.00 [0.97, 1.02] | 0.99 [0.97, 1.02] | 0.999 [0.998, 1.000] | 8 |
| coupling=0.05 | 128 | 1.00 [0.99, 1.00] | 1.00 [0.99, 1.00] | 1.000 [1.000, 1.000] | 8 |
| coupling=0.05 | 1024 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.000 [1.000, 1.000] | 8 |
| coupling=0.15 | 16 | 1.00 [0.97, 1.02] | 1.00 [0.97, 1.02] | 0.998 [0.995, 1.001] | 8 |
| coupling=0.15 | 128 | 1.00 [1.00, 1.01] | 1.00 [1.00, 1.01] | 1.000 [0.999, 1.002] | 8 |
| coupling=0.15 | 1024 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.000 [1.000, 1.001] | 8 |
| coupling=0.4 | 16 | 0.98 [0.96, 1.01] | 0.98 [0.97, 1.00] | 0.999 [0.990, 1.008] | 8 |
| coupling=0.4 | 128 | 1.00 [0.99, 1.00] | 1.00 [0.99, 1.00] | 0.999 [0.998, 1.001] | 8 |
| coupling=0.4 | 1024 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.000 [1.000, 1.000] | 8 |
| coupling=1 | 16 | 1.21 [1.04, 1.45] | 1.22 [1.04, 1.46] | 1.006 [0.990, 1.022] | 8 |
| coupling=1 | 128 | 1.04 [0.92, 1.18] | 1.05 [0.92, 1.18] | 1.001 [0.998, 1.004] | 8 |
| coupling=1 | 1024 | 0.99 [0.98, 1.01] | 0.99 [0.98, 1.01] | 1.000 [0.999, 1.001] | 8 |

### FeedbackLinear, size 123M

| knob | H | past_cov / oracle | joint / oracle | joint/past_cov | n |
|---|---|---|---|---|---|
| coupling=0 | 16 | 1.00 [0.98, 1.02] | 1.00 [0.98, 1.02] | 1.000 [1.000, 1.001] | 4 |
| coupling=0 | 128 | 1.00 [0.98, 1.01] | 1.00 [0.98, 1.01] | 1.000 [1.000, 1.000] | 4 |
| coupling=0.15 | 16 | 0.97 [0.91, 1.03] | 0.97 [0.91, 1.03] | 0.998 [0.993, 1.004] | 4 |
| coupling=0.15 | 128 | 0.99 [0.98, 1.01] | 0.99 [0.98, 1.01] | 0.998 [0.997, 0.999] | 4 |
| coupling=1 | 16 | 1.16 [1.04, 1.30] | 1.16 [1.05, 1.27] | 0.995 [0.978, 1.013] | 4 |
| coupling=1 | 128 | 0.98 [0.86, 1.12] | 0.99 [0.85, 1.14] | 1.004 [0.989, 1.019] | 4 |

### FeedbackLinear, size 209M

| knob | H | past_cov / oracle | joint / oracle | joint/past_cov | n |
|---|---|---|---|---|---|
| coupling=0 | 16 | 1.00 [0.98, 1.02] | 1.00 [0.98, 1.02] | 0.999 [0.997, 1.001] | 4 |
| coupling=0 | 128 | 1.00 [0.98, 1.01] | 1.00 [0.98, 1.01] | 1.000 [1.000, 1.000] | 4 |
| coupling=0.15 | 16 | 0.97 [0.92, 1.00] | 0.96 [0.93, 1.00] | 0.995 [0.985, 1.006] | 4 |
| coupling=0.15 | 128 | 0.99 [0.99, 1.00] | 0.99 [0.99, 1.00] | 0.997 [0.995, 1.000] | 4 |
| coupling=1 | 16 | 1.23 [1.13, 1.41] | 1.21 [1.11, 1.37] | 0.982 [0.973, 0.991] | 4 |
| coupling=1 | 128 | 1.10 [0.98, 1.24] | 1.09 [0.98, 1.22] | 0.990 [0.978, 0.998] | 4 |

### FeedbackPredPrey, size 5M

| knob | H | past_cov / oracle | joint / oracle | joint/past_cov | n |
|---|---|---|---|---|---|
| coupling=0 | 16 | 1.12 [1.01, 1.25] | 1.12 [1.01, 1.25] | 1.004 [1.001, 1.007] | 8 |
| coupling=0 | 128 | 1.24 [1.08, 1.44] | 1.25 [1.08, 1.45] | 1.005 [1.001, 1.009] | 8 |
| coupling=0 | 1024 | 1.07 [0.99, 1.16] | 1.08 [0.99, 1.17] | 1.002 [1.000, 1.004] | 8 |
| coupling=0.25 | 16 | 0.93 [0.85, 1.04] | 0.94 [0.86, 1.04] | 1.007 [0.990, 1.022] | 8 |
| coupling=0.25 | 128 | 1.15 [1.04, 1.31] | 1.15 [1.04, 1.30] | 0.997 [0.989, 1.005] | 8 |
| coupling=0.25 | 1024 | 1.03 [1.00, 1.07] | 1.03 [1.00, 1.07] | 1.000 [0.999, 1.001] | 8 |
| coupling=0.5 | 16 | 0.95 [0.77, 1.15] | 0.95 [0.78, 1.14] | 1.006 [0.969, 1.040] | 8 |
| coupling=0.5 | 128 | 0.97 [0.90, 1.05] | 0.97 [0.90, 1.05] | 1.005 [0.997, 1.011] | 8 |
| coupling=0.5 | 1024 | 1.02 [1.00, 1.05] | 1.02 [1.00, 1.05] | 1.000 [0.998, 1.001] | 8 |
| coupling=1 | 16 | 1.13 [0.94, 1.38] | 1.15 [0.96, 1.39] | 1.017 [0.922, 1.127] | 8 |
| coupling=1 | 128 | 1.00 [0.92, 1.08] | 1.00 [0.92, 1.09] | 0.998 [0.980, 1.019] | 8 |
| coupling=1 | 1024 | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] | 0.999 [0.993, 1.004] | 8 |
| coupling=2 | 16 | 1.11 [0.98, 1.26] | 1.11 [0.99, 1.27] | 1.002 [0.945, 1.072] | 8 |
| coupling=2 | 128 | 1.06 [1.00, 1.12] | 1.07 [0.99, 1.14] | 1.005 [0.993, 1.016] | 8 |
| coupling=2 | 1024 | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] | 1.000 [0.999, 1.002] | 8 |

### FeedbackPredPrey, size 28M

| knob | H | past_cov / oracle | joint / oracle | joint/past_cov | n |
|---|---|---|---|---|---|
| coupling=0 | 16 | 1.14 [1.05, 1.31] | 1.14 [1.03, 1.30] | 0.993 [0.981, 1.001] | 8 |
| coupling=0 | 128 | 1.18 [0.98, 1.46] | 1.17 [0.98, 1.45] | 0.997 [0.994, 1.000] | 8 |
| coupling=0 | 1024 | 1.04 [0.97, 1.11] | 1.04 [0.97, 1.10] | 0.997 [0.994, 0.999] | 8 |
| coupling=0.25 | 16 | 0.95 [0.85, 1.07] | 0.95 [0.85, 1.07] | 1.005 [0.997, 1.019] | 8 |
| coupling=0.25 | 128 | 1.11 [1.02, 1.26] | 1.10 [1.02, 1.25] | 0.995 [0.984, 1.003] | 8 |
| coupling=0.25 | 1024 | 1.03 [1.00, 1.06] | 1.03 [1.00, 1.06] | 0.999 [0.998, 1.002] | 8 |
| coupling=0.5 | 16 | 0.94 [0.73, 1.24] | 0.95 [0.75, 1.20] | 1.010 [0.950, 1.055] | 8 |
| coupling=0.5 | 128 | 1.02 [0.90, 1.15] | 1.01 [0.90, 1.14] | 0.995 [0.965, 1.014] | 8 |
| coupling=0.5 | 1024 | 1.02 [1.00, 1.05] | 1.02 [1.00, 1.05] | 0.998 [0.994, 1.001] | 8 |
| coupling=1 | 16 | 1.06 [0.87, 1.30] | 1.10 [1.00, 1.20] | 1.033 [0.862, 1.245] | 8 |
| coupling=1 | 128 | 1.01 [0.95, 1.08] | 1.01 [0.96, 1.07] | 1.001 [0.975, 1.023] | 8 |
| coupling=1 | 1024 | 1.00 [0.99, 1.02] | 1.00 [0.99, 1.02] | 1.003 [1.000, 1.006] | 8 |
| coupling=2 | 16 | 0.99 [0.90, 1.13] | 0.94 [0.88, 1.02] | 0.951 [0.878, 1.041] | 8 |
| coupling=2 | 128 | 1.06 [0.98, 1.15] | 1.03 [0.97, 1.10] | 0.976 [0.952, 1.005] | 8 |
| coupling=2 | 1024 | 1.00 [0.99, 1.02] | 1.00 [0.99, 1.01] | 0.997 [0.992, 1.002] | 8 |

### FeedbackPredPrey, size 123M

| knob | H | past_cov / oracle | joint / oracle | joint/past_cov | n |
|---|---|---|---|---|---|
| coupling=0 | 16 | 1.18 [1.07, 1.34] | 1.19 [1.08, 1.37] | 1.012 [1.003, 1.021] | 4 |
| coupling=0 | 128 | 1.28 [0.98, 1.65] | 1.29 [0.97, 1.68] | 1.009 [0.994, 1.026] | 4 |
| coupling=0.5 | 16 | 0.97 [0.62, 1.61] | 0.94 [0.63, 1.41] | 0.968 [0.870, 1.037] | 4 |
| coupling=0.5 | 128 | 1.00 [0.85, 1.14] | 0.96 [0.86, 1.08] | 0.965 [0.911, 1.021] | 4 |
| coupling=2 | 16 | 1.10 [0.99, 1.21] | 0.96 [0.90, 1.05] | 0.870 [0.773, 0.980] | 4 |
| coupling=2 | 128 | 1.05 [0.98, 1.13] | 1.03 [0.98, 1.09] | 0.979 [0.951, 0.999] | 4 |

### FeedbackPredPrey, size 209M

| knob | H | past_cov / oracle | joint / oracle | joint/past_cov | n |
|---|---|---|---|---|---|
| coupling=0 | 16 | 1.22 [1.09, 1.45] | 1.23 [1.09, 1.46] | 1.007 [1.003, 1.012] | 4 |
| coupling=0 | 128 | 1.30 [0.94, 1.79] | 1.30 [0.92, 1.79] | 0.996 [0.983, 1.004] | 4 |
| coupling=0.5 | 16 | 1.01 [0.60, 1.77] | 1.04 [0.68, 1.66] | 1.035 [0.904, 1.176] | 4 |
| coupling=0.5 | 128 | 1.05 [0.79, 1.39] | 1.02 [0.86, 1.32] | 0.975 [0.867, 1.109] | 4 |
| coupling=2 | 16 | 1.17 [0.98, 1.39] | 0.95 [0.89, 1.02] | 0.812 [0.687, 0.959] | 4 |
| coupling=2 | 128 | 1.07 [1.01, 1.13] | 1.03 [0.98, 1.11] | 0.970 [0.943, 0.991] | 4 |

## Table 3. LeakyKnown: distance to the oracle when the known covariate reveals the future (T = 2048)

Knob `noise_ratio` = noise sd of x over the innovation sd of y. MASE ratio to the oracle matched to the condition (the exact posterior given x's future for `known_cov`; the marginal oracle otherwise), geometric mean [95% CI]. `oracle gain` = MASE(uni) / MASE(known_cov oracle).

### Size 5M

| knob | H | uni | past_cov | known_cov | joint | oracle gain | known_cov gain |
|---|---|---|---|---|---|---|---|
| noise_ratio=0.05 | 16 | 0.99 [0.88, 1.11] | 0.98 [0.87, 1.08] | 8.47 [6.64, 10.75] | 0.99 [0.88, 1.10] | 14.61 [10.10, 20.51] | 1.73 [1.40, 2.19] |
| noise_ratio=0.05 | 128 | 1.05 [1.02, 1.08] | 1.05 [1.02, 1.08] | 20.92 [17.57, 26.16] | 1.05 [1.02, 1.08] | 37.78 [34.36, 41.32] | 1.81 [1.48, 2.13] |
| noise_ratio=0.05 | 1024 | 1.07 [1.04, 1.10] | 1.06 [1.04, 1.09] | 42.71 [41.39, 44.38] | 1.06 [1.04, 1.10] | 45.93 [43.79, 48.49] | 1.08 [1.05, 1.10] |
| noise_ratio=0.25 | 16 | 1.04 [0.95, 1.13] | 1.04 [0.96, 1.12] | 3.11 [2.58, 3.67] | 1.04 [0.95, 1.13] | 5.23 [4.11, 6.95] | 1.68 [1.30, 2.14] |
| noise_ratio=0.25 | 128 | 1.03 [1.00, 1.06] | 1.03 [1.00, 1.06] | 4.47 [4.17, 4.77] | 1.03 [0.99, 1.06] | 8.86 [7.56, 10.39] | 1.98 [1.67, 2.34] |
| noise_ratio=0.25 | 1024 | 1.06 [1.04, 1.09] | 1.06 [1.04, 1.08] | 9.66 [9.38, 9.97] | 1.06 [1.04, 1.08] | 10.34 [10.08, 10.58] | 1.07 [1.04, 1.11] |
| noise_ratio=1 | 16 | 1.10 [0.97, 1.28] | 1.10 [0.97, 1.27] | 1.47 [1.22, 1.77] | 1.10 [0.97, 1.27] | 2.82 [2.04, 3.66] | 1.92 [1.44, 2.43] |
| noise_ratio=1 | 128 | 1.03 [0.98, 1.10] | 1.03 [0.98, 1.10] | 1.63 [1.56, 1.70] | 1.03 [0.98, 1.10] | 3.20 [2.87, 3.54] | 1.97 [1.80, 2.16] |
| noise_ratio=1 | 1024 | 1.07 [1.04, 1.09] | 1.07 [1.04, 1.09] | 3.24 [3.13, 3.36] | 1.07 [1.04, 1.09] | 3.52 [3.42, 3.64] | 1.09 [1.07, 1.10] |
| noise_ratio=4 | 16 | 1.01 [0.90, 1.13] | 1.00 [0.90, 1.12] | 1.11 [0.99, 1.23] | 1.02 [0.90, 1.13] | 1.85 [1.31, 2.57] | 1.66 [1.13, 2.47] |
| noise_ratio=4 | 128 | 1.05 [0.98, 1.13] | 1.05 [0.98, 1.13] | 1.20 [1.15, 1.25] | 1.05 [0.98, 1.14] | 1.75 [1.49, 2.14] | 1.46 [1.26, 1.78] |
| noise_ratio=4 | 1024 | 1.06 [1.03, 1.09] | 1.06 [1.03, 1.09] | 1.63 [1.56, 1.73] | 1.06 [1.03, 1.09] | 1.74 [1.64, 1.87] | 1.07 [1.04, 1.09] |
| noise_ratio=16 | 16 | 1.05 [0.93, 1.17] | 1.05 [0.93, 1.17] | 1.04 [0.93, 1.16] | 1.04 [0.93, 1.16] | 1.07 [0.94, 1.24] | 1.03 [0.96, 1.10] |
| noise_ratio=16 | 128 | 1.05 [1.01, 1.09] | 1.04 [1.01, 1.09] | 1.02 [0.92, 1.12] | 1.04 [1.01, 1.09] | 1.04 [0.92, 1.18] | 1.02 [0.98, 1.07] |
| noise_ratio=16 | 1024 | 1.05 [1.04, 1.07] | 1.05 [1.03, 1.07] | 1.14 [1.11, 1.18] | 1.05 [1.03, 1.07] | 1.15 [1.10, 1.19] | 1.00 [0.99, 1.01] |

### Size 28M

| knob | H | uni | past_cov | known_cov | joint | oracle gain | known_cov gain |
|---|---|---|---|---|---|---|---|
| noise_ratio=0.05 | 16 | 1.01 [0.86, 1.14] | 1.01 [0.86, 1.13] | 3.81 [3.06, 4.63] | 1.02 [0.88, 1.14] | 14.86 [10.07, 21.45] | 3.90 [3.11, 4.90] |
| noise_ratio=0.05 | 128 | 1.03 [1.00, 1.07] | 1.03 [0.99, 1.07] | 9.94 [8.93, 11.01] | 1.03 [1.00, 1.07] | 37.16 [34.22, 40.45] | 3.74 [3.42, 4.11] |
| noise_ratio=0.05 | 1024 | 1.03 [1.01, 1.06] | 1.03 [1.01, 1.05] | 38.41 [36.26, 40.77] | 1.03 [1.01, 1.05] | 44.48 [42.86, 46.27] | 1.16 [1.10, 1.22] |
| noise_ratio=0.25 | 16 | 1.04 [0.93, 1.15] | 1.03 [0.94, 1.14] | 1.72 [1.41, 2.16] | 1.03 [0.93, 1.14] | 5.22 [4.10, 7.08] | 3.03 [2.17, 4.24] |
| noise_ratio=0.25 | 128 | 1.01 [0.99, 1.04] | 1.01 [0.98, 1.04] | 2.46 [2.25, 2.67] | 1.01 [0.98, 1.04] | 8.70 [7.39, 10.37] | 3.54 [2.98, 4.15] |
| noise_ratio=0.25 | 1024 | 1.05 [1.02, 1.09] | 1.04 [1.01, 1.08] | 8.95 [8.39, 9.51] | 1.04 [1.01, 1.08] | 10.19 [9.77, 10.66] | 1.14 [1.04, 1.24] |
| noise_ratio=1 | 16 | 1.13 [1.01, 1.30] | 1.14 [1.01, 1.28] | 1.06 [0.94, 1.19] | 1.12 [1.00, 1.27] | 2.91 [2.10, 3.77] | 2.73 [2.16, 3.43] |
| noise_ratio=1 | 128 | 1.03 [0.96, 1.12] | 1.02 [0.95, 1.09] | 1.18 [1.13, 1.24] | 1.03 [0.96, 1.10] | 3.21 [2.89, 3.55] | 2.72 [2.34, 3.11] |
| noise_ratio=1 | 1024 | 1.03 [1.01, 1.05] | 1.03 [1.01, 1.05] | 2.95 [2.82, 3.12] | 1.03 [1.01, 1.05] | 3.39 [3.31, 3.47] | 1.15 [1.09, 1.19] |
| noise_ratio=4 | 16 | 1.02 [0.92, 1.13] | 1.02 [0.93, 1.11] | 0.97 [0.84, 1.10] | 1.02 [0.93, 1.12] | 1.86 [1.33, 2.66] | 1.92 [1.43, 2.67] |
| noise_ratio=4 | 128 | 1.01 [0.95, 1.08] | 1.01 [0.95, 1.07] | 1.08 [1.05, 1.12] | 1.01 [0.95, 1.08] | 1.67 [1.42, 2.04] | 1.55 [1.34, 1.87] |
| noise_ratio=4 | 1024 | 1.03 [1.00, 1.07] | 1.03 [1.00, 1.07] | 1.49 [1.41, 1.58] | 1.03 [1.00, 1.07] | 1.69 [1.59, 1.82] | 1.13 [1.10, 1.17] |
| noise_ratio=16 | 16 | 0.97 [0.85, 1.10] | 0.96 [0.84, 1.10] | 0.96 [0.82, 1.11] | 0.96 [0.84, 1.09] | 0.99 [0.86, 1.17] | 1.03 [0.96, 1.09] |
| noise_ratio=16 | 128 | 1.03 [0.99, 1.07] | 1.03 [0.99, 1.07] | 1.00 [0.91, 1.09] | 1.03 [0.99, 1.07] | 1.02 [0.91, 1.15] | 1.02 [0.98, 1.06] |
| noise_ratio=16 | 1024 | 1.03 [1.02, 1.05] | 1.03 [1.02, 1.05] | 1.11 [1.07, 1.15] | 1.03 [1.01, 1.05] | 1.13 [1.09, 1.17] | 1.02 [1.00, 1.03] |

### Size 123M

| knob | H | uni | past_cov | known_cov | joint | oracle gain | known_cov gain |
|---|---|---|---|---|---|---|---|
| noise_ratio=0.05 | 16 | 1.05 [0.96, 1.14] | 1.05 [0.96, 1.15] | 3.04 [2.29, 4.04] | 1.07 [0.98, 1.18] | 19.10 [14.57, 25.04] | 6.28 [4.28, 8.64] |
| noise_ratio=0.05 | 128 | 1.03 [1.01, 1.06] | 1.02 [1.00, 1.05] | 3.70 [3.60, 3.78] | 1.02 [1.00, 1.05] | 35.51 [32.14, 38.25] | 9.60 [8.89, 10.26] |
| noise_ratio=1 | 16 | 1.15 [0.94, 1.42] | 1.12 [0.94, 1.34] | 1.02 [0.82, 1.24] | 1.11 [0.93, 1.29] | 3.48 [2.54, 4.87] | 3.43 [2.10, 5.61] |
| noise_ratio=1 | 128 | 1.12 [0.98, 1.27] | 1.08 [0.97, 1.20] | 1.06 [1.05, 1.08] | 1.09 [0.98, 1.20] | 3.45 [2.70, 4.02] | 3.24 [2.55, 3.76] |
| noise_ratio=16 | 16 | 1.00 [0.94, 1.06] | 1.00 [0.94, 1.06] | 1.01 [0.96, 1.10] | 1.00 [0.95, 1.05] | 1.02 [0.94, 1.14] | 1.00 [0.98, 1.04] |
| noise_ratio=16 | 128 | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.00] | 0.90 [0.83, 0.95] | 0.99 [0.98, 1.00] | 0.90 [0.83, 0.95] | 1.00 [0.98, 1.02] |

### Size 209M

| knob | H | uni | past_cov | known_cov | joint | oracle gain | known_cov gain |
|---|---|---|---|---|---|---|---|
| noise_ratio=0.05 | 16 | 1.06 [0.97, 1.18] | 1.06 [0.97, 1.16] | 3.25 [2.76, 4.01] | 1.07 [0.99, 1.16] | 19.39 [14.28, 26.32] | 5.96 [4.78, 7.46] |
| noise_ratio=0.05 | 128 | 1.03 [1.00, 1.06] | 1.02 [1.00, 1.06] | 2.89 [2.66, 3.13] | 1.02 [0.99, 1.05] | 35.36 [32.06, 38.24] | 12.23 [11.54, 13.35] |
| noise_ratio=1 | 16 | 1.10 [0.92, 1.31] | 1.08 [0.91, 1.29] | 1.03 [0.84, 1.27] | 1.07 [0.93, 1.24] | 3.32 [2.58, 4.72] | 3.24 [2.11, 4.98] |
| noise_ratio=1 | 128 | 1.03 [0.95, 1.10] | 1.02 [0.94, 1.08] | 1.06 [1.03, 1.09] | 1.02 [0.94, 1.07] | 3.19 [2.64, 3.59] | 3.01 [2.42, 3.38] |
| noise_ratio=16 | 16 | 0.97 [0.91, 1.05] | 0.97 [0.91, 1.04] | 0.95 [0.85, 1.08] | 0.97 [0.92, 1.04] | 0.99 [0.90, 1.14] | 1.04 [0.99, 1.09] |
| noise_ratio=16 | 128 | 1.01 [0.98, 1.04] | 1.00 [0.98, 1.03] | 0.91 [0.82, 0.98] | 1.00 [0.98, 1.04] | 0.92 [0.84, 0.98] | 1.01 [0.99, 1.02] |

## Table 4. Context-length check, 5M (T = 512, 2048, 8192)

Covariate gain [95% CI] and MASE ratio to the oracle per condition, same samples at every T.

### LaggedDriver

| knob | H | T | past_cov: gain / ratio | known_cov: gain / ratio | joint: gain / ratio | uni: ratio |
|---|---|---|---|---|---|---|
| lag=0 | 16 | 512 | 0.99 [0.97, 1.01] / 0.98 [0.80, 1.21] | 2.33 [1.76, 3.08] / 2.28 [1.79, 3.07] | 0.99 [0.96, 1.01] / 0.98 [0.80, 1.21] | 0.97 [0.79, 1.20] |
| lag=0 | 16 | 2048 | 0.98 [0.96, 1.00] / 1.02 [0.90, 1.13] | 2.55 [1.89, 3.20] / 2.16 [1.72, 2.91] | 0.98 [0.95, 1.00] / 1.02 [0.90, 1.14] | 1.00 [0.89, 1.12] |
| lag=0 | 16 | 8192 | 0.99 [0.97, 1.00] / 1.00 [0.93, 1.08] | 2.57 [1.92, 3.21] / 2.12 [1.71, 2.80] | 0.99 [0.97, 1.00] / 1.00 [0.93, 1.08] | 0.99 [0.91, 1.07] |
| lag=0 | 128 | 512 | 1.00 [1.00, 1.00] / 1.03 [1.00, 1.08] | 2.08 [1.75, 2.37] / 3.61 [3.09, 4.08] | 1.00 [1.00, 1.00] / 1.03 [1.00, 1.08] | 1.04 [1.00, 1.08] |
| lag=0 | 128 | 2048 | 1.00 [1.00, 1.00] / 1.00 [0.95, 1.04] | 1.95 [1.63, 2.23] / 3.69 [3.16, 4.25] | 1.00 [1.00, 1.00] / 1.00 [0.95, 1.04] | 1.00 [0.95, 1.04] |
| lag=0 | 128 | 8192 | 1.00 [1.00, 1.00] / 1.01 [0.97, 1.05] | 2.03 [1.72, 2.29] / 3.59 [3.11, 4.07] | 1.00 [1.00, 1.00] / 1.01 [0.97, 1.05] | 1.00 [0.96, 1.05] |
| lag=0 | 1024 | 512 | 1.00 [1.00, 1.00] / 1.17 [1.13, 1.21] | 1.07 [1.04, 1.10] / 8.26 [7.62, 9.01] | 1.00 [1.00, 1.00] / 1.17 [1.13, 1.21] | 1.17 [1.13, 1.21] |
| lag=0 | 1024 | 2048 | 1.00 [1.00, 1.00] / 1.07 [1.05, 1.10] | 1.04 [0.98, 1.08] / 7.80 [7.14, 8.57] | 1.00 [1.00, 1.00] / 1.07 [1.05, 1.10] | 1.08 [1.05, 1.10] |
| lag=0 | 1024 | 8192 | 1.00 [1.00, 1.00] / 1.04 [1.03, 1.06] | 1.03 [1.00, 1.06] / 7.61 [7.05, 8.27] | 1.00 [1.00, 1.00] / 1.04 [1.03, 1.06] | 1.04 [1.03, 1.06] |
| lag=8 | 16 | 512 | 1.25 [1.01, 1.56] / 1.43 [1.10, 1.83] | 1.45 [1.13, 1.91] / 4.16 [3.31, 5.31] | 1.26 [1.00, 1.64] / 1.41 [1.09, 1.81] | 1.78 [1.27, 2.45] |
| lag=8 | 16 | 2048 | 1.18 [0.94, 1.54] / 1.50 [1.22, 1.87] | 1.59 [1.26, 2.10] / 3.75 [3.10, 4.57] | 1.20 [0.93, 1.62] / 1.47 [1.19, 1.87] | 1.76 [1.39, 2.33] |
| lag=8 | 16 | 8192 | 1.37 [1.05, 1.84] / 1.32 [1.11, 1.58] | 1.75 [1.42, 2.27] / 3.49 [2.91, 4.24] | 1.39 [1.07, 1.90] / 1.30 [1.09, 1.56] | 1.81 [1.44, 2.40] |
| lag=8 | 128 | 512 | 1.04 [0.99, 1.09] / 1.05 [0.94, 1.17] | 1.74 [1.56, 1.96] / 4.68 [4.41, 4.93] | 1.04 [0.99, 1.10] / 1.04 [0.94, 1.17] | 1.09 [0.99, 1.22] |
| lag=8 | 128 | 2048 | 1.04 [1.00, 1.08] / 0.98 [0.91, 1.04] | 1.73 [1.59, 1.87] / 4.40 [4.14, 4.63] | 1.04 [1.00, 1.08] / 0.98 [0.91, 1.04] | 1.02 [0.96, 1.07] |
| lag=8 | 128 | 8192 | 1.05 [1.02, 1.09] / 0.97 [0.91, 1.02] | 1.80 [1.70, 1.91] / 4.25 [4.07, 4.40] | 1.05 [1.02, 1.09] / 0.97 [0.91, 1.02] | 1.02 [0.97, 1.06] |
| lag=8 | 1024 | 512 | 1.02 [1.00, 1.03] / 1.17 [1.13, 1.21] | 1.11 [1.07, 1.14] / 8.63 [8.26, 8.98] | 1.02 [1.00, 1.04] / 1.17 [1.13, 1.22] | 1.20 [1.16, 1.23] |
| lag=8 | 1024 | 2048 | 1.01 [1.00, 1.02] / 1.04 [1.03, 1.05] | 1.07 [1.04, 1.09] / 7.88 [7.49, 8.25] | 1.01 [1.00, 1.02] / 1.04 [1.03, 1.05] | 1.06 [1.05, 1.06] |
| lag=8 | 1024 | 8192 | 1.02 [1.01, 1.03] / 1.02 [1.00, 1.04] | 1.07 [1.05, 1.10] / 7.71 [7.34, 8.16] | 1.02 [1.01, 1.03] / 1.02 [1.00, 1.04] | 1.04 [1.02, 1.06] |
| lag=32 | 16 | 512 | 1.04 [0.99, 1.09] / 8.08 [5.74, 11.19] | 1.07 [0.97, 1.18] / 7.91 [5.86, 10.79] | 1.04 [0.99, 1.09] / 8.09 [5.74, 11.17] | 8.39 [5.87, 11.61] |
| lag=32 | 16 | 2048 | 1.07 [1.00, 1.16] / 7.69 [5.48, 10.15] | 1.15 [1.03, 1.31] / 7.19 [5.49, 9.18] | 1.08 [1.00, 1.17] / 7.67 [5.46, 10.12] | 8.26 [5.66, 11.74] |
| lag=32 | 16 | 8192 | 1.08 [1.00, 1.19] / 7.13 [5.22, 9.30] | 1.17 [1.04, 1.34] / 6.64 [5.12, 8.49] | 1.09 [1.00, 1.20] / 7.11 [5.22, 9.26] | 7.72 [5.37, 10.76] |
| lag=32 | 128 | 512 | 1.01 [1.00, 1.03] / 1.28 [1.15, 1.44] | 1.03 [0.98, 1.10] / 7.10 [6.30, 8.03] | 1.01 [1.00, 1.03] / 1.28 [1.15, 1.44] | 1.30 [1.15, 1.47] |
| lag=32 | 128 | 2048 | 1.03 [1.00, 1.07] / 1.24 [1.13, 1.34] | 1.14 [1.06, 1.22] / 6.32 [5.68, 7.17] | 1.03 [0.99, 1.07] / 1.24 [1.13, 1.34] | 1.27 [1.15, 1.40] |
| lag=32 | 128 | 8192 | 1.05 [1.01, 1.08] / 1.20 [1.13, 1.26] | 1.27 [1.21, 1.35] / 5.57 [5.04, 6.27] | 1.05 [1.01, 1.09] / 1.20 [1.13, 1.26] | 1.25 [1.17, 1.34] |
| lag=32 | 1024 | 512 | 1.00 [0.99, 1.00] / 1.17 [1.14, 1.21] | 0.99 [0.98, 1.00] / 8.59 [8.08, 9.16] | 1.00 [1.00, 1.00] / 1.17 [1.14, 1.20] | 1.17 [1.14, 1.21] |
| lag=32 | 1024 | 2048 | 1.00 [1.00, 1.01] / 1.07 [1.06, 1.09] | 1.01 [1.00, 1.02] / 7.74 [7.14, 8.39] | 1.00 [1.00, 1.01] / 1.07 [1.06, 1.08] | 1.08 [1.07, 1.09] |
| lag=32 | 1024 | 8192 | 1.00 [1.00, 1.01] / 1.05 [1.04, 1.06] | 1.02 [1.01, 1.03] / 7.44 [6.84, 8.08] | 1.00 [1.00, 1.01] / 1.05 [1.04, 1.06] | 1.05 [1.04, 1.07] |
| lag=128 | 16 | 512 | 0.99 [0.95, 1.03] / 7.16 [5.79, 9.10] | 0.97 [0.83, 1.11] / 7.33 [5.79, 9.23] | 0.98 [0.91, 1.03] / 7.24 [5.84, 9.10] | 7.11 [5.58, 9.21] |
| lag=128 | 16 | 2048 | 0.98 [0.95, 1.01] / 7.35 [5.92, 9.19] | 0.96 [0.88, 1.05] / 7.48 [5.96, 9.40] | 0.98 [0.94, 1.02] / 7.36 [5.93, 9.18] | 7.23 [5.87, 8.97] |
| lag=128 | 16 | 8192 | 1.00 [0.97, 1.02] / 7.66 [6.00, 9.90] | 1.01 [0.96, 1.07] / 7.50 [5.85, 9.72] | 1.00 [0.96, 1.02] / 7.65 [6.00, 9.89] | 7.63 [6.07, 9.83] |
| lag=128 | 128 | 512 | 1.02 [1.01, 1.03] / 7.84 [7.09, 8.96] | 1.04 [0.98, 1.09] / 7.70 [6.95, 8.73] | 1.02 [1.01, 1.03] / 7.85 [7.11, 8.96] | 7.99 [7.24, 9.13] |
| lag=128 | 128 | 2048 | 1.01 [0.99, 1.02] / 7.85 [7.32, 8.56] | 1.01 [0.97, 1.05] / 7.81 [7.19, 8.68] | 1.01 [0.99, 1.02] / 7.85 [7.32, 8.56] | 7.91 [7.31, 8.66] |
| lag=128 | 128 | 8192 | 1.00 [0.99, 1.01] / 7.64 [6.83, 8.55] | 0.99 [0.97, 1.02] / 7.68 [6.93, 8.62] | 1.00 [1.00, 1.01] / 7.63 [6.82, 8.54] | 7.65 [6.82, 8.57] |
| lag=128 | 1024 | 512 | 1.00 [0.99, 1.01] / 1.33 [1.25, 1.43] | 1.00 [0.98, 1.04] / 9.39 [8.85, 10.15] | 1.00 [1.00, 1.01] / 1.33 [1.25, 1.42] | 1.34 [1.25, 1.44] |
| lag=128 | 1024 | 2048 | 1.00 [1.00, 1.00] / 1.18 [1.14, 1.23] | 0.99 [0.98, 1.00] / 8.46 [7.95, 9.03] | 1.00 [1.00, 1.01] / 1.18 [1.14, 1.23] | 1.18 [1.15, 1.23] |
| lag=128 | 1024 | 8192 | 1.00 [1.00, 1.00] / 1.16 [1.12, 1.19] | 0.99 [0.97, 0.99] / 8.27 [7.81, 8.74] | 1.00 [1.00, 1.00] / 1.15 [1.12, 1.19] | 1.15 [1.12, 1.19] |
| lag=512 | 16 | 512 | 0.97 [0.93, 1.02] / 7.16 [5.73, 8.95] | 1.02 [0.98, 1.07] / 6.79 [5.59, 8.14] | 0.97 [0.92, 1.02] / 7.21 [5.77, 9.10] | 6.97 [5.59, 8.54] |
| lag=512 | 16 | 2048 | 1.01 [0.96, 1.07] / 6.30 [4.92, 8.12] | 1.00 [0.94, 1.06] / 6.30 [5.24, 7.64] | 1.01 [0.95, 1.07] / 6.33 [4.93, 8.19] | 6.37 [5.10, 8.01] |
| lag=512 | 16 | 8192 | 1.01 [0.97, 1.05] / 5.94 [4.75, 7.45] | 0.97 [0.92, 1.02] / 6.10 [5.14, 7.22] | 1.01 [0.96, 1.05] / 5.95 [4.75, 7.50] | 5.98 [4.87, 7.33] |
| lag=512 | 128 | 512 | 1.01 [1.00, 1.02] / 8.72 [7.69, 9.94] | 1.03 [0.98, 1.09] / 8.44 [7.56, 9.53] | 1.01 [0.99, 1.02] / 8.74 [7.70, 9.96] | 8.80 [7.75, 10.13] |
| lag=512 | 128 | 2048 | 1.01 [0.99, 1.03] / 8.24 [7.41, 9.42] | 1.03 [0.99, 1.07] / 8.02 [7.30, 9.03] | 1.01 [0.99, 1.03] / 8.25 [7.42, 9.43] | 8.34 [7.49, 9.56] |
| lag=512 | 128 | 8192 | 1.01 [1.00, 1.02] / 7.84 [7.17, 8.99] | 1.01 [0.99, 1.03] / 7.74 [7.08, 8.81] | 1.01 [1.00, 1.02] / 7.84 [7.17, 8.99] | 7.91 [7.21, 9.11] |
| lag=512 | 1024 | 512 | 1.00 [0.99, 1.02] / 2.15 [1.98, 2.32] | 1.02 [1.00, 1.03] / 9.23 [8.66, 10.07] | 1.00 [0.99, 1.02] / 2.15 [1.98, 2.33] | 2.15 [1.97, 2.34] |
| lag=512 | 1024 | 2048 | 1.01 [1.00, 1.01] / 1.91 [1.81, 2.02] | 1.00 [0.98, 1.01] / 8.41 [7.98, 8.99] | 1.01 [1.00, 1.01] / 1.91 [1.80, 2.02] | 1.92 [1.81, 2.04] |
| lag=512 | 1024 | 8192 | 1.00 [0.99, 1.01] / 1.84 [1.76, 1.92] | 0.99 [0.97, 1.00] / 8.14 [7.87, 8.50] | 1.00 [0.99, 1.01] / 1.84 [1.76, 1.92] | 1.84 [1.76, 1.93] |

### FeedbackLinear

| knob | H | T | past_cov: gain / ratio | known_cov: gain / ratio | joint: gain / ratio | uni: ratio |
|---|---|---|---|---|---|---|
| coupling=0 | 16 | 512 | 1.01 [0.99, 1.04] / 1.04 [1.01, 1.07] | - | 1.01 [0.99, 1.04] / 1.04 [1.02, 1.07] | 1.06 [1.02, 1.10] |
| coupling=0 | 16 | 2048 | 1.01 [1.00, 1.02] / 1.03 [1.00, 1.06] | - | 1.01 [1.00, 1.02] / 1.03 [1.00, 1.06] | 1.03 [1.00, 1.07] |
| coupling=0 | 16 | 8192 | 1.01 [1.00, 1.02] / 1.01 [0.99, 1.03] | - | 1.01 [1.00, 1.02] / 1.01 [0.99, 1.03] | 1.02 [1.00, 1.04] |
| coupling=0 | 128 | 512 | 1.00 [1.00, 1.01] / 1.00 [1.00, 1.01] | - | 1.00 [1.00, 1.01] / 1.00 [1.00, 1.01] | 1.01 [1.00, 1.03] |
| coupling=0 | 128 | 2048 | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.00] | 1.00 [0.99, 1.01] |
| coupling=0 | 128 | 8192 | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.00] | 1.00 [0.99, 1.00] |
| coupling=0 | 1024 | 512 | 1.00 [1.00, 1.02] / 1.00 [1.00, 1.01] | - | 1.01 [1.00, 1.02] / 1.00 [1.00, 1.01] | 1.01 [1.00, 1.03] |
| coupling=0 | 1024 | 2048 | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| coupling=0 | 1024 | 8192 | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| coupling=0.05 | 16 | 512 | 0.99 [0.98, 1.00] / 1.02 [0.98, 1.06] | - | 0.99 [0.98, 1.00] / 1.02 [0.98, 1.06] | 1.01 [0.98, 1.04] |
| coupling=0.05 | 16 | 2048 | 1.01 [0.99, 1.02] / 1.01 [0.99, 1.03] | - | 1.01 [0.99, 1.02] / 1.01 [0.99, 1.03] | 1.02 [1.00, 1.04] |
| coupling=0.05 | 16 | 8192 | 1.01 [1.00, 1.02] / 1.00 [1.00, 1.01] | - | 1.01 [1.00, 1.02] / 1.00 [1.00, 1.01] | 1.01 [1.00, 1.03] |
| coupling=0.05 | 128 | 512 | 1.00 [1.00, 1.00] / 1.01 [1.00, 1.02] | - | 1.00 [1.00, 1.00] / 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] |
| coupling=0.05 | 128 | 2048 | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.01] | - | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.01] | 1.00 [1.00, 1.01] |
| coupling=0.05 | 128 | 8192 | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.00] | 1.00 [0.99, 1.01] |
| coupling=0.05 | 1024 | 512 | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | 1.00 [1.00, 1.01] |
| coupling=0.05 | 1024 | 2048 | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| coupling=0.05 | 1024 | 8192 | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| coupling=0.15 | 16 | 512 | 1.02 [0.99, 1.05] / 1.02 [0.99, 1.06] | - | 1.01 [0.99, 1.05] / 1.02 [0.99, 1.07] | 1.04 [0.99, 1.10] |
| coupling=0.15 | 16 | 2048 | 1.03 [0.99, 1.11] / 1.01 [0.97, 1.04] | - | 1.03 [0.98, 1.11] / 1.01 [0.97, 1.04] | 1.04 [1.01, 1.08] |
| coupling=0.15 | 16 | 8192 | 1.03 [0.99, 1.08] / 1.02 [1.00, 1.03] | - | 1.03 [0.99, 1.09] / 1.02 [1.01, 1.03] | 1.05 [1.01, 1.10] |
| coupling=0.15 | 128 | 512 | 1.00 [0.99, 1.01] / 1.01 [0.99, 1.02] | - | 1.00 [0.99, 1.01] / 1.01 [0.99, 1.02] | 1.00 [0.99, 1.02] |
| coupling=0.15 | 128 | 2048 | 1.00 [0.99, 1.01] / 1.00 [0.99, 1.01] | - | 1.00 [0.99, 1.01] / 1.00 [0.99, 1.01] | 1.00 [1.00, 1.01] |
| coupling=0.15 | 128 | 8192 | 1.00 [0.99, 1.01] / 1.00 [1.00, 1.01] | - | 1.00 [0.99, 1.01] / 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] |
| coupling=0.15 | 1024 | 512 | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.01] | - | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.01] | 1.00 [1.00, 1.01] |
| coupling=0.15 | 1024 | 2048 | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| coupling=0.15 | 1024 | 8192 | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] |
| coupling=0.4 | 16 | 512 | 1.01 [0.97, 1.05] / 0.99 [0.96, 1.03] | - | 1.01 [0.97, 1.05] / 0.99 [0.96, 1.02] | 1.00 [0.96, 1.05] |
| coupling=0.4 | 16 | 2048 | 1.00 [0.96, 1.03] / 1.00 [0.97, 1.03] | - | 0.99 [0.96, 1.03] / 1.00 [0.97, 1.04] | 1.00 [0.96, 1.04] |
| coupling=0.4 | 16 | 8192 | 0.99 [0.95, 1.03] / 1.01 [0.98, 1.03] | - | 0.98 [0.93, 1.03] / 1.02 [0.98, 1.05] | 1.00 [0.96, 1.03] |
| coupling=0.4 | 128 | 512 | 1.00 [1.00, 1.01] / 1.00 [0.99, 1.01] | - | 1.00 [0.99, 1.01] / 1.01 [1.00, 1.01] | 1.01 [0.99, 1.02] |
| coupling=0.4 | 128 | 2048 | 1.00 [1.00, 1.01] / 0.99 [0.99, 1.00] | - | 1.00 [0.99, 1.01] / 1.00 [0.99, 1.00] | 1.00 [0.99, 1.00] |
| coupling=0.4 | 128 | 8192 | 1.00 [0.99, 1.00] / 1.00 [0.99, 1.00] | - | 1.00 [0.99, 1.00] / 1.00 [0.99, 1.01] | 1.00 [0.99, 1.00] |
| coupling=0.4 | 1024 | 512 | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.01] | - | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.01] | 1.00 [1.00, 1.01] |
| coupling=0.4 | 1024 | 2048 | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| coupling=0.4 | 1024 | 8192 | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | - | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| coupling=1 | 16 | 512 | 0.98 [0.91, 1.06] / 1.15 [0.94, 1.50] | - | 0.98 [0.91, 1.06] / 1.15 [0.93, 1.50] | 1.12 [0.92, 1.45] |
| coupling=1 | 16 | 2048 | 1.06 [1.02, 1.12] / 1.10 [0.85, 1.42] | - | 1.06 [1.01, 1.11] / 1.11 [0.86, 1.42] | 1.17 [0.87, 1.55] |
| coupling=1 | 16 | 8192 | 1.02 [0.97, 1.06] / 1.21 [0.98, 1.50] | - | 1.01 [0.96, 1.05] / 1.22 [0.99, 1.51] | 1.23 [0.97, 1.55] |
| coupling=1 | 128 | 512 | 1.00 [0.99, 1.02] / 1.11 [0.97, 1.27] | - | 1.00 [0.98, 1.01] / 1.11 [0.97, 1.27] | 1.11 [0.98, 1.26] |
| coupling=1 | 128 | 2048 | 1.00 [0.98, 1.02] / 1.08 [0.98, 1.20] | - | 1.00 [0.98, 1.02] / 1.08 [0.98, 1.20] | 1.08 [0.99, 1.20] |
| coupling=1 | 128 | 8192 | 0.99 [0.98, 1.01] / 1.10 [1.01, 1.20] | - | 0.99 [0.98, 1.01] / 1.10 [1.01, 1.20] | 1.09 [1.01, 1.18] |
| coupling=1 | 1024 | 512 | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.02] | - | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.02] | 1.00 [0.99, 1.02] |
| coupling=1 | 1024 | 2048 | 1.00 [0.99, 1.00] / 1.00 [0.98, 1.01] | - | 1.00 [0.99, 1.00] / 0.99 [0.98, 1.01] | 0.99 [0.98, 1.01] |
| coupling=1 | 1024 | 8192 | 1.00 [1.00, 1.00] / 1.00 [0.98, 1.01] | - | 1.00 [1.00, 1.00] / 1.00 [0.98, 1.01] | 1.00 [0.98, 1.01] |

### FeedbackPredPrey

| knob | H | T | past_cov: gain / ratio | known_cov: gain / ratio | joint: gain / ratio | uni: ratio |
|---|---|---|---|---|---|---|
| coupling=0 | 16 | 512 | 1.03 [1.00, 1.09] / 1.17 [1.05, 1.37] | - | 1.04 [1.00, 1.10] / 1.17 [1.05, 1.36] | 1.21 [1.05, 1.48] |
| coupling=0 | 16 | 2048 | 1.03 [1.00, 1.07] / 1.12 [1.01, 1.25] | - | 1.03 [1.00, 1.06] / 1.12 [1.01, 1.25] | 1.15 [1.03, 1.32] |
| coupling=0 | 16 | 8192 | 1.02 [1.00, 1.04] / 1.09 [1.02, 1.17] | - | 1.02 [1.00, 1.03] / 1.10 [1.02, 1.17] | 1.11 [1.04, 1.19] |
| coupling=0 | 128 | 512 | 1.00 [0.98, 1.02] / 1.18 [0.95, 1.53] | - | 1.00 [0.97, 1.02] / 1.18 [0.95, 1.53] | 1.18 [0.94, 1.53] |
| coupling=0 | 128 | 2048 | 1.03 [0.98, 1.07] / 1.24 [1.08, 1.44] | - | 1.02 [0.98, 1.06] / 1.25 [1.08, 1.45] | 1.28 [1.08, 1.53] |
| coupling=0 | 128 | 8192 | 1.00 [0.98, 1.01] / 1.06 [0.94, 1.19] | - | 1.00 [0.98, 1.01] / 1.06 [0.94, 1.19] | 1.06 [0.93, 1.18] |
| coupling=0 | 1024 | 512 | 1.00 [1.00, 1.01] / 1.08 [0.97, 1.20] | - | 1.00 [0.99, 1.01] / 1.08 [0.97, 1.20] | 1.08 [0.98, 1.20] |
| coupling=0 | 1024 | 2048 | 1.02 [1.01, 1.04] / 1.07 [0.99, 1.16] | - | 1.02 [1.01, 1.03] / 1.08 [0.99, 1.17] | 1.10 [1.00, 1.20] |
| coupling=0 | 1024 | 8192 | 1.01 [1.00, 1.02] / 1.01 [0.95, 1.06] | - | 1.01 [1.00, 1.02] / 1.01 [0.95, 1.06] | 1.02 [0.96, 1.07] |
| coupling=0.25 | 16 | 512 | 1.00 [0.98, 1.02] / 0.98 [0.86, 1.13] | - | 1.00 [0.97, 1.03] / 0.98 [0.87, 1.11] | 0.98 [0.85, 1.13] |
| coupling=0.25 | 16 | 2048 | 0.98 [0.95, 1.01] / 0.93 [0.85, 1.04] | - | 0.97 [0.93, 1.02] / 0.94 [0.86, 1.04] | 0.91 [0.81, 1.03] |
| coupling=0.25 | 16 | 8192 | 1.00 [0.97, 1.04] / 0.99 [0.85, 1.16] | - | 1.00 [0.97, 1.05] / 0.99 [0.85, 1.17] | 0.99 [0.84, 1.19] |
| coupling=0.25 | 128 | 512 | 1.01 [0.98, 1.04] / 1.28 [1.12, 1.52] | - | 1.02 [0.98, 1.05] / 1.28 [1.12, 1.52] | 1.30 [1.11, 1.58] |
| coupling=0.25 | 128 | 2048 | 1.02 [0.99, 1.04] / 1.15 [1.04, 1.31] | - | 1.02 [0.99, 1.05] / 1.15 [1.04, 1.30] | 1.17 [1.05, 1.34] |
| coupling=0.25 | 128 | 8192 | 1.02 [1.00, 1.04] / 1.06 [0.93, 1.21] | - | 1.02 [1.00, 1.05] / 1.05 [0.93, 1.20] | 1.08 [0.95, 1.25] |
| coupling=0.25 | 1024 | 512 | 1.00 [0.99, 1.01] / 1.03 [1.01, 1.06] | - | 1.00 [0.99, 1.01] / 1.03 [1.01, 1.06] | 1.04 [1.01, 1.06] |
| coupling=0.25 | 1024 | 2048 | 1.00 [1.00, 1.01] / 1.03 [1.00, 1.07] | - | 1.00 [1.00, 1.01] / 1.03 [1.00, 1.07] | 1.04 [1.00, 1.08] |
| coupling=0.25 | 1024 | 8192 | 1.01 [1.00, 1.01] / 1.03 [1.01, 1.06] | - | 1.01 [1.00, 1.02] / 1.03 [1.01, 1.06] | 1.04 [1.02, 1.07] |
| coupling=0.5 | 16 | 512 | 0.93 [0.79, 1.09] / 0.81 [0.64, 1.04] | - | 0.91 [0.76, 1.06] / 0.83 [0.64, 1.07] | 0.76 [0.57, 1.00] |
| coupling=0.5 | 16 | 2048 | 0.96 [0.87, 1.06] / 0.95 [0.77, 1.15] | - | 0.95 [0.84, 1.09] / 0.95 [0.78, 1.14] | 0.91 [0.74, 1.16] |
| coupling=0.5 | 16 | 8192 | 0.94 [0.84, 1.06] / 0.96 [0.84, 1.10] | - | 0.93 [0.82, 1.09] / 0.97 [0.86, 1.08] | 0.90 [0.74, 1.14] |
| coupling=0.5 | 128 | 512 | 1.07 [1.00, 1.16] / 1.05 [0.92, 1.20] | - | 1.06 [0.97, 1.16] / 1.06 [0.93, 1.20] | 1.12 [0.98, 1.30] |
| coupling=0.5 | 128 | 2048 | 1.03 [0.98, 1.12] / 0.97 [0.90, 1.05] | - | 1.02 [0.97, 1.11] / 0.97 [0.90, 1.05] | 1.00 [0.90, 1.11] |
| coupling=0.5 | 128 | 8192 | 0.97 [0.92, 1.04] / 0.98 [0.91, 1.08] | - | 0.97 [0.91, 1.05] / 0.98 [0.91, 1.08] | 0.96 [0.87, 1.05] |
| coupling=0.5 | 1024 | 512 | 1.02 [1.00, 1.04] / 1.08 [1.01, 1.17] | - | 1.02 [1.00, 1.04] / 1.09 [1.01, 1.17] | 1.10 [1.03, 1.18] |
| coupling=0.5 | 1024 | 2048 | 1.01 [1.00, 1.02] / 1.02 [1.00, 1.05] | - | 1.01 [1.00, 1.02] / 1.02 [1.00, 1.05] | 1.03 [1.01, 1.06] |
| coupling=0.5 | 1024 | 8192 | 1.00 [0.99, 1.01] / 1.02 [0.99, 1.04] | - | 1.00 [0.99, 1.01] / 1.02 [0.99, 1.04] | 1.02 [0.99, 1.04] |
| coupling=1 | 16 | 512 | 1.07 [0.84, 1.31] / 1.10 [0.90, 1.34] | - | 1.07 [0.80, 1.41] / 1.10 [0.84, 1.41] | 1.18 [1.00, 1.38] |
| coupling=1 | 16 | 2048 | 1.01 [0.76, 1.28] / 1.13 [0.94, 1.38] | - | 1.00 [0.71, 1.35] / 1.15 [0.96, 1.39] | 1.15 [0.92, 1.41] |
| coupling=1 | 16 | 8192 | 0.93 [0.71, 1.18] / 1.11 [0.95, 1.30] | - | 0.91 [0.63, 1.24] / 1.14 [0.96, 1.35] | 1.03 [0.71, 1.37] |
| coupling=1 | 128 | 512 | 1.00 [0.97, 1.04] / 1.03 [0.97, 1.10] | - | 1.01 [0.97, 1.06] / 1.03 [0.96, 1.09] | 1.04 [0.95, 1.12] |
| coupling=1 | 128 | 2048 | 1.01 [0.96, 1.06] / 1.00 [0.92, 1.08] | - | 1.01 [0.96, 1.06] / 1.00 [0.92, 1.09] | 1.01 [0.93, 1.10] |
| coupling=1 | 128 | 8192 | 1.01 [0.96, 1.04] / 1.00 [0.91, 1.08] | - | 1.01 [0.97, 1.04] / 1.00 [0.92, 1.08] | 1.00 [0.92, 1.09] |
| coupling=1 | 1024 | 512 | 1.00 [1.00, 1.01] / 1.01 [1.00, 1.03] | - | 1.00 [1.00, 1.01] / 1.01 [1.00, 1.03] | 1.02 [1.00, 1.03] |
| coupling=1 | 1024 | 2048 | 1.00 [0.99, 1.00] / 1.01 [1.00, 1.02] | - | 1.00 [0.99, 1.01] / 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] |
| coupling=1 | 1024 | 8192 | 1.00 [0.99, 1.00] / 1.01 [1.00, 1.02] | - | 1.00 [1.00, 1.00] / 1.01 [1.00, 1.01] | 1.01 [1.00, 1.02] |
| coupling=2 | 16 | 512 | 1.04 [0.92, 1.17] / 0.94 [0.77, 1.14] | - | 1.08 [0.93, 1.30] / 0.90 [0.68, 1.15] | 0.98 [0.85, 1.13] |
| coupling=2 | 16 | 2048 | 1.02 [0.88, 1.22] / 1.11 [0.98, 1.26] | - | 1.02 [0.90, 1.15] / 1.11 [0.99, 1.27] | 1.14 [0.97, 1.31] |
| coupling=2 | 16 | 8192 | 1.01 [0.85, 1.17] / 1.12 [1.00, 1.26] | - | 1.01 [0.85, 1.17] / 1.12 [1.02, 1.23] | 1.13 [0.93, 1.34] |
| coupling=2 | 128 | 512 | 0.99 [0.96, 1.02] / 1.07 [1.00, 1.17] | - | 0.99 [0.96, 1.02] / 1.07 [1.00, 1.17] | 1.06 [1.00, 1.15] |
| coupling=2 | 128 | 2048 | 1.01 [0.99, 1.04] / 1.06 [1.00, 1.12] | - | 1.01 [0.98, 1.03] / 1.07 [0.99, 1.14] | 1.07 [1.00, 1.16] |
| coupling=2 | 128 | 8192 | 1.00 [0.96, 1.05] / 1.04 [0.98, 1.09] | - | 1.01 [0.97, 1.04] / 1.04 [0.97, 1.10] | 1.04 [0.97, 1.12] |
| coupling=2 | 1024 | 512 | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.01] | - | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] |
| coupling=2 | 1024 | 2048 | 1.00 [1.00, 1.00] / 1.01 [1.00, 1.02] | - | 1.00 [1.00, 1.00] / 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] |
| coupling=2 | 1024 | 8192 | 1.00 [0.99, 1.00] / 1.00 [0.99, 1.01] | - | 1.00 [1.00, 1.00] / 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] |
