---
title: "Off the Prior: Mapping Where a Synthetic-Prior Time-Series Foundation Model Fails, and Why"
author: "Yousef Anas (Cúper)"
date: "October 2026"
abstract: |
  RESULTS-PENDING: the abstract is written last, from the result files.
---

<!--
Writing rule for this file: every number must trace to a committed result file produced by a
committed script. Each number carries a pointer comment like <!-- src: results/... --> next to it.
Sections marked RESULTS-PENDING are filled only from those files.
-->

# 1. Introduction

Timestral is a family of open time-series foundation models (5M, 28M, 123M and 209M
parameters) released by Mistral on October 15, 2026. It forecasts up to 1,024 steps per
forward pass from up to 32,768 steps of history, outputs 21 quantiles per step, accepts
past-only and known-future covariates, and was pretrained mostly on synthetic data: simple
trend, seasonality and noise processes combined into multivariate series. Mistral reports
first place among pretrained and zero-shot models on GIFT-Eval, TIME and fev-bench.

A model trained on a synthetic distribution learns a prior over what series look like.
Benchmarks of real data measure how well that prior matches the series people happen to
collect. They do not say where the prior ends. This paper asks that question directly: we
build parameterized generators whose difficulty can be turned up smoothly, forecast them
with all four Timestral sizes, and measure the gap to forecasters that know the true
generating process. Then we look inside the model to explain the gaps, package the hard
cases as benchmarks, and test whether cheap fine-tuning closes them.

Contributions:

1. **A failure map** of Timestral across four signal families (chaos and dynamics, regime
   switches, heavy tails and fractal processes, and a learned mixture of primitives), along
   every difficulty knob, at four model sizes, four context lengths and horizons from 16
   steps to rollouts past one 1,024-step block, measured against a true-dynamics oracle and
   classical baselines.
2. **A covariate test** of Timestral's one-way covariate attention on coupled systems.
3. **A comparison of search methods** (CMA-ES, MAP-Elites quality-diversity, and gradient
   search through the model) for finding failures automatically.
4. **A mechanism analysis** with attention measurements on the forecast tokens and linear
   readouts of internal states.
5. **Two benchmarks** (a tagged diagnostic set and a leaderboard set) with generator code,
   seeds and a one-command evaluation.
6. **A fine-tuning study** comparing full fine-tuning, LoRA and data mixing under
   leave-one-family-out and held-out-knob protocols, with a regression check on normal series.

# 2. Setup

## 2.1 The model

Timestral is a causal patch transformer. A series is cut into patches of 32 steps; each
patch is normalized with the mean and standard deviation of everything observed up to the
end of that patch, squashed with `asinh`, and embedded. The sequence holds the context
patches followed by learned forecast tokens ("Decoupled Horizon Forecasting"): each forecast
token reads the whole context and the earlier forecast tokens, and a head maps it to 21
non-crossing quantiles for each of its 32 steps. Layers alternate attention over time with
attention across the variates of a series. Covariate rows are read by the main-series rows
but never read them. One forward pass covers up to 32 forecast patches (1,024 steps); longer
horizons are produced block by block, feeding each block's median back as context. The
released forecaster also averages the forecast of the series with the reflected forecast
of its negation ("symmetric averaging") and clips at zero when the context is non-negative.
We use the released code and weights with these defaults throughout.

Because a forecast block is causal across its forecast tokens, a forecast's first `h`
steps do not depend on the requested horizon beyond `h`. We checked that slicing a
1,024-step forecast reproduces separate shorter forecasts to float precision
(`tests/test_harness.py`), and evaluate all horizons up to the longest from one call.

## 2.2 References: what counts as a failure

A forecast error is only a failure relative to something that could have done better. We
use two references:

- **True-dynamics oracle.** Every generator is a simulator with an explicit state. To build
  a test case we simulate a warmup and the context, copy the final state 501 times, and
  advance every copy over the horizon with fresh noise. Copy 0 is the realized future; the
  other 500 form the oracle's predictive distribution, from which we take the same 21
  quantiles Timestral outputs. The oracle knows the equations, the parameters and the latent
  state at the forecast origin, so it is the best achievable forecast given that state (up
  to Monte Carlo error) and slightly stronger than anything that sees only the noisy
  observations. Deterministic systems get small dynamic noise and observation noise so the
  oracle's spread grows with the horizon instead of being zero.
- **Classical models.** Seasonal naive, ETS (additive error, damped trend, additive season
  when the period allows) and ARIMA (small order grid chosen by AIC), each turned into the
  same 21 quantiles from its Gaussian forecast distribution, fitted on the last 1,024 steps.

For every series we report the ratio of Timestral's error to the oracle's and to the best
classical model on that series, and the gap (difference) to the oracle.

## 2.3 Metrics

- **MASE**: mean absolute error of the median, divided by the in-sample mean absolute error
  of the seasonal naive forecast on the context.
- **sCRPS**: the continuous ranked probability score approximated by the average pinball
  loss over the 21 quantile levels (times 2), divided by the same scale as MASE.
- **Coverage and calibration**: the fraction of realized values inside the 80% and 98%
  intervals, and the mean absolute difference between each quantile level and the fraction
  of realized values below that quantile.
- **WQL** (GIFT-Eval's weighted quantile loss on levels 0.1 to 0.9), reported for the
  leaderboard only, since it divides by the series' magnitude and breaks near zero.

Ratios are aggregated over seeds by geometric mean, with 95% bootstrap intervals.

# 3. Signal families

Each family is a set of simulators with a difficulty knob (full definitions in
`otp/generators/`):

- **Chaos and dynamics**: Lorenz-63 (knob ρ through the onset of chaos; also the sampling
  interval), Rössler (knob c through period doubling), Mackey-Glass (knob: delay τ), the
  logistic map (knob r through the onset of chaos at about 3.57), and Kuramoto coupled
  oscillators (knob: coupling K through the synchronization threshold).
- **Regime switches**: jumps in mean, variance or seasonal frequency, placed inside the
  context at a chosen distance before the forecast origin or occurring in the horizon at a
  known hazard rate; and a Markov-switching AR process (knob: switching probability).
- **Heavy tails and fractal**: AR(1) and random walks with α-stable innovations (knob α),
  fractional Brownian motion (knob: Hurst exponent), a self-exciting (Hawkes) count process
  (knob: branching ratio), and a Markov-switching multifractal cascade (knob: number of
  multipliers).
- **Learned mixture of primitives**: a genome of four slots, each choosing one component
  (trend, season, AR noise, or any generator above), a weight, an additive or multiplicative
  role and three parameters, plus a global noise level (`otp/generators/mixture.py`). Search
  methods evolve genomes to maximize Timestral's error relative to the oracle while
  requiring the oracle to beat seasonal naive by a margin, so the series stays forecastable.
- **Controls**: in-prior series (random trend + season + AR noise, the kind of data Timestral
  was pretrained on).

# 4. Failure map

RESULTS-PENDING.

# 5. Covariates and one-way attention

RESULTS-PENDING.

# 6. Searching for failures

RESULTS-PENDING.

# 7. Why it fails

RESULTS-PENDING.

# 8. Benchmarks

RESULTS-PENDING.

# 9. Fixing it

RESULTS-PENDING.

# 10. Limitations

RESULTS-PENDING (compute actually used, what ran on CPU vs. GPU, what is not run).

# Reproducibility

All code, generator seeds and result files are in the repository. Every number above
points to the result file it comes from; every result file has a `provenance.json` naming
the script, the commit, the host and the wall time.
