---
title: "Off the Prior: Mapping Where a Synthetic-Prior Time-Series Foundation Model Fails, and Why"
author: "Yousef Anas (Cúper)"
date: "October 2026"
abstract: |
  Timestral (Mistral, October 2026; 5M to 209M parameters) is a time-series foundation model
  pretrained mostly on synthetic trend + season + noise series. We map where its forecasts fall
  behind the best achievable forecast, using simulators whose true state is known: a Monte Carlo
  continuation from that state serves as an oracle, so a large error counts as a failure only when
  the oracle does better. On series like the pretraining data, all four sizes stay at
  {{num:inprior_short_min}}–{{num:inprior_short_max}}× the oracle's MASE up to horizon 128; heavy tails, long
  memory, bursts and multifractal cascades stay at {{num:tails_min}}–{{num:tails_max}}×. The gaps are on
  smooth, nearly deterministic dynamics (Mackey-Glass at τ = 30: {{num:chaos_mg30_5M}}× at 5M,
  {{num:chaos_mg30_209M}}× at 209M; for Lorenz the gap is confined to the first Lyapunov times) and on
  recent regime switches ({{num:reg_mj4_16_5M}}× at 5M and {{num:reg_mj4_16_209M}}× at 209M sixteen steps after a
  level jump). Linear readouts of the forecast tokens recover the new level after a jump with
  R² {{num:ro_mj16_ro_min}}–{{num:ro_mj16_ro_max}} while the forecast reaches {{num:ro_mj16_out_min}}–{{num:ro_mj16_out_max}}: the change
  is represented but not used. A nearest-neighbour forecaster built from the context beats every size
  on the delay and smooth-chaos systems, and a training-free switch that backtests the two on the
  context lowers chaos errors to {{num:sv_chaos_5M}}× (5M) and {{num:sv_chaos_209M}}× (209M) of Timestral's on fresh
  samples, with in-prior and regime series unchanged and tails at most {{num:sv_tails_max}}×. We release two benchmarks with oracle quantiles
  ({{num:bench_n_diag}} tagged diagnostic series; {{num:bench_n_lb}} leaderboard series), on which Timestral-209M
  is at {{num:bench_diag_209M_oracle}}× and {{num:bench_lb_209M_oracle}}× the oracle's MASE.
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

All ratios below are the geometric mean over seeds of Timestral's MASE divided by the
oracle's MASE on the same series, context and horizon (1.0 = as good as the true dynamics).
"vs best classical" divides by the best of seasonal naive, ETS and ARIMA on each series
instead. Unless stated otherwise: context 2,048 steps, horizon 128 steps, 8 seeds per setting.

## 4.1 Control: series like the training data

On random trend + season + AR-noise series (the kind Timestral was pretrained on), every size
stays within {{num:inprior_short_min}}–{{num:inprior_short_max}} of the oracle at horizons up
to 128 steps. The gap grows at long horizons:

{{table:inprior}}

Part of that long-horizon gap is not a prior failure: the oracle knows the exact trend slope
and seasonal shape, while any forecaster must estimate them from the context, and estimation
error in a slope grows linearly with the horizon. Family results are read against this control.

## 4.2 Chaos and dynamics

![MASE ratio to the oracle along each chaos knob (context 2,048, horizon 128; shaded: 95% bootstrap interval over 8 seeds; dashed: as good as the oracle).](figures/chaos_curves.png)

{{table:chaos_main}}

Timestral beats the best classical model on most of these systems (the 209M model's median
ratio to the best classical model across the chaos configs is
{{num:chaos_209m_vs_classical_median}}) and is still far from the oracle on smooth, nearly
deterministic oscillators: Mackey-Glass with long delays ({{num:chaos_mg30_5M}} at 5M,
{{num:chaos_mg30_209M}} at 209M for τ = 30), weakly coupled Kuramoto oscillators
({{num:chaos_kur1_5M}} and {{num:chaos_kur1_209M}} at K = 1), and the Rössler system
({{num:chaos_ros9_5M}} and {{num:chaos_ros9_209M}} at c = 9). Size matters for some of these
and not others: strongly coupled Kuramoto goes from {{num:chaos_kur8_5M}} at 5M to
{{num:chaos_kur8_209M}} at 209M, while the densely sampled Lorenz system stays at
{{num:chaos_lor_oe2_5M}} (5M) and {{num:chaos_lor_oe2_209M}} (209M). The chaotic logistic map
(r = 3.9), whose predictability is gone after a few steps, stays near 1 at every size.

**The gap lives inside the predictability horizon.** A Lyapunov time is how long it takes a
small error in a chaotic system to grow by a factor of e (for Lorenz at ρ = 28 the exponent is
{{num:lorenz_lyap}} per time unit). Putting four sampling intervals of the same Lorenz system
on that common axis shows that the ratio is set by how many Lyapunov times the forecast
reaches, not by the step count:

![Lorenz at four sampling intervals on one axis: forecast reach in Lyapunov times.](figures/lorenz_lyapunov.png)

{{table:lorenz_lyap}}

Inside one to three Lyapunov times the oracle can still track the trajectory and Timestral
cannot; past about ten, nobody can, and the ratio returns to about 1. The noise level of the
generator sets how good the oracle can be inside that window. Sweeping the noise level
(dynamic and observation noise together) shows how much of each ratio that choice sets:

![Ratio to the oracle against the noise level of the generator (16 seeds, horizon 128).](figures/noise_sweep.png)

{{table:chaos_noise}}

At 0.1% noise the oracle is nearly exact and Mackey-Glass sits at
{{num:noise_mg_lo_min}}–{{num:noise_mg_lo_max}} times its error; at 10% the gap is
{{num:noise_mg_hi_min}}–{{num:noise_mg_hi_max}}. The ratio is therefore a measure of how much
predictable structure Timestral leaves unused, and its size depends on how much predictable
structure the generator provides. The ordering of model sizes holds at every noise level.

## 4.3 Heavy tails and fractal processes

Across α-stable AR processes and random walks, fractional Brownian motion, Hawkes counts and
multifractal cascades, all four sizes stay between {{num:tails_min}} and {{num:tails_max}} of
the oracle:

{{table:tails_main}}

Heavy tails and long memory, which a trend + season + Gaussian-noise description of series
does not cover, do not produce failures of the size seen for deterministic dynamics.

## 4.4 Regime switches

Three kinds of switch (a level jump, a change of seasonal period, a change of noise level) placed
a chosen number of steps before the forecast origin, so the context shows the switch:

![Ratio to the oracle against the number of steps since the switch (context 8,192, horizon 128).](figures/regime_distance.png)

{{table:regime}}

The failure is concentrated right after a switch. Sixteen steps after a level jump of 4 noise
standard deviations, the error is {{num:reg_mj4_16_5M}} times the oracle's at 5M and
{{num:reg_mj4_16_209M}} at 209M; a thousand steps later it is {{num:reg_mj4_1024_5M}} and
{{num:reg_mj4_1024_209M}}. Period changes last longer: 64 steps after the period triples, the
ratio is still {{num:reg_fj3_64_5M}} (5M) and {{num:reg_fj3_64_209M}} (209M). A direct measure of
how much of a level jump the forecast adopts (1 = forecasts at the new level, like the oracle;
0 = still at the old level), over the first 16 forecast steps and pooled over jump sizes, is
{{num:adopt_5M_16}} / {{num:adopt_5M_64}} / {{num:adopt_5M_256}} at 16 / 64 / 256 steps after the
switch for 5M and {{num:adopt_209M_16}} / {{num:adopt_209M_64}} / {{num:adopt_209M_256}} for 209M.
Section 7.3 shows that the change is nevertheless represented inside the model.

When the switch happens during the horizon at a known rate (so nobody can see it coming),
Timestral stays at {{num:reg_horizon_min}}–{{num:reg_horizon_max}} of the oracle, and on
Markov-switching AR processes at {{num:reg_markov_min}}–{{num:reg_markov_max}}.

## 4.5 Does more history help?

All four sizes with contexts up to 32,768 steps (Colab job B, 28 settings × 8 seeds):

{{table:long_ctx}}

More history helps on Mackey-Glass (τ = 17: {{num:long_mg17_5M_T2048}} → {{num:long_mg17_5M_T32768}}
for 5M and {{num:long_mg17_209M_T2048}} → {{num:long_mg17_209M_T32768}} for 209M from 2,048 to
32,768 steps) and slightly on Rössler and Lorenz; it does not help on the in-prior control,
which gets slightly worse at 32,768 steps. {{num:long_nonfinite_oracle_samples}} samples
(Rössler, c = 5.7) had oracle paths that left the attractor at long horizons; their oracle
ratios are excluded.

## 4.6 Real series that stand in for the families

Real data has no oracle, so it cannot show a gap to the best achievable forecast; it can show whether
the families' pattern against classical models carries over. We gathered open-license series
(sources, licenses and hashes in `bench/regression/sources.csv`; windows whose license does not allow
redistribution are kept out of git and rebuilt by `scripts/fetch_realdata.py`). On {{num:rd_n_real}} windows
from GIFT-Eval and fev-bench datasets, Timestral's MASE relative to seasonal naive is {{num:rd_real_5M}} at 5M
and {{num:rd_real_209M}} at 209M (ETS {{num:rd_real_ets}}, ARIMA {{num:rd_real_arima}}; GIFT-Eval-style geometric mean over
dataset groups). These datasets may overlap Timestral's pretraining data.

A separate subset of {{num:rd_n_analogues}} windows was picked as real analogues of the failure families:
sunspot numbers, lynx and hormone series and El Niño indices for delay feedback; grid frequency,
pedestrian counts and activity recordings for coupled oscillators; river discharge, earthquake counts,
financial and crypto series for heavy tails; macro and energy series with documented breaks, traffic
and COVID-19 cases for regime switches. The table compares Timestral's ratio to the best classical
model on these windows with the same ratio on the synthetic family (context 2,048, horizon 128):

{{table:rd_analogue}}

On the synthetic delay and oscillator systems Timestral beats the classical models by a wide margin
while staying far from the oracle (Section 4.2); on the real analogues its margin over classical models
is smaller. Heavy tails and regime switches sit near 1 on both. With no oracle on the real side, these
numbers do not say how far Timestral is from the best achievable forecast there.

# 5. Covariates and one-way attention

Timestral reads covariates as conditioning rows: the main series reads them, they never read the
main series. Several main rows given together read each other both ways. We built six coupled
systems with known equations (`otp/generators/coupled.py`): an exogenous driver whose past sets the
main series after a lag (LaggedDriver), the same through a nonlinearity (NonlinearDriver), linear and
predator-prey feedback loops, a hidden driver seen through a noisy proxy (HiddenDriver), and a
known-future covariate that is a noisy copy of the main series shifted by one step (LeakyKnown). The
main series is forecast four ways: alone, with the other channel as a past-only covariate, as a
known-future covariate where that makes sense, and as a second main row ("joint"). The oracle is
matched to the information each condition gives. Sizes 5M and 28M ran on the full grid (5 knob values,
8 seeds, horizons 16 / 128 / 1,024); 123M and 209M on a reduced grid (3 knob values, 4 seeds, horizons
16 / 128), so the oracle gains below differ slightly between the two pairs of sizes. Details and all
tables: `results/covariates/README.md`.

**Past covariates are used partly, and more with size.** When the driver's past determines the main
series over the whole horizon, a forecaster that used it fully would cut the error several-fold. Each
cell below gives Timestral's covariate gain (MASE alone / MASE with the past covariate) and the matched
oracle's gain, at context 2,048 and horizon 16:

{{table:cov_past}}

At lag 32, 209M reaches a gain of {{num:cov_lag32_209M_gain}} out of {{num:cov_lag32_209M_ogain}} available.

**Known-future covariates help only when the relation is immediate.** With the driver's future given,
the 28M gain on LaggedDriver falls from {{num:cov_known_lag0_28M}} at lag 0 to {{num:cov_known_lag512_28M}} at lag 512, while the
matched oracle's gain stays at {{num:cov_known_ogain_min}}–{{num:cov_known_ogain_max}} (horizon 16):

{{table:cov_known_lag}}

![Gain from a known-future driver (MASE alone / MASE with the covariate) on LaggedDriver against the lag, for Timestral (solid) and the oracle that reads the same covariate (dashed); context 2,048, horizon 16, 8 seeds.](figures/covariate_lag.png)

When the known covariate is the main series itself, one step late with small noise (LeakyKnown at
noise ratio 0.05), Timestral's error is several times the oracle's that reads it (MASE over matched-oracle
MASE):

{{table:cov_leaky}}

**One-way routing is not the bottleneck.** Giving the same channel as a second main row (two-way
attention) instead of a past covariate (one-way) changes the error little: joint/past-covariate MASE is
within 0.95–1.05 in {{num:cov_jp_share_within}}% of the {{num:cov_jp_n}} system × knob × size × horizon cells.
Among the cells whose 95% interval lies below 1, two-way routing gains most on NonlinearDriver
with the nonlinearity off (123M; `results/covariates/summary.csv`), the predator-prey loop
({{num:cov_jp_FeedbackPredPrey_min}}, 95% interval {{num:cov_jp_FeedbackPredPrey_min_lo}}–{{num:cov_jp_FeedbackPredPrey_min_hi}}) and
LaggedDriver ({{num:cov_jp_LaggedDriver_min}}, {{num:cov_jp_LaggedDriver_min_lo}}–{{num:cov_jp_LaggedDriver_min_hi}}); on the linear feedback loop
the ratio stays within {{num:cov_jp_FeedbackLinear_min}}–{{num:cov_jp_FeedbackLinear_max}}. The gap to the oracle comes from how much
of the covariate's information is used at all, not from which way it is routed.

# 6. Searching for failures

Sections 4 and 5 test failure families we chose by hand. Here a search procedure chooses. The search space
is the mixture-of-primitives generator (`otp/generators/mixture.py`): a genome of numbers in [0, 1] decodes
into four component slots (each a trend, season or noise piece, or one of the chaos, regime, heavy-tail or
fractal generators, with a weight, a mode of combination and its own parameters) plus an observation-noise
level. Every genome is a simulator with an exact Monte Carlo oracle, so every candidate can be scored as in
Section 4.

**Objective.** The fitness of a genome is the geometric mean, over a few fixed seeds, of MASE(Timestral-5M) /
MASE(oracle) at context 2,048 and horizon 128 (`otp/search/objective.py`). A genome counts only if the series
it makes is forecastable by the oracle: the oracle's sCRPS must be well below seasonal naive's on the same
samples (a fixed fraction set in the objective), values must be finite and of sane scale. Infeasible genomes
score zero. Every Timestral forward call is counted, and results are cached by genome.

**Methods, at equal budget.** Each method gets the same number of Timestral calls and three search seeds.
(1) CMA-ES over the bounded genome. (2) MAP-Elites, a quality-diversity search: two behaviour descriptors (the
dominant component family and a spectral statistic of the context) define a grid of cells, and each cell keeps
its best genome, so the archive covers different kinds of failure instead of one peak. (3) Gradient search:
a differentiable re-implementation of part of the generator (sinusoids, trend, AR noise with fixed noise
draws, a smooth regime gate) is pushed by gradients through Timestral's differentiable forward pass to raise
its pinball loss above the oracle's; the final parameters are then scored with the standard pipeline, so all
three methods are compared on the same metric. A separate input-space study finds, for in-prior series, the
smallest perturbation of the context that doubles Timestral's MASE on a fixed future; it measures sensitivity,
not a generator failure, since the oracle is unaffected.

{{table:search_comparison}}

**Transfer.** The top elites of each method are re-scored on 28M, 123M and 209M with fresh seeds, to see whether
failures found on the smallest model are failures of the family or of one size.

{{table:search_transfer}}

# 7. Why it fails

## 7.1 Two suspects that turned out innocent

**Symmetric averaging.** The released forecaster averages its forecast of the series with the
mirrored forecast of the negated series. That forces a symmetric treatment of asymmetric
processes (bursty counts, one-sided maps), so it was a natural suspect. Turning it off changes
the sCRPS ratio to the oracle by a factor between {{num:symavg_min}} and {{num:symavg_max}}
across 14 settings and both small sizes: it is not where the failures come from.

**Cumulative normalization.** Timestral normalizes every patch with the mean and spread of
everything since the start of the context, so old history keeps weighing on the scale after a
regime switch. After a variance drop 64 steps back, intervals are indeed far too wide (80%
intervals cover at least {{num:norm_vdrop_cov80_min}} of points; sCRPS ratio
{{num:norm_vdrop_scrps_min}}–{{num:norm_vdrop_scrps_max}}). But cutting the context down to
the post-switch part, which removes the old history from the statistics entirely, lowered the
sCRPS ratio in only {{num:norm_post_better}} of {{num:norm_cells}} cells (those with at least
256 post-switch steps) and raised it in the rest. The anchoring is not the main cause.

## 7.2 Template or retrieval?

For a deterministic system, the context itself is a library of past trajectories: find the
moments whose recent history looks most like the present and see what happened next (the
"method of analogues", a few lines of nearest-neighbour code with no training). A second
context-only forecaster continues the series as a periodic pattern (the average of its last
cycles at the dominant period). On the chaos family (MASE, lower is better):

{{table:analog}}

The method of analogues beats every Timestral size on Mackey-Glass and Rössler and loses to
Timestral on weakly coupled Kuramoto and on Lorenz. Timestral's forecasts correlate more with the analogue forecast
than with the periodic template, but lean toward the template more than the oracle does
(`results/mechanism/analog/summary.csv`). Our reading: Timestral uses the context partly as a
trajectory library and partly as a pattern to repeat, and on smooth nonlinear oscillators,
which its trend + season + noise prior represents as near-periodic, the second pull wins.

## 7.3 Inside the model: represented but not used

For each setting we ran one plain forward pass on 300 samples and asked two questions about
the oracle's expected change over the first forecast patch (and at single steps 1 and 16):
how well does Timestral's own forecast predict it, and how well does a linear readout (a
ridge regression, cross-validated) of the hidden state of the first forecast token predict
it, at the best layer? If the readout does much better than the output, the information the
oracle uses is inside the model but does not reach the forecast. The readout is a lower bound
on what is linearly present: with 300 samples and 192–768 features per token, the ridge fit
loses some signal, which shows on the in-prior control, where it scores
{{num:ro_ip_ro_min}}–{{num:ro_ip_ro_max}} against {{num:ro_ip_out_min}}–{{num:ro_ip_out_max}}
for the output. Gaps in the readout's favour are therefore conservative.

![How well the oracle's expected change can be predicted from Timestral's forecast (solid) and from a linear readout of its first forecast token (dashed), against the time since a mean jump.](figures/readout_vs_output.png)

{{table:readouts}}

**Recent regime switches: represented, attended to, not used.** Sixteen steps after a mean
jump, the readout reaches R² {{num:ro_mj16_ro_min}}–{{num:ro_mj16_ro_max}} at every size while
the forecast itself reaches {{num:ro_mj16_out_min}}–{{num:ro_mj16_out_max}}, and the forecast
error is {{num:ro_mj16_ratio_min}}–{{num:ro_mj16_ratio_max}} times the oracle's. The forecast
tokens are not looking elsewhere: they put {{num:ro_mj16_post_min}}–{{num:ro_mj16_post_max}}
times their proportional share of attention on the post-switch patches. Sixty-four steps
after a frequency jump, one step ahead, the readout reaches {{num:ro_fj64_s1ro_min}}–{{num:ro_fj64_s1ro_max}}
against {{num:ro_fj64_s1out_min}}–{{num:ro_fj64_s1out_max}} for the output. By 256 steps after
a mean jump the two agree ({{num:ro_mj256_out_min}}–{{num:ro_mj256_out_max}} vs.
{{num:ro_mj256_ro_min}}–{{num:ro_mj256_ro_max}}). The model sees and encodes a recent change and
then forecasts as if it may not last. Our interpretation, which we cannot check without the
pretraining data: if lasting breaks are rare in the synthetic trend + season + noise mix, a very
recent departure from the pattern is better explained as noise under that prior than as a new
regime.

**Deterministic oscillators: the gap builds later.** On Mackey-Glass (τ = 17) output and readout
agree for the first patch ({{num:ro_mg17_out_min}}–{{num:ro_mg17_out_max}} vs.
{{num:ro_mg17_ro_min}}–{{num:ro_mg17_ro_max}}): the near-term forecast uses what is there, and the
error accumulates over the following cycles. There the forecast correlates with a periodic
continuation of the context more than the oracle's forecast does
({{num:ro_mg17_tm_5M}} at 5M, {{num:ro_mg17_tm_209M}} at 209M, against
{{num:ro_mg17_to_min}} for the oracle), and the excess shrinks with size together with the error.

# 8. A neuron atlas

Section 7 looked at a few readouts and attention patterns. This section looks at every MLP neuron of every
size. Timestral's MLP is SwiGLU, so a neuron's value (gate times up-projection) takes both signs; each neuron
is studied as two poles, one per sign.

**Corpus.** A labelled corpus of series from the project's generators: in-prior trend, season and noise with
controlled parameters, and the chaos, regime, tails and mixture families. For every series the generating
variables are stored, and for every 32-step patch the patch-level variables that can be computed exactly from
the simulator: local level, slope and noise, seasonal phase, regime indicator, steps since the last switch,
whether the patch holds an outlier, and the oracle's one-step predictability. A seeded split of series is
held out and never used for labelling. A mixed-domain set of open-license real windows is added for a
transfer test (`results/neurons/real/`).

**Labels.** For each pole, the known variable (or simple input motif) that best explains its activation on the
tokens where that pole is active is found by mutual information and cross-validated linear readout R²
(single variable and best pair). Each pole stores its best label, score, runner-up and the gap between them.

**One job or many.** A pole has one main job if its best single label explains at least a fraction θ of the
explainable variance, and is unexplained if no label clears a floor. The fractions of one-job, mixed and
unexplained poles are reported as curves over θ, per layer and size, never at a single threshold.

{{table:nn_one_job}}

**Checks.** Predictive: on the held-out series, each pole's activation is predicted from its label alone and
the held-out R² is compared with the labelling-set score. Causal: for a stratified sample of poles, the pole is
scaled during the forward pass and the change in the forecast is compared with a prediction written down, and
committed, before any steering run (`results/neurons/steering_preregistration.md`).

{{table:nn_steering}}

**Link to the failure map.** Which poles are active on the Mackey-Glass, Rössler and Kuramoto contexts where
Timestral falls behind the oracle, whether seasonal or periodic-pattern poles are among them, and whether
damping them moves the forecast toward the oracle or toward the analogue forecaster of Section 7.2.

{{table:nn_failure_link}}

# 9. Benchmarks

The failure map is released as two benchmark sets built from the same simulators (code and data in
`bench/`). Each series comes with the oracle's 21 quantiles, so a model is scored against the best
achievable forecast as well as against seasonal naive, the GIFT-Eval convention.

**OffPrior-Diag** has {{num:bench_n_diag}} series: {{num:bench_nconf_diag}} configs with 12 seeds each, at
context 2,048 and horizon 128. Every config was picked to cover one failure type from Sections 4 and 7,
or to serve as a control (inside the prior, an old switch, a switch inside the horizon that nobody can
see coming, a map nobody can forecast); each series carries its failure type and a one-line reason.
**OffPrior-LB** has {{num:bench_n_lb}} series: every config of the chaos, regime, tails and in-prior
grids, 40 fixed random mixture genomes and four more in-prior periods ({{num:bench_nconf_lb}} configs), 16 seeds per config spread
over contexts 512 / 2,048 / 8,192 and horizons 16 / 128 / 1,024, split into public and held-out halves by
seed parity. The seeds and code are public, so the held-out half is a convenience, not a secret test set.
`bench/eval.py` scores one model in one command; the summary is the geometric mean over series of MASE
and sCRPS divided by seasonal naive's, and of MASE divided by the oracle's.

{{table:bench_diag}}

{{table:bench_lb}}

On OffPrior-Diag, Timestral's MASE is {{num:bench_diag_209M_oracle}}× the oracle's at 209M and
{{num:bench_diag_5M_oracle}}× at 5M, against {{num:bench_diag_ets_oracle}}× for ETS,
{{num:bench_diag_arima_oracle}}× for ARIMA and {{num:bench_diag_seasonal_naive_oracle}}× for seasonal naive.
On OffPrior-LB the same ratios are {{num:bench_lb_209M_oracle}}× and {{num:bench_lb_5M_oracle}}× for Timestral,
and {{num:bench_lb_ets_oracle}}×, {{num:bench_lb_arima_oracle}}× and
{{num:bench_lb_seasonal_naive_oracle}}× for the classical models. By family on OffPrior-LB, 209M is at
{{num:bench_lb_fam_chaos}}× the oracle on chaos, {{num:bench_lb_fam_mixture}}× on the mixture genomes and
{{num:bench_lb_fam_tails}}× on tails.

{{table:bench_diag_ft}}

By failure type, the diagnostic set reproduces the map at 209M: coupled oscillators
({{num:bench_ft_coupled_oscillators}}×), delay oscillators ({{num:bench_ft_delay_oscillator}}×), chaos within the
first Lyapunov times ({{num:bench_ft_chaos_short_reach}}×), smooth chaos ({{num:bench_ft_smooth_chaos}}×) and
recent switches ({{num:bench_ft_recent_switch}}×), while the controls and the heavy-tailed, bursty and
multifractal types stay at {{num:bench_ft_ctrl_min}}–{{num:bench_ft_ctrl_max}}×.

The published files were built on a Windows laptop. A rebuild on Linux with newer NumPy matched them
array-for-array for {{num:bench_xp_same}} of {{num:bench_xp_n}} diagnostic configs; in the others
(Mackey-Glass, Kuramoto near synchronization, and one fractional Brownian motion setting) platform
floating-point differences change the simulated paths, and the chaotic ones diverge. Scores should
therefore be reported on the published files (`bench/data/cross_platform_diag.json` has the per-config
comparison).

# 10. Fixing it

## 10.1 Without training: let the context decide

Section 7.2 showed that a nearest-neighbour forecaster built from the context beats Timestral
on some deterministic systems and loses badly on others. A forecaster can find out which case
it is in from the context alone: forecast the last stretch of the context from what comes
before it, with both methods, and compare. The rule we test switches to the analogue forecast
(keeping Timestral's quantile spread around the new median) only when the analogues' backtest
error is below 0.7 times Timestral's on each of the last two 128-step windows.

How the rule was chosen matters. A first pre-set procedure (tune on even seeds, report on odd
seeds) picked a one-window rule that helped chaos but cost the tails family
{{num:sv_firstrule_tails_min}}–{{num:sv_firstrule_tails_max}} on the held-out seeds. We then
picked the two-window rule from that held-out table, which is a choice made after seeing
it, so we froze it and evaluated it once more on fresh samples generated for that purpose
(`results/fix/README.md` lists all three stages). On the fresh samples (MASE of the fix divided
by Timestral's MASE; below 1 is better):

{{table:selfval}}

The fix improves chaos by a factor of {{num:sv_chaos_5M}} at 5M and {{num:sv_chaos_209M}} at
209M, leaves in-prior and regime series untouched (the rule never switches there), and costs at
most a factor of {{num:sv_tails_max}} on tails. Its benefit shrinks with model size, in line with
larger models closing more of the chaos gap on their own.

The other zero-training idea, cutting the context to the part after a detected change
(Section 7.1), helps only when at least 256 post-switch steps are available.

## 10.2 Fine-tuning: full, LoRA, and data mixing

The direct fix is to train on the failures. We compare full fine-tuning (every weight), LoRA (low-rank adapters
on the attention and MLP projections, base weights frozen) and data mixing (a share of every batch replaced by
in-prior trend, season and noise series, on top of either method), with training code in `otp/finetune.py`
(pinball loss over the 21 quantiles, scaled by the context's naive error; AdamW with warmup and cosine decay).
Two held-out protocols keep the test away from the training distribution: leave-one-family-out (train on three
of chaos, regime, tails and mixture, test on the fourth) and held-out knob ranges (train on one range of each
family's knob, test on the other; splits in `otp/finetune_splits.py`). A regression check scores the
fine-tuned models on normal series (the in-prior set and, where available, the real-data set) against the
pretrained model of the same size. Gain is reported as the paired MASE ratio fine-tuned / pretrained on the
held-out samples (500-path oracle), with 95% intervals, next to cost (trainable parameters, steps, series
seen, wall time).

A first pass on the 5M model ran on CPU with a short budget, to validate the pipeline end to end:

{{table:ft_local_lofo}}

{{table:ft_local_knob}}

{{table:ft_local_regression}}

The full-budget runs on all four sizes ran on GPUs:

{{table:ft_colab}}

# 11. Limitations

- **The oracle knows the latent state.** It is the best forecast given the true state, which
  is stronger than the best forecast given only the observed history. Ratios to it therefore
  bound the possible improvement from above; part of every gap is unavoidable for any
  forecaster that sees only the data.
- **Noise sets the size of chaos gaps.** For nearly deterministic systems the oracle's error,
  and so every ratio to it, depends on the noise level we chose (1% by default); Section 4.2
  reports the dependence. Orderings (between settings and between model sizes) are more
  robust than the magnitudes.
- **Seeds.** Most settings use 8 seeds (16 for the noise sweep, 300 samples per setting for
  the mechanism runs). Intervals in the result files are bootstrap intervals over seeds; with
  8 seeds they are wide for heavy-tailed settings.
- **Diverged simulations.** A few Rössler simulations left the attractor under noise; their
  oracle could not be scored and they are excluded (counts are given where they occur).
- **Univariate main series.** Except for the covariate study, every test series is a single
  variate.
- **No access to the pretraining data.** Statements about what the synthetic prior contains
  are inferences from behaviour, labelled as such.
- **Compute.** Small-model sweeps ran on a 2-core CPU VM; the large-model sweeps, the
  long-context runs and the mechanism runs at all sizes ran on Colab A100 GPUs; one mechanism
  test ran on a laptop RTX 3060 GPU. GPU and CPU forecasts of the same series agree to a few
  parts in 100,000 in MASE (`results/mechanism/analog/parity_vm_cpu_vs_laptop_gpu.json`).

# Reproducibility

All code, generator seeds and result files are in the repository. Every number above
points to the result file it comes from; every result file has a `provenance.json` naming
the script, the commit, the host and the wall time.
