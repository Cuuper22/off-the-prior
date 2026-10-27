# Pre-registration of the neuron-atlas follow-up (parts A and C)

Written and committed before any run of the analyses below. The earlier pre-registration
(`steering_preregistration.md`: prediction table, matching criterion, thresholds, test series) stays in force and is
called the *original criterion* here. Nothing below changes the original criterion or its thresholds; it adds checks.
Part B (corpus and library dependence) was done separately (`followup/README.md`).

Question: can the steering test detect an effect when one is known to exist, and how big must an effect be before a
group of poles or a direction can be told from a random one?

## Definitions shared by A1, A2, A3

- Test series, measures, effects, bootstrap interval and the three conditions (monotone, size, interval) are those of
  `otp/neurons/steer.py` (level, slope, spread, amp, phase; effects divided by the context standard deviation; 500
  bootstrap resamples of test series). Test series: the 96 held-out mixed univariate series and the 64 held-out
  seasonal series (period 8..64) of `scripts/neurons_steer.py` (`test_sets`, seed 0), context 1024, horizon 128.
  Plain forward pass, no symmetric averaging.
- `E(s)` is the mean effect on a quantity at scale (or readout move) s, against the unsteered pass. The statistic that the
  size condition uses is `D = E(top) - E(0)`.
- **Calibrated criterion (added).** A steered object (a single pole, a group of poles, a direction) is *big* when
  `|D|` exceeds the 95th percentile of `|D|` over its matched controls (same model size, same group size or same
  residual block and site, same predicted quantity, same test series, same scale). It *matches under the calibrated
  criterion* when it is monotone (rho condition of the original criterion, same direction rule), big, and its bootstrap
  interval excludes 0. The original criterion is reported next to it, always.
- **Minimum detectable effect (MDE)** = that 95th percentile of `|D|` of the controls, reported next to the fixed
  threshold of the original criterion (0.05; 0.1 for phase).
- A control match rate is the share of controls that pass the same criterion as the labeled objects.

## A1. Positive control by direction steering

Residual stream: the output of transformer block `b` (`model.transformer.layers[b]`), shape (series, tokens, d_model),
tokens = 32 context patches then 4 forecast tokens. Blocks (0-based): early `n//6`, middle `n//2`, late `n-1`, with `n`
the number of blocks (5M: 1, 4, 7; 28M: 2, 6, 11; 123M: 3, 10, 19; 209M: 4, 12, 23).

Readout variables (label-set series of `corpus/main.npz`, `split == 0`; tokens where both `lat_level` and
`lat_noise_sd` are defined):

| name | response | role |
|---|---|---|
| `level` | `lat_level` of the token's patch (local level in context standard deviations; for a forecast token, of the patch it forecasts) | main positive control |
| `noise` | natural log of `lat_noise_sd` of the token's patch | second positive control, predicted quantity `spread` |
| `postjump` | `lat_level` minus the median `lat_level` of the pre-jump patches (`lat_regime_frac == 0`) of the same series; only `mean_jump` series (120 label series, 30 held-out series) | contrasting case: the mechanism study (`results/mechanism/readouts/`) found a post-jump level change represented but not used by the forecast |

Ridge readout (features standardized, intercept free, penalty chosen from 0.01, 0.1, 1, 10, 100, 1e3, 1e4 by 3-fold
cross-validation over the label series' `fold`, refit on all label series). Three token sets per variable and block:
`pooled` (context and forecast tokens), `ctx` (context tokens), `fc` (forecast tokens). Held-out R^2 of every readout
is reported (held-out series, `split == 1`). A readout with held-out R^2 below 0.1 is reported and still steered.

Steering vector for a move `m` of the readout: `v = m * w / (w . w)` with `w` the readout weights mapped back to the raw
residual space, so that the readout of the steered residual changes by exactly `m` at the site and by nothing else on
the readout axis. Moves: `m` in {-2, -1, -0.5, 0.5, 1, 2} (level and postjump in context standard deviations; noise in
natural-log units of the local noise sd, because the label is a log). The criterion is judged on the move list
(0, 0.5, 1, 2) with the original rho threshold (0.8, positive) and `D = E(2) - E(0)`; moves of negative sign are
reported to check antisymmetry.

Sites where `v` is added (to the output of block `b`): `F` all four forecast tokens; `C1` the last context token;
`C4` the last four context tokens. Direction used: the `pooled` direction at every site (the literal spec), and the
matched direction as secondary (`fc` at `F`, `ctx` at `C1` and `C4`).

Predicted quantities: `level` and `postjump` -> `level`, direction `+`; `noise` -> `spread`, direction `+`. All five
measures are stored for every condition (specificity: the level direction should move `level` more than `spread`, and
the noise direction the reverse).

Controls: 32 random directions per (size, block, site, variable): Gaussian in the residual space, rescaled to the
norm of the steering vector of that variable at each of the moves 0.5, 1, 2 (so every random control has the norm of
a real steering vector). Seeds fixed from (size, block, site, variable, index).

Test series: the 96 mixed series for `level` and `noise`; the 30 held-out `mean_jump` series for `postjump`.

**Gain** = `E(2)/2` on the predicted quantity (1 means the forecast moved as much as the readout was moved).

**Verdict rule for the positive control (per model size).** The steering test *detects a known effect* if, for the plain
`level` direction, at least one (block, site, direction kind) cell matches both the original and the calibrated
criterion. If no cell does, the single-pole and group steering results of that size are reported as insensitive and
the MDE tables are the statement of what the test could not see. `noise` and `postjump` get no verdict; they are
reported as second and contrasting cases. Prediction recorded before the run: the `postjump` gain is smaller than the
`level` gain in the cells where both are measured (consistent with the mechanism study), and the `level` direction
moves `level` more than `spread`; the run may contradict both.

## A2. Group steering

Pole pools per model size (`results/neurons/<size>/poles.csv`):

- `one_job` (the spec): live poles that are one-job at theta 0.7 and floor 0.1, whose best label is in the prediction
  table of the original pre-registration.
- `label_top` (extension, because the one-job pool is small): live poles with `r2_cv >= 0.1` whose best label is in the
  table.

Groups: for each pool, label and sign of the pole-label correlation (`label_corr > 0` or `< 0`; signs are never mixed
inside a group), the top `k` poles by `r2_cv` (ties by layer, neuron), `k` in {8, 32, 128}, for every group with at least `k`
poles. A group of positive correlation is predicted to move its quantity in the table's direction; a group of
negative correlation in the opposite direction (direction `0` and `level|spread` as in the table). The pole scales are
those of the original steering (0, 0.5, 2, 4 on the tokens where each pole is active, all poles of the group jointly in
one pass). Test series: seasonal set for labels predicting `amp` or `phase`, mixed set otherwise.

Controls: 16 random groups per labeled group, same `k`, with the layer and sign (positive/negative pole) counts of the
labeled group, drawn from live poles not in the labeled group. Each random group is judged against the prediction of
the labeled group it was matched to. Seeds fixed from (size, pool, label, sign, k, index).

Reported per size: for both criteria the match rate of labeled groups and of random groups (by `k` and pool); the
effect-size distributions `|D|` (median, 75th, 95th percentile) of labeled and random groups; for each labeled group the
share of its 16 random groups with smaller `|D|` (rank); the control 95th percentile pooled over all random groups of
the same (size, `k`, quantity) as the calibration constant of the calibrated criterion.

## A3. Minimum detectable effect

One table (`followup/mde.csv`) per size with, for single poles (from `results/neurons/steering_results.csv`, the 432
control rows per size: 144 random poles x level, slope, spread), groups (`k` = 8, 32, 128) and directions (move 0.5, 1,
2; by block and site): the 95th percentile of `|D|` of the controls, the fixed threshold, and their ratio. For single
poles the same file gives the calibrated-criterion match rate of the labeled poles of the original run (re-judged from
the stored `rho`, `effect_size`, `sig` columns; no new run).

## Sizes and compute

5M and 28M on the laptop GPU lane through the lead; 123M and 209M on Colab (`scripts/colab/run_job.sh`, GPU list
G4,H100,A100, `CODE_ONLY=1`, inputs as extra files, session names starting `otp-nn2-`). Runs write a partial result
table after every chunk of conditions and resume from it.

## C. Motif labels without a variable list

Exemplars: the 8 top and 8 mid-range exemplar tokens of each pole (`top_tok`, `mid_tok` of `scores_partial.csv`),
mapped back to the corpus of the run. The patch of a token is its 32 raw values (a forecast token: the realized patch
it forecasts, the patch the library variables describe).

Normalizations: *shape* (primary): subtract the patch mean, divide by the patch standard deviation (a patch with sd below
1e-9 becomes the zero vector); *scaled* (sensitivity): subtract the context mean, divide by the context sd of the series.

Vocabulary: k-means (k-means++ start, 5 restarts, seed 0) with K = 16 (sensitivity K = 8, 32) on the top-exemplar
patches of all live poles of the four sizes in the main-corpus atlas run (`nn_<size>_main`), pooled over sizes. The
vocabulary is then fixed and every other run (other corpora, subsamples) is assigned to the nearest centroid; a second
analysis refits the vocabulary inside each run. Cluster names: each member patch (shape normalization) is classified by
the first matching rule below and the cluster takes its plurality class (purity reported):

1. `step`: the best two-level piecewise-constant fit with the break between steps 6 and 26 leaves at most 20% of the
   variance and the two levels differ by at least 1.0 shape-sd (`step up` / `step down` by the sign);
2. `oscillation`: at least 4 sign changes of the linearly detrended patch and lag-1 autocorrelation of the detrended
   patch at least 0.3;
3. `rising` / `falling`: linear-fit R^2 at least 0.6 (sign of the slope);
4. `peak` / `trough`: quadratic-fit R^2 at least 0.5 and the extremum of the fit inside steps 6..26 (concave / convex);
5. `flat with noise`: everything else.

Cohesion: a pole is *motif-cohesive* when at least 6 of its 8 top exemplars fall in one cluster (sensitivity: at least
5 of 8; 8 of 8). Reported per model size and run: the share of cohesive poles among live poles, the same for the 8
mid-range exemplars, the share expected if the exemplar cluster ids were permuted across poles of the run (null; 20
permutations), the share of poles whose top exemplars are all context tokens, and the known-variable one-job share of the same
run (theta 0.7, floor 0.1; theta 0.5 and 0.9 in the table), plus the 2x2 agreement of the two labels per pole.

Matched series counts (follow-up B showed that the number of series moves the one-job share): the runs compared are
the main corpus `main.npz` in full (3100 series; `ref_main_full` of follow-up B), subsamples of it with 1034, 620 and
48 series (seeds 0, 1, 2 where available), and the short (1034 series), long (620) and xlong (48) corpora of the
atlas. Pairs for the comparison: short with the 1034 subsample, long with the 620 subsample, xlong with the 48 subsample.

Prediction recorded before the run (the motif share behaves like the one-job share): the cohesive share is larger on the
48-series subsample than on the full main corpus, and the gap between a context-length corpus and its matched
subsample is smaller than the gap between that corpus and the full main corpus.

## Outputs

`results/neurons/followup/`: `direction_readouts.csv`, `direction_steering.csv`, `group_steering.csv`,
`group_summary.csv`, `mde.csv`, `single_pole_calibrated.csv`, `motif_clusters.csv`, `motif_cluster_names.csv`,
`provenance_followup_ac.json`, and sections A and C of `followup/README.md` (exact commands, tables copied from the
files). Code: `otp/neurons/followup.py`, `scripts/neurons_followup_a.py`, `scripts/neurons_followup_c.py`,
`scripts/neurons_followup_report.py`, `tests/test_neurons_followup.py`.
