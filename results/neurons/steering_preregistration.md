# Steering pre-registration

Committed before any steering run. The predictions below are fixed; the matching criterion is fixed.

## What scaling does

For a pole (neuron, sign) and a scale c, `z = gate * silu(up)` of that neuron is multiplied by c on every token where `z` has the pole's sign, in the plain forward pass (no symmetric averaging). Scales: [0.0, 0.5, 2.0, 4.0] (0 is ablation, 1 is the unsteered pass). Effects are differences between the steered and unsteered 128-step median / quantile forecast, divided by the context standard deviation.

## Quantities

- level: mean over the horizon of the median forecast
- slope: OLS slope of the median forecast times 128
- spread: log ratio of the mean 5%-95% interval width
- amp: amplitude of the sinusoid at the series' known period fitted to the median forecast (seasonal test series)
- phase: phase of that sinusoid, circular difference in radians (seasonal test series)

## Predictions by label

Direction `+` means: increasing the pole moves Q in the direction in which the label variable moves it (sign of the correlation between the pole's activation and the label variable on label-set tokens); `-` the opposite; `0` any direction (monotone in scale is still required).

| predicted quantity | direction | labels |
|---|---|---|
| level | + | lat_level, m_dlevel_prev, m_level_ctx, prev_level |
| slope | + | lat_slope, m_slope, prev_slope, s_trend, x_trend |
| spread | + | lat_inn_sd, lat_noise_sd, lat_outlier, lat_outlier_mag, m_jump, m_logsd, m_max_z, m_resid_sd, m_roughness, prev_noise_sd, s_noise_level, x_noise_level |
| spread | - | lat_pred1, s_alpha, x_tail |
| amp | + | s_has_season, s_season_amp, x_season_strength |
| phase | 0 | lat_phase_cos, lat_phase_sin, prev_phase_cos, prev_phase_sin, x_phase_cos, x_phase_sin |
| level|spread | 0 | lat_regime, lat_regime_frac, prev_regime, x_post_cp |

Labels not listed (periods, family and generator indicators, position, persistence, Lyapunov, clustering, coupling, motif shapes without a direct forecast quantity) have no prediction; their steered poles are reported without a match decision.

## Matching criterion

A steered pole matches its pre-registered prediction when all three hold for the predicted quantity Q: (1) monotone: |Spearman rho| between the scale list [0.0, 0.5, 2.0, 4.0] and the mean effect on Q is at least 0.8, with the sign of rho equal to the predicted direction when a direction is given; (2) size: |effect(scale 4) - effect(scale 0)| is at least the threshold of Q (level 0.05, slope 0.05, spread 0.05, amp 0.05, phase 0.1); (3) the 95% bootstrap interval (500 resamples of test series) of effect(scale 4) - effect(scale 0) excludes 0. For the label group with quantity 'level|spread' the three conditions are checked for each of the two quantities and the pole matches if either passes (direction not prescribed).

## Pole sample

Per model size, from the pole table of the main-context run. Candidates: live poles that are one-job at theta 0.7 and floor 0.1 (explained share of the explainable variance, see `otp/neurons/score.py`) whose best label is in the table above. Cells: layer third (early / middle / late) x predicted-quantity group (level, slope, spread, amp, phase, level|spread). In each cell 8 candidate poles are drawn uniformly at random (seed 0), not by score; a cell with fewer candidates contributes all of them. For the first 4 poles of each cell the other pole of the same neuron is steered as well (negative-pole consistency, below).

## Control

144 control poles per size are drawn uniformly at random from the live poles of the size, whatever their label (unexplained poles included). Each control pole is judged with the same criterion for each of the quantities level, slope and spread, with a random direction (sign drawn once from seed 0). The fraction of (pole, quantity) pairs that pass is the null match rate reported next to the match rate of the labeled poles.

## Test series

Held-out series (never used for labeling) of the audit corpus, context 1024 steps. 96 series drawn at random from all held-out univariate series for level, slope and spread; 64 held-out series with a seasonal component of period 8..64 steps for amp and phase. Fixed seed.

## Consistency checks (reported next to the match rate)

- Downstream: the effect of ablation (scale 0) on each of the five quantities is stored for every steered pole. A label is flagged as broken by this check when the pole matches no prediction and its largest ablation effect, in units of that quantity's size threshold, is on a quantity other than the predicted one and is at least 1.
- Negative-pole consistency: for neurons with both poles steered, whose two poles have the same best label variable, the scale-4 effects of the two poles on the predicted quantity should have opposite signs when the two label correlations have opposite signs, and equal signs when they have the same sign (each pole's predicted direction follows the sign of its own label correlation). The share of such neurons that satisfy this is reported.
- Upstream: the three earlier-layer poles most correlated with the pole on a fixed subsample of tokens (`scripts/neurons_match.py`); a label is flagged when none of them has a label in the same group and the best motif label of the pole explains less than half of its label's R^2.
