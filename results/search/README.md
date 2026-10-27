# results/search: learned failure generator, CMA-ES vs MAP-Elites vs gradient search

Facts only. Every number below is copied from the CSV/JSON files in this folder by `scripts/search_report.py`.

## Files

- `runs/<method>_seed<k>/history.csv`: one row per fresh genome evaluation (evaluation index, cumulative Timestral calls, best-so-far, the genome, fitness, feasibility, MASE and sCRPS ratios, cov80, oracle/seasonal-naive sCRPS ratio, descriptors). `elites.json`: top 20 feasible genomes with `describe()` recipes (MAP-Elites also its archive cells). `run.json`: calls, evaluations, wall time. Gradient runs also have `grad_trace.csv` (one row per step and start).
- `comparison.csv`: per method and search seed, plus a `mean` row per method.
- `map_elites_cells.csv`: for every run of every method, the best fitness per cell when all its feasible genomes are binned into the MAP-Elites grid.
- `transfer.csv`, `transfer_summary.csv`, `transfer_spearman.csv`, `transfer.json`, `elites.npz`: transfer to other sizes.
- `input_space.csv`, `input_space_summary.csv`: the input-space sensitivity study (b).
- `provenance.json`: command lines, git commits, hosts, wall times.

## Terms

- Genome: 25 numbers in [0, 1] that decode (`otp/generators/mixture.py`) into a mixture of four component slots (trend, season, AR noise, chaos, regime and heavy-tail pieces) plus an observation-noise level.
- MASE ratio: MASE of Timestral divided by MASE of the true-dynamics oracle (Monte Carlo continuation of the true process from its true state, 500 paths), on the same sample, context 2048, horizon 128. Fitness is the geometric mean of this ratio over 3 fixed seeds (`SEARCH_SEEDS`).
- Feasible: oracle sCRPS is at most 0.8 times seasonal-naive sCRPS (geometric mean over the 3 seeds), everything finite, context std within [1e-3, 1e6], oracle MASE at least 1e-6. Infeasible genomes have fitness 0; for the optimizers' ranking they carry a search score of minus the violation (see `otp/search/objective.py`).
- Call: one Timestral forecast of one series. A genome evaluation is 3 calls (3 seeds). A gradient pass of one series (forward and backward) is 2 calls. Symmetric averaging inside the forecaster is part of one call. Repeated genomes in a run cost nothing.
- Grid: 4 dominant families (primitive, chaos, regime, tails; family with the largest summed slot weight) x 10 bins of spectral entropy of the context (mean over seeds, equal bins on [0, 1]) = 40 cells.

## Comparison (comparison.csv, `mean` rows)

| method | evaluations | calls | n_feasible | best_fitness | median_top20 | calls_to_1.5 | calls_to_2.0 | calls_to_3.0 | n_runs_reaching_1.5 | n_runs_reaching_2.0 | n_runs_reaching_3.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cmaes | 1000 | 3000 | 981.3 | 123 | 38.17 | 16 | 17 | 32 | 3 | 3 | 3 |
| map_elites | 1000 | 3000 | 976.7 | 217.8 | 101.2 | 8 | 9 | 43 | 3 | 3 | 3 |
| gradient | 15 | 2997 | 14 | 21.17 | 3.981 | 505 | 505 | 757 | 3 | 3 | 3 |
| gradient_r12 | 60 | 2964 | 58.33 | 471.8 | 6.629 | 13 | 248 | 718 | 3 | 3 | 3 |

`calls_to_x` is the mean over the runs that reached fitness x (the `n_runs_reaching_x` columns count them out of the runs of that method).

### Coverage of the same grid (mean over seeds)

| method | cells_filled | cells_ge_1.5 | cells_ge_2.0 | cells_ge_3.0 | signatures_ge_1.5 | signatures_ge_2.0 | signatures_ge_3.0 | families_ge_1.5 |
|---|---|---|---|---|---|---|---|---|
| cmaes | 36.67 | 25.33 | 22.67 | 19 | 274 | 212.7 | 149 | 4 |
| map_elites | 35.67 | 32.33 | 30.67 | 26.67 | 199.7 | 153.3 | 106 | 4 |
| gradient | 5 | 1.667 | 1.667 | 1.333 | 2.667 | 2.667 | 2 | 1.333 |
| gradient_r12 | 13.33 | 6.667 | 6 | 5.333 | 9.333 | 7.333 | 5.667 | 2 |

`signatures_ge_x` is the number of distinct sets of components (weight above 0.05) among feasible genomes with fitness at least x.

### Per search seed

| method | seed | evaluations | calls | best_fitness | median_top20 | calls_to_1.5 | calls_to_2.0 | calls_to_3.0 | cells_filled | cells_ge_1.5 | cells_ge_2.0 | cells_ge_3.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmaes | 0 | 1000 | 3000 | 243.1 | 32.5 | 30 | 30 | 54 | 38 | 24 | 21 | 17 |
| cmaes | 1 | 1000 | 3000 | 68.89 | 51.27 | 12 | 15 | 36 | 36 | 24 | 21 | 19 |
| cmaes | 2 | 1000 | 3000 | 57.12 | 30.75 | 6 | 6 | 6 | 36 | 28 | 26 | 21 |
| map_elites | 0 | 1000 | 3000 | 101.3 | 79.32 | 9 | 9 | 9 | 37 | 31 | 28 | 27 |
| map_elites | 1 | 1000 | 3000 | 200.3 | 85.91 | 3 | 6 | 12 | 35 | 35 | 34 | 32 |
| map_elites | 2 | 1000 | 3000 | 351.7 | 138.5 | 12 | 12 | 108 | 35 | 31 | 30 | 21 |
| gradient | 0 | 15 | 2997 | 7.516 | 1.378 | 3 | 3 | 756 | 6 | 1 | 1 | 1 |
| gradient | 1 | 15 | 2997 | 47.13 | 8.55 | 756 | 756 | 756 | 5 | 2 | 2 | 2 |
| gradient | 2 | 15 | 2997 | 8.876 | 2.014 | 756 | 756 | 759 | 4 | 2 | 2 | 1 |
| gradient_r12 | 0 | 60 | 2964 | 33.37 | 6.433 | 3 | 3 | 711 | 14 | 7 | 6 | 6 |
| gradient_r12 | 1 | 60 | 2964 | 8.868 | 6.62 | 12 | 717 | 717 | 12 | 7 | 7 | 6 |
| gradient_r12 | 2 | 60 | 2964 | 1373 | 6.836 | 24 | 24 | 726 | 14 | 6 | 5 | 4 |

## Run facts (run.json, history.csv)

| method | seed | budget | calls | evaluations | replayed_from_checkpoint | wall_seconds_this_process | calls_at_last_improvement | best_fitness |
|---|---|---|---|---|---|---|---|---|
| cmaes | 0 | 3000 | 3000 | 1000 | 767 | 592.7 | 774 | 243.1 |
| cmaes | 1 | 3000 | 3000 | 1000 | 0 | 2400 | 1716 | 68.89 |
| cmaes | 2 | 3000 | 3000 | 1000 | 0 | 2466 | 2682 | 57.12 |
| gradient_r12 | 0 | 3000 | 2964 | 60 | 0 | 31.31 | 2958 | 33.37 |
| gradient_r12 | 1 | 3000 | 2964 | 60 | 0 | 25.55 | 717 | 8.868 |
| gradient_r12 | 2 | 3000 | 2964 | 60 | 0 | 25.72 | 2958 | 1373 |
| gradient | 0 | 3000 | 2997 | 15 | 0 | 44.16 | 756 | 7.516 |
| gradient | 1 | 3000 | 2997 | 15 | 0 | 40.05 | 756 | 47.13 |
| gradient | 2 | 3000 | 2997 | 15 | 0 | 41.94 | 759 | 8.876 |
| map_elites | 0 | 3000 | 3000 | 1000 | 0 | 2541 | 2340 | 101.3 |
| map_elites | 1 | 3000 | 3000 | 1000 | 0 | 1936 | 2496 | 200.3 |
| map_elites | 2 | 3000 | 3000 | 1000 | 0 | 1845 | 2853 | 351.7 |

`wall_seconds_this_process` counts the process that wrote the run; a run resumed from a checkpoint does not include the time spent before the checkpoint. `calls_at_last_improvement` is the cumulative call count at the last evaluation that raised the best-so-far fitness.

Budget choice: the timing job (`results/colab/srch-timing/`) ran each method with 360 calls on the same A100 runtime; cmaes: 360 calls in 214 s; gradient: 357 calls in 6 s; map_elites: 360 calls in 247 s. The equal budget of every run is 3000 Timestral calls on 5M.

## MAP-Elites archives: elite per cell

### seed 0: 37 cells filled

| family | entropy_bin | fitness | recipe (* = modulate, weight) | noise_sd |
|---|---|---|---|---|
| primitive | 1 | 8.218 | mackey_glass(0.18) trend(0.78) | 0.001 |
| primitive | 2 | 10.51 | rossler(0.11) season(1.14) kuramoto(0.50) markov_ar*(0.11) | 0.001 |
| primitive | 3 | 23.88 | rossler(0.11) season(0.66) kuramoto(0.44) | 0.001044 |
| primitive | 4 | 21.81 | mackey_glass(0.11) season(0.66) kuramoto(0.50) frequency_jump*(0.11) | 0.001044 |
| primitive | 5 | 6.167 | rossler(0.11) trend*(1.14) mean_jump(0.50) stable_ar(0.06) | 0.003548 |
| primitive | 6 | 6.527 | rossler(0.05) trend(0.63) mean_jump(0.50) stable_ar(0.29) | 0.001 |
| primitive | 7 | 4.964 | season(1.09) trend*(0.78) stable_ar*(0.67) | 0.001851 |
| primitive | 8 | 4.242 | season(1.09) trend(0.78) stable_ar*(0.69) | 0.001851 |
| primitive | 9 | 4.975 | rossler(1.15) stable_ar*(0.89) frequency_jump(1.04) season(2.00) | 0.002221 |
| chaos | 0 | 1.388 | logistic_map(0.66) fbm(0.11) | 0.01022 |
| chaos | 1 | 8.46 | mackey_glass*(0.20) kuramoto(1.73) | 0.004438 |
| chaos | 2 | 29.92 | rossler(1.50) | 0.001537 |
| chaos | 3 | 21.88 | kuramoto(2.00) season(0.67) logistic_map(1.59) | 0.005105 |
| chaos | 4 | 27.15 | kuramoto(1.27) season(0.67) mackey_glass(0.12) | 0.005105 |
| chaos | 5 | 101.3 | lorenz63(1.58) | 0.001 |
| chaos | 6 | 81.48 | mackey_glass(0.86) lorenz63(1.58) | 0.001 |
| chaos | 7 | 14.33 | stable_ar*(0.11) kuramoto(1.27) season(0.67) mean_jump(0.12) | 0.01956 |
| chaos | 8 | 56.04 | stable_ar*(0.30) kuramoto(1.27) season(0.67) season(0.07) | 0.005105 |
| chaos | 9 | 2.832 | rossler(2.00) stable_ar*(0.64) frequency_jump(1.04) rossler(1.93) | 0.008575 |
| regime | 2 | 5.555 | mackey_glass(0.18) frequency_jump*(0.65) | 0.001 |
| regime | 3 | 6.341 | rossler(0.08) mackey_glass*(0.11) mean_jump(0.50) | 0.001376 |
| regime | 4 | 8.712 | rossler(0.06) season*(0.47) mean_jump(0.50) | 0.001031 |
| regime | 5 | 6.899 | rossler(0.11) trend(0.35) mean_jump(0.50) stable_ar(0.09) | 0.001102 |
| regime | 6 | 3.281 | rossler(0.11) ar_noise(0.35) mean_jump(0.50) variance_jump(0.06) | 0.001 |
| regime | 7 | 1.966 | frequency_jump(0.21) ar_noise(0.07) lorenz63(0.21) frequency_jump*(0.48) | 0.00297 |
| regime | 8 | 1.608 | lorenz63(0.21) frequency_jump*(0.48) | 0.00297 |
| regime | 9 | 1.273 | lorenz63(0.21) frequency_jump*(0.48) | 0.00297 |
| tails | 0 | 1.028 | fbm(0.19) | 0.0119 |
| tails | 1 | 1.354 | fbm(0.19) | 0.0119 |
| tails | 2 | 1.782 | logistic_map(0.06) fbm(0.54) | 0.01914 |
| tails | 3 | 5.773 | mackey_glass*(0.88) kuramoto(0.21) msm_cascade(1.77) variance_jump(0.13) | 0.004464 |
| tails | 4 | 4.429 | mackey_glass*(0.84) kuramoto(0.35) msm_cascade(1.77) frequency_jump(0.13) | 0.004464 |
| tails | 5 | 8.283 | lorenz63*(1.45) kuramoto(0.21) msm_cascade(1.69) | 0.03026 |
| tails | 6 | 3.944 | lorenz63*(1.50) kuramoto(0.21) msm_cascade(1.69) stable_ar(0.08) | 0.03026 |
| tails | 7 | 1.207 | frequency_jump(0.76) hawkes(1.51) msm_cascade(0.11) rossler(0.93) | 0.03009 |
| tails | 8 | 1.265 | frequency_jump(0.56) hawkes(1.51) msm_cascade(0.11) rossler(0.93) | 0.01817 |
| tails | 9 | 6.538 | lorenz63(0.21) stable_ar*(0.48) | 0.00297 |

### seed 1: 35 cells filled

| family | entropy_bin | fitness | recipe (* = modulate, weight) | noise_sd |
|---|---|---|---|---|
| primitive | 1 | 40.36 | rossler(0.10) season(0.84) | 0.001 |
| primitive | 2 | 200.3 | season(1.95) season(1.03) | 0.001008 |
| primitive | 3 | 22.46 | rossler(0.42) season(0.63) | 0.01577 |
| primitive | 4 | 40.83 | rossler(0.62) season(0.66) | 0.004657 |
| primitive | 5 | 15.53 | fbm*(0.41) lorenz63(0.35) trend(0.64) | 0.002562 |
| primitive | 6 | 38.84 | markov_ar(1.16) stable_ar*(1.44) season*(1.59) rossler(1.46) | 0.04577 |
| primitive | 7 | 3.082 | frequency_jump(1.16) stable_ar*(1.02) season*(1.59) rossler(1.06) | 0.04577 |
| primitive | 8 | 2.826 | markov_ar(0.56) stable_ar*(1.13) season*(1.49) rossler(1.43) | 0.009346 |
| primitive | 9 | 8.29 | markov_ar(0.72) stable_ar*(1.78) season*(1.09) ar_noise(1.06) | 0.01516 |
| chaos | 1 | 23.13 | rossler(0.24) kuramoto(1.12) | 0.0113 |
| chaos | 2 | 27.88 | season(1.31) kuramoto(1.83) | 0.002261 |
| chaos | 3 | 86.42 | season(1.31) mackey_glass(1.50) | 0.001045 |
| chaos | 4 | 31.37 | rossler(0.62) season(0.49) | 0.01397 |
| chaos | 5 | 29.89 | mackey_glass(0.91) lorenz63*(2.00) | 0.001952 |
| chaos | 6 | 56.94 | lorenz63(1.37) season*(0.07) msm_cascade(0.31) lorenz63*(1.54) | 0.06118 |
| chaos | 7 | 21.18 | lorenz63(1.37) markov_ar(0.31) lorenz63*(2.00) | 0.1275 |
| chaos | 8 | 4.421 | lorenz63(1.37) season*(0.09) msm_cascade(0.31) lorenz63*(1.25) | 0.1243 |
| chaos | 9 | 8.473 | frequency_jump(1.16) stable_ar*(1.02) season*(1.44) mackey_glass(2.00) | 0.03886 |
| regime | 2 | 3.882 | frequency_jump*(0.95) trend(0.37) | 0.007373 |
| regime | 3 | 4.187 | frequency_jump*(0.95) trend(0.31) | 0.00497 |
| regime | 4 | 3.694 | frequency_jump(1.16) mean_jump(1.14) trend*(1.78) rossler(1.04) | 0.01432 |
| regime | 5 | 3.268 | frequency_jump(1.16) mean_jump*(0.57) season*(1.67) rossler(1.04) | 0.04644 |
| regime | 6 | 3.864 | frequency_jump(1.29) markov_ar*(1.03) season*(1.78) rossler(1.04) | 0.04577 |
| regime | 7 | 3.564 | stable_ar*(1.78) mean_jump(1.81) trend(0.07) | 0.002836 |
| regime | 8 | 6.301 | stable_ar*(1.66) variance_jump(2.00) season(0.53) | 0.00492 |
| regime | 9 | 1.959 | msm_cascade*(1.78) mean_jump(2.00) | 0.001 |
| tails | 1 | 2.735 | fbm*(0.41) rossler(0.09) trend(0.15) | 0.002562 |
| tails | 2 | 6.749 | fbm*(0.41) rossler(0.23) trend(0.15) | 0.002562 |
| tails | 3 | 5.366 | fbm*(0.41) rossler(0.23) trend(0.15) | 0.002562 |
| tails | 4 | 9.971 | fbm(0.38) lorenz63(0.30) trend(0.27) | 0.002155 |
| tails | 5 | 17.71 | fbm*(0.41) lorenz63(0.35) trend(0.15) | 0.002562 |
| tails | 6 | 31.94 | markov_ar*(1.24) stable_ar*(1.78) season*(1.09) rossler(1.06) | 0.02571 |
| tails | 7 | 15.51 | markov_ar(0.72) stable_ar*(1.78) season*(1.09) rossler(1.06) | 0.01516 |
| tails | 8 | 5.882 | stable_ar*(1.78) mean_jump(1.70) season(0.31) | 0.00492 |
| tails | 9 | 10.43 | frequency_jump(1.16) stable_ar*(1.78) season*(1.68) mackey_glass(0.91) | 0.0103 |

### seed 2: 35 cells filled

| family | entropy_bin | fitness | recipe (* = modulate, weight) | noise_sd |
|---|---|---|---|---|
| primitive | 2 | 76.52 | season(1.83) lorenz63(0.34) season(0.84) | 0.001837 |
| primitive | 3 | 303.5 | season(1.83) season(1.01) season*(1.08) | 0.001837 |
| primitive | 4 | 351.7 | season*(2.00) rossler(0.34) season(0.97) | 0.001837 |
| primitive | 5 | 53.71 | season*(1.99) lorenz63(0.56) season(0.89) | 0.00313 |
| primitive | 6 | 3.758 | mean_jump(0.23) lorenz63(0.16) season*(2.00) kuramoto(0.14) | 0.002724 |
| primitive | 7 | 3.665 | mean_jump(0.52) season(0.30) season*(2.00) kuramoto(0.14) | 0.005798 |
| primitive | 8 | 1.461 | mean_jump(0.36) season(0.30) ar_noise*(1.92) kuramoto(0.10) | 0.01058 |
| primitive | 9 | 1.097 | mean_jump(0.36) trend(0.75) ar_noise*(1.92) kuramoto(0.15) | 0.01058 |
| chaos | 1 | 4.746 | logistic_map(0.76) trend(0.61) mackey_glass*(0.56) mackey_glass(0.34) | 0.04864 |
| chaos | 2 | 38.99 | season(1.85) mackey_glass(1.85) logistic_map*(0.70) | 0.009328 |
| chaos | 3 | 43.91 | season(1.85) mackey_glass(1.85) logistic_map*(0.87) | 0.009328 |
| chaos | 4 | 77.53 | mackey_glass*(2.00) lorenz63(0.56) season(0.89) | 0.00313 |
| chaos | 5 | 77.24 | mackey_glass*(2.00) lorenz63(0.56) season(0.89) | 0.00313 |
| chaos | 6 | 23.89 | mackey_glass*(2.00) lorenz63(0.56) frequency_jump*(0.19) | 0.005407 |
| chaos | 7 | 8.391 | rossler*(2.00) lorenz63*(0.05) season(0.64) season(0.81) | 0.001033 |
| chaos | 8 | 1.483 | mackey_glass(1.50) trend(0.30) lorenz63*(1.92) season(0.22) | 0.007532 |
| chaos | 9 | 1.402 | frequency_jump(0.23) trend(0.30) lorenz63*(1.92) season(0.22) | 0.007532 |
| regime | 1 | 2.96 | mean_jump*(0.58) logistic_map(0.41) fbm*(0.45) | 0.109 |
| regime | 2 | 2.592 | season(0.41) lorenz63*(0.11) frequency_jump*(0.61) | 0.002606 |
| regime | 3 | 3.657 | logistic_map(1.39) variance_jump*(1.85) mackey_glass*(0.22) | 0.04864 |
| regime | 4 | 6.559 | stable_ar(0.39) frequency_jump(0.15) mean_jump(1.93) mackey_glass(0.29) | 0.02822 |
| regime | 5 | 9.158 | mackey_glass(1.14) season(0.64) kuramoto*(0.22) variance_jump*(2.00) | 0.01647 |
| regime | 6 | 7.774 | mackey_glass(1.14) season(0.71) stable_ar*(0.22) variance_jump*(2.00) | 0.01184 |
| regime | 7 | 2.47 | lorenz63(0.48) fbm*(0.38) trend(0.11) variance_jump*(0.84) | 0.009142 |
| regime | 8 | 2.534 | msm_cascade*(1.28) frequency_jump(1.95) | 0.006219 |
| regime | 9 | 2.03 | msm_cascade*(1.28) frequency_jump(2.00) | 0.006219 |
| tails | 1 | 2.911 | mean_jump*(0.57) logistic_map(0.41) fbm*(0.82) | 0.109 |
| tails | 2 | 9.695 | season(0.41) lorenz63(0.11) fbm*(0.61) | 0.002606 |
| tails | 3 | 2.357 | season(0.31) lorenz63(0.11) msm_cascade*(1.12) | 0.05906 |
| tails | 4 | 3.691 | frequency_jump(0.64) hawkes(0.12) fbm*(0.54) | 0.03047 |
| tails | 5 | 3.401 | frequency_jump(0.34) msm_cascade(0.15) fbm*(0.25) | 0.05656 |
| tails | 6 | 3.504 | mean_jump*(0.57) lorenz63(0.41) fbm*(0.82) | 0.06231 |
| tails | 7 | 2.416 | frequency_jump(0.15) msm_cascade(0.12) fbm*(0.54) | 0.05656 |
| tails | 8 | 2.303 | msm_cascade*(1.19) frequency_jump(1.13) | 0.003044 |
| tails | 9 | 1.967 | msm_cascade*(2.00) frequency_jump(1.16) | 0.005469 |

## Gradient search (a): run facts

| method | seed | starts | screened_draws | steps | rejected_steps_per_start | calls | scored_genomes | best_fitness |
|---|---|---|---|---|---|---|---|---|
| gradient_r12 | 0 | 12 | 12 | 58 | [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] | 2964 | 60 | 33.37 |
| gradient_r12 | 1 | 12 | 12 | 58 | [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 2964 | 60 | 8.868 |
| gradient_r12 | 2 | 12 | 13 | 58 | [0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0] | 2964 | 60 | 1373 |
| gradient | 0 | 3 | 3 | 246 | [0, 0, 0] | 2997 | 15 | 7.516 |
| gradient | 1 | 3 | 3 | 246 | [0, 0, 0] | 2997 | 15 | 47.13 |
| gradient | 2 | 3 | 3 | 246 | [0, 0, 0] | 2997 | 15 | 8.876 |

## Transfer to other sizes (fresh seeds)

| method | size | n | n_feasible_fresh | frac_gt_1.5 | frac_gt_2 | median_fitness | median_fitness_search_5M |
|---|---|---|---|---|---|---|---|
| cmaes | 209M | 20 | 20 | 1 | 1 | 45.91 | 59.54 |
| cmaes | 123M | 20 | 20 | 1 | 1 | 49.72 | 59.54 |
| cmaes | 28M | 20 | 20 | 1 | 1 | 51.47 | 59.54 |
| cmaes | 5M | 20 | 20 | 1 | 1 | 57.66 | 59.54 |
| gradient_r12 | 209M | 20 | 20 | 1 | 0.95 | 11.71 | 22.65 |
| gradient_r12 | 123M | 20 | 20 | 1 | 1 | 11.25 | 22.65 |
| gradient_r12 | 28M | 20 | 20 | 1 | 1 | 17.65 | 22.65 |
| gradient_r12 | 5M | 20 | 20 | 1 | 1 | 24.92 | 22.65 |
| gradient | 209M | 20 | 20 | 0.85 | 0.8 | 8.042 | 8.291 |
| gradient | 123M | 20 | 20 | 0.8 | 0.8 | 7.447 | 8.291 |
| gradient | 28M | 20 | 20 | 0.95 | 0.85 | 7.687 | 8.291 |
| gradient | 5M | 20 | 20 | 1 | 1 | 8.239 | 8.291 |
| map_elites | 209M | 20 | 20 | 1 | 1 | 63.15 | 199.4 |
| map_elites | 123M | 20 | 20 | 1 | 1 | 71.38 | 199.4 |
| map_elites | 28M | 20 | 20 | 1 | 1 | 101.6 | 199.4 |
| map_elites | 5M | 20 | 20 | 1 | 1 | 189.2 | 199.4 |
| all | 209M | 80 | 80 | 0.9625 | 0.9375 | 36.49 | 53.62 |
| all | 123M | 80 | 80 | 0.95 | 0.95 | 39.2 | 53.62 |
| all | 28M | 80 | 80 | 0.9875 | 0.9625 | 45.24 | 53.62 |
| all | 5M | 80 | 80 | 1 | 1 | 57.66 | 53.62 |

`frac_gt_x` is the share of the method's top genomes whose fitness on that size, recomputed on fresh seeds, exceeds x (a genome infeasible on the fresh seeds counts as below). `median_fitness_search_5M` is the same genomes' fitness on the search seeds.

### Spearman correlation of the MASE ratio between sizes

| method | a | b | n | spearman |
|---|---|---|---|---|
| cmaes | search_5M | 209M | 20 | 0.1985 |
| cmaes | search_5M | 123M | 20 | 0.2932 |
| cmaes | search_5M | 28M | 20 | 0.2857 |
| cmaes | search_5M | 5M_fresh | 20 | 0.3173 |
| cmaes | 209M | 123M | 20 | 0.9203 |
| cmaes | 209M | 28M | 20 | 0.8436 |
| cmaes | 209M | 5M_fresh | 20 | 0.8316 |
| cmaes | 123M | 28M | 20 | 0.9639 |
| cmaes | 123M | 5M_fresh | 20 | 0.9609 |
| cmaes | 28M | 5M_fresh | 20 | 0.985 |
| gradient_r12 | search_5M | 209M | 20 | 0.8421 |
| gradient_r12 | search_5M | 123M | 20 | 0.8812 |
| gradient_r12 | search_5M | 28M | 20 | 0.9414 |
| gradient_r12 | search_5M | 5M_fresh | 20 | 0.991 |
| gradient_r12 | 209M | 123M | 20 | 0.9504 |
| gradient_r12 | 209M | 28M | 20 | 0.8797 |
| gradient_r12 | 209M | 5M_fresh | 20 | 0.8662 |
| gradient_r12 | 123M | 28M | 20 | 0.9429 |
| gradient_r12 | 123M | 5M_fresh | 20 | 0.8992 |
| gradient_r12 | 28M | 5M_fresh | 20 | 0.9609 |
| gradient | search_5M | 209M | 20 | 0.9833 |
| gradient | search_5M | 123M | 20 | 0.9152 |
| gradient | search_5M | 28M | 20 | 0.9727 |
| gradient | search_5M | 5M_fresh | 20 | 0.9439 |
| gradient | 209M | 123M | 20 | 0.9 |
| gradient | 209M | 28M | 20 | 0.9848 |
| gradient | 209M | 5M_fresh | 20 | 0.9439 |
| gradient | 123M | 28M | 20 | 0.9364 |
| gradient | 123M | 5M_fresh | 20 | 0.7909 |
| gradient | 28M | 5M_fresh | 20 | 0.9106 |
| map_elites | search_5M | 209M | 20 | 0.7214 |
| map_elites | search_5M | 123M | 20 | 0.6837 |
| map_elites | search_5M | 28M | 20 | 0.6943 |
| map_elites | search_5M | 5M_fresh | 20 | 0.9021 |
| map_elites | 209M | 123M | 20 | 0.914 |
| map_elites | 209M | 28M | 20 | 0.7496 |
| map_elites | 209M | 5M_fresh | 20 | 0.6229 |
| map_elites | 123M | 28M | 20 | 0.7285 |
| map_elites | 123M | 5M_fresh | 20 | 0.5913 |
| map_elites | 28M | 5M_fresh | 20 | 0.4932 |
| all | search_5M | 209M | 80 | 0.9033 |
| all | search_5M | 123M | 80 | 0.9302 |
| all | search_5M | 28M | 80 | 0.9542 |
| all | search_5M | 5M_fresh | 80 | 0.9545 |
| all | 209M | 123M | 80 | 0.9802 |
| all | 209M | 28M | 80 | 0.9482 |
| all | 209M | 5M_fresh | 80 | 0.8962 |
| all | 123M | 28M | 80 | 0.973 |
| all | 123M | 5M_fresh | 80 | 0.923 |
| all | 28M | 5M_fresh | 80 | 0.9583 |

## Input-space sensitivity (b), not a generator failure

For in-prior series (`in_prior(seed)`, seeds 0..n-1), the smallest relative perturbation found by projected gradient ascent that doubles Timestral's MASE on the fixed realized future. rho is the per-step RMS of the perturbation divided by the context std; `l2_over_std` is the perturbation's L2 norm divided by the context std.

| size | n_series | n_reached | median_rho | min_rho | max_rho | median_l2_over_std | median_mase_clean |
|---|---|---|---|---|---|---|---|
| 5M | 20 | 20 | 0.03402 | 0.006043 | 0.08058 | 1.54 | 0.6457 |
| 28M | 20 | 20 | 0.02579 | 0.004988 | 0.08254 | 1.167 | 0.6587 |
| 123M | 20 | 20 | 0.02742 | 0.003398 | 0.06813 | 1.241 | 0.6555 |
| 209M | 20 | 20 | 0.02746 | 0.002943 | 0.0619 | 1.242 | 0.6563 |

## Exact commands

Colab job scripts (`results/search/colab_jobs/`), each run with `scripts/colab/run_job.sh <job> <script> a100` (timing and smoke jobs only chose the budget and checked the code path):

`job_search_a.sh`

```
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/search.py --methods input_space --sizes 5M 28M 123M 209M --n-series 20 --out results/colab_out --flat-prefix srchis_ --resume > results/colab_out/input_space.log 2>&1 &
IS=$!
python3 scripts/search.py --methods cmaes gradient --seeds 0 1 2 --budget 3000 --size 5M --workers 11 --out results/colab_out --flat-prefix srch_ --resume
wait $IS
cat results/colab_out/input_space.log
cp results/colab_out/input_space.log results/colab_out/srchis_input_space/input_space.log
```

`job_search_b.sh`

```
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/search.py --methods map_elites --seeds 0 1 2 --budget 3000 --size 5M --workers 11 --out results/colab_out --flat-prefix srch_ --resume
python3 scripts/search.py --methods gradient --grad-restarts 12 --label gradient_r12 --seeds 0 1 2 --budget 3000 --size 5M --workers 11 --out results/colab_out --flat-prefix srch_ --resume
```

`job_transfer.sh`

```
nvidia-smi --query-gpu=name --format=csv,noheader
python3 scripts/search_transfer.py --runs results/search/runs --top 20 --workers 11 --out results/colab_out/srcht_transfer
```

Jobs (from the pulled job folders, also in `provenance.json`):

| job | bundle_commit | code_commit | started_utc | ended_utc | exit_code | wall_seconds |
|---|---|---|---|---|---|---|
| srch-a | fa5f369 | fa5f369 | 2026-10-27T01:06:44Z | 2026-10-27T01:45:36Z | None | 2332 |
| srch-a2 | 9fb9396 | b5d7bce | 2026-10-27T02:05:09Z | 2026-10-27T03:41:07Z | 0 | 5758 |
| srch-b | 0e09977 | 2bd3089 | 2026-10-27T03:46:34Z | 2026-10-27T05:39:31Z | 0 | 6777 |
| srch-smoke | e334b2f | e334b2f | 2026-10-27T00:13:00Z | 2026-10-27T00:25:16Z | 0 | 736 |
| srch-timing | 683c36b | 683c36b | 2026-10-27T00:45:41Z | 2026-10-27T00:57:47Z | 0 | 726 |
| srch-transfer | 22c6fb7 | b16990e | 2026-10-27T05:46:05Z | 2026-10-27T05:50:13Z | 0 | 248 |

`bundle_commit` is the commit of the code archive the job ran; the driver was launched from a slim worktree (result folders, bench, paper, notebooks and tests removed, code unchanged) because uploads of the full 97 MB archive to fresh Colab runtimes failed with HTTP 500 and 400 errors on repeated attempts, and `code_commit` is the task-branch commit that slim commit was made from. The per-process invocation records written on Colab (`*provenance.json` next to the run folders) are not among the files the driver pulls, so `provenance.json` holds the jobs table, per-run `run.json` and `transfer.json`. The first CMA-ES seed 0 run was cut when its job (`srch-a`) was ended by the driver; the checkpoint pulled before that held 767 evaluations, and the run was resumed in `srch-a2` from it.

Bundle commits are slim copies of `code_commit` made for upload and are not published; the code in them is identical (see results/BUNDLE_COMMITS.md).
