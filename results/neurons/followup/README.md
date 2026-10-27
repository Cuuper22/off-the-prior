# Neuron atlas follow-up B: corpus, number of series and variable library

Question: the atlas's one-job share was 4.5% on the main corpus but 24 to 47% on the short, long and xlong corpora
(`../context_comparison.csv`). Is that a property of the neurons, of the variable library, or of the corpus size?

Runs (all with `scripts/neurons_run.py` at commit c4eecd7 on the laptop GPU; plan in `plan.csv`, run folders not
committed): the main corpus with the full library (reference) and with the library common to all four corpora
(`libraries/common.txt`, 82 of the 101 variables); short, long and xlong with the common library; the main corpus
subsampled at random to 1034, 620 and 48 series (the series counts of short, long and xlong), 3 seeds each; random
libraries of 10, 25 and 50 variables drawn from the common library, 3 seeds each. 28M: reference, common library,
1034 and 48 series. Classification as in the atlas (theta 0.7, floor 0.1 below; other thresholds in
`corpus_library_control.csv`), written by `scripts/neurons_followup_b.py`. `same_label_as_reference`: share of poles
whose best label equals the reference run's (same size).

## Summary (theta 0.7, floor 0.1)

| size | role | corpus | n_series | library | one_job_mean | one_job_min | one_job_max | E_mean | runs |
|---|---|---|---|---|---|---|---|---|---|
| 28M | common_library | main | 3100 | common | 0.03187 | 0.03187 | 0.03187 | 0.2322 | 1 |
| 28M | reference | main | 3100 | full | 0.03244 | 0.03244 | 0.03244 | 0.2334 | 1 |
| 28M | series_count | main | 48 | full | 0.3027 | 0.3027 | 0.3027 | 0.1159 | 1 |
| 28M | series_count | main | 1034 | full | 0.1031 | 0.1031 | 0.1031 | 0.1823 | 1 |
| 5M | common_library | long | 620 | common | 0.2258 | 0.2258 | 0.2258 | 0.1926 | 1 |
| 5M | common_library | main | 3100 | common | 0.04775 | 0.04775 | 0.04775 | 0.2615 | 1 |
| 5M | common_library | short | 1034 | common | 0.4081 | 0.4081 | 0.4081 | 0.2105 | 1 |
| 5M | common_library | xlong | 48 | common | 0.4739 | 0.4739 | 0.4739 | 0.1644 | 1 |
| 5M | library_size | main | 3100 | random10 | 0.06387 | 0.05765 | 0.07181 | 0.1298 | 3 |
| 5M | library_size | main | 3100 | random25 | 0.0504 | 0.03188 | 0.06168 | 0.1837 | 3 |
| 5M | library_size | main | 3100 | random50 | 0.055 | 0.04629 | 0.06204 | 0.2349 | 3 |
| 5M | reference | main | 3100 | full | 0.04812 | 0.04812 | 0.04812 | 0.2623 | 1 |
| 5M | series_count | main | 48 | full | 0.3263 | 0.2569 | 0.3742 | 0.1244 | 3 |
| 5M | series_count | main | 620 | full | 0.2641 | 0.2617 | 0.2659 | 0.156 | 3 |
| 5M | series_count | main | 1034 | full | 0.1148 | 0.1108 | 0.1221 | 0.2137 | 3 |

## All runs (theta 0.7, floor 0.1)

| tag | size | corpus | library | n_series | seed | n_live | one_job | unexplained | E_mean | same_label_as_reference |
|---|---|---|---|---|---|---|---|---|---|---|
| 5M_ref_main_full | 5M | main | full | 3100 | 0 | 8188 | 0.04812 | 0.115 | 0.2623 | 1 |
| 5M_main_common | 5M | main | common | 3100 | 0 | 8188 | 0.04775 | 0.1159 | 0.2615 | 0.9891 |
| 5M_short_common | 5M | short | common | 1034 | 0 | 8187 | 0.4081 | 0.2399 | 0.2105 | 0.1941 |
| 5M_long_common | 5M | long | common | 620 | 0 | 8188 | 0.2258 | 0.3035 | 0.1926 | 0.516 |
| 5M_xlong_common | 5M | xlong | common | 48 | 0 | 8185 | 0.4739 | 0.3806 | 0.1644 | 0.1716 |
| 5M_main_sub1034_s0 | 5M | main | full | 1034 | 0 | 8188 | 0.1115 | 0.2443 | 0.2153 | 0.7883 |
| 5M_main_sub1034_s1 | 5M | main | full | 1034 | 1 | 8188 | 0.1221 | 0.2548 | 0.2098 | 0.7792 |
| 5M_main_sub1034_s2 | 5M | main | full | 1034 | 2 | 8188 | 0.1108 | 0.2443 | 0.2158 | 0.7976 |
| 5M_main_sub620_s0 | 5M | main | full | 620 | 0 | 8188 | 0.2647 | 0.438 | 0.1534 | 0.739 |
| 5M_main_sub620_s1 | 5M | main | full | 620 | 1 | 8188 | 0.2659 | 0.4244 | 0.1574 | 0.7242 |
| 5M_main_sub620_s2 | 5M | main | full | 620 | 2 | 8188 | 0.2617 | 0.4249 | 0.1573 | 0.7302 |
| 5M_main_sub48_s0 | 5M | main | full | 48 | 0 | 8179 | 0.3478 | 0.5051 | 0.1337 | 0.3118 |
| 5M_main_sub48_s1 | 5M | main | full | 48 | 1 | 8178 | 0.2569 | 0.6236 | 0.1043 | 0.2985 |
| 5M_main_sub48_s2 | 5M | main | full | 48 | 2 | 8181 | 0.3742 | 0.4959 | 0.1353 | 0.2803 |
| 5M_main_k10_s0 | 5M | main | random10 | 3100 | 0 | 8188 | 0.07181 | 0.5609 | 0.1099 | 0.1191 |
| 5M_main_k10_s1 | 5M | main | random10 | 3100 | 1 | 8188 | 0.06216 | 0.4295 | 0.1366 | 0.1984 |
| 5M_main_k10_s2 | 5M | main | random10 | 3100 | 2 | 8188 | 0.05765 | 0.4265 | 0.143 | 0.2709 |
| 5M_main_k25_s0 | 5M | main | random25 | 3100 | 0 | 8188 | 0.06168 | 0.248 | 0.1902 | 0.2142 |
| 5M_main_k25_s1 | 5M | main | random25 | 3100 | 1 | 8188 | 0.03188 | 0.3144 | 0.1643 | 0.2434 |
| 5M_main_k25_s2 | 5M | main | random25 | 3100 | 2 | 8188 | 0.05765 | 0.2263 | 0.1967 | 0.3655 |
| 5M_main_k50_s0 | 5M | main | random50 | 3100 | 0 | 8188 | 0.06204 | 0.148 | 0.2342 | 0.5905 |
| 5M_main_k50_s1 | 5M | main | random50 | 3100 | 1 | 8188 | 0.04629 | 0.138 | 0.2355 | 0.6224 |
| 5M_main_k50_s2 | 5M | main | random50 | 3100 | 2 | 8188 | 0.05667 | 0.1462 | 0.2351 | 0.7107 |
| 28M_ref_main_full | 28M | main | full | 3100 | 0 | 24565 | 0.03244 | 0.1381 | 0.2334 | 1 |
| 28M_main_common | 28M | main | common | 3100 | 0 | 24565 | 0.03187 | 0.1391 | 0.2322 | 0.9822 |
| 28M_main_sub48_s0 | 28M | main | full | 48 | 0 | 24490 | 0.3027 | 0.5727 | 0.1159 | 0.2312 |
| 28M_main_sub1034_s0 | 28M | main | full | 1034 | 0 | 24559 | 0.1031 | 0.3073 | 0.1823 | 0.768 |

## Reproduce

```
python3 scripts/neurons_run.py --size <S> --corpus results/neurons/corpus/<corpus>.npz --out results/colab_out/fb/<tag> \
    --layers-per-chunk 4 --batch <32|16|4> --resume [--keep-vars results/neurons/followup/libraries/<file>] \
    [--subsample <n> --subsample-seed <seed>]          # one line per row of plan.csv
cp results/neurons/followup/plan.csv results/colab_out/fb/runs.csv
python3 scripts/neurons_followup_b.py --runs results/colab_out/fb --out results/neurons/followup
```

<!-- follow-up A and C -->

## A. Steering power (pre-registration: `../steering_preregistration_followup.md`)

Commands (each size writes one folder `results/colab_out/nn2_a_<S>`; the capture runs for part C write `results/colab_out/nn2c_<tag>`; both run through `scripts/colab/job_nn2_common.sh`, which `scripts/colab/job_nn2_123M.sh` and `job_nn2_209M.sh` call, and which the 5M and 28M jobs also used):

```
python3 scripts/neurons_followup_a.py --size <S> --poles results/neurons/<S>/poles.csv --corpus results/neurons/corpus/main.npz \
    --out results/colab_out/nn2_a_<S> --batch 128 --checkpoint-every 10 --resume
python3 scripts/neurons_run.py --size <S> --corpus results/neurons/corpus/main.npz --out results/colab_out/nn2c_<tag> \
    --layers-per-chunk 8 --batch <batch> --resume [--subsample <n> --subsample-seed <seed>]      # rows of followup/plan_c.csv
python3 scripts/neurons_followup_c.py --atlas <atlas colab_out> --fb <follow-up B colab_out/fb> --flat results/colab_out \
    --corpus-dir results/neurons/corpus --out results/neurons/followup
python3 scripts/neurons_followup_report.py --a-runs results/colab_out --steering results/neurons/steering_results.csv --out results/neurons/followup
```

The `nn2_a_<S>` folders (scores, readouts, meta) are committed under `results/colab_out/`; the `nn2c_<tag>` capture files (about 18 MB each) are not, and the cohesion tables in section C are derived from them by `scripts/neurons_followup_c.py`. Job logs and environment files are in `results/colab/nn2-small/`, `nn2-123M/`, `nn2-209M/`.

Quantities, scales, test series and the matching criterion are those of the original steering. The calibrated criterion counts an effect as big when `|D|` (D = effect at the top scale minus effect at scale 0) exceeds the 95th percentile of `|D|` over the matched controls; it also requires the monotone and interval conditions. Files: `direction_readouts.csv`, `direction_steering.csv`, `group_steering.csv`, `group_summary.csv`, `mde.csv`, `single_pole_calibrated.csv`. `direction` in the group and pole tables is the predicted sign of the effect.

Runs (from `meta.json` of each folder):

| size | device | n_tasks_run | wall_s | git | n_series_mixed | n_series_seasonal | n_series_jump | n_random_dir | n_random_group |
|---|---|---|---|---|---|---|---|---|---|
| 5M | cuda | 148 | 52.8 | 5337a6a | 96 | 64 | 30 | 32 | 16 |
| 28M | cuda | 168 | 179.2 | 5337a6a | 96 | 64 | 30 | 32 | 16 |
| 123M | cuda | 201 | 678.9 | 5337a6a | 96 | 64 | 30 | 32 | 16 |
| 209M | cuda | 208 | 1065 | 5337a6a | 96 | 64 | 30 | 32 | 16 |

Late block, forecast-token site, pooled direction (`postjump` on the jump series, the others on the mixed series; `gain` = effect on the predicted quantity divided by the move in context standard deviations, so 1 would be a 1:1 response):

| size | variable | block | site | gain | effect_size_D | p95_random | orig_match | cal_match |
|---|---|---|---|---|---|---|---|---|
| 123M | level | late | F | 0.01212 | 0.02424 | 0.02023 | False | True |
| 209M | level | late | F | 0.008622 | 0.01724 | 0.0107 | False | True |
| 28M | level | late | F | 0.04399 | 0.08797 | 0.05085 | True | True |
| 5M | level | late | F | 0.349 | 0.698 | 0.2196 | True | True |
| 123M | noise | late | F | 0.02699 | 0.05398 | 0.02499 | True | True |
| 209M | noise | late | F | 0.03174 | 0.06348 | 0.0286 | True | True |
| 28M | noise | late | F | 0.02812 | 0.05624 | 0.03887 | True | True |
| 5M | noise | late | F | 0.07844 | 0.1569 | 0.09941 | True | True |
| 123M | postjump | late | F | 0.003046 | 0.006091 | 0.01591 | False | False |
| 209M | postjump | late | F | 0.005693 | 0.01139 | 0.01646 | False | False |
| 28M | postjump | late | F | 0.005603 | 0.01121 | 0.03945 | False | False |
| 5M | postjump | late | F | 0.09011 | 0.1802 | 0.1211 | True | True |

Pre-registered prediction (post-jump gain smaller than the level gain), late block, forecast-token site, pooled direction:

| size | level_gain | postjump_gain | postjump_gain_smaller |
|---|---|---|---|
| 123M | 0.01212 | 0.003046 | True |
| 209M | 0.008622 | 0.005693 | True |
| 28M | 0.04399 | 0.005603 | True |
| 5M | 0.349 | 0.09011 | True |

Late-block context sites (`C1`, `C4`): the output of the last block at context tokens is not read by anything after it, and the measured effects there are at most 2e-07 in absolute value over all sizes, variables and kinds (rows `late`, `C1`/`C4` in the cell tables); these cells are not a test of the readout.

Group steering is read together with the minimum detectable effect (`mde.csv`): `ratio_p95_to_threshold` = 95th percentile of the control effects divided by the original threshold of the quantity.

### A1. Readouts of known variables from the residual stream (`direction_readouts.csv`)

Held-out R^2 of the ridge readout at the output of the early / middle / late block, by token set (`pooled`: context and forecast tokens).

| size | variable | block_name | ctx | fc | pooled |
|---|---|---|---|---|---|
| 123M | level | early | 0.5655 | 0.7842 | 0.5947 |
| 123M | level | late | 0.5438 | 0.9018 | 0.6085 |
| 123M | level | mid | 0.6215 | 0.8962 | 0.6744 |
| 123M | noise | early | 0.9339 | 0.9479 | 0.9258 |
| 123M | noise | late | 0.9243 | 0.9486 | 0.9204 |
| 123M | noise | mid | 0.9392 | 0.9539 | 0.9358 |
| 123M | postjump | early | 0.9016 | 0.6789 | 0.8613 |
| 123M | postjump | late | 0.8995 | 0.8772 | 0.8977 |
| 123M | postjump | mid | 0.9157 | 0.9041 | 0.9171 |
| 209M | level | early | 0.5736 | 0.7905 | 0.6062 |
| 209M | level | late | 0.5648 | 0.9056 | 0.6276 |
| 209M | level | mid | 0.648 | 0.8939 | 0.6966 |
| 209M | noise | early | 0.9427 | 0.9557 | 0.9378 |
| 209M | noise | late | 0.9348 | 0.954 | 0.9316 |
| 209M | noise | mid | 0.9476 | 0.9585 | 0.945 |
| 209M | postjump | early | 0.9082 | 0.704 | 0.8722 |
| 209M | postjump | late | 0.8993 | 0.8889 | 0.9034 |
| 209M | postjump | mid | 0.9166 | 0.9016 | 0.9081 |
| 28M | level | early | 0.5228 | 0.7749 | 0.557 |
| 28M | level | late | 0.4516 | 0.8994 | 0.5396 |
| 28M | level | mid | 0.5874 | 0.8798 | 0.6399 |
| 28M | noise | early | 0.8907 | 0.9236 | 0.8768 |
| 28M | noise | late | 0.8258 | 0.9227 | 0.8255 |
| 28M | noise | mid | 0.9054 | 0.9418 | 0.9033 |
| 28M | postjump | early | 0.8953 | 0.6868 | 0.8549 |
| 28M | postjump | late | 0.8977 | 0.8861 | 0.8954 |
| 28M | postjump | mid | 0.9211 | 0.9229 | 0.915 |
| 5M | level | early | 0.4957 | 0.7652 | 0.4927 |
| 5M | level | late | 0.3474 | 0.8863 | 0.4519 |
| 5M | level | mid | 0.5031 | 0.8434 | 0.5535 |
| 5M | noise | early | 0.8466 | 0.8835 | 0.8238 |
| 5M | noise | late | 0.5686 | 0.8655 | 0.56 |
| 5M | noise | mid | 0.8026 | 0.9004 | 0.7957 |
| 5M | postjump | early | 0.8904 | 0.6151 | 0.834 |
| 5M | postjump | late | 0.7307 | 0.8821 | 0.7364 |
| 5M | postjump | mid | 0.9049 | 0.8488 | 0.8833 |

### A1. Positive control: direction steering (`direction_steering.csv`)

`effect_size` = D at move 2 on the predicted quantity (`level` for the level and post-jump directions, `spread` for the noise direction), in context standard deviations (spread: log width ratio); `gain` = D / 2; `p95_random` = 95th percentile of `|D|` over the random directions of equal norm at the same size, variable, kind, block and site; `orig_match` and `cal_match` are the two criteria.

Verdict rule (plain level direction, all blocks, sites, direction kinds, mixed test set):

| size | cells | orig_match_cells | cal_match_cells | both_cells | max_gain | max_abs_D | verdict |
|---|---|---|---|---|---|---|---|
| 123M | 18 | 0 | 8 | 0 | 0.02233 | 0.04465 | insensitive |
| 209M | 18 | 0 | 7 | 0 | 0.01686 | 0.03372 | insensitive |
| 28M | 18 | 2 | 11 | 2 | 0.1009 | 0.2017 | detects a known effect |
| 5M | 18 | 3 | 9 | 3 | 0.349 | 0.698 | detects a known effect |

Variable `level`, pooled direction, all cells:

| size | variable | kind | block_name | site | readout_r2_heldout | gain | effect_size | p95_random | threshold | rho | orig_match | cal_match | random_orig_match_rate | random_cal_match_rate | E2_level | E2_spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 123M | level | pooled | early | F | 0.5947 | -0.004121 | -0.008242 | 0.004584 | 0.05 | -1 | False | False | 0 | 0.0625 | -0.008242 | -0.002334 |
| 123M | level | pooled | early | C1 | 0.5947 | 0.002548 | 0.005095 | 0.002252 | 0.05 | 1 | False | True | 0 | 0.03125 | 0.005095 | 0.005535 |
| 123M | level | pooled | early | C4 | 0.5947 | 0.005219 | 0.01044 | 0.004711 | 0.05 | 1 | False | True | 0 | 0 | 0.01044 | 0.004423 |
| 123M | level | pooled | mid | F | 0.6744 | 0.004269 | 0.008538 | 0.008765 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.008538 | 0.001572 |
| 123M | level | pooled | mid | C1 | 0.6744 | 0.0005238 | 0.001048 | 0.001835 | 0.05 | 1 | False | False | 0 | 0.0625 | 0.001048 | 2.893e-06 |
| 123M | level | pooled | mid | C4 | 0.6744 | 0.001722 | 0.003444 | 0.0045 | 0.05 | 1 | False | False | 0 | 0.0625 | 0.003444 | 0.001275 |
| 123M | level | pooled | late | F | 0.6085 | 0.01212 | 0.02424 | 0.02023 | 0.05 | 1 | False | True | 0 | 0.0625 | 0.02424 | 0.001723 |
| 123M | level | pooled | late | C1 | 0.6085 | -1.766e-09 | -3.531e-09 | 2.065e-08 | 0.05 | 0 | False | False | 0 | 0 | -3.531e-09 | -1.783e-08 |
| 123M | level | pooled | late | C4 | 0.6085 | -1.766e-09 | -3.531e-09 | 2.065e-08 | 0.05 | 0 | False | False | 0 | 0 | -3.531e-09 | -1.783e-08 |
| 209M | level | pooled | early | F | 0.6062 | -0.001782 | -0.003563 | 0.006235 | 0.05 | -1 | False | False | 0 | 0.0625 | -0.003563 | 0.001206 |
| 209M | level | pooled | early | C1 | 0.6062 | 0.001112 | 0.002225 | 0.00155 | 0.05 | 1 | False | True | 0 | 0.03125 | 0.002225 | 0.002214 |
| 209M | level | pooled | early | C4 | 0.6062 | 0.002393 | 0.004787 | 0.003479 | 0.05 | 1 | False | True | 0 | 0.03125 | 0.004787 | 0.00358 |
| 209M | level | pooled | mid | F | 0.6966 | 0.002325 | 0.00465 | 0.008316 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.00465 | -0.004859 |
| 209M | level | pooled | mid | C1 | 0.6966 | 2.533e-07 | 5.065e-07 | 0.001409 | 0.05 | 0.4 | False | False | 0 | 0 | 5.065e-07 | 0.00205 |
| 209M | level | pooled | mid | C4 | 0.6966 | 5.632e-06 | 1.126e-05 | 0.003444 | 0.05 | 0.4 | False | False | 0 | 0.03125 | 1.126e-05 | 0.004558 |
| 209M | level | pooled | late | F | 0.6276 | 0.008622 | 0.01724 | 0.0107 | 0.05 | 1 | False | True | 0 | 0 | 0.01724 | 0.003403 |
| 209M | level | pooled | late | C1 | 0.6276 | 2.538e-08 | 5.076e-08 | 5.31e-08 | 0.05 | 0.7746 | False | False | 0 | 0 | 5.076e-08 | 3.602e-08 |
| 209M | level | pooled | late | C4 | 0.6276 | 2.538e-08 | 5.076e-08 | 5.31e-08 | 0.05 | 0.7746 | False | False | 0 | 0 | 5.076e-08 | 3.602e-08 |
| 28M | level | pooled | early | F | 0.557 | 0.002258 | 0.004516 | 0.009615 | 0.05 | 1 | False | False | 0 | 0.0625 | 0.004516 | 0.000622 |
| 28M | level | pooled | early | C1 | 0.557 | 0.004351 | 0.008701 | 0.005753 | 0.05 | 1 | False | True | 0 | 0.03125 | 0.008701 | 0.002376 |
| 28M | level | pooled | early | C4 | 0.557 | 0.007028 | 0.01406 | 0.008849 | 0.05 | 1 | False | True | 0 | 0.03125 | 0.01406 | 0.006607 |
| 28M | level | pooled | mid | F | 0.6399 | 0.009133 | 0.01827 | 0.02723 | 0.05 | 1 | False | False | 0 | 0 | 0.01827 | 0.01019 |
| 28M | level | pooled | mid | C1 | 0.6399 | 0.004459 | 0.008919 | 0.005444 | 0.05 | 1 | False | True | 0 | 0.0625 | 0.008919 | -0.001133 |
| 28M | level | pooled | mid | C4 | 0.6399 | 0.008894 | 0.01779 | 0.01277 | 0.05 | 1 | False | True | 0 | 0.0625 | 0.01779 | -0.002736 |
| 28M | level | pooled | late | F | 0.5396 | 0.04399 | 0.08797 | 0.05085 | 0.05 | 1 | True | True | 0.0625 | 0.0625 | 0.08797 | 0.05063 |
| 28M | level | pooled | late | C1 | 0.5396 | 1.715e-08 | 3.431e-08 | 5.073e-08 | 0.05 | 0.7746 | False | False | 0 | 0 | 3.431e-08 | 8.746e-08 |
| 28M | level | pooled | late | C4 | 0.5396 | 1.715e-08 | 3.431e-08 | 5.073e-08 | 0.05 | 0.7746 | False | False | 0 | 0 | 3.431e-08 | 8.746e-08 |
| 5M | level | pooled | early | F | 0.4927 | -0.006748 | -0.0135 | 0.02887 | 0.05 | -1 | False | False | 0 | 0.0625 | -0.0135 | 0.03359 |
| 5M | level | pooled | early | C1 | 0.4927 | 0.004797 | 0.009594 | 0.01532 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.009594 | 0.002312 |
| 5M | level | pooled | early | C4 | 0.4927 | 0.0122 | 0.0244 | 0.02418 | 0.05 | 1 | False | True | 0 | 0.03125 | 0.0244 | 0.002539 |
| 5M | level | pooled | mid | F | 0.5535 | -0.01119 | -0.02239 | 0.0599 | 0.05 | -1 | False | False | 0.125 | 0.0625 | -0.02239 | -0.01181 |
| 5M | level | pooled | mid | C1 | 0.5535 | 0.008861 | 0.01772 | 0.008956 | 0.05 | 1 | False | True | 0 | 0.03125 | 0.01772 | 0.0012 |
| 5M | level | pooled | mid | C4 | 0.5535 | 0.0203 | 0.04061 | 0.02721 | 0.05 | 1 | False | True | 0 | 0.0625 | 0.04061 | 0.01457 |
| 5M | level | pooled | late | F | 0.4519 | 0.349 | 0.698 | 0.2196 | 0.05 | 1 | True | True | 0.25 | 0 | 0.698 | 0.4357 |
| 5M | level | pooled | late | C1 | 0.4519 | -5.887e-09 | -1.177e-08 | 3.759e-08 | 0.05 | -0.7746 | False | False | 0 | 0 | -1.177e-08 | 3.291e-08 |
| 5M | level | pooled | late | C4 | 0.4519 | -5.887e-09 | -1.177e-08 | 3.759e-08 | 0.05 | -0.7746 | False | False | 0 | 0 | -1.177e-08 | 3.291e-08 |

Variable `noise`, pooled direction, all cells:

| size | variable | kind | block_name | site | readout_r2_heldout | gain | effect_size | p95_random | threshold | rho | orig_match | cal_match | random_orig_match_rate | random_cal_match_rate | E2_level | E2_spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 123M | noise | pooled | early | F | 0.9258 | -0.0001327 | -0.0002654 | 0.002849 | 0.05 | -1 | False | False | 0 | 0.03125 | -0.001408 | -0.0002654 |
| 123M | noise | pooled | early | C1 | 0.9258 | 0.0004139 | 0.0008278 | 0.00113 | 0.05 | 1 | False | False | 0 | 0.0625 | 0.0001923 | 0.0008278 |
| 123M | noise | pooled | early | C4 | 0.9258 | 0.001039 | 0.002078 | 0.001912 | 0.05 | 1 | False | True | 0 | 0.0625 | 0.0004318 | 0.002078 |
| 123M | noise | pooled | mid | F | 0.9358 | 0.003597 | 0.007195 | 0.008883 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.002115 | 0.007195 |
| 123M | noise | pooled | mid | C1 | 0.9358 | 0.0005529 | 0.001106 | 0.001858 | 0.05 | 1 | False | False | 0 | 0.0625 | 0.0002432 | 0.001106 |
| 123M | noise | pooled | mid | C4 | 0.9358 | 0.001147 | 0.002294 | 0.004789 | 0.05 | 1 | False | False | 0 | 0.0625 | 2.277e-05 | 0.002294 |
| 123M | noise | pooled | late | F | 0.9204 | 0.02699 | 0.05398 | 0.02499 | 0.05 | 1 | True | True | 0 | 0.03125 | 0.001515 | 0.05398 |
| 123M | noise | pooled | late | C1 | 0.9204 | -8.915e-09 | -1.783e-08 | 3.254e-08 | 0.05 | -0.7746 | False | False | 0 | 0 | -3.531e-09 | -1.783e-08 |
| 123M | noise | pooled | late | C4 | 0.9204 | -8.915e-09 | -1.783e-08 | 3.254e-08 | 0.05 | -0.7746 | False | False | 0 | 0 | -3.531e-09 | -1.783e-08 |
| 209M | noise | pooled | early | F | 0.9378 | 0.001069 | 0.002139 | 0.002592 | 0.05 | 1 | False | False | 0 | 0.0625 | 0.0004318 | 0.002139 |
| 209M | noise | pooled | early | C1 | 0.9378 | 0.0001559 | 0.0003119 | 0.00124 | 0.05 | 1 | False | False | 0 | 0.0625 | -0.0002643 | 0.0003119 |
| 209M | noise | pooled | early | C4 | 0.9378 | 0.0005953 | 0.001191 | 0.002067 | 0.05 | 1 | False | False | 0 | 0.0625 | -0.0002379 | 0.001191 |
| 209M | noise | pooled | mid | F | 0.945 | 0.006818 | 0.01364 | 0.005611 | 0.05 | 1 | False | True | 0 | 0.03125 | -1.75e-05 | 0.01364 |
| 209M | noise | pooled | mid | C1 | 0.945 | 8.047e-05 | 0.0001609 | 0.001254 | 0.05 | 1 | False | False | 0 | 0.0625 | 0.0004598 | 0.0001609 |
| 209M | noise | pooled | mid | C4 | 0.945 | -0.0001516 | -0.0003031 | 0.002719 | 0.05 | -1 | False | False | 0 | 0.0625 | 0.0009942 | -0.0003031 |
| 209M | noise | pooled | late | F | 0.9316 | 0.03174 | 0.06348 | 0.0286 | 0.05 | 1 | True | True | 0 | 0 | -0.0003891 | 0.06348 |
| 209M | noise | pooled | late | C1 | 0.9316 | 1.801e-08 | 3.602e-08 | 1.095e-07 | 0.05 | 0.7746 | False | False | 0 | 0 | 5.076e-08 | 3.602e-08 |
| 209M | noise | pooled | late | C4 | 0.9316 | 1.801e-08 | 3.602e-08 | 1.095e-07 | 0.05 | 0.7746 | False | False | 0 | 0 | 5.076e-08 | 3.602e-08 |
| 28M | noise | pooled | early | F | 0.8768 | 0.005265 | 0.01053 | 0.007284 | 0.05 | 1 | False | True | 0 | 0.0625 | 0.001102 | 0.01053 |
| 28M | noise | pooled | early | C1 | 0.8768 | 0.001142 | 0.002283 | 0.002849 | 0.05 | 1 | False | False | 0 | 0.0625 | -0.0007346 | 0.002283 |
| 28M | noise | pooled | early | C4 | 0.8768 | 0.002545 | 0.00509 | 0.006147 | 0.05 | 1 | False | False | 0 | 0.03125 | -0.001266 | 0.00509 |
| 28M | noise | pooled | mid | F | 0.9033 | 0.01597 | 0.03194 | 0.01458 | 0.05 | 1 | False | True | 0 | 0.0625 | 0.001123 | 0.03194 |
| 28M | noise | pooled | mid | C1 | 0.9033 | 0.001188 | 0.002376 | 0.003006 | 0.05 | 1 | False | False | 0 | 0.03125 | -0.001378 | 0.002376 |
| 28M | noise | pooled | mid | C4 | 0.9033 | 0.001795 | 0.003591 | 0.007336 | 0.05 | 1 | False | False | 0 | 0.0625 | -0.002026 | 0.003591 |
| 28M | noise | pooled | late | F | 0.8255 | 0.02812 | 0.05624 | 0.03887 | 0.05 | 1 | True | True | 0 | 0.03125 | 0.008629 | 0.05624 |
| 28M | noise | pooled | late | C1 | 0.8255 | 4.373e-08 | 8.746e-08 | 1.361e-07 | 0.05 | 0.7746 | False | False | 0 | 0 | 3.431e-08 | 8.746e-08 |
| 28M | noise | pooled | late | C4 | 0.8255 | 4.373e-08 | 8.746e-08 | 1.361e-07 | 0.05 | 0.7746 | False | False | 0 | 0 | 3.431e-08 | 8.746e-08 |
| 5M | noise | pooled | early | F | 0.8238 | 0.02071 | 0.04142 | 0.02016 | 0.05 | 1 | False | True | 0 | 0 | 0.001452 | 0.04142 |
| 5M | noise | pooled | early | C1 | 0.8238 | 0.002516 | 0.005031 | 0.01001 | 0.05 | 1 | False | False | 0 | 0.0625 | -0.001295 | 0.005031 |
| 5M | noise | pooled | early | C4 | 0.8238 | 0.003259 | 0.006519 | 0.01285 | 0.05 | 1 | False | False | 0 | 0.0625 | -0.00327 | 0.006519 |
| 5M | noise | pooled | mid | F | 0.7957 | 0.01375 | 0.0275 | 0.03749 | 0.05 | 1 | False | False | 0 | 0 | 0.002306 | 0.0275 |
| 5M | noise | pooled | mid | C1 | 0.7957 | 0.002061 | 0.004122 | 0.002999 | 0.05 | 1 | False | True | 0 | 0.0625 | 0.0009771 | 0.004122 |
| 5M | noise | pooled | mid | C4 | 0.7957 | 0.005914 | 0.01183 | 0.01026 | 0.05 | 1 | False | True | 0 | 0.03125 | 0.001052 | 0.01183 |
| 5M | noise | pooled | late | F | 0.56 | 0.07844 | 0.1569 | 0.09941 | 0.05 | 1 | True | True | 0.2812 | 0.03125 | 0.01086 | 0.1569 |
| 5M | noise | pooled | late | C1 | 0.56 | 1.645e-08 | 3.291e-08 | 1.042e-07 | 0.05 | 0.7746 | False | False | 0 | 0 | -1.177e-08 | 3.291e-08 |
| 5M | noise | pooled | late | C4 | 0.56 | 1.645e-08 | 3.291e-08 | 1.042e-07 | 0.05 | 0.7746 | False | False | 0 | 0 | -1.177e-08 | 3.291e-08 |

Variable `postjump`, pooled direction, all cells:

| size | variable | kind | block_name | site | readout_r2_heldout | gain | effect_size | p95_random | threshold | rho | orig_match | cal_match | random_orig_match_rate | random_cal_match_rate | E2_level | E2_spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 123M | postjump | pooled | early | F | 0.8613 | -0.006927 | -0.01385 | 0.01535 | 0.05 | -1 | False | False | 0 | 0.03125 | -0.01385 | -0.000417 |
| 123M | postjump | pooled | early | C1 | 0.8613 | 0.005927 | 0.01185 | 0.01292 | 0.05 | 1 | False | False | 0 | 0.0625 | 0.01185 | 0.0003752 |
| 123M | postjump | pooled | early | C4 | 0.8613 | 0.01161 | 0.02322 | 0.01785 | 0.05 | 1 | False | True | 0 | 0.0625 | 0.02322 | -0.008602 |
| 123M | postjump | pooled | mid | F | 0.9171 | 0.003832 | 0.007664 | 0.01421 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.007664 | -0.01015 |
| 123M | postjump | pooled | mid | C1 | 0.9171 | -0.001462 | -0.002925 | 0.00745 | 0.05 | -1 | False | False | 0 | 0 | -0.002925 | -0.0007538 |
| 123M | postjump | pooled | mid | C4 | 0.9171 | -0.0008678 | -0.001736 | 0.01136 | 0.05 | -1 | False | False | 0 | 0 | -0.001736 | 0.001369 |
| 123M | postjump | pooled | late | F | 0.8977 | 0.003046 | 0.006091 | 0.01591 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.006091 | -0.009355 |
| 123M | postjump | pooled | late | C1 | 0.8977 | 2.645e-08 | 5.29e-08 | 1.987e-08 | 0.05 | 0.7746 | False | False | 0 | 0 | 5.29e-08 | 1.335e-07 |
| 123M | postjump | pooled | late | C4 | 0.8977 | 2.645e-08 | 5.29e-08 | 1.987e-08 | 0.05 | 0.7746 | False | False | 0 | 0 | 5.29e-08 | 1.335e-07 |
| 209M | postjump | pooled | early | F | 0.8722 | -0.005794 | -0.01159 | 0.01388 | 0.05 | -1 | False | False | 0 | 0.03125 | -0.01159 | 0.001539 |
| 209M | postjump | pooled | early | C1 | 0.8722 | 0.006038 | 0.01208 | 0.008943 | 0.05 | 1 | False | True | 0 | 0 | 0.01208 | -0.003576 |
| 209M | postjump | pooled | early | C4 | 0.8722 | 0.01197 | 0.02395 | 0.01131 | 0.05 | 1 | False | True | 0 | 0.03125 | 0.02395 | -0.006739 |
| 209M | postjump | pooled | mid | F | 0.9081 | 0.000424 | 0.0008481 | 0.01113 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.0008481 | 0.005501 |
| 209M | postjump | pooled | mid | C1 | 0.9081 | 0.0002962 | 0.0005925 | 0.004109 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.0005925 | -0.002803 |
| 209M | postjump | pooled | mid | C4 | 0.9081 | 0.0006765 | 0.001353 | 0.007985 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.001353 | -0.005026 |
| 209M | postjump | pooled | late | F | 0.9034 | 0.005693 | 0.01139 | 0.01646 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.01139 | -0.002119 |
| 209M | postjump | pooled | late | C1 | 0.9034 | -2.844e-08 | -5.687e-08 | 4.483e-08 | 0.05 | -0.7746 | False | False | 0 | 0 | -5.687e-08 | 1.08e-08 |
| 209M | postjump | pooled | late | C4 | 0.9034 | -2.844e-08 | -5.687e-08 | 4.483e-08 | 0.05 | -0.7746 | False | False | 0 | 0 | -5.687e-08 | 1.08e-08 |
| 28M | postjump | pooled | early | F | 0.8549 | -0.006374 | -0.01275 | 0.01728 | 0.05 | -1 | False | False | 0 | 0 | -0.01275 | 0.009117 |
| 28M | postjump | pooled | early | C1 | 0.8549 | 0.008273 | 0.01655 | 0.01123 | 0.05 | 1 | False | True | 0 | 0.03125 | 0.01655 | -0.002598 |
| 28M | postjump | pooled | early | C4 | 0.8549 | 0.01242 | 0.02484 | 0.02785 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.02484 | -0.006801 |
| 28M | postjump | pooled | mid | F | 0.915 | -0.01167 | -0.02334 | 0.04059 | 0.05 | -1 | False | False | 0 | 0 | -0.02334 | -0.0003108 |
| 28M | postjump | pooled | mid | C1 | 0.915 | 0.002581 | 0.005162 | 0.01781 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.005162 | -0.004214 |
| 28M | postjump | pooled | mid | C4 | 0.915 | 0.008611 | 0.01722 | 0.02312 | 0.05 | 1 | False | False | 0 | 0.0625 | 0.01722 | -0.01165 |
| 28M | postjump | pooled | late | F | 0.8954 | 0.005603 | 0.01121 | 0.03945 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.01121 | 0.01157 |
| 28M | postjump | pooled | late | C1 | 0.8954 | -1.001e-07 | -2.002e-07 | 1.547e-07 | 0.05 | -0.7746 | False | False | 0 | 0 | -2.002e-07 | -2.958e-08 |
| 28M | postjump | pooled | late | C4 | 0.8954 | -1.001e-07 | -2.002e-07 | 1.547e-07 | 0.05 | -0.7746 | False | False | 0 | 0 | -2.002e-07 | -2.958e-08 |
| 5M | postjump | pooled | early | F | 0.834 | 0.004931 | 0.009862 | 0.03844 | 0.05 | 1 | False | False | 0 | 0.03125 | 0.009862 | 0.001393 |
| 5M | postjump | pooled | early | C1 | 0.834 | 0.01609 | 0.03217 | 0.03181 | 0.05 | 1 | False | True | 0 | 0.03125 | 0.03217 | -0.003648 |
| 5M | postjump | pooled | early | C4 | 0.834 | 0.03219 | 0.06438 | 0.05741 | 0.05 | 1 | True | True | 0.03125 | 0 | 0.06438 | -0.0009501 |
| 5M | postjump | pooled | mid | F | 0.8833 | -0.03527 | -0.07055 | 0.07463 | 0.05 | -1 | False | False | 0.03125 | 0 | -0.07055 | 0.06467 |
| 5M | postjump | pooled | mid | C1 | 0.8833 | 0.01934 | 0.03867 | 0.02172 | 0.05 | 1 | False | True | 0 | 0 | 0.03867 | 0.00406 |
| 5M | postjump | pooled | mid | C4 | 0.8833 | 0.04146 | 0.08293 | 0.03583 | 0.05 | 1 | True | True | 0 | 0 | 0.08293 | 0.006534 |
| 5M | postjump | pooled | late | F | 0.7364 | 0.09011 | 0.1802 | 0.1211 | 0.05 | 1 | True | True | 0.4062 | 0.03125 | 0.1802 | 0.2138 |
| 5M | postjump | pooled | late | C1 | 0.7364 | -1.614e-09 | -3.229e-09 | 5.824e-08 | 0.05 | 0 | False | False | 0 | 0 | -3.229e-09 | 5.588e-09 |
| 5M | postjump | pooled | late | C4 | 0.7364 | -1.614e-09 | -3.229e-09 | 5.824e-08 | 0.05 | 0 | False | False | 0 | 0 | -3.229e-09 | 5.588e-09 |

### A2. Group steering (`group_summary.csv`, `group_steering.csv`)

Match rates and effect-size distributions of labeled groups against matched random groups (16 per labeled group); `cal` = calibrated criterion with the pooled random-group 95th percentile of the same size, k and predicted quantity.

| size | pool | k | n_groups | n_random | labeled_orig_match | random_orig_match | labeled_cal_match | random_cal_match | labeled_abs_D_median | labeled_abs_D_p75 | labeled_abs_D_p95 | random_abs_D_median | random_abs_D_p75 | random_abs_D_p95 | mean_rank_among_random | share_rank_ge_0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 123M | label_top | 8 | 35 | 560 | 0 | 0 | 0.3714 | 0.03214 | 0.002937 | 0.008307 | 0.01701 | 0.0009721 | 0.001999 | 0.004859 | 0.6454 | 0.2 |
| 123M | label_top | 32 | 28 | 448 | 0.03571 | 0 | 0.2143 | 0.02232 | 0.005017 | 0.01848 | 0.05221 | 0.002166 | 0.005045 | 0.01187 | 0.6352 | 0.2143 |
| 123M | label_top | 128 | 18 | 288 | 0.05556 | 0 | 0.3889 | 0.03472 | 0.01857 | 0.04085 | 0.07941 | 0.004522 | 0.009667 | 0.02456 | 0.8139 | 0.5 |
| 123M | one_job | 8 | 10 | 160 | 0 | 0 | 0.4 | 0.025 | 0.002593 | 0.006474 | 0.01283 | 0.0007221 | 0.00133 | 0.002875 | 0.7547 | 0.5 |
| 123M | one_job | 32 | 1 | 16 | 0 | 0 | 0 | 0.125 | 0.02266 | 0.02266 | 0.02266 | 0.005663 | 0.01147 | 0.02212 | 0.6406 | 0 |
| 123M | one_job | 128 | 1 | 16 | 0 | 0 | 0 | 0.0625 | 0.04223 | 0.04223 | 0.04223 | 0.02574 | 0.0323 | 0.03864 | 0.9375 | 0 |
| 209M | label_top | 8 | 37 | 592 | 0 | 0 | 0.3243 | 0.0152 | 0.003474 | 0.007627 | 0.02388 | 0.0007373 | 0.001602 | 0.00492 | 0.6793 | 0.1351 |
| 209M | label_top | 32 | 29 | 464 | 0.06897 | 0.00431 | 0.5172 | 0.02802 | 0.009048 | 0.01858 | 0.06924 | 0.001673 | 0.003785 | 0.01034 | 0.7455 | 0.3103 |
| 209M | label_top | 128 | 24 | 384 | 0.125 | 0.002604 | 0.5 | 0.01823 | 0.01624 | 0.04474 | 0.1107 | 0.004419 | 0.009299 | 0.02024 | 0.8162 | 0.5 |
| 209M | one_job | 8 | 8 | 128 | 0 | 0 | 0.375 | 0 | 0.002546 | 0.002844 | 0.009595 | 0.0005189 | 0.0009694 | 0.002331 | 0.9082 | 0.625 |
| 209M | one_job | 32 | 1 | 16 | 0 | 0 | 0 | 0.125 | 0.01088 | 0.01088 | 0.01088 | 0.007468 | 0.01156 | 0.02372 | 0.4375 | 0 |
| 209M | one_job | 128 | 1 | 16 | 0 | 0.0625 | 0 | 0.1875 | 0.02454 | 0.02454 | 0.02454 | 0.01928 | 0.02652 | 0.04215 | 0.5312 | 0 |
| 28M | label_top | 8 | 30 | 480 | 0.03333 | 0.002083 | 0.2667 | 0.03333 | 0.006307 | 0.01561 | 0.05326 | 0.002148 | 0.004115 | 0.009022 | 0.8082 | 0.3 |
| 28M | label_top | 32 | 22 | 352 | 0.1818 | 0.002841 | 0.3636 | 0.03409 | 0.01616 | 0.05018 | 0.1077 | 0.00611 | 0.01246 | 0.02438 | 0.8783 | 0.4545 |
| 28M | label_top | 128 | 5 | 80 | 0.2 | 0.025 | 0.2 | 0.0375 | 0.08383 | 0.1063 | 0.124 | 0.01168 | 0.02354 | 0.06506 | 0.9583 | 0.8 |
| 28M | one_job | 8 | 2 | 32 | 0 | 0 | 0 | 0.03125 | 0.008765 | 0.01245 | 0.0154 | 0.001995 | 0.003172 | 0.01744 | 0.7188 | 0 |
| 28M | one_job | 32 | 1 | 16 | 0 | 0 | 0 | 0 | 0.04187 | 0.04187 | 0.04187 | 0.01653 | 0.02789 | 0.04687 | 0.8125 | 0 |
| 5M | label_top | 8 | 27 | 432 | 0.1111 | 0.00463 | 0.1481 | 0.02778 | 0.008714 | 0.01874 | 0.06501 | 0.007253 | 0.01336 | 0.0308 | 0.8021 | 0.3333 |
| 5M | label_top | 32 | 10 | 160 | 0.2 | 0.05625 | 0.2 | 0.03125 | 0.02866 | 0.06165 | 0.139 | 0.01356 | 0.02534 | 0.06455 | 0.8656 | 0.5 |
| 5M | label_top | 128 | 1 | 16 | 0 | 0.25 | 0 | 0.0625 | 0.2515 | 0.2515 | 0.2515 | 0.1508 | 0.1896 | 0.2388 | 1 | 1 |
| 5M | one_job | 8 | 1 | 16 | 0 | 0 | 0 | 0.25 | 0.02888 | 0.02888 | 0.02888 | 0.01528 | 0.03196 | 0.04385 | 0.9375 | 0 |
| 5M | one_job | 32 | 1 | 16 | 0 | 0.3125 | 0 | 0.0625 | 0.0818 | 0.0818 | 0.0818 | 0.04408 | 0.0674 | 0.1061 | 0.9531 | 1 |

Labeled groups with the highest rank among their random groups, per size:

| size | gid | predicted | direction | effect_size | p95_random_pooled | rank_among_random | rho | orig_match | cal_match |
|---|---|---|---|---|---|---|---|---|---|
| 209M | label_top/x_trend/p/k128 | slope | + | 0.014 | 0.008086 | 1 | 1 | False | True |
| 209M | label_top/m_roughness/n/k32 | spread | + | -0.06124 | 0.01475 | 1 | -1 | True | True |
| 209M | label_top/m_resid_sd/n/k32 | spread | + | 0.056 | 0.01475 | 1 | 1 | False | False |
| 209M | label_top/m_level_ctx/n/k32 | level | + | -0.006211 | 0.002966 | 1 | -1 | False | True |
| 209M | label_top/lat_slope/n/k32 | slope | + | -0.02891 | 0.00448 | 1 | -1 | False | True |
| 28M | label_top/lat_pred1/p/k8 | spread | - | -0.02715 | 0.01185 | 1 | -1 | False | True |
| 28M | label_top/m_logsd/n/k8 | spread | + | 0.2691 | 0.01185 | 1 | 1 | False | False |
| 28M | label_top/m_level_ctx/n/k8 | level | + | -0.03041 | 0.004335 | 1 | -1 | False | True |
| 123M | label_top/m_roughness/n/k8 | spread | + | -0.04142 | 0.00598 | 1 | -1 | False | True |
| 28M | label_top/m_roughness/n/k32 | spread | + | -0.1087 | 0.03203 | 1 | -1 | True | True |
| 123M | label_top/prev_noise_sd/p/k8 | spread | + | -0.004619 | 0.00598 | 1 | -1 | False | False |
| 28M | label_top/s_trend/n/k32 | slope | + | -0.01144 | 0.01207 | 1 | -1 | False | False |
| 5M | label_top/m_roughness/p/k128 | spread | + | 0.2515 | 0.2388 | 1 | 0.8 | False | False |
| 5M | label_top/x_trend/p/k32 | slope | + | 0.08714 | 0.02792 | 1 | 1 | True | True |
| 5M | label_top/x_season_strength/p/k32 | amp | + | -0.06532 | 0.05727 | 1 | -1 | False | False |
| 5M | label_top/m_resid_sd/p/k32 | spread | + | 0.1815 | 0.1009 | 1 | 0.8 | False | False |
| 5M | label_top/m_dlevel_prev/n/k32 | level | + | -0.05063 | 0.01934 | 1 | -1 | True | True |
| 123M | label_top/m_roughness/n/k32 | spread | + | -0.07153 | 0.01941 | 1 | -0.8 | False | False |
| 123M | label_top/lat_slope/n/k32 | slope | + | -0.01962 | 0.006727 | 1 | -1 | False | True |
| 123M | label_top/lat_slope/p/k32 | slope | + | 0.02787 | 0.006727 | 1 | 1 | False | True |

### A3. Single poles, re-judged with the calibrated criterion (`single_pole_calibrated.csv`)

From the original run (`../steering_results.csv`): 144 random control poles per size judged for level, slope and spread. Labeled poles of the amp and phase groups have no control distribution and are not in this table.

| size | quantity | n_labeled | labeled_orig_match | labeled_cal_match | n_control | control_orig_match | control_cal_match | p95_control_abs_D | threshold | max_labeled_abs_D | median_labeled_abs_D | labeled_abs_D_above_p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 123M | level | 11 | 0 | 0 | 144 | 0 | 0.03472 | 0.0006731 | 0.05 | 0.0005906 | 0.0001949 | 0 |
| 123M | slope | 24 | 0 | 0.25 | 144 | 0 | 0.02778 | 0.0006988 | 0.05 | 0.006002 | 0.0004833 | 0.3333 |
| 123M | spread | 21 | 0 | 0 | 144 | 0 | 0.03472 | 0.00157 | 0.05 | 0.003219 | 0.0003943 | 0.09524 |
| 209M | level | 8 | 0 | 0 | 144 | 0 | 0.02083 | 0.000332 | 0.05 | 0.0004247 | 0.000103 | 0.375 |
| 209M | slope | 24 | 0 | 0.08333 | 144 | 0 | 0.01389 | 0.0005906 | 0.05 | 0.001372 | 0.0003072 | 0.25 |
| 209M | spread | 24 | 0 | 0.04167 | 144 | 0 | 0.04861 | 0.001614 | 0.05 | 0.002302 | 0.0003264 | 0.04167 |
| 28M | level | 3 | 0 | 0 | 144 | 0 | 0.03472 | 0.002078 | 0.05 | 0.001774 | 0.0014 | 0 |
| 28M | slope | 17 | 0 | 0 | 144 | 0 | 0.02778 | 0.001993 | 0.05 | 0.006073 | 0.000386 | 0.05882 |
| 28M | spread | 15 | 0 | 0.06667 | 144 | 0 | 0.03472 | 0.004571 | 0.05 | 0.007285 | 0.0002806 | 0.06667 |
| 5M | level | 4 | 0 | 0 | 144 | 0.01389 | 0.03472 | 0.006622 | 0.05 | 0.003371 | 0.002325 | 0 |
| 5M | slope | 13 | 0 | 0 | 144 | 0 | 0.03472 | 0.006384 | 0.05 | 0.01206 | 0.001725 | 0.07692 |
| 5M | spread | 8 | 0 | 0 | 144 | 0.02083 | 0.04167 | 0.01739 | 0.05 | 0.01722 | 0.001871 | 0 |

### A4. Minimum detectable effect (`mde.csv`)

95th percentile of `|D|` over matched controls (single poles: 144 random poles; groups: random groups of the same size; directions: random directions of equal norm). The original thresholds are in the `threshold` column of `mde.csv`. Single poles and groups, quantity `level`:

| size | group k=128 | group k=32 | group k=8 | pole k=1 |
|---|---|---|---|---|
| 123M | 0.006482 | 0.003613 | 0.001526 | 0.0006731 |
| 209M | 0.00451 | 0.002966 | 0.001129 | 0.0003321 |
| 28M |  | 0.01499 | 0.004335 | 0.002077 |
| 5M |  | 0.01934 | 0.01643 | 0.006622 |

Directions (selected cells; all cells are in `mde.csv`):

| size | level/pooled/early/F | level/pooled/late/F | level/pooled/mid/F | noise/pooled/late/F |
|---|---|---|---|---|
| 123M | 0.004584 | 0.02023 | 0.008766 | 0.02499 |
| 209M | 0.006235 | 0.0107 | 0.008316 | 0.0286 |
| 28M | 0.009615 | 0.05085 | 0.02723 | 0.03887 |
| 5M | 0.02887 | 0.2196 | 0.0599 | 0.09941 |


## C. Motif labels without a variable list (`motif_clusters.csv`, `motif_cluster_names.csv`, `motif_by_name.csv`)

Top and mid-range exemplar patches of every pole, normalized per patch (subtract mean, divide by sd), assigned to a fixed vocabulary of K = 16 shapes (k-means on the top-exemplar patches of the four atlas main-corpus runs, pooled over sizes). A pole is motif-cohesive when at least 6 of its 8 top exemplars share a cluster. `null` = the same share when exemplar cluster ids are permuted across poles (mean of 20 permutations); `one_job` = known-variable share of the same run (theta 0.7, floor 0.1).

Cluster names (`motif_cluster_names.csv`):

| cluster | name | purity | centroid_name | n |
|---|---|---|---|---|
| 0 | flat with noise | 0.5304 | oscillation | 33832 |
| 1 | oscillation | 0.6005 | flat with noise | 21789 |
| 2 | oscillation | 0.4232 | falling | 41242 |
| 3 | oscillation | 0.4385 | peak | 26575 |
| 4 | oscillation | 0.7003 | oscillation | 20303 |
| 5 | oscillation | 0.5453 | flat with noise | 22052 |
| 6 | oscillation | 0.4445 | step up | 35137 |
| 7 | oscillation | 0.3905 | step up | 42148 |
| 8 | flat with noise | 0.6068 | oscillation | 53650 |
| 9 | oscillation | 0.5158 | flat with noise | 22791 |
| 10 | flat with noise | 1 | flat with noise | 10894 |
| 11 | flat with noise | 1 | flat with noise | 8444 |
| 12 | oscillation | 0.6309 | oscillation | 20600 |
| 13 | oscillation | 0.5989 | oscillation | 22696 |
| 14 | oscillation | 0.4105 | step down | 31105 |
| 15 | oscillation | 0.4159 | trough | 27246 |

Vocabulary fitted on 440504 top-exemplar patches (K = 16); 56 capture runs; wall time of the clustering script 411.3 s (`provenance_motifs.json`).

Names come from a fixed rule applied to the patches of each cluster (step, oscillation, rising or falling, peak or trough, flat with noise; first rule that fires, in that order). Clusters named by name: oscillation 12, flat with noise 4. `purity` is the share of the cluster's patches that the rule assigns to the cluster's name; `centroid_name` is the rule applied to the cluster mean. The rule does not separate most clusters from each other, so the cluster id, not the name, is the label; the tables below use cluster ids only.

Share of poles with mostly one cluster (at least 6 of 8 top exemplars) next to the known-variable one-job share, full main corpus (3100 series):

| size | n_series | n_live | cohesive_top6 | null_top6_mean | one_job_0.7 | both_known_and_motif | motif_only | known_only | neither |
|---|---|---|---|---|---|---|---|---|---|
| 5M | 3100 | 8188 | 0.1424 | 0.0001649 | 0.04812 | 0.00513 | 0.1373 | 0.04299 | 0.8146 |
| 28M | 3100 | 24565 | 0.1292 | 0.0001547 | 0.03245 | 0.003949 | 0.1252 | 0.0285 | 0.8423 |
| 123M | 3100 | 68461 | 0.108 | 0.0001483 | 0.03 | 0.003798 | 0.1042 | 0.0262 | 0.8658 |
| 209M | 3100 | 98268 | 0.1075 | 0.0001242 | 0.02609 | 0.002809 | 0.1047 | 0.02328 | 0.8693 |

Same quantities on same-size subsamples of the main corpus (mean over seeds):

| size | n_series | cohesive_top6 | one_job | runs |
|---|---|---|---|---|
| 123M | 3100 | 0.108 | 0.03 | 1 |
| 123M | 1034 | 0.08795 | 0.1012 | 3 |
| 123M | 620 | 0.08168 | 0.2409 | 3 |
| 123M | 48 | 0.05789 | 0.246 | 3 |
| 209M | 3100 | 0.1075 | 0.02609 | 1 |
| 209M | 1034 | 0.08843 | 0.09343 | 3 |
| 209M | 620 | 0.08151 | 0.229 | 3 |
| 209M | 48 | 0.05958 | 0.2378 | 3 |
| 28M | 3100 | 0.1292 | 0.03245 | 1 |
| 28M | 1034 | 0.1072 | 0.1036 | 3 |
| 28M | 620 | 0.09961 | 0.2559 | 3 |
| 28M | 48 | 0.06421 | 0.2777 | 3 |
| 5M | 3100 | 0.1424 | 0.04812 | 1 |
| 5M | 1034 | 0.1169 | 0.1148 | 3 |
| 5M | 620 | 0.1069 | 0.2641 | 3 |
| 5M | 48 | 0.06765 | 0.3263 | 3 |

Pre-registered prediction (motif cohesion rises with fewer series, as the one-job share does):

| size | cohesive_at_48 | cohesive_at_3100 | one_job_at_48 | one_job_at_3100 | cohesion_rises_at_48 |
|---|---|---|---|---|---|
| 123M | 0.05789 | 0.108 | 0.246 | 0.03 | False |
| 209M | 0.05958 | 0.1075 | 0.2378 | 0.02609 | False |
| 28M | 0.06421 | 0.1292 | 0.2777 | 0.03245 | False |
| 5M | 0.06765 | 0.1424 | 0.3263 | 0.04812 | False |

### 5M

| tag | role | n_series | n_live | cohesive_top6 | null_top6_mean | cohesive_mid6 | one_job_0.7 | both_known_and_motif | motif_only | known_only | neither |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5M_atlas_main | atlas_main | 3580 | 8188 | 0.1371 | 0.0001466 | 0.0006107 | 0.04531 | 0.004641 | 0.1325 | 0.04067 | 0.8222 |
| 5M_ref_main_full | full_main | 3100 | 8188 | 0.1424 | 0.0001649 | 0.0003664 | 0.04812 | 0.00513 | 0.1373 | 0.04299 | 0.8146 |
| 5M_atlas_short | context | 1034 | 8187 | 0.09576 | 0.000116 | 0.0002443 | 0.4147 | 0.03237 | 0.06339 | 0.3823 | 0.5219 |
| 5M_main_sub1034_s0 | subsample | 1034 | 8188 | 0.1166 | 0.0002015 | 0.0004885 | 0.1115 | 0.01136 | 0.1053 | 0.1002 | 0.7832 |
| 5M_main_sub1034_s1 | subsample | 1034 | 8188 | 0.1166 | 0.0001221 | 0.0002443 | 0.1221 | 0.01172 | 0.1049 | 0.1104 | 0.773 |
| 5M_main_sub1034_s2 | subsample | 1034 | 8188 | 0.1175 | 0.0001893 | 0.0003664 | 0.1108 | 0.01075 | 0.1067 | 0.1 | 0.7825 |
| 5M_atlas_long | context | 620 | 8188 | 0.1293 | 0.000171 | 0.0006107 | 0.2419 | 0.03053 | 0.0988 | 0.2114 | 0.6593 |
| 5M_main_sub620_s0 | subsample | 620 | 8188 | 0.1077 | 0.0001221 | 0 | 0.2647 | 0.02907 | 0.07865 | 0.2356 | 0.6567 |
| 5M_main_sub620_s1 | subsample | 620 | 8188 | 0.1065 | 0.0001282 | 0.0007328 | 0.2659 | 0.03102 | 0.07548 | 0.2349 | 0.6586 |
| 5M_main_sub620_s2 | subsample | 620 | 8188 | 0.1066 | 9.77e-05 | 0.0002443 | 0.2617 | 0.03273 | 0.07389 | 0.229 | 0.6644 |
| 5M_atlas_xlong | context | 48 | 8185 | 0.1371 | 8.552e-05 | 0.01136 | 0.4747 | 0.07294 | 0.06414 | 0.4017 | 0.4612 |
| 5M_main_sub48_s0 | subsample | 48 | 8179 | 0.0758 | 9.17e-05 | 0.001345 | 0.3478 | 0.03607 | 0.03974 | 0.3118 | 0.6124 |
| 5M_main_sub48_s1 | subsample | 48 | 8178 | 0.04622 | 6.725e-05 | 0.0003668 | 0.2569 | 0.01284 | 0.03338 | 0.2441 | 0.7097 |
| 5M_main_sub48_s2 | subsample | 48 | 8181 | 0.08092 | 0.0001711 | 0.005867 | 0.3742 | 0.04694 | 0.03398 | 0.3272 | 0.5919 |

### 28M

| tag | role | n_series | n_live | cohesive_top6 | null_top6_mean | cohesive_mid6 | one_job_0.7 | both_known_and_motif | motif_only | known_only | neither |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 28M_atlas_main | atlas_main | 3580 | 24566 | 0.1257 | 0.000118 | 0.0002442 | 0.02858 | 0.003297 | 0.1224 | 0.02528 | 0.8491 |
| 28M_ref_main_full | full_main | 3100 | 24565 | 0.1292 | 0.0001547 | 0.0002035 | 0.03245 | 0.003949 | 0.1252 | 0.0285 | 0.8423 |
| 28M_atlas_short | context | 1034 | 24528 | 0.07942 | 9.581e-05 | 0.0002446 | 0.36 | 0.02418 | 0.05524 | 0.3359 | 0.5847 |
| 28M_main_sub1034_s0 | subsample | 1034 | 24559 | 0.1088 | 0.0001425 | 0.0004072 | 0.1031 | 0.01063 | 0.09821 | 0.09251 | 0.7986 |
| 28M_main_sub1034_s1 | subsample | 1034 | 24559 | 0.1075 | 0.0001059 | 0.0002443 | 0.1074 | 0.01006 | 0.09744 | 0.09732 | 0.7952 |
| 28M_main_sub1034_s2 | subsample | 1034 | 24560 | 0.1052 | 0.0001527 | 0.0001629 | 0.1002 | 0.009446 | 0.09573 | 0.0908 | 0.804 |
| 28M_atlas_long | context | 620 | 24562 | 0.1201 | 0.0001445 | 0.0002443 | 0.2497 | 0.0263 | 0.09376 | 0.2233 | 0.6566 |
| 28M_main_sub620_s0 | subsample | 620 | 24552 | 0.09624 | 0.0001018 | 0.0002851 | 0.2546 | 0.02676 | 0.06949 | 0.2279 | 0.6759 |
| 28M_main_sub620_s1 | subsample | 620 | 24555 | 0.1028 | 8.756e-05 | 0.0003665 | 0.2608 | 0.02826 | 0.07457 | 0.2325 | 0.6646 |
| 28M_main_sub620_s2 | subsample | 620 | 24551 | 0.09975 | 9.776e-05 | 0.0003666 | 0.2524 | 0.0277 | 0.07205 | 0.2247 | 0.6756 |
| 28M_atlas_xlong | context | 48 | 24554 | 0.1351 | 0.0001222 | 0.008023 | 0.4423 | 0.06516 | 0.06989 | 0.3772 | 0.4878 |
| 28M_main_sub48_s0 | subsample | 48 | 24490 | 0.06986 | 0.0001062 | 0.0005308 | 0.3027 | 0.03312 | 0.03675 | 0.2696 | 0.6606 |
| 28M_main_sub48_s1 | subsample | 48 | 24481 | 0.04077 | 7.761e-05 | 0.0001634 | 0.2061 | 0.009518 | 0.03125 | 0.1966 | 0.7626 |
| 28M_main_sub48_s2 | subsample | 48 | 24497 | 0.08201 | 0.0001878 | 0.002694 | 0.3242 | 0.04498 | 0.03703 | 0.2792 | 0.6388 |

### 123M

| tag | role | n_series | n_live | cohesive_top6 | null_top6_mean | cohesive_mid6 | one_job_0.7 | both_known_and_motif | motif_only | known_only | neither |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 123M_atlas_main | atlas_main | 3580 | 68463 | 0.1029 | 0.0001176 | 0.0003798 | 0.02674 | 0.003082 | 0.09979 | 0.02366 | 0.8735 |
| 123M_ref_main_full | full_main | 3100 | 68461 | 0.108 | 0.0001483 | 0.000336 | 0.03 | 0.003798 | 0.1042 | 0.0262 | 0.8658 |
| 123M_atlas_short | context | 1034 | 68361 | 0.06226 | 9.655e-05 | 0.0003365 | 0.3289 | 0.01862 | 0.04364 | 0.3102 | 0.6275 |
| 123M_main_sub1034_s0 | subsample | 1034 | 68451 | 0.08999 | 0.000114 | 0.000336 | 0.09988 | 0.009072 | 0.08092 | 0.09081 | 0.8192 |
| 123M_main_sub1034_s1 | subsample | 1034 | 68452 | 0.08868 | 0.000114 | 0.0002922 | 0.1015 | 0.00897 | 0.0797 | 0.09249 | 0.8188 |
| 123M_main_sub1034_s2 | subsample | 1034 | 68453 | 0.0852 | 0.0001658 | 0.0003798 | 0.1024 | 0.009189 | 0.07601 | 0.0932 | 0.8216 |
| 123M_atlas_long | context | 620 | 68461 | 0.1024 | 0.0001256 | 0.0002775 | 0.2507 | 0.02341 | 0.07902 | 0.2273 | 0.6703 |
| 123M_main_sub620_s0 | subsample | 620 | 68448 | 0.08165 | 0.0001154 | 0.0003652 | 0.2389 | 0.02143 | 0.06022 | 0.2175 | 0.7008 |
| 123M_main_sub620_s1 | subsample | 620 | 68445 | 0.08389 | 0.0001081 | 0.0002922 | 0.2447 | 0.02431 | 0.05958 | 0.2203 | 0.6958 |
| 123M_main_sub620_s2 | subsample | 620 | 68447 | 0.07949 | 9.131e-05 | 0.0003945 | 0.2392 | 0.02231 | 0.05718 | 0.2169 | 0.7036 |
| 123M_atlas_xlong | context | 48 | 68422 | 0.1613 | 0.0001308 | 0.006577 | 0.4322 | 0.0781 | 0.08316 | 0.3541 | 0.4847 |
| 123M_main_sub48_s0 | subsample | 48 | 68256 | 0.06098 | 0.0001421 | 0.0008351 | 0.2725 | 0.02943 | 0.03154 | 0.2431 | 0.6959 |
| 123M_main_sub48_s1 | subsample | 48 | 68180 | 0.03483 | 6.674e-05 | 0.0002053 | 0.1708 | 0.007598 | 0.02724 | 0.1632 | 0.802 |
| 123M_main_sub48_s2 | subsample | 48 | 68255 | 0.07785 | 0.0001934 | 0.004073 | 0.2946 | 0.04104 | 0.03682 | 0.2536 | 0.6686 |

### 209M

| tag | role | n_series | n_live | cohesive_top6 | null_top6_mean | cohesive_mid6 | one_job_0.7 | both_known_and_motif | motif_only | known_only | neither |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 209M_atlas_main | atlas_main | 3580 | 98273 | 0.1032 | 0.0001145 | 0.0002239 | 0.02335 | 0.002401 | 0.1008 | 0.02095 | 0.8758 |
| 209M_ref_main_full | full_main | 3100 | 98268 | 0.1075 | 0.0001242 | 0.0004172 | 0.02609 | 0.002809 | 0.1047 | 0.02328 | 0.8693 |
| 209M_atlas_short | context | 1034 | 98144 | 0.05911 | 9.017e-05 | 0.0002649 | 0.3082 | 0.01718 | 0.04193 | 0.291 | 0.6499 |
| 209M_main_sub1034_s0 | subsample | 1034 | 98238 | 0.09122 | 0.0001369 | 0.0003156 | 0.09293 | 0.008856 | 0.08236 | 0.08407 | 0.8247 |
| 209M_main_sub1034_s1 | subsample | 1034 | 98231 | 0.08832 | 0.0001232 | 0.0003054 | 0.09267 | 0.008134 | 0.08019 | 0.08453 | 0.8271 |
| 209M_main_sub1034_s2 | subsample | 1034 | 98236 | 0.08576 | 0.0001578 | 0.0003665 | 0.09469 | 0.008195 | 0.07757 | 0.0865 | 0.8277 |
| 209M_atlas_long | context | 620 | 98257 | 0.1015 | 0.0001318 | 0.0003359 | 0.2419 | 0.0221 | 0.07944 | 0.2198 | 0.6787 |
| 209M_main_sub620_s0 | subsample | 620 | 98210 | 0.08127 | 0.0001288 | 0.0004888 | 0.2275 | 0.02088 | 0.06038 | 0.2066 | 0.7122 |
| 209M_main_sub620_s1 | subsample | 620 | 98205 | 0.08414 | 9.623e-05 | 0.000336 | 0.2343 | 0.02387 | 0.06027 | 0.2104 | 0.7055 |
| 209M_main_sub620_s2 | subsample | 620 | 98201 | 0.07911 | 9.47e-05 | 0.0004277 | 0.2252 | 0.0214 | 0.05772 | 0.2038 | 0.7171 |
| 209M_atlas_xlong | context | 48 | 98177 | 0.1688 | 0.0001375 | 0.006122 | 0.431 | 0.08062 | 0.0882 | 0.3503 | 0.4808 |
| 209M_main_sub48_s0 | subsample | 48 | 97959 | 0.06494 | 0.0001521 | 0.0006329 | 0.264 | 0.03131 | 0.03363 | 0.2326 | 0.7024 |
| 209M_main_sub48_s1 | subsample | 48 | 97852 | 0.03543 | 7.562e-05 | 0.0002146 | 0.1591 | 0.007225 | 0.02821 | 0.1519 | 0.8127 |
| 209M_main_sub48_s2 | subsample | 48 | 97936 | 0.07839 | 0.0002134 | 0.003278 | 0.2903 | 0.04176 | 0.03663 | 0.2486 | 0.673 |

### Matched series counts

Each context-length corpus next to a random subsample of the main corpus with the same number of series (seeds pooled) and next to the full main corpus.

| size | context_run | n_series | context_cohesive_top6 | cohesive_top6_mean | cohesive_top6_min | cohesive_top6_max | full_main_cohesive_top6 | context_one_job | one_job_mean | full_main_one_job | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5M | 5M_atlas_short | 1034 | 0.09576 | 0.1169 | 0.1166 | 0.1175 | 0.1424 | 0.4147 | 0.1148 | 0.04812 | 3 |
| 5M | 5M_atlas_long | 620 | 0.1293 | 0.1069 | 0.1065 | 0.1077 | 0.1424 | 0.2419 | 0.2641 | 0.04812 | 3 |
| 5M | 5M_atlas_xlong | 48 | 0.1371 | 0.06765 | 0.04622 | 0.08092 | 0.1424 | 0.4747 | 0.3263 | 0.04812 | 3 |
| 28M | 28M_atlas_short | 1034 | 0.07942 | 0.1072 | 0.1052 | 0.1088 | 0.1292 | 0.36 | 0.1036 | 0.03245 | 3 |
| 28M | 28M_atlas_long | 620 | 0.1201 | 0.09961 | 0.09624 | 0.1028 | 0.1292 | 0.2497 | 0.2559 | 0.03245 | 3 |
| 28M | 28M_atlas_xlong | 48 | 0.1351 | 0.06421 | 0.04077 | 0.08201 | 0.1292 | 0.4423 | 0.2777 | 0.03245 | 3 |
| 123M | 123M_atlas_short | 1034 | 0.06226 | 0.08795 | 0.0852 | 0.08999 | 0.108 | 0.3289 | 0.1012 | 0.03 | 3 |
| 123M | 123M_atlas_long | 620 | 0.1024 | 0.08168 | 0.07949 | 0.08389 | 0.108 | 0.2507 | 0.2409 | 0.03 | 3 |
| 123M | 123M_atlas_xlong | 48 | 0.1613 | 0.05789 | 0.03483 | 0.07785 | 0.108 | 0.4322 | 0.246 | 0.03 | 3 |
| 209M | 209M_atlas_short | 1034 | 0.05911 | 0.08843 | 0.08576 | 0.09122 | 0.1075 | 0.3082 | 0.09343 | 0.02609 | 3 |
| 209M | 209M_atlas_long | 620 | 0.1015 | 0.08151 | 0.07911 | 0.08414 | 0.1075 | 0.2419 | 0.229 | 0.02609 | 3 |
| 209M | 209M_atlas_xlong | 48 | 0.1688 | 0.05958 | 0.03543 | 0.07839 | 0.1075 | 0.431 | 0.2378 | 0.02609 | 3 |

### Sensitivity (vocabulary size, threshold, normalization, vocabulary refitted inside each run)

| tag | cohesive_top6 | cohesive_top6_K8 | cohesive_top6_K32 | cohesive_top5 | cohesive_top8 | cohesive_top6_scaled | cohesive_top6_refit | all_top_context_tokens | cohesive_top6_context_only_poles | cluster_share_max |
|---|---|---|---|---|---|---|---|---|---|---|
| 5M_atlas_main | 0.1371 | 0.186 | 0.102 | 0.2474 | 0.03297 | 0.1357 | 0.1484 | 0.8068 | 0.1487 | 0.1148 |
| 28M_atlas_main | 0.1257 | 0.1802 | 0.08789 | 0.233 | 0.02736 | 0.1347 | 0.1314 | 0.8515 | 0.1301 | 0.1212 |
| 123M_atlas_main | 0.1029 | 0.1553 | 0.07226 | 0.1976 | 0.02098 | 0.1318 | 0.1008 | 0.7943 | 0.1075 | 0.1244 |
| 209M_atlas_main | 0.1032 | 0.1534 | 0.07311 | 0.1968 | 0.02071 | 0.1285 | 0.11 | 0.7938 | 0.1079 | 0.1219 |
| 5M_ref_main_full | 0.1424 | 0.1936 | 0.1028 | 0.2532 | 0.03371 | 0.1376 | 0.1431 | 0.7998 | 0.1562 | 0.1149 |
| 28M_ref_main_full | 0.1292 | 0.1861 | 0.09013 | 0.2408 | 0.02837 | 0.1389 | 0.1306 | 0.8447 | 0.1341 | 0.1215 |
| 123M_ref_main_full | 0.108 | 0.1644 | 0.07483 | 0.2082 | 0.02163 | 0.1385 | 0.1039 | 0.7838 | 0.1143 | 0.1246 |
| 209M_ref_main_full | 0.1075 | 0.1611 | 0.07564 | 0.2064 | 0.0216 | 0.1352 | 0.1086 | 0.7838 | 0.1132 | 0.1231 |

