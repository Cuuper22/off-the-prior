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
