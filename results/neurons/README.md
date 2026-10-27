# Neuron audit of the Timestral MLPs (task t10_neurons)

Ground-truth audit of whether the MLP neurons of the four Timestral sizes (5M, 28M, 123M, 209M) are monosemantic: every neuron, both poles, scored against variables that are known exactly because the audit series are generated. All numbers below are read from files in this folder by `scripts/neurons_readme.py`; the folder names the file each table comes from.

## Definitions

- Neuron: one hidden unit of the SwiGLU MLP. Its value is `z = gate * silu(up)`, the input of the down projection `w3`. Positive pole = `max(z, 0)`, negative pole = `max(-z, 0)`; a pole is one atlas entry. Pole counts: 5M 8,192, 28M 24,576, 123M 68,480, 209M 98,304.
- Forward pass: the plain pass (`model.predict_block` followed by the model's `_refine_block`), without symmetric averaging. Captured activations are stored in fp16 (`n_clipped_fp16` is in `provenance_runs.json`).
- Sign regime of a firing: `2*(gate>0) + (up>0)`. Positive-pole firings come from regime 3 (gate>0, up>0) or regime 0 (both negative); negative-pole firings from regime 2 (gate>0, up<0) or 1 (gate<0, up>0).
- Tokens: a token is a 32-step patch. A 1024-step series gives 32 context tokens and 4 forecast tokens (36 per series).
- Label: for every pole and every variable of the library (`otp/neurons/variables.py`), the cross-validated R^2 of a 12-bin quantile bin-mean readout of the pole's activation from the variable, and the mutual information. Folds and the held-out split are by series. Undefined variable values form their own bin. `label` = variable with the largest R^2; `pair_label` = best pair of core variables (5 x 5 bins); `r2_add` = additive ridge ceiling over the core variables. `E = max(single, pair, additive)` at score time, `ratio = r2_cv / E`.
- Classes at threshold theta and floor f: dead (fewer than 30 label tokens with activation above zero), unexplained (`E < f`), one_job (`ratio >= theta`), mixed (the rest). Shares are over live poles. Intervals: multinomial bootstrap over poles (`_lo`, `_hi`), and range over the 3 folds (`_fold_lo`, `_fold_hi`, per-fold label R^2 with pooled E); `_heldout` uses the held-out series.

## Corpus (`results/neurons/corpus/`)

| variant | T | n_series | n_label | n_heldout | bytes | sha256 |
|---|---|---|---|---|---|---|
| main | 1024 | 3100 | 2480 | 620 | 22048501 | 4c235b6225feb962 |
| mv | 1024 | 480 | 384 | 96 | 3199265 | 9f98f96930328a56 |
| short | 128 | 1034 | 831 | 203 | 1729719 | e0d6bee8059cc215 |
| long | 4096 | 620 | 495 | 125 | 15856565 | e467f4beb108b6b3 |
| xlong | 32768 | 48 | 38 | 10 | 9070615 | e97db2d79d20a8ee |

`main` is univariate; `mv` holds grouped two-row systems (merged with `main` for the main-context runs: 3580 series in total). `short`, `long` and `xlong` are the context-length variants of the same generator recipe (recipe and seeds in `corpus/recipe.json` and each `*_manifest.json`). The npz files are git-ignored and rebuilt with `scripts/neurons_corpus.py` (hashes above).

## Commands that produced the files

```
# corpus (VM)
python3 scripts/neurons_corpus.py
python3 scripts/neurons_real.py --raw-dir <dir> --out results/neurons/real --per-domain 64   # re-fetch: scripts/fetch_realdata.py
# capture + labeling, per size S in 5M 28M 123M 209M (scripts/colab/job_nn_common.sh has the same lines)
python3 scripts/neurons_run.py --size $S --corpus results/neurons/corpus/main.npz,results/neurons/corpus/mv.npz \
    --real results/neurons/real/real.npz --out results/colab_out/nn_${S}_main --acts-sub 2048 --acts-dir results/colab_out/acts \
    --layers-per-chunk 8 --batch 256 --resume          # short: --batch 256; long: --batch 64; xlong: --batch 16
python3 scripts/neurons_steer.py --size $S --runs results/colab_out --corpus results/neurons/corpus/main.npz --out results/colab_out/nn_steer
python3 scripts/neurons_failure.py --size $S --runs results/colab_out --out results/colab_out/nn_failure
python3 scripts/neurons_heads.py --size $S --corpus results/neurons/corpus/main.npz --mv results/neurons/corpus/mv.npz --out results/colab_out/nn_heads_$S
# squeeze
python3 scripts/neurons_squeeze.py train --out results/colab_out/nn_squeeze --steps 3000
python3 scripts/neurons_run.py --size results/colab_out/nn_squeeze/w$W --corpus results/neurons/corpus/main.npz --out results/colab_out/nn_squeeze_w${W}_main   # W in 4 32 256
# collect
python3 scripts/neurons_score.py --runs results/colab_out --corpus-dir results/neurons/corpus --out results/neurons
python3 scripts/neurons_match.py --acts-dir results/colab_out/acts --runs results/colab_out --out results/colab_out/nn_match
python3 scripts/neurons_summarize.py --runs results/colab_out --out results/neurons
python3 scripts/neurons_export_browser.py --poles results/neurons --corpus results/neurons/corpus/main.npz,results/neurons/corpus/mv.npz --real results/neurons/real/real.npz --steering results/neurons/steering_results.csv --out results/neurons/browser
python3 scripts/neurons_export_neuronpedia.py --size 5M --poles results/neurons/5M/poles.npz --corpus results/neurons/corpus/main.npz,results/neurons/corpus/mv.npz --real results/neurons/real/real.npz --out results/neurons/neuronpedia --n-poles 12
python3 scripts/neurons_readme.py --runs results/colab_out --out results/neurons
```

Large intermediate files (`results/colab_out/`: `scores_partial.csv` per size and context, `acts_sub_<size>.npz`, squeeze weights) are not committed; every table below is rebuilt from them with the commands above. The Colab jobs ran from a code-only copy of commit `e9e880a` (branch `task/t10_neurons_colab`, same `otp/`, `scripts/` and `tests/`), because the full repository archive exceeds the 40 MB limit of the Colab upload endpoint; the `git` field of those runs in `provenance_runs.json` is that copy's hash.

## Where the runs ran (`provenance_runs.json`)

| file | size | host | device | git | wall_s | n_series | n_label_series | n_heldout_series | status_counts |
|---|---|---|---|---|---|---|---|---|---|
| nn_123M_main/meta.json | 123M | 85cfa68b4b8b | cuda | 69140f1 | 33.2 | 3580 | 2864 | 716 |  |
| nn_209M_main/meta.json | 209M | de1eefdb6c0c | cuda | 69140f1 | 46.7 | 3580 | 2864 | 716 |  |
| nn_28M_main/meta.json | 28M | LAPTOP-TTI9OLBG | cuda | 6466e97+labels.onehot-device-fix | 527.8 | 3580 | 2864 | 716 |  |
| nn_5M_main/meta.json | 5M | LAPTOP-TTI9OLBG | cuda | 6466e97+labels.onehot-device-fix | 27.9 | 3580 | 2864 | 716 |  |
| nn_123M_short/meta.json | 123M | 85cfa68b4b8b | cuda | 69140f1 | 6.7 | 1034 | 831 | 203 |  |
| nn_209M_short/meta.json | 209M | de1eefdb6c0c | cuda | 69140f1 | 9.1 | 1034 | 831 | 203 |  |
| nn_28M_short/meta.json | 28M | LAPTOP-TTI9OLBG | cuda | 6466e97+labels.onehot-device-fix | 12.2 | 1034 | 831 | 203 |  |
| nn_5M_short/meta.json | 5M | LAPTOP-TTI9OLBG | cuda | 6466e97+labels.onehot-device-fix | 5.6 | 1034 | 831 | 203 |  |
| nn_123M_long/meta.json | 123M | 85cfa68b4b8b | cuda | 69140f1 | 17.9 | 620 | 495 | 125 |  |
| nn_209M_long/meta.json | 209M | de1eefdb6c0c | cuda | 69140f1 | 25.4 | 620 | 495 | 125 |  |
| nn_28M_long/meta.json | 28M | LAPTOP-TTI9OLBG | cuda | 6466e97+labels.onehot-device-fix | 83.1 | 620 | 495 | 125 |  |
| nn_5M_long/meta.json | 5M | LAPTOP-TTI9OLBG | cuda | 6466e97+labels.onehot-device-fix | 15.0 | 620 | 495 | 125 |  |
| nn_123M_xlong/meta.json | 123M | 85cfa68b4b8b | cuda | 69140f1 | 13.3 | 48 | 38 | 10 |  |
| nn_209M_xlong/meta.json | 209M | de1eefdb6c0c | cuda | 69140f1 | 18.6 | 48 | 38 | 10 |  |
| nn_28M_xlong/meta.json | 28M | LAPTOP-TTI9OLBG | cuda | 6466e97+labels.onehot-device-fix | 27.9 | 48 | 38 | 10 |  |
| nn_5M_xlong/meta.json | 5M | LAPTOP-TTI9OLBG | cuda | 6466e97+labels.onehot-device-fix | 10.5 | 48 | 38 | 10 |  |
| nn_squeeze_w256_main/meta.json | results/colab_out/nn_squeeze/w256 | vm | cpu | 66dc7d3 | 241.4 | 3100 | 2480 | 620 |  |
| nn_squeeze_w32_main/meta.json | results/colab_out/nn_squeeze/w32 | vm | cpu | 66dc7d3 | 117.8 | 3100 | 2480 | 620 |  |
| nn_squeeze_w4_main/meta.json | results/colab_out/nn_squeeze/w4 | vm | cpu | 66dc7d3 | 105.1 | 3100 | 2480 | 620 |  |
| nn_steer/meta_123M.json | 123M | 85cfa68b4b8b | cuda | 69140f1 | 31.3 |  |  |  |  |
| nn_steer/meta_209M.json | 209M | de1eefdb6c0c | cuda | 69140f1 | 46.8 |  |  |  |  |
| nn_steer/meta_28M.json | 28M | LAPTOP-TTI9OLBG | cuda | 8b2ae29 | 75.8 |  |  |  |  |
| nn_steer/meta_5M.json | 5M | LAPTOP-TTI9OLBG | cuda | 8b2ae29 | 22.9 |  |  |  |  |
| nn_failure/meta_123M.json | 123M | 1bff3bbb5a82 | cuda | 81b092b | 41.0 |  |  |  | {'failing': 112, 'reference': 67, 'other': 40} |
| nn_failure/meta_209M.json | 209M | 1bff3bbb5a82 | cuda | 81b092b | 80.1 |  |  |  | {'failing': 88, 'reference': 75, 'other': 56} |
| nn_failure/meta_28M.json | 28M | LAPTOP-TTI9OLBG | cuda | 8b2ae29 | 43.8 |  |  |  | {'failing': 144, 'reference': 35, 'other': 40} |
| nn_failure/meta_5M.json | 5M | vm | cpu | e0b4a1e | 26.8 |  |  |  | {'failing': 160, 'reference': 35, 'other': 24} |
| nn_heads_123M/meta.json | 123M | 85cfa68b4b8b |  | 69140f1 | 9.7 | 3100 |  |  |  |
| nn_heads_209M/meta.json | 209M | de1eefdb6c0c |  | 69140f1 | 12.2 | 3100 |  |  |  |
| nn_heads_28M/meta.json | 28M | LAPTOP-TTI9OLBG |  | 8b2ae29 | 18.4 | 3100 |  |  |  |
| nn_heads_5M/meta.json | 5M | LAPTOP-TTI9OLBG |  | 8b2ae29 | 15.8 | 3100 |  |  |  |
| nn_match/meta.json |  | vm |  | 66dc7d3 |  |  |  |  |  |

Sizes 5M and 28M were captured and labeled on the laptop GPU lane (host `LAPTOP-TTI9OLBG`, git `6466e97+labels.onehot-device-fix`: the code of `6466e97` with the one-line `labels.onehot` device fix that `7ae89c2` commits with a test). Sizes 123M and 209M ran as two parallel Colab sessions on a G4 GPU (RTX PRO 6000 Blackwell, 97,887 MiB; setup logs in `results/colab/nn-123M/setup.txt` and `nn-209M/setup.txt`). The `host`, `device` and `git` columns above give the lane of every steering, failure-link and head run; the full map used one laptop GPU and two Colab GPU sessions at once, so it did not need more than 2 A100 sessions.

## 1. One-job, mixed and unexplained poles (`one_job_curves.csv`, `scaling.csv`)

Share of live poles by class, theta 0.5 to 0.9, for three floors (all layers, both poles).

one_job, floor 0.05

| theta | 5M | 28M | 123M | 209M |
|---|---|---|---|---|
| 0.5 | 0.1915 | 0.1734 | 0.1656 | 0.1552 |
| 0.6 | 0.09587 | 0.08027 | 0.07633 | 0.06892 |
| 0.7 | 0.04592 | 0.03029 | 0.02889 | 0.02515 |
| 0.8 | 0.01856 | 0.01034 | 0.008837 | 0.007215 |
| 0.9 | 0.006961 | 0.003541 | 0.002235 | 0.00173 |

unexplained, floor 0.05

| theta | 5M | 28M | 123M | 209M |
|---|---|---|---|---|
| 0.5 | 0.03285 | 0.0256 | 0.02197 | 0.02425 |
| 0.6 | 0.03285 | 0.0256 | 0.02197 | 0.02425 |
| 0.7 | 0.03285 | 0.0256 | 0.02197 | 0.02425 |
| 0.8 | 0.03285 | 0.0256 | 0.02197 | 0.02425 |
| 0.9 | 0.03285 | 0.0256 | 0.02197 | 0.02425 |

one_job, floor 0.1

| theta | 5M | 28M | 123M | 209M |
|---|---|---|---|---|
| 0.5 | 0.1833 | 0.157 | 0.1496 | 0.14 |
| 0.6 | 0.09319 | 0.07478 | 0.06991 | 0.06332 |
| 0.7 | 0.04531 | 0.02858 | 0.02674 | 0.02335 |
| 0.8 | 0.01844 | 0.01001 | 0.008121 | 0.006716 |
| 0.9 | 0.006839 | 0.00346 | 0.00203 | 0.001598 |

unexplained, floor 0.1

| theta | 5M | 28M | 123M | 209M |
|---|---|---|---|---|
| 0.5 | 0.1205 | 0.1417 | 0.1601 | 0.1653 |
| 0.6 | 0.1205 | 0.1417 | 0.1601 | 0.1653 |
| 0.7 | 0.1205 | 0.1417 | 0.1601 | 0.1653 |
| 0.8 | 0.1205 | 0.1417 | 0.1601 | 0.1653 |
| 0.9 | 0.1205 | 0.1417 | 0.1601 | 0.1653 |

one_job, floor 0.2

| theta | 5M | 28M | 123M | 209M |
|---|---|---|---|---|
| 0.5 | 0.1409 | 0.1091 | 0.1011 | 0.09335 |
| 0.6 | 0.07426 | 0.05251 | 0.04924 | 0.04385 |
| 0.7 | 0.03823 | 0.02145 | 0.01997 | 0.01717 |
| 0.8 | 0.01759 | 0.008182 | 0.006631 | 0.00521 |
| 0.9 | 0.006717 | 0.003053 | 0.001811 | 0.001343 |

unexplained, floor 0.2

| theta | 5M | 28M | 123M | 209M |
|---|---|---|---|---|
| 0.5 | 0.3989 | 0.4726 | 0.5236 | 0.5352 |
| 0.6 | 0.3989 | 0.4726 | 0.5236 | 0.5352 |
| 0.7 | 0.3989 | 0.4726 | 0.5236 | 0.5352 |
| 0.8 | 0.3989 | 0.4726 | 0.5236 | 0.5352 |
| 0.9 | 0.3989 | 0.4726 | 0.5236 | 0.5352 |

Mixed share, floor 0.1

| theta | 5M | 28M | 123M | 209M |
|---|---|---|---|---|
| 0.5 | 0.6961 | 0.7013 | 0.6903 | 0.6947 |
| 0.6 | 0.7863 | 0.7835 | 0.77 | 0.7714 |
| 0.7 | 0.8341 | 0.8297 | 0.8131 | 0.8114 |
| 0.8 | 0.861 | 0.8483 | 0.8317 | 0.828 |
| 0.9 | 0.8726 | 0.8548 | 0.8378 | 0.8331 |

Scaling at theta 0.7, floor 0.1 with bootstrap interval (poles) and fold range (`scaling.csv`)

| size | n_poles | n_dead | n_live | one_job | one_job_lo | one_job_hi | one_job_fold_lo | one_job_fold_hi | one_job_heldout | mixed | mixed_lo | mixed_hi | unexplained | unexplained_lo | unexplained_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5M | 8192 | 4 | 8188 | 0.04531 | 0.04079 | 0.04995 | 0.04665 | 0.04763 | 0.04104 | 0.8341 | 0.8263 | 0.8418 | 0.1205 | 0.1138 | 0.1272 |
| 28M | 24576 | 10 | 24566 | 0.02858 | 0.02642 | 0.03063 | 0.03012 | 0.03354 | 0.02398 | 0.8297 | 0.825 | 0.8342 | 0.1417 | 0.1373 | 0.1457 |
| 123M | 68480 | 17 | 68463 | 0.02674 | 0.02556 | 0.02806 | 0.02796 | 0.03151 | 0.0221 | 0.8131 | 0.8101 | 0.8158 | 0.1601 | 0.1575 | 0.1629 |
| 209M | 98304 | 31 | 98273 | 0.02335 | 0.0223 | 0.02435 | 0.0251 | 0.02741 | 0.01908 | 0.8114 | 0.8089 | 0.8139 | 0.1653 | 0.1628 | 0.1677 |

By layer third and by layer, theta 0.7, floor 0.1: rows with `scope` early / mid / late and layer_k in `one_job_curves.csv` (columns `one_job`, `mixed`, `unexplained`, with intervals and the held-out share). Layer-third rows:

| size | scope | n_live | one_job | one_job_lo | one_job_hi | mixed | unexplained |
|---|---|---|---|---|---|---|---|
| 5M | early | 3072 | 0.04785 | 0.04102 | 0.05486 | 0.8838 | 0.06836 |
| 5M | mid | 3068 | 0.04303 | 0.03536 | 0.04954 | 0.9166 | 0.04042 |
| 5M | late | 2048 | 0.04492 | 0.03565 | 0.05371 | 0.6362 | 0.3189 |
| 28M | early | 8185 | 0.03653 | 0.03268 | 0.04081 | 0.8354 | 0.128 |
| 28M | mid | 8192 | 0.02136 | 0.01819 | 0.02466 | 0.9058 | 0.07288 |
| 28M | late | 8189 | 0.02784 | 0.02424 | 0.03181 | 0.748 | 0.2242 |
| 123M | early | 23965 | 0.03372 | 0.03138 | 0.03587 | 0.826 | 0.1403 |
| 123M | mid | 23967 | 0.01953 | 0.01759 | 0.02122 | 0.8191 | 0.1614 |
| 123M | late | 20531 | 0.02703 | 0.02484 | 0.0292 | 0.7911 | 0.1819 |
| 209M | early | 32766 | 0.02777 | 0.02589 | 0.02959 | 0.8238 | 0.1484 |
| 209M | mid | 32767 | 0.01852 | 0.01704 | 0.01995 | 0.8168 | 0.1647 |
| 209M | late | 32740 | 0.02376 | 0.02222 | 0.02538 | 0.7934 | 0.1828 |

By sign, theta 0.7, floor 0.1

| size | sign | n_live | one_job | mixed | unexplained |
|---|---|---|---|---|---|
| 5M | pos | 4093 | 0.04618 | 0.8317 | 0.1222 |
| 5M | neg | 4095 | 0.04444 | 0.8366 | 0.1189 |
| 28M | pos | 12283 | 0.02841 | 0.8318 | 0.1398 |
| 28M | neg | 12283 | 0.02874 | 0.8276 | 0.1436 |
| 123M | pos | 34235 | 0.02728 | 0.8119 | 0.1608 |
| 123M | neg | 34228 | 0.02621 | 0.8143 | 0.1595 |
| 209M | pos | 49138 | 0.02359 | 0.81 | 0.1664 |
| 209M | neg | 49135 | 0.02312 | 0.8127 | 0.1642 |

Label groups of the one-job poles (`label_groups.csv`)

| group | 5M | 28M | 123M | 209M |
|---|---|---|---|---|
| chaos | 0.01348 | 0.04986 | 0.2294 | 0.2911 |
| clustering |  | 0.009971 | 0.01147 | 0.01917 |
| coupling |  | 0.007123 | 0.01638 | 0.01002 |
| curvature | 0.005391 | 0.008547 | 0.01201 | 0.01307 |
| family | 0.01348 | 0.04701 | 0.07318 | 0.06972 |
| level | 0.01078 | 0.004274 | 0.008192 | 0.003486 |
| noise | 0.124 | 0.1581 | 0.136 | 0.125 |
| period | 0.1159 | 0.1937 | 0.1333 | 0.1381 |
| persistence | 0.002695 | 0.007123 | 0.0071 | 0.006536 |
| phase | 0.08895 | 0.06125 | 0.04041 | 0.04183 |
| position | 0.5876 | 0.4103 | 0.2807 | 0.2318 |
| regime |  |  | 0.0005462 |  |
| slope | 0.03504 | 0.03419 | 0.03932 | 0.03704 |
| tail | 0.002695 | 0.008547 | 0.01201 | 0.01307 |

## 2. Predictive check on held-out series (`heldout_check.csv`)

| size | poles | n | r2_cv_mean | r2_heldout_mean | gap_mean | gap_median | gap_p95 | pearson_cv_heldout | share_heldout_ratio_ge_theta |
|---|---|---|---|---|---|---|---|---|---|
| 5M | all_live | 8188 | 0.1044 | 0.1031 | 0.001291 | 0.0006285 | 0.0215 | 0.9878 | 0.0414 |
| 5M | one_job | 371 | 0.3256 | 0.3233 | 0.002305 | 0.0007 | 0.03218 | 0.9934 | 0.8518 |
| 5M | mixed | 6830 | 0.1042 | 0.1029 | 0.001371 | 0.000718 | 0.02261 | 0.9769 | 0.002782 |
| 28M | all_live | 24566 | 0.0895 | 0.08769 | 0.001811 | 0.0008195 | 0.0245 | 0.9783 | 0.02589 |
| 28M | one_job | 702 | 0.2808 | 0.2734 | 0.007432 | 0.0035 | 0.0642 | 0.9815 | 0.7407 |
| 28M | mixed | 20383 | 0.09394 | 0.09213 | 0.00181 | 0.00096 | 0.02547 | 0.968 | 0.003385 |
| 123M | all_live | 68463 | 0.0837 | 0.08137 | 0.002338 | 0.000967 | 0.02701 | 0.9752 | 0.02466 |
| 123M | one_job | 1831 | 0.2634 | 0.2542 | 0.009177 | 0.004571 | 0.06613 | 0.9751 | 0.7116 |
| 123M | mixed | 55669 | 0.08939 | 0.08692 | 0.002475 | 0.00124 | 0.02805 | 0.9655 | 0.003664 |
| 209M | all_live | 98273 | 0.08064 | 0.07872 | 0.001922 | 0.00079 | 0.02536 | 0.9739 | 0.02141 |
| 209M | one_job | 2295 | 0.2502 | 0.2417 | 0.00855 | 0.00422 | 0.06421 | 0.9688 | 0.7137 |
| 209M | mixed | 79734 | 0.08729 | 0.08529 | 0.001997 | 0.000946 | 0.0266 | 0.9655 | 0.003098 |

## 3. Steering (`steering_preregistration.md`, `steering_results.csv`, `steering_effects.csv`, `steering_summary.csv`)

The pre-registration (prediction table, matching criterion, sampling rule) was committed before any steering run. `role` labeled = poles drawn for the table of predictions; `complement` = the other pole of a drawn neuron; `control` = random poles, each judged for level, slope and spread against a random direction (the null rate).

Essay style (a qualitative label per pole plus a steering check) is this section together with the `label` and `motif_*` columns of `poles.npz`; this project's known-variable scoring (R^2 and mutual information against the generating variables, as curves over thresholds) is sections 1 and 2. The browser rows carry both for each pole.

| size | role | n | n_match | match_rate | ci_lo | ci_hi | monotone_rate | size_rate | interval_rate |
|---|---|---|---|---|---|---|---|---|---|
| 123M | labeled | 68 | 0 | 0 | 0 | 0.05347 | 0.5735 | 0 | 0.3971 |
| 123M | complement | 26 | 0 | 0 | 0 | 0.1287 | 0.5 | 0 | 0.5769 |
| 123M | control | 432 | 0 | 0 | 0 | 0.008814 | 0.4861 | 0 | 0.375 |
| 209M | labeled | 63 | 0 | 0 | 0 | 0.05747 | 0.6032 | 0 | 0.381 |
| 209M | complement | 20 | 0 | 0 | 0 | 0.1611 | 0.65 | 0 | 0.35 |
| 209M | control | 432 | 0 | 0 | 0 | 0.008814 | 0.5208 | 0 | 0.3403 |
| 28M | labeled | 47 | 0 | 0 | 0 | 0.07556 | 0.5319 | 0 | 0.3191 |
| 28M | complement | 20 | 0 | 0 | 0 | 0.1611 | 0.6 | 0 | 0.45 |
| 28M | control | 432 | 0 | 0 | 0 | 0.008814 | 0.4815 | 0 | 0.4005 |
| 5M | labeled | 25 | 0 | 0 | 0 | 0.1332 | 0.32 | 0 | 0.4 |
| 5M | complement | 11 | 0 | 0 | 0 | 0.2588 | 0.3636 | 0 | 0.4546 |
| 5M | control | 432 | 5 | 0.01157 | 0.004953 | 0.0268 | 0.4491 | 0.01157 | 0.463 |

By predicted quantity group (labeled poles)

| size | group | n | n_match | match_rate | ci_lo | ci_hi | monotone_rate | size_rate | interval_rate |
|---|---|---|---|---|---|---|---|---|---|
| 123M | level | 11 | 0 | 0 | 0 | 0.2588 | 0.5454 | 0 | 0.1818 |
| 123M | slope | 24 | 0 | 0 | 0 | 0.138 | 0.7083 | 0 | 0.4167 |
| 123M | spread | 21 | 0 | 0 | 0 | 0.1546 | 0.5714 | 0 | 0.381 |
| 123M | amp | 12 | 0 | 0 | 0 | 0.2425 | 0.3333 | 0 | 0.5833 |
| 123M | all | 68 | 0 | 0 | 0 | 0.05347 | 0.5735 | 0 | 0.3971 |
| 209M | level | 8 | 0 | 0 | 0 | 0.3244 | 0.25 | 0 | 0.125 |
| 209M | slope | 24 | 0 | 0 | 0 | 0.138 | 0.75 | 0 | 0.3333 |
| 209M | spread | 24 | 0 | 0 | 0 | 0.138 | 0.625 | 0 | 0.4167 |
| 209M | amp | 7 | 0 | 0 | 0 | 0.3543 | 0.4286 | 0 | 0.7143 |
| 209M | all | 63 | 0 | 0 | 0 | 0.05747 | 0.6032 | 0 | 0.381 |
| 28M | level | 3 | 0 | 0 | 0 | 0.5615 | 0.6667 | 0 | 1 |
| 28M | slope | 17 | 0 | 0 | 0 | 0.1843 | 0.4118 | 0 | 0.1177 |
| 28M | spread | 15 | 0 | 0 | 0 | 0.2039 | 0.6 | 0 | 0.2667 |
| 28M | amp | 12 | 0 | 0 | 0 | 0.2425 | 0.5833 | 0 | 0.5 |
| 28M | all | 47 | 0 | 0 | 0 | 0.07556 | 0.5319 | 0 | 0.3191 |
| 5M | level | 4 | 0 | 0 | 0 | 0.4899 | 0.25 | 0 | 0.5 |
| 5M | slope | 13 | 0 | 0 | 0 | 0.2281 | 0.4615 | 0 | 0.3846 |
| 5M | spread | 8 | 0 | 0 | 0 | 0.3244 | 0.125 | 0 | 0.375 |
| 5M | all | 25 | 0 | 0 | 0 | 0.1332 | 0.32 | 0 | 0.4 |

Largest |effect(scale 4) - effect(scale 0)| over the 203 steered labeled poles of all sizes, in units of the context standard deviation (column `effect_size`): 0.04115; the size thresholds of the pre-registration are 0.05 (level, slope, spread, amp) and 0.1 (phase). Median |effect_size|: 0.0003437.

## 4. Failure-map link (`failure_link.csv`, `failure_configs.csv`, `failure_samples.csv`)

Contexts: the three chaos families of the failure map (Mackey-Glass, Rossler, Kuramoto), 8 seeds per config, last 2048 steps of an 8192-step context, horizon 128, oracle with 200 paths (the realized futures can differ from `results/map/chaos`); a config is failing for a size when the geometric-mean MASE of Timestral is at least 1.5 times the oracle's and the method of analogues is lower than Timestral's. Reference contexts: configs with ratio below 1.5, plus the held-out in-prior seasonal series (period 8 to 64, first 2048 steps) of the long corpus. Elevated poles: Cohen's d >= 0.8 for activation on failing versus reference contexts (last 16 context patches and forecast tokens).

Configs per status

| size | failing | other | reference |
|---|---|---|---|
| 5M | 20 | 3 | 0 |
| 28M | 18 | 5 | 0 |
| 123M | 14 | 5 | 4 |
| 209M | 11 | 7 | 5 |

Activation of period- and phase-labeled one-job poles (`mean_act_*`), and of all live poles

| size | set | n_poles | mean_act_failing | mean_act_reference | share_higher_on_failing | share_d_ge_thresh | share_d_le_neg_thresh |
|---|---|---|---|---|---|---|---|
| 123M | periodic_one_job | 318 | 0.09008 | 0.1434 | 0.2327 | 0.0566 | 0.2484 |
| 123M | all_live | 68463 | 0.1413 | 0.1341 | 0.4739 | 0.1335 | 0.1558 |
| 209M | periodic_one_job | 413 | 0.08067 | 0.1212 | 0.2833 | 0.05811 | 0.1017 |
| 209M | all_live | 98273 | 0.1318 | 0.1228 | 0.4811 | 0.1393 | 0.1063 |
| 28M | periodic_one_job | 179 | 0.1815 | 0.3491 | 0.3296 | 0.1229 | 0.3743 |
| 28M | all_live | 24566 | 0.1871 | 0.164 | 0.4737 | 0.2087 | 0.3015 |
| 5M | periodic_one_job | 76 | 0.0735 | 0.1462 | 0.3158 | 0.06579 | 0.3421 |
| 5M | all_live | 8188 | 0.3272 | 0.2769 | 0.4577 | 0.2057 | 0.307 |

Elevated poles: count and share that are not one-job (mixed, unexplained or dead)

| size | n_poles | share_not_one_job |
|---|---|---|
| 123M | 9142 | 0.9734 |
| 209M | 13686 | 0.9767 |
| 28M | 5128 | 0.9778 |
| 5M | 1684 | 0.965 |

Label groups among elevated poles against the base rate of the groups (enrichment = share / base rate)

| set | 5M | 28M | 123M | 209M |
|---|---|---|---|---|
| chaos | 2.487 | 2.072 | 1.957 | 2.017 |
| clustering | 2.431 | 1.127 | 0.8109 | 1.053 |
| coupling | 0.5995 | 0.718 | 0.7799 | 0.7145 |
| curvature | 2.012 | 2.033 | 1.359 | 0.9325 |
| family | 1.408 | 1.278 | 0.8778 | 1.071 |
| level | 0.6847 | 0.5107 | 1.08 | 1.04 |
| noise | 0.6733 | 0.7163 | 0.8445 | 0.8213 |
| period | 0.668 | 0.6714 | 0.4875 | 0.5208 |
| persistence | 1.309 | 1.315 | 1.557 | 1.191 |
| phase | 0.7943 | 0.7049 | 0.4941 | 0.4779 |
| position | 0.9161 | 0.8249 | 0.7095 | 0.7583 |
| regime | 1.897 | 1.265 | 0.5327 | 0.6662 |
| slope | 1.331 | 1.202 | 1.603 | 1.107 |
| tail | 0.6617 | 0.7442 | 0.758 | 0.9538 |

Scaling the elevated period/phase poles (all at once) and the same number of random poles of the same layers, on failing contexts

| size | set | n_poles | scale | n_samples | mase_before | mase_after | dist_oracle_before | dist_oracle_after | dist_analog_before | dist_analog_after | corr_template_before | corr_template_after | share_closer_to_oracle | share_closer_to_analog | share_lower_mase |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 123M | periodic_elevated | 18 | 0 | 112 | 1.685 | 1.688 | 1.631 | 1.634 | 1.412 | 1.416 | 0.7677 | 0.7675 | 0.4554 | 0.4821 | 0.4732 |
| 123M | periodic_elevated | 18 | 0.5 | 112 | 1.685 | 1.687 | 1.631 | 1.633 | 1.412 | 1.414 | 0.7677 | 0.7676 | 0.4554 | 0.4732 | 0.4821 |
| 123M | random_same_layers | 18 | 0 | 112 | 1.685 | 1.683 | 1.631 | 1.628 | 1.412 | 1.412 | 0.7677 | 0.7678 | 0.5536 | 0.4821 | 0.5536 |
| 123M | random_same_layers | 18 | 0.5 | 112 | 1.685 | 1.684 | 1.631 | 1.629 | 1.412 | 1.412 | 0.7677 | 0.7678 | 0.5625 | 0.4732 | 0.5714 |
| 209M | periodic_elevated | 24 | 0 | 88 | 2.05 | 2.05 | 1.989 | 1.989 | 1.742 | 1.741 | 0.7117 | 0.7113 | 0.5114 | 0.4773 | 0.4773 |
| 209M | periodic_elevated | 24 | 0.5 | 88 | 2.05 | 2.05 | 1.989 | 1.989 | 1.742 | 1.742 | 0.7117 | 0.7115 | 0.5114 | 0.5 | 0.4773 |
| 209M | random_same_layers | 24 | 0 | 88 | 2.05 | 2.053 | 1.989 | 1.992 | 1.742 | 1.745 | 0.7117 | 0.7117 | 0.4659 | 0.5 | 0.5 |
| 209M | random_same_layers | 24 | 0.5 | 88 | 2.05 | 2.051 | 1.989 | 1.991 | 1.742 | 1.744 | 0.7117 | 0.7117 | 0.4659 | 0.5 | 0.5 |
| 28M | periodic_elevated | 22 | 0 | 144 | 1.754 | 1.759 | 1.679 | 1.685 | 1.419 | 1.426 | 0.8234 | 0.8223 | 0.375 | 0.3403 | 0.3819 |
| 28M | periodic_elevated | 22 | 0.5 | 144 | 1.754 | 1.757 | 1.679 | 1.682 | 1.419 | 1.422 | 0.8234 | 0.8228 | 0.3889 | 0.375 | 0.3889 |
| 28M | random_same_layers | 22 | 0 | 144 | 1.754 | 1.755 | 1.679 | 1.678 | 1.419 | 1.418 | 0.8234 | 0.8233 | 0.4931 | 0.4861 | 0.4792 |
| 28M | random_same_layers | 22 | 0.5 | 144 | 1.754 | 1.754 | 1.679 | 1.678 | 1.419 | 1.418 | 0.8234 | 0.8234 | 0.5139 | 0.5208 | 0.5069 |
| 5M | periodic_elevated | 5 | 0 | 160 | 1.85 | 1.85 | 1.774 | 1.773 | 1.512 | 1.511 | 0.832 | 0.8322 | 0.475 | 0.5062 | 0.4625 |
| 5M | periodic_elevated | 5 | 0.5 | 160 | 1.85 | 1.85 | 1.774 | 1.773 | 1.512 | 1.511 | 0.832 | 0.8321 | 0.4875 | 0.5062 | 0.4625 |
| 5M | random_same_layers | 5 | 0 | 160 | 1.85 | 1.851 | 1.774 | 1.775 | 1.512 | 1.513 | 0.832 | 0.832 | 0.4813 | 0.45 | 0.5062 |
| 5M | random_same_layers | 5 | 0.5 | 160 | 1.85 | 1.851 | 1.774 | 1.774 | 1.512 | 1.512 | 0.832 | 0.832 | 0.4875 | 0.4562 | 0.5125 |

Largest single-pole change of MASE among the 10 elevated period/phase poles with the largest d (ablated alone), per size

| size | set | d | mase_before | mase_after | dist_oracle_before | dist_oracle_after |
|---|---|---|---|---|---|---|
| 28M | L8.600.pos:s_log2_period | 1.258 | 1.754 | 1.758 | 1.679 | 1.683 |
| 123M | L1.629.neg:m_trough_pos | 1.063 | 1.685 | 1.687 | 1.631 | 1.633 |
| 209M | L13.694.neg:s_log2_period | 1.477 | 2.05 | 2.049 | 1.989 | 1.988 |
| 5M | L4.101.neg:s_log2_period | 1.079 | 1.85 | 1.849 | 1.774 | 1.772 |

## 5. Real-data transfer (`transfer.csv`, `transfer_by_group.csv`, `real/`)

468 windows of 1152 steps (context 1024) from 10 domains (windows per domain: astronomy 41, energy 64, epidemiology 38, finance 44, hydrology 43, physiology 24, sensors 64, traffic 64, weather 49, web traffic 37); cut-off 2026-09-30; failed sources: 0. The license manifest is `real/sources.csv` (name, URL, license, license evidence, redistribution flag, date fetched, sha256), the windows are in `real/windows.csv`, the re-fetch script is `scripts/fetch_realdata.py` with `scripts/neurons_real.py`. The raw series and the window array (`real.npz`) are not committed; the manifest holds its sha256. Events: weekend, us_holiday, covid_spring_2020.

| source_id | license | redistributable_derived_windows | date_fetched_utc |
|---|---|---|---|
| chronos_wiki | CC0 (config license field of the Hugging Face dataset card; Wikimedia page view data) | True | 2026-10-27 |
| fred | public domain: citation requested (series tag on each FRED page) | True | 2026-10-27 |
| ghcn | public domain (US Government work) | True | 2026-10-27 |
| monash | cc-by-4.0 (Hugging Face dataset card) | True | 2026-10-27 |
| obf | CC BY 4.0 | True | 2026-10-27 |
| owid | CC BY 4.0 (OWID) | True | 2026-10-27 |
| seattle | Public Domain (Socrata dataset metadata) | True | 2026-10-27 |
| usgs_nwis | public domain (US Government work) | True | 2026-10-27 |

Transfer: the bin-mean readout fitted on synthetic label tokens is applied to the real tokens; `tr_corr` = Pearson correlation of the readout with the pole's real activation, for poles whose label has a measurable analog on real series.

| size | poles | n_with_analog | median_tr_corr | median_tr_r2 | share_corr_ge_0.1 | share_corr_ge_0.3 | share_corr_ge_0.5 | share_r2_gt_0 | share_abs_event0_corr_ge_0.2 | share_abs_event1_corr_ge_0.2 | share_abs_event2_corr_ge_0.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5M | one_job | 361 | 0.4444 | 0.14 | 0.9363 | 0.7562 | 0.421 | 0.7756 | 0 | 0 | 0.00277 |
| 5M | mixed | 6119 | 0.2162 | 0.01667 | 0.8268 | 0.2716 | 0.03693 | 0.6037 | 0.004249 | 0 | 0 |
| 5M | unexplained | 874 | 0.1087 | 0.0007504 | 0.5515 | 0.01602 | 0 | 0.5126 | 0.002288 | 0 | 0 |
| 28M | one_job | 621 | 0.3678 | 0.08798 | 0.9179 | 0.6312 | 0.2947 | 0.7182 | 0.00161 | 0 | 0 |
| 28M | mixed | 16724 | 0.1805 | -0.002748 | 0.7288 | 0.1982 | 0.01949 | 0.4828 | 0.004485 | 0 | 0 |
| 28M | unexplained | 2648 | 0.108 | -0.002096 | 0.5366 | 0.03437 | 0.003399 | 0.4585 | 0.001888 | 0 | 0.0003776 |
| 123M | one_job | 1226 | 0.3496 | 0.0695 | 0.894 | 0.5791 | 0.2692 | 0.6949 | 0.001631 | 0 | 0 |
| 123M | mixed | 42350 | 0.1622 | -0.01026 | 0.674 | 0.1659 | 0.01806 | 0.4341 | 0.003235 | 0 | 2.361e-05 |
| 123M | unexplained | 8188 | 0.1003 | -0.005918 | 0.5012 | 0.02235 | 0.001832 | 0.4062 | 0.001832 | 0 | 0.0002443 |
| 209M | one_job | 1397 | 0.3173 | 0.05804 | 0.8633 | 0.5362 | 0.2226 | 0.6714 | 0.002148 | 0 | 0 |
| 209M | mixed | 59327 | 0.154 | -0.01148 | 0.6565 | 0.155 | 0.01627 | 0.4224 | 0.002697 | 0 | 1.686e-05 |
| 209M | unexplained | 11627 | 0.09049 | -0.007032 | 0.4552 | 0.01496 | 0.001634 | 0.3832 | 0.002408 | 0 | 8.601e-05 |

One-job poles by label group

| size | group | n | median_tr_corr | share_corr_ge_0.3 |
|---|---|---|---|---|
| 5M | curvature | 2 | 0.5005 | 1 |
| 5M | level | 4 | 0.4776 | 1 |
| 5M | noise | 46 | 0.3189 | 0.587 |
| 5M | period | 43 | 0.2718 | 0.4186 |
| 5M | persistence | 1 | 0.04529 | 0 |
| 5M | phase | 33 | 0.2885 | 0.4242 |
| 5M | position | 218 | 0.5455 | 0.8991 |
| 5M | slope | 13 | 0.4323 | 0.8461 |
| 5M | tail | 1 | 0.6476 | 1 |
| 28M | curvature | 6 | 0.4755 | 0.6667 |
| 28M | level | 3 | 0.5143 | 1 |
| 28M | noise | 110 | 0.2526 | 0.3182 |
| 28M | period | 136 | 0.2602 | 0.4485 |
| 28M | persistence | 5 | 0.03836 | 0 |
| 28M | phase | 43 | 0.3436 | 0.6977 |
| 28M | position | 288 | 0.4963 | 0.8438 |
| 28M | slope | 24 | 0.3911 | 0.6667 |
| 28M | tail | 6 | 0.09023 | 0 |
| 123M | curvature | 22 | 0.5279 | 0.6364 |
| 123M | level | 15 | 0.2965 | 0.4667 |
| 123M | noise | 249 | 0.2412 | 0.2329 |
| 123M | period | 244 | 0.2589 | 0.4385 |
| 123M | persistence | 13 | 0.01234 | 0 |
| 123M | phase | 74 | 0.3428 | 0.6622 |
| 123M | position | 514 | 0.4829 | 0.8463 |
| 123M | regime | 1 | 0.1831 | 0 |
| 123M | slope | 72 | 0.3235 | 0.5417 |
| 123M | tail | 22 | 0.07985 | 0.04546 |
| 209M | curvature | 30 | 0.5278 | 0.8 |
| 209M | level | 8 | 0.2716 | 0.375 |
| 209M | noise | 284 | 0.2385 | 0.2359 |
| 209M | period | 317 | 0.1892 | 0.347 |
| 209M | persistence | 15 | 0.05176 | 0 |
| 209M | phase | 96 | 0.3505 | 0.6875 |
| 209M | position | 532 | 0.4498 | 0.8233 |
| 209M | slope | 85 | 0.2926 | 0.4824 |
| 209M | tail | 30 | 0.0725 | 0 |

## 6. Essay-derived checks

### 6.1 Both poles as distinctions (`pole_pairs.csv`, theta 0.7, floor 0.1)

| size | scope | theta | floor | n_neurons | n_both_explained | n_one_pole_dead | n_both_dead | both_one_job | same_label | same_group_not_label | related_groups | unrelated_groups | oj_same_label | oj_same_group | oj_related | oj_unrelated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5M | all | 0.7 | 0.1 | 4096 | 3300 | 4 | 0 | 0.002727 | 0.1715 | 0.1303 | 0.06394 | 0.6342 | 0.6667 | 0.2222 | 0.1111 | 0 |
| 28M | all | 0.7 | 0.1 | 12288 | 9305 | 10 | 0 | 0.002472 | 0.1583 | 0.108 | 0.05427 | 0.6794 | 0.6087 | 0.2174 | 0 | 0.1739 |
| 123M | all | 0.7 | 0.1 | 34240 | 24684 | 17 | 0 | 0.002471 | 0.163 | 0.1027 | 0.05028 | 0.6841 | 0.7705 | 0.1148 | 0 | 0.1148 |
| 209M | all | 0.7 | 0.1 | 49152 | 34972 | 31 | 0 | 0.001373 | 0.1596 | 0.09662 | 0.05307 | 0.6907 | 0.7083 | 0.125 | 0 | 0.1667 |

`same_label` / `same_group_not_label` / `related_groups` / `unrelated_groups`: shares of neurons with both poles explained, by the relation of the two best labels; `oj_*` the same for neurons whose two poles are both one-job.

### 6.2 SwiGLU sign regimes (`sign_regimes.csv`)

| size | sign | n_live | share_of_mass_in_agreeing_regime_mean | dominant_one_regime | two_regime | two_regime_both_explained | of_those_labels_differ_group | of_those_labels_differ_variable | r2_regA_mean | r2_regB_mean | r2_regime_split_gain_over_single |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5M | neg | 4095 | 0.438 | 0.5121 | 0.4879 | 0.1084 | 0.6689 | 0.8356 | 0.1307 | 0.1035 | 0.07051 |
| 5M | pos | 4093 | 0.5755 | 0.5231 | 0.4769 | 0.1065 | 0.7018 | 0.8326 | 0.1044 | 0.134 | 0.0731 |
| 28M | neg | 12283 | 0.5109 | 0.4803 | 0.5197 | 0.08939 | 0.7077 | 0.8197 | 0.1134 | 0.08661 | 0.05897 |
| 28M | pos | 12283 | 0.4935 | 0.4822 | 0.5178 | 0.08288 | 0.7004 | 0.8271 | 0.0853 | 0.1145 | 0.05901 |
| 123M | neg | 34228 | 0.546 | 0.4647 | 0.5353 | 0.07359 | 0.7574 | 0.8511 | 0.1076 | 0.07469 | 0.05196 |
| 123M | pos | 34235 | 0.4542 | 0.4668 | 0.5332 | 0.07224 | 0.7457 | 0.8419 | 0.07519 | 0.107 | 0.05123 |
| 209M | neg | 49135 | 0.56 | 0.457 | 0.543 | 0.06873 | 0.7575 | 0.8401 | 0.104 | 0.07364 | 0.05095 |
| 209M | pos | 49138 | 0.4415 | 0.4552 | 0.5448 | 0.06797 | 0.7461 | 0.841 | 0.07258 | 0.1039 | 0.05042 |

### 6.3 Summary motif: token type and position

Best-label kind among explained poles (`kinds.csv`)

| size | sign | n_explained | best_is_motif | best_is_position | best_is_latent | best_is_family | motif_poles_position_explains_ge_half |
|---|---|---|---|---|---|---|---|
| 5M | neg | 3608 | 0.3301 | 0.224 | 0.4246 | 0.02134 | 0.2107 |
| 5M | pos | 3593 | 0.3421 | 0.2266 | 0.4083 | 0.0231 | 0.2221 |
| 28M | neg | 10519 | 0.2817 | 0.1728 | 0.5123 | 0.03318 | 0.2251 |
| 28M | pos | 10566 | 0.2894 | 0.17 | 0.5057 | 0.03492 | 0.2237 |
| 123M | neg | 28769 | 0.2407 | 0.1384 | 0.5804 | 0.04053 | 0.1944 |
| 123M | pos | 28731 | 0.2458 | 0.1366 | 0.5753 | 0.04232 | 0.1981 |
| 209M | neg | 41068 | 0.2362 | 0.1226 | 0.5982 | 0.04298 | 0.1905 |
| 209M | pos | 40961 | 0.2344 | 0.1232 | 0.6005 | 0.04187 | 0.1926 |

One-job poles with a position-group label (token type or position), by label

| label | 5M | 28M | 123M | 209M |
|---|---|---|---|---|
| pos_from_end | 24 | 66 | 86 | 117 |
| pos_logj | 9 | 23 | 21 | 32 |
| tok_fc | 2 | 3 | 1 | 1 |
| tok_fc_step | 183 | 196 | 406 | 382 |

Share of live poles whose best label is in the position group, by layer third

| size | third | n_live | share_label_position |
|---|---|---|---|
| 123M | early | 23965 | 0.1592 |
| 123M | mid | 23967 | 0.1504 |
| 123M | late | 20531 | 0.1187 |
| 209M | early | 32766 | 0.145 |
| 209M | mid | 32767 | 0.1519 |
| 209M | late | 32740 | 0.09337 |
| 28M | early | 8185 | 0.1611 |
| 28M | mid | 8192 | 0.1982 |
| 28M | late | 8189 | 0.159 |
| 5M | early | 3072 | 0.1667 |
| 5M | mid | 3068 | 0.2539 |
| 5M | late | 2048 | 0.2715 |

Median R^2 of the best label restricted to context tokens (`r2_ctx`) and to forecast tokens (`r2_fc`)

| size | median_r2_ctx | median_r2_fc | median_r2_cv |
|---|---|---|---|
| 123M | 0.06501 | 0.1357 | 0.06302 |
| 209M | 0.06346 | 0.1323 | 0.06139 |
| 28M | 0.07014 | 0.1593 | 0.06877 |
| 5M | 0.08045 | 0.1803 | 0.08 |

### 6.4 Context length (`context_comparison.csv`)

Same pipeline on corpora of 128, 1024, 4096 and 32768 steps; `same_label_as_main`, `same_group_as_main`, `r2_corr_with_main` over poles explained in both.

| size | context | n_poles | n_live | one_job | unexplained | dead | E_mean | n_both_explained | same_label_as_main | same_group_as_main | r2_corr_with_main |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5M | main | 8192 | 8188 | 0.04531 | 0.1205 | 0.0004883 | 0.2568 | 7201 | 1 | 1 | 1 |
| 28M | main | 24576 | 24566 | 0.02858 | 0.1417 | 0.0004069 | 0.2285 | 21085 | 1 | 1 | 1 |
| 123M | main | 68480 | 68463 | 0.02674 | 0.1601 | 0.0002482 | 0.2158 | 57500 | 1 | 1 | 1 |
| 209M | main | 98304 | 98273 | 0.02335 | 0.1653 | 0.0003153 | 0.2118 | 82029 | 1 | 1 | 1 |
| 5M | short | 8192 | 8187 | 0.4147 | 0.2405 | 0.0006103 | 0.2094 | 5913 | 0.2119 | 0.3834 | 0.4556 |
| 28M | short | 24576 | 24528 | 0.36 | 0.3403 | 0.001953 | 0.1656 | 15285 | 0.2313 | 0.3772 | 0.4519 |
| 123M | short | 68480 | 68361 | 0.3289 | 0.4134 | 0.001738 | 0.1468 | 37524 | 0.23 | 0.3713 | 0.3927 |
| 209M | short | 98304 | 98144 | 0.3082 | 0.4566 | 0.001628 | 0.1361 | 50037 | 0.2337 | 0.3648 | 0.3659 |
| 5M | long | 8192 | 8188 | 0.2419 | 0.3137 | 0.0004883 | 0.1885 | 5586 | 0.5514 | 0.6735 | 0.7608 |
| 28M | long | 24576 | 24562 | 0.2497 | 0.377 | 0.0005697 | 0.1628 | 15141 | 0.5436 | 0.663 | 0.7435 |
| 123M | long | 68480 | 68461 | 0.2507 | 0.4105 | 0.0002774 | 0.1545 | 39734 | 0.5439 | 0.6593 | 0.7672 |
| 209M | long | 98304 | 98257 | 0.2419 | 0.4183 | 0.0004781 | 0.1528 | 56207 | 0.5466 | 0.6606 | 0.7591 |
| 5M | xlong | 8192 | 8185 | 0.4747 | 0.3805 | 0.0008545 | 0.1644 | 4956 | 0.1983 | 0.3386 | 0.4148 |
| 28M | xlong | 24576 | 24554 | 0.4423 | 0.4328 | 0.0008952 | 0.1526 | 13382 | 0.1537 | 0.276 | 0.3972 |
| 123M | xlong | 68480 | 68422 | 0.4322 | 0.4438 | 0.000847 | 0.1478 | 35846 | 0.1308 | 0.2519 | 0.4222 |
| 209M | xlong | 98304 | 98177 | 0.431 | 0.4439 | 0.001292 | 0.146 | 51194 | 0.1223 | 0.2466 | 0.4267 |

### 6.5 Shared motif of exemplar lists (`motif_cohesion.csv`)

Cohesion = mean over the 18 motif variables of the standard deviation inside the exemplar list over the standard deviation over all label tokens (1 = random tokens, below 1 = shared motif); `coh_top` = top exemplars, `coh_mid` = mid-range exemplars. Per-pole motif descriptions are the columns `motif_top`, `motif_mid` (synthetic) and the real-data exemplars `real_top_*`, `real_mid_*` in each `<size>/poles.npz`.

| size | sign | poles | n | coh_top_mean | coh_mid_mean | coh_top_median | coh_mid_median |
|---|---|---|---|---|---|---|---|
| 5M | pos | all_live | 4093 | 0.858 | 0.831 | 0.7526 | 0.8152 |
| 5M | pos | one_job | 189 | 0.8324 | 0.8246 | 0.7086 | 0.8055 |
| 5M | pos | mixed | 3404 | 0.8548 | 0.8311 | 0.7499 | 0.8151 |
| 5M | pos | unexplained | 500 | 0.8887 | 0.8328 | 0.7982 | 0.8217 |
| 5M | neg | all_live | 4095 | 0.8599 | 0.8351 | 0.7621 | 0.8168 |
| 5M | neg | one_job | 182 | 0.8464 | 0.8207 | 0.7277 | 0.8099 |
| 5M | neg | mixed | 3426 | 0.8556 | 0.8358 | 0.7537 | 0.8161 |
| 5M | neg | unexplained | 487 | 0.8953 | 0.8358 | 0.8296 | 0.8272 |
| 28M | pos | all_live | 12283 | 0.8348 | 0.835 | 0.7498 | 0.8165 |
| 28M | pos | one_job | 349 | 0.7587 | 0.8384 | 0.686 | 0.813 |
| 28M | pos | mixed | 10217 | 0.8478 | 0.8345 | 0.758 | 0.8153 |
| 28M | pos | unexplained | 1717 | 0.7728 | 0.8377 | 0.7175 | 0.8234 |
| 28M | neg | all_live | 12283 | 0.8281 | 0.8328 | 0.7458 | 0.8161 |
| 28M | neg | one_job | 353 | 0.7486 | 0.8418 | 0.6897 | 0.8255 |
| 28M | neg | mixed | 10166 | 0.8352 | 0.831 | 0.7501 | 0.8144 |
| 28M | neg | unexplained | 1764 | 0.8031 | 0.8411 | 0.7331 | 0.8231 |
| 123M | pos | all_live | 34235 | 0.8211 | 0.8347 | 0.7449 | 0.8167 |
| 123M | pos | one_job | 934 | 0.7393 | 0.8452 | 0.6651 | 0.8226 |
| 123M | pos | mixed | 27797 | 0.8305 | 0.8333 | 0.7536 | 0.8157 |
| 123M | pos | unexplained | 5504 | 0.7879 | 0.8398 | 0.7234 | 0.8213 |
| 123M | neg | all_live | 34228 | 0.8217 | 0.8342 | 0.7439 | 0.8163 |
| 123M | neg | one_job | 897 | 0.715 | 0.8349 | 0.6544 | 0.8118 |
| 123M | neg | mixed | 27872 | 0.8326 | 0.833 | 0.7523 | 0.8154 |
| 123M | neg | unexplained | 5459 | 0.7837 | 0.8406 | 0.7199 | 0.8221 |
| 209M | pos | all_live | 49138 | 0.8275 | 0.8351 | 0.75 | 0.8163 |
| 209M | pos | one_job | 1159 | 0.6996 | 0.8335 | 0.6438 | 0.8132 |
| 209M | pos | mixed | 39802 | 0.8369 | 0.8335 | 0.7585 | 0.8149 |
| 209M | pos | unexplained | 8177 | 0.7998 | 0.8433 | 0.7286 | 0.8237 |
| 209M | neg | all_live | 49135 | 0.8255 | 0.8343 | 0.7484 | 0.8163 |
| 209M | neg | one_job | 1136 | 0.712 | 0.8398 | 0.6533 | 0.8208 |
| 209M | neg | mixed | 39932 | 0.8331 | 0.8336 | 0.7528 | 0.8152 |
| 209M | neg | unexplained | 8067 | 0.804 | 0.8371 | 0.7414 | 0.822 |

### 6.6 Scale ladder (`scale_match.csv`)

For one-job poles of the smaller model: activation correlation with every pole of the larger model on the same 2048 label-set tokens (`acts_sub_<size>.npz`, same token ids for all sizes). `share_two_or_more_ge_0.5`: share of small-model poles that have two or more larger-model poles with correlation >= 0.5.

| small | large | n_one_job_small | median_best_corr | share_best_ge_0.5 | share_best_ge_0.7 | share_two_or_more_ge_0.5 | mean_n_ge50 | n_matched | matched_same_label | matched_same_group | matched_large_r2_minus_small_r2 | matched_large_r2_higher |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5M | 28M | 371 | 0.6247 | 0.8194 | 0.3477 | 0.752 | 17.44 | 129 | 0.8992 | 0.9767 | 0.08054 | 0.5271 |
| 28M | 123M | 702 | 0.5929 | 0.7293 | 0.2806 | 0.6539 | 20.97 | 197 | 0.9239 | 0.9797 | 0.0634 | 0.5787 |
| 123M | 209M | 1828 | 0.5796 | 0.7194 | 0.2412 | 0.6565 | 21.07 | 441 | 0.9138 | 0.9796 | 0.03199 | 0.4989 |
| 5M | 209M | 371 | 0.6215 | 0.8383 | 0.3288 | 0.7682 | 27.74 | 122 | 0.9262 | 0.9754 | 0.06565 | 0.459 |

Distinct labels among one-job poles

| size | n_one_job | distinct_labels | distinct_groups |
|---|---|---|---|
| 123M | 1831 | 35 | 14 |
| 209M | 2295 | 40 | 13 |
| 28M | 702 | 34 | 13 |
| 5M | 371 | 21 | 11 |

### 6.7 Squeeze test (`squeeze_summary.csv`, `squeeze/train_w*.json`)

Three small Timestral-style models (patch 32, d_model 96, 3 heads, 4 time-attention blocks, 21 quantile knots) that differ only in MLP hidden width (4, 32, 256), same initialization seed and training batches, 3000 steps on `otp.generators.primitives.in_prior`. Labeled with the same pipeline on the main corpus.

| model | n_neurons | n_poles | n_live | one_job | mixed | unexplained | E_mean | r2_cv_mean | distinct_labels_one_job | distinct_groups_one_job | mean_best_over_pair_r2 | share_one_job_group_level_only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w256 | 1024 | 2048 | 2048 | 0.02539 | 0.9233 | 0.05127 | 0.2578 | 0.1039 | 6 | 4 | 0.8173 | 0.4615 |
| w32 | 128 | 256 | 256 | 0.01562 | 0.9453 | 0.03906 | 0.3309 | 0.1397 | 4 | 4 | 0.8489 | 0.25 |
| w4 | 16 | 32 | 31 | 0.06452 | 0.871 | 0.06452 | 0.3296 | 0.1471 | 1 | 1 | 0.9456 | 0 |

| mlp_hidden_dim | params | steps | batch_size | lr | final_loss | wall_s | host | git |
|---|---|---|---|---|---|---|---|---|
| 256 | 907776 | 3000 | 32 | 0.001 | 0.7053 | 431.1 | vm | 7ae89c2 |
| 32 | 649728 | 3000 | 32 | 0.001 | 0.7272 | 409.4 | vm | 7ae89c2 |
| 4 | 617472 | 3000 | 32 | 0.001 | 0.726 | 402.3 | vm | de5866f |

### 6.8 Validation beyond correlation (`steering_checks.csv`)

`downstream_flagged`: unmatched steered labels whose largest ablation effect is on another forecast quantity and at least the size threshold; `negative_pole_*`: neurons where a pole and its complement share a label, share with the effect signs the label correlations imply; `upstream_flagged`: steered poles with none of the three most correlated earlier-layer poles in the same label group and a best motif label that explains less than half of the label's R^2 (`upstream.csv` columns `up*` in each `poles.npz`).

| size | n_labeled | n_unmatched | downstream_flagged | downstream_flagged_share | negative_pole_pairs | negative_pole_consistent | negative_pole_share | upstream_checked | upstream_flagged | upstream_flagged_share |
|---|---|---|---|---|---|---|---|---|---|---|
| 123M | 68 | 68 | 0 | 0 | 15 | 8 | 0.5333 | 62 | 0 | 0 |
| 209M | 63 | 63 | 0 | 0 | 15 | 7 | 0.4667 | 57 | 0 | 0 |
| 28M | 47 | 47 | 0 | 0 | 13 | 10 | 0.7692 | 40 | 0 | 0 |
| 5M | 25 | 25 | 0 | 0 | 6 | 4 | 0.6667 | 13 | 0 | 0 |

## 7. Attention heads (`heads_time.csv`, `heads_variate.csv`, `heads_summary.csv`)

Every time-attention head and every variate-attention head of the four models, described by where its forecast-token and context-token queries attend on the main corpus (recency, seasonal lag, post-switch patches, self, first-patch sink, covariate rows for variate heads); `heads_variate.csv` has the two-row systems.

| query | description | 5M | 28M | 123M | 209M |
|---|---|---|---|---|---|
| context | diffuse | 3 | 8 | 29 | 41 |
| context | mixed | 4 | 12 | 33 | 63 |
| context | post-switch patches | 5 | 24 | 90 | 110 |
| context | recent patches | 8 | 19 | 35 | 50 |
| context | seasonal lag | 1 | 4 | 1 | 3 |
| context | self | 3 | 5 | 12 | 19 |
| context | sink (first patch) | 0 | 0 | 0 | 2 |
| forecast | diffuse | 5 | 18 | 45 | 64 |
| forecast | mixed | 5 | 12 | 37 | 47 |
| forecast | post-switch patches | 8 | 27 | 86 | 137 |
| forecast | recent patches | 5 | 14 | 30 | 35 |
| forecast | seasonal lag | 0 | 0 | 0 | 1 |
| forecast | self | 1 | 1 | 2 | 3 |
| forecast | sink (first patch) | 0 | 0 | 0 | 1 |

## 8. Browser and Neuronpedia exports

`browser/`: compact JSON per size and layer (`index.json` describes the field order and the snippet encoding); one row per pole with label, scores, held-out R^2, class, sign-regime share, transfer correlation, steering result and short exemplar snippets (top, mid-range, real). The largest file is 1.63 MB.

`neuronpedia/`: sample of 15 poles of Timestral-5M in the `upload-batch` layout, written by `scripts/neurons_export_neuronpedia.py` (which checks the field limits, no upload). Draft e-mail (not sent): `paper/outreach/neuronpedia_email.md`.

Sources read: the Neuronpedia documentation (`https://docs.neuronpedia.org`, pages on uploading and feature dashboards, read on 2026-10-26) and the repository `https://github.com/hijohnnylin/neuronpedia` at commit `a8a8c331a2954fd2a84cb2a099ece4001618716b` (route `apps/webapp/app/api/feature/upload-batch/route.tsx`, Python client `neuronpedia.np_feature.Feature`).

What does not fit the schema:
- Tokens are patches of 32 real numbers, not text; the export shows block-character sparkline strings and keeps the raw values in `neuronpedia/patches/`.
- The output is 21 quantiles per forecast step, not a token distribution: `topLogits` and `bottomLogits` are empty (no logit lens).
- Four sizes are four separate models; the two poles of a neuron are two features (source sets `ts<size>-mlp-pos`, `ts<size>-mlp-neg`) linked only through explanation text.
- Real-data exemplars have a domain and a license that the schema has no field for (kept in `neuronpedia/extra/`).
- Steering results and attention-head entries have no place on a feature page.
- Label scores (R^2, mutual information, class) go into the explanation text (at most 512 characters) and a side file.

## 9. Cuts, failures and limits

- Colab upload: the first launches for 123M and 209M failed at the upload of the repository archive (more than 40 MB); the jobs were relaunched from a code-only copy and ran once each.
- The failure-link script changed after the first Colab pass of 123M and 209M (seasonal reference contexts, cache of the size-independent contexts); the tables here come from the second pass of all four sizes (the first-pass chaos-only files are in `results/colab/nn-123M/failure_chaos_only/` and `nn-209M/failure_chaos_only/`).
- Per-context tables (`poles_short/long/xlong.npz`) keep the id, label, score and class columns only; exemplar columns exist in `poles.npz`.
- The xlong corpus has 48 series (32768 steps); its per-pole numbers rest on 38 label series.
- Reference contexts of the failure link: 35 seasonal series (the long corpus has 35 held-out seasonal series with period 8 to 64).
- Pole tables are numpy archives (`otp.neurons.score.read_poles`) with float32 numbers, plus a CSV of the core columns per size (`<size>/poles.csv`); parquet files are not committed because the project word checker reads compressed bytes as text and finds list words by chance.

## 10. Follow-up (`followup/`)

- `followup/README.md` holds the follow-up tables: corpus, series-count and library controls for the one-job share (B); steering power, with a positive control by residual-stream directions, group steering and minimum detectable effects (A, pre-registered in `steering_preregistration_followup.md`); and motif labels from patch-shape clusters (C).
- Single-pole steering results in section 3 are re-judged against matched controls in `followup/single_pole_calibrated.csv`.

## Files

`README.md`, `context_comparison.csv`, `failure_configs.csv`, `failure_link.csv`, `failure_samples.csv`, `heads_summary.csv`, `heads_time.csv`, `heads_variate.csv`, `heldout_check.csv`, `kinds.csv`, `label_groups.csv`, `motif_cohesion.csv`, `one_job_curves.csv`, `pole_pairs.csv`, `provenance_readme.json`, `provenance_runs.json`, `provenance_score.json`, `provenance_summarize.json`, `scale_match.csv`, `scaling.csv`, `sign_regimes.csv`, `squeeze_summary.csv`, `steering_checks.csv`, `steering_effects.csv`, `steering_preregistration.md`, `steering_results.csv`, `steering_summary.csv`, `transfer.csv`, `transfer_by_group.csv`; folders `<size>/` (`poles.npz`, `poles_exemplars.npz`, `poles.csv`, `poles_short|long|xlong.npz`), `corpus/`, `real/`, `browser/`, `neuronpedia/`, `squeeze/`.
