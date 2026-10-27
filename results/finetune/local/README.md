# results/finetune/local

Fine-tuning Timestral on failure families: full fine-tune, LoRA (low-rank adapters on the attention and MLP linears; base weights frozen), and data mixing (a share of every batch is replay of in-prior series: random trend + season + AR-noise, the kind of data Timestral was pretrained on). Files:

- `scores.csv`: one row per eval set x run x forecaster x sample (`forecaster` is `oracle`, `pretrained`, `pretrained_truncate` or `finetuned`).
- `summary.csv`: one row per run x evaluated set. `pre_*` is the pretrained model of the same size, `ft_*` the fine-tuned one. `*_ratio` is the geometric mean over samples of the metric divided by the oracle's metric on the same sample (1 = as good as the oracle; MASE = mean absolute scaled error, sCRPS = scaled continuous ranked probability score). `delta_*` = ft minus pre. `gap_closed_*` = (pre - ft) / (pre - 1). `paired_mase_ft_over_pre` = geometric mean over samples of fine-tuned MASE / pretrained MASE. `cov80` = share of future points inside the 10%-90% quantile interval (0.8 is calibrated). `trunc_*` = the pretrained model with the context cut to the last max(d, 64) steps (regime sets only).
- `costs.csv`: per run: trainable parameters, steps, batch size, learning rate, time steps seen (context + future values over all training series), training wall time, final loss, peak memory. `trained_as` is filled when the run id reuses the model trained under another run id (knob protocol: all held-out families are tested on one trained model per size, method and replay share); its cost columns then repeat that model's.
- `provenance.json`: command line, commit, host, wall time, splits.
- `runs/<run_id>/run.json`, `runs/<run_id>/loss.csv`: per-run configuration, cost record and loss curve.

Evaluation: 32 held-out samples per set (500 oracle paths each, context 2048, horizon 128); regression set: 300 fresh in-prior series (seeds 100000 and up). Real-data regression: pending (`bench/regression/real.npz` was not on master when this ran; rerun with `--real-npz`).
Evaluation wall time 1.2 s; training wall time summed over trained models 2564.2 s (both from `provenance.json`).

## Leave-one-family-out (train on three families, test on the fourth)

### Size 5M: ratio to oracle, before and after

Cells: MASE ratio pre -> ft, sCRPS ratio pre -> ft, cov80 pre -> ft.

| held-out family | n | full+mix0.3 | full | lora+mix0.3 | lora |
|---|---|---|---|---|---|
| chaos | 32 | M 2.197 -> 2.203; S 2.238 -> 2.255; C 0.714 -> 0.689 | M 2.197 -> 2.201; S 2.238 -> 2.251; C 0.714 -> 0.694 | M 2.197 -> 2.214; S 2.238 -> 2.255; C 0.714 -> 0.701 | M 2.197 -> 2.213; S 2.238 -> 2.251; C 0.714 -> 0.713 |
| regime | 32 | M 1.171 -> 1.171; S 1.189 -> 1.191; C 0.769 -> 0.764 | M 1.171 -> 1.169; S 1.189 -> 1.187; C 0.769 -> 0.767 | M 1.171 -> 1.167; S 1.189 -> 1.188; C 0.769 -> 0.767 | M 1.171 -> 1.160; S 1.189 -> 1.178; C 0.769 -> 0.780 |
| tails | 32 | M 1.066 -> 1.059; S 1.060 -> 1.050; C 0.767 -> 0.755 | M 1.066 -> 1.060; S 1.060 -> 1.051; C 0.767 -> 0.754 | M 1.066 -> 1.058; S 1.060 -> 1.046; C 0.767 -> 0.739 | M 1.066 -> 1.063; S 1.060 -> 1.050; C 0.767 -> 0.742 |
| mixture | 32 | M 1.273 -> 1.283; S 1.310 -> 1.311; C 0.752 -> 0.750 | M 1.273 -> 1.284; S 1.310 -> 1.313; C 0.752 -> 0.745 | M 1.273 -> 1.309; S 1.310 -> 1.336; C 0.752 -> 0.742 | M 1.273 -> 1.305; S 1.310 -> 1.336; C 0.752 -> 0.736 |

### Size 5M: gain versus cost

| held-out family | method | paired MASE ft/pre | 95% interval | gap closed (MASE) | gap closed (sCRPS) | trainable params | steps | time steps seen | train wall s |
|---|---|---|---|---|---|---|---|---|---|
| chaos | full+mix0.3 | 1.003 | 0.982 to 1.027 | -0.005 | -0.014 | 5,208,960 | 180 | 2,164,992 | 100.8 |
| chaos | full | 1.002 | 0.984 to 1.021 | -0.004 | -0.011 | 5,208,960 | 180 | 2,164,992 | 113.3 |
| chaos | lora+mix0.3 | 1.008 | 0.959 to 1.067 | -0.014 | -0.014 | 215,040 | 180 | 2,164,992 | 74.3 |
| chaos | lora | 1.008 | 0.954 to 1.068 | -0.014 | -0.011 | 215,040 | 180 | 2,164,992 | 74.9 |
| regime | full+mix0.3 | 1.000 | 0.995 to 1.007 | -0.002 | -0.007 | 5,208,960 | 180 | 2,164,992 | 91.8 |
| regime | full | 0.999 | 0.994 to 1.004 | 0.007 | 0.013 | 5,208,960 | 180 | 2,164,992 | 87.6 |
| regime | lora+mix0.3 | 0.997 | 0.987 to 1.009 | 0.018 | 0.006 | 215,040 | 180 | 2,164,992 | 65.5 |
| regime | lora | 0.991 | 0.978 to 1.002 | 0.062 | 0.058 | 215,040 | 180 | 2,164,992 | 83.0 |
| tails | full+mix0.3 | 0.994 | 0.976 to 1.012 | 0.102 | 0.168 | 5,208,960 | 180 | 2,164,992 | 72.7 |
| tails | full | 0.994 | 0.975 to 1.010 | 0.092 | 0.138 | 5,208,960 | 180 | 2,164,992 | 66.5 |
| tails | lora+mix0.3 | 0.993 | 0.968 to 1.020 | 0.116 | 0.224 | 215,040 | 180 | 2,164,992 | 71.5 |
| tails | lora | 0.997 | 0.969 to 1.024 | 0.043 | 0.160 | 215,040 | 180 | 2,164,992 | 71.5 |
| mixture | full+mix0.3 | 1.008 | 0.991 to 1.031 | -0.038 | -0.004 | 5,208,960 | 180 | 2,164,992 | 95.9 |
| mixture | full | 1.008 | 0.990 to 1.034 | -0.039 | -0.011 | 5,208,960 | 180 | 2,164,992 | 89.2 |
| mixture | lora+mix0.3 | 1.028 | 0.992 to 1.082 | -0.132 | -0.085 | 215,040 | 180 | 2,164,992 | 70.6 |
| mixture | lora | 1.025 | 0.986 to 1.084 | -0.118 | -0.085 | 215,040 | 180 | 2,164,992 | 69.4 |
| regime | truncate (no training) |  |  | 0.114 | -0.338 | 0 | 0 | 0 | 0.0 |

## Held-out knob ranges (train on one range of each family's knob, test on the other)

### Size 5M: ratio to oracle, before and after

Cells: MASE ratio pre -> ft, sCRPS ratio pre -> ft, cov80 pre -> ft.

| held-out family | n | full+mix0.3 | full | lora+mix0.3 | lora |
|---|---|---|---|---|---|
| chaos | 32 | M 3.423 -> 3.413; S 3.576 -> 3.578; C 0.713 -> 0.665 | M 3.423 -> 3.564; S 3.576 -> 3.654; C 0.713 -> 0.674 | M 3.423 -> 3.577; S 3.576 -> 3.714; C 0.713 -> 0.678 | M 3.423 -> 3.569; S 3.576 -> 3.659; C 0.713 -> 0.684 |
| regime | 32 | M 1.729 -> 1.750; S 1.864 -> 1.911; C 0.652 -> 0.590 | M 1.729 -> 1.737; S 1.864 -> 1.885; C 0.652 -> 0.615 | M 1.729 -> 1.785; S 1.864 -> 1.965; C 0.652 -> 0.577 | M 1.729 -> 1.742; S 1.864 -> 1.893; C 0.652 -> 0.613 |
| tails | 32 | M 1.033 -> 1.059; S 1.046 -> 1.052; C 0.760 -> 0.710 | M 1.033 -> 1.056; S 1.046 -> 1.052; C 0.760 -> 0.718 | M 1.033 -> 1.142; S 1.046 -> 1.113; C 0.760 -> 0.688 | M 1.033 -> 1.094; S 1.046 -> 1.078; C 0.760 -> 0.698 |
| mixture | 32 | M 1.388 -> 1.393; S 1.410 -> 1.415; C 0.786 -> 0.751 | M 1.388 -> 1.390; S 1.410 -> 1.411; C 0.786 -> 0.760 | M 1.388 -> 1.393; S 1.410 -> 1.416; C 0.786 -> 0.759 | M 1.388 -> 1.383; S 1.410 -> 1.405; C 0.786 -> 0.775 |

### Size 5M: gain versus cost

| held-out family | method | paired MASE ft/pre | 95% interval | gap closed (MASE) | gap closed (sCRPS) | trainable params | steps | time steps seen | train wall s |
|---|---|---|---|---|---|---|---|---|---|
| chaos | full+mix0.3 | 0.997 | 0.950 to 1.052 | 0.004 | -0.001 | 5,208,960 | 180 | 2,164,992 | 75.3 |
| chaos | full | 1.041 | 0.997 to 1.102 | -0.058 | -0.030 | 5,208,960 | 180 | 2,164,992 | 92.5 |
| chaos | lora+mix0.3 | 1.045 | 0.988 to 1.134 | -0.063 | -0.054 | 215,040 | 180 | 2,164,992 | 67.0 |
| chaos | lora | 1.043 | 0.997 to 1.109 | -0.060 | -0.032 | 215,040 | 180 | 2,164,992 | 61.7 |
| regime | full+mix0.3 | 1.012 | 1.001 to 1.024 | -0.029 | -0.055 | 5,208,960 | 180 | 2,164,992 | 74.0 |
| regime | full | 1.005 | 0.997 to 1.013 | -0.011 | -0.025 | 5,208,960 | 180 | 2,164,992 | 70.7 |
| regime | lora+mix0.3 | 1.032 | 1.013 to 1.056 | -0.077 | -0.117 | 215,040 | 180 | 2,164,992 | 90.9 |
| regime | lora | 1.007 | 0.993 to 1.021 | -0.018 | -0.035 | 215,040 | 180 | 2,164,992 | 80.6 |
| tails | full+mix0.3 | 1.025 | 1.005 to 1.050 | -0.787 | -0.118 | 5,208,960 | 180 | 2,164,992 | 120.5 |
| tails | full | 1.023 | 1.009 to 1.041 | -0.707 | -0.129 | 5,208,960 | 180 | 2,164,992 | 125.8 |
| tails | lora+mix0.3 | 1.106 | 1.017 to 1.264 | -3.305 | -1.437 | 215,040 | 180 | 2,164,992 | 77.0 |
| tails | lora | 1.059 | 1.014 to 1.125 | -1.827 | -0.684 | 215,040 | 180 | 2,164,992 | 65.6 |
| mixture | full+mix0.3 | 1.004 | 0.995 to 1.013 | -0.013 | -0.012 | 5,208,960 | 180 | 2,164,992 | 70.0 |
| mixture | full | 1.002 | 0.993 to 1.011 | -0.007 | -0.002 | 5,208,960 | 180 | 2,164,992 | 69.7 |
| mixture | lora+mix0.3 | 1.004 | 0.992 to 1.018 | -0.015 | -0.014 | 215,040 | 180 | 2,164,992 | 61.8 |
| mixture | lora | 0.996 | 0.983 to 1.011 | 0.013 | 0.012 | 215,040 | 180 | 2,164,992 | 62.6 |
| regime | truncate (no training) |  |  | 0.058 | -0.143 | 0 | 0 | 0 | 0.0 |

## Regression check on normal series

Fine-tuned minus pretrained of the same size, on normal series that are outside the failure families. `paired MASE ft/pre` is the geometric mean over series of fine-tuned MASE / pretrained MASE (above 1: the fine-tuned model is worse). Run id column shows which checkpoint.

### regression/in_prior

| run | n | pre MASE ratio | ft MASE ratio | pre sCRPS ratio | ft sCRPS ratio | paired MASE ft/pre | 95% interval | paired sCRPS ft/pre | pre cov80 | ft cov80 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5M__knob-chaos__full__mix0.3 | 300 | 1.062 | 1.069 | 1.066 | 1.073 | 1.006 | 1.000 to 1.013 | 1.006 | 0.818 | 0.769 |
| 5M__knob-chaos__full__nomix | 300 | 1.062 | 1.083 | 1.066 | 1.085 | 1.020 | 1.012 to 1.028 | 1.018 | 0.818 | 0.781 |
| 5M__knob-chaos__lora__mix0.3 | 300 | 1.062 | 1.059 | 1.066 | 1.062 | 0.997 | 0.993 to 1.001 | 0.996 | 0.818 | 0.789 |
| 5M__knob-chaos__lora__nomix | 300 | 1.062 | 1.097 | 1.066 | 1.097 | 1.033 | 1.022 to 1.045 | 1.029 | 0.818 | 0.777 |
| 5M__knob-mixture__full__mix0.3 | 300 | 1.062 | 1.069 | 1.066 | 1.073 | 1.006 | 1.000 to 1.013 | 1.006 | 0.818 | 0.769 |
| 5M__knob-mixture__full__nomix | 300 | 1.062 | 1.083 | 1.066 | 1.085 | 1.020 | 1.012 to 1.028 | 1.018 | 0.818 | 0.781 |
| 5M__knob-mixture__lora__mix0.3 | 300 | 1.062 | 1.059 | 1.066 | 1.062 | 0.997 | 0.993 to 1.001 | 0.996 | 0.818 | 0.789 |
| 5M__knob-mixture__lora__nomix | 300 | 1.062 | 1.097 | 1.066 | 1.097 | 1.033 | 1.022 to 1.045 | 1.029 | 0.818 | 0.777 |
| 5M__knob-regime__full__mix0.3 | 300 | 1.062 | 1.069 | 1.066 | 1.073 | 1.006 | 1.000 to 1.013 | 1.006 | 0.818 | 0.769 |
| 5M__knob-regime__full__nomix | 300 | 1.062 | 1.083 | 1.066 | 1.085 | 1.020 | 1.012 to 1.028 | 1.018 | 0.818 | 0.781 |
| 5M__knob-regime__lora__mix0.3 | 300 | 1.062 | 1.059 | 1.066 | 1.062 | 0.997 | 0.993 to 1.001 | 0.996 | 0.818 | 0.789 |
| 5M__knob-regime__lora__nomix | 300 | 1.062 | 1.097 | 1.066 | 1.097 | 1.033 | 1.022 to 1.045 | 1.029 | 0.818 | 0.777 |
| 5M__knob-tails__full__mix0.3 | 300 | 1.062 | 1.069 | 1.066 | 1.073 | 1.006 | 1.000 to 1.013 | 1.006 | 0.818 | 0.769 |
| 5M__knob-tails__full__nomix | 300 | 1.062 | 1.083 | 1.066 | 1.085 | 1.020 | 1.012 to 1.028 | 1.018 | 0.818 | 0.781 |
| 5M__knob-tails__lora__mix0.3 | 300 | 1.062 | 1.059 | 1.066 | 1.062 | 0.997 | 0.993 to 1.001 | 0.996 | 0.818 | 0.789 |
| 5M__knob-tails__lora__nomix | 300 | 1.062 | 1.097 | 1.066 | 1.097 | 1.033 | 1.022 to 1.045 | 1.029 | 0.818 | 0.777 |
| 5M__lofo-chaos__full__mix0.3 | 300 | 1.062 | 1.076 | 1.066 | 1.080 | 1.013 | 1.007 to 1.020 | 1.013 | 0.818 | 0.760 |
| 5M__lofo-chaos__full__nomix | 300 | 1.062 | 1.125 | 1.066 | 1.130 | 1.059 | 1.045 to 1.074 | 1.060 | 0.818 | 0.737 |
| 5M__lofo-chaos__lora__mix0.3 | 300 | 1.062 | 1.082 | 1.066 | 1.085 | 1.019 | 1.011 to 1.029 | 1.018 | 0.818 | 0.769 |
| 5M__lofo-chaos__lora__nomix | 300 | 1.062 | 1.129 | 1.066 | 1.133 | 1.063 | 1.048 to 1.078 | 1.063 | 0.818 | 0.746 |
| 5M__lofo-mixture__full__mix0.3 | 300 | 1.062 | 1.077 | 1.066 | 1.081 | 1.014 | 1.008 to 1.022 | 1.014 | 0.818 | 0.762 |
| 5M__lofo-mixture__full__nomix | 300 | 1.062 | 1.104 | 1.066 | 1.106 | 1.040 | 1.029 to 1.052 | 1.037 | 0.818 | 0.762 |
| 5M__lofo-mixture__lora__mix0.3 | 300 | 1.062 | 1.089 | 1.066 | 1.091 | 1.025 | 1.016 to 1.036 | 1.023 | 0.818 | 0.776 |
| 5M__lofo-mixture__lora__nomix | 300 | 1.062 | 1.149 | 1.066 | 1.150 | 1.081 | 1.064 to 1.102 | 1.079 | 0.818 | 0.756 |
| 5M__lofo-regime__full__mix0.3 | 300 | 1.062 | 1.073 | 1.066 | 1.076 | 1.010 | 1.004 to 1.016 | 1.009 | 0.818 | 0.766 |
| 5M__lofo-regime__full__nomix | 300 | 1.062 | 1.111 | 1.066 | 1.114 | 1.046 | 1.034 to 1.060 | 1.045 | 0.818 | 0.743 |
| 5M__lofo-regime__lora__mix0.3 | 300 | 1.062 | 1.084 | 1.066 | 1.085 | 1.020 | 1.012 to 1.030 | 1.018 | 0.818 | 0.778 |
| 5M__lofo-regime__lora__nomix | 300 | 1.062 | 1.126 | 1.066 | 1.130 | 1.060 | 1.046 to 1.076 | 1.060 | 0.818 | 0.744 |
| 5M__lofo-tails__full__mix0.3 | 300 | 1.062 | 1.067 | 1.066 | 1.070 | 1.004 | 1.000 to 1.009 | 1.004 | 0.818 | 0.791 |
| 5M__lofo-tails__full__nomix | 300 | 1.062 | 1.081 | 1.066 | 1.083 | 1.018 | 1.012 to 1.025 | 1.016 | 0.818 | 0.809 |
| 5M__lofo-tails__lora__mix0.3 | 300 | 1.062 | 1.063 | 1.066 | 1.066 | 1.001 | 0.996 to 1.006 | 1.000 | 0.818 | 0.792 |
| 5M__lofo-tails__lora__nomix | 300 | 1.062 | 1.105 | 1.066 | 1.106 | 1.040 | 1.029 to 1.052 | 1.038 | 0.818 | 0.807 |

## Regime family: zero-training context truncation

`pretrained_truncate` keeps only the last max(d, 64) context steps when the scheduled switch is inside the context (no training). Same samples as the regime rows above.

| protocol | size | pre MASE ratio | truncate MASE ratio | pre sCRPS ratio | truncate sCRPS ratio | pre cov80 | truncate cov80 |
|---|---|---|---|---|---|---|---|
| knob | 5M | 1.729 | 1.687 | 1.864 | 1.987 | 0.652 | 0.916 |
| lofo | 5M | 1.171 | 1.151 | 1.189 | 1.253 | 0.769 | 0.853 |

Lead's normalization test (`results/mechanism/normalization/summary.csv`, T = 2048; other samples and seeds than the table above), rows copied:

| case | size | condition | n | mase_ratio_gmean | scrps_ratio_gmean | cov80 |
|---|---|---|---|---|---|---|
| mean_jump/delta4/d1024 | 5M | full | 24 | 1.043 | 1.052 | 0.849 |
| mean_jump/delta4/d1024 | 5M | post_only | 24 | 1.032 | 1.034 | 0.826 |
| mean_jump/delta4/d256 | 5M | full | 24 | 1.056 | 1.074 | 0.875 |
| mean_jump/delta4/d256 | 5M | post_only | 24 | 1.056 | 1.066 | 0.855 |
| mean_jump/delta4/d64 | 5M | full | 24 | 1.450 | 1.486 | 0.874 |
| mean_jump/delta4/d64 | 5M | post_only | 24 | 1.332 | 1.541 | 0.963 |
| stable_ar/alpha1.3/w256 | 5M | full | 24 | 1.044 | 1.082 | 0.885 |
| stable_ar/alpha1.3/w256 | 5M | post_only | 24 | 1.225 | 1.343 | 0.901 |
| stable_ar/alpha2/w256 | 5M | full | 24 | 1.001 | 0.999 | 0.830 |
| stable_ar/alpha2/w256 | 5M | post_only | 24 | 1.097 | 1.146 | 0.906 |
| variance_jump/f0.25/d1024 | 5M | full | 24 | 1.101 | 1.116 | 0.835 |
| variance_jump/f0.25/d1024 | 5M | post_only | 24 | 1.072 | 1.078 | 0.834 |
| variance_jump/f0.25/d256 | 5M | full | 24 | 1.117 | 1.172 | 0.918 |
| variance_jump/f0.25/d256 | 5M | post_only | 24 | 1.110 | 1.132 | 0.856 |
| variance_jump/f0.25/d64 | 5M | full | 24 | 1.236 | 1.734 | 1.000 |
| variance_jump/f0.25/d64 | 5M | post_only | 24 | 1.612 | 2.351 | 0.977 |
| variance_jump/f4/d1024 | 5M | full | 24 | 1.017 | 1.015 | 0.792 |
| variance_jump/f4/d1024 | 5M | post_only | 24 | 1.025 | 1.022 | 0.799 |
| variance_jump/f4/d256 | 5M | full | 24 | 1.015 | 1.017 | 0.755 |
| variance_jump/f4/d256 | 5M | post_only | 24 | 1.042 | 1.042 | 0.830 |
| variance_jump/f4/d64 | 5M | full | 24 | 1.018 | 1.052 | 0.663 |
| variance_jump/f4/d64 | 5M | post_only | 24 | 1.154 | 1.260 | 0.925 |

## Budgets used

From `costs.csv`.

| size | method | runs | steps | batch size | learning rate | trainable params |
|---|---|---|---|---|---|---|
| 5M | full | 16 | 180 | 8 | 5e-05 | 5,208,960 |
| 5M | lora | 16 | 180 | 8 | 0.001 | 215,040 |

## Notes

- Local pass: 5M model only, run on CPU (1 thread). Each run took 180 optimizer steps at batch size 8 over a pool of 60 simulated batches, so every pool batch was used 3 times (the pool is cycled); simulating the training series dominated the cost, so the pool size was kept small.
- Methods: full, lora (rank 8, alpha 16) and each with replay share 0.3; learning rates 5e-5 (full) and 1e-3 (lora), picked from the 60-step training-loss pilot in results/finetune/local/lr_pilot/ (loss on training series only; the held-out sets were not used). They were not tuned further.
- Training contexts 512, 1024, 2048; horizons 32, 128, 512. Held-out samples: context 2048, horizon 128.
- Real-data regression check: pending. bench/regression/real.npz and otp/realdata.py were not on master when this ran. The in-prior regression set here is 300 fresh series (seeds 100000 to 100299), not the 500-series bench/regression/in_prior.npz.
- Knob protocol: the four runs with the same method and replay share (one per held-out family) trained on identical data in identical order, so their final_loss values in costs.csv are equal and so are their regression rows. They were trained separately in this local pass (scripts/finetune_job.py, used for the Colab run, trains that model once).
- The learning-rate pilot, the number of steps and the choice to cycle the pool were fixed before any held-out set was scored.

