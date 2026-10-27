# Model card draft: Timestral-5M fine-tuned (full+mix0.3), knob / regime

Run id: `5M__knob-regime__full__mix0.3`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `regime`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the regime family (ranges in `otp/finetune_splits.py`).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 5e-05
- Time steps seen: 2,164,992
- Training wall time: 74.0 s; peak resident memory: 675,282,944 bytes; final loss: 1.2162

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `knob/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.729 | 1.750 |
| sCRPS ratio to oracle | 1.864 | 1.911 |
| cov80 | 0.652 | 0.590 |

Paired MASE, fine-tuned / pretrained: 1.012 (95% interval 1.001 to 1.024).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.069 |
| sCRPS ratio to oracle | 1.073 | 1.074 |
| cov80 | 0.810 | 0.765 |

Paired MASE, fine-tuned / pretrained: 1.000.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/local/summary.csv`): paired MASE, fine-tuned / pretrained: 1.009 (95% interval 1.000 to 1.019); cov80 0.784 pretrained, 0.745 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

