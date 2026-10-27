# Model card draft: Timestral-5M fine-tuned (full), knob / regime

Run id: `5M__knob-regime__full__nomix`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `regime`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the regime family (ranges in `otp/finetune_splits.py`).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 5e-05
- Time steps seen: 2,164,992
- Training wall time: 70.7 s; peak resident memory: 668,774,400 bytes; final loss: 2.4733

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `knob/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.729 | 1.737 |
| sCRPS ratio to oracle | 1.864 | 1.885 |
| cov80 | 0.652 | 0.615 |

Paired MASE, fine-tuned / pretrained: 1.005 (95% interval 0.997 to 1.013).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.088 |
| sCRPS ratio to oracle | 1.073 | 1.090 |
| cov80 | 0.810 | 0.773 |

Paired MASE, fine-tuned / pretrained: 1.017.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/local/summary.csv`): paired MASE, fine-tuned / pretrained: 1.012 (95% interval 1.003 to 1.025); cov80 0.784 pretrained, 0.754 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

