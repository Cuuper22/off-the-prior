# Model card draft: Timestral-5M fine-tuned (full), knob / tails

Run id: `5M__knob-tails__full__nomix`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `tails`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the tails family (ranges in `otp/finetune_splits.py`).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 5e-05
- Time steps seen: 2,164,992
- Training wall time: 125.8 s; peak resident memory: 675,803,136 bytes; final loss: 2.4733

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `knob/tails`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.033 | 1.056 |
| sCRPS ratio to oracle | 1.046 | 1.052 |
| cov80 | 0.760 | 0.718 |

Paired MASE, fine-tuned / pretrained: 1.023 (95% interval 1.009 to 1.041).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 300 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.062 | 1.083 |
| sCRPS ratio to oracle | 1.066 | 1.085 |
| cov80 | 0.818 | 0.781 |

Paired MASE, fine-tuned / pretrained: 1.020.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

