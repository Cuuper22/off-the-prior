# Model card draft: Timestral-123M fine-tuned (full), knob / tails

Run id: `123M__knob-tails__full__nomix`

## Model

- Base model: `mistralai/Timestral-123M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `tails`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the tails family (ranges in `otp/finetune_splits.py`).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 122,606,720 (base model: 122,606,720)
- Steps: 300; batch size: 16; learning rate: 2e-05
- Time steps seen: 5,898,240
- Training wall time: 35.9 s; peak resident memory: 2,461,941,760 bytes; final loss: 2.2109
- Peak GPU memory allocated: 3,844,695,552 bytes on NVIDIA A100-SXM4-40GB
- This is the same trained model as run `123M__knob-chaos__full__nomix` (knob protocol: one model per size, method and replay share is tested on every held-out family); the cost figures above are that model's.

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `knob/tails`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.015 | 1.024 |
| sCRPS ratio to oracle | 1.025 | 1.014 |
| cov80 | 0.767 | 0.738 |

Paired MASE, fine-tuned / pretrained: 1.010 (95% interval 0.991 to 1.039).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.032 | 1.048 |
| sCRPS ratio to oracle | 1.031 | 1.048 |
| cov80 | 0.803 | 0.782 |

Paired MASE, fine-tuned / pretrained: 1.016.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.005 (95% interval 1.000 to 1.010); cov80 0.774 pretrained, 0.746 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

