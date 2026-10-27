# Model card draft: Timestral-123M fine-tuned (full+mix0.5), knob / regime

Run id: `123M__knob-regime__full__mix0.5`

## Model

- Base model: `mistralai/Timestral-123M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `regime`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the regime family (ranges in `otp/finetune_splits.py`).
- 0.5 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 122,606,720 (base model: 122,606,720)
- Steps: 300; batch size: 16; learning rate: 2e-05
- Time steps seen: 5,898,240
- Training wall time: 35.1 s; peak resident memory: 2,461,216,768 bytes; final loss: 1.4542
- Peak GPU memory allocated: 3,844,695,552 bytes on NVIDIA A100-SXM4-40GB
- This is the same trained model as run `123M__knob-chaos__full__mix0.5` (knob protocol: one model per size, method and replay share is tested on every held-out family); the cost figures above are that model's.

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `knob/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.445 | 1.441 |
| sCRPS ratio to oracle | 1.643 | 1.620 |
| cov80 | 0.697 | 0.681 |

Paired MASE, fine-tuned / pretrained: 0.997 (95% interval 0.981 to 1.012).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.032 | 1.029 |
| sCRPS ratio to oracle | 1.031 | 1.028 |
| cov80 | 0.803 | 0.786 |

Paired MASE, fine-tuned / pretrained: 0.997.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.002 (95% interval 0.995 to 1.009); cov80 0.774 pretrained, 0.741 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

