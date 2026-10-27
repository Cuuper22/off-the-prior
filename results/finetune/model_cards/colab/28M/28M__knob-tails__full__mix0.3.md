# Model card draft: Timestral-28M fine-tuned (full+mix0.3), knob / tails

Run id: `28M__knob-tails__full__mix0.3`

## Model

- Base model: `mistralai/Timestral-28M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `tails`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the tails family (ranges in `otp/finetune_splits.py`).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 27,821,184 (base model: 27,821,184)
- Steps: 500; batch size: 32; learning rate: 3e-05
- Time steps seen: 19,478,528
- Training wall time: 37.6 s; peak resident memory: 2,075,152,384 bytes; final loss: 1.1431
- Peak GPU memory allocated: 1,814,089,216 bytes on NVIDIA A100-SXM4-40GB
- This is the same trained model as run `28M__knob-chaos__full__mix0.3` (knob protocol: one model per size, method and replay share is tested on every held-out family); the cost figures above are that model's.

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `knob/tails`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.019 | 1.014 |
| sCRPS ratio to oracle | 1.033 | 1.007 |
| cov80 | 0.759 | 0.729 |

Paired MASE, fine-tuned / pretrained: 0.995 (95% interval 0.980 to 1.007).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.043 | 1.036 |
| sCRPS ratio to oracle | 1.045 | 1.036 |
| cov80 | 0.803 | 0.785 |

Paired MASE, fine-tuned / pretrained: 0.993.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.003 (95% interval 0.994 to 1.013); cov80 0.756 pretrained, 0.743 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

