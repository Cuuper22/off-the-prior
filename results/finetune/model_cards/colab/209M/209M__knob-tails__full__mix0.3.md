# Model card draft: Timestral-209M fine-tuned (full+mix0.3), knob / tails

Run id: `209M__knob-tails__full__mix0.3`

## Model

- Base model: `mistralai/Timestral-209M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `tails`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the tails family (ranges in `otp/finetune_splits.py`).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 209,312,256 (base model: 209,312,256)
- Steps: 300; batch size: 16; learning rate: 1e-05
- Time steps seen: 5,898,240
- Training wall time: 43.2 s; peak resident memory: 2,776,154,112 bytes; final loss: 1.8335
- Peak GPU memory allocated: 5,989,050,880 bytes on NVIDIA A100-SXM4-40GB
- This is the same trained model as run `209M__knob-chaos__full__mix0.3` (knob protocol: one model per size, method and replay share is tested on every held-out family); the cost figures above are that model's.

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `knob/tails`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.011 | 1.012 |
| sCRPS ratio to oracle | 1.027 | 1.015 |
| cov80 | 0.768 | 0.739 |

Paired MASE, fine-tuned / pretrained: 1.001 (95% interval 0.995 to 1.007).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.029 | 1.027 |
| sCRPS ratio to oracle | 1.029 | 1.027 |
| cov80 | 0.808 | 0.782 |

Paired MASE, fine-tuned / pretrained: 0.998.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.005 (95% interval 1.001 to 1.009); cov80 0.770 pretrained, 0.739 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

