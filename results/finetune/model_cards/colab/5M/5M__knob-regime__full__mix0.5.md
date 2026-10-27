# Model card draft: Timestral-5M fine-tuned (full+mix0.5), knob / regime

Run id: `5M__knob-regime__full__mix0.5`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `regime`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the regime family (ranges in `otp/finetune_splits.py`).
- 0.5 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 500; batch size: 32; learning rate: 5e-05
- Time steps seen: 19,478,528
- Training wall time: 28.2 s; peak resident memory: 1,958,141,952 bytes; final loss: 1.0713
- Peak GPU memory allocated: 578,694,656 bytes on NVIDIA A100-SXM4-40GB
- This is the same trained model as run `5M__knob-chaos__full__mix0.5` (knob protocol: one model per size, method and replay share is tested on every held-out family); the cost figures above are that model's.

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `knob/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.729 | 1.729 |
| sCRPS ratio to oracle | 1.864 | 1.900 |
| cov80 | 0.652 | 0.611 |

Paired MASE, fine-tuned / pretrained: 1.000 (95% interval 0.969 to 1.036).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.055 |
| sCRPS ratio to oracle | 1.073 | 1.058 |
| cov80 | 0.810 | 0.799 |

Paired MASE, fine-tuned / pretrained: 0.986.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.006 (95% interval 0.994 to 1.020); cov80 0.784 pretrained, 0.772 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

