# Model card draft: Timestral-5M fine-tuned (lora+mix0.5), knob / tails

Run id: `5M__knob-tails__lora__mix0.5`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `tails`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the tails family (ranges in `otp/finetune_splits.py`).
- 0.5 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 215,040 (base model: 5,208,960)
- Steps: 500; batch size: 32; learning rate: 0.001
- Time steps seen: 19,478,528
- Training wall time: 33.0 s; peak resident memory: 1,978,908,672 bytes; final loss: 1.0335
- Peak GPU memory allocated: 461,856,256 bytes on NVIDIA A100-SXM4-40GB
- This is the same trained model as run `5M__knob-chaos__lora__mix0.5` (knob protocol: one model per size, method and replay share is tested on every held-out family); the cost figures above are that model's.

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `knob/tails`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.033 | 1.020 |
| sCRPS ratio to oracle | 1.046 | 1.014 |
| cov80 | 0.760 | 0.757 |

Paired MASE, fine-tuned / pretrained: 0.987 (95% interval 0.972 to 1.002).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.052 |
| sCRPS ratio to oracle | 1.073 | 1.056 |
| cov80 | 0.810 | 0.816 |

Paired MASE, fine-tuned / pretrained: 0.984.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.012 (95% interval 0.992 to 1.035); cov80 0.784 pretrained, 0.783 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

