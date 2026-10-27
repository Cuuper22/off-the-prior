# Model card draft: Timestral-209M fine-tuned (lora+mix0.3), knob / chaos

Run id: `209M__knob-chaos__lora__mix0.3`

## Model

- Base model: `mistralai/Timestral-209M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `chaos`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the chaos family (ranges in `otp/finetune_splits.py`).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 2,580,480 (base model: 209,312,256)
- Steps: 300; batch size: 16; learning rate: 0.0005
- Time steps seen: 5,898,240
- Training wall time: 47.6 s; peak resident memory: 2,792,472,576 bytes; final loss: 1.7981
- Peak GPU memory allocated: 3,203,901,440 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `knob/chaos`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 2.668 | 2.600 |
| sCRPS ratio to oracle | 2.681 | 2.614 |
| cov80 | 0.770 | 0.780 |

Paired MASE, fine-tuned / pretrained: 0.974 (95% interval 0.948 to 1.002).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.029 | 1.026 |
| sCRPS ratio to oracle | 1.029 | 1.026 |
| cov80 | 0.808 | 0.801 |

Paired MASE, fine-tuned / pretrained: 0.996.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.004 (95% interval 0.995 to 1.013); cov80 0.770 pretrained, 0.754 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

