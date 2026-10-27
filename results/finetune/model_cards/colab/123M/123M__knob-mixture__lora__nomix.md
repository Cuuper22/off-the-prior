# Model card draft: Timestral-123M fine-tuned (lora), knob / mixture

Run id: `123M__knob-mixture__lora__nomix`

## Model

- Base model: `mistralai/Timestral-123M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `mixture`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the mixture family (ranges in `otp/finetune_splits.py`).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 1,794,560 (base model: 122,606,720)
- Steps: 300; batch size: 16; learning rate: 0.0005
- Time steps seen: 5,898,240
- Training wall time: 41.1 s; peak resident memory: 2,478,239,744 bytes; final loss: 2.1796
- Peak GPU memory allocated: 2,156,770,816 bytes on NVIDIA A100-SXM4-40GB
- This is the same trained model as run `123M__knob-chaos__lora__nomix` (knob protocol: one model per size, method and replay share is tested on every held-out family); the cost figures above are that model's.

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `knob/mixture`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.306 | 1.298 |
| sCRPS ratio to oracle | 1.330 | 1.323 |
| cov80 | 0.788 | 0.781 |

Paired MASE, fine-tuned / pretrained: 0.994 (95% interval 0.986 to 1.001).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.032 | 1.034 |
| sCRPS ratio to oracle | 1.031 | 1.034 |
| cov80 | 0.803 | 0.803 |

Paired MASE, fine-tuned / pretrained: 1.003.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.004 (95% interval 0.996 to 1.013); cov80 0.774 pretrained, 0.764 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

