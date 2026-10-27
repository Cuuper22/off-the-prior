# Model card draft: Timestral-123M fine-tuned (lora+mix0.5), lofo / mixture

Run id: `123M__lofo-mixture__lora__mix0.5`

## Model

- Base model: `mistralai/Timestral-123M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `mixture`.
- Failure families used for training: chaos, regime, tails. Held out for testing: mixture (random mixtures of the primitives and the other families).
- 0.5 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 1,794,560 (base model: 122,606,720)
- Steps: 300; batch size: 16; learning rate: 0.0005
- Time steps seen: 5,898,240
- Training wall time: 40.9 s; peak resident memory: 2,480,537,600 bytes; final loss: 1.2285
- Peak GPU memory allocated: 2,156,770,816 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/mixture`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.266 | 1.243 |
| sCRPS ratio to oracle | 1.278 | 1.263 |
| cov80 | 0.764 | 0.764 |

Paired MASE, fine-tuned / pretrained: 0.982 (95% interval 0.956 to 1.003).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.032 | 1.026 |
| sCRPS ratio to oracle | 1.031 | 1.026 |
| cov80 | 0.803 | 0.806 |

Paired MASE, fine-tuned / pretrained: 0.995.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 0.991 (95% interval 0.969 to 1.010); cov80 0.774 pretrained, 0.757 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

