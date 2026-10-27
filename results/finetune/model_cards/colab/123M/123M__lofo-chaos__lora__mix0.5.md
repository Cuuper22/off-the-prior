# Model card draft: Timestral-123M fine-tuned (lora+mix0.5), lofo / chaos

Run id: `123M__lofo-chaos__lora__mix0.5`

## Model

- Base model: `mistralai/Timestral-123M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `chaos`.
- Failure families used for training: regime, tails, mixture. Held out for testing: chaos (chaotic and dynamical-system series (Lorenz63, Rossler, Mackey-Glass, logistic map, Kuramoto)).
- 0.5 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 1,794,560 (base model: 122,606,720)
- Steps: 300; batch size: 16; learning rate: 0.0005
- Time steps seen: 5,898,240
- Training wall time: 41.1 s; peak resident memory: 2,480,545,792 bytes; final loss: 1.4492
- Peak GPU memory allocated: 2,156,770,816 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/chaos`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.807 | 1.824 |
| sCRPS ratio to oracle | 1.829 | 1.840 |
| cov80 | 0.794 | 0.781 |

Paired MASE, fine-tuned / pretrained: 1.010 (95% interval 0.992 to 1.029).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.032 | 1.026 |
| sCRPS ratio to oracle | 1.031 | 1.026 |
| cov80 | 0.803 | 0.802 |

Paired MASE, fine-tuned / pretrained: 0.995.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 0.997 (95% interval 0.980 to 1.014); cov80 0.774 pretrained, 0.753 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

