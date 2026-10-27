# Model card draft: Timestral-28M fine-tuned (full+mix0.5), lofo / chaos

Run id: `28M__lofo-chaos__full__mix0.5`

## Model

- Base model: `mistralai/Timestral-28M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `chaos`.
- Failure families used for training: regime, tails, mixture. Held out for testing: chaos (chaotic and dynamical-system series (Lorenz63, Rossler, Mackey-Glass, logistic map, Kuramoto)).
- 0.5 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 27,821,184 (base model: 27,821,184)
- Steps: 500; batch size: 32; learning rate: 3e-05
- Time steps seen: 19,478,528
- Training wall time: 39.1 s; peak resident memory: 2,077,495,296 bytes; final loss: 1.3167
- Peak GPU memory allocated: 1,814,089,216 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/chaos`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 2.031 | 2.025 |
| sCRPS ratio to oracle | 2.056 | 2.041 |
| cov80 | 0.762 | 0.770 |

Paired MASE, fine-tuned / pretrained: 0.997 (95% interval 0.978 to 1.021).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.043 | 1.038 |
| sCRPS ratio to oracle | 1.045 | 1.038 |
| cov80 | 0.803 | 0.804 |

Paired MASE, fine-tuned / pretrained: 0.995.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.000 (95% interval 0.989 to 1.013); cov80 0.756 pretrained, 0.757 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

