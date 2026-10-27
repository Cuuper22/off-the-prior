# Model card draft: Timestral-123M fine-tuned (full), lofo / chaos

Run id: `123M__lofo-chaos__full__nomix`

## Model

- Base model: `mistralai/Timestral-123M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `chaos`.
- Failure families used for training: regime, tails, mixture. Held out for testing: chaos (chaotic and dynamical-system series (Lorenz63, Rossler, Mackey-Glass, logistic map, Kuramoto)).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 122,606,720 (base model: 122,606,720)
- Steps: 300; batch size: 16; learning rate: 2e-05
- Time steps seen: 5,898,240
- Training wall time: 35.3 s; peak resident memory: 2,596,130,816 bytes; final loss: 1.6830
- Peak GPU memory allocated: 3,844,695,552 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/chaos`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.807 | 1.810 |
| sCRPS ratio to oracle | 1.829 | 1.839 |
| cov80 | 0.794 | 0.774 |

Paired MASE, fine-tuned / pretrained: 1.002 (95% interval 0.981 to 1.024).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.032 | 1.040 |
| sCRPS ratio to oracle | 1.031 | 1.040 |
| cov80 | 0.803 | 0.779 |

Paired MASE, fine-tuned / pretrained: 1.008.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 0.995 (95% interval 0.990 to 0.999); cov80 0.774 pretrained, 0.757 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

