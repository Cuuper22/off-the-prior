# Model card draft: Timestral-123M fine-tuned (full), lofo / mixture

Run id: `123M__lofo-mixture__full__nomix`

## Model

- Base model: `mistralai/Timestral-123M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `mixture`.
- Failure families used for training: chaos, regime, tails. Held out for testing: mixture (random mixtures of the primitives and the other families).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 122,606,720 (base model: 122,606,720)
- Steps: 300; batch size: 16; learning rate: 2e-05
- Time steps seen: 5,898,240
- Training wall time: 35.0 s; peak resident memory: 2,464,251,904 bytes; final loss: 1.7265
- Peak GPU memory allocated: 3,844,695,552 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/mixture`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.266 | 1.267 |
| sCRPS ratio to oracle | 1.278 | 1.279 |
| cov80 | 0.764 | 0.761 |

Paired MASE, fine-tuned / pretrained: 1.001 (95% interval 0.992 to 1.009).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.032 | 1.037 |
| sCRPS ratio to oracle | 1.031 | 1.037 |
| cov80 | 0.803 | 0.782 |

Paired MASE, fine-tuned / pretrained: 1.005.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 0.994 (95% interval 0.988 to 0.998); cov80 0.774 pretrained, 0.759 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

