# Model card draft: Timestral-5M fine-tuned (full), lofo / tails

Run id: `5M__lofo-tails__full__nomix`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `tails`.
- Failure families used for training: chaos, regime, mixture. Held out for testing: tails (heavy-tailed and fractal series (alpha-stable AR, fractional Brownian motion, Hawkes counts, multifractal cascade)).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 500; batch size: 32; learning rate: 5e-05
- Time steps seen: 19,478,528
- Training wall time: 28.9 s; peak resident memory: 1,900,908,544 bytes; final loss: 1.1461
- Peak GPU memory allocated: 578,694,656 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/tails`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.066 | 1.053 |
| sCRPS ratio to oracle | 1.060 | 1.038 |
| cov80 | 0.767 | 0.762 |

Paired MASE, fine-tuned / pretrained: 0.988 (95% interval 0.974 to 1.003).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.100 |
| sCRPS ratio to oracle | 1.073 | 1.104 |
| cov80 | 0.810 | 0.817 |

Paired MASE, fine-tuned / pretrained: 1.028.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.009 (95% interval 0.995 to 1.025); cov80 0.784 pretrained, 0.800 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

