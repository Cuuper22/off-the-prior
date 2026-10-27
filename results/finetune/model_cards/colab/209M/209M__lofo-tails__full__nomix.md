# Model card draft: Timestral-209M fine-tuned (full), lofo / tails

Run id: `209M__lofo-tails__full__nomix`

## Model

- Base model: `mistralai/Timestral-209M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `tails`.
- Failure families used for training: chaos, regime, mixture. Held out for testing: tails (heavy-tailed and fractal series (alpha-stable AR, fractional Brownian motion, Hawkes counts, multifractal cascade)).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 209,312,256 (base model: 209,312,256)
- Steps: 300; batch size: 16; learning rate: 1e-05
- Time steps seen: 5,898,240
- Training wall time: 43.3 s; peak resident memory: 2,718,830,592 bytes; final loss: 1.0836
- Peak GPU memory allocated: 5,989,050,880 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/tails`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.062 | 1.060 |
| sCRPS ratio to oracle | 1.062 | 1.052 |
| cov80 | 0.757 | 0.748 |

Paired MASE, fine-tuned / pretrained: 0.998 (95% interval 0.988 to 1.008).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.029 | 1.030 |
| sCRPS ratio to oracle | 1.029 | 1.030 |
| cov80 | 0.808 | 0.805 |

Paired MASE, fine-tuned / pretrained: 1.001.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 0.998 (95% interval 0.994 to 1.002); cov80 0.770 pretrained, 0.759 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

