# Model card draft: Timestral-5M fine-tuned (full+mix0.3), lofo / tails

Run id: `5M__lofo-tails__full__mix0.3`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `tails`.
- Failure families used for training: chaos, regime, mixture. Held out for testing: tails (heavy-tailed and fractal series (alpha-stable AR, fractional Brownian motion, Hawkes counts, multifractal cascade)).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 5e-05
- Time steps seen: 2,164,992
- Training wall time: 72.7 s; peak resident memory: 616,620,032 bytes; final loss: 15.4650

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `lofo/tails`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.066 | 1.059 |
| sCRPS ratio to oracle | 1.060 | 1.050 |
| cov80 | 0.767 | 0.755 |

Paired MASE, fine-tuned / pretrained: 0.994 (95% interval 0.976 to 1.012).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.070 |
| sCRPS ratio to oracle | 1.073 | 1.074 |
| cov80 | 0.810 | 0.789 |

Paired MASE, fine-tuned / pretrained: 1.001.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/local/summary.csv`): paired MASE, fine-tuned / pretrained: 1.000 (95% interval 0.991 to 1.009); cov80 0.784 pretrained, 0.780 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

