# Model card draft: Timestral-5M fine-tuned (full+mix0.3), lofo / mixture

Run id: `5M__lofo-mixture__full__mix0.3`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `mixture`.
- Failure families used for training: chaos, regime, tails. Held out for testing: mixture (random mixtures of the primitives and the other families).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 5e-05
- Time steps seen: 2,164,992
- Training wall time: 95.9 s; peak resident memory: 680,161,280 bytes; final loss: 1.5871

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `lofo/mixture`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.273 | 1.283 |
| sCRPS ratio to oracle | 1.310 | 1.311 |
| cov80 | 0.752 | 0.750 |

Paired MASE, fine-tuned / pretrained: 1.008 (95% interval 0.991 to 1.031).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 300 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.062 | 1.077 |
| sCRPS ratio to oracle | 1.066 | 1.081 |
| cov80 | 0.818 | 0.762 |

Paired MASE, fine-tuned / pretrained: 1.014.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

