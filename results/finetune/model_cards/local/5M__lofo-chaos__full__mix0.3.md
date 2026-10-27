# Model card draft: Timestral-5M fine-tuned (full+mix0.3), lofo / chaos

Run id: `5M__lofo-chaos__full__mix0.3`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `chaos`.
- Failure families used for training: regime, tails, mixture. Held out for testing: chaos (chaotic and dynamical-system series (Lorenz63, Rossler, Mackey-Glass, logistic map, Kuramoto)).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 5e-05
- Time steps seen: 2,164,992
- Training wall time: 100.8 s; peak resident memory: 676,950,016 bytes; final loss: 16.2230

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `lofo/chaos`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 2.197 | 2.203 |
| sCRPS ratio to oracle | 2.238 | 2.255 |
| cov80 | 0.714 | 0.689 |

Paired MASE, fine-tuned / pretrained: 1.003 (95% interval 0.982 to 1.027).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 300 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.062 | 1.076 |
| sCRPS ratio to oracle | 1.066 | 1.080 |
| cov80 | 0.818 | 0.760 |

Paired MASE, fine-tuned / pretrained: 1.013.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

