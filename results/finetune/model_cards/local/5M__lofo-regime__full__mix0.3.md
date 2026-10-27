# Model card draft: Timestral-5M fine-tuned (full+mix0.3), lofo / regime

Run id: `5M__lofo-regime__full__mix0.3`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `regime`.
- Failure families used for training: chaos, tails, mixture. Held out for testing: regime (regime switches (mean, variance and frequency jumps; Markov-switching AR)).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 5e-05
- Time steps seen: 2,164,992
- Training wall time: 91.8 s; peak resident memory: 677,322,752 bytes; final loss: 16.3187

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `lofo/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.171 | 1.171 |
| sCRPS ratio to oracle | 1.189 | 1.191 |
| cov80 | 0.769 | 0.764 |

Paired MASE, fine-tuned / pretrained: 1.000 (95% interval 0.995 to 1.007).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 300 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.062 | 1.073 |
| sCRPS ratio to oracle | 1.066 | 1.076 |
| cov80 | 0.818 | 0.766 |

Paired MASE, fine-tuned / pretrained: 1.010.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

