# Model card draft: Timestral-5M fine-tuned (lora), lofo / chaos

Run id: `5M__lofo-chaos__lora__nomix`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `chaos`.
- Failure families used for training: regime, tails, mixture. Held out for testing: chaos (chaotic and dynamical-system series (Lorenz63, Rossler, Mackey-Glass, logistic map, Kuramoto)).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 215,040 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 0.001
- Time steps seen: 2,164,992
- Training wall time: 74.9 s; peak resident memory: 609,628,160 bytes; final loss: 16.5772

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `lofo/chaos`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 2.197 | 2.213 |
| sCRPS ratio to oracle | 2.238 | 2.251 |
| cov80 | 0.714 | 0.713 |

Paired MASE, fine-tuned / pretrained: 1.008 (95% interval 0.954 to 1.068).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.128 |
| sCRPS ratio to oracle | 1.073 | 1.132 |
| cov80 | 0.810 | 0.744 |

Paired MASE, fine-tuned / pretrained: 1.055.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/local/summary.csv`): paired MASE, fine-tuned / pretrained: 1.001 (95% interval 0.986 to 1.017); cov80 0.784 pretrained, 0.753 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

