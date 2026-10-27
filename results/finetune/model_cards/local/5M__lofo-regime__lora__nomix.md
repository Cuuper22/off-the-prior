# Model card draft: Timestral-5M fine-tuned (lora), lofo / regime

Run id: `5M__lofo-regime__lora__nomix`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `regime`.
- Failure families used for training: chaos, tails, mixture. Held out for testing: regime (regime switches (mean, variance and frequency jumps; Markov-switching AR)).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 215,040 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 0.001
- Time steps seen: 2,164,992
- Training wall time: 83.0 s; peak resident memory: 606,687,232 bytes; final loss: 16.7211

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `lofo/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.171 | 1.160 |
| sCRPS ratio to oracle | 1.189 | 1.178 |
| cov80 | 0.769 | 0.780 |

Paired MASE, fine-tuned / pretrained: 0.991 (95% interval 0.978 to 1.002).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.126 |
| sCRPS ratio to oracle | 1.073 | 1.129 |
| cov80 | 0.810 | 0.744 |

Paired MASE, fine-tuned / pretrained: 1.053.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

