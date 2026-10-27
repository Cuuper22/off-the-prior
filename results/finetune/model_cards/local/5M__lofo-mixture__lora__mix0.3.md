# Model card draft: Timestral-5M fine-tuned (lora+mix0.3), lofo / mixture

Run id: `5M__lofo-mixture__lora__mix0.3`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `mixture`.
- Failure families used for training: chaos, regime, tails. Held out for testing: mixture (random mixtures of the primitives and the other families).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 215,040 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 0.001
- Time steps seen: 2,164,992
- Training wall time: 70.6 s; peak resident memory: 606,527,488 bytes; final loss: 1.5685

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `lofo/mixture`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.273 | 1.309 |
| sCRPS ratio to oracle | 1.310 | 1.336 |
| cov80 | 0.752 | 0.742 |

Paired MASE, fine-tuned / pretrained: 1.028 (95% interval 0.992 to 1.082).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.089 |
| sCRPS ratio to oracle | 1.073 | 1.092 |
| cov80 | 0.810 | 0.777 |

Paired MASE, fine-tuned / pretrained: 1.019.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

