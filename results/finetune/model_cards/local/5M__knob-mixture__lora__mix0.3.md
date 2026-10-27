# Model card draft: Timestral-5M fine-tuned (lora+mix0.3), knob / mixture

Run id: `5M__knob-mixture__lora__mix0.3`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `mixture`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the mixture family (ranges in `otp/finetune_splits.py`).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 215,040 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 0.001
- Time steps seen: 2,164,992
- Training wall time: 61.8 s; peak resident memory: 598,454,272 bytes; final loss: 1.0977

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `knob/mixture`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.388 | 1.393 |
| sCRPS ratio to oracle | 1.410 | 1.416 |
| cov80 | 0.786 | 0.759 |

Paired MASE, fine-tuned / pretrained: 1.004 (95% interval 0.992 to 1.018).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.062 |
| sCRPS ratio to oracle | 1.073 | 1.066 |
| cov80 | 0.810 | 0.783 |

Paired MASE, fine-tuned / pretrained: 0.993.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

