# Model card draft: Timestral-5M fine-tuned (lora), knob / regime

Run id: `5M__knob-regime__lora__nomix`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `regime`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the regime family (ranges in `otp/finetune_splits.py`).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/local/costs.csv`.

- Trainable parameters: 215,040 (base model: 5,208,960)
- Steps: 180; batch size: 8; learning rate: 0.001
- Time steps seen: 2,164,992
- Training wall time: 80.6 s; peak resident memory: 600,858,624 bytes; final loss: 2.3652

## Held-out evaluation

Copied from `results/finetune/local/summary.csv` (set `knob/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.729 | 1.742 |
| sCRPS ratio to oracle | 1.864 | 1.893 |
| cov80 | 0.652 | 0.613 |

Paired MASE, fine-tuned / pretrained: 1.007 (95% interval 0.993 to 1.021).

## Regression check (normal series)

Copied from `results/finetune/local/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.097 |
| sCRPS ratio to oracle | 1.073 | 1.098 |
| cov80 | 0.810 | 0.774 |

Paired MASE, fine-tuned / pretrained: 1.025.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/local/summary.csv`): paired MASE, fine-tuned / pretrained: 1.006 (95% interval 0.989 to 1.025); cov80 0.784 pretrained, 0.765 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

