# Model card draft: Timestral-28M fine-tuned (lora+mix0.3), knob / regime

Run id: `28M__knob-regime__lora__mix0.3`

## Model

- Base model: `mistralai/Timestral-28M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `regime`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the regime family (ranges in `otp/finetune_splits.py`).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 645,120 (base model: 27,821,184)
- Steps: 500; batch size: 32; learning rate: 0.001
- Time steps seen: 19,478,528
- Training wall time: 46.4 s; peak resident memory: 2,096,672,768 bytes; final loss: 1.1128
- Peak GPU memory allocated: 1,320,841,216 bytes on NVIDIA A100-SXM4-40GB
- This is the same trained model as run `28M__knob-chaos__lora__mix0.3` (knob protocol: one model per size, method and replay share is tested on every held-out family); the cost figures above are that model's.

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `knob/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.559 | 1.574 |
| sCRPS ratio to oracle | 1.714 | 1.763 |
| cov80 | 0.674 | 0.637 |

Paired MASE, fine-tuned / pretrained: 1.010 (95% interval 0.983 to 1.031).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.043 | 1.035 |
| sCRPS ratio to oracle | 1.045 | 1.035 |
| cov80 | 0.803 | 0.801 |

Paired MASE, fine-tuned / pretrained: 0.992.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.005 (95% interval 0.986 to 1.029); cov80 0.756 pretrained, 0.761 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

