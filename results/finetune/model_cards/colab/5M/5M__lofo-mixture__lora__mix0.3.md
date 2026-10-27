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

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 215,040 (base model: 5,208,960)
- Steps: 500; batch size: 32; learning rate: 0.001
- Time steps seen: 19,478,528
- Training wall time: 32.9 s; peak resident memory: 1,980,833,792 bytes; final loss: 1.8103
- Peak GPU memory allocated: 461,856,256 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/mixture`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.273 | 1.290 |
| sCRPS ratio to oracle | 1.310 | 1.313 |
| cov80 | 0.752 | 0.770 |

Paired MASE, fine-tuned / pretrained: 1.013 (95% interval 0.985 to 1.051).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.061 |
| sCRPS ratio to oracle | 1.073 | 1.064 |
| cov80 | 0.810 | 0.806 |

Paired MASE, fine-tuned / pretrained: 0.992.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 0.996 (95% interval 0.982 to 1.012); cov80 0.784 pretrained, 0.797 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

