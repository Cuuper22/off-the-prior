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

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 500; batch size: 32; learning rate: 5e-05
- Time steps seen: 19,478,528
- Training wall time: 28.8 s; peak resident memory: 1,960,378,368 bytes; final loss: 1.8387
- Peak GPU memory allocated: 578,694,656 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/mixture`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.273 | 1.271 |
| sCRPS ratio to oracle | 1.310 | 1.300 |
| cov80 | 0.752 | 0.763 |

Paired MASE, fine-tuned / pretrained: 0.998 (95% interval 0.985 to 1.012).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.072 |
| sCRPS ratio to oracle | 1.073 | 1.073 |
| cov80 | 0.810 | 0.791 |

Paired MASE, fine-tuned / pretrained: 1.002.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.001 (95% interval 0.991 to 1.012); cov80 0.784 pretrained, 0.781 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

