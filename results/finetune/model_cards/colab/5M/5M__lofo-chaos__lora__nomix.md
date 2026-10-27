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

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 215,040 (base model: 5,208,960)
- Steps: 500; batch size: 32; learning rate: 0.001
- Time steps seen: 19,478,528
- Training wall time: 32.4 s; peak resident memory: 1,980,923,904 bytes; final loss: 3.8630
- Peak GPU memory allocated: 461,856,256 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/chaos`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 2.197 | 2.197 |
| sCRPS ratio to oracle | 2.238 | 2.249 |
| cov80 | 0.714 | 0.745 |

Paired MASE, fine-tuned / pretrained: 1.000 (95% interval 0.975 to 1.026).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.089 |
| sCRPS ratio to oracle | 1.073 | 1.092 |
| cov80 | 0.810 | 0.817 |

Paired MASE, fine-tuned / pretrained: 1.019.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.007 (95% interval 0.991 to 1.025); cov80 0.784 pretrained, 0.787 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

