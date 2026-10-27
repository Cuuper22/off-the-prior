# Model card draft: Timestral-28M fine-tuned (lora), lofo / mixture

Run id: `28M__lofo-mixture__lora__nomix`

## Model

- Base model: `mistralai/Timestral-28M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `mixture`.
- Failure families used for training: chaos, regime, tails. Held out for testing: mixture (random mixtures of the primitives and the other families).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 645,120 (base model: 27,821,184)
- Steps: 500; batch size: 32; learning rate: 0.001
- Time steps seen: 19,478,528
- Training wall time: 44.9 s; peak resident memory: 2,098,233,344 bytes; final loss: 3.6849
- Peak GPU memory allocated: 1,320,841,216 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/mixture`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.285 | 1.300 |
| sCRPS ratio to oracle | 1.301 | 1.321 |
| cov80 | 0.759 | 0.771 |

Paired MASE, fine-tuned / pretrained: 1.011 (95% interval 0.980 to 1.049).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.043 | 1.054 |
| sCRPS ratio to oracle | 1.045 | 1.056 |
| cov80 | 0.803 | 0.817 |

Paired MASE, fine-tuned / pretrained: 1.010.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.009 (95% interval 0.984 to 1.038); cov80 0.756 pretrained, 0.761 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

