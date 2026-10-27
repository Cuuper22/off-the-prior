# Model card draft: Timestral-28M fine-tuned (lora), lofo / regime

Run id: `28M__lofo-regime__lora__nomix`

## Model

- Base model: `mistralai/Timestral-28M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `regime`.
- Failure families used for training: chaos, tails, mixture. Held out for testing: regime (regime switches (mean, variance and frequency jumps; Markov-switching AR)).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 645,120 (base model: 27,821,184)
- Steps: 500; batch size: 32; learning rate: 0.001
- Time steps seen: 19,478,528
- Training wall time: 47.2 s; peak resident memory: 2,098,376,704 bytes; final loss: 3.9053
- Peak GPU memory allocated: 1,320,841,216 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.137 | 1.129 |
| sCRPS ratio to oracle | 1.151 | 1.146 |
| cov80 | 0.782 | 0.791 |

Paired MASE, fine-tuned / pretrained: 0.993 (95% interval 0.982 to 1.003).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.043 | 1.049 |
| sCRPS ratio to oracle | 1.045 | 1.051 |
| cov80 | 0.803 | 0.818 |

Paired MASE, fine-tuned / pretrained: 1.005.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.006 (95% interval 0.990 to 1.022); cov80 0.756 pretrained, 0.763 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

