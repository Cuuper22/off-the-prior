# Model card draft: Timestral-209M fine-tuned (lora), lofo / regime

Run id: `209M__lofo-regime__lora__nomix`

## Model

- Base model: `mistralai/Timestral-209M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `regime`.
- Failure families used for training: chaos, tails, mixture. Held out for testing: regime (regime switches (mean, variance and frequency jumps; Markov-switching AR)).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 2,580,480 (base model: 209,312,256)
- Steps: 300; batch size: 16; learning rate: 0.0005
- Time steps seen: 5,898,240
- Training wall time: 48.2 s; peak resident memory: 2,794,262,528 bytes; final loss: 1.6847
- Peak GPU memory allocated: 3,203,901,440 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.117 | 1.112 |
| sCRPS ratio to oracle | 1.139 | 1.135 |
| cov80 | 0.781 | 0.783 |

Paired MASE, fine-tuned / pretrained: 0.995 (95% interval 0.980 to 1.006).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.029 | 1.038 |
| sCRPS ratio to oracle | 1.029 | 1.039 |
| cov80 | 0.808 | 0.798 |

Paired MASE, fine-tuned / pretrained: 1.008.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 0.997 (95% interval 0.990 to 1.002); cov80 0.770 pretrained, 0.759 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

