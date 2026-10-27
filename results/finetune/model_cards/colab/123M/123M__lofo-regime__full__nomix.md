# Model card draft: Timestral-123M fine-tuned (full), lofo / regime

Run id: `123M__lofo-regime__full__nomix`

## Model

- Base model: `mistralai/Timestral-123M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `regime`.
- Failure families used for training: chaos, tails, mixture. Held out for testing: regime (regime switches (mean, variance and frequency jumps; Markov-switching AR)).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 122,606,720 (base model: 122,606,720)
- Steps: 300; batch size: 16; learning rate: 2e-05
- Time steps seen: 5,898,240
- Training wall time: 35.1 s; peak resident memory: 2,463,838,208 bytes; final loss: 1.7703
- Peak GPU memory allocated: 3,844,695,552 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.115 | 1.114 |
| sCRPS ratio to oracle | 1.134 | 1.135 |
| cov80 | 0.772 | 0.774 |

Paired MASE, fine-tuned / pretrained: 1.000 (95% interval 0.994 to 1.007).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.032 | 1.040 |
| sCRPS ratio to oracle | 1.031 | 1.041 |
| cov80 | 0.803 | 0.788 |

Paired MASE, fine-tuned / pretrained: 1.009.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 0.995 (95% interval 0.990 to 0.998); cov80 0.774 pretrained, 0.766 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

