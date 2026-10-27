# Model card draft: Timestral-5M fine-tuned (full+mix0.5), lofo / regime

Run id: `5M__lofo-regime__full__mix0.5`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `regime`.
- Failure families used for training: chaos, tails, mixture. Held out for testing: regime (regime switches (mean, variance and frequency jumps; Markov-switching AR)).
- 0.5 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 500; batch size: 32; learning rate: 5e-05
- Time steps seen: 19,478,528
- Training wall time: 29.2 s; peak resident memory: 1,960,288,256 bytes; final loss: 1.5043
- Peak GPU memory allocated: 578,694,656 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `lofo/regime`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.171 | 1.158 |
| sCRPS ratio to oracle | 1.189 | 1.180 |
| cov80 | 0.769 | 0.771 |

Paired MASE, fine-tuned / pretrained: 0.989 (95% interval 0.980 to 0.998).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.062 |
| sCRPS ratio to oracle | 1.073 | 1.064 |
| cov80 | 0.810 | 0.799 |

Paired MASE, fine-tuned / pretrained: 0.993.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 0.999 (95% interval 0.991 to 1.009); cov80 0.784 pretrained, 0.778 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

