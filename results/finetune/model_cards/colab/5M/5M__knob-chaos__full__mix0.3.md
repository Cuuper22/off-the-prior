# Model card draft: Timestral-5M fine-tuned (full+mix0.3), knob / chaos

Run id: `5M__knob-chaos__full__mix0.3`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: all parameters trained.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `chaos`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the chaos family (ranges in `otp/finetune_splits.py`).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Copied from `results/finetune/colab/run1/costs.csv`.

- Trainable parameters: 5,208,960 (base model: 5,208,960)
- Steps: 500; batch size: 32; learning rate: 5e-05
- Time steps seen: 19,478,528
- Training wall time: 28.2 s; peak resident memory: 1,959,600,128 bytes; final loss: 1.1853
- Peak GPU memory allocated: 578,694,656 bytes on NVIDIA A100-SXM4-40GB

## Held-out evaluation

Copied from `results/finetune/colab/run1/summary.csv` (set `knob/chaos`, 32 samples, 500-path true-dynamics oracle). Ratio = geometric mean over samples of the metric divided by the oracle's.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 3.423 | 3.471 |
| sCRPS ratio to oracle | 3.576 | 3.631 |
| cov80 | 0.713 | 0.713 |

Paired MASE, fine-tuned / pretrained: 1.014 (95% interval 0.973 to 1.059).

## Regression check (normal series)

Copied from `results/finetune/colab/run1/summary.csv` (set `regression/in_prior`, 500 series).

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | 1.070 | 1.057 |
| sCRPS ratio to oracle | 1.073 | 1.060 |
| cov80 | 0.810 | 0.792 |

Paired MASE, fine-tuned / pretrained: 0.988.

Real-data windows (`regression/real`, 40 windows, no oracle; from `results/finetune/colab/run1/summary.csv`): paired MASE, fine-tuned / pretrained: 1.008 (95% interval 0.997 to 1.021); cov80 0.784 pretrained, 0.773 fine-tuned.

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

