# Model card draft: Timestral-5M fine-tuned (lora), knob / chaos

Run id: `5M__knob-chaos__lora__nomix`

## Model

- Base model: `mistralai/Timestral-5M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `knob`, held-out family: `chaos`.
- Failure families used for training: all four, on one range of each family's difficulty knob; the test set is the other range of the chaos family (ranges in `otp/finetune_splits.py`).
- No replay: every training batch is failure-family series.
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Placeholders name the result file and column the number comes from (results/finetune/colab/<RUN_ID>/costs.csv).

- Trainable parameters: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=5M__knob-chaos__lora__nomix :: trainable_params}} (base model: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=5M__knob-chaos__lora__nomix :: base_params}})
- Steps: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=5M__knob-chaos__lora__nomix :: steps}}; batch size: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=5M__knob-chaos__lora__nomix :: batch_size}}; learning rate: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=5M__knob-chaos__lora__nomix :: lr}}
- Time steps seen: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=5M__knob-chaos__lora__nomix :: time_steps_seen}}
- Training wall time: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=5M__knob-chaos__lora__nomix :: wall_sec}} s on {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=5M__knob-chaos__lora__nomix :: gpu}}; final loss: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=5M__knob-chaos__lora__nomix :: final_loss}}

## Held-out evaluation

From `results/finetune/colab/<RUN_ID>/summary.csv`, set `knob/chaos`.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=knob/chaos :: pre_mase_ratio}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=knob/chaos :: ft_mase_ratio}} |
| sCRPS ratio to oracle | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=knob/chaos :: pre_scrps_ratio}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=knob/chaos :: ft_scrps_ratio}} |
| cov80 | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=knob/chaos :: pre_cov80}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=knob/chaos :: ft_cov80}} |

Paired MASE, fine-tuned / pretrained: {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=knob/chaos :: paired_mase_ft_over_pre}}.

## Regression check (normal series)

From `results/finetune/colab/<RUN_ID>/summary.csv`, sets `regression/in_prior` and `regression/real`.

| set | metric | pretrained | fine-tuned |
|---|---|---|---|
| regression/in_prior | MASE ratio to oracle | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=regression/in_prior :: pre_mase_ratio}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=regression/in_prior :: ft_mase_ratio}} |
| regression/in_prior | cov80 | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=regression/in_prior :: pre_cov80}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=regression/in_prior :: ft_cov80}} |
| regression/in_prior | paired MASE ft / pre | | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=regression/in_prior :: paired_mase_ft_over_pre}} |
| regression/real | cov80 | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=regression/real :: pre_cov80}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=regression/real :: ft_cov80}} |
| regression/real | paired MASE ft / pre | | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=5M__knob-chaos__lora__nomix, eval_set=regression/real :: paired_mase_ft_over_pre}} |

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

