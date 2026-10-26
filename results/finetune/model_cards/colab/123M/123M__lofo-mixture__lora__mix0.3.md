# Model card draft: Timestral-123M fine-tuned (lora+mix0.3), lofo / mixture

Run id: `123M__lofo-mixture__lora__mix0.3`

## Model

- Base model: `mistralai/Timestral-123M` (license and terms: those of the base model; see its model card).
- Method: LoRA adapters (rank r) on the attention and MLP linears, merged into the weights at export.
- Format: `config.json` + `model.safetensors`, loadable with `otp.model.forecast(size=<checkpoint dir>, ...)` (shipped defaults: symmetric averaging on, non-negative clipping on).

## Training data

- Protocol: `lofo`, held-out family: `mixture`.
- Failure families used for training: chaos, regime, tails. Held out for testing: mixture (random mixtures of the primitives and the other families).
- 0.3 of every training batch is replay of in-prior series (random trend + season + AR noise).
- Series are simulated by the generators in `otp/generators/`; training seeds are disjoint from evaluation seeds.

## Training cost

Placeholders name the result file and column the number comes from (results/finetune/colab/<RUN_ID>/costs.csv).

- Trainable parameters: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=123M__lofo-mixture__lora__mix0.3 :: trainable_params}} (base model: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=123M__lofo-mixture__lora__mix0.3 :: base_params}})
- Steps: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=123M__lofo-mixture__lora__mix0.3 :: steps}}; batch size: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=123M__lofo-mixture__lora__mix0.3 :: batch_size}}; learning rate: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=123M__lofo-mixture__lora__mix0.3 :: lr}}
- Time steps seen: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=123M__lofo-mixture__lora__mix0.3 :: time_steps_seen}}
- Training wall time: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=123M__lofo-mixture__lora__mix0.3 :: wall_sec}} s on {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=123M__lofo-mixture__lora__mix0.3 :: gpu}}; final loss: {{results/finetune/colab/<RUN_ID>/costs.csv :: run_id=123M__lofo-mixture__lora__mix0.3 :: final_loss}}

## Held-out evaluation

From `results/finetune/colab/<RUN_ID>/summary.csv`, set `lofo/mixture`.

| metric | pretrained | fine-tuned |
|---|---|---|
| MASE ratio to oracle | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=lofo/mixture :: pre_mase_ratio}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=lofo/mixture :: ft_mase_ratio}} |
| sCRPS ratio to oracle | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=lofo/mixture :: pre_scrps_ratio}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=lofo/mixture :: ft_scrps_ratio}} |
| cov80 | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=lofo/mixture :: pre_cov80}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=lofo/mixture :: ft_cov80}} |

Paired MASE, fine-tuned / pretrained: {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=lofo/mixture :: paired_mase_ft_over_pre}}.

## Regression check (normal series)

From `results/finetune/colab/<RUN_ID>/summary.csv`, sets `regression/in_prior` and `regression/real`.

| set | metric | pretrained | fine-tuned |
|---|---|---|---|
| regression/in_prior | MASE ratio to oracle | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=regression/in_prior :: pre_mase_ratio}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=regression/in_prior :: ft_mase_ratio}} |
| regression/in_prior | cov80 | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=regression/in_prior :: pre_cov80}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=regression/in_prior :: ft_cov80}} |
| regression/in_prior | paired MASE ft / pre | | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=regression/in_prior :: paired_mase_ft_over_pre}} |
| regression/real | cov80 | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=regression/real :: pre_cov80}} | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=regression/real :: ft_cov80}} |
| regression/real | paired MASE ft / pre | | {{results/finetune/colab/<RUN_ID>/summary.csv :: run_id=123M__lofo-mixture__lora__mix0.3, eval_set=regression/real :: paired_mase_ft_over_pre}} |

## Scope

Evaluated on synthetic held-out samples at context 2048 and horizon 128 and on the regression series listed above. No other settings were evaluated.

