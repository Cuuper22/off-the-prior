# results/fix: zero-training fixes

## Self-validation switch (Timestral or the method of analogues, decided by the context)

`scripts/selfval_fix.py` backtests Timestral and the method of analogues on the last two
128-step windows of each series' own context. The rule "switch to the analogue forecast when its
backtest error is below m x Timestral's on each of the last K windows" is then applied; the
final quantiles are Timestral's, shifted to the chosen median. No training, no oracle, no future
values: deployable as is.

Three stages, kept separate so that no number reported for the final rule was used to choose it:

1. `selfval_v1_onewindow/` (superseded): one window, soft blends. Helped chaos, hurt tails and
   regime: one backtest window favours the analogues by chance on random walks.
2. `selfval/`: two windows, all four families, all four sizes (8 seeds per config).
   `scripts/selfval_rules.py` tuned (K, m) on even seed indices with the constraint that no other
   family loses more than 0.5% (`chosen_rule.json`: K = 1, m = 0.6 for every size) and reports it on
   odd seeds (`chosen_rule_test_half.csv`): chaos improved, but tails lost 3.1-4.3%, so the
   constraint held on the tuning half only. `rules_test_half.csv` shows every rule on the odd half;
   K = 2, m = 0.7 had the best tradeoff there, which is a choice made after seeing that half.
3. `selfval_confirm/`: the K = 2, m = 0.7 rule was frozen (`frozen_rule.json`) and evaluated once on
   FRESH samples (seed offset 100; generated for this run by `scripts/colab/job_selfval_confirm.sh`).
   `frozen_rule_confirmation.csv` holds the result reported in the paper.
