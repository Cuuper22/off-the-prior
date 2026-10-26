# results/finetune/colab_smoke

GPU-path check of `notebooks/finetune_colab.ipynb`: the notebook's code cells run with `TINY = True` on a Colab T4 runtime
(driver `scripts/colab/run_job.sh`, job script `job_ft_smoke.sh` in this folder). Budget: 5M model, leave-one-family-out with
regime and tails held out, 2 steps of batch size 4, 3 held-out samples per set, 20 oracle paths. These numbers check the
code path only; they are not results.

- `costs.csv`, `summary.csv`, `provenance.json`, `colab_meta.json`: the files the notebook wrote on the runtime.
- `started_utc.txt`, `ended_utc.txt`: job start and end.
- `../colab_spend.log`: compute-unit balance before and after each Colab job started by this task (one line per job; the lines for
  `ft-smoke` cover three launches, two of which stopped on notebook errors that were then fixed).
