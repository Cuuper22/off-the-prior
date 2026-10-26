# Off-the-Prior benchmarks

Two benchmark sets built from this project's simulators. Every series comes with the **true-dynamics
oracle's** 21 quantiles (the best possible forecast given the simulator's true state), so a model can be
scored not only against seasonal naive (as on GIFT-Eval) but against the best achievable forecast.

## OffPrior-Diag (diagnostic)

(not built yet). Each config was chosen to cover one failure type found in the failure map (delay oscillators,
smooth chaos, coupled oscillators, chaos within the first Lyapunov times, recent regime switches, heavy-tailed
random walks, long memory, bursts) plus controls inside the prior and controls that nobody can forecast. Every
series carries `failure_type`, its generator tags and a one-line `reason` (see `data/diag_manifest.csv`).
Context 2,048 steps, horizon 128.

## OffPrior-LB (leaderboard)

(not built yet): every config of the chaos, regime, tails and in-prior grids plus 40 fixed random
mixture-of-primitives genomes, 16 seeds per config spread over contexts 512 / 2,048 / 8,192 and horizons
16 / 128 / 1,024. Seed indices alternate between a public and a held-out split. The seeds and the code are
public, so the held-out split is a convenience (futures left out of the default Hugging Face config), not
a secret test set.

## Run it

```bash
python bench/build.py --set diag            # rebuilds bench/data/diag_*.npz (deterministic; checks: diag_SHA256SUMS)
python bench/build.py --set lb --workers 8  # the leaderboard set (large; not in git)
python bench/eval.py --set diag --model timestral-5M
python bench/eval.py --set lb --split public --predictions my_preds/ --name mymodel
```

`--predictions DIR`: for every data file `bench/data/<set>_<split>_T<T>_H<H>.npz` put a file with the same
name in DIR holding `quantiles`, an array (N, H, 21) in the data file's series order, at levels
0.01, 0.05, 0.10, ..., 0.95, 0.99.

## Scores

Per series: MASE, scaled CRPS (sCRPS), WQL, 80% coverage, calibration error (definitions in `otp/metrics.py`).
Summary, GIFT-Eval style: geometric mean over series of MASE and sCRPS divided by seasonal naive's; plus the
geometric mean of MASE divided by the oracle's (1.0 = as good as the true dynamics).

### Baselines, OffPrior-Diag (all splits)

_Baselines not scored yet._

By failure type (Timestral-209M):

_not scored yet_

### Baselines, OffPrior-LB (public + held-out)

_Baselines not scored yet._

By family (Timestral-209M):

_not scored yet_

Numbers above are read from `bench/leaderboard/*/summary.json` by `bench/make_docs.py`.
