"""Write bench/README.md and bench/dataset_card.md, with baseline tables read from
bench/leaderboard/*/summary.json (no number is typed by hand).

python bench/make_docs.py
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "bench")
ORDER = ["oracle", "timestral-209M", "timestral-123M", "timestral-28M", "timestral-5M", "ets", "arima", "seasonal_naive"]


def table(set_name: str, split: str = "all") -> str:
    rows = []
    for m in ORDER:
        f = os.path.join(B, "leaderboard", f"{m}_{set_name}_{split}", "summary.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f))["overall"]
        rows.append(f"| {m} | {s['n']} | {s['mase_rel_snaive']:.3f} | {s['scrps_rel_snaive']:.3f} | {s['mase_rel_oracle']:.3f} | {s['cov80']:.3f} |")
    if not rows:
        return "_Baselines not scored yet._"
    head = "| model | series | MASE / seasonal naive | sCRPS / seasonal naive | MASE / oracle | 80% coverage |\n|---|---|---|---|---|---|"
    return head + "\n" + "\n".join(rows)


def family_table(set_name: str, model: str, key: str = "by_family", split: str = "all") -> str:
    f = os.path.join(B, "leaderboard", f"{model}_{set_name}_{split}", "summary.json")
    if not os.path.exists(f):
        return ""
    s = json.load(open(f))[key]
    lines = [f"| {key.replace('by_', '').replace('_', ' ')} | series | MASE / seasonal naive | MASE / oracle |", "|---|---|---|---|"]
    for k, v in sorted(s.items()):
        lines.append(f"| {k} | {v['n']} | {v['mase_rel_snaive']:.3f} | {v['mase_rel_oracle']:.3f} |")
    return "\n".join(lines)


def counts(set_name: str) -> str:
    f = os.path.join(B, "data", f"{set_name}_manifest.csv")
    if not os.path.exists(f):
        return "(not built yet)"
    m = pd.read_csv(f)
    return f"{len(m)} series ({(m.split == 'public').sum()} public, {(m.split == 'heldout').sum()} held-out)"


README = """# Off-the-Prior benchmarks

Two benchmark sets built from this project's simulators. Every series comes with the **true-dynamics
oracle's** 21 quantiles (the best possible forecast given the simulator's true state), so a model can be
scored not only against seasonal naive (as on GIFT-Eval) but against the best achievable forecast.

## OffPrior-Diag (diagnostic)

{diag_counts}. Each config was chosen to cover one failure type found in the failure map (delay oscillators,
smooth chaos, coupled oscillators, chaos within the first Lyapunov times, recent regime switches, heavy-tailed
random walks, long memory, bursts) plus controls inside the prior and controls that nobody can forecast. Every
series carries `failure_type`, its generator tags and a one-line `reason` (see `data/diag_manifest.csv`).
Context 2,048 steps, horizon 128.

## OffPrior-LB (leaderboard)

{lb_counts}: every config of the chaos, regime, tails and in-prior grids plus 40 fixed random
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

{diag_table}

By failure type (Timestral-209M):

{diag_ft}

### Baselines, OffPrior-LB (public + held-out)

{lb_table}

By family (Timestral-209M):

{lb_fam}

Numbers above are read from `bench/leaderboard/*/summary.json` by `bench/make_docs.py`.
"""

CARD = """---
license: apache-2.0
pretty_name: Off-the-Prior benchmarks
task_categories:
- time-series-forecasting
tags:
- time-series
- forecasting
- synthetic
- chaos
- regime-switching
- heavy-tails
- benchmark
---

# Off-the-Prior benchmarks

Synthetic forecasting benchmarks built to find where time-series foundation models trained on simple
synthetic priors stop working: chaotic and delay-feedback dynamics, coupled oscillators, regime switches,
heavy tails, long memory, bursts, learned mixtures of these, and in-prior controls. Every series includes the
**oracle's 21 quantiles**: a Monte Carlo continuation of the true simulator from its true state, so models can be
compared with the best achievable forecast, not only with each other.

- **diag**: {diag_counts}, each tagged with why it is hard.
- **lb**: {lb_counts}, all simulator configs, several context lengths and horizons.

Files are NumPy `.npz` archives (load with `numpy.load`; the Hugging Face `datasets` loader is not used).
Each `.npz` file holds `series_id`, `context` (N, T), `future` (N, H), `oracle_q` (N, H, 21), `period` (N,) and a
JSON `meta` string per series (generator, parameters, tags, reason). Files are rebuilt bit-for-bit (array contents)
by `bench/build.py` from the config lists in `bench/configs/`; `*_SHA256SUMS` lists content hashes.

Evaluation: `python bench/eval.py --set diag --predictions <dir>`; see the repository README.

The seeds and generator code are public. The held-out leaderboard files exist for convenience, not secrecy.

License: the data are generated by the project's own code (Apache 2.0). No third-party data are included.
"""


def main():
    fmt = {"diag_counts": counts("diag"), "lb_counts": counts("lb"), "diag_table": table("diag"), "lb_table": table("lb"),
           "diag_ft": family_table("diag", "timestral-209M", "by_failure_type") or "_not scored yet_",
           "lb_fam": family_table("lb", "timestral-209M") or "_not scored yet_"}
    open(os.path.join(B, "README.md"), "w").write(README.format(**fmt))
    open(os.path.join(B, "dataset_card.md"), "w").write(CARD.format(**fmt))
    print("wrote bench/README.md and bench/dataset_card.md")


if __name__ == "__main__":
    main()
