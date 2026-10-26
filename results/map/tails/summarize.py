"""Build summary.csv and README.md for a tails sweep folder from its scores.csv and provenance.json.

Usage (from the repository root):
    python results/map/tails/summarize.py --dir results/map/tails
    python results/map/tails/summarize.py --dir results/map/tails_big --big

summary.csv: one row per [config_id, generator, knob_value, forecaster, T, H] with
  - n (number of samples = seeds)
  - for mase_ratio_oracle, scrps_ratio_oracle, mase_ratio_classical: geometric mean, mean, std,
    95% bootstrap interval of the geometric mean (otp.analysis.curve) and the median over seeds
  - cov80_mean, cov98_mean, calib_mean: mean over seeds
The oracle is evaluated once per sample at the longest context (it conditions on the true state and
does not depend on the context length); its rows are copied to every context length so that tables
can put it next to each Timestral context.

README.md is written from summary.csv and provenance.json: every number in its tables is copied from
summary.csv (printed to 4 significant digits; coverage to 3 decimals).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from otp.analysis import curve, ratios  # noqa: E402
from otp.grids.tails import BIG_IDS  # noqa: E402
from otp.io import read_json  # noqa: E402

BY = ["config_id", "generator", "knob_value", "forecaster", "T", "H"]
METRICS = ["mase_ratio_oracle", "scrps_ratio_oracle", "mase_ratio_classical"]


def build_summary(scores_path: str) -> pd.DataFrame:
    df = ratios(pd.read_csv(scores_path))
    orc = df[df.forecaster == "oracle"]
    rest = df[df.forecaster != "oracle"]
    dup = [orc.assign(T=t) for t in sorted(rest["T"].unique())]
    df = pd.concat([rest] + dup, ignore_index=True)
    out = None
    for m in METRICS:
        c = curve(df, BY, m)
        med = df.groupby(BY, dropna=False)[m].median().rename(f"{m}_median").reset_index()
        c = c.merge(med, on=BY)
        out = c if out is None else out.merge(c.drop(columns=["n"]), on=BY)
    cov = df.groupby(BY, dropna=False)[["cov80", "cov98", "calib"]].mean().rename(
        columns=lambda s: f"{s}_mean").reset_index()
    out = out.merge(cov, on=BY)
    return out.sort_values(["generator", "config_id", "T", "H", "forecaster"]).reset_index(drop=True)


def variant(cid: str) -> str:
    """Config id without the knob value: the table a config belongs to."""
    parts = cid.split("/")
    gen, tail = parts[1], parts[2]
    if gen == "stable_ar":
        return f"tails/stable_ar, rho = {tail.split('_rho')[1]}"
    if gen == "msm":
        return "tails/msm, observed series = " + {"ret": "returns", "cum": "cumulative sum"}[tail.split("_")[1]]
    return f"tails/{gen}"


KNOB_NAME = {"stable_ar": "alpha", "fbm": "H_hurst", "hawkes": "eta", "msm_cascade": "k"}


def g(x, nd=4):
    return f"{x:.{nd}g}"


def table(s: pd.DataFrame, T: int, H: int) -> str:
    lines = []
    sub = s[(s["T"] == T) & (s["H"] == H)].copy()
    sub["variant"] = sub.config_id.map(variant)
    for var, gv in sub.groupby("variant", sort=False):
        gen = gv.generator.iloc[0]
        lines.append(f"\n**{var}** (knob `{KNOB_NAME[gen]}`)\n")
        lines.append(f"| {KNOB_NAME[gen]} | forecaster | MASE/oracle gmean | MASE/oracle median | sCRPS/oracle gmean | "
                     "sCRPS/oracle median | MASE/classical gmean | cov80 | cov98 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for (kv, cid), gk in gv.groupby(["knob_value", "config_id"], sort=True):
            order = {"oracle": 0}
            gk = gk.assign(o=gk.forecaster.map(lambda f: order.get(f, 1 if f.startswith("timestral") else 2)),
                           sz=gk.forecaster.map(lambda f: float(f.split("-")[1][:-1]) if f.startswith("timestral") else 0))
            for _, r in gk.sort_values(["o", "sz", "forecaster"]).iterrows():
                if r.forecaster == "oracle":
                    cells = ["-"] * 5
                else:
                    cells = [g(r.mase_ratio_oracle_gmean), g(r.mase_ratio_oracle_median), g(r.scrps_ratio_oracle_gmean),
                             g(r.scrps_ratio_oracle_median), g(r.mase_ratio_classical_gmean)]
                lines.append(f"| {kv:g} | {r.forecaster} | " + " | ".join(cells) +
                             f" | {r.cov80_mean:.3f} | {r.cov98_mean:.3f} |")
    return "\n".join(lines)


def write_readme(d: str, s: pd.DataFrame, big: bool) -> None:
    prov = read_json(os.path.join(d, "provenance.json"))
    argv = " ".join(prov["argv"][1:]) if "argv" in prov else "?"
    sizes = sorted({f for f in s.forecaster if f.startswith("timestral")})
    contexts = sorted(s["T"].unique().tolist())
    horizons = sorted(s["H"].unique().tolist())
    classical = sorted({f for f in s.forecaster if f in ("seasonal_naive", "ets", "arima")})
    n_cfg = s.config_id.nunique()
    seeds = int(s[s.forecaster == "oracle"].n.max())
    folder = os.path.relpath(d)
    if big:
        sw = ("python scripts/sweep.py --grid otp.grids.tails --only " + " ".join(BIG_IDS) +
              " --sizes 123M 209M --contexts 2048 --horizons 16 128 1024 --seeds 4 --classical seasonal_naive"
              f" --n-paths 2000 --out {folder}")
    else:
        sw = ("python scripts/sweep.py --grid otp.grids.tails --sizes 5M 28M --contexts 512 2048 8192 "
              f"--horizons 16 128 1024 --seeds 8 --n-paths 2000 --out {folder}")
    extra = ""
    if not big:
        extra += """
## 123M and 209M models

The 123M and 209M runs are not in this folder: they moved to Colab job A (full grid). This folder holds the
5M and 28M main sweep only.
"""
        cal = os.path.join(d, "oracle_calibration.csv")
        if os.path.exists(cal):
            c = pd.read_csv(cal)
            extra += f"""
## Oracle calibration check

`oracle_calibration.py` builds {int(c.n_samples.iloc[0])} independent samples per config (context 1024, horizon 128,
500 oracle paths) and measures how often the realized future falls inside the oracle's own 80% and 98% intervals
(`oracle_calibration.csv`; `se` = standard error over samples). A calibrated oracle gives 0.80 and 0.98.
The oracle cov80 and cov98 in the tables above come from {seeds} samples per config at 128 correlated steps, so they
move more than these. Hawkes counts are integers, so an interval with inclusive end points covers more than its nominal level.

| config | oracle cov80 | se | oracle cov98 | se |
|---|---|---|---|---|
""" + "\n".join(f"| {r.config_id} | {r.oracle_cov80:.3f} | {r.cov80_se:.3f} | {r.oracle_cov98:.3f} | {r.cov98_se:.3f} |"
                 for r in c.itertuples()) + "\n"
    txt = f"""# {folder}: heavy-tail and fractal family, Timestral failure map

Facts only. Every number below is copied from `summary.csv` in this folder.

## Contents

- `scores.csv`: one row per sample x forecaster x context length T x horizon H (written by `scripts/sweep.py`).
- `provenance.json`: command line, git commit, host, threads, wall time of the sweep.
- `summary.csv`: built from `scores.csv` by `results/map/tails/summarize.py` with `otp.analysis.ratios` and
  `otp.analysis.curve`, grouped by `[config_id, generator, knob_value, forecaster, T, H]`.
- `samples.npz`: the generated samples (contexts, realized futures, oracle quantiles). Not committed if over 50 MB;
  see the regeneration commands.
- `README.md`: this file (written by `summarize.py`).

Sweep: {n_cfg} configs, {seeds} seeds each ({n_cfg * seeds} samples), Timestral {", ".join(s.replace("timestral-", "") for s in sizes)},
context lengths {contexts}, horizons {horizons}, classical models: {", ".join(classical)}.
Wall time of the sweep (`provenance.json`): {prov.get("wall_sec")} s on {prov.get("threads")} thread(s), git `{prov.get("git")}`.

## Regeneration

```
export PYTHONPATH=$PWD OTP_THREADS=1
{sw}
python results/map/tails/summarize.py --dir {folder}{" --big" if big else ""}
```

Sample generation is seeded and deterministic (`samples.npz` is rebuilt by the first command if absent).
`--n-paths 2000` is used instead of the harness default of 500: the oracle quantiles at levels 0.01 and 0.99
are Monte Carlo estimates from 2000 simulated continuations (1% of 500 is 5 paths; heavy tails make that noisy).

## Terms

- Sample: one simulated series of length 8192 (context) plus 1024 (realized future) for one config and seed.
- T: context length used by the forecaster (the last T steps of the 8192). H: horizon; horizons 16 and 128 are
  prefixes of the 1024-step forecast.
- Oracle: Monte Carlo continuation of the true generating process from its true state with known parameters
  (2000 paths). It is computed once per sample; its rows are copied to every T in `summary.csv`.
- MASE: mean absolute error of the median forecast divided by the in-sample one-step naive error on the context.
  sCRPS: scaled CRPS (average pinball loss over the 21 quantile levels, times 2, divided by the same scale).
  Ratio to oracle = metric of the forecaster / metric of the oracle on the same sample, T and H.
  Ratio to classical = metric / metric of the best classical model on that sample ({", ".join(classical)}).
- gmean: geometric mean over seeds of the per-sample ratio (the mean of the logs, exponentiated). median: median over
  seeds of the per-sample ratio. Heavy-tailed series can make a few samples dominate a mean; the median is reported
  next to the geometric mean for that reason.
- cov80, cov98: fraction of realized future points inside the forecaster's [0.1, 0.9] and [0.01, 0.99] quantile
  intervals, averaged over seeds. A calibrated forecast has 0.80 and 0.98.

## Generators (`otp/generators/tails.py`)

- `stable_ar`: x_t = rho x_(t-1) + e_t with e_t symmetric alpha-stable (scale 1, sampled by the Chambers-Mallows-Stuck
  formula); rho = 1.0 is a random walk. alpha = 2.0 is Gaussian (standard deviation sqrt(2)). Oracle state: last value.
- `fbm`: fractional Brownian motion, unit time step, increments are fractional Gaussian noise. Context increments are
  exact (Davies-Harte); future increments are drawn from the exact Gaussian conditional given the last 2048 increments
  (the oracle conditions on that window, not on the whole 8192-step history).
- `hawkes`: discrete-time approximation of a Hawkes process with exponential kernel: counts N_t ~ Poisson(mu + s_t),
  s_(t+1) = d s_t + eta (1 - d) N_t, d = exp(-1/4); mean count per bin is 5.0 for every eta (mu = 5 (1 - eta)).
- `msm_cascade`: Markov-switching multifractal, multipliers in {{1.4, 0.6}}, renewal probability of the fastest multiplier
  0.5, frequency ratio 3 between neighbouring multipliers, returns = sqrt(product of multipliers) x standard normal.
  Config id suffix `ret` = observed series is the returns, `cum` = the cumulative sum of the returns.

## Tables: T = 2048, H = 128

Columns: knob value, forecaster, MASE and sCRPS ratios to the oracle (gmean and median over seeds), MASE ratio to
the best classical model (gmean), cov80, cov98. The `oracle` row shows the oracle's own coverage on the same samples.
{table(s, 2048, 128)}
{extra}"""
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--big", action="store_true")
    a = ap.parse_args()
    s = build_summary(os.path.join(a.dir, "scores.csv"))
    s.to_csv(os.path.join(a.dir, "summary.csv"), index=False)
    write_readme(a.dir, s, a.big)
    print(f"wrote {len(s)} summary rows and README.md to {a.dir}")


if __name__ == "__main__":
    main()
