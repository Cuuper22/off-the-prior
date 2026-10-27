"""Build summary.csv and README.md for a regime-family sweep folder.

Usage:
  python scripts/regime_summary.py --dir results/map/regime
  python scripts/regime_summary.py --dir results/map/regime_big --big

summary.csv: one row per [config_id, generator, knob_value, forecaster, T, H] with, over seeds,
the geometric mean / mean / std / 95% bootstrap interval of mase_ratio_oracle,
scrps_ratio_oracle and mase_ratio_classical (otp.analysis.ratios + otp.analysis.curve), plus
explicit `placement` ("context", "horizon" or "markov") and `d` columns (parsed from config_id),
`switch_visible` (placement context: d <= T), the mean cov80 of the forecaster and of the
oracle for the same config and H. The README tables are written from summary.csv by this script.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp.analysis import curve, ratios  # noqa: E402
from otp.grids.regime import parse_config_id  # noqa: E402
from otp.io import read_json, write_json  # noqa: E402

BY = ["config_id", "generator", "knob_value", "forecaster", "T", "H"]
METRICS = ["mase_ratio_oracle", "scrps_ratio_oracle", "mase_ratio_classical"]
GEN_ORDER = ["mean_jump", "variance_jump", "frequency_jump", "markov_ar"]
KNOB_NAME = {"mean_jump": "delta", "variance_jump": "factor", "frequency_jump": "ratio", "markov_ar": "p_switch"}


def build_summary(scores_path: str) -> pd.DataFrame:
    df = ratios(pd.read_csv(scores_path))
    df = df[df.forecaster != "oracle"]
    out = None
    for m in METRICS:
        c = curve(df, BY, m)
        out = c if out is None else out.merge(c.drop(columns=["n"]), on=BY)
    cov = df.groupby(BY, dropna=False)["cov80"].mean().rename("cov80_mean").reset_index()
    out = out.merge(cov, on=BY)
    full = pd.read_csv(scores_path)
    orc = full[full.forecaster == "oracle"].groupby(["config_id", "H"])["cov80"].mean()
    orc = orc.rename("oracle_cov80_mean").reset_index()
    out = out.merge(orc, on=["config_id", "H"], how="left")
    info = pd.DataFrame([parse_config_id(c) for c in out.config_id])
    out["placement"] = info.placement.values
    out["d"] = info.d.values
    out["switch_visible"] = np.where(out.placement == "context", out.d <= out["T"], np.nan)
    cols = ["config_id", "generator", "placement", "d", "knob_value", "forecaster", "T", "H", "switch_visible", "n"]
    rest = [c for c in out.columns if c not in cols]
    return out[cols + rest]


# ------------------------------------------------------------------ README tables
def md(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def f3(v) -> str:
    return "" if pd.isna(v) else f"{v:.3f}"


def cell(s: pd.DataFrame, **kw) -> pd.Series | None:
    q = s
    for k, v in kw.items():
        q = q[q[k].isna()] if v is None else q[q[k] == v]
    return q.iloc[0] if len(q) else None


def val(s, col, **kw) -> str:
    r = cell(s, **kw)
    return "" if r is None else f3(r[col])


def tables(summary: pd.DataFrame, sizes: list[str], t_main: int, h_main: int) -> str:
    out = []
    S = summary
    d_vals = sorted(S.d.dropna().unique().astype(int))
    short = lambda f: f.replace("timestral-", "")
    for metric, col, title in [
            ("mase", "mase_ratio_oracle_gmean", "MASE ratio to oracle (geometric mean over seeds)"),
            ("scrps", "scrps_ratio_oracle_gmean", "sCRPS ratio to oracle (geometric mean over seeds)"),
            ("cl", "mase_ratio_classical_gmean", "MASE ratio to the best classical model (geometric mean over seeds)"),
            ("cov", "cov80_mean", "cov80 (fraction of horizon points inside the 10%-90% interval; 0.8 is calibrated)")]:
        out.append(f"### {title}, T = {t_main}, H = {h_main}\n")
        for gen in GEN_ORDER:
            g = S[(S.generator == gen) & (S["T"] == t_main) & (S.H == h_main)]
            if g.empty:
                continue
            knob = KNOB_NAME[gen]
            if gen == "markov_ar":
                header = [knob] + [short(f) for f in sizes] + (["oracle cov80"] if metric == "cov" else [])
                rows = []
                for k in sorted(g.knob_value.unique()):
                    r = [f"{k:g}"] + [val(g, col, knob_value=k, forecaster=f) for f in sizes]
                    if metric == "cov":
                        r.append(val(g, "oracle_cov80_mean", knob_value=k, forecaster=sizes[0] if sizes[0] in set(g.forecaster) else g.forecaster.iloc[0]))
                    rows.append(r)
                out.append(f"**{gen}** (switching through context and horizon)\n\n" + md(header, rows))
                continue
            header = ["size", knob] + [f"d={d}" for d in d_vals] + ["horizon"]
            rows = []
            for f in sizes:
                for k in sorted(g.knob_value.unique()):
                    r = [short(f), f"{k:g}"]
                    r += [val(g, col, knob_value=k, forecaster=f, placement="context", d=d) for d in d_vals]
                    r.append(val(g, col, knob_value=k, forecaster=f, placement="horizon"))
                    rows.append(r)
            out.append(f"**{gen}** (columns d: placement context with the switch d steps before the origin; horizon: placement horizon)\n\n" + md(header, rows))
    # d x T
    out.append(f"### MASE ratio to oracle by d and T, H = {h_main} (geometric mean over seeds)\n")
    Ts = sorted(S["T"].unique())
    for gen in GEN_ORDER[:3]:
        g = S[(S.generator == gen) & (S.placement == "context") & (S.H == h_main)]
        if g.empty:
            continue
        header = [KNOB_NAME[gen], "d"] + [f"{short(f)} T={t}" for f in sizes for t in Ts]
        rows = []
        for k in sorted(g.knob_value.unique()):
            for d in d_vals:
                r = [f"{k:g}", str(d)]
                for f in sizes:
                    for t in Ts:
                        c = cell(g, knob_value=k, d=d, forecaster=f, T=t)
                        mark = "" if c is None else ("" if c.switch_visible == 1 else "*")
                        r.append("" if c is None else f3(c.mase_ratio_oracle_gmean) + mark)
                rows.append(r)
        out.append(f"**{gen}** (a * marks cells where d > T: the switch is before the visible window)\n\n" + md(header, rows))
    g = S[(S.generator == "markov_ar") & (S.H == h_main)]
    if not g.empty:
        header = ["p_switch"] + [f"{short(f)} T={t}" for f in sizes for t in Ts]
        rows = [[f"{k:g}"] + [val(g, "mase_ratio_oracle_gmean", knob_value=k, forecaster=f, T=t) for f in sizes for t in Ts]
                for k in sorted(g.knob_value.unique())]
        out.append(f"**markov_ar** (MASE ratio to oracle, H = {h_main})\n\n" + md(header, rows))
    # H dependence
    Hs = sorted(S.H.unique())
    out.append(f"### MASE ratio to oracle by horizon, T = {t_main} (geometric mean over seeds)\n")
    for gen in GEN_ORDER:
        g = S[(S.generator == gen) & (S["T"] == t_main)]
        if g.empty:
            continue
        header = [KNOB_NAME[gen], "placement", "d"] + [f"{short(f)} H={h}" for f in sizes for h in Hs]
        rows = []
        keys = g[["knob_value", "placement", "d"]].drop_duplicates().sort_values(["placement", "d", "knob_value"], na_position="last")
        for _, kr in keys.iterrows():
            r = [f"{kr.knob_value:g}", kr.placement, "" if pd.isna(kr.d) else str(int(kr.d))]
            for f in sizes:
                for h in Hs:
                    r.append(val(g, "mase_ratio_oracle_gmean", knob_value=kr.knob_value, placement=kr.placement,
                                 d=None if pd.isna(kr.d) else kr.d, forecaster=f, H=h))
            rows.append(r)
        out.append(f"**{gen}**\n\n" + md(header, rows))
    # references
    out.append("### Reference rows (T, H as in the header)\n")
    cl = S[(S["T"] == t_main) & (S.H == h_main) & (~S.forecaster.str.startswith("timestral-"))]
    if not cl.empty:
        rows = []
        for gen in GEN_ORDER:
            for fc in sorted(cl.forecaster.unique()):
                x = cl[(cl.generator == gen) & (cl.forecaster == fc)]
                if len(x):
                    rows.append([gen, fc, f3(np.exp(np.log(x.mase_ratio_oracle_gmean.clip(lower=1e-9)).mean())), f3(x.cov80_mean.mean())])
        out.append("Classical models: mean over configs of the per-config geometric-mean MASE ratio to oracle, and mean cov80.\n\n"
                   + md(["generator", "forecaster", "mean of config gmeans (mase_ratio_oracle)", "mean cov80"], rows))
    orc = S[(S["T"] == t_main) & (S.forecaster == sizes[0])]
    out.append("Oracle cov80 (mean over configs, from the oracle rows of scores.csv), by H: "
               + ", ".join(f"H={h}: {orc[orc.H == h].oracle_cov80_mean.mean():.3f}" for h in Hs) + "\n")
    return "\n".join(out)


HEADER = """# Regime switches: failure map for Timestral 5M, 28M, 123M, 209M

Family `regime`. A base series that sits inside Timestral's pretraining prior (a level, a seasonal
term, AR(1) noise) with one thing that is not: a switch of regime. Question: does Timestral notice
a switch, and does it matter whether the switch is inside the context window (and how far back) or
happens in the forecast horizon?

## Contents

| file | what it is |
|---|---|
| `samples.npz` | the 880 generated samples (110 configs x 8 seeds, context 8192, horizon 1024, 500 oracle paths each). Not committed (207 MB, listed in `.gitignore`). md5 `{md5}` |
| `scores.csv` | long table, {nrows} rows: one row per sample x forecaster x T x H. Forecasters: `oracle`, `seasonal_naive`, `ets`, `arima`, `timestral-5M`, `timestral-28M`, `timestral-123M`, `timestral-209M`. Contexts T = 512, 2048, 8192; horizons H = 16, 128, 1024 |
| `provenance.json`, `colab_job/` | how `scores.csv` was produced (two runs on a Colab A100, concatenated; see `colab_job/README.md`) |
| `summary.csv` | {nsum} rows, grouped by `[config_id, generator, knob_value, forecaster, T, H]` plus explicit `placement` and `d` columns. For `mase_ratio_oracle`, `scrps_ratio_oracle`, `mase_ratio_classical`: geometric mean (`_gmean`), mean, std and 95% bootstrap interval (`_lo95`, `_hi95`) over the 8 seeds; `cov80_mean`; `oracle_cov80_mean`; `switch_visible` (placement context: 1 if d <= T) |
| `adoption.csv`, `adoption_by_seed.csv`, `adoption_provenance.json` | the level-adoption diagnostic (section at the end) |
| `tables.md` | the tables below, as generated |

The 123M and 209M runs (`regime_big` in the task spec) are covered by this folder's `scores.csv`: all four sizes were run on the full grid of 110 configs, T = 512 / 2048 / 8192, so there is no separate `regime_big` folder. The Colab run used the same `samples.npz`.

## Generators (`otp/generators/regime.py`, grid in `otp/grids/regime.py`)

Base series: `y_t = level + season_t + x_t`. `season_t` is a sum of 1 or 2 sinusoids at period 24 (amplitude drawn per seed in 0.8 to 1.5). `x_t` is Gaussian AR(1) noise, `x_t = rho x_(t-1) + sigma e_t`, with `rho` drawn per seed in 0.2 to 0.8 and the stationary std of `x` drawn per seed in 0.2 to 0.6. The AR noise starts from its stationary distribution. The simulator state carries the time index, the forecast origin and the current regime, so the oracle continues each of its 500 paths from the true regime.

| generator | what changes at the switch | knob (values in the grid) |
|---|---|---|
| `mean_jump` | the level jumps by `delta` x (stationary std of the noise) | `delta` = 0.5, 1, 2, 4, 8 |
| `variance_jump` | the AR innovation std is multiplied by `factor` | `factor` = 0.25, 0.5, 2, 4, 8 |
| `frequency_jump` | the seasonal period goes from 24 to 24 x `ratio` (phase continuous) | `ratio` = 0.5, 0.75, 1.5, 2, 3 |
| `markov_ar` | two AR(1) regimes; regime 0: mean 0, rho = per-seed value drawn in 0.2 to 0.8; regime 1: mean 2 x (regime-0 stationary std), rho 0.9, same sigma. Symmetric switching probability per step throughout context and horizon; the season is shared | `p_switch` = 0.001, 0.003, 0.01, 0.03, 0.1 |

Placement (for the first three generators):

- `context`: one scheduled switch, `d` steps before the forecast origin (the last `d` observations before the origin are post-switch), `d` = 16, 64, 256, 1024, 4096, 16384. No later switch, so the oracle knows the post-switch regime exactly. The samples are 8192 steps long: for d = 16384 the whole sample is post-switch (the switch is before the visible window at every T).
- `horizon`: no switch in the context; during the horizon the series switches with a constant hazard of 1/256 per step (once switched it stays switched). The oracle knows the hazard and both regimes but not the switch time; each oracle path draws its own switch time.

Config ids: `regime/<generator>/context/<knob><value>/d<d>`, `regime/<generator>/horizon/<knob><value>`, `regime/markov_ar/p<p_switch>`. The explicit `placement` (context, horizon, markov) and `d` columns of `summary.csv` repeat what the id encodes. For `frequency_jump` with placement context the seasonal period given to the classical models and to the MASE scale is the post-switch period (24 x ratio); with placement horizon it is 24.

## Reading the tables

- Every sample is one realized series; there are 8 samples (seeds) per config. Ratios are per-sample metric ratios, aggregated over the 8 seeds by geometric mean. The 95% bootstrap intervals are in `summary.csv`.
- `mase_ratio_oracle` = MASE of the forecaster / MASE of the true-dynamics oracle on the same sample, T and H. `scrps_ratio_oracle` is the same for scaled CRPS. `mase_ratio_classical` = MASE of the forecaster / MASE of the best of seasonal naive, ETS and ARIMA on that sample. MASE is the mean absolute error of the median over the horizon divided by the in-sample seasonal-naive error of the context the forecaster saw. A ratio of 1 means equal to the reference. Each sample has one realized future, so a forecaster can score below the oracle on a sample and geometric means just under 1 occur.
- cov80 is the fraction of horizon points inside the 10% to 90% quantile interval; 0.8 is calibrated.
- `d` and the context length `T` interact. The samples are generated once at T = 8192 and shorter contexts are the last T steps. A switch d steps before the origin is visible only if d <= T: with T = 512 a switch at d = 1024 is not in the window, and the forecaster sees a series that is entirely post-switch. The oracle conditions on the true state and does not depend on T. `summary.csv` can be sliced by both (`d`, `T`, `switch_visible`); the d-by-T tables below do this, with a * on cells where d > T.
- Classical models are run once per sample on the full 8192-step context (ETS and ARIMA fit its last 1024 steps; seasonal naive repeats the last season and takes its residual spread from the whole context). Their forecasts do not depend on T; only the MASE scale in their rows does.

## Regenerating

Samples (deterministic given the seeds; this is the command that produced `samples.npz`, with the scores that were then computed elsewhere):

```
python scripts/sweep.py --grid otp.grids.regime --sizes 5M 28M --contexts 512 2048 8192 --horizons 16 128 1024 --seeds 8 --out results/map/regime
```

The VM run of this command was stopped after `samples.npz` was written and before any scores were saved. `scores.csv` was produced on a Colab A100 from the same `samples.npz` by two runs (arguments as recorded in `colab_job/provenance_*.json`):

```
python scripts/sweep.py --grid otp.grids.regime --samples-from samples/regime --sizes --contexts 512 2048 8192 --horizons 16 128 1024 --workers 12 --out results/colab_out/regime_refs
python scripts/sweep.py --grid otp.grids.regime --samples-from samples/regime --no-references --sizes 5M 28M 123M 209M --contexts 512 2048 8192 --horizons 16 128 1024 --batch-size 32 --out results/colab_out/regime_big
```

(the first run: oracle, seasonal naive, ETS, ARIMA; the second: Timestral; wall time {wall_ref} s and {wall_ts} s in `colab_job/provenance_*.json`). Then:

```
python scripts/regime_summary.py --dir results/map/regime
python scripts/regime_adoption.py --dir results/map/regime --sizes 5M 28M 123M 209M --contexts 512 2048 8192
python scripts/regime_summary.py --dir results/map/regime --readme
```

Tests: `python3 -m pytest -q tests/test_regime.py`.

## Tables

All tables are generated from `summary.csv` by `scripts/regime_summary.py`. Geometric means over 8 seeds; T = 2048 and H = 128 unless the heading says otherwise.

"""


def adoption_tables(path: str) -> str:
    A = pd.read_csv(path)
    out = ["## Level adoption at H = 16 (placement context, mean_jump)\n",
           "Definition: `adoption = 1 + mean over the first 16 horizon steps of (Timestral median - oracle median) / shift`, with `shift = delta x (stationary noise std)`. "
           "1 means the median sits at the oracle's (post-switch) level; 0 means it sits where the oracle would be had the old level continued (oracle median minus shift). "
           "The oracle knows the AR noise state exactly and Timestral does not, so values around 1 carry noise of order (noise std) / shift per sample; the d = 16384 column (switch before the visible window at every T) is the no-switch-visible reference. "
           "Source: `adoption.csv` (mean over seeds; `adoption_by_seed.csv` has the per-sample values). "
           "`delta = all` pools the five deltas (40 samples per cell).\n"]
    sizes = sorted(dict.fromkeys(A["size"]), key=lambda z: int(z[:-1]))
    d_vals = sorted(A.d.unique())
    for t in sorted(A["T"].unique()):
        for stat in ("adoption_mean", "adoption_median"):
            rows = []
            for sz in sizes:
                x = A[(A["size"] == sz) & (A["T"] == t) & (A.delta.astype(str) == "all")]
                rows.append([sz] + [f3(x[x.d == d][stat].iloc[0]) if len(x[x.d == d]) else "" for d in d_vals])
            out.append(f"**T = {t}, {stat.replace('adoption_', '')} over seeds, delta = all**\n\n" + md(["size"] + [f"d={d}" for d in d_vals], rows))
    deltas = sorted(float(v) for v in A.delta.unique() if str(v) != "all")
    for t in (2048,):
        rows = []
        for sz in sizes:
            for dl in deltas:
                x = A[(A["size"] == sz) & (A["T"] == t) & (pd.to_numeric(A.delta, errors="coerce") == dl)]
                rows.append([sz, f"{dl:g}"] + [f3(x[x.d == d].adoption_mean.iloc[0]) if len(x[x.d == d]) else "" for d in d_vals])
        out.append(f"**T = {t}, mean over seeds, by delta**\n\n" + md(["size", "delta"] + [f"d={d}" for d in d_vals], rows))
    return "\n".join(out)


def write_readme(a, S, body):
    prov = read_json(os.path.join(a.dir, "provenance.json"))
    nrows = len(pd.read_csv(os.path.join(a.dir, "scores.csv")))
    head = HEADER.format(md5=prov["samples_md5"], nrows=nrows, nsum=len(S),
                         wall_ref=prov["runs"]["references"]["wall_sec"], wall_ts=prov["runs"]["timestral"]["wall_sec"])
    apath = os.path.join(a.dir, "adoption.csv")
    text = head + body + ("\n" + adoption_tables(apath) if os.path.exists(apath) else "")
    with open(os.path.join(a.dir, "README.md"), "w") as f:
        f.write(text)
    print("wrote README.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--readme", action="store_true", help="also write README.md (after adoption.csv exists)")
    a = ap.parse_args()
    S = build_summary(os.path.join(a.dir, "scores.csv"))
    S.to_csv(os.path.join(a.dir, "summary.csv"), index=False)
    print(f"wrote {len(S)} rows to {a.dir}/summary.csv")
    jobs = {n: read_json(os.path.join(a.dir, "colab_job", f"provenance_{n}.json")) for n in ("references", "timestral")
            if os.path.exists(os.path.join(a.dir, "colab_job", f"provenance_{n}.json"))}
    if jobs:  # scores.csv was assembled from separate runs; collect their provenance in one file
        write_json(os.path.join(a.dir, "provenance.json"),
                   {"note": "scores.csv is the concatenation of the runs below; see colab_job/README.md",
                    "samples_md5": "71cca8ecd1e7f1a73d04cf9059be930c", "runs": jobs})
    sizes = sorted([f for f in S.forecaster.unique() if f.startswith("timestral-")], key=lambda f: int(f.split("-")[1][:-1]))
    body = tables(S, sizes, 2048, 128)
    with open(os.path.join(a.dir, "tables.md"), "w") as f:
        f.write(body)
    print("wrote tables.md")
    if a.readme:
        write_readme(a, S, body)


if __name__ == "__main__":
    main()
