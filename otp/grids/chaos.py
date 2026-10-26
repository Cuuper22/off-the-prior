"""Chaos and dynamics grid: five generators, each swept over its difficulty knob.

`configs()` returns the full grid (the main sweep). `BIG_ONLY` lists the reduced subset used
for the 123M and 209M runs (three knob values per generator, from periodic to chaotic).
`param_list()` returns `(config_id, generator_class, params)` rows, which the Lyapunov table
in `otp.generators.chaos` also uses.

Also holds the post-processing used for every chaos results folder:

    python -m otp.grids.chaos summarize --dir results/map/chaos
    python -m otp.grids.chaos readme --dir results/map/chaos --grid otp.grids.chaos --command "..." --note "..."

`summarize` builds `summary.csv` from `scores.csv` with `otp.analysis.ratios` and
`otp.analysis.curve` grouped by `[config_id, generator, knob_value, forecaster, T, H]`
(`mase_ratio_oracle`, `scrps_ratio_oracle`, `mase_ratio_classical`, merged on the group
keys, plus the mean of `cov80` and the mean `mase`).
"""

from __future__ import annotations

from otp.generators.chaos import Kuramoto, LogisticMap, Lorenz63, MackeyGlass, Rossler

LORENZ_RHO = (10.0, 15.0, 20.0, 24.0, 26.0, 28.0, 35.0, 45.0)
LORENZ_OBS_EVERY = (2, 5, 10, 20)       # at rho = 28 (10 is the default, already in the rho sweep)
ROSSLER_C = (2.0, 3.0, 4.0, 4.5, 5.0, 5.7, 6.5, 9.0)
MG_TAU = (6.0, 10.0, 14.0, 17.0, 20.0, 25.0, 30.0)
LOGISTIC_R = (2.8, 3.2, 3.5, 3.56, 3.6, 3.7, 3.83, 3.9, 4.0)
KURAMOTO_K = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)


def _f(v: float) -> str:
    return f"{v:g}"


def param_list(full: bool = True):
    """Rows `(config_id, class, params)` for the main grid."""
    rows = []
    for rho in LORENZ_RHO:
        rows.append((f"lorenz63/rho{_f(rho)}", Lorenz63, {"rho": rho}))
    for oe in LORENZ_OBS_EVERY:
        if oe != Lorenz63.defaults["obs_every"]:
            rows.append((f"lorenz63/rho28_oe{oe}", Lorenz63, {"rho": 28.0, "obs_every": oe}))
    for c in ROSSLER_C:
        rows.append((f"rossler/c{_f(c)}", Rossler, {"c": c}))
    for tau in MG_TAU:
        rows.append((f"mackey_glass/tau{_f(tau)}", MackeyGlass, {"tau": tau}))
    for r in LOGISTIC_R:
        rows.append((f"logistic_map/r{_f(r)}", LogisticMap, {"r": r}))
    for k in KURAMOTO_K:
        rows.append((f"kuramoto/K{_f(k)}", Kuramoto, {"K": k}))
    return rows


# Reduced grid for the large models: about 3 knob values per generator, periodic -> chaotic.
BIG_ONLY = ["lorenz63/rho15", "lorenz63/rho26", "lorenz63/rho28", "lorenz63/rho28_oe20",
            "rossler/c3", "rossler/c5", "rossler/c5.7",
            "mackey_glass/tau10", "mackey_glass/tau17", "mackey_glass/tau30",
            "logistic_map/r3.2", "logistic_map/r3.6", "logistic_map/r4",
            "kuramoto/K0", "kuramoto/K2", "kuramoto/K8"]


def make_factory(cls, params):
    if cls.name == "kuramoto":
        return lambda seed: cls(**params, freq_seed=int(seed))
    return lambda seed: cls(**params)


def configs():
    return [(cid, make_factory(cls, params)) for cid, cls, params in param_list()]


# ---------------------------------------------------------------------------
# Post-processing: summary.csv and README.md for a results folder
# ---------------------------------------------------------------------------
BY = ["config_id", "generator", "knob_value", "forecaster", "T", "H"]
METRICS = ("mase_ratio_oracle", "scrps_ratio_oracle", "mase_ratio_classical")


def summarize(folder: str):
    """Build `<folder>/summary.csv` from `<folder>/scores.csv`."""
    import os

    import pandas as pd

    from otp.analysis import curve, ratios

    df = ratios(pd.read_csv(os.path.join(folder, "scores.csv")))
    out = None
    for m in METRICS:
        c = curve(df, BY, m)
        if out is not None:
            c = c.drop(columns=["n"])
        out = c if out is None else out.merge(c, on=BY, how="outer")
    extra = df.groupby(BY, dropna=False).agg(cov80_mean=("cov80", "mean"), mase_mean_raw=("mase", "mean")).reset_index()
    out = out.merge(extra, on=BY, how="left")
    out.to_csv(os.path.join(folder, "summary.csv"), index=False)
    return out


def _fmt(x) -> str:
    return "" if x != x else f"{x:.3f}"


def readme(folder: str, grid_name: str, commands: list[str], notes: list[str]):
    """Write `<folder>/README.md`: contents, regeneration commands and curve tables
    (every number is read from `summary.csv`)."""
    import os

    import pandas as pd

    from otp.io import read_json

    s = pd.read_csv(os.path.join(folder, "summary.csv"))
    prov = read_json(os.path.join(folder, "provenance.json"))
    name = os.path.basename(os.path.normpath(folder))
    forecasters = [f for f in s.forecaster.unique() if f.startswith("timestral-")]
    classical = [f for f in ("seasonal_naive", "ets", "arima") if f in set(s.forecaster)]
    sizes = sorted(forecasters, key=lambda f: float(f.split("-")[1][:-1]))
    Ts, Hs = sorted(s["T"].unique()), sorted(s["H"].unique())
    cfg_order = list(dict.fromkeys(s.config_id))
    lines = [f"# results/map/{name}", ""]
    lines += [f"Chaos and dynamics family, grid module `{grid_name}`. Files in this folder:", "",
              "- `scores.csv`: one row per sample x forecaster x context length T x horizon H (`oracle` is the "
              "true-dynamics oracle, a Monte Carlo continuation of the true process from its true state).",
              "- `provenance.json`: command line, git commit, host, threads, wall time.",
              "- `summary.csv`: per `[config_id, generator, knob_value, forecaster, T, H]`: `n` samples; for "
              "`mase_ratio_oracle`, `scrps_ratio_oracle` and `mase_ratio_classical` the geometric mean over seeds "
              "(`_gmean`), mean, std and 95% bootstrap interval; `cov80_mean` (share of horizon points inside the "
              "10%-90% interval, 0.8 is calibrated) and `mase_mean_raw`.",
              "- `samples.npz`: the generated samples (see the regeneration note below).", ""]
    lines += [f"Sweep: {prov.get('n_samples')} samples, {prov.get('n_rows')} score rows, wall time "
              f"{prov.get('wall_sec')} s (from `provenance.json`; threads {prov.get('threads')}, "
              f"git `{prov.get('git')}`).", ""]
    lines += ["## Regenerate", ""] + ["```", *commands, "```", ""]
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]

    def table(T, H, cols, title):
        sub = s[(s["T"] == T) & (s.H == H)]
        out = [f"### {title}", ""]
        head = ["config_id", "knob"] + [f"{c} ratio" for c in cols] + [f"{c} cov80" for c in cols] + ["oracle cov80"]
        out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for cid in cfg_order:
            g = sub[sub.config_id == cid]
            if g.empty:
                continue
            row = [cid, f"{g.knob_value.iloc[0]:g}"]
            for c in cols:
                r = g[g.forecaster == c]
                row.append(_fmt(r.mase_ratio_oracle_gmean.iloc[0]) if len(r) else "")
            for c in cols:
                r = g[g.forecaster == c]
                row.append(_fmt(r.cov80_mean.iloc[0]) if len(r) else "")
            r = g[g.forecaster == "oracle"]
            row.append(_fmt(r.cov80_mean.iloc[0]) if len(r) else "")
            out.append("| " + " | ".join(row) + " |")
        return out + [""]

    lines += ["## Main curves", "",
              "Ratio columns are `mase_ratio_oracle_gmean` (geometric mean over seeds of the forecaster's MASE "
              "divided by the oracle's MASE on the same sample; 1 means as good as the oracle) copied from "
              "`summary.csv`. `cov80` columns are `cov80_mean`.", ""]
    H0 = 128 if 128 in Hs else Hs[0]
    cols = sizes + classical
    lines += table(2048, H0, cols, f"T = 2048, H = {H0}")
    Hmax = max(Hs)
    if Hmax != H0:
        lines += table(2048, Hmax, cols, f"T = 2048, H = {Hmax}")
    if len(Ts) > 1:
        lines += ["### Context length: ratio to oracle by T", ""]
        first = sizes[0]
        for H in [H0]:
            head = ["config_id", "knob"] + [f"{first} T={t}" for t in Ts]
            lines += ["Forecaster " + first + f", H = {H}.", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            for cid in cfg_order:
                row = [cid, f"{s[s.config_id == cid].knob_value.iloc[0]:g}"]
                for t in Ts:
                    r = s[(s.config_id == cid) & (s.forecaster == first) & (s["T"] == t) & (s.H == H)]
                    row.append(_fmt(r.mase_ratio_oracle_gmean.iloc[0]) if len(r) else "")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    with open(os.path.join(folder, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summarize", "readme"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--grid", default="otp.grids.chaos")
    ap.add_argument("--command", action="append", default=[], help="regeneration command line (repeatable)")
    ap.add_argument("--note", action="append", default=[], help="fact to list under Notes (repeatable)")
    a = ap.parse_args()
    if a.cmd == "summarize":
        summarize(a.dir)
    else:
        readme(a.dir, a.grid, a.command, a.note)
