"""Build the writeup from committed result files.

paper/src.md holds the text with placeholders:
  {{num:KEY}}    a number computed below from a result file
  {{table:KEY}}  a markdown table computed below from a result file
This script fills them, writes paper/off_the_prior.md, and writes paper/numbers.json: for every
key, the value, the source file(s) and a one-line description of how it was computed. Any
placeholder without a definition stops the build, so no number can be typed in by hand.

python paper/build.py
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from otp.analysis import ratios  # noqa: E402

NUMS: dict[str, dict] = {}
TABLES: dict[str, dict] = {}
SIZES = ["timestral-5M", "timestral-28M", "timestral-123M", "timestral-209M"]
SIZE_LABEL = {s: s.replace("timestral-", "") for s in SIZES}


def p(*parts):
    return os.path.join(ROOT, *parts)


def gm(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if x.size else float("nan")


def num(key, value, src, how, fmt="{:.2f}"):
    NUMS[key] = {"value": value, "text": fmt.format(value), "src": src, "how": how}


def table(key, df: pd.DataFrame, src, how, floatfmt="{:.2f}"):
    def f(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("–" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f(v) for v in r.values) + " |")
    TABLES[key] = {"text": "\n".join(lines), "src": src, "how": how}


def load_summary(path):
    s = pd.read_csv(p(path))
    s["knob_value"] = s["knob_value"].astype(str)
    return s


# --------------------------------------------------------------------------------------------
def in_prior():
    src = "results/map/in_prior/summary.csv"
    s = load_summary(src)
    g = s.groupby(["forecaster", "T", "H"]).mase_ratio_oracle_gmean.apply(gm).reset_index()
    short = g[g.H <= 128]
    num("inprior_short_min", short.mase_ratio_oracle_gmean.min(), src, "min over sizes, T, H<=128 of the geometric mean over configs of the per-config geometric-mean MASE ratio to oracle")
    num("inprior_short_max", short.mase_ratio_oracle_gmean.max(), src, "max of the same")
    rows = []
    for fc in SIZES:
        r = {"size": SIZE_LABEL[fc]}
        for h in (16, 128, 1024, 2048):
            v = g[(g.forecaster == fc) & (g["T"] == 2048) & (g.H == h)].mase_ratio_oracle_gmean
            r[f"H={h}"] = float(v.iloc[0]) if len(v) else float("nan")
        rows.append(r)
    table("inprior", pd.DataFrame(rows), src, "geometric mean over the 4 in-prior configs of the per-config geometric-mean MASE ratio to oracle, context 2048")


def family_table(src, key, configs, T=2048, H=128, label=None):
    s = load_summary(src)
    q = s[(s["T"] == T) & (s.H == H) & s.config_id.isin(configs)]
    rows = []
    for cid in configs:
        r = {"series": label(cid) if label else cid}
        for fc in SIZES:
            v = q[(q.config_id == cid) & (q.forecaster == fc)].mase_ratio_oracle_gmean
            r[SIZE_LABEL[fc]] = float(v.iloc[0]) if len(v) else float("nan")
        v = q[(q.config_id == cid) & (q.forecaster == "timestral-209M")].mase_ratio_classical_gmean
        r["209M vs best classical"] = float(v.iloc[0]) if len(v) else float("nan")
        rows.append(r)
    table(key, pd.DataFrame(rows), src, f"per-config geometric mean over seeds of MASE ratio to oracle (and to the best classical model), context {T}, horizon {H}")
    return q


def chaos():
    src = "results/map/chaos/summary_all_sizes.csv"
    cfg = ["lorenz63/rho20", "lorenz63/rho28", "lorenz63/rho28_oe2", "rossler/c3", "rossler/c5.7", "rossler/c9",
           "mackey_glass/tau10", "mackey_glass/tau17", "mackey_glass/tau30", "logistic_map/r3.5", "logistic_map/r3.9",
           "kuramoto/K1", "kuramoto/K8"]
    q = family_table(src, "chaos_main", cfg)
    v = q[q.forecaster == "timestral-209M"].mase_ratio_classical_gmean
    num("chaos_209m_vs_classical_median", float(np.median(v)), src, "median over chaos configs of the 209M geometric-mean MASE ratio to the best classical model, T=2048, H=128")
    for cid, k in [("mackey_glass/tau30", "mg30"), ("kuramoto/K1", "kur1"), ("kuramoto/K8", "kur8"), ("rossler/c9", "ros9"),
                   ("lorenz63/rho28_oe2", "lor_oe2")]:
        for fc in ("timestral-5M", "timestral-209M"):
            r = q[(q.config_id == cid) & (q.forecaster == fc)]
            num(f"chaos_{k}_{SIZE_LABEL[fc]}", float(r.mase_ratio_oracle_gmean.iloc[0]), src, f"{cid}, {fc}, T=2048, H=128, geometric-mean MASE ratio to oracle")
    # Lyapunov-time view for Lorenz rho=28 at four sampling intervals
    s = load_summary(src)
    dt = {"lorenz63/rho28_oe2": 0.02, "lorenz63/rho28_oe5": 0.05, "lorenz63/rho28": 0.1, "lorenz63/rho28_oe20": 0.2}
    lyap = pd.read_csv(p("results/map/chaos/lyapunov.csv"))
    lam = float(lyap[lyap.config_id == "lorenz63/rho28"].lyap_per_time.mean())
    num("lorenz_lyap", lam, "results/map/chaos/lyapunov.csv", "largest Lyapunov exponent per time unit, Lorenz rho=28, mean over seeds", "{:.3f}")
    r = s[(s["T"] == 2048) & (s.forecaster == "timestral-209M") & s.config_id.isin(dt)].copy()
    r["lyapunov_times"] = r.config_id.map(dt) * r.H.astype(float) * lam
    r = r.sort_values("lyapunov_times")
    out = pd.DataFrame({"sampling step (time units)": r.config_id.map(dt).values, "horizon (steps)": r.H.astype(int).values,
                        "horizon (Lyapunov times)": r.lyapunov_times.values, "209M MASE / oracle": r.mase_ratio_oracle_gmean.values})
    table("lorenz_lyap", out, src + " + results/map/chaos/lyapunov.csv", "209M, context 2048; Lyapunov times = sampling step x horizon x exponent")


def tails():
    src = "results/map/tails/summary_all_sizes.csv"
    s = load_summary(src)
    q = s[(s["T"] == 2048) & (s.H == 128)]
    num("tails_min", q.mase_ratio_oracle_gmean.min(), src, "min over all tails configs and sizes, T=2048, H=128")
    num("tails_max", q.mase_ratio_oracle_gmean.max(), src, "max of the same")
    cfg = ["tails/stable_ar/alpha2_rho0.9", "tails/stable_ar/alpha1.3_rho0.9", "tails/stable_ar/alpha1.3_rho1.0",
           "tails/fbm/H0.2", "tails/fbm/H0.8", "tails/hawkes/eta0.3", "tails/hawkes/eta0.95", "tails/msm/k8_ret", "tails/msm/k8_cum"]
    family_table(src, "tails_main", cfg)


def long_ctx():
    src = "results/map/long_ctx/scores.csv"
    df = ratios(pd.read_csv(p(src)))
    ts = df[df.forecaster.isin(SIZES) & (df.H == 128)]
    g = ts.groupby(["config_id", "forecaster", "T"]).mase_ratio_oracle.apply(gm).reset_index()
    rows = []
    for cid in ["mackey_glass/tau17", "kuramoto/K1", "rossler/c5.7", "lorenz63/rho28", "regime/mean_jump/context/delta4/d64",
                "tails/fbm/H0.8", "in_prior/p24"]:
        for fc in ("timestral-5M", "timestral-209M"):
            r = {"series": cid, "size": SIZE_LABEL[fc]}
            for t in (2048, 8192, 32768):
                v = g[(g.config_id == cid) & (g.forecaster == fc) & (g["T"] == t)].mase_ratio_oracle
                r[f"T={t}"] = float(v.iloc[0]) if len(v) else float("nan")
            rows.append(r)
    table("long_ctx", pd.DataFrame(rows), src, "geometric mean over 8 seeds of MASE ratio to oracle at H=128 (oracle rows matched per context length)")
    for cid, k in [("mackey_glass/tau17", "mg17")]:
        for fc in ("timestral-5M", "timestral-209M"):
            for t in (2048, 32768):
                v = float(g[(g.config_id == cid) & (g.forecaster == fc) & (g["T"] == t)].mase_ratio_oracle.iloc[0])
                num(f"long_{k}_{SIZE_LABEL[fc]}_T{t}", v, src, f"{cid}, {fc}, T={t}, H=128")
    bad = df[(df.forecaster == "oracle") & ~np.isfinite(df.mase)].sample_id.nunique()
    num("long_nonfinite_oracle_samples", bad, src, "samples with a non-finite oracle (diverged Monte Carlo paths)", "{:d}")


def symavg():
    src = "results/mechanism/symavg/off_over_on.csv"
    w = pd.read_csv(p(src))
    q = w[w.H == 128]
    num("symavg_min", q.scrps_off_over_on.min(), src, "min over configs and sizes of sCRPS ratio (averaging off / on), H=128", "{:.3f}")
    num("symavg_max", q.scrps_off_over_on.max(), src, "max of the same", "{:.3f}")


def normalization():
    src = "results/mechanism/normalization/summary.csv"
    s = pd.read_csv(p(src))
    full = s[s.condition == "full"].set_index(["case", "T", "size"])
    post = s[s.condition == "post_only"].set_index(["case", "T", "size"])
    j = full.join(post, lsuffix="_full", rsuffix="_post")
    num("norm_cells", len(j), src, "case x context x size cells with both conditions", "{:d}")
    num("norm_post_better", int((j.scrps_ratio_gmean_post < j.scrps_ratio_gmean_full).sum()), src, "cells where cutting to the post-switch part lowered the sCRPS ratio", "{:d}")
    v = s[(s.case == "variance_jump/f0.25/d64") & (s.condition == "full")]
    num("norm_vdrop_cov80_min", v.cov80.min(), src, "variance drop x0.25, 64 steps back, full context: min cov80 over sizes and contexts", "{:.3f}")
    num("norm_vdrop_scrps_min", v.scrps_ratio_gmean.min(), src, "same cells: min sCRPS ratio to oracle")
    num("norm_vdrop_scrps_max", v.scrps_ratio_gmean.max(), src, "same cells: max sCRPS ratio to oracle")


def analog():
    src = "results/mechanism/analog/summary.csv"
    s = pd.read_csv(p(src))
    cfg = ["mackey_glass/tau17", "mackey_glass/tau30", "rossler/c5.7", "rossler/c9", "kuramoto/K1", "kuramoto/K8",
           "lorenz63/rho28", "logistic_map/r3.9"]
    cols = [c for c in ["mase_oracle", "mase_analog", "mase_template", "mase_5M", "mase_28M", "mase_123M", "mase_209M"] if c in s]
    out = s[s.config_id.isin(cfg)].set_index("config_id").loc[cfg, cols].reset_index()
    out.columns = ["series"] + [c.replace("mase_", "") for c in cols]
    table("analog", out, src, "geometric mean over 8 seeds of MASE (context 2048, horizon 128) for the oracle, the method of analogues, the seasonal template, and Timestral")


def main():
    for fn in (in_prior, chaos, tails, long_ctx, symavg, normalization, analog):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e}")
    src = open(p("paper", "src.md")).read()
    missing = []

    def sub(m):
        kind, key = m.group(1), m.group(2)
        d = (NUMS if kind == "num" else TABLES).get(key)
        if d is None:
            missing.append(m.group(0))
            return m.group(0)
        return d["text"]

    out = re.sub(r"\{\{(num|table):([A-Za-z0-9_]+)\}\}", sub, src)
    if missing:
        raise SystemExit(f"undefined placeholders: {sorted(set(missing))}")
    open(p("paper", "off_the_prior.md"), "w").write(out)
    audit = {k: {kk: (vv if kk != "value" else float(vv)) for kk, vv in v.items()} for k, v in NUMS.items()}
    audit.update({f"table:{k}": {"src": v["src"], "how": v["how"]} for k, v in TABLES.items()})
    with open(p("paper", "numbers.json"), "w") as f:
        f.write(json.JSONEncoder(indent=1).encode(audit) + "\n")
    print(f"wrote paper/off_the_prior.md ({len(NUMS)} numbers, {len(TABLES)} tables)")


if __name__ == "__main__":
    main()
