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
            header = [knob] + [f"{short(f)} d={d}" for f in sizes for d in d_vals] + [f"{short(f)} horizon" for f in sizes]
            rows = []
            for k in sorted(g.knob_value.unique()):
                r = [f"{k:g}"]
                for f in sizes:
                    r += [val(g, col, knob_value=k, forecaster=f, placement="context", d=d) for d in d_vals]
                r += [val(g, col, knob_value=k, forecaster=f, placement="horizon") for f in sizes]
                rows.append(r)
            out.append(f"**{gen}** (columns: size and d for placement context; then placement horizon)\n\n" + md(header, rows))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--no-readme", action="store_true")
    a = ap.parse_args()
    S = build_summary(os.path.join(a.dir, "scores.csv"))
    S.to_csv(os.path.join(a.dir, "summary.csv"), index=False)
    print(f"wrote {len(S)} rows to {a.dir}/summary.csv")
    if a.no_readme:
        return
    sizes = sorted([f for f in S.forecaster.unique() if f.startswith("timestral-")], key=lambda f: int(f.split("-")[1][:-1]))
    body = tables(S, sizes, 2048, 128)
    with open(os.path.join(a.dir, "tables.md"), "w") as f:
        f.write(body)
    print("wrote tables.md")


if __name__ == "__main__":
    main()
