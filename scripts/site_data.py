"""Export the data files the site (docs/) reads. Every value comes from a committed result file; each output
JSON carries a `sources` list naming the files it was built from.

    python scripts/site_data.py

Writes docs/data/*.json. Sections whose results are not in the repository yet (fine-tuning, search, neuron atlas)
get a file with "status": "pending" and the result path the site is waiting for; rerunning this script after
those results land fills them, with no change to the site code.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "data")
SIZES = ["timestral-5M", "timestral-28M", "timestral-123M", "timestral-209M"]


def p(*a):
    return os.path.join(ROOT, *a)


def r4(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else float(f"{x:.4g}")


def write(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(json.JSONEncoder(separators=(",", ":"), allow_nan=False).encode(obj))
    print(f"docs/data/{name}: {os.path.getsize(os.path.join(OUT, name)):,} bytes")


def pending(name, expect, what):
    write(name, {"status": "pending", "expect": expect, "what": what})


# ------------------------------------------------------------------------------------------------
def numbers():
    src = "paper/numbers.json"
    d = json.load(open(p(src)))
    out = {k: {"text": v["text"], "src": v["src"], "how": v["how"]} for k, v in d.items() if not k.startswith("table:") and "text" in v}
    write("numbers.json", {"sources": [src], "numbers": out})


def failure_map():
    rows, srcs = [], []
    for fam in ("chaos", "regime", "tails"):
        src = f"results/map/{fam}/summary_all_sizes.csv"
        s = pd.read_csv(p(src))
        srcs.append(src)
        for r in s.itertuples(index=False):
            rows.append([fam, r.generator, r.config_id, str(r.knob_value), r.forecaster.replace("timestral-", ""), int(r.T), int(r.H),
                         r4(r.mase_ratio_oracle_gmean), r4(r.mase_ratio_oracle_lo95), r4(r.mase_ratio_oracle_hi95),
                         r4(r.mase_ratio_classical_gmean), int(r.n)])
    src = "results/map/in_prior/summary.csv"
    s = pd.read_csv(p(src))
    srcs.append(src)
    for r in s.itertuples(index=False):
        rows.append(["in_prior", "in_prior", r.config_id, str(r.knob_value), r.forecaster.replace("timestral-", ""), int(r.T), int(r.H),
                     r4(r.mase_ratio_oracle_gmean), r4(getattr(r, "mase_ratio_oracle_lo95", np.nan)), r4(getattr(r, "mase_ratio_oracle_hi95", np.nan)),
                     r4(getattr(r, "mase_ratio_classical_gmean", np.nan)), int(r.n)])
    cols = ["family", "generator", "config_id", "knob_value", "size", "T", "H", "ratio_oracle", "lo95", "hi95", "ratio_classical", "n"]
    write("failure_map.json", {"sources": srcs, "columns": cols, "rows": rows,
                               "note": "ratio_oracle: geometric mean over seeds of MASE(Timestral) / MASE(oracle); ratio_classical: same against the best of seasonal naive, ETS, ARIMA"})


def lorenz():
    src = "results/map/chaos/summary_all_sizes.csv"
    lsrc = "results/map/chaos/lyapunov.csv"
    s = pd.read_csv(p(src))
    lam = float(pd.read_csv(p(lsrc)).query("config_id == 'lorenz63/rho28'").lyap_per_time.mean())
    dt = {"lorenz63/rho28_oe2": 0.02, "lorenz63/rho28_oe5": 0.05, "lorenz63/rho28": 0.1, "lorenz63/rho28_oe20": 0.2}
    q = s[(s["T"] == 2048) & s.config_id.isin(dt)].copy()
    q["reach"] = q.config_id.map(dt) * q.H * lam
    series = {fc.replace("timestral-", ""): [[r4(a), r4(b)] for a, b in q[q.forecaster == fc].sort_values("reach")[["reach", "mase_ratio_oracle_gmean"]].values]
              for fc in SIZES}
    write("lorenz.json", {"sources": [src, lsrc], "lyapunov_exponent": r4(lam), "series": series,
                          "x": "forecast reach in Lyapunov times", "y": "MASE / oracle MASE"})


def noise():
    src = "results/mechanism/chaos_noise/summary.csv"
    s = pd.read_csv(p(src))
    s["system"] = s.config_id.str.split("/").str[1]
    s["noise"] = s.config_id.str.extract(r"/n([\d.]+)$")[0].astype(float) * 100
    q = s[s.H == 128]
    out = {}
    for k in sorted(q.system.unique()):
        g = q[q.system == k]
        out[k] = {fc.replace("timestral-", ""): [[r4(a), r4(b)] for a, b in g[g.forecaster == fc].sort_values("noise")[["noise", "mase_ratio_oracle_gmean"]].values]
                  for fc in SIZES}
    write("noise.json", {"sources": [src], "systems": out, "x": "noise, % of signal scale", "y": "MASE / oracle MASE (H = 128)"})


def readouts():
    src = "results/mechanism/readouts/summary.csv"
    s = pd.read_csv(p(src))
    g = s[(s.study == "mean_jump_d") & s.valid]
    out = {}
    for size in sorted(g["size"].unique()):
        h = g[g["size"] == size].sort_values("knob_value")
        out[size] = {"steps": [r4(x) for x in h.knob_value], "readout_r2": [r4(x) for x in h.best_readout_r2], "forecast_r2": [r4(x) for x in h.output_r2]}
    write("readouts.json", {"sources": [src], "mean_jump": out})


def covariates():
    src = "results/covariates/summary.csv"
    s = pd.read_csv(p(src))
    s = s[(s["T"] == 2048) & (s.condition != "classical_best")]
    keep = ["system", "knob", "knob_value", "size", "H", "condition", "n", "covariate_gain_gmean", "oracle_gain_gmean",
            "joint_vs_past_gmean", "mase_ratio_oracle_gmean"]
    rows = [[r.system, r.knob, r4(r.knob_value), r.size, int(r.H), r.condition, int(r.n), r4(r.covariate_gain_gmean), r4(r.oracle_gain_gmean),
             r4(r.joint_vs_past_gmean), r4(r.mase_ratio_oracle_gmean)] for r in s[keep].itertuples(index=False)]
    write("covariates.json", {"sources": [src], "columns": ["system", "knob", "knob_value", "size", "H", "condition", "n", "covariate_gain",
                                                            "oracle_gain", "joint_over_past", "ratio_oracle"], "rows": rows})


def realdata():
    src = "results/regression_baseline/analogue_vs_synthetic.csv"
    d = pd.read_csv(p(src))
    d = d[(d.level == "family") & (d.metric == "mase")]
    rows = [[r.family_analogue, r.forecaster.replace("timestral-", ""), int(r.n_windows), r4(r.real_ratio_to_best_classical_gmean),
             r4(r.synthetic_T2048_H128)] for r in d.itertuples(index=False)]
    s2 = "results/regression_baseline/summary.csv"
    a = pd.read_csv(p(s2))
    a = a[(a.group == "ALL") & (a.basis == "common")]
    sets = [[r.set, r.forecaster, int(r.n), r4(r.mase_rel_snaive), r4(r.scrps_rel_snaive)] for r in a.itertuples(index=False)]
    write("realdata.json", {"sources": [src, s2], "analogue_columns": ["family", "size", "windows", "real_ratio_best_classical", "synthetic_ratio_best_classical"],
                            "analogues": rows, "set_columns": ["set", "forecaster", "windows", "mase_rel_snaive", "scrps_rel_snaive"], "sets": sets})


def selfval():
    src = "results/fix/selfval_confirm/frozen_rule_confirmation.csv"
    r = pd.read_csv(p(src))
    rows = [[x.family, x.size, int(x.n), r4(x.mase_fix_over_timestral), r4(x.switched_share)] for x in r.itertuples(index=False)]
    write("selfval.json", {"sources": [src], "columns": ["family", "size", "samples", "mase_fix_over_timestral", "share_switched"], "rows": rows,
                           "rule": json.load(open(p("results/fix/selfval_confirm/frozen_rule.json")))})


def bench():
    srcs, rows, ft = [], [], {}
    for set_name in ("diag", "lb"):
        for f in sorted(glob.glob(p("bench", "leaderboard", f"*_{set_name}_all", "summary.json"))):
            d = json.load(open(f))
            rel = os.path.relpath(f, ROOT)
            srcs.append(rel)
            o = d["overall"]
            rows.append([set_name, d["model"], int(o["n"]), r4(o["mase_rel_snaive"]), r4(o["scrps_rel_snaive"]), r4(o["mase_rel_oracle"]), r4(o["cov80"])])
            if set_name == "diag":
                ft[d["model"]] = {k: r4(v["mase_rel_oracle"]) for k, v in d["by_failure_type"].items()}
    write("bench.json", {"sources": srcs, "columns": ["set", "model", "series", "mase_rel_snaive", "scrps_rel_snaive", "mase_rel_oracle", "cov80"],
                         "rows": rows, "diag_by_failure_type_mase_rel_oracle": ft})


# ------------------------------------------------------------------------------------------------ slots
def finetune():
    found = sorted(glob.glob(p("results", "finetune", "*", "summary.csv")))
    if not found:
        return pending("finetune.json", "results/finetune/<local|colab>/summary.csv", "fine-tuning gain and cost per method and protocol")
    out = {"sources": [], "runs": {}}
    for f in found:
        rel = os.path.relpath(f, ROOT)
        s = pd.read_csv(f)
        out["sources"].append(rel)
        out["runs"][rel.split("/")[2]] = {"columns": list(s.columns), "rows": json.loads(s.to_json(orient="values", double_precision=4))}
    out["status"] = "ready"
    write("finetune.json", out)


def search():
    f = p("results", "search", "comparison.csv")
    if not os.path.exists(f):
        return pending("search.json", "results/search/comparison.csv", "CMA-ES vs MAP-Elites vs gradient search at equal budget")
    s = pd.read_csv(f)
    out = {"status": "ready", "sources": ["results/search/comparison.csv"], "columns": list(s.columns),
           "rows": json.loads(s.to_json(orient="values", double_precision=4))}
    t = p("results", "search", "transfer.csv")
    if os.path.exists(t):
        tt = pd.read_csv(t)
        out["sources"].append("results/search/transfer.csv")
        out["transfer"] = {"columns": list(tt.columns), "rows": json.loads(tt.to_json(orient="values", double_precision=4))}
    write("search.json", out)


def neurons():
    files = sorted(glob.glob(p("results", "neurons", "browser", "*.json")))
    if not files:
        return pending("neurons.json", "results/neurons/browser/*.json", "neuron atlas: per-size pole labels, scores and exemplars")
    out = {"status": "ready", "sources": [os.path.relpath(f, ROOT) for f in files],
           "files": [os.path.relpath(f, os.path.join(ROOT, "docs")) for f in files]}
    c = p("results", "neurons", "one_job_curves.csv")
    if os.path.exists(c):
        cc = pd.read_csv(c)
        out["sources"].append("results/neurons/one_job_curves.csv")
        out["one_job_curves"] = {"columns": list(cc.columns), "rows": json.loads(cc.to_json(orient="values", double_precision=4))}
    write("neurons.json", out)


def main():
    for fn in (numbers, failure_map, lorenz, noise, readouts, covariates, realdata, selfval, bench, finetune, search, neurons):
        fn()


if __name__ == "__main__":
    main()
