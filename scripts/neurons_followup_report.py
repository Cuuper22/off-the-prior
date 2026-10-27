"""Tables and README sections of the neuron-atlas follow-up (parts A and C).

    python scripts/neurons_followup_report.py --a-runs results/colab_out --steering results/neurons/steering_results.csv \
        --out results/neurons/followup

Reads `<a-runs>/nn2_a_<size>/{scores_partial.csv,readouts.csv,meta.json}` (scripts/neurons_followup_a.py) and the
single-pole steering results of the atlas; writes `direction_readouts.csv`, `direction_steering.csv`,
`group_steering.csv`, `group_summary.csv`, `mde.csv`, `single_pole_calibrated.csv`, and the sections A and C of
`README.md` (everything after the line `<!-- follow-up A and C -->`) from those files and from `motif_*.csv`.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from otp.io import provenance, write_json  # noqa: E402
from otp.neurons import steer as St  # noqa: E402

SIZES = ("5M", "28M", "123M", "209M")
MARK = "<!-- follow-up A and C -->"
THRESH = St.THRESH


def p95_abs(x) -> float:
    x = np.abs(np.asarray(x, float))
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 95)) if len(x) else np.nan


def md(df: pd.DataFrame, cols=None, fmt="{:.4g}") -> str:
    cols = list(cols or df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and not np.isfinite(v)) else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)) for v in r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
def load_a(root: str):
    parts, readouts, metas = [], [], {}
    for s in SIZES:
        d = os.path.join(root, f"nn2_a_{s}")
        f = os.path.join(d, "scores_partial.csv")
        if not os.path.exists(f):
            continue
        parts.append(pd.read_csv(f))
        readouts.append(pd.read_csv(os.path.join(d, "readouts.csv")))
        mf = os.path.join(d, "meta.json")
        if os.path.exists(mf):
            metas[s] = json.load(open(mf))
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), (pd.concat(readouts, ignore_index=True) if readouts else pd.DataFrame()), metas


def direction_tables(d: pd.DataFrame) -> pd.DataFrame:
    dd = d[d.family == "dir"].copy()
    key = ["size", "variable", "kind", "block_name", "site"]
    rnd = dd[dd.role == "random"]
    p95 = rnd.groupby(key).effect_size.apply(p95_abs).rename("p95_random")
    nrand = rnd.groupby(key).size().rename("n_random")
    rmatch = rnd.groupby(key).apply(lambda g: pd.Series({
        "random_orig_match_rate": float(g["match"].astype(bool).mean()),
        "random_cal_match_rate": float((g["monotone"].astype(bool) & g["sig"].astype(bool) & (g.effect_size.abs() > p95_abs(g.effect_size))).mean())}), include_groups=False)
    real = dd[dd.role == "real"].merge(p95, on=key, how="left").merge(nrand, on=key, how="left").merge(rmatch, on=key, how="left")
    real["orig_match"] = real["match"].astype(bool)
    real["big_cal"] = real.effect_size.abs() > real.p95_random
    real["cal_match"] = real.monotone.astype(bool) & real.sig.astype(bool) & real.big_cal
    real["threshold"] = real.predicted.map(THRESH)
    return real


def group_tables(d: pd.DataFrame):
    g = d[d.family == "group"].copy()
    if g.empty:
        return g, pd.DataFrame()
    key = ["size", "k", "predicted"]
    rnd = g[g.role == "random"]
    p95 = rnd.groupby(key).effect_size.apply(p95_abs).rename("p95_random_pooled")
    nr = rnd.groupby(key).size().rename("n_random_pooled")
    lab = g[g.role == "labeled"].merge(p95, on=key, how="left").merge(nr, on=key, how="left")
    ranks = []
    rr = rnd.set_index("gid")
    for r in lab.itertuples():
        x = rr.loc[[r.gid]].effect_size.abs().to_numpy() if r.gid in rr.index else np.array([])
        ranks.append(float(((x < abs(r.effect_size)).sum() + 0.5 * (x == abs(r.effect_size)).sum()) / len(x)) if len(x) else np.nan)
    lab["rank_among_random"] = ranks
    lab["orig_match"] = lab["match"].astype(bool)
    lab["big_cal"] = lab.effect_size.abs() > lab.p95_random_pooled
    lab["cal_match"] = lab.monotone.astype(bool) & lab.sig.astype(bool) & lab.big_cal
    rnd = rnd.merge(p95, on=key, how="left")
    rnd["orig_match"] = rnd["match"].astype(bool)
    rnd["cal_match"] = rnd.monotone.astype(bool) & rnd.sig.astype(bool) & (rnd.effect_size.abs() > rnd.p95_random_pooled)
    rows = []
    for (s, pool, k), x in lab.groupby(["size", "pool", "k"]):
        r = rnd[(rnd["size"] == s) & (rnd.pool == pool) & (rnd.k == k)]
        rows.append({"size": s, "pool": pool, "k": k, "n_groups": len(x), "n_random": len(r),
                     "labeled_orig_match": float(x.orig_match.mean()), "random_orig_match": float(r.orig_match.mean()) if len(r) else np.nan,
                     "labeled_cal_match": float(x.cal_match.mean()), "random_cal_match": float(r.cal_match.mean()) if len(r) else np.nan,
                     "labeled_abs_D_median": float(x.effect_size.abs().median()), "labeled_abs_D_p75": float(x.effect_size.abs().quantile(.75)),
                     "labeled_abs_D_p95": p95_abs(x.effect_size),
                     "random_abs_D_median": float(r.effect_size.abs().median()) if len(r) else np.nan,
                     "random_abs_D_p75": float(r.effect_size.abs().quantile(.75)) if len(r) else np.nan, "random_abs_D_p95": p95_abs(r.effect_size),
                     "mean_rank_among_random": float(x.rank_among_random.mean()), "share_rank_ge_0.95": float((x.rank_among_random >= 0.95).mean())})
    return lab.drop(columns=["match"]), pd.DataFrame(rows)


def single_pole_tables(steer: pd.DataFrame):
    s = steer.copy()
    ctrl = s[s.role == "control"]
    p95 = ctrl.groupby(["size", "quantity"]).effect_size.apply(p95_abs).rename("p95_control").reset_index()
    lab = s[(s.role == "labeled") & s.quantity.isin(["level", "slope", "spread"])].merge(p95, on=["size", "quantity"], how="left")
    lab["big_cal"] = lab.effect_size.abs() > lab.p95_control
    lab["cal_match"] = lab.monotone.astype(bool) & lab["sig"].astype(bool) & lab.big_cal
    c2 = ctrl.merge(p95, on=["size", "quantity"], how="left")
    c2["cal_match"] = c2.monotone.astype(bool) & c2["sig"].astype(bool) & (c2.effect_size.abs() > c2.p95_control)
    rows = []
    for (sz, q), x in lab.groupby(["size", "quantity"]):
        c = c2[(c2["size"] == sz) & (c2.quantity == q)]
        rows.append({"size": sz, "quantity": q, "n_labeled": len(x), "labeled_orig_match": float(x["match"].astype(bool).mean()),
                     "labeled_cal_match": float(x.cal_match.mean()), "n_control": len(c), "control_orig_match": float(c["match"].astype(bool).mean()),
                     "control_cal_match": float(c.cal_match.mean()), "p95_control_abs_D": float(x.p95_control.iloc[0]), "threshold": THRESH[q],
                     "max_labeled_abs_D": float(x.effect_size.abs().max()), "median_labeled_abs_D": float(x.effect_size.abs().median()),
                     "labeled_abs_D_above_p95": float((x.effect_size.abs() > x.p95_control).mean())})
    return pd.DataFrame(rows), p95


def mde_table(real: pd.DataFrame, glab: pd.DataFrame, d: pd.DataFrame, p95_pole: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in p95_pole.itertuples():
        rows.append({"size": r.size, "object": "pole", "setting": "k=1", "quantity": r.quantity, "p95_control_abs_D": r.p95_control,
                     "threshold": THRESH[r.quantity], "ratio_p95_to_threshold": r.p95_control / THRESH[r.quantity]})
    if len(glab):
        for (s, k, q), x in glab.groupby(["size", "k", "predicted"]):
            qq = q.split("|")[0]
            rows.append({"size": s, "object": "group", "setting": f"k={k}", "quantity": q, "p95_control_abs_D": float(x.p95_random_pooled.iloc[0]),
                         "threshold": THRESH[qq], "ratio_p95_to_threshold": float(x.p95_random_pooled.iloc[0]) / THRESH[qq]})
    for r in real.itertuples():
        if r.test_set in ("mixed", "jump"):
            rows.append({"size": r.size, "object": "direction", "setting": f"{r.variable}/{r.kind}/{r.block_name}/{r.site}", "quantity": r.predicted,
                         "p95_control_abs_D": r.p95_random, "threshold": r.threshold, "ratio_p95_to_threshold": r.p95_random / r.threshold})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
def section_a(readouts, real, glab, gsum, single, metas) -> str:
    L = ["## A. Steering power (pre-registration: `../steering_preregistration_followup.md`)", ""]
    if real.empty and glab.empty:
        return "\n".join(L + ["No run files found."])
    L += ["Quantities, scales, test series and the matching criterion are those of the original steering. The calibrated criterion "
          "counts an effect as big when `|D|` (D = effect at the top scale minus effect at scale 0) exceeds the 95th percentile of `|D|` over the "
          "matched controls; it also requires the monotone and interval conditions. Files: `direction_readouts.csv`, `direction_steering.csv`, "
          "`group_steering.csv`, `group_summary.csv`, `mde.csv`, `single_pole_calibrated.csv`.", ""]
    if not readouts.empty:
        L += ["### A1. Readouts of known variables from the residual stream (`direction_readouts.csv`)", "",
              "Held-out R^2 of the ridge readout at the output of the early / middle / late block, by token set (`pooled`: context and forecast tokens).", "",
              md(readouts.pivot_table(index=["size", "variable", "block_name"], columns="tokens", values="r2_heldout").reset_index()), ""]
    if not real.empty:
        r = real[real.test_set.isin(["mixed", "jump"])]
        L += ["### A1. Positive control: direction steering (`direction_steering.csv`)", "",
              "`effect_size` = D at move 2 on the predicted quantity (`level` for the level and post-jump directions, `spread` for the noise direction), in "
              "context standard deviations (spread: log width ratio); `gain` = D / 2; `p95_random` = 95th percentile of `|D|` over the random directions of equal norm "
              "at the same size, variable, kind, block and site; `orig_match` and `cal_match` are the two criteria.", ""]
        verdict = []
        for s, x in r[r.variable == "level"].groupby("size"):
            verdict.append({"size": s, "cells": len(x), "orig_match_cells": int(x.orig_match.sum()), "cal_match_cells": int(x.cal_match.sum()),
                            "both_cells": int((x.orig_match & x.cal_match).sum()), "max_gain": float(x.gain.max()), "max_abs_D": float(x.effect_size.abs().max()),
                            "verdict": "detects a known effect" if (x.orig_match & x.cal_match).any() else "insensitive"})
        L += ["Verdict rule (plain level direction, all blocks, sites, direction kinds, mixed test set):", "", md(pd.DataFrame(verdict)), ""]
        cols = ["size", "variable", "kind", "block_name", "site", "readout_r2_heldout", "gain", "effect_size", "p95_random", "threshold", "rho", "orig_match", "cal_match",
                "random_orig_match_rate", "random_cal_match_rate", "E2_level", "E2_spread"]
        for var in ("level", "noise", "postjump"):
            x = r[(r.variable == var) & (r.kind == "pooled")]
            if len(x):
                L += [f"Variable `{var}`, pooled direction, all cells:", "", md(x.sort_values(["size", "block"] if "block" in x else ["size"]), cols), ""]
    if len(gsum):
        L += ["### A2. Group steering (`group_summary.csv`, `group_steering.csv`)", "",
              "Match rates and effect-size distributions of labeled groups against matched random groups (16 per labeled group); `cal` = calibrated criterion with the pooled random-group "
              "95th percentile of the same size, k and predicted quantity.", "", md(gsum), ""]
        top = glab.sort_values("rank_among_random", ascending=False).groupby("size").head(5)
        L += ["Labeled groups with the highest rank among their random groups, per size:", "",
              md(top, ["size", "gid", "predicted", "direction", "effect_size", "p95_random_pooled", "rank_among_random", "rho", "orig_match", "cal_match"]), ""]
    if len(single):
        L += ["### A3. Single poles, re-judged with the calibrated criterion (`single_pole_calibrated.csv`)", "",
              "From the original run (`../steering_results.csv`): 144 random control poles per size judged for level, slope and spread. Labeled poles of the amp and phase groups have no control "
              "distribution and are not in this table.", "", md(single), ""]
    return "\n".join(L)


def section_c(motif: pd.DataFrame, names: pd.DataFrame, byname: pd.DataFrame) -> str:
    L = ["## C. Motif labels without a variable list (`motif_clusters.csv`, `motif_cluster_names.csv`, `motif_by_name.csv`)", ""]
    if motif.empty:
        return "\n".join(L + ["No motif files found."])
    L += ["Top and mid-range exemplar patches of every pole, normalized per patch (subtract mean, divide by sd), assigned to a fixed vocabulary of K = 16 shapes (k-means on the "
          "top-exemplar patches of the four atlas main-corpus runs, pooled over sizes). A pole is motif-cohesive when at least 6 of its 8 top exemplars share a cluster. "
          "`null` = the same share when exemplar cluster ids are permuted across poles (mean of 20 permutations); `one_job` = known-variable share of the same run "
          "(theta 0.7, floor 0.1).", "", "Cluster names (`motif_cluster_names.csv`):", ""]
    if len(names):
        L += [md(names[[c for c in ["cluster", "name", "purity", "n"] if c in names]]), ""]
    cols = ["tag", "role", "n_series", "n_live", "cohesive_top6", "null_top6_mean", "cohesive_mid6", "one_job_0.7", "both_known_and_motif", "motif_only", "known_only", "neither"]
    for s in SIZES:
        x = motif[motif["size"] == s]
        if len(x):
            L += [f"### {s}", "", md(x.sort_values(["n_series", "tag"], ascending=[False, True]), cols), ""]
    agg = motif[motif.role.isin(["full_main", "subsample"])].groupby(["size", "n_series"]).agg(
        cohesive_top6_mean=("cohesive_top6", "mean"), cohesive_top6_min=("cohesive_top6", "min"), cohesive_top6_max=("cohesive_top6", "max"),
        one_job_mean=("one_job_0.7", "mean"), runs=("tag", "count")).reset_index()
    ctx = motif[motif.role == "context"][["size", "n_series", "tag", "cohesive_top6", "one_job_0.7"]].rename(
        columns={"tag": "context_run", "cohesive_top6": "context_cohesive_top6", "one_job_0.7": "context_one_job"})
    pair = ctx.merge(agg, on=["size", "n_series"], how="left")
    full = agg[agg.n_series == 3100][["size", "cohesive_top6_mean", "one_job_mean"]].rename(columns={"cohesive_top6_mean": "full_main_cohesive_top6", "one_job_mean": "full_main_one_job"})
    pair = pair.merge(full, on="size", how="left")
    L += ["### Matched series counts", "", "Each context-length corpus next to a random subsample of the main corpus with the same number of series (seeds pooled) and next to the full main corpus.", "",
          md(pair[["size", "context_run", "n_series", "context_cohesive_top6", "cohesive_top6_mean", "cohesive_top6_min", "cohesive_top6_max", "full_main_cohesive_top6",
                   "context_one_job", "one_job_mean", "full_main_one_job", "runs"]]), ""]
    sens = ["tag", "cohesive_top6", "cohesive_top6_K8", "cohesive_top6_K32", "cohesive_top5", "cohesive_top8", "cohesive_top6_scaled", "cohesive_top6_refit", "all_top_context_tokens",
            "cohesive_top6_context_only_poles", "cluster_share_max"]
    L += ["### Sensitivity (vocabulary size, threshold, normalization, vocabulary refitted inside each run)", "", md(motif[motif.role.isin(["atlas_main", "full_main"])][sens]), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-runs", default="results/colab_out")
    ap.add_argument("--steering", default="results/neurons/steering_results.csv")
    ap.add_argument("--out", default="results/neurons/followup")
    a = ap.parse_args()
    d, readouts, metas = load_a(a.a_runs)
    steer = pd.read_csv(a.steering)
    single, p95_pole = single_pole_tables(steer)
    real = direction_tables(d) if len(d) else pd.DataFrame()
    glab, gsum = group_tables(d) if len(d) else (pd.DataFrame(), pd.DataFrame())
    mde = mde_table(real, glab, d, p95_pole) if len(real) or len(glab) else pd.DataFrame()
    fl = "%.5g"
    if len(readouts):
        readouts.to_csv(os.path.join(a.out, "direction_readouts.csv"), index=False, float_format=fl)
    if len(real):
        real.drop(columns=["match"]).to_csv(os.path.join(a.out, "direction_steering.csv"), index=False, float_format=fl)
    if len(glab):
        glab.to_csv(os.path.join(a.out, "group_steering.csv"), index=False, float_format=fl)
        gsum.to_csv(os.path.join(a.out, "group_summary.csv"), index=False, float_format=fl)
    if len(mde):
        mde.to_csv(os.path.join(a.out, "mde.csv"), index=False, float_format=fl)
    single.to_csv(os.path.join(a.out, "single_pole_calibrated.csv"), index=False, float_format=fl)
    motif = pd.read_csv(os.path.join(a.out, "motif_clusters.csv")) if os.path.exists(os.path.join(a.out, "motif_clusters.csv")) else pd.DataFrame()
    names = pd.read_csv(os.path.join(a.out, "motif_cluster_names.csv")) if os.path.exists(os.path.join(a.out, "motif_cluster_names.csv")) else pd.DataFrame()
    byname = pd.read_csv(os.path.join(a.out, "motif_by_name.csv")) if os.path.exists(os.path.join(a.out, "motif_by_name.csv")) else pd.DataFrame()
    text = section_a(readouts, real, glab, gsum, single, metas) + "\n\n" + section_c(motif, names, byname)
    rp = os.path.join(a.out, "README.md")
    body = open(rp).read()
    body = body.split(MARK)[0].rstrip() + "\n\n" + MARK + "\n\n" + text + "\n"
    open(rp, "w").write(body)
    write_json(os.path.join(a.out, "provenance_followup_ac.json"), provenance("scripts/neurons_followup_report.py", argv=sys.argv, a_meta=metas))
    print("tables written")


if __name__ == "__main__":
    main()
