"""Summaries of the steering, transfer, cross-size, upstream and squeeze results (definitions in the docstrings)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from otp.neurons import labels as Lb
from otp.neurons import score as Sc
from otp.neurons import steer as St


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def steering_summary(res: pd.DataFrame) -> pd.DataFrame:
    """Match rate of steered poles with the pre-registered prediction, by size and predicted quantity group, next to the
    null rate of random poles (each control pole judged for level, slope and spread with a random direction).
    Rows: role `labeled` (poles drawn for the table of predictions), `complement` (the other pole of a drawn neuron, judged
    by its own label when that label has a prediction), `control` (pairs of pole and quantity)."""
    rows = []
    for size, d in res.groupby("size", sort=False):
        for role in ("labeled", "complement", "control"):
            x = d[(d["role"] == role) & d["match"].notna()]
            if role == "control":
                groups = {"level": x[x.quantity == "level"], "slope": x[x.quantity == "slope"], "spread": x[x.quantity == "spread"],
                          "all": x}
            else:
                groups = {g: x[x["predicted"].map(lambda q, g=g: q == g)] for g in St.GROUPS if g != "level|spread"}
                groups["level|spread"] = x[x["predicted"] == "level|spread"]
                groups["all"] = x
            for g, y in groups.items():
                n = len(y)
                if not n:
                    continue
                k = int(y["match"].astype(bool).sum())
                lo, hi = wilson(k, n)
                rows.append({"size": size, "role": role, "group": g, "n": n, "n_match": k, "match_rate": k / n, "ci_lo": lo, "ci_hi": hi,
                             "monotone_rate": float(y["monotone"].astype(bool).mean()), "size_rate": float(y["big"].astype(bool).mean()),
                             "interval_rate": float(y["sig"].astype(bool).mean())})
    return pd.DataFrame(rows)


def steering_checks(res: pd.DataFrame, eff: pd.DataFrame, upstream: pd.DataFrame | None, poles: pd.DataFrame) -> pd.DataFrame:
    """Consistency checks of the pre-registration, per size.

    downstream: among labeled poles with no match, the share whose largest ablation effect, in units of the quantity's size
      threshold, is on a quantity other than the predicted one and at least 1 (these labels are flagged as broken by this check).
    negative_pole: among neurons where a labeled pole and its complement have the same best label, the share whose scale-4
      effects on the predicted quantity have the signs the labels' correlations imply (opposite signs when the two label
      correlations have opposite signs, equal signs otherwise).
    upstream: among steered labeled poles with upstream rows, the share flagged: none of the three most correlated
      earlier-layer poles has a label in the pole's group, and the best motif label explains less than half of the label's R^2."""
    rows = []
    for size, d in res.groupby("size", sort=False):
        e = eff[eff["size"] == size]
        lab = d[(d["role"] == "labeled") & d["match"].notna()]
        fail = lab[~lab["match"].astype(bool)]
        n_dn = 0
        for r in fail.itertuples():
            ee = e[(e.layer == r.layer) & (e.neuron == r.neuron) & (e.sign == r.sign) & (e.scale == 0.0) & (e.role == r.role)]
            if ee.empty:
                continue
            rel = ee.set_index("quantity")["mean_effect"].abs() / pd.Series(St.THRESH)[ee["quantity"].values].values
            pq = str(r.predicted).split("|")
            n_dn += int(rel.idxmax() not in pq and rel.max() >= 1.0)
        row = {"size": size, "n_labeled": len(lab), "n_unmatched": len(fail), "downstream_flagged": n_dn,
               "downstream_flagged_share": n_dn / len(fail) if len(fail) else np.nan}
        comp = d[(d["role"] == "complement")]
        n_pair = n_ok = 0
        for r in lab.itertuples():
            c = comp[(comp.layer == r.layer) & (comp.neuron == r.neuron) & (comp.sign != r.sign)]
            if c.empty or c.iloc[0]["label"] != r.label or r.quantity in ("", None) or not isinstance(r.quantity, str):
                continue
            q = r.quantity
            e4 = e[(e.layer == r.layer) & (e.neuron == r.neuron) & (e.scale == 4.0) & (e.quantity == q)]
            ea = e4[(e4.sign == r.sign) & (e4.role == "labeled")]["mean_effect"]
            eb = e4[(e4.sign != r.sign) & (e4.role == "complement")]["mean_effect"]
            if ea.empty or eb.empty:
                continue
            same_corr = (np.sign(r.label_corr) == np.sign(c.iloc[0]["label_corr"]))
            n_pair += 1
            n_ok += int((np.sign(ea.iloc[0]) == np.sign(eb.iloc[0])) == bool(same_corr))
        row.update({"negative_pole_pairs": n_pair, "negative_pole_consistent": n_ok,
                    "negative_pole_share": n_ok / n_pair if n_pair else np.nan})
        if upstream is not None:
            u = upstream[upstream["size"] == size]
            p = poles[poles["size"] == size].set_index(["layer", "neuron", "sign"])
            n_u = n_f = 0
            for r in lab.itertuples():
                k = (r.layer, r.neuron, r.sign)
                m = u[(u.layer == r.layer) & (u.neuron == r.neuron) & (u.sign == r.sign)]
                if m.empty or k not in p.index:
                    continue
                n_u += 1
                grp = Lb.GROUP_OF.get(r.label, "")
                same = any(m.iloc[0][f"up{i}_group"] == grp for i in range(3))
                weak = p.loc[k, "r2_best_motif"] < 0.5 * p.loc[k, "r2_cv"]
                n_f += int((not same) and weak)
            row.update({"upstream_checked": n_u, "upstream_flagged": n_f, "upstream_flagged_share": n_f / n_u if n_u else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def transfer(df: pd.DataFrame, theta: float = 0.7, floor: float = 0.1) -> pd.DataFrame:
    """Share of poles whose synthetic label predicts their activation on real series (Pearson correlation `tr_corr` of the
    label's bin-mean readout with the real activation), at three thresholds, for one-job poles whose label has a measurable
    analog, and, as a reference, for unexplained poles with an analog (their best label carries no information)."""
    rows = []
    for size, d in df.groupby("size", sort=False):
        if "tr_corr" not in d:
            continue
        cls = Sc.klass(d, theta, floor)
        for name, m in (("one_job", cls == "one_job"), ("mixed", cls == "mixed"), ("unexplained", cls == "unexplained")):
            x = d[m & d["tr_corr"].notna().to_numpy()]
            if not len(x):
                continue
            row = {"size": size, "poles": name, "n_with_analog": len(x), "median_tr_corr": float(x["tr_corr"].median()),
                   "median_tr_r2": float(x["tr_r2"].median())}
            for t in (0.1, 0.3, 0.5):
                row[f"share_corr_ge_{t}"] = float((x["tr_corr"] >= t).mean())
            row["share_r2_gt_0"] = float((x["tr_r2"] > 0).mean())
            for k in range(3):
                if f"ev{k}_corr" in x:
                    row[f"share_abs_event{k}_corr_ge_0.2"] = float((x[f"ev{k}_corr"].abs() >= 0.2).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def transfer_by_group(df: pd.DataFrame, theta: float = 0.7, floor: float = 0.1) -> pd.DataFrame:
    rows = []
    for size, d in df.groupby("size", sort=False):
        if "tr_corr" not in d:
            continue
        cls = Sc.klass(d, theta, floor)
        x = d[(cls == "one_job") & d["tr_corr"].notna().to_numpy()]
        for g, y in x.groupby("group"):
            rows.append({"size": size, "group": g, "n": len(y), "median_tr_corr": float(y["tr_corr"].median()),
                         "share_corr_ge_0.3": float((y["tr_corr"] >= 0.3).mean())})
    return pd.DataFrame(rows)


def scale_match(m: pd.DataFrame, poles: pd.DataFrame, theta: float = 0.7, floor: float = 0.1) -> pd.DataFrame:
    """For one-job poles of the smaller model of each size pair: how well does a pole of the larger model match its activation?
    `n_ge50` counts larger-model poles with correlation >= 0.5 on the shared token subsample (more than one means the pole is
    represented by several poles in the larger model). For matched pairs (best correlation >= 0.7): same label, same
    label group, and the larger model's R^2 against the smaller's."""
    rows = []
    for (small, large), d in m.groupby(["small", "large"], sort=False):
        t = poles[poles["size"] == small].copy()
        t["cls"] = Sc.klass(t, theta, floor)
        oj = t[t["cls"] == "one_job"][["layer", "neuron", "sign"]]
        x = d.merge(oj, on=["layer", "neuron", "sign"])
        if not len(x):
            continue
        g = x[x["best_corr"] >= 0.7]
        rows.append({"small": small, "large": large, "n_one_job_small": len(x), "median_best_corr": float(x["best_corr"].median()),
                     "share_best_ge_0.5": float((x["best_corr"] >= 0.5).mean()), "share_best_ge_0.7": float((x["best_corr"] >= 0.7).mean()),
                     "share_two_or_more_ge_0.5": float((x["n_ge50"] >= 2).mean()), "mean_n_ge50": float(x["n_ge50"].mean()),
                     "n_matched": len(g), "matched_same_label": float((g["label"] == g["match_label"]).mean()) if len(g) else np.nan,
                     "matched_same_group": float((g["group"] == g["match_group"]).mean()) if len(g) else np.nan,
                     "matched_large_r2_minus_small_r2": float((g["match_r2_cv"] - g["r2_cv"]).mean()) if len(g) else np.nan,
                     "matched_large_r2_higher": float((g["match_r2_cv"] > g["r2_cv"]).mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


def heads_summary(ht: pd.DataFrame) -> pd.DataFrame:
    """Number of time-attention heads per description, by size and query type."""
    return (ht.groupby(["size", "query", "description"]).size().rename("n_heads").reset_index()
            .assign(share=lambda x: x["n_heads"] / x.groupby(["size", "query"])["n_heads"].transform("sum")))


def squeeze(tables: dict[str, pd.DataFrame], theta: float = 0.7, floor: float = 0.1) -> pd.DataFrame:
    """Label tables of models that differ only in MLP width: class shares, mean E, spread of labels over neurons."""
    rows = []
    for name, t in tables.items():
        cls = Sc.klass(t, theta, floor)
        live = cls != "dead"
        lab = t.loc[(cls == "one_job"), "label"]
        grp = t.loc[(cls == "one_job"), "group"]
        rows.append({"model": name, "n_neurons": int(t.groupby(["layer", "neuron"]).ngroups), "n_poles": len(t), "n_live": int(live.sum()),
                     "one_job": float((cls == "one_job").sum() / max(live.sum(), 1)), "mixed": float((cls == "mixed").sum() / max(live.sum(), 1)),
                     "unexplained": float((cls == "unexplained").sum() / max(live.sum(), 1)),
                     "E_mean": float(t.loc[live, "E"].mean()), "r2_cv_mean": float(t.loc[live, "r2_cv"].mean()),
                     "distinct_labels_one_job": int(lab.nunique()), "distinct_groups_one_job": int(grp.nunique()),
                     "mean_best_over_pair_r2": float((t.loc[live, "r2_cv"] / t.loc[live, "r2_pair"].clip(lower=1e-9)).mean()),
                     "share_one_job_group_level_only": float(grp.isin(["family", "position"]).mean()) if len(grp) else np.nan})
    return pd.DataFrame(rows)
