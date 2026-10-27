"""Turn per-pole label tables into the audit's summary tables.

Definitions used here (also written in `results/neurons/README.md`):

- A pole is *dead* when it has fewer than `labels.MIN_ACTIVE` active label-set tokens.
- E, the explainable variance of a live pole, is the largest of the cross-validated R^2 of its best single label,
  of its best pair label and of the additive model on all library variables.
- ratio = R^2(best single label) / E. A pole is *unexplained* when E is below the floor, *one-job* when E is at
  least the floor and ratio is at least theta, and *mixed* otherwise (E at least the floor, ratio below theta).
- The held-out version uses the label chosen on the label set and the held-out series' R^2 for the label, the pair
  label and the additive model.
- Intervals: `lo`/`hi` are the 2.5% and 97.5% points of 500 bootstrap resamples of the poles (category counts drawn
  from a multinomial with the observed shares, so neuron pairs are treated as independent); `fold_lo`/`fold_hi`
  are the smallest and largest share when the label's R^2 is taken from each of the 3 folds of series in turn
  (E stays pooled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from otp.neurons import labels as Lb
from otp.neurons import variables as V

THETAS = (0.5, 0.6, 0.7, 0.8, 0.9)
FLOORS = (0.05, 0.1, 0.2)
N_BOOT = 500
RELATED = [{"level", "slope", "curvature"}, {"noise", "tail", "clustering"}, {"period", "phase"},
           {"regime", "persistence"}, {"chaos", "persistence"}, {"coupling"}, {"position", "family"}]
CLASSES = ("one_job", "mixed", "unexplained")


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Columns E, ratio, E_ho, ratio_ho, group, label_kind."""
    df = df.copy()
    e = np.fmax.reduce([df["r2_cv"], df["r2_pair"], df["r2_add"]])
    e_ho = np.fmax.reduce([df["r2_heldout"], df["r2_pair_heldout"], df["r2_add_heldout"]])
    df["E"] = np.maximum(e, 0.0)
    df["E_ho"] = np.maximum(e_ho, 0.0)
    df["ratio"] = df["r2_cv"] / df["E"].clip(lower=1e-9)
    df["ratio_ho"] = df["r2_heldout"] / df["E_ho"].clip(lower=1e-9)
    for k in range(3):
        df[f"ratio_f{k}"] = df[f"r2_fold{k}"] / df["E"].clip(lower=1e-9)
    df["group"] = df["label"].map(lambda x: Lb.GROUP_OF.get(x, "dead"))
    return df


def klass(df: pd.DataFrame, theta: float, floor: float, ratio: str = "ratio", e: str = "E") -> np.ndarray:
    """Per-pole class: dead / unexplained / one_job / mixed."""
    dead = df["dead"].to_numpy().astype(bool)
    out = np.where(df[e].to_numpy() < floor, "unexplained", np.where(df[ratio].to_numpy() >= theta, "one_job", "mixed"))
    out = np.where(dead, "dead", out)
    return out


def _shares(cls: np.ndarray) -> tuple[dict, int]:
    live = cls != "dead"
    n = int(live.sum())
    return {k: (float((cls == k).sum()) / n if n else np.nan) for k in CLASSES}, n


def _boot(counts: np.ndarray, rng) -> np.ndarray:
    n = int(counts.sum())
    if n == 0:
        return np.full((N_BOOT, len(counts)), np.nan)
    return rng.multinomial(n, counts / n, size=N_BOOT) / n


def curves(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """one_job / mixed / unexplained shares of live poles by size, scope, sign, floor and theta."""
    rng = np.random.default_rng(seed)
    rows = []
    for size, ds in df.groupby("size", sort=False):
        nl = int(ds["layer"].max()) + 1
        scopes = {"all": np.ones(len(ds), bool)}
        for i, nm in enumerate(("early", "mid", "late")):
            scopes[nm] = ((ds["layer"] * 3 // nl) == i).to_numpy()
        for li in range(nl):
            scopes[f"layer_{li}"] = (ds["layer"] == li).to_numpy()
        for scope, msk in scopes.items():
            for sign in ("both", "pos", "neg"):
                sm = msk if sign == "both" else msk & (ds["sign"] == sign).to_numpy()
                d = ds[sm]
                if not len(d):
                    continue
                for floor in FLOORS:
                    for theta in THETAS:
                        cls = klass(d, theta, floor)
                        sh, n = _shares(cls)
                        clsh = klass(d, theta, floor, "ratio_ho", "E_ho")
                        sh_ho, _ = _shares(clsh)
                        counts = np.array([(cls == k).sum() for k in CLASSES])
                        b = _boot(counts, rng)
                        lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)
                        fold = []
                        for k in range(3):
                            dd = d.copy()
                            dd["ratio"] = dd[f"ratio_f{k}"]
                            fold.append(_shares(klass(dd, theta, floor))[0]["one_job"])
                        rows.append({
                            "size": size, "scope": scope, "sign": sign, "floor": floor, "theta": theta,
                            "n_poles": len(d), "n_dead": int((cls == "dead").sum()), "n_live": n,
                            "one_job": sh["one_job"], "mixed": sh["mixed"], "unexplained": sh["unexplained"],
                            "one_job_lo": lo[0], "one_job_hi": hi[0], "mixed_lo": lo[1], "mixed_hi": hi[1],
                            "unexplained_lo": lo[2], "unexplained_hi": hi[2],
                            "one_job_fold_lo": float(np.nanmin(fold)), "one_job_fold_hi": float(np.nanmax(fold)),
                            "one_job_heldout": sh_ho["one_job"], "mixed_heldout": sh_ho["mixed"],
                            "unexplained_heldout": sh_ho["unexplained"]})
    return pd.DataFrame(rows)


def related(a: str, b: str) -> str:
    if a == b:
        return "same_group"
    if any(a in g and b in g for g in RELATED):
        return "related_groups"
    return "unrelated_groups"


def pole_pairs(df: pd.DataFrame, theta: float = 0.7, floor: float = 0.1) -> pd.DataFrame:
    """The two poles of a neuron: how are their labels related? Only neurons whose two poles are both live and
    explained (E >= floor) are classified; the rest are counted by what the other pole is."""
    rows = []
    for size, ds in df.groupby("size", sort=False):
        d = ds.copy()
        d["cls"] = klass(d, theta, floor)
        p = d[d["sign"] == "pos"].set_index(["layer", "neuron"])
        n = d[d["sign"] == "neg"].set_index(["layer", "neuron"]).loc[p.index]
        both = (p["cls"] != "dead") & (n["cls"] != "dead") & (p["E"] >= floor).to_numpy() & (n["E"] >= floor).to_numpy()
        for scope, ix in (("all", np.ones(len(p), bool)),) + tuple((f"layer_{li}", (p.index.get_level_values(0) == li)) for li in sorted(set(p.index.get_level_values(0)))):
            ix = np.asarray(ix)
            b = np.asarray(both) & ix
            row = {"size": size, "scope": scope, "theta": theta, "floor": floor, "n_neurons": int(ix.sum()),
                   "n_both_explained": int(b.sum()),
                   "n_one_pole_dead": int((((p["cls"] == "dead") ^ (n["cls"] == "dead")).to_numpy() & ix).sum()),
                   "n_both_dead": int((((p["cls"] == "dead") & (n["cls"] == "dead")).to_numpy() & ix).sum())}
            if b.sum():
                pl, nlb = p["label"].to_numpy()[b], n["label"].to_numpy()[b]
                pg, ng = p["group"].to_numpy()[b], n["group"].to_numpy()[b]
                oj = (p["cls"].to_numpy()[b] == "one_job") & (n["cls"].to_numpy()[b] == "one_job")
                rel = np.array([related(a, c) for a, c in zip(pg, ng)])
                row.update({"both_one_job": float(oj.mean()), "same_label": float((pl == nlb).mean()),
                            "same_group_not_label": float(((rel == "same_group") & (pl != nlb)).mean()),
                            "related_groups": float((rel == "related_groups").mean()),
                            "unrelated_groups": float((rel == "unrelated_groups").mean())})
                # among neurons with both poles one-job
                if oj.sum():
                    row.update({"oj_same_label": float((pl[oj] == nlb[oj]).mean()),
                                "oj_same_group": float(((rel[oj] == "same_group") & (pl[oj] != nlb[oj])).mean()),
                                "oj_related": float((rel[oj] == "related_groups").mean()),
                                "oj_unrelated": float((rel[oj] == "unrelated_groups").mean())})
            rows.append(row)
    return pd.DataFrame(rows)


def sign_regimes(df: pd.DataFrame, floor: float = 0.1, dom: float = 0.9) -> pd.DataFrame:
    """SwiGLU sign regimes. For a positive pole the regimes are (gate>0, up>0) and (gate<0, up<0); for a negative
    pole (gate>0, up<0) and (gate<0, up>0). `dominant`: one regime holds at least `dom` of the pole's activation
    mass. `two_regime`: both hold at least 10%; of those, `differ`: the regime-specific best labels fall in different
    groups, and `both_explained`: each regime's R^2 is at least `floor`."""
    rows = []
    for (size, sign), d in df.groupby(["size", "sign"], sort=False):
        d = d[~d["dead"]]
        if sign == "pos":
            a, b = d["mass_g1u1"], d["mass_g0u0"]
        else:
            a, b = d["mass_g1u0"], d["mass_g0u1"]
        tot = (a + b).clip(lower=1e-9)
        fa = a / tot
        dominant = (fa >= dom) | (fa <= 1 - dom)
        two = (fa > 0.1) & (fa < 0.9)
        ga = d["label_regA"].map(lambda x: Lb.GROUP_OF.get(x, "dead"))
        gb = d["label_regB"].map(lambda x: Lb.GROUP_OF.get(x, "dead"))
        ok = two & (d["r2_regA"] >= floor) & (d["r2_regB"] >= floor)
        rows.append({"size": size, "sign": sign, "n_live": len(d), "share_of_mass_in_agreeing_regime_mean": float(fa.mean()),
                     "dominant_one_regime": float(dominant.mean()), "two_regime": float(two.mean()),
                     "two_regime_both_explained": float(ok.mean()),
                     "of_those_labels_differ_group": float((ga[ok] != gb[ok]).mean()) if ok.any() else np.nan,
                     "of_those_labels_differ_variable": float((d["label_regA"][ok] != d["label_regB"][ok]).mean()) if ok.any() else np.nan,
                     "r2_regA_mean": float(d["r2_regA"].mean()), "r2_regB_mean": float(d["r2_regB"].mean()),
                     "r2_regime_split_gain_over_single": float((np.fmax(d["r2_regA"], d["r2_regB"]) - d["r2_cv"]).mean())})
    return pd.DataFrame(rows)


def kinds(df: pd.DataFrame, floor: float = 0.1) -> pd.DataFrame:
    """Which kind of variable explains a pole best: summary motif, token type / position, latent, family."""
    rows = []
    for (size, sign), d in df.groupby(["size", "sign"], sort=False):
        d = d[(~d["dead"]) & (d["E"] >= floor)]
        mat = d[["r2_best_motif", "r2_best_position", "r2_best_latent", "r2_best_family"]]
        win = mat.idxmax(axis=1).str.replace("r2_best_", "")
        row = {"size": size, "sign": sign, "n_explained": len(d)}
        for k in ("motif", "position", "latent", "family"):
            row[f"best_is_{k}"] = float((win == k).mean())
            row[f"r2_best_{k}_mean"] = float(d[f"r2_best_{k}"].mean())
        m = win == "motif"
        row["motif_poles_position_r2_mean"] = float(d.loc[m, "r2_best_position"].mean()) if m.any() else np.nan
        row["motif_poles_position_explains_ge_half"] = (
            float((d.loc[m, "r2_best_position"] >= 0.5 * d.loc[m, "r2_best_motif"]).mean()) if m.any() else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def group_table(df: pd.DataFrame, theta: float = 0.7, floor: float = 0.1) -> pd.DataFrame:
    """Share of one-job poles by label group and size."""
    rows = []
    for size, d in df.groupby("size", sort=False):
        d = d.copy()
        d["cls"] = klass(d, theta, floor)
        oj = d[d["cls"] == "one_job"]
        for g, n in oj["group"].value_counts().items():
            rows.append({"size": size, "group": g, "n_one_job": int(n), "share_of_one_job": n / len(oj),
                         "share_of_live": n / int((d["cls"] != "dead").sum())})
    return pd.DataFrame(rows)


def scaling(cv: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """One-job share at the headline setting (theta 0.7, floor 0.1) by size, with intervals, plus mean E and R^2."""
    c = cv[(cv.scope == "all") & (cv.sign == "both") & (cv.theta == 0.7) & (cv.floor == 0.1)]
    return c[["size", "n_poles", "n_dead", "n_live", "one_job", "one_job_lo", "one_job_hi", "one_job_fold_lo",
              "one_job_fold_hi", "one_job_heldout", "mixed", "mixed_lo", "mixed_hi", "unexplained", "unexplained_lo",
              "unexplained_hi"]]


def heldout_check(df: pd.DataFrame, theta: float = 0.7, floor: float = 0.1) -> pd.DataFrame:
    """Do labels chosen on the label set keep their explained variance on held-out series?"""
    rows = []
    for size, d in df.groupby("size", sort=False):
        d = d[~d["dead"]]
        cls = klass(d, theta, floor)
        for name, m in (("all_live", np.ones(len(d), bool)), ("one_job", cls == "one_job"), ("mixed", cls == "mixed")):
            x = d[m]
            if len(x) < 5:
                continue
            gap = x["r2_cv"] - x["r2_heldout"]
            rows.append({"size": size, "poles": name, "n": len(x), "r2_cv_mean": float(x["r2_cv"].mean()),
                         "r2_heldout_mean": float(x["r2_heldout"].mean()), "gap_mean": float(gap.mean()),
                         "gap_median": float(gap.median()), "gap_p95": float(gap.quantile(0.95)),
                         "pearson_cv_heldout": float(np.corrcoef(x["r2_cv"], x["r2_heldout"])[0, 1]),
                         "share_heldout_ratio_ge_theta": float((x["ratio_ho"] >= theta).mean())})
    return pd.DataFrame(rows)


def context_comparison(tables: dict[str, pd.DataFrame], theta: float = 0.7, floor: float = 0.1) -> pd.DataFrame:
    """Compare label tables of the same size computed on corpora with different context lengths.
    `tables` maps a context name ('main', 'short', 'long', 'xlong') to a derived table (`add_derived`)."""
    rows = []
    base = tables["main"]
    for name, t in tables.items():
        for size in t["size"].unique():
            a = base[base["size"] == size].set_index(["layer", "neuron", "sign"])
            b = t[t["size"] == size].set_index(["layer", "neuron", "sign"])
            ix = a.index.intersection(b.index)
            a, b = a.loc[ix], b.loc[ix]
            ca, cb = klass(a, theta, floor), klass(b, theta, floor)
            live = (ca != "dead") & (cb != "dead")
            both = live & (ca != "unexplained") & (cb != "unexplained")
            row = {"size": size, "context": name, "n_poles": len(ix), "n_live": int((cb != "dead").sum()),
                   "one_job": float((cb == "one_job").sum() / max((cb != "dead").sum(), 1)),
                   "unexplained": float((cb == "unexplained").sum() / max((cb != "dead").sum(), 1)),
                   "dead": float((cb == "dead").mean()), "E_mean": float(b.loc[cb != "dead", "E"].mean()),
                   "n_both_explained": int(both.sum())}
            if both.any():
                row["same_label_as_main"] = float((a["label"][both] == b["label"][both]).mean())
                row["same_group_as_main"] = float((a["group"][both] == b["group"][both]).mean())
                row["r2_corr_with_main"] = float(np.corrcoef(a["r2_cv"][both], b["r2_cv"][both])[0, 1])
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# shared motif over exemplar lists
# ---------------------------------------------------------------------------------------------
def motif_cohesion(df: pd.DataFrame, X: np.ndarray, label_mask: np.ndarray, chunk: int = 20000) -> pd.DataFrame:
    """For each pole, how alike are the motif features of its top and of its mid-range exemplar tokens?

    X (n_tokens, len(LIBRARY)) with NaN, label_mask (n_tokens,) tokens of the label set. cohesion = mean over the 18
    motif features of (standard deviation over the exemplar list) / (standard deviation over all label tokens); 1 is
    the value for random tokens, smaller means the list shares the motif. Adds `coh_top`, `coh_mid`, and a plain-words
    description `motif_top`, `motif_mid`: the two motif features whose exemplar mean is furthest from the label-set mean
    (in label-set standard deviations), among those whose spread over the list is below the label-set spread, written
    as `name high` or `name low`.
    """
    names = [n for n in V.LIBRARY if n.startswith("m_")]
    mot = [V.VAR_INDEX[n] for n in names]
    Xm = X[:, mot]
    sd = np.nanstd(Xm[label_mask], axis=0).clip(min=1e-9)
    mu = np.nanmean(Xm[label_mask], axis=0)
    out = {}
    for name in ("top", "mid"):
        res = np.full(len(df), np.nan, np.float32)
        desc = [""] * len(df)
        col = df[f"{name}_tok"].to_numpy()
        for s in range(0, len(df), chunk):
            ids = np.array([[int(t) for t in str(r).split()] if isinstance(r, str) else [0] * 8 for r in col[s:s + chunk]])
            F = Xm[ids]                                                   # (c, k, nm)
            ratio = np.nanstd(F, axis=1) / sd[None]
            res[s:s + chunk] = np.nanmean(ratio, axis=1)
            z = (np.nanmean(F, axis=1) - mu[None]) / sd[None]
            z = np.where(ratio < 1.0, z, 0.0)
            top2 = np.argsort(-np.abs(z), axis=1)[:, :2]
            for i in range(len(z)):
                parts = [f"{names[j]} {'high' if z[i, j] > 0 else 'low'}" for j in top2[i] if abs(z[i, j]) >= 0.5]
                desc[s + i] = "; ".join(parts) if parts else "no shared motif"
        out[f"coh_{name}"] = res
        out[f"motif_{name}"] = desc
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------------------------
# table files: numpy archives (the project's text checker skips .npz; random bytes of compressed parquet can match its word list)
# ---------------------------------------------------------------------------------------------
CORE_COLUMNS = ("size", "layer", "neuron", "sign", "n_active_label", "mean", "std", "max", "label", "label_corr", "r2_cv", "label_runner_up",
                "r2_runner_up", "r2_heldout", "pair_label", "r2_pair", "r2_add", "dead", "E", "ratio", "group")


def write_table(t: pd.DataFrame, path: str) -> None:
    """Write a table as a compressed numpy archive: one array per column (float32 for floats, fixed-width unicode for text)."""
    cols = {}
    for c in t.columns:
        v = t[c]
        if v.dtype == object:
            cols[c] = v.fillna("").astype(str).to_numpy().astype("U")
        elif v.dtype == "float64":
            cols[c] = v.to_numpy().astype(np.float32)
        else:
            cols[c] = v.to_numpy()
    np.savez_compressed(path, __columns__=np.asarray(list(t.columns), dtype="U"), **cols)


def read_table(path: str) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as z:
        return pd.DataFrame({c: z[c] for c in z["__columns__"]})
