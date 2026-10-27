"""Score the four Timestral sizes and the classical models on the two regression sets.

Sets (built by `scripts/build_regression_sets.py`): `bench/regression/real.npz` (real windows, no oracle)
and `bench/regression/in_prior.npz` (synthetic, with oracle quantiles).

Outputs in `--out` (default `results/regression_baseline`):
  scores.csv      one row per window x forecaster: MASE, sCRPS, WQL, cov80, cov98, calib
  summary.csv     aggregates per set, group and forecaster (definitions below)
  provenance.json command line, commit, host, per-forecaster counts and wall time, any subset used
  README.md       tables copied from summary.csv

Aggregation (the GIFT-Eval way). Within a group (a dataset/term pair for the real set, a seasonal
period for the in-prior set) the group score of a forecaster is the arithmetic mean over windows of
its MASE (or sCRPS). The relative score is the group score divided by seasonal naive's group score.
The overall relative score is the geometric mean of the group relative scores over groups. Windows
with a non-finite value for the metric are left out of that metric. For the in-prior set the same is
done against the oracle (`*_rel_oracle`). The oracle is never used for the real set.

python scripts/score_regression.py --sets real in_prior --out results/regression_baseline
python scripts/score_regression.py --summarize-only --out results/regression_baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp import realdata  # noqa: E402
from otp.evaluate import METRIC_COLS, score_samples  # noqa: E402
from otp.generators.base import load_samples  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.model import SIZES, set_threads  # noqa: E402

BENCH = "bench/regression"
SETS = ("real", "real_extra", "analogues", "in_prior")
CLASSICAL = ("seasonal_naive", "ets", "arima")
REL_METRICS = ("mase", "scrps")


def load_set(name: str, bench: str = BENCH):
    """Windows of a set in manifest order (real sets: those present on this machine; see `realdata.load_union`)."""
    if name == "in_prior":
        return load_samples(os.path.join(bench, "in_prior.npz"))
    return realdata.load_union(bench, name)


def window_number(samples, i: int) -> int:
    """Position of window `i` in its set's manifest (stable across machines)."""
    return int(samples[i].meta.get("manifest_row", i))


def forecasters_for(name: str, sizes) -> list[str]:
    """Forecasters to run on a set. The oracle is added for the in-prior set only."""
    fs = [f"timestral-{s}" for s in sizes] + list(CLASSICAL)
    return fs + (["oracle"] if name == "in_prior" else [])


def group_of(df: pd.DataFrame, set_name: str) -> pd.Series:
    """Group label per row: dataset (with term) for the real set, seasonal period for the in-prior set."""
    if set_name == "analogues":
        return df["family_analogue"].astype(str) + "/" + df["generator"].astype(str)
    if set_name in ("real", "real_extra"):
        return df["generator"].astype(str)
    return "period=" + df["period"].astype(int).astype(str)


def subset_indices(samples, n: int | None, seed: int) -> list[int]:
    """Seeded subset of about n windows, stratified so every group keeps at least one window."""
    idx = np.arange(len(samples))
    if n is None or n >= len(samples):
        return idx.tolist()
    rng = np.random.default_rng(seed)
    keys = np.asarray([s.meta["generator"] if s.meta.get("family") == "real" else f"p{s.period}" for s in samples])
    keep = []
    for k in sorted(set(keys)):
        members = idx[keys == k]
        take = max(1, int(round(n * len(members) / len(samples))))
        keep += sorted(rng.permutation(members)[:take].tolist())
    return sorted(keep)


def part_path(out_dir: str, set_name: str, forecaster: str, k: int, tag: str = "") -> str:
    """Cache file of chunk `k`. A forecaster run on a subset carries a tag (`sub<N>s<seed>_`) in the chunk
    part of the name, so subset chunks and full-set chunks can never be mistaken for each other on a resume."""
    return os.path.join(out_dir, "parts", f"{set_name}__{forecaster}__{tag}{k:03d}.csv")


def run_scores(set_name: str, forecaster: str, samples, ids: list[int], out_dir: str, chunk: int = 40,
               worker: tuple[int, int] = (0, 1), tag: str = "") -> float:
    """Score one forecaster on the chosen windows in chunks; each chunk is cached under `out_dir/parts`
    so an interrupted run resumes where it stopped. `worker=(i, n)` runs only the chunks whose number is i modulo n,
    so n processes can share one set without repeating work. `tag` goes into the chunk file names (see `part_path`). Returns the seconds spent in this call."""
    t0 = time.time()
    for k in range(0, len(ids), chunk):
        part = part_path(out_dir, set_name, forecaster, k // chunk, tag)
        if (k // chunk) % worker[1] != worker[0] or os.path.exists(part):
            continue
        cid = ids[k:k + chunk]
        sub = [samples[i] for i in cid]
        df = score_samples(sub, [forecaster])
        df["set"] = set_name
        df["period"] = [s.period for s in sub for _ in range(s.context.shape[0])]
        df["window_index"] = [window_number(samples, i) for i in cid for _ in range(samples[i].context.shape[0])]
        df["family_analogue"] = [samples[i].meta.get("family_analogue", "") for i in cid for _ in range(samples[i].context.shape[0])]
        df["domain"] = [samples[i].meta.get("domain", "") for i in cid for _ in range(samples[i].context.shape[0])]
        os.makedirs(os.path.dirname(part), exist_ok=True)
        df.to_csv(part + ".tmp", index=False)
        os.replace(part + ".tmp", part)
        print(f"    chunk {k // chunk} ({len(cid)} windows) done, {time.time() - t0:.0f}s", flush=True)
    return time.time() - t0


# ----------------------------------------------------------------------------- aggregation


def _gmean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if x.size else float("nan")


def _finite_mean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _summarize_one(d: pd.DataFrame, set_name: str) -> pd.DataFrame:
    d = d.copy()
    d["group"] = group_of(d, set_name)
    agg = d.groupby(["group", "forecaster"]).agg(
        n=("mase", "size"), T_mean=("T", "mean"), H_mean=("H", "mean"),
        **{m: (m, _finite_mean) for m in METRIC_COLS}).reset_index()
    refs = {"snaive": "seasonal_naive"}
    if set_name == "in_prior":
        refs["oracle"] = "oracle"
    rel_cols = []
    for tag, ref in refs.items():
        base = agg[agg.forecaster == ref].set_index("group")
        for m in REL_METRICS:
            col = f"{m}_rel_{tag}"
            rel_cols.append(col)
            agg[col] = [r[m] / base.loc[r["group"], m] if r["group"] in base.index and base.loc[r["group"], m] > 0
                        else np.nan for _, r in agg.iterrows()]
    agg.insert(0, "set", set_name)
    allrows = []
    for f, g in agg.groupby("forecaster"):
        r = {"set": set_name, "group": "ALL", "forecaster": f, "n": int(g["n"].sum()), "n_groups": int(len(g)),
             "T_mean": np.nan, "H_mean": np.nan}
        dd = d[d.forecaster == f]
        for m in METRIC_COLS:
            r[m] = _finite_mean(dd[m])
        for c in rel_cols:
            r[c] = _gmean(g[c])
        allrows.append(r)
    return pd.concat([agg, pd.DataFrame(allrows)], ignore_index=True)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Per set, group and forecaster: window count, mean raw metrics, relative scores; plus an `ALL` row
    per forecaster holding the overall (geometric mean over groups) relative scores.

    Two bases, in column `basis`:
    - `own`: each forecaster on the windows it scored, with the references (seasonal naive, oracle)
      restricted to the same windows. Forecasters run on a subset are therefore compared with seasonal
      naive on that subset.
    - `common`: every forecaster on the windows that all forecasters of the set scored (like for like).
    """
    out = []
    for set_name, d in scores.groupby("set", sort=False):
        d = d.copy()
        parts = []
        for f, df in d.groupby("forecaster"):
            ids = set(df["window_index"])
            sub = d[d["window_index"].isin(ids) & d["forecaster"].isin([f, "seasonal_naive", "oracle"])]
            r = _summarize_one(sub, set_name)
            parts.append(r[r.forecaster == f])
        own = pd.concat(parts, ignore_index=True)
        own.insert(1, "basis", "own")
        nf = d["forecaster"].nunique()
        cnt = d.groupby("window_index")["forecaster"].nunique()
        common_ids = cnt[cnt == nf].index
        com = _summarize_one(d[d["window_index"].isin(common_ids)], set_name)
        com.insert(1, "basis", "common")
        out += [own, com]
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- README tables


def md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else format(v, fmt.get(c, ".4g")))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


README = """# Regression baseline: Timestral sizes and classical models

Baseline scores of the four released Timestral sizes (5M, 28M, 123M, 209M), seasonal naive, ETS and ARIMA
on the regression sets in `bench/regression/` (see the README there for how the sets are built). A fine-tuned
model is checked against these numbers to see whether normal forecasting got worse. Real-data numbers are not
a leaderboard claim; the data may overlap Timestral's pretraining data in unknown ways.

Files: `scores.csv` (one row per window and forecaster), `summary.csv` (the tables below),
`provenance.json` (command, commit, host, compute seconds, subsets). Produced by `scripts/score_regression.py`;
`parts/` holds the cached chunks and is not committed.

Metrics (definitions in `otp/metrics.py`): MASE is the mean absolute error of the median forecast divided by the
in-sample seasonal-naive error of the context. sCRPS is the average pinball loss over the 21 quantile levels, times 2,
divided by the same scale. WQL is the GIFT-Eval weighted quantile loss on levels 0.1 to 0.9. cov80 is the fraction of
horizon points inside the 0.1 to 0.9 interval (0.8 is perfectly calibrated). calib is the mean over the 21 levels of
the absolute gap between empirical coverage and the level (0 is perfect).

Aggregation, as GIFT-Eval does it: for each group (dataset/term pair in the real sets, family/dataset in the
analogue set, seasonal period in the in-prior set) a forecaster's score is the mean over windows of its MASE or
sCRPS; the relative score is that value divided by seasonal naive's value for the same group; the overall relative
score is the geometric mean of the group relative scores over groups. The `ALL` raw metrics are means over windows.
Windows with a non-finite metric value are left out of that metric (WQL is undefined for windows whose horizon
values are all zero). The oracle exists only for the in-prior set, where `*_rel_oracle` is computed the same way
against it; it is never used for the real sets.

`summary.csv` has two bases. `own`: each forecaster on the windows it scored, with seasonal naive (and the oracle)
restricted to the same windows. `common`: every forecaster on the windows that all forecasters of the set scored.
The tables show the `common` basis, then the `own` basis.
{notes}
{sections}
"""

ORDER = [f"timestral-{s}" for s in SIZES] + list(CLASSICAL) + ["oracle"]
SET_TITLES = {"real": "Real set (GIFT-Eval and fev-bench)", "real_extra": "Extra real set (further open-license sources)",
              "analogues": "Analogue subset (real series standing in for the synthetic failure families)", "in_prior": "In-prior set"}


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(_o=df.forecaster.map({f: i for i, f in enumerate(ORDER)})).sort_values("_o").drop(columns="_o")


def write_readme(out_dir: str, summ: pd.DataFrame, notes: str) -> None:
    cols = ["forecaster", "n", "mase", "scrps", "wql", "cov80", "calib", "mase_rel_snaive", "scrps_rel_snaive"]
    pcols = cols + ["mase_rel_oracle", "scrps_rel_oracle"]
    allrows = summ[summ.group == "ALL"]
    parts = []
    for set_name in SETS:
        if set_name not in set(summ.set):
            continue
        c = pcols if set_name == "in_prior" else cols
        com = _sorted(allrows[(allrows.set == set_name) & (allrows.basis == "common")])
        own = _sorted(allrows[(allrows.set == set_name) & (allrows.basis == "own")])
        nwin = int(com["n"].max()) if len(com) else 0
        ng = int(com["n_groups"].max()) if len(com) else 0
        parts.append(f"## {SET_TITLES[set_name]}, common windows ({nwin} windows, {ng} groups)\n\n"
                     + md_table(com, [x for x in c if x in com.columns]))
        parts.append(f"## {SET_TITLES[set_name]}, each forecaster on its own windows\n\n"
                     + md_table(own, [x for x in c if x in own.columns]))
        if set_name != "in_prior":
            gr = summ[(summ.set == set_name) & (summ.group != "ALL") & (summ.basis == "own")]
            for m, lab in (("mase_rel_snaive", "MASE"), ("scrps_rel_snaive", "sCRPS")):
                t = gr.pivot(index="group", columns="forecaster", values=m).reset_index()
                cc = ["group"] + [x for x in ORDER if x in t.columns]
                parts.append(f"### {SET_TITLES[set_name]}: relative {lab} to seasonal naive per group (own windows)\n\n" + md_table(t[cc], cc))
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(README.format(notes=notes, sections="\n\n".join(parts)))


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["real", "in_prior"], choices=list(SETS))
    ap.add_argument("--only", choices=["timestral", "classical"], default=None, help="run only these forecasters")
    ap.add_argument("--sizes", nargs="+", default=list(SIZES))
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--out", default="results/regression_baseline")
    ap.add_argument("--subset", nargs="*", default=[], metavar="FORECASTER=N",
                    help="score this forecaster on a seeded stratified subset of about N windows per set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker", default="0/1", metavar="I/N", help="run only chunks with number I modulo N (N processes share the work)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--parts-only", action="store_true", help="run the forecasters and cache the chunks, skip assembling tables")
    a = ap.parse_args()
    set_threads(1)
    os.makedirs(a.out, exist_ok=True)
    worker = tuple(int(x) for x in a.worker.split("/"))
    sub_n = {k: int(v) for k, v in (s.split("=") for s in a.subset)}
    t_start = time.time()
    if not a.summarize_only:
        for set_name in a.sets:
            samples = load_set(set_name, a.bench)
            for f in forecasters_for(set_name, a.sizes):
                if a.only == "timestral" and not f.startswith("timestral"):
                    continue
                if a.only == "classical" and f.startswith("timestral"):
                    continue
                ids = subset_indices(samples, sub_n.get(f), a.seed)
                print(f"[{set_name}] {f}: {len(ids)}/{len(samples)} windows", flush=True)
                tag = f"sub{sub_n[f]}s{a.seed}_" if f in sub_n and len(ids) < len(samples) else ""
                dt = run_scores(set_name, f, samples, ids, a.out, worker=worker, tag=tag)
                print(f"    {f} done in {dt:.0f}s this call", flush=True)
    if a.parts_only:
        return
    pdir = os.path.join(a.out, "parts")
    parts = sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []
    scores = pd.concat([pd.read_csv(os.path.join(pdir, p)) for p in parts if p.endswith(".csv")], ignore_index=True)
    dup = scores.duplicated(["set", "forecaster", "window_index", "variate"], keep=False)
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} rows repeat a (set, forecaster, window, variate): stale or overlapping chunk files in {pdir}")
    scores["set"] = pd.Categorical(scores["set"], list(SETS))
    scores = scores.sort_values(["set", "window_index", "forecaster"]).reset_index(drop=True)
    scores["set"] = scores["set"].astype(str)
    scores.to_csv(os.path.join(a.out, "scores.csv"), index=False)
    summ = summarize(scores)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    nwin = scores.groupby(["set", "forecaster"])["window_index"].nunique()
    total = {s: int(scores[scores.set == s]["window_index"].nunique()) for s in scores.set.unique()}
    lines = []
    for s in total:
        sub = {f: int(n) for (ss, f), n in nwin.items() if ss == s}
        if len(set(sub.values())) > 1:
            lines.append(f"- {s} set ({total[s]} windows): windows scored per forecaster: "
                         + ", ".join(f"{k} {v}" for k, v in sub.items()))
    notes = ""
    if lines:
        notes = ("\nSubsets. Forecasters not run on every window were run on a seeded subset stratified by group "
                 "(seed " + str(a.seed) + ", `--subset` in provenance.json):\n\n" + "\n".join(lines) + "\n")
    for s_name, d in scores[scores.forecaster == "seasonal_naive"].groupby("set", sort=False):
        w = d["wql"].astype(float)
        top = d.loc[w.idxmax()]
        if w.max() > 100 * w.median():
            notes += (f"\nWQL note, {s_name} set: the raw WQL columns are means over windows. For seasonal naive the median window WQL is "
                      f"{w.median():.4g} and the largest is {w.max():.4g} (window `{top['sample_id']}`, MASE {top['mase']:.4g}), so the mean WQL "
                      "is set by a few windows with a near-zero or zero in-sample scale; MASE, sCRPS and the relative scores are the columns to compare.\n")
    plat = os.path.join(a.out, "chunk_platforms.csv")
    if os.path.exists(plat):
        pl = pd.read_csv(plat)
        notes += ("\nPlatforms. Chunks of 40 windows were scored on different machines (`chunk_platforms.csv` lists set, forecaster, "
                  "chunk numbers and platform). Seasonal naive and the oracle are deterministic. ETS is fitted by a numerical optimizer and "
                  "is not bit-identical across platforms: a cross-check of one ETS chunk between the laptop and the VM found small "
                  "differences in per-window scores, while ARIMA agreed more closely. So the ETS rows of the real set mix 3 laptop chunks "
                  "(Windows) with 8 VM chunks (Linux), and those of the in-prior set mix 2 laptop chunks with 11 VM chunks. Timestral "
                  "was run on CPU (VM) for the real and in-prior sets and on a Colab T4 GPU for the extra real and analogue sets.\n\n"
                  + md_table(pl, list(pl.columns)) + "\n")
    write_readme(a.out, summ, notes)
    comp = {f"{s}/{f}": float(d.groupby("window_index")["sec_per_sample"].first().sum())
            for (s, f), d in scores.groupby(["set", "forecaster"])}
    prov = provenance("scripts/score_regression.py", argv=sys.argv[1:], sets=a.sets, sizes=a.sizes, subset=sub_n,
                      seed=a.seed, chunk_platforms_file="results/regression_baseline/chunk_platforms.csv",
                      wall_seconds_this_invocation=time.time() - t_start, compute_seconds=comp,
                      compute_seconds_total=float(sum(comp.values())),
                      windows_scored={f"{s}/{f}": int(n) for (s, f), n in nwin.items()})
    write_json(os.path.join(a.out, "provenance.json"), prov)
    print(f"wrote {len(scores)} score rows, {len(summ)} summary rows to {a.out}")


if __name__ == "__main__":
    main()
