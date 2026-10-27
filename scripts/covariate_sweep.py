"""Covariate sweep: does one-way covariate attention cost accuracy on coupled systems?

For every sample (a coupled system at one knob value and seed) Timestral forecasts the main series y
under up to four input conditions:

  uni        y alone
  past_cov   x as a past-only covariate (x's rows never read y)
  known_cov  x as a known-future covariate (only for systems where x's future can be given)
  joint      x as a second main row, context (B, 2, T): y and x read each other; only row 0 is scored

The true-dynamics oracle and the classical models (seasonal naive, ETS, ARIMA) are scored on y alone.

Commands (all write under --out, default results/covariates):
  samples   build the samples and save them in <out>/samples/<system>.npz (seeded, deterministic)
  refs      score the oracles and classical models                 -> <out>/parts/refs__<system>__T<T>.csv
  models    score Timestral under the conditions                   -> <out>/parts/models__<label>__<size>__T<T>__<system>.csv
  merge     join the parts, add ratios, write scores.csv, summary.csv, tables.md
Existing part files are skipped, so an interrupted run resumes.

Example (see results/covariates/README.md for the exact commands that produced the committed files):
  python scripts/covariate_sweep.py samples
  python scripts/covariate_sweep.py refs --contexts 512 2048 8192
  python scripts/covariate_sweep.py models --label full --sizes 5M 28M --contexts 2048
  python scripts/covariate_sweep.py merge
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp import analysis, metrics  # noqa: E402
from otp.classical import CLASSICAL, classical_forecast  # noqa: E402
from otp.evaluate import METRIC_COLS  # noqa: E402
from otp.generators import coupled as C  # noqa: E402
from otp.io import provenance, read_json, write_json  # noqa: E402
from otp.model import forecast, set_threads  # noqa: E402

CONDITIONS = ("uni", "past_cov", "known_cov", "joint")
CONTEXT_CHECK_SYSTEMS = ("LaggedDriver", "FeedbackLinear", "FeedbackPredPrey")
HMAX = 1024
COND_H = (16, 128, 1024)
N_FULL_SEEDS = 8


# ------------------------------------------------------------------------------------------ helpers

def sample_path(out: str, system: str) -> str:
    return os.path.join(out, "samples", f"{system}.npz")


def selected(samples, grid: str, seeds: int):
    """Subset of the saved samples: `reduced` keeps knob values 1, 3 and 5 of the 5, and the first `seeds` seeds."""
    if grid == "full" and seeds >= N_FULL_SEEDS:
        return list(samples)
    vals = None
    out = []
    for s in samples:
        system = s.meta["system"]
        vals = C.SYSTEMS[system]["values"]
        keep = [vals[i] for i in C.REDUCED_KNOB_IDX] if grid == "reduced" else list(vals)
        if s.meta["knob_value"] in keep and int(s.meta["sample_id"].rsplit("/s", 1)[1]) < seeds:
            out.append(s)
    return out


def contexts_for(system: str, requested: list[int]) -> list[int]:
    t_gen = C.SYSTEMS[system]["t_gen"]
    return [t for t in requested if t <= t_gen and (t == 2048 or system in CONTEXT_CHECK_SYSTEMS)]


def base_row(s: C.CovSample, forecaster: str, condition: str, size: str, t: int, h: int, v: int = 0) -> dict:
    return {"sample_id": s.meta["sample_id"], "system": s.meta["system"], "config_id": s.meta["config_id"],
            "knob": s.meta["knob"], "knob_value": s.meta["knob_value"], "seed": s.meta["seed"],
            "forecaster": forecaster, "condition": condition, "size": size, "T": t, "H": h, "variate": v}


def score_rows(s: C.CovSample, q: np.ndarray, forecaster: str, condition: str, size: str, t: int, horizons,
               extra: dict | None = None, per_h_q=None) -> list[dict]:
    """One row per horizon. `q` is `(1, Hmax, 21)`; horizons are prefixes. `per_h_q(h)` overrides `q` per horizon."""
    ctx = s.context[..., -t:]
    rows = []
    for h in horizons:
        qh = per_h_q(h) if per_h_q is not None else q[..., :h, :]
        sc = metrics.score(ctx, s.future[..., :h], qh, s.period)
        r = base_row(s, forecaster, condition, size, t, h)
        r.update({k: float(sc[k][0]) for k in METRIC_COLS})
        r.update(extra or {})
        rows.append(r)
    return rows


def slice_for(s: C.CovSample, t: int, h: int):
    """Context, past covariate and known covariate cut to the last `t` context steps and `h` horizon steps."""
    ctx = s.context[..., -t:]
    pc = s.past_cov[..., -t:]
    kc = None if s.known_cov is None else s.known_cov[..., s.T - t: s.T + h]
    return ctx, pc, kc


def update_provenance(out: str, key: str, entry: dict) -> None:
    path = os.path.join(out, "provenance.json")
    cur = read_json(path) if os.path.exists(path) else {"runs": {}}
    cur["runs"][key] = entry
    write_json(path, cur)


# ------------------------------------------------------------------------------------------ commands

def cmd_samples(a):
    t0 = time.time()
    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True)
    n = 0
    for system in a.systems:
        spec = C.SYSTEMS[system]
        ss = [C.build_sample(system, v, seed, horizon=HMAX, cond_horizons=COND_H, n_paths=a.n_paths)
              for v in spec["values"] for seed in range(N_FULL_SEEDS)]
        C.save_cov_samples(sample_path(a.out, system), ss)
        n += len(ss)
        print(f"{system}: {len(ss)} samples ({time.time() - t0:.0f}s)", flush=True)
    update_provenance(a.out, "samples", provenance("scripts/covariate_sweep.py", argv=sys.argv, n_samples=n,
                                                   wall_sec=round(time.time() - t0, 1)))


def _refs_one(system: str, a):
    ss = C.load_cov_samples(sample_path(a.out, system))
    ts = contexts_for(system, a.contexts)
    rows = []
    for i, s in enumerate(ss):
        base = s.context[..., -1024:][None]                              # classical fits use the last 1024 steps
        cls = {name: classical_forecast(name, base, HMAX, s.period)[0] for name in a.classical}
        for t in ts:
            rows += score_rows(s, s.oracle_q, "oracle", "ref", "-", t, a.horizons,
                               per_h_q=lambda h, s=s: s.oracle_for(True, h))
            rows += score_rows(s, s.oracle_marg_q, "oracle_marg", "ref", "-", t, a.horizons)
            for name, q in cls.items():
                rows += score_rows(s, q, name, "ref", "-", t, a.horizons)
        if (i + 1) % 10 == 0:
            print(f"  refs {system} {i + 1}/{len(ss)}", flush=True)
    return rows


def cmd_refs(a):
    t0 = time.time()
    os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
    for system in a.systems:
        ts = contexts_for(system, a.contexts)
        path = os.path.join(a.out, "parts", f"refs__{system}__T{'-'.join(map(str, ts))}.csv")
        if os.path.exists(path):
            print("skip", path)
            continue
        pd.DataFrame(_refs_one(system, a)).to_csv(path, index=False)
        print(f"refs {system} done ({time.time() - t0:.0f}s)", flush=True)
    update_provenance(a.out, "refs", provenance("scripts/covariate_sweep.py", argv=sys.argv,
                                                wall_sec=round(time.time() - t0, 1)))


def _run_condition(size, ss, cond, t, horizons, bs):
    """Quantile forecasts for one condition: dict h -> `(N, 1, h, 21)`. Prefix-exact conditions run once at max horizon."""
    hmax = max(horizons)
    ctx = np.stack([slice_for(s, t, hmax)[0] for s in ss])                    # (N, 1, T)
    pc = np.stack([slice_for(s, t, hmax)[1] for s in ss])                     # (N, 1, T)
    if cond == "uni":
        q = forecast(size, ctx, hmax, batch_size=bs)[:, :1]
    elif cond == "past_cov":
        q = forecast(size, ctx, hmax, past_covariates=pc, batch_size=bs)[:, :1]
    elif cond == "joint":
        q = forecast(size, np.concatenate([ctx, pc], axis=1), hmax, batch_size=bs)[:, :1]
    elif cond == "known_cov":
        out = {}
        for h in horizons:               # the known future has length h, so each horizon is its own call
            kc = np.stack([slice_for(s, t, h)[2] for s in ss])
            out[h] = forecast(size, ctx, h, known_covariates=kc, batch_size=bs)[:, :1]
        return out
    else:
        raise ValueError(cond)
    return {h: q[:, :, :h] for h in horizons}


def cmd_models(a):
    t0 = time.time()
    os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
    nrows = 0
    for size in a.sizes:
        for t in a.contexts:
            for system in a.systems:
                if t not in contexts_for(system, [t]):
                    continue
                path = os.path.join(a.out, "parts", f"models__{a.label}__{size}__T{t}__{system}.csv")
                if os.path.exists(path):
                    print("skip", path)
                    continue
                ss = selected(C.load_cov_samples(sample_path(a.out, system)), a.grid, a.seeds)
                conds = [c for c in a.conditions if c != "known_cov" or C.SYSTEMS[system]["known"]]
                rows = []
                for cond in conds:
                    tc = time.time()
                    qs = _run_condition(size, ss, cond, t, a.horizons, a.batch_size)
                    dt = (time.time() - tc) / len(ss)
                    for j, s in enumerate(ss):
                        for h in a.horizons:
                            rows += score_rows(s, qs[h][j], f"timestral-{size}", cond, size, t, [h],
                                               extra={"sec_per_sample": dt})
                    print(f"{size} T={t} {system} {cond}: {len(ss)} samples, {dt:.1f}s/sample "
                          f"(total {time.time() - t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(path, index=False)
                nrows += len(rows)
    update_provenance(a.out, f"models:{a.label}", provenance(
        "scripts/covariate_sweep.py", argv=sys.argv, sizes=a.sizes, contexts=a.contexts, grid=a.grid, seeds=a.seeds,
        horizons=a.horizons, n_rows=nrows, wall_sec=round(time.time() - t0, 1)))


# ------------------------------------------------------------------------------------------ merge

def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios to the oracle and to the best classical model, with the oracle matched on T (otp.analysis, per T)."""
    parts = []
    for t, g in df.groupby("T"):
        g = analysis.ratios(g[g.forecaster != "oracle_marg"].reset_index(drop=True))
        marg = df[(df["T"] == t) & (df.forecaster == "oracle_marg")][["sample_id", "H", "variate", "mase", "scrps"]]
        marg = marg.rename(columns={"mase": "mase_oracle_marg", "scrps": "scrps_oracle_marg"})
        g = g.merge(marg, on=["sample_id", "H", "variate"], how="left")
        for m in ("mase", "scrps"):
            g[f"{m}_ratio_oracle_marg"] = g[m] / g[f"{m}_oracle_marg"].clip(lower=1e-9)
            known = g.condition == "known_cov"
            g[f"{m}_ratio_oracle_matched"] = np.where(known, g[f"{m}_ratio_oracle"], g[f"{m}_ratio_oracle_marg"])
            g[f"{m}_oracle_matched"] = np.where(known, g[f"{m}_oracle"], g[f"{m}_oracle_marg"])
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _agg(x) -> tuple[float, float, float]:
    lo, hi = analysis._boot_gm(x)
    return analysis._gm(x), lo, hi


def build_summary(sc: pd.DataFrame) -> pd.DataFrame:
    tm = sc[sc.forecaster.str.startswith("timestral-")].copy()
    key = ["system", "knob", "knob_value", "size", "T", "H"]
    uni = tm[tm.condition == "uni"][["sample_id", "size", "T", "H", "mase"]].rename(columns={"mase": "mase_uni"})
    pcv = tm[tm.condition == "past_cov"][["sample_id", "size", "T", "H", "mase"]].rename(columns={"mase": "mase_past"})
    tm = tm.merge(uni, on=["sample_id", "size", "T", "H"], how="left").merge(pcv, on=["sample_id", "size", "T", "H"], how="left")
    tm["gain"] = tm.mase_uni / tm.mase.clip(lower=1e-9)
    tm["oracle_gain_matched"] = tm.mase_uni / tm.mase_oracle_matched.clip(lower=1e-9)
    tm["joint_vs_past"] = np.where(tm.condition == "joint", tm.mase / tm.mase_past.clip(lower=1e-9), np.nan)
    rows = []
    for k, g in tm.groupby(key + ["condition"]):
        r = dict(zip(key + ["condition"], k))
        r["n"] = len(g)
        for name, col in (("mase_ratio_oracle", "mase_ratio_oracle_matched"), ("mase_ratio_oracle_cond", "mase_ratio_oracle"),
                          ("scrps_ratio_oracle", "scrps_ratio_oracle_matched"), ("mase_ratio_classical", "mase_ratio_classical"),
                          ("covariate_gain", "gain"), ("oracle_gain", "oracle_gain_matched"), ("joint_vs_past", "joint_vs_past")):
            x = g[col].dropna()
            if len(x):
                m, lo, hi = _agg(x)
                r[f"{name}_gmean"], r[f"{name}_lo95"], r[f"{name}_hi95"] = m, lo, hi
        r["mase_mean"] = float(g.mase.mean())
        rows.append(r)
    ref = sc[sc.forecaster.isin(CLASSICAL_NAMES)].copy()
    if len(ref):
        best = ref.groupby(["sample_id", "T", "H"], as_index=False).agg(
            system=("system", "first"), knob=("knob", "first"), knob_value=("knob_value", "first"),
            mase=("mase", "min"), r=("mase_ratio_oracle_marg", "min"))
        for k, g in best.groupby(["system", "knob", "knob_value", "T", "H"]):
            m, lo, hi = _agg(g.r)
            rows.append({**dict(zip(["system", "knob", "knob_value", "T", "H"], k)), "size": "-",
                         "condition": "classical_best", "n": len(g), "mase_ratio_oracle_gmean": m,
                         "mase_ratio_oracle_lo95": lo, "mase_ratio_oracle_hi95": hi, "mase_mean": float(g.mase.mean())})
    return pd.DataFrame(rows)


CLASSICAL_NAMES = tuple(CLASSICAL)


def fmt(m, lo=None, hi=None, d=2):
    if m is None or not np.isfinite(m):
        return "-"
    s = f"{m:.{d}f}"
    return s if lo is None or not np.isfinite(lo) else f"{s} [{lo:.{d}f}, {hi:.{d}f}]"


def _cell(row, name, d=2):
    if row is None or f"{name}_gmean" not in row or not np.isfinite(row[f"{name}_gmean"]):
        return "-"
    return fmt(row[f"{name}_gmean"], row.get(f"{name}_lo95"), row.get(f"{name}_hi95"), d)


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def build_tables(sm: pd.DataFrame) -> str:
    """Markdown tables copied from summary.csv values."""
    idx = sm.set_index(["system", "knob_value", "size", "T", "H", "condition"]).sort_index()
    look = {k: r for k, r in zip(idx.index, idx.to_dict("records"))}
    sizes = [s for s in ("5M", "28M", "123M", "209M") if s in set(sm["size"])]
    out = []
    knobs = {k: (C.SYSTEMS[k]["knob"], C.SYSTEMS[k]["values"]) for k in C.SYSTEMS}

    def get(system, v, size, t, h, cond):
        return look.get((system, v, size, t, h, cond))

    out.append("## Table 1. Covariate gain per system and knob (T = 2048)\n")
    out.append("Covariate gain = MASE(uni) / MASE(condition), geometric mean over samples, with 95% bootstrap CI in "
               "brackets. Above 1 means the covariate helped. `oracle` is the same ratio for the oracle that matches "
               "the condition's information (MASE(uni) / MASE(oracle)): the largest gain available.\n")
    for size in sizes:
        for h in sorted(set(sm[(sm["size"] == size) & (sm["T"] == 2048)]["H"])):
            out.append(f"### Size {size}, H = {h}\n")
            rows = []
            for system, (kn, vals) in knobs.items():
                for v in vals:
                    if get(system, v, size, 2048, h, "uni") is None:
                        continue
                    r = [system, f"{kn}={v:g}"]
                    for cond in ("past_cov", "known_cov", "joint"):
                        r.append(_cell(get(system, v, size, 2048, h, cond), "covariate_gain"))
                    og = get(system, v, size, 2048, h, "known_cov") or get(system, v, size, 2048, h, "past_cov")
                    r.append(_cell(og, "oracle_gain"))
                    rows.append(r)
            out.append(md_table(["system", "knob", "past_cov", "known_cov", "joint", "oracle (past_cov, or known_cov where offered)"], rows))
    out.append("## Table 1b. MASE ratio to the oracle per condition (T = 2048)\n")
    out.append("Geometric mean over samples [95% bootstrap CI] of MASE(Timestral) / MASE(oracle matched to the "
               "condition's information). `classical` is the best of seasonal naive, ETS, ARIMA per sample over the "
               "same oracle (marginal oracle).\n")
    for size in sizes:
        for h in sorted(set(sm[(sm["size"] == size) & (sm["T"] == 2048)]["H"])):
            out.append(f"### Size {size}, H = {h}\n")
            rows = []
            for system, (kn, vals) in knobs.items():
                for v in vals:
                    if get(system, v, size, 2048, h, "uni") is None:
                        continue
                    r = [system, f"{kn}={v:g}"]
                    for cond in CONDITIONS:
                        r.append(_cell(get(system, v, size, 2048, h, cond), "mase_ratio_oracle"))
                    r.append(_cell(get(system, v, "-", 2048, h, "classical_best"), "mase_ratio_oracle"))
                    rows.append(r)
            out.append(md_table(["system", "knob", "uni", "past_cov", "known_cov", "joint", "classical"], rows))
    out.append("## Table 2. Joint versus past_cov on the Feedback systems (T = 2048)\n")
    out.append("`joint/past_cov` = MASE(joint) / MASE(past_cov) per sample, geometric mean [95% CI]; below 1 means "
               "two-way attention is more accurate. The two ratio columns are MASE ratios to the oracle.\n")
    for system in ("FeedbackLinear", "FeedbackPredPrey"):
        kn, vals = knobs[system]
        for size in sizes:
            hs = sorted(set(sm[(sm["size"] == size) & (sm["T"] == 2048) & (sm.system == system)]["H"]))
            if not hs:
                continue
            out.append(f"### {system}, size {size}\n")
            rows = []
            for v in vals:
                for h in hs:
                    j, p = get(system, v, size, 2048, h, "joint"), get(system, v, size, 2048, h, "past_cov")
                    if j is None:
                        continue
                    rows.append([f"{kn}={v:g}", h, _cell(p, "mase_ratio_oracle"), _cell(j, "mase_ratio_oracle"),
                                 _cell(j, "joint_vs_past", 3), j["n"]])
            out.append(md_table(["knob", "H", "past_cov / oracle", "joint / oracle", "joint/past_cov", "n"], rows))
    out.append("## Table 3. LeakyKnown: distance to the oracle when the known covariate reveals the future (T = 2048)\n")
    out.append("Knob `noise_ratio` = noise sd of x over the innovation sd of y. MASE ratio to the oracle matched to "
               "the condition (the exact posterior given x's future for `known_cov`; the marginal oracle otherwise), "
               "geometric mean [95% CI]. `oracle gain` = MASE(uni) / MASE(known_cov oracle).\n")
    for size in sizes:
        hs = sorted(set(sm[(sm["size"] == size) & (sm["T"] == 2048) & (sm.system == "LeakyKnown")]["H"]))
        if not hs:
            continue
        out.append(f"### Size {size}\n")
        rows = []
        for v in knobs["LeakyKnown"][1]:
            for h in hs:
                kc = get("LeakyKnown", v, size, 2048, h, "known_cov")
                if kc is None:
                    continue
                rows.append([f"noise_ratio={v:g}", h] + [_cell(get("LeakyKnown", v, size, 2048, h, c), "mase_ratio_oracle")
                                                           for c in CONDITIONS] + [_cell(kc, "oracle_gain"), _cell(kc, "covariate_gain")])
        out.append(md_table(["knob", "H", "uni", "past_cov", "known_cov", "joint", "oracle gain", "known_cov gain"], rows))
    ctx = sm[(sm["size"] == "5M") & (sm.system.isin(CONTEXT_CHECK_SYSTEMS)) & (sm["T"].isin([512, 8192]))]
    if len(ctx):
        out.append("## Table 4. Context-length check, 5M (T = 512, 2048, 8192)\n")
        out.append("Covariate gain [95% CI] and MASE ratio to the oracle per condition, same samples at every T.\n")
        for system in CONTEXT_CHECK_SYSTEMS:
            kn, vals = knobs[system]
            out.append(f"### {system}\n")
            rows = []
            for v in vals:
                for h in sorted(set(ctx[ctx.system == system]["H"])):
                    for t in (512, 2048, 8192):
                        if get(system, v, "5M", t, h, "uni") is None:
                            continue
                        r = [f"{kn}={v:g}", h, t]
                        for cond in ("past_cov", "known_cov", "joint"):
                            c = get(system, v, "5M", t, h, cond)
                            r.append(f"{_cell(c, 'covariate_gain')} / {_cell(c, 'mase_ratio_oracle')}" if c else "-")
                        r.append(_cell(get(system, v, "5M", t, h, "uni"), "mase_ratio_oracle"))
                        rows.append(r)
            out.append(md_table(["knob", "H", "T", "past_cov: gain / ratio", "known_cov: gain / ratio",
                                 "joint: gain / ratio", "uni: ratio"], rows))
    return "\n".join(out)


def cmd_merge(a):
    t0 = time.time()
    refs = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(a.out, "parts", "refs__*.csv")))]
    mods = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(a.out, "parts", "models__*.csv")))]
    df = pd.concat(refs + mods, ignore_index=True)
    # keep reference rows only for the (sample, T, H) cells that some Timestral row covers
    cells = df[df.forecaster.str.startswith("timestral-")][["sample_id", "T", "H"]].drop_duplicates()
    df = df.merge(cells.assign(_k=1), on=["sample_id", "T", "H"], how="left")
    df = df[(df._k == 1)].drop(columns="_k").reset_index(drop=True)
    sc = add_ratios(df)
    sc = sc.sort_values(["system", "knob_value", "seed", "T", "H", "size", "forecaster", "condition"]).reset_index(drop=True)
    sc.to_csv(os.path.join(a.out, "scores.csv"), index=False)
    sm = build_summary(sc)
    sm.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write(build_tables(sm))
    update_provenance(a.out, "merge", provenance("scripts/covariate_sweep.py", argv=sys.argv, n_rows_scores=len(sc),
                                                 n_rows_summary=len(sm), wall_sec=round(time.time() - t0, 1)))
    print(f"scores.csv {len(sc)} rows; summary.csv {len(sm)} rows ({time.time() - t0:.0f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["samples", "refs", "models", "merge"])
    ap.add_argument("--out", default="results/covariates")
    ap.add_argument("--systems", nargs="+", default=list(C.SYSTEMS))
    ap.add_argument("--sizes", nargs="+", default=["5M"])
    ap.add_argument("--contexts", nargs="+", type=int, default=[2048])
    ap.add_argument("--horizons", nargs="+", type=int, default=[16, 128, 1024])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--grid", choices=["full", "reduced"], default="full")
    ap.add_argument("--seeds", type=int, default=N_FULL_SEEDS)
    ap.add_argument("--label", default="full")
    ap.add_argument("--classical", nargs="*", default=list(CLASSICAL))
    ap.add_argument("--n-paths", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    set_threads(a.threads)
    {"samples": cmd_samples, "refs": cmd_refs, "models": cmd_models, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
