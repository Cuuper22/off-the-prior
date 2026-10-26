"""Score a forecaster on an Off-the-Prior benchmark set. One command:

  python bench/eval.py --set diag --model timestral-5M
  python bench/eval.py --set lb --split public --model seasonal_naive
  python bench/eval.py --set diag --predictions my_preds/ --name mymodel

--model: timestral-<5M|28M|123M|209M> | seasonal_naive | ets | arima | oracle
--predictions DIR: your own forecasts. For every data file bench/data/<set>_<split>_T<T>_H<H>.npz,
  put a file with the same name in DIR holding `quantiles`, float array (N, H, 21), in the same
  series order as the data file (the data file's `series_id` array), at levels
  0.01, 0.05, 0.1, 0.15, ..., 0.9, 0.95, 0.99.

Scores per series (otp.metrics): MASE, sCRPS, WQL, cov80, calib, against the true-dynamics oracle
and against seasonal naive. Summary (GIFT-Eval style): geometric mean over series of
MASE / MASE(seasonal naive) and sCRPS / sCRPS(seasonal naive), overall and per family (and per
failure type for the diagnostic set); plus the geometric mean of MASE / MASE(oracle), which
GIFT-Eval cannot report. Series whose oracle could not be scored (diverged Monte Carlo paths)
are left out of oracle ratios only.

Writes bench/leaderboard/<name>_<set>_<split>/{scores.csv, summary.json, table.md}.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from otp import metrics  # noqa: E402
from otp.classical import CLASSICAL, classical_forecast  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.model import forecast  # noqa: E402

DATA = os.path.join(ROOT, "bench", "data")


def gm(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if x.size else float("nan")


def _classical_one(job):
    model, x, h, per = job
    return classical_forecast(model, x[None, None], h, per)[0, 0]


WORKERS = 1


def predict(model: str, z, pred_dir: str | None, fname: str) -> np.ndarray:
    ctx = z["context"].astype(np.float64)
    h = z["future"].shape[1]
    if pred_dir:
        q = np.load(os.path.join(pred_dir, fname))["quantiles"]
        assert q.shape == (ctx.shape[0], h, 21), f"{fname}: predictions must be {(ctx.shape[0], h, 21)}, got {q.shape}"
        return q.astype(np.float64)
    if model == "oracle":
        return z["oracle_q"].astype(np.float64)
    if model in CLASSICAL:
        per = z["period"]
        jobs = [(model, ctx[i], h, int(per[i])) for i in range(len(ctx))]
        if WORKERS > 1 and model != "seasonal_naive":
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                return np.stack(list(ex.map(_classical_one, jobs, chunksize=4)))
        return np.stack([_classical_one(j) for j in jobs])
    assert model.startswith("timestral-"), model
    return forecast(model[len("timestral-"):], ctx, h, batch_size=16)[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["diag", "lb"])
    ap.add_argument("--split", default="public", choices=["public", "heldout", "all"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "leaderboard"))
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--workers", type=int, default=1, help="processes for ETS/ARIMA fits")
    a = ap.parse_args()
    global WORKERS
    WORKERS = a.workers
    assert a.model or a.predictions, "give --model or --predictions"
    name = a.name or a.model
    splits = ["public", "heldout"] if a.split == "all" else [a.split]
    files = sorted(f for s in splits for f in glob.glob(os.path.join(a.data, f"{a.set}_{s}_T*_H*.npz")))
    assert files, f"no data files for {a.set}/{a.split}; build them with bench/build.py --set {a.set}"
    rows = []
    for path in files:
        fname = os.path.basename(path)
        with np.load(path, allow_pickle=False) as z:
            z = {k: z[k] for k in z.files}
        ctx, fut = z["context"].astype(np.float64), z["future"].astype(np.float64)
        q = predict(a.model, z, a.predictions, fname)
        q_sn = predict("seasonal_naive", z, None, fname)
        q_or = z["oracle_q"].astype(np.float64)
        for i in range(len(ctx)):
            m = json.loads(str(z["meta"][i]))
            per = int(z["period"][i])
            sc = metrics.score(ctx[i], fut[i], q[i], per)
            sn = metrics.score(ctx[i], fut[i], q_sn[i], per)
            orc = metrics.score(ctx[i], fut[i], q_or[i], per)
            rows.append({"series_id": m["series_id"], "config_id": m["config_id"], "family": m["family"],
                         "failure_type": m.get("failure_type", ""), "T": m["T"], "H": m["H"], "split": m["split"],
                         **{k: float(sc[k]) for k in ("mase", "scrps", "wql", "cov80", "calib")},
                         "mase_rel_snaive": float(sc["mase"] / sn["mase"]), "scrps_rel_snaive": float(sc["scrps"] / sn["scrps"]),
                         "mase_rel_oracle": float(sc["mase"] / orc["mase"]) if np.isfinite(orc["mase"]) else np.nan})
        print(f"{fname}: {len(ctx)} series", flush=True)
    df = pd.DataFrame(rows)
    od = os.path.join(a.out, f"{name}_{a.set}_{a.split}")
    os.makedirs(od, exist_ok=True)
    df.to_csv(os.path.join(od, "scores.csv"), index=False)

    def agg(g):
        return {"n": len(g), "mase_rel_snaive": gm(g.mase_rel_snaive), "scrps_rel_snaive": gm(g.scrps_rel_snaive),
                "mase_rel_oracle": gm(g.mase_rel_oracle), "cov80": float(g.cov80.mean())}

    summary = {"model": name, "set": a.set, "split": a.split, "overall": agg(df),
               "by_family": {k: agg(g) for k, g in df.groupby("family")}}
    if a.set == "diag":
        summary["by_failure_type"] = {k: agg(g) for k, g in df.groupby("failure_type")}
    summary["provenance"] = provenance("bench/eval.py", argv=sys.argv)
    write_json(os.path.join(od, "summary.json"), summary)
    lines = [f"# {name} on {a.set} ({a.split})", "", "| group | n | MASE / seasonal naive | sCRPS / seasonal naive | MASE / oracle | cov80 |",
             "|---|---|---|---|---|---|"]
    for k, v in [("overall", summary["overall"])] + sorted(summary["by_family"].items()):
        lines.append(f"| {k} | {v['n']} | {v['mase_rel_snaive']:.3f} | {v['scrps_rel_snaive']:.3f} | {v['mase_rel_oracle']:.3f} | {v['cov80']:.3f} |")
    open(os.path.join(od, "table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
