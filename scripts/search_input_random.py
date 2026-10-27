"""Random-direction baseline for the input-space sensitivity study of the failure search.

results/search/input_space.csv gives, per in-prior series, the smallest perturbation radius (found by projected
gradient ascent) that doubles Timestral's MASE. This script perturbs the same series by random directions of the
same L2 norm (Gaussian noise, and a smooth random walk), with the same forecast function (otp.search.gradient,
`forecast_torch`, median MASE on the fixed future), and reports the MASE ratio perturbed / clean. It also scores the
clean and randomly perturbed contexts with the shipped forecaster (`otp.model.forecast`, symmetric averaging on).

    python scripts/search_input_random.py [--size 5M] [--draws 20]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from otp import metrics  # noqa: E402
from otp.generators.base import make_sample  # noqa: E402
from otp.generators.primitives import in_prior  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.model import forecast  # noqa: E402
from otp.search.gradient import _mase_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="5M")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "search", "input_space_random.csv"))
    a = ap.parse_args()
    inp = pd.read_csv(os.path.join(ROOT, "results", "search", "input_space.csv"))
    inp = inp[(inp["size"] == a.size) & inp.reached.astype(bool)]
    T, H = 2048, 128
    rows = []
    for r in inp.itertuples(index=False):
        s = make_sample(in_prior(int(r.series_seed)), T, H, seed=int(r.series_seed), n_paths=2)
        x = torch.tensor(s.context[0], dtype=torch.float32)
        y = torch.tensor(s.future[0], dtype=torch.float32)
        scale = float(metrics.naive_scale(s.context, s.period)[0])
        f = _mase_fn(a.size, y, scale, H)
        with torch.no_grad():
            m0 = float(f(x))
        radius = float(r.rho) * float(x.std()) * math.sqrt(T)
        rng = np.random.default_rng(1000 + int(r.series_seed))
        ctx_batch = [s.context[0]]
        for kind in ("gauss", "walk"):
            for k in range(a.draws):
                z = rng.standard_normal(T)
                if kind == "walk":
                    z = np.cumsum(z)
                    z -= z.mean()
                d = torch.tensor(z * radius / np.linalg.norm(z), dtype=torch.float32)
                with torch.no_grad():
                    m = float(f(x + d))
                rows.append({"size": a.size, "series_seed": int(r.series_seed), "kind": kind, "draw": k, "rho": float(r.rho),
                             "mase_clean": m0, "mase_perturbed": m, "ratio": m / m0, "optimized_ratio": float(r.mase_perturbed) / float(r.mase_clean)})
                if k == 0:
                    ctx_batch.append((x + d).numpy())
        # shipped forecaster on the clean context and on the first random draw of each kind
        q = forecast(a.size, np.stack(ctx_batch)[:, None], H)
        ms = [float(metrics.score(c[None], s.future, q[i], s.period)["mase"][0]) for i, c in enumerate(ctx_batch)]
        for kind, mk in zip(("gauss", "walk"), ms[1:]):
            rows.append({"size": a.size, "series_seed": int(r.series_seed), "kind": f"{kind}_shipped", "draw": 0, "rho": float(r.rho),
                         "mase_clean": ms[0], "mase_perturbed": mk, "ratio": mk / ms[0], "optimized_ratio": np.nan})
        print(r.series_seed, f"rho {r.rho:.4f}", f"random gauss median ratio "
              f"{np.median([x['ratio'] for x in rows if x['series_seed'] == r.series_seed and x['kind'] == 'gauss']):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    summ = df.groupby("kind").ratio.agg(["count", "median", "max", lambda v: float(np.quantile(v, 0.95))]).reset_index()
    summ.columns = ["kind", "n", "median_ratio", "max_ratio", "p95_ratio"]
    summ.to_csv(a.out.replace(".csv", "_summary.csv"), index=False)
    write_json(a.out.replace(".csv", "_provenance.json"), provenance("scripts/search_input_random.py", argv=sys.argv))
    print(summ.to_string())


if __name__ == "__main__":
    main()
