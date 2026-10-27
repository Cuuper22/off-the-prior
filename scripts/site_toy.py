"""Precompute the "try to beat it" toy for the site: every combination of a few mixing controls is built with the
mixture-of-primitives generator (otp/generators/mixture.py), forecast by Timestral-5M and Timestral-209M, and scored
against the true-dynamics oracle. The site only looks results up; nothing is invented in the browser.

    python scripts/site_toy.py [--n-paths 300]

Writes docs/data/toy.json. Controls:
  season    weight of a period-24 season: 0, 0.7, 1.4
  wild      one extra piece: none, mackey_glass, kuramoto, rossler, lorenz63, mean_jump (16 steps before the end of
            the context), variance_jump (16 steps before the end), stable_ar (alpha 1.3), hawkes
  strength  weight of the wild piece: 0.7 or 1.4 (ignored for none)
  noise     observation noise sd: 0.01, 0.1, 0.4
Context 512, horizon 64, one fixed seed per combination.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from otp import metrics  # noqa: E402
from otp.generators import mixture as M  # noqa: E402
from otp.generators.base import make_sample  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.model import forecast  # noqa: E402

T, H = 512, 64
SEASON = [0.0, 0.7, 1.4]
WILD = ["none", "mackey_glass", "kuramoto", "rossler", "lorenz63", "mean_jump", "variance_jump", "stable_ar", "hawkes"]
STRENGTH = [0.7, 1.4]
NOISE = [0.01, 0.1, 0.4]
# parameter genes (u1, u2, u3) for each wild piece, mapped by the component's own ranges in mixture.py
WILD_GENES = {
    "mackey_glass": (0.46, 0.4, 0.2),    # tau about 17, observed every 5 steps
    "kuramoto": (0.25, 0.3, 0.2),        # K about 1
    "rossler": (0.53, 0.4, 0.2),         # c about 5.7
    "lorenz63": (0.45, 0.4, 0.2),        # rho about 28
    "mean_jump": (0.9, 0.0, 0.4),        # +6.4 sd, 16 steps before the end of the context
    "variance_jump": (0.95, 0.0, 0.4),   # noise sd x6.7, 16 steps before the end of the context
    "stable_ar": (0.22, 0.9, 0.0),       # alpha about 1.3, rho 0.95
    "hawkes": (0.95, 0.5, 0.0),          # near-critical bursts
}
QI = [2, 10, 18]  # quantile indices for 0.1, 0.5, 0.9


def gene_for(name: str) -> float:
    names = M.component_names()
    return (names.index(name) + 0.5) / len(names)


def genome(season: float, wild: str, strength: float, noise: float) -> np.ndarray:
    g = np.zeros(M.GENOME_DIM)
    # slot 0: season (period 24, 2 harmonics); weight = 2 * gene^2
    g[0:6] = [gene_for("season"), np.sqrt(season / 2), 0.0, 0.3, 0.3, 0.5]
    # slot 1: AR noise, fixed small weight, so every series has some memory
    g[6:12] = [gene_for("ar_noise"), np.sqrt(0.3 / 2), 0.0, 0.7, 0.0, 0.0]
    # slot 2: the wild piece
    if wild != "none":
        g[12:18] = [gene_for(wild), np.sqrt(strength / 2), 0.0, *WILD_GENES[wild]]
    # slot 3 unused (weight 0)
    g[18:24] = [gene_for("trend"), 0.0, 0.0, 0.5, 0.0, 1.0]
    g[-1] = (np.log10(noise) + 3) / 2.5
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "toy.json"))
    a = ap.parse_args()
    combos = []
    for season, wild, strength, noise in itertools.product(SEASON, WILD, STRENGTH, NOISE):
        if wild == "none" and strength != STRENGTH[0]:
            continue
        combos.append((season, wild, strength, noise))
    samples = []
    for k, (season, wild, strength, noise) in enumerate(combos):
        key = f"s{season}|{wild}|{strength if wild != 'none' else 0}|n{noise}"
        seed = zlib.crc32(("toy/" + key).encode()) % (2**31)
        proc = M.decode(genome(season, wild, strength, noise))
        samples.append((key, make_sample(proc, T, H, seed=seed, n_paths=a.n_paths, sample_id=key)))
        print(f"{k + 1}/{len(combos)} {key}", flush=True)
    ctx = np.stack([s.context[0] for _, s in samples])
    fut = np.stack([s.future[0] for _, s in samples])
    orq = np.stack([s.oracle_q[0] for _, s in samples])
    per = [int(s.period) for _, s in samples]
    fc = {size: forecast(size, ctx[:, None], H, batch_size=16)[:, 0] for size in ("5M", "209M")}
    items = []
    for i, (key, s) in enumerate(samples):
        so = metrics.score(ctx[i], fut[i], orq[i], per[i])
        rec = {"key": key, "period": per[i], "context": [round(float(x), 4) for x in ctx[i, -256:]],
               "future": [round(float(x), 4) for x in fut[i]],
               "oracle": [[round(float(x), 4) for x in orq[i][:, j]] for j in QI], "oracle_mase": round(float(so["mase"]), 4),
               "nonfinite_oracle_paths": int(s.meta.get("oracle_paths_nonfinite", 0))}
        for size, q in fc.items():
            st = metrics.score(ctx[i], fut[i], q[i], per[i])
            rec[f"q{size}"] = [[round(float(x), 4) for x in q[i][:, j]] for j in QI]
            rec[f"mase{size}"] = round(float(st["mase"]), 4)
            rec[f"ratio{size}"] = round(float(st["mase"] / so["mase"]), 4) if np.isfinite(so["mase"]) and so["mase"] > 0 else None
        items.append(rec)
    write_json(a.out, {"controls": {"season": SEASON, "wild": WILD, "strength": STRENGTH, "noise": NOISE}, "T": T, "H": H,
                       "context_shown": 256, "quantiles": [0.1, 0.5, 0.9], "items": items,
                       "provenance": provenance("scripts/site_toy.py", argv=sys.argv)})
    print(f"wrote {a.out} ({len(items)} combinations)")


if __name__ == "__main__":
    main()
