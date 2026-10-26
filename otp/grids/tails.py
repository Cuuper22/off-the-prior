"""Grid for the heavy-tail and fractal family.

Config ids (the first column of every results table):
  tails/stable_ar/alpha<A>_rho<R>   alpha-stable AR(1); rho = 1.0 is a random walk
  tails/fbm/H<H>                    fractional Brownian motion
  tails/hawkes/eta<E>               Hawkes counts, mean rate fixed
  tails/msm/k<K>_<ret|cum>          Markov-switching multifractal; ret = returns, cum = cumulative sum

`configs()` is the full grid; `BIG_IDS` is the reduced grid (three knob values per generator)
used for the 123M and 209M models.
"""
from __future__ import annotations

from otp.generators.tails import FractionalBM, Hawkes, MSMCascade, StableAR

ALPHAS = (2.0, 1.9, 1.7, 1.5, 1.3, 1.1)
RHOS = (0.9, 1.0)
HURSTS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)
ETAS = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95)
KS = (1, 2, 4, 6, 8)
OBS = (("ret", "returns"), ("cum", "cumsum"))


def configs():
    out = []
    for rho in RHOS:
        for a in ALPHAS:
            out.append((f"tails/stable_ar/alpha{a:g}_rho{rho:.1f}",
                        lambda seed, a=a, rho=rho: StableAR(alpha=a, rho=rho)))
    for h in HURSTS:
        out.append((f"tails/fbm/H{h:g}", lambda seed, h=h: FractionalBM(H_hurst=h)))
    for e in ETAS:
        out.append((f"tails/hawkes/eta{e:g}", lambda seed, e=e: Hawkes(eta=e)))
    for tag, obs in OBS:
        for k in KS:
            out.append((f"tails/msm/k{k}_{tag}", lambda seed, k=k, obs=obs: MSMCascade(k=k, observe=obs)))
    return out


BIG_IDS = [
    "tails/stable_ar/alpha2_rho0.9", "tails/stable_ar/alpha1.5_rho0.9", "tails/stable_ar/alpha1.1_rho0.9",
    "tails/fbm/H0.2", "tails/fbm/H0.5", "tails/fbm/H0.8",
    "tails/hawkes/eta0", "tails/hawkes/eta0.8", "tails/hawkes/eta0.95",
    "tails/msm/k1_ret", "tails/msm/k4_ret", "tails/msm/k8_ret",
]
