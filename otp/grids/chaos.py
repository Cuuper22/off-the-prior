"""Chaos and dynamics grid: five generators, each swept over its difficulty knob.

`configs()` returns the full grid (the main sweep). `BIG_ONLY` lists the reduced subset used
for the 123M and 209M runs (three knob values per generator, from periodic to chaotic).
`param_list()` returns `(config_id, generator_class, params)` rows, which the Lyapunov table
in `otp.generators.chaos` also uses.

Also holds the post-processing used for every chaos results folder:

    python -m otp.grids.chaos summarize --dir results/map/chaos
    python -m otp.grids.chaos readme --dir results/map/chaos --grid otp.grids.chaos

`summarize` builds `summary.csv` from `scores.csv` with `otp.analysis.ratios` and
`otp.analysis.curve` grouped by `[config_id, generator, knob_value, forecaster, T, H]`
(`mase_ratio_oracle`, `scrps_ratio_oracle`, `mase_ratio_classical`, merged on the group
keys, plus the mean of `cov80` and the mean `mase`).
"""

from __future__ import annotations

from otp.generators.chaos import Kuramoto, LogisticMap, Lorenz63, MackeyGlass, Rossler

LORENZ_RHO = (10.0, 15.0, 20.0, 24.0, 26.0, 28.0, 35.0, 45.0)
LORENZ_OBS_EVERY = (2, 5, 10, 20)       # at rho = 28 (10 is the default, already in the rho sweep)
ROSSLER_C = (2.0, 3.0, 4.0, 4.5, 5.0, 5.7, 6.5, 9.0)
MG_TAU = (6.0, 10.0, 14.0, 17.0, 20.0, 25.0, 30.0)
LOGISTIC_R = (2.8, 3.2, 3.5, 3.56, 3.6, 3.7, 3.83, 3.9, 4.0)
KURAMOTO_K = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)


def _f(v: float) -> str:
    return f"{v:g}"


def param_list(full: bool = True):
    """Rows `(config_id, class, params)` for the main grid."""
    rows = []
    for rho in LORENZ_RHO:
        rows.append((f"lorenz63/rho{_f(rho)}", Lorenz63, {"rho": rho}))
    for oe in LORENZ_OBS_EVERY:
        if oe != Lorenz63.defaults["obs_every"]:
            rows.append((f"lorenz63/rho28_oe{oe}", Lorenz63, {"rho": 28.0, "obs_every": oe}))
    for c in ROSSLER_C:
        rows.append((f"rossler/c{_f(c)}", Rossler, {"c": c}))
    for tau in MG_TAU:
        rows.append((f"mackey_glass/tau{_f(tau)}", MackeyGlass, {"tau": tau}))
    for r in LOGISTIC_R:
        rows.append((f"logistic_map/r{_f(r)}", LogisticMap, {"r": r}))
    for k in KURAMOTO_K:
        rows.append((f"kuramoto/K{_f(k)}", Kuramoto, {"K": k}))
    return rows


# Reduced grid for the large models: about 3 knob values per generator, periodic -> chaotic.
BIG_ONLY = ["lorenz63/rho15", "lorenz63/rho26", "lorenz63/rho28", "lorenz63/rho28_oe20",
            "rossler/c3", "rossler/c5", "rossler/c5.7",
            "mackey_glass/tau10", "mackey_glass/tau17", "mackey_glass/tau30",
            "logistic_map/r3.2", "logistic_map/r3.6", "logistic_map/r4",
            "kuramoto/K0", "kuramoto/K2", "kuramoto/K8"]


def make_factory(cls, params):
    if cls.name == "kuramoto":
        return lambda seed: cls(**params, freq_seed=int(seed))
    return lambda seed: cls(**params)


def configs():
    return [(cid, make_factory(cls, params)) for cid, cls, params in param_list()]
