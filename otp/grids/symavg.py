"""Configs for the symmetric-averaging ablation: skewed / asymmetric processes plus controls.

Timestral's shipped forecaster averages the forecast of the series with the reflected
forecast of its negation. That is harmless for symmetric processes and may cost accuracy on
asymmetric ones (bursty counts, one-sided maps). Ids are reused from the family grids.
"""
from otp.grids import chaos, in_prior, tails

PICK = {
    "tails": ["tails/hawkes/eta0", "tails/hawkes/eta0.6", "tails/hawkes/eta0.9", "tails/msm/k4_ret",
              "tails/msm/k4_cum", "tails/stable_ar/alpha1.3_rho0.9", "tails/fbm/H0.5"],
    "chaos": ["logistic_map/r3.2", "logistic_map/r3.9", "lorenz63/rho28", "rossler/c5.7", "mackey_glass/tau17"],
    "in_prior": ["in_prior/p24", "in_prior/p96"],
}


def configs():
    out = []
    for mod, ids in (("tails", PICK["tails"]), ("chaos", PICK["chaos"]), ("in_prior", PICK["in_prior"])):
        table = dict({"tails": tails, "chaos": chaos, "in_prior": in_prior}[mod].configs())
        out += [(i, table[i]) for i in ids]
    return out
