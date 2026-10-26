"""Long-context grid for Colab job B: does history up to 32,768 steps fix failures?

Two settings per generator (one easy, one hard) across all families, plus in-prior controls.
Regime switches use distances that only become visible with long contexts.
"""
from otp.grids import chaos_long, in_prior, regime, tails

REGIME_IDS = ["regime/mean_jump/context/delta4/d64", "regime/mean_jump/context/delta4/d4096",
              "regime/variance_jump/context/factor0.25/d64", "regime/variance_jump/context/factor0.25/d4096",
              "regime/frequency_jump/context/ratio2/d256", "regime/frequency_jump/context/ratio2/d16384",
              "regime/markov_ar/p0.003", "regime/markov_ar/p0.03"]
TAILS_IDS = ["tails/stable_ar/alpha2_rho0.9", "tails/stable_ar/alpha1.3_rho0.9", "tails/fbm/H0.2", "tails/fbm/H0.8",
             "tails/hawkes/eta0.3", "tails/hawkes/eta0.95", "tails/msm/k2_cum", "tails/msm/k8_cum"]
PRIOR_IDS = ["in_prior/p24", "in_prior/p168"]


def configs():
    out = list(chaos_long.configs())
    for mod, ids in ((regime, REGIME_IDS), (tails, TAILS_IDS), (in_prior, PRIOR_IDS)):
        table = dict(mod.configs())
        out += [(i, table[i]) for i in ids]
    return out
