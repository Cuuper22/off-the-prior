"""Extra benchmark configs: 40 fixed random mixture-of-primitives genomes and more in-prior periods.

Genome j is `mixture.random_genome(np.random.default_rng(7000 + j))`, so each config is fixed;
the sample seed only changes the noise realization.
"""
import numpy as np

from otp.generators import mixture
from otp.generators.primitives import in_prior

N_MIX = 40
EXTRA_PERIODS = (7, 12, 48, 365)


def genome(j: int) -> np.ndarray:
    return mixture.random_genome(np.random.default_rng(7000 + j))


def configs():
    out = [(f"mixture/g{j:02d}", (lambda seed, j=j: mixture.decode(genome(j)))) for j in range(N_MIX)]
    out += [(f"in_prior_extra/p{p}", (lambda seed, p=p: in_prior(seed, period=p))) for p in EXTRA_PERIODS]
    return out
