"""CMA-ES over the genome box [0, 1]^D (the `cma` package), maximizing `search_score`.

Covariance matrix adaptation is rank based, so only the ordering of `search_score` matters:
feasible genomes rank by fitness above all infeasible ones, infeasible ones by closeness to
feasibility (see `objective.py`). Restarts (IPOP style: population size doubled, new random
start) are used when a run converges before the budget is spent. One population is evaluated
as one batch, so the sample generation parallelizes.
"""

from __future__ import annotations

import cma
import numpy as np

from otp.generators import mixture


def run_cmaes(ev, seed: int, sigma0: float = 0.3, popsize: int | None = None, max_popsize: int = 64):
    rng = np.random.default_rng(seed)
    d = mixture.GENOME_DIM
    pop = popsize or (4 + int(3 * np.log(d)))
    restart = 0
    info = {"restarts": 0, "generations": 0}
    while not ev.done:
        x0 = rng.random(d)
        opts = {"bounds": [0.0, 1.0], "popsize": pop, "seed": int(seed * 1000 + restart + 1), "verbose": -9,
                "tolfun": 1e-9, "tolx": 1e-6}
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop() and not ev.done:
            xs = es.ask()
            res = ev.evaluate([np.clip(x, 0, 1) for x in xs], origin=f"cmaes_r{restart}_g{info['generations']}")
            info["generations"] += 1
            if len(res) < len(xs):          # budget ran out inside this population
                break
            es.tell(xs, [-r.search_score for r in res])
        restart += 1
        info["restarts"] = restart
        pop = min(max_popsize, pop * 2)
    return info
