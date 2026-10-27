"""MAP-Elites (quality-diversity) over the genome box.

Behaviour descriptors (see `objective.py`): the dominant component family by summed slot weight
(4 values) and the spectral entropy of the context (10 equal bins on [0, 1]); 40 cells. Each cell
keeps its best feasible genome. Start with random genomes (a fraction of the budget), then
repeat: pick archive members uniformly, mutate, evaluate the batch, insert children that beat
the cell's elite or fill an empty cell. Mutation: each gene moves with probability `p_gene` by a
Gaussian step of `sigma`; with probability `p_slot` a child also re-draws one whole slot.
"""

from __future__ import annotations

import numpy as np

from otp.generators import mixture
from otp.search.objective import GRID_SHAPE, cell_of


class Archive:
    def __init__(self):
        self.cells: dict[tuple[int, int], tuple[float, np.ndarray, str]] = {}

    def add(self, genome, res) -> bool:
        if not res.feasible:
            return False
        c = cell_of(res.family, res.spec_entropy)
        cur = self.cells.get(c)
        if cur is None or res.fitness > cur[0]:
            self.cells[c] = (res.fitness, np.asarray(genome, float).copy(), res.hash)
            return True
        return False

    def __len__(self):
        return len(self.cells)


def mutate(g: np.ndarray, rng: np.random.Generator, sigma: float = 0.12, p_gene: float = 0.2,
           p_slot: float = 0.15) -> np.ndarray:
    c = g.copy()
    m = rng.random(c.size) < p_gene
    if not m.any():
        m[rng.integers(c.size)] = True
    c[m] += rng.normal(0.0, sigma, int(m.sum()))
    if rng.random() < p_slot:
        i = int(rng.integers(mixture.N_SLOTS))
        c[i * mixture.GENES_PER_SLOT:(i + 1) * mixture.GENES_PER_SLOT] = rng.random(mixture.GENES_PER_SLOT)
    return np.clip(c, 0.0, 1.0)


def run_map_elites(ev, seed: int, init_frac: float = 0.15, batch: int = 12):
    rng = np.random.default_rng(seed)
    arch = Archive()
    info = {"batches": 0}
    n_init = max(batch, int(init_frac * ev.budget / ev.n_seeds))
    done_init = 0
    while done_init < n_init and not ev.done:
        k = min(batch, n_init - done_init)
        gs = [mixture.random_genome(rng) for _ in range(k)]
        res = ev.evaluate(gs, origin="init")
        for g, r in zip(gs, res):
            arch.add(g, r)
        done_init += k
        info["batches"] += 1
    while not ev.done:
        if len(arch) == 0:
            gs = [mixture.random_genome(rng) for _ in range(batch)]
            origin = "random"
        else:
            keys = list(arch.cells.keys())
            keys.sort()
            parents = [arch.cells[keys[int(rng.integers(len(keys)))]][1] for _ in range(batch)]
            gs = [mutate(p, rng) for p in parents]
            origin = "mutation"
        res = ev.evaluate(gs, origin=origin)
        for g, r in zip(gs, res):
            arch.add(g, r)
        info["batches"] += 1
        if not res:
            break
    info["cells"] = len(arch)
    info["grid"] = GRID_SHAPE
    return info, arch
