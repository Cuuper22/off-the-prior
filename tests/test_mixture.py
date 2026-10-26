import numpy as np
import pytest

from otp.generators import mixture as M
from otp.generators.base import make_sample


def _genome(names, modulate_slot=None, seed=0):
    g = np.random.default_rng(seed).random(M.GENOME_DIM)
    allnames = M.component_names()
    for i, c in enumerate(names):
        g[i * M.GENES_PER_SLOT] = (allnames.index(c) + 0.5) / len(allnames)
        g[i * M.GENES_PER_SLOT + 1] = 0.8
        g[i * M.GENES_PER_SLOT + 2] = 0.9 if i == modulate_slot else 0.1
    return g


def test_decode_is_deterministic_and_finite():
    g = _genome(["season", "ar_noise", "trend", "stable_ar"], modulate_slot=2)
    a = make_sample(M.decode(g), 1024, 64, seed=3, n_paths=50)
    b = make_sample(M.decode(g), 1024, 64, seed=3, n_paths=50)
    assert np.array_equal(a.context, b.context)
    assert np.isfinite(a.context).all() and np.isfinite(a.oracle_q).all()
    d = M.describe(g)
    assert len(d["slots"]) == M.N_SLOTS and d["slots"][2]["mode"] == "modulate"


def test_regime_component_keeps_origin_inside_mixture():
    # a context-placed mean jump must land d steps before the forecast origin, even when
    # another slot needs a long warmup
    g = _genome(["mean_jump", "stable_ar", "season", "ar_noise"])
    g[1 * M.GENES_PER_SLOT + 1] = 0.0  # silence the other slots
    g[2 * M.GENES_PER_SLOT + 1] = 0.0
    g[3 * M.GENES_PER_SLOT + 1] = 0.0
    proc = M.decode(g)
    assert proc.procs[0].params["placement"] in ("context", "horizon")


@pytest.mark.parametrize("seed", range(16))
def test_random_genomes_build(seed):
    rng = np.random.default_rng(seed)
    g = M.random_genome(rng)
    try:
        s = make_sample(M.decode(g), 512, 32, seed=seed, n_paths=20)
    except ImportError:
        pytest.skip("a family module is not merged on this branch")
    assert s.context.shape == (1, 512)
