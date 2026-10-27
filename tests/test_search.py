import csv
import os

import numpy as np
import pytest
import torch

from otp.classical import seasonal_naive
from otp.generators import mixture
from otp.search import torch_gen as tg
from otp.search.objective import (Evaluator, EvalResult, Objective, assess, cell_of, dominant_family, genome_hash,
                                  spectral_entropy)


# ------------------------------------------------------------------ stub objective (no simulation, no model)
class StubObjective:
    seeds = (1, 2, 3)
    size = "5M"

    def __init__(self):
        self.calls = 0
        self.sample_generations = 0
        self.n_eval = 0

    def evaluate(self, genomes, size=None, keep_samples=False):
        out = []
        for g in genomes:
            self.n_eval += 1
            fit = 1.0 + 4.0 * float(np.exp(-2.0 * np.sum((g - 0.3) ** 2) / 5))
            feas = bool(g[0] < 0.9)
            out.append(EvalResult(hash=genome_hash(g), size="5M", fitness=fit if feas else 0.0, feasible=feas,
                                  search_score=fit if feas else -float(g[0] - 0.9), ratio_mase=fit, ratio_scrps=fit,
                                  cov80=0.5, orc_sn=0.3, family=dominant_family(g), signature="x",
                                  spec_entropy=float(g[1])))
        return out


def test_evaluator_budget_cache_and_replay(tmp_path):
    obj = StubObjective()
    rng = np.random.default_rng(0)
    gs = [rng.random(mixture.GENOME_DIM) for _ in range(10)]
    ev = Evaluator(obj, 3 * 7, str(tmp_path / "r"), "t", 0)
    res = ev.evaluate(gs + [gs[0]])
    assert len(res) == 7 and ev.calls == 21 and ev.done           # truncated at the budget, no duplicate charge
    again = Evaluator(obj, 3 * 7, str(tmp_path / "r"), "t", 0)    # not resumed: partial file is removed
    assert not again.replay
    # a repeated genome within a run is free
    ev2 = Evaluator(obj, 30, str(tmp_path / "r2"), "t", 0)
    ev2.evaluate([gs[0], gs[0], gs[1]])
    assert ev2.calls == 6 and len(ev2.rows) == 2
    # resume replays the partial file without calling the objective
    n0 = obj.n_eval
    ev3 = Evaluator(obj, 3 * 7, str(tmp_path / "r"), "t", 0, resume=False)
    ev3.evaluate(gs[:4])
    ev4 = Evaluator(obj, 3 * 7, str(tmp_path / "r"), "t", 0, resume=True)
    assert len(ev4.replay) == 4
    n1 = obj.n_eval
    r4 = ev4.evaluate(gs[:4])
    assert obj.n_eval == n1 and ev4.n_replayed == 4 and ev4.calls == 12
    assert [r.hash for r in r4] == [genome_hash(g) for g in gs[:4]]
    ev4.finish()
    assert os.path.exists(tmp_path / "r" / "history.csv") and not os.path.exists(tmp_path / "r" / "scores_partial.csv")
    rows = list(csv.DictReader(open(tmp_path / "r" / "history.csv")))
    assert len(rows) == 4 and float(rows[-1]["best_so_far"]) >= float(rows[0]["best_so_far"])


def test_cmaes_budget_and_progress(tmp_path):
    from otp.search.cmaes import run_cmaes
    ev = Evaluator(StubObjective(), 3 * 260, str(tmp_path), "cmaes", 1)
    run_cmaes(ev, 1)
    assert ev.calls <= 3 * 260 and ev.calls > 3 * 255
    first = np.mean([r["best_so_far"] for r in ev.rows[:13]])
    assert ev.best > first and ev.best > 4.0
    # deterministic given the seed
    ev2 = Evaluator(StubObjective(), 3 * 60, str(tmp_path / "b"), "cmaes", 1)
    run_cmaes(ev2, 1)
    ev3 = Evaluator(StubObjective(), 3 * 60, str(tmp_path / "c"), "cmaes", 1)
    run_cmaes(ev3, 1)
    assert [r["hash"] for r in ev2.rows] == [r["hash"] for r in ev3.rows]


def test_map_elites_archive(tmp_path):
    from otp.search.map_elites import run_map_elites
    ev = Evaluator(StubObjective(), 3 * 300, str(tmp_path), "map_elites", 2)
    info, arch = run_map_elites(ev, 2)
    assert ev.calls <= 3 * 300 and len(arch) >= 5 and info["cells"] == len(arch)
    for (fi, eb), (fit, g, h) in arch.cells.items():
        assert 0 <= fi < 4 and 0 <= eb < 10 and fit >= 1.0
        assert cell_of(dominant_family(g), g[1]) == (fi, eb)


# ------------------------------------------------------------------ objective logic
def _rec(**kw):
    r = {"error": None, "seed": 1, "ctx_ok": True, "orc_ok": True, "period": 24, "ctx_std": 1.0, "spec_entropy": 0.3,
         "mase_orc": 0.5, "scrps_orc": 0.3, "cov80_orc": 0.8, "mase_sn": 1.0, "scrps_sn": 1.0}
    r.update(kw)
    return r


def _ts(mase=1.0, scrps=0.6, cov=0.7, finite=True):
    return {"mase": mase, "scrps": scrps, "cov80": cov, "finite": finite}


def test_assess_feasibility_rules():
    g = np.random.default_rng(1).random(mixture.GENOME_DIM)
    ok = assess(g, "5M", [_rec(), _rec(), _rec()], [_ts(), _ts(), _ts()])
    assert ok.feasible and abs(ok.fitness - 2.0) < 1e-9 and abs(ok.orc_sn - 0.3) < 1e-9
    bad = assess(g, "5M", [_rec(scrps_orc=0.9)] * 3, [_ts()] * 3)           # oracle not clearly better than seasonal naive
    assert not bad.feasible and bad.fitness == 0.0 and bad.search_score < 0 and np.isfinite(bad.ratio_mase)
    worse = assess(g, "5M", [_rec(scrps_orc=1.5)] * 3, [_ts()] * 3)
    assert worse.search_score < bad.search_score                              # closer to feasibility ranks higher
    assert not assess(g, "5M", [_rec(ctx_std=1e7)] * 3, [_ts()] * 3).feasible
    assert not assess(g, "5M", [_rec(ctx_std=1e-4)] * 3, [_ts()] * 3).feasible
    assert not assess(g, "5M", [_rec(mase_orc=1e-8)] * 3, [_ts()] * 3).feasible
    assert not assess(g, "5M", [_rec(orc_ok=False)] * 3, [_ts()] * 3).feasible
    assert not assess(g, "5M", [_rec()] * 3, [_ts(finite=False)] * 3).feasible
    assert not assess(g, "5M", [_rec(error="boom")] * 3, [None] * 3).feasible


def test_descriptors():
    t = np.arange(2048)
    sine = np.sin(2 * np.pi * t / 32)
    noise = np.random.default_rng(0).standard_normal(2048)
    assert spectral_entropy(sine) < 0.3 < 0.8 < spectral_entropy(noise)
    assert spectral_entropy(np.ones(100)) == 1.0
    assert genome_hash(np.ones(25)) == genome_hash(np.ones(25) * 1.5)       # clipped to the box first


def test_objective_real_small():
    g = tg.restricted_genome(np.random.default_rng(4), ["season", "ar_noise", "trend", "season"])
    obj = Objective("5M", n_paths=60)
    r = obj.evaluate([g], keep_samples=True)[0]
    assert obj.calls == 3 and obj.sample_generations == 3
    assert np.isfinite(r.ratio_mase) and r.size == "5M" and len(r.seed_ratios) == 3
    r2 = obj.evaluate([g])[0]
    assert obj.calls == 3 and r2 is r                                         # cached by genome hash
    r3 = obj.evaluate([g], size="28M")[0]                                     # same samples, other size
    assert obj.calls == 6 and obj.sample_generations == 3 and r3.size == "28M"


# ------------------------------------------------------------------ differentiable generator
@pytest.mark.parametrize("kind", tg.SUPPORTED)
def test_pilot_stats_match_numpy(kind):
    rng = np.random.default_rng(11)
    for horizon_switch in (False, True):
        g = tg.restricted_genome(rng, [kind] * 4)
        if horizon_switch:
            for i in range(4):
                g[i * 6 + 4] = 0.9
        m = tg.TorchMixture(g, B=1, K=2, seed=0)
        for i, sl in enumerate(mixture._slots(g)[0]):
            _, _, yp = m._slot(i, torch.tensor(g), 1e-3)
            mu, sd = mixture._pilot_stats(sl.component.name, sl.genes)
            assert abs(yp.mean().item() - mu) < 2e-3 * max(1.0, abs(mu)) + 2e-3
            assert abs(yp.std(unbiased=False).item() - sd) < 2e-3 * max(1.0, sd)


def test_period_of_matches_mixture():
    rng = np.random.default_rng(2)
    for _ in range(25):
        g = tg.restricted_genome(rng)
        assert tg.period_of(g) == mixture.decode(g).period


def test_torch_generator_matches_numpy_statistics():
    g = tg.restricted_genome(np.random.default_rng(8), ["ar_noise", "season", "variance_jump", "mean_jump"])
    g[0 * 6 + 2] = 0.1
    for i in range(4):
        g[i * 6 + 2] = 0.1
    g[1] = 0.8                       # AR weight 1.28
    g[7] = 0.7
    B = 40
    m = tg.TorchMixture(g, B=B, K=20, seed=3)
    c, f = m.forward(torch.tensor(g), tau=1e-3)
    c, f = c.numpy(), f.numpy()
    from otp.generators.base import make_sample
    proc = mixture.decode(g)
    S = [make_sample(proc, 2048, 128, seed=50 + s, n_paths=20) for s in range(B)]
    nc = np.stack([s.context[0] for s in S])
    nf = np.stack([s.future[0] for s in S])
    noq = np.stack([s.oracle_q[0] for s in S])

    def ac1(x):
        return np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in x])

    assert abs(c.std() / nc.std() - 1) < 0.15
    assert abs(ac1(c) - ac1(nc)) < 0.05
    assert abs(f[:, 0].std() / nf.std() - 1) < 0.2
    w_t = (np.quantile(f[:, 1:, 64], 0.95, axis=1) - np.quantile(f[:, 1:, 64], 0.05, axis=1)).mean()
    w_n = (noq[:, 64, 19] - noq[:, 64, 1]).mean()
    assert abs(w_t / w_n - 1) < 0.15


def test_gradients_flow_and_gate():
    g = tg.restricted_genome(np.random.default_rng(5), ["trend", "ar_noise", "mean_jump", "variance_jump"])
    for i in range(4):
        g[i * 6 + 2] = 0.1
        g[i * 6 + 4] = min(g[i * 6 + 4], 0.5) if i >= 2 else g[i * 6 + 4]
    m = tg.TorchMixture(g, B=2, K=8, seed=1)
    gt = torch.tensor(g, requires_grad=True)
    c, f = m.forward(gt, tau=4.0)
    (c.pow(2).mean() + f[:, 0].pow(2).mean()).backward()
    lo, hi, free = tg.gene_bounds(g)
    gr = gt.grad.numpy()
    assert np.isfinite(gr).all()
    assert (np.abs(gr[free]) > 0).mean() > 0.6 and hi[4 * 0 + 4 + 12] <= 0.8   # switch position gene has a gradient and a box
    assert abs(gr[2 * 6 + 4]) > 0                                               # through the sigmoid gate
    # hard limit: smaller tau moves toward a step
    c1, _ = m.forward(torch.tensor(g), tau=0.01)
    c2, _ = m.forward(torch.tensor(g), tau=0.001)
    assert (c1 - c2).abs().max() < 0.2 * c2.abs().max()


def test_torch_scoring_pieces_match_numpy():
    rng = np.random.default_rng(0)
    ctx = np.cumsum(rng.standard_normal((2, 600)), axis=1) + np.sin(np.arange(600) / 4)
    for m in (1, 24):
        q = tg.seasonal_naive_quantiles(torch.tensor(ctx), 64, m).numpy()
        ref = np.stack([seasonal_naive(c, 64, m) for c in ctx])
        assert np.allclose(q, ref, atol=1e-8)
    paths = rng.standard_normal((3, 50, 16))
    fut = torch.tensor(np.concatenate([paths[:, :1], paths], axis=1))
    q = tg.oracle_quantiles(fut).numpy()
    from otp.model import QUANTILE_LEVELS
    ref = np.moveaxis(np.quantile(paths, QUANTILE_LEVELS, axis=1), 0, -1)
    assert np.allclose(q, ref, atol=1e-9)
    from otp import metrics
    y = rng.standard_normal((3, 16))
    pl = tg.pinball(torch.tensor(y), torch.tensor(ref)).numpy()
    assert np.allclose(2 * pl, metrics.score(np.ones((3, 600)) * np.arange(600), y, ref, 1)["scrps"] * metrics.naive_scale(np.ones((3, 600)) * np.arange(600), 1))


def test_grad_search_runs_and_respects_budget(tmp_path):
    from otp.search.gradient import grad_search, plan_steps
    obj = Objective("5M", n_paths=60)
    ev = Evaluator(obj, 80, str(tmp_path), "gradient", 0)
    info = grad_search(ev, 0, n_restarts=1, n_draws=2, K=16, log=lambda *_: None)
    assert ev.calls <= 80 and info["n_steps"] == plan_steps(80, 1, 2, 5)
    assert len(ev.rows) == 5 and all(r["feasible"] for r in ev.rows)           # starts and iterates stay feasible
    ev.finish()
    assert os.path.exists(tmp_path / "grad_trace.csv")


def test_input_sensitivity_small():
    from otp.search.gradient import input_sensitivity
    r = input_sensitivity("5M", 0, T=512, H=32, n_iter=6, grid=[0.01, 0.3, 3.0], n_bisect=1)
    assert r["mase_clean"] > 0 and r["passes"] > 0
    if r["reached"]:
        assert r["mase_perturbed"] >= 2 * r["mase_clean"] - 1e-6
