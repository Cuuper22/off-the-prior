import numpy as np
import pytest

from otp.generators.base import make_sample
from otp.generators.regime import (FrequencyJump, MarkovSwitchingAR, MeanJump, VarianceJump, randomized_base,
                                   stationary_sd)
from otp.grids.regime import BIG_IDS, configs, parse_config_id


def _sample(proc, T=256, H=32, seed=3, n_paths=60):
    return make_sample(proc, T, H, seed=seed, n_paths=n_paths)


def test_shapes():
    for proc in (MeanJump(), VarianceJump(), FrequencyJump(), MarkovSwitchingAR(),
                 MeanJump(placement="horizon")):
        s = _sample(proc)
        assert s.context.shape == (1, 256)
        assert s.future.shape == (1, 32)
        assert s.oracle_q.shape == (1, 32, 21)
        assert np.isfinite(s.context).all() and np.isfinite(s.oracle_q).all()
        assert s.meta["family"] == "regime"


def test_determinism_by_seed():
    for proc_fn in (lambda: MeanJump(d=32), lambda: MarkovSwitchingAR(p_switch=0.05),
                    lambda: VarianceJump(placement="horizon", hazard=0.05)):
        a, b = _sample(proc_fn(), seed=7), _sample(proc_fn(), seed=7)
        c = _sample(proc_fn(), seed=8)
        assert np.array_equal(a.context, b.context) and np.array_equal(a.oracle_q, b.oracle_q)
        assert np.array_equal(a.future, b.future)
        assert not np.array_equal(a.context, c.context)


def test_context_mean_jump_oracle_median_at_post_switch_level():
    # no season, weak AR: the oracle median at H = 16 sits at the post-switch level
    kw = dict(amplitude=0.0, rho=0.2, sigma=0.3, delta=8.0, d=64, placement="context")
    proc = MeanJump(**kw)
    sd = stationary_sd(0.2, 0.3)
    post = 8.0 * sd
    for seed in range(5):
        s = make_sample(MeanJump(**kw), 512, 16, seed=seed, n_paths=400)
        med = s.oracle_q[0, :, 10]
        assert abs(med.mean() - post) < 0.5 * sd, (med.mean(), post)
        # the context really switched d = 64 steps before the origin
        assert s.context[0, -64:].mean() - s.context[0, :-64].mean() > 5 * sd
    assert proc.period == 24


def test_switch_position_exact():
    # level shift only, no noise, no season: the first post-switch value is at index T - d
    kw = dict(amplitude=0.0, sigma=1e-9, delta=1000.0, d=20, placement="context")
    s = make_sample(MeanJump(**kw), 100, 8, seed=1, n_paths=5)
    jump = np.flatnonzero(s.context[0] > 0.5 * s.context[0].max())
    assert jump.min() == 100 - 20 and jump.max() == 99


def test_switch_before_window_means_all_post_switch():
    kw = dict(amplitude=0.0, sigma=1e-9, delta=1000.0, d=500, placement="context")
    proc = MeanJump(**kw)
    s = make_sample(proc, 100, 8, seed=1, n_paths=5)
    assert (s.context[0] > 0.5 * s.context[0].max()).all()
    assert "switch_before_window" in s.meta["tags"]


def test_horizon_placement_no_switch_in_context_and_oracle_mixes():
    kw = dict(amplitude=0.0, rho=0.2, sigma=0.3, delta=8.0, placement="horizon", hazard=0.05)
    sd = stationary_sd(0.2, 0.3)
    s = make_sample(MeanJump(**kw), 256, 64, seed=2, n_paths=400)
    assert abs(s.context[0].mean()) < 3 * sd             # pre-switch level throughout the context
    assert s.meta["tags"] == ["switch_in_horizon"]
    med = s.oracle_q[0, :, 10]
    assert med[-1] > med[0] + 2 * sd                     # more paths have switched late in the horizon
    q = s.oracle_q[0]
    assert q[-1, 19] - q[-1, 1] > 4 * sd                 # wide, two-humped spread at the end


def test_variance_and_frequency_jump_change_what_they_say():
    kw = dict(amplitude=0.0, rho=0.0, sigma=0.5, factor=4.0, d=400)
    s = make_sample(VarianceJump(**kw), 800, 8, seed=0, n_paths=5)
    before, after = s.context[0, :400].std(), s.context[0, 400:].std()
    assert 3.0 < after / before < 5.0
    f = FrequencyJump(amplitude=1.0, harmonics=1, sigma=1e-6, rho=0.0, ratio=2.0, d=480, period=24)
    s = make_sample(f, 960, 8, seed=0, n_paths=5)
    x = s.context[0]
    zc = lambda seg: np.sum(np.diff(np.sign(seg - seg.mean())) != 0)
    assert zc(x[:480]) > 1.6 * zc(x[480:])               # half as many zero crossings at twice the period
    assert f.period == 48
    assert FrequencyJump(placement="horizon", ratio=2.0).period == 24


def test_markov_switching_visits_both_regimes():
    p = MarkovSwitchingAR(p_switch=0.05, mean1=60.0, sigma=0.1, amplitude=0.0)
    s = make_sample(p, 2000, 8, seed=4, n_paths=5)
    x = s.context[0]
    assert x.max() - x.min() > 3.0 and (x > 1.0).mean() > 0.2 and (x < 1.0).mean() > 0.2
    assert s.meta["tags"] == ["markov_switching"]


def test_oracle_paths_share_the_true_regime_for_context_placement():
    kw = dict(amplitude=0.0, rho=0.0, sigma=1e-6, delta=50.0, d=10)
    s = make_sample(MeanJump(**kw), 64, 16, seed=0, n_paths=30)
    spread = s.oracle_q[0, :, -1] - s.oracle_q[0, :, 0]
    assert spread.max() < 1e-3


def test_every_grid_config_gives_finite_samples():
    cfgs = configs()
    assert len(cfgs) == len({c for c, _ in cfgs}) == 110
    for cid, fac in cfgs:
        s = make_sample(fac(1), 256, 32, seed=5, n_paths=20)
        assert np.isfinite(s.context).all() and np.isfinite(s.future).all() and np.isfinite(s.oracle_q).all(), cid
        info = parse_config_id(cid)
        assert info["generator"] == s.meta["generator"]
        assert info["knob_value"] == s.meta["knob_value"]
        if info["placement"] == "context":
            assert s.meta["params"]["d"] == info["d"]


def test_big_ids_are_in_grid():
    ids = {c for c, _ in configs()}
    assert set(BIG_IDS) <= ids and len(BIG_IDS) == len(set(BIG_IDS)) == 24


def test_randomized_base_in_range():
    for seed in range(20):
        b = randomized_base(seed)
        assert 0.2 <= b["rho"] <= 0.8 and 0.2 <= stationary_sd(b["rho"], b["sigma"]) <= 0.6
