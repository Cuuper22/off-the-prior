import numpy as np
import pytest

from otp.generators.base import make_sample
from otp.generators.chaos import Kuramoto, LogisticMap, Lorenz63, MackeyGlass, Rossler, noise_free
from otp.grids.chaos import configs, param_list

REPS = [
    (Lorenz63, {"rho": 28.0}),
    (Rossler, {"c": 5.7}),
    (MackeyGlass, {"tau": 17.0}),
    (LogisticMap, {"r": 3.9}),
    (Kuramoto, {"K": 2.0, "freq_seed": 3}),
]


@pytest.mark.parametrize("cls,params", REPS)
def test_shapes(cls, params):
    proc = cls(**params)
    rng = np.random.default_rng(0)
    state = proc.init_state(rng, 7)
    obs, new = proc.step(state, 20, rng)
    assert obs.shape == (7, 1, 20)
    assert set(new) == set(state)
    for k in state:
        assert new[k].shape == state[k].shape
    s = make_sample(proc, 128, 16, seed=1, n_paths=20)
    assert s.context.shape == (1, 128) and s.future.shape == (1, 16) and s.oracle_q.shape == (1, 16, 21)


@pytest.mark.parametrize("cls,params", REPS)
def test_same_seed_same_sample(cls, params):
    a = make_sample(cls(**params), 96, 16, seed=5, n_paths=10)
    b = make_sample(cls(**params), 96, 16, seed=5, n_paths=10)
    assert np.array_equal(a.context, b.context)
    assert np.array_equal(a.future, b.future)
    assert np.array_equal(a.oracle_q, b.oracle_q)
    c = make_sample(cls(**params), 96, 16, seed=6, n_paths=10)
    assert not np.array_equal(a.context, c.context)


@pytest.mark.parametrize("cls,params", REPS)
def test_oracle_median_close_to_future_with_tiny_noise(cls, params):
    proc = cls(**params, dyn_rel=1e-8, obs_rel=1e-8)
    s = make_sample(proc, 256, 16, seed=2, n_paths=40)
    med = s.oracle_q[0, :, 10]
    sd = s.context.std()
    assert np.abs(med - s.future[0]).max() < 0.02 * sd


def test_oracle_spread_grows_with_default_noise():
    s = make_sample(Lorenz63(rho=28.0), 256, 128, seed=2, n_paths=100)
    width = s.oracle_q[0, :, 18] - s.oracle_q[0, :, 2]
    assert width[-1] > 10 * width[0]


def test_logistic_state_stays_in_unit_interval():
    proc = LogisticMap(r=4.0, dyn_rel=0.5)
    rng = np.random.default_rng(0)
    state = proc.init_state(rng, 50)
    _, new = proc.step(state, 500, rng)
    assert (new["x"] > 0).all() and (new["x"] < 1).all()


def test_mackey_glass_buffer_shape():
    proc = MackeyGlass(tau=17.0)
    state = proc.init_state(np.random.default_rng(0), 3)
    assert state["hist"].shape == (3, 86)


def test_kuramoto_frequencies_fixed_per_seed():
    a, b, c = Kuramoto(freq_seed=1), Kuramoto(freq_seed=1), Kuramoto(freq_seed=2)
    assert np.array_equal(a.omega, b.omega) and not np.array_equal(a.omega, c.omega)
    assert a.omega.shape == (16,)
    assert abs(Kuramoto.critical_coupling(1.0) - 1.5958) < 1e-3


def test_noise_free_run_is_deterministic_given_state():
    proc = noise_free(Lorenz63, {"rho": 28.0})
    s = make_sample(proc, 64, 16, seed=1, n_paths=5)
    # no noise: every oracle path equals the realized future
    assert np.allclose(s.oracle_q[0, :, 0], s.oracle_q[0, :, 20])
    assert np.allclose(s.oracle_q[0, :, 10], s.future[0])


def test_samples_finite_for_every_grid_config():
    cfgs = configs()
    assert len(cfgs) == len(param_list())
    ids = [c[0] for c in cfgs]
    assert len(set(ids)) == len(ids)
    for cid, factory in cfgs:
        proc = factory(0)
        assert proc.family == "chaos"
        s = make_sample(proc, 64, 16, seed=0, n_paths=8)
        assert np.isfinite(s.context).all(), cid
        assert np.isfinite(s.future).all(), cid
        assert np.isfinite(s.oracle_q).all(), cid
        assert isinstance(proc.tags(), list)
