"""Tests for the heavy-tail and fractal family."""
import numpy as np
import pytest

from otp.generators.base import make_sample
from otp.generators.tails import (FractionalBM, Hawkes, MSMCascade, StableAR, davies_harte, fgn_lag1_autocorr,
                                  stable_noise)
from otp.grids.tails import BIG_IDS, configs


def _run(proc, n, P=1, seed=0):
    rng = np.random.default_rng(seed)
    st = proc.init_state(rng, P)
    w = proc.warmup_steps()
    if w:
        _, st = proc.step(st, w, rng)
    return proc.step(st, n, rng)


ALL = [StableAR(alpha=1.5), StableAR(alpha=2.0, rho=1.0), FractionalBM(H_hurst=0.2), FractionalBM(H_hurst=0.8),
       Hawkes(eta=0.8), MSMCascade(k=4), MSMCascade(k=6, observe="cumsum")]


@pytest.mark.parametrize("proc", ALL, ids=lambda p: p.name)
def test_shapes_and_determinism(proc):
    obs, st = _run(proc, 300, P=3, seed=5)
    assert obs.shape == (3, 1, 300)
    assert np.isfinite(obs).all()
    obs2, _ = _run(proc, 300, P=3, seed=5)
    assert np.array_equal(obs, obs2)
    obs3, _ = _run(proc, 300, P=3, seed=6)
    assert not np.array_equal(obs, obs3)
    # continuing from the returned state keeps the leading path axis
    more, _ = proc.step(st, 40, np.random.default_rng(1))
    assert more.shape == (3, 1, 40) and np.isfinite(more).all()


def test_make_sample_determinism_and_meta():
    for proc in (StableAR(alpha=1.3), FractionalBM(H_hurst=0.7), Hawkes(), MSMCascade(k=4)):
        a = make_sample(proc, 600, 64, seed=3, n_paths=50)
        b = make_sample(proc, 600, 64, seed=3, n_paths=50)
        assert a.context.shape == (1, 600) and a.future.shape == (1, 64) and a.oracle_q.shape == (1, 64, 21)
        assert np.array_equal(a.context, b.context) and np.array_equal(a.oracle_q, b.oracle_q)
        assert a.meta["family"] == "tails" and a.meta["tags"]


def test_stable_noise_tails():
    rng = np.random.default_rng(0)
    n = 400_000
    g = stable_noise(rng, 2.0, n)
    h = stable_noise(rng, 1.3, n)
    # alpha = 2 is a Gaussian with std sqrt(2): excess kurtosis near 0
    assert abs(g.var() - 2.0) < 0.03
    kurt = ((g - g.mean()) ** 4).mean() / g.var() ** 2 - 3
    assert abs(kurt) < 0.1

    # tail fraction beyond 6 robust standard deviations (IQR / 1.349), so that the variance is not used
    def tail_frac(x):
        s = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
        return np.mean(np.abs(x) > 6 * s)

    assert tail_frac(g) < 1e-4
    assert tail_frac(h) > 0.005


def test_stable_noise_matches_scipy_cdf():
    from scipy.stats import levy_stable
    rng = np.random.default_rng(1)
    x = stable_noise(rng, 1.5, 200_000)
    for t in (1.0, 3.0):
        emp = np.mean(x > t)
        ref = levy_stable.sf(t, 1.5, 0.0)
        assert abs(emp - ref) < 0.004


def test_stable_ar_random_walk_is_cumulative_sum():
    proc = StableAR(alpha=1.7, rho=1.0)
    obs, _ = _run(proc, 50, P=1, seed=2)
    e = stable_noise(np.random.default_rng(2), 1.7, (1, 50))
    assert np.allclose(obs[0, 0], np.cumsum(e[0]))


@pytest.mark.parametrize("H", [0.2, 0.8])
def test_fgn_lag1_autocorrelation(H):
    # increments of the whole simulated process: context from Davies-Harte, continuation from the conditional
    proc = FractionalBM(H_hurst=H)
    rng = np.random.default_rng(0)
    P = 400
    st = proc.init_state(rng, 1)
    ctx, st = proc.step(st, 2048, rng)
    big = {k: np.repeat(v, P, axis=0) for k, v in st.items()}
    fut, _ = proc.step(big, 256, rng)
    inc_fut = np.diff(np.concatenate([np.repeat(ctx[:, :, -1:], P, axis=0), fut], axis=-1), axis=-1)[:, 0]
    rho1 = fgn_lag1_autocorr(H)
    est = np.mean(inc_fut[:, 1:] * inc_fut[:, :-1]) / np.mean(inc_fut**2)
    assert abs(est - rho1) < 0.03, (est, rho1)
    # Davies-Harte context alone
    z = davies_harte(np.random.default_rng(1), H, 2048, 200)
    est2 = np.mean(z[:, 1:] * z[:, :-1]) / np.mean(z**2)
    assert abs(est2 - rho1) < 0.02, (est2, rho1)
    assert abs(rho1 - 0.5 * (2 ** (2 * H) - 2)) < 1e-12


def test_fbm_conditional_matches_history_window():
    # a window of 1024 or 2048 increments gives nearly the same one-step predictor for H = 0.8
    from otp.generators.tails import _conditional_operator
    W1, L1 = _conditional_operator(0.8, 1.0, 1024, 1)
    W2, L2 = _conditional_operator(0.8, 1.0, 2048, 1)
    assert abs(L1[0, 0] - L2[0, 0]) / L2[0, 0] < 5e-3
    assert abs(W1.sum() - W2.sum()) < 0.02


def test_hawkes_counts():
    rate = 5.0
    for eta in (0.0, 0.6, 0.9):
        proc = Hawkes(eta=eta, rate=rate)
        obs, _ = _run(proc, 3000, P=60, seed=4)
        assert (obs >= 0).all() and np.array_equal(obs, np.round(obs))
        assert abs(obs.mean() - rate) / rate < 0.1, (eta, obs.mean())
    # clustering grows with the branching ratio: Fano factor (variance / mean) of counts summed over 64 bins
    def fano64(e):
        x = _run(Hawkes(eta=e), 3200, P=60, seed=4)[0][:, 0].reshape(60, 50, 64).sum(-1)
        return x.var() / x.mean()

    f = [fano64(e) for e in (0.0, 0.6, 0.9)]
    assert f[0] < 1.3 < f[1] < f[2]


def test_msm_structure():
    p = MSMCascade(k=6)
    g = p.renewal_probs()
    assert g.shape == (6,) and np.all(np.diff(g) > 0) and abs(g[-1] - 0.5) < 1e-12
    obs, st = _run(MSMCascade(k=6), 4000, P=20, seed=1)
    assert abs(obs.var() - 1.0) < 0.25                     # E[prod M] = 1
    ret, _ = _run(MSMCascade(k=6, observe="returns"), 200, P=1, seed=9)
    cum, _ = _run(MSMCascade(k=6, observe="cumsum"), 200, P=1, seed=9)
    assert np.allclose(np.cumsum(ret), cum)
    # volatility clustering: autocorrelation of |r| is positive for k > 1
    a = np.abs(_run(MSMCascade(k=8), 20000, P=1, seed=2)[0][0, 0])
    assert np.corrcoef(a[1:], a[:-1])[0, 1] > 0.05


def test_grid_configs_finite():
    cfgs = configs()
    ids = [c[0] for c in cfgs]
    assert len(ids) == len(set(ids)) == 12 + 7 + 6 + 10
    assert set(BIG_IDS) <= set(ids)
    for cid, factory in cfgs:
        proc = factory(0)
        s = make_sample(proc, 600, 32, seed=1, n_paths=40)
        assert np.isfinite(s.context).all() and np.isfinite(s.future).all() and np.isfinite(s.oracle_q).all(), cid
        assert s.meta["tags"], cid


def test_fbm_conditional_operator_matches_direct_formula():
    from scipy.linalg import toeplitz
    from otp.generators.tails import _conditional_operator, fgn_autocov
    H, h, n = 0.8, 40, 6
    S = toeplitz(fgn_autocov(H, h + n))
    Shh, Sfh, Sff = S[:h, :h], S[h:, :h], S[h:, h:]
    W_ref = Sfh @ np.linalg.inv(Shh)
    C_ref = Sff - W_ref @ Sfh.T
    W, L = _conditional_operator(H, 1.0, h, n)
    assert np.allclose(W, W_ref, atol=1e-6)
    assert np.allclose(L @ L.T, C_ref, atol=1e-6)
