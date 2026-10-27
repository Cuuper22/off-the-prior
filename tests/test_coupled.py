"""Tests for the coupled systems: structure, lag/map identities, oracle behaviour and the exact smoother."""
import numpy as np
import pytest
from scipy.stats import norm

from otp import metrics
from otp.generators import coupled as C
from otp.generators.base import LEVELS, make_sample, tile_state

ALL = list(C.SYSTEMS)


@pytest.mark.parametrize("system", ALL)
def test_shapes_and_determinism(system):
    spec = C.SYSTEMS[system]
    v = spec["values"][2]
    a = C.build_sample(system, v, 0, horizon=64, cond_horizons=(16, 64), n_paths=100, context_len=300)
    b = C.build_sample(system, v, 0, horizon=64, cond_horizons=(16, 64), n_paths=100, context_len=300)
    assert a.context.shape == (1, 300) and a.future.shape == (1, 64)
    assert a.oracle_q.shape == (1, 64, 21) and a.oracle_marg_q.shape == (1, 64, 21)
    assert a.past_cov.shape == (1, 300)
    assert (a.known_cov is not None) == spec["known"]
    if spec["known"]:
        assert a.known_cov.shape == (1, 364)
    assert np.isfinite(a.context).all() and np.isfinite(a.future).all() and np.isfinite(a.oracle_q).all()
    assert np.array_equal(a.context, b.context) and np.array_equal(a.oracle_q, b.oracle_q)
    assert (np.diff(a.oracle_q, axis=-1) >= -1e-9).all()
    assert a.meta["family"] == "coupled" and a.meta["knob"] == spec["knob"]


def test_lagged_driver_identity_without_noise():
    for lag in (0, 5, 40):
        p = C.LaggedDriver(lag=lag, noise_ratio=1e-9, period=12)
        s = C.make_cov_sample(p, 400, 50, seed=1, n_paths=20)
        x = s.known_cov[0]                     # x over context + horizon
        y = np.concatenate([s.context[0], s.future[0]])
        t = np.arange(lag, 450)
        assert np.allclose(y[t], p.params["a"] * x[t - lag], atol=1e-5)


def test_lagged_driver_conditional_oracle_shares_x_future_and_marginal_does_not():
    p = C.LaggedDriver(lag=0, period=12, noise_ratio=0.1)
    st = p.init_state(np.random.default_rng(0), 1)
    _, st = p.step(st, p.warmup_steps(), np.random.default_rng(1), np.random.default_rng(2))
    big = tile_state(st, 50)
    p.share_exo = True
    fut, _ = p.step(big, 30, np.random.default_rng(3), np.random.default_rng(4))
    assert np.allclose(fut[:, 1], fut[:1, 1])               # identical x across paths
    assert fut[:, 0].std(axis=0).max() > 0                  # y still varies (own noise)
    p.share_exo = False
    fut2, _ = p.step(tile_state(st, 50), 30, np.random.default_rng(3), np.random.default_rng(4))
    p.share_exo = True
    assert fut2[:, 1].std(axis=0).min() > 0                 # x varies across paths


def test_oracle_gap_between_conditional_and_marginal_when_x_leads_y():
    # lag 0: y_t follows x_t, so knowing x's future must give a much tighter oracle than not knowing it
    p = C.LaggedDriver(lag=0, period=24, noise_ratio=0.05)
    s = C.make_cov_sample(p, 600, 100, seed=3, n_paths=400)
    w_cond = (s.oracle_q[..., -1] - s.oracle_q[..., 0]).mean()
    w_marg = (s.oracle_marg_q[..., -1] - s.oracle_marg_q[..., 0]).mean()
    assert w_cond < 0.5 * w_marg
    # lag longer than the horizon: x's future is irrelevant, both oracles agree in width
    p = C.LaggedDriver(lag=200, period=24, noise_ratio=0.05)
    s = C.make_cov_sample(p, 600, 100, seed=3, n_paths=400)
    w_cond = (s.oracle_q[..., -1] - s.oracle_q[..., 0]).mean()
    w_marg = (s.oracle_marg_q[..., -1] - s.oracle_marg_q[..., 0]).mean()
    assert abs(w_cond - w_marg) < 0.2 * w_marg


def test_nonlinear_driver_maps():
    p0 = C.NonlinearDriver(nonlin=0.0, noise_ratio=1e-9)
    u = np.linspace(-2, 2, 9)
    assert np.allclose(p0._map(u), u)
    p1 = C.NonlinearDriver(nonlin=1.0, g="sin", omega=2.0)
    assert np.allclose(p1._map(u), np.sin(2 * u))
    p2 = C.NonlinearDriver(nonlin=1.0, g="threshold")
    assert set(np.unique(p2._map(u))) == {-1.0, 1.0}
    s = C.make_cov_sample(C.NonlinearDriver(nonlin=1.0, lag=7, noise_ratio=1e-9, g="threshold"), 300, 20, 2, n_paths=10)
    y = np.concatenate([s.context[0], s.future[0]])
    sd = C.NonlinearDriver(nonlin=1.0, lag=7).sd_x
    assert np.allclose(np.abs(y[10:]), sd, atol=1e-5)


def test_feedback_linear_coupling_zero_is_independent_and_stable_grid():
    for k in C.SYSTEMS["FeedbackLinear"]["values"]:
        assert C.FeedbackLinear(coupling=k).spectral_radius() < 1.0
    p = C.FeedbackLinear(coupling=0.0)
    out, _ = p.step(p.init_state(np.random.default_rng(0), 4000), 400, np.random.default_rng(1))
    # with no coupling the cross-correlation of y and x is zero up to sampling error
    assert abs(np.corrcoef(out[:, 0, -1], out[:, 1, -1])[0, 1]) < 0.06
    q = C.FeedbackLinear(coupling=1.0)
    out, _ = q.step(q.init_state(np.random.default_rng(0), 4000), 1500, np.random.default_rng(1))
    c1 = np.corrcoef(out[:, 0, -1], out[:, 1, -1])[0, 1]
    assert abs(c1) > 0.1


def test_predprey_positive_bounded_and_oscillates_only_when_coupled():
    for k in C.SYSTEMS["FeedbackPredPrey"]["values"]:
        p = C.FeedbackPredPrey(coupling=k)
        st = p.init_state(np.random.default_rng(0), 1)
        _, st = p.step(st, p.warmup_steps(), np.random.default_rng(1))
        out, _ = p.step(st, 3000, np.random.default_rng(2))
        assert (out > 0).all() and out.max() < 50
    def acf_neg(k):
        p = C.FeedbackPredPrey(coupling=k)
        st = p.init_state(np.random.default_rng(0), 1)
        _, st = p.step(st, p.warmup_steps(), np.random.default_rng(1))
        y = p.step(st, 6000, np.random.default_rng(2))[0][0, 0]
        y = y - y.mean()
        lags = np.arange(1, 400)
        return min(np.dot(y[:-l], y[l:]) / np.dot(y, y) for l in lags)
    assert acf_neg(1.0) < -0.2
    assert acf_neg(0.0) > -0.1


def test_hidden_driver_x_noise_and_lag_identity():
    p = C.HiddenDriver(lag=10, noise_ratio=1e-9, x_noise=1e-9, period=12)
    s = C.make_cov_sample(p, 300, 40, seed=5, n_paths=10)
    x = np.concatenate([s.past_cov[0]])
    y = s.context[0]
    t = np.arange(10, 300)
    assert np.allclose(y[t], x[t - 10], atol=1e-5)
    p = C.HiddenDriver(lag=10, x_noise=2.0)
    out, _ = p.step(p.init_state(np.random.default_rng(0), 3000), 600, np.random.default_rng(1))
    # x - z has sd x_noise * sd_z
    z = out[:, 0, 10:]  # not used directly; check x noise through variance of x relative to sd_z
    assert out[:, 1, -1].std() > 1.5 * p.sd_z


def test_leaky_known_smoother_matches_brute_force_gaussian():
    rho, se, ratio, h = 0.9, 0.3, 1.5, 12
    p = C.LeakyKnown(rho=rho, sigma_e=se, noise_ratio=ratio, period=12)
    rng = np.random.default_rng(0)
    st = p.init_state(rng, 1)
    _, st = p.step(st, 300, rng)
    ctx, st1 = p.step(st, 5, rng)
    big, _ = p.step(tile_state(st1, 1), h, rng)
    xf = big[0, 1]
    got = p.conditional_oracle(st1, xf, [h])[h][0]          # (h, 21)
    # direct Gaussian conditioning: u_k = rho^{k+1} u_prev + noise; Cov(u_i, u_j) = s2 rho^{|i-j|} (1-rho^{2(min+1)})/(1-rho^2)
    u_prev = float(st1["u"][0])
    k = np.arange(h)
    cov = np.array([[se**2 * rho ** abs(i - j) * (1 - rho ** (2 * (min(i, j) + 1))) / (1 - rho**2) for j in k] for i in k])
    mu = rho ** (k + 1) * u_prev
    t0 = float(st1["t"][0])
    m = p.season_at(t0 + k)
    obs = np.arange(h - 1)
    z = xf[1:h] - m[: h - 1]
    r = (ratio * se) ** 2
    sxx = cov[np.ix_(obs, obs)] + r * np.eye(h - 1)
    kg = cov[:, obs] @ np.linalg.inv(sxx)
    post_mu = mu + kg @ (z - mu[obs])
    post_var = np.diag(cov) - np.einsum("ij,ji->i", kg, cov[obs, :])
    want = (m + post_mu)[:, None] + np.sqrt(post_var)[:, None] * C.Z[None, :]
    assert np.allclose(got, want, atol=1e-8)


def test_leaky_known_oracle_calibrated_and_sharper_than_marginal():
    cover = []
    wc, wm = [], []
    for seed in range(60):
        s = C.build_sample("LeakyKnown", 0.25, seed, horizon=64, cond_horizons=(16, 64), n_paths=300, context_len=400)
        q = s.oracle_cond_by_h[64]
        cover.append((s.future[0][:, None] <= q[0]).mean(axis=0))
        wc.append((q[..., 15] - q[..., 5]).mean())
        wm.append((s.oracle_marg_q[..., 15] - s.oracle_marg_q[..., 5]).mean())
    cov = np.mean(cover, axis=0)
    assert np.abs(cov - LEVELS).max() < 0.05
    assert np.mean(wc) < 0.6 * np.mean(wm)


def test_leaky_known_cond_by_h_prefix_differs_but_matches_h():
    s = C.build_sample("LeakyKnown", 1.0, 2, horizon=128, cond_horizons=(16, 128), n_paths=100, context_len=300)
    assert s.oracle_cond_by_h[16].shape == (1, 16, 21) and s.oracle_cond_by_h[128].shape == (1, 128, 21)
    assert np.array_equal(s.oracle_for(True, 128), s.oracle_cond_by_h[128])
    assert np.array_equal(s.oracle_for(False, 16), s.oracle_marg_q[..., :16, :])
    # more future x lowers or keeps posterior width at the same step (smoothing uses extra measurements)
    w16 = (s.oracle_cond_by_h[16][0, :15, 15] - s.oracle_cond_by_h[16][0, :15, 5]).sum()
    w128 = (s.oracle_cond_by_h[128][0, :15, 15] - s.oracle_cond_by_h[128][0, :15, 5]).sum()
    assert w128 <= w16 + 1e-9


def test_save_load_roundtrip(tmp_path):
    ss = [C.build_sample("LeakyKnown", 1.0, i, horizon=32, cond_horizons=(16, 32), n_paths=50, context_len=200) for i in range(2)]
    C.save_cov_samples(str(tmp_path / "s.npz"), ss)
    back = C.load_cov_samples(str(tmp_path / "s.npz"))
    assert np.array_equal(back[1].known_cov, ss[1].known_cov)
    assert np.array_equal(back[1].oracle_cond_by_h[16], ss[1].oracle_cond_by_h[16])
    assert back[0].meta["sample_id"] == ss[0].meta["sample_id"]
    ss = [C.build_sample("FeedbackLinear", 0.15, i, horizon=32, n_paths=50, context_len=200) for i in range(2)]
    C.save_cov_samples(str(tmp_path / "f.npz"), ss)
    back = C.load_cov_samples(str(tmp_path / "f.npz"))
    assert back[0].known_cov is None and back[0].oracle_cond_by_h is None
    assert np.array_equal(back[0].oracle_marg_q, ss[0].oracle_marg_q)


def test_oracle_scores_calibrated_on_mc_systems():
    # the MC oracle quantiles should be calibrated against realized futures drawn from the same law
    for system in ("FeedbackLinear", "FeedbackPredPrey", "HiddenDriver", "LaggedDriver"):
        v = C.SYSTEMS[system]["values"][2]
        cover = []
        for seed in range(20):
            s = C.build_sample(system, v, seed, horizon=32, cond_horizons=(32,), n_paths=300, context_len=300)
            cover.append((s.future[0][:, None] <= s.oracle_for(True, 32)[0]).mean(axis=0))
        assert np.abs(np.mean(cover, axis=0) - LEVELS).max() < 0.08, system
