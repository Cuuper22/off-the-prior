import numpy as np

from otp import metrics
from otp.evaluate import score_samples, with_reference_ratios
from otp.generators.base import make_sample, save_samples, load_samples
from otp.generators.primitives import in_prior
from otp.model import forecast, forecast_torch, set_threads


def test_metrics_perfect_and_calibration():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((1, 2000))
    from scipy.stats import norm
    q = np.broadcast_to(norm.ppf(metrics.LEVELS), (1, 2000, 21))
    sc = metrics.score(rng.standard_normal((1, 500)), y, q)
    assert sc["calib"][0] < 0.03
    assert abs(sc["cov80"][0] - 0.8) < 0.03


def test_pipeline_small(tmp_path):
    set_threads(1)
    samples = [make_sample(in_prior(s), 512, 64, seed=s, n_paths=200) for s in range(4)]
    p = tmp_path / "s.npz"
    save_samples(str(p), samples)
    back = load_samples(str(p))
    assert np.allclose(back[0].context, samples[0].context)
    df = score_samples(samples, ["timestral-5M", "seasonal_naive", "oracle"])
    df = with_reference_ratios(df)
    assert set(df.forecaster) == {"timestral-5M", "seasonal_naive", "oracle"}
    assert np.isfinite(df.mase_ratio_oracle).all()


def test_forecast_torch_matches_forecast():
    import torch
    set_threads(1)
    x = np.sin(np.arange(1000) / 9.0) + 0.05 * np.random.default_rng(1).standard_normal(1000) + 3.0
    q_np = forecast("5M", x[None], 64, clip_nonnegative=False)[0, 0]
    xt = torch.tensor(x[None], dtype=torch.float32, requires_grad=True)
    q_t = forecast_torch("5M", xt, 64)
    q_t[..., 10].sum().backward()
    assert xt.grad is not None and torch.isfinite(xt.grad).all()
    assert np.allclose(q_t[0].detach().numpy(), q_np, atol=1e-3 * (1 + np.abs(q_np).max()))


def test_nonfinite_quantiles_score_as_nan():
    q = np.zeros((1, 8, 21)) + np.linspace(-1, 1, 21)
    q[0, 3, 20] = np.inf
    sc = metrics.score(np.random.default_rng(0).standard_normal((1, 64)), np.zeros((1, 8)), q)
    assert np.isnan(sc["mase"][0]) and np.isnan(sc["cov80"][0]) and np.isnan(sc["coverage"][0]).all()
