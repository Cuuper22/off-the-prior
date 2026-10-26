import numpy as np

from otp.internals import attention_profile, ridge_readout, trace
from otp.model import forecast, set_threads


def test_trace_matches_forecast_and_rows_sum_to_one():
    set_threads(1)
    rng = np.random.default_rng(0)
    x = np.sin(np.arange(1500) / 7.0)[None].repeat(2, 0) + 0.05 * rng.standard_normal((2, 1500)) + 2
    tr = trace("5M", x, 96)
    ref = forecast("5M", x, 96, symmetric_averaging=False, clip_nonnegative=False)[:, 0]
    assert np.allclose(tr["quantiles"], ref, atol=1e-3 * (1 + np.abs(ref).max()))
    for w in tr["attn"].values():
        assert np.allclose(w.sum(-1), 1.0, atol=1e-4)
    prof = attention_profile(tr, period=96 * 2, change_step=300)
    assert prof["recent"].shape == (2, len(tr["attn"]))


def test_ridge_readout_recovers_linear_signal():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((200, 10))
    y = f[:, 0] * 2 + 0.1 * rng.standard_normal(200)
    assert ridge_readout(f, y)["r2"] > 0.9
