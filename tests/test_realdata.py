"""Tests for the real-data loaders and the regression-set scripts (no network needed)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

from otp import realdata as R
from otp.generators.base import load_samples

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
BENCH = os.path.join(ROOT, "bench", "regression")


def test_gift_eval_horizon_convention():
    assert R.gift_eval_horizon("H") == 48
    assert R.gift_eval_horizon("15T") == 48
    assert R.gift_eval_horizon("10T") == 48
    assert R.gift_eval_horizon("D") == 30
    assert R.gift_eval_horizon("W-SUN") == 8
    assert R.gift_eval_horizon("M") == 12
    assert R.gift_eval_horizon("D", "m4_daily") == 14
    assert R.gift_eval_horizon("W", "m4_weekly") == 13
    assert R.gift_eval_horizon("M", "m4_monthly") == 18
    assert R.gift_eval_horizon("H", term="medium") == 480
    assert R.gift_eval_horizon("H", term="long") == 720
    assert R.gift_eval_horizon("T", term="long") == 720
    assert R.gift_eval_horizon("S", term="long") == R.MAX_HORIZON  # 900 capped


def test_seasonal_period():
    assert R.seasonal_period("H") == R.seasonal_period("1H") == 24
    assert R.seasonal_period("15T") == 96
    assert R.seasonal_period("10T") == 144
    assert R.seasonal_period("D") == 7
    assert R.seasonal_period("W-SUN") == 52
    assert R.seasonal_period("M") == 12
    assert R.seasonal_period("A") == 1


def test_fill_missing():
    x = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])
    assert np.allclose(R.fill_missing(x), [1, 1, 1, 1, 4, 4])
    y = np.arange(5.0)
    assert np.array_equal(R.fill_missing(y), y)


def test_cut_window_shapes_and_rules():
    x = np.arange(5000, dtype=float) + np.sin(np.arange(5000))
    ctx, fut, info = R.cut_window(x, 48)
    assert len(ctx) == 2048 and len(fut) == 48
    assert np.array_equal(fut, x[-48:]) and np.array_equal(ctx, x[-2096:-48])
    assert info["origin"] == 5000 - 48 and info["ctx_start"] == 5000 - 48 - 2048
    # short series: context is whatever is left, as long as it is at least 256
    ctx, fut, info = R.cut_window(x[:400], 48)
    assert len(ctx) == 352
    assert R.cut_window(x[:300], 48)[2] == "too_short"
    # missing value in the future: rejected; in the context: filled
    z = x[:1000].copy()
    z[-3] = np.nan
    assert R.cut_window(z, 48)[2] == "missing_in_future"
    z = x[:1000].copy()
    z[100] = np.nan
    ctx, _, info = R.cut_window(z, 48)
    assert np.isfinite(ctx).all() and info["n_filled"] == 1
    z = x[:1000].copy()
    z[:200] = np.nan
    assert R.cut_window(z, 48)[2] == "too_many_missing"
    assert R.cut_window(np.ones(1000), 48)[2] == "constant_or_empty"


class FakeSource(R.SeriesSource):
    source_file = "fake/file"
    revision = "rev"
    sha256 = "abc"

    def __init__(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        self.series = [np.cumsum(rng.standard_normal(int(rng.integers(100, 1500)))) for _ in range(n)]
        self.series[3][-2] = np.nan

    def __len__(self):
        return len(self.series)

    def get(self, i):
        return self.series[i]

    def series_id(self, i):
        return f"s{i}"


def test_load_pair_is_deterministic_and_sample_compatible():
    spec = R.Spec("gifteval", "fake", "H", "energy", n_series=10)
    s1, r1 = R.load_pair(spec, src=FakeSource())
    s2, r2 = R.load_pair(spec, src=FakeSource())
    assert len(s1) == 10 and len(r1) == 10
    for a, b in zip(s1, s2):
        assert np.array_equal(a.context, b.context) and a.meta["sample_id"] == b.meta["sample_id"]
    for s, r in zip(s1, r1):
        assert s.context.shape == (1, s.T) and s.future.shape == (1, 48) and s.T >= 256
        assert s.period == 24 and r["m"] == 24 and r["H"] == 48 and r["T"] == s.T
        assert np.isnan(s.oracle_q).all() and s.oracle_q.shape == (1, 48, 21)
        assert not R.has_oracle(s)
        assert np.isfinite(s.context).all() and np.isfinite(s.future).all()
    # a different seed picks a different order
    s3, _ = R.load_pair(spec, seed=1, src=FakeSource())
    assert [s.meta["sample_id"] for s in s3] != [s.meta["sample_id"] for s in s1]


def test_save_load_real_roundtrip(tmp_path):
    spec = R.Spec("gifteval", "fake", "D", "sales", n_series=6)
    s, _ = R.load_pair(spec, src=FakeSource())
    p = str(tmp_path / "r.npz")
    R.save_real(p, s)
    back = R.load_real(p)
    assert len(back) == len(s)
    for a, b in zip(s, back):
        assert np.array_equal(a.context, b.context) and np.array_equal(a.future, b.future)
        assert a.period == b.period and a.meta == b.meta
        assert np.isnan(b.oracle_q).all()


def test_summary_skips_oracle_for_real_and_uses_snaive_reference():
    import score_regression as S

    rows = []
    for g in ("a", "b"):
        for w in range(3):
            for f, mase in (("seasonal_naive", 2.0), ("timestral-5M", 1.0), ("ets", 4.0)):
                rows.append({"set": "real", "window_index": w, "generator": g, "family": "real", "period": 24,
                             "forecaster": f, "T": 300, "H": 48, "mase": mase, "scrps": mase / 2, "wql": 0.5,
                             "cov80": 0.8, "cov98": 0.98, "calib": 0.05})
    summ = S.summarize(pd.DataFrame(rows))
    assert "mase_rel_oracle" not in summ.columns
    assert set(summ.basis) == {"own", "common"}
    all_ = summ[(summ.group == "ALL") & (summ.basis == "common")].set_index("forecaster")
    assert np.isclose(all_.loc["timestral-5M", "mase_rel_snaive"], 0.5)
    assert np.isclose(all_.loc["ets", "mase_rel_snaive"], 2.0)
    assert np.isclose(all_.loc["seasonal_naive", "scrps_rel_snaive"], 1.0)
    assert S.forecasters_for("real", ["5M"]) == ["timestral-5M", "seasonal_naive", "ets", "arima"]
    assert S.forecasters_for("in_prior", ["5M"])[-1] == "oracle"


def test_subset_indices_keep_every_group():
    import score_regression as S

    spec = R.Spec("gifteval", "fake", "H", "energy", n_series=12)
    s, _ = R.load_pair(spec, src=FakeSource())
    for i, x in enumerate(s):
        x.meta["generator"] = "g" + str(i % 3)
    idx = S.subset_indices(s, 6, seed=0)
    assert {s[i].meta["generator"] for i in idx} == {"g0", "g1", "g2"}
    assert idx == S.subset_indices(s, 6, seed=0)
    assert S.subset_indices(s, None, 0) == list(range(len(s)))


@pytest.mark.skipif(not os.path.exists(os.path.join(BENCH, "real.npz")), reason="regression sets not built")
def test_built_sets_match_manifest():
    man = pd.read_csv(os.path.join(BENCH, "manifest.csv"))
    real = R.load_real(os.path.join(BENCH, "real.npz"))
    prior = load_samples(os.path.join(BENCH, "in_prior.npz"))
    mr, mp = man[man.set == "real"], man[man.set == "in_prior"]
    assert len(real) == len(mr) and len(prior) == len(mp) == 500
    assert 300 <= len(real) <= 600
    assert list(mr.sample_id) == [s.meta["sample_id"] for s in real]
    for s, (_, r) in zip(real, mr.iterrows()):
        assert s.T == r["T"] and s.H == r["H"] and s.period == r["m"] and 256 <= s.T <= 2048 and s.H <= 720
        assert np.isfinite(s.context).all() and np.isfinite(s.future).all() and not R.has_oracle(s)
    assert mr["source_sha256"].notna().all() and mr["revision"].notna().all()
    assert all(s.T == 2048 and s.H == 128 and R.has_oracle(s) for s in prior)
    assert [s.meta["seed"] for s in prior] == list(range(500))
    assert len(set(mr.sample_id)) == len(mr)
    for f in ("real.npz", "in_prior.npz"):
        assert os.path.getsize(os.path.join(BENCH, f)) < 50e6
