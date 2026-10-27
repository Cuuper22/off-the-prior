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
    real = R.load_union(BENCH, "real")
    prior = load_samples(os.path.join(BENCH, "in_prior.npz"))
    mr, mp = man[man.set == "real"], man[man.set == "in_prior"]
    assert len(prior) == len(mp) == 500
    assert 300 <= len(mr) <= 600
    # on a machine without the restricted file, only redistributable windows load
    n_ok = int(mr.redistributable.sum())
    assert len(real) >= n_ok
    byid = {s.meta["sample_id"]: s for s in real}
    for _, r in mr.iterrows():
        if r["sample_id"] not in byid:
            continue
        s = byid[r["sample_id"]]
        assert s.T == r["T"] and s.H == r["H"] and s.period == r["m"] and 256 <= s.T <= 2048 and s.H <= 720
        assert np.isfinite(s.context).all() and np.isfinite(s.future).all() and not R.has_oracle(s)
    assert mr["source_sha256"].notna().all() and mr["revision"].notna().all()
    assert all(s.T == 2048 and s.H == 128 and R.has_oracle(s) for s in prior)
    assert [s.meta["seed"] for s in prior] == list(range(500))
    assert len(set(mr.sample_id)) == len(mr)
    for f in os.listdir(BENCH):
        if f.endswith(".npz") and not f.endswith("_restricted.npz"):
            assert os.path.getsize(os.path.join(BENCH, f)) < 50e6


# ----------------------------------------------------------------------------- sources, license split, analogues

from otp import realdata_sources as RS  # noqa: E402


def test_read_tsf():
    text = ("# comment\n@relation x\n@attribute series_name string\n@attribute start_timestamp date\n@frequency hourly\n"
            "@horizon 24\n@data\nT1:2009-05-01 00-00-00:1,2,?,4\nT2:2010-01-01 05-00-00:5,6\n")
    hdr, ser = RS.read_tsf(text)
    assert hdr["frequency"] == "hourly" and len(ser) == 2
    name, start, vals = ser[0]
    assert name == "T1" and start == pd.Timestamp("2009-05-01 00:00:00")
    assert np.isnan(vals[2]) and vals[3] == 4
    assert ser[1][1] == pd.Timestamp("2010-01-01 05:00:00")


def test_read_ghcn_dly(tmp_path):
    def line(element, year, month, vals):
        s = f"USW00094728{year:04d}{month:02d}{element}"
        for d in range(31):
            v = vals.get(d, -9999)
            s += f"{v:5d}" + "   "
        return s + "\n"

    p = tmp_path / "x.dly"
    p.write_text(line("TMAX", 2020, 1, {0: 100, 1: 250}) + line("PRCP", 2020, 1, {0: 5}))
    s = R_ghcn = RS.read_ghcn_dly(str(p), "TMAX")
    assert s.loc["2020-01-01"] == 10.0 and s.loc["2020-01-02"] == 25.0 and np.isnan(s.loc["2020-01-03"])
    assert RS.read_ghcn_dly(str(p), "PRCP").loc["2020-01-01"] == 0.5


def test_item_window_origin_and_business_days():
    idx = pd.bdate_range("2015-01-01", periods=3000)
    v = np.cumsum(np.random.default_rng(0).standard_normal(3000)) + 1000
    v[500] = np.nan  # a holiday: dropped for business-day series
    it = RS.Item("fred", "d", "x", v, "B", "finance", idx, horizon=30, origin=str(idx[2000].date()), m=5, family="regime_switches")
    ctx, fut, info = RS.item_window(it)
    assert len(fut) == 30 and len(ctx) == 2000 - 1  # one holiday dropped before the origin
    assert info["origin_time"].startswith(str(idx[2000].date()))
    it2 = RS.Item("fred", "d", "x", v, "B", "finance", idx, horizon=30, transform="log_return")
    ctx2, fut2, info2 = RS.item_window(it2)
    assert len(fut2) == 30 and np.isfinite(ctx2).all() and np.abs(ctx2).max() < 1
    late = RS.Item("fred", "d", "x", v, "B", "finance", idx, horizon=30, origin="2100-01-01")
    assert RS.item_window(late)[2] == "origin_outside_record"


def test_cutoff_applies_to_live_series():
    idx = pd.date_range("2025-01-01", periods=700, freq="D")
    v = np.random.default_rng(1).standard_normal(700).cumsum()
    it = RS.Item("fred", "d", "x", v, "D", "finance", idx, horizon=30, t_min=100)
    ctx, fut, info = RS.item_window(it)
    assert pd.Timestamp(info["origin_time"]) + pd.Timedelta(days=29) <= RS.CUTOFF


def test_business_day_conventions():
    assert R.gift_eval_horizon("B") == 30 and R.seasonal_period("B") == 5


def test_save_split_and_load_union(tmp_path):
    spec = R.Spec("gifteval", "fake", "H", "energy", n_series=6)
    s, rows = R.load_pair(spec, src=FakeSource())
    flags = [i % 2 == 0 for i in range(len(s))]
    counts = R.save_split(str(tmp_path), "demo", s, flags)
    assert counts == {"redistributable": 3, "restricted": 3}
    man = pd.DataFrame(rows)
    man["set"] = "demo"
    man.to_csv(tmp_path / "manifest.csv", index=False)
    full = R.load_union(str(tmp_path), "demo")
    assert [x.meta["sample_id"] for x in full] == [x.meta["sample_id"] for x in s]
    os.remove(tmp_path / "demo_restricted.npz")
    part = R.load_union(str(tmp_path), "demo")
    assert [x.meta["manifest_row"] for x in part] == [0, 2, 4]


def test_licence_records_complete():
    for sid, s in RS.SOURCES.items():
        lic = s["license"]
        assert lic.name and lic.url.startswith("http") and lic.evidence
    assert RS.SOURCES["silso"]["license"].redistributable is False
    assert RS.SOURCES["fev"]["license"].redistributable is False
    assert RS.SOURCES["fred"]["license"].redistributable is True


@pytest.mark.skipif(not os.path.exists(os.path.join(BENCH, "analogues.npz")), reason="regression sets not built")
def test_built_analogues_and_license_split():
    man = pd.read_csv(os.path.join(BENCH, "manifest.csv"))
    ana = man[man.set == "analogues"]
    assert set(ana.family_analogue) == {"delay_feedback", "coupled_oscillators", "heavy_tails", "regime_switches"}
    assert ana.reason.notna().all() and (ana.reason.str.len() > 10).all()
    src = pd.read_csv(os.path.join(BENCH, "sources.csv"))
    assert src.license.notna().all() and src.sha256.notna().all() and src.date_fetched_utc.notna().all()
    for name in ("real", "real_extra", "analogues"):
        committed = {s.meta["sample_id"] for s in R.load_real(os.path.join(BENCH, f"{name}.npz"))}
        sub = man[(man.set == name) & man.sample_id.isin(committed)]
        assert sub.redistributable.all() and len(sub) == len(committed)
    # restricted sources never appear in committed files
    assert not (man[(man.set != "in_prior") & ~man.redistributable].sample_id.isin(
        {s.meta["sample_id"] for n in ("real", "real_extra", "analogues") for s in R.load_real(os.path.join(BENCH, f"{n}.npz"))})).any()


def test_subset_chunks_have_their_own_file_names():
    """A subset run must not write files that a later full run would pick up as its own chunks."""
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location("score_regression", os.path.join(os.path.dirname(__file__), "..", "scripts", "score_regression.py"))
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    full = sr.part_path("out", "real", "arima", 3)
    sub = sr.part_path("out", "real", "arima", 3, "sub100s0_")
    assert full != sub
    assert os.path.basename(full) == "real__arima__003.csv"
    assert os.path.basename(sub) == "real__arima__sub100s0_003.csv"
