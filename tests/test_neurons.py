"""Tests for the neuron audit package (small corpora, CPU, the 5M model only where a forward pass is needed)."""

import numpy as np
import pandas as pd
import pytest
import torch

from otp.neurons import corpus as C
from otp.neurons import labels as Lb
from otp.neurons import realdata as R
from otp.neurons import score as Sc
from otp.neurons import steer
from otp.neurons import variables as V

T = 256


@pytest.fixture(scope="module")
def small_corpus():
    specs = [s for s in C.make_specs(0.05) if s["generator"] in ("in_prior", "ctl_season", "ctl_trend", "ctl_ar", "stable_ar",
                                                                 "mean_jump", "variance_jump")][:60]
    c = C.build_corpus(specs, T, workers=1)
    allspecs = C.make_specs(0.05)
    split, fold = C.split_of(allspecs)
    c["sid"] = np.asarray([s["sid"] for s in specs])
    c["split"], c["fold"] = split[c["sid"]], fold[c["sid"]]
    return c, specs


def test_corpus_is_deterministic():
    specs = [s for s in C.make_specs(0.05) if s["generator"] == "in_prior"][:3]
    a = C.build_corpus(specs, T, workers=1)
    b = C.build_corpus(specs, T, workers=1)
    for k in ("values", "pv", "sv"):
        assert np.array_equal(a[k], b[k], equal_nan=True)


def test_token_table_shapes(small_corpus):
    c, specs = small_corpus
    tab = Lb.token_table(c)
    p = T // 32 + 4
    assert tab["X"].shape == (len(specs) * p, len(V.LIBRARY))
    ix = V.VAR_INDEX
    fc = tab["X"][:, ix["tok_fc"]]
    assert fc.reshape(len(specs), p)[:, -4:].min() == 1 and fc.reshape(len(specs), p)[:, :-4].max() == 0


def test_bins_ignore_undefined():
    x = np.array([np.nan, 1, 2, 3, 4, np.nan] * 40, np.float32)[:, None]
    codes, nbins, edges = Lb.make_bins(x, np.ones(len(x), bool), nb=2)
    assert (codes[np.isnan(x[:, 0])] == 2).all()
    assert codes[~np.isnan(x[:, 0])].max() <= 1


def test_planted_neurons_are_recovered(small_corpus):
    c, specs = small_corpus
    tab = Lb.token_table(c)
    X, ix = tab["X"], V.VAR_INDEX
    rng = np.random.default_rng(0)

    def fill(v):
        x = X[:, ix[v]].copy()
        x[~np.isfinite(x)] = 0
        return x

    S = X.shape[0]
    Z = rng.standard_normal((S, 6)).astype(np.float32) * 0.05
    Z[:, 0] += 3 * fill("lat_slope")
    Z[:, 1] += -2 * fill("m_level_ctx")
    Z[:, 2] += 1.5 * (fill("tok_fc") > 0.5)
    Z[:, 3] = rng.standard_normal(S)
    prep = Lb.prepare(tab, "cpu")
    res = Lb.label_layer(prep, torch.as_tensor(Z).half(), torch.as_tensor((Z > 0).astype(np.int8) * 3))
    pos = pd.DataFrame(Lb.summarize_sign(res["pos"], 0, "pos", "t"))
    neg = pd.DataFrame(Lb.summarize_sign(res["neg"], 0, "neg", "t"))
    assert pos.loc[0, "label"] in ("lat_slope", "m_slope", "s_trend", "x_trend", "m_monotone") and pos.loc[0, "r2_cv"] > 0.3
    assert neg.loc[1, "label"] in ("lat_level", "m_level_ctx", "prev_level", "m_dlevel_prev") and neg.loc[1, "r2_cv"] > 0.2
    assert pos.loc[2, "label"] in ("tok_fc", "tok_fc_step", "pos_from_end") and pos.loc[2, "r2_cv"] > 0.5
    assert neg.loc[1, "label_corr"] < 0 or pos.loc[1, "label_corr"] != 0
    assert pos.loc[3, "r2_cv"] < 0.05                      # pure noise: nothing explains it
    assert pos.loc[0, "r2_heldout"] > 0.15                 # the label still works on held-out series


def test_class_rules():
    df = pd.DataFrame({"dead": [False, False, False, True], "E": [0.5, 0.5, 0.02, 0.9], "ratio": [0.95, 0.4, 1.0, 1.0]})
    assert list(Sc.klass(df, 0.7, 0.1)) == ["one_job", "mixed", "unexplained", "dead"]
    assert list(Sc.klass(df, 0.3, 0.1)) == ["one_job", "one_job", "unexplained", "dead"]


def test_related_groups():
    assert Sc.related("level", "level") == "same_group"
    assert Sc.related("level", "slope") == "related_groups"
    assert Sc.related("level", "tail") == "unrelated_groups"


def test_prereg_has_every_label_in_the_library():
    assert set(steer.PREREG) <= set(V.LIBRARY)
    md = steer.prereg_markdown()
    assert "Matching criterion" in md and "Pole sample" in md


def test_judge_accepts_a_clean_monotone_effect():
    rng = np.random.default_rng(1)
    eff = [{"level": 0.2 * s + rng.normal(0, 0.01, 40)} for s in steer.SCALES]
    r = steer.judge(eff, "level", "+", 1.0)
    assert r["match"] and r["rho"] > 0.9
    r_neg = steer.judge(eff, "level", "+", -1.0)           # label correlation negative: increasing the pole should lower the level
    assert not r_neg["match"]


def test_real_window_rule():
    rng = np.random.default_rng(0)
    v = np.cumsum(rng.normal(size=5000))
    v[100:400] = np.nan
    st = R._starts(v, rng, 3)
    assert st and all(not np.isnan(v[s:s + R.WIN]).any() or np.isnan(v[s:s + R.WIN]).mean() <= R.MAX_NAN_FRAC for s in st)
    assert all(abs(a - b) >= R.WIN for i, a in enumerate(st) for b in st[i + 1:])
    flat = np.ones(5000)
    assert R._starts(flat, rng, 3) == []


def test_steering_scale_one_is_the_unsteered_pass():
    rng = np.random.default_rng(0)
    ctx = rng.normal(size=(3, 128)).astype(np.float32).cumsum(1)
    q = steer.run_conditions("5M", ctx, [[], [(2, 5, +1, 1.0)], [(2, 5, +1, 0.0)]], device="cpu")
    assert np.allclose(q[0], q[1], atol=1e-5)
    assert np.abs(q[0] - q[2]).max() > 0                     # ablating a pole changes the forecast


def test_wilson_interval():
    from otp.neurons import summary as Sm
    lo, hi = Sm.wilson(0, 10)
    assert lo == 0 and 0.2 < hi < 0.4
    lo, hi = Sm.wilson(5, 10)
    assert lo < 0.5 < hi


def test_squeeze_and_scale_match_tables():
    from otp.neurons import summary as Sm
    n = 40
    rng = np.random.default_rng(0)
    t = pd.DataFrame({"size": "5M", "layer": 0, "neuron": np.arange(n), "sign": "pos", "dead": False, "label": "lat_level",
                      "group": "level", "r2_cv": rng.uniform(0.05, 0.4, n), "E": 0.4, "ratio": rng.uniform(0.3, 1.0, n),
                      "r2_pair": 0.4})
    sq = Sm.squeeze({"w4": t, "w32": t})
    assert list(sq["model"]) == ["w4", "w32"] and sq["n_poles"].iloc[0] == n
    m = pd.DataFrame({"small": "5M", "large": "28M", "layer": 0, "sign": "pos", "neuron": np.arange(n), "label": "lat_level",
                      "group": "level", "r2_cv": 0.2, "best_corr": rng.uniform(0.3, 1, n), "second_corr": 0.2, "n_ge50": 2, "n_ge70": 1,
                      "match_layer": 0, "match_sign": "pos", "match_neuron": 0, "match_label": "lat_level", "match_group": "level",
                      "match_r2_cv": 0.3})
    sm = Sm.scale_match(m, t)
    assert len(sm) == 1 and sm["n_one_job_small"].iloc[0] > 0


def test_onehot_moves_codes_to_the_requested_device():
    codes = torch.tensor([[0, 3], [1, 2]], dtype=torch.int16)          # CPU codes
    o = Lb.onehot(codes, 4, torch.device("meta"))                        # shape-only device: fails if codes are not moved
    assert o.shape == (2, 8) and o.device.type == "meta"
    o = Lb.onehot(codes, 4, "cpu")
    assert o.sum().item() == 4 and o[0, 0] == 1 and o[0, 4 + 3] == 1
