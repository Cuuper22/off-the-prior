"""Tests for the follow-up A tools (CPU, 5M model for the forward passes)."""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
import torch

from otp.model import load
from otp.neurons import followup as F
from otp.neurons import steer as St

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ridge_from_statistics_matches_direct_solution():
    rng = np.random.default_rng(0)
    n, d = 400, 6
    X = rng.standard_normal((n, d)) * rng.uniform(0.5, 3, d) + rng.uniform(-2, 2, d)
    y = X @ rng.standard_normal(d) + 0.3 * rng.standard_normal(n) + 1.0
    s = F.stat_from_arrays(X, y)
    for a in (0.1, 10.0):
        w, b0 = F.ridge_path(s, "y", (a,))[a]
        mu, sd = X.mean(0), X.std(0)
        Z = (X - mu) / sd
        ws = np.linalg.solve(Z.T @ Z + a * np.eye(d), Z.T @ (y - y.mean()))
        assert np.allclose(w, ws / sd, atol=1e-8)
        assert np.isclose(b0, y.mean() - mu @ (ws / sd))
        pred = X @ w + b0
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        assert np.isclose(F.r2_of(s, "y", w, b0), r2)


def test_steer_vector_moves_readout_by_move():
    rng = np.random.default_rng(1)
    w = rng.standard_normal(32)
    for m in (-2.0, 0.5, 2.0):
        v = F.steer_vector(w, m)
        assert np.isclose(w @ v, m)
        assert np.isclose(np.linalg.norm(v), abs(m) / np.linalg.norm(w))


def test_fit_direction_recovers_planted_readout():
    rng = np.random.default_rng(2)
    d = 8
    w_true = rng.standard_normal(d)
    stats = {}
    for g in range(4):
        X = rng.standard_normal((300, d))
        y = X @ w_true + 0.1 * rng.standard_normal(300)
        stats[g] = F.stat_from_arrays(X, y)
    fit = F.fit_direction(stats, "y")
    assert fit["r2_heldout"] > 0.95 and fit["r2_cv"] > 0.95
    assert np.corrcoef(fit["w"], w_true)[0, 1] > 0.99


def test_judge_scales_agrees_with_original_judge():
    rng = np.random.default_rng(3)
    n = 60
    base = rng.standard_normal(n) * 0.01
    effs = [{"level": base + 0.05 * k + 0.002 * rng.standard_normal(n)} for k in range(len(St.SCALES))]
    a = St.judge(effs, "level", "+", 1.0, seed=0)
    b = F.judge_scales(effs, St.SCALES, "level", "+", 1.0, seed=0)
    for k in ("rho", "size", "monotone", "big", "sig", "match"):
        assert a[k] == pytest.approx(b[k], abs=1e-9) or a[k] == b[k]
    assert abs(a["ci_lo"] - b["ci_lo"]) < 0.01 and abs(a["ci_hi"] - b["ci_hi"]) < 0.01
    flat = [{"level": base} for _ in range(4)]
    assert not F.judge_scales(flat, St.SCALES, "level", "+", 1.0)["match"]


def test_test_sets_equal_the_original_script():
    path = os.path.join(ROOT, "scripts", "neurons_steer.py")
    spec = importlib.util.spec_from_file_location("neurons_steer_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n = 400
    rng = np.random.default_rng(0)
    from otp.neurons import variables as V
    sv = np.full((n, len(V.SERIES_COLS)), np.nan, np.float32)
    sc = {k: i for i, k in enumerate(V.SERIES_COLS)}
    sv[:, sc["s_has_season"]] = rng.integers(0, 2, n)
    sv[:, sc["s_log2_period"]] = rng.uniform(2, 8, n)
    c = {"values": rng.standard_normal((n, 1152)).astype(np.float32), "split": (rng.random(n) < 0.3).astype(np.int8), "sv": sv}
    a1, a2 = mod.test_sets(c)
    b1, b2 = F.test_sets(c)
    assert np.array_equal(a1["ids"], b1["ids"]) and np.array_equal(a2["ids"], b2["ids"])
    assert np.array_equal(a2["period"], b2["period"])


@pytest.fixture(scope="module")
def ctx():
    rng = np.random.default_rng(0)
    t = np.arange(1024)
    return np.stack([np.sin(2 * np.pi * t / p) + 0.1 * rng.standard_normal(1024) + 0.002 * t * s
                     for p, s in ((24, 1), (64, -1), (40, 0), (16, 2))]).astype(np.float32)


def test_group_run_equals_original_run_conditions(ctx):
    conds = [[(2, 5, 1, 0.0), (2, 7, -1, 2.0), (4, 11, 1, 2.0)], [(1, 3, -1, 4.0)], []]
    ref = St.run_conditions("5M", ctx, conds, device="cpu")
    plan = F.poles_plan(pd.DataFrame({"layer": [2, 2, 4], "neuron": [5, 7, 11], "sign": ["pos", "neg", "pos"]}))
    # same poles, scaled by one factor each: run the three conditions of the reference one by one
    c1 = St.run_conditions("5M", ctx, [[(2, 5, 1, 2.0), (2, 7, -1, 2.0), (4, 11, 1, 2.0)]], device="cpu")
    g1 = F.run_group_conditions("5M", ctx, [(plan, 2.0)], device="cpu")
    assert np.allclose(c1, g1, atol=1e-4)
    plan2 = F.poles_plan(pd.DataFrame({"layer": [1], "neuron": [3], "sign": ["neg"]}))
    g2 = F.run_group_conditions("5M", ctx, [(plan2, 4.0), None], device="cpu")
    assert np.allclose(ref[1], g2[0], atol=1e-4) and np.allclose(ref[2], g2[1], atol=1e-4)


def test_poles_plan_merges_both_poles_of_a_neuron():
    plan = F.poles_plan(pd.DataFrame({"layer": [3, 3, 3], "neuron": [4, 4, 9], "sign": ["pos", "neg", "neg"]}))
    idx, sg = plan[3]
    assert list(idx) == [4, 9] and list(sg) == [0.0, -1.0]


def test_add_condition_changes_forecast_only_where_asked(ctx):
    d = 192
    rng = np.random.default_rng(0)
    v = rng.standard_normal(d).astype(np.float32)
    out = F.run_add_conditions("5M", ctx, [None, (1, "F", np.zeros(d, np.float32)), (1, "F", 3 * v), (1, "C1", 3 * v)], device="cpu")
    assert np.allclose(out[0], out[1], atol=1e-5)
    assert np.abs(out[2] - out[0]).max() > 1e-3
    assert np.abs(out[3] - out[0]).max() > 1e-4
    # added vector is seen by the next block: readout at the site moves by exactly the requested amount
    layers = load("5M", "cpu").model.transformer.layers
    seen = {}
    h = layers[2].register_forward_pre_hook(lambda m, a: seen.__setitem__("o", a[0].detach().clone()))
    try:
        w = rng.standard_normal(d)
        steer = F.steer_vector(w, 1.5).astype(np.float32)
        F.run_add_conditions("5M", ctx, [None, (1, "F", steer)], device="cpu")
    finally:
        h.remove()
    o = seen["o"].numpy()          # last call: rows of the steered condition
    B = ctx.shape[0]
    assert o.shape[0] == 2 * B
    delta = (o[B:] - o[:B]) @ w
    assert np.allclose(delta[:, F.L_CTX:], 1.5, atol=1e-3)
    assert np.allclose(delta[:, :F.L_CTX], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------------------------
# follow-up C helpers
# ---------------------------------------------------------------------------------------------
def test_shape_rules_name_planted_shapes():
    from otp.neurons import motifs as M
    t = np.arange(32)
    rng = np.random.default_rng(0)
    shapes = {"rising": t + 0.3 * rng.standard_normal(32), "falling": -t + 0.3 * rng.standard_normal(32),
              "peak": -(t - 16) ** 2 / 40 + 0.1 * rng.standard_normal(32), "trough": (t - 16) ** 2 / 40 + 0.1 * rng.standard_normal(32),
              "step up": (t > 15) * 5.0 + 0.3 * rng.standard_normal(32), "step down": -5.0 * (t > 10) + 0.3 * rng.standard_normal(32),
              "oscillation": np.sin(2 * np.pi * t / 6) + 0.05 * rng.standard_normal(32), "flat with noise": rng.standard_normal(32)}
    got = M.classify(M.shape_norm(np.stack(list(shapes.values()))))
    assert list(got) == list(shapes)


def test_shape_norm_constant_patch_is_zero_and_scale_free():
    from otp.neurons import motifs as M
    x = np.stack([np.ones(32) * 7, np.arange(32) * 3.0 + 100])
    z = M.shape_norm(x)
    assert np.all(z[0] == 0) and np.allclose(z[1], M.shape_norm(np.arange(32)[None])[0], atol=1e-5)


def test_majority_count_and_vocabulary():
    from otp.neurons import motifs as M
    ids = np.array([[0, 0, 0, 1, 1, 2, 2, 2], [3, 3, 3, 3, 3, 3, 1, 1]])
    cnt, cl = M.majority_count(ids, 4)
    assert list(cnt) == [3, 6] and list(cl) == [0, 3]
    rng = np.random.default_rng(0)
    a = M.shape_norm(np.arange(32) + 0.05 * rng.standard_normal((300, 32)))
    b = M.shape_norm(-np.arange(32) + 0.05 * rng.standard_normal((300, 32)))
    Z = np.concatenate([a, b])
    cen = M.fit_vocabulary(Z, 2, seed=0)
    lab = M.assign(Z, cen)
    assert len(set(lab[:300])) == 1 and len(set(lab[300:])) == 1 and lab[0] != lab[-1]


def test_token_ids_map_to_patches():
    import importlib.util
    path = os.path.join(ROOT, "scripts", "neurons_followup_c.py")
    spec = importlib.util.spec_from_file_location("neurons_followup_c_script", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    T = 256
    vals = np.arange(3 * (T + 128), dtype=np.float32).reshape(3, T + 128)
    tok = T // 32 + 4
    ids = np.array([[0, 1 * tok + 2, 2 * tok + tok - 1]])
    P, ser = m.patches(vals, ids, tok)
    assert list(ser[0]) == [0, 1, 2]
    assert np.array_equal(P[0, 0], vals[0, 0:32]) and np.array_equal(P[0, 1], vals[1, 64:96])
    assert np.array_equal(P[0, 2], vals[2, (tok - 1) * 32:tok * 32])       # last forecast token = last patch of the extended series
