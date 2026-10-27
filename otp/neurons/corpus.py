"""Audit corpus: series with known generating variables, for labeling Timestral's neurons.

A *spec* is a small dict (`sid`, `family`, `generator`, `config`, `params`, `seed`). `generate(spec, T)`
simulates `T + 128` steps (the context plus the 4 patches the forecast tokens stand for) and returns the
series with per-step channels (noise-free signal, residual, innovation scale, regime, phase, period) from
which `build_arrays` derives patch-level and series-level variables (see `variables.py`).

Recipe (`RECIPE`) is recorded in `results/neurons/corpus/recipe.json`. The simulators are the project's
generators (`otp.generators`); where a generator hides the quantity we need, a recording copy of its step
function is used here (same random draws, so the series is the generator's own).

Splits: `split_of(spec list)` gives `label` / `heldout` (20% of series per generator, seeded) and a fold
(0..2) among the label series; the held-out series are never used for labeling.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from otp.generators.base import Process
from otp.generators.chaos import Kuramoto, LogisticMap, Lorenz63, MackeyGlass, Rossler
from otp.generators.primitives import ARNoise, Season, Trend, in_prior
from otp.generators.regime import (FrequencyJump, MarkovSwitchingAR, MeanJump, VarianceJump, randomized_base,
                                   stationary_sd)
from otp.generators.tails import FractionalBM, Hawkes, MSMCascade, StableAR, fgn_autocov
from otp.neurons import variables as V

PATCH = 32
N_FC = 4                    # forecast tokens (horizon of 128 steps)
H_FC = N_FC * PATCH
CORPUS_SEED = 20261015
SPLIT_SEED = 7
PERIODS_PRIOR = (7, 12, 24, 48, 96, 168)
PERIODS_CTL = (8, 12, 16, 24, 32, 48, 64, 96, 128, 168, 256)
D_FRACS = (0.03, 0.1, 0.25, 0.5, 0.7)

RECIPE = {
    "version": 1,
    "context_steps_main": 1024,
    "forecast_steps": H_FC,
    "mix_main": {"in_prior": 700, "controlled": 300, "regime": 600, "chaos": 700, "tails": 700, "mixture": 100},
    "notes": ("in_prior: otp.generators.primitives.in_prior(seed) components stepped separately; controlled: single or "
              "paired components with gridded parameters; regime: mean/variance/frequency jump with the switch "
              "d = d_frac * T steps before the origin (context placement) and Markov-switching AR; chaos: configs "
              "of otp.grids.chaos.param_list; tails: stable AR, fBm, Hawkes, MSM grids of otp.grids.tails; "
              "mixture: random genomes of otp.generators.mixture."),
}


# ---------------------------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------------------------
def _chaos_configs():
    from otp.grids.chaos import param_list
    return param_list(full=True)


def _tails_configs():
    from otp.grids import tails as G
    out = []
    for rho in G.RHOS:
        for a in G.ALPHAS:
            out.append(("stable_ar", f"tails/stable_ar/alpha{a:g}_rho{rho:.1f}", {"alpha": a, "rho": rho}))
    for h in G.HURSTS:
        out.append(("fbm", f"tails/fbm/H{h:g}", {"H_hurst": h}))
    for e in G.ETAS:
        out.append(("hawkes", f"tails/hawkes/eta{e:g}", {"eta": e}))
    for tag, obs in G.OBS:
        for k in G.KS:
            out.append((f"msm_{'returns' if obs == 'returns' else 'cum'}", f"tails/msm/k{k}_{tag}",
                        {"k": k, "observe": obs}))
    return out


def make_specs(scale: float = 1.0, seed: int = CORPUS_SEED) -> list[dict]:
    """The main audit corpus (about 3100 series at scale 1). Deterministic."""
    rng = np.random.default_rng(seed)
    specs: list[dict] = []

    def add(family, generator, config, params):
        sid = len(specs)
        specs.append({"sid": sid, "family": family, "generator": generator, "config": config, "params": params,
                      "seed": int(rng.integers(2 ** 31 - 1))})

    n = lambda k: max(1, int(round(RECIPE["mix_main"][k] * scale)))   # noqa: E731
    for i in range(n("in_prior")):
        add("in_prior", "in_prior", "in_prior", {"period": int(rng.choice(PERIODS_PRIOR))})
    # controlled: season only / trend + noise / AR only / mixed grid
    nc = n("controlled")
    for i in range(nc):
        kind = ["ctl_season", "ctl_trend", "ctl_ar", "ctl_mix"][i % 4]
        if kind == "ctl_season":
            p = {"period": int(rng.choice(PERIODS_CTL)), "harmonics": int(rng.choice([1, 2, 3])),
                 "noise": float(rng.choice([0.02, 0.1, 0.3])), "rho": 0.0}
        elif kind == "ctl_trend":
            p = {"slope": float(rng.choice([-1, 1]) * rng.choice([0.0005, 0.002, 0.008])),
                 "noise": float(rng.choice([0.1, 0.3, 1.0])), "rho": float(rng.choice([0.0, 0.5, 0.9]))}
        elif kind == "ctl_ar":
            p = {"rho": float(rng.choice([0.0, 0.3, 0.6, 0.9, 0.98])), "noise": 1.0}
        else:
            p = {"period": int(rng.choice(PERIODS_CTL)), "harmonics": int(rng.choice([1, 2])),
                 "slope": float(rng.choice([-1, 1]) * rng.choice([0.0005, 0.002])),
                 "noise": float(rng.choice([0.1, 0.3, 1.0])), "rho": float(rng.choice([0.0, 0.5, 0.9])),
                 "season_w": float(rng.choice([0.5, 1.0, 2.0]))}
        add("controlled", kind, f"{kind}", p)
    # regime
    nr = n("regime")
    knobs = {"mean_jump": (0.5, 1, 2, 4, 8), "variance_jump": (0.25, 0.5, 2, 4, 8),
             "frequency_jump": (0.5, 0.75, 1.5, 2, 3)}
    for i in range(nr):
        g = ["mean_jump", "variance_jump", "frequency_jump", "markov_ar"][i % 4]
        base = {"period": int(rng.choice([12, 24, 48, 96, 168]))}
        if g == "markov_ar":
            base["p_switch"] = float(rng.choice([0.001, 0.003, 0.01, 0.03, 0.1]))
            base["mean1"] = float(rng.choice([1.0, 2.0, 4.0]))
            base["rho1"] = float(rng.choice([0.5, 0.9]))
            base["sigma1_mult"] = float(rng.choice([1.0, 2.0]))
            cfg = f"regime/markov_ar/p{base['p_switch']:g}"
        else:
            base["knob"] = float(rng.choice(knobs[g]))
            base["d_frac"] = float(rng.choice(D_FRACS))
            cfg = f"regime/{g}/{base['knob']:g}/dfrac{base['d_frac']:g}"
        add("regime", g, cfg, base)
    # chaos
    cfgs = _chaos_configs()
    for i in range(n("chaos")):
        cid, cls, params = cfgs[i % len(cfgs)]
        add("chaos", cls.name, cid, {"cls": cls.name, "params": params})
    # tails
    tc = _tails_configs()
    for i in range(n("tails")):
        g, cid, params = tc[i % len(tc)]
        add("tails", g, cid, params)
    # mixture
    for i in range(n("mixture")):
        add("mixture", "mixture", "mixture", {"genome_seed": int(rng.integers(2 ** 31 - 1))})
    return specs


def split_of(specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(split, fold) by position in `specs`: split 1 = held-out (20% of each generator), 0 = label; fold in 0..2."""
    rng = np.random.default_rng(SPLIT_SEED)
    n = len(specs)
    split = np.zeros(n, np.int8)
    fold = np.full(n, -1, np.int8)
    by: dict[str, list[int]] = {}
    for pos, s in enumerate(specs):
        by.setdefault(s["generator"], []).append(pos)
    for g in sorted(by):
        ids = np.array(by[g])
        rng.shuffle(ids)
        n_h = int(round(0.2 * len(ids)))
        split[ids[:n_h]] = 1
        for k, sid in enumerate(ids[n_h:]):
            fold[sid] = k % 3
    return split, fold


# ---------------------------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------------------------
def _nan(n):
    return np.full(n, np.nan)


def _components_series(rng, total, trend=None, season=None, season_w=1.0, ar=None):
    """Trend / season / AR noise stepped separately. Returns y, sig, res, inn, phase1, period, extras."""
    sig = np.zeros(total)
    res = np.zeros(total)
    inn = np.zeros(total)
    phase1 = None
    period = None
    extra = {}
    if trend is not None:
        pr = Trend(**trend)
        y, _ = pr.step(pr.init_state(rng, 1), total, rng)
        sig += y[0, 0]
    if season is not None:
        pr = Season(**season)
        st = pr.init_state(rng, 1)
        y, _ = pr.step(st, total, rng)
        sig += season_w * y[0, 0]
        phase1 = 2 * np.pi * np.arange(total) / season["period"] + pr._phases[0]
        period = np.full(total, float(season["period"]))
        extra["season_amp_sd"] = float(np.std(season_w * y[0, 0]))
    if ar is not None:
        pr = ARNoise(**ar)
        y, _ = pr.step(pr.init_state(rng, 1), total, rng)
        res += y[0, 0]
        inn += ar["sigma"]
    return sig + res, sig, res, inn, phase1, period, extra


def gen_in_prior(spec, T, H):
    pr = in_prior(spec["seed"], period=spec["params"]["period"])
    (_, tr), (w_s, se), (_, ar) = pr.components
    rng = np.random.default_rng(spec["seed"] + 1)
    total = T + H
    y, sig, res, inn, ph, per, ex = _components_series(
        rng, total, trend=dict(tr.params), season=dict(se.params), season_w=w_s, ar=dict(ar.params))
    sv = {"s_log2_period": np.log2(se.params["period"]), "s_has_season": 1.0, "s_ar_rho": ar.params["rho"]}
    return {"y": y, "sig": sig, "res": res, "inn": inn, "phase1": ph, "period": per, "sv": sv,
            "season_sd": ex.get("season_amp_sd")}


def gen_controlled(spec, T, H):
    p = spec["params"]
    kind = spec["generator"]
    rng = np.random.default_rng(spec["seed"] + 1)
    total = T + H
    trend = season = ar = None
    w_s = p.get("season_w", 1.0)
    if kind == "ctl_season":
        season = {"period": p["period"], "amplitude": 1.0, "harmonics": p["harmonics"],
                  "phase_seed": int(spec["seed"] % 1000003)}
        ar = {"rho": 0.0, "sigma": p["noise"]}
    elif kind == "ctl_trend":
        trend = {"level": 0.0, "slope": p["slope"], "phi": 1.0}
        ar = {"rho": p["rho"], "sigma": p["noise"]}
    elif kind == "ctl_ar":
        ar = {"rho": p["rho"], "sigma": p["noise"]}
    else:
        trend = {"level": 0.0, "slope": p["slope"], "phi": 1.0}
        season = {"period": p["period"], "amplitude": 1.0, "harmonics": p["harmonics"],
                  "phase_seed": int(spec["seed"] % 1000003)}
        ar = {"rho": p["rho"], "sigma": p["noise"]}
    y, sig, res, inn, ph, per, ex = _components_series(rng, total, trend, season, w_s, ar)
    sv = {"s_ar_rho": ar["rho"], "s_has_season": 1.0 if season else 0.0}
    if season:
        sv["s_log2_period"] = float(np.log2(season["period"]))
    return {"y": y, "sig": sig, "res": res, "inn": inn, "phase1": ph, "period": per, "sv": sv,
            "season_sd": ex.get("season_amp_sd")}


def _regime_step_rec(proc, state, n, rng):
    """Copy of `_RegimeBase.step` that also returns the pieces (same random draws)."""
    st, first = proc._begin(state, n, rng)
    P = st["t"].shape[0]
    R = proc._regimes(st, n, rng)
    if first:
        proc._start_x(st, R)
    sig, rho = proc._sigma_rho(R)
    e = rng.standard_normal((P, n)) * sig
    x = st["x"].copy()
    xs = np.empty((P, n))
    for i in range(n):
        x = rho[:, i] * x + e[:, i]
        xs[:, i] = x
    inc = proc._phase_inc(R)
    phase_path = st["phase"][:, None] + np.cumsum(inc, axis=1)
    lev = proc._levels(R)
    seas = proc._season(phase_path)
    y = lev + seas + xs
    new = {"t": st["t"] + n, "origin": st["origin"], "regime": R[:, -1].astype(np.int8), "x": x,
           "phase": phase_path[:, -1], "u0": st["u0"]}
    rec = {"sig": (lev + seas)[0], "res": xs[0], "inn": sig[0], "regime": R[0].astype(float),
           "phase1": phase_path[0] + proc._phases[0], "period": 2 * np.pi / inc[0], "seas": seas[0]}
    return y[:, None, :], new, rec


def _markov_step_rec(proc, state, n, rng):
    st, first = proc._begin(state, n, rng)
    P = st["t"].shape[0]
    if first:
        r0 = st["regime"].astype(int)
        st["x"] = st["u0"] * proc._sig[r0] / np.sqrt(1.0 - proc._rho[r0] ** 2)
    flips = rng.random((P, n)) < proc.params["p_switch"]
    z = rng.standard_normal((P, n))
    r = st["regime"].astype(int)
    x = st["x"].copy()
    xs = np.empty((P, n))
    rs = np.empty((P, n), np.int8)
    for i in range(n):
        if i > 0 or not first:
            r = np.where(flips[:, i], 1 - r, r)
        x = proc._rho[r] * x + proc._sig[r] * z[:, i]
        xs[:, i] = x
        rs[:, i] = r
    phase_path = st["phase"][:, None] + (np.arange(1, n + 1) * 2 * np.pi / proc.params["period"])[None, :]
    lev = proc._mean[rs]
    seas = proc._season(phase_path)
    y = lev + seas + xs
    new = {"t": st["t"] + n, "origin": st["origin"], "regime": r.astype(np.int8), "x": x,
           "phase": phase_path[:, -1], "u0": st["u0"]}
    rec = {"sig": (lev + seas)[0], "res": xs[0], "inn": proc._sig[rs][0], "regime": rs[0].astype(float),
           "phase1": phase_path[0] + proc._phases[0], "period": np.full(n, float(proc.params["period"])),
           "seas": seas[0]}
    return y[:, None, :], new, rec


def gen_regime(spec, T, H):
    p = spec["params"]
    g = spec["generator"]
    base = randomized_base(spec["seed"])
    rng = np.random.default_rng(spec["seed"] + 1)
    if g == "markov_ar":
        proc = MarkovSwitchingAR(**base, period=p["period"], p_switch=p["p_switch"], mean1=p["mean1"],
                                 rho1=p["rho1"], sigma1_mult=p["sigma1_mult"])
        stepf = _markov_step_rec
    else:
        cls = {"mean_jump": MeanJump, "variance_jump": VarianceJump, "frequency_jump": FrequencyJump}[g]
        knob = {"mean_jump": "delta", "variance_jump": "factor", "frequency_jump": "ratio"}[g]
        d = int(round(p["d_frac"] * T))
        proc = cls(**base, period=p["period"], placement="context", d=d, **{knob: p["knob"]})
        stepf = _regime_step_rec
    state = proc.init_state(rng, 1)
    y1, state, r1 = stepf(proc, state, T, rng)
    y2, state, r2 = stepf(proc, state, H, rng)
    y = np.concatenate([y1[0, 0], y2[0, 0]])
    rec = {k: np.concatenate([r1[k], r2[k]]) for k in r1}
    sv = {"s_log2_period": float(np.log2(p["period"])), "s_has_season": 1.0, "s_ar_rho": float(base["rho"])}
    if g == "markov_ar":
        sv["s_markov_p"] = p["p_switch"]
        sv["s_jump_mag"] = p["mean1"]
        sv["s_switch_pos"] = np.nan
    else:
        sv["s_switch_pos"] = float(1.0 - p["d_frac"])
        sv["s_jump_mag"] = {"mean_jump": p["knob"], "variance_jump": float(np.log2(p["knob"])),
                            "frequency_jump": float(np.log2(p["knob"]))}[g]
    seas = rec.pop("seas")
    return {"y": y, **rec, "sv": sv, "season_sd": float(np.std(seas[:T]))}


_LYAP = None


def lyap_obs_step(config_id: str, seed: int) -> float:
    """Lyapunov exponent per observed step from results/map/chaos/lyapunov.csv (Kuramoto: by freq seed)."""
    global _LYAP
    if _LYAP is None:
        import pandas as pd
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _LYAP = pd.read_csv(os.path.join(here, "results", "map", "chaos", "lyapunov.csv"))
    d = _LYAP[_LYAP.config_id == config_id]
    if "kuramoto" in config_id:
        d = d[d.seed == seed % 4]
    return float(d.lyap_per_obs_step.iloc[0]) if len(d) else np.nan


def gen_chaos(spec, T, H):
    cls = {"lorenz63": Lorenz63, "rossler": Rossler, "mackey_glass": MackeyGlass, "logistic_map": LogisticMap,
           "kuramoto": Kuramoto}[spec["params"]["cls"]]
    params = dict(spec["params"]["params"])
    if cls is Kuramoto:
        params["freq_seed"] = spec["seed"] % 4
    out = {}
    for tag, pr_params in (("obs", params), ("clean", {**params, "obs_rel": 0.0})):
        proc = cls(**pr_params)
        rng = np.random.default_rng(spec["seed"] + 1)
        st = proc.init_state(rng, 1)
        w = proc.warmup_steps()
        if w > 0:
            _, st = proc.step(st, w, rng)
        a, st = proc.step(st, T, rng)
        b, st = proc.step(st, H, rng)
        out[tag] = np.concatenate([a[0, 0], b[0, 0]])
    proc = cls(**params)
    dt_obs = proc.dt_obs
    sd_ref = cls.sd_ref
    noise = sd_ref * np.sqrt(proc.params["obs_rel"] ** 2 + proc.params["dyn_rel"] ** 2 * dt_obs)
    y, sig = out["obs"], out["clean"]
    sv = {"s_lyap": lyap_obs_step(spec["config"], spec["seed"])}
    return {"y": y, "sig": sig, "res": y - sig, "inn": np.full(T + H, noise), "sv": sv}


def _stable_ar(spec, T, H):
    p = spec["params"]
    proc = StableAR(**p)
    rng = np.random.default_rng(spec["seed"] + 1)
    st = proc.init_state(rng, 1)
    w = proc.warmup_steps()
    if w > 0:
        _, st = proc.step(st, w, rng)
    x0 = st["x"][0]
    a, st = proc.step(st, T, rng)
    b, st = proc.step(st, H, rng)
    y = np.concatenate([a[0, 0], b[0, 0]])
    e = y - p["rho"] * np.concatenate([[x0], y[:-1]])
    return {"y": y, "sig": None, "res": e, "inn": np.full(T + H, proc.params["sigma"]),
            "sv": {"s_alpha": p["alpha"], "s_ar_rho": p["rho"]}}


def _dl_innovation_var(gamma: np.ndarray) -> float:
    """One-step prediction error variance of a stationary process with autocovariance gamma[0..n] (Durbin-Levinson)."""
    v = float(gamma[0])
    phi = np.zeros(0)
    for k in range(1, len(gamma)):
        kappa = (gamma[k] - float(np.dot(phi, gamma[1:k][::-1]))) / v
        phi = np.concatenate([phi - kappa * phi[::-1], [kappa]])
        v *= 1.0 - kappa ** 2
    return v


def _fbm(spec, T, H):
    p = spec["params"]
    proc = FractionalBM(**p)
    rng = np.random.default_rng(spec["seed"] + 1)
    st = proc.init_state(rng, 1)
    a, st = proc.step(st, T, rng)
    b, st = proc.step(st, H, rng)
    y = np.concatenate([a[0, 0], b[0, 0]])
    inc = np.diff(np.concatenate([[0.0], y]))
    # one-step conditional sd of an fGn increment given the past (Durbin-Levinson on 256 lags)
    v = _dl_innovation_var(fgn_autocov(p["H_hurst"], 257, proc.params["sigma"]))
    return {"y": y, "sig": None, "res": inc, "inn": np.full(T + H, np.sqrt(max(v, 1e-12))),
            "sv": {"s_hurst": p["H_hurst"]}}


def _hawkes(spec, T, H):
    p = spec["params"]
    proc = Hawkes(**p)
    rng = np.random.default_rng(spec["seed"] + 1)
    st = proc.init_state(rng, 1)
    _, st = proc.step(st, proc.warmup_steps(), rng)
    s = st["s"].copy()
    d = np.exp(-1.0 / proc.params["tau"])
    mu = proc.params["rate"] * (1.0 - proc.params["eta"])
    total = T + H
    y = np.empty(total)
    lam = np.empty(total)
    for t in range(total):
        lam[t] = (mu + s)[0]
        c = rng.poisson(mu + s)
        y[t] = c[0]
        s = d * s + proc.params["eta"] * (1.0 - d) * c
    return {"y": y, "sig": lam, "res": y - lam, "inn": np.sqrt(lam), "sv": {"s_eta": p["eta"]}}


def _msm(spec, T, H):
    p = spec["params"]
    proc = MSMCascade(**p)
    rng = np.random.default_rng(spec["seed"] + 1)
    st = proc.init_state(rng, 1)
    gam = proc.renewal_probs()
    M, level = st["M"].copy(), st["level"].copy()
    total = T + H
    z = rng.standard_normal((1, total))
    r = np.empty(total)
    vol = np.empty(total)
    for t in range(total):
        renew = rng.random((1, M.shape[1])) < gam
        M = np.where(renew, proc._draw(rng, (1, M.shape[1])), M)
        vol[t] = proc.params["sigma"] * np.sqrt(M.prod(axis=1))[0]
        r[t] = vol[t] * z[0, t]
    y = r if p["observe"] == "returns" else np.cumsum(r)
    return {"y": y, "sig": np.zeros(total) if p["observe"] == "returns" else None, "res": r, "inn": vol,
            "sv": {"s_msm_k": float(p["k"])}}


def gen_tails(spec, T, H):
    g = spec["generator"]
    return {"stable_ar": _stable_ar, "fbm": _fbm, "hawkes": _hawkes, "msm_returns": _msm, "msm_cum": _msm}[g](
        spec, T, H)


def gen_mixture(spec, T, H):
    from otp.generators.mixture import decode, random_genome
    rng0 = np.random.default_rng(spec["params"]["genome_seed"])
    proc = decode(random_genome(rng0))
    rng = np.random.default_rng(spec["seed"] + 1)
    st = proc.init_state(rng, 1)
    a, st = proc.step(st, T, rng)
    b, st = proc.step(st, H, rng)
    y = np.concatenate([a[0, 0], b[0, 0]])
    return {"y": y, "sig": None, "res": None, "inn": None, "sv": {}}


GEN = {"in_prior": gen_in_prior, "controlled": gen_controlled, "regime": gen_regime, "chaos": gen_chaos,
       "tails": gen_tails, "mixture": gen_mixture}


def generate(spec: dict, T: int, H: int = H_FC) -> dict:
    out = GEN[spec["family"]](spec, T, H)
    out["y"] = np.asarray(out["y"], float)
    for k in ("sig", "res", "inn", "regime", "phase1", "period"):
        if out.get(k) is not None:
            out[k] = np.asarray(out[k], float)
    return out


# ---------------------------------------------------------------------------------------------
# arrays
# ---------------------------------------------------------------------------------------------
def series_arrays(spec: dict, gen: dict, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-series patch table (P_tot, len(PATCH_COLS)) and series vector (len(SERIES_COLS),)."""
    y = gen["y"]
    sd_obs = max(float(np.std(y[:T])), 1e-12)
    lat = V.latent_patch({k: gen.get(k) for k in ("y", "sig", "res", "inn", "regime", "phase1", "period")},
                         T, sd_obs)
    mot = V.motif_features(y[None], T)[0]
    sm = V.series_measured(y[:T])
    mp = V.patch_measured(y, sm, T)
    pv = np.concatenate([lat, mot, mp], axis=1).astype(np.float32)
    assert pv.shape[1] == len(V.PATCH_COLS)
    sv = np.full(len(V.SERIES_COLS), np.nan, np.float32)
    ix = {n: i for i, n in enumerate(V.SERIES_COLS)}
    for k, v in gen["sv"].items():
        sv[ix[k]] = v
    sig, res = gen.get("sig"), gen.get("res")
    if res is not None and np.isfinite(res[:T]).all():
        sv[ix["s_noise_level"]] = np.sqrt(np.mean(res[:T] ** 2)) / sd_obs
    if sig is not None and np.isfinite(sig[:T]).all():
        sv[ix["s_trend"]] = (np.polyfit(np.arange(T), sig[:T], 1)[0] * T) / sd_obs if spec["family"] in (
            "in_prior", "controlled") else np.nan
    if gen.get("season_sd") is not None:
        sv[ix["s_season_amp"]] = gen["season_sd"] / sd_obs
    for k in V.S_MEAS:
        sv[ix[k]] = sm[k]
    return pv, sv


def cat_vector(spec: dict) -> np.ndarray:
    v = np.zeros(len(V.CATS), np.float32)
    v[V.CATS.index("fam_" + spec["family"])] = 1.0
    v[V.CATS.index("gen_" + spec["generator"])] = 1.0
    return v


def _worker(args):
    k, spec, T = args
    gen = generate(spec, T)
    pv, sv = series_arrays(spec, gen, T)
    return k, gen["y"].astype(np.float32), pv, sv


def build_corpus(specs: list[dict], T: int, workers: int = 1, log=print) -> dict:
    """Generate every series at context length T. Returns arrays: values (N, T+128), pv, sv, cat, meta."""
    t0 = time.time()
    n = len(specs)
    P = T // PATCH + N_FC
    values = np.zeros((n, T + H_FC), np.float32)
    pv = np.zeros((n, P, len(V.PATCH_COLS)), np.float32)
    sv = np.zeros((n, len(V.SERIES_COLS)), np.float32)
    jobs = [(k, s, T) for k, s in enumerate(specs)]
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            it = pool.imap(_worker, jobs, chunksize=4)
            for k, (sid, y, p, s) in enumerate(it):
                values[sid], pv[sid], sv[sid] = y, p, s
                if k % 200 == 0:
                    log(f"  {k}/{n} series ({time.time() - t0:.0f}s)")
    else:
        for k, j in enumerate(jobs):
            sid, y, p, s = _worker(j)
            values[sid], pv[sid], sv[sid] = y, p, s
            if k % 200 == 0:
                log(f"  {k}/{n} series ({time.time() - t0:.0f}s)")
    cat = np.stack([cat_vector(s) for s in specs])
    return {"values": values, "pv": pv, "sv": sv, "cat": cat, "T": T, "sid": np.asarray([s["sid"] for s in specs])}


def save_corpus(path: str, c: dict, specs: list[dict]) -> None:
    arrays = {k: v for k, v in c.items() if k not in ("T", "cat")}
    arrays["T"] = np.asarray(c["T"])
    arrays["spec_json"] = np.asarray([json.JSONEncoder(default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode(s)
                                      for s in specs])
    np.savez_compressed(path, **arrays)


def load_corpus(path: str) -> tuple[dict, list[dict]]:
    with np.load(path, allow_pickle=False) as z:
        c = {k: z[k] for k in z.files if k != "spec_json"}
        specs = [json.loads(str(s)) for s in z["spec_json"]]
    c["T"] = int(c["T"])
    # indicators are derived from the specs (the stored copy may predate a change of the library)
    if all("family" in s for s in specs):
        c["cat"] = np.stack([cat_vector(s) for s in specs])
    else:
        c["cat"] = np.full((len(specs), len(V.CATS)), np.nan, np.float32)
    if "gid" not in c:
        c["gid"] = np.arange(len(specs))
    if "row" not in c:
        c["row"] = np.zeros(len(specs), np.int8)
    return c, specs


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def measured_arrays(values: np.ndarray, T: int) -> dict:
    """Corpus-format arrays for series with no latent variables (real data): motifs and measured analogs only.

    `values` `(N, T + 128)`. Latent columns are NaN; `cat` is NaN (unknown family).
    """
    n = values.shape[0]
    P = values.shape[1] // PATCH
    pv = np.full((n, P, len(V.PATCH_COLS)), np.nan, np.float32)
    sv = np.full((n, len(V.SERIES_COLS)), np.nan, np.float32)
    ix = {k: i for i, k in enumerate(V.SERIES_COLS)}
    mot = V.motif_features(values.astype(float), T)
    pv[:, :, len(V.LAT_PATCH):len(V.LAT_PATCH) + len(V.MOTIF)] = mot
    for i in range(n):
        sm = V.series_measured(values[i, :T])
        mp = V.patch_measured(values[i].astype(float), sm, T)
        pv[i, :, len(V.LAT_PATCH) + len(V.MOTIF):] = mp
        for k in V.S_MEAS:
            sv[i, ix[k]] = sm[k]
    return {"values": values.astype(np.float32), "pv": pv, "sv": sv, "cat": np.full((n, len(V.CATS)), np.nan, np.float32),
            "T": T, "sid": np.arange(n), "split": np.zeros(n, np.int8), "fold": np.zeros(n, np.int8)}


# ---------------------------------------------------------------------------------------------
# multivariate systems (rows of one system attend each other in the variate attention)
# ---------------------------------------------------------------------------------------------
MV_SID0 = 10_000


def make_mv_specs(n: int = 240, seed: int = CORPUS_SEED + 1) -> list[dict]:
    """Coupled two-row systems: lagged driver x -> y (lag, strength a) and linear feedback (coupling k)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        sid = MV_SID0 + i
        if i % 2 == 0:
            lag = int(rng.choice([0, 8, 32, 128]))
            a = float(rng.choice([0.0, 0.25, 0.5, 1.0]))
            out.append({"sid": sid, "family": "coupled", "generator": "lagged_driver",
                        "config": f"lagged_driver/lag{lag}/a{a:g}", "params": {"lag": lag, "a": a,
                        "period": int(rng.choice([12, 24, 48, 96])), "phase_seed": int(rng.integers(10 ** 9))},
                        "seed": int(rng.integers(2 ** 31 - 1))})
        else:
            k = float(rng.choice([0.0, 0.05, 0.15, 0.4, 1.0]))
            out.append({"sid": sid, "family": "coupled", "generator": "feedback_linear",
                        "config": f"feedback_linear/k{k:g}", "params": {"coupling": k},
                        "seed": int(rng.integers(2 ** 31 - 1))})
    return out


def generate_mv(spec: dict, T: int, H: int = H_FC) -> np.ndarray:
    """(2, T + H): row 0 = y, row 1 = x."""
    from otp.generators.coupled import FeedbackLinear, LaggedDriver
    p = spec["params"]
    if spec["generator"] == "lagged_driver":
        proc = LaggedDriver(lag=p["lag"], a=p["a"], period=p["period"], phase_seed=p["phase_seed"], rho_x=0.9)
    else:
        proc = FeedbackLinear(coupling=p["coupling"])
    rng = np.random.default_rng(spec["seed"] + 1)
    exo = np.random.default_rng(spec["seed"] + 2)
    st = proc.init_state(rng, 1)
    w = proc.warmup_steps()
    if w > 0:
        _, st = proc.step(st, w, rng, exo)
    a, st = proc.step(st, T, rng, exo)
    b, st = proc.step(st, H, rng, exo)
    return np.concatenate([a[0], b[0]], axis=1)


def build_mv(specs: list[dict], T: int, log=print) -> dict:
    """Corpus dict with two rows per system; rows of one system share `gid`, split and fold."""
    n = len(specs)
    P = T // PATCH + N_FC
    values = np.zeros((2 * n, T + H_FC), np.float32)
    pv = np.full((2 * n, P, len(V.PATCH_COLS)), np.nan, np.float32)
    sv = np.full((2 * n, len(V.SERIES_COLS)), np.nan, np.float32)
    six = {k: i for i, k in enumerate(V.SERIES_COLS)}
    for i, s in enumerate(specs):
        ys = generate_mv(s, T)
        base = measured_arrays(ys.astype(np.float32), T)
        for r in range(2):
            values[2 * i + r] = ys[r]
            pv[2 * i + r] = base["pv"][r]
            sv[2 * i + r] = base["sv"][r]
            if s["generator"] == "lagged_driver":
                sv[2 * i + r, six["s_coupling"]] = s["params"]["a"]
                sv[2 * i + r, six["s_lag"]] = s["params"]["lag"]
            else:
                sv[2 * i + r, six["s_coupling"]] = s["params"]["coupling"]
    rows = [dict(s) for s in specs for _ in range(2)]
    cat = np.stack([cat_vector(s) for s in rows])
    return {"values": values, "pv": pv, "sv": sv, "cat": cat, "T": T, "sid": np.repeat([s["sid"] for s in specs], 2),
            "gid": np.repeat(np.arange(n), 2), "row": np.tile(np.array([0, 1], np.int8), n)}, rows


def merge_corpora(cs: list[dict], spec_lists: list[list[dict]]) -> tuple[dict, list[dict]]:
    """Concatenate corpora along the series axis (same T). Group ids are offset so groups stay distinct."""
    out = {"T": cs[0]["T"]}
    gid_off = 0
    gids = []
    for c in cs:
        g = c.get("gid", np.arange(len(c["values"])))
        gids.append(g + gid_off)
        gid_off += int(g.max()) + 1
    out["gid"] = np.concatenate(gids)
    for k in ("values", "pv", "sv", "cat", "sid", "split", "fold"):
        out[k] = np.concatenate([c[k] for c in cs])
    out["row"] = np.concatenate([c.get("row", np.zeros(len(c["values"]), np.int8)) for c in cs])
    return out, [s for sl in spec_lists for s in sl]
