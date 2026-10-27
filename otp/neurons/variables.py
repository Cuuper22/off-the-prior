"""The variable library: every quantity a pole can be labeled with.

Groups (names are the column names used everywhere):

- `lat_*`   patch-level latent variables computed exactly from the simulator or its noise-free signal
            (local level, slope, curvature, noise sd, innovation sd, oracle one-step predictability,
            seasonal phase, regime indicator, steps since the last switch, period in force, outlier).
- `prev_*`  the same for the previous patch (for a forecast token: the patch just before the one it forecasts).
- `m_*`     input motifs of a patch, computed from the raw values alone (also computable on real series).
- `x_*`     measured analogs of latent variables (dominant period, change point, ...), computed from the
            raw values alone with the same code on synthetic and real series.
- `s_*`     series-level latent variables (period, season amplitude, trend, noise level, AR coefficient,
            Lyapunov exponent per observed step, stable alpha, Hurst exponent, branching ratio, ...).
- `pos_*`, `tok_*`  position of the token.
- `fam_*`, `gen_*`  indicator of the family / generator the series came from.

A value of NaN means "not defined for this series" (for example `s_alpha` on a seasonal series).
`LIBRARY` lists all names in a fixed order; `ANALOG` maps a latent variable to the measured variable
that stands in for it on real series.
"""

from __future__ import annotations

import numpy as np

PATCH = 32
EPS = 1e-9

LAT_PATCH = ["lat_level", "lat_slope", "lat_curv", "lat_noise_sd", "lat_inn_sd", "lat_pred1", "lat_phase_sin",
             "lat_phase_cos", "lat_regime", "lat_regime_frac", "lat_since_switch", "lat_log2_period",
             "lat_outlier", "lat_outlier_mag"]
PREV_SRC = ["lat_level", "lat_slope", "lat_noise_sd", "lat_phase_sin", "lat_phase_cos", "lat_regime"]
PREV = ["prev_" + n[4:] for n in PREV_SRC]
MOTIF = ["m_level_ctx", "m_slope", "m_curv", "m_logsd", "m_resid_sd", "m_jump", "m_peak_pos", "m_trough_pos",
         "m_zero_cross", "m_skew", "m_kurt", "m_monotone", "m_dlevel_prev", "m_dsd_prev", "m_range", "m_acf1",
         "m_max_z", "m_roughness"]
MEAS_PATCH = ["x_post_cp", "x_since_cp", "x_phase_sin", "x_phase_cos"]
S_LAT = ["s_log2_period", "s_has_season", "s_season_amp", "s_trend", "s_noise_level", "s_ar_rho", "s_lyap",
         "s_alpha", "s_hurst", "s_eta", "s_msm_k", "s_switch_pos", "s_jump_mag", "s_markov_p", "s_coupling",
         "s_lag"]
S_MEAS = ["x_log2_period", "x_season_strength", "x_trend", "x_noise_level", "x_ar1", "x_hurst", "x_tail",
          "x_cp_pos", "x_cp_strength"]
POSITION = ["pos_logj", "pos_from_end", "tok_fc", "tok_fc_step", "tok_cov_row"]
FAMILIES = ["in_prior", "controlled", "regime", "chaos", "tails", "mixture", "coupled"]
GENERATORS = ["in_prior", "ctl_season", "ctl_trend", "ctl_ar", "ctl_mix", "mean_jump", "variance_jump",
              "frequency_jump", "markov_ar", "lorenz63", "rossler", "mackey_glass", "logistic_map", "kuramoto",
              "stable_ar", "fbm", "hawkes", "msm_returns", "msm_cum", "mixture", "lagged_driver", "feedback_linear"]
CATS = ["fam_" + f for f in FAMILIES] + ["gen_" + g for g in GENERATORS]

PATCH_COLS = LAT_PATCH + MOTIF + MEAS_PATCH          # stored per patch
SERIES_COLS = S_LAT + S_MEAS                         # stored per series
LIBRARY = LAT_PATCH + PREV + MOTIF + MEAS_PATCH + S_LAT + S_MEAS + POSITION + CATS
VAR_INDEX = {n: i for i, n in enumerate(LIBRARY)}
IS_CAT = np.array([n in CATS or n in ("tok_fc", "tok_cov_row") for n in LIBRARY])

ANALOG = {
    "lat_level": "m_level_ctx", "lat_slope": "m_slope", "lat_curv": "m_curv", "lat_noise_sd": "m_resid_sd",
    "lat_regime": "x_post_cp", "lat_regime_frac": "x_post_cp", "lat_since_switch": "x_since_cp",
    "lat_phase_sin": "x_phase_sin", "lat_phase_cos": "x_phase_cos", "lat_log2_period": "x_log2_period",
    "lat_outlier_mag": "m_max_z", "lat_outlier": "m_max_z", "lat_inn_sd": "m_resid_sd",
    "s_log2_period": "x_log2_period", "s_has_season": "x_season_strength", "s_season_amp": "x_season_strength",
    "s_trend": "x_trend", "s_noise_level": "x_noise_level", "s_ar_rho": "x_ar1", "s_hurst": "x_hurst",
    "s_alpha": "x_tail", "s_switch_pos": "x_cp_pos", "s_jump_mag": "x_cp_strength",
}
# variables that are already measured from the raw values (usable on real series as they are)
MEASURABLE = set(MOTIF) | set(MEAS_PATCH) | set(S_MEAS) | set(POSITION)

# core subset for pair labels (24 variables with distinct meaning)
CORE = ["lat_level", "lat_slope", "lat_curv", "lat_noise_sd", "lat_pred1", "lat_phase_sin", "lat_phase_cos",
        "lat_regime", "lat_since_switch", "lat_log2_period", "lat_outlier_mag", "m_level_ctx", "m_slope",
        "m_jump", "m_peak_pos", "s_log2_period", "s_season_amp", "s_noise_level", "s_lyap", "s_alpha",
        "s_hurst", "x_log2_period", "pos_from_end", "tok_fc"]


# ---------------------------------------------------------------------------------------------
# patch-level helpers
# ---------------------------------------------------------------------------------------------
_T = np.arange(PATCH, dtype=float)
_TC = (_T - _T.mean()) / (PATCH / 2)
_V2 = np.stack([np.ones(PATCH), _TC, _TC ** 2 - (_TC ** 2).mean()], axis=1)       # orthogonal basis 1, t, t^2
_PROJ2 = np.linalg.pinv(_V2)                                                        # (3, 32)


def _quad_fit(x: np.ndarray):
    """x (..., 32) -> coefficients on the orthogonal basis (..., 3) and residual rms (...)."""
    c = x @ _PROJ2.T
    fit = c @ _V2.T
    return c, np.sqrt(np.nanmean((x - fit) ** 2, axis=-1))


def _slope_per_patch(x: np.ndarray) -> np.ndarray:
    """OLS slope across a patch, expressed as total change over the patch (per 32 steps)."""
    xc = x - np.nanmean(x, axis=-1, keepdims=True)
    return (xc * (_T - _T.mean())).sum(-1) / ((_T - _T.mean()) ** 2).sum() * PATCH


def _curv_per_patch(x: np.ndarray) -> np.ndarray:
    c, _ = _quad_fit(x)
    # coefficient of (t_c^2 - mean) with t_c in [-1, 1]: change of slope scale over the patch
    return c[..., 2]


def motif_features(vals: np.ndarray, n_ctx_steps: int) -> np.ndarray:
    """Input motifs per patch. vals (N, T_tot) -> (N, P, len(MOTIF)); normalized by context mean and sd."""
    n, ttot = vals.shape
    p = ttot // PATCH
    ctx = vals[:, :n_ctx_steps]
    c = ctx.mean(1)[:, None, None]
    s = np.maximum(ctx.std(1), 1e-12)[:, None, None]
    x = vals.reshape(n, p, PATCH)
    xs = (x - c) / s
    out = np.full((n, p, len(MOTIF)), np.nan, np.float32)
    mean = xs.mean(-1)
    sd = xs.std(-1)
    coefs, rres = _quad_fit(xs)
    out[..., 0] = mean
    out[..., 1] = _slope_per_patch(xs)
    out[..., 2] = coefs[..., 2]
    out[..., 3] = np.log10(sd + 1e-4)
    out[..., 4] = np.log10(rres + 1e-4)
    d = np.diff(xs, axis=-1)
    mad = np.median(np.abs(d - np.median(d, -1, keepdims=True)), -1) * 1.4826
    out[..., 5] = np.log1p(np.abs(d).max(-1) / (mad + 1e-6))
    out[..., 6] = xs.argmax(-1) / (PATCH - 1)
    out[..., 7] = xs.argmin(-1) / (PATCH - 1)
    xc = xs - mean[..., None]
    sg = np.sign(xc)
    out[..., 8] = (sg[..., 1:] * sg[..., :-1] < 0).sum(-1) / (PATCH - 1)
    sd_safe = np.maximum(sd, 1e-9)
    z = xc / sd_safe[..., None]
    out[..., 9] = (z ** 3).mean(-1)
    out[..., 10] = (z ** 4).mean(-1) - 3.0
    up = (d > 0).mean(-1)
    out[..., 11] = np.abs(2 * up - 1)
    out[..., 12] = np.concatenate([np.zeros((n, 1)), np.diff(mean, axis=1)], axis=1)
    lsd = np.log10(sd + 1e-4)
    out[..., 13] = np.concatenate([np.zeros((n, 1)), np.diff(lsd, axis=1)], axis=1)
    out[..., 14] = xs.max(-1) - xs.min(-1)
    out[..., 15] = (xc[..., 1:] * xc[..., :-1]).mean(-1) / (xc ** 2).mean(-1).clip(1e-12)
    out[..., 16] = np.abs(xs).max(-1)
    out[..., 17] = np.log10(d.std(-1) + 1e-4)
    out[:, 0, 12] = np.nan
    out[:, 0, 13] = np.nan
    return out


# ---------------------------------------------------------------------------------------------
# measured analogs
# ---------------------------------------------------------------------------------------------
def _acf(x: np.ndarray, nmax: int) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / max(ac[0], 1e-12)


def dominant_period(x: np.ndarray, min_p: int = 4, max_p: int = 1024) -> tuple[int, float]:
    """Lag of the highest autocorrelation peak in [min_p, max_p] after the first zero crossing, and its height.

    Same rule as `otp.mechanism.dominant_period` (last 4096 steps), returning the height as well.
    """
    x = np.asarray(x[-4096:], float)
    n = len(x)
    ac = _acf(x, n)
    hi = min(max_p, n // 3)
    if hi <= min_p + 1:
        return 1, 0.0
    neg = np.where(ac[1:] < 0)[0]
    zc = int(neg[0] + 1) if len(neg) else min_p
    lo = max(min_p, zc)
    if lo >= hi:
        lo = min_p
    k = int(lo + np.argmax(ac[lo:hi]))
    return k, float(ac[k])


def series_measured(ctx: np.ndarray) -> dict:
    """Measured series-level analogs from the context values alone."""
    x = np.asarray(ctx, float)
    n = len(x)
    sd = max(x.std(), 1e-12)
    t = np.arange(n)
    tc = t - t.mean()
    slope = (tc * (x - x.mean())).sum() / (tc ** 2).sum()
    p, strength = dominant_period(x - slope * tc)     # linear trend removed before the autocorrelation
    trend = slope * n / sd
    r = x - x.mean() - slope * tc
    if strength > 0.3 and p >= 2 and n >= 2 * p:
        k = (n // p) * p
        prof = r[n - k:].reshape(-1, p).mean(0)
        # sample index i uses profile position (i - (n - k)) mod p
        idx = (np.arange(n) - (n - k)) % p
        r = r - prof[idx]
    noise_level = r.std() / sd
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if r.std() > 1e-12 else 0.0
    # aggregated-increment scaling exponent
    d = np.diff(x)
    ms = [1, 2, 4, 8, 16, 32, 64]
    v = []
    for m in ms:
        k = (len(d) // m) * m
        if k < 8 * m:
            v.append(np.nan)
            continue
        agg = d[:k].reshape(-1, m).sum(1)
        v.append(agg.var())
    v = np.asarray(v)
    ok = np.isfinite(v) & (v > 0)
    hurst = np.polyfit(np.log(np.asarray(ms)[ok]), np.log(v[ok]), 1)[0] / 2 if ok.sum() >= 3 else np.nan
    # Hill tail index on |increments|
    a = np.sort(np.abs(d - np.median(d)))[::-1]
    kk = max(10, int(0.05 * len(a)))
    kk = min(kk, len(a) - 1)
    hill = np.log(a[:kk] / max(a[kk], 1e-12)).mean() if a[kk] > 0 else np.nan
    tail = 1.0 / hill if hill and hill > 0 else np.nan
    tail = float(np.clip(tail, 0.3, 8.0)) if np.isfinite(tail) else np.nan
    # change point on block means of the seasonally-flattened residual plus trend-free level
    nb = n // PATCH
    pm = x[:nb * PATCH].reshape(nb, PATCH).mean(1)
    best, kbest = 0.0, nb // 2
    csum = np.cumsum(pm)
    tot = csum[-1]
    for kidx in range(2, nb - 1):
        m1 = csum[kidx - 1] / kidx
        m2 = (tot - csum[kidx - 1]) / (nb - kidx)
        stat = abs(m2 - m1) / sd
        if stat > best:
            best, kbest = stat, kidx
    return {"x_log2_period": float(np.log2(max(p, 1))), "x_season_strength": float(strength), "x_trend": float(trend),
            "x_noise_level": float(noise_level), "x_ar1": ac1, "x_hurst": float(hurst), "x_tail": tail,
            "x_cp_pos": kbest / nb, "x_cp_strength": float(best), "_period": int(p), "_kbest": int(kbest),
            "_nb": int(nb)}


def patch_measured(vals: np.ndarray, sm: dict, n_ctx_steps: int) -> np.ndarray:
    """Per-patch measured analogs (x_post_cp, x_since_cp, x_phase_sin, x_phase_cos) for one series."""
    ttot = len(vals)
    p_tot = ttot // PATCH
    out = np.zeros((p_tot, 4), np.float32)
    kb, strength = sm["_kbest"], sm["x_cp_strength"]
    j = np.arange(p_tot)
    post = (j >= kb) & (strength > 0.5)
    out[:, 0] = post
    out[:, 1] = np.where(post, np.log1p(np.maximum(j - kb, 0) * PATCH), 0.0)
    per = sm["_period"]
    if sm["x_season_strength"] > 0.3 and per >= 2:
        ctx = vals[:n_ctx_steps].astype(float)
        t = np.arange(len(ctx))
        w = 2 * np.pi / per
        a = np.stack([np.sin(w * t), np.cos(w * t), np.ones_like(t, float)], 1)
        coef = np.linalg.lstsq(a, ctx, rcond=None)[0]
        phi = np.arctan2(coef[1], coef[0])                  # y ~ A sin(w t + phi)
        t0 = j * PATCH
        ph = w * t0 + phi
        out[:, 2] = np.sin(ph)
        out[:, 3] = np.cos(ph)
    else:
        out[:, 2:] = np.nan
    return out


# ---------------------------------------------------------------------------------------------
# latent patch variables from per-step channels
# ---------------------------------------------------------------------------------------------
def latent_patch(ch: dict, n_ctx_steps: int, sd_obs: float) -> np.ndarray:
    """ch: per-step channels (len T_tot): y, sig, res, inn, regime, phase1, period. NaN = undefined.

    Returns (P, len(LAT_PATCH)).
    """
    y = ch["y"]
    ttot = len(y)
    p = ttot // PATCH
    out = np.full((p, len(LAT_PATCH)), np.nan, np.float32)

    def patches(a):
        return None if a is None else np.asarray(a, float).reshape(p, PATCH)

    sig, res, inn = patches(ch.get("sig")), patches(ch.get("res")), patches(ch.get("inn"))
    reg, ph, per = patches(ch.get("regime")), patches(ch.get("phase1")), patches(ch.get("period"))
    sd = max(sd_obs, 1e-12)
    if sig is not None and np.isfinite(sig).all():
        sm = np.nanmean(np.asarray(ch["sig"], float)[:n_ctx_steps])
        out[:, 0] = (sig.mean(-1) - sm) / sd
        out[:, 1] = _slope_per_patch(sig) / sd
        out[:, 2] = _curv_per_patch(sig) / sd
    if res is not None and np.isfinite(res).all():
        out[:, 3] = np.sqrt((res ** 2).mean(-1)) / sd
        rr = np.asarray(ch["res"], float)[:n_ctx_steps]
        rob = max(1.4826 * np.median(np.abs(rr - np.median(rr))), 1e-12)
        mx = np.abs(res).max(-1) / rob
        out[:, 12] = (mx > 4.0).astype(np.float32)
        out[:, 13] = np.log1p(mx)
    if inn is not None and np.isfinite(inn).all():
        s1 = inn.mean(-1)
        out[:, 4] = s1 / sd
        out[:, 5] = np.clip(1.0 - (s1 / sd) ** 2, 0.0, 1.0)
    if ph is not None and np.isfinite(ph).all():
        out[:, 6] = np.sin(ph[:, 0])
        out[:, 7] = np.cos(ph[:, 0])
    if reg is not None and np.isfinite(reg).all():
        out[:, 8] = reg[:, -1]
        out[:, 9] = reg.mean(-1)
        r = np.asarray(ch["regime"], float)
        flips = np.where(np.diff(r) != 0)[0] + 1
        mark = np.zeros(ttot, bool)
        mark[flips] = True
        idx = np.where(mark, np.arange(ttot), -1)
        idx = np.maximum.accumulate(idx)
        since = np.where(idx >= 0, np.arange(ttot) - idx, 0.0)
        end = np.arange(PATCH - 1, ttot, PATCH)
        out[:, 10] = np.log1p(since[end])
    if per is not None and np.isfinite(per).all():
        out[:, 11] = np.log2(np.maximum(per[:, 0], 1.0))
    return out
