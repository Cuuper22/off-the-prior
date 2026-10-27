"""Motif labels without a variable list (follow-up C).

The top and mid-range exemplar tokens of a pole are patches of 32 raw values. Each patch is normalized, the patches are
clustered into a fixed vocabulary of shapes (k-means), clusters are named by simple shape rules, and a pole is
*motif-cohesive* when most of its exemplars fall in one cluster. Definitions are fixed in
`results/neurons/steering_preregistration_followup.md` (section C).
"""

from __future__ import annotations

import numpy as np

PATCH = 32
SD_FLOOR = 1e-9
NAMES = ("step up", "step down", "oscillation", "rising", "falling", "peak", "trough", "flat with noise")

_T = np.arange(PATCH, dtype=float)
_TC = _T - _T.mean()
_QUAD = np.stack([np.ones(PATCH), _TC, _TC ** 2 - (_TC ** 2).mean()], axis=1)
_QPROJ = np.linalg.pinv(_QUAD)


def shape_norm(P: np.ndarray) -> np.ndarray:
    """Subtract the patch mean, divide by the patch sd; a patch with sd below 1e-9 becomes the zero vector."""
    P = np.asarray(P, np.float64)
    m = P.mean(-1, keepdims=True)
    s = P.std(-1, keepdims=True)
    Z = (P - m) / np.maximum(s, SD_FLOOR)
    return np.where(s < SD_FLOOR, 0.0, Z).astype(np.float32)


def scaled_norm(P: np.ndarray, ctx_mean: np.ndarray, ctx_sd: np.ndarray) -> np.ndarray:
    """Subtract the context mean and divide by the context sd of the series (keeps level and amplitude)."""
    return ((np.asarray(P, np.float64) - ctx_mean[..., None]) / np.maximum(ctx_sd, SD_FLOOR)[..., None]).astype(np.float32)


def classify(Z: np.ndarray) -> np.ndarray:
    """Name each shape-normalized patch (n, 32) by the first matching rule (names in NAMES)."""
    Z = np.asarray(Z, np.float64)
    n = len(Z)
    nrm = (Z ** 2).sum(1)
    safe = np.maximum(nrm, 1e-12)
    # linear fit
    slope = (Z * _TC).sum(1) / (_TC ** 2).sum()
    lin = slope[:, None] * _TC
    r2_lin = 1 - ((Z - lin) ** 2).sum(1) / safe
    # quadratic fit and the vertex of the fitted parabola
    coef = Z @ _QPROJ.T
    fitq = coef @ _QUAD.T
    r2_quad = 1 - ((Z - fitq) ** 2).sum(1) / safe
    a2 = coef[:, 2]
    b1 = coef[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = np.where(np.abs(a2) > 1e-12, -b1 / (2 * a2) + _T.mean(), np.nan)
    inside = (vertex >= 6) & (vertex <= 26)
    # best two-level fit with the break between steps b-1 and b, b in 6..26
    cs = np.cumsum(Z, axis=1)
    tot = cs[:, -1]
    best_sse = np.full(n, np.inf)
    best_diff = np.zeros(n)
    for b in range(6, 27):
        ml = cs[:, b - 1] / b
        mr = (tot - cs[:, b - 1]) / (PATCH - b)
        sse = nrm - b * ml ** 2 - (PATCH - b) * mr ** 2           # Z has mean zero overall, so SSE of the two-level fit
        sse = ((Z[:, :b] - ml[:, None]) ** 2).sum(1) + ((Z[:, b:] - mr[:, None]) ** 2).sum(1)
        better = sse < best_sse
        best_sse = np.where(better, sse, best_sse)
        best_diff = np.where(better, mr - ml, best_diff)
    step = (best_sse <= 0.2 * nrm) & (np.abs(best_diff) >= 1.0) & (nrm > 1e-9)
    # oscillation: sign changes and smoothness of the linearly detrended patch
    Zd = Z - lin
    sg = np.sign(Zd)
    changes = (sg[:, 1:] * sg[:, :-1] < 0).sum(1)
    ac1 = (Zd[:, 1:] * Zd[:, :-1]).sum(1) / np.maximum((Zd ** 2).sum(1), 1e-12)
    osc = (changes >= 4) & (ac1 >= 0.3)
    out = np.full(n, "flat with noise", dtype=object)
    peak = (r2_quad >= 0.5) & inside & (a2 < 0)
    trough = (r2_quad >= 0.5) & inside & (a2 > 0)
    out = np.where(trough, "trough", out)
    out = np.where(peak, "peak", out)
    out = np.where((r2_lin >= 0.6) & (slope < 0), "falling", out)
    out = np.where((r2_lin >= 0.6) & (slope > 0), "rising", out)
    out = np.where(osc, "oscillation", out)
    out = np.where(step & (best_diff > 0), "step up", out)
    out = np.where(step & (best_diff < 0), "step down", out)
    return out.astype(str)


def fit_vocabulary(Z: np.ndarray, K: int, seed: int = 0, n_fit: int = 500_000) -> np.ndarray:
    """k-means centers (K, 32) on (a random subset of) the shape-normalized patches Z."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    if len(Z) > n_fit:
        Z = Z[rng.choice(len(Z), n_fit, replace=False)]
    km = KMeans(n_clusters=K, n_init=5, random_state=seed)
    km.fit(np.asarray(Z, np.float32))
    return km.cluster_centers_.astype(np.float32)


def assign(Z: np.ndarray, centers: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Nearest-center index of each row of Z."""
    c2 = (centers ** 2).sum(1)
    out = np.empty(len(Z), np.int16)
    for s in range(0, len(Z), chunk):
        z = np.asarray(Z[s:s + chunk], np.float32)
        d = c2[None] - 2 * z @ centers.T
        out[s:s + chunk] = d.argmin(1)
    return out


def name_clusters(Z: np.ndarray, ids: np.ndarray, K: int, rng_seed: int = 0, max_per: int = 20_000) -> list[dict]:
    """Name each cluster by the plurality class of its member patches (up to `max_per` sampled)."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for k in range(K):
        ix = np.where(ids == k)[0]
        if len(ix) == 0:
            rows.append({"cluster": k, "name": "empty", "purity": np.nan, "n": 0})
            continue
        if len(ix) > max_per:
            ix = rng.choice(ix, max_per, replace=False)
        cl = classify(Z[ix])
        vals, cnt = np.unique(cl, return_counts=True)
        j = cnt.argmax()
        rows.append({"cluster": k, "name": str(vals[j]), "purity": float(cnt[j] / cnt.sum()), "n": int((ids == k).sum()),
                     **{f"share_{v}": float(c / cnt.sum()) for v, c in zip(vals, cnt)}})
    return rows


def majority_count(ids: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """ids (n_poles, k_exemplars) -> (largest cluster count per pole, cluster of that count)."""
    n = ids.shape[0]
    cnt = np.zeros((n, K), np.int16)
    for j in range(ids.shape[1]):
        np.add.at(cnt, (np.arange(n), ids[:, j]), 1)
    return cnt.max(1), cnt.argmax(1)
