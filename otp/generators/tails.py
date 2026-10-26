"""Heavy-tail and fractal family ("tails"): four batched simulators with a difficulty knob each.

Pretraining data for Timestral is trend + seasonality + light-tailed noise. These processes
leave that region in four different ways:

1. `StableAR`      AR(1) / random walk driven by symmetric alpha-stable innovations
                   (knob `alpha`; alpha = 2 is Gaussian, smaller alpha means heavier tails:
                   the variance is infinite for alpha < 2 and the mean for alpha <= 1).
2. `FractionalBM`  fractional Brownian motion (knob Hurst exponent `H_hurst`). H < 0.5 means
                   increments are negatively correlated (anti-persistent), H > 0.5 positively
                   correlated (long memory), H = 0.5 is ordinary Brownian motion.
3. `Hawkes`        self-exciting point process observed as counts per time bin (knob
                   branching ratio `eta`, the expected number of events directly caused by one
                   event; the mean rate is held fixed, so only the clustering changes).
4. `MSMCascade`    Markov-switching multifractal volatility (Calvet-Fisher): volatility is a
                   product of `k` multipliers renewed at geometrically spaced frequencies
                   (knob `k`).

All four are vectorized over the leading path axis, as the generator contract in
`otp.generators.base` requires, and each oracle continues the exact state:

- StableAR: the state is the last value (the process is Markov).
- FractionalBM: not Markov. The state is the last `max_hist` increments (fractional Gaussian
  noise) plus the current level; the context itself is generated exactly (Davies-Harte
  circulant embedding) and future increments are drawn from the exact Gaussian conditional
  given those `max_hist` increments (default 2048, at least 1024). Conditioning on only the
  last `max_hist` increments instead of the whole history is the one approximation; paths
  0..n of a sample all use the same conditioning, so the realized future is a draw from the
  oracle's own distribution.
- Hawkes: discrete-time approximation (see the class). The excitation level is a Markov state.
- MSMCascade: the state is the vector of current multipliers (and the level for the
  cumulative observation); the oracle is Monte Carlo from that state.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular, toeplitz
from scipy.signal import lfilter

from otp.generators.base import Process

FAMILY = "tails"


def _fmt(x: float) -> str:
    return f"{x:g}"


# ----------------------------------------------------------------------------------------
# 1. alpha-stable AR(1) / random walk
# ----------------------------------------------------------------------------------------

def stable_noise(rng: np.random.Generator, alpha: float, shape, scale: float = 1.0) -> np.ndarray:
    """Symmetric alpha-stable samples S_alpha(scale, beta=0, mu=0) by the Chambers-Mallows-Stuck formula.

    With U uniform on (-pi/2, pi/2) and W standard exponential,
        X = sin(alpha U) / cos(U)^(1/alpha) * (cos((1 - alpha) U) / W)^((1 - alpha) / alpha).
    In this parameterization the characteristic function is exp(-scale^alpha |t|^alpha), so
    alpha = 2 gives a Gaussian with standard deviation scale * sqrt(2). Valid for alpha != 1;
    alpha = 1 (Cauchy) is handled separately. Vectorized NumPy, no scipy sampler.
    """
    assert 0 < alpha <= 2
    u = rng.uniform(-np.pi / 2, np.pi / 2, shape)
    if alpha == 1.0:
        return scale * np.tan(u)
    w = rng.exponential(1.0, shape)
    x = np.sin(alpha * u) / np.cos(u) ** (1.0 / alpha) * (np.cos((1.0 - alpha) * u) / w) ** ((1.0 - alpha) / alpha)
    return scale * x


class StableAR(Process):
    """x_t = rho x_{t-1} + sigma * e_t with e_t symmetric alpha-stable (scale 1). rho = 1 is a random walk.

    Observed series: x_t. Knob: `alpha` (2.0, 1.9, 1.7, 1.5, 1.3, 1.1 in the grid).
    """

    family, name = FAMILY, "stable_ar"
    defaults = {"alpha": 1.7, "rho": 0.9, "sigma": 1.0}
    knob = "alpha"

    def warmup_steps(self) -> int:
        return 0 if self.params["rho"] >= 1.0 else 500

    def init_state(self, rng, n_paths):
        return {"x": np.zeros(n_paths)}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        x0 = state["x"]
        e = stable_noise(rng, p["alpha"], (x0.shape[0], n), p["sigma"])
        y, _ = lfilter([1.0], [1.0, -p["rho"]], e, axis=-1, zi=(p["rho"] * x0)[:, None])
        return y[:, None, :], {"x": y[:, -1].copy()}

    def tags(self):
        t = []
        if self.params["alpha"] < 2.0:
            t.append(f"heavy_tail_alpha{_fmt(self.params['alpha'])}")
        else:
            t.append("gaussian_innovations")
        if self.params["rho"] >= 1.0:
            t.append("random_walk")
        return t


# ----------------------------------------------------------------------------------------
# 2. fractional Brownian motion
# ----------------------------------------------------------------------------------------

def fgn_autocov(H: float, n: int, sigma: float = 1.0) -> np.ndarray:
    """Autocovariance of fractional Gaussian noise at lags 0..n-1 (unit time step)."""
    k = np.arange(n, dtype=float)
    return 0.5 * sigma**2 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def fgn_lag1_autocorr(H: float) -> float:
    """Lag-1 autocorrelation of fractional Gaussian noise: 0.5 * (2^(2H) - 2)."""
    return 0.5 * (2.0 ** (2.0 * H) - 2.0)


def davies_harte(rng: np.random.Generator, H: float, n: int, n_paths: int = 1, sigma: float = 1.0) -> np.ndarray:
    """Exact fractional Gaussian noise `(n_paths, n)` by circulant embedding (Davies-Harte).

    The covariance is embedded in a circulant matrix of size 2n whose eigenvalues are computed
    by one FFT; they are non-negative for fractional Gaussian noise for every H in (0, 1).
    """
    gam = fgn_autocov(H, n + 1, sigma)
    c = np.concatenate([gam, gam[-2:0:-1]])           # circulant first row, length 2n
    lam = np.fft.fft(c).real
    assert lam.min() > -1e-8 * lam.max(), "circulant embedding is not non-negative definite"
    lam = np.maximum(lam, 0.0)
    N = c.shape[0]
    w = np.sqrt(lam) * (rng.standard_normal((n_paths, N)) + 1j * rng.standard_normal((n_paths, N)))
    x = np.fft.fft(w, axis=-1) / np.sqrt(N)            # real and imaginary parts: two independent samples
    return x.real[:, :n].copy()


_COND_CACHE: dict = {}


def _conditional_operator(H: float, sigma: float, h: int, n: int):
    """Predictor weights `W (n, h)` and noise factor `L (n, n)` such that, for history `x_h` (h increments),
    the next n increments are `W x_h + L z`, z ~ N(0, I): the exact Gaussian conditional.

    Computed from one Cholesky factor of the (h + n) Toeplitz covariance, using the block structure
    Sigma = [[L_hh, 0], [L_fh, L_ff]] [[L_hh, 0], [L_fh, L_ff]]^T: W = L_fh L_hh^-1, noise factor L_ff.
    The result depends only on (H, sigma, h, n), so it is computed once and applied to all paths.
    """
    key = (H, sigma, h, n)
    if key not in _COND_CACHE:
        if len(_COND_CACHE) > 8:
            _COND_CACHE.clear()
        S = toeplitz(fgn_autocov(H, h + n, sigma))
        S[np.diag_indices_from(S)] += 1e-10 * sigma**2
        L = cholesky(S, lower=True, check_finite=False)
        Lhh, Lfh, Lff = L[:h, :h], L[h:, :h], L[h:, h:]
        W = solve_triangular(Lhh, Lfh.T, lower=True, trans="T", check_finite=False).T
        _COND_CACHE[key] = (np.ascontiguousarray(W), np.ascontiguousarray(Lff))
    return _COND_CACHE[key]


class FractionalBM(Process):
    """Fractional Brownian motion observed on a unit grid; the series is B_H(t), started at 0.

    Increments are fractional Gaussian noise with autocovariance
    0.5 sigma^2 (|k+1|^2H - 2|k|^2H + |k-1|^2H). Knob: `H_hurst` (0.1 ... 0.9 in the grid).

    State: `hist (P, h)` = last `min(steps so far, max_hist)` increments, `level (P,)`.
    The first call from an empty history (the context) uses exact Davies-Harte; later calls
    (the horizon) draw increments from the exact Gaussian conditional given `hist`.
    `max_hist` (default 2048, documented minimum 1024) bounds the conditioning window.
    Long continuations are produced in chunks of 1024 increments, each conditioned on the
    last `max_hist` increments including the earlier chunks.
    """

    family, name = FAMILY, "fbm"
    defaults = {"H_hurst": 0.5, "sigma": 1.0, "max_hist": 2048}
    knob = "H_hurst"
    CHUNK = 1024

    def init_state(self, rng, n_paths):
        return {"hist": np.zeros((n_paths, 0)), "level": np.zeros(n_paths)}

    def _increments(self, hist: np.ndarray, n: int, rng) -> np.ndarray:
        p = self.params
        P = hist.shape[0]
        if hist.shape[1] == 0:
            return davies_harte(rng, p["H_hurst"], n, P, p["sigma"])
        out = np.empty((P, n))
        done = 0
        while done < n:
            m = min(self.CHUNK, n - done)
            hh = hist[:, -p["max_hist"]:]
            W, L = _conditional_operator(p["H_hurst"], p["sigma"], hh.shape[1], m)
            inc = hh @ W.T + rng.standard_normal((P, m)) @ L.T
            out[:, done: done + m] = inc
            done += m
            hist = np.concatenate([hist, inc], axis=1)[:, -p["max_hist"]:]
        return out

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        hist = state["hist"]
        inc = self._increments(hist, n, rng)
        obs = state["level"][:, None] + np.cumsum(inc, axis=1)
        new_hist = np.concatenate([hist, inc], axis=1)[:, -p["max_hist"]:]
        return obs[:, None, :], {"hist": new_hist, "level": obs[:, -1].copy()}

    def tags(self):
        H = self.params["H_hurst"]
        if H < 0.5:
            return [f"anti_persistent_H{_fmt(H)}"]
        if H > 0.5:
            return [f"long_memory_H{_fmt(H)}"]
        return ["brownian_control"]


# ----------------------------------------------------------------------------------------
# 3. Hawkes counts
# ----------------------------------------------------------------------------------------

class Hawkes(Process):
    """Self-exciting counts per time bin with an exponential kernel, in discrete time.

    This is a discrete-time approximation of a Hawkes process, not an event-level simulation.
    In bin t the count is N_t ~ Poisson(mu + s_t), and the excitation level follows
        s_{t+1} = d s_t + eta (1 - d) N_t,    d = exp(-1 / tau).
    Each event adds a kernel eta (1 - d) d^(k-1) to the intensity k bins later; the kernel sums
    to `eta`, the branching ratio. The base rate is mu = rate (1 - eta), so the stationary mean
    count per bin is `rate` for every eta and only the clustering changes. The excitation
    level s is the Markov state, so the oracle continues it exactly. Knob: `eta`.
    """

    family, name = FAMILY, "hawkes"
    defaults = {"eta": 0.6, "rate": 5.0, "tau": 4.0}
    knob = "eta"

    def warmup_steps(self) -> int:
        return 2000

    def init_state(self, rng, n_paths):
        p = self.params
        return {"s": np.full(n_paths, p["eta"] * p["rate"])}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        s = state["s"].copy()
        d = np.exp(-1.0 / p["tau"])
        mu = p["rate"] * (1.0 - p["eta"])
        out = np.empty((s.shape[0], n))
        for t in range(n):
            c = rng.poisson(mu + s)
            out[:, t] = c
            s = d * s + p["eta"] * (1.0 - d) * c
        return out[:, None, :], {"s": s}

    def tags(self):
        t = ["bursty"]
        if self.params["eta"] >= 0.9:
            t.append("near_critical")
        return t


# ----------------------------------------------------------------------------------------
# 4. Markov-switching multifractal
# ----------------------------------------------------------------------------------------

class MSMCascade(Process):
    """Markov-switching multifractal (Calvet-Fisher) volatility cascade.

    Volatility squared is sigma^2 * prod_i M_i, with k multipliers M_i that take the values m0 or
    2 - m0 with equal probability (so E[M_i] = 1). Multiplier i is renewed (redrawn) each step
    with probability gamma_i = 1 - (1 - gamma_k)^(b^(i-k)), i = 1..k, so frequencies are geometric
    with ratio b: multiplier k renews fastest (probability `gamma_k`) and multiplier 1 slowest.
    Returns are r_t = sigma sqrt(prod_i M_i) z_t with z_t standard normal.
    `observe` chooses the observed series: "returns" (r_t) or "cumsum" (the cumulative sum of r_t).
    The state is the multiplier vector (and the level, for "cumsum"); the oracle is Monte Carlo from it.
    Knob: `k` (1, 2, 4, 6, 8 in the grid).
    """

    family, name = FAMILY, "msm_cascade"
    defaults = {"k": 4, "m0": 1.4, "gamma_k": 0.5, "b": 3.0, "sigma": 1.0, "observe": "returns"}
    knob = "k"

    def __init__(self, **params):
        super().__init__(**params)
        assert self.params["observe"] in ("returns", "cumsum")
        assert 1.0 <= self.params["m0"] < 2.0

    def renewal_probs(self) -> np.ndarray:
        p = self.params
        i = np.arange(1, int(p["k"]) + 1)
        return 1.0 - (1.0 - p["gamma_k"]) ** (p["b"] ** (i - p["k"]))

    def _draw(self, rng, shape):
        m0 = self.params["m0"]
        return np.where(rng.random(shape) < 0.5, m0, 2.0 - m0)

    def init_state(self, rng, n_paths):
        return {"M": self._draw(rng, (n_paths, int(self.params["k"]))), "level": np.zeros(n_paths)}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        M, level = state["M"].copy(), state["level"].copy()
        gam = self.renewal_probs()
        P, k = M.shape
        z = rng.standard_normal((P, n))
        r = np.empty((P, n))
        for t in range(n):
            renew = rng.random((P, k)) < gam
            M = np.where(renew, self._draw(rng, (P, k)), M)
            r[:, t] = p["sigma"] * np.sqrt(M.prod(axis=1)) * z[:, t]
        if p["observe"] == "returns":
            obs, new_level = r, level
        else:
            obs = level[:, None] + np.cumsum(r, axis=1)
            new_level = obs[:, -1].copy()
        return obs[:, None, :], {"M": M, "level": new_level}

    def tags(self):
        t = ["multifractal"]
        if self.params["observe"] == "cumsum":
            t.append("stochastic_vol_random_walk")
        return t
