"""Regime-switch family: an in-prior-like base series with one thing out of the prior, a switch.

Base series (all four generators):

    y_t = level_t + season_t + x_t

- `level_t`: a constant level (0), plus a jump after the switch for `MeanJump`.
- `season_t`: a sum of `harmonics` sinusoids at period `period`; the phase advances by
  2*pi/period each step, so a period change keeps the phase continuous.
- `x_t`: Gaussian AR(1) noise, x_t = rho * x_{t-1} + sigma_t * e_t.

Generators (class -> what changes at the switch):

- `MeanJump`      level jumps by `delta` stationary noise std units (std of x before the switch).
- `VarianceJump`  the innovation std sigma is multiplied by `factor`.
- `FrequencyJump` the seasonal period goes from `period` to `period * ratio`.
- `MarkovSwitchingAR` two AR(1) regimes (different mean, rho and sigma), switching back and
  forth throughout context and horizon with probability `p_switch` per step.

Placement of the switch (param `placement`, for the first three generators):

- "context": one scheduled switch, `d` steps before the forecast origin (the first `d`
  observations before the origin are post-switch). No later switch. If `d` is at least the
  length of the simulated context, the whole simulated series is post-switch (the switch
  happened before the visible window). The oracle therefore knows the post-switch regime exactly.
- "horizon": the series is pre-switch for the whole context. During the horizon, a switch
  happens with constant hazard `hazard` per step (once switched it stays switched). The
  oracle knows the hazard and both regimes but not the switch time; its Monte Carlo paths
  draw their own switch times.

Forecast origin: the origin is the end of the first `step` call on a fresh state, which is
the context call in `make_sample` (no warmup is used; the AR noise starts from the
stationary distribution of the starting regime). `d` counts back from that origin. When a
sweep slices shorter contexts from the end of one long sample, `d` and the context length
`T` interact: the switch is visible only if `d <= T`.

State (a dict of arrays with a leading path axis): `t` (time index), `origin`
(forecast origin index, -1 until set), `regime` (0 or 1 now), `x` (AR noise), `phase`
(seasonal phase), `u0` (a standard normal used once to start `x` from stationarity).
"""

from __future__ import annotations

import numpy as np

from otp.generators.base import Process

PLACEMENTS = ("context", "horizon")


def _season_coefs(amplitude: float, harmonics: int, phase_seed: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(phase_seed)
    amps = amplitude / (1 + np.arange(harmonics)) * r.uniform(0.5, 1.0, harmonics)
    phases = r.uniform(0, 2 * np.pi, harmonics)
    return amps, phases


def stationary_sd(rho: float, sigma: float) -> float:
    return float(sigma / np.sqrt(max(1e-9, 1.0 - rho ** 2)))


class _RegimeBase(Process):
    """Shared machinery: state handling, regime path, seasonal term, AR loop."""

    family = "regime"
    base_defaults = {"level": 0.0, "amplitude": 1.0, "harmonics": 2, "phase_seed": 0, "period": 24,
                     "rho": 0.5, "sigma": 0.25, "placement": "context", "d": 64, "hazard": 1.0 / 256.0}
    defaults = dict(base_defaults)

    def __init__(self, **params):
        super().__init__(**params)
        assert self.params["placement"] in PLACEMENTS, self.params["placement"]
        self._origin_seen: int | None = None
        self._amps, self._phases = _season_coefs(self.params["amplitude"], self.params["harmonics"],
                                                 self.params["phase_seed"])

    # ---- hooks for the three single-switch generators -------------------------------------
    def level_shift(self) -> float:
        return 0.0

    def sigma_mult(self) -> float:
        return 1.0

    def period_after(self) -> float:
        return float(self.params["period"])

    # ---------------------------------------------------------------------------------------
    @property
    def period(self) -> int:
        """Seasonal period of the regime in force at the forecast origin."""
        p = self.params
        if p["placement"] == "context":
            return int(round(self.period_after()))
        return int(round(p["period"]))

    def init_state(self, rng, n_paths):
        return {"t": np.zeros(n_paths, np.int64), "origin": np.full(n_paths, -1, np.int64),
                "regime": np.zeros(n_paths, np.int8), "x": np.zeros(n_paths), "phase": np.zeros(n_paths),
                "u0": rng.standard_normal(n_paths)}

    def tags(self):
        p = self.params
        if p["placement"] == "context":
            tags = [f"switch_in_context_d{int(p['d'])}"]
            if self._origin_seen is not None and p["d"] >= self._origin_seen:
                tags.append("switch_before_window")
            return tags
        return ["switch_in_horizon"]

    # ---- regime path ---------------------------------------------------------------------
    def _regimes(self, state, n, rng) -> np.ndarray:
        """Regime (0/1) at each of the next n steps, shape (P, n)."""
        p = self.params
        t0 = state["t"]
        tt = t0[:, None] + np.arange(n)[None, :]
        if p["placement"] == "context":
            return (tt >= (state["origin"] - int(p["d"]))[:, None]).astype(np.int8)
        u = rng.random((t0.shape[0], n))
        sw = (tt >= state["origin"][:, None]) & (u < p["hazard"])
        return (np.logical_or.accumulate(sw, axis=1) | (state["regime"][:, None] == 1)).astype(np.int8)

    def _sigma_rho(self, R):
        """Per-step sigma and rho arrays (P, n) from the regime path."""
        p = self.params
        sig = p["sigma"] * np.where(R == 1, self.sigma_mult(), 1.0)
        return sig, np.full(R.shape, p["rho"])

    def _levels(self, R):
        return self.params["level"] + self.level_shift() * R

    def _phase_inc(self, R):
        p0 = float(self.params["period"])
        p1 = self.period_after()
        return 2 * np.pi / np.where(R == 1, p1, p0)

    def _season(self, phase_path):
        y = np.zeros_like(phase_path)
        for j in range(len(self._amps)):
            y += self._amps[j] * np.sin((j + 1) * phase_path + self._phases[j])
        return y

    def _begin(self, state, n, rng):
        """Set origin on the first call; returns a working copy of the state."""
        st = {k: v.copy() for k, v in state.items()}
        if (st["origin"] < 0).all():
            assert (st["t"] == 0).all(), "the first step call must start at t = 0"
            st["origin"][:] = n
            self._origin_seen = int(n)
            first = True
        else:
            first = False
        return st, first

    def _start_x(self, st, R):
        """Start the AR noise from the stationary law of the starting regime (first call only)."""
        sig, rho = self._sigma_rho(R[:, :1])
        st["x"] = st["u0"] * sig[:, 0] / np.sqrt(1.0 - rho[:, 0] ** 2)

    def step(self, state, n, rng, exo_rng=None):
        st, first = self._begin(state, n, rng)
        P = st["t"].shape[0]
        R = self._regimes(st, n, rng)
        if first:
            self._start_x(st, R)
        sig, rho = self._sigma_rho(R)
        e = rng.standard_normal((P, n)) * sig
        x = st["x"].copy()
        xs = np.empty((P, n))
        for i in range(n):
            x = rho[:, i] * x + e[:, i]
            xs[:, i] = x
        phase_path = st["phase"][:, None] + np.cumsum(self._phase_inc(R), axis=1)
        y = self._levels(R) + self._season(phase_path) + xs
        new = {"t": st["t"] + n, "origin": st["origin"], "regime": R[:, -1].astype(np.int8), "x": x,
               "phase": phase_path[:, -1], "u0": st["u0"]}
        return y[:, None, :], new


class MeanJump(_RegimeBase):
    """Level jumps by `delta` x (stationary std of the AR noise before the switch)."""

    name = "mean_jump"
    defaults = {**_RegimeBase.base_defaults, "delta": 2.0}
    knob = "delta"

    def level_shift(self):
        return self.params["delta"] * stationary_sd(self.params["rho"], self.params["sigma"])


class VarianceJump(_RegimeBase):
    """Innovation std multiplied by `factor` at the switch."""

    name = "variance_jump"
    defaults = {**_RegimeBase.base_defaults, "factor": 2.0}
    knob = "factor"

    def sigma_mult(self):
        return float(self.params["factor"])


class FrequencyJump(_RegimeBase):
    """Seasonal period goes from `period` to `period * ratio` at the switch (phase continuous)."""

    name = "frequency_jump"
    defaults = {**_RegimeBase.base_defaults, "ratio": 2.0}
    knob = "ratio"

    def period_after(self):
        return float(self.params["period"]) * float(self.params["ratio"])


class MarkovSwitchingAR(_RegimeBase):
    """Two AR(1) regimes with a symmetric Markov switch of probability `p_switch` per step.

    Regime 0: mean 0, `rho0`, `sigma`. Regime 1: mean `mean1` x (regime-0 stationary std),
    `rho1`, `sigma * sigma1_mult`. Switching runs through context and horizon; the oracle
    continues each path from the true current regime. The season is shared by both regimes.
    """

    name = "markov_ar"
    defaults = {**_RegimeBase.base_defaults, "p_switch": 0.01, "mean1": 2.0, "rho1": 0.9, "sigma1_mult": 1.0}
    knob = "p_switch"

    def __init__(self, **params):
        params.setdefault("rho", 0.3)
        super().__init__(**params)
        self._rho = np.array([self.params["rho"], self.params["rho1"]])
        self._sig = self.params["sigma"] * np.array([1.0, self.params["sigma1_mult"]])
        self._mean = np.array([0.0, self.params["mean1"] * stationary_sd(self.params["rho"], self.params["sigma"])])

    @property
    def period(self):
        return int(round(self.params["period"]))

    def init_state(self, rng, n_paths):
        s = super().init_state(rng, n_paths)
        s["regime"] = (rng.random(n_paths) < 0.5).astype(np.int8)
        return s

    def tags(self):
        return ["markov_switching"]

    def step(self, state, n, rng, exo_rng=None):
        st, first = self._begin(state, n, rng)
        P = st["t"].shape[0]
        if first:
            r0 = st["regime"].astype(int)
            st["x"] = st["u0"] * self._sig[r0] / np.sqrt(1.0 - self._rho[r0] ** 2)
        flips = rng.random((P, n)) < self.params["p_switch"]
        z = rng.standard_normal((P, n))
        r = st["regime"].astype(int)
        x = st["x"].copy()
        xs = np.empty((P, n))
        rs = np.empty((P, n), np.int8)
        for i in range(n):
            if i > 0 or not first:
                r = np.where(flips[:, i], 1 - r, r)
            x = self._rho[r] * x + self._sig[r] * z[:, i]
            xs[:, i] = x
            rs[:, i] = r
        phase_path = st["phase"][:, None] + (np.arange(1, n + 1) * 2 * np.pi / self.params["period"])[None, :]
        y = self._mean[rs] + self._season(phase_path) + xs
        new = {"t": st["t"] + n, "origin": st["origin"], "regime": r.astype(np.int8), "x": x,
               "phase": phase_path[:, -1], "u0": st["u0"]}
        return y[:, None, :], new


def randomized_base(seed: int) -> dict:
    """Per-seed base-series parameters: AR coefficient, noise level, season shape. In-prior-like."""
    r = np.random.default_rng(10_000 + seed)
    rho = float(r.uniform(0.2, 0.8))
    sd = float(r.uniform(0.2, 0.6))                    # stationary std of the AR noise
    return {"rho": rho, "sigma": sd * float(np.sqrt(1 - rho ** 2)), "harmonics": int(r.integers(1, 3)),
            "phase_seed": int(r.integers(1_000_000_000)), "amplitude": float(r.uniform(0.8, 1.5))}
