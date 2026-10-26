"""Mistral-style "in-prior" primitives: trend + seasonality + noise, and a weighted sum.

These approximate the kind of synthetic processes Timestral was pretrained on (simple
trend, seasonality and noise combined). They serve three roles:

1. sanity checks (Timestral should be close to the oracle here),
2. building blocks for the learned mixture-of-primitives generator,
3. replay data for fine-tuning and the "normal series" regression check.
"""

from __future__ import annotations

import numpy as np

from otp.generators.base import Process


class Trend(Process):
    """Level + slope, with optional random-walk drift on the slope. Dampening phi in (0, 1]."""

    family, name = "primitive", "trend"
    defaults = {"level": 0.0, "slope": 0.01, "slope_noise": 0.0, "phi": 1.0}
    knob = "slope"

    def init_state(self, rng, n_paths):
        p = self.params
        return {"level": np.full((n_paths,), p["level"], float), "slope": np.full((n_paths,), p["slope"], float)}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        lv, sl = state["level"].copy(), state["slope"].copy()
        out = np.empty((lv.shape[0], 1, n))
        for t in range(n):
            sl = p["phi"] * sl + p["slope_noise"] * rng.standard_normal(lv.shape[0])
            lv = lv + sl
            out[:, 0, t] = lv
        return out, {"level": lv, "slope": sl}


class Season(Process):
    """Sum of `harmonics` sinusoids at period `period`, fixed random phases/amplitudes per seed."""

    family, name = "primitive", "season"
    defaults = {"period": 24, "amplitude": 1.0, "harmonics": 2, "phase_seed": 0}
    knob = "period"

    def init_state(self, rng, n_paths):
        r = np.random.default_rng(self.params["phase_seed"])
        k = self.params["harmonics"]
        self._amps = self.params["amplitude"] / (1 + np.arange(k)) * r.uniform(0.5, 1.0, k)
        self._phases = r.uniform(0, 2 * np.pi, k)
        return {"t": np.zeros((n_paths,), float)}

    def step(self, state, n, rng, exo_rng=None):
        t0 = state["t"]
        tt = t0[:, None] + np.arange(n)[None, :]
        per = float(self.params["period"])
        y = np.zeros_like(tt)
        for j, (a, ph) in enumerate(zip(self._amps, self._phases)):
            y += a * np.sin(2 * np.pi * (j + 1) * tt / per + ph)
        return y[:, None, :], {"t": t0 + n}


class ARNoise(Process):
    """Gaussian AR(1) noise: x_t = rho x_{t-1} + sigma e_t."""

    family, name = "primitive", "ar_noise"
    defaults = {"rho": 0.5, "sigma": 0.1}
    knob = "rho"

    def init_state(self, rng, n_paths):
        p = self.params
        sd = p["sigma"] / np.sqrt(max(1e-6, 1 - p["rho"] ** 2))
        return {"x": rng.standard_normal(n_paths) * sd if n_paths == 1 else np.zeros(n_paths)}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        x = state["x"].copy()
        out = np.empty((x.shape[0], 1, n))
        e = rng.standard_normal((x.shape[0], n)) * p["sigma"]
        for t in range(n):
            x = p["rho"] * x + e[:, t]
            out[:, 0, t] = x
        return out, {"x": x}


class SumProcess(Process):
    """Weighted sum of univariate component processes with independent states.

    `components` is a list of `(weight, Process)`. Parameters of the sum are the weights;
    component parameters live on the components. `period` is the first seasonal
    component's period (for MASE scaling).
    """

    family, name = "primitive", "sum"
    defaults = {}

    def __init__(self, components, name: str | None = None, family: str | None = None):
        super().__init__()
        self.components = components
        if name:
            self.name = name
        if family:
            self.family = family
        self.params = {f"w{i}_{c.name}": w for i, (w, c) in enumerate(components)}
        for i, (_, c) in enumerate(components):
            for k, v in c.params.items():
                self.params[f"c{i}_{c.name}.{k}"] = v

    @property
    def period(self) -> int:
        for _, c in self.components:
            if c.period > 1:
                return c.period
        return 1

    def warmup_steps(self) -> int:
        return max([c.warmup_steps() for _, c in self.components] + [0])

    def init_state(self, rng, n_paths):
        return {f"c{i}": c.init_state(rng, n_paths) for i, (_, c) in enumerate(self.components)}

    def step(self, state, n, rng, exo_rng=None):
        total, new = None, {}
        for i, (w, c) in enumerate(self.components):
            y, new[f"c{i}"] = c.step(state[f"c{i}"], n, rng, exo_rng)
            y = w * y[:, list(c.main_rows)[:1]]
            total = y if total is None else total + y
        return total, new


def in_prior(seed: int, period: int | None = None) -> SumProcess:
    """A random trend + season + AR-noise series of the kind Timestral was pretrained on."""
    r = np.random.default_rng(seed)
    period = period or int(r.choice([7, 12, 24, 48, 96, 168]))
    comps = [
        (1.0, Trend(level=float(r.normal(0, 1)), slope=float(r.normal(0, 0.005)), phi=float(r.choice([1.0, 0.999])))),
        (float(r.uniform(0.5, 2.0)), Season(period=period, harmonics=int(r.integers(1, 4)), phase_seed=int(r.integers(1e9)))),
        (1.0, ARNoise(rho=float(r.uniform(0, 0.9)), sigma=float(r.uniform(0.05, 0.5)))),
    ]
    return SumProcess(comps, name="in_prior", family="in_prior")
