"""Coupled systems: a main series `y` (channel 0) and a covariate `x` (channel 1) with known equations.

Used to ask when Timestral's one-way covariate attention (covariate rows never read the main
series) costs accuracy. Every system has a difficulty knob and is vectorized over paths, so
the Monte Carlo true-dynamics oracle works as in `otp.generators.base`.

Systems (family "coupled"):

- `LaggedDriver`      x -> y. x is an exogenous AR + season series, y_t = a * x_{t-L} + small AR noise.
                      Knob `lag` L. x is offered as past-only or known-future covariate.
- `NonlinearDriver`   x -> y with y_t = a * sd_x * ((1 - s) u + s g(u)) + noise, u = x_{t-L} / sd_x.
                      g is sin(omega u) or a sign threshold. Knob `nonlin` s in [0, 1].
- `FeedbackLinear`    y <-> x: y_t = phi_y y_{t-1} + k x_{t-1} + noise, x_t = phi_x x_{t-1} - r k y_{t-1} + noise.
                      Knob `coupling` k (0 = independent).
- `FeedbackPredPrey`  y <-> x: noisy predator-prey in log space, y = prey, x = predator, linearized
                      around the equilibrium (1, 1) with self-damping. Knob `coupling` k (0 = independent).
- `HiddenDriver`      latent z (AR + season) drives y with lag; x_t = z_t + noise. Knob `x_noise`
                      (noise sd of x as a multiple of sd of z). Only x is shown, never z.
- `LeakyKnown`        x_t = y_{t-1} + noise, offered as a known-future covariate. Knob `noise_ratio`
                      (noise sd over the innovation sd of y). The oracle for the known-covariate condition
                      is a Gaussian smoother (exact), built in `LeakyKnown.conditional_oracle`.

Oracles. Each sample stores two sets of quantiles on y:

- `oracle_q`        conditions on the realized future of x when x is an exogenous driver (shared exogenous
                    noise across paths); this is the best forecast of a model that is given x's future.
                    For LeakyKnown it is the exact posterior given the noisy copy of y, per horizon H
                    (`oracle_cond_by_h`), because the posterior at a step depends on how far x extends.
- `oracle_marg_q`   averages over the future of x (independent exogenous noise across paths): the best
                    forecast of a model that is not given x's future. Equal to `oracle_q` for systems
                    where x's future is not offered.

`CovSample.oracle_for(known, h)` returns the oracle that matches a condition's information set.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from otp.generators.base import LEVELS, Process, Sample, shared_normal, tile_state

Z = norm.ppf(LEVELS)


def _season_terms(r: np.random.Generator, amp: float, harmonics: int) -> tuple[np.ndarray, np.ndarray]:
    amps = amp / (1 + np.arange(harmonics)) * r.uniform(0.5, 1.0, harmonics)
    phases = r.uniform(0, 2 * np.pi, harmonics)
    return amps, phases


def _season(tt: np.ndarray, period: float, amps: np.ndarray, phases: np.ndarray) -> np.ndarray:
    out = np.zeros_like(tt, dtype=float)
    for j, (a, ph) in enumerate(zip(amps, phases)):
        out += a * np.sin(2 * np.pi * (j + 1) * tt / period + ph)
    return out


def _ar_filter(x0: np.ndarray, rho: float, inn: np.ndarray) -> np.ndarray:
    """AR(1) recursion from state `x0` (P,) with innovations `inn` (P, n); returns (P, n)."""
    out = np.empty_like(inn)
    x = x0
    for i in range(inn.shape[1]):
        x = rho * x + inn[:, i]
        out[:, i] = x
    return out


class CoupledProcess(Process):
    """Base for the coupled systems: channel 0 is y (main), channel 1 is x (covariate)."""

    family = "coupled"
    n_channels = 2
    main_rows = (0,)
    past_cov_rows = (1,)
    known_cov_rows: tuple[int, ...] = ()
    exo_known = False       # x is an exogenous driver whose future can be given and conditioned on
    share_exo = True        # exogenous noise identical across paths (conditional oracle)

    def _exo_normal(self, exo_rng, rng, p: int, shape: tuple[int, ...]) -> np.ndarray:
        # one path (the context) always draws from exo_rng so both oracle modes see the same context
        if self.share_exo or p == 1:
            return shared_normal(exo_rng, p, shape)
        return rng.standard_normal((p, *shape))


# ---------------------------------------------------------------------------------------------
# x -> y drivers
# ---------------------------------------------------------------------------------------------

class _ExoDriver(CoupledProcess):
    """x exogenous (AR + season); y is a lagged map of x plus small AR noise."""

    known_cov_rows = (1,)
    exo_known = True
    defaults = {"lag": 32, "a": 1.0, "period": 24, "amp": 1.0, "harmonics": 2, "phase_seed": 0,
                "rho_x": 0.9, "sigma_x": 0.3, "rho_e": 0.5, "noise_ratio": 0.1}

    def __init__(self, **params):
        super().__init__(**params)
        p = self.params
        r = np.random.default_rng(p["phase_seed"])
        self._amps, self._phases = _season_terms(r, p["amp"], int(p["harmonics"]))
        self.sd_x = float(np.sqrt(0.5 * np.sum(self._amps ** 2) + p["sigma_x"] ** 2 / (1 - p["rho_x"] ** 2)))
        self.sigma_e = p["noise_ratio"] * self.sd_x * np.sqrt(1 - p["rho_e"] ** 2)

    def warmup_steps(self) -> int:
        return int(self.params["lag"]) + 300

    def _map(self, u: np.ndarray) -> np.ndarray:
        """Map from normalized lagged driver u to y / (a * sd_x). Identity here."""
        return u

    def init_state(self, rng, n_paths):
        p = self.params
        sd_u = p["sigma_x"] / np.sqrt(1 - p["rho_x"] ** 2)
        sd_e = self.sigma_e / np.sqrt(1 - p["rho_e"] ** 2)
        draw = (lambda s: rng.standard_normal(n_paths) * s) if n_paths == 1 else (lambda s: np.zeros(n_paths))
        return {"t": np.zeros(n_paths), "u": draw(sd_u), "e": draw(sd_e),
                "xbuf": np.zeros((n_paths, int(p["lag"])))}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        npaths = state["u"].shape[0]
        t0 = float(state["t"][0])
        m = _season(t0 + np.arange(n), p["period"], self._amps, self._phases)
        z = self._exo_normal(exo_rng, rng, npaths, (n,))
        xs = _ar_filter(state["u"], p["rho_x"], p["sigma_x"] * z)
        u_end = xs[:, -1].copy()
        xs = xs + m[None, :]
        xcat = np.concatenate([state["xbuf"], xs], axis=1)        # column i holds x_{t0 + i - L}
        drive = xcat[:, :n]
        e = _ar_filter(state["e"], p["rho_e"], self.sigma_e * rng.standard_normal((npaths, n)))
        y = p["a"] * self.sd_x * self._map(drive / self.sd_x) + e
        new = {"t": state["t"] + n, "u": u_end, "e": e[:, -1].copy(), "xbuf": xcat[:, n:].copy()}
        return np.stack([y, xs], axis=1), new

    def tags(self):
        return ["lag>=1" if self.params["lag"] >= 1 else "contemporaneous"]


class LaggedDriver(_ExoDriver):
    family, name = "coupled", "lagged_driver"
    knob = "lag"


class NonlinearDriver(_ExoDriver):
    family, name = "coupled", "nonlinear_driver"
    defaults = {**_ExoDriver.defaults, "lag": 32, "nonlin": 0.5, "g": "sin", "omega": 2.0}
    knob = "nonlin"

    def _map(self, u):
        s = self.params["nonlin"]
        if self.params["g"] == "sin":
            g = np.sin(self.params["omega"] * u)
        elif self.params["g"] == "threshold":
            g = np.where(u > 0.0, 1.0, -1.0)
        else:
            raise ValueError(f"unknown g {self.params['g']!r}")
        return (1.0 - s) * u + s * g


# ---------------------------------------------------------------------------------------------
# y <-> x feedback
# ---------------------------------------------------------------------------------------------

class FeedbackLinear(CoupledProcess):
    """Bidirectional linear coupling (a VAR(1)). Simultaneous update from the previous state."""

    family, name = "coupled", "feedback_linear"
    defaults = {"coupling": 0.15, "phi_y": 0.5, "phi_x": 0.95, "sigma_y": 1.0, "sigma_x": 0.5, "fb_ratio": 0.5}
    knob = "coupling"

    def warmup_steps(self):
        return 1500

    def spectral_radius(self) -> float:
        p = self.params
        a = np.array([[p["phi_y"], p["coupling"]], [-p["fb_ratio"] * p["coupling"], p["phi_x"]]])
        return float(np.abs(np.linalg.eigvals(a)).max())

    def init_state(self, rng, n_paths):
        return {"y": np.zeros(n_paths), "x": np.zeros(n_paths)}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        k = p["coupling"]
        y, x = state["y"].copy(), state["x"].copy()
        npaths = y.shape[0]
        ey = p["sigma_y"] * rng.standard_normal((npaths, n))
        ex = p["sigma_x"] * rng.standard_normal((npaths, n))
        out = np.empty((npaths, 2, n))
        for i in range(n):
            y, x = p["phi_y"] * y + k * x + ey[:, i], p["phi_x"] * x - p["fb_ratio"] * k * y + ex[:, i]
            out[:, 0, i], out[:, 1, i] = y, x
        return out, {"y": y, "x": x}

    def tags(self):
        return [f"spectral_radius={self.spectral_radius():.3f}"]


class FeedbackPredPrey(CoupledProcess):
    """Noisy predator-prey in log space. y = prey u, x = predator v, both positive.

    d log u = dt * (-a (u - 1) - k b (v - 1)) + s sqrt(dt) e,
    d log v = dt * (-a (v - 1) + k d (u - 1)) + s sqrt(dt) e'  (the v step uses the updated u).
    Near the equilibrium (1, 1) this is damped (rate a) with oscillation frequency k sqrt(b d) per
    unit time; the noise keeps it going. k = 0 gives two independent damped noisy series.
    """

    family, name = "coupled", "feedback_predprey"
    defaults = {"coupling": 1.0, "a": 0.2, "b": 1.0, "d": 1.0, "dt": 0.1, "noise": 0.2}
    knob = "coupling"

    def warmup_steps(self):
        return 2000

    def init_state(self, rng, n_paths):
        return {"lu": np.zeros(n_paths), "lv": np.zeros(n_paths)}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        k, dt, s = p["coupling"], p["dt"], p["noise"]
        lu, lv = state["lu"].copy(), state["lv"].copy()
        npaths = lu.shape[0]
        eu = s * np.sqrt(dt) * rng.standard_normal((npaths, n))
        ev = s * np.sqrt(dt) * rng.standard_normal((npaths, n))
        out = np.empty((npaths, 2, n))
        for i in range(n):
            u, v = np.exp(lu), np.exp(lv)
            lu = lu + dt * (-p["a"] * (u - 1.0) - k * p["b"] * (v - 1.0)) + eu[:, i]
            u = np.exp(lu)                      # semi-implicit: the predator step sees the updated prey
            lv = lv + dt * (-p["a"] * (v - 1.0) + k * p["d"] * (u - 1.0)) + ev[:, i]
            out[:, 0, i], out[:, 1, i] = u, np.exp(lv)
        return out, {"lu": lu, "lv": lv}


# ---------------------------------------------------------------------------------------------
# latent driver
# ---------------------------------------------------------------------------------------------

class HiddenDriver(CoupledProcess):
    """Latent z (AR + season) drives y with lag L. The covariate x is z plus measurement noise."""

    family, name = "coupled", "hidden_driver"
    defaults = {"lag": 32, "a": 1.0, "period": 24, "amp": 1.0, "harmonics": 2, "phase_seed": 0,
                "rho_z": 0.95, "sigma_z": 0.2, "rho_e": 0.5, "noise_ratio": 0.1, "x_noise": 1.0}
    knob = "x_noise"

    def __init__(self, **params):
        super().__init__(**params)
        p = self.params
        r = np.random.default_rng(p["phase_seed"])
        self._amps, self._phases = _season_terms(r, p["amp"], int(p["harmonics"]))
        self.sd_z = float(np.sqrt(0.5 * np.sum(self._amps ** 2) + p["sigma_z"] ** 2 / (1 - p["rho_z"] ** 2)))
        self.sigma_e = p["noise_ratio"] * self.sd_z * np.sqrt(1 - p["rho_e"] ** 2)

    def warmup_steps(self):
        return int(self.params["lag"]) + 500

    def init_state(self, rng, n_paths):
        p = self.params
        draw = (lambda s: rng.standard_normal(n_paths) * s) if n_paths == 1 else (lambda s: np.zeros(n_paths))
        return {"t": np.zeros(n_paths), "u": draw(p["sigma_z"] / np.sqrt(1 - p["rho_z"] ** 2)),
                "e": draw(self.sigma_e / np.sqrt(1 - p["rho_e"] ** 2)), "zbuf": np.zeros((n_paths, int(p["lag"])))}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        npaths = state["u"].shape[0]
        t0 = float(state["t"][0])
        m = _season(t0 + np.arange(n), p["period"], self._amps, self._phases)
        u = _ar_filter(state["u"], p["rho_z"], p["sigma_z"] * rng.standard_normal((npaths, n)))
        u_end = u[:, -1].copy()
        z = u + m[None, :]
        zcat = np.concatenate([state["zbuf"], z], axis=1)
        e = _ar_filter(state["e"], p["rho_e"], self.sigma_e * rng.standard_normal((npaths, n)))
        y = p["a"] * zcat[:, :n] + e
        x = z + p["x_noise"] * self.sd_z * rng.standard_normal((npaths, n))
        new = {"t": state["t"] + n, "u": u_end, "e": e[:, -1].copy(), "zbuf": zcat[:, n:].copy()}
        return np.stack([y, x], axis=1), new


# ---------------------------------------------------------------------------------------------
# leaky known-future covariate
# ---------------------------------------------------------------------------------------------

class LeakyKnown(CoupledProcess):
    """y = season + AR(1); x_t = y_{t-1} + noise; x is given with its future.

    Then x_{t+1} is a noisy copy of y_t, so the covariate's future reveals the main series' future
    (except y at the last horizon step). Noise sd = `noise_ratio` times the innovation sd of y.
    """

    family, name = "coupled", "leaky_known"
    known_cov_rows = (1,)
    analytic_cond = True
    defaults = {"period": 24, "amp": 1.0, "harmonics": 2, "phase_seed": 0, "rho": 0.9, "sigma_e": 0.3,
                "noise_ratio": 1.0}
    knob = "noise_ratio"

    def __init__(self, **params):
        super().__init__(**params)
        p = self.params
        r = np.random.default_rng(p["phase_seed"])
        self._amps, self._phases = _season_terms(r, p["amp"], int(p["harmonics"]))
        self.noise_sd = p["noise_ratio"] * p["sigma_e"]

    def warmup_steps(self):
        return 300

    def season_at(self, tt: np.ndarray) -> np.ndarray:
        return _season(tt, self.params["period"], self._amps, self._phases)

    def init_state(self, rng, n_paths):
        p = self.params
        sd = p["sigma_e"] / np.sqrt(1 - p["rho"] ** 2)
        u = rng.standard_normal(n_paths) * sd if n_paths == 1 else np.zeros(n_paths)
        return {"t": np.zeros(n_paths), "u": u, "ylast": self.season_at(np.array([-1.0]))[0] + u}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        npaths = state["u"].shape[0]
        t0 = float(state["t"][0])
        m = self.season_at(t0 + np.arange(n))
        u = _ar_filter(state["u"], p["rho"], p["sigma_e"] * rng.standard_normal((npaths, n)))
        y = u + m[None, :]
        yprev = np.concatenate([state["ylast"][:, None], y[:, :-1]], axis=1)
        x = yprev + self.noise_sd * rng.standard_normal((npaths, n))
        return np.stack([y, x], axis=1), {"t": state["t"] + n, "u": u[:, -1].copy(), "ylast": y[:, -1].copy()}

    def conditional_oracle(self, state: dict, x_future: np.ndarray, horizons: list[int]) -> dict[int, np.ndarray]:
        """Exact quantiles of y given the realized future of x, for each horizon length.

        `state` is the single-path state at the forecast origin (u_{T-1} known exactly), `x_future` the
        realized x_T .. x_{T+Hmax-1}. For horizon H the measurements are z_k = x_{T+k+1} - m_{T+k} = u_{T+k} +
        noise, k = 0 .. H-2; u is AR(1), so a Kalman filter plus RTS smoother gives each step's posterior.
        The last step (k = H-1) has no measurement and is predicted from the smoothed k = H-2 step.
        """
        p = self.params
        rho, s2, r = p["rho"], p["sigma_e"] ** 2, self.noise_sd ** 2
        t0 = float(state["t"][0])
        out = {}
        for h in horizons:
            m = self.season_at(t0 + np.arange(h))
            mean = np.empty(h)
            var = np.empty(h)
            if h >= 2:
                zobs = x_future[1:h] - m[: h - 1]
                nm = h - 1
                fm, fp, pm, pp = np.empty(nm), np.empty(nm), np.empty(nm), np.empty(nm)
                mu, pv = float(state["u"][0]), 0.0
                for k in range(nm):
                    pm[k], pp[k] = rho * mu, rho * rho * pv + s2
                    gain = pp[k] / (pp[k] + r)
                    mu = pm[k] + gain * (zobs[k] - pm[k])
                    pv = (1 - gain) * pp[k]
                    fm[k], fp[k] = mu, pv
                sm, sp = fm.copy(), fp.copy()
                for k in range(nm - 2, -1, -1):
                    g = fp[k] * rho / pp[k + 1]
                    sm[k] = fm[k] + g * (sm[k + 1] - pm[k + 1])
                    sp[k] = fp[k] + g * g * (sp[k + 1] - pp[k + 1])
                mean[: h - 1], var[: h - 1] = sm, sp
                mean[h - 1], var[h - 1] = rho * sm[-1], rho * rho * sp[-1] + s2
            else:
                mean[0], var[0] = rho * float(state["u"][0]), s2
            q = (m + mean)[:, None] + np.sqrt(np.maximum(var, 0.0))[:, None] * Z[None, :]
            out[h] = q[None]
        return out


# ---------------------------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------------------------

@dataclass
class CovSample(Sample):
    oracle_marg_q: np.ndarray | None = None                       # (1, H, 21) oracle that does not see x's future
    oracle_cond_by_h: dict[int, np.ndarray] | None = None         # LeakyKnown: exact oracle per horizon

    def oracle_for(self, known: bool, h: int) -> np.ndarray:
        """Oracle quantiles `(1, h, 21)` matching the information set: with x's future (`known`) or without."""
        if not known:
            return self.oracle_marg_q[..., :h, :]
        if self.oracle_cond_by_h is not None:
            return self.oracle_cond_by_h[h]
        return self.oracle_q[..., :h, :]


def _quantiles(paths: np.ndarray) -> np.ndarray:
    return np.moveaxis(np.quantile(paths, LEVELS, axis=0), 0, -1)     # (V, H, 21)


def make_cov_sample(proc: CoupledProcess, context_len: int, horizon: int, seed: int, n_paths: int = 500,
                    sample_id: str | None = None, cond_horizons: list[int] | None = None) -> CovSample:
    """Like `base.make_sample`, plus the oracle that does not see x's future and the exact LeakyKnown oracle."""
    rng = np.random.default_rng(seed)
    exo_seed = int(rng.integers(2**63 - 1))
    exo_rng = np.random.default_rng(exo_seed)
    proc.share_exo = True
    state = proc.init_state(rng, 1)
    w = proc.warmup_steps()
    if w > 0:
        _, state = proc.step(state, w, rng, exo_rng)
    ctx, state = proc.step(state, context_len, rng, exo_rng)               # (1, C, T)
    fut, _ = proc.step(tile_state(state, n_paths + 1), horizon, rng, np.random.default_rng(exo_seed + 1))
    main = list(proc.main_rows)
    oracle_q = _quantiles(fut[1:, main])
    marg_q = oracle_q
    if proc.exo_known:
        proc.share_exo = False
        fut2, _ = proc.step(tile_state(state, n_paths + 1), horizon, np.random.default_rng(exo_seed + 2),
                            np.random.default_rng(exo_seed + 1))
        proc.share_exo = True
        marg_q = _quantiles(fut2[1:, main])
    cond_by_h = None
    if getattr(proc, "analytic_cond", False):
        cond_by_h = proc.conditional_oracle(state, fut[0, 1], cond_horizons or [horizon])
        oracle_q = cond_by_h[horizon]
        marg_q = _quantiles(fut[1:, main])
    kr = list(proc.known_cov_rows)
    known = np.concatenate([ctx[0, kr], fut[0, kr]], axis=-1) if kr else None
    meta = {"family": proc.family, "generator": proc.name, "knob": proc.knob, "knob_value": proc.params.get(proc.knob),
            "params": dict(proc.params), "seed": seed, "T": context_len, "H": horizon, "tags": proc.tags(),
            "sample_id": sample_id or f"{proc.name}-{seed}"}
    return CovSample(ctx[0, main].astype(np.float64), fut[0, main].astype(np.float64), oracle_q, proc.period, meta,
                     ctx[0, list(proc.past_cov_rows)].astype(np.float64),
                     None if known is None else known.astype(np.float64), marg_q, cond_by_h)


def save_cov_samples(path: str, samples: list[CovSample]) -> None:
    enc = json.JSONEncoder(default=lambda o: o.item() if hasattr(o, "item") else str(o))
    arr = {
        "context": np.stack([s.context for s in samples]), "future": np.stack([s.future for s in samples]),
        "oracle_q": np.stack([s.oracle_q for s in samples]), "oracle_marg_q": np.stack([s.oracle_marg_q for s in samples]),
        "past_cov": np.stack([s.past_cov for s in samples]), "period": np.asarray([s.period for s in samples]),
        "meta": np.asarray([enc.encode(s.meta) for s in samples]),
    }
    if samples[0].known_cov is not None:
        arr["known_cov"] = np.stack([s.known_cov for s in samples])
    if samples[0].oracle_cond_by_h is not None:
        for h in samples[0].oracle_cond_by_h:
            arr[f"oracle_cond_h{h}"] = np.stack([s.oracle_cond_by_h[h] for s in samples])
    np.savez_compressed(path, **arr)


def load_cov_samples(path: str) -> list[CovSample]:
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    hs = sorted(int(k[len("oracle_cond_h"):]) for k in a if k.startswith("oracle_cond_h"))
    out = []
    for i in range(a["context"].shape[0]):
        out.append(CovSample(
            a["context"][i].copy(), a["future"][i].copy(), a["oracle_q"][i].copy(), int(a["period"][i]),
            json.loads(str(a["meta"][i])), a["past_cov"][i].copy(),
            a["known_cov"][i].copy() if "known_cov" in a else None, a["oracle_marg_q"][i].copy(),
            {h: a[f"oracle_cond_h{h}"][i].copy() for h in hs} if hs else None))
    return out


# ---------------------------------------------------------------------------------------------
# registry: systems, knob values, factories
# ---------------------------------------------------------------------------------------------

def _series_params(seed: int) -> dict:
    """Random season/AR parameters for an in-prior-like series, fixed by the sample seed."""
    r = np.random.default_rng(10_000 + seed)
    return {"period": int(r.choice([12, 24, 48, 96])), "amp": float(r.uniform(0.7, 1.5)),
            "harmonics": int(r.integers(1, 4)), "phase_seed": int(r.integers(1_000_000_000))}


SYSTEMS: dict[str, dict] = {
    "LaggedDriver": {"knob": "lag", "values": (0, 8, 32, 128, 512), "known": True, "t_gen": 8192,
                     "make": lambda seed, v: LaggedDriver(lag=int(v), rho_x=0.9, **_series_params(seed))},
    "NonlinearDriver": {"knob": "nonlin", "values": (0.0, 0.25, 0.5, 0.75, 1.0), "known": True, "t_gen": 2048,
                        "make": lambda seed, v: NonlinearDriver(nonlin=float(v), lag=32, g="sin", **_series_params(seed))},
    "FeedbackLinear": {"knob": "coupling", "values": (0.0, 0.05, 0.15, 0.4, 1.0), "known": False, "t_gen": 8192,
                       "make": lambda seed, v: FeedbackLinear(coupling=float(v))},
    "FeedbackPredPrey": {"knob": "coupling", "values": (0.0, 0.25, 0.5, 1.0, 2.0), "known": False, "t_gen": 8192,
                         "make": lambda seed, v: FeedbackPredPrey(coupling=float(v))},
    "HiddenDriver": {"knob": "x_noise", "values": (0.05, 0.25, 0.5, 1.0, 2.0), "known": False, "t_gen": 2048,
                     "make": lambda seed, v: HiddenDriver(x_noise=float(v), lag=32, **_series_params(seed))},
    "LeakyKnown": {"knob": "noise_ratio", "values": (0.05, 0.25, 1.0, 4.0, 16.0), "known": True, "t_gen": 2048,
                   "make": lambda seed, v: LeakyKnown(noise_ratio=float(v), **_series_params(seed))},
}
REDUCED_KNOB_IDX = (0, 2, 4)


def config_id(system: str, value) -> str:
    return f"{system}/{SYSTEMS[system]['knob']}={value:g}"


def sample_seed(cid: str, seed: int) -> int:
    return zlib.crc32(f"{cid}/{seed}".encode()) % (2**31)


def build_sample(system: str, value, seed: int, horizon: int = 1024, cond_horizons=(16, 128, 1024),
                 n_paths: int = 500, context_len: int | None = None) -> CovSample:
    spec = SYSTEMS[system]
    cid = config_id(system, value)
    proc = spec["make"](seed, value)
    s = make_cov_sample(proc, context_len or spec["t_gen"], horizon, sample_seed(cid, seed), n_paths=n_paths,
                        sample_id=f"{cid}/s{seed}", cond_horizons=list(cond_horizons))
    s.meta.update({"system": system, "config_id": cid})
    return s
