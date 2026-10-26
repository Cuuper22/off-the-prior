"""Chaos and dynamics family: five batched simulators with a difficulty knob each.

Every generator is a `Process` (see `base.py`) whose state has a leading path axis, so the
true-dynamics oracle can advance 500 noisy continuations at once. Loops run over time steps
only, never over paths.

Noise convention (same for all five)
------------------------------------
Each generator has two noise params, both relative to a reference scale `sd_ref`:

- `dyn_rel` (default 0.01): dynamic noise. After every internal integration step a
  Gaussian increment is added to every state variable (Euler-Maruyama style):
  `state += dyn_rel * sd_ref * sqrt(dt) * N(0, 1)`. For the logistic map `dt = 1`.
  For Kuramoto the increment is added to the phases (radians), with the same formula.
- `obs_rel` (default 0.01): observation noise, `obs = signal + obs_rel * sd_ref * N(0, 1)`,
  drawn fresh at every observed step. It is not part of the state.

`sd_ref` is a fixed constant per generator (class attribute, same for every knob value, so
the noise size does not change while the knob is swept). It is the standard deviation of the
observed noise-free signal on a chaotic reference configuration, measured by
`python -m otp.generators.chaos calibrate` and stored in `results/map/chaos/calibration.csv`.
The oracle starts from the true latent state at the forecast origin and re-draws both noises,
so its spread grows with the horizon.

Units: `dt` is the internal step; one observed step is `obs_every` internal steps, so
`dt_obs = dt * obs_every` time units pass per observed step. Lyapunov exponent: the average
exponential rate at which two nearby noise-free trajectories separate (positive means
chaos). `lyapunov` estimates it with the two-trajectory separation method (renormalise the
gap back to a small size every few steps and average the log growth), with no noise.

Regime tags returned by `tags()` come from the measured exponents in
`results/map/chaos/lyapunov.csv` (rule per generator in each class docstring).
"""

from __future__ import annotations

import numpy as np

from otp.generators.base import Process, tile_state

CHUNK = 1024  # internal steps of noise drawn at once (bounds memory for 501 paths)


def _angle_wrap(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


class _Integrated(Process):
    """Shared loop for ODE-like generators with state `u` of shape `(D, P)` internally.

    Subclasses give `_pack(state) -> u`, `_unpack(u) -> state`, `_rhs(u) -> du/dt`,
    `_observe(u) -> (P,)` and a class constant `sd_ref`. State is stored with the path axis
    first, `(P, D)`.
    """

    family = "chaos"
    n_channels = 1
    sd_ref = 1.0
    warmup_time = 100.0   # time units simulated before the context starts
    angle_keys: tuple[str, ...] = ()
    renorm_time = 1.0     # Lyapunov renormalisation interval (time units)

    @property
    def dt(self) -> float:
        return float(self.params["dt"])

    @property
    def dt_obs(self) -> float:
        return self.dt * int(self.params["obs_every"])

    def warmup_steps(self) -> int:
        return int(round(self.warmup_time / self.dt_obs))

    # --- subclass hooks -------------------------------------------------
    def _pack(self, state):
        return np.ascontiguousarray(state["u"].T)

    def _unpack(self, u):
        return {"u": np.ascontiguousarray(u.T)}

    def _rhs(self, u):
        raise NotImplementedError

    def _observe(self, u):
        return u[0]

    # --- integration ------------------------------------------------------
    def _rk4(self, u):
        h = self.dt
        k1 = self._rhs(u)
        k2 = self._rhs(u + (0.5 * h) * k1)
        k3 = self._rhs(u + (0.5 * h) * k2)
        k4 = self._rhs(u + h * k3)
        return u + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        u = self._pack(state)
        n_paths = u.shape[1]
        k = int(p["obs_every"])
        sig = p["dyn_rel"] * self.sd_ref * np.sqrt(self.dt)
        obs_sd = p["obs_rel"] * self.sd_ref
        out = np.empty((n_paths, 1, n))
        total = n * k
        noise = None
        for j in range(total):
            c = j % CHUNK
            if c == 0:
                noise = rng.standard_normal((min(CHUNK, total - j), *u.shape)) * sig
            u = self._rk4(u) + noise[c]
            if (j + 1) % k == 0:
                out[:, 0, (j + 1) // k - 1] = self._observe(u)
        out[:, 0, :] += obs_sd * rng.standard_normal((n_paths, n))
        return out, self._unpack(u)

    # --- Lyapunov helpers -------------------------------------------------
    def _diff(self, state):
        """Difference (path 1 minus path 0) of every state array, angles wrapped."""
        d = {}
        for key, a in state.items():
            x = a[1] - a[0]
            d[key] = _angle_wrap(x) if key in self.angle_keys else x
        return d


class Lorenz63(_Integrated):
    """Lorenz 1963 system, `x' = sigma (y - x)`, `y' = x (rho - z) - y`, `z' = x y - beta z`.

    RK4 with internal step `dt` = 0.01; observe `x` every `obs_every` = 10 internal steps
    (0.1 time units). With sigma = 10 and beta = 8/3 the origin loses stability at rho = 1,
    the two fixed points `(+-sqrt(beta (rho - 1)), ..., rho - 1)` lose stability at
    rho = 24.74 (the onset of the standard attractor), and chaos at rho = 28 has a largest
    Lyapunov exponent near 0.9 per time unit. Knob: `rho`. Second axis: `obs_every`, how many
    internal steps (hence how much of a Lyapunov time) pass per observed step.

    Warmup: 100 time units. Noise: `dyn_rel = obs_rel = 0.01` of `sd_ref` (see module
    docstring). Tags: "chaotic" and "aperiodic" when the measured exponent is positive,
    "fixed_point" when it is negative; "near_bifurcation" within 1.5 of rho = 24.74.
    """

    name = "lorenz63"
    defaults = {"rho": 28.0, "sigma": 10.0, "beta": 8.0 / 3.0, "dt": 0.01, "obs_every": 10,
                "dyn_rel": 0.01, "obs_rel": 0.01}
    knob = "rho"
    sd_ref = 7.93
    warmup_time = 100.0
    renorm_time = 1.0

    def init_state(self, rng, n_paths):
        u = np.array([1.0, 1.0, 20.0]) + rng.standard_normal((n_paths, 3))
        return {"u": u}

    def _rhs(self, u):
        p = self.params
        x, y, z = u
        out = np.empty_like(u)
        out[0] = p["sigma"] * (y - x)
        out[1] = x * (p["rho"] - z) - y
        out[2] = x * y - p["beta"] * z
        return out

    def tags(self):
        return _tags_from_table(self.name, self.params["rho"], near=(24.74, 1.5))


class Rossler(_Integrated):
    """Rossler system, `x' = -y - z`, `y' = x + a y`, `z' = b + z (x - c)`, a = b = 0.2.

    RK4 with `dt` = 0.02, observe `x` every `obs_every` = 10 internal steps (0.2 time
    units). Knob: `c`, swept across the period-doubling route to chaos (the grid values
    are listed in `otp/grids/chaos.py`; the measured exponents are in `lyapunov.csv`). Warmup: 500 time
    units. Noise: `dyn_rel = obs_rel = 0.01` of `sd_ref`. Tags: "periodic" where the
    measured exponent is zero, "chaotic" and "aperiodic" where it is positive (see `TAG_TABLE`).
    """

    name = "rossler"
    defaults = {"c": 5.7, "a": 0.2, "b": 0.2, "dt": 0.02, "obs_every": 10,
                "dyn_rel": 0.01, "obs_rel": 0.01}
    knob = "c"
    sd_ref = 5.13
    warmup_time = 500.0
    renorm_time = 5.0

    def init_state(self, rng, n_paths):
        u = np.array([1.0, 1.0, 0.5]) + 0.1 * rng.standard_normal((n_paths, 3))
        return {"u": u}

    def _rhs(self, u):
        p = self.params
        x, y, z = u
        out = np.empty_like(u)
        out[0] = -y - z
        out[1] = x + p["a"] * y
        out[2] = p["b"] + z * (x - p["c"])
        return out

    def tags(self):
        return _tags_from_table(self.name, self.params["c"])


class Kuramoto(_Integrated):
    """`N` Kuramoto phase oscillators, `theta_i' = w_i + (K/N) sum_j sin(theta_j - theta_i)`.

    Natural frequencies `w_i = omega0 + sigma_w z_i`, `z_i` standard normal drawn once from
    `default_rng(freq_seed)`, so they are fixed per seed (the grid sets `freq_seed` to the
    sample seed). Defaults: N = 16, omega0 = 1.5 rad per time unit, sigma_w = 1.0. RK4 with
    `dt` = 0.05, observe `mean_i sin(theta_i)` every `obs_every` = 2 internal steps (0.1
    time units; the mean frequency then has a period of about 42 observed steps).

    Theory for the synchronization threshold: for a Gaussian frequency density with spread
    sigma_w and infinitely many oscillators, `K_c = 2 / (pi g(0)) = sigma_w sqrt(8 / pi)`,
    which is 1.5958 for sigma_w = 1 (computed in `critical_coupling`). With N = 16 and
    one random draw of frequencies, the transition is blurred and shifted, so this is a
    guide, not a sharp edge. Knob: `K`. Phase noise: `dyn_rel` times `sd_ref` radians per
    root time unit added to each phase. Warmup: 600 time units. Tags: "quasi_periodic"
    where the measured exponent is zero or small (incoherent), "synchronized" where it is
    negative, and "near_bifurcation" near the finite-size threshold.
    """

    name = "kuramoto"
    defaults = {"K": 2.0, "N": 16, "omega0": 1.5, "sigma_w": 1.0, "freq_seed": 0, "dt": 0.05,
                "obs_every": 2, "dyn_rel": 0.01, "obs_rel": 0.01}
    knob = "K"
    sd_ref = 0.688
    warmup_time = 600.0
    renorm_time = 5.0
    angle_keys = ("theta",)

    def __init__(self, **params):
        super().__init__(**params)
        r = np.random.default_rng(int(self.params["freq_seed"]))
        self.omega = self.params["omega0"] + self.params["sigma_w"] * r.standard_normal(int(self.params["N"]))

    @staticmethod
    def critical_coupling(sigma_w: float) -> float:
        """Infinite-N threshold for Gaussian natural frequencies of spread `sigma_w`."""
        return float(sigma_w * np.sqrt(8.0 / np.pi))

    def init_state(self, rng, n_paths):
        return {"theta": rng.uniform(-np.pi, np.pi, (n_paths, int(self.params["N"])))}

    def _pack(self, state):
        return state["theta"].T.copy()          # (N, P)

    def _unpack(self, u):
        return {"theta": _angle_wrap(u.T).copy()}

    def _rhs(self, u):
        s, c = np.sin(u), np.cos(u)
        ms, mc = s.mean(axis=0), c.mean(axis=0)
        return self.omega[:, None] + self.params["K"] * (ms * c - mc * s)

    def _observe(self, u):
        return np.sin(u).mean(axis=0)

    def tags(self):
        return _tags_from_table(self.name, self.params["K"], near=(self.critical_coupling(self.params["sigma_w"]), 0.5))


class MackeyGlass(Process):
    """Mackey-Glass delay equation, `x'(t) = beta x(t - tau) / (1 + x(t - tau)^10) - gamma x(t)`.

    beta = 0.2, gamma = 0.1. Integrated with Heun's method (second order) at internal step
    `h` = 0.2 time units; the history of the last `tau / h + 1` internal values is kept in the
    state, shape `(P, buffer_len)` with the newest value last. Observe `x` every `obs_every`
    = 5 internal steps (1 time unit). `tau` must be a multiple of `h`. Knob: delay `tau`.
    The exponent is zero (closed loops) at the small grid values and positive from
    tau = 17, the common benchmark value (see `lyapunov.csv`). Warmup: 2000 time units; the initial history is constant 1.2
    plus small noise. Noise: `dyn_rel` is applied to the new value at every internal step
    as `dyn_rel * sd_ref * sqrt(h) * N(0, 1)`, and `obs_rel` to the observation; `sd_ref` as
    in the module docstring. Tags from measured exponents: "delay_memory" always,
    plus "chaotic", "aperiodic" where the exponent is positive and "periodic" otherwise.
    """

    family = "chaos"
    name = "mackey_glass"
    defaults = {"tau": 17.0, "beta": 0.2, "gamma": 0.1, "n_exp": 10.0, "h": 0.2, "obs_every": 5,
                "dyn_rel": 0.01, "obs_rel": 0.01}
    knob = "tau"
    sd_ref = 0.226
    warmup_time = 2000.0
    renorm_time = 10.0
    angle_keys: tuple[str, ...] = ()

    @property
    def dt(self) -> float:
        return float(self.params["h"])

    @property
    def dt_obs(self) -> float:
        return self.dt * int(self.params["obs_every"])

    @property
    def delay_steps(self) -> int:
        d = self.params["tau"] / self.params["h"]
        assert abs(d - round(d)) < 1e-6, "tau must be a multiple of h"
        return int(round(d))

    @property
    def buffer_len(self) -> int:
        return self.delay_steps + 1

    def warmup_steps(self) -> int:
        return int(round(self.warmup_time / self.dt_obs))

    def init_state(self, rng, n_paths):
        hist = 1.2 + 0.05 * rng.standard_normal((n_paths, self.buffer_len))
        return {"hist": hist}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        h, k, ls = p["h"], int(p["obs_every"]), self.buffer_len
        beta, gamma, ne = p["beta"], p["gamma"], p["n_exp"]
        hist = state["hist"]
        n_paths = hist.shape[0]
        total = n * k
        sig = p["dyn_rel"] * self.sd_ref * np.sqrt(h)
        work = np.empty((ls + total, n_paths))   # time-major: one row per internal step
        work[:ls] = hist.T
        noise = None
        for j in range(total):
            c = j % CHUNK
            if c == 0:
                noise = rng.standard_normal((min(CHUNK, total - j), n_paths)) * sig
            x = work[ls - 1 + j]
            d0 = work[j]
            d1 = work[j + 1]
            f0 = beta * d0 / (1.0 + d0 ** ne) - gamma * x
            f1 = beta * d1 / (1.0 + d1 ** ne) - gamma * (x + h * f0)
            work[ls + j] = x + 0.5 * h * (f0 + f1) + noise[c]
        idx = ls - 1 + k * np.arange(1, n + 1)
        obs = work[idx].T[:, None, :] + p["obs_rel"] * self.sd_ref * rng.standard_normal((n_paths, 1, n))
        return obs, {"hist": np.ascontiguousarray(work[-ls:].T)}

    def _diff(self, state):
        return {key: a[1] - a[0] for key, a in state.items()}

    def tags(self):
        return ["delay_memory"] + _tags_from_table(self.name, self.params["tau"])


class LogisticMap(Process):
    """Noisy logistic map, `x <- r x (1 - x) + noise`, observed directly.

    Dynamic noise per step: `dyn_rel * sd_ref * N(0, 1)` (dt = 1), added after the map. To
    keep the state inside (0, 1) it is reflected at the borders once (`x -> |x|`, then
    `x -> 1 - |1 - x|`) and clipped to `[1e-9, 1 - 1e-9]`. Observation noise
    `obs_rel * sd_ref * N(0, 1)` is added to the output only, so observed values can leave
    (0, 1). Knob: `r`. Period doubling at r = 3, 3.449, 3.544 and 3.5645, chaos from about
    r = 3.5699, a period-3 window near r = 3.83, full chaos at r = 4. Warmup: 2000 steps.
    Tags from measured exponents: "periodic" (with "near_bifurcation" at the grid values
    next to a doubling point) or "chaotic" and "aperiodic".
    """

    family = "chaos"
    name = "logistic_map"
    defaults = {"r": 3.9, "dyn_rel": 0.01, "obs_rel": 0.01}
    knob = "r"
    sd_ref = 0.299
    warmup_time = 2000.0
    renorm_time = 1.0
    angle_keys: tuple[str, ...] = ()

    @property
    def dt(self) -> float:
        return 1.0

    @property
    def dt_obs(self) -> float:
        return 1.0

    def warmup_steps(self) -> int:
        return int(self.warmup_time)

    def init_state(self, rng, n_paths):
        return {"x": rng.uniform(0.2, 0.8, n_paths)}

    def step(self, state, n, rng, exo_rng=None):
        p = self.params
        r = p["r"]
        x = state["x"].copy()
        out = np.empty((x.shape[0], 1, n))
        e = rng.standard_normal((x.shape[0], n)) * (p["dyn_rel"] * self.sd_ref)
        for t in range(n):
            x = r * x * (1.0 - x) + e[:, t]
            x = np.abs(x)
            x = 1.0 - np.abs(1.0 - x)
            x = np.clip(x, 1e-9, 1.0 - 1e-9)
            out[:, 0, t] = x
        out[:, 0, :] += p["obs_rel"] * self.sd_ref * rng.standard_normal((x.shape[0], n))
        return out, {"x": x}

    def _diff(self, state):
        return {key: a[1] - a[0] for key, a in state.items()}

    def tags(self):
        return _tags_from_table(self.name, self.params["r"])


# ---------------------------------------------------------------------------
# Regime tags. Each row: generator -> list of (upper_bound_exclusive, tags). Filled from the
# measured exponents in results/map/chaos/lyapunov.csv.
# ---------------------------------------------------------------------------
INF = float("inf")
TAG_TABLE: dict[str, list[tuple[float, list[str]]]] = {
    # rho 10, 15, 20: exponent below 0 (settles to a fixed point). rho 24: exponent above 0 in
    # the noise-free run (long chaotic transient before the fixed point wins). rho >= 26: above 0.
    "lorenz63": [(22.0, ["fixed_point"]), (24.74, ["chaotic_transient"]), (INF, ["chaotic", "aperiodic"])],
    # c 2, 3, 4: exponent 0 (closed loops). c >= 4.5: above 0.
    "rossler": [(4.25, ["periodic"]), (INF, ["chaotic", "aperiodic"])],
    # tau 6, 10, 14: exponent 0 (closed loops). tau >= 17: above 0.
    "mackey_glass": [(16.0, ["periodic"]), (INF, ["chaotic", "aperiodic"])],
    # r 2.8: fixed point; 3.2, 3.5, 3.56: cycles; 3.6 and 3.7: chaos; 3.83: period-3 window.
    "logistic_map": [(3.0, ["fixed_point"]), (3.55, ["periodic"]), (3.5699, ["periodic", "near_bifurcation"]),
                     (3.8, ["chaotic", "aperiodic"]), (3.86, ["periodic", "period3_window"]),
                     (INF, ["chaotic", "aperiodic"])],
    # K 0, 0.5: exponent about 0 (sum of incommensurate sinusoids). K 1 to 1.5: above 0 for
    # some frequency seeds. K >= 2: 0 (one rotating locked state).
    "kuramoto": [(0.75, ["quasi_periodic"]), (1.75, ["partial_sync"]), (INF, ["synchronized", "periodic"])],
}


def _tags_from_table(name: str, value: float, near: tuple[float, float] | None = None) -> list[str]:
    tags: list[str] = []
    for upper, t in TAG_TABLE.get(name, []):
        if value < upper:
            tags = list(t)
            break
    if near is not None and abs(value - near[0]) <= near[1]:
        tags = tags + ["near_bifurcation"]
    return tags


# ---------------------------------------------------------------------------
# Lyapunov estimation and calibration
# ---------------------------------------------------------------------------
def lyapunov(proc: Process, n_renorm: int = 1000, discard: int = 100, d0: float = 1e-8, seed: int = 0,
             n_blocks: int = 5) -> dict:
    """Largest Lyapunov exponent by two-trajectory separation, no noise.

    `proc` should have `dyn_rel = obs_rel = 0`. Two copies of the state are advanced; every
    `renorm_time` the gap (over the full state, angles wrapped) is measured, its log growth
    is added to the average and the second copy is pulled back to distance `d0`. The first
    `discard` intervals let the gap align with the unstable direction and are not counted.
    Returns the exponent per time unit with a standard error from `n_blocks` block means.
    """
    rng = np.random.default_rng(seed)
    state = proc.init_state(rng, 1)
    w = proc.warmup_steps()
    if w > 0:
        _, state = proc.step(state, w, rng)
    state = tile_state(state, 2)
    keys = list(state.keys())
    direction = {k: rng.standard_normal(state[k].shape[1:]) for k in keys}
    norm = np.sqrt(sum(float((v ** 2).sum()) for v in direction.values()))
    for k in keys:
        state[k][1] = state[k][0] + d0 * direction[k] / norm
    m = max(1, int(round(proc.renorm_time / proc.dt_obs)))
    interval = m * proc.dt_obs
    logs = []
    for i in range(n_renorm + discard):
        _, state = proc.step(state, m, rng)
        diff = proc._diff(state)
        d = np.sqrt(sum(float((v ** 2).sum()) for v in diff.values()))
        if not np.isfinite(d) or d <= 0:
            logs.append(np.nan)
            break
        if i >= discard:
            logs.append(np.log(d / d0) / interval)
        for k in keys:
            state[k][1] = state[k][0] + diff[k] * (d0 / d)
    arr = np.asarray(logs, float)
    blocks = np.array_split(arr, n_blocks)
    means = np.array([b.mean() for b in blocks if b.size])
    return {"lyap_per_time": float(np.nanmean(arr)), "lyap_se": float(means.std(ddof=1) / np.sqrt(len(means))),
            "n_intervals": int(arr.size), "renorm_time": interval, "dt_obs": proc.dt_obs}


def noise_free(proc_cls, params: dict) -> Process:
    return proc_cls(**{**params, "dyn_rel": 0.0, "obs_rel": 0.0})


def signal_stats(proc: Process, n: int = 20000, n_paths: int = 4, seed: int = 0) -> dict:
    """Mean and standard deviation of the observed noise-free signal after warmup."""
    rng = np.random.default_rng(seed)
    state = proc.init_state(rng, n_paths)
    w = proc.warmup_steps()
    if w > 0:
        _, state = proc.step(state, w, rng)
    obs, _ = proc.step(state, n, rng)
    return {"mean": float(obs.mean()), "sd": float(obs.std(axis=-1).mean())}


if __name__ == "__main__":
    import argparse
    import os
    import time

    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "lyapunov"])
    ap.add_argument("--out", default="results/map/chaos")
    ap.add_argument("--n-renorm", type=int, default=1000)
    ap.add_argument("--kuramoto-seeds", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from otp.grids import chaos as grid

    if a.cmd == "calibrate":
        refs = [(Lorenz63, {"rho": 28.0}), (Rossler, {"c": 5.7}), (MackeyGlass, {"tau": 17.0}),
                (LogisticMap, {"r": 3.9}), (Kuramoto, {"K": 4.0, "freq_seed": 0})]
        rows = []
        for cls, params in refs:
            st = signal_stats(noise_free(cls, params))
            rows.append({"generator": cls.name, "reference_params": str(params), **st,
                         "sd_ref_in_code": cls.sd_ref})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "calibration.csv"), index=False)
    else:
        rows = []
        t0 = time.time()
        for cid, cls, params in grid.param_list(full=True):
            is_kur = cls.name == "kuramoto"
            seeds = range(a.kuramoto_seeds) if is_kur else [0]
            for sd in seeds:
                pr = {**params, "freq_seed": sd} if is_kur else params
                proc = noise_free(cls, pr)
                res = lyapunov(proc, n_renorm=a.n_renorm, seed=sd)
                lam = res["lyap_per_time"]
                rows.append({"config_id": cid, "generator": cls.name, "knob_value": params[cls.knob],
                             "seed": sd, **res, "lyap_per_obs_step": lam * proc.dt_obs,
                             "lyap_time_obs_steps": (1.0 / (lam * proc.dt_obs)) if lam > 0 else float("nan")})
                print(f"{cid} seed {sd}: {lam:.4f} +- {res['lyap_se']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "lyapunov.csv"), index=False)
