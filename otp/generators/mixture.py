"""Learned failure generator: a mixture of primitives with evolvable weights and parameters.

A genome is a vector in [0, 1]^GENOME_DIM. It decodes into `N_SLOTS` component slots plus a
global observation-noise gene. Each slot picks one component from `COMPONENTS` (trend,
season and AR noise pieces, and the hand-made chaos / regime / heavy-tail / fractal
generators), a weight, a combination mode, and three parameter genes that map into that
component's own parameter ranges.

    y_t = (sum over "add" slots of w_i * z_i(t)) * prod over "modulate" slots of exp(0.5 * w_i * z_i(t))
          + noise_sd * e_t

where z_i is component i standardized by a fixed pilot estimate of its mean and standard
deviation (a deterministic function of its parameters, so the transform is part of the
process and the oracle sees the same thing). "Modulate" slots make multiplicative
interactions (e.g. a heavy-tailed volatility riding on a season).

Each component is warmed up separately inside `init_state`, so the mixture's first `step`
call is the context; regime components that fix their switch relative to the forecast
origin keep working inside a mixture.

API used by the search code: `GENOME_DIM`, `random_genome(rng)`, `decode(genome)`,
`describe(genome)`, `component_names()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from otp.generators.base import Process
from otp.generators.primitives import ARNoise, Season, Trend

N_SLOTS = 4
GENES_PER_SLOT = 6  # type, weight, mode, u1, u2, u3
GENOME_DIM = N_SLOTS * GENES_PER_SLOT + 1
MODULATE_AT = 0.75  # mode gene >= this -> multiplicative slot
PILOT_STEPS = 2048


def _lin(u, lo, hi):
    return float(lo + (hi - lo) * u)


def _log(u, lo, hi):
    return float(np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * u))


def _pick(u, options):
    return options[min(int(u * len(options)), len(options) - 1)]


@dataclass(frozen=True)
class Component:
    name: str
    family: str
    build: Callable[[float, float, float], Process]


def _chaos(cls_name: str, fn):
    def build(u1, u2, u3):
        from otp.generators import chaos

        return getattr(chaos, cls_name)(**fn(u1, u2, u3))
    return build


def _regime(cls_name: str, fn):
    def build(u1, u2, u3):
        from otp.generators import regime

        return getattr(regime, cls_name)(**fn(u1, u2, u3))
    return build


def _tails(cls_name: str, fn):
    def build(u1, u2, u3):
        from otp.generators import tails

        return getattr(tails, cls_name)(**fn(u1, u2, u3))
    return build


def _placement(u):
    return {"placement": "context", "d": int(round(_log(u / 0.8, 16, 2048)))} if u < 0.8 else {"placement": "horizon"}


COMPONENTS: list[Component] = [
    Component("trend", "primitive", lambda a, b, c: Trend(slope=_lin(a, -0.01, 0.01), slope_noise=_log(b, 1e-5, 1e-2),
                                                         phi=_lin(c, 0.95, 1.0))),
    Component("season", "primitive", lambda a, b, c: Season(period=_pick(a, [7, 12, 24, 48, 96, 168, 365]),
                                                           harmonics=_pick(b, [1, 2, 3, 4]), phase_seed=int(c * 1e6))),
    Component("ar_noise", "primitive", lambda a, b, c: ARNoise(rho=_lin(a, 0.0, 0.99), sigma=1.0)),
    Component("lorenz63", "chaos", _chaos("Lorenz63", lambda a, b, c: {
        "rho": _lin(a, 10.0, 50.0), "obs_every": _pick(b, [2, 5, 10, 20]), "dyn_rel": _log(c, 1e-3, 0.02)})),
    Component("rossler", "chaos", _chaos("Rossler", lambda a, b, c: {
        "c": _lin(a, 2.0, 9.0), "obs_every": _pick(b, [5, 10, 20, 40]), "dyn_rel": _log(c, 1e-3, 0.01)})),
    Component("mackey_glass", "chaos", _chaos("MackeyGlass", lambda a, b, c: {
        "tau": round(_lin(a, 6.0, 30.0) / 0.2) * 0.2, "obs_every": _pick(b, [2, 5, 10]), "dyn_rel": _log(c, 1e-3, 0.02)})),
    Component("logistic_map", "chaos", _chaos("LogisticMap", lambda a, b, c: {
        "r": _lin(a, 2.8, 4.0), "dyn_rel": _log(b, 1e-3, 0.02)})),
    Component("kuramoto", "chaos", _chaos("Kuramoto", lambda a, b, c: {
        "K": _lin(a, 0.0, 4.0), "freq_seed": int(b * 1e6), "dyn_rel": _log(c, 1e-3, 0.02)})),
    Component("mean_jump", "regime", _regime("MeanJump", lambda a, b, c: {
        "delta": _lin(a, -8.0, 8.0), **_placement(b), "period": _pick(c, [12, 24, 48, 96])})),
    Component("variance_jump", "regime", _regime("VarianceJump", lambda a, b, c: {
        "factor": _log(a, 0.2, 8.0), **_placement(b), "period": _pick(c, [12, 24, 48, 96])})),
    Component("frequency_jump", "regime", _regime("FrequencyJump", lambda a, b, c: {
        "ratio": _log(a, 0.33, 3.0), **_placement(b), "period": _pick(c, [12, 24, 48, 96])})),
    Component("markov_ar", "regime", _regime("MarkovSwitchingAR", lambda a, b, c: {
        "p_switch": _log(a, 1e-3, 0.1), "mean1": _lin(b, 0.5, 4.0), "rho1": _lin(c, 0.0, 0.98)})),
    Component("stable_ar", "tails", _tails("StableAR", lambda a, b, c: {
        "alpha": _lin(a, 1.1, 2.0), "rho": _lin(b, 0.5, 1.0)})),
    Component("fbm", "tails", _tails("FractionalBM", lambda a, b, c: {"H_hurst": _lin(a, 0.1, 0.9)})),
    Component("hawkes", "tails", _tails("Hawkes", lambda a, b, c: {
        "eta": _lin(a, 0.0, 0.95), "tau": _log(b, 1.0, 20.0)})),
    Component("msm_cascade", "tails", _tails("MSMCascade", lambda a, b, c: {
        "k": _pick(a, [1, 2, 3, 4, 5, 6, 7, 8]), "m0": _lin(b, 1.1, 1.9),
        "observe": "returns" if c < 0.5 else "cumsum"})),
]


def component_names() -> list[str]:
    return [c.name for c in COMPONENTS]


@lru_cache(maxsize=4096)
def _pilot_stats(name: str, genes: tuple[float, float, float]) -> tuple[float, float]:
    """Mean and std of a component over PILOT_STEPS after its warmup, fixed seed."""
    comp = next(c for c in COMPONENTS if c.name == name)
    proc = comp.build(*genes)
    rng = np.random.default_rng(12345)
    st = proc.init_state(rng, 1)
    w = proc.warmup_steps()
    if w:
        _, st = proc.step(st, w, rng)
    y, _ = proc.step(st, PILOT_STEPS, rng)
    y = y[0, list(proc.main_rows)[0]]
    mu, sd = float(np.mean(y)), float(np.std(y))
    if not (np.isfinite(mu) and np.isfinite(sd)):
        return 0.0, 1.0  # the component diverged in the pilot; samples will be non-finite and fail feasibility
    return mu, sd if sd > 1e-9 else 1.0


@dataclass
class Slot:
    component: Component
    weight: float
    modulate: bool
    genes: tuple[float, float, float]


def _slots(genome: np.ndarray) -> tuple[list[Slot], float]:
    g = np.clip(np.asarray(genome, float), 0.0, 1.0)
    assert g.shape == (GENOME_DIM,), f"genome must have {GENOME_DIM} entries, got {g.shape}"
    comps = COMPONENTS  # fixed list: a genome means the same thing everywhere
    slots = []
    for i in range(N_SLOTS):
        s = g[i * GENES_PER_SLOT:(i + 1) * GENES_PER_SLOT]
        comp = _pick(s[0], comps)
        weight = 2.0 * float(s[1]) ** 2
        slots.append(Slot(comp, weight, bool(s[2] >= MODULATE_AT), (float(s[3]), float(s[4]), float(s[5]))))
    noise_sd = 10 ** (-3 + 2.5 * float(g[-1]))
    return slots, noise_sd


class MixtureProcess(Process):
    family, name = "mixture", "mixture"
    defaults = {}

    def __init__(self, genome: np.ndarray):
        super().__init__()
        self.genome = np.clip(np.asarray(genome, float), 0.0, 1.0)
        self.slots, self.noise_sd = _slots(self.genome)
        self.procs = [s.component.build(*s.genes) for s in self.slots]
        self.stats = [_pilot_stats(s.component.name, s.genes) for s in self.slots]
        self.params = {"genome": [round(float(x), 6) for x in self.genome]}

    @property
    def period(self) -> int:
        best, best_w = 1, 0.1
        for s, p in zip(self.slots, self.procs):
            if not s.modulate and s.weight > best_w and p.period > 1:
                best, best_w = p.period, s.weight
        return best

    def init_state(self, rng, n_paths):
        st = {}
        for i, p in enumerate(self.procs):
            s = p.init_state(rng, n_paths)
            w = p.warmup_steps()
            if w:
                _, s = p.step(s, w, rng)
            st[f"c{i}"] = s
        return st

    def step(self, state, n, rng, exo_rng=None):
        new = {}
        add = None
        mod = None
        for i, (slot, p, (mu, sd)) in enumerate(zip(self.slots, self.procs, self.stats)):
            y, new[f"c{i}"] = p.step(state[f"c{i}"], n, rng, exo_rng)
            z = (y[:, list(p.main_rows)[0]] - mu) / sd  # (P, n)
            if slot.modulate:
                m = np.exp(np.clip(0.5 * slot.weight * z, -20, 20))
                mod = m if mod is None else mod * m
            else:
                add = slot.weight * z if add is None else add + slot.weight * z
        if add is None:  # every slot modulates: modulate a unit constant
            add = np.ones_like(mod)
        out = add if mod is None else add * mod
        out = out + self.noise_sd * rng.standard_normal(out.shape)
        return out[:, None, :], new

    def tags(self):
        names = [s.component.name for s in self.slots if s.weight > 0.05]
        return ["mixture"] + sorted(set(names))


def decode(genome: np.ndarray) -> MixtureProcess:
    return MixtureProcess(genome)


def random_genome(rng: np.random.Generator) -> np.ndarray:
    return rng.random(GENOME_DIM)


def describe(genome: np.ndarray) -> dict:
    slots, noise_sd = _slots(genome)
    out = []
    for s in slots:
        p = s.component.build(*s.genes)
        out.append({"component": s.component.name, "family": s.component.family, "weight": round(s.weight, 4),
                    "mode": "modulate" if s.modulate else "add",
                    "params": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.params.items()}})
    return {"slots": out, "noise_sd": round(noise_sd, 6)}
