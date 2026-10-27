"""Differentiable re-implementation of a subset of the mixture-of-primitives generator.

Supported components (the same equations as `otp.generators.mixture`, with the same genome
meaning): `trend`, `season`, `ar_noise`, and two regime pieces with a smooth (sigmoid) switch,
`mean_jump` and `variance_jump`. A genome whose four slots all use these components can be
optimized by gradient and then decoded by `mixture.decode` for scoring with the standard
pipeline, because the parameters ARE the genome genes.

What is differentiable (a gradient flows from the series to the gene):
  slot weights, the observation-noise gene, trend slope / slope noise / damping, the AR
  coefficient, the jump size / variance factor, and the switch position (through a sigmoid
  gate whose width `tau` is annealed by the caller; the standard pipeline uses a hard switch).
What is held fixed per optimization start (not differentiable): which component each slot
uses, add vs modulate, the season period / harmonics / phase seed, the regime base period,
and whether the switch lies in the context or in the horizon.

Noise is reparameterized: all Gaussian / uniform draws are made once at construction and are
constants afterwards, so the series is a deterministic differentiable function of the genes.
The standardization of each component by a pilot run (mean and std over 2048 steps, seed
12345) is reproduced with the same random stream as the numpy generator and is differentiated
too. Linear recursions (AR noise, trend slope) are computed as a product with a lower-triangular
matrix of powers, so there is no time loop.

Oracle: `forward` returns the context (B, T) and `1 + K` futures (B, 1 + K, H) that continue
from the SAME end-of-context state with independent noise. Future 0 is the realized
future; futures 1..K are the Monte Carlo continuation whose quantiles are the oracle forecast.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm

from otp.generators import mixture
from otp.generators.primitives import Season
from otp.generators.regime import _season_coefs
from otp.model import QUANTILE_LEVELS

SUPPORTED = ("trend", "season", "ar_noise", "mean_jump", "variance_jump")
N_SLOTS, GPS, GENOME_DIM = mixture.N_SLOTS, mixture.GENES_PER_SLOT, mixture.GENOME_DIM
PILOT = mixture.PILOT_STEPS
_COMP_INDEX = {c.name: i for i, c in enumerate(mixture.COMPONENTS)}
SEASON_PERIODS = [7, 12, 24, 48, 96, 168, 365]
REGIME_PERIODS = [12, 24, 48, 96]
REG_RHO, REG_SIGMA, REG_HAZARD = 0.5, 0.25, 1.0 / 256.0


# ---------------------------------------------------------------- genome helpers
def slot_types(genome) -> list[str]:
    g = np.clip(np.asarray(genome, float), 0, 1)
    return [mixture._pick(g[i * GPS], mixture.COMPONENTS).name for i in range(N_SLOTS)]


def is_supported(genome) -> bool:
    return all(t in SUPPORTED for t in slot_types(genome))


def restricted_genome(rng: np.random.Generator, types=None) -> np.ndarray:
    """Random genome whose slots all use supported components (centered in each type's gene interval)."""
    g = rng.random(GENOME_DIM)
    if types is None:
        types = [SUPPORTED[int(i)] for i in rng.integers(0, len(SUPPORTED), N_SLOTS)]
    for i, t in enumerate(types):
        g[i * GPS] = (_COMP_INDEX[t] + 0.5) / len(mixture.COMPONENTS)
    return g


def gene_bounds(genome) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lo, hi, free): box and which genes the gradient may move for this genome's slot types."""
    g = np.clip(np.asarray(genome, float), 0, 1)
    lo, hi = np.zeros(GENOME_DIM), np.ones(GENOME_DIM)
    free = np.zeros(GENOME_DIM, bool)
    for i, t in enumerate(slot_types(g)):
        b = i * GPS
        free[b + 1] = True                      # weight
        if t == "trend":
            free[b + 3:b + 6] = True
        elif t == "ar_noise":
            free[b + 3] = True
        elif t in ("mean_jump", "variance_jump"):
            free[b + 3] = True
            if g[b + 4] < 0.8:                  # switch inside the context: its position is a gene
                free[b + 4] = True
                hi[b + 4] = 0.7999
    free[-1] = True                              # observation noise
    return lo, hi, free


def period_of(genome) -> int:
    """Seasonal period the mixture reports (scales MASE / drives seasonal naive); same rule as MixtureProcess."""
    g = np.clip(np.asarray(genome, float), 0, 1)
    best, best_w = 1, 0.1
    for i, t in enumerate(slot_types(g)):
        b = i * GPS
        w, mod = 2.0 * g[b + 1] ** 2, g[b + 2] >= mixture.MODULATE_AT
        per = 1
        if t == "season":
            per = mixture._pick(g[b + 3], SEASON_PERIODS)
        elif t in ("mean_jump", "variance_jump"):
            per = mixture._pick(g[b + 5], REGIME_PERIODS)
        if not mod and w > best_w and per > 1:
            best, best_w = per, w
    return int(best)


# ---------------------------------------------------------------- linear recursion
_LAG: dict = {}


def _lag(n: int, device):
    key = (n, str(device))
    if key not in _LAG:
        t = torch.arange(n, device=device)
        lag = t[:, None] - t[None, :]
        _LAG[key] = (lag.clamp_min(0).to(torch.float64), (lag >= 0).to(torch.float64), (t + 1).to(torch.float64))
    return _LAG[key]


def ar1(rho: torch.Tensor, inp: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """x_t = rho x_{t-1} + inp_t with x_{-1} = x0, along the last axis. `x0` broadcasts against inp[..., 0]."""
    n = inp.shape[-1]
    lag, mask, t1 = _lag(n, inp.device)
    logr = torch.log(rho.clamp_min(1e-6))
    w = torch.exp(lag * logr) * mask
    return torch.matmul(inp, w.T) + x0[..., None] * torch.exp(t1 * logr)


def _lin(u, lo, hi):
    return lo + (hi - lo) * u


def _log(u, lo, hi):
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * u)


def _pilot_noise(kind: str, horizon_switch: bool) -> dict:
    """The random numbers `mixture._pilot_stats` uses for this component kind (seed 12345, same order)."""
    rng = np.random.default_rng(12345)
    if kind == "trend":
        return {"e": rng.standard_normal(PILOT)}
    u0 = float(rng.standard_normal(1)[0])
    if kind in ("mean_jump", "variance_jump") and horizon_switch:
        rng.random(PILOT)   # the hazard draws, consumed before the innovations
    return {"u0": u0, "e": rng.standard_normal(PILOT)}


class TorchMixture:
    """Fixed-noise differentiable mixture generator for a given slot layout (see module docstring)."""

    def __init__(self, genome, B: int = 2, K: int = 64, T: int = 2048, H: int = 128, seed: int = 0,
                 device: str = "cpu"):
        g = np.clip(np.asarray(genome, float), 0, 1)
        assert is_supported(g), f"unsupported component in {slot_types(g)}"
        self.B, self.K, self.T, self.H, self.P = B, K, T, H, K + 1
        self.device = device
        self.types = slot_types(g)
        self.modulate = [bool(g[i * GPS + 2] >= mixture.MODULATE_AT) for i in range(N_SLOTS)]
        self.horizon_switch = [t in ("mean_jump", "variance_jump") and g[i * GPS + 4] >= 0.8
                               for i, t in enumerate(self.types)]
        self.fixed: list[dict] = []
        tg = torch.Generator().manual_seed(seed)
        kw = dict(dtype=torch.float64, device=device)

        def randn(*s):
            return torch.randn(*s, generator=tg, dtype=torch.float64).to(device)

        def rand(*s):
            return torch.rand(*s, generator=tg, dtype=torch.float64).to(device)

        self.noise = []
        for i, t in enumerate(self.types):
            self.noise.append({"u0": randn(B), "e_c": randn(B, T), "e_f": randn(B, self.P, H),
                               "hz": rand(B, self.P, H)})
            pn = _pilot_noise(t, self.horizon_switch[i])
            self.noise[-1]["pilot_e"] = torch.tensor(pn["e"], **kw)
            self.noise[-1]["pilot_u0"] = torch.tensor(pn.get("u0", 0.0), **kw)
            b = i * GPS
            fx: dict = {}
            if t == "season":
                per = mixture._pick(g[b + 3], SEASON_PERIODS)
                har = mixture._pick(g[b + 4], [1, 2, 3, 4])
                s = Season(period=per, harmonics=har, phase_seed=int(g[b + 5] * 1e6))
                s.init_state(np.random.default_rng(0), 1)
                fx = {"period": float(per), "amps": torch.tensor(s._amps, **kw), "phases": torch.tensor(s._phases, **kw)}
            elif t in ("mean_jump", "variance_jump"):
                per = mixture._pick(g[b + 5], REGIME_PERIODS)
                amps, phases = _season_coefs(1.0, 2, 0)
                fx = {"period": float(per), "amps": torch.tensor(amps, **kw), "phases": torch.tensor(phases, **kw)}
            self.fixed.append(fx)
        self.obs_c = randn(B, T)
        self.obs_f = randn(B, self.P, H)

    # -- components: each returns (context (B, T), future (B, P, H), pilot (PILOT,)) -----------
    def _season_y(self, fx, t, offset: float):
        ph = (t + offset) * 2 * math.pi / fx["period"]
        y = 0.0
        for j in range(len(fx["amps"])):
            y = y + fx["amps"][j] * torch.sin((j + 1) * ph + fx["phases"][j])
        return y

    def _slot(self, i: int, g: torch.Tensor, tau: float):
        t, nz, fx, b = self.types[i], self.noise[i], self.fixed[i], i * GPS
        B, T, H, P = self.B, self.T, self.H, self.P
        dev = self.device
        u1, u2, u3 = g[b + 3], g[b + 4], g[b + 5]
        if t == "season":
            tc = torch.arange(T, dtype=torch.float64, device=dev)
            tf = torch.arange(T, T + H, dtype=torch.float64, device=dev)
            tp = torch.arange(PILOT, dtype=torch.float64, device=dev)
            yc = self._season_y(fx, tc, 0.0)[None].expand(B, T)
            yf = self._season_y(fx, tf, 0.0)[None, None].expand(B, P, H)
            return yc, yf, self._season_y(fx, tp, 0.0)
        if t == "trend":
            slope, sn, phi = _lin(u1, -0.01, 0.01), _log(u2, 1e-5, 1e-2), _lin(u3, 0.95, 1.0)
            sl_c = ar1(phi, sn * nz["e_c"], slope.expand(B))
            lv_c = torch.cumsum(sl_c, -1)
            sl_f = ar1(phi, sn * nz["e_f"], sl_c[:, -1:])
            lv_f = lv_c[:, -1][:, None, None] + torch.cumsum(sl_f, -1)
            sl_p = ar1(phi, sn * nz["pilot_e"][None], slope[None])
            return lv_c, lv_f, torch.cumsum(sl_p, -1)[0]
        if t == "ar_noise":
            rho = _lin(u1, 0.0, 0.99)
            sd = 1.0 / torch.sqrt(1.0 - rho ** 2)
            xc = ar1(rho, nz["e_c"], nz["u0"] * sd)
            xf = ar1(rho, nz["e_f"], xc[:, -1:])
            xp = ar1(rho, nz["pilot_e"][None], (nz["pilot_u0"] * sd)[None])
            return xc, xf, xp[0]
        # regime pieces
        rho = torch.tensor(REG_RHO, dtype=torch.float64, device=dev)
        sd0 = REG_SIGMA / math.sqrt(1.0 - REG_RHO ** 2)
        tc = torch.arange(T, dtype=torch.float64, device=dev)
        tf = torch.arange(T, T + H, dtype=torch.float64, device=dev)
        tp = torch.arange(PILOT, dtype=torch.float64, device=dev)
        if self.horizon_switch[i]:
            gate_c = torch.zeros(B, T, dtype=torch.float64, device=dev)
            sw = (nz["hz"] < REG_HAZARD).to(torch.float64)
            gate_f = torch.cummax(sw, dim=-1).values
            gate_p = torch.zeros(PILOT, dtype=torch.float64, device=dev)
        else:
            d = torch.exp(math.log(16) + (math.log(2048) - math.log(16)) * (u2 / 0.8))
            gate_c = torch.sigmoid((tc - (T - d)) / tau)[None].expand(B, T)
            gate_f = torch.sigmoid((tf - (T - d)) / tau)[None, None].expand(B, P, H)
            gate_p = torch.sigmoid((tp - (PILOT - d)) / tau)
        if t == "variance_jump":
            factor = _log(u1, 0.2, 8.0)
            sig_c, sig_f, sig_p = (REG_SIGMA * (1 + (factor - 1) * x) for x in (gate_c, gate_f, gate_p))
            lev = (0.0, 0.0, 0.0)
        else:
            delta = _lin(u1, -8.0, 8.0)
            sig_c, sig_f, sig_p = (REG_SIGMA * torch.ones_like(x) for x in (gate_c, gate_f, gate_p))
            lev = (delta * sd0 * gate_c, delta * sd0 * gate_f, delta * sd0 * gate_p)
        k = 1.0 / math.sqrt(1.0 - REG_RHO ** 2)
        xc = ar1(rho, sig_c * nz["e_c"], nz["u0"] * sig_c[:, 0] * k)
        xf = ar1(rho, sig_f * nz["e_f"], xc[:, -1:])
        xp = ar1(rho, (sig_p * nz["pilot_e"])[None], (nz["pilot_u0"] * sig_p[0] * k)[None])[0]
        yc = lev[0] + self._season_y(fx, tc, 1.0)[None] + xc
        yf = lev[1] + self._season_y(fx, tf, 1.0)[None, None] + xf
        yp = lev[2] + self._season_y(fx, tp, 1.0) + xp
        return yc, yf, yp

    def forward(self, genome: torch.Tensor, tau: float = 1.0):
        """Series for the genome tensor `(GENOME_DIM,)`: context (B, T), futures (B, 1 + K, H)."""
        g = genome.to(torch.float64)
        add_c = add_f = mod_c = mod_f = None
        for i in range(N_SLOTS):
            yc, yf, yp = self._slot(i, g, tau)
            mu, sd = yp.mean(), yp.std(unbiased=False).clamp_min(1e-9)
            w = 2.0 * g[i * GPS + 1] ** 2
            zc, zf = (yc - mu) / sd, (yf - mu) / sd
            if self.modulate[i]:
                mc, mf = torch.exp(torch.clamp(0.5 * w * zc, -20, 20)), torch.exp(torch.clamp(0.5 * w * zf, -20, 20))
                mod_c = mc if mod_c is None else mod_c * mc
                mod_f = mf if mod_f is None else mod_f * mf
            else:
                add_c = w * zc if add_c is None else add_c + w * zc
                add_f = w * zf if add_f is None else add_f + w * zf
        if add_c is None:
            add_c, add_f = torch.ones_like(mod_c), torch.ones_like(mod_f)
        out_c = add_c if mod_c is None else add_c * mod_c
        out_f = add_f if mod_f is None else add_f * mod_f
        noise_sd = 10 ** (-3 + 2.5 * g[-1])
        return out_c + noise_sd * self.obs_c, out_f + noise_sd * self.obs_f


# ---------------------------------------------------------------- scoring pieces (torch)
LEVELS = torch.tensor(QUANTILE_LEVELS, dtype=torch.float64)
Z = torch.tensor(norm.ppf(np.asarray(QUANTILE_LEVELS)), dtype=torch.float64)


def pinball(y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss over horizon and levels. y (B, H), q (B, H, 21) -> (B,)."""
    lv = LEVELS.to(q.device, q.dtype)
    diff = y[..., None] - q
    return torch.maximum(lv * diff, (lv - 1.0) * diff).mean(dim=(-1, -2))


def oracle_quantiles(fut: torch.Tensor) -> torch.Tensor:
    """Quantiles (B, H, 21) of the Monte Carlo continuation futures 1..K of `fut` (B, 1 + K, H)."""
    q = torch.quantile(fut[:, 1:], LEVELS.to(fut.device, fut.dtype), dim=1)   # (21, B, H)
    return q.permute(1, 2, 0)


def seasonal_naive_quantiles(ctx: torch.Tensor, H: int, m: int) -> torch.Tensor:
    """Torch version of `otp.classical.seasonal_naive` for a batch of contexts (B, T) -> (B, H, 21)."""
    T = ctx.shape[-1]
    if m < 1 or m >= T // 2:
        m = 1
    last = ctx[:, -m:]
    reps = int(math.ceil(H / m))
    mean = last.repeat(1, reps)[:, :H]
    resid = ctx[:, m:] - ctx[:, :-m]
    sigma = resid.std(dim=-1, unbiased=False)
    k = torch.arange(H, device=ctx.device) // m + 1
    sd = sigma[:, None] * torch.sqrt(k.to(ctx.dtype))[None]
    sd = sd.clamp_min(1e-9)
    return mean[..., None] + sd[..., None] * Z.to(ctx.device, ctx.dtype)


def naive_scale(ctx: torch.Tensor, m: int) -> torch.Tensor:
    T = ctx.shape[-1]
    if m < 1 or m >= T // 2:
        m = 1
    return (ctx[:, m:] - ctx[:, :-m]).abs().mean(dim=-1).clamp_min(1e-8)
