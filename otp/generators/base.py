"""Generator interface: batched simulators with a Monte Carlo true-dynamics oracle.

Every signal family is a `Process`: a simulator whose state carries a leading "path"
axis, so one call can advance many independent paths. `make_sample` uses that to:

1. simulate a warmup and the context from one initial state (path axis = 1),
2. copy the final state into `n_paths + 1` paths,
3. advance all of them over the horizon with fresh noise: path 0 is the realized future,
   paths 1..n are the oracle's ensemble.

The oracle therefore knows the true equations, the true parameters and the true latent
state at the forecast origin. Its quantiles are the best achievable forecast given that
state (up to Monte Carlo error); this is slightly stronger than a forecaster that only sees
noisy observations, which is the right side to err on for "best possible".

Deterministic dynamics with no noise would give the oracle zero error, so chaotic
families should include dynamic noise and/or observation noise (as knobs).

Channels: `step` returns all observed channels `(P, C, n)`. `main_rows` are the series to
forecast; `past_cov_rows` / `known_cov_rows` are offered as Timestral covariates.
Exogenous drivers that must be identical across oracle paths (known-future covariates)
should draw noise with `shared_normal(exo_rng, ...)`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from otp.model import QUANTILE_LEVELS

LEVELS = np.asarray(QUANTILE_LEVELS)


def shared_normal(rng: np.random.Generator, n_paths: int, shape: tuple[int, ...]) -> np.ndarray:
    """Standard normals identical across the path axis: `(n_paths, *shape)`."""
    z = rng.standard_normal((1, *shape))
    return np.broadcast_to(z, (n_paths, *shape)).copy()


class Process:
    """Base simulator. Subclasses set `family`, `name`, `defaults`, `knob` and implement
    `init_state` and `step`."""

    family: str = "base"
    name: str = "base"
    defaults: dict[str, Any] = {}
    knob: str | None = None  # the main difficulty parameter
    n_channels: int = 1
    main_rows: tuple[int, ...] = (0,)
    past_cov_rows: tuple[int, ...] = ()
    known_cov_rows: tuple[int, ...] = ()

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.defaults)
        assert not unknown, f"{self.name}: unknown params {sorted(unknown)}"
        self.params = {**self.defaults, **params}

    @property
    def period(self) -> int:
        """Seasonal period for MASE scaling and classical models (1 = none)."""
        return int(self.params.get("period", 1) or 1)

    def warmup_steps(self) -> int:
        return 0

    def init_state(self, rng: np.random.Generator, n_paths: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def step(self, state: dict, n: int, rng: np.random.Generator,
             exo_rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
        """Advance `n` steps. Returns observations `(P, C, n)` and the new state."""
        raise NotImplementedError

    def tags(self) -> list[str]:
        """Short reasons this instance may be hard (used by the diagnostic benchmark)."""
        return []


def tile_state(state: dict, n: int) -> dict:
    out = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = tile_state(v, n)
        elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == 1:
            out[k] = np.repeat(v, n, axis=0)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Sample:
    context: np.ndarray            # (V, T) main series
    future: np.ndarray             # (V, H) realized continuation
    oracle_q: np.ndarray           # (V, H, 21) true-dynamics oracle quantiles
    period: int
    meta: dict = field(default_factory=dict)
    past_cov: np.ndarray | None = None    # (C, T)
    known_cov: np.ndarray | None = None   # (C, T + H)

    @property
    def T(self) -> int:
        return self.context.shape[-1]

    @property
    def H(self) -> int:
        return self.future.shape[-1]


def make_sample(proc: Process, context_len: int, horizon: int, seed: int, n_paths: int = 500,
                sample_id: str | None = None) -> Sample:
    rng = np.random.default_rng(seed)
    exo_seed = int(rng.integers(2**63 - 1))
    exo_rng = np.random.default_rng(exo_seed)
    state = proc.init_state(rng, 1)
    w = proc.warmup_steps()
    if w > 0:
        _, state = proc.step(state, w, rng, exo_rng)
    ctx, state = proc.step(state, context_len, rng, exo_rng)  # (1, C, T)
    big = tile_state(state, n_paths + 1)
    fut, _ = proc.step(big, horizon, rng, np.random.default_rng(exo_seed + 1))  # (P+1, C, H)
    main = list(proc.main_rows)
    paths = fut[1:, main]
    oracle_q = np.quantile(paths, LEVELS, axis=0)          # (21, V, H)
    oracle_q = np.moveaxis(oracle_q, 0, -1)                # (V, H, 21)
    past_cov = ctx[0, list(proc.past_cov_rows)] if proc.past_cov_rows else None
    known_cov = None
    if proc.known_cov_rows:
        kr = list(proc.known_cov_rows)
        known_cov = np.concatenate([ctx[0, kr], fut[0, kr]], axis=-1)
    meta = {
        "family": proc.family,
        "generator": proc.name,
        "knob": proc.knob,
        "knob_value": proc.params.get(proc.knob) if proc.knob else None,
        "params": dict(proc.params),
        "seed": seed,
        "T": context_len,
        "H": horizon,
        "tags": proc.tags(),
        "sample_id": sample_id or f"{proc.name}-{seed}",
    }
    return Sample(ctx[0, main].astype(np.float64), fut[0, main].astype(np.float64), oracle_q, proc.period,
                  meta, past_cov, known_cov)


def save_samples(path: str, samples: list[Sample]) -> None:
    """Stack equal-shape samples into one .npz (context, future, oracle_q, covariates) plus a JSON meta column."""
    import json

    enc = json.JSONEncoder(default=lambda o: o.item() if hasattr(o, "item") else str(o))
    arrays = {
        "context": np.stack([s.context for s in samples]),
        "future": np.stack([s.future for s in samples]),
        "oracle_q": np.stack([s.oracle_q for s in samples]),
        "period": np.asarray([s.period for s in samples]),
        "meta": np.asarray([enc.encode(s.meta) for s in samples]),
    }
    if samples[0].past_cov is not None:
        arrays["past_cov"] = np.stack([s.past_cov for s in samples])
    if samples[0].known_cov is not None:
        arrays["known_cov"] = np.stack([s.known_cov for s in samples])
    np.savez_compressed(path, **arrays)


def load_samples(path: str) -> list[Sample]:
    import json

    with np.load(path, allow_pickle=False) as z:
        # read every array once (indexing z[...] repeatedly would decompress a fresh copy each time)
        arr = {k: z[k] for k in z.files}
    n = arr["context"].shape[0]
    pc, kc = arr.get("past_cov"), arr.get("known_cov")
    return [
        Sample(arr["context"][i].copy(), arr["future"][i].copy(), arr["oracle_q"][i].copy(), int(arr["period"][i]),
               json.loads(str(arr["meta"][i])), None if pc is None else pc[i].copy(), None if kc is None else kc[i].copy())
        for i in range(n)
    ]
