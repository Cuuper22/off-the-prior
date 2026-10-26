"""Thin wrapper around the released Timestral forecaster.

Everything in this project calls Timestral through this module, so every experiment uses
the same settings. Defaults match the shipped forecaster (symmetric averaging on,
non-negative clipping on, full context up to 32,768 steps).

Two entry points:

- `forecast(size, context, horizon, ...)`: NumPy in, NumPy `(B, V, H, 21)` quantiles out.
- `forecast_torch(size, x, horizon)`: a differentiable single-block forecast for gradient
  search in input space. It goes through the same `predict_block` + refine-frame code the
  forecaster uses, minus `torch.inference_mode`. Note: Timestral computes its per-patch
  normalization statistics under `no_grad`, so gradients flow through the normalized values
  but not through the location/scale statistics themselves.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np
import torch

SIZES = ("5M", "28M", "123M", "209M")
QUANTILE_LEVELS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
MEDIAN_INDEX = QUANTILE_LEVELS.index(0.5)
PATCH = 32
BLOCK_STEPS = 1024
MAX_CONTEXT = 32768


def set_threads(n: int | None = None) -> None:
    """Pin torch CPU threads. Defaults to env OTP_THREADS, else 1 (safe when several jobs share the box)."""
    n = n or int(os.environ.get("OTP_THREADS", "1"))
    torch.set_num_threads(n)


def repo_id(size: str) -> str:
    """Hub id, or a local directory when `size` is a path (used for fine-tuned checkpoints)."""
    if os.path.isdir(size):
        return size
    assert size in SIZES, f"unknown size {size!r}; expected one of {SIZES} or a local checkpoint dir"
    return f"mistralai/Timestral-{size}"


@lru_cache(maxsize=4)
def load(size: str, device: str = "cpu"):
    from timestral import TimestralForecaster

    fc = TimestralForecaster.from_pretrained(repo_id(size), device=device)
    for p in fc.model.parameters():
        p.requires_grad_(False)
    return fc


def forecast(
    size: str,
    context: np.ndarray,
    horizon: int,
    past_covariates: np.ndarray | None = None,
    known_covariates: np.ndarray | None = None,
    max_context: int | None = None,
    batch_size: int = 32,
    symmetric_averaging: bool = True,
    clip_nonnegative: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """Quantile forecast `(B, V, H, 21)` for `context` `(B, V, T)` or `(B, T)`.

    `max_context` truncates the context to its most recent steps before the call (the
    context-length knob). Covariates follow the Timestral shapes: past `(B, C, T)`, known
    `(B, C, T + H)`; they are truncated consistently with the context.
    """
    fc = load(size, device)
    x = np.asarray(context, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, None]
    assert x.ndim == 3, f"context must be (B, V, T) or (B, T), got {x.shape}"
    t = x.shape[-1]
    if max_context is not None and t > max_context:
        x = x[..., -max_context:]
        if past_covariates is not None:
            past_covariates = np.asarray(past_covariates)[..., -max_context:]
        if known_covariates is not None:
            known_covariates = np.asarray(known_covariates)[..., -(max_context + horizon):]
    out = []
    for i in range(0, x.shape[0], batch_size):
        sl = slice(i, i + batch_size)
        kw = {}
        if past_covariates is not None:
            kw["past_covariates"] = np.asarray(past_covariates, dtype=np.float32)[sl]
        if known_covariates is not None:
            kw["known_covariates"] = np.asarray(known_covariates, dtype=np.float32)[sl]
        f = fc.forecast(x[sl], prediction_length=horizon, symmetric_averaging=symmetric_averaging,
                        clip_nonnegative=clip_nonnegative, **kw)
        out.append(f.quantiles.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def forecast_torch(size: str, x: torch.Tensor, horizon: int, symmetric_averaging: bool = True) -> torch.Tensor:
    """Differentiable forecast `(B, H, 21)` for univariate `x` `(B, T)`, `H <= 1024` (one decode block).

    No covariates, no non-negative clipping, no extreme-value cap (those are piecewise and
    only matter at the edges). Gradients flow into `x`.
    """
    assert horizon <= BLOCK_STEPS, "forecast_torch covers one decode block (<= 1024 steps)"
    fc = load(size)

    def one_pass(xx: torch.Tensor) -> torch.Tensor:
        b, t = xx.shape
        pad = (-t) % PATCH
        mask = torch.ones(b, t + pad, dtype=torch.bool, device=xx.device)
        if pad:
            xx = torch.nn.functional.pad(xx, (pad, 0))
            mask[:, :pad] = False
        gid = torch.arange(b, device=xx.device)
        q = fc.model.predict_block(xx, mask, gid, math.ceil(horizon / PATCH))
        q = fc._refine_block(q, xx, mask, None)
        return q[:, :horizon]

    q = one_pass(x)
    if symmetric_averaging:
        q = 0.5 * (q + (-one_pass(-x)).flip(-1))
    return q


def median(q: np.ndarray) -> np.ndarray:
    return q[..., MEDIAN_INDEX]
