"""Fine-tuning library for Timestral: full fine-tune, LoRA, and data mixing.

- `load_trainable(size, device)`: a fresh, trainable copy of the model plus a forecaster
  object sharing it (for the refine-frame step), independent of `otp.model.load`'s cache.
- `add_lora(model, rank, alpha, layers)`: wraps the attention (`wQKV`, `wO`) and MLP (`w12`,
  `w3`) linears with low-rank adapters; base weights frozen. `merge_lora` folds them back.
- `forecast_train(fc, x, horizon)`: the training forward pass, one decode block, no symmetric
  averaging: pad, `predict_block`, refine frame (same path as `otp.model.forecast_torch`).
- `pinball_loss(q, y, scale)`: pinball loss over the 21 levels, per series divided by the
  naive scale of its context (the MASE scale), averaged.
- `Sampler`: draws (context, future) pairs on the fly from a list of process factories, with a
  `replay_frac` share of in-prior series; context lengths and horizons are randomized.
- `train(...)`: AdamW + warmup/cosine schedule + gradient clipping; returns a cost record.
- `export(fc, path)`: writes `config.json` + `model.safetensors` (LoRA merged), loadable by
  `otp.model.forecast(size=path, ...)`.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn

from otp.model import PATCH, QUANTILE_LEVELS, repo_id

LEVELS_T = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32)
LORA_LAYERS = ("wQKV", "wO", "w12", "w3")


def load_trainable(size: str, device: str = "cpu"):
    from timestral import TimestralForecaster
    from timestral.loading import load_model

    model = load_model(repo_id(size), device=device)
    model.train()
    fc = TimestralForecaster(model)
    return fc


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    @property
    def weight(self):  # timestral reads `.weight.dtype` in places
        return self.base.weight

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def add_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0, layers=LORA_LAYERS) -> int:
    """Freeze everything, wrap the listed linears with LoRA. Returns the number of trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in layers and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def merge_lora(model: nn.Module) -> None:
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight += (child.B @ child.A) * child.scale
                setattr(module, name, child.base)


def forecast_train(fc, x: torch.Tensor, horizon: int) -> torch.Tensor:
    """Training forward `(B, H, 21)` for `x` `(B, T)`; `horizon` a multiple of 32, <= 1024."""
    b, t = x.shape
    pad = (-t) % PATCH
    mask = torch.ones(b, t + pad, dtype=torch.bool, device=x.device)
    if pad:
        x = torch.nn.functional.pad(x, (pad, 0))
        mask[:, :pad] = False
    gid = torch.arange(b, device=x.device)
    q = fc.model.predict_block(x, mask, gid, math.ceil(horizon / PATCH))
    q = fc._refine_block(q, x, mask, None)
    return q[:, :horizon]


def naive_scale_t(x: torch.Tensor) -> torch.Tensor:
    return (x[:, 1:] - x[:, :-1]).abs().mean(-1).clamp_min(1e-6)


def pinball_loss(q: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    lv = LEVELS_T.to(q.device)
    diff = y[..., None] - q
    pl = torch.maximum(lv * diff, (lv - 1.0) * diff)  # (B, H, 21)
    return (pl.mean(dim=(-1, -2)) / scale).mean() * 2.0


@dataclass
class Sampler:
    """On-the-fly training pairs. `factories`: list of callables seed -> Process (the failure mix).

    Seeds are drawn from [seed_lo, seed_hi); evaluation must use seeds outside that range.
    """

    factories: list[Callable]
    replay_frac: float = 0.0
    contexts: tuple[int, ...] = (256, 512, 1024, 2048)
    horizons: tuple[int, ...] = (32, 128, 512)
    seed_lo: int = 1_000_000
    seed_hi: int = 2_000_000
    rng_seed: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)

    def batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        from otp.generators.base import make_sample
        from otp.generators.primitives import in_prior

        t = int(self.rng.choice(self.contexts))
        h = int(self.rng.choice(self.horizons))
        xs, ys = [], []
        for _ in range(batch_size):
            seed = int(self.rng.integers(self.seed_lo, self.seed_hi))
            if self.factories and self.rng.random() >= self.replay_frac:
                proc = self.factories[int(self.rng.integers(len(self.factories)))](seed)
            else:
                proc = in_prior(seed)
            s = make_sample(proc, t, h, seed=seed, n_paths=1)
            xs.append(s.context[0])
            ys.append(s.future[0])
        return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def train(fc, sampler: Sampler, steps: int, batch_size: int = 16, lr: float = 1e-4, warmup: int = 20,
          weight_decay: float = 0.01, clip: float = 1.0, log_every: int = 10, device: str = "cpu",
          log: Callable[[str], None] = print) -> dict:
    params = [p for p in fc.model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    fc.model.train()
    t0 = time.time()
    losses, tokens = [], 0
    for step in range(steps):
        x, y = sampler.batch(batch_size)
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        q = forecast_train(fc, xt, yt.shape[1])
        loss = pinball_loss(q, yt, naive_scale_t(xt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()
        losses.append(loss.item())
        tokens += x.size + y.size
        if log_every and (step + 1) % log_every == 0:
            log(f"step {step + 1}/{steps} loss {np.mean(losses[-log_every:]):.4f} ({time.time() - t0:.0f}s)")
    fc.model.eval()
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() and device.startswith("cuda") else None
    return {"trainable_params": n_train, "steps": steps, "batch_size": batch_size, "lr": lr,
            "time_steps_seen": tokens, "wall_sec": round(time.time() - t0, 1), "final_loss": float(np.mean(losses[-10:])),
            "loss_curve": [round(v, 5) for v in losses], "peak_cuda_bytes": peak}


def export(fc, path: str) -> None:
    from safetensors.torch import save_file

    merge_lora(fc.model)
    os.makedirs(path, exist_ok=True)
    fc.model.config.save(os.path.join(path, "config.json"))
    sd = {k: v.detach().cpu().contiguous() for k, v in fc.model.state_dict().items()}
    save_file(sd, os.path.join(path, "model.safetensors"))
