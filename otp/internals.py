"""Look inside Timestral: attention of the forecast tokens and per-layer hidden states.

`trace(size, context, horizon)` runs one forward pass of the network (the plain pass, not
the mirrored pass used by symmetric averaging) on univariate contexts and returns:

- `attn[layer_idx]`: `(B, heads, H_patches, L + H_patches)` softmax weights of every
  forecast (decoder) token over context patches and earlier forecast tokens, for every
  time-attention sublayer. Recomputed from the same q/k (after RoPE) and mask that the
  network uses, so the weights are exactly the ones inside scaled-dot-product attention.
- `hidden[layer]`: `(B, L + H_patches, D)` residual stream after each transformer block.
- `quantiles`: `(B, H, 21)` raw-space quantiles from this pass (refine frame applied).
- `n_ctx_patches`: L.

Summaries (`attention_profile`) reduce the weights to interpretable numbers per series:
mass on the most recent patches, mass at seasonal lags, mass after a change point, and
entropy.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from otp.model import PATCH, load

_ROPE_APPLY = None


def _apply_rotary(q, k, cos, sin):
    global _ROPE_APPLY
    if _ROPE_APPLY is None:
        from timestral.layers import _apply_rotary as f

        _ROPE_APPLY = f
    return _ROPE_APPLY(q, k, cos, sin)


@torch.no_grad()
def trace(size: str, context: np.ndarray, horizon: int, keep_hidden: bool = True) -> dict:
    fc = load(size)
    model = fc.model
    x = torch.as_tensor(np.asarray(context, dtype=np.float32), device=fc.device)
    if x.dim() == 1:
        x = x[None]
    b, t = x.shape
    pad = (-t) % PATCH
    mask = torch.ones(b, t + pad, dtype=torch.bool, device=x.device)
    if pad:
        x = torch.nn.functional.pad(x, (pad, 0))
        mask[:, :pad] = False
    h_patches = math.ceil(horizon / PATCH)
    n_ctx = (t + pad) // PATCH

    attn: dict[int, torch.Tensor] = {}
    hidden: dict[int, torch.Tensor] = {}
    hooks = []
    time_idx = 0
    for li, layer in enumerate(model.transformer.layers):
        for a in layer.attns:
            if not a.is_time:
                continue

            def hook(mod, args, out, key=time_idx):
                xin, msk, cos, sin = args
                q, k, _ = mod._qkv(xin)
                q, k = _apply_rotary(q, k, cos, sin)
                n = msk.enc_len
                full_mask = msk.mask  # (R, 1, S, S)
                qd = q[:, n:].transpose(1, 2)  # (R, heads, Hd, hd)
                kk = k.transpose(1, 2)  # (R, heads, S, hd)
                scores = qd @ kk.transpose(-1, -2) / math.sqrt(qd.shape[-1])
                scores = scores.masked_fill(~full_mask[:, :, n:, :], float("-inf"))
                attn[key] = torch.softmax(scores.float(), dim=-1).cpu()

            hooks.append(a.register_forward_hook(hook))
            time_idx += 1
        if keep_hidden:
            hooks.append(layer.register_forward_hook(lambda m, i, o, li=li: hidden.__setitem__(li, o.detach().cpu())))
    try:
        gid = torch.arange(b, device=x.device)
        q = model.predict_block(x, mask, gid, h_patches)
        q = fc._refine_block(q, x, mask, None)[:, :horizon]
    finally:
        for hk in hooks:
            hk.remove()
    return {"attn": {k: v.numpy() for k, v in attn.items()},
            "hidden": {k: v.numpy() for k, v in hidden.items()},
            "quantiles": q.cpu().numpy(), "n_ctx_patches": n_ctx, "pad": pad}


def attention_profile(tr: dict, period: int | None = None, change_step: int | None = None,
                      recent_patches: int = 4, dec_token: int = 0) -> dict[str, np.ndarray]:
    """Per-series, per-layer summaries of where forecast token `dec_token` looks, averaged over heads.

    - `recent`: mass on the last `recent_patches` context patches.
    - `seasonal`: mass on context patches whose distance to the end is within half a patch
      of a multiple of `period` (only when period >= 2 * PATCH), minus the mass a uniform
      spread would put there (so 0 = no preference).
    - `post_change`: mass on patches after the change point (`change_step` = steps before
      the forecast origin), divided by the share of valid patches that lie after it
      (1 = proportional, < 1 = under-weights the new regime).
    - `entropy`: entropy of the context part of the weights, in nats.
    - `ctx_mass`: total mass on context patches (the rest goes to forecast tokens).
    """
    L = tr["n_ctx_patches"]
    pad_patches = tr["pad"] // PATCH  # left padding patches are fully masked anyway
    layers = sorted(tr["attn"])
    out = {k: [] for k in ("recent", "seasonal", "post_change", "entropy", "ctx_mass")}
    for li in layers:
        w = tr["attn"][li][:, :, dec_token, :L].mean(1)  # (B, L)
        ctx_mass = w.sum(-1)
        wn = w / np.clip(ctx_mass[:, None], 1e-12, None)
        valid = np.arange(L) >= pad_patches
        nvalid = valid.sum()
        out["ctx_mass"].append(ctx_mass)
        out["recent"].append(wn[:, -recent_patches:].sum(-1))
        ent = -(wn * np.log(np.clip(wn, 1e-12, None))).sum(-1)
        out["entropy"].append(ent)
        if period and period >= 2 * PATCH:
            dist = (L - 1 - np.arange(L)) * PATCH + PATCH / 2  # steps from patch centre to origin
            r = np.mod(dist, period)
            near = valid & ((r < PATCH / 2) | (period - r < PATCH / 2)) & (dist > PATCH)
            out["seasonal"].append(wn[:, near].sum(-1) - near.sum() / nvalid)
        else:
            out["seasonal"].append(np.full(w.shape[0], np.nan))
        if change_step is not None:
            first_post = L - math.ceil(change_step / PATCH)
            post = valid & (np.arange(L) >= first_post)
            share = post.sum() / nvalid
            out["post_change"].append(wn[:, post].sum(-1) / max(share, 1e-12) if post.any() else np.full(w.shape[0], np.nan))
        else:
            out["post_change"].append(np.full(w.shape[0], np.nan))
    return {k: np.stack(v, axis=1) for k, v in out.items()}  # (B, n_time_layers)


def ridge_readout(features: np.ndarray, y: np.ndarray, alphas=(1e-2, 1e-1, 1, 10, 100, 1e3), folds: int = 5,
                  seed: int = 0) -> dict:
    """Cross-validated ridge linear readout: R^2 on held-out folds (features standardized in-fold).

    Returns the best mean held-out R^2 over `alphas`, its alpha, and per-fold scores.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    parts = np.array_split(idx, folds)
    best = (-np.inf, None, None)
    for a in alphas:
        scores = []
        for f in range(folds):
            te = parts[f]
            tr = np.concatenate([parts[g] for g in range(folds) if g != f])
            mu, sd = features[tr].mean(0), features[tr].std(0) + 1e-8
            xtr, xte = (features[tr] - mu) / sd, (features[te] - mu) / sd
            ym = y[tr].mean()
            d = xtr.shape[1]
            w = np.linalg.solve(xtr.T @ xtr + a * np.eye(d), xtr.T @ (y[tr] - ym))
            pred = xte @ w + ym
            ss = ((y[te] - y[te].mean()) ** 2).sum()
            scores.append(1 - ((y[te] - pred) ** 2).sum() / max(ss, 1e-12))
        m = float(np.mean(scores))
        if m > best[0]:
            best = (m, a, scores)
    return {"r2": best[0], "alpha": best[1], "fold_r2": best[2]}
