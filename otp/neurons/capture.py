"""Capture the input of every `w3` in Timestral's SwiGLU MLPs (one coordinate of `gate * silu(up)` per neuron).

A *neuron* is one hidden unit of a block's MLP. Its value on a token is `z = gate * silu(up)`, where
`[gate, up] = w12(x)`. The positive pole is `max(z, 0)` and the negative pole `max(-z, 0)`. The sign regime of a
firing is the pair (sign of gate, sign of up); `silu(up)` has the sign of `up`, so a positive `z` comes from
(+, +) or (-, -) and a negative `z` from (+, -) or (-, +). We store a 2-bit code per token and neuron:
`code = 2 * (gate > 0) + (up > 0)`.

Tokens of a univariate series: `L` context-patch tokens followed by `N_FC` forecast tokens (the learned query
tokens), in that order. The pass is the plain forward pass of the network (`predict_block`): no symmetric
averaging, no refine frame for the activations.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from otp.model import PATCH, load

N_FC = 4


def layer_dims(size: str, device: str = "cpu") -> tuple[int, int]:
    """(number of blocks, MLP hidden width)."""
    m = load(size, device).model
    return len(m.transformer.layers), m.transformer.layers[0].mlp.w3.in_features


@torch.no_grad()
def forward_with_mlp(size: str, x: torch.Tensor, n_fc: int = N_FC, device: str | None = None,
                     keep: tuple[int, ...] | None = None, edit=None, gid: torch.Tensor | None = None):
    """One plain forward pass on contexts `x` `(B, T)` (T a multiple of 32).

    Returns `(q, z, code)`: raw-space quantiles `(B, n_fc * 32, 21)` (refine frame applied), and, per kept layer,
    `z` `(B, tokens, H)` float32 and `code` `(B, tokens, H)` int8. `edit(layer, z, gate, up) -> z'` may replace the
    neuron values before `w3` (used for scaling and ablation); the codes returned are those of the unedited pass.
    """
    fc = load(size, device)
    model = fc.model
    dev = fc.device
    x = x.to(dev)
    b, t = x.shape
    assert t % PATCH == 0
    layers = list(model.transformer.layers)
    keep_set = set(range(len(layers))) if keep is None else set(keep)
    zs, codes = {}, {}
    hooks = []

    def mk(li):
        def hook(mod, args, out):
            gate, up = out.chunk(2, dim=-1)
            z = gate * F.silu(up)
            if li in keep_set:
                zs[li] = z.float()
                codes[li] = (2 * (gate > 0).to(torch.int8) + (up > 0).to(torch.int8))
            if edit is not None:
                z2 = edit(li, z, gate, up)
                if z2 is not None:
                    layers[li].mlp._edited = z2      # picked up by the pre-hook on w3
            return out
        return hook

    def mk_pre(li):
        def pre(mod, args):
            ed = getattr(layers[li].mlp, "_edited", None)
            if ed is not None:
                layers[li].mlp._edited = None
                return (ed,)
            return None
        return pre

    for li, layer in enumerate(layers):
        hooks.append(layer.mlp.w12.register_forward_hook(mk(li)))
        hooks.append(layer.mlp.w3.register_forward_pre_hook(mk_pre(li)))
    try:
        mask = torch.ones(b, t, dtype=torch.bool, device=dev)
        gid = torch.arange(b, device=dev) if gid is None else gid.to(dev)
        q = model.predict_block(x, mask, gid, n_fc)
        q = fc._refine_block(q, x, mask, None)[:, : n_fc * PATCH]
    finally:
        for h in hooks:
            h.remove()
        for layer in layers:
            layer.mlp._edited = None
    return q, zs, codes


@torch.no_grad()
def capture_corpus(size: str, values: np.ndarray, T: int, batch: int = 32, device: str | None = None,
                   layers: tuple[int, ...] | None = None, log=print, gid: np.ndarray | None = None):
    """Activations of every neuron on every token of a corpus.

    `values` `(N, >= T)`: the first `T` steps of each series are the context. Returns
    `z[layer]` `(N * tokens, H)` float16 and `code[layer]` int8, tokens in series order, each series
    contributing `T // 32 + N_FC` tokens (context patches, then forecast tokens).
    """
    fc = load(size, device)
    n = values.shape[0]
    n_blocks = len(fc.model.transformer.layers)
    use = list(range(n_blocks)) if layers is None else list(layers)
    tok = T // PATCH + N_FC
    h = fc.model.transformer.layers[0].mlp.w3.in_features
    z = {li: torch.zeros(n * tok, h, dtype=torch.float16) for li in use}
    code = {li: torch.zeros(n * tok, h, dtype=torch.int8) for li in use}
    clip = 60000.0
    n_clipped = 0
    # batches never split a group of rows (rows of one group attend each other in the variate attention)
    starts = [0]
    while starts[-1] < n:
        j = min(starts[-1] + batch, n)
        if gid is not None:
            while j < n and gid[j] == gid[j - 1]:
                j += 1
        starts.append(j)
    for bi, (i, j) in enumerate(zip(starts[:-1], starts[1:])):
        xb = torch.as_tensor(values[i:j, :T], dtype=torch.float32)
        gb = None
        if gid is not None:
            _, gb = np.unique(gid[i:j], return_inverse=True)
            gb = torch.as_tensor(gb)
        _, zs, cs = forward_with_mlp(size, xb, device=device, keep=tuple(use), gid=gb)
        for li in use:
            zl = zs[li]
            n_clipped += int((zl.abs() > clip).sum())
            z[li][i * tok:j * tok] = zl.clamp(-clip, clip).reshape(-1, h).to(torch.float16).cpu()
            code[li][i * tok:j * tok] = cs[li].reshape(-1, h).cpu()
        if bi % 20 == 0:
            log(f"  captured {j}/{n} rows")
    return z, code, {"tokens_per_series": tok, "n_clipped_fp16": n_clipped}
