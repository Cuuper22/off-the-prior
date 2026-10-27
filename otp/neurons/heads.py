"""Attention heads: what do context-token and forecast-token queries attend to?

For every time-attention head (every block has one) and every variate-attention head (blocks of kind
`time_variate`), per series and query type (context-patch queries, forecast-token queries), we compute from the
attention weights of the plain forward pass:

- `self`     weight on the query's own position (context queries)
- `recent`   weight on the 4 patches before the query (context queries) / the last 4 context patches (forecast queries)
- `first`    weight on the first patch (an attention sink if large)
- `entropy`  entropy of the weights over the valid keys divided by log(number of valid keys)
- `loglag`   expected log1p(lag in patches)
- `seas`     weight on keys at a multiple of the series' period (within half a patch) minus the weight a uniform
             spread would put there (only for series with period >= 64 steps)
- `post`     weight on patches at or after the switch, divided by the share of valid keys there (series with a
             scheduled switch in the context; context queries beyond switch + 2 patches, and forecast queries)
- `peak`     lag (in patches) with the largest mean weight

`summarize` turns per-series arrays into one row per head: means over all series, over seasonal series and over
switch series, the cross-series correlation of the head's peak lag with the period, and a rule-based description.
Variate heads (rows of one coupled system attending each other) report the weight a row puts on the other row and
its correlation with the coupling strength.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from otp.model import PATCH, load

METRICS = ["self", "recent", "first", "entropy", "loglag", "seas", "post", "peak", "peakmass", "postpl"]
NM = len(METRICS)


def _apply_rotary(q, k, cos, sin):
    from timestral.layers import _apply_rotary as f
    return f(q, k, cos, sin)


def _metrics(W: torch.Tensor, qpos: torch.Tensor, kpos: torch.Tensor, valid_cnt: torch.Tensor, lam: torch.Tensor,
             switch: torch.Tensor, switch_pl: torch.Tensor, ctx_q: bool, L: int) -> torch.Tensor:
    """W (B, heads, Q, K) -> (B, heads, NM). lam (B,) period in patches (0 = none), switch (B,) patch (-1 = none)."""
    B, Hh, Q, K = W.shape
    lag = (qpos[:, None] - kpos[None, :]).float()                         # (Q, K)
    pos_lag = lag > 0
    qsel = (qpos >= 8) if ctx_q else torch.ones_like(qpos, dtype=torch.bool)
    out = torch.zeros(B, Hh, NM, device=W.device)
    nsel = qsel.float().sum().clamp(min=1)
    Ws = W[:, :, qsel, :]
    ls = lag[qsel]
    vs = valid_cnt[qsel].float()
    if ctx_q:
        self_m = (ls == 0).float()
        rec_m = ((ls >= 1) & (ls <= 4)).float()
    else:
        self_m = ((kpos[None, :] == qpos[qsel][:, None])).float()
        rec_m = ((kpos[None, :] >= L - 4) & (kpos[None, :] < L)).float()
    out[:, :, 0] = (Ws * self_m).sum(-1).mean(-1)
    out[:, :, 1] = (Ws * rec_m).sum(-1).mean(-1)
    out[:, :, 2] = Ws[..., 0].mean(-1)
    ent = -(Ws * torch.log(Ws.clamp(min=1e-12))).sum(-1)
    out[:, :, 3] = (ent / torch.log(vs.clamp(min=2))[None, None]).mean(-1)
    out[:, :, 4] = (Ws * torch.log1p(ls.clamp(min=0))[None, None]).sum(-1).mean(-1)
    # seasonal excess
    has = lam > 0
    if has.any():
        lb = lam.clamp(min=1e-6)[:, None, None]
        ratio = ls[None] / lb
        near = ((ratio - ratio.round()).abs() * lb <= 0.5) & (ratio.round() >= 1) & (ls[None] > 0)
        nearf = near.float()                                               # (B, Qs, K)
        mass = (Ws * nearf[:, None]).sum(-1)                               # (B, heads, Qs)
        exp = nearf.sum(-1) / vs[None]                                     # (B, Qs)
        ok = (nearf.sum(-1) > 0).float()                                   # queries that have at least one key
        ex = ((mass - exp[:, None]) * ok[:, None]).sum(-1) / ok.sum(-1).clamp(min=1)[:, None]
        out[:, :, 5] = torch.where(has[:, None], ex, torch.zeros_like(ex))
    # post-switch ratio (real switch) and the same ratio at a placebo switch patch
    for col, sw_arr in ((6, switch), (9, switch_pl)):
        hs = sw_arr >= 0
        if not hs.any():
            out[:, :, col] = float("nan")
            continue
        sw = sw_arr.clamp(min=0)[:, None, None].float()
        after = (kpos[None, None, :].float() >= sw).float()                # (B, 1, K)
        q_ok = (qpos[qsel][None, :].float() >= sw[:, :, 0] + 2).float()    # (B, Qs)
        share = (after * (kpos[None, None, :] <= qpos[qsel][None, :, None]).float()).sum(-1) / vs[None]
        mass = (Ws * after[:, None]).sum(-1)
        r = mass / share[:, None].clamp(min=1e-6)
        val = (r * q_ok[:, None]).sum(-1) / q_ok.sum(-1).clamp(min=1)[:, None]
        out[:, :, col] = torch.where(hs[:, None] & (q_ok.sum(-1) > 0)[:, None], val,
                                     torch.full_like(val, float("nan")))
    # peak lag: mean weight by lag (integer patches, 1..K)
    lag_i = ls.long().clamp(min=0, max=K)                                  # (Qs, K)
    prof = torch.zeros(B, Hh, K + 1, device=W.device)
    idx = lag_i[None, None].expand(B, Hh, -1, -1).reshape(B, Hh, -1)
    prof.scatter_add_(2, idx, Ws.reshape(B, Hh, -1))
    prof = prof[:, :, 1:] / nsel                                           # lags 1..K
    m, a = prof.max(-1)
    out[:, :, 7] = a.float() + 1
    out[:, :, 8] = m
    return out


@torch.no_grad()
def collect_time(size: str, values: np.ndarray, T: int, lam: np.ndarray, switch: np.ndarray,
                 switch_pl: np.ndarray, batch: int = 16,
                 device: str | None = None, log=print) -> np.ndarray:
    """Per-series metrics for every time-attention head: (N, n_layers, heads, 2, NM); query type 0 = context, 1 = forecast."""
    fc = load(size, device)
    model = fc.model
    dev = fc.device
    layers = list(model.transformer.layers)
    nl = len(layers)
    heads = layers[0].attns[0].num_heads
    n = values.shape[0]
    L = T // PATCH
    out = np.zeros((n, nl, heads, 2, NM), np.float32)
    cur: dict = {}
    hooks = []

    def mk(li):
        def hook(mod, args, o):
            xin, msk, cos, sin = args
            q, k, _ = mod._qkv(xin)
            q, k = _apply_rotary(q, k, cos, sin)
            nn = msk.enc_len
            full = msk.mask
            res = []
            for qt in (0, 1):
                if qt == 0:
                    qq, kk, mm = q[:, :nn], k[:, :nn], full[:, :, :nn, :nn]
                else:
                    qq, kk, mm = q[:, nn:], k, full[:, :, nn:, :]
                sc = qq.transpose(1, 2) @ kk.transpose(1, 2).transpose(-1, -2) / math.sqrt(qq.shape[-1])
                sc = sc.masked_fill(~mm, float("-inf"))
                W = torch.softmax(sc.float(), dim=-1)
                if qt == 0:
                    qpos = torch.arange(nn, device=dev)
                    kpos = torch.arange(nn, device=dev)
                    vc = qpos + 1
                else:
                    ks = k.shape[1]
                    qpos = torch.arange(nn, ks, device=dev)
                    kpos = torch.arange(ks, device=dev)
                    vc = qpos + 1
                res.append(_metrics(W, qpos, kpos, vc, cur["lam"], cur["switch"], cur["switch_pl"], qt == 0, L))
            cur[li] = torch.stack(res, dim=2).cpu().numpy()
        return hook

    for li, layer in enumerate(layers):
        for a in layer.attns:
            if a.is_time:
                hooks.append(a.register_forward_hook(mk(li)))
    try:
        for i in range(0, n, batch):
            xb = torch.as_tensor(values[i:i + batch, :T], dtype=torch.float32, device=dev)
            b = xb.shape[0]
            cur["lam"] = torch.as_tensor(lam[i:i + b], dtype=torch.float32, device=dev)
            cur["switch"] = torch.as_tensor(switch[i:i + b], dtype=torch.long, device=dev)
            cur["switch_pl"] = torch.as_tensor(switch_pl[i:i + b], dtype=torch.long, device=dev)
            mask = torch.ones(b, T, dtype=torch.bool, device=dev)
            model.predict_block(xb, mask, torch.arange(b, device=dev), 4)
            for li in range(nl):
                out[i:i + b, li] = cur[li]
            if (i // batch) % 40 == 0:
                log(f"  heads: {min(i + batch, n)}/{n}")
    finally:
        for h in hooks:
            h.remove()
    return out


@torch.no_grad()
def collect_variate(size: str, values: np.ndarray, T: int, gid: np.ndarray, row: np.ndarray, batch: int = 32,
                    device: str | None = None) -> tuple[np.ndarray, list[int]]:
    """Per-row weight put on the other row of the same group, per variate head and query type.

    Returns (n_rows, n_variate_layers, heads, 2) and the block indices of the variate layers.
    """
    fc = load(size, device)
    model = fc.model
    dev = fc.device
    layers = list(model.transformer.layers)
    vlayers = [li for li, ly in enumerate(layers) if any(not a.is_time for a in ly.attns)]
    heads = layers[0].attns[0].num_heads
    n = values.shape[0]
    L = T // PATCH
    out = np.zeros((n, len(vlayers), heads, 2), np.float32)
    cur: dict = {}
    hooks = []

    def mk(vi):
        def hook(mod, args, o):
            x, msk, cos, sin = args                          # x (R, N, D)
            inp = x.permute(1, 0, 2)                         # (N, R, D)
            q, k, _ = mod._qkv(inp)                          # (N, R, heads, hd)
            sc = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / math.sqrt(q.shape[-1])   # (N, heads, R, R)
            if msk is not None:
                sc = sc.masked_fill(~msk, float("-inf"))
            W = torch.softmax(sc.float(), dim=-1)
            R = W.shape[-1]
            eye = torch.eye(R, device=W.device, dtype=torch.bool)
            other = (W * (~eye)).sum(-1)                     # (N, heads, R) weight on other rows
            cur[vi] = other.permute(2, 1, 0).cpu().numpy()   # (R, heads, N)
        return hook

    vi = 0
    for li, ly in enumerate(layers):
        for a in ly.attns:
            if not a.is_time:
                hooks.append(a.register_forward_hook(mk(vi)))
                vi += 1
    starts = [0]
    while starts[-1] < n:
        j = min(starts[-1] + batch, n)
        while j < n and gid[j] == gid[j - 1]:
            j += 1
        starts.append(j)
    try:
        for i, j in zip(starts[:-1], starts[1:]):
            xb = torch.as_tensor(values[i:j, :T], dtype=torch.float32, device=dev)
            _, gb = np.unique(gid[i:j], return_inverse=True)
            mask = torch.ones(j - i, T, dtype=torch.bool, device=dev)
            model.predict_block(xb, mask, torch.as_tensor(gb, device=dev), 4)
            for v in range(len(vlayers)):
                o = cur[v]                                   # (R, heads, N)
                out[i:j, v, :, 0] = o[:, :, :L].mean(-1)
                out[i:j, v, :, 1] = o[:, :, L:].mean(-1)
    finally:
        for h in hooks:
            h.remove()
    return out, vlayers


def describe(row: dict) -> str:
    """Rule-based description of a time head from its summary (thresholds fixed here)."""
    if row["first_all"] > 0.5:
        return "sink (first patch)"
    if row["self_all"] > 0.5:
        return "self"
    if row["recent_all"] > 0.5:
        return "recent patches"
    if row["seas_seasonal"] > 0.1 and row["peak_vs_period_r2"] > 0.5:
        return "seasonal lag"
    if row["post_switch"] > 1.5 and row["post_switch"] > 1.25 * row["post_placebo"]:
        return "post-switch patches"
    if row["entropy_all"] > 0.8:
        return "diffuse"
    return "mixed"


def summarize_time(arr: np.ndarray, lam: np.ndarray, switch: np.ndarray, size: str) -> list[dict]:
    """One row per (layer, head, query type) from per-series metrics `arr` (N, layers, heads, 2, NM)."""
    n, nl, heads, _, _ = arr.shape
    seasonal = lam > 0
    sw = switch >= 0
    rows = []
    mi = {m: i for i, m in enumerate(METRICS)}
    for li in range(nl):
        for h in range(heads):
            for qt, qn in ((0, "context"), (1, "forecast")):
                a = arr[:, li, h, qt, :]
                peak = a[:, mi["peak"]]
                if seasonal.sum() > 5:
                    x, y = lam[seasonal], peak[seasonal]
                    r = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else 0.0
                    r2 = float(r * r) if np.isfinite(r) else 0.0
                else:
                    r2 = float("nan")
                row = {"size": size, "layer": li, "head": h, "query": qn,
                       "self_all": float(np.nanmean(a[:, mi["self"]])), "recent_all": float(np.nanmean(a[:, mi["recent"]])),
                       "first_all": float(np.nanmean(a[:, mi["first"]])), "entropy_all": float(np.nanmean(a[:, mi["entropy"]])),
                       "loglag_all": float(np.nanmean(a[:, mi["loglag"]])),
                       "peak_lag_all": float(np.nanmedian(peak)), "peak_mass_all": float(np.nanmean(a[:, mi["peakmass"]])),
                       "seas_seasonal": float(np.nanmean(a[seasonal, mi["seas"]])) if seasonal.any() else float("nan"),
                       "post_switch": float(np.nanmean(a[sw, mi["post"]])) if sw.any() else float("nan"),
                       "post_placebo": float(np.nanmean(a[sw, mi["postpl"]])) if sw.any() else float("nan"),
                       "peak_vs_period_r2": r2}
                row["description"] = describe(row)
                rows.append(row)
    return rows
