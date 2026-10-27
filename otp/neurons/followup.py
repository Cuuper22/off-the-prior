"""Follow-up A of the neuron atlas: can the steering test detect an effect that is known to exist?

Pre-registered in `results/neurons/steering_preregistration_followup.md`. This module holds

- residual-stream directions: ridge readouts of a known variable (local level, local noise sd, post-jump level)
  from the output of a transformer block, the steering vector that moves the readout by a chosen amount, and a
  forward pass that adds such a vector at chosen tokens (`run_add_conditions`);
- groups of poles: the top-k poles of a label, matched random groups, and a vectorized version of the scaling pass of
  `otp.neurons.steer.run_conditions` for many poles at once (`run_group_conditions`);
- the matching criterion of `otp.neurons.steer` for any list of scales (`judge_scales`) and the calibrated criterion
  (effect larger than the 95th percentile of the matched controls).

Tokens of a 1024-step univariate series: 32 context patches (positions 0..31) then 4 forecast tokens (32..35).
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import torch

from otp.model import PATCH, load
from otp.neurons import capture as Cap
from otp.neurons import steer as St
from otp.neurons import variables as V

CTX = 1024
L_CTX = CTX // PATCH
N_TOK = L_CTX + Cap.N_FC
SITES = ("F", "C1", "C4")
VARIABLES = ("level", "noise", "postjump")
PREDICTED = {"level": "level", "noise": "spread", "postjump": "level"}
MOVES = (0.0, 0.5, 1.0, 2.0)                  # judged list of readout moves
NEG_MOVES = (-2.0, -1.0, -0.5)                # reported for antisymmetry
N_RANDOM_DIR = 32
N_RANDOM_GROUP = 16
KS = (8, 32, 128)
POOLS = ("one_job", "label_top")
LABEL_TOP_R2 = 0.10
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4)
QUANTITIES = St.QUANTITIES


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


def blocks_for(n_blocks: int) -> dict[str, int]:
    return {"early": n_blocks // 6, "mid": n_blocks // 2, "late": n_blocks - 1}


def site_slice(site: str) -> tuple[int, int]:
    return {"F": (L_CTX, N_TOK), "C1": (L_CTX - 1, L_CTX), "C4": (L_CTX - 4, L_CTX)}[site]


# ---------------------------------------------------------------------------------------------
# test series
# ---------------------------------------------------------------------------------------------
def test_sets(c: dict) -> tuple[dict, dict]:
    """Held-out mixed set and held-out seasonal set (period 8..64): same rule and seed as scripts/neurons_steer.py."""
    held = np.where(c["split"] == 1)[0]
    sc = {n: i for i, n in enumerate(V.SERIES_COLS)}
    rng = np.random.default_rng(St.SAMPLE_SEED)
    mixed = np.sort(rng.choice(held, min(St.N_MIXED, len(held)), replace=False))
    per = 2.0 ** c["sv"][:, sc["s_log2_period"]]
    seas = held[(c["sv"][held, sc["s_has_season"]] == 1) & (per[held] >= 8) & (per[held] <= 64)]
    seas = np.sort(rng.choice(seas, min(St.N_SEASONAL, len(seas)), replace=False))
    return ({"ctx": c["values"][mixed, :CTX], "period": None, "ids": mixed},
            {"ctx": c["values"][seas, :CTX], "period": per[seas].astype(float), "ids": seas})


def jump_set(c: dict, specs: list[dict]) -> dict:
    """All held-out `mean_jump` series (the jump lies inside the context)."""
    gen = np.array([s["generator"] for s in specs])
    ids = np.where((gen == "mean_jump") & (c["split"] == 1))[0]
    return {"ctx": c["values"][ids, :CTX], "period": None, "ids": ids}


# ---------------------------------------------------------------------------------------------
# residual-stream readouts
# ---------------------------------------------------------------------------------------------
def token_responses(c: dict, specs: list[dict]) -> dict:
    """Responses per series and token (N, 36): level, noise (log local noise sd), postjump; NaN = not used.
    `ok_all` marks tokens where level and noise are both defined; `mj` marks mean_jump series."""
    pc = {n: i for i, n in enumerate(V.PATCH_COLS)}
    pv = c["pv"]
    lvl = pv[:, :, pc["lat_level"]].astype(np.float64)
    nsd = pv[:, :, pc["lat_noise_sd"]].astype(np.float64)
    reg = pv[:, :, pc["lat_regime_frac"]].astype(np.float64)
    ok = np.isfinite(lvl) & np.isfinite(nsd) & (nsd > 0)
    gen = np.array([s["generator"] for s in specs])
    mj = gen == "mean_jump"
    pj = np.full_like(lvl, np.nan)
    for i in np.where(mj)[0]:
        pre = (reg[i, :L_CTX] == 0) & np.isfinite(lvl[i, :L_CTX])
        if pre.sum() >= 2:
            pj[i] = lvl[i] - np.median(lvl[i, :L_CTX][pre])
    ok_pj = ok & np.isfinite(pj) & mj[:, None]
    return {"level": np.where(ok, lvl, np.nan), "noise": np.where(ok, np.log(np.where(nsd > 0, nsd, 1.0)), np.nan),
            "postjump": np.where(ok_pj, pj, np.nan), "ok_all": ok, "ok_pj": ok_pj, "mj": mj}


class _Stat:
    """Sufficient statistics of a ridge regression of several responses on one feature matrix."""

    def __init__(self, d: int, responses: tuple[str, ...]):
        self.n = 0
        self.sx = np.zeros(d)
        self.sxx = np.zeros((d, d))
        self.responses = responses
        self.sy = {t: 0.0 for t in responses}
        self.syy = {t: 0.0 for t in responses}
        self.sxy = {t: np.zeros(d) for t in responses}

    def add(self, X: torch.Tensor, ys: dict[str, torch.Tensor]):
        if X.shape[0] == 0:
            return
        X = X.double()
        self.n += X.shape[0]
        self.sx += X.sum(0).cpu().numpy()
        self.sxx += (X.T @ X).cpu().numpy()
        for t in self.responses:
            y = ys[t].double()
            self.sy[t] += float(y.sum())
            self.syy[t] += float((y * y).sum())
            self.sxy[t] += (X.T @ y).cpu().numpy()

    def __add__(self, o: "_Stat") -> "_Stat":
        r = _Stat(len(self.sx), self.responses)
        r.n = self.n + o.n
        r.sx = self.sx + o.sx
        r.sxx = self.sxx + o.sxx
        for t in self.responses:
            r.sy[t] = self.sy[t] + o.sy[t]
            r.syy[t] = self.syy[t] + o.syy[t]
            r.sxy[t] = self.sxy[t] + o.sxy[t]
        return r


def stat_from_arrays(X: np.ndarray, y: np.ndarray, t: str = "y") -> _Stat:
    s = _Stat(X.shape[1], (t,))
    s.add(torch.as_tensor(X), {t: torch.as_tensor(y)})
    return s


def ridge_path(s: _Stat, t: str, alphas=ALPHAS) -> dict[float, tuple[np.ndarray, float]]:
    """Ridge on standardized features for each alpha: raw-space weights and intercept."""
    n = s.n
    mu = s.sx / n
    cxx = s.sxx - n * np.outer(mu, mu)
    sd = np.sqrt(np.clip(np.diag(cxx) / n, 1e-16, None))
    sd = np.maximum(sd, 1e-8)
    cs = cxx / np.outer(sd, sd)
    ybar = s.sy[t] / n
    b = (s.sxy[t] - n * mu * ybar) / sd
    lam, U = np.linalg.eigh(cs)
    ub = U.T @ b
    out = {}
    for a in alphas:
        w_std = U @ (ub / (lam + a))
        w = w_std / sd
        out[a] = (w, float(ybar - mu @ w))
    return out


def r2_of(s: _Stat, t: str, w: np.ndarray, a: float) -> float:
    n = s.n
    if n < 3:
        return float("nan")
    sse = (s.syy[t] - 2 * (a * s.sy[t] + w @ s.sxy[t]) + (n * a * a + 2 * a * (w @ s.sx) + w @ s.sxx @ w))
    sst = s.syy[t] - s.sy[t] ** 2 / n
    return float(1 - sse / max(sst, 1e-12))


def fit_direction(stats_g: dict[int, _Stat], t: str, alphas=ALPHAS) -> dict:
    """stats_g: groups 0,1,2 (label folds) and 3 (held-out series). Penalty chosen by 3-fold CV over the label folds,
    final fit on the three folds together. Returns weights (raw space), intercept, alpha, cv R^2, held-out R^2."""
    best = (-np.inf, None)
    cv = {}
    for a in alphas:
        r = []
        for f in range(3):
            tr = stats_g[(f + 1) % 3] + stats_g[(f + 2) % 3]
            w, b0 = ridge_path(tr, t, (a,))[a]
            r.append(r2_of(stats_g[f], t, w, b0))
        cv[a] = float(np.nanmean(r))
        if cv[a] > best[0]:
            best = (cv[a], a)
    a = best[1]
    allt = stats_g[0] + stats_g[1] + stats_g[2]
    w, b0 = ridge_path(allt, t, (a,))[a]
    return {"w": w, "b0": b0, "alpha": a, "r2_cv": best[0], "r2_train": r2_of(allt, t, w, b0),
            "r2_heldout": r2_of(stats_g[3], t, w, b0) if stats_g.get(3) is not None and stats_g[3].n > 2 else float("nan"),
            "n_train": allt.n, "n_heldout": stats_g[3].n if stats_g.get(3) is not None else 0}


def steer_vector(w: np.ndarray, move: float) -> np.ndarray:
    """Minimum-norm raw-space vector that changes the readout `x @ w` by exactly `move`."""
    return move * w / float(w @ w)


@torch.no_grad()
def collect_stats(size: str, c: dict, specs: list[dict], blocks: dict[str, int], device: str | None = None,
                  batch: int = 64, log=print) -> dict:
    """Sufficient statistics of the residual stream (output of the given blocks) against the three responses.

    Returns stats[(block, subset, tokenset, group)] -> _Stat with subset 'all' (responses level, noise) or 'mj'
    (response postjump), tokenset 'ctx' or 'fc', group 0..2 = label folds, 3 = held-out series."""
    fc = load(size, device)
    layers = fc.model.transformer.layers
    d = layers[0].mlp.w12.in_features
    resp = token_responses(c, specs)
    grp = np.where(c["split"] == 0, c["fold"], 3).astype(int)
    stats: dict = {}
    for blk in blocks.values():
        for sub, tg in (("all", ("level", "noise")), ("mj", ("postjump",))):
            for ts in ("ctx", "fc"):
                for g in range(4):
                    stats[(blk, sub, ts, g)] = _Stat(d, tg)
    n = c["values"].shape[0]
    ctx = c["values"][:, :CTX]
    dev = fc.device
    ts_idx = {"ctx": slice(0, L_CTX), "fc": slice(L_CTX, N_TOK)}
    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        cap: dict[int, torch.Tensor] = {}
        hooks = [layers[b].register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(b, o.detach().float())) for b in set(blocks.values())]
        try:
            Cap.forward_with_mlp(size, torch.as_tensor(ctx[i0:i1], dtype=torch.float32), device=device, keep=())
        finally:
            for h in hooks:
                h.remove()
        for blk in set(blocks.values()):
            H = cap[blk]
            for g in range(4):
                rows = np.where(grp[i0:i1] == g)[0]
                if len(rows) == 0:
                    continue
                for ts, sl in ts_idx.items():
                    Hs = H[torch.as_tensor(rows, device=H.device)][:, sl]
                    gi = i0 + rows
                    ok = resp["ok_all"][gi][:, sl]
                    okt = torch.as_tensor(ok, device=H.device)
                    X = Hs[okt]
                    ys = {t: torch.as_tensor(resp[t][gi][:, sl][ok], device=H.device) for t in ("level", "noise")}
                    stats[(blk, "all", ts, g)].add(X, ys)
                    okp = resp["ok_pj"][gi][:, sl]
                    if okp.any():
                        okpt = torch.as_tensor(okp, device=H.device)
                        stats[(blk, "mj", ts, g)].add(Hs[okpt], {"postjump": torch.as_tensor(resp["postjump"][gi][:, sl][okp], device=H.device)})
        if (i0 // batch) % 10 == 0:
            log(f"  readout statistics {i1}/{n} series")
    return stats


def fit_all_directions(stats: dict, blocks: dict[str, int]) -> dict:
    """fits[(variable, block_name, tokenset)] -> fit dict, tokenset in ctx / fc / pooled."""
    fits = {}
    for bname, blk in blocks.items():
        for var, sub in (("level", "all"), ("noise", "all"), ("postjump", "mj")):
            for ts in ("ctx", "fc", "pooled"):
                sg = {}
                for g in range(4):
                    if ts == "pooled":
                        sg[g] = stats[(blk, sub, "ctx", g)] + stats[(blk, sub, "fc", g)]
                    else:
                        sg[g] = stats[(blk, sub, ts, g)]
                if min(sg[g].n for g in range(3)) < 10:
                    continue
                fits[(var, bname, ts)] = fit_direction(sg, var)
    return fits


# ---------------------------------------------------------------------------------------------
# forward passes with edits
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def run_add_conditions(size: str, ctx: np.ndarray, conds: list, device: str | None = None, max_rows: int = 2048) -> np.ndarray:
    """Quantile forecasts (n_cond, B, 128, 21). A condition is (block, site, vector) or None (unsteered): the vector
    (d_model,) is added to the output of the block at the tokens of the site ('F', 'C1', 'C4')."""
    fc = load(size, device)
    layers = fc.model.transformer.layers
    B = ctx.shape[0]
    per = max(1, max_rows // B)
    out = np.zeros((len(conds), B, St.H_STEPS, 21), np.float32)
    x_all = torch.as_tensor(ctx, dtype=torch.float32)
    for c0 in range(0, len(conds), per):
        cb = conds[c0:c0 + per]
        plan: dict[int, list] = {}
        for k, cd in enumerate(cb):
            if cd is None:
                continue
            blk, site, vec = cd
            plan.setdefault(blk, []).append((k, site_slice(site), torch.as_tensor(np.asarray(vec), dtype=torch.float32)))
        hooks = []
        for blk, items in plan.items():
            def hook(mod, args, o, items=items):
                o2 = o.clone()
                for (k, (a, b), v) in items:
                    o2[k * B:(k + 1) * B, a:b, :] += v.to(device=o.device, dtype=o.dtype)
                return o2
            hooks.append(layers[blk].register_forward_hook(hook))
        try:
            q, _, _ = Cap.forward_with_mlp(size, x_all.repeat(len(cb), 1), device=device, keep=())
        finally:
            for h in hooks:
                h.remove()
        out[c0:c0 + len(cb)] = q.reshape(len(cb), B, St.H_STEPS, 21).float().cpu().numpy()
    return out


def poles_plan(poles: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """layer -> (neuron indices, sign vector); sign 0 when both poles of a neuron are in the set."""
    plan = {}
    for li, d in poles.groupby("layer"):
        m: dict[int, int] = {}
        for nn, sg in zip(d["neuron"].astype(int), d["sign"]):
            s = 1 if sg == "pos" else -1
            m[nn] = 0 if (nn in m and m[nn] != s) else s
        idx = np.array(sorted(m), dtype=np.int64)
        plan[int(li)] = (idx, np.array([m[i] for i in idx], dtype=np.float32))
    return plan


@torch.no_grad()
def run_group_conditions(size: str, ctx: np.ndarray, conds: list, device: str | None = None, max_rows: int = 2048) -> np.ndarray:
    """Like `steer.run_conditions` for many poles at once. A condition is (plan, scale) with plan from `poles_plan`
    (None = unsteered): z of each listed neuron is multiplied by `scale` on the tokens where it has the pole's sign
    (sign 0: both poles). Returns (n_cond, B, 128, 21)."""
    B = ctx.shape[0]
    per = max(1, max_rows // B)
    out = np.zeros((len(conds), B, St.H_STEPS, 21), np.float32)
    x_all = torch.as_tensor(ctx, dtype=torch.float32)
    for c0 in range(0, len(conds), per):
        cb = conds[c0:c0 + per]
        x = x_all.repeat(len(cb), 1)
        by_layer: dict[int, list] = {}
        for k, cd in enumerate(cb):
            if cd is None:
                continue
            plan, sc = cd
            for li, (idx, sg) in plan.items():
                by_layer.setdefault(li, []).append((k, idx, sg, float(sc)))
        cache: dict = {}

        def edit(li, z, gate, up):
            items = by_layer.get(li)
            if not items:
                return None
            z2 = z.clone()
            for (k, idx, sg, sc) in items:
                key = (li, id(idx), z.device)
                if key not in cache:
                    cache[key] = (torch.as_tensor(idx, device=z.device), torch.as_tensor(sg, device=z.device, dtype=z.dtype))
                it, st = cache[key]
                sub = z2[k * B:(k + 1) * B]
                col = sub[:, :, it]
                keep = (col * st > 0) | (st == 0)
                sub[:, :, it] = torch.where(keep, sc * col, col)
            return z2

        q, _, _ = Cap.forward_with_mlp(size, x, device=device, keep=(), edit=edit)
        out[c0:c0 + len(cb)] = q.reshape(len(cb), B, St.H_STEPS, 21).float().cpu().numpy()
    return out


# ---------------------------------------------------------------------------------------------
# criterion for any list of scales
# ---------------------------------------------------------------------------------------------
def _circ(d: np.ndarray, axis=-1) -> np.ndarray:
    return np.arctan2(np.sin(d).mean(axis), np.cos(d).mean(axis))


def mean_effect(q: str, e: np.ndarray) -> float:
    return float(_circ(e)) if q == "phase" else float(e.mean())


def judge_scales(effs: list[dict], scales: tuple, quantity: str, direction: str, sign_corr: float, seed: int = 0,
                 n_boot: int = St.N_BOOT, thresh: dict | None = None) -> dict:
    """The original criterion (monotone, size, interval) for an arbitrary scale list. effs[i][q]: per-series effects at
    scales[i] against the unsteered pass; the size statistic is D = E(scales[-1]) - E(scales[0]); the interval is a
    bootstrap over series. `quantity` may be 'a|b' (either passes)."""
    thresh = thresh or St.THRESH
    rng = np.random.default_rng(seed)
    best = None
    for q in quantity.split("|"):
        if q not in effs[0]:
            continue
        e = np.array([mean_effect(q, effs[i][q]) for i in range(len(scales))])
        rho = St._spearman4(np.array(scales, float), e)
        n = len(effs[0][q])
        ix = rng.integers(0, n, (n_boot, n))
        top, bot = effs[-1][q][ix], effs[0][q][ix]
        diffs = (_circ(top) - _circ(bot)) if q == "phase" else (top.mean(1) - bot.mean(1))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        size = float(e[-1] - e[0])
        if q == "phase":
            size = float((size + np.pi) % (2 * np.pi) - np.pi)
        want = 0 if direction == "0" else (1 if direction == "+" else -1) * (1 if sign_corr >= 0 else -1)
        mono = abs(rho) >= St.RHO_MIN and (want == 0 or np.sign(rho) == want)
        big = abs(size) >= thresh[q]
        sig = bool((lo > 0) or (hi < 0))
        res = {"quantity": q, "rho": rho, "size": size, "ci_lo": float(lo), "ci_hi": float(hi), "monotone": bool(mono),
               "big": bool(big), "sig": sig, "match": bool(mono and big and sig), "e": e.tolist()}
        if best is None or (res["match"] and not best["match"]) or (res["match"] == best["match"] and abs(size) > abs(best["size"])):
            best = res
    return best or {"quantity": quantity, "match": False}


def all_D(effs: list[dict]) -> dict[str, float]:
    """E(top) - E(first) for every quantity present."""
    out = {}
    for q in QUANTITIES:
        if q in effs[0]:
            d = mean_effect(q, effs[-1][q]) - mean_effect(q, effs[0][q])
            out[f"D_{q}"] = float((d + np.pi) % (2 * np.pi) - np.pi) if q == "phase" else float(d)
    return out


def effects_of(base_m: dict, ctx: np.ndarray, qs: np.ndarray, period) -> dict:
    return St.effects(base_m, St.measures(ctx, qs, period))


# ---------------------------------------------------------------------------------------------
# groups of poles
# ---------------------------------------------------------------------------------------------
def make_groups(poles: pd.DataFrame, pools=POOLS, ks=KS) -> list[dict]:
    """Labeled groups from a pole table (`poles.csv` columns; adds the class at theta 0.7, floor 0.1)."""
    from otp.neurons import score as Sc
    t = poles.copy()
    t["cls"] = Sc.klass(t, St.THETA_SAMPLE, St.FLOOR_SAMPLE)
    live = t[~t["dead"].astype(bool)]
    out = []
    for pool in pools:
        cand = live[live["cls"] == "one_job"] if pool == "one_job" else live[live["r2_cv"] >= LABEL_TOP_R2]
        cand = cand[cand["label"].isin(St.PREREG.keys())]
        for (lab, pos), d in cand.groupby(["label", cand["label_corr"] > 0]):
            d = d.sort_values(["r2_cv", "layer", "neuron"], ascending=[False, True, True])
            for k in ks:
                if len(d) >= k:
                    qn, dr = St.PREREG[lab]
                    out.append({"gid": f"{pool}/{lab}/{'p' if pos else 'n'}/k{k}", "pool": pool, "label": lab,
                                "sign": 1 if pos else -1, "k": k, "quantity": qn, "direction": dr, "poles": d.iloc[:k]})
    return out


def random_group(live: pd.DataFrame, group_poles: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Random poles with the (layer, sign) counts of the labeled group, from live poles outside it."""
    rng = np.random.default_rng(seed)
    member = set(zip(group_poles["layer"], group_poles["neuron"], group_poles["sign"]))
    parts = []
    for (li, sg), n in group_poles.groupby(["layer", "sign"]).size().items():
        pool = live[(live["layer"] == li) & (live["sign"] == sg)]
        pool = pool[[(a, b, s) not in member for a, b, s in zip(pool["layer"], pool["neuron"], pool["sign"])]]
        take = rng.permutation(len(pool))[:n]
        parts.append(pool.iloc[take])
    return pd.concat(parts)
