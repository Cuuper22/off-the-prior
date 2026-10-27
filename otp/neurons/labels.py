"""Label every pole with the known variable (or pair of variables) that best explains its activation.

Definitions (all computed per pole = per neuron and sign):

- Activation of the positive pole on a token: `a = max(z, 0)`; of the negative pole: `a = max(-z, 0)`, with
  `z = gate * silu(up)` (see `capture.py`).
- Readout of a variable `v` (single label): the *bin-mean readout*. `v` is cut into at most 12 quantile bins
  (one bin per value for variables with at most 12 values; indicators have 2 bins) fitted on label-set tokens;
  the readout predicts the mean activation of the bin. A token whose `v` is undefined (NaN) is predicted by the
  overall training mean, so an undefined pattern never counts as information about the variable.
  Cross-validated R^2 = 1 - sum of squared errors / sum of squares around the training mean, pooled over
  3 folds of series (fold = group of series, never split a series). A pair label uses 5 x 5 quantile bins.
- Mutual information (bits) between the bin of `v` and the activation binned into 6 levels (0 = inactive,
  1..5 = quintiles of the pole's active values), over tokens where `v` is defined. Plug-in estimate.
- Explainable variance E of a pole: the larger of (a) the cross-validated R^2 of the additive model with the
  bins of all library variables as dummy features (ridge, lambda = 1) and (b) the best pair label's R^2. It is a
  lower bound on what the library can explain, so ratios R^2(label) / E are upper-biased when E is small.
- Held-out R^2: the label's readout fitted on all label-set tokens, evaluated on held-out series (R^2 around the
  label-set training mean).
"""

from __future__ import annotations

import numpy as np
import torch

from otp.neurons import variables as V

NB = 12            # bins per variable
NBP = 5            # bins per variable in pair labels
UNDEF = NB         # code of an undefined value in the single-variable one-hot (NB + 1 columns per variable)
NA = 6             # activation levels for mutual information
N_FOLD = 3
K_TOP = 8
K_MID = 8
MIN_ACTIVE = 30      # a pole with fewer active label-set tokens is reported as dead (no label)
LN2 = float(np.log(2.0))

GROUP_OF: dict[str, str] = {}
for _g, _names in {
    "level": ["lat_level", "prev_level", "m_level_ctx", "m_dlevel_prev"],
    "slope": ["lat_slope", "prev_slope", "m_slope", "m_monotone", "s_trend", "x_trend"],
    "curvature": ["lat_curv", "m_curv"],
    "noise": ["lat_noise_sd", "prev_noise_sd", "lat_inn_sd", "lat_pred1", "m_logsd", "m_resid_sd", "m_dsd_prev",
              "m_roughness", "m_acf1", "s_noise_level", "x_noise_level", "s_ar_rho", "x_ar1"],
    "phase": ["lat_phase_sin", "lat_phase_cos", "prev_phase_sin", "prev_phase_cos", "x_phase_sin", "x_phase_cos",
              "m_peak_pos", "m_trough_pos", "m_zero_cross"],
    "period": ["lat_log2_period", "s_log2_period", "x_log2_period", "s_has_season", "s_season_amp",
               "x_season_strength"],
    "regime": ["lat_regime", "lat_regime_frac", "lat_since_switch", "prev_regime", "x_post_cp", "x_since_cp",
               "s_switch_pos", "s_jump_mag", "s_markov_p", "x_cp_pos", "x_cp_strength"],
    "tail": ["lat_outlier", "lat_outlier_mag", "m_jump", "m_max_z", "m_skew", "m_kurt", "s_alpha", "x_tail",
             "m_range"],
    "persistence": ["s_hurst", "x_hurst"],
    "chaos": ["s_lyap"],
    "clustering": ["s_eta", "s_msm_k"],
    "coupling": ["s_coupling", "s_lag"],
    "position": ["pos_logj", "pos_from_end", "tok_fc", "tok_fc_step", "tok_cov_row"],
}.items():
    for _n in _names:
        GROUP_OF[_n] = _g
for _n in V.CATS:
    GROUP_OF[_n] = "family"
missing = [n for n in V.LIBRARY if n not in GROUP_OF]
assert not missing, missing


# ---------------------------------------------------------------------------------------------
# token variables and bins
# ---------------------------------------------------------------------------------------------
def token_table(c: dict) -> dict:
    """Variable matrix for every token of a corpus (series-major): X (S, len(LIBRARY)) with NaN = undefined."""
    pv, sv, cat, T = c["pv"], c["sv"], c["cat"], c["T"]
    n, p, _ = pv.shape
    L = T // V.PATCH
    X = np.full((n, p, len(V.LIBRARY)), np.nan, np.float32)
    pc = {n_: i for i, n_ in enumerate(V.PATCH_COLS)}
    sc = {n_: i for i, n_ in enumerate(V.SERIES_COLS)}
    for k, name in enumerate(V.LIBRARY):
        if name in pc:
            X[:, :, k] = pv[:, :, pc[name]]
        elif name.startswith("prev_"):
            src = pv[:, :, pc["lat_" + name[5:]]]
            X[:, 1:, k] = src[:, :-1]
        elif name in sc:
            X[:, :, k] = sv[:, sc[name]][:, None]
        elif name in V.CATS:
            X[:, :, k] = cat[:, V.CATS.index(name)][:, None]
    t = np.arange(p)
    is_fc = t >= L
    k_ix = {n_: i for i, n_ in enumerate(V.LIBRARY)}
    X[:, :, k_ix["pos_logj"]] = np.log1p(t)[None]
    X[:, :, k_ix["pos_from_end"]] = np.where(is_fc, -np.log1p(np.maximum(t - L + 1, 0)), np.log1p(np.maximum(L - 1 - t, 0)))[None]
    X[:, :, k_ix["tok_fc"]] = is_fc[None].astype(np.float32)
    X[:, :, k_ix["tok_fc_step"]] = np.where(is_fc, t - L + 1, 0)[None]
    row = c.get("row")
    X[:, :, k_ix["tok_cov_row"]] = (np.zeros(n) if row is None else np.asarray(row, float))[:, None]
    ser = np.repeat(np.arange(n), p)
    tpos = np.tile(t, n)
    return {"X": X.reshape(n * p, -1), "series": ser, "tpos": tpos, "is_fc": np.tile(is_fc, n), "L": L, "P": p,
            "split": np.repeat(c["split"], p), "fold": np.repeat(c["fold"], p)}


def make_bins(X: np.ndarray, fit_mask: np.ndarray, nb: int = NB, undef_code: int | None = None):
    """Per-variable bin edges from tokens in `fit_mask`, and the bin code of every token.

    Returns `codes` (S, V) int16 (undefined = `undef_code`, default `nb`), `nbins` (V,) number of bins in use,
    and `edges` (list of arrays).
    """
    undef = nb if undef_code is None else undef_code
    S, nv = X.shape
    codes = np.zeros((S, nv), np.int16)
    nbins = np.zeros(nv, np.int16)
    edges = []
    for v in range(nv):
        x = X[:, v]
        fin = np.isfinite(x)
        d = x[fin & fit_mask]
        if V.IS_CAT[v] or len(d) == 0:
            e = np.array([0.5]) if V.IS_CAT[v] else np.array([])
        else:
            u = np.unique(d)
            if len(u) <= nb:
                e = (u[1:] + u[:-1]) / 2
            else:
                e = np.unique(np.quantile(d, np.linspace(0, 1, nb + 1)[1:-1]))
        edges.append(e)
        code = np.searchsorted(e, np.where(fin, x, 0.0), side="right")
        code[~fin] = undef
        codes[:, v] = code
        nbins[v] = len(e) + 1
    return codes, nbins, edges


def onehot(codes: torch.Tensor, ncol: int, device) -> torch.Tensor:
    """codes (S, V) int -> (S, V * ncol) float32 one-hot, column v * ncol + code."""
    S, nv = codes.shape
    off = (torch.arange(nv, device=device) * ncol)[None]
    O = torch.zeros(S, nv * ncol, dtype=torch.float32, device=device)
    O.scatter_(1, (codes.long() + off), 1.0)
    return O


# ---------------------------------------------------------------------------------------------
# readout statistics
# ---------------------------------------------------------------------------------------------
def _fold_slices(fold: np.ndarray):
    """Tokens must be ordered fold 0, 1, 2 (label) then held-out; returns slices."""
    sl = []
    start = 0
    for f in range(N_FOLD):
        n = int((fold == f).sum())
        sl.append(slice(start, start + n))
        start += n
    return sl, slice(start, start + int((fold < 0).sum()))


def cv_binmean(O: torch.Tensor, A: torch.Tensor, sl, ho: slice, nv: int, ncol: int, undef_col: int | None,
               M: torch.Tensor | None = None, want_ho: bool = True, keep_means: bool = False):
    """Cross-validated and held-out R^2 of the bin-mean readout, for every variable at once.

    O (S, nv * ncol) one-hot, A (S, H) activations (label tokens first, ordered by fold, then held-out),
    M (S, H) optional 0/1 mask (the readout is fitted and scored on masked tokens only).
    Returns dict: r2 (nv, H), r2_fold (N_FOLD, nv, H), r2_ho (nv, H) (if want_ho), sst (H,).
    """
    H = A.shape[1]
    Am = A if M is None else A * M
    S1, C1, Sq, Sm, Nn = [], [], [], [], []
    for f in range(N_FOLD):
        Of, Af = O[sl[f]], Am[sl[f]]
        S1.append(Of.T @ Af)                                    # (nv*ncol, H) sum of a per bin
        if M is None:
            C1.append(Of.sum(0)[:, None].expand(-1, H))
            Nn.append(torch.full((H,), float(Of.shape[0]), device=A.device))
        else:
            Mf = M[sl[f]]
            C1.append(Of.T @ Mf)
            Nn.append(Mf.sum(0))
        Sq.append((Af * Af).sum(0))
        Sm.append(Af.sum(0))
    S1t, C1t, Nt, Smt = sum(S1), sum(C1), sum(Nn), sum(Sm)

    def means(s1, c1, n, sm):
        mu = sm / n.clamp(min=1)
        m = torch.where(c1 > 0, s1 / c1.clamp(min=1), mu[None])
        m = m.reshape(nv, ncol, H)
        if undef_col is not None:
            m[:, undef_col, :] = mu[None]
        return m, mu

    sse = torch.zeros(N_FOLD, nv, H, device=A.device)
    sst = torch.zeros(N_FOLD, H, device=A.device)
    for f in range(N_FOLD):
        m, mu = means(S1t - S1[f], C1t - C1[f], Nt - Nn[f], Smt - Sm[f])
        s1f = S1[f].reshape(nv, ncol, H)
        c1f = C1[f].reshape(nv, ncol, H)
        sse[f] = Sq[f][None] - 2 * (m * s1f).sum(1) + (c1f * m * m).sum(1)
        sst[f] = Sq[f] - 2 * mu * Sm[f] + Nn[f] * mu * mu
    out = {"r2": 1 - sse.sum(0) / sst.sum(0).clamp(min=1e-12)[None], "r2_fold": 1 - sse / sst.clamp(min=1e-12)[:, None],
           "sst": sst.sum(0)}
    if keep_means:
        out["means"], out["mu"] = means(S1t, C1t, Nt, Smt)       # (nv, ncol, H) bin means on all label tokens
    if want_ho:
        m, mu = means(S1t, C1t, Nt, Smt)
        Oh, Ah = O[ho], Am[ho]
        s1h = (Oh.T @ Ah).reshape(nv, ncol, H)
        if M is None:
            c1h = Oh.sum(0)[:, None].expand(-1, H).reshape(nv, ncol, H)
            nh = torch.full((H,), float(Oh.shape[0]), device=A.device)
        else:
            Mh = M[ho]
            c1h = (Oh.T @ Mh).reshape(nv, ncol, H)
            nh = Mh.sum(0)
        sqh, smh = (Ah * Ah).sum(0), Ah.sum(0)
        sse_h = sqh[None] - 2 * (m * s1h).sum(1) + (c1h * m * m).sum(1)
        sst_h = sqh - 2 * mu * smh + nh * mu * mu
        out["r2_ho"] = 1 - sse_h / sst_h.clamp(min=1e-12)[None]
    return out


def mutual_info(O: torch.Tensor, Acode: torch.Tensor, nv: int, ncol: int, undef_col: int | None) -> torch.Tensor:
    """MI (bits) between each variable's bin and the activation level, tokens where the variable is defined."""
    H = Acode.shape[1]
    J = torch.stack([O.T @ (Acode == b).float() for b in range(NA)], dim=0)      # (NA, nv*ncol, H)
    J = J.reshape(NA, nv, ncol, H)
    if undef_col is not None:
        J[:, :, undef_col, :] = 0
    Nv = J.sum((0, 2)).clamp(min=1)                                               # (nv, H)
    cb = J.sum(2, keepdim=True)                                                   # (NA, nv, 1, H)
    cv = J.sum(0, keepdim=True)                                                   # (1, nv, ncol, H)
    ratio = J * Nv[None, :, None, :] / (cb * cv).clamp(min=1e-12)
    mi = (J / Nv[None, :, None, :] * torch.log(ratio.clamp(min=1e-12)) * (J > 0)).sum((0, 2)) / LN2
    return mi


def ridge_additive(O: torch.Tensor, A: torch.Tensor, sl, ho: slice, lam: float = 1.0):
    """CV and held-out R^2 of the additive model on all bin dummies (ridge)."""
    H = A.shape[1]
    d = O.shape[1]
    XtX = [O[sl[f]].T @ O[sl[f]] for f in range(N_FOLD)]
    XtA = [O[sl[f]].T @ A[sl[f]] for f in range(N_FOLD)]
    Sm = [A[sl[f]].sum(0) for f in range(N_FOLD)]
    Sq = [(A[sl[f]] ** 2).sum(0) for f in range(N_FOLD)]
    Nf = [sl[f].stop - sl[f].start for f in range(N_FOLD)]
    eye = lam * torch.eye(d, device=A.device)
    sse = torch.zeros(H, device=A.device)
    sst = torch.zeros(H, device=A.device)
    for f in range(N_FOLD):
        G = sum(XtX[g] for g in range(N_FOLD) if g != f) + eye
        B = sum(XtA[g] for g in range(N_FOLD) if g != f)
        n_tr = sum(Nf[g] for g in range(N_FOLD) if g != f)
        mu = sum(Sm[g] for g in range(N_FOLD) if g != f) / n_tr
        W = torch.linalg.solve(G, B)          # no intercept column: the bins of every variable span it
        pred = O[sl[f]] @ W
        sse += ((A[sl[f]] - pred) ** 2).sum(0)
        sst += Sq[f] - 2 * mu * Sm[f] + Nf[f] * mu * mu
    out = {"r2": 1 - sse / sst.clamp(min=1e-12)}
    G = sum(XtX) + eye
    W = torch.linalg.solve(G, sum(XtA))
    mu = sum(Sm) / sum(Nf)
    Ah = A[ho]
    pred = O[ho] @ W
    sse_h = ((Ah - pred) ** 2).sum(0)
    sst_h = ((Ah - mu[None]) ** 2).sum(0)
    out["r2_ho"] = 1 - sse_h / sst_h.clamp(min=1e-12)
    return out


# ---------------------------------------------------------------------------------------------
# one layer
# ---------------------------------------------------------------------------------------------
def pair_index(core_idx: list[int]):
    return [(i, j) for a, i in enumerate(core_idx) for j in core_idx[a + 1:]]


def prepare(tab: dict, device: str = "cpu"):
    """Reorder tokens (fold 0,1,2, then held-out) and build the one-hot matrices once per corpus."""
    fold = tab["fold"]
    label = tab["split"] == 0
    order = np.concatenate([np.where(fold == f)[0] for f in range(N_FOLD)] + [np.where(fold < 0)[0]])
    X = tab["X"][order]
    codes, nbins, edges = make_bins(X, np.isin(np.arange(len(order)), np.arange(int(label.sum()))))
    ncol = NB + 1
    O = onehot(torch.as_tensor(codes), ncol, device)
    core_idx = [V.VAR_INDEX[n] for n in V.CORE]
    cX = X[:, core_idx]
    pcodes, pnb, pedges = make_bins(cX, np.isin(np.arange(len(order)), np.arange(int(label.sum()))), nb=NBP)
    pairs = pair_index(list(range(len(core_idx))))
    fold_o = fold[order]
    sl, ho = _fold_slices(fold_o)
    nl = int(label.sum())
    Xl = X[:nl]
    Md = np.isfinite(Xl)
    nd = np.maximum(Md.sum(0), 1)
    mu_x = np.where(Md, Xl, 0.0).sum(0) / nd
    Xc = np.where(Md, Xl - mu_x, 0.0).astype(np.float32)
    sd_x = np.sqrt((Xc ** 2).sum(0) / nd).clip(min=1e-12)
    return {"order": order, "Xc": torch.as_tensor(Xc, device=device), "Md": torch.as_tensor(Md.astype(np.float32), device=device),
            "sd_x": torch.as_tensor(sd_x.astype(np.float32), device=device), "nd_x": torch.as_tensor(nd.astype(np.float32), device=device), "O": O, "codes": codes, "nbins": nbins, "edges": edges, "pcodes": torch.as_tensor(pcodes),
            "pairs": pairs, "core_idx": core_idx, "sl": sl, "ho": ho, "n_label": int(label.sum()), "X": X,
            "tpos": tab["tpos"][order], "is_fc": tab["is_fc"][order], "L": tab["L"], "series": tab["series"][order],
            "device": device}


def _sorted_edges_and_exemplars(Al: torch.Tensor):
    """Activation levels, quintile edges and exemplar indices from label-set tokens only.

    Al (n, H). Returns Acode (n, H) int8 in 0..NA-1, top idx (K_TOP, H), mid idx (K_MID, H), n_active (H,).
    """
    n, H = Al.shape
    vals, idx = torch.sort(Al, dim=0)                       # ascending; zeros first
    act = (Al > 0)
    k = act.sum(0)                                          # (H,)
    start = n - k
    qs = torch.tensor([0.2, 0.4, 0.6, 0.8], device=Al.device)
    pos = (start[None] + (qs[:, None] * (k[None] - 1).clamp(min=0)).long()).clamp(max=n - 1)   # (4, H)
    edges = torch.gather(vals, 0, pos)                      # (4, H)
    code = torch.zeros_like(Al, dtype=torch.int8)
    lvl = 1 + sum((Al >= edges[i][None]).to(torch.int8) for i in range(4))
    code = torch.where(act, lvl.to(torch.int8), code)
    top = idx[n - K_TOP:].flip(0)                           # (K_TOP, H), largest first
    f = (0.4 + 0.2 * (torch.arange(K_MID, device=Al.device) + 0.5) / K_MID)[:, None]
    mpos = (start[None] + (f * (k[None] - 1).clamp(min=0)).long()).clamp(max=n - 1)
    mid = torch.gather(idx, 0, mpos)
    return code, top, mid, k, vals


def _position_groups(tpos: np.ndarray, is_fc: np.ndarray, L: int) -> np.ndarray:
    """Group id per token: 0 = first context patch, 1 = patches 1..3, 2..5 = quartiles of the rest, 6..9 = forecast k."""
    g = np.zeros(len(tpos), np.int64)
    ctx = ~is_fc
    g[ctx & (tpos == 0)] = 0
    g[ctx & (tpos >= 1) & (tpos <= 3)] = 1
    rest = ctx & (tpos >= 4)
    if rest.any():
        q = np.minimum(((tpos[rest] - 4) * 4 // max(L - 4, 1)), 3)
        g[rest] = 2 + q
    g[is_fc] = 6 + np.minimum(tpos[is_fc] - L, 3)
    return g


@torch.no_grad()
def label_layer(prep: dict, Z: torch.Tensor, code: torch.Tensor, regime_pairs=((3, 0), (2, 1)),
                real: dict | None = None) -> dict:
    """Label the two poles of every neuron of one layer.

    Z (S, H) float16 and code (S, H) int8 in *corpus* token order (the function applies `prep['order']`).
    `real` (optional): {"Z": (Sr, H) float16 activations on real-data tokens, "codes": (Sr, nv) bin codes of the real
    tokens under the synthetic bin edges}; adds transfer scores (see `transfer_scores`).
    Returns {sign: dict of numpy arrays (length H)} for sign in ('pos', 'neg').
    """
    dev = prep["device"]
    order = torch.as_tensor(prep["order"], device="cpu")
    Zo = Z[order].to(dev).float()
    Co = code[order].to(dev)
    O = prep["O"]
    sl, ho, nl = prep["sl"], prep["ho"], prep["n_label"]
    nv = len(V.LIBRARY)
    ncol = NB + 1
    H = Zo.shape[1]
    cat = torch.as_tensor(V.IS_CAT, device=dev)
    pg = torch.as_tensor(_position_groups(prep["tpos"], prep["is_fc"], prep["L"]), device=dev)
    Gm = torch.zeros(10, Zo.shape[0], device=dev)
    Gm[pg, torch.arange(Zo.shape[0], device=dev)] = 1.0
    pairs = prep["pairs"]
    pc = prep["pcodes"].to(dev)
    ncp = NBP + 1
    res = {}
    for sign, (reg_a, reg_b) in zip(("pos", "neg"), regime_pairs):
        A = torch.relu(Zo) if sign == "pos" else torch.relu(-Zo)
        out: dict[str, np.ndarray] = {}
        Al = A[:nl]
        Acode, top, mid, nact, vals = _sorted_edges_and_exemplars(Al)
        out["n_active_label"] = nact.cpu().numpy()
        out["top_idx"] = top.T.cpu().numpy()                 # positions in the reordered token list
        out["top_act"] = torch.gather(Al, 0, top).T.cpu().numpy()
        out["mid_idx"] = mid.T.cpu().numpy()
        out["mid_act"] = torch.gather(Al, 0, mid).T.cpu().numpy()
        out["mean"] = Al.mean(0).cpu().numpy()
        out["std"] = Al.std(0).cpu().numpy()
        out["max"] = vals[-1].cpu().numpy()
        kk = (nact.float() * 0.001).clamp(min=1).long()
        out["p999"] = torch.gather(vals, 0, (Al.shape[0] - kk)[None])[0].cpu().numpy()
        # single-variable readouts (all tokens)
        cv = cv_binmean(O, A, sl, ho, nv, ncol, UNDEF, keep_means=real is not None)
        mi = mutual_info(O[:nl], Acode, nv, ncol, UNDEF)
        if real is not None:
            out.update(transfer_scores(cv, real, dev, sign, prep))
        # sign and size of the Pearson correlation of the activation with each variable (label tokens, defined values)
        Md, Xc, sd_x, nd_x = prep["Md"], prep["Xc"], prep["sd_x"], prep["nd_x"]
        s_a, s_a2 = Md.T @ Al, Md.T @ (Al * Al)                                  # (nv, H)
        cov = (Xc.T @ Al) / nd_x[:, None]
        var_a = (s_a2 / nd_x[:, None] - (s_a / nd_x[:, None]) ** 2).clamp(min=1e-12)
        out["_corr"] = (cov / (sd_x[:, None] * var_a.sqrt())).cpu().numpy()
        out["_r2"] = cv["r2"].cpu().numpy()
        out["_r2_ho"] = cv["r2_ho"].cpu().numpy()
        out["_r2_fold"] = cv["r2_fold"].cpu().numpy()
        out["_mi"] = mi.cpu().numpy()
        out["sst_label"] = cv["sst"].cpu().numpy()
        # active-token readout (tokens above 10% of the 99th percentile of active values)
        tau = 0.1 * out["p999"]
        tau_t = torch.as_tensor(tau, device=dev)[None]
        Mact = (A >= tau_t).float() * (A > 0).float()
        cva = cv_binmean(O, A, sl, ho, nv, ncol, UNDEF, M=Mact)
        out["_r2_act"] = cva["r2"].cpu().numpy()
        out["frac_active_hi"] = Mact[:nl].mean(0).cpu().numpy()
        # readouts restricted to context tokens and to forecast tokens
        isfc = torch.as_tensor(prep["is_fc"], device=dev).float()[:, None]
        for tag, mk in (("ctx", (1 - isfc).expand(-1, H)), ("fc", isfc.expand(-1, H))):
            cvm = cv_binmean(O, A, sl, ho, nv, ncol, UNDEF, M=mk, want_ho=False)
            out[f"_r2_{tag}"] = cvm["r2"].cpu().numpy()
        # regime shares and regime-specific labels
        tot = Al.sum(0).clamp(min=1e-12)
        mass = {}
        for r in range(4):
            mass[r] = ((Al * (Co[:nl] == r)).sum(0) / tot).cpu().numpy()
        out["mass_g0u0"], out["mass_g0u1"], out["mass_g1u0"], out["mass_g1u1"] = mass[0], mass[1], mass[2], mass[3]
        for tag, r in (("regA", reg_a), ("regB", reg_b)):
            Ar = A * (Co == r)
            cvr = cv_binmean(O, Ar, sl, ho, nv, ncol, UNDEF, want_ho=False)
            out[f"_r2_{tag}"] = cvr["r2"].cpu().numpy()
        # position profile
        cnt = Gm[:, :nl].sum(1, keepdim=True).clamp(min=1)
        out["pos_mean"] = ((Gm[:, :nl] @ Al) / cnt).T.cpu().numpy()                    # (H, 10)
        out["pos_active"] = ((Gm[:, :nl] @ (Al > 0).float()) / cnt).T.cpu().numpy()
        # additive ceiling
        add = ridge_additive(O, A, sl, ho)
        out["r2_add"] = add["r2"].cpu().numpy()
        out["r2_add_ho"] = add["r2_ho"].cpu().numpy()
        # pair labels
        best_r2 = torch.full((H,), -1e9, device=dev)
        best_ho = torch.zeros(H, device=dev)
        best_pair = torch.zeros(H, dtype=torch.long, device=dev)
        CH = 40
        for p0 in range(0, len(pairs), CH):
            chunk = pairs[p0:p0 + CH]
            cells = torch.stack([pc[:, i] * ncp + pc[:, j] for i, j in chunk], dim=1)          # (S, chunk)
            Op = onehot(cells, ncp * ncp, dev)
            cvp = cv_binmean_pairs(Op, A, sl, ho, len(chunk), ncp)
            m, idx = cvp["r2"].max(0)
            upd = m > best_r2
            best_r2 = torch.where(upd, m, best_r2)
            best_ho = torch.where(upd, cvp["r2_ho"].gather(0, idx[None])[0], best_ho)
            best_pair = torch.where(upd, idx + p0, best_pair)
        out["pair_r2"] = best_r2.cpu().numpy()
        out["pair_r2_ho"] = best_ho.cpu().numpy()
        out["pair_idx"] = best_pair.cpu().numpy()
        res[sign] = out
    return res


def cv_binmean_pairs(Op: torch.Tensor, A: torch.Tensor, sl, ho: slice, npairs: int, ncp: int):
    """Bin-mean readout on cells of variable pairs; a cell with an undefined member predicts the training mean."""
    H = A.shape[1]
    ncell = ncp * ncp
    undef = torch.zeros(ncell, dtype=torch.bool, device=A.device)
    for a in range(ncp):
        undef[a * ncp + NBP] = True
        undef[NBP * ncp + a] = True
    S1 = [Op[sl[f]].T @ A[sl[f]] for f in range(N_FOLD)]
    C1 = [Op[sl[f]].sum(0) for f in range(N_FOLD)]
    Sq = [(A[sl[f]] ** 2).sum(0) for f in range(N_FOLD)]
    Sm = [A[sl[f]].sum(0) for f in range(N_FOLD)]
    Nf = [float(sl[f].stop - sl[f].start) for f in range(N_FOLD)]
    S1t, C1t, Smt, Nt = sum(S1), sum(C1), sum(Sm), sum(Nf)

    def means(s1, c1, sm, n):
        mu = sm / n
        m = torch.where(c1[:, None] > 0, s1 / c1.clamp(min=1)[:, None], mu[None])
        m = m.reshape(npairs, ncell, H)
        m[:, undef, :] = mu[None, None]
        return m, mu

    sse = torch.zeros(npairs, H, device=A.device)
    sst = torch.zeros(H, device=A.device)
    for f in range(N_FOLD):
        m, mu = means(S1t - S1[f], C1t - C1[f], Smt - Sm[f], Nt - Nf[f])
        s1f = S1[f].reshape(npairs, ncell, H)
        c1f = C1[f].reshape(npairs, ncell)
        sse += Sq[f][None] - 2 * (m * s1f).sum(1) + (c1f[:, :, None] * m * m).sum(1)
        sst += Sq[f] - 2 * mu * Sm[f] + Nf[f] * mu * mu
    out = {"r2": 1 - sse / sst.clamp(min=1e-12)[None]}
    m, mu = means(S1t, C1t, Smt, Nt)
    Oh, Ah = Op[ho], A[ho]
    s1h = (Oh.T @ Ah).reshape(npairs, ncell, H)
    c1h = Oh.sum(0).reshape(npairs, ncell)
    sqh, smh, nh = (Ah * Ah).sum(0), Ah.sum(0), float(Oh.shape[0])
    sse_h = sqh[None] - 2 * (m * s1h).sum(1) + (c1h[:, :, None] * m * m).sum(1)
    sst_h = sqh - 2 * mu * smh + nh * mu * mu
    out["r2_ho"] = 1 - sse_h / sst_h.clamp(min=1e-12)[None]
    return out


def apply_bins(X: np.ndarray, edges: list, nb: int = NB) -> np.ndarray:
    """Bin codes of new tokens under existing edges (NaN -> undefined code)."""
    S, nv = X.shape
    codes = np.zeros((S, nv), np.int16)
    for v in range(nv):
        x = X[:, v]
        fin = np.isfinite(x)
        code = np.searchsorted(edges[v], np.where(fin, x, 0.0), side="right")
        code[~fin] = nb
        codes[:, v] = code
    return codes


def analog_table() -> np.ndarray:
    """For each library variable the index of the variable that can be measured on real series (-1 = none)."""
    out = np.full(len(V.LIBRARY), -1, np.int64)
    for i, n in enumerate(V.LIBRARY):
        if n in V.MEASURABLE:
            out[i] = i
        elif n in V.ANALOG:
            out[i] = V.VAR_INDEX[V.ANALOG[n]]
    return out


@torch.no_grad()
def transfer_scores(cv: dict, real: dict, dev, sign: str, prep: dict) -> dict:
    """Does the synthetic label predict the pole's activation on real series?

    For each pole: take its best label `v` (by cross-validated R^2 on synthetic data), the measurable variable `u`
    standing in for it (`v` itself for motifs and measured analogs, the analog of a latent variable otherwise),
    fit the bin-mean readout of the pole on `u` over all synthetic label tokens, and apply it to the real tokens.
    Returns per pole: Pearson correlation and R^2 (around the real mean) of the prediction on real tokens, the
    analog variable index, and the synthetic cross-validated R^2 of `u`.
    """
    r2 = torch.nan_to_num(cv["r2"], nan=-1.0)
    best = r2.argmax(0)                                              # (H,)
    an = torch.as_tensor(analog_table(), device=dev)[best]           # (H,)
    H = best.shape[0]
    ok = an >= 0
    u = an.clamp(min=0)
    ar = torch.arange(H, device=dev)
    means = cv["means"]                                              # (nv, ncol, H)
    m_sel = means[u, :, ar]                                          # (H, ncol)
    codes = real["codes"].to(dev).long()                             # (Sr, nv)
    idx = codes[:, u]                                                # (Sr, H)
    pred = torch.gather(m_sel.T, 0, idx)                             # (Sr, H)
    Zr = real["Z"].to(dev).float()
    Ar = torch.relu(Zr) if sign == "pos" else torch.relu(-Zr)
    pc = pred - pred.mean(0, keepdim=True)
    ac = Ar - Ar.mean(0, keepdim=True)
    corr = (pc * ac).sum(0) / (pc.pow(2).sum(0).sqrt() * ac.pow(2).sum(0).sqrt()).clamp(min=1e-12)
    sst = ac.pow(2).sum(0).clamp(min=1e-12)
    r2r = 1 - ((Ar - pred) ** 2).sum(0) / sst
    nan = torch.full((H,), float("nan"), device=dev)
    ev = {}
    if real.get("events") is not None:
        E = torch.as_tensor(real["events"], device=dev).float()              # (Sr, n_events), NaN where no calendar
        for k in range(E.shape[1]):
            m = torch.isfinite(E[:, k])
            if int(m.sum()) < 50 or float(E[m, k].std()) < 1e-6:
                ev[f"ev{k}_corr"] = nan.cpu().numpy()
                continue
            e = E[m, k] - E[m, k].mean()
            a = Ar[m] - Ar[m].mean(0, keepdim=True)
            ev[f"ev{k}_corr"] = ((e[:, None] * a).sum(0) / (e.pow(2).sum().sqrt() * a.pow(2).sum(0).sqrt()).clamp(min=1e-12)).cpu().numpy()
    _, rtop, rmid, rnact, _ = _sorted_edges_and_exemplars(Ar)
    ev.update({"real_top_idx": rtop.T.cpu().numpy(), "real_top_act": torch.gather(Ar, 0, rtop).T.cpu().numpy(),
               "real_mid_idx": rmid.T.cpu().numpy(), "real_mid_act": torch.gather(Ar, 0, rmid).T.cpu().numpy(),
               "real_n_active": rnact.cpu().numpy()})
    return {**ev, "tr_corr": torch.where(ok, corr, nan).cpu().numpy(), "tr_r2": torch.where(ok, r2r, nan).cpu().numpy(),
            "tr_analog": an.cpu().numpy(), "tr_r2_syn": torch.where(ok, r2[u, ar], nan).cpu().numpy(),
            "real_mean": Ar.mean(0).cpu().numpy(), "real_active": (Ar > 0).float().mean(0).cpu().numpy()}


# ---------------------------------------------------------------------------------------------
# turn raw per-layer outputs into table rows
# ---------------------------------------------------------------------------------------------
def summarize_sign(out: dict, layer: int, sign: str, size: str) -> dict:
    """Per-pole columns from the full (variable x neuron) arrays of one layer and sign."""
    prs = pair_index(list(range(len(V.CORE))))
    names = V.LIBRARY
    r2 = np.nan_to_num(out["_r2"], nan=-1.0)                         # (nv, H)
    H = r2.shape[1]
    order = np.argsort(-r2, axis=0)
    b1, b2 = order[0], order[1]
    ar = np.arange(H)
    r2_1, r2_2 = r2[b1, ar], r2[b2, ar]
    mi = out["_mi"]
    mi_best = np.argmax(np.nan_to_num(mi, nan=-1.0), axis=0)
    r2a = np.nan_to_num(out["_r2_act"], nan=-1.0)
    ba = np.argmax(r2a, axis=0)
    rho = out["_r2_ho"]
    fold = out["_r2_fold"]
    dead = out["n_active_label"] < MIN_ACTIVE
    row = {
        "size": size, "layer": layer, "neuron": ar, "sign": sign,
        "n_active_label": out["n_active_label"], "mean": out["mean"], "std": out["std"], "max": out["max"],
        "p999": out["p999"], "frac_active_hi": out["frac_active_hi"],
        "label": [names[i] for i in b1], "label_corr": out["_corr"][b1, ar], "r2_cv": r2_1, "label_runner_up": [names[i] for i in b2],
        "r2_runner_up": r2_2, "gap": r2_1 - r2_2,
        "mi_of_label": mi[b1, ar], "label_mi": [names[i] for i in mi_best], "mi_best": mi[mi_best, ar],
        "label_active": [names[i] for i in ba], "r2_active": r2a[ba, ar],
        "r2_heldout": rho[b1, ar], "r2_fold0": fold[0][b1, ar], "r2_fold1": fold[1][b1, ar],
        "r2_fold2": fold[2][b1, ar],
        "pair_label": [f"{V.CORE[prs[i][0]]}|{V.CORE[prs[i][1]]}" for i in out["pair_idx"]],
        "r2_pair": out["pair_r2"], "r2_pair_heldout": out["pair_r2_ho"],
        "r2_add": out["r2_add"], "r2_add_heldout": out["r2_add_ho"],
        "mass_g0u0": out["mass_g0u0"], "mass_g0u1": out["mass_g0u1"], "mass_g1u0": out["mass_g1u0"],
        "mass_g1u1": out["mass_g1u1"],
    }
    if "tr_corr" in out:
        row["tr_corr"], row["tr_r2"] = out["tr_corr"], out["tr_r2"]
        row["tr_analog"] = [names[i] if i >= 0 else "" for i in out["tr_analog"]]
        row["tr_r2_syn"], row["real_mean"], row["real_active"] = out["tr_r2_syn"], out["real_mean"], out["real_active"]
        for k in range(3):
            if f"ev{k}_corr" in out:
                row[f"ev{k}_corr"] = out[f"ev{k}_corr"]
    kinds = {"position": [i for i, n_ in enumerate(names) if GROUP_OF[n_] == "position"],
             "motif": [i for i, n_ in enumerate(names) if n_.startswith("m_")],
             "family": [i for i, n_ in enumerate(names) if GROUP_OF[n_] == "family"],
             "latent": [i for i, n_ in enumerate(names) if n_.startswith(("lat_", "prev_", "s_", "x_"))]}
    for k_, ix_ in kinds.items():
        row[f"r2_best_{k_}"] = r2[ix_].max(0)
    for tag in ("regA", "regB", "ctx", "fc"):
        rr = np.nan_to_num(out[f"_r2_{tag}"], nan=-1.0)
        bi = np.argmax(rr, axis=0)
        row[f"label_{tag}"] = [names[i] for i in bi]
        row[f"r2_{tag}"] = rr[bi, ar]
    for g in range(10):
        row[f"pos_mean_g{g}"] = out["pos_mean"][:, g]
        row[f"pos_active_g{g}"] = out["pos_active"][:, g]
    row["dead"] = dead
    for k in ("label_corr", "r2_cv", "r2_runner_up", "gap", "mi_of_label", "mi_best", "r2_active", "r2_heldout", "r2_fold0",
              "r2_fold1", "r2_fold2", "r2_pair", "r2_pair_heldout", "r2_add", "r2_add_heldout", "r2_regA",
              "r2_regB", "r2_ctx", "r2_fc", "r2_best_position", "r2_best_motif", "r2_best_family",
              "r2_best_latent"):
        row[k] = np.where(dead, np.nan, row[k])
    for k in ("label", "label_runner_up", "label_mi", "label_active", "pair_label", "label_regA", "label_regB", "label_ctx",
              "label_fc"):
        row[k] = ["dead" if d else v for v, d in zip(row[k], dead)]
    return row
