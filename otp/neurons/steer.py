"""Steering: scale a pole during the forward pass and measure how the forecast changes.

Scaling a positive pole by `c` multiplies `z` by `c` on every token where `z > 0` (a negative pole: where `z < 0`),
leaving the rest of the neuron untouched; `c = 0` is ablation. The effect is read from the quantile forecast of the
same plain forward pass (no symmetric averaging), 128 steps, in raw units divided by the context standard deviation.

Measured quantities (differences between the steered and the unsteered forecast, per series):

- `level`   mean over the horizon of the median forecast
- `slope`   OLS slope of the median forecast times 128 (total drift over the horizon)
- `spread`  log ratio of the mean 5%-95% interval width
- `amp`     amplitude of the sinusoid at the series' known period fitted to the median forecast (seasonal series)
- `phase`   phase of that sinusoid (radians, circular difference)

Pre-registration: `PREREG` maps a label's group to the predicted quantity and direction; `CRITERION` fixes the matching
rule. `prereg_markdown()` writes the table committed in `results/neurons/steering_preregistration.md`.
"""

from __future__ import annotations

import numpy as np
import torch

from otp.neurons.capture import forward_with_mlp

SCALES = (0.0, 0.5, 2.0, 4.0)
H_STEPS = 128
QUANTITIES = ("level", "slope", "spread", "amp", "phase")
THRESH = {"level": 0.05, "slope": 0.05, "spread": 0.05, "amp": 0.05, "phase": 0.10}
RHO_MIN = 0.8
N_BOOT = 500
THETA_SAMPLE, FLOOR_SAMPLE = 0.7, 0.1      # a pole enters the labeled sample when its class at these settings is one_job
PER_CELL = 8                                # poles per (layer third x predicted quantity group x size)
N_CONTROL = 144                             # control poles per size
N_COMPLEMENT = 4                            # per cell, the other pole of the same neuron is steered too
N_MIXED, N_SEASONAL = 96, 64                # test series
SAMPLE_SEED = 0
GROUPS = ("level", "slope", "spread", "amp", "phase", "level|spread")

# label -> (quantity, direction rule). direction: "+" = same sign as the pole-label correlation, "-" = opposite, "0" = any
PREREG: dict[str, tuple[str, str]] = {}
for _n in ("lat_level", "prev_level", "m_level_ctx", "m_dlevel_prev"):
    PREREG[_n] = ("level", "+")
for _n in ("lat_slope", "prev_slope", "m_slope", "s_trend", "x_trend"):
    PREREG[_n] = ("slope", "+")
for _n in ("lat_noise_sd", "prev_noise_sd", "lat_inn_sd", "m_logsd", "m_resid_sd", "m_roughness", "s_noise_level",
           "x_noise_level", "lat_outlier", "lat_outlier_mag", "m_max_z", "m_jump"):
    PREREG[_n] = ("spread", "+")
for _n in ("lat_pred1", "s_alpha", "x_tail"):
    PREREG[_n] = ("spread", "-")
for _n in ("s_season_amp", "x_season_strength", "s_has_season"):
    PREREG[_n] = ("amp", "+")
for _n in ("lat_phase_sin", "lat_phase_cos", "prev_phase_sin", "prev_phase_cos", "x_phase_sin", "x_phase_cos"):
    PREREG[_n] = ("phase", "0")
for _n in ("lat_regime", "lat_regime_frac", "prev_regime", "x_post_cp"):
    PREREG[_n] = ("level|spread", "0")

CRITERION = (
    "A steered pole matches its pre-registered prediction when all three hold for the predicted quantity Q: "
    f"(1) monotone: |Spearman rho| between the scale list {list(SCALES)} and the mean effect on Q is at least {RHO_MIN}, "
    "with the sign of rho equal to the predicted direction when a direction is given; "
    "(2) size: |effect(scale 4) - effect(scale 0)| is at least the threshold of Q "
    f"({', '.join(f'{k} {v}' for k, v in THRESH.items())}); "
    f"(3) the 95% bootstrap interval ({N_BOOT} resamples of test series) of effect(scale 4) - effect(scale 0) excludes 0. "
    "For the label group with quantity 'level|spread' the three conditions are checked for each of the two quantities "
    "and the pole matches if either passes (direction not prescribed)."
)


def group_of_label(label: str) -> str | None:
    q = PREREG.get(label)
    return None if q is None else q[0]


def prereg_markdown() -> str:
    rows = {}
    for lab, (q, d) in PREREG.items():
        rows.setdefault((q, d), []).append(lab)
    lines = ["# Steering pre-registration", "",
             "Committed before any steering run. The predictions below are fixed; the matching criterion is fixed.", "",
             "## What scaling does", "",
             "For a pole (neuron, sign) and a scale c, `z = gate * silu(up)` of that neuron is multiplied by c on every "
             "token where `z` has the pole's sign, in the plain forward pass (no symmetric averaging). "
             f"Scales: {list(SCALES)} (0 is ablation, 1 is the unsteered pass). Effects are differences between the "
             "steered and unsteered 128-step median / quantile forecast, divided by the context standard deviation.", "",
             "## Quantities", "",
             "- level: mean over the horizon of the median forecast",
             "- slope: OLS slope of the median forecast times 128",
             "- spread: log ratio of the mean 5%-95% interval width",
             "- amp: amplitude of the sinusoid at the series' known period fitted to the median forecast (seasonal test series)",
             "- phase: phase of that sinusoid, circular difference in radians (seasonal test series)", "",
             "## Predictions by label", "",
             "Direction `+` means: increasing the pole moves Q in the direction in which the label variable moves it "
             "(sign of the correlation between the pole's activation and the label variable on label-set tokens); "
             "`-` the opposite; `0` any direction (monotone in scale is still required).", "",
             "| predicted quantity | direction | labels |", "|---|---|---|"]
    for (q, d), labs in rows.items():
        lines.append(f"| {q} | {d} | {', '.join(sorted(labs))} |")
    lines += ["", "Labels not listed (periods, family and generator indicators, position, persistence, Lyapunov, "
              "clustering, coupling, motif shapes without a direct forecast quantity) have no prediction; "
              "their steered poles are reported without a match decision.", "",
              "## Matching criterion", "", CRITERION, "",
              "## Pole sample", "",
              f"Per model size, from the pole table of the main-context run. Candidates: live poles that are one-job at "
              f"theta {THETA_SAMPLE} and floor {FLOOR_SAMPLE} (explained share of the explainable variance, see "
              "`otp/neurons/score.py`) whose best label is in the table above. Cells: layer third (early / middle / "
              f"late) x predicted-quantity group ({', '.join(GROUPS)}). In each cell {PER_CELL} candidate poles are "
              f"drawn uniformly at random (seed {SAMPLE_SEED}), not by score; a cell with fewer candidates contributes "
              f"all of them. For the first {N_COMPLEMENT} poles of each cell the other pole of the same neuron is "
              "steered as well (negative-pole consistency, below).", "",
              "## Control", "",
              f"{N_CONTROL} control poles per size are drawn uniformly at random from the live poles of the size, "
              "whatever their label (unexplained poles included). Each control pole is judged with the same criterion "
              "for each of the quantities level, slope and spread, with a random direction (sign drawn once from "
              f"seed {SAMPLE_SEED}). The fraction of (pole, quantity) pairs that pass is the null match rate reported "
              "next to the match rate of the labeled poles.", "",
              "## Test series", "",
              f"Held-out series (never used for labeling) of the audit corpus, context 1024 steps. {N_MIXED} series "
              f"drawn at random from all held-out univariate series for level, slope and spread; {N_SEASONAL} "
              "held-out series with a seasonal component of period 8..64 steps for amp and phase. Fixed seed.", "",
              "## Consistency checks (reported next to the match rate)", "",
              "- Downstream: the effect of ablation (scale 0) on each of the five quantities is stored for every steered "
              "pole. A label is flagged as broken by this check when the pole matches no prediction and its largest "
              "ablation effect, in units of that quantity's size threshold, is on a quantity other than the predicted one.",
              "- Negative-pole consistency: for neurons with both poles steered, whose two poles have the same best "
              "label variable, the scale-4 effects of the two poles on the predicted quantity should have opposite "
              "signs when the two label correlations have opposite signs, and equal signs when they have the same sign "
              "(each pole's predicted direction follows the sign of its own label correlation). The share of "
              "such neurons that satisfy this is reported.",
              "- Upstream: the three earlier-layer poles most correlated with the pole on a fixed subsample of "
              "tokens (`scripts/neurons_match.py`); a label is flagged when none of them has a label in the same "
              "group and the best motif label of the pole explains less than half of its label's R^2.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# running steered forward passes
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def run_conditions(size: str, ctx: np.ndarray, conditions: list[list[tuple[int, int, int, float]]],
                   device: str | None = None, max_rows: int = 2048) -> np.ndarray:
    """Quantile forecasts for each condition on every context. ctx (B, T). A condition is a list of
    (layer, neuron, sign (+1 / -1), scale). Returns (n_cond, B, 128, 21). An empty condition is the unsteered pass."""
    B = ctx.shape[0]
    per = max(1, max_rows // B)
    out = np.zeros((len(conditions), B, H_STEPS, 21), np.float32)
    x_all = torch.as_tensor(ctx, dtype=torch.float32)
    for c0 in range(0, len(conditions), per):
        conds = conditions[c0:c0 + per]
        x = x_all.repeat(len(conds), 1)
        plan: dict[int, list] = {}
        for k, cond in enumerate(conds):
            for (li, nn, sg, sc) in cond:
                plan.setdefault(li, []).append((k, nn, sg, sc))

        def edit(li, z, gate, up, plan=plan, B=B):
            if li not in plan:
                return None
            z2 = z.clone()
            for (k, nn, sg, sc) in plan[li]:
                col = z2[k * B:(k + 1) * B, :, nn]
                z2[k * B:(k + 1) * B, :, nn] = torch.where(sg * col > 0, sc * col, col)
            return z2

        q, _, _ = forward_with_mlp(size, x, device=device, keep=(), edit=edit)
        out[c0:c0 + len(conds)] = q.reshape(len(conds), B, H_STEPS, 21).float().cpu().numpy()
    return out


# ---------------------------------------------------------------------------------------------
# effect measures
# ---------------------------------------------------------------------------------------------
def measures(ctx: np.ndarray, q: np.ndarray, period: np.ndarray | None) -> dict[str, np.ndarray]:
    """Per-series measures of a forecast. ctx (B, T), q (B, 128, 21); period (B,) in steps or None."""
    sd = np.maximum(ctx.std(1), 1e-9)
    med = q[..., 10] / sd[:, None]
    t = np.arange(H_STEPS)
    tc = t - t.mean()
    m = {"level": med.mean(1), "slope": (med * tc).sum(1) / (tc ** 2).sum() * H_STEPS,
         "spread": np.log(np.maximum((q[..., 19] - q[..., 1]).mean(1) / sd, 1e-9))}
    if period is not None:
        T = ctx.shape[1]
        w = 2 * np.pi / period[:, None]
        tt = (T + t)[None, :]
        sn, cs = np.sin(w * tt), np.cos(w * tt)
        mc = med - med.mean(1, keepdims=True)
        a = 2 * (mc * sn).mean(1)
        b = 2 * (mc * cs).mean(1)
        m["amp"] = np.sqrt(a ** 2 + b ** 2)
        m["phase"] = np.arctan2(b, a)
    return m


def effects(base: dict, steered: dict) -> dict[str, np.ndarray]:
    out = {}
    for k in ("level", "slope", "spread", "amp"):
        if k in base:
            out[k] = steered[k] - base[k]
    if "phase" in base:
        d = steered["phase"] - base["phase"]
        out["phase"] = (d + np.pi) % (2 * np.pi) - np.pi
    return out


def _spearman4(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if np.allclose(y, y[0]):
        return 0.0
    r = spearmanr(x, y).statistic
    return 0.0 if not np.isfinite(r) else float(r)


def _circ_mean(d: np.ndarray) -> float:
    return float(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))


def judge(eff_by_scale: list[dict], quantity: str, direction: str, sign_corr: float, seed: int = 0) -> dict:
    """Apply the pre-registered criterion. eff_by_scale[i][quantity]: per-series effects at SCALES[i]."""
    qs = quantity.split("|")
    best = None
    rng = np.random.default_rng(seed)
    for q in qs:
        if q not in eff_by_scale[0]:
            continue
        mean_fn = _circ_mean if q == "phase" else np.mean
        e = np.array([mean_fn(eff_by_scale[i][q]) for i in range(len(SCALES))])
        rho = _spearman4(np.array(SCALES), e)
        size = float(e[-1] - e[0])
        n = len(eff_by_scale[0][q])
        diffs = []
        for _ in range(N_BOOT):
            ix = rng.integers(0, n, n)
            diffs.append(mean_fn(eff_by_scale[-1][q][ix]) - mean_fn(eff_by_scale[0][q][ix]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        if q == "phase":
            size = float((e[-1] - e[0] + np.pi) % (2 * np.pi) - np.pi)
        want = 0 if direction == "0" else (1 if direction == "+" else -1) * (1 if sign_corr >= 0 else -1)
        mono = abs(rho) >= RHO_MIN and (want == 0 or np.sign(rho) == want)
        big = abs(size) >= THRESH[q]
        sig = (lo > 0) or (hi < 0)
        res = {"quantity": q, "rho": rho, "size": size, "ci_lo": float(lo), "ci_hi": float(hi), "monotone": bool(mono),
               "big": bool(big), "sig": bool(sig), "match": bool(mono and big and sig), "e": e.tolist()}
        if best is None or (res["match"] and not best["match"]) or (res["match"] == best["match"] and abs(size) > abs(best["size"])):
            best = res
    return best or {"quantity": quantity, "match": False}
