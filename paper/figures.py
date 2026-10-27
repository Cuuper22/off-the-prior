"""Paper figures, drawn only from committed result files (sources listed in figures/sources.json).

python paper/figures.py

Encoding: model size is ordered, so it uses one blue ramp (light = 5M, dark = 209M; validated as
an ordinal ramp on the light surface). A dashed gray line marks a ratio of 1 (as good as the
oracle). One y-axis per panel, log scale for ratios.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "paper", "figures")
SIZES = ["timestral-5M", "timestral-28M", "timestral-123M", "timestral-209M"]
RAMP = {"timestral-5M": "#86b6ef", "timestral-28M": "#3987e5", "timestral-123M": "#1c5cab", "timestral-209M": "#0d366b"}
LABEL = {s: s.replace("timestral-", "") for s in SIZES}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SOURCES: dict[str, list[str]] = {}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "lines.linewidth": 2})


def p(*a):
    return os.path.join(ROOT, *a)


def ref1(ax):
    ax.axhline(1.0, color=INK2, lw=1, ls="--", zorder=0)


def plain_log(ax, x=False):
    """Log axes with plain-number tick labels (1, 2, 5, 10 ... instead of 10^0 notation)."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    axes = [ax.yaxis] + ([ax.xaxis] if x else [])
    for a in axes:
        a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        a.set_major_formatter(fmt)
        a.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def save(fig, name, srcs):
    os.makedirs(OUT, exist_ok=True)
    for ax in fig.axes:
        if ax.get_yscale() == "log":
            plain_log(ax, x=ax.get_xscale() == "log")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    SOURCES[name] = srcs


def lyapunov():
    src = "results/map/chaos/summary_all_sizes.csv"
    s = pd.read_csv(p(src))
    lam = float(pd.read_csv(p("results/map/chaos/lyapunov.csv")).query("config_id == 'lorenz63/rho28'").lyap_per_time.mean())
    dt = {"lorenz63/rho28_oe2": 0.02, "lorenz63/rho28_oe5": 0.05, "lorenz63/rho28": 0.1, "lorenz63/rho28_oe20": 0.2}
    q = s[(s["T"] == 2048) & s.config_id.isin(dt)].copy()
    q["reach"] = q.config_id.map(dt) * q.H * lam
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for fc in SIZES:
        g = q[q.forecaster == fc].sort_values("reach")
        ax.plot(g.reach, g.mase_ratio_oracle_gmean, "o-", ms=4, color=RAMP[fc], label=LABEL[fc])
    ref1(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("forecast reach (Lyapunov times)")
    ax.set_ylabel("MASE / oracle MASE")
    ax.set_title("Lorenz (rho = 28), four sampling intervals", fontsize=9, color=INK)
    ax.legend(title="Timestral", fontsize=8, title_fontsize=8)
    save(fig, "lorenz_lyapunov.png", [src, "results/map/chaos/lyapunov.csv"])


def chaos_curves():
    src = "results/map/chaos/summary_all_sizes.csv"
    s = pd.read_csv(p(src))
    q = s[(s["T"] == 2048) & (s.H == 128)].copy()
    q["kv"] = pd.to_numeric(q.knob_value, errors="coerce")
    panels = [("mackey_glass", "delay tau"), ("rossler", "c"), ("kuramoto", "coupling K"), ("lorenz63", "rho")]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, (gen, xl) in zip(axes, panels):
        g = q[(q.generator == gen) & ~q.config_id.str.contains("_oe")]
        for fc in SIZES:
            h = g[g.forecaster == fc].sort_values("kv")
            ax.plot(h.kv, h.mase_ratio_oracle_gmean, "o-", ms=3, color=RAMP[fc], label=LABEL[fc])
            ax.fill_between(h.kv, h.mase_ratio_oracle_lo95, h.mase_ratio_oracle_hi95, color=RAMP[fc], alpha=0.12, lw=0)
        ref1(ax)
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_title(gen.replace("63", "").replace("_", "-"), fontsize=9, color=INK)
    axes[0].set_ylabel("MASE / oracle MASE")
    axes[-1].legend(title="Timestral", fontsize=7, title_fontsize=7)
    save(fig, "chaos_curves.png", [src])


def regime_distance():
    src = "results/map/regime/summary_all_sizes.csv"
    if not os.path.exists(p(src)):
        return
    s = pd.read_csv(p(src))
    q = s[(s["T"] == 8192) & (s.H == 128)].copy()
    q["d"] = pd.to_numeric(q.config_id.str.extract(r"/d(\d+)$")[0], errors="coerce")
    picks = [("regime/mean_jump/context/delta4/", "mean jump (4 sd)"), ("regime/frequency_jump/context/ratio2/", "period x2"),
             ("regime/variance_jump/context/factor0.25/", "noise sd x0.25")]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8), sharey=True)
    for ax, (prefix, title) in zip(axes, picks):
        g = q[q.config_id.str.startswith(prefix)]
        for fc in SIZES:
            h = g[g.forecaster == fc].sort_values("d")
            ax.plot(h.d, h.mase_ratio_oracle_gmean, "o-", ms=3, color=RAMP[fc], label=LABEL[fc])
        ref1(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("steps since the switch")
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel("MASE / oracle MASE")
    axes[-1].legend(title="Timestral", fontsize=7, title_fontsize=7)
    save(fig, "regime_distance.png", [src])


def noise():
    src = "results/mechanism/chaos_noise/summary.csv"
    s = pd.read_csv(p(src))
    s["system"] = s.config_id.str.split("/").str[1]
    s["noise"] = s.config_id.str.extract(r"/n([\d.]+)$")[0].astype(float) * 100
    q = s[s.H == 128]
    names = {"mackey_glass_tau17": "Mackey-Glass (tau = 17)", "rossler_c5.7": "Rossler (c = 5.7)", "lorenz63_rho28": "Lorenz (rho = 28)"}
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8))
    for ax, (k, title) in zip(axes, names.items()):
        g = q[q.system == k]
        for fc in SIZES:
            h = g[g.forecaster == fc].sort_values("noise")
            ax.plot(h.noise, h.mase_ratio_oracle_gmean, "o-", ms=3, color=RAMP[fc], label=LABEL[fc])
        ref1(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("noise (% of signal scale)")
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel("MASE / oracle MASE")
    axes[-1].legend(title="Timestral", fontsize=7, title_fontsize=7)
    save(fig, "noise_sweep.png", [src])


def readouts():
    src = "results/mechanism/readouts/summary.csv"
    s = pd.read_csv(p(src))
    g = s[(s.study == "mean_jump_d") & s.valid]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for size in ("5M", "209M"):
        fc = f"timestral-{size}"
        h = g[g["size"] == size].sort_values("knob_value")
        ax.plot(h.knob_value, h.best_readout_r2, "o--", ms=4, color=RAMP[fc], label=f"{size}: linear readout")
        ax.plot(h.knob_value, h.output_r2, "s-", ms=4, color=RAMP[fc], label=f"{size}: forecast")
    ax.set_xscale("log")
    ax.set_xlabel("steps since a mean jump")
    ax.set_ylabel("R² for the oracle's expected change")
    ax.set_title("Represented but not used", fontsize=9, color=INK)
    ax.legend(fontsize=7)
    save(fig, "readout_vs_output.png", [src])


def covariates():
    src = "results/covariates/summary.csv"
    s = pd.read_csv(p(src))
    g = s[(s.system == "LaggedDriver") & (s["T"] == 2048) & (s.H == 16) & (s.condition == "known_cov")]
    lags = [0, 8, 32, 128, 512]
    x = np.arange(len(lags))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for size in ("5M", "28M"):
        fc = f"timestral-{size}"
        h = g[g["size"] == size].set_index(g[g["size"] == size].knob_value.astype(float)).loc[[float(v) for v in lags]]
        ax.plot(x, h.covariate_gain_gmean, "o-", ms=4, color=RAMP[fc], label=f"{size}: Timestral")
        ax.plot(x, h.oracle_gain_gmean, "s--", ms=4, color=RAMP[fc], label=f"{size}: matched oracle")
    ref1(ax)
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_xticks(x, [str(v) for v in lags])
    ax.set_xlabel("lag between driver and main series (steps)")
    ax.set_ylabel("gain from the known covariate")
    ax.set_title("Known-future driver: gain fades with lag", fontsize=9, color=INK)
    ax.legend(fontsize=7)
    save(fig, "covariate_lag.png", [src])


def main():
    for fn in (lyapunov, chaos_curves, regime_distance, noise, readouts, covariates):
        fn()
    with open(os.path.join(OUT, "sources.json"), "w") as f:
        f.write(json.JSONEncoder(indent=1).encode(SOURCES) + "\n")
    print("figures:", ", ".join(SOURCES))


if __name__ == "__main__":
    main()
