"""Write the benchmark config lists (bench/configs/*.json). Each entry fully determines one series:
set, split, config id (with the grid module that defines it), seed index, context length T and
horizon H, plus tags and a plain-English reason for the diagnostic set.

python bench/make_configs.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---- OffPrior-Diag: configs chosen to cover each failure type found in the map, plus controls ----
DIAG = [
    # (grid module, config id, failure type, reason)
    ("otp.grids.chaos", "mackey_glass/tau17", "delay_oscillator", "smooth delay-feedback oscillation; Timestral leans toward a periodic template and a nearest-neighbour forecaster beats it"),
    ("otp.grids.chaos", "mackey_glass/tau20", "delay_oscillator", "longer delay, less regular cycles; same pattern as tau = 17, larger gap"),
    ("otp.grids.chaos", "mackey_glass/tau30", "delay_oscillator", "long-delay chaos in the delay equation; the largest Mackey-Glass gap in the map"),
    ("otp.grids.chaos", "rossler/c5.7", "smooth_chaos", "Rossler chaos: near-periodic loops whose timing drifts"),
    ("otp.grids.chaos", "rossler/c9", "smooth_chaos", "Rossler with more irregular loop sizes"),
    ("otp.grids.chaos", "kuramoto/K0", "coupled_oscillators", "incoherent coupled oscillators: a sum of many drifting phases"),
    ("otp.grids.chaos", "kuramoto/K1", "coupled_oscillators", "partially synchronized oscillators; the largest gap to the oracle in the map"),
    ("otp.grids.chaos", "kuramoto/K1.5", "coupled_oscillators", "near the synchronization threshold"),
    ("otp.grids.chaos", "lorenz63/rho28", "chaos_short_reach", "Lorenz chaos: the gap lives within one to three Lyapunov times"),
    ("otp.grids.chaos", "lorenz63/rho28_oe2", "chaos_short_reach", "densely sampled Lorenz: a 128-step horizon is about two Lyapunov times"),
    ("otp.grids.chaos", "logistic_map/r3.5", "periodic_map", "period-4 cycle of the logistic map (classical models fail here; Timestral does not)"),
    ("otp.grids.chaos", "logistic_map/r4", "control_unpredictable", "fully chaotic map: predictability is gone after a few steps, so nobody beats the mean"),
    ("otp.grids.regime", "regime/mean_jump/context/delta2/d16", "recent_switch", "level jumped 16 steps ago; the change is represented inside the model but not used"),
    ("otp.grids.regime", "regime/mean_jump/context/delta4/d16", "recent_switch", "large level jump 16 steps ago"),
    ("otp.grids.regime", "regime/mean_jump/context/delta2/d64", "recent_switch", "level jumped 64 steps ago"),
    ("otp.grids.regime", "regime/frequency_jump/context/ratio1.5/d16", "recent_switch", "seasonal period changed 16 steps ago"),
    ("otp.grids.regime", "regime/frequency_jump/context/ratio2/d64", "recent_switch", "seasonal period doubled 64 steps ago"),
    ("otp.grids.regime", "regime/frequency_jump/context/ratio0.75/d64", "recent_switch", "seasonal period shortened 64 steps ago"),
    ("otp.grids.regime", "regime/variance_jump/context/factor0.25/d64", "recent_switch", "noise dropped to a quarter 64 steps ago: intervals stay too wide"),
    ("otp.grids.regime", "regime/markov_ar/p0.003", "switching", "Markov switching between two AR regimes"),
    ("otp.grids.regime", "regime/mean_jump/horizon/delta4", "control_unknowable", "a jump may happen during the horizon at a known rate: no forecaster can see it coming"),
    ("otp.grids.regime", "regime/mean_jump/context/delta4/d1024", "control_old_switch", "jump long ago: back to normal"),
    ("otp.grids.tails", "tails/stable_ar/alpha1.3_rho1.0", "heavy_tail_walk", "random walk with alpha-stable (alpha = 1.3) steps"),
    ("otp.grids.tails", "tails/fbm/H0.8", "long_memory", "fractional Brownian motion with long memory"),
    ("otp.grids.tails", "tails/msm/k8_cum", "multifractal", "cumulative multifractal cascade"),
    ("otp.grids.tails", "tails/hawkes/eta0.95", "bursty", "near-critical self-exciting counts"),
    ("otp.grids.tails", "tails/stable_ar/alpha2_rho0.9", "control_in_prior", "Gaussian AR(1): inside the prior"),
    ("otp.grids.tails", "tails/fbm/H0.5", "control_in_prior", "ordinary random walk"),
    ("otp.grids.in_prior", "in_prior/p24", "control_in_prior", "trend + season (period 24) + AR noise, like the pretraining data"),
    ("otp.grids.in_prior", "in_prior/p96", "control_in_prior", "trend + season (period 96) + AR noise"),
    ("otp.grids.in_prior", "in_prior/p168", "control_in_prior", "trend + season (period 168) + AR noise"),
]
DIAG_SEEDS = 12
DIAG_T, DIAG_H = 2048, 128

# ---- OffPrior-LB: every grid config, seeds spread over context/horizon shapes ----
LB_GRIDS = ["otp.grids.chaos", "otp.grids.regime", "otp.grids.tails", "otp.grids.in_prior", "otp.grids.bench_extra"]
# seed index -> (T, H); 16 seeds per config
LB_SHAPES = [(2048, 128)] * 6 + [(512, 128)] * 2 + [(8192, 128)] * 2 + [(2048, 1024)] * 3 + [(8192, 1024)] + [(2048, 16)] * 2


def main():
    import importlib

    os.makedirs(os.path.join(ROOT, "bench", "configs"), exist_ok=True)
    diag = []
    for mod, cid, ftype, reason in DIAG:
        assert cid in dict(importlib.import_module(mod).configs()), cid
        for k in range(DIAG_SEEDS):
            diag.append({"set": "diag", "split": "public", "grid": mod, "config_id": cid, "seed_index": k,
                         "T": DIAG_T, "H": DIAG_H, "failure_type": ftype, "reason": reason})
    lb = []
    for mod in LB_GRIDS:
        for cid, _ in importlib.import_module(mod).configs():
            for k, (T, H) in enumerate(LB_SHAPES):
                lb.append({"set": "lb", "split": "public" if k % 2 == 0 else "heldout", "grid": mod, "config_id": cid,
                           "seed_index": k, "T": T, "H": H})
    for name, rows in (("diag", diag), ("lb", lb)):
        with open(os.path.join(ROOT, "bench", "configs", f"{name}.json"), "w") as f:
            f.write(json.JSONEncoder(indent=0).encode(rows) + "\n")
        print(f"{name}: {len(rows)} series")


if __name__ == "__main__":
    main()
