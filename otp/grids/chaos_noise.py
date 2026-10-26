"""Noise-level axis for chaotic generators: how the ratio to the oracle depends on noise.

With little noise the oracle is nearly exact, so any forecaster's ratio to it grows; with
more noise the oracle's own error grows. Dynamic and observation noise are set equal.
"""
from otp.generators.chaos import Lorenz63, MackeyGlass, Rossler

NOISE = (0.001, 0.003, 0.01, 0.03, 0.1)


def configs():
    out = []
    for nz in NOISE:
        out.append((f"chaos_noise/mackey_glass_tau17/n{nz:g}", lambda s, nz=nz: MackeyGlass(tau=17.0, dyn_rel=nz, obs_rel=nz)))
        out.append((f"chaos_noise/lorenz63_rho28/n{nz:g}", lambda s, nz=nz: Lorenz63(rho=28.0, dyn_rel=nz, obs_rel=nz)))
        if nz <= 0.01:  # Rossler leaves its attractor at higher dynamic noise
            out.append((f"chaos_noise/rossler_c5.7/n{nz:g}", lambda s, nz=nz: Rossler(c=5.7, dyn_rel=nz, obs_rel=nz)))
    return out
