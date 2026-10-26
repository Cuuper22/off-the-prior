"""Train / test splits for the fine-tuning study (committed config, imported by the scripts).

Four failure families: `chaos`, `regime`, `tails`, `mixture`. Two protocols:

1. `lofo` (leave one family out): train on the other three families, test on the held-out one.
   Test samples come from every config of the held-out family.
2. `knob` (held-out knob ranges): inside each family, train on one range of the main difficulty
   knob and test on another range. Values between the two ranges are left out of both sides.
   Rules per family (implemented in `knob_side` and `side_configs`):

   - chaos: Lorenz63 rho <= 24 | >= 28; Rossler c <= 4.5 | >= 5.7; Mackey-Glass tau <= 14 | >= 20;
     logistic map r <= 3.6 | >= 3.83; Kuramoto K <= 1.5 | >= 3.
   - regime: switch distance d >= 256 | d <= 64 (context-placed switches only; the horizon-placed
     and Markov-switching configs are in neither side).
   - tails: stable-AR alpha >= 1.5 | <= 1.3; fBm Hurst in {0.3, 0.5, 0.7} | in {0.1, 0.2, 0.8, 0.9};
     Hawkes eta <= 0.6 | >= 0.9; multifractal cascade k <= 4 | >= 6.
   - mixture: no config grid. The knob is the observation-noise gene (last genome entry):
     < 0.5 train | >= 0.5 test.

Seed ranges (disjoint): training draws seeds in `TRAIN_SEEDS`; evaluation samples use seeds from
`EVAL_SEED_BASE` upward; the in-prior regression series use seeds from `REGRESSION_SEED_START`.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

FAMILIES = ("chaos", "regime", "tails", "mixture")
PROTOCOLS = ("lofo", "knob")

TRAIN_SEEDS = (1_000_000, 2_000_000)
EVAL_SEED_BASE = 5_000_000
REGRESSION_SEED_START = 100_000
EVAL_T, EVAL_H = 2048, 128
N_EVAL_DEFAULT = 32
N_REGRESSION_DEFAULT = 300


# ---------------------------------------------------------------------------
# Config grids per family
# ---------------------------------------------------------------------------
def _mixture_factory(lo: float = 0.0, hi: float = 1.0):
    """seed -> random mixture process, with the observation-noise gene mapped into [lo, hi)."""
    def make(seed: int):
        from otp.generators import mixture

        rng = np.random.default_rng(seed)
        g = mixture.random_genome(rng)
        g[-1] = lo + (hi - lo) * g[-1]
        return mixture.decode(g)
    return make


def family_configs(family: str) -> list[tuple[str, object]]:
    """`[(config_id, factory)]` for a family; factory(seed) -> Process."""
    if family == "chaos":
        from otp.grids import chaos
        return chaos.configs()
    if family == "regime":
        from otp.grids import regime
        return regime.configs()
    if family == "tails":
        from otp.grids import tails
        return tails.configs()
    if family == "mixture":
        return [("mixture/random", _mixture_factory())]
    raise ValueError(family)


def _num(s: str) -> float:
    return float(s)


def knob_side(family: str, config_id: str) -> str | None:
    """'train', 'test' or None (left out) for a config under the held-out-knob protocol."""
    if family == "chaos":
        gen, rest = config_id.split("/")
        m = re.match(r"(rho|c|tau|r|K)([0-9.]+)", rest)
        v = _num(m.group(2))
        lim = {"lorenz63": (24.0, 28.0), "rossler": (4.5, 5.7), "mackey_glass": (14.0, 20.0),
               "logistic_map": (3.6, 3.83), "kuramoto": (1.5, 3.0)}[gen]
        return "train" if v <= lim[0] else "test" if v >= lim[1] else None
    if family == "regime":
        from otp.grids.regime import parse_config_id
        p = parse_config_id(config_id)
        if p["placement"] != "context":
            return None
        return "train" if p["d"] >= 256 else "test" if p["d"] <= 64 else None
    if family == "tails":
        parts = config_id.split("/")
        gen, rest = parts[1], parts[2]
        if gen == "stable_ar":
            a = _num(re.match(r"alpha([0-9.]+)", rest).group(1))
            return "train" if a >= 1.5 else "test" if a <= 1.3 else None
        if gen == "fbm":
            h = _num(rest[1:])
            return "train" if h in (0.3, 0.5, 0.7) else "test"
        if gen == "hawkes":
            e = _num(rest[3:])
            return "train" if e <= 0.6 else "test" if e >= 0.9 else None
        if gen == "msm":
            k = int(re.match(r"k(\d+)", rest).group(1))
            return "train" if k <= 4 else "test" if k >= 6 else None
    if family == "mixture":
        return None  # handled by the noise-gene factories below
    raise ValueError((family, config_id))



def side_configs(family: str, side: str) -> list[tuple[str, object]]:
    """Configs of a family on the 'train' or 'test' side of the held-out-knob protocol."""
    if family == "mixture":
        lo, hi = (0.0, 0.5) if side == "train" else (0.5, 1.0)
        return [(f"mixture/noise_gene_{lo:g}_{hi:g}", _mixture_factory(lo, hi))]
    return [(cid, f) for cid, f in family_configs(family) if knob_side(family, cid) == side]


# ---------------------------------------------------------------------------
# Training factories (one composite factory per family, so each family gets equal weight)
# ---------------------------------------------------------------------------
def _balanced(configs: list[tuple[str, object]]):
    """seed -> Process, choosing a config of the family uniformly (deterministic in the seed)."""
    ids = [c for c, _ in configs]
    facs = [f for _, f in configs]

    def make(seed: int):
        k = zlib.crc32(f"cfg/{seed}".encode()) % len(facs)
        return facs[k](seed)
    make.config_ids = ids
    return make


def train_families(protocol: str, holdout: str) -> list[str]:
    """Families that contribute failure data to a run."""
    if protocol == "lofo":
        return [f for f in FAMILIES if f != holdout]
    if protocol == "knob":
        return list(FAMILIES)
    raise ValueError(protocol)


def family_train_factory(protocol: str, family: str):
    """Factory (seed -> Process) for one family's part of the training mix: every config of the family
    under `lofo`, the train-side configs under `knob`."""
    if protocol == "lofo":
        return _balanced(family_configs(family))
    if protocol == "knob":
        return _balanced(side_configs(family, "train"))
    raise ValueError(protocol)


def test_configs(protocol: str, holdout: str) -> list[tuple[str, object]]:
    """Configs the held-out samples are drawn from."""
    if protocol == "lofo":
        return family_configs(holdout)
    if protocol == "knob":
        return side_configs(holdout, "test")
    raise ValueError(protocol)


def split_summary() -> dict:
    """The splits as plain data (config ids per side), written into provenance files."""
    out = {"lofo": {f: {"train_families": [g for g in FAMILIES if g != f], "test_family": f,
                        "n_test_configs": len(family_configs(f))} for f in FAMILIES},
           "knob": {}}
    for f in FAMILIES:
        tr = [c for c, _ in side_configs(f, "train")]
        te = [c for c, _ in side_configs(f, "test")]
        out["knob"][f] = {"train_configs": tr, "test_configs": te}
    out["seeds"] = {"train": list(TRAIN_SEEDS), "eval_base": EVAL_SEED_BASE, "regression_start": REGRESSION_SEED_START}
    out["eval"] = {"T": EVAL_T, "H": EVAL_H}
    return out


# ---------------------------------------------------------------------------
# Held-out sample sets
# ---------------------------------------------------------------------------
def make_eval_samples(protocol: str, holdout: str, n: int = N_EVAL_DEFAULT, n_paths: int = 500,
                      T: int = EVAL_T, H: int = EVAL_H) -> list:
    """`n` held-out samples with the true-dynamics oracle (500 paths). Config order and seeds are
    deterministic. Samples whose context, future or oracle quantiles are not finite are skipped."""
    from otp.generators.base import make_sample

    configs = test_configs(protocol, holdout)
    rng = np.random.default_rng(zlib.crc32(f"eval/{protocol}/{holdout}".encode()))
    base = EVAL_SEED_BASE + (zlib.crc32(f"base/{protocol}/{holdout}".encode()) % 1000) * 1000
    out, tries = [], 0
    while len(out) < n and tries < 20 * n:
        cid, fac = configs[int(rng.integers(len(configs)))]
        seed = base + tries
        tries += 1
        s = make_sample(fac(seed), T, H, seed=seed, n_paths=n_paths,
                        sample_id=f"{protocol}/{holdout}/{len(out):03d}")
        if not (np.isfinite(s.context).all() and np.isfinite(s.future).all() and np.isfinite(s.oracle_q).all()):
            continue
        if float(np.std(s.context)) < 1e-9:
            continue
        s.meta["config_id"] = cid
        s.meta["protocol"], s.meta["holdout"] = protocol, holdout
        out.append(s)
    return out


def make_regression_samples(n: int = N_REGRESSION_DEFAULT, n_paths: int = 500, T: int = EVAL_T, H: int = EVAL_H) -> list:
    """Fresh in-prior series (seeds REGRESSION_SEED_START + i) with the oracle."""
    from otp.generators.base import make_sample
    from otp.generators.primitives import in_prior

    out = []
    for i in range(n):
        seed = REGRESSION_SEED_START + i
        s = make_sample(in_prior(seed), T, H, seed=seed, n_paths=n_paths, sample_id=f"in_prior/{i:04d}")
        s.meta["config_id"] = "in_prior"
        out.append(s)
    return out
