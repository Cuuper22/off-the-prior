"""Regime-switch grid: mean, variance and frequency jumps (switch in context or in horizon) and Markov switching.

Config ids:
  regime/<generator>/context/<knob><value>/d<d>   scheduled switch d steps before the forecast origin
  regime/<generator>/horizon/<knob><value>        switch with constant hazard during the horizon
  regime/markov_ar/p<p_switch>                    Markov switching through context and horizon

`parse_config_id` recovers generator, placement, knob value and d from an id. `BIG_IDS` is the
reduced grid used for the 123M and 209M runs (see results/map/regime_big/README.md).
"""
from otp.generators.regime import FrequencyJump, MarkovSwitchingAR, MeanJump, VarianceJump, randomized_base

D_VALUES = (16, 64, 256, 1024, 4096, 16384)
KNOBS = {
    "mean_jump": (MeanJump, "delta", (0.5, 1, 2, 4, 8)),
    "variance_jump": (VarianceJump, "factor", (0.25, 0.5, 2, 4, 8)),
    "frequency_jump": (FrequencyJump, "ratio", (0.5, 0.75, 1.5, 2, 3)),
}
P_SWITCH = (0.001, 0.003, 0.01, 0.03, 0.1)
HAZARD = 1.0 / 256.0


def _g(v):
    return f"{v:g}"


def _factory(cls, **kw):
    return lambda seed: cls(**randomized_base(seed), **kw)


def configs():
    out = []
    for gen, (cls, knob, values) in KNOBS.items():
        for v in values:
            for d in D_VALUES:
                out.append((f"regime/{gen}/context/{knob}{_g(v)}/d{d}",
                            _factory(cls, placement="context", d=d, **{knob: v})))
            out.append((f"regime/{gen}/horizon/{knob}{_g(v)}",
                        _factory(cls, placement="horizon", hazard=HAZARD, **{knob: v})))
    for p in P_SWITCH:
        out.append((f"regime/markov_ar/p{_g(p)}", _factory(MarkovSwitchingAR, p_switch=p)))
    return out


def _big_ids():
    pick = {"mean_jump": (1, 2, 8), "variance_jump": (0.25, 2, 8), "frequency_jump": (0.5, 1.5, 3)}
    mid = {"mean_jump": 2, "variance_jump": 2, "frequency_jump": 1.5}
    ids = []
    for gen, (_, knob, _) in KNOBS.items():
        for v in pick[gen]:
            for d in (64, 1024):
                ids.append(f"regime/{gen}/context/{knob}{_g(v)}/d{d}")
        ids.append(f"regime/{gen}/horizon/{knob}{_g(mid[gen])}")
    ids += [f"regime/markov_ar/p{_g(p)}" for p in (0.003, 0.03, 0.1)]
    return ids


BIG_IDS = _big_ids()


def parse_config_id(cid: str) -> dict:
    """{'generator', 'placement', 'knob_value', 'd'} from a config id (d is None unless placement is context)."""
    parts = cid.split("/")
    gen = parts[1]
    if gen == "markov_ar":
        return {"generator": gen, "placement": "markov", "knob_value": float(parts[2][1:]), "d": None}
    knob = KNOBS[gen][1]
    val = float(parts[3][len(knob):])
    d = int(parts[4][1:]) if parts[2] == "context" else None
    return {"generator": gen, "placement": parts[2], "knob_value": val, "d": d}
