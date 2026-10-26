"""Long-context chaos grid: two knob values per generator, one non-chaotic and one chaotic.

Ids are the same as in `otp.grids.chaos`. The non-chaotic Lorenz value (rho = 20) settles to a
fixed point rather than a cycle; the Kuramoto pair is K = 8 (locked, exponent 0) and K = 1
(partial synchrony, exponent above 0 for some frequency seeds; see
`results/map/chaos/lyapunov.csv`).
"""

from otp.grids.chaos import make_factory, param_list

LONG_IDS = ["lorenz63/rho20", "lorenz63/rho28", "rossler/c3", "rossler/c5.7",
            "mackey_glass/tau10", "mackey_glass/tau17", "logistic_map/r3.2", "logistic_map/r3.9",
            "kuramoto/K8", "kuramoto/K1"]


def configs():
    rows = {cid: (cls, params) for cid, cls, params in param_list()}
    return [(cid, make_factory(*rows[cid])) for cid in LONG_IDS]
