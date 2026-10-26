"""Run the mechanism measurements along each family's difficulty knob.

For each study (a generator swept across its knob from passing to failing settings), build
`--n` samples per knob value, run `otp.mechanism.measure` for each size, and write
`results/mechanism/<study>.json` (one record per size x knob value) plus `provenance.json`.

python scripts/mechanism_run.py --studies all --sizes 5M 28M --n 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp.generators.base import make_sample  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.mechanism import measure  # noqa: E402
from otp.model import set_threads  # noqa: E402


def studies():
    """name -> (knob name, values, factory(value, seed) -> Process, change_step(value) or None)."""
    from otp.generators import chaos, regime, tails
    from otp.generators.primitives import in_prior

    base = regime.randomized_base
    return {
        "in_prior": ("period", [24, 96],
                     lambda v, s: in_prior(s, period=v), None),
        "lorenz_rho": ("rho", [15.0, 24.0, 28.0, 45.0],
                       lambda v, s: chaos.Lorenz63(rho=v), None),
        "logistic_r": ("r", [3.2, 3.5, 3.7, 3.9, 4.0],
                       lambda v, s: chaos.LogisticMap(r=v), None),
        "mackey_tau": ("tau", [10.0, 17.0, 30.0],
                       lambda v, s: chaos.MackeyGlass(tau=v), None),
        "mean_jump_d": ("d", [16, 64, 256, 1024],
                        lambda v, s: regime.MeanJump(delta=4.0, placement="context", d=v, **base(s)), lambda v: v),
        "freq_jump_d": ("d", [64, 256, 1024],
                        lambda v, s: regime.FrequencyJump(ratio=2.0, placement="context", d=v, **base(s)), lambda v: v),
        "var_jump_d": ("d", [64, 256, 1024],
                       lambda v, s: regime.VarianceJump(factor=4.0, placement="context", d=v, **base(s)), lambda v: v),
        "stable_alpha": ("alpha", [2.0, 1.7, 1.3, 1.1],
                         lambda v, s: tails.StableAR(alpha=v), None),
        "fbm_hurst": ("H_hurst", [0.2, 0.5, 0.8],
                      lambda v, s: tails.FractionalBM(H_hurst=v), None),
    }


def to_jsonable(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if k == "attn_per_sample":
            continue
        if isinstance(v, dict):
            out[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="+", default=["all"])
    ap.add_argument("--sizes", nargs="+", default=["5M", "28M"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--n-paths", type=int, default=300)
    ap.add_argument("--out", default="results/mechanism")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    set_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    allst = studies()
    names = list(allst) if a.studies == ["all"] else a.studies
    t0 = time.time()
    for name in names:
        knob, values, factory, change = allst[name]
        records = []
        for v in values:
            samples = []
            for s in range(a.n):
                seed = zlib.crc32(f"mech/{name}/{v}/{s}".encode()) % (2**31)
                samples.append(make_sample(factory(v, s), a.T, a.H, seed=seed, n_paths=a.n_paths))
            for size in a.sizes:
                t1 = time.time()
                r = measure(size, samples, horizon=a.H, change_step=change(v) if change else None)
                rec = {"study": name, "knob": knob, "knob_value": v, "size": size, "T": a.T, "H": a.H,
                       "sec": round(time.time() - t1, 1), **to_jsonable(r)}
                records.append(rec)
                print(f"{name} {knob}={v} {size}: ratio {r['mase_ratio_oracle_gmean']:.3f} "
                      f"output_r2 {r['output_r2']:.3f} best_readout {max(r['readout_r2'].values()):.3f} "
                      f"tmpl model/oracle {r['template_corr_model']:.2f}/{r['template_corr_oracle']:.2f}", flush=True)
        write_json(os.path.join(a.out, f"{name}.json"), records)
    write_json(os.path.join(a.out, f"provenance_{'_'.join(names) if len(names) < 4 else 'all'}.json"),
               provenance("scripts/mechanism_run.py", argv=sys.argv, wall_sec=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()
