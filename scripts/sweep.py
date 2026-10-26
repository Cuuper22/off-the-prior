"""Generic failure-map sweep: grid of generator configs x seeds x context lengths x horizons.

A grid module (e.g. `otp.grids.chaos`) exposes `configs()` returning a list of
`(config_id, factory)` where `factory(seed) -> Process`. For every config and seed this
script builds one sample at the longest context and longest horizon, then:

- Timestral: one forecast per (size, context length) at the longest horizon; shorter
  horizons are prefixes. (A DHF decode block is causal across horizon patches, so a
  prefix equals a shorter forecast to float precision; checked in tests/test_harness.py.)
- classical models and the oracle: computed once per sample (classical fits use the last
  1024 steps regardless of the context-length knob; the oracle conditions on the true state).

Output (in --out):
  samples.npz           the generated samples (re-used if present and --regen not set)
  scores.csv            long table: one row per sample x forecaster x context x horizon x variate
  provenance.json       command line, git commit, host, threads, wall time

Usage:
  python scripts/sweep.py --grid otp.grids.chaos --sizes 5M 28M --contexts 512 2048 8192 \
      --horizons 16 128 1024 --seeds 8 --out results/map/chaos
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp import metrics  # noqa: E402
from otp.classical import CLASSICAL, classical_forecast  # noqa: E402
from otp.evaluate import METRIC_COLS  # noqa: E402
from otp.generators.base import Sample, load_samples, make_sample, save_samples  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.model import forecast, set_threads  # noqa: E402


def slice_sample(s: Sample, t: int, h: int) -> Sample:
    return Sample(s.context[..., -t:], s.future[..., :h], s.oracle_q[..., :h, :], s.period, s.meta,
                  None if s.past_cov is None else s.past_cov[..., -t:],
                  None if s.known_cov is None else s.known_cov[..., s.T - t: s.T + h])


def rows_for(s: Sample, q: np.ndarray, name: str, t: int, horizons: list[int], extra: dict) -> list[dict]:
    out = []
    ctx = s.context[..., -t:]
    for h in horizons:
        sc = metrics.score(ctx, s.future[..., :h], q[..., :h, :], s.period)
        for v in range(s.context.shape[0]):
            r = {"config_id": s.meta.get("config_id"), "sample_id": s.meta["sample_id"], "family": s.meta["family"],
                 "generator": s.meta["generator"], "knob": s.meta["knob"], "knob_value": s.meta["knob_value"],
                 "seed": s.meta["seed"], "T": t, "H": h, "variate": v, "forecaster": name, **extra}
            for k in METRIC_COLS:
                r[k] = float(sc[k][v])
            out.append(r)
    return out


def _gen_one(job):
    """Build one sample in a worker process (grid factories are lambdas, so re-import the grid)."""
    grid_name, cid, seed, tmax, hmax, n_paths = job
    factory = dict(importlib.import_module(grid_name).configs())[cid]
    s = make_sample(factory(seed), tmax, hmax, seed=zlib.crc32(f"{cid}/{seed}".encode()) % (2**31), n_paths=n_paths,
                    sample_id=f"{cid}/s{seed}")
    s.meta["config_id"] = cid
    return s


def _refs_one(job):
    s, names, hmax = job
    return [classical_forecast(name, s.context[None], hmax, s.period)[0] for name in names]


def _pool_map(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs, chunksize=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--sizes", nargs="*", default=["5M"], help="Timestral sizes; give none to compute references only")
    ap.add_argument("--contexts", nargs="+", type=int, default=[2048])
    ap.add_argument("--horizons", nargs="+", type=int, default=[16, 128, 1024])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--classical", nargs="*", default=list(CLASSICAL))
    ap.add_argument("--n-paths", type=int, default=500)
    ap.add_argument("--covariates", action="store_true", help="also run timestral-<size>+cov")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these config ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-symmetric", action="store_true",
                    help="ablation: turn off Timestral's symmetric averaging (forecaster name gets '-nosym')")
    ap.add_argument("--workers", type=int, default=1, help="processes for sample generation and classical fits")
    ap.add_argument("--samples-from", default=None, help="load samples.npz from this folder instead of generating")
    ap.add_argument("--no-references", action="store_true", help="skip oracle/classical rows (merge them from the "
                    "folder the samples came from when summarizing)")
    a = ap.parse_args(argv)
    set_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    grid = importlib.import_module(a.grid)
    cfgs = grid.configs()
    if a.only:
        cfgs = [c for c in cfgs if c[0] in set(a.only)]
    tmax, hmax = max(a.contexts), max(a.horizons)
    spath = os.path.join(a.samples_from or a.out, "samples.npz")
    if os.path.exists(spath) and not a.regen:
        samples = load_samples(spath)
        print(f"loaded {len(samples)} samples from {spath}")
    else:
        jobs = [(a.grid, cid, a.seed_offset + k, tmax, hmax, a.n_paths) for cid, _ in cfgs for k in range(a.seeds)]
        samples = _pool_map(_gen_one, jobs, a.workers)
        print(f"generated {len(samples)} samples ({time.time() - t_start:.0f}s)", flush=True)
        if a.only:
            print("note: --only given; samples are saved for this subset")
        try:
            save_samples(spath, samples)
        except ValueError:
            print("samples have mixed shapes; not saved as one npz")

    rows = []
    ref_samples = [] if a.no_references else samples
    ref_q = _pool_map(_refs_one, [(s, list(a.classical), hmax) for s in ref_samples], a.workers)
    for s, qs in zip(ref_samples, ref_q):  # oracle + classical once per sample (classical uses the last 1024 steps)
        for t in a.contexts:  # the oracle's quantiles do not depend on T, but MASE/sCRPS scaling does
            rows += rows_for(s, s.oracle_q, "oracle", t, a.horizons, {})
        for name, q in zip(a.classical, qs):
            for t in a.contexts:
                rows += rows_for(s, q, name, t, a.horizons, {})
    print(f"references done ({time.time() - t_start:.0f}s)", flush=True)

    specs = [(size, False) for size in a.sizes] + ([(size, True) for size in a.sizes] if a.covariates else [])
    for size, use_cov in specs:
        name = f"timestral-{size}" + ("+cov" if use_cov else "") + ("-nosym" if a.no_symmetric else "")
        for t in a.contexts:
            t0 = time.time()
            groups: dict = {}
            for i, s in enumerate(samples):
                groups.setdefault((s.context.shape, s.past_cov is None, s.known_cov is None), []).append(i)
            for idx in groups.values():
                ctx = np.stack([samples[i].context[..., -t:] for i in idx])
                pc = kc = None
                if use_cov and samples[idx[0]].past_cov is not None:
                    pc = np.stack([samples[i].past_cov[..., -t:] for i in idx])
                if use_cov and samples[idx[0]].known_cov is not None:
                    kc = np.stack([samples[i].known_cov[..., samples[i].T - t: samples[i].T + hmax] for i in idx])
                q = forecast(size, ctx, hmax, past_covariates=pc, known_covariates=kc, batch_size=a.batch_size,
                             symmetric_averaging=not a.no_symmetric)
                for j, i in enumerate(idx):
                    rows += rows_for(samples[i], q[j], name, t, a.horizons,
                                     {"sec_per_sample": (time.time() - t0) / len(samples)})
            print(f"{name} T={t} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "scores.csv"), index=False)
    write_json(os.path.join(a.out, "provenance.json"),
               provenance("scripts/sweep.py", argv=sys.argv, n_samples=len(samples), n_rows=len(df),
                          wall_sec=round(time.time() - t_start, 1)))
    print(f"wrote {len(df)} rows to {a.out}/scores.csv in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
