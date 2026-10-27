"""Build the Off-the-Prior benchmark files from bench/configs/*.json. Deterministic.

Series seed = crc32("bench/<set>/<config_id>/<seed_index>"), used both for the generator's own
random parameters (where a config has them, e.g. in-prior series) and for make_sample's noise,
so benchmark series never coincide with the failure-map samples. 500 Monte Carlo paths give the
oracle's 21 quantiles.

Output: bench/data/<set>_<split>_T<T>_H<H>.npz (float32 arrays: series_id, context (N, T),
future (N, H), oracle_q (N, H, 21), period (N,), meta JSON per series) and bench/data/<set>_manifest.csv,
plus bench/data/<set>_SHA256SUMS (hashes of the array contents of each file).

python bench/build.py --set diag [--workers 8]
python bench/build.py --set lb --workers 12
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from otp.generators.base import make_sample  # noqa: E402

N_PATHS = 500


def series_seed(e: dict) -> int:
    return zlib.crc32(f"bench/{e['set']}/{e['config_id']}/{e['seed_index']}".encode()) % (2**31)


def build_one(e: dict):
    factory = dict(importlib.import_module(e["grid"]).configs())[e["config_id"]]
    seed = series_seed(e)
    s = make_sample(factory(seed), e["T"], e["H"], seed=seed, n_paths=N_PATHS,
                    sample_id=f"{e['set']}/{e['config_id']}/s{e['seed_index']}")
    meta = {k: v for k, v in e.items() if k not in ("grid",)}
    meta.update({"series_id": s.meta["sample_id"], "family": s.meta["family"], "generator": s.meta["generator"],
                 "knob": s.meta["knob"], "knob_value": s.meta["knob_value"], "tags": s.meta["tags"],
                 "oracle_paths_nonfinite": s.meta.get("oracle_paths_nonfinite", 0), "seed": seed})
    return (s.context[0].astype(np.float32), s.future[0].astype(np.float32), s.oracle_q[0].astype(np.float32),
            int(s.period), meta)


def _map(fn, items, workers):
    if workers <= 1:
        return [fn(i) for i in items]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=4))


def content_sha256(path: str) -> str:
    """Hash of the arrays inside an .npz (zip timestamps would make file hashes unstable)."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for k in sorted(z.files):
            v = z[k]
            h.update(k.encode())
            h.update(v.tobytes() if v.dtype.kind != "U" else "\n".join(v.tolist()).encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["diag", "lb"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "data"))
    ap.add_argument("--limit", type=int, default=None, help="build only the first N entries (tests)")
    a = ap.parse_args()
    entries = json.load(open(os.path.join(ROOT, "bench", "configs", f"{a.set}.json")))
    if a.limit:
        entries = entries[: a.limit]
    built = _map(build_one, entries, a.workers)
    os.makedirs(a.out, exist_ok=True)
    groups = defaultdict(list)
    for e, b in zip(entries, built):
        groups[(e["split"], e["T"], e["H"])].append(b)
    enc = json.JSONEncoder(default=lambda o: o.item() if hasattr(o, "item") else str(o))
    rows = []
    for (split, T, H), items in sorted(groups.items()):
        path = os.path.join(a.out, f"{a.set}_{split}_T{T}_H{H}.npz")
        np.savez_compressed(path, series_id=np.asarray([m["series_id"] for *_, m in items]),
                            context=np.stack([c for c, *_ in items]), future=np.stack([f for _, f, *_ in items]),
                            oracle_q=np.stack([q for _, _, q, *_ in items]), period=np.asarray([p for *_, p, _ in items]),
                            meta=np.asarray([enc.encode(m) for *_, m in items]))
        for *_, m in items:
            rows.append({**{k: m[k] for k in ("series_id", "set", "split", "config_id", "family", "generator", "knob",
                                              "knob_value", "T", "H", "seed", "oracle_paths_nonfinite")},
                         "failure_type": m.get("failure_type", ""), "reason": m.get("reason", ""),
                         "tags": ";".join(m["tags"]), "file": os.path.basename(path)})
        print(f"wrote {path} ({len(items)} series)")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{a.set}_manifest.csv"), index=False)
    sums = {f: content_sha256(os.path.join(a.out, f)) for f in sorted(os.listdir(a.out)) if f.startswith(a.set + "_") and f.endswith(".npz")}
    with open(os.path.join(a.out, f"{a.set}_SHA256SUMS"), "w") as fh:
        fh.writelines(f"{v}  {k}\n" for k, v in sums.items())


if __name__ == "__main__":
    main()
