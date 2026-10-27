"""Fetch the open-license real series, cut the transfer windows, and write the real-data corpus.

    python scripts/neurons_real.py --raw-dir <cache dir> --out results/neurons/real

Writes in --out:
  real.npz             corpus-format arrays (values (N, 1152), motifs, measured analogs); git-ignored, rebuilt by this script
  real_events.npy      (N, 36, 3) known-event shares per patch (weekend, US holiday, COVID spring 2020)
  sources.csv          one row per raw file: source, URL, license, license evidence, date fetched, bytes, SHA-256
  windows.csv          one row per window: domain, dataset, series, start, license, redistributable flag
  real_manifest.json   counts per domain, sources that failed, hash of real.npz, command

Raw files are never committed. Sources with licenses that do not allow redistribution contribute windows to
real.npz only (the file is not committed either); only the manifest, this script and derived statistics are.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from otp.io import provenance, write_json, read_json  # noqa: E402
from otp.neurons import corpus as C  # noqa: E402
from otp.neurons import realdata as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default="results/neurons/real")
    ap.add_argument("--per-domain", type=int, default=R.PER_DOMAIN)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fx, items, fails = R.collect_items(a.raw_dir)
    print(f"{len(items)} candidate series, {len(fails)} failed sources", flush=True)
    win = R.build_windows(items, per_domain=a.per_domain)
    v = win["values"]
    print("windows", v.shape, flush=True)
    c = C.measured_arrays(v, R.T_CTX)
    specs = [{"sid": i, **{k: m[k] for k in ("domain", "dataset", "series_id")}} for i, m in enumerate(win["meta"])]
    path = os.path.join(a.out, "real.npz")
    C.save_corpus(path, c, specs)
    np.save(os.path.join(a.out, "real_events.npy"), win["events"])
    R.write_manifest(fx, win, fails, a.out, f"python scripts/neurons_real.py --raw-dir <dir> --out {a.out} --per-domain {a.per_domain}")
    m = read_json(os.path.join(a.out, "real_manifest.json"))
    m["sha256_real_npz"] = C.sha256(path)
    m["bytes_real_npz"] = os.path.getsize(path)
    m["provenance"] = provenance("scripts/neurons_real.py")
    write_json(os.path.join(a.out, "real_manifest.json"), m)
    print("wrote", path, m["windows_per_domain"])


if __name__ == "__main__":
    main()
