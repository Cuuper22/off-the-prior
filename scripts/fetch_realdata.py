"""Download every open-license source used by the extra real-data and analogue sets, check its license, and
write `bench/regression/sources.csv` (name, URL, license, license evidence, date fetched, bytes, SHA-256).

Raw files go to --raw-dir (not committed). Re-running reuses files already present. Live services are cut at the
date in `otp.realdata_sources.CUTOFF`.

python scripts/fetch_realdata.py --raw-dir <dir> [--out bench/regression/sources.csv]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp import realdata  # noqa: E402
from otp import realdata_sources as S  # noqa: E402


def fetch_everything(raw_dir: str, seed: int = realdata.SEED):
    """Returns `(fetcher, extra_items, analogue_items, hub_rows)`; every raw file is recorded in the fetcher."""
    fx = S.Fetcher(raw_dir)
    extra = S.collect_extra_items(fx, seed)
    analogues = S.collect_analogue_items(fx, seed)
    hub = S.hub_file_rows(realdata.PAIRS, os.path.join(raw_dir, "hf"))
    return fx, extra, analogues, hub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default="bench/regression/sources.csv")
    a = ap.parse_args()
    fx, extra, analogues, hub = fetch_everything(a.raw_dir)
    fx.write_sources_csv(a.out, hub)
    print(f"{len(extra)} extra series, {len(analogues)} analogue series, {len(fx.records) + len(hub)} source files; wrote {a.out}")


if __name__ == "__main__":
    main()
