"""Build the fixed regression sets used to check that fine-tuning does not damage normal forecasting.

- `real`        last windows of series from GIFT-Eval and fev-bench datasets (`otp/realdata.py`)
- `real_extra`  last windows from further open-license sources (`otp/realdata_sources.py`)
- `analogues`   real series that stand in for the synthetic failure families, tagged with family and reason
- `in_prior`    500 series from `otp.generators.primitives.in_prior(seed)`, seeds 0..499,
                context 2048, horizon 128, with Monte Carlo oracle quantiles
- `manifest.csv`, `sources.csv`, `README.md`

Windows from sources whose license allows redistribution go to `<set>.npz` (committed); the rest go to
`<set>_restricted.npz` (git-ignored, rebuilt here from the raw files).

python scripts/build_regression_sets.py --out bench/regression --cache-dir <dir for raw downloads>

Raw source data stays in the cache directory and is never committed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from otp import realdata  # noqa: E402
from otp.generators.base import make_sample, save_samples  # noqa: E402
from otp.generators.primitives import in_prior  # noqa: E402
from fetch_realdata import fetch_everything  # noqa: E402
from otp import realdata_sources as S  # noqa: E402

IN_PRIOR_N, IN_PRIOR_T, IN_PRIOR_H = 500, 2048, 128

MANIFEST_COLS = ["set", "sample_id", "source", "revision", "source_file", "source_sha256", "dataset", "term",
                 "domain", "freq", "series_id", "series_index", "series_len", "window_start", "origin_index",
                 "origin_time", "T", "H", "m", "n_filled", "first_variate_of_multivariate", "value_col", "redistributable",
                 "family_analogue", "reason"]

README = """# Regression sets for the Timestral fine-tune check

Fixed sets of forecasting windows. They are used to check that a fine-tuned Timestral still forecasts
normal series as well as the released model. They are not a leaderboard.

| set | windows | what it is |
|---|---|---|
{set_table}

Files: `<set>.npz` (windows from sources whose license allows redistribution), `<set>_restricted.npz`
(windows from other sources; git-ignored, rebuilt locally), `manifest.csv` (one row per window: set, sample id,
source, revision, source file, file hash, dataset, term, domain, frequency, series id, window start, origin,
T, H, m, redistributable, family_analogue, reason), `sources.csv` (every raw file: name, URL, license, license
evidence, date fetched, bytes, SHA-256).

Load a set with `otp.realdata.load_union("bench/regression", "<set>")` (real sets; windows missing on this machine
are skipped) or `otp.generators.base.load_samples("bench/regression/in_prior.npz")`. Both return lists of
`otp.generators.base.Sample`. Rebuild everything with
`python scripts/build_regression_sets.py --cache-dir <dir>`; fetch only with `python scripts/fetch_realdata.py`.

## License rule

Windows are committed only when the source states a license that allows redistribution (public domain or US
Government work, CC0, CC BY, Apache-2.0, MIT). The Hub card of `Salesforce/GiftEval` states apache-2.0; its
per-dataset upstream terms were not separately verified, and `restaurant` (competition data) is treated as
restricted. `autogluon/fev_datasets` states no license and says research use only, so all windows from it are
restricted. SILSO monthly sunspots (CC BY-NC) and the R datasets (lynx, lh) are restricted. The Transnet BW part
of the grid-frequency record has no open license and is not used. `sources.csv` has the license and the
check behind every row. Restricted counts: {restricted_counts}.

## Window rule

- Real set: the last window of the series. H is GIFT-Eval's prediction length for the dataset frequency (hourly,
  10-minute and 15-minute 48; daily 30; weekly 8; monthly 12; the M4 datasets use the M4 lengths: hourly 48, daily 14,
  weekly 13, monthly 18). Pairs with the term `medium` use 10 times that length. The cap is {max_h}.
- Context `T = min({max_t}, L - H)`; a series needs `T >= {min_t}` for the real and extra sets. The analogue subset
  allows shorter contexts for short natural records (lynx, hormone series); `T` is in the manifest.
- Only the first variate of multivariate series is used. For `fred_md_2025` every column is one series.
- Series are visited in a seeded random order (seed {seed}). A series is skipped when its future window has a
  missing value, more than 5% of its context is missing, or its context is constant. Missing values inside an
  accepted context are filled by forward then backward fill (`n_filled`).
- `m` is a calendar period used for MASE scaling and seasonal naive: 24 hourly, 96 for 15-minute, 144 for 10-minute,
  7 daily, 5 for business-day series, 52 weekly, 12 monthly; 1 where the manifest says so.
- Live sources (FRED, USGS, NOAA, Seattle open data, OWID) are cut at {cutoff}; `sources.csv` has the hash of the
  exact bytes used. Event windows in the analogue subset start at a fixed origin date given in the manifest.
- `oracle_q` is NaN for every real window. Score these windows against the classical models and seasonal naive.
- `hierarchical_sales/D` of GIFT-Eval was tried and has no usable series (all series have missing days inside the
  last 30 steps).

Overlap with Timestral's training data: Timestral was pretrained mostly on synthetic data, and the mix is not fully
documented. GIFT-Eval, fev-bench and the other public series may overlap it in unknown ways. The sets are used only
to detect regressions between Timestral and its fine-tuned versions, not as a leaderboard claim. Medium-term pairs
reuse series of the short-term pairs with their own seeded order, so windows may overlap.

## Real set pairs

{pair_table}

## Extra real set (further open-license sources)

{extra_table}

## Analogue subset

Real series that stand in for the synthetic failure families (`family_analogue` and `reason` in the manifest):

{ana_table}

## In-prior set

{n_prior} series from `in_prior(seed)` for seeds 0 to {last_seed}, context {prior_t}, horizon {prior_h}, built with
`make_sample(in_prior(seed), {prior_t}, {prior_h}, seed=seed)` (500 oracle paths). The oracle quantiles are the Monte
Carlo continuation of the true process from its true state and are meaningful here. Seasonal periods: {periods}.
"""


def table(man: pd.DataFrame, by: list[str]) -> str:
    g = man.groupby(by, sort=False)
    cols = by + ["windows", "T min", "T max", "H", "m", "redistributable"]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for k, d in g:
        k = k if isinstance(k, tuple) else (k,)
        hs = sorted(d["H"].unique())
        rows.append("| " + " | ".join([str(x) for x in k] + [str(len(d)), str(d["T"].min()), str(d["T"].max()),
                                                           "/".join(map(str, hs)), "/".join(map(str, sorted(d["m"].unique()))),
                                                           "yes" if d["redistributable"].all() else ("no" if not d["redistributable"].any() else "mixed")]) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/regression")
    ap.add_argument("--cache-dir", required=True, help="where the raw files are downloaded (not committed)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    counts = {}

    real, rows = realdata.build_real(cache_dir=os.path.join(a.cache_dir, "hf"))
    counts["real"] = realdata.save_split(a.out, "real", real, [r["redistributable"] for r in rows])

    fx, extra_items, ana_items, hub = fetch_everything(a.cache_dir)
    ex, ex_rows = S.items_to_samples(extra_items, "real_extra", realdata.SEED)
    counts["real_extra"] = realdata.save_split(a.out, "real_extra", ex, [r["redistributable"] for r in ex_rows])
    an, an_rows = S.items_to_samples(ana_items, "analogues", realdata.SEED)
    counts["analogues"] = realdata.save_split(a.out, "analogues", an, [r["redistributable"] for r in an_rows])
    rows += ex_rows + an_rows
    fx.write_sources_csv(os.path.join(a.out, "sources.csv"), hub)

    prior = [make_sample(in_prior(s), IN_PRIOR_T, IN_PRIOR_H, seed=s) for s in range(IN_PRIOR_N)]
    save_samples(os.path.join(a.out, "in_prior.npz"), prior)
    for s in prior:
        rows.append({"set": "in_prior", "sample_id": s.meta["sample_id"], "source": "synthetic",
                     "source_file": "otp.generators.primitives.in_prior(seed)", "dataset": "in_prior",
                     "term": "short", "domain": "synthetic", "freq": "", "series_id": s.meta["seed"],
                     "series_index": s.meta["seed"], "window_start": 0, "T": s.T, "H": s.H, "m": s.period,
                     "n_filled": 0, "redistributable": True})
    man = pd.DataFrame(rows).reindex(columns=MANIFEST_COLS)
    for c in ("series_index", "series_len", "window_start", "origin_index", "T", "H", "m", "n_filled"):
        man[c] = pd.to_numeric(man[c], errors="coerce").astype("Int64")
    man["redistributable"] = man["redistributable"].astype(bool)
    man.to_csv(os.path.join(a.out, "manifest.csv"), index=False)

    n = man.groupby("set").size().to_dict()
    set_rows = [("real", "last windows of GIFT-Eval and fev-bench series"), ("real_extra", "last windows from further open-license sources"),
                ("analogues", "real series standing in for the synthetic failure families, tagged"),
                ("in_prior", "synthetic trend plus season plus AR noise series with oracle quantiles")]
    set_table = "\n".join(f"| `{k}` | {n.get(k, 0)} | {d} |" for k, d in set_rows)
    r = man[man.set == "real"]
    periods = ", ".join(str(p) for p in sorted(man[man.set == "in_prior"]["m"].unique()))
    ana = man[man.set == "analogues"]
    ana_tab = table(ana, ["family_analogue", "dataset"]) + "\n\n" + "\n".join(
        f"- `{x.sample_id}`: {x.reason}" for x in ana.itertuples())
    text = README.format(set_table=set_table, restricted_counts=", ".join(f"{k} {v['restricted']}" for k, v in counts.items()),
                         max_h=realdata.MAX_HORIZON, max_t=realdata.MAX_CONTEXT, min_t=realdata.MIN_CONTEXT, seed=realdata.SEED,
                         cutoff=str(S.CUTOFF.date()), pair_table=table(r, ["dataset", "term", "domain", "freq"]),
                         extra_table=table(man[man.set == "real_extra"], ["dataset", "domain", "freq"]), ana_table=ana_tab,
                         n_prior=IN_PRIOR_N, last_seed=IN_PRIOR_N - 1, prior_t=IN_PRIOR_T, prior_h=IN_PRIOR_H, periods=periods)
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(text)
    sizes = {nme: os.path.getsize(os.path.join(a.out, nme)) for nme in sorted(os.listdir(a.out)) if nme.endswith(".npz")}
    print(f"built {n} windows per set in {time.time() - t0:.0f}s; counts {counts}; sizes {sizes}")


if __name__ == "__main__":
    main()
