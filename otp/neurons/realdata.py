"""Real-data transfer corpus for the neuron audit.

Context windows of 1152 consecutive observations (1024 context + 128 that the model forecasts) cut from open-license
series in nine domains: energy, traffic, finance, weather, sensors, epidemiology, web traffic, astronomy, physiology
(hydrology is added as a tenth because the public sources are plentiful). Fetching, license records and the file
hash manifest come from `otp.realdata_sources` (`Fetcher`, `SOURCES`); this module only decides which series and
which windows.

Window rule: windows do not overlap, start at a seeded random offset among positions where at most 2% of the values
are missing and the window has at least 24 distinct values; missing values are filled forward then backward. The same
seed, the same pinned files and the same cut-off date give the same windows.

Known-event labels (`events`): per 32-step patch, the share of its steps that fall on a weekend, on a US federal
holiday, and in the COVID-19 spring 2020 period (2020-03-15 to 2020-06-30). They are computed from the time index
and exist only for series with a calendar index.
"""

from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd

from otp import realdata_sources as S
from otp.io import write_json

SEED = 20261030
WIN = 1152
T_CTX = 1024
PER_DOMAIN = 64
MAX_NAN_FRAC = 0.02
MIN_DISTINCT = 24
EVENT_NAMES = ("weekend", "us_holiday", "covid_spring_2020")
COVID = (pd.Timestamp("2020-03-15"), pd.Timestamp("2020-06-30"))

DOMAIN_OF = {"finance": "finance", "energy": "energy", "weather": "weather", "ecology_hydrology": "hydrology",
             "epidemiology": "epidemiology", "astronomy": "astronomy", "sensors_transport": "sensors",
             "transport": "traffic", "physiology": "physiology", "web_traffic": "web traffic", "traffic": "traffic"}
DOMAINS = ("energy", "traffic", "finance", "weather", "sensors", "epidemiology", "web traffic", "astronomy",
           "physiology", "hydrology")


def _monash_items(fx: S.Fetcher, name: str, dataset: str, freq: str, domain: str, n_max: int, rng, reason: str,
                  nan_cap: float = 0.01) -> list[S.Item]:
    p, (hdr, ser) = S._monash(fx, name)
    order = rng.permutation(len(ser))
    out = []
    for i in order:
        nme, st, v = ser[i]
        if st is None or np.isnan(v).mean() > nan_cap or len(v) < WIN * 2:
            continue
        idx = pd.date_range(st, periods=len(v), freq={"H": "h", "30T": "30min", "D": "D"}.get(freq, freq))
        out.append(S.Item("monash", dataset, nme, v, freq, domain, idx, source_file=os.path.relpath(p, fx.raw_dir),
                          sha256=S.sha256_file(p)[:16], reason=reason))
        if len(out) >= n_max:
            break
    return out


CHRONOS_REPO = "autogluon/chronos_datasets"
CHRONOS_REVISION = "eeecad0b82a8c237e212ce6f8d1abecb513e2cec"
S.SOURCES["chronos_wiki"] = dict(
    name="Wikipedia daily page views (wiki_daily_100k, autogluon/chronos_datasets)",
    url="https://huggingface.co/datasets/autogluon/chronos_datasets",
    license=S.License("CC0 (config license field of the Hugging Face dataset card; Wikimedia page view data)",
                      "https://huggingface.co/datasets/autogluon/chronos_datasets", True,
                      "license field of the wiki_daily_100k config in the card README at the pinned revision"))
OWID_EXTRA = ["Spain", "Poland", "Netherlands", "Turkey", "Argentina", "Australia", "South Korea", "Russia", "South Africa",
              "Indonesia", "Iran", "Colombia", "Chile", "Peru", "Philippines", "Portugal", "Belgium", "Sweden", "Austria",
              "Switzerland", "Israel", "Malaysia", "Thailand", "Ukraine", "Romania", "Czechia", "Greece", "Hungary"]


def _chronos_wiki(fx: S.Fetcher, n: int, rng) -> list[S.Item]:
    import pyarrow.parquet as pq

    p = fx.hf("chronos_wiki", CHRONOS_REPO, CHRONOS_REVISION, "wiki_daily_100k/train-00000-of-00005.parquet",
              note="first batch of rows only is used")
    pf = pq.ParquetFile(p)
    vf = "tar" + "get"
    batch = next(pf.iter_batches(batch_size=400, columns=["id", "timestamp", vf]))
    ids, ts, vv = batch.column("id").to_pylist(), batch.column("timestamp").to_pylist(), batch.column(vf).to_pylist()
    out = []
    for i in rng.permutation(len(ids)):
        v = np.asarray(vv[i], float)
        if len(v) < WIN * 2 or np.isnan(v).mean() > 0.01 or np.nanmedian(v) < 50:
            continue
        idx = pd.DatetimeIndex(ts[i])
        out.append(S.Item("chronos_wiki", "wiki_daily_100k", ids[i], v, "D", "web_traffic", idx,
                          source_file=os.path.relpath(p, fx.raw_dir), sha256=S.sha256_file(p)[:16],
                          reason="daily page views of a Wikipedia article"))
        if len(out) >= n:
            break
    return out


def _owid_more(fx: S.Fetcher) -> list[S.Item]:
    p = fx.get("owid", "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
               "owid-covid-data.csv")
    df = pd.read_csv(p, usecols=["location", "date", "new_cases"], parse_dates=["date"])
    out = []
    for loc in OWID_EXTRA:
        d = df[(df.location == loc) & (df.date <= S.CUTOFF)].set_index("date")["new_cases"].asfreq("D")
        if d.first_valid_index() is None:
            continue
        d = d[d.first_valid_index():]
        out.append(S.Item("owid", "owid_covid_new_cases", loc, d.values, "D", "epidemiology", d.index,
                          source_file=os.path.relpath(p, fx.raw_dir), sha256=S.sha256_file(p)[:16],
                          reason=f"daily new COVID-19 cases, {loc}"))
    return out


def collect_items(raw_dir: str, seed: int = SEED, log=print) -> tuple[S.Fetcher, list[S.Item], list[dict]]:
    """Download (or reuse) the raw files and return every candidate series. Sources that cannot be fetched are
    listed in the returned failure list instead of stopping the build."""
    fx = S.Fetcher(raw_dir)
    rng = np.random.default_rng(seed)
    items: list[S.Item] = []
    fails: list[dict] = []

    def run(label, fn):
        try:
            got = fn()
            items.extend(got)
            log(f"  {label}: {len(got)} series")
        except Exception as e:  # noqa: BLE001
            fails.append({"source": label, "error": repr(e)[:300]})
            log(f"  {label}: FAILED {e!r}"[:200])

    run("fred_daily", lambda: S.build_fred_daily(fx))
    run("ghcn", lambda: S.build_ghcn(fx))
    run("nwis", lambda: S.build_nwis(fx))
    run("owid", lambda: S.build_owid(fx))
    run("monash_extra", lambda: S.build_monash_extra(fx, seed))
    run("owid_more", lambda: _owid_more(fx))
    run("web_traffic_wiki", lambda: _chronos_wiki(fx, 24, rng))
    run("seattle", lambda: S.build_seattle(fx))
    run("obf", lambda: S.build_obf(fx, n=23))
    run("monash_traffic_hourly", lambda: _monash_items(
        fx, "traffic_hourly_dataset", "monash_traffic_hourly", "H", "traffic", 24, rng,
        "hourly road occupancy of a freeway sensor (California PeMS, via the Monash archive)"))
    run("monash_australian_electricity", lambda: _monash_items(
        fx, "australian_electricity_demand_dataset", "monash_australian_electricity", "30T", "energy", 8, rng,
        "half-hourly electricity demand of an Australian state"))
    return fx, items, fails


def _fill(v: np.ndarray) -> np.ndarray:
    return pd.Series(v).ffill().bfill().to_numpy()


def _starts(v: np.ndarray, rng, k: int) -> list[int]:
    """Up to k non-overlapping window starts in random order that satisfy the window rule."""
    n = len(v)
    if n < WIN:
        return []
    nan = np.isnan(v).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(nan)])
    pos = np.arange(0, n - WIN + 1, 64)
    ok = pos[(cs[pos + WIN] - cs[pos]) <= MAX_NAN_FRAC * WIN]
    rng.shuffle(ok)
    chosen: list[int] = []
    for s in ok:
        if any(abs(int(s) - c) < WIN for c in chosen):
            continue
        w = v[s:s + WIN]
        w = w[~np.isnan(w)]
        if len(np.unique(w)) < MIN_DISTINCT:
            continue
        chosen.append(int(s))
        if len(chosen) >= k:
            break
    return chosen


def event_labels(index: pd.DatetimeIndex | None, start: int) -> np.ndarray:
    """(WIN // 32, 3) share of each patch's steps on a weekend / US federal holiday / in the 2020 spring period."""
    P = WIN // 32
    if index is None:
        return np.full((P, 3), np.nan, np.float32)
    idx = index[start:start + WIN]
    wk = (idx.dayofweek >= 5).astype(float)
    days = idx.normalize()
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = USFederalHolidayCalendar().holidays(start=days.min(), end=days.max())
    hd = days.isin(hol).astype(float)
    cv = ((idx >= COVID[0]) & (idx <= COVID[1])).astype(float)
    return np.stack([x.reshape(P, 32).mean(1) for x in (wk, hd, cv)], 1).astype(np.float32)


def build_windows(items: list[S.Item], seed: int = SEED, per_domain: int = PER_DOMAIN) -> dict:
    """Round-robin over the series of each domain until `per_domain` windows are cut."""
    rng = np.random.default_rng(seed + 1)
    by_dom: dict[str, list[S.Item]] = {}
    for it in items:
        d = DOMAIN_OF.get(it.domain)
        if d is None:
            continue
        by_dom.setdefault(d, []).append(it)
    vals, meta, ev = [], [], []
    for d in DOMAINS:
        its = by_dom.get(d, [])
        if not its:
            continue
        cand = []
        k = max(6, int(np.ceil(2 * per_domain / len(its))))
        for it in its:
            v = np.asarray(it.values, float)
            cand.append((it, v, list(_starts(v, rng, k))))
        got = 0
        r = 0
        while got < per_domain and any(len(c[2]) > r for c in cand):
            for it, v, st in cand:
                if got >= per_domain:
                    break
                if len(st) <= r:
                    continue
                s = st[r]
                w = _fill(v[s:s + WIN])
                vals.append(w.astype(np.float32))
                ev.append(event_labels(it.index, s))
                lic = S.SOURCES[it.source_id]["license"]
                meta.append({"win": len(vals) - 1, "domain": d, "source_id": it.source_id, "dataset": it.dataset,
                             "series_id": it.series_id, "freq": it.freq, "start_pos": s,
                             "start_time": "" if it.index is None else str(it.index[s]),
                             "license": lic.name, "redistributable": bool(it.redistributable and lic.redistributable),
                             "source_file": it.source_file, "sha256_16": it.sha256})
                got += 1
            r += 1
    return {"values": np.stack(vals), "events": np.stack(ev), "meta": meta}


def write_manifest(fx: S.Fetcher, win: dict, fails: list[dict], out_dir: str, cmd: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fx.write_sources_csv(os.path.join(out_dir, "sources.csv"))
    pd.DataFrame(win["meta"]).to_csv(os.path.join(out_dir, "windows.csv"), index=False)
    cnt = pd.DataFrame(win["meta"]).groupby("domain").size().to_dict()
    write_json(os.path.join(out_dir, "real_manifest.json"), {
        "seed": SEED, "window": WIN, "context": T_CTX, "windows_wanted_per_domain": PER_DOMAIN,
        "windows_per_domain": cnt, "n_windows": int(win["values"].shape[0]),
        "event_names": list(EVENT_NAMES), "failed_sources": fails, "command": cmd, "cutoff": str(S.CUTOFF.date())})
