"""Open-license real-data sources, a license manifest, and series builders.

Every source has a license record (name, URL, license, evidence, whether derived windows may be
committed). `Fetcher` downloads the raw files into a cache directory (never committed), and writes
one row per file to `bench/regression/sources.csv`: name, URL, license, date fetched, size and
SHA-256. Live services (FRED, USGS, NOAA, Seattle open data, OWID) are cut at `CUTOFF` so a rebuild
uses the same observations; the file hashes identify the exact bytes used.

Series builders return `Item`s. An `Item` carries the values, a time index, the window rule
(last window, or a fixed origin date for event windows), the horizon, and for the analogue subset
the failure family it stands in for and a one-line reason.

Redistribution rule: derived windows are committed only for sources whose license allows
redistribution (public domain / US Government work, CC0, CC BY, Apache-2.0, MIT). Windows from other
sources go to `*_restricted.npz`, which is git-ignored and rebuilt by `scripts/build_regression_sets.py`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-09-30")
UAS = ({}, {"User-Agent": "curl/8.5.0"})  # some hosts drop one or the other; try both
MONASH_REPO = "Monash-University/monash_tsf"
MONASH_REVISION = "58aafbe2712ff481c014f562e42723f2820fd5d4"

# ----------------------------------------------------------------------------- license records


@dataclass(frozen=True)
class License:
    name: str                 # license identifier or short description
    url: str                  # where the license is stated
    redistributable: bool     # may derived windows be committed?
    evidence: str             # how this was checked


PUBLIC_DOMAIN_US = "public domain (US Government work)"
# Upstream licenses of the GIFT-Eval datasets used in the `real` set, checked on 2026-10-27 (UTC) at the original
# publisher (the Salesforce/GiftEval card's apache-2.0 tag covers the collection, not each dataset's origin).
# Only datasets whose origin grants redistribution are committed; see GE_REDISTRIBUTABLE in otp/realdata.py.
_GE_CHECKED = "checked 2026-10-27 UTC"
GE_UPSTREAM: dict[str, License] = {
    "electricity/H": License("CC BY 4.0 (UCI ElectricityLoadDiagrams20112014)", "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014", True,
                             f"UCI dataset page states CC BY 4.0; {_GE_CHECKED}"),
    "electricity/D": License("CC BY 4.0 (UCI ElectricityLoadDiagrams20112014)", "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014", True,
                             f"UCI dataset page states CC BY 4.0; daily totals of the same data; {_GE_CHECKED}"),
    "solar/H": License("NREL data notice (Solar Power Data for Integration Studies)", "https://data.nrel.gov/node/1/license", False,
                       "NREL terms allow use and copying if the entire NREL notice appears in all copies; the notice text is not yet "
                       f"included in this repository, so windows stay restricted until it is; {_GE_CHECKED}"),
    "solar/10T": License("NREL data notice (Solar Power Data for Integration Studies)", "https://data.nrel.gov/node/1/license", False,
                         "NREL terms allow use and copying if the entire NREL notice appears in all copies; the notice text is not yet "
                         f"included in this repository, so windows stay restricted until it is; {_GE_CHECKED}"),
    "LOOP_SEATTLE/H": License("no license found (origin github.com/zhiyongc/Seattle-Loop-Data, via LibCity)", "https://github.com/zhiyongc/Seattle-Loop-Data", False,
                              f"origin repository has no LICENSE file and its README states no terms; {_GE_CHECKED}"),
    "M_DENSE/H": License("no license found (origin github.com/rdemedrano/crann_traffic, via LibCity)", "https://github.com/rdemedrano/crann_traffic", False,
                         f"origin repository has no LICENSE file; {_GE_CHECKED}"),
    "SZ_TAXI/15T": License("no license found (origin github.com/lehaifeng/T-GCN data folder, via LibCity)", "https://github.com/lehaifeng/T-GCN/tree/master/data", False,
                           f"origin repository has no LICENSE file and its README states no terms; {_GE_CHECKED}"),
    "bitbrains_fast_storage/H": License("terms not verified (Grid Workloads Archive GWA-T-12 Bitbrains)", "http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains", False,
                                        f"archive page could not be reached to read its terms; {_GE_CHECKED}"),
    "kdd_cup_2018_with_missing/H": License("CC BY 4.0 on the Monash/Zenodo copy; original KDD Cup 2018 competition terms not found", "https://zenodo.org/records/4656719", False,
                                           f"Zenodo record 4656719 states CC BY 4.0, but the competition data's own terms (kdd.org/kdd2018/kdd-cup) could not be confirmed; {_GE_CHECKED}"),
}
for _k in ("m4_hourly", "m4_daily", "m4_weekly", "m4_monthly"):
    GE_UPSTREAM[_k] = License("no license found (M4 competition data, github.com/Mcompetitions/M4-methods)", "https://github.com/Mcompetitions/M4-methods", False,
                              "origin repository has no LICENSE file and its README states no terms (the Monash archive copy on Zenodo "
                              f"is CC BY 4.0, but that does not establish the origin's grant); {_GE_CHECKED}")

SOURCES: dict[str, dict] = {
    "gifteval": dict(name="GIFT-Eval (Salesforce/GiftEval on Hugging Face)", url="https://huggingface.co/datasets/Salesforce/GiftEval",
                     license=License("apache-2.0 (Hugging Face dataset card)", "https://huggingface.co/datasets/Salesforce/GiftEval", True,
                                     "dataset card metadata read through the Hub API; per-dataset upstream terms not separately verified")),
    "gifteval_restricted": dict(name="GIFT-Eval datasets built from competition data (restaurant)", url="https://huggingface.co/datasets/Salesforce/GiftEval",
                                license=License("apache-2.0 on the Hub card, but upstream is a Kaggle competition dataset", "https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/rules", False,
                                                "competition data: derived windows kept out of git")),
    "fev": dict(name="fev-bench datasets (autogluon/fev_datasets on Hugging Face)", url="https://huggingface.co/datasets/autogluon/fev_datasets",
                license=License("other: card says 'refer to the original sources for licensing', research use only", "https://huggingface.co/datasets/autogluon/fev_datasets", False,
                                "dataset card text; no redistribution grant, so derived windows are kept out of git")),
    "monash": dict(name="Monash Time Series Forecasting Repository (HF mirror)", url="https://huggingface.co/datasets/Monash-University/monash_tsf",
                   license=License("cc-by-4.0 (Hugging Face dataset card)", "https://huggingface.co/datasets/Monash-University/monash_tsf", True,
                                   "dataset card metadata read through the Hub API")),
    "fred": dict(name="FRED (St. Louis Fed), series with the tag 'public domain: citation requested'", url="https://fred.stlouisfed.org/",
                 license=License("public domain: citation requested (series tag on each FRED page)", "https://fred.stlouisfed.org/legal/", True,
                                 "series page tag read per series at fetch time")),
    "silso": dict(name="SILSO monthly mean total sunspot number V2.0", url="https://www.sidc.be/SILSO/datafiles",
                  license=License("CC BY-NC 4.0 (SILSO data policy)", "https://www.sidc.be/SILSO/datafiles", False,
                                  "data policy text on the SILSO data files page; non-commercial, so windows are kept out of git")),
    "rdatasets": dict(name="R datasets package (lynx, lh) via Rdatasets", url="https://vincentarelbundock.github.io/Rdatasets/",
                      license=License("GPL (R datasets package); original records Elton and Nicholson 1942 (lynx)", "https://vincentarelbundock.github.io/Rdatasets/", False,
                                      "R package license; redistribution terms for derived windows not clear, so kept out of git")),
    "noaa_psl": dict(name="NOAA PSL Nino 3.4 anomaly index", url="https://psl.noaa.gov/data/timeseries/month/", license=License(
        PUBLIC_DOMAIN_US, "https://www.noaa.gov/information-technology/open-data-dissemination", True, "NOAA open data policy")),
    "ghcn": dict(name="GHCN-Daily station files (NOAA NCEI)", url="https://www.ncei.noaa.gov/pub/data/ghcn/daily/", license=License(
        PUBLIC_DOMAIN_US, "https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt", True, "NOAA open data policy")),
    "usgs_nwis": dict(name="USGS National Water Information System, daily discharge", url="https://waterservices.usgs.gov/", license=License(
        PUBLIC_DOMAIN_US, "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits", True, "USGS policy page")),
    "usgs_quakes": dict(name="USGS earthquake catalog (ComCat)", url="https://earthquake.usgs.gov/fdsnws/event/1/", license=License(
        PUBLIC_DOMAIN_US, "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits", True, "USGS policy page")),
    "owid": dict(name="Our World in Data COVID-19 dataset", url="https://github.com/owid/covid-19-data", license=License(
        "CC BY 4.0 (OWID)", "https://github.com/owid/covid-19-data", True, "stated in the OWID data repository and on ourworldindata.org")),
    "fingrid": dict(name="Pre-processed power grid frequency 2020-2024, Fingrid (Nordic) part", url="https://doi.org/10.5281/zenodo.15784548",
                    license=License("CC BY 4.0 (Fingrid subfolders)", "https://zenodo.org/api/records/15784548/files/LICENSE.md/content", True,
                                    "LICENSE.md in the record: Fingrid subfolders CC-BY 4.0; the TransnetBW part has no open license and is not used")),
    "obf": dict(name="OBF-Psychiatric motor activity dataset (control group)", url="https://doi.org/10.5281/zenodo.13754984", license=License(
        "CC BY 4.0", "https://zenodo.org/records/13754984", True, "Zenodo record metadata")),
    "seattle": dict(name="Fremont Bridge bicycle and scooter counter (City of Seattle)", url="https://data.seattle.gov/d/65db-xm6k", license=License(
        "Public Domain (Socrata dataset metadata)", "https://data.seattle.gov/api/views/65db-xm6k.json", True, "licenseId PUBLIC_DOMAIN in the dataset metadata")),
}

# ----------------------------------------------------------------------------- fetching


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Fetcher:
    """Downloads files into `raw_dir` and records them (source, URL, license, date, size, hash)."""

    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir
        os.makedirs(raw_dir, exist_ok=True)
        self.records: dict[str, dict] = {}

    def _record(self, source_id: str, url: str, path: str, note: str = "", license_override: License | None = None):
        s = SOURCES[source_id]
        lic = license_override or s["license"]
        self.records[path] = {
            "source_id": source_id, "name": s["name"], "url": url, "license": lic.name, "license_url": lic.url,
            "license_evidence": lic.evidence, "redistributable_derived_windows": lic.redistributable,
            "file": os.path.relpath(path, self.raw_dir), "bytes": os.path.getsize(path), "sha256": sha256_file(path),
            "date_fetched_utc": time.strftime("%Y-%m-%d", time.gmtime(os.path.getmtime(path))), "note": note}

    def get(self, source_id: str, url: str, fname: str, note: str = "", license_override: License | None = None,
            retries: int = 3) -> str:
        path = os.path.join(self.raw_dir, source_id, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            last = None
            for k in range(retries):
                try:
                    req = urllib.request.Request(url, headers=UAS[k % len(UAS)])
                    with urllib.request.urlopen(req, timeout=300) as r, open(path + ".tmp", "wb") as f:
                        while True:
                            b = r.read(1 << 20)
                            if not b:
                                break
                            f.write(b)
                    os.replace(path + ".tmp", path)
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
                    time.sleep(2 * (k + 1))
            else:
                raise RuntimeError(f"download failed for {url}: {last!r}")
        self._record(source_id, url, path, note, license_override)
        return path

    def hf(self, source_id: str, repo: str, revision: str, filename: str, note: str = "") -> str:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=os.path.join(self.raw_dir, "hf"))
        self._record(source_id, f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}", p, note)
        return p

    def write_sources_csv(self, path: str, extra_rows: list[dict] | None = None) -> None:
        rows = list(self.records.values()) + (extra_rows or [])
        cols = ["source_id", "name", "url", "license", "license_url", "license_evidence", "redistributable_derived_windows",
                "file", "bytes", "sha256", "date_fetched_utc", "note"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r["source_id"], r["file"])):
                w.writerow({c: r.get(c, "") for c in cols})


# ----------------------------------------------------------------------------- items


@dataclass
class Item:
    """One real series plus the rule for cutting its window."""
    source_id: str
    dataset: str
    series_id: str
    values: np.ndarray
    freq: str                       # frequency string (see realdata.freq_letter), e.g. "H", "15T", "B", "M"
    domain: str
    index: pd.DatetimeIndex | None = None
    horizon: int | None = None      # None: GIFT-Eval convention for the frequency
    origin: str | None = None       # first forecast time; None = last window of the (cut-off) series
    t_min: int = 256
    t_max: int = 2048
    m: int | None = None
    family: str = ""                # analogue family; "" for the wide real set
    reason: str = ""
    transform: str = ""
    redistributable: bool = True
    source_file: str = ""
    sha256: str = ""
    pick_key: str = ""              # items sharing a key are candidates; `pick` of them (seeded order, valid windows) are kept
    pick: int = 0


def log_returns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(np.where(x > 0, x, np.nan)))
    return r


def cut_at(values: np.ndarray, index: pd.DatetimeIndex | None, end: pd.Timestamp = CUTOFF):
    """Drop observations after `end`."""
    if index is None:
        return values, index
    k = int(index.searchsorted(end, side="right"))
    return values[:k], index[:k]


# ----------------------------------------------------------------------------- parsers


def read_tsf(text_or_file) -> tuple[dict, list[tuple[str, pd.Timestamp | None, np.ndarray]]]:
    """Monash .tsf format. Returns `(header, [(series_name, start, values)])`; `?` is NaN."""
    f = io.StringIO(text_or_file) if isinstance(text_or_file, str) else text_or_file
    attrs, header, out, in_data = [], {}, [], False
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not in_data:
            if line.lower().startswith("@attribute"):
                attrs.append(line.split()[1])
            elif line.lower().startswith("@data"):
                in_data = True
            elif line.startswith("@"):
                k, _, v = line[1:].partition(" ")
                header[k.lower()] = v
            continue
        parts = line.split(":", len(attrs))
        meta, vals = parts[:-1], parts[-1]
        start = None
        for a, v in zip(attrs, meta):
            if a == "start_timestamp":
                try:
                    start = pd.Timestamp(v[:10] + " " + v[11:].replace("-", ":"))
                except Exception:  # noqa: BLE001
                    start = None
        out.append((meta[0], start, np.asarray([np.nan if t == "?" else float(t) for t in vals.split(",")])))
    return header, out


def read_ghcn_dly(path: str, element: str) -> pd.Series:
    """GHCN-Daily fixed-width station file -> daily series of one element (TMAX/TMIN in degrees C, PRCP in mm).
    Missing values (-9999) and values with a quality flag are NaN."""
    rows = []
    with open(path) as f:
        for line in f:
            if line[17:21] != element:
                continue
            year, month = int(line[11:15]), int(line[15:17])
            for d in range(31):
                base = 21 + 8 * d
                v = int(line[base:base + 5])
                qflag = line[base + 6]
                try:
                    date = pd.Timestamp(year=year, month=month, day=d + 1)
                except ValueError:
                    continue
                rows.append((date, np.nan if v == -9999 or qflag.strip() else v / 10.0))
    s = pd.Series(dict(rows)).sort_index()
    return s.asfreq("D")


def read_rdb(path: str) -> pd.DataFrame:
    """USGS RDB (tab separated, # comments, a column-type row after the header)."""
    lines = [ln for ln in open(path) if not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", dtype=str).iloc[1:]
    return df


# ----------------------------------------------------------------------------- builders: wide real set


FRED_DAILY = {  # series id: (description, domain)
    "DEXUSEU": ("US dollars per euro", "finance"), "DEXJPUS": ("yen per US dollar", "finance"),
    "DEXUSUK": ("US dollars per pound", "finance"), "DEXCAUS": ("Canadian dollars per US dollar", "finance"),
    "DEXCHUS": ("yuan per US dollar", "finance"), "DEXINUS": ("rupees per US dollar", "finance"),
    "DGS10": ("10-year Treasury yield", "finance"), "DGS2": ("2-year Treasury yield", "finance"),
    "DCOILWTICO": ("WTI crude oil price", "energy"), "DCOILBRENTEU": ("Brent crude oil price", "energy"),
    "DHHNGSP": ("Henry Hub natural gas price", "energy"),
}
FRED_MONTHLY = ["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "FEDFUNDS", "HOUST", "RSAFS", "M2SL"]


def fred_series(fx: Fetcher, sid: str, start: str = "1900-01-01") -> tuple[pd.Series, bool, str]:
    """Fetch one FRED series (observations up to the cut-off) and check the public-domain tag on its page."""
    page = fx.get("fred", f"https://fred.stlouisfed.org/series/{sid}", f"{sid}.html", note="series page, license tag")
    tags = re.findall(r'name="series-tag" content="([^"]*)"', open(page, errors="ignore").read())
    pd_tag = any(t.startswith("public domain") for t in tags)
    p = fx.get("fred", f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}&coed={CUTOFF.date()}",
               f"{sid}.csv", note="observations up to the cut-off date")
    df = pd.read_csv(p)
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(df.iloc[:, 0]))
    return s, pd_tag, p


def build_fred_daily(fx: Fetcher, ids=None) -> list[Item]:
    out = []
    for sid, (desc, dom) in FRED_DAILY.items():
        if ids and sid not in ids:
            continue
        s, pdtag, p = fred_series(fx, sid)
        s = s[s.index >= "1990-01-01"]
        out.append(Item("fred", "fred_daily", sid, s.values, "B", dom, s.index, redistributable=pdtag,
                        source_file=os.path.relpath(p, fx.raw_dir), sha256=sha256_file(p)[:16], reason=desc))
    return out


def build_ghcn(fx: Fetcher) -> list[Item]:
    stations = {"USW00094728": "New York Central Park", "USW00023174": "Los Angeles airport", "USW00012960": "Houston airport",
                "USW00014922": "Minneapolis airport", "USW00026451": "Anchorage airport", "USW00022521": "Honolulu airport"}
    out = []
    for sid, name in stations.items():
        p = fx.get("ghcn", f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{sid}.dly", f"{sid}.dly")
        for el, unit in (("TMAX", "deg C"), ("PRCP", "mm")):
            if el == "PRCP" and sid not in ("USW00012960", "USW00022521", "USW00023174"):
                continue
            s = read_ghcn_dly(p, el)
            s = s[(s.index >= "2000-01-01") & (s.index <= CUTOFF)]
            out.append(Item("ghcn", f"ghcn_daily_{el}", f"{sid}:{el}", s.values, "D", "weather", s.index,
                            source_file=os.path.relpath(p, fx.raw_dir), sha256=sha256_file(p)[:16], reason=f"{name}, {el} ({unit})"))
    return out


NWIS_SITES = {"01646500": "Potomac River near Washington DC", "07010000": "Mississippi River at St. Louis",
              "11447650": "Sacramento River at Freeport", "09380000": "Colorado River at Lees Ferry",
              "08158000": "Colorado River at Austin TX", "12113000": "Green River, Washington",
              "06191500": "Yellowstone River at Corwin Springs", "03612600": "Ohio River at Olmsted"}


def build_nwis(fx: Fetcher) -> list[Item]:
    out = []
    for site, name in NWIS_SITES.items():
        url = (f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={site}&parameterCd=00060&statCd=00003"
               f"&startDT=2000-01-01&endDT={CUTOFF.date()}")
        p = fx.get("usgs_nwis", url, f"{site}.rdb")
        df = read_rdb(p)
        col = [c for c in df.columns if re.match(r"\d+_00060_00003$", c)]
        if not col or df.empty:
            continue
        s = pd.Series(pd.to_numeric(df[col[0]], errors="coerce").values, index=pd.to_datetime(df["datetime"])).sort_index().asfreq("D")
        out.append(Item("usgs_nwis", "usgs_daily_discharge", site, s.values, "D", "ecology_hydrology", s.index,
                        source_file=os.path.relpath(p, fx.raw_dir), sha256=sha256_file(p)[:16], reason=name))
    return out


def build_owid(fx: Fetcher) -> list[Item]:
    p = fx.get("owid", "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv", "owid-covid-data.csv")
    df = pd.read_csv(p, usecols=["location", "date", "new_cases"], parse_dates=["date"])
    out = []
    for loc in ["United States", "Germany", "Brazil", "India", "Japan", "France", "Italy", "Mexico", "United Kingdom", "Canada"]:
        d = df[(df.location == loc) & (df.date <= CUTOFF)].set_index("date")["new_cases"].asfreq("D")
        d = d[d.first_valid_index():]
        out.append(Item("owid", "owid_covid_new_cases", loc, d.values, "D", "epidemiology", d.index,
                        source_file=os.path.relpath(p, fx.raw_dir), sha256=sha256_file(p)[:16], reason=f"daily new COVID-19 cases, {loc}"))
    return out


def _monash(fx: Fetcher, name: str):
    p = fx.hf("monash", MONASH_REPO, MONASH_REVISION, f"data/{name}.zip")
    z = zipfile.ZipFile(p)
    tsf = [n for n in z.namelist() if n.endswith(".tsf")][0]
    return p, read_tsf(io.TextIOWrapper(z.open(tsf), encoding="utf-8", errors="ignore"))


def build_monash_extra(fx: Fetcher, seed: int) -> list[Item]:
    out = []
    p, (hdr, ser) = _monash(fx, "sunspot_dataset_with_missing_values")
    for n, st, v in ser:
        idx = pd.date_range(st, periods=len(v), freq="D")
        out.append(Item("monash", "monash_sunspot_daily", n, v, "D", "astronomy", idx, source_file=os.path.relpath(p, fx.raw_dir),
                        sha256=sha256_file(p)[:16], reason="daily sunspot number"))
    p, (hdr, ser) = _monash(fx, "pedestrian_counts_dataset")
    for n, st, v in ser:
        idx = pd.date_range(st, periods=len(v), freq="h")
        out.append(Item("monash", "monash_pedestrian_counts", n, v, "H", "sensors_transport", idx, source_file=os.path.relpath(p, fx.raw_dir),
                        sha256=sha256_file(p)[:16], reason="hourly pedestrian count, sensor of an Australian city"))
    p, (hdr, ser) = _monash(fx, "bitcoin_dataset_with_missing_values")
    for n, st, v in ser:
        idx = pd.date_range(st, periods=len(v), freq="D")
        out.append(Item("monash", "monash_bitcoin", n, v, "D", "finance", idx, source_file=os.path.relpath(p, fx.raw_dir),
                        sha256=sha256_file(p)[:16], reason=f"bitcoin network series {n}"))
    return out


def build_web_traffic(fx: Fetcher, seed: int, n: int = 16) -> list[Item]:
    """Kaggle web traffic (Wikipedia page views), daily, from the Monash mirror: `n` seeded series."""
    p = fx.hf("monash", MONASH_REPO, MONASH_REVISION, "data/kaggle_web_traffic_dataset_with_missing_values.zip")
    z = zipfile.ZipFile(p)
    tsf = [n_ for n_ in z.namelist() if n_.endswith(".tsf")][0]
    rng = np.random.default_rng(seed)
    want = None
    items = []
    with z.open(tsf) as raw:
        lines = io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")
        data_lines = []
        in_data = False
        for line in lines:
            if in_data:
                data_lines.append(line)
            elif line.strip().lower().startswith("@data"):
                in_data = True
    order = rng.permutation(len(data_lines))
    for i in order:
        _, ser = read_tsf("@attribute series_name string\n@attribute start_timestamp date\n@data\n" + data_lines[i])
        nme, st, v = ser[0]
        if np.isnan(v).mean() > 0.01:
            continue
        idx = pd.date_range(st, periods=len(v), freq="D")
        items.append(Item("monash", "monash_web_traffic_daily", nme, v, "D", "web_traffic", idx, source_file=os.path.relpath(p, fx.raw_dir),
                          sha256=sha256_file(p)[:16], reason="daily page views"))
        if len(items) >= n:
            break
    return items


def build_seattle(fx: Fetcher) -> list[Item]:
    p = fx.get("seattle", "https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD", "fremont_bridge.csv")
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p")
    s = df.set_index("Date").iloc[:, 0].sort_index()
    s = s[~s.index.duplicated()].asfreq("h")
    s = s[s.index <= CUTOFF]
    return [Item("seattle", "seattle_fremont_bridge", "fremont_total", s.values, "H", "transport", s.index,
                 source_file=os.path.relpath(p, fx.raw_dir), sha256=sha256_file(p)[:16], reason="hourly bicycle and scooter crossings")]


def build_obf(fx: Fetcher, n: int = 8) -> list[Item]:
    p = fx.get("obf", "https://zenodo.org/api/records/13754984/files/OBF-Psychiatric%20Dataset.zip/content", "obf.zip")
    z = zipfile.ZipFile(p)
    names = sorted(x for x in z.namelist() if "/control/" in x and x.endswith(".csv"))
    out = []
    for nme in names[:n]:
        df = pd.read_csv(z.open(nme), parse_dates=["timestamp"])
        s = df.set_index("timestamp")["activity"].astype(float)
        s = s[~s.index.duplicated()].asfreq("min")
        s = s.resample("15min").sum(min_count=10)  # 15-minute activity counts
        out.append(Item("obf", "obf_control_activity_15min", os.path.basename(nme)[:-4], s.values, "15T", "physiology", s.index,
                        horizon=48, t_min=256, m=96, source_file=os.path.relpath(p, fx.raw_dir), sha256=sha256_file(p)[:16],
                        reason="wrist actigraphy activity counts, healthy control, 15-minute bins"))
    return out


# ----------------------------------------------------------------------------- builders: analogues


def _item_from_series(src, dataset, sid, s: pd.Series, freq, domain, fx, p, **kw) -> Item:
    return Item(src, dataset, sid, s.values.astype(float), freq, domain, s.index, source_file=os.path.relpath(p, fx.raw_dir),
                sha256=sha256_file(p)[:16], **kw)


def build_analogues(fx: Fetcher, seed: int) -> list[Item]:
    """Series that stand in for the synthetic failure families. Each has a family tag and a reason."""
    A: list[Item] = []
    # ---- delay feedback / chaos-like
    p = fx.get("rdatasets", "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/lynx.csv", "lynx.csv")
    df = pd.read_csv(p)
    idx = pd.date_range("1821-01-01", periods=len(df), freq="YS")
    A.append(Item("rdatasets", "lynx", "lynx_annual", df.iloc[:, -1].values.astype(float), "A", "ecology", idx, horizon=10, t_min=96, m=1,
                  family="delay_feedback", reason="Canadian lynx trappings 1821-1934: predator-prey cycle of about 10 years with delayed density dependence",
                  redistributable=False, source_file=os.path.relpath(p, fx.raw_dir), sha256=sha256_file(p)[:16]))
    p = fx.get("rdatasets", "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/lh.csv", "lh.csv")
    df = pd.read_csv(p)
    idx = pd.date_range("2000-01-01", periods=len(df), freq="10min")
    A.append(Item("rdatasets", "lh", "lh_hormone", df.iloc[:, -1].values.astype(float), "10T", "physiology", idx, horizon=12, t_min=24, m=1,
                  family="delay_feedback", reason="luteinizing hormone in blood samples every 10 minutes: pulsatile secretion under hormonal feedback",
                  redistributable=False, source_file=os.path.relpath(p, fx.raw_dir), sha256=sha256_file(p)[:16]))
    # sunspots, Monash daily (CC BY): origins at documented points of solar cycles 24 and 25
    for it in build_monash_extra(fx, seed):
        if it.dataset == "monash_sunspot_daily":
            for origin, why in (("2008-12-01", "solar minimum between cycles 23 and 24 (December 2008)"),
                                ("2014-04-01", "maximum of solar cycle 24 (smoothed maximum in April 2014)"),
                                (None, "last window of the record (May 2020, close to the minimum before cycle 25)")):
                A.append(Item(**{**it.__dict__, "origin": origin, "family": "delay_feedback", "t_min": 256,
                                 "reason": f"daily sunspot number; origin at {why}; solar dynamo cycle of about 11 years"}))
    # sunspots, SILSO monthly (CC BY-NC): restricted
    p = fx.get("silso", "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv", "SN_m_tot_V2.0.csv")
    df = pd.read_csv(p, sep=";", header=None, names=["y", "m", "dec", "sn", "sd", "n", "prov"])
    idx = pd.to_datetime(dict(year=df.y, month=df.m, day=1)) + pd.offsets.MonthBegin(0)
    s = pd.Series(df.sn.values.astype(float), index=pd.DatetimeIndex(idx))
    s = s[s >= 0]
    for origin, why in (("2008-12-01", "solar minimum between cycles 23 and 24"), ("2014-04-01", "maximum of cycle 24"),
                        ("2019-12-01", "solar minimum between cycles 24 and 25")):
        A.append(_item_from_series("silso", "silso_monthly_sunspots", "silso_monthly", s, "M", "astronomy", fx, p, horizon=12, origin=origin, m=12,
                                   family="delay_feedback", reason=f"monthly mean sunspot number; origin at {why}", redistributable=False))
    # ENSO delayed oscillator (Nino 3.4 anomaly)
    p = fx.get("noaa_psl", "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data", "nino34.long.anom.data")
    rows = []
    for line in open(p):
        t = line.split()
        if len(t) == 13 and t[0].isdigit():
            for mth, v in enumerate(t[1:], 1):
                rows.append((pd.Timestamp(int(t[0]), mth, 1), float(v)))
    s = pd.Series(dict(rows)).sort_index()
    s = s[(s > -90) & (s.index <= CUTOFF)]
    for origin, why in (("1997-03-01", "onset of the 1997-98 El Nino"), ("2015-03-01", "onset of the 2015-16 El Nino"),
                        ("2020-08-01", "onset of the 2020-23 La Nina"), (None, "last window of the record")):
        A.append(_item_from_series("noaa_psl", "nino34_monthly", "nino34", s, "M", "climate", fx, p, horizon=12, origin=origin, m=12,
                                   family="delay_feedback", reason=f"Nino 3.4 anomaly, ENSO is a delayed oscillator; {why}"))
    return A


def build_analogues_part2(fx: Fetcher, seed: int) -> list[Item]:
    A: list[Item] = []

    def keep(items, fam, reason_fn=None, **over):
        for it in items:
            it.family = fam
            for k, v in over.items():
                setattr(it, k, v)
            if reason_fn:
                it.reason = reason_fn(it)
            A.append(it)

    # ---- coupled oscillators
    from remotezip import RemoteZip

    zurl = "https://zenodo.org/api/records/15784548/files/Data_cleansed.zip/content"
    local = os.path.join(fx.raw_dir, "fingrid", "Fingrid_2024.zip")
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with RemoteZip(zurl) as z:
            data = z.read("Data_cleansed/2024_cleansed/Fingrid/2024.zip")
        with open(local, "wb") as f:
            f.write(data)
    fx._record("fingrid", zurl + "#Data_cleansed/2024_cleansed/Fingrid/2024.zip", local, "member of the record's Data_cleansed.zip")
    fx.get("fingrid", "https://zenodo.org/api/records/15784548/files/LICENSE.md/content", "LICENSE.md", note="record license statement")
    df = pd.read_csv(zipfile.ZipFile(local).open("2024.csv"), header=None, names=["t", "f"], parse_dates=["t"])
    sf = df.set_index("t")["f"].astype(float)
    sf = sf[~sf.index.duplicated()].asfreq("s")
    for start in ("2024-01-08 03:00", "2024-01-15 15:00", "2024-02-05 03:00", "2024-02-19 15:00", "2024-03-04 03:00", "2024-03-18 15:00"):
        origin = str(pd.Timestamp(start) + pd.Timedelta(seconds=2048))
        A.append(_item_from_series("fingrid", "fingrid_frequency_1s", f"fingrid@{start}", sf, "S", "energy", fx, local, horizon=300, origin=origin,
                                   t_min=256, m=1, family="coupled_oscillators",
                                   reason="Nordic grid frequency in Hz at 1 s: many synchronous generators coupled through the network, inter-area oscillation modes"))
    # circadian activity
    obf = build_obf(fx, n=8)
    keep(obf, "coupled_oscillators", lambda it: "wrist activity of a healthy control, 15-minute bins: circadian rhythm with ultradian components")
    # pedestrian sensors before the pandemic: coupled city-wide daily cycles
    ped = [it for it in build_monash_extra(fx, seed) if it.dataset == "monash_pedestrian_counts"]
    for it in ped:
        A.append(Item(**{**it.__dict__, "origin": "2019-11-04", "horizon": 48, "family": "coupled_oscillators", "pick_key": "ped_2019", "pick": 5,
                         "reason": "hourly pedestrian counts at sensors of an Australian city in a normal week: sensors driven by the same daily and weekly cycles"}))
    # ---- heavy tails
    fxs = [it for it in build_fred_daily(fx, ids={"DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXCAUS", "DEXINUS"})]
    for it in fxs:
        it.family, it.transform, it.m = "heavy_tails", "log_return", 5
        it.reason = f"daily log return of {it.reason}: fat-tailed returns with volatility clustering"
        A.append(it)
    for it in build_fred_daily(fx, ids={"DCOILWTICO", "DCOILBRENTEU", "DHHNGSP"}):
        it.family, it.transform, it.m = "heavy_tails", "log_return", 5
        it.reason = f"daily log return of {it.reason}: fat-tailed returns"
        A.append(it)
    btc = [it for it in build_monash_extra(fx, seed) if it.dataset == "monash_bitcoin"]
    for it in btc:
        A.append(Item(**{**it.__dict__, "transform": "log_return", "family": "heavy_tails", "pick_key": "btc", "pick": 2, "m": 7, "t_min": 256,
                         "reason": f"daily log change of bitcoin network series {it.series_id}: fat-tailed"}))
    # earthquakes
    frames = []
    for yr in range(2000, 2027):
        end = f"{yr + 1}-01-01" if yr < 2026 else str(CUTOFF.date() + pd.Timedelta(days=1))
        url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&minmagnitude=4.5&orderby=time-asc"
               f"&starttime={yr}-01-01&endtime={end}")
        pth = fx.get("usgs_quakes", url, f"quakes_{yr}.csv")
        frames.append(pd.read_csv(pth, usecols=["time", "mag"], parse_dates=["time"]))
    q = pd.concat(frames)
    q["day"] = q["time"].dt.tz_localize(None).dt.floor("D")
    days = pd.date_range("2000-01-01", CUTOFF, freq="D")
    cnt = q.groupby("day").size().reindex(days, fill_value=0).astype(float)
    mx = q.groupby("day")["mag"].max().reindex(days).fillna(4.4)
    qp = os.path.join(fx.raw_dir, "usgs_quakes", "quakes_2026.csv")
    for nm, ser, why in (("quake_daily_count_M4.5", cnt, "daily count of M4.5+ earthquakes worldwide: aftershock clusters, heavy-tailed counts"),
                         ("quake_daily_max_magnitude", mx, "daily maximum magnitude of M4.5+ earthquakes worldwide (4.4 on days without one): magnitudes follow a power-law tail")):
        A.append(_item_from_series("usgs_quakes", "usgs_quakes_daily", nm, ser, "D", "geophysics", fx, qp, family="heavy_tails", reason=why))
    # floods: daily discharge of rivers with flashy peaks
    for it in build_nwis(fx):
        if it.series_id in {"01646500", "12113000", "06191500", "03612600"}:
            it.family = "heavy_tails"
            it.reason = f"daily discharge, {it.reason}: flood peaks with a heavy upper tail"
            A.append(it)
    # ---- regime switches
    macro = {("INDPRO", "2020-02-01"): "COVID-19 shutdown of US industry from March 2020",
             ("INDPRO", "2008-09-01"): "global financial crisis after September 2008",
             ("PAYEMS", "2020-02-01"): "COVID-19 job losses from March 2020",
             ("UNRATE", "2020-02-01"): "COVID-19 unemployment jump in April 2020",
             ("RSAFS", "2020-02-01"): "COVID-19 retail sales collapse and rebound",
             ("CPIAUCSL", "2021-02-01"): "inflation surge from spring 2021",
             ("FEDFUNDS", "2022-02-01"): "start of the Federal Reserve tightening cycle in March 2022",
             ("HOUST", "2007-06-01"): "US housing starts during the housing bust",
             ("M2SL", "2020-03-01"): "jump in the money supply from March 2020"}
    for (sid, origin), why in macro.items():
        s, pdtag, pth = fred_series(fx, sid)
        A.append(Item("fred", "fred_monthly_breaks", f"{sid}@{origin}", s.values, "M", "economics", s.index, horizon=12, origin=origin, t_min=256, m=12,
                      family="regime_switches", reason=f"{sid}: {why}", redistributable=pdtag,
                      source_file=os.path.relpath(pth, fx.raw_dir), sha256=sha256_file(pth)[:16]))
    energy = {("DCOILWTICO", "2020-04-01", 30): "WTI front-month price fell below zero on 2020-04-20",
              ("DHHNGSP", "2021-02-01", 30): "Henry Hub gas spot price spike during Winter Storm Uri, 13-17 February 2021",
              ("DCOILBRENTEU", "2014-09-30", 60): "oil price collapse from mid-2014",
              ("DCOILWTICO", "2022-02-15", 60): "oil price spike after the invasion of Ukraine on 2022-02-24"}
    for (sid, origin, h), why in energy.items():
        s, pdtag, pth = fred_series(fx, sid)
        s = s[s.index >= "1990-01-01"]
        A.append(Item("fred", "fred_energy_breaks", f"{sid}@{origin}", s.values, "B", "energy", s.index, horizon=h, origin=origin, m=5,
                      family="regime_switches", reason=f"{sid}: {why}", redistributable=pdtag, source_file=os.path.relpath(pth, fx.raw_dir),
                      sha256=sha256_file(pth)[:16]))
    for it in build_seattle(fx):
        for origin, h, why in (("2020-03-09", 480, "Washington stay-home order of 2020-03-23 inside the horizon"),
                               ("2019-12-18", 168, "Christmas week"), ("2019-11-25", 168, "Thanksgiving week (2019-11-28)")):
            A.append(Item(**{**it.__dict__, "origin": origin, "horizon": h, "family": "regime_switches",
                             "reason": f"hourly bicycle and scooter crossings, Fremont Bridge Seattle: {why}"}))
    for it in ped:
        A.append(Item(**{**it.__dict__, "origin": "2020-03-09", "horizon": 480, "family": "regime_switches", "pick_key": "ped_covid", "pick": 4,
                         "reason": "hourly pedestrian counts at sensors of an Australian city: COVID-19 restrictions in March 2020 inside the horizon"}))
    for it in build_owid(fx):
        if it.series_id in ("United States", "Germany", "Japan"):
            A.append(Item(**{**it.__dict__, "origin": "2021-12-01", "horizon": 60, "family": "regime_switches",
                             "reason": f"daily COVID-19 cases, {it.series_id}: Omicron wave after the variant was named on 2021-11-26"}))
    return A


# ----------------------------------------------------------------------------- items -> windows


def item_window(it: Item):
    """Window for one item: `(context, future, info)` or `(None, None, reason)`."""
    from otp import realdata as R

    v, idx = it.values.astype(float), it.index
    v, idx = cut_at(v, idx)
    if it.freq == "B":  # business-day series: holidays are missing days, not steps; keep observed days only
        ok = np.isfinite(v)
        v, idx = v[ok], idx[ok]
    if it.origin is None and idx is not None:  # last window: ignore a trailing run of missing values
        fin = np.flatnonzero(np.isfinite(v))
        if len(fin):
            v, idx = v[:fin[-1] + 1], idx[:fin[-1] + 1]
    if it.transform == "log_return":
        v, idx = log_returns(v), (idx[1:] if idx is not None else None)
    h = it.horizon or R.gift_eval_horizon(it.freq, it.dataset)
    if it.origin is not None:
        pos = int(idx.searchsorted(pd.Timestamp(it.origin)))
        if pos >= len(idx) or pos + h > len(v):
            return None, None, "origin_outside_record"
        v, idx = v[:pos + h], idx[:pos + h]
    ctx, fut, info = R.cut_window(v, h, it.t_max, it.t_min)
    if ctx is None:
        return None, None, info
    info["origin_time"] = str(idx[info["origin"]]) if idx is not None else ""
    info["horizon"] = h
    return ctx, fut, info


def items_to_samples(items: list[Item], set_name: str, seed: int, log=print):
    """Samples and manifest rows (shared columns with `realdata.load_pair`, plus family and reason)."""
    from otp import realdata as R
    from otp.generators.base import Sample
    from otp.model import QUANTILE_LEVELS
    import zlib

    out, rows = [], []
    groups: dict[str, list[Item]] = {}
    plain: list[Item] = []
    for it in items:
        (groups.setdefault(f"{it.dataset}|{it.family}|{it.origin}|{it.pick_key}", []) if it.pick_key else plain).append(it)
    selected = list(plain)
    for key, g in groups.items():
        order = np.random.default_rng([seed, zlib.crc32(key.encode())]).permutation(len(g))
        ok = 0
        for i in order:
            if ok >= g[0].pick:
                break
            ctx, fut, info = item_window(g[i])
            if ctx is not None:
                selected.append(g[i])
                ok += 1
    skipped = {}
    for it in selected:
        ctx, fut, info = item_window(it)
        if ctx is None:
            skipped[info] = skipped.get(info, 0) + 1
            log(f"  skipped {it.dataset}/{it.series_id} origin={it.origin}: {info}")
            continue
        m = it.m or R.seasonal_period(it.freq)
        sid = f"{set_name}/{it.dataset}/{it.series_id}" + (f"@{it.origin}" if it.origin and "@" not in it.series_id else "")
        h = info["horizon"]
        meta = {"family": "real", "generator": it.dataset, "knob": None, "knob_value": None, "seed": int(seed), "T": len(ctx), "H": h, "tags": [],
                "sample_id": sid, "domain": it.domain, "freq": it.freq, "source": it.source_id, "family_analogue": it.family}
        out.append(Sample(ctx[None, :], fut[None, :], np.full((1, h, len(QUANTILE_LEVELS)), np.nan), m, meta))
        rows.append({
            "set": set_name, "sample_id": sid, "source": it.source_id, "revision": "", "source_file": it.source_file, "source_sha256": it.sha256,
            "dataset": it.dataset, "term": "short", "domain": it.domain, "freq": it.freq, "series_id": it.series_id, "series_index": "",
            "series_len": info["length"], "window_start": info["ctx_start"], "origin_index": info["origin"], "origin_time": info["origin_time"],
            "T": len(ctx), "H": h, "m": m, "n_filled": info["n_filled"], "first_variate_of_multivariate": False,
            "value_col": it.transform or "", "redistributable": bool(it.redistributable), "family_analogue": it.family, "reason": it.reason})
    return out, rows


# ----------------------------------------------------------------------------- collections


def collect_extra_items(fx: Fetcher, seed: int) -> list[Item]:
    """Wide real set: last windows from open-license sources across domains."""
    items = build_fred_daily(fx) + build_ghcn(fx) + build_nwis(fx) + build_owid(fx) + build_web_traffic(fx, seed) + build_seattle(fx) + build_obf(fx)
    for it in build_monash_extra(fx, seed):
        if it.dataset == "monash_pedestrian_counts":
            it.pick_key, it.pick = "ped_last", 12
        elif it.dataset == "monash_bitcoin":
            it.pick_key, it.pick = "btc_last", 6
        items.append(it)
    return items


def collect_analogue_items(fx: Fetcher, seed: int) -> list[Item]:
    return build_analogues(fx, seed) + build_analogues_part2(fx, seed)


def hub_file_rows(pairs, cache_dir: str) -> list[dict]:
    """sources.csv rows for the GIFT-Eval and fev-bench files used by the first real set."""
    from otp import realdata as R

    rows, seen = [], set()
    for sp in pairs:
        if (sp.source, sp.path) in seen:
            continue
        seen.add((sp.source, sp.path))
        if sp.source == "gifteval":
            fn, repo, rev = f"{sp.path}/data-00000-of-00001.arrow", R.GE_REPO, R.GE_REVISION
            sid = "gifteval" if (sp.redistributable or sp.path in GE_UPSTREAM) else "gifteval_restricted"
        else:
            fn, repo, rev = f"{sp.path}/train-00000-of-00001.parquet", R.FEV_REPO, R.FEV_REVISION
            sid = "fev"
        local = R._download(repo, rev, fn, cache_dir)
        lic = SOURCES[sid]["license"]
        if sp.source == "gifteval" and sp.path in GE_UPSTREAM:
            lic = GE_UPSTREAM[sp.path]
        rows.append({"source_id": sid, "name": SOURCES[sid]["name"] + f" / {sp.path}", "url": f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{fn}",
                     "license": lic.name, "license_url": lic.url, "license_evidence": lic.evidence,
                     "redistributable_derived_windows": lic.redistributable, "file": fn, "bytes": os.path.getsize(local),
                     "sha256": sha256_file(local), "date_fetched_utc": time.strftime("%Y-%m-%d", time.gmtime(os.path.getmtime(local))),
                     "note": f"revision {rev}"})
    return rows
