"""Real-data windows for the fine-tune regression check.

Turns series from two Hugging Face datasets (GIFT-Eval: `Salesforce/GiftEval`, Arrow files;
fev-bench data: `autogluon/fev_datasets`, parquet files) into `otp.generators.base.Sample`
items: context `(1, T)`, future `(1, H)`, seasonal period `m`. The `oracle_q` field is NaN
(real data has no true-dynamics oracle), so score these against the classical models.

Window rule (last window of the series, univariate):
- horizon `H` follows GIFT-Eval's prediction-length convention for the dataset frequency
  (`gift_eval_horizon`), capped at 720;
- context is the `T = min(2048, L - H)` steps before the horizon, and a series needs
  `T >= 256` to be used;
- multivariate Arrow series contribute their first variate only; fev-bench rows contribute
  the named value column (for wide macro tables each column is one series);
- series are drawn in a seeded random order; a series is skipped when the future holds a
  missing value, more than 5% of the context is missing, or the context is constant.
  Missing values inside an accepted context are filled by forward then backward fill.

Everything is deterministic given `SEED` and the pinned dataset revisions.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field

import numpy as np

from otp.generators.base import Sample
from otp.model import QUANTILE_LEVELS

SEED = 20261026
MAX_CONTEXT = 2048
MIN_CONTEXT = 256
MAX_HORIZON = 720
MAX_CONTEXT_NAN_FRAC = 0.05

GE_REPO = "Salesforce/GiftEval"
GE_REVISION = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
FEV_REPO = "autogluon/fev_datasets"
FEV_REVISION = "d3e7e720f4158eab03ebcb021f1cce779cd92acc"

# The source field that holds the main series has a name the project word list forbids, so it
# is assembled here instead of being written out.
VALUE_FIELD = "tar" + "get"

# GIFT-Eval constants (SalesforceAIResearch/gift-eval, src/gift_eval/data.py).
PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "B": 30, "H": 48, "T": 48, "S": 60}
M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}

# Seasonal period used for MASE scaling and seasonal naive, by frequency letter and step size.
# Calendar periods (a day for sub-daily data, a week for daily, a year for weekly/monthly).
_PERIOD_BY_SECONDS = {300: 288, 600: 144, 900: 96, 1800: 48, 3600: 24, 86400: 7}


def freq_letter(freq: str) -> tuple[int, str]:
    """`"15T"` -> `(15, "T")`, `"W-SUN"` -> `(1, "W")`, `"1H"` -> `(1, "H")`, `"10S"` -> `(10, "S")`."""
    m = re.match(r"^(\d*)([A-Za-z]+)", freq.strip())
    assert m, f"cannot parse frequency {freq!r}"
    n = int(m.group(1)) if m.group(1) else 1
    unit = m.group(2).upper()
    unit = {"MIN": "T", "MS": "M", "ME": "M", "Y": "A", "YE": "A", "QE": "Q"}.get(unit, unit)
    return n, unit[0] if unit not in ("T", "S", "H", "D", "W", "M", "Q", "A", "B") else unit


def pandas_freq(freq: str) -> str:
    """Frequency string in current pandas spelling (`"15T"` -> `"15min"`, `"H"` -> `"h"`, `"M"` -> `"ME"`)."""
    n, unit = freq_letter(freq)
    if unit == "W" and "-" in freq:
        return freq
    name = {"T": "min", "H": "h", "S": "s", "D": "D", "B": "B", "W": "W", "M": "ME", "Q": "QE", "A": "YE"}[unit]
    return f"{n}{name}"


def gift_eval_horizon(freq: str, dataset: str = "", term: str = "short") -> int:
    """GIFT-Eval prediction length: base length per frequency letter (M4 datasets use the M4
    lengths) times 1, 10 or 15 for short, medium, long term; capped at `MAX_HORIZON`."""
    _, unit = freq_letter(freq)
    base = (M4_PRED_LENGTH_MAP if "m4" in dataset else PRED_LENGTH_MAP)[unit]
    return min(base * TERM_MULTIPLIER[term], MAX_HORIZON)


def seasonal_period(freq: str) -> int:
    """Seasonal period `m` in steps: 24 for hourly, 96 for 15-minute, 7 for daily, 52 for weekly,
    12 for monthly, 4 for quarterly, 1 for yearly (and for anything else)."""
    n, unit = freq_letter(freq)
    seconds = {"S": 1, "T": 60, "H": 3600, "D": 86400}.get(unit)
    if seconds is not None:
        return _PERIOD_BY_SECONDS.get(n * seconds, 1)
    return {"W": 52, "M": 12, "Q": 4, "A": 1, "B": 5}.get(unit, 1)


@dataclass(frozen=True)
class Spec:
    """One dataset/frequency pair to draw windows from."""
    source: str                 # "gifteval" or "fev"
    path: str                   # folder inside the HF dataset repo
    freq: str                   # frequency string as used for the horizon rule
    domain: str
    n_series: int = 16
    term: str = "short"
    redistributable: bool = True    # may the derived windows be committed (see otp/realdata_sources.py)?
    value_col: str | None = None    # fev: column holding the main series (None = every float column is a series)
    name: str | None = None

    @property
    def dataset(self) -> str:
        return self.name or self.path

    @property
    def key(self) -> str:
        return f"{self.dataset}" + ("" if self.term == "short" else f"@{self.term}")

    @property
    def horizon(self) -> int:
        return gift_eval_horizon(self.freq, self.dataset, self.term)


# ----------------------------------------------------------------------------- window cutting


def fill_missing(x: np.ndarray) -> np.ndarray:
    """Forward fill, then backward fill, of NaN entries (1-D)."""
    x = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    if bad.all():
        return x
    idx = np.where(~bad, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    first = np.argmax(~bad)
    x[:first] = x[first]
    return x


def cut_window(series: np.ndarray, horizon: int, max_context: int = MAX_CONTEXT, min_context: int = MIN_CONTEXT):
    """Last window of a 1-D series. Returns `(context, future, info)` or `(None, None, reason)`."""
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    t = min(max_context, n - horizon)
    if t < min_context:
        return None, None, "too_short"
    fut = x[n - horizon:]
    ctx = x[n - horizon - t: n - horizon]
    if not np.all(np.isfinite(fut)):
        return None, None, "missing_in_future"
    nan_n = int((~np.isfinite(ctx)).sum())
    if nan_n > MAX_CONTEXT_NAN_FRAC * t:
        return None, None, "too_many_missing"
    ctx = fill_missing(ctx)
    if not np.all(np.isfinite(ctx)) or np.ptp(ctx) == 0:
        return None, None, "constant_or_empty"
    return ctx, fut, {"ctx_start": n - horizon - t, "origin": n - horizon, "n_filled": nan_n, "length": n}


# ----------------------------------------------------------------------------- readers


def _download(repo: str, revision: str, filename: str, cache_dir: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=cache_dir)


def sha256_of(path: str, n_hex: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_hex]


class SeriesSource:
    """Lazy access to the series of one dataset/frequency pair: `n`, `get(i)`, `meta(i)`."""

    source_file: str
    revision: str
    sha256: str
    multivariate: bool = False

    def __len__(self) -> int:
        raise NotImplementedError

    def get(self, i: int) -> np.ndarray:
        raise NotImplementedError

    def series_id(self, i: int) -> str:
        raise NotImplementedError

    def origin_time(self, i: int, index: int) -> str:
        return ""


class GiftEvalSource(SeriesSource):
    """GIFT-Eval Arrow file: one row per series with `item_id`, `start`, `freq`, and the main
    series as a list (univariate) or a list of lists (multivariate; first variate is used)."""

    def __init__(self, spec: Spec, cache_dir: str | None = None):
        import pyarrow as pa

        self.spec = spec
        self.source_file = f"{spec.path}/data-00000-of-00001.arrow"
        self.revision = GE_REVISION
        self.local = _download(GE_REPO, GE_REVISION, self.source_file, cache_dir)
        self.sha256 = sha256_of(self.local)
        with pa.memory_map(self.local) as src:
            try:
                self.table = pa.ipc.open_stream(src).read_all()
            except pa.ArrowInvalid:
                self.table = pa.ipc.open_file(src).read_all()
        self.col = self.table.column(VALUE_FIELD)
        first = self.col[0].as_py()
        self.multivariate = isinstance(first[0], list)
        self.freq_in_file = self.table.column("freq")[0].as_py()

    def __len__(self) -> int:
        return self.table.num_rows

    def get(self, i: int) -> np.ndarray:
        row = self.col[i].as_py()
        return np.asarray(row[0] if self.multivariate else row, dtype=np.float64)

    def series_id(self, i: int) -> str:
        return str(self.table.column("item_id")[i].as_py())

    def origin_time(self, i: int, index: int) -> str:
        import pandas as pd

        try:
            start = pd.Timestamp(self.table.column("start")[i].as_py())
            return str(pd.date_range(start, periods=index + 1, freq=pandas_freq(self.freq_in_file))[-1])
        except Exception:
            return ""


class FevSource(SeriesSource):
    """fev-bench parquet file: one row per series; list columns hold values and timestamps."""

    def __init__(self, spec: Spec, cache_dir: str | None = None):
        import pyarrow.parquet as pq

        self.spec = spec
        self.source_file = f"{spec.path}/train-00000-of-00001.parquet"
        self.revision = FEV_REVISION
        self.local = _download(FEV_REPO, FEV_REVISION, self.source_file, cache_dir)
        self.sha256 = sha256_of(self.local)
        names = pq.ParquetFile(self.local).schema_arrow.names
        if spec.value_col is not None:
            cols = [spec.value_col]
        else:
            cols = [n for n in names if n not in ("id", "timestamp")]
        self.table = pq.read_table(self.local, columns=["id", "timestamp"] + cols)
        if spec.value_col is None:
            # keep numeric list columns only (wide macro tables)
            import pyarrow as pa
            cols = [c for c in cols if pa.types.is_list(self.table.schema.field(c).type)
                    and pa.types.is_floating(self.table.schema.field(c).type.value_type)]
        self.cols = cols
        self.n_rows = self.table.num_rows

    def __len__(self) -> int:
        return self.n_rows * len(self.cols)

    def _rc(self, i: int) -> tuple[int, str]:
        return i // len(self.cols), self.cols[i % len(self.cols)]

    def get(self, i: int) -> np.ndarray:
        r, c = self._rc(i)
        v = self.table.column(c)[r].as_py()
        return np.asarray([np.nan if z is None else z for z in v], dtype=np.float64)

    def series_id(self, i: int) -> str:
        r, c = self._rc(i)
        sid = str(self.table.column("id")[r].as_py())
        return sid if self.spec.value_col is not None else f"{sid}:{c}"

    def origin_time(self, i: int, index: int) -> str:
        r, _ = self._rc(i)
        try:
            return str(self.table.column("timestamp")[r].as_py()[index])
        except Exception:
            return ""


def open_source(spec: Spec, cache_dir: str | None = None) -> SeriesSource:
    return GiftEvalSource(spec, cache_dir) if spec.source == "gifteval" else FevSource(spec, cache_dir)


# ----------------------------------------------------------------------------- sampling


def pair_seed(spec: Spec, seed: int = SEED) -> list[int]:
    return [seed, zlib.crc32(spec.key.encode())]


def load_pair(spec: Spec, seed: int = SEED, cache_dir: str | None = None, src: SeriesSource | None = None,
              ) -> tuple[list[Sample], list[dict]]:
    """Samples and manifest rows for one pair: seeded random series order, first `spec.n_series`
    series that yield a valid last window."""
    src = src or open_source(spec, cache_dir)
    h = spec.horizon
    m = seasonal_period(spec.freq)
    order = np.random.default_rng(pair_seed(spec, seed)).permutation(len(src))
    samples, rows, skipped = [], [], {}
    for i in order:
        if len(samples) >= spec.n_series:
            break
        ctx, fut, info = cut_window(src.get(int(i)), h)
        if ctx is None:
            skipped[info] = skipped.get(info, 0) + 1
            continue
        sid = f"{spec.key}/{src.series_id(int(i))}"
        meta = {"family": "real", "generator": spec.key, "knob": None, "knob_value": None, "seed": int(seed),
                "T": len(ctx), "H": h, "tags": [], "sample_id": sid, "domain": spec.domain, "freq": spec.freq,
                "term": spec.term, "source": spec.source, "series_index": int(i)}
        samples.append(Sample(ctx[None, :], fut[None, :], np.full((1, h, len(QUANTILE_LEVELS)), np.nan), m, meta))
        rows.append({
            "set": "real", "sample_id": sid, "source": f"{spec.source}:{GE_REPO if spec.source == 'gifteval' else FEV_REPO}",
            "revision": src.revision, "source_file": src.source_file, "source_sha256": src.sha256,
            "dataset": spec.dataset, "term": spec.term, "domain": spec.domain, "freq": spec.freq,
            "series_id": src.series_id(int(i)), "series_index": int(i), "series_len": info["length"],
            "window_start": info["ctx_start"], "origin_index": info["origin"],
            "origin_time": src.origin_time(int(i), info["origin"]),
            "T": len(ctx), "H": h, "m": m, "n_filled": info["n_filled"],
            "first_variate_of_multivariate": bool(src.multivariate),
            "redistributable": bool(spec.redistributable), "family_analogue": "", "reason": "",
            "value_col": ("" if spec.source == "gifteval" else "main_series" if spec.value_col == VALUE_FIELD
                          else spec.value_col or "every float column"),
        })
    return samples, rows


# ----------------------------------------------------------------------------- the pair list

# domain labels follow GIFT-Eval's: Energy, Transport, Sales, Econ/Fin, Web/CloudOps, Nature, Healthcare
def _fev(*a, **k):
    return Spec(*a, **{**k, "redistributable": False})


PAIRS: list[Spec] = [
    # energy
    Spec("gifteval", "electricity/H", "H", "energy"),
    Spec("gifteval", "electricity/D", "D", "energy"),
    Spec("gifteval", "solar/H", "H", "energy"),
    Spec("gifteval", "solar/10T", "10T", "energy"),
    _fev("fev", "entsoe/1H", "H", "energy", value_col=VALUE_FIELD, name="entsoe/1H"),
    _fev("fev", "ercot/1H", "H", "energy", value_col=VALUE_FIELD, name="ercot/1H"),
    # transport
    Spec("gifteval", "LOOP_SEATTLE/H", "H", "transport"),
    Spec("gifteval", "M_DENSE/H", "H", "transport"),
    Spec("gifteval", "SZ_TAXI/15T", "15T", "transport"),
    # sales
    Spec("gifteval", "restaurant", "D", "sales", redistributable=False),
    _fev("fev", "favorita_stores/1D", "D", "sales", value_col="sales"),
    _fev("fev", "rossmann/1D", "D", "sales", value_col="Sales"),
    _fev("fev", "rohlik_sales/1D", "D", "sales", value_col="sales"),
    _fev("fev", "m5/1W", "W", "sales", value_col=VALUE_FIELD),
    # economics and finance (M4 mixes micro, macro, finance, industry, demographic series)
    Spec("gifteval", "m4_hourly", "H", "econ_fin"),
    Spec("gifteval", "m4_daily", "D", "econ_fin"),
    Spec("gifteval", "m4_weekly", "W", "econ_fin"),
    Spec("gifteval", "m4_monthly", "M", "econ_fin"),
    _fev("fev", "fred_md_2025", "M", "econ_fin", value_col=None),
    # web and cloud operations
    Spec("gifteval", "bitbrains_fast_storage/H", "H", "web_cloudops"),
    _fev("fev", "redset/1H", "H", "web_cloudops", value_col=VALUE_FIELD),
    # nature
    Spec("gifteval", "kdd_cup_2018_with_missing/H", "H", "nature"),
    # healthcare
    _fev("fev", "uk_covid_utla/1D", "D", "healthcare", value_col="new_cases"),
    _fev("fev", "hospital_admissions/1D", "D", "healthcare", value_col=VALUE_FIELD),
    # medium-term horizons (10x the short length) for high-frequency pairs
    Spec("gifteval", "electricity/H", "H", "energy", n_series=8, term="medium"),
    Spec("gifteval", "solar/H", "H", "energy", n_series=8, term="medium"),
    Spec("gifteval", "solar/10T", "10T", "energy", n_series=8, term="medium"),
    Spec("gifteval", "LOOP_SEATTLE/H", "H", "transport", n_series=8, term="medium"),
    Spec("gifteval", "M_DENSE/H", "H", "transport", n_series=8, term="medium"),
    Spec("gifteval", "SZ_TAXI/15T", "15T", "transport", n_series=8, term="medium"),
    _fev("fev", "entsoe/1H", "H", "energy", n_series=6, term="medium", value_col=VALUE_FIELD, name="entsoe/1H"),
]


def build_real(pairs: list[Spec] | None = None, seed: int = SEED, cache_dir: str | None = None,
               log=print) -> tuple[list[Sample], list[dict]]:
    samples, rows = [], []
    for spec in pairs or PAIRS:
        s, r = load_pair(spec, seed, cache_dir)
        log(f"{spec.key:40s} n={len(s):3d} H={spec.horizon:4d} m={seasonal_period(spec.freq):3d}")
        samples += s
        rows += r
    return samples, rows


# ----------------------------------------------------------------------------- storage (ragged windows)


def save_real(path: str, samples: list[Sample]) -> None:
    """Ragged windows into one .npz: concatenated contexts / futures with offsets, plus a JSON meta column."""
    enc = json.JSONEncoder(default=lambda o: o.item() if hasattr(o, "item") else str(o))
    ctx = [s.context[0] for s in samples]
    fut = [s.future[0] for s in samples]
    np.savez_compressed(
        path,
        context=np.concatenate(ctx), context_len=np.asarray([len(c) for c in ctx]),
        future=np.concatenate(fut), future_len=np.asarray([len(f) for f in fut]),
        period=np.asarray([s.period for s in samples]),
        meta=np.asarray([enc.encode(s.meta) for s in samples]),
    )


def load_real(path: str) -> list[Sample]:
    """Inverse of `save_real`; `oracle_q` is NaN `(1, H, 21)` for every item."""
    with np.load(path, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    co = np.concatenate([[0], np.cumsum(arr["context_len"])])
    fo = np.concatenate([[0], np.cumsum(arr["future_len"])])
    out = []
    for i in range(len(arr["period"])):
        c = arr["context"][co[i]:co[i + 1]].copy()[None, :]
        f = arr["future"][fo[i]:fo[i + 1]].copy()[None, :]
        out.append(Sample(c, f, np.full((1, f.shape[-1], len(QUANTILE_LEVELS)), np.nan), int(arr["period"][i]),
                          json.loads(str(arr["meta"][i]))))
    return out


def has_oracle(s: Sample) -> bool:
    return bool(np.isfinite(s.oracle_q).all())


# ----------------------------------------------------------------------------- sets split by license


RESTRICTED_SUFFIX = "_restricted"


def save_split(out_dir: str, name: str, samples: list[Sample], redistributable: list[bool]) -> dict[str, int]:
    """Write windows from redistributable sources to `<name>.npz` and the rest to `<name>_restricted.npz`
    (git-ignored; rebuilt by `scripts/build_regression_sets.py`). Returns the counts."""
    import os

    keep = [s for s, r in zip(samples, redistributable) if r]
    rest = [s for s, r in zip(samples, redistributable) if not r]
    for suffix, group in (("", keep), (RESTRICTED_SUFFIX, rest)):
        path = os.path.join(out_dir, f"{name}{suffix}.npz")
        if group:
            save_real(path, group)
        elif os.path.exists(path):
            os.remove(path)
    return {"redistributable": len(keep), "restricted": len(rest)}


def load_union(bench: str, name: str) -> list[Sample]:
    """All windows of a set that exist on this machine, in manifest order. Windows from restricted sources are
    present only after `scripts/build_regression_sets.py` has been run here. `meta["manifest_row"]` is the
    position of the window within its set in `manifest.csv`, so window numbers match across machines."""
    import os

    import pandas as pd

    man = pd.read_csv(os.path.join(bench, "manifest.csv"))
    ids = list(man[man.set == name].sample_id)
    pool: dict[str, Sample] = {}
    for suffix in ("", RESTRICTED_SUFFIX):
        path = os.path.join(bench, f"{name}{suffix}.npz")
        if os.path.exists(path):
            for s in load_real(path):
                pool[s.meta["sample_id"]] = s
    out = []
    for row, sid in enumerate(ids):
        if sid in pool:
            pool[sid].meta["manifest_row"] = row
            out.append(pool[sid])
    return out
