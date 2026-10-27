"""Search objective: how badly Timestral fails on a mixture-of-primitives series, relative to the oracle.

fitness(genome, size) = geometric mean over `n_seeds` fixed seeds of
    MASE(Timestral) / MASE(true-dynamics oracle)
at context 2048 and horizon 128. sCRPS ratio and cov80 are recorded too.

Feasibility ("the series stays forecastable by the oracle"), all of these must hold:
  1. every context, future and oracle quantile is finite, and Timestral's quantiles are finite;
  2. the geometric mean over seeds of  sCRPS(oracle) / sCRPS(seasonal naive)  is at most 0.8
     (seasonal naive uses the generator's own period; scored on the same samples);
  3. the context std of every seed lies in [1e-3, 1e6] (the generator standardizes its own
     components, so a std of about 1 is expected; this rejects collapsed or exploded series);
  4. the oracle's MASE is at least 1e-6 on every seed (a ratio against a numerically zero error
     says nothing).

Infeasible genomes: `fitness` is 0. For the search algorithms there is also `search_score`:
the fitness when feasible, otherwise  -violation  where violation = max(0, orc_sn - 0.8)
(10 when something is not finite). Every feasible genome therefore ranks above every infeasible
one, and among infeasible genomes the ones closer to feasibility rank higher, which pushes a
rank-based optimizer back toward the feasible region.

Counting: one sample (one genome at one seed) is one Timestral forecast call; symmetric
averaging inside the forecaster counts as part of that one call. A genome evaluation with
3 seeds is 3 calls, whether or not the genome turns out feasible. Cached genomes (same
hash within a run) cost nothing.

Also here: behaviour descriptors (dominant component family by weight, spectral entropy of
the context) used by MAP-Elites and by the report, and the `Evaluator` that wraps an
`Objective` for one search run (budget, history file, resume by replay).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field

import numpy as np

from otp import metrics
from otp.classical import seasonal_naive
from otp.generators import mixture
from otp.generators.base import make_sample

T_CTX = 2048
HORIZON = 128
N_PATHS = 500
SEARCH_SEEDS = (11001, 11002, 11003)   # fixed seeds used while searching
FRESH_SEEDS = (21001, 21002, 21003)    # fresh seeds used for transfer / benchmark samples
ORC_SN_MAX = 0.8
STD_LO, STD_HI = 1e-3, 1e6
ORC_MASE_MIN = 1e-6
FAMILIES = ("primitive", "chaos", "regime", "tails")
N_ENTROPY_BINS = 10
GRID_SHAPE = (len(FAMILIES), N_ENTROPY_BINS)


def _enc_list(xs) -> str:
    """JSON text of a list of floats (repr round-trips float64 exactly)."""
    return "[" + ", ".join(repr(float(x)) for x in xs) + "]"


def genome_hash(genome) -> str:
    g = np.clip(np.asarray(genome, dtype="<f8"), 0.0, 1.0)
    return hashlib.sha1(g.tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- descriptors
def dominant_family(genome) -> str:
    """Component family with the largest summed slot weight (ties go to the earlier family)."""
    w = {f: 0.0 for f in FAMILIES}
    for s in mixture.describe(genome)["slots"]:
        w[s["family"]] += float(s["weight"])
    return max(FAMILIES, key=lambda f: w[f])


def component_signature(genome, min_weight: float = 0.05) -> str:
    """Sorted names of the components whose weight exceeds `min_weight`, joined by '+'."""
    names = sorted({s["component"] for s in mixture.describe(genome)["slots"] if s["weight"] > min_weight})
    return "+".join(names) if names else "none"


def spectral_entropy(x: np.ndarray) -> float:
    """Normalized entropy in [0, 1] of the Hann-windowed periodogram (0 = one frequency, 1 = flat)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    if not np.isfinite(x).all() or x.std() < 1e-12:
        return 1.0
    p = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    p = p[1:]
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return 1.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p) if len(p) > 1 else 2))


def cell_of(family: str, entropy: float) -> tuple[int, int]:
    return FAMILIES.index(family), min(int(max(entropy, 0.0) * N_ENTROPY_BINS), N_ENTROPY_BINS - 1)


# ---------------------------------------------------------------- one sample
def _gen_record(job):
    """Worker: build one sample and everything about it that does not need Timestral."""
    genome, seed, T, H, n_paths = job
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            proc = mixture.decode(genome)
            s = make_sample(proc, T, H, seed=seed, n_paths=n_paths, sample_id=f"{genome_hash(genome)}/s{seed}")
            ctx_ok = bool(np.isfinite(s.context).all() and np.isfinite(s.future).all())
            orc_ok = bool(np.isfinite(s.oracle_q).all())
            rec = {"error": None, "seed": seed, "ctx_ok": ctx_ok, "orc_ok": orc_ok, "period": int(s.period),
                   "ctx_std": float(np.std(s.context)) if ctx_ok else float("nan"),
                   "spec_entropy": spectral_entropy(s.context[0]) if ctx_ok else 1.0}
            if ctx_ok and orc_ok:
                so = metrics.score(s.context, s.future, s.oracle_q, s.period)
                sn_q = seasonal_naive(s.context[0], H, s.period)[None]
                sn = metrics.score(s.context, s.future, sn_q, s.period)
                rec.update(mase_orc=float(so["mase"][0]), scrps_orc=float(so["scrps"][0]),
                           cov80_orc=float(so["cov80"][0]), mase_sn=float(sn["mase"][0]),
                           scrps_sn=float(sn["scrps"][0]))
            rec["sample"] = s
            return rec
    except Exception as e:  # a simulator that fails is an infeasible genome
        return {"error": f"{type(e).__name__}: {e}", "seed": seed, "ctx_ok": False, "orc_ok": False,
                "spec_entropy": 1.0, "ctx_std": float("nan"), "period": 1}


# ---------------------------------------------------------------- result
@dataclass
class EvalResult:
    hash: str
    size: str
    fitness: float = 0.0           # geometric mean MASE ratio if feasible, else 0
    feasible: bool = False
    search_score: float = -10.0
    ratio_mase: float = float("nan")      # geometric mean MASE(Timestral)/MASE(oracle), also when infeasible
    ratio_scrps: float = float("nan")
    cov80: float = float("nan")           # mean over seeds of Timestral's 80% interval coverage
    orc_sn: float = float("nan")          # geometric mean sCRPS(oracle)/sCRPS(seasonal naive)
    ctx_std_min: float = float("nan")
    ctx_std_max: float = float("nan")
    spec_entropy: float = 1.0
    family: str = "primitive"
    signature: str = "none"
    reason: str = ""
    seed_ratios: list = field(default_factory=list)   # per-seed MASE ratios
    samples: list | None = None

    def row(self) -> dict:
        d = {k: getattr(self, k) for k in ("hash", "size", "fitness", "feasible", "search_score", "ratio_mase",
                                           "ratio_scrps", "cov80", "orc_sn", "ctx_std_min", "ctx_std_max",
                                           "spec_entropy", "family", "signature", "reason")}
        d["seed_ratios"] = _enc_list(self.seed_ratios)
        return d

    @staticmethod
    def from_row(r: dict) -> "EvalResult":
        def f(k):
            v = r.get(k)
            return float("nan") if v is None or v == "" else float(v)
        sr = r.get("seed_ratios")
        return EvalResult(hash=str(r["hash"]), size=str(r["size"]), fitness=f("fitness"),
                          feasible=str(r["feasible"]) in ("True", "true", "1"), search_score=f("search_score"),
                          ratio_mase=f("ratio_mase"), ratio_scrps=f("ratio_scrps"), cov80=f("cov80"),
                          orc_sn=f("orc_sn"), ctx_std_min=f("ctx_std_min"), ctx_std_max=f("ctx_std_max"),
                          spec_entropy=f("spec_entropy"), family=str(r["family"]), signature=str(r["signature"]),
                          reason="" if r.get("reason") in (None, "nan") or r.get("reason") != r.get("reason")
                          else str(r["reason"]),
                          seed_ratios=json.loads(sr) if isinstance(sr, str) and sr else [])


def _gm(x) -> float:
    x = np.asarray(x, float)
    if x.size == 0 or not np.isfinite(x).all() or (x <= 0).any():
        return float("nan")
    return float(np.exp(np.log(x).mean()))


def assess(genome, size: str, recs: list[dict], ts_scores: list[dict | None]) -> EvalResult:
    """Combine per-seed records and Timestral scores into fitness / feasibility (pure function)."""
    h = genome_hash(genome)
    res = EvalResult(hash=h, size=size, family=dominant_family(genome), signature=component_signature(genome))
    ent = [r["spec_entropy"] for r in recs]
    res.spec_entropy = float(np.mean(ent)) if ent else 1.0
    stds = [r["ctx_std"] for r in recs]
    if np.isfinite(stds).all() and stds:
        res.ctx_std_min, res.ctx_std_max = float(min(stds)), float(max(stds))
    errs = [r["error"] for r in recs if r.get("error")]
    if errs:
        res.reason = "generator error: " + errs[0][:80]
        res.search_score = -10.0
        return res
    if not all(r["ctx_ok"] for r in recs):
        res.reason = "non-finite context or future"
        return res
    if not all(r["orc_ok"] for r in recs):
        res.reason = "non-finite oracle quantiles"
        return res
    if any(t is None or not t["finite"] for t in ts_scores):
        res.reason = "non-finite Timestral quantiles"
        return res
    mo = np.array([r["mase_orc"] for r in recs])
    mt = np.array([t["mase"] for t in ts_scores])
    so = np.array([r["scrps_orc"] for r in recs])
    st = np.array([t["scrps"] for t in ts_scores])
    ssn = np.array([r["scrps_sn"] for r in recs])
    res.ratio_mase = _gm(mt / np.maximum(mo, 1e-9))
    res.ratio_scrps = _gm(st / np.maximum(so, 1e-9))
    res.cov80 = float(np.mean([t["cov80"] for t in ts_scores]))
    res.seed_ratios = list((mt / np.maximum(mo, 1e-9)).tolist())
    res.orc_sn = _gm(so / np.maximum(ssn, 1e-12))
    why = []
    if not np.isfinite(res.orc_sn):
        why.append("orc_sn undefined")
    elif res.orc_sn > ORC_SN_MAX:
        why.append(f"oracle sCRPS above {ORC_SN_MAX} x seasonal naive")
    if not (min(stds) >= STD_LO and max(stds) <= STD_HI):
        why.append("context std outside [1e-3, 1e6]")
    if (mo < ORC_MASE_MIN).any():
        why.append("oracle MASE below 1e-6")
    if not np.isfinite(res.ratio_mase):
        why.append("MASE ratio undefined")
    if why:
        res.reason = "; ".join(why)
        viol = max(0.0, res.orc_sn - ORC_SN_MAX) if np.isfinite(res.orc_sn) else 10.0
        res.search_score = -max(viol, 1e-3) if np.isfinite(res.ratio_mase) else -10.0
        return res
    res.feasible = True
    res.fitness = res.ratio_mase
    res.search_score = res.fitness
    return res


# ---------------------------------------------------------------- objective
class Objective:
    """Scores genomes on one Timestral size. Generation of samples can use a process pool."""

    def __init__(self, size: str = "5M", seeds=SEARCH_SEEDS, T: int = T_CTX, H: int = HORIZON,
                 n_paths: int = N_PATHS, workers: int = 1, batch_size: int = 32):
        self.size, self.seeds, self.T, self.H, self.n_paths = size, tuple(seeds), T, H, n_paths
        self.workers, self.batch_size = workers, batch_size
        self.cache: dict[tuple, EvalResult] = {}
        self.rec_cache: dict[str, list[dict]] = {}
        self.calls = 0              # Timestral forward calls made by this object (cache hits excluded)
        self.sample_generations = 0
        self._pool = None

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _map(self, jobs):
        if self.workers <= 1:
            return [_gen_record(j) for j in jobs]
        if self._pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"))
        return list(self._pool.map(_gen_record, jobs, chunksize=1))

    def records(self, genomes, keep_samples: bool = False) -> list[list[dict]]:
        """Per-genome list of per-seed records (generation only, no Timestral), cached by genome hash."""
        todo, order = [], []
        for g in genomes:
            h = genome_hash(g)
            need = h not in self.rec_cache or (keep_samples and any("sample" not in r for r in self.rec_cache[h]))
            if need and h not in order:
                order.append(h)
                todo.append(np.clip(np.asarray(g, float), 0, 1))
        jobs = [(g, s, self.T, self.H, self.n_paths) for g in todo for s in self.seeds]
        out = self._map(jobs)
        self.sample_generations += len(jobs)
        for i, h in enumerate(order):
            recs = out[i * len(self.seeds):(i + 1) * len(self.seeds)]
            self.rec_cache[h] = recs
        return [self.rec_cache[genome_hash(g)] for g in genomes]

    def _timestral(self, size: str, recs_list: list[list[dict]]) -> list[list[dict | None]]:
        from otp.model import forecast
        idx, ctxs = [], []
        for gi, recs in enumerate(recs_list):
            if all(r["ctx_ok"] and r["orc_ok"] for r in recs) and not any(r.get("error") for r in recs):
                for si, r in enumerate(recs):
                    idx.append((gi, si))
                    ctxs.append(r["sample"].context)
        res: list[list[dict | None]] = [[None] * len(r) for r in recs_list]
        if not ctxs:
            return res
        q = forecast(size, np.stack(ctxs), self.H, batch_size=self.batch_size)   # (N, 1, H, 21)
        self.calls += len(ctxs)
        for k, (gi, si) in enumerate(idx):
            s = recs_list[gi][si]["sample"]
            sc = metrics.score(s.context, s.future, q[k], s.period)
            res[gi][si] = {"mase": float(sc["mase"][0]), "scrps": float(sc["scrps"][0]),
                           "cov80": float(sc["cov80"][0]), "finite": bool(np.isfinite(q[k]).all())}
        return res

    def evaluate(self, genomes, size: str | None = None, keep_samples: bool = False) -> list[EvalResult]:
        size = size or self.size
        genomes = [np.clip(np.asarray(g, float), 0, 1) for g in genomes]
        fresh = [g for g in genomes if (genome_hash(g), size) not in self.cache
                 or (keep_samples and self.cache[(genome_hash(g), size)].samples is None)]
        uniq, seen = [], set()
        for g in fresh:
            h = genome_hash(g)
            if h not in seen:
                seen.add(h)
                uniq.append(g)
        if uniq:
            recs_list = self.records(uniq, keep_samples=True)
            ts = self._timestral(size, recs_list)
            for g, recs, t in zip(uniq, recs_list, ts):
                r = assess(g, size, recs, t)
                if keep_samples:
                    r.samples = [x["sample"] for x in recs if "sample" in x]
                self.cache[(r.hash, size)] = r
            if not keep_samples:   # samples are large; keep only what the numbers need
                for g in uniq:
                    for rec in self.rec_cache[genome_hash(g)]:
                        rec.pop("sample", None)
        return [self.cache[(genome_hash(g), size)] for g in genomes]


# ---------------------------------------------------------------- one search run
HISTORY_COLS = ["eval", "calls", "method", "run_seed", "origin", "best_so_far", "genome", *EvalResult(
    hash="", size="").row().keys()]


class Evaluator:
    """Budgeted view of an `Objective` for one search run.

    * `budget` counts Timestral calls: `len(seeds)` per fresh genome, plus whatever `charge()` adds
      (gradient steps charge 2 per series: one forward and one backward).
    * Re-evaluating a genome already seen in this run is free and writes no history row.
    * `evaluate` truncates a batch when the budget would be exceeded and returns results for the
      processed prefix only.
    * History rows go to `<run_dir>/scores_partial.csv` as they are produced. With `resume`, rows of
      that file are replayed (the search algorithms are deterministic given their seed), so a run
      cut off by a lost runtime continues where it stopped. `finish()` writes `history.csv`.
    """

    def __init__(self, objective: Objective, budget: int, run_dir: str, method: str, run_seed: int,
                 resume: bool = False):
        self.obj, self.budget, self.run_dir, self.method, self.run_seed = objective, budget, run_dir, method, run_seed
        self.n_seeds = len(objective.seeds)
        self.calls = 0
        self.rows: list[dict] = []
        self.seen: dict[str, EvalResult] = {}
        self.best = 0.0
        self.best_genome: np.ndarray | None = None
        self.genomes: dict[str, np.ndarray] = {}
        self.replay: dict[str, dict] = {}
        os.makedirs(run_dir, exist_ok=True)
        self.partial = os.path.join(run_dir, "scores_partial.csv")
        self.final = os.path.join(run_dir, "history.csv")
        self.n_replayed = 0
        if resume and os.path.exists(self.partial):
            with open(self.partial, newline="") as f:
                for r in csv.DictReader(f):
                    self.replay[r["hash"]] = r
        if not resume and os.path.exists(self.partial):
            os.remove(self.partial)

    @property
    def remaining(self) -> int:
        return self.budget - self.calls

    @property
    def done(self) -> bool:
        return self.remaining < self.n_seeds

    def charge(self, n: int):
        """Charge Timestral calls made outside `evaluate` (gradient forward + backward passes)."""
        self.calls += int(n)

    def _append(self, row: dict):
        new = not os.path.exists(self.partial)
        with open(self.partial, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in HISTORY_COLS})

    def evaluate(self, genomes, origin: str = "") -> list[EvalResult]:
        genomes = [np.clip(np.asarray(g, float), 0, 1) for g in genomes]
        take, fresh, seen_now = [], [], set()
        afford = self.remaining // self.n_seeds
        for g in genomes:
            h = genome_hash(g)
            if h in self.seen or h in seen_now:
                take.append(g)
                continue
            if len(fresh) >= afford:
                break
            seen_now.add(h)
            fresh.append(g)
            take.append(g)
        to_compute = [g for g in fresh if genome_hash(g) not in self.replay]
        if to_compute:
            self.obj.evaluate(to_compute)
        for g in fresh:
            h = genome_hash(g)
            if h in self.replay:
                res = EvalResult.from_row(self.replay[h])
                self.n_replayed += 1
                is_new_row = False
            else:
                res = self.obj.evaluate([g])[0]
                is_new_row = True
            self.calls += self.n_seeds
            self.seen[h] = res
            self.genomes[h] = g
            if res.feasible and res.fitness > self.best:
                self.best, self.best_genome = res.fitness, g
            row = {"eval": len(self.rows) + 1, "calls": self.calls, "method": self.method,
                   "run_seed": self.run_seed, "origin": origin, "best_so_far": self.best,
                   "genome": _enc_list(g), **res.row()}
            self.rows.append(row)
            if is_new_row:
                self._append(row)
        return [self.seen[genome_hash(g)] for g in take]

    def finish(self):
        with open(self.final, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in HISTORY_COLS})
        if os.path.exists(self.partial):
            os.remove(self.partial)
