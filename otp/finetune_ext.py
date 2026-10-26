"""Extensions to `otp.finetune` (the lead's library is not edited; this module adds what the study needs).

Why this module exists:

- `otp.finetune.Sampler` simulates every training pair on the fly and does not check that the
  simulated series are finite. Chaotic and mixture processes can diverge, and one non-finite
  series makes the whole batch loss NaN. `build_pool` draws the same kind of pairs, skips
  non-finite or constant ones, and stores them, so that every method in a comparison trains on the
  identical sequence of batches (a paired comparison) and the simulation cost is paid once.
- `CompositeSampler` serves those stored batches through the same `.batch(batch_size)` interface that
  `otp.finetune.train` expects, applying `replay_frac` (the share of each batch taken from in-prior
  replay series instead of failure-family series). Pools are per family, so the runs of one
  protocol that share a family also share its simulated data.
- cost helpers: peak resident memory of the process, parameter counts.
- report helpers: summary tables, README text and model-card drafts built from result files.
"""

from __future__ import annotations

import os
import resource
import sys
import zlib

import numpy as np

from otp.generators.base import make_sample
from otp.generators.primitives import in_prior


# ---------------------------------------------------------------------------
# Training pools
# ---------------------------------------------------------------------------
def _draw(factory, seed: int, t: int, h: int):
    s = make_sample(factory(seed), t, h, seed=seed, n_paths=1)
    x, y = s.context[0], s.future[0]
    ok = np.isfinite(x).all() and np.isfinite(y).all() and float(np.std(x)) > 1e-9
    return (x.astype(np.float32), y.astype(np.float32)) if ok else None


def schedule(n_batches: int, seed: int = 0, contexts=(256, 512, 1024, 2048), horizons=(32, 128, 512)) -> list[tuple[int, int]]:
    """(context length, horizon) of each training batch. All pools of a run share one schedule, so
    rows drawn from different pools can be stacked into one batch."""
    rng = np.random.default_rng(seed)
    return [(int(rng.choice(contexts)), int(rng.choice(horizons))) for _ in range(n_batches)]


_WORK = {}  # set before forking worker processes: the factory and the draw settings


def _one_batch(b: int):
    factory, sched, rows, seed, seed_range = (_WORK[k] for k in ("factory", "sched", "rows", "seed", "seed_range"))
    t, h = sched[b]
    rng = np.random.default_rng([seed, b])  # one stream per batch: the result does not depend on the worker count
    xs, ys = [], []
    while len(xs) < rows:
        got = _draw(factory, int(rng.integers(*seed_range)), t, h)
        if got is not None:
            xs.append(got[0])
            ys.append(got[1])
    return np.stack(xs), np.stack(ys)


def build_pool(factory, sched: list[tuple[int, int]], rows: int, seed: int = 0,
               seed_range=(1_000_000, 2_000_000), workers: int = 1, log=print) -> dict:
    """Pre-draw `rows` (context, future) pairs from `factory` for every batch in `sched`.
    Non-finite or constant series are skipped and redrawn. Each batch has its own random stream, so
    the pool is identical for any `workers` (worker processes are forked, so factories may be lambdas)."""
    _WORK.update(factory=factory, sched=sched, rows=rows, seed=seed, seed_range=seed_range)
    n = len(sched)
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers) as pool_:
            out = pool_.map(_one_batch, range(n), chunksize=max(1, n // (workers * 4)))
    else:
        out = []
        for b in range(n):
            out.append(_one_batch(b))
            if log and (b + 1) % 10 == 0:
                log(f"pool {b + 1}/{n}")
    return {"t": [t for t, _ in sched], "h": [h for _, h in sched], "x": [o[0] for o in out], "y": [o[1] for o in out]}


def save_pool(path: str, pool: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {"t": np.asarray(pool["t"]), "h": np.asarray(pool["h"])}
    for i in range(len(pool["t"])):
        arrays[f"x_{i}"] = pool["x"][i]
        arrays[f"y_{i}"] = pool["y"][i]
    np.savez_compressed(path, **arrays)


def load_pool(path: str) -> dict:
    with np.load(path) as z:
        arr = {k: z[k] for k in z.files}
    n = len(arr["t"])
    return {"t": [int(v) for v in arr["t"]], "h": [int(v) for v in arr["h"]],
            "x": [arr[f"x_{i}"] for i in range(n)], "y": [arr[f"y_{i}"] for i in range(n)]}


def pool_file(pool_dir: str, name: str, n_batches: int, rows: int, seed: int, contexts, horizons) -> str:
    c = "-".join(map(str, contexts))
    h = "-".join(map(str, horizons))
    return os.path.join(pool_dir, f"pool__{name}__n{n_batches}_r{rows}_s{seed}_c{c}_h{h}.npz")


def get_pool(pool_dir: str, name: str, factory, sched: list, rows: int, seed: int, contexts, horizons,
             workers: int = 1, log=print) -> tuple[dict, str, str]:
    """Load a cached pool or build and cache it. Returns (pool, 'cached'|'built', file)."""
    path = pool_file(pool_dir, name, len(sched), rows, seed, contexts, horizons)
    if os.path.exists(path):
        return load_pool(path), "cached", path
    name_seed = seed * 1000 + (zlib.crc32(name.encode()) % 1000)
    pool = build_pool(factory, sched, rows, seed=name_seed, workers=workers, log=log)
    save_pool(path, pool)
    return pool, "built", path


class CompositeSampler:
    """Serves stored batches through the `.batch(batch_size)` interface of `otp.finetune.train`.

    `family_pools`: one pool per failure family (same schedule). With `k = round(replay_frac * B)`
    replay rows (taken from `replay_pool`), the remaining `B - k` rows go round-robin over the
    family pools, so each family contributes (nearly) equally. Batches are served in order and the
    pool is cycled when it is shorter than the run.
    """

    def __init__(self, family_pools: list[dict], replay_pool: dict, replay_frac: float = 0.0):
        self.fam = family_pools
        self.rep = replay_pool
        self.replay_frac = replay_frac
        self.i = 0

    def batch(self, batch_size: int):
        n = len(self.fam[0]["t"])
        j = self.i % n
        self.i += 1
        k = int(round(self.replay_frac * batch_size))
        xs, ys = [], []
        if k:
            assert self.rep["x"][j].shape[0] >= k, "replay pool batch is smaller than the replay share"
            xs.append(self.rep["x"][j][:k])
            ys.append(self.rep["y"][j][:k])
        nf = len(self.fam)
        for r in range(batch_size - k):
            pool = self.fam[r % nf]
            xs.append(pool["x"][j][r // nf:r // nf + 1])
            ys.append(pool["y"][j][r // nf:r // nf + 1])
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------
def peak_rss_bytes() -> int:
    """Peak resident memory of this process (Linux reports kilobytes)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def count_params(model) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr, tot


def run_id(size: str, protocol: str, holdout: str, method: str, replay_frac: float) -> str:
    mix = f"mix{replay_frac:g}" if replay_frac > 0 else "nomix"
    return f"{size}__{protocol}-{holdout}__{method}__{mix}"


def parse_run_id(rid: str) -> dict:
    size, ph, method, mix = rid.split("__")
    protocol, holdout = ph.split("-", 1)
    return {"run_id": rid, "size": size, "protocol": protocol, "holdout": holdout, "method": method,
            "replay_frac": 0.0 if mix == "nomix" else float(mix[3:])}


def method_label(method: str, replay_frac: float) -> str:
    return method + (f"+mix{replay_frac:g}" if replay_frac > 0 else "")


def log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Result tables
# ---------------------------------------------------------------------------
COST_COLS = ["run_id", "size", "protocol", "holdout", "method", "replay_frac", "trainable_params", "base_params",
             "params_with_adapters", "rank", "steps", "batch_size", "lr", "time_steps_seen", "wall_sec",
             "pool_sec", "final_loss", "peak_rss_bytes", "peak_cuda_bytes", "device", "gpu"]


def cost_row(run: dict) -> dict:
    """One `costs.csv` row from a `run.json` dict."""
    t = run["train"]
    r = {k: run.get(k) for k in COST_COLS if k in run}
    r.update({"steps": t["steps"], "batch_size": t["batch_size"], "lr": t["lr"],
              "time_steps_seen": t["time_steps_seen"], "wall_sec": t["wall_sec"], "final_loss": t["final_loss"],
              "peak_cuda_bytes": t.get("peak_cuda_bytes"), "pool_sec": run["pool"]["sec"]})
    return {k: r.get(k) for k in COST_COLS}


def _gm(x) -> float:
    from otp.analysis import _gm as gm
    return gm(x)


def summarize(scores, runs: list[dict], own_sets: set) -> "pd.DataFrame":  # noqa: F821
    """One row per run x evaluated set: ratio to oracle before / after, paired change, coverage."""
    import pandas as pd

    from otp.analysis import _boot_gm, ratios

    keys = ["sample_id", "T", "H", "variate"]
    rows = []
    for set_name, g in scores.groupby("eval_set"):
        has_oracle = (g.forecaster == "oracle").any()
        gr = ratios(g.reset_index(drop=True)) if has_oracle else g.assign(
            mase_ratio_oracle=float("nan"), scrps_ratio_oracle=float("nan"))
        orc = gr[gr.forecaster == "oracle"]
        for r in runs:
            if set_name not in (f"{r['protocol']}/{r['holdout']}", "regression/in_prior", "regression/real"):
                continue
            ft = gr[(gr.forecaster == "finetuned") & (gr.run_id == r["run_id"])]
            if ft.empty:
                continue
            pre = gr[(gr.forecaster == "pretrained") & (gr["size"] == r["size"])]
            pair = ft.merge(pre, on=keys, suffixes=("_ft", "_pre"))
            row = {k: r[k] for k in ("run_id", "size", "protocol", "holdout", "method", "replay_frac")}
            row.update({"eval_set": set_name, "role": ft.role.iloc[0], "n": int(ft.sample_id.nunique())})
            for m in ("mase", "scrps"):
                a, b = _gm(pre[f"{m}_ratio_oracle"]), _gm(ft[f"{m}_ratio_oracle"])
                row[f"pre_{m}_ratio"], row[f"ft_{m}_ratio"] = a, b
                row[f"delta_{m}_ratio"] = b - a
                row[f"gap_closed_{m}"] = (a - b) / (a - 1.0) if a > 1.0 else float("nan")
            row["pre_cov80"], row["ft_cov80"] = float(pre.cov80.mean()), float(ft.cov80.mean())
            row["delta_cov80"] = row["ft_cov80"] - row["pre_cov80"]
            per = pair.mase_ft / pair.mase_pre.clip(lower=1e-12)
            row["paired_mase_ft_over_pre"] = _gm(per)
            row["paired_mase_lo95"], row["paired_mase_hi95"] = _boot_gm(per)
            row["paired_scrps_ft_over_pre"] = _gm(pair.scrps_ft / pair.scrps_pre.clip(lower=1e-12))
            tr = gr[(gr.forecaster == "pretrained_truncate") & (gr["size"] == r["size"])]
            row["trunc_mase_ratio"] = _gm(tr.mase_ratio_oracle) if len(tr) else float("nan")
            row["trunc_scrps_ratio"] = _gm(tr.scrps_ratio_oracle) if len(tr) else float("nan")
            row["trunc_cov80"] = float(tr.cov80.mean()) if len(tr) else float("nan")
            row["oracle_cov80"] = float(orc.cov80.mean()) if len(orc) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def method_label(method: str, replay_frac: float) -> str:  # noqa: F811
    return method + (f"+mix{replay_frac:g}" if replay_frac > 0 else "")


def _f(x, nd: int = 3) -> str:
    return "" if x is None or x != x else f"{x:.{nd}f}"


def _md(headers: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(r) + " |" for r in rows]


def write_readme(out: str, summary, costs, scores) -> None:
    """`<out>/README.md`: every number is read from summary.csv, costs.csv or provenance.json."""
    from otp.io import read_json

    prov = read_json(os.path.join(out, "provenance.json"))
    name = os.path.basename(os.path.normpath(out))
    L = [f"# results/finetune/{name}", ""]
    L += ["Fine-tuning Timestral on failure families: full fine-tune, LoRA (low-rank adapters on the attention and MLP "
          "linears; base weights frozen), and data mixing (a share of every batch is replay of in-prior series: random "
          "trend + season + AR-noise, the kind of data Timestral was pretrained on). Files:", "",
          "- `scores.csv`: one row per eval set x run x forecaster x sample (`forecaster` is `oracle`, `pretrained`, "
          "`pretrained_truncate` or `finetuned`).",
          "- `summary.csv`: one row per run x evaluated set. `pre_*` is the pretrained model of the same size, `ft_*` the "
          "fine-tuned one. `*_ratio` is the geometric mean over samples of the metric divided by the oracle's metric on "
          "the same sample (1 = as good as the oracle; MASE = mean absolute scaled error, sCRPS = scaled continuous "
          "ranked probability score). `delta_*` = ft minus pre. `gap_closed_*` = (pre - ft) / (pre - 1). "
          "`paired_mase_ft_over_pre` = geometric mean over samples of fine-tuned MASE / pretrained MASE. `cov80` = share "
          "of future points inside the 10%-90% quantile interval (0.8 is calibrated). `trunc_*` = the pretrained model "
          "with the context cut to the last max(d, 64) steps (regime sets only).",
          "- `costs.csv`: per run: trainable parameters, steps, batch size, learning rate, time steps seen "
          "(context + future values over all training series), training wall time, final loss, peak memory.",
          "- `provenance.json`: command line, commit, host, wall time, splits.",
          "- `runs/<run_id>/run.json`, `runs/<run_id>/loss.csv`: per-run configuration, cost record and loss curve.", ""]
    ev = prov.get("sets", {})
    L += [f"Evaluation: {prov['n_eval']} held-out samples per set (500 oracle paths each, context {prov['splits']['eval']['T']}, "
          f"horizon {prov['splits']['eval']['H']}); regression set: {ev.get('regression/in_prior')} fresh in-prior series "
          f"(seeds {prov['splits']['seeds']['regression_start']} and up). Real-data regression: "
          + ("`" + str(prov.get("real_npz")) + "` (" + str(ev.get("regression/real")) + " windows)." if prov.get("real_npz")
             else "pending (`bench/regression/real.npz` was not on master when this ran; rerun with `--real-npz`).") + "",
          f"Evaluation wall time {prov['wall_sec']} s; training wall time summed over runs "
          f"{prov['train_wall_sec_total']} s (both from `provenance.json`).", ""]

    tests = summary[summary.role == "test"].copy()
    tests["label"] = [method_label(m, f) for m, f in zip(tests.method, tests.replay_frac)]
    order = list(dict.fromkeys(tests.label))
    for protocol in ("lofo", "knob"):
        t = tests[tests.protocol == protocol]
        if t.empty:
            continue
        title = {"lofo": "Leave-one-family-out (train on three families, test on the fourth)",
                 "knob": "Held-out knob ranges (train on one range of each family's knob, test on the other)"}[protocol]
        L += [f"## {title}", ""]
        for size in sorted(t["size"].unique(), key=lambda s: float(s[:-1])):
            ts = t[t["size"] == size]
            L += [f"### Size {size}: ratio to oracle, before and after", "",
                  "Cells: MASE ratio pre -> ft, sCRPS ratio pre -> ft, cov80 pre -> ft.", ""]
            hdr = ["held-out family", "n"] + order
            rows = []
            for fam in S_FAMILIES:
                g = ts[ts.holdout == fam]
                if g.empty:
                    continue
                row = [fam, str(int(g.n.iloc[0]))]
                for lab in order:
                    r = g[g.label == lab]
                    if r.empty:
                        row.append("")
                        continue
                    r = r.iloc[0]
                    row.append(f"M {_f(r.pre_mase_ratio)} -> {_f(r.ft_mase_ratio)}; S {_f(r.pre_scrps_ratio)} -> "
                               f"{_f(r.ft_scrps_ratio)}; C {_f(r.pre_cov80)} -> {_f(r.ft_cov80)}")
                rows.append(row)
            L += _md(hdr, rows) + [""]
            hdr2 = ["held-out family", "method", "paired MASE ft/pre", "95% interval", "gap closed (MASE)",
                    "gap closed (sCRPS)", "trainable params", "steps", "time steps seen", "train wall s"]
            rows2 = []
            cs = costs.set_index("run_id")
            for fam in S_FAMILIES:
                for lab in order:
                    r = ts[(ts.holdout == fam) & (ts.label == lab)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    c = cs.loc[r.run_id]
                    rows2.append([fam, lab, _f(r.paired_mase_ft_over_pre), f"{_f(r.paired_mase_lo95)} to {_f(r.paired_mase_hi95)}",
                                  _f(r.gap_closed_mase), _f(r.gap_closed_scrps), f"{int(c.trainable_params):,}",
                                  str(int(c.steps)), f"{int(c.time_steps_seen):,}", _f(c.wall_sec, 1)])
            L += [f"### Size {size}: gain versus cost", ""] + _md(hdr2, rows2) + [""]

    reg = summary[summary.role == "regression"].copy()
    if not reg.empty:
        reg["label"] = [method_label(m, f) for m, f in zip(reg.method, reg.replay_frac)]
        L += ["## Regression check on normal series", "",
              "Fine-tuned minus pretrained of the same size, on normal series that are outside the failure families. "
              "`paired MASE ft/pre` is the geometric mean over series of fine-tuned MASE / pretrained MASE "
              "(above 1: the fine-tuned model is worse). Run id column shows which checkpoint.", ""]
        for setname in sorted(reg.eval_set.unique()):
            g = reg[reg.eval_set == setname]
            hdr = ["run", "n", "pre MASE ratio", "ft MASE ratio", "pre sCRPS ratio", "ft sCRPS ratio", "paired MASE ft/pre",
                   "95% interval", "paired sCRPS ft/pre", "pre cov80", "ft cov80"]
            rows = [[r.run_id, str(int(r.n)), _f(r.pre_mase_ratio), _f(r.ft_mase_ratio), _f(r.pre_scrps_ratio),
                     _f(r.ft_scrps_ratio), _f(r.paired_mase_ft_over_pre), f"{_f(r.paired_mase_lo95)} to {_f(r.paired_mase_hi95)}",
                     _f(r.paired_scrps_ft_over_pre), _f(r.pre_cov80), _f(r.ft_cov80)] for r in g.itertuples()]
            L += [f"### {setname}", ""] + _md(hdr, rows) + [""]

    tr = tests[tests.trunc_mase_ratio.notna()]
    if not tr.empty:
        L += ["## Regime family: zero-training context truncation", "",
              "`pretrained_truncate` keeps only the last max(d, 64) context steps when the scheduled switch is inside the "
              "context (no training). Same samples as the regime rows above.", ""]
        hdr = ["protocol", "size", "pre MASE ratio", "truncate MASE ratio", "pre sCRPS ratio", "truncate sCRPS ratio",
               "pre cov80", "truncate cov80"]
        seen, rows = set(), []
        for r in tr.itertuples():
            key = (r.protocol, r.size)
            if key in seen:
                continue
            seen.add(key)
            rows.append([r.protocol, r.size, _f(r.pre_mase_ratio), _f(r.trunc_mase_ratio), _f(r.pre_scrps_ratio),
                         _f(r.trunc_scrps_ratio), _f(r.pre_cov80), _f(r.trunc_cov80)])
        L += _md(hdr, rows) + [""]
        norm = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/mechanism/normalization/summary.csv")
        if os.path.exists(norm):
            import pandas as pd
            n = pd.read_csv(norm)
            sizes = sorted(tr["size"].unique())
            n = n[n["size"].isin(sizes) & (n["T"] == 2048) & n.condition.isin(["full", "post_only"])]
            if len(n):
                L += ["Lead's normalization test (`results/mechanism/normalization/summary.csv`, T = 2048; other samples "
                      "and seeds than the table above), rows copied:", ""]
                hdr = ["case", "size", "condition", "n", "mase_ratio_gmean", "scrps_ratio_gmean", "cov80"]
                rows = [[r.case, r.size, r.condition, str(int(r.n)), _f(r.mase_ratio_gmean), _f(r.scrps_ratio_gmean),
                         _f(r.cov80)] for r in n.itertuples()]
                L += _md(hdr, rows) + [""]
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(L) + "\n")


S_FAMILIES = ("chaos", "regime", "tails", "mixture")
