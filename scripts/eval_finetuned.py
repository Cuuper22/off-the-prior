"""Score fine-tuned Timestral checkpoints against the pretrained model on held-out samples.

For every run found under `--runs-root` (written by `scripts/finetune.py`):

- the run's held-out sample set (`lofo/<family>` or `knob/<family>`: `--n-eval` samples, each with
  a 500-path true-dynamics oracle),
- the regression set: `--n-regression` fresh in-prior series (seeds 100000 and up) with oracle, and,
  when `--real-npz` is given, the real-data windows (no oracle; only the paired change versus the
  pretrained model is reported for them).

are scored with `otp.evaluate.score_samples` for the pretrained model of the same size, the
fine-tuned checkpoint, and the oracle. For the regime family the pretrained model is also scored with
the zero-training fix "keep only the last max(d, 64) steps of the context" (`pretrained_truncate`), for
samples whose scheduled switch lies inside the context.

Outputs in `--out`:
  scores.csv       long table, one row per sample x forecaster x run x eval set
  summary.csv      one row per run x eval set: ratio to oracle before / after, paired change, coverage
  costs.csv        one row per run: trainable parameters, steps, time steps seen, wall time, memory
  provenance.json  commands, sizes, sample counts, wall time, splits
  README.md        tables copied from the csv files
With `--cards-dir`, model-card drafts (numbers copied from summary.csv and costs.csv) are written for the scored runs;
`--cards-colab-placeholders DIR` writes drafts for the Colab grid with placeholders that name the result file and column.
Per-run score parts are cached in `<out>/parts/` so an interrupted evaluation resumes.

  python scripts/eval_finetuned.py --runs-root results/finetune/local/runs \
      --ckpt-root checkpoints/finetune/local --out results/finetune/local --n-eval 32
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from otp import finetune_ext as X  # noqa: E402
from otp import finetune_splits as S  # noqa: E402
from otp.evaluate import score_samples  # noqa: E402
from otp.generators.base import Sample, load_samples, save_samples  # noqa: E402
from otp.io import provenance, read_json, write_json  # noqa: E402
from otp.model import set_threads  # noqa: E402

TRUNC_MIN = 64


def truncate_regime(samples: list[Sample]) -> list[Sample]:
    """Copies of regime samples with the context cut to the last max(d, 64) steps, when the scheduled
    switch (placement `context`, switch distance d) lies inside the context. Other samples are copied unchanged."""
    out = []
    for s in samples:
        p = s.meta.get("params", {})
        L = s.T
        if s.meta.get("family") == "regime" and p.get("placement") == "context" and int(p["d"]) < s.T:
            L = max(int(p["d"]), TRUNC_MIN)
        out.append(Sample(s.context[..., -L:], s.future, s.oracle_q, s.period, dict(s.meta),
                          None if s.past_cov is None else s.past_cov[..., -L:], s.known_cov))
    return out


def get_set(cache_dir: str, name: str, maker) -> list[Sample]:
    path = os.path.join(cache_dir, name + ".npz")
    if os.path.exists(path):
        return load_samples(path)
    samples = maker()
    os.makedirs(cache_dir, exist_ok=True)
    save_samples(path, samples)
    return samples


def tag(df: pd.DataFrame, **cols) -> pd.DataFrame:
    for k, v in cols.items():
        df[k] = v
    return df


def score_part(path: str, fn) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    df = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="results/finetune/local/runs")
    ap.add_argument("--ckpt-root", default="checkpoints/finetune/local")
    ap.add_argument("--out", default="results/finetune/local")
    ap.add_argument("--cache-dir", default=None, help="held-out sample cache (default <out>/eval_cache; not for git)")
    ap.add_argument("--n-eval", type=int, default=S.N_EVAL_DEFAULT)
    ap.add_argument("--n-regression", type=int, default=S.N_REGRESSION_DEFAULT)
    ap.add_argument("--n-paths", type=int, default=500, help="oracle Monte Carlo paths per sample")
    ap.add_argument("--real-npz", default=None, help="real-data regression windows (bench/regression/real.npz)")
    ap.add_argument("--sizes", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-readme", action="store_true")
    ap.add_argument("--note", action="append", default=[], help="fact to list under Notes in README.md (repeatable)")
    ap.add_argument("--cards-dir", default=None, help="also write model-card drafts for the scored runs here")
    ap.add_argument("--cards-colab-placeholders", default=None, metavar="DIR",
                    help="write model-card drafts with result-file placeholders for the Colab grid, then exit")
    a = ap.parse_args(argv)
    if a.cards_colab_placeholders:
        print("wrote", X.write_cards_colab_placeholders(a.cards_colab_placeholders), "cards to", a.cards_colab_placeholders)
        return
    set_threads(a.threads)
    t0 = time.time()
    cache = a.cache_dir or os.path.join(a.out, "eval_cache")
    parts = os.path.join(a.out, "parts")

    runs = [read_json(p) for p in sorted(glob.glob(os.path.join(a.runs_root, "*", "run.json")))]
    if a.sizes:
        runs = [r for r in runs if r["size"] in a.sizes]
    assert runs, f"no runs under {a.runs_root}"
    sizes = sorted({r["size"] for r in runs}, key=lambda s: float(s[:-1]))

    # ---- held-out sample sets
    sets: dict[str, list[Sample]] = {}
    for protocol, holdout in sorted({(r["protocol"], r["holdout"]) for r in runs}):
        sets[f"{protocol}/{holdout}"] = get_set(
            cache, f"eval__{protocol}-{holdout}__n{a.n_eval}_p{a.n_paths}",
            lambda protocol=protocol, holdout=holdout: S.make_eval_samples(protocol, holdout, a.n_eval, a.n_paths))
    sets["regression/in_prior"] = get_set(
        cache, f"regression__in_prior__n{a.n_regression}_p{a.n_paths}",
        lambda: S.make_regression_samples(a.n_regression, a.n_paths))
    if a.real_npz:
        from otp.realdata import load_real
        sets["regression/real"] = load_real(a.real_npz)
        for i, s in enumerate(sets["regression/real"]):
            s.meta.setdefault("config_id", "real")
    trunc_sets = {k: truncate_regime(v) for k, v in sets.items() if k.endswith("/regime")}
    print({k: len(v) for k, v in sets.items()}, flush=True)

    # ---- oracle rows (once per set), pretrained rows (once per size and set), fine-tuned rows (per run)
    frames = []
    for name, ss in sets.items():
        if name != "regression/real":
            df = score_part(os.path.join(parts, f"oracle__{name.replace('/', '-')}.csv"),
                            lambda ss=ss: score_samples(ss, ["oracle"]))
            frames.append(tag(df, eval_set=name, size="", run_id="", role=None))
    for name, ss in trunc_sets.items():
        df = score_part(os.path.join(parts, f"oracle__{name.replace('/', '-')}__trunc.csv"),
                        lambda ss=ss: score_samples(ss, ["oracle"]))
        frames.append(tag(df, eval_set=name, size="", run_id="", role=None, forecaster="oracle"))
    for size in sizes:
        for name, ss in sets.items():
            df = score_part(os.path.join(parts, f"pretrained__{size}__{name.replace('/', '-')}.csv"),
                            lambda ss=ss: score_samples(ss, [f"timestral-{size}"]))
            df["forecaster"] = "pretrained"
            frames.append(tag(df, eval_set=name, size=size, run_id="", role=None))
        for name, ss in trunc_sets.items():
            df = score_part(os.path.join(parts, f"pretrained_truncate__{size}__{name.replace('/', '-')}.csv"),
                            lambda ss=ss: score_samples(ss, [f"timestral-{size}"]))
            df["forecaster"] = "pretrained_truncate"
            frames.append(tag(df, eval_set=name, size=size, run_id="", role=None))
    for r in runs:
        rid = r["run_id"]
        ck = os.path.join(a.ckpt_root, rid)
        assert os.path.isdir(ck), f"missing checkpoint {ck}"
        own = f"{r['protocol']}/{r['holdout']}"
        for name in [own] + [k for k in sets if k.startswith("regression/")]:
            ss = sets[name]
            df = score_part(os.path.join(parts, f"run__{rid}__{name.replace('/', '-')}.csv"),
                            lambda ss=ss, ck=ck: score_samples(ss, [f"timestral-{ck}"]))
            df["forecaster"] = "finetuned"
            frames.append(tag(df, eval_set=name, size=r["size"], run_id=rid,
                              role="test" if name == own else "regression"))
        print("scored", rid, f"{time.time() - t0:.0f}s", flush=True)

    scores = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame([{k: r[k] for k in ("run_id", "protocol", "holdout", "method", "replay_frac")} for r in runs])
    scores = scores.merge(meta, on="run_id", how="left")
    os.makedirs(a.out, exist_ok=True)
    keep = ["eval_set", "size", "run_id", "protocol", "holdout", "method", "replay_frac", "forecaster", "sample_id",
            "family", "generator", "knob", "knob_value", "seed", "T", "H", "variate", "mase", "scrps", "wql", "cov80",
            "cov98", "calib"]
    scores[keep].to_csv(os.path.join(a.out, "scores.csv"), index=False)

    costs = pd.DataFrame([X.cost_row(r) for r in runs])
    costs.to_csv(os.path.join(a.out, "costs.csv"), index=False)
    summary = X.summarize(scores, runs, own_sets={(r["protocol"], r["holdout"]) for r in runs})
    summary.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    import torch
    write_json(os.path.join(a.out, "provenance.json"), provenance(
        "scripts/eval_finetuned.py", argv=sys.argv[1:] if argv is None else list(argv), wall_sec=round(time.time() - t0, 1),
        sizes=sizes, n_runs=len(runs), n_eval=a.n_eval, n_regression=a.n_regression, n_paths=a.n_paths,
        sets={k: len(v) for k, v in sets.items()}, real_npz=a.real_npz,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        train_wall_sec_total=round(float(costs.wall_sec.sum()), 1), splits=S.split_summary()))
    if not a.no_readme:
        X.write_readme(a.out, summary, costs, scores, a.note)
    if a.cards_dir:
        print("wrote", X.write_cards_local(a.cards_dir, summary, costs, a.out), "cards to", a.cards_dir)
    print("wrote", a.out, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
