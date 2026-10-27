"""Train and score the whole fine-tuning grid in one process tree, saving partial results after every run.

Built for a Colab GPU job (see `scripts/colab/job_finetune.sh`). For each run of the grid
(sizes x protocols x held-out families x methods x replay shares):

  1. `scripts/finetune.py run`  trains it (its own subprocess, so peak memory is per run),
  2. `scripts/eval_finetuned.py --run-ids <id> --parts-only` scores it on its held-out set and the regression set,
  3. every `--checkpoint-every` runs, the score parts and the run records are written to
     `<out>/scores_partial.csv` (atomically: temp file, then rename).

If the machine is lost, relaunch with `--resume` and the partial file uploaded to the same path: the score parts
and run records are restored from it and finished runs are skipped. A run lost in the middle is trained again from
the start (a run is the unit of resume; a checkpoint inside a run would not survive the machine either).

When the grid is finished, one `eval_finetuned.py` call over all runs writes `scores.csv`, `summary.csv`,
`costs.csv`, `provenance.json` and `README.md` into `--out`, the loss curves of all runs go to
`loss_curves.csv`, and the partial file is removed.

Budgets that differ by size are given as `SIZE=VALUE` pairs, e.g. `--steps 5M=500 28M=500 123M=300 209M=300`.

  python scripts/finetune_job.py --out results/colab_out/ft --sizes 5M 28M --steps 5M=500 28M=500 \
      --batch 5M=32 28M=32 --lr-full 5M=5e-5 28M=3e-5 --lr-lora 5M=1e-3 28M=1e-3 --resume
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from otp import finetune_ext as X  # noqa: E402
from otp import finetune_splits as S  # noqa: E402
from otp.io import read_json, write_json  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRA = ["part", "run_json", "loss_csv"]


def per_size(pairs: list[str], sizes: list[str], cast) -> dict:
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        out[k] = cast(v)
    missing = [s for s in sizes if s not in out]
    if missing:
        raise SystemExit(f"no value for sizes {missing} in {pairs}")
    return out


def grid(a) -> list[tuple]:
    """(size, protocol, holdout, method, rf); the held-out families of one (size, protocol, method, rf) are consecutive."""
    return [(size, protocol, holdout, method, rf)
            for size in a.sizes for protocol in a.protocols for method in a.methods for rf in a.replay_fracs
            for holdout in a.holdouts]


def alias_run(a, base_rid: str, rid: str) -> None:
    """Under `knob` every held-out family is tested on the same trained model (the training data are all four families
    on the train side of each knob, whatever the held-out family), so it is trained once, under the first family's run
    id, and the other run ids point at the same checkpoint: same run record (plus `trained_as`), same loss curve."""
    brdir, rdir = os.path.join(a.runs_root, base_rid), os.path.join(a.runs_root, rid)
    os.makedirs(rdir, exist_ok=True)
    rec = read_json(os.path.join(brdir, "run.json"))
    rec.update(X.parse_run_id(rid))
    rec["trained_as"] = base_rid
    rec["checkpoint"] = os.path.join(os.path.dirname(rec["checkpoint"]), rid)
    write_json(os.path.join(rdir, "run.json"), rec)
    shutil.copy(os.path.join(brdir, "loss.csv"), os.path.join(rdir, "loss.csv"))
    link = os.path.join(a.ckpt_root, rid)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.abspath(os.path.join(a.ckpt_root, base_rid)), link)


# ---------------------------------------------------------------------------
# Partial file
# ---------------------------------------------------------------------------
def partial_path(a) -> str:
    return os.path.join(a.out, "scores_partial.csv")


def last_part(a, rid: str) -> str:
    """Score part written last for a run (a run counts as scored once it exists)."""
    use_real = bool(a.real_npz and os.path.exists(a.real_npz))
    return os.path.join(a.out, "parts", f"run__{rid}__regression-{'real' if use_real else 'in_prior'}.csv")


def collect_rows(a, included: set) -> pd.DataFrame:
    """Score parts and run records not yet in the partial file, as one table."""
    frames = []
    pdir = os.path.join(a.out, "parts")
    for path in sorted(glob.glob(os.path.join(pdir, "*.csv"))):
        name = os.path.basename(path)
        if name in included:
            continue
        df = pd.read_csv(path)
        df.insert(0, "part", name)
        frames.append(df)
        included.add(name)
    for path in sorted(glob.glob(os.path.join(a.runs_root, "*", "run.json"))):
        rid = os.path.basename(os.path.dirname(path))
        key = f"run_record__{rid}"
        if key in included or not os.path.exists(last_part(a, rid)):
            continue  # the record goes in once all of the run's scores are in
        rec = pd.DataFrame([{"part": key, "run_json": open(path).read(),
                             "loss_csv": open(os.path.join(os.path.dirname(path), "loss.csv")).read()}])
        frames.append(rec)
        included.add(key)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def flush_partial(a, included: set) -> int:
    new = collect_rows(a, included)
    if new.empty:
        return 0
    path = partial_path(a)
    tmp = path + ".tmp"
    if os.path.exists(path):
        old = pd.read_csv(path)
        cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
        merged = pd.concat([old, new], ignore_index=True).reindex(columns=cols)
    else:
        merged = new
    merged.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return len(new)


def restore_partial(a) -> set:
    """Rebuild `parts/` and `runs/<id>/{run.json,loss.csv}` from the partial file; return the names it holds."""
    path = partial_path(a)
    included: set = set()
    if not os.path.exists(path):
        return included
    df = pd.read_csv(path)
    pdir = os.path.join(a.out, "parts")
    os.makedirs(pdir, exist_ok=True)
    for name, g in df.groupby("part", sort=False):
        included.add(name)
        if name.startswith("run_record__"):
            rid = name[len("run_record__"):]
            rdir = os.path.join(a.runs_root, rid)
            os.makedirs(rdir, exist_ok=True)
            open(os.path.join(rdir, "run.json"), "w").write(g.run_json.iloc[0])
            open(os.path.join(rdir, "loss.csv"), "w").write(g.loss_csv.iloc[0])
        else:
            g.drop(columns=[c for c in EXTRA if c in g.columns]).dropna(axis=1, how="all").to_csv(
                os.path.join(pdir, name), index=False)
    print(f"restored {len([n for n in included if n.startswith('run_record__')])} runs and "
          f"{len([n for n in included if not n.startswith('run_record__')])} score parts from {path}", flush=True)
    return included


def is_done(a, rid: str) -> bool:
    return os.path.exists(os.path.join(a.runs_root, rid, "run.json")) and os.path.exists(last_part(a, rid))


# ---------------------------------------------------------------------------
def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"command failed with exit code {r.returncode}: {' '.join(cmd)}")


def eval_args(a) -> list[str]:
    cmd = ["--runs-root", a.runs_root, "--ckpt-root", a.ckpt_root, "--out", a.out, "--cache-dir", a.cache_dir,
           "--n-eval", str(a.n_eval), "--n-regression", str(a.n_regression), "--n-paths", str(a.n_paths)]
    if a.real_npz and os.path.exists(a.real_npz):
        cmd += ["--real-npz", a.real_npz]
    if a.in_prior_npz and os.path.exists(a.in_prior_npz):
        cmd += ["--in-prior-npz", a.in_prior_npz]
    if a.threads:
        cmd += ["--threads", str(a.threads)]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results folder (scores_partial.csv is written here)")
    ap.add_argument("--runs-root", default=None, help="default <out>/runs")
    ap.add_argument("--ckpt-root", required=True, help="checkpoints (large; not for git)")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--cache-dir", default=None, help="held-out sample cache (default <pool-dir>/eval_cache)")
    ap.add_argument("--sizes", nargs="+", default=["5M", "28M", "123M", "209M"])
    ap.add_argument("--protocols", nargs="+", default=list(S.PROTOCOLS), choices=list(S.PROTOCOLS))
    ap.add_argument("--holdouts", nargs="+", default=list(S.FAMILIES), choices=list(S.FAMILIES))
    ap.add_argument("--methods", nargs="+", default=["full", "lora"], choices=["full", "lora"])
    ap.add_argument("--replay-fracs", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    ap.add_argument("--steps", nargs="+", required=True, metavar="SIZE=N")
    ap.add_argument("--batch", nargs="+", required=True, metavar="SIZE=N")
    ap.add_argument("--lr-full", nargs="+", required=True, metavar="SIZE=LR")
    ap.add_argument("--lr-lora", nargs="+", required=True, metavar="SIZE=LR")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-batches", type=int, default=500)
    ap.add_argument("--pool-workers", type=int, default=8)
    ap.add_argument("--contexts", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--horizons", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--n-eval", type=int, default=S.N_EVAL_DEFAULT)
    ap.add_argument("--n-regression", type=int, default=S.N_REGRESSION_DEFAULT)
    ap.add_argument("--n-paths", type=int, default=500)
    ap.add_argument("--real-npz", default="bench/regression/real.npz")
    ap.add_argument("--in-prior-npz", default="bench/regression/in_prior.npz")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--checkpoint-every", type=int, default=1, help="write scores_partial.csv every this many finished runs")
    ap.add_argument("--resume", action="store_true", help="restore finished runs from scores_partial.csv and skip them")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="keep each checkpoint after it is scored (default: delete it; the 123M and 209M ones are large)")
    ap.add_argument("--keep-partial", action="store_true", help="do not remove scores_partial.csv when the grid is finished")
    ap.add_argument("--note", action="append", default=[], help="fact for the README (repeatable)")
    a = ap.parse_args(argv)

    a.runs_root = a.runs_root or os.path.join(a.out, "runs")
    a.cache_dir = a.cache_dir or os.path.join(a.pool_dir, "eval_cache")
    steps = per_size(a.steps, a.sizes, int)
    batch = per_size(a.batch, a.sizes, int)
    lr_full = per_size(a.lr_full, a.sizes, float)
    lr_lora = per_size(a.lr_lora, a.sizes, float)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    pth = partial_path(a)
    if a.resume:
        included = restore_partial(a)
    else:
        for d in (a.runs_root, os.path.join(a.out, "parts")):
            shutil.rmtree(d, ignore_errors=True)
        if os.path.exists(pth):
            os.remove(pth)
        included = set()

    todo = grid(a)
    n_done = 0
    base_holdout = a.holdouts[0]
    for i, (size, protocol, holdout, method, rf) in enumerate(todo):
        rid = X.run_id(size, protocol, holdout, method, rf)
        if is_done(a, rid):
            print(f"[{i + 1}/{len(todo)}] skip {rid}", flush=True)
            continue
        shared = protocol == "knob" and holdout != base_holdout
        base_rid = X.run_id(size, protocol, base_holdout, method, rf)
        if shared and os.path.isdir(os.path.join(a.ckpt_root, base_rid)):
            print(f"[{i + 1}/{len(todo)}] share {rid} <- {base_rid} ({time.time() - t0:.0f}s)", flush=True)
            alias_run(a, base_rid, rid)
        else:
            if shared:  # the base checkpoint is gone (session lost in the middle of this group): train it again
                rid_train = base_rid
                holdout_train = base_holdout
            else:
                rid_train, holdout_train = rid, holdout
            print(f"[{i + 1}/{len(todo)}] train {rid_train} ({time.time() - t0:.0f}s)", flush=True)
            lr = lr_full[size] if method == "full" else lr_lora[size]
            run([sys.executable, os.path.join(HERE, "finetune.py"), "run", "--size", size, "--protocol", protocol,
                 "--holdout", holdout_train, "--method", method, "--replay-frac", str(rf), "--steps", str(steps[size]),
                 "--batch-size", str(batch[size]), "--lr", str(lr), "--rank", str(a.rank), "--alpha", str(a.alpha),
                 "--seed", str(a.seed), "--warmup", str(max(1, steps[size] // 20)), "--log-every", str(a.log_every),
                 "--pool-batches", str(a.pool_batches), "--pool-workers", str(a.pool_workers),
                 "--pool-batch-size", str(max(batch.values())),
                 "--contexts", *map(str, a.contexts), "--horizons", *map(str, a.horizons),
                 "--runs-root", a.runs_root, "--ckpt-root", a.ckpt_root, "--pool-dir", a.pool_dir]
                + (["--threads", str(a.threads)] if a.threads else []))
            if shared:
                alias_run(a, base_rid, rid)
        print(f"[{i + 1}/{len(todo)}] score {rid} ({time.time() - t0:.0f}s)", flush=True)
        run([sys.executable, os.path.join(HERE, "eval_finetuned.py"), *eval_args(a), "--run-ids", rid, "--parts-only"])
        last_of_group = protocol != "knob" or holdout == a.holdouts[-1]
        if not a.keep_checkpoints and last_of_group:
            for h in (a.holdouts if protocol == "knob" else [holdout]):
                shutil.rmtree(os.path.join(a.ckpt_root, X.run_id(size, protocol, h, method, rf)), ignore_errors=True)
                p = os.path.join(a.ckpt_root, X.run_id(size, protocol, h, method, rf))
                if os.path.islink(p):
                    os.remove(p)
        n_done += 1
        if n_done % max(1, a.checkpoint_every) == 0:
            flush_partial(a, included)
            print(f"[{i + 1}/{len(todo)}] partial saved ({time.time() - t0:.0f}s)", flush=True)
    flush_partial(a, included)

    # ---- final tables from the cached parts (no checkpoints are needed for runs that are already scored)
    note_args = [x for n in a.note for x in ("--note", n)]
    run([sys.executable, os.path.join(HERE, "eval_finetuned.py"), *eval_args(a), *note_args])
    rows = []
    for rid in sorted(os.listdir(a.runs_root)):
        p = os.path.join(a.runs_root, rid, "loss.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.insert(0, "run_id", rid)
            rows.append(df)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(os.path.join(a.out, "loss_curves.csv"), index=False)
    write_json(os.path.join(a.out, "job.json"), {
        "grid_runs": len(todo), "trained_in_this_process": n_done, "resumed": bool(a.resume),
        "wall_sec_this_process": round(time.time() - t0, 1),
        "steps": steps, "batch": batch, "lr_full": lr_full, "lr_lora": lr_lora, "rank": a.rank, "alpha": a.alpha,
        "pool_batches": a.pool_batches, "contexts": a.contexts, "horizons": a.horizons,
        "keep_checkpoints": bool(a.keep_checkpoints)})
    if os.path.exists(pth) and not a.keep_partial:
        os.remove(pth)
    print(f"finished {len(todo)} runs in {time.time() - t0:.0f}s (this process)", flush=True)


if __name__ == "__main__":
    main()
