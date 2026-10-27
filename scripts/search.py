"""Run the failure searches: CMA-ES, MAP-Elites and gradient search (plus the input-space study).

Equal budget: every run of every method gets `--budget` Timestral forward calls on `--size`
(3 per genome: one per fixed seed; gradient passes charge 2 per series, forward + backward).

  python scripts/search.py --methods cmaes map_elites gradient --seeds 0 1 2 --budget 2400 \
      --workers 11 --out results/search
  python scripts/search.py --methods input_space --sizes 5M 28M 123M 209M --n-series 20 --out results/search

Outputs per run, in <out>/runs/<method>_seed<k>/ (or <out>/<flat-prefix><method>_seed<k>/ with --flat-prefix):
  history.csv   one row per fresh genome evaluation (evaluation index, cumulative calls, best-so-far,
                genome, fitness, feasibility, descriptors); gradient runs also write grad_trace.csv
  elites.json   top 20 feasible genomes with their `describe()` recipes (+ the archive cells for MAP-Elites)
  run.json      budget, calls, evaluations, sample generations, wall time
A run interrupted by a lost runtime leaves scores_partial.csv; --resume replays it.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp.generators import mixture  # noqa: E402
from otp.io import provenance, read_json, write_json  # noqa: E402
from otp.model import set_threads  # noqa: E402
from otp.search.objective import Evaluator, Objective  # noqa: E402

TOP_N = 20


def run_dir_for(args, method: str, seed: int) -> str:
    name = f"{args.label or method}_seed{seed}"
    if args.flat_prefix:
        return os.path.join(args.out, args.flat_prefix + name)
    return os.path.join(args.out, "runs", name)


def top_elites(rows: list[dict], n: int = TOP_N) -> list[dict]:
    import json
    feas = [r for r in rows if str(r["feasible"]) == "True"]
    feas.sort(key=lambda r: -float(r["fitness"]))
    out = []
    for rank, r in enumerate(feas[:n], 1):
        g = np.array(json.loads(r["genome"]))
        out.append({"rank": rank, "hash": r["hash"], "fitness": float(r["fitness"]), "ratio_scrps": float(r["ratio_scrps"]),
                    "cov80": float(r["cov80"]), "family": r["family"], "signature": r["signature"],
                    "spec_entropy": float(r["spec_entropy"]), "eval": int(r["eval"]), "genome": g.tolist(),
                    "recipe": mixture.describe(g)})
    return out


def do_search(args, method: str, seed: int, obj: Objective):
    from otp.search.cmaes import run_cmaes
    from otp.search.gradient import grad_search
    from otp.search.map_elites import run_map_elites

    rd = run_dir_for(args, method, seed)
    if os.path.exists(os.path.join(rd, "history.csv")) and os.path.exists(os.path.join(rd, "elites.json")) and not args.force:
        print(f"{method} seed {seed}: already done, skipping", flush=True)
        return
    t0 = time.time()
    label = args.label or method
    ev = Evaluator(obj, args.budget, rd, label, seed, resume=args.resume)
    gen0, calls0 = obj.sample_generations, obj.calls
    archive = None
    if method == "cmaes":
        info = run_cmaes(ev, seed)
    elif method == "map_elites":
        info, archive = run_map_elites(ev, seed)
    elif method == "gradient":
        info = grad_search(ev, seed, size=args.size, n_restarts=args.grad_restarts, n_draws=args.grad_draws)
    else:
        raise ValueError(method)
    ev.finish()
    elites = {"method": label, "seed": seed, "size": args.size, "budget": args.budget, "calls": ev.calls,
              "top": top_elites(ev.rows)}
    if archive is not None:
        elites["archive"] = [{"family_index": c[0], "entropy_bin": c[1], "fitness": v[0], "hash": v[2],
                              "genome": v[1].tolist(), "recipe": mixture.describe(v[1])}
                             for c, v in sorted(archive.cells.items())]
    write_json(os.path.join(rd, "elites.json"), elites)
    wall = time.time() - t0
    write_json(os.path.join(rd, "run.json"), {
        "method": label, "algorithm": method, "grad_restarts": args.grad_restarts, "grad_draws": args.grad_draws, "seed": seed, "size": args.size, "budget": args.budget, "calls": ev.calls,
        "evaluations": len(ev.rows), "replayed_from_checkpoint": ev.n_replayed,
        "sample_generations_this_process": obj.sample_generations - gen0,
        "timestral_calls_made_by_objective": obj.calls - calls0, "best_fitness": ev.best,
        "wall_seconds": wall, "info": info})
    print(f"{method} seed {seed}: {len(ev.rows)} evals, calls {ev.calls}, best fitness {ev.best:.4f}, "
          f"{wall:.0f}s", flush=True)


def do_input_space(args):
    from otp.search.gradient import input_sensitivity

    out_dir = args.out if not args.flat_prefix else os.path.join(args.out, args.flat_prefix + "input_space")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "input_space.csv")
    rows = []
    done = set()
    if os.path.exists(path) and args.resume:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        done = {(r["size"], int(r["series_seed"])) for r in rows}
    for size in args.sizes:
        for s in range(args.n_series):
            if (size, s) in done:
                continue
            t0 = time.time()
            r = input_sensitivity(size, s)
            r["seconds"] = time.time() - t0
            rows.append(r)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"input_space {size}: {args.n_series} series done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["cmaes", "map_elites", "gradient"],
                    choices=["cmaes", "map_elites", "gradient", "input_space"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=2400, help="Timestral forward calls per run")
    ap.add_argument("--size", default="5M")
    ap.add_argument("--sizes", nargs="+", default=["5M", "28M", "123M", "209M"], help="input_space only")
    ap.add_argument("--n-series", type=int, default=20, help="input_space only")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--label", default="", help="name for the run folders and the method column (default: the method name); "
                    "use it to run a variant, e.g. --methods gradient --grad-restarts 12 --label gradient_r12")
    ap.add_argument("--grad-restarts", type=int, default=3)
    ap.add_argument("--grad-draws", type=int, default=2)
    ap.add_argument("--out", default="results/search")
    ap.add_argument("--flat-prefix", default="", help="write run folders as <out>/<prefix><method>_seed<k> (Colab driver layout)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)
    set_threads(args.threads)
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    obj = Objective(args.size, workers=args.workers)
    try:
        for method in args.methods:
            if method == "input_space":
                do_input_space(args)
                continue
            for seed in args.seeds:
                do_search(args, method, seed, obj)
    finally:
        obj.close()
    prov = provenance("scripts/search.py", argv=sys.argv[1:], wall_seconds=time.time() - t_start,
                      budget=args.budget, size=args.size, workers=args.workers)
    pp = os.path.join(args.out, "provenance_search.json" if not args.flat_prefix else f"{args.flat_prefix}provenance.json")
    old = read_json(pp) if os.path.exists(pp) else {"invocations": []}
    old["invocations"].append(prov)
    write_json(pp, old)


if __name__ == "__main__":
    main()
