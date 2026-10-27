"""Re-score the top elites of each search method on Timestral 209M, 123M, 28M and 5M with fresh seeds.

Elites: for each method, the top-N feasible genomes pooled over that method's search runs
(each run's `elites.json`), deduplicated by genome hash. Every genome is simulated once per fresh
seed (`objective.FRESH_SEEDS`, none used during search) and the same samples are forecast by each
size, so differences between sizes are differences between models. 209M is scored first.

  python scripts/search_transfer.py --runs results/search/runs --top 20 --workers 11 --out results/search

Outputs: transfer.csv (one row per method x genome x size), transfer_summary.csv (fractions above
1.5 and 2 per method and size), transfer_spearman.csv (rank correlation of the MASE ratio between
sizes), elites.npz (the samples, T = 2048, H = 128, built by `make_sample`), transfer.json (timings).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp.generators.base import save_samples  # noqa: E402
from otp.io import provenance, read_json, write_json  # noqa: E402
from otp.model import set_threads  # noqa: E402
from otp.search.objective import FRESH_SEEDS, Objective, genome_hash  # noqa: E402

ORDER = ["209M", "123M", "28M", "5M"]


def pool_elites(runs_glob: str, top: int) -> dict[str, list[dict]]:
    by_method: dict[str, dict[str, dict]] = {}
    for p in sorted(glob.glob(runs_glob)):
        e = read_json(p)
        d = by_method.setdefault(e["method"], {})
        for t in e["top"]:
            if t["hash"] not in d or t["fitness"] > d[t["hash"]]["fitness"]:
                d[t["hash"]] = {**t, "run_seed": e["seed"]}
    return {m: sorted(d.values(), key=lambda t: -t["fitness"])[:top] for m, d in by_method.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/search/runs")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--sizes", nargs="+", default=ORDER)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/search")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    set_threads(a.threads)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    pooled = pool_elites(os.path.join(a.runs, "*", "elites.json") if "*" not in a.runs else a.runs, a.top)
    genomes: dict[str, np.ndarray] = {}
    for m, lst in pooled.items():
        for t in lst:
            genomes[t["hash"]] = np.array(t["genome"])
    order = list(genomes)
    print({m: len(v) for m, v in pooled.items()}, "unique genomes:", len(order), flush=True)
    obj = Objective("5M", seeds=FRESH_SEEDS, workers=a.workers)
    gl = [genomes[h] for h in order]
    t1 = time.time()
    obj.records(gl, keep_samples=True)
    timings = {"generate_samples_s": time.time() - t1, "n_samples": len(gl) * len(FRESH_SEEDS)}
    print(f"samples generated ({timings['generate_samples_s']:.0f}s)", flush=True)
    res: dict[tuple[str, str], object] = {}
    for size in a.sizes:
        t1 = time.time()
        for h, r in zip(order, obj.evaluate(gl, size=size, keep_samples=True)):
            res[(h, size)] = r
        timings[f"score_{size}_s"] = time.time() - t1
        print(f"{size} scored ({timings[f'score_{size}_s']:.0f}s)", flush=True)
    obj.close()
    rows = []
    for m, lst in pooled.items():
        for t in lst:
            for size in a.sizes:
                r = res[(t["hash"], size)]
                rows.append({"method": m, "rank": t["rank"], "run_seed": t["run_seed"], "hash": t["hash"],
                             "search_fitness_5M": t["fitness"], "size": size, "fitness": r.fitness,
                             "feasible": r.feasible, "ratio_mase": r.ratio_mase, "ratio_scrps": r.ratio_scrps,
                             "cov80": r.cov80, "orc_sn": r.orc_sn, "family": r.family, "signature": r.signature,
                             "reason": r.reason})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "transfer.csv"), index=False)
    # fractions above thresholds (of the genomes in the list, counting infeasible-on-fresh-seeds as below)
    srows = []
    for m in list(pooled) + ["all"]:
        d = df if m == "all" else df[df.method == m]
        for size in a.sizes:
            x = d[d["size"] == size].drop_duplicates("hash") if m == "all" else d[d["size"] == size]
            srows.append({"method": m, "size": size, "n": len(x), "n_feasible_fresh": int(x.feasible.sum()),
                          "frac_gt_1.5": float((x.fitness > 1.5).mean()), "frac_gt_2": float((x.fitness > 2).mean()),
                          "median_fitness": float(x.fitness.median()),
                          "median_fitness_search_5M": float(x.search_fitness_5M.median())})
    pd.DataFrame(srows).to_csv(os.path.join(a.out, "transfer_summary.csv"), index=False)
    # Spearman of the MASE ratio between sizes (genomes feasible on the fresh seeds), plus search-seed 5M
    wide = df.pivot_table(index=["method", "hash"], columns="size", values="ratio_mase")
    feas = df[df["size"] == a.sizes[0]].set_index(["method", "hash"]).feasible
    srch = df[df["size"] == a.sizes[0]].set_index(["method", "hash"]).search_fitness_5M
    wide = wide.join(feas.rename("feasible_fresh")).join(srch.rename("search_5M"))
    wide = wide[wide.feasible_fresh]
    cols = ["search_5M"] + [s for s in ORDER if s in a.sizes]
    prow = []
    for m in list(pooled) + ["all"]:
        w = wide if m == "all" else wide.xs(m, level="method", drop_level=False)
        if m == "all":
            w = w.reset_index().drop_duplicates("hash").set_index(["method", "hash"])
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                ok = w[[c1, c2]].dropna()
                rho = spearmanr(ok[c1], ok[c2])[0] if len(ok) >= 3 else float("nan")
                prow.append({"method": m, "a": c1 if c1 != "5M" else "5M_fresh", "b": c2 if c2 != "5M" else "5M_fresh",
                             "n": len(ok), "spearman": float(rho)})
    pd.DataFrame(prow).to_csv(os.path.join(a.out, "transfer_spearman.csv"), index=False)
    # elites.npz: the fresh-seed samples of all genomes, with method/rank in the meta
    samples = []
    tag: dict[str, list] = {}
    for m, lst in pooled.items():
        for t in lst:
            tag.setdefault(t["hash"], []).append({"method": m, "rank": t["rank"]})
    for h in order:
        for s in res[(h, a.sizes[0])].samples:
            s.meta["sample_id"] = f"search/{h}/s{s.meta['seed']}"
            s.meta["elite_of"] = tag[h]
            s.meta["genome"] = [float(x) for x in genomes[h]]
            samples.append(s)
    save_samples(os.path.join(a.out, "elites.npz"), samples)
    timings["wall_seconds"] = time.time() - t0
    write_json(os.path.join(a.out, "transfer.json"), {
        "timings": timings, "sizes": a.sizes, "fresh_seeds": list(FRESH_SEEDS), "n_genomes": len(order),
        "n_samples": len(samples),
        "provenance": provenance("scripts/search_transfer.py", argv=sys.argv[1:], wall_seconds=timings["wall_seconds"],
                                 workers=a.workers)})
    print("done", flush=True)


if __name__ == "__main__":
    main()
