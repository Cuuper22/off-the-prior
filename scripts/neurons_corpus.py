"""Build the audit corpus files.

    python scripts/neurons_corpus.py --variant main  --out results/neurons/corpus
    python scripts/neurons_corpus.py --variant short --out results/neurons/corpus
    python scripts/neurons_corpus.py --variant long  --out results/neurons/corpus
    python scripts/neurons_corpus.py --variant xlong --out results/neurons/corpus

Variants (same recipe, same series ids; the context length T differs):
  main   all series, T = 1024
  short  series with sid % 3 == 0, T = 128
  long   series with sid % 5 == 1, T = 4096
  xlong  48 series (8 per family group, cheapest chaos generators), T = 32768
  mv     240 two-row coupled systems (lagged driver, linear feedback), T = 1024; rows of one system share a group
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from otp.io import provenance, write_json  # noqa: E402
from otp.neurons import corpus as C  # noqa: E402

VARIANTS = {"main": 1024, "short": 128, "long": 4096, "xlong": 32768, "mv": 1024}


def pick(specs, variant):
    if variant == "main":
        return specs
    if variant == "short":
        return [s for s in specs if s["sid"] % 3 == 0]
    if variant == "long":
        return [s for s in specs if s["sid"] % 5 == 1]
    out = []
    groups = {"in_prior": ["in_prior", "ctl_season", "ctl_mix"], "regime": ["mean_jump", "variance_jump",
              "frequency_jump", "markov_ar"], "tails": ["stable_ar", "hawkes", "msm_returns", "fbm"],
              "chaos": ["logistic_map", "mackey_glass"]}
    for fam, gens in groups.items():
        c = [s for s in specs if s["generator"] in gens]
        k = max(1, 12 // len(gens)) if fam != "chaos" else 6
        # spread over generators
        per = {}
        for s in c:
            per.setdefault(s["generator"], []).append(s)
        for g in gens:
            out += per[g][:k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--out", default="results/neurons/corpus")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = VARIANTS[a.variant]
    if a.variant == "mv":
        sys_specs = C.make_mv_specs()
        c, specs = C.build_mv(sys_specs, T)
        sp, fo = C.split_of(sys_specs)
        c["split"], c["fold"] = np.repeat(sp, 2), np.repeat(fo, 2)
    else:
        allspecs = C.make_specs(a.scale)
        specs = pick(allspecs, a.variant)
        print(f"{a.variant}: {len(specs)} series, T={T}", flush=True)
        c = C.build_corpus(specs, T, workers=a.workers)
        # keep the global series id; arrays are indexed by position in `specs`
        c["sid"] = np.asarray([s["sid"] for s in specs])
        split, fold = C.split_of(allspecs)
        c["split"], c["fold"] = split[c["sid"]], fold[c["sid"]]
    path = os.path.join(a.out, f"{a.variant}.npz")
    C.save_corpus(path, c, specs)
    write_json(os.path.join(a.out, f"{a.variant}_manifest.json"), {
        "variant": a.variant, "T": T, "n_series": len(specs), "sha256": C.sha256(path),
        "bytes": os.path.getsize(path), "recipe": C.RECIPE,
        "by_generator": {g: int(sum(1 for s in specs if s["generator"] == g)) for g in sorted({s["generator"] for s in specs})},
        "n_label": int((c["split"] == 0).sum()), "n_heldout": int((c["split"] == 1).sum()),
        "provenance": provenance("scripts/neurons_corpus.py", variant=a.variant)})
    print("wrote", path, os.path.getsize(path) / 1e6, "MB")


if __name__ == "__main__":
    main()
