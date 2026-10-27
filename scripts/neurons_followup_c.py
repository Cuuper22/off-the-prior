"""Follow-up C: motif labels without a variable list (pre-registered in
results/neurons/steering_preregistration_followup.md, section C).

    python scripts/neurons_followup_c.py --atlas <dir with nn_<size>_main|short|long|xlong> \
        --fb <dir with runs.csv and run folders> [--fb <more>] --corpus-dir results/neurons/corpus \
        --out results/neurons/followup

For every capture run (scripts/neurons_run.py output `scores_partial.csv`) the 8 top and 8 mid-range exemplar tokens of
each pole are mapped to their 32-value patches, normalized, assigned to a fixed vocabulary of shapes (k-means centers
fitted on the top-exemplar patches of the four atlas main-corpus runs), and a pole is motif-cohesive when at least 6 of
its 8 top exemplars share a cluster. Writes `motif_clusters.csv` (one row per run), `motif_cluster_names.csv` (names of
the vocabulary clusters), `motif_vocabulary_K16.csv` (centers), `motif_by_name.csv` and a provenance file.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from otp.io import provenance, write_json  # noqa: E402
from otp.neurons import corpus as C  # noqa: E402
from otp.neurons import motifs as M  # noqa: E402
from otp.neurons import score as Sc  # noqa: E402

SIZES = ("5M", "28M", "123M", "209M")
KS_VOCAB = (8, 16, 32)
K_MAIN = 16
N_PERM = 20
USECOLS = ["layer", "neuron", "sign", "dead", "label", "r2_cv", "r2_heldout", "r2_pair", "r2_pair_heldout", "r2_add", "r2_add_heldout",
           "r2_fold0", "r2_fold1", "r2_fold2", "top_tok", "mid_tok"]


# ---------------------------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------------------------
def build_manifest(atlas: str, fb_dirs: list[str], flat_dirs: list[str] = (), plan: str = "") -> pd.DataFrame:
    rows = []
    for S in SIZES:
        for ctxname, role in (("main", "atlas_main"), ("short", "context"), ("long", "context"), ("xlong", "context")):
            p = os.path.join(atlas, f"nn_{S}_{ctxname}", "scores_partial.csv")
            if os.path.exists(p):
                rows.append({"tag": f"{S}_atlas_{ctxname}", "size": S, "path": p, "corpus": "main.npz;mv.npz" if ctxname == "main" else f"{ctxname}.npz",
                             "subsample": 0, "seed": 0, "role": role, "n_series": {"main": 3580, "short": 1034, "long": 620, "xlong": 48}[ctxname]})
    for d in fb_dirs:
        f = os.path.join(d, "runs.csv")
        if not os.path.exists(f):
            continue
        r = pd.read_csv(f)
        for x in r.itertuples(index=False):
            if x.role not in ("reference", "series_count") or x.library != "full":
                continue
            p = os.path.join(d, x.tag, "scores_partial.csv")
            if os.path.exists(p):
                rows.append({"tag": x.tag, "size": x.size, "path": p, "corpus": x.corpus_file, "subsample": int(x.subsample), "seed": int(x.seed),
                             "role": "full_main" if x.role == "reference" else "subsample", "n_series": int(x.n_series)})
    if plan and os.path.exists(plan):                     # runs of this follow-up: flat folders nn2c_<tag>/scores_partial.csv
        pl = pd.read_csv(plan)
        for d in flat_dirs:
            for x in pl.itertuples(index=False):
                p = os.path.join(d, f"nn2c_{x.tag}", "scores_partial.csv")
                if os.path.exists(p):
                    rows.append({"tag": x.tag, "size": x.size, "path": p, "corpus": x.corpus_file, "subsample": int(x.subsample), "seed": int(x.seed),
                                 "role": "full_main" if x.role == "reference" else "subsample", "n_series": int(x.n_series)})
    m = pd.DataFrame(rows).drop_duplicates("tag", keep="last")
    return m


def run_values(corpus_dir: str, corpus: str, subsample: int, seed: int) -> np.ndarray:
    """Series values (n, T + 128) of a run, with the subsample rule of scripts/neurons_run.py."""
    vals = []
    n_specs = 0
    for f in corpus.split(";"):
        c, specs = C.load_corpus(os.path.join(corpus_dir, f))
        vals.append(c["values"])
        n_specs += len(specs)
    v = np.concatenate(vals) if len(vals) > 1 else vals[0]
    if subsample and subsample < n_specs:
        keep = np.sort(np.random.default_rng(seed).choice(n_specs, subsample, replace=False))
        v = v[keep]
    return v


def parse_ids(col: pd.Series, k: int = 8) -> np.ndarray:
    return np.array([[int(t) for t in str(r).split()[:k]] if isinstance(r, str) else [0] * k for r in col], dtype=np.int64)


def patches(vals: np.ndarray, ids: np.ndarray, tok: int) -> tuple[np.ndarray, np.ndarray]:
    """Patches (n, k, 32) of token ids and the series index (n, k)."""
    s = ids // tok
    tp = ids % tok
    cols = (tp * M.PATCH)[..., None] + np.arange(M.PATCH)
    return vals[s[..., None], cols], s


class Run:
    def __init__(self, row, corpus_dir):
        self.row = row
        self.vals = run_values(corpus_dir, row["corpus"], int(row["subsample"]), int(row["seed"]))
        self.T = self.vals.shape[1] - 128
        self.L = self.T // M.PATCH
        self.tok = self.L + 4
        df = pd.read_csv(row["path"], usecols=lambda c: c in USECOLS)
        self.df = Sc.add_derived(df)
        self.live = ~self.df["dead"].to_numpy().astype(bool)
        self.top = parse_ids(self.df["top_tok"])
        self.mid = parse_ids(self.df["mid_tok"])
        assert self.top.max() < len(self.vals) * self.tok, "token id outside the run's corpus"
        self.cm = self.vals[:, :self.T].mean(1)
        self.cs = self.vals[:, :self.T].std(1)

    def normalized(self, ids: np.ndarray, how: str, chunk: int = 20000):
        """Yield normalized patches (m, k, 32) over chunks of poles."""
        for s0 in range(0, len(ids), chunk):
            P, ser = patches(self.vals, ids[s0:s0 + chunk], self.tok)
            yield s0, (M.shape_norm(P) if how == "shape" else M.scaled_norm(P, self.cm[ser], self.cs[ser]))

    def clusters(self, ids: np.ndarray, centers: np.ndarray, how: str) -> np.ndarray:
        out = np.empty(ids.shape, np.int16)
        for s0, Z in self.normalized(ids, how):
            out[s0:s0 + len(Z)] = M.assign(Z.reshape(-1, M.PATCH), centers).reshape(Z.shape[:2])
        return out

    def sample_patches(self, ids: np.ndarray, how: str, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        live_ix = np.where(self.live)[0]
        take = rng.choice(live_ix, min(len(live_ix), max(1, n // ids.shape[1])), replace=False)
        P, ser = patches(self.vals, ids[take], self.tok)
        Z = M.shape_norm(P) if how == "shape" else M.scaled_norm(P, self.cm[ser], self.cs[ser])
        return Z.reshape(-1, M.PATCH)


# ---------------------------------------------------------------------------------------------
def cohesion_stats(top_ids, mid_ids, live, K, rng) -> dict:
    tm, _ = M.majority_count(top_ids[live], K)
    mm, _ = M.majority_count(mid_ids[live], K)
    out = {"top6": float((tm >= 6).mean()), "top5": float((tm >= 5).mean()), "top8": float((tm >= 8).mean()), "mid6": float((mm >= 6).mean())}
    return out, tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--fb", action="append", default=[])
    ap.add_argument("--flat", action="append", default=[], help="folder holding nn2c_<tag>/scores_partial.csv runs listed in --plan")
    ap.add_argument("--plan", default="results/neurons/followup/plan_c.csv")
    ap.add_argument("--corpus-dir", default="results/neurons/corpus")
    ap.add_argument("--out", default="results/neurons/followup")
    ap.add_argument("--n-fit", type=int, default=500_000)
    ap.add_argument("--sizes", nargs="*", default=list(SIZES))
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    man = build_manifest(a.atlas, a.fb, a.flat, a.plan)
    man = man[man["size"].isin(a.sizes)].reset_index(drop=True)
    print(man[["tag", "size", "role", "n_series", "subsample", "seed"]].to_string(), flush=True)

    # ---- vocabularies from the atlas main-corpus runs ----
    fit = {"shape": [], "scaled": []}
    for r in man[man.role == "atlas_main"].itertuples():
        run = Run(r._asdict(), a.corpus_dir)
        for how in fit:
            fit[how].append(run.sample_patches(run.top, how, a.n_fit // len(SIZES), seed=0))
        del run
    vocab = {}
    for how in fit:
        Zf = np.concatenate(fit[how])
        for K in (KS_VOCAB if how == "shape" else (K_MAIN,)):
            vocab[(how, K)] = M.fit_vocabulary(Zf, K, seed=0, n_fit=a.n_fit)
        if how == "shape":
            Zshape = Zf
    print(f"[{time.time() - t0:.0f}s] vocabularies fitted ({len(Zshape)} patches)", flush=True)
    cen = vocab[("shape", K_MAIN)]
    ids_f = M.assign(Zshape, cen)
    names = M.name_clusters(Zshape, ids_f, K_MAIN)
    for n_, cn in zip(names, M.classify(M.shape_norm(cen))):      # descriptive extra: the rule applied to the cluster mean
        n_["centroid_name"] = str(cn)
    pd.DataFrame(names).to_csv(os.path.join(a.out, "motif_cluster_names.csv"), index=False, float_format="%.4g")
    pd.DataFrame(cen).to_csv(os.path.join(a.out, "motif_vocabulary_K16.csv"), index=False, float_format="%.4g", header=[f"t{i}" for i in range(M.PATCH)])
    cname = {n["cluster"]: n["name"] for n in names}

    # ---- every run ----
    rows, byname = [], []
    for r in man.itertuples():
        run = Run(r._asdict(), a.corpus_dir)
        live = run.live
        rng = np.random.default_rng(0)
        row = {"tag": r.tag, "size": r.size, "role": r.role, "corpus": r.corpus, "n_series": len(run.vals), "subsample": r.subsample,
               "seed": r.seed, "n_poles": len(live), "n_live": int(live.sum())}
        for theta in (0.5, 0.7, 0.9):
            cls = Sc.klass(run.df, theta, 0.1)
            row[f"one_job_{theta}"] = float((cls[live] == "one_job").mean())
            if theta == 0.7:
                one = cls == "one_job"
        for K in KS_VOCAB:
            cen_k = vocab[("shape", K)]
            tid = run.clusters(run.top, cen_k, "shape")
            mid = run.clusters(run.mid, cen_k, "shape")
            st, tm = cohesion_stats(tid, mid, live, K, rng)
            suf = "" if K == K_MAIN else f"_K{K}"
            row[f"cohesive_top6{suf}"] = st["top6"]
            row[f"cohesive_mid6{suf}"] = st["mid6"]
            if K == K_MAIN:
                row["cohesive_top5"], row["cohesive_top8"] = st["top5"], st["top8"]
                flat = tid[live].reshape(-1)
                nulls = []
                for _ in range(N_PERM):
                    perm = rng.permutation(flat).reshape(-1, tid.shape[1])
                    m_, _ = M.majority_count(perm, K)
                    nulls.append((m_ >= 6).mean())
                row["null_top6_mean"], row["null_top6_sd"] = float(np.mean(nulls)), float(np.std(nulls))
                coh = np.zeros(len(live), bool)
                maj, mcl = M.majority_count(tid, K)
                coh = (maj >= 6) & live
                row["frac_cohesive_among_one_job"] = float(coh[one & live].mean()) if (one & live).any() else np.nan
                row["frac_one_job_among_cohesive"] = float(one[coh].mean()) if coh.any() else np.nan
                row["both_known_and_motif"] = float((coh & one)[live].mean())
                row["motif_only"] = float((coh & ~one)[live].mean())
                row["known_only"] = float((~coh & one)[live].mean())
                row["neither"] = float((~coh & ~one)[live].mean())
                tc = (run.top % run.tok) < run.L
                allctx = tc.all(1) & live
                row["all_top_context_tokens"] = float(allctx.sum() / live.sum())
                row["cohesive_top6_context_only_poles"] = float((maj[allctx] >= 6).mean()) if allctx.any() else np.nan
                row["cluster_share_max"] = float(np.bincount(tid[live].reshape(-1), minlength=K).max() / tid[live].size)
                nm = pd.Series([cname[k] for k in mcl[coh]]).value_counts()
                for k_, v_ in nm.items():
                    byname.append({"tag": r.tag, "size": r.size, "role": r.role, "n_series": len(run.vals), "name": k_, "n_poles": int(v_),
                                   "share_of_cohesive": float(v_ / coh.sum()), "share_of_live": float(v_ / live.sum())})
        # scaled normalization (K=16) and a vocabulary refitted inside the run (shape, K=16)
        cen_s = vocab[("scaled", K_MAIN)]
        sid = run.clusters(run.top, cen_s, "scaled")
        m_, _ = M.majority_count(sid[live], K_MAIN)
        row["cohesive_top6_scaled"] = float((m_ >= 6).mean())
        own = M.fit_vocabulary(run.sample_patches(run.top, "shape", 200_000, seed=1), K_MAIN, seed=0, n_fit=200_000)
        rid = run.clusters(run.top, own, "shape")
        m_, _ = M.majority_count(rid[live], K_MAIN)
        row["cohesive_top6_refit"] = float((m_ >= 6).mean())
        rows.append(row)
        print(f"[{time.time() - t0:.0f}s] {r.tag}: cohesive {row['cohesive_top6']:.4f} null {row['null_top6_mean']:.4f} one_job {row['one_job_0.7']:.4f}", flush=True)
        del run
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "motif_clusters.csv"), index=False, float_format="%.5g")
    pd.DataFrame(byname).to_csv(os.path.join(a.out, "motif_by_name.csv"), index=False, float_format="%.5g")
    write_json(os.path.join(a.out, "provenance_motifs.json"), provenance("scripts/neurons_followup_c.py", argv=sys.argv, wall_s=round(time.time() - t0, 1),
                                                                         n_runs=len(rows), vocab_patches=int(len(Zshape)), K_main=K_MAIN))
    print("done", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
