"""Follow-up A: positive control by direction steering and group steering (pre-registered in
results/neurons/steering_preregistration_followup.md).

    python scripts/neurons_followup_a.py --size 5M --poles results/neurons/5M/poles.csv \
        --corpus results/neurons/corpus/main.npz --out results/colab_out/nn2_a_5M --checkpoint-every 20 --resume

Writes into --out: `readouts.npz` and `readouts.csv` (ridge directions and their R^2), `scores_partial.csv` (one row per
steered object and test set; appended after every `--checkpoint-every` tasks; with `--resume` finished tasks are
skipped) and `meta.json`. `scripts/neurons_followup_report.py` turns the partial files into the committed tables.

Task kinds (column `family` / `role`): dir/real (a readout direction of a known variable added to the residual stream at
a block and site), dir/random (random directions of the same norm), group/labeled (top-k poles of one label scaled
together), group/random (matched random groups of poles).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from otp.io import provenance, write_json  # noqa: E402
from otp.model import default_device, set_threads  # noqa: E402
from otp.neurons import capture as Cap  # noqa: E402
from otp.neurons import corpus as C  # noqa: E402
from otp.neurons import followup as F  # noqa: E402
from otp.neurons import steer as St  # noqa: E402

SCALES = St.SCALES


def zero_effs(effs_like: dict) -> dict:
    return {k: np.zeros_like(v) for k, v in effs_like.items()}


def judged_row(base: dict, effs: list, scales, quantity: str, direction: str, sign_corr: float, seed: int) -> dict:
    j = F.judge_scales(effs, scales, quantity, direction, sign_corr, seed=seed)
    row = {"quantity": j["quantity"], "predicted": quantity, "direction": direction, "rho": j.get("rho"),
           "effect_size": j.get("size"), "ci_lo": j.get("ci_lo"), "ci_hi": j.get("ci_hi"), "monotone": j.get("monotone"),
           "big": j.get("big"), "sig": j.get("sig"), "match": j.get("match")}
    for i, v in enumerate(j.get("e", [])):
        row[f"e{i}"] = v
    row.update(F.all_D(effs))
    return row


class Runner:
    def __init__(self, a):
        self.a = a
        self.dev = a.device or default_device()
        set_threads()
        self.c, self.specs = C.load_corpus(a.corpus)
        self.poles = pd.read_csv(a.poles)
        self.n_blocks, self.d_hidden = Cap.layer_dims(a.size, self.dev)
        self.blocks = F.blocks_for(self.n_blocks)
        self.sets = dict(zip(("mixed", "seasonal"), F.test_sets(self.c)))
        self.sets["jump"] = F.jump_set(self.c, self.specs)
        self.base, self.basem = {}, {}
        for name, s in self.sets.items():
            self.base[name] = St.run_conditions(a.size, s["ctx"], [[]], device=self.dev)[0]
            self.basem[name] = St.measures(s["ctx"], self.base[name], s["period"])

    # ---- readouts ----
    def readouts(self, out):
        path = os.path.join(out, "readouts.npz")
        if os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            fits = {}
            for k in z["keys"]:
                var, bname, ts = str(k).split("|")
                fits[(var, bname, ts)] = {"w": z[f"w|{k}"], "b0": float(z[f"b0|{k}"]), "alpha": float(z[f"alpha|{k}"]),
                                          "r2_cv": float(z[f"r2_cv|{k}"]), "r2_heldout": float(z[f"r2_heldout|{k}"])}
            print("readouts loaded", len(fits), flush=True)
            return fits
        t0 = time.time()
        stats = F.collect_stats(self.a.size, self.c, self.specs, self.blocks, device=self.dev, batch=self.a.batch)
        fits = F.fit_all_directions(stats, self.blocks)
        arrs = {"keys": np.array(["|".join(k) for k in fits])}
        rows = []
        for k, f in fits.items():
            kk = "|".join(k)
            arrs[f"w|{kk}"] = f["w"]
            arrs[f"b0|{kk}"] = np.array(f["b0"])
            for n in ("alpha", "r2_cv", "r2_heldout"):
                arrs[f"{n}|{kk}"] = np.array(f[n])
            rows.append({"size": self.a.size, "variable": k[0], "block_name": k[1], "block": self.blocks[k[1]], "tokens": k[2],
                         "alpha": f["alpha"], "r2_cv": f["r2_cv"], "r2_train": f["r2_train"], "r2_heldout": f["r2_heldout"],
                         "n_train": f["n_train"], "n_heldout": f["n_heldout"], "w_norm": float(np.linalg.norm(f["w"]))})
        np.savez(path, **arrs)
        pd.DataFrame(rows).to_csv(os.path.join(out, "readouts.csv"), index=False, float_format="%.5g")
        print(f"readouts fitted in {time.time() - t0:.0f}s", flush=True)
        return fits

    # ---- direction tasks ----
    def dir_sets(self, var):
        return ("jump",) if var == "postjump" else ("mixed", "seasonal")

    def fit_for(self, fits, var, bname, site, kind):
        ts = "pooled" if kind == "pooled" else ("fc" if site == "F" else "ctx")
        return fits.get((var, bname, ts))

    def dir_real(self, fits, var, bname, site, kind):
        f = self.fit_for(fits, var, bname, site, kind)
        if f is None:
            return []
        blk = self.blocks[bname]
        moves = list(F.MOVES[1:]) + list(F.NEG_MOVES)
        conds = [(blk, site, F.steer_vector(f["w"], m)) for m in moves]
        rows = []
        for sname in self.dir_sets(var):
            s = self.sets[sname]
            if len(s["ctx"]) == 0:
                continue
            q = F.run_add_conditions(self.a.size, s["ctx"], conds, device=self.dev)
            eff = {m: St.effects(self.basem[sname], St.measures(s["ctx"], q[i], s["period"])) for i, m in enumerate(moves)}
            effs = [zero_effs(eff[0.5])] + [eff[m] for m in F.MOVES[1:]]
            pq = F.PREDICTED[var]
            r = judged_row(self.basem[sname], effs, F.MOVES, pq, "+", 1.0, seed=F.seed_of("dirjudge", var, bname, site, kind))
            r["gain"] = float(r["e3"] / F.MOVES[-1]) if pq in effs[0] else np.nan
            for m in F.NEG_MOVES:
                r[{2.0: 'em2', 1.0: 'em1', 0.5: 'em05'}[abs(m)]] = F.mean_effect(pq, eff[m][pq])
            for qn in eff[2.0]:
                r[f"E2_{qn}"] = F.mean_effect(qn, eff[2.0][qn])
            r.update({"size": self.a.size, "family": "dir", "role": "real", "variable": var, "kind": kind, "block_name": bname,
                      "block": blk, "site": site, "test_set": sname, "n_series": len(s["ctx"]), "readout_r2_heldout": f["r2_heldout"],
                      "rep": -1})
            rows.append(r)
        return rows

    def dir_random(self, fits, var, bname, site, kind):
        f = self.fit_for(fits, var, bname, site, kind)
        if f is None:
            return []
        blk = self.blocks[bname]
        sname = self.dir_sets(var)[0]
        s = self.sets[sname]
        if len(s["ctx"]) == 0:
            return []
        d = len(f["w"])
        unit = 1.0 / float(np.linalg.norm(f["w"]))
        conds, meta = [], []
        for j in range(self.a.n_random_dir):
            r = np.random.default_rng(F.seed_of(self.a.size, var, bname, site, kind, j)).standard_normal(d)
            r /= np.linalg.norm(r)
            for m in F.MOVES[1:]:
                conds.append((blk, site, (m * unit * r).astype(np.float32)))
                meta.append((j, m))
        q = F.run_add_conditions(self.a.size, s["ctx"], conds, device=self.dev)
        rows = []
        for j in range(self.a.n_random_dir):
            effs = [zero_effs(St.effects(self.basem[sname], St.measures(s["ctx"], q[0], s["period"])))]
            for i, (jj, m) in enumerate(meta):
                if jj == j:
                    effs.append(St.effects(self.basem[sname], St.measures(s["ctx"], q[i], s["period"])))
            pq = F.PREDICTED[var]
            r = judged_row(self.basem[sname], effs, F.MOVES, pq, "+", 1.0, seed=F.seed_of("dirrandjudge", var, bname, site, kind, j))
            r.update({"size": self.a.size, "family": "dir", "role": "random", "variable": var, "kind": kind, "block_name": bname,
                      "block": blk, "site": site, "test_set": sname, "n_series": len(s["ctx"]), "rep": j})
            rows.append(r)
        return rows

    # ---- group tasks ----
    def group_task(self, g):
        live = self.poles[~self.poles["dead"].astype(bool)]
        qn = g["quantity"]
        first = qn.split("|")[0]
        sname = "seasonal" if first in ("amp", "phase") else "mixed"
        s = self.sets[sname]
        members = [("labeled", -1, g["poles"])]
        for j in range(self.a.n_random_group):
            members.append(("random", j, F.random_group(live, g["poles"], F.seed_of(self.a.size, g["gid"], j))))
        conds = []
        for _, _, pl in members:
            plan = F.poles_plan(pl)
            conds += [(plan, sc) for sc in SCALES]
        q = F.run_group_conditions(self.a.size, s["ctx"], conds, device=self.dev)
        rows = []
        for mi, (role, j, pl) in enumerate(members):
            effs = [St.effects(self.basem[sname], St.measures(s["ctx"], q[mi * len(SCALES) + k], s["period"])) for k in range(len(SCALES))]
            r = judged_row(self.basem[sname], effs, SCALES, qn, g["direction"], float(g["sign"]), seed=F.seed_of("grpjudge", g["gid"], j))
            r.update({"size": self.a.size, "family": "group", "role": role, "gid": g["gid"], "pool": g["pool"], "label": g["label"],
                      "sign": g["sign"], "k": g["k"], "rep": j, "test_set": sname, "n_series": len(s["ctx"]), "n_poles": len(pl),
                      "mean_r2_cv": float(g["poles"]["r2_cv"].mean()) if role == "labeled" else np.nan})
            rows.append(r)
        return rows

    # ---- task list ----
    def tasks(self, fits):
        out = []
        if self.a.only in ("all", "dir"):
            for var in F.VARIABLES:
                for bname in self.blocks:
                    for site in F.SITES:
                        for kind in ("pooled", "matched"):
                            out.append((f"dir/real/{var}/{bname}/{site}/{kind}", lambda v=var, b=bname, s=site, k=kind: self.dir_real(fits, v, b, s, k)))
            for var in F.VARIABLES:
                for bname in self.blocks:
                    for site in F.SITES:
                        for kind in ("pooled", "matched"):
                            out.append((f"dir/random/{var}/{bname}/{site}/{kind}", lambda v=var, b=bname, s=site, k=kind: self.dir_random(fits, v, b, s, k)))
        if self.a.only in ("all", "group"):
            groups = F.make_groups(self.poles)
            groups.sort(key=lambda g: (g["k"], g["pool"] != "one_job", g["label"], g["sign"]))
            for g in groups:
                out.append((f"group/{g['gid']}", lambda g=g: self.group_task(g)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--poles", required=True, help="results/neurons/<size>/poles.csv")
    ap.add_argument("--corpus", default="results/neurons/corpus/main.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", default="all", choices=["all", "dir", "group"])
    ap.add_argument("--n-random-dir", type=int, default=F.N_RANDOM_DIR)
    ap.add_argument("--n-random-group", type=int, default=F.N_RANDOM_GROUP)
    ap.add_argument("--max-tasks", type=int, default=0, help="stop after n tasks (tests)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    R = Runner(a)
    part = os.path.join(a.out, "scores_partial.csv")
    done = set()
    if a.resume and os.path.exists(part):
        done = set(pd.read_csv(part, usecols=["task_id"]).task_id.unique())
        print("resume: tasks done", len(done), flush=True)
    fits = R.readouts(a.out)
    tasks = [(tid, fn) for tid, fn in R.tasks(fits) if tid not in done]
    if a.max_tasks:
        tasks = tasks[:a.max_tasks]
    print(f"{a.size}: {len(tasks)} tasks to run (blocks {R.blocks})", flush=True)
    buf = []

    def flush():
        if buf:
            df = pd.concat([pd.DataFrame(b) for b in buf], ignore_index=True)
            df.to_csv(part, mode="a", header=not os.path.exists(part), index=False, float_format="%.6g")
            buf.clear()

    for i, (tid, fn) in enumerate(tasks):
        rows = fn()
        for r in rows:
            r["task_id"] = tid
        buf.append(rows)
        if len(buf) >= a.checkpoint_every:
            flush()
        if i % 10 == 0:
            print(f"[{time.time() - t0:.0f}s] task {i + 1}/{len(tasks)} {tid}", flush=True)
    flush()
    write_json(os.path.join(a.out, "meta.json"), provenance(
        "scripts/neurons_followup_a.py", size=a.size, device=R.dev, wall_s=round(time.time() - t0, 1), blocks=R.blocks,
        n_blocks=R.n_blocks, d_hidden=R.d_hidden, n_series={k: len(v["ctx"]) for k, v in R.sets.items()},
        n_random_dir=a.n_random_dir, n_random_group=a.n_random_group, n_tasks_run=len(tasks), only=a.only))
    print("done", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
