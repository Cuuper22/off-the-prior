"""Fine-tune Timestral on failure families: full fine-tune, LoRA, and data mixing.

Two sub-commands.

`run` trains one model (one size, one protocol / held-out family, one method, one replay share)
and writes

    <runs-root>/<run_id>/run.json     config, cost record (trainable params, steps, time steps seen,
                                      wall time, peak memory), final loss
    <runs-root>/<run_id>/loss.csv     per-step training loss
    <ckpt-root>/<run_id>/             checkpoint loadable by `otp.model.forecast(size=<dir>)`

`suite` loops `run` over sizes x protocols x held-out families x methods, one subprocess per
run (so the peak-memory number of each run is its own), and skips runs whose `run.json` exists.

Run id: `<size>__<protocol>-<holdout>__<method>__<nomix|mix<frac>>`.

Methods: `full` (all parameters), `lora` (rank-r adapters on the attention and MLP linears, base
weights frozen). A replay share above 0 mixes in-prior series into every batch ("mixing").

Examples:
  OTP_THREADS=1 python scripts/finetune.py run --size 5M --protocol lofo --holdout chaos --method lora \
      --replay-frac 0.3 --steps 40 --batch-size 8
  python scripts/finetune.py suite --sizes 5M --steps 40 --batch-size 8 --replay-fracs 0 0.3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from otp import finetune_ext as X  # noqa: E402
from otp import finetune_splits as S  # noqa: E402
from otp.generators.primitives import in_prior  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402

DEFAULT_LR = {"full": 5e-5, "lora": 1e-3}


def cmd_run(a) -> None:
    import torch

    from otp import finetune as F
    from otp.model import default_device, set_threads

    set_threads(a.threads)
    device = default_device()
    rid = X.run_id(a.size, a.protocol, a.holdout, a.method, a.replay_frac)
    rdir = os.path.join(a.runs_root, rid)
    ckdir = os.path.join(a.ckpt_root, rid)
    os.makedirs(rdir, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    contexts, horizons = tuple(a.contexts), tuple(a.horizons)
    t0 = time.time()
    n_pool = a.pool_batches or a.steps  # pools hold `n_pool` batches; a run uses the first `steps` of them
    sched = X.schedule(n_pool, a.seed, contexts, horizons)
    pool_bs = max(a.batch_size, a.pool_batch_size or 0)  # a pool made for a larger batch also serves smaller ones
    rows_f = -(-pool_bs // len(S.train_families(a.protocol, a.holdout)))  # ceil: rows one family can supply
    fam_pools, pool_files, pool_src = [], [], {}
    for fam in S.train_families(a.protocol, a.holdout):
        pool, src, path = X.get_pool(a.pool_dir, f"{a.protocol}-{fam}", S.family_train_factory(a.protocol, fam),
                                     sched, rows_f, a.seed, contexts, horizons, workers=a.pool_workers, log=X.log_stderr)
        fam_pools.append(pool)
        pool_files.append(os.path.basename(path))
        pool_src[fam] = src
    rep_pool, src, path = X.get_pool(a.pool_dir, "replay", in_prior, sched, pool_bs, a.seed, contexts, horizons,
                                     workers=a.pool_workers, log=X.log_stderr)
    pool_files.append(os.path.basename(path))
    pool_src["replay"] = src
    pool_sec = round(time.time() - t0, 1)

    fc = F.load_trainable(a.size, device)
    tr0, total = X.count_params(fc.model)
    if a.method == "lora":
        F.add_lora(fc.model, rank=a.rank, alpha=a.alpha)
    elif a.method != "full":
        raise SystemExit(f"unknown method {a.method}")
    trainable, total_with_adapters = X.count_params(fc.model)
    lr = a.lr if a.lr else DEFAULT_LR[a.method]
    sampler = X.CompositeSampler(fam_pools, rep_pool, a.replay_frac)
    rec = F.train(fc, sampler, steps=a.steps, batch_size=a.batch_size, lr=lr, warmup=a.warmup,
                  device=device, log=X.log_stderr, log_every=a.log_every)
    F.export(fc, ckdir)
    losses = rec.pop("loss_curve")
    with open(os.path.join(rdir, "loss.csv"), "w") as f:
        f.write("step,loss\n" + "".join(f"{i + 1},{v}\n" for i, v in enumerate(losses)))
    info = X.parse_run_id(rid)
    info.update({
        "trainable_params": trainable, "base_params": total, "params_with_adapters": total_with_adapters,
        "rank": a.rank if a.method == "lora" else None, "alpha": a.alpha if a.method == "lora" else None,
        "contexts": list(contexts), "horizons": list(horizons), "seed": a.seed, "warmup": a.warmup,
        "pool": {"source": pool_src, "files": pool_files, "sec": pool_sec},
        "train": {k: v for k, v in rec.items() if k != "trainable_params"},
        "peak_rss_bytes": X.peak_rss_bytes(), "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": os.path.relpath(ckdir, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "provenance": provenance("scripts/finetune.py", argv=sys.argv[1:]),
    })
    write_json(os.path.join(rdir, "run.json"), info)
    X.log_stderr(f"done {rid}: trainable {trainable}, wall {rec['wall_sec']} s")


def cmd_suite(a) -> None:
    here = os.path.abspath(__file__)
    protocols = a.protocols
    for size in a.sizes:
        for protocol in protocols:
            for holdout in a.holdouts:
                for method in a.methods:
                    for rf in a.replay_fracs:
                        rid = X.run_id(size, protocol, holdout, method, rf)
                        if os.path.exists(os.path.join(a.runs_root, rid, "run.json")):
                            print("skip", rid, flush=True)
                            continue
                        cmd = [sys.executable, here, "run", "--size", size, "--protocol", protocol,
                               "--holdout", holdout, "--method", method, "--replay-frac", str(rf),
                               "--steps", str(a.steps), "--batch-size", str(a.batch_size),
                               "--runs-root", a.runs_root, "--ckpt-root", a.ckpt_root, "--pool-dir", a.pool_dir,
                               "--rank", str(a.rank), "--alpha", str(a.alpha), "--seed", str(a.seed),
                               "--warmup", str(a.warmup), "--log-every", str(a.log_every),
                               "--pool-workers", str(a.pool_workers),
                               *(["--pool-batch-size", str(a.pool_batch_size)] if a.pool_batch_size else []),
                               "--contexts", *map(str, a.contexts), "--horizons", *map(str, a.horizons)]
                        if a.pool_batches:
                            cmd += ["--pool-batches", str(a.pool_batches)]
                        if a.lr_full and method == "full":
                            cmd += ["--lr", str(a.lr_full)]
                        if a.lr_lora and method == "lora":
                            cmd += ["--lr", str(a.lr_lora)]
                        if a.threads:
                            cmd += ["--threads", str(a.threads)]
                        print("run", rid, flush=True)
                        r = subprocess.run(cmd)
                        if r.returncode != 0:
                            raise SystemExit(f"run {rid} failed with exit code {r.returncode}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--runs-root", default="results/finetune/local/runs")
        p.add_argument("--ckpt-root", default="checkpoints/finetune/local", help="checkpoints (large; not for git)")
        p.add_argument("--pool-dir", default="results/finetune/local/pools", help="cached training batches (not for git)")
        p.add_argument("--steps", type=int, default=40)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--rank", type=int, default=8)
        p.add_argument("--alpha", type=float, default=16.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--warmup", type=int, default=5)
        p.add_argument("--log-every", type=int, default=10)
        p.add_argument("--threads", type=int, default=None)
        p.add_argument("--pool-batches", type=int, default=None, help="batches per cached pool (default: steps); a run uses the first `steps`")
        p.add_argument("--pool-batch-size", type=int, default=None,
                       help="batch size the cached pools are built for (default: --batch-size); sizes with smaller batches share the pools")
        p.add_argument("--pool-workers", type=int, default=1, help="processes for simulating training series")
        p.add_argument("--contexts", type=int, nargs="+", default=[256, 512, 1024, 2048])
        p.add_argument("--horizons", type=int, nargs="+", default=[32, 128, 512])

    r = sub.add_parser("run", help="train one model")
    common(r)
    r.add_argument("--size", required=True, choices=["5M", "28M", "123M", "209M"])
    r.add_argument("--protocol", required=True, choices=list(S.PROTOCOLS))
    r.add_argument("--holdout", required=True, choices=list(S.FAMILIES))
    r.add_argument("--method", required=True, choices=["full", "lora"])
    r.add_argument("--replay-frac", type=float, default=0.0)
    r.add_argument("--lr", type=float, default=None, help=f"default {DEFAULT_LR}")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("suite", help="loop `run` over sizes x protocols x held-out families x methods")
    common(s)
    s.add_argument("--sizes", nargs="+", default=["5M"])
    s.add_argument("--holdouts", nargs="+", default=list(S.FAMILIES), choices=list(S.FAMILIES))
    s.add_argument("--protocols", nargs="+", default=list(S.PROTOCOLS), choices=list(S.PROTOCOLS))
    s.add_argument("--methods", nargs="+", default=["full", "lora"], choices=["full", "lora"])
    s.add_argument("--replay-fracs", type=float, nargs="+", default=[0.0, 0.3])
    s.add_argument("--lr-full", type=float, default=None)
    s.add_argument("--lr-lora", type=float, default=None)
    s.set_defaults(fn=cmd_suite)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
