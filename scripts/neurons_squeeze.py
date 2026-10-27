"""Squeeze test: a tiny Timestral-style model with a very narrow MLP and wider twins, trained the same way on the
synthetic in-prior series, then labeled with the same pipeline as the released models.

    python scripts/neurons_squeeze.py train --out results/colab_out/nn_squeeze --steps 3000
    python scripts/neurons_run.py --size results/colab_out/nn_squeeze/w4 --corpus results/neurons/corpus/main.npz \
        --out results/colab_out/nn_squeeze_w4_main        # and w32, w256

Both models: patch 32, d_model 96, 3 heads, 4 time-attention blocks, the shipped 21 quantile knots; they differ only in the
MLP hidden width (`--widths`, default 4 32 256; directories `w4`, `w32`, `w256`). Same initialization seed, same training batches (same
sampler seed), same steps, learning rate and schedule. Training data: `otp.generators.primitives.in_prior`.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from otp import finetune as FT  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.model import default_device, set_threads  # noqa: E402


def build(mlp: int, seed: int, device: str):
    from timestral import TimestralForecaster
    from timestral.config import TimestralConfig
    from timestral.model import TimestralModel

    cfg = TimestralConfig(patch_size=32, d_model=96, num_heads=3, mlp_hidden_dim=mlp, block_schedule=("time",) * 4,
                          quantile_knots=tuple(FT.QUANTILE_LEVELS), block_size=32, context_length=2048,
                          quantile_width_factor=3.0)
    torch.manual_seed(seed)
    model = TimestralModel(cfg).to(device)
    model.train()
    return TimestralForecaster(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--widths", type=int, nargs="+", default=[4, 32, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    set_threads()
    dev = a.device or default_device()
    for mlp in a.widths:
        name = f"w{mlp}"
        t0 = time.time()
        fc = build(mlp, a.seed, dev)
        n_par = sum(p.numel() for p in fc.model.parameters())
        sampler = FT.Sampler(factories=[], replay_frac=1.0, contexts=(256, 512), horizons=(32, 128), rng_seed=a.seed)
        rec = FT.train(fc, sampler, a.steps, batch_size=a.batch, lr=a.lr, warmup=100, log_every=100, device=dev)
        FT.export(fc, os.path.join(a.out, name))
        write_json(os.path.join(a.out, f"train_{name}.json"), provenance(
            "scripts/neurons_squeeze.py", mlp_hidden_dim=mlp, params=n_par, device=dev, wall_s=round(time.time() - t0, 1),
            **{k: v for k, v in rec.items() if k != "loss_curve"}, loss_curve=rec["loss_curve"][::10]))
        print(name, "params", n_par, "final loss", rec["final_loss"], flush=True)


if __name__ == "__main__":
    main()
