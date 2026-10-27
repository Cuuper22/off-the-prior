"""Export what the live toy needs to run Timestral-5M in the browser.

    python scripts/export_site_weights.py                       # fp16 weights into docs/live/
    python scripts/export_site_weights.py --dtype f32 --out D   # fp32 copy for the parity test

Writes, next to each other in the output folder:

  timestral5m.<f16|f32>   every tensor of the checkpoint, flat little-endian, in the order of timestral5m.json
  timestral5m.json        config (from the checkpoint), tensor table (name, shape, offset in elements), dtype, source
  live_spec.json          the toy's generator pieces as resolved by otp/generators/mixture.py: parameters of the
                          season, AR-noise and wild pieces, their pilot mean and std (so the browser standardizes
                          them exactly like the Python mixture does), season amplitudes and phases, Kuramoto
                          natural frequencies. The browser cannot reproduce numpy's random streams, so the values
                          that numpy drew once from a fixed seed are exported instead.

The forward pass is written by hand in docs/live/timestral.js; tests/test_site_parity.py checks it against
otp.model.forecast (symmetric averaging on).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from otp import model as otp_model  # noqa: E402
from otp.generators import mixture as M  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402

SIZE = "5M"


def export_weights(out_dir: str, dtype: str = "f16") -> dict:
    """Write the weights file and its table. Returns the table (config + tensors)."""
    fc = otp_model.load(SIZE)
    sd = fc.model.state_dict()
    np_dtype = {"f16": np.float16, "f32": np.float32}[dtype]
    tensors, chunks, offset = [], [], 0
    for name, t in sd.items():
        a = t.detach().cpu().numpy().astype(np.float32)
        tensors.append({"name": name, "shape": list(a.shape), "offset": offset})
        chunks.append(a.astype(np_dtype).ravel())
        offset += a.size
    os.makedirs(out_dir, exist_ok=True)
    flat = np.concatenate(chunks).astype("<" + np.dtype(np_dtype).str[1:])
    bin_name = f"timestral5m.{dtype}"
    with open(os.path.join(out_dir, bin_name), "wb") as f:
        f.write(flat.tobytes())
    table = {"model": f"Timestral-{SIZE}", "source": otp_model.repo_id(SIZE), "dtype": dtype, "file": bin_name,
             "elements": int(offset), "config": fc.config.to_dict(), "tensors": tensors,
             "provenance": provenance(script="scripts/export_site_weights.py")}
    write_json(os.path.join(out_dir, "timestral5m.json"), table)
    return table


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def _piece(slot, proc, stats) -> dict:
    out = {"component": slot.component.name, "params": {k: _clean(v) for k, v in proc.params.items()},
           "pilot": {"mean": stats[0], "sd": stats[1]}, "period": int(proc.period)}
    name = slot.component.name
    if name == "season":
        proc.init_state(np.random.default_rng(0), 1)         # draws the amplitudes and phases from phase_seed
        out["amps"] = [float(x) for x in proc._amps]
        out["phases"] = [float(x) for x in proc._phases]
    if name == "kuramoto":
        out["omega"] = [float(x) for x in proc.omega]
    if name in ("mean_jump", "variance_jump"):
        out["level_shift"] = float(proc.level_shift())
        out["sigma_mult"] = float(proc.sigma_mult())
        out["amps"] = [float(x) for x in proc._amps]
        out["phases"] = [float(x) for x in proc._phases]
    if hasattr(proc, "sd_ref"):
        out["sd_ref"] = float(proc.sd_ref)                      # reference scale of the chaos generators' noise
    if hasattr(proc, "warmup_steps"):
        out["warmup_steps"] = int(proc.warmup_steps())
    return out


def export_spec(out_dir: str) -> dict:
    import site_toy as S

    pieces = {}
    mix = M.decode(S.genome(1.0, "none", S.STRENGTH[0], 0.1))
    pieces["season"] = _piece(mix.slots[0], mix.procs[0], mix.stats[0])
    pieces["ar_noise"] = _piece(mix.slots[1], mix.procs[1], mix.stats[1])
    wild = {}
    for w in S.WILD:
        if w == "none":
            continue
        mix = M.decode(S.genome(1.0, w, 1.0, 0.1))
        slot = mix.slots[2]
        assert slot.component.name == w, (slot.component.name, w)
        wild[w] = _piece(slot, mix.procs[2], mix.stats[2])
    spec = {"T": S.T, "H": S.H, "controls": {"season": S.SEASON, "wild": S.WILD, "strength": S.STRENGTH,
                                             "noise": S.NOISE},
            "ar_noise_weight": 0.3, "pieces": pieces, "wild": wild,
            "note": "y = season_w*z_season + 0.3*z_ar + strength*z_wild + noise*e, each z standardized by its pilot "
                    "mean and sd (otp/generators/mixture.py). Period for scoring follows MixtureProcess.period.",
            "provenance": provenance(script="scripts/export_site_weights.py")}
    write_json(os.path.join(out_dir, "live_spec.json"), spec)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "live"))
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--no-spec", action="store_true", help="weights only (the parity test does not need the spec)")
    a = ap.parse_args()
    otp_model.set_threads(1)
    t = export_weights(a.out, a.dtype)
    size = os.path.getsize(os.path.join(a.out, t["file"]))
    print(f"wrote {t['file']}: {t['elements']} values, {size / 1e6:.2f} MB ({a.dtype})")
    if not a.no_spec:
        s = export_spec(a.out)
        print(f"wrote live_spec.json: {len(s['wild'])} wild pieces")


if __name__ == "__main__":
    main()
