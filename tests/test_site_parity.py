"""Parity of the live toy's JavaScript against the Python it was ported from. Needs node on the PATH.

1. docs/live/timestral.js against otp.model.forecast (symmetric averaging on, non-negative clip on):
   - fp32 weights: agrees to a tiny fraction of the series' spread (float32 rounding only);
   - the fp16 weights shipped to the site: agrees to a small fraction (weight rounding).
2. docs/live/generators.js against otp/generators: the random streams differ, so statistics are compared.
   - each piece alone (season, AR noise and the eight wild pieces), standardized with the exported pilot mean and sd:
     quantiles and autocorrelations of one long run;
   - the whole mixture with the referee: spread of the context, width of the referee's 80% band at the last step,
     the naive-forecast scale and the referee's own MASE, averaged over seeds.

    python3 -m pytest -q tests/test_site_parity.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

LIVE = os.path.join(ROOT, "docs", "live")


def run_node(tmp_path, source: str, *args: str) -> str:
    p = tmp_path / "run.js"
    p.write_text(source)
    r = subprocess.run([NODE, str(p), *args], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    return r.stdout


# ---------------------------------------------------------------- forward pass

FORWARD_JS = r"""
const fs = require("fs");
const T = require(process.argv[2] + "/timestral.js");
const dir = process.argv[3], dtype = process.argv[4];
const table = JSON.parse(fs.readFileSync(dir + "/timestral5m.json"));
const b = fs.readFileSync(dir + "/timestral5m." + dtype);
const M = T.load(table, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
const cases = JSON.parse(fs.readFileSync(dir + "/cases.json"));
console.log(JSON.stringify(cases.map(c => Array.from(T.forecast(M, c.x, c.h).q))));
"""


@pytest.fixture(scope="module")
def forward(tmp_path_factory):
    from otp import model as otp_model
    import export_site_weights as E

    otp_model.set_threads(1)
    dirs = {"f32": tmp_path_factory.mktemp("w32"), "f16": tmp_path_factory.mktemp("w16")}
    for dtype, d in dirs.items():
        E.export_weights(str(d), dtype)
    rng = np.random.default_rng(3)
    t = np.arange(512)
    a = 3 + np.sin(t / 9.0) + 0.1 * rng.standard_normal(512)
    cases = [(a, 64),                                            # positive: the non-negative clip is active
             (a - 3.5, 64),                                      # has negative values
             (np.cumsum(rng.standard_normal(256)), 100),         # 256 steps, horizon of 4 patches cut to 100
             (a, 32),                                            # one patch: no refine frame
             (a * 100 + 500, 128)]                               # large level and scale
    cases = [(np.asarray(x, np.float32).astype(float), h) for x, h in cases]
    for d in dirs.values():
        (d / "cases.json").write_text(json.JSONEncoder().encode([{"x": [float(v) for v in x], "h": h} for x, h in cases]))
    ref = [otp_model.forecast("5M", x[None].astype(np.float32), h)[0, 0] for x, h in cases]
    out = {}
    for dtype in ("f32", "f16"):
        raw = run_node(dirs[dtype], FORWARD_JS, LIVE, str(dirs[dtype]), dtype)
        out[dtype] = [np.array(q).reshape(r.shape) for q, r in zip(json.loads(raw), ref)]
    return cases, ref, out


def _worst(cases, ref, got):
    """Largest |difference| over all quantiles and steps, and over the median, in units of the context's sd."""
    allq, med = 0.0, 0.0
    for (x, _), r, g in zip(cases, ref, got):
        sd = float(np.std(x))
        allq = max(allq, float(np.abs(r - g).max() / sd))
        med = max(med, float(np.abs(r[:, 10] - g[:, 10]).max() / sd))
    return allq, med


def test_forward_fp32_matches_python(forward):
    cases, ref, out = forward
    allq, med = _worst(cases, ref, out["f32"])
    assert allq < 1e-4, allq
    assert med < 1e-4, med


def test_forward_fp16_matches_python(forward):
    cases, ref, out = forward
    allq, med = _worst(cases, ref, out["f16"])
    assert allq < 2e-2, allq
    assert med < 5e-3, med


def test_forward_quantiles_ordered_and_clipped(forward):
    cases, _, out = forward
    for dtype in ("f32", "f16"):
        for (x, _), g in zip(cases, out[dtype]):
            assert np.all(np.diff(g, axis=1) >= -1e-6 * np.abs(g).max())
            if np.all(x >= 0):
                assert g.min() >= 0


# ---------------------------------------------------------------- generators

PIECES_JS = r"""
const G = require(process.argv[2] + "/generators.js");
const spec = JSON.parse(require("fs").readFileSync(process.argv[2] + "/live_spec.json"));
const N = +process.argv[3], seeds = +process.argv[4];
function stats(y) {
  const n = y.length, s = Array.from(y).sort((a, b) => a - b), q = p => s[Math.floor(p * (n - 1))];
  const m = y.reduce((a, b) => a + b, 0) / n; let v = 0; for (const x of y) v += (x - m) ** 2; v /= n;
  const ac = k => { let c = 0; for (let i = k; i < n; i++) c += (y[i] - m) * (y[i - k] - m); return c / (n - k) / v; };
  return { q05: q(.05), q25: q(.25), q50: q(.5), q75: q(.75), q95: q(.95), ac1: ac(1), ac5: ac(5), ac12: ac(12) };
}
const out = {};
for (const nm of ["season", "ar_noise", ...Object.keys(spec.wild)]) {
  const P = nm === "season" || nm === "ar_noise" ? spec.pieces[nm] : spec.wild[nm], acc = [];
  for (let sd = 0; sd < seeds; sd++) {
    const rng = new G.Rng(1000 + sd);
    const proc = nm === "season" ? G.pieces.seasonPiece(P) : nm === "ar_noise" ? G.pieces.arPiece(P) : G.pieces.wildPiece(nm, P);
    const st = proc.init(rng);
    if (proc.warmup) proc.step(st, proc.warmup, rng, new Float64Array(proc.warmup));
    const y = new Float64Array(N); proc.step(st, N, rng, y);
    acc.push(stats(y.map(v => (v - P.pilot.mean) / P.pilot.sd)));
  }
  out[nm] = {}; for (const k of Object.keys(acc[0])) out[nm][k] = acc.reduce((a, b) => a + b[k], 0) / acc.length;
}
console.log(JSON.stringify(out));
"""

MIX_JS = r"""
const G = require(process.argv[2] + "/generators.js");
const spec = JSON.parse(require("fs").readFileSync(process.argv[2] + "/live_spec.json"));
const cfgs = JSON.parse(process.argv[3]), seeds = +process.argv[4];
console.log(JSON.stringify(cfgs.map(c => {
  const acc = { sd: [], width: [], oracle_mase: [], scale: [] };
  for (let s = 0; s < seeds; s++) {
    const r = G.makeSample(spec, c, 5000 + s, { nPaths: 100 });
    const n = r.context.length, m = r.context.reduce((a, b) => a + b, 0) / n; let v = 0; for (const x of r.context) v += (x - m) ** 2;
    const sd = Math.sqrt(v / n);
    acc.sd.push(sd); acc.width.push((r.oracle[63 * 21 + 18] - r.oracle[63 * 21 + 2]) / sd);
    acc.oracle_mase.push(G.mase(r.context, r.future, r.oracle, r.period)); acc.scale.push(G.naiveScale(r.context, r.period) / sd);
  }
  const o = {}; for (const k in acc) o[k] = acc[k].reduce((a, b) => a + b, 0) / seeds; return o;
})));
"""

# tolerance per piece: (quantiles, autocorrelations), in standardized units
PIECE_TOL = {"lorenz63": (0.2, 0.1), "stable_ar": (0.4, 0.1), "hawkes": (0.3, 0.2)}
DEFAULT_TOL = (0.1, 0.05)


def test_pieces_statistics_match_python(tmp_path):
    import site_toy as S
    from otp.generators import mixture as M

    spec_path = os.path.join(LIVE, "live_spec.json")
    assert os.path.exists(spec_path), "run scripts/export_site_weights.py first"
    n, seeds = 4000, 3
    js = json.loads(run_node(tmp_path, PIECES_JS, LIVE, str(n), str(seeds)))
    mix = M.decode(S.genome(1.0, "none", 0.7, 0.1))
    items = {"season": (mix.procs[0], mix.stats[0]), "ar_noise": (mix.procs[1], mix.stats[1])}
    for w in S.WILD:
        if w != "none":
            mx = M.decode(S.genome(1.0, w, 1.0, 0.1))
            items[w] = (mx.procs[2], mx.stats[2])
    assert set(items) == set(js)
    for name, (proc, (mu, sd)) in items.items():
        acc = []
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            st = proc.init_state(rng, 1)
            w = proc.warmup_steps()
            if w:
                _, st = proc.step(st, w, rng)
            y, _ = proc.step(st, n, rng)
            y = (y[0, 0] - mu) / sd
            srt = np.sort(y)
            m, v = y.mean(), y.var()
            row = {f"q{int(p * 100):02d}": srt[int(np.floor(p * (n - 1)))] for p in (.05, .25, .5, .75, .95)}
            row.update({f"ac{k}": float(((y[k:] - m) * (y[:-k] - m)).mean() / v) for k in (1, 5, 12)})
            acc.append(row)
        py = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        tq, ta = PIECE_TOL.get(name, DEFAULT_TOL)
        for k, v in py.items():
            tol = ta if k.startswith("ac") else tq
            assert abs(js[name][k] - v) <= tol, (name, k, js[name][k], v)


MIX_CONFIGS = [
    {"season": 1.4, "wild": "none", "strength": 0.7, "noise": 0.1},
    {"season": 0.7, "wild": "lorenz63", "strength": 1.4, "noise": 0.1},
    {"season": 1.4, "wild": "mean_jump", "strength": 1.4, "noise": 0.1},
    {"season": 0.0, "wild": "hawkes", "strength": 1.4, "noise": 0.01},
    {"season": 0.7, "wild": "kuramoto", "strength": 0.7, "noise": 0.4},
]
MIX_TOL = {"sd": 0.15, "width": 0.2, "scale": 0.1, "oracle_mase": 0.3}     # relative


def test_mixture_and_referee_match_python(tmp_path):
    import site_toy as S
    from otp import metrics
    from otp.generators import mixture as M
    from otp.generators.base import make_sample

    seeds = 12
    js = json.loads(run_node(tmp_path, MIX_JS, LIVE, json.JSONEncoder().encode(MIX_CONFIGS), str(seeds)))
    for c, got in zip(MIX_CONFIGS, js):
        acc = {"sd": [], "width": [], "oracle_mase": [], "scale": []}
        for s in range(seeds):
            proc = M.decode(S.genome(c["season"], c["wild"], c["strength"], c["noise"]))
            smp = make_sample(proc, 512, 64, seed=7000 + s, n_paths=100)
            ctx, q = smp.context[0], smp.oracle_q[0]
            sd = ctx.std()
            acc["sd"].append(sd)
            acc["width"].append((q[63, 18] - q[63, 2]) / sd)
            acc["oracle_mase"].append(float(metrics.score(ctx, smp.future[0], q, smp.period)["mase"]))
            acc["scale"].append(float(metrics.naive_scale(ctx, smp.period)) / sd)
        for k, v in acc.items():
            py = float(np.mean(v))
            assert abs(got[k] - py) <= MIX_TOL[k] * abs(py), (c["wild"], k, got[k], py)
