"""Export the data files the site (docs/) reads. Every value comes from a committed result file; each output
JSON carries a `sources` list naming the files it was built from.

    python scripts/site_data.py

Writes docs/data/*.json. Sections whose results are not in the repository yet (fine-tuning, search, neuron atlas)
get a file with "status": "pending" and the result path the site is waiting for; rerunning this script after
those results land fills them, with no change to the site code.

Besides the chart files it writes
  docs/data/site_numbers.json   numbers and generated sentences that the story uses ({{site:KEY}}, {{text:NAME}},
                                {{if:FLAG}} blocks in docs/src/index.src.html; scripts/make_site.py fills them)
  docs/data/neurons/            a slimmed copy of results/neurons/browser (see neurons() for what is kept)
  docs/data/scaling.json        error ratio against model size for the chaos and regime families

Environment variables (all optional, for building before a branch is merged; the names written into the JSON
are always the repository paths):
  OTP_NEURONS_ROOT, OTP_SEARCH_ROOT, OTP_FINETUNE_ROOT   folders to read instead of results/neurons, results/search,
                                                         results/finetune
  OTP_SITE_OUT                                           output folder instead of docs/data
"""

from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# Output folder. OTP_SITE_OUT can point elsewhere (for a preview build); the default is docs/data.
OUT = os.environ.get("OTP_SITE_OUT") or os.path.join(ROOT, "docs", "data")
SIZES = ["timestral-5M", "timestral-28M", "timestral-123M", "timestral-209M"]


def p(*a):
    return os.path.join(ROOT, *a)


def r4(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else float(f"{x:.4g}")


def write(name, obj, quiet=False):
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(json.JSONEncoder(separators=(",", ":"), allow_nan=False).encode(obj))
    if not quiet:
        print(f"docs/data/{name}: {os.path.getsize(path):,} bytes")
    return os.path.getsize(path)


class Root:
    """A result folder. Files are read from `base` (the repository folder, or the folder an environment variable names,
    for example a worktree of a branch that is not merged yet) and always named by their repository path."""

    def __init__(self, env, rel):
        self.rel = rel
        self.base = os.environ.get(env) or p(*rel.split("/"))

    def path(self, *a):
        return os.path.join(self.base, *a)

    def name(self, *a):
        return "/".join((self.rel,) + a)

    def has(self, *a):
        return os.path.exists(self.path(*a))


# numbers and generated sentences for the story (written to site_numbers.json)
NUM = {}
READY = {}
TEXT = {}
SRCS = []


_PAPER_NUMS = None


def num(key, value, text, src, how):
    """A number for the story. Keys that paper/build.py defines are taken from paper/numbers.json (one definition
    for paper and site); the value computed here is only a fallback for keys the paper does not have."""
    global _PAPER_NUMS
    if _PAPER_NUMS is None:
        f = os.path.join(ROOT, "paper", "numbers.json")
        _PAPER_NUMS = json.load(open(f)) if os.path.exists(f) else {}
    pv = _PAPER_NUMS.get(key)
    if pv and "text" in pv:
        NUM[key] = {"text": pv["text"], "src": pv["src"], "how": pv["how"], "value": r4(pv.get("value", value))}
    else:
        NUM[key] = {"text": text, "src": src, "how": how, "value": r4(value)}
    return NUM[key]


def span(text, src):
    return f'<span class="num" title="{src}">{text}</span>'


def pending(name, expect, what):
    write(name, {"status": "pending", "expect": expect, "what": what})


# ------------------------------------------------------------------------------------------------
def numbers():
    src = "paper/numbers.json"
    d = json.load(open(p(src)))
    out = {k: {"text": v["text"], "src": v["src"], "how": v["how"]} for k, v in d.items() if not k.startswith("table:") and "text" in v}
    write("numbers.json", {"sources": [src], "numbers": out})


def failure_map():
    rows, srcs = [], []
    for fam in ("chaos", "regime", "tails"):
        src = f"results/map/{fam}/summary_all_sizes.csv"
        s = pd.read_csv(p(src))
        srcs.append(src)
        for r in s.itertuples(index=False):
            rows.append([fam, r.generator, r.config_id, str(r.knob_value), r.forecaster.replace("timestral-", ""), int(r.T), int(r.H),
                         r4(r.mase_ratio_oracle_gmean), r4(r.mase_ratio_oracle_lo95), r4(r.mase_ratio_oracle_hi95),
                         r4(r.mase_ratio_classical_gmean), int(r.n)])
    src = "results/map/in_prior/summary.csv"
    s = pd.read_csv(p(src))
    srcs.append(src)
    for r in s.itertuples(index=False):
        rows.append(["in_prior", "in_prior", r.config_id, str(r.knob_value), r.forecaster.replace("timestral-", ""), int(r.T), int(r.H),
                     r4(r.mase_ratio_oracle_gmean), r4(getattr(r, "mase_ratio_oracle_lo95", np.nan)), r4(getattr(r, "mase_ratio_oracle_hi95", np.nan)),
                     r4(getattr(r, "mase_ratio_classical_gmean", np.nan)), int(r.n)])
    cols = ["family", "generator", "config_id", "knob_value", "size", "T", "H", "ratio_oracle", "lo95", "hi95", "ratio_classical", "n"]
    write("failure_map.json", {"sources": srcs, "columns": cols, "rows": rows,
                               "note": "ratio_oracle: geometric mean over seeds of MASE(Timestral) / MASE(oracle); ratio_classical: same against the best of seasonal naive, ETS, ARIMA"})


def lorenz():
    src = "results/map/chaos/summary_all_sizes.csv"
    lsrc = "results/map/chaos/lyapunov.csv"
    s = pd.read_csv(p(src))
    lam = float(pd.read_csv(p(lsrc)).query("config_id == 'lorenz63/rho28'").lyap_per_time.mean())
    dt = {"lorenz63/rho28_oe2": 0.02, "lorenz63/rho28_oe5": 0.05, "lorenz63/rho28": 0.1, "lorenz63/rho28_oe20": 0.2}
    q = s[(s["T"] == 2048) & s.config_id.isin(dt)].copy()
    q["reach"] = q.config_id.map(dt) * q.H * lam
    series = {fc.replace("timestral-", ""): [[r4(a), r4(b)] for a, b in q[q.forecaster == fc].sort_values("reach")[["reach", "mase_ratio_oracle_gmean"]].values]
              for fc in SIZES}
    write("lorenz.json", {"sources": [src, lsrc], "lyapunov_exponent": r4(lam), "series": series,
                          "x": "forecast reach in Lyapunov times", "y": "MASE / oracle MASE"})


def noise():
    src = "results/mechanism/chaos_noise/summary.csv"
    s = pd.read_csv(p(src))
    s["system"] = s.config_id.str.split("/").str[1]
    s["noise"] = s.config_id.str.extract(r"/n([\d.]+)$")[0].astype(float) * 100
    q = s[s.H == 128]
    out = {}
    for k in sorted(q.system.unique()):
        g = q[q.system == k]
        out[k] = {fc.replace("timestral-", ""): [[r4(a), r4(b)] for a, b in g[g.forecaster == fc].sort_values("noise")[["noise", "mase_ratio_oracle_gmean"]].values]
                  for fc in SIZES}
    write("noise.json", {"sources": [src], "systems": out, "x": "noise, % of signal scale", "y": "MASE / oracle MASE (H = 128)"})


def readouts():
    src = "results/mechanism/readouts/summary.csv"
    s = pd.read_csv(p(src))
    g = s[(s.study == "mean_jump_d") & s.valid]
    out = {}
    for size in sorted(g["size"].unique()):
        h = g[g["size"] == size].sort_values("knob_value")
        out[size] = {"steps": [r4(x) for x in h.knob_value], "readout_r2": [r4(x) for x in h.best_readout_r2], "forecast_r2": [r4(x) for x in h.output_r2]}
    write("readouts.json", {"sources": [src], "mean_jump": out})


def covariates():
    src = "results/covariates/summary.csv"
    s = pd.read_csv(p(src))
    s = s[(s["T"] == 2048) & (s.condition != "classical_best")]
    keep = ["system", "knob", "knob_value", "size", "H", "condition", "n", "covariate_gain_gmean", "oracle_gain_gmean",
            "joint_vs_past_gmean", "mase_ratio_oracle_gmean"]
    rows = [[r.system, r.knob, r4(r.knob_value), r.size, int(r.H), r.condition, int(r.n), r4(r.covariate_gain_gmean), r4(r.oracle_gain_gmean),
             r4(r.joint_vs_past_gmean), r4(r.mase_ratio_oracle_gmean)] for r in s[keep].itertuples(index=False)]
    write("covariates.json", {"sources": [src], "columns": ["system", "knob", "knob_value", "size", "H", "condition", "n", "covariate_gain",
                                                            "oracle_gain", "joint_over_past", "ratio_oracle"], "rows": rows})


def realdata():
    src = "results/regression_baseline/analogue_vs_synthetic.csv"
    d = pd.read_csv(p(src))
    d = d[(d.level == "family") & (d.metric == "mase")]
    rows = [[r.family_analogue, r.forecaster.replace("timestral-", ""), int(r.n_windows), r4(r.real_ratio_to_best_classical_gmean),
             r4(r.synthetic_T2048_H128)] for r in d.itertuples(index=False)]
    s2 = "results/regression_baseline/summary.csv"
    a = pd.read_csv(p(s2))
    a = a[(a.group == "ALL") & (a.basis == "common")]
    sets = [[r.set, r.forecaster, int(r.n), r4(r.mase_rel_snaive), r4(r.scrps_rel_snaive)] for r in a.itertuples(index=False)]
    write("realdata.json", {"sources": [src, s2], "analogue_columns": ["family", "size", "windows", "real_ratio_best_classical", "synthetic_ratio_best_classical"],
                            "analogues": rows, "set_columns": ["set", "forecaster", "windows", "mase_rel_snaive", "scrps_rel_snaive"], "sets": sets})


def selfval():
    src = "results/fix/selfval_confirm/frozen_rule_confirmation.csv"
    r = pd.read_csv(p(src))
    rows = [[x.family, x.size, int(x.n), r4(x.mase_fix_over_timestral), r4(x.switched_share)] for x in r.itertuples(index=False)]
    write("selfval.json", {"sources": [src], "columns": ["family", "size", "samples", "mase_fix_over_timestral", "share_switched"], "rows": rows,
                           "rule": json.load(open(p("results/fix/selfval_confirm/frozen_rule.json")))})


def bench():
    srcs, rows, ft = [], [], {}
    for set_name in ("diag", "lb"):
        for f in sorted(glob.glob(p("bench", "leaderboard", f"*_{set_name}_all", "summary.json"))):
            d = json.load(open(f))
            rel = os.path.relpath(f, ROOT)
            srcs.append(rel)
            o = d["overall"]
            rows.append([set_name, d["model"], int(o["n"]), r4(o["mase_rel_snaive"]), r4(o["scrps_rel_snaive"]), r4(o["mase_rel_oracle"]), r4(o["cov80"])])
            if set_name == "diag":
                ft[d["model"]] = {k: r4(v["mase_rel_oracle"]) for k, v in d["by_failure_type"].items()}
    write("bench.json", {"sources": srcs, "columns": ["set", "model", "series", "mase_rel_snaive", "scrps_rel_snaive", "mase_rel_oracle", "cov80"],
                         "rows": rows, "diag_by_failure_type_mase_rel_oracle": ft})


# ------------------------------------------------------------------------------------------------ slots
# ------------------------------------------------------------------------------------------------ scaling
def failure_frame():
    """The failure-map rows as one frame (same files as failure_map())."""
    frames = []
    for fam in ("chaos", "regime"):
        s = pd.read_csv(p("results", "map", fam, "summary_all_sizes.csv"))
        s["family"] = fam
        frames.append(s)
    return pd.concat(frames, ignore_index=True), ["results/map/chaos/summary_all_sizes.csv", "results/map/regime/summary_all_sizes.csv"]


PARAMS_M = {"5M": 5, "28M": 28, "123M": 123, "209M": 209}   # the sizes' names, in millions of parameters


def scaling():
    """Error ratio against model size. Per family, horizon and generator two lines: the setting where the smallest
    model has its largest ratio ("hardest", one config) and the geometric mean over all settings of that generator
    ("average"). Chaos: context 2048, without the observation-step variants; regime: context 8192, switches inside
    the context."""
    s, srcs = failure_frame()
    s = s.copy()
    s["size"] = s.forecaster.str.replace("timestral-", "", regex=False)
    keep = ((s.family == "chaos") & (s["T"] == 2048) & ~s.config_id.str.contains("_oe")) | \
           ((s.family == "regime") & (s["T"] == 8192) & s.config_id.str.contains("/context/"))
    s = s[keep]
    rows = []
    for (fam, gen, H), g in s.groupby(["family", "generator", "H"]):
        wide = g.pivot_table(index="config_id", columns="size", values="mase_ratio_oracle_gmean")
        wide = wide.dropna()
        if wide.empty or not set(PARAMS_M) <= set(wide.columns):
            continue
        hard = wide["5M"].idxmax()
        rows.append([fam, gen, int(H), "hardest", hard] + [r4(wide.loc[hard, k]) for k in PARAMS_M])
        gm = np.exp(np.log(wide[list(PARAMS_M)]).mean())
        rows.append([fam, gen, int(H), "average", f"{len(wide)} settings"] + [r4(gm[k]) for k in PARAMS_M])
    write("scaling.json", {"sources": srcs, "params_millions": list(PARAMS_M.values()), "sizes": list(PARAMS_M),
                           "columns": ["family", "generator", "H", "stat", "config", "r5M", "r28M", "r123M", "r209M"], "rows": rows,
                           "note": "MASE(Timestral) / MASE(oracle), geometric mean over seeds per config; 'average' is the geometric mean over a generator's configs"})


# ------------------------------------------------------------------------------------------------ neuron atlas
KTOP = 60          # poles per layer (best R^2 first) that keep exemplars, besides every one-job pole and every steered pole


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}%"


def neurons():
    R = Root("OTP_NEURONS_ROOT", "results/neurons")
    if not R.has("browser", "index.json"):
        READY["atlas"] = False
        return pending("neurons.json", "results/neurons/browser/*.json", "neuron atlas: per-size pole labels, scores and exemplars")
    idx = json.load(open(R.path("browser", "index.json")))
    fields = idx["fields"]
    F = {k: i for i, k in enumerate(fields)}
    theta, floor = idx["theta"], idx["floor"]
    sources = [R.name("browser", "index.json")]
    files = []
    sizes = {}
    total_in, total_out = 0, 0
    for size, info in idx["sizes"].items():
        labels0 = None
        per_layer = []
        for li in range(info["layers"]):
            fn = R.path("browser", size, f"layer_{li}.json")
            raw = open(fn, "rb").read()
            total_in += len(raw)
            d = json.loads(raw)
            if labels0 is None:
                labels0 = d["labels"]
            assert d["labels"] == labels0, "label list differs between layers"
            poles = d["poles"]
            order = sorted(range(len(poles)), key=lambda i: (-(poles[i][F["r2"]] if poles[i][F["r2"]] is not None else -1.0), i))
            ids, lab, r2, ratio, cls, detail = [], [], [], [], [], {}
            for rank, i in enumerate(order):
                r = poles[i]
                ids.append(r[F["neuron"]] * 2 + r[F["sign"]])
                lab.append(r[F["label"]])
                r2.append(-1 if r[F["r2"]] is None else int(round(r[F["r2"]] * 10000)))
                ratio.append(-1 if r[F["ratio"]] is None else int(round(r[F["ratio"]] * 1000)))
                cls.append(r[F["class"]])
                has_ex = bool(r[F["top"]])
                if has_ex and (rank < KTOP or r[F["class"]] == "o" or r[F["steer"]]):
                    snip = lambda L: [[s[1], int(round(s[2] * 1000)) / 1000, s[3], s[4], s[5]] for s in L]
                    detail[str(rank)] = [r[F["E"]], r[F["heldout"]], r[F["corr_sign"]], r[F["pair_label"]], r[F["regime_mass"]],
                                         r[F["transfer_corr"]], r[F["steer"]], snip(r[F["top"]]), snip(r[F["mid"]]), snip(r[F["real"]])]
            out = {"size": size, "layer": li, "n": len(poles), "id": ids, "label": lab, "r2": r2, "ratio": ratio, "cls": "".join(cls), "detail": detail}
            rel = f"neurons/{size}/layer_{li}.json"
            total_out += write(rel, out, quiet=True)
            files.append("data/" + rel)
            sources.append(R.name("browser", size, f"layer_{li}.json"))
            per_layer.append(len(detail))
        sizes[size] = dict(info, labels=labels0, with_exemplars=sum(per_layer))
    write("neurons/index.json", {"fields": ["id = 2 * neuron + sign (0 positive pole, 1 negative)", "label (index into labels)", "r2 x 10000", "ratio x 1000",
                                             "class: o one job, m mixed, u unexplained, d dead"],
                                 "detail_fields": ["E", "heldout_r2", "label_sign", "pair_label", "regime_mass", "transfer_corr", "steering", "top", "mid", "real"],
                                 "snippet": "[token, activation, 32 hexadecimal digits (patch scaled 0..15), min, max]",
                                 "token": "0..31 context patches, then forecast patches", "theta": theta, "floor": floor, "k_top": KTOP,
                                 "kept": f"exemplars kept for every one-job pole, every steered pole and the {KTOP} best-scoring poles of each layer; "
                                         "activations to 3 decimals, r2 to 4, ratio to 3",
                                 "sizes": sizes, "source_index": idx.get("provenance")}, quiet=True)
    print(f"docs/data/neurons/: {len(files)} layer files, {total_out:,} bytes from {total_in:,} bytes")

    # ---- chart tables
    out = {"status": "ready", "sources": sources[:1], "files": files, "browser_index": "data/neurons/index.json", "theta": theta, "floor": floor,
           "bytes_in": total_in, "bytes_out": total_out}
    c = pd.read_csv(R.path("one_job_curves.csv"))
    out["sources"].append(R.name("one_job_curves.csv"))
    a = c[(c.scope == "all") & (c.sign == "both")]
    out["curves"] = {"columns": ["size", "floor", "theta", "one_job", "mixed", "unexplained", "one_job_lo", "one_job_hi", "n_live"],
                     "rows": [[r.size, r4(r.floor), r4(r.theta), r4(r.one_job), r4(r.mixed), r4(r.unexplained), r4(r.one_job_lo), r4(r.one_job_hi), int(r.n_live)]
                              for r in a.itertuples(index=False)]}
    lay = c[c.scope.str.startswith("layer_") & (c.sign == "both") & np.isclose(c.floor, floor) & np.isclose(c.theta, theta)]
    rows = []
    for r in lay.itertuples(index=False):
        k = int(r.scope.split("_")[1])
        nl = idx["sizes"][r.size]["layers"]
        rows.append([r.size, k, nl, r4(k / (nl - 1)), int(r.n_live), r4(r.one_job), r4(r.one_job_lo), r4(r.one_job_hi), r4(r.mixed), r4(r.unexplained)])
    out["layers"] = {"columns": ["size", "layer", "n_layers", "depth", "n_live", "one_job", "one_job_lo", "one_job_hi", "mixed", "unexplained"], "rows": rows}
    cc = pd.read_csv(R.path("context_comparison.csv"))
    out["sources"].append(R.name("context_comparison.csv"))
    out["corpora"] = {"columns": ["size", "corpus", "one_job", "unexplained", "E_mean", "same_label_as_main"],
                      "rows": [[r.size, r.context, r4(r.one_job), r4(r.unexplained), r4(r.E_mean), r4(r.same_label_as_main)] for r in cc.itertuples(index=False)]}
    info = {}
    for cname in ("main", "short", "long", "xlong"):
        mf = R.path("corpus", f"{cname}_manifest.json")
        if os.path.exists(mf):
            m = json.load(open(mf))
            info[cname] = {"T": int(m["T"]), "n_series": int(m["n_series"])}
            out["sources"].append(R.name("corpus", f"{cname}_manifest.json"))
    out["corpus_info"] = info
    fb = R.path("followup", "corpus_library_control.csv")
    if os.path.exists(fb):
        # follow-up B: the main corpus subsampled to fewer series, a library common to all corpora, random libraries
        b = pd.read_csv(fb)
        b = b[np.isclose(b.theta, theta) & np.isclose(b.floor, floor)]
        g = b.groupby(["size", "role", "corpus", "n_series", "library"]).agg(
            one_job=("one_job", "mean"), lo=("one_job", "min"), hi=("one_job", "max"), E=("E_mean", "mean"),
            agree=("same_label_as_reference", "mean"), runs=("tag", "count")).reset_index()
        out["controls"] = {"columns": ["size", "role", "corpus", "n_series", "library", "runs", "one_job", "lo", "hi", "E_mean", "same_label_as_reference"],
                           "rows": [[r.size, r.role, r.corpus, int(r.n_series), r.library, int(r.runs), r4(r.one_job), r4(r.lo), r4(r.hi), r4(r.E), r4(r.agree)]
                                    for r in g.itertuples(index=False)]}
        out["sources"].append(R.name("followup", "corpus_library_control.csv"))
        sub = sorted(g[(g.role == "series_count") & (g["size"] == "5M")].n_series.unique(), reverse=True)
        n_main = int(g[g.role == "reference"].n_series.max())
        ks = sorted(int(x.replace("random", "")) for x in g[g.role == "library_size"].library.unique())
        src_fb = R.name("followup", "corpus_library_control.csv")
        num("nnb_n_main", n_main, f"{n_main:,}", src_fb, "series in the main corpus")
        num("nnb_n_mid", int(sub[0]), f"{int(sub[0]):,}", src_fb, "largest subsample of the main corpus")
        num("nnb_n_small", int(sub[-1]), f"{int(sub[-1]):,}", src_fb, "smallest subsample of the main corpus")
        num("nnb_k_lo", ks[0], str(ks[0]), src_fb, "smallest random library size")
        num("nnb_k_hi", ks[-1], str(ks[-1]), src_fb, "largest random library size")
    st = pd.read_csv(R.path("steering_summary.csv"))
    out["sources"].append(R.name("steering_summary.csv"))
    st = st[st.group == "all"]
    out["steering"] = {"columns": ["size", "role", "n", "n_match", "match_rate", "ci_lo", "ci_hi"],
                       "rows": [[r.size, r.role, int(r.n), int(r.n_match), r4(r.match_rate), r4(r.ci_lo), r4(r.ci_hi)] for r in st.itertuples(index=False)]}
    lg = pd.read_csv(R.path("label_groups.csv"))
    out["sources"].append(R.name("label_groups.csv"))
    out["groups"] = {"columns": ["size", "group", "n_one_job", "share_of_one_job"],
                     "rows": [[r.size, r.group, int(r.n_one_job), r4(r.share_of_one_job)] for r in lg.itertuples(index=False)]}
    sc = pd.read_csv(R.path("scaling.csv"))
    out["sources"].append(R.name("scaling.csv"))
    out["scaling"] = {"columns": ["size", "n_poles", "n_live", "one_job", "mixed", "unexplained"],
                      "rows": [[r.size, int(r.n_poles), int(r.n_live), r4(r.one_job), r4(r.mixed), r4(r.unexplained)] for r in sc.itertuples(index=False)]}
    write("neurons.json", out)

    # ---- numbers for the story that the paper's numbers.json does not carry
    big = sc.set_index("size")
    num("nn_unexpl_5M", 100 * big.loc["5M", "unexplained"], f"{100 * big.loc['5M', 'unexplained']:.0f}", R.name("scaling.csv"), "percent of live poles unexplained at floor 0.1 (theta does not change it)")
    num("nn_unexpl_209M", 100 * big.loc["209M", "unexplained"], f"{100 * big.loc['209M', 'unexplained']:.0f}", R.name("scaling.csv"), "same, 209M")
    num("nn_poles_5M", big.loc["5M", "n_poles"], f"{int(big.loc['5M', 'n_poles']):,}", R.name("scaling.csv"), "poles (two per neuron)")
    num("nn_poles_209M", big.loc["209M", "n_poles"], f"{int(big.loc['209M', 'n_poles']):,}", R.name("scaling.csv"), "poles (two per neuron)")
    g = lg.set_index(["size", "group"]).share_of_one_job
    for size in ("5M", "209M"):
        for grp in ("position", "chaos"):
            v = g.get((size, grp))
            if v is not None:
                num(f"nn_grp_{grp}_{size}", 100 * v, f"{100 * v:.0f}", R.name("label_groups.csv"), f"percent of {size}'s one-job poles whose label is in the {grp} group")
    srow = st[st.role == "labeled"]
    num("nn_steer_matches", int(srow.n_match.sum()), f"{int(srow.n_match.sum())}", R.name("steering_summary.csv"), "labeled poles whose steering matched the pre-registered prediction, all sizes")
    ctl = st[st.role == "control"]
    num("nn_steer_control_n", int(ctl.n.sum()), f"{int(ctl.n.sum()):,}", R.name("steering_summary.csv"), "random control poles steered, all sizes")
    num("nn_steer_control_matches", int(ctl.n_match.sum()), f"{int(ctl.n_match.sum())}", R.name("steering_summary.csv"), "control poles that met the match criterion, all sizes")
    fl = pd.read_csv(R.path("failure_link.csv"))
    ab = fl[(fl.analysis == "ablate") & np.isclose(fl.scale, 0.0)]
    for size in ("209M",):
        for st_name, key in (("periodic_elevated", "elev"), ("random_same_layers", "rand")):
            r = ab[(ab["size"] == size) & (ab.set == st_name)]
            if len(r):
                num(f"nn_abl_{key}_before_{size}", r.mase_before.iloc[0], f"{r.mase_before.iloc[0]:.2f}", R.name("failure_link.csv"), f"{size}: MASE on failing contexts before ablation")
                num(f"nn_abl_{key}_after_{size}", r.mase_after.iloc[0], f"{r.mase_after.iloc[0]:.2f}", R.name("failure_link.csv"), f"{size}: MASE after setting the {st_name} poles to zero")
                num(f"nn_abl_{key}_n_{size}", r.n_poles.iloc[0], f"{int(r.n_poles.iloc[0])}", R.name("failure_link.csv"), f"{size}: number of poles set to zero ({st_name})")
    SRCS.append(R.name("browser"))
    READY["atlas"] = True


# ------------------------------------------------------------------------------------------------ search
SEARCH_LABEL = {"cmaes": "CMA-ES", "map_elites": "MAP-Elites", "gradient": "gradient search", "gradient_r12": "gradient search with 12 starts"}
SEARCH_ORDER = ["cmaes", "map_elites", "gradient", "gradient_r12"]
SEARCH_FAMILIES = ["primitive", "chaos", "regime", "tails"]   # scripts/search_report.py: the four dominant families
SEARCH_BINS = 10                                               # spectral-entropy bins of the same grid
SEARCH_FLOOR = 0.05                                            # oracle sCRPS / seasonal naive floor used in the paper (section 6)


def search():
    R = Root("OTP_SEARCH_ROOT", "results/search")
    if not R.has("comparison.csv"):
        READY["search"] = False
        return pending("search.json", "results/search/comparison.csv", "CMA-ES vs MAP-Elites vs gradient search at equal budget")
    comp = pd.read_csv(R.path("comparison.csv"))
    src = [R.name("comparison.csv")]
    comp["seed"] = comp["seed"].astype(str)
    mean = comp[comp.seed == "mean"]
    per = comp[comp.seed != "mean"]
    methods = [m for m in SEARCH_ORDER if m in set(comp.method)]
    out = {"status": "ready", "sources": src, "methods": [[m, SEARCH_LABEL[m]] for m in methods]}
    # best-so-far against Timestral calls: per method, median and range over the search seeds on a log grid of calls.
    # "curves" keeps only series whose referee error is at least SEARCH_FLOOR x seasonal naive's (scripts/search_floor_report.py);
    # "curves_raw" is the search's own best-so-far, which near-noise-free series dominate (results/search/oracle_check.csv).
    curves, curves_raw, cells_fl = {}, {}, {}
    for m in methods:
        runs, runs_raw, per_cell = [], [], {}
        for f in sorted(glob.glob(R.path("runs", f"{m}_seed*", "history.csv"))):
            h = pd.read_csv(f).sort_values("eval")
            if not len(h):
                continue
            ok = (h.feasible.astype(str) == "True") & (h.orc_sn >= SEARCH_FLOOR)
            runs_raw.append((h["calls"].to_numpy(float), h["best_so_far"].to_numpy(float)))
            runs.append((h["calls"].to_numpy(float), h.fitness.where(ok).cummax().ffill().to_numpy(float)))
            seen = {}
            for r in h[ok].itertuples(index=False):
                k = f"{r.family}|{min(int(max(r.spec_entropy, 0.0) * SEARCH_BINS), SEARCH_BINS - 1)}"
                seen[k] = max(seen.get(k, 0.0), r.fitness)
            for k, v in seen.items():
                b, n = per_cell.get(k, (0.0, 0))
                per_cell[k] = (max(b, v), n + 1)
            src.append(R.name("runs", os.path.basename(os.path.dirname(f)), "history.csv"))
        if not runs:
            continue
        for store, rr in ((curves, runs), (curves_raw, runs_raw)):
            lo_c = min(r[0][0] for r in rr)
            hi_c = min(r[0][-1] for r in rr)
            grid = np.unique(np.round(np.geomspace(lo_c, hi_c, 90)))
            vals = np.array([[b[np.searchsorted(c, g, side="right") - 1] if c[0] <= g else np.nan for g in grid] for c, b in rr])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med, lo, hi = np.nanmedian(vals, 0), np.nanmin(vals, 0), np.nanmax(vals, 0)
            store[m] = {"calls": [int(g) for g in grid], "median": [r4(x) if np.isfinite(x) else None for x in med],
                        "lo": [r4(x) if np.isfinite(x) else None for x in lo], "hi": [r4(x) if np.isfinite(x) else None for x in hi], "runs": len(rr)}
        cells_fl[m] = {"cells": {k: [r4(v[0]), int(v[1])] for k, v in per_cell.items()}, "runs": len(runs)}
    out["floor"] = SEARCH_FLOOR
    out["curves"] = curves
    out["curves_raw"] = curves_raw
    keep = [c for c in comp.columns if c in ("method", "seed", "evaluations", "calls", "n_feasible", "best_fitness", "median_top20", "calls_to_1.5", "calls_to_2.0", "calls_to_3.0",
                                              "n_runs_reaching_1.5", "n_runs_reaching_2.0", "n_runs_reaching_3.0", "cells_filled", "cells_ge_1.5", "cells_ge_2.0", "cells_ge_3.0",
                                              "signatures_ge_2.0", "families_ge_1.5")]
    out["comparison"] = {"columns": keep, "rows": [[(r4(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v) for v in row]
                                                   for row in mean[keep].itertuples(index=False)]}
    out["grid"] = {"families": SEARCH_FAMILIES, "bins": SEARCH_BINS, "methods": cells_fl}
    for name, key in (("transfer_floored.csv", "transfer"), ("transfer_spearman.csv", "spearman"), ("input_space_summary.csv", "input_space"),
                      ("input_space_random_summary.csv", "input_space_random")):
        if R.has(name):
            t = pd.read_csv(R.path(name))
            if key == "transfer":
                t = t[np.isclose(t["floor"], SEARCH_FLOOR) & (t.n > 0)].drop(columns="floor")
            src.append(R.name(name))
            out[key] = {"columns": list(t.columns), "rows": json.loads(t.to_json(orient="values", double_precision=4))}
    write("search.json", out)
    # sentences
    nr = per.groupby("method").size()
    li = []
    for m in methods:
        r = mean[mean.method == m].iloc[0]
        bits = [f"best series found: {span(f'{r.best_fitness:.3g}', src[0])}× the referee's error"]
        for t in ("2.0", "3.0"):
            c, n = r.get(f"calls_to_{t}"), r.get(f"n_runs_reaching_{t}")
            if c is not None and pd.notna(c):
                bits.append(f"reached {float(t):g}× after {span(f'{c:,.0f}', src[0])} calls on average ({span(f'{int(n)}', src[0])} of {int(nr[m])} runs)")
            elif n is not None:
                bits.append(f"reached {float(t):g}× in {span(f'{int(n)}', src[0])} of {int(nr[m])} runs")
        if "cells_filled" in r and pd.notna(r.cells_filled):
            bits.append(f"filled {span(f'{r.cells_filled:.1f}', src[0])} cells of the shared grid on average")
        li.append(f"<li><strong>{SEARCH_LABEL[m]}</strong>: " + "; ".join(bits) + ".</li>")
    calls = mean.calls.dropna()
    head = (f"Each method got about {span(f'{calls.mean():,.0f}', src[0])} Timestral calls on the 5M model per search seed. "
            "Fitness is Timestral's error divided by the referee's, with the referee required to be clearly better than \"repeat last season\" so the series stays forecastable. Means over search seeds:")
    tail = ""
    if "transfer" in out:
        tail = " The charts below also show how many of the failures found on 5M still fail on the larger sizes."
    TEXT["search"] = head + "<ul class=\"plain\">" + "".join(li) + "</ul>" + tail
    SRCS.extend(src[:1])
    READY["search"] = True


# ------------------------------------------------------------------------------------------------ fine-tuning
def finetune():
    R = Root("OTP_FINETUNE_ROOT", "results/finetune")
    found = sorted(glob.glob(R.path("local", "summary.csv")) + glob.glob(R.path("colab", "summary.csv")))   # the two lanes; smoke runs are left out
    if not found:
        READY["finetune"] = False
        return pending("finetune.json", "results/finetune/<local|colab>/summary.csv", "fine-tuning gain and cost per method and protocol")
    src, frames = [], []
    for f in found:
        lane = os.path.basename(os.path.dirname(f))
        s = pd.read_csv(f)
        src.append(R.name(lane, "summary.csv"))
        cf = os.path.join(os.path.dirname(f), "costs.csv")
        if os.path.exists(cf):
            c = pd.read_csv(cf)
            src.append(R.name(lane, "costs.csv"))
            s = s.merge(c[["run_id", "trainable_params", "steps", "time_steps_seen", "wall_sec"]], on="run_id", how="left")
        s["lane"] = lane
        frames.append(s)
    s = pd.concat(frames, ignore_index=True)
    sizes = [z for z in ("5M", "28M", "123M", "209M") if z in set(s["size"])]
    cols = ["size", "protocol", "holdout", "method", "replay", "paired", "lo", "hi", "gap_mase", "gap_scrps", "pre_ratio", "ft_ratio", "cov80_pre", "cov80_ft",
            "trainable_params", "time_steps_seen", "wall_sec"]

    def row(r):
        return [r.size, r.protocol, r.holdout, r.method, r4(r.replay_frac), r4(r.paired_mase_ft_over_pre), r4(r.paired_mase_lo95), r4(r.paired_mase_hi95),
                r4(r.gap_closed_mase), r4(r.gap_closed_scrps), r4(r.pre_mase_ratio), r4(r.ft_mase_ratio), r4(r.pre_cov80), r4(r.ft_cov80),
                None if pd.isna(getattr(r, "trainable_params", np.nan)) else int(r.trainable_params),
                None if pd.isna(getattr(r, "time_steps_seen", np.nan)) else int(r.time_steps_seen), r4(getattr(r, "wall_sec", np.nan))]
    t = s[s.role == "test"]
    g = s[s.role == "regression"]
    gcols = ["size", "protocol", "holdout", "method", "replay", "set", "paired", "lo", "hi", "cov80_pre", "cov80_ft", "run"]
    out = {"status": "ready", "sources": src, "sizes": sizes,
           "tests": {"columns": cols, "rows": [row(r) for r in t.itertuples(index=False)]},
           "regression": {"columns": gcols, "rows": [[r.size, r.protocol, r.holdout, r.method, r4(r.replay_frac), r.eval_set.split("/")[1], r4(r.paired_mase_ft_over_pre),
                                                       r4(r.paired_mase_lo95), r4(r.paired_mase_hi95), r4(r.pre_cov80), r4(r.ft_cov80), r.run_id]
                                                      for r in g.itertuples(index=False)]}}
    write("finetune.json", out)
    s0 = src[0]
    parts = [f"Sizes with results: {', '.join(sizes)}. Each run trained one model on some failure families and was scored on 32 held-out series per family against the referee."]
    for proto, what in (("lofo", "a family left out of training"), ("knob", "a range of the difficulty setting left out of training")):
        x = t[t.protocol == proto].paired_mase_ft_over_pre.dropna()
        if len(x):
            parts.append(f"With {what}, fine-tuned error divided by the original error runs from {span(f'{x.min():.3f}', s0)} to {span(f'{x.max():.3f}', s0)} "
                         f"over {span(str(len(x)), s0)} runs (1 means no change, below 1 means better).")
    gi = g[g.eval_set == "regression/in_prior"]
    for rf in sorted(gi.replay_frac.unique()):
        x = gi[gi.replay_frac == rf].paired_mase_ft_over_pre.dropna()
        if len(x):
            parts.append(f"On normal in-prior series ({'with' if rf > 0 else 'without'} replay of such series in training) the same ratio runs from "
                         f"{span(f'{x.min():.3f}', s0)} to {span(f'{x.max():.3f}', s0)}.")
    TEXT["finetune"] = " ".join(parts)
    SRCS.extend(src)
    READY["finetune"] = True


def site_numbers():
    write("site_numbers.json", {"sources": sorted(set(SRCS)), "ready": READY, "numbers": NUM, "text": TEXT})


def main():
    for fn in (numbers, failure_map, lorenz, noise, readouts, covariates, realdata, selfval, bench, scaling, finetune, search, neurons, site_numbers):
        fn()


if __name__ == "__main__":
    main()
