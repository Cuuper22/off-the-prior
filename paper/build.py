"""Build the writeup from committed result files.

paper/src.md holds the text with placeholders:
  {{num:KEY}}    a number computed below from a result file
  {{table:KEY}}  a markdown table computed below from a result file
This script fills them, writes paper/off_the_prior.md, and writes paper/numbers.json: for every
key, the value, the source file(s) and a one-line description of how it was computed. Any
placeholder without a definition stops the build, so no number can be typed in by hand.

python paper/build.py
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from otp.analysis import ratios  # noqa: E402

NUMS: dict[str, dict] = {}
TABLES: dict[str, dict] = {}
SIZES = ["timestral-5M", "timestral-28M", "timestral-123M", "timestral-209M"]
SIZE_LABEL = {s: s.replace("timestral-", "") for s in SIZES}


def p(*parts):
    return os.path.join(ROOT, *parts)


def gm(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if x.size else float("nan")


def num(key, value, src, how, fmt="{:.2f}"):
    NUMS[key] = {"value": value, "text": fmt.format(value), "src": src, "how": how}


def table(key, df: pd.DataFrame, src, how, floatfmt="{:.2f}"):
    def f(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("–" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    int_cols = {c for c in df.columns if pd.api.types.is_integer_dtype(df[c])}
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(int(v)) if c in int_cols else f(v) for c, v in zip(df.columns, row)) + " |")
    TABLES[key] = {"text": "\n".join(lines), "src": src, "how": how}


def load_summary(path):
    s = pd.read_csv(p(path))
    s["knob_value"] = s["knob_value"].astype(str)
    return s


# --------------------------------------------------------------------------------------------
def in_prior():
    src = "results/map/in_prior/summary.csv"
    s = load_summary(src)
    g = s.groupby(["forecaster", "T", "H"]).mase_ratio_oracle_gmean.apply(gm).reset_index()
    short = g[g.H <= 128]
    num("inprior_short_min", short.mase_ratio_oracle_gmean.min(), src, "min over sizes, T, H<=128 of the geometric mean over configs of the per-config geometric-mean MASE ratio to oracle")
    num("inprior_short_max", short.mase_ratio_oracle_gmean.max(), src, "max of the same")
    rows = []
    for fc in SIZES:
        r = {"size": SIZE_LABEL[fc]}
        for h in (16, 128, 1024, 2048):
            v = g[(g.forecaster == fc) & (g["T"] == 2048) & (g.H == h)].mase_ratio_oracle_gmean
            r[f"H={h}"] = float(v.iloc[0]) if len(v) else float("nan")
        rows.append(r)
    table("inprior", pd.DataFrame(rows), src, "geometric mean over the 4 in-prior configs of the per-config geometric-mean MASE ratio to oracle, context 2048")


def family_table(src, key, configs, T=2048, H=128, label=None):
    s = load_summary(src)
    q = s[(s["T"] == T) & (s.H == H) & s.config_id.isin(configs)]
    rows = []
    for cid in configs:
        r = {"series": label(cid) if label else cid}
        for fc in SIZES:
            v = q[(q.config_id == cid) & (q.forecaster == fc)].mase_ratio_oracle_gmean
            r[SIZE_LABEL[fc]] = float(v.iloc[0]) if len(v) else float("nan")
        v = q[(q.config_id == cid) & (q.forecaster == "timestral-209M")].mase_ratio_classical_gmean
        r["209M vs best classical"] = float(v.iloc[0]) if len(v) else float("nan")
        rows.append(r)
    table(key, pd.DataFrame(rows), src, f"per-config geometric mean over seeds of MASE ratio to oracle (and to the best classical model), context {T}, horizon {H}")
    return q


def chaos():
    src = "results/map/chaos/summary_all_sizes.csv"
    cfg = ["lorenz63/rho20", "lorenz63/rho28", "lorenz63/rho28_oe2", "rossler/c3", "rossler/c5.7", "rossler/c9",
           "mackey_glass/tau10", "mackey_glass/tau17", "mackey_glass/tau30", "logistic_map/r3.5", "logistic_map/r3.9",
           "kuramoto/K1", "kuramoto/K8"]
    q = family_table(src, "chaos_main", cfg)
    v = q[q.forecaster == "timestral-209M"].mase_ratio_classical_gmean
    num("chaos_209m_vs_classical_median", float(np.median(v)), src, "median over chaos configs of the 209M geometric-mean MASE ratio to the best classical model, T=2048, H=128")
    for cid, k in [("mackey_glass/tau30", "mg30"), ("kuramoto/K1", "kur1"), ("kuramoto/K8", "kur8"), ("rossler/c9", "ros9"),
                   ("lorenz63/rho28_oe2", "lor_oe2")]:
        for fc in ("timestral-5M", "timestral-209M"):
            r = q[(q.config_id == cid) & (q.forecaster == fc)]
            num(f"chaos_{k}_{SIZE_LABEL[fc]}", float(r.mase_ratio_oracle_gmean.iloc[0]), src, f"{cid}, {fc}, T=2048, H=128, geometric-mean MASE ratio to oracle")
    # Lyapunov-time view for Lorenz rho=28 at four sampling intervals
    s = load_summary(src)
    dt = {"lorenz63/rho28_oe2": 0.02, "lorenz63/rho28_oe5": 0.05, "lorenz63/rho28": 0.1, "lorenz63/rho28_oe20": 0.2}
    lyap = pd.read_csv(p("results/map/chaos/lyapunov.csv"))
    lam = float(lyap[lyap.config_id == "lorenz63/rho28"].lyap_per_time.mean())
    num("lorenz_lyap", lam, "results/map/chaos/lyapunov.csv", "largest Lyapunov exponent per time unit, Lorenz rho=28, mean over seeds", "{:.3f}")
    r = s[(s["T"] == 2048) & (s.forecaster == "timestral-209M") & s.config_id.isin(dt)].copy()
    r["lyapunov_times"] = r.config_id.map(dt) * r.H.astype(float) * lam
    r = r.sort_values("lyapunov_times")
    out = pd.DataFrame({"sampling step (time units)": r.config_id.map(dt).values, "horizon (steps)": r.H.astype(int).values,
                        "horizon (Lyapunov times)": r.lyapunov_times.values, "209M MASE / oracle": r.mase_ratio_oracle_gmean.values})
    table("lorenz_lyap", out, src + " + results/map/chaos/lyapunov.csv", "209M, context 2048; Lyapunov times = sampling step x horizon x exponent")


def tails():
    src = "results/map/tails/summary_all_sizes.csv"
    s = load_summary(src)
    q = s[(s["T"] == 2048) & (s.H == 128)]
    num("tails_min", q.mase_ratio_oracle_gmean.min(), src, "min over all tails configs and sizes, T=2048, H=128")
    num("tails_max", q.mase_ratio_oracle_gmean.max(), src, "max of the same")
    cfg = ["tails/stable_ar/alpha2_rho0.9", "tails/stable_ar/alpha1.3_rho0.9", "tails/stable_ar/alpha1.3_rho1.0",
           "tails/fbm/H0.2", "tails/fbm/H0.8", "tails/hawkes/eta0.3", "tails/hawkes/eta0.95", "tails/msm/k8_ret", "tails/msm/k8_cum"]
    family_table(src, "tails_main", cfg)


def long_ctx():
    src = "results/map/long_ctx/scores.csv"
    df = ratios(pd.read_csv(p(src)))
    ts = df[df.forecaster.isin(SIZES) & (df.H == 128)]
    g = ts.groupby(["config_id", "forecaster", "T"]).mase_ratio_oracle.apply(gm).reset_index()
    rows = []
    for cid in ["mackey_glass/tau17", "kuramoto/K1", "rossler/c5.7", "lorenz63/rho28", "regime/mean_jump/context/delta4/d64",
                "tails/fbm/H0.8", "in_prior/p24"]:
        for fc in ("timestral-5M", "timestral-209M"):
            r = {"series": cid, "size": SIZE_LABEL[fc]}
            for t in (2048, 8192, 32768):
                v = g[(g.config_id == cid) & (g.forecaster == fc) & (g["T"] == t)].mase_ratio_oracle
                r[f"T={t}"] = float(v.iloc[0]) if len(v) else float("nan")
            rows.append(r)
    table("long_ctx", pd.DataFrame(rows), src, "geometric mean over 8 seeds of MASE ratio to oracle at H=128 (oracle rows matched per context length)")
    for cid, k in [("mackey_glass/tau17", "mg17")]:
        for fc in ("timestral-5M", "timestral-209M"):
            for t in (2048, 32768):
                v = float(g[(g.config_id == cid) & (g.forecaster == fc) & (g["T"] == t)].mase_ratio_oracle.iloc[0])
                num(f"long_{k}_{SIZE_LABEL[fc]}_T{t}", v, src, f"{cid}, {fc}, T={t}, H=128")
    bad = df[(df.forecaster == "oracle") & ~np.isfinite(df.mase)].sample_id.nunique()
    num("long_nonfinite_oracle_samples", bad, src, "samples with a non-finite oracle (diverged Monte Carlo paths)", "{:d}")


def symavg():
    src = "results/mechanism/symavg/off_over_on.csv"
    w = pd.read_csv(p(src))
    q = w[w.H == 128]
    num("symavg_min", q.scrps_off_over_on.min(), src, "min over configs and sizes of sCRPS ratio (averaging off / on), H=128", "{:.3f}")
    num("symavg_max", q.scrps_off_over_on.max(), src, "max of the same", "{:.3f}")


def normalization():
    src = "results/mechanism/normalization/summary.csv"
    s = pd.read_csv(p(src))
    full = s[s.condition == "full"].set_index(["case", "T", "size"])
    post = s[s.condition == "post_only"].set_index(["case", "T", "size"])
    j = full.join(post, lsuffix="_full", rsuffix="_post")
    num("norm_cells", len(j), src, "case x context x size cells with both conditions", "{:d}")
    num("norm_post_better", int((j.scrps_ratio_gmean_post < j.scrps_ratio_gmean_full).sum()), src, "cells where cutting to the post-switch part lowered the sCRPS ratio", "{:d}")
    v = s[(s.case == "variance_jump/f0.25/d64") & (s.condition == "full")]
    num("norm_vdrop_cov80_min", v.cov80.min(), src, "variance drop x0.25, 64 steps back, full context: min cov80 over sizes and contexts", "{:.3f}")
    num("norm_vdrop_scrps_min", v.scrps_ratio_gmean.min(), src, "same cells: min sCRPS ratio to oracle")
    num("norm_vdrop_scrps_max", v.scrps_ratio_gmean.max(), src, "same cells: max sCRPS ratio to oracle")


def analog():
    src = "results/mechanism/analog/summary.csv"
    s = pd.read_csv(p(src))
    cfg = ["mackey_glass/tau17", "mackey_glass/tau30", "rossler/c5.7", "rossler/c9", "kuramoto/K1", "kuramoto/K8",
           "lorenz63/rho28", "logistic_map/r3.9"]
    cols = [c for c in ["mase_oracle", "mase_analog", "mase_template", "mase_5M", "mase_28M", "mase_123M", "mase_209M"] if c in s]
    out = s[s.config_id.isin(cfg)].set_index("config_id").loc[cfg, cols].reset_index()
    out.columns = ["series"] + [c.replace("mase_", "") for c in cols]
    table("analog", out, src, "geometric mean over 8 seeds of MASE (context 2048, horizon 128) for the oracle, the method of analogues, the seasonal template, and Timestral")


def chaos_noise():
    src = "results/mechanism/chaos_noise/summary.csv"
    s = load_summary(src)
    s["system"] = s.config_id.str.split("/").str[1]
    s["noise"] = s.config_id.str.extract(r"/n([\d.]+)$")[0].astype(float)
    q = s[s.H == 128]
    rows = []
    for (sysname, nz), g in q.groupby(["system", "noise"]):
        r = {"system": sysname, "noise (% of scale)": nz * 100}
        for fc in SIZES:
            v = g[g.forecaster == fc].mase_ratio_oracle_gmean
            r[SIZE_LABEL[fc]] = float(v.iloc[0]) if len(v) else float("nan")
        rows.append(r)
    table("chaos_noise", pd.DataFrame(rows), src, "geometric mean over 16 seeds of MASE ratio to oracle, context 2048, horizon 128")
    mg = q[q.system == "mackey_glass_tau17"]
    for nz, k in [(0.001, "lo"), (0.1, "hi")]:
        v = mg[mg.noise == nz].mase_ratio_oracle_gmean
        num(f"noise_mg_{k}_min", float(v.min()), src, f"Mackey-Glass tau=17, noise {nz}, min over sizes")
        num(f"noise_mg_{k}_max", float(v.max()), src, f"Mackey-Glass tau=17, noise {nz}, max over sizes")


def readouts():
    src = "results/mechanism/readouts/summary.csv"
    s = pd.read_csv(p(src))
    s = s[s.valid]
    for study, kv, k in [("mean_jump_d", 16.0, "mj16"), ("mean_jump_d", 256.0, "mj256"), ("freq_jump_d", 64.0, "fj64"),
                         ("mackey_tau", 17.0, "mg17")]:
        g = s[(s.study == study) & (s.knob_value == kv)]
        for col, name in [("output_r2", "out"), ("best_readout_r2", "ro"), ("mase_ratio_oracle_gmean", "ratio"),
                          ("step1_output_r2", "s1out"), ("step1_readout_r2", "s1ro"), ("attn_post_change_mean", "post"),
                          ("template_corr_model", "tm"), ("template_corr_oracle", "to")]:
            if col in g and g[col].notna().any():
                num(f"ro_{k}_{name}_min", float(g[col].min()), src, f"{study} {kv}: min over the four sizes of {col}", "{:.3f}" if "r2" in col or "corr" in col else "{:.2f}")
                num(f"ro_{k}_{name}_max", float(g[col].max()), src, f"{study} {kv}: max over the four sizes of {col}", "{:.3f}" if "r2" in col or "corr" in col else "{:.2f}")
    g = s[(s.study == "in_prior") & (s.knob_value == 24.0)]
    for col, name in [("output_r2", "out"), ("best_readout_r2", "ro")]:
        num(f"ro_ip_{name}_min", float(g[col].min()), src, f"in_prior period 24: min over sizes of {col}", "{:.3f}")
        num(f"ro_ip_{name}_max", float(g[col].max()), src, f"in_prior period 24: max over sizes of {col}", "{:.3f}")
    for size in ("5M", "209M"):
        g = s[(s.study == "mackey_tau") & (s.knob_value == 17.0) & (s["size"] == size)]
        num(f"ro_mg17_tm_{size}", float(g.template_corr_model.iloc[0]), src, f"Mackey-Glass tau=17, {size}: correlation of Timestral's median with the periodic template", "{:.2f}")
    rows = []
    for study, kv in [("mean_jump_d", 16.0), ("mean_jump_d", 64.0), ("mean_jump_d", 256.0), ("freq_jump_d", 64.0),
                      ("freq_jump_d", 1024.0), ("mackey_tau", 17.0), ("lorenz_rho", 28.0), ("in_prior", 24.0)]:
        for size in ("5M", "209M"):
            g = s[(s.study == study) & (s.knob_value == kv) & (s["size"] == size)]
            if not len(g):
                continue
            r = g.iloc[0]
            rows.append({"setting": f"{study} {kv:g}", "size": size, "MASE / oracle": r.mase_ratio_oracle_gmean,
                         "output R2": r.output_r2, "best readout R2": r.best_readout_r2,
                         "step-1 output R2": r.get("step1_output_r2", np.nan), "step-1 readout R2": r.get("step1_readout_r2", np.nan)})
    table("readouts", pd.DataFrame(rows), src, "300 samples per setting; cross-validated R2 of predicting the oracle's expected change (first patch, or step 1) from Timestral's output vs. from the best per-layer linear readout of the first forecast token", "{:.3f}")


def selfval():
    src = "results/fix/selfval_confirm/frozen_rule_confirmation.csv"
    r = pd.read_csv(p(src))
    rows = []
    for fam in ("chaos", "regime", "tails", "in_prior"):
        row = {"family": fam}
        for fc in SIZES:
            v = r[(r.family == fam) & (r["size"] == SIZE_LABEL[fc])].mase_fix_over_timestral
            row[SIZE_LABEL[fc]] = float(v.iloc[0]) if len(v) else float("nan")
        v = r[r.family == fam].switched_share
        row["share switched (range over sizes)"] = f"{v.min():.3f}-{v.max():.3f}"
        rows.append(row)
    table("selfval", pd.DataFrame(rows), src, "geometric mean over fresh samples of MASE(fix) / MASE(Timestral); frozen rule K=2, m=0.7", "{:.3f}")
    c = r[r.family == "chaos"].set_index("size").mase_fix_over_timestral
    for size in ("5M", "209M"):
        num(f"sv_chaos_{size}", float(c[size]), src, f"chaos, {size}: MASE(fix)/MASE(Timestral) on fresh seeds", "{:.3f}")
    t = r[r.family == "tails"].mase_fix_over_timestral
    num("sv_tails_max", float(t.max()), src, "tails: max over sizes of MASE(fix)/MASE(Timestral)", "{:.3f}")
    s2 = pd.read_csv(p("results/fix/selfval/chosen_rule_test_half.csv"))
    tt = s2[s2.family == "tails"].mase_fix_over_timestral
    num("sv_firstrule_tails_min", float(tt.min()), "results/fix/selfval/chosen_rule_test_half.csv", "first pre-set rule, odd seeds: min over sizes, tails", "{:.3f}")
    num("sv_firstrule_tails_max", float(tt.max()), "results/fix/selfval/chosen_rule_test_half.csv", "first pre-set rule, odd seeds: max over sizes, tails", "{:.3f}")


def regime():
    src = "results/map/regime/summary_all_sizes.csv"
    s = load_summary(src)
    q = s[(s["T"] == 2048) & (s.H == 128)].copy()
    q["d"] = pd.to_numeric(q.config_id.str.extract(r"/d(\d+)$")[0], errors="coerce")
    rows = []
    for prefix, label in [("regime/mean_jump/context/delta4/", "level +4 sd"), ("regime/frequency_jump/context/ratio2/", "period x2"),
                          ("regime/variance_jump/context/factor0.25/", "noise sd x0.25")]:
        for fc in ("timestral-5M", "timestral-209M"):
            r = {"switch": label, "size": SIZE_LABEL[fc]}
            for d in (16, 64, 256, 1024):
                v = q[(q.config_id == f"{prefix}d{d}") & (q.forecaster == fc)].mase_ratio_oracle_gmean
                r[f"{d} steps ago"] = float(v.iloc[0]) if len(v) else float("nan")
            rows.append(r)
    table("regime", pd.DataFrame(rows), src, "geometric mean over 8 seeds of MASE ratio to oracle, context 2048, horizon 128")
    for gen, kv, d, k in [("mean_jump", "delta4", 16, "mj4_16"), ("mean_jump", "delta4", 1024, "mj4_1024"), ("frequency_jump", "ratio3", 64, "fj3_64")]:
        for fc in ("timestral-5M", "timestral-209M"):
            v = float(q[(q.config_id == f"regime/{gen}/context/{kv}/d{d}") & (q.forecaster == fc)].mase_ratio_oracle_gmean.iloc[0])
            num(f"reg_{k}_{SIZE_LABEL[fc]}", v, src, f"{gen} {kv} switch {d} steps back, {fc}, T=2048, H=128")
    hz = q[q.config_id.str.contains("/horizon/")].mase_ratio_oracle_gmean
    num("reg_horizon_min", hz.min(), src, "switches during the horizon (all generators, knob values, sizes): min")
    num("reg_horizon_max", hz.max(), src, "same: max")
    mk = q[q.generator == "markov_ar"].mase_ratio_oracle_gmean
    num("reg_markov_min", mk.min(), src, "Markov-switching AR (all switching rates, sizes): min")
    num("reg_markov_max", mk.max(), src, "same: max")
    asrc = "results/map/regime/adoption.csv"
    a = pd.read_csv(p(asrc))
    a = a[(a["T"] == 2048) & (a.delta.astype(str) == "all")]
    for size in ("5M", "209M"):
        for d in (16, 64, 256):
            v = float(a[(a["size"] == size) & (a.d == d)].adoption_mean.iloc[0])
            num(f"adopt_{size}_{d}", v, asrc, f"level adoption at H<=16, all deltas pooled, {size}, switch {d} steps back, T=2048")


def covariates():
    src = "results/covariates/summary.csv"
    s = pd.read_csv(p(src))
    s = s[s["T"] == 2048]

    def row(system, kv, size, H, cond):
        r = s[(s.system == system) & (s.knob_value.astype(float) == kv) & (s["size"] == size) & (s.H == H) & (s.condition == cond)]
        assert len(r) == 1, (system, kv, size, H, cond, len(r))
        return r.iloc[0]

    # Table: past covariate gain vs the gain available to the matched oracle, H = 16
    cases = [("LaggedDriver", 32.0, "lag 32"), ("NonlinearDriver", 0.5, "nonlinearity 0.5"), ("HiddenDriver", 0.05, "x noise 0.05")]
    rows = []
    for system, kv, lab in cases:
        r = {"system": system, "knob": lab}
        for size in ("5M", "28M", "123M", "209M"):
            x = row(system, kv, size, 16, "past_cov")
            r[size] = f"{x.covariate_gain_gmean:.2f} / {x.oracle_gain_gmean:.2f}"
        rows.append(r)
    table("cov_past", pd.DataFrame(rows), src, "past_cov rows at T=2048, H=16: covariate gain (MASE(uni)/MASE(past_cov)) / oracle gain (MASE(uni)/MASE(matched oracle)), geometric means over samples")
    lag = row("LaggedDriver", 32.0, "209M", 16, "past_cov")
    num("cov_lag32_209M_gain", lag.covariate_gain_gmean, src, "LaggedDriver lag 32, 209M, H=16, past_cov covariate gain")
    num("cov_lag32_209M_ogain", lag.oracle_gain_gmean, src, "LaggedDriver lag 32, 209M, H=16, matched-oracle gain")
    # known-future covariate on LaggedDriver: gain decays with lag
    rows = []
    for kv in (0.0, 8.0, 32.0, 128.0, 512.0):
        r = {"lag": int(kv)}
        for size in ("5M", "28M"):
            x = row("LaggedDriver", kv, size, 16, "known_cov")
            r[f"{size} gain"] = x.covariate_gain_gmean
            r[f"{size} oracle gain"] = x.oracle_gain_gmean
        rows.append(r)
    table("cov_known_lag", pd.DataFrame(rows), src, "LaggedDriver known_cov, T=2048, H=16: covariate gain and matched-oracle gain (geometric means)")
    for kv in (0.0, 512.0):
        num(f"cov_known_lag{int(kv)}_28M", row("LaggedDriver", kv, "28M", 16, "known_cov").covariate_gain_gmean, src, f"LaggedDriver lag {int(kv)}, 28M, H=16, known_cov covariate gain")
    og = [row("LaggedDriver", kv, "28M", 16, "known_cov").oracle_gain_gmean for kv in (0.0, 8.0, 32.0, 128.0, 512.0)]
    num("cov_known_ogain_min", min(og), src, "min over lags of the matched-oracle gain, LaggedDriver known_cov, 28M, H=16")
    num("cov_known_ogain_max", max(og), src, "max of the same")
    # joint vs past_cov (two-way vs one-way routing of the same information)
    jv = s[s.condition == "joint"].dropna(subset=["joint_vs_past_gmean"])
    for system in ("FeedbackLinear", "FeedbackPredPrey", "LaggedDriver"):
        g = jv[jv.system == system]
        lo = g.loc[g.joint_vs_past_gmean.idxmin()]
        num(f"cov_jp_{system}_min", lo.joint_vs_past_gmean, src, f"{system}: smallest joint/past_cov (geometric mean) over knobs, sizes, H at T=2048")
        num(f"cov_jp_{system}_min_lo", lo.joint_vs_past_lo95, src, "its 95% bootstrap lower bound")
        num(f"cov_jp_{system}_min_hi", lo.joint_vs_past_hi95, src, "its 95% bootstrap upper bound")
        num(f"cov_jp_{system}_max", g.joint_vs_past_gmean.max(), src, f"{system}: largest joint/past_cov over knobs, sizes, H at T=2048")
    jall = jv.joint_vs_past_gmean
    num("cov_jp_share_within", 100 * float(((jall > 0.95) & (jall < 1.05)).mean()), src, "percent of joint rows (all systems, knobs, sizes, H, T=2048) with joint/past_cov in (0.95, 1.05)", "{:.0f}")
    num("cov_jp_n", len(jall), src, "number of such rows", "{:d}")
    # LeakyKnown: ratio to the matched oracle = oracle gain / covariate gain (exact for geometric means over the same samples)
    rows = []
    for size in ("5M", "28M", "123M", "209M"):
        r = {"size": size}
        for H in (16, 128, 1024):
            q = s[(s.system == "LeakyKnown") & (s.knob_value.astype(float) == 0.05) & (s["size"] == size) & (s.H == H) & (s.condition == "known_cov")]
            r[f"H={H}"] = float(q.oracle_gain_gmean.iloc[0] / q.covariate_gain_gmean.iloc[0]) if len(q) else float("nan")
        rows.append(r)
    table("cov_leaky", pd.DataFrame(rows), src, "LeakyKnown noise ratio 0.05, known_cov, T=2048: MASE / matched-oracle MASE = oracle gain / covariate gain (geometric means over the same samples)")
    lk = pd.DataFrame(rows).set_index("size")
    num("cov_leaky_5M_16", lk.loc["5M", "H=16"], src, "LeakyKnown 0.05, 5M, H=16, ratio to matched oracle")
    num("cov_leaky_209M_16", lk.loc["209M", "H=16"], src, "LeakyKnown 0.05, 209M, H=16, ratio to matched oracle")
    num("cov_leaky_5M_1024", lk.loc["5M", "H=1024"], src, "LeakyKnown 0.05, 5M, H=1024, ratio to matched oracle")


def realdata():
    src = "results/regression_baseline/summary.csv"
    s = pd.read_csv(p(src))
    a = s[(s.group == "ALL") & (s.basis == "common")]
    for st in ("real", "analogues"):
        for fc in ("timestral-5M", "timestral-209M", "ets", "arima"):
            v = float(a[(a.set == st) & (a.forecaster == fc)].mase_rel_snaive.iloc[0])
            num(f"rd_{st}_{fc.replace('timestral-', '')}", v, src, f"{st} set, common windows: geometric mean over groups of mean MASE / seasonal-naive mean MASE, {fc}")
        num(f"rd_n_{st}", int(a[a.set == st].n.iloc[0]), src, f"windows in the {st} set (common basis)", "{:d}")
    asrc = "results/regression_baseline/analogue_vs_synthetic.csv"
    d = pd.read_csv(p(asrc))
    d = d[(d.level == "family") & (d.metric == "mase")]
    names = {"delay_feedback": "delay feedback", "coupled_oscillators": "coupled oscillators", "heavy_tails": "heavy tails", "regime_switches": "regime switches"}
    rows = []
    for fam, lab in names.items():
        r = {"family": lab, "real windows": int(d[d.family_analogue == fam].n_windows.iloc[0])}
        for size in ("5M", "209M"):
            q = d[(d.family_analogue == fam) & (d.forecaster == f"timestral-{size}")].iloc[0]
            r[f"real {size}"] = q.real_ratio_to_best_classical_gmean
            r[f"synthetic {size}"] = q.synthetic_T2048_H128
        rows.append(r)
    table("rd_analogue", pd.DataFrame(rows), asrc, "Timestral MASE / best classical MASE (geometric mean): real analogue windows vs the synthetic family at T=2048, H=128")


# ------------------------------------------------------------------------------ sections whose results land later
# Each key below may be referenced in src.md before its result file exists; until then it renders as a
# "results pending" note naming the file. Once the file exists, the function that defines the key must fill it.
OPTIONAL = {
    "search_comparison": "results/search/comparison.csv",
    "search_transfer": "results/search/transfer.csv",
    "ft_local_lofo": "results/finetune/local/summary.csv",
    "ft_local_knob": "results/finetune/local/summary.csv",
    "ft_local_regression": "results/finetune/local/summary.csv",
    "ft_colab": "results/finetune/colab/summary.csv",
    "nn_one_job": "results/neurons/one_job_curves.csv",
    "nn_steering": "results/neurons/steering_results.csv",
    "nn_failure_link": "results/neurons/failure_link.csv",
}


def generic_table(key, path, how, max_rows=40, cols=None):
    d = pd.read_csv(p(path))
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    table(key, d.head(max_rows), path, how + (f" (first {max_rows} of {len(d)} rows)" if len(d) > max_rows else ""), "{:.3f}")


def search_results():
    f = "results/search/comparison.csv"
    if os.path.exists(p(f)):
        generic_table("search_comparison", f, "per search method, from results/search/comparison.csv")
    f = "results/search/transfer.csv"
    if os.path.exists(p(f)):
        generic_table("search_transfer", f, "elites re-scored on larger sizes, from results/search/transfer.csv")


def finetune_results():
    f = "results/finetune/local/summary.csv"
    if os.path.exists(p(f)):
        s = pd.read_csv(p(f))
        label = lambda r: r.method + (f"+mix{r.replay_frac:g}" if r.replay_frac > 0 else "")
        for proto, key in (("lofo", "ft_local_lofo"), ("knob", "ft_local_knob")):
            q = s[(s.protocol == proto) & (s.role == "test")].copy()
            if len(q):
                q["method"] = [label(r) for r in q.itertuples()]
                rows = [{"held out": r.holdout, "method": r.method, "n": int(r.n), "ratio to oracle before": r.pre_mase_ratio,
                         "after": r.ft_mase_ratio, "paired MASE after/before": r.paired_mase_ft_over_pre,
                         "95% low": r.paired_mase_lo95, "95% high": r.paired_mase_hi95} for r in q.sort_values(["holdout", "method"]).itertuples()]
                table(key, pd.DataFrame(rows), f, f"5M, protocol {proto}, held-out test sets: geometric-mean MASE ratio to oracle before and after, paired ratio with 95% interval", "{:.3f}")
        q = s[s.role == "regression"].copy() if "role" in s else pd.DataFrame()
        if len(q):
            q["method"] = [label(r) for r in q.itertuples()]
            rows = [{"run": r.run_id, "set": r.eval_set, "n": int(r.n), "paired MASE after/before": r.paired_mase_ft_over_pre,
                     "95% low": r.paired_mase_lo95, "95% high": r.paired_mase_hi95} for r in q.itertuples()]
            table("ft_local_regression", pd.DataFrame(rows), f, "regression sets: paired MASE ratio fine-tuned / pretrained", "{:.3f}")
    f = "results/finetune/colab/summary.csv"
    if os.path.exists(p(f)):
        generic_table("ft_colab", f, "GPU runs, from results/finetune/colab/summary.csv")


def label_group_share(lg, size, grp):
    """Share of a size's one-job poles whose label is in group `grp` (label_groups.csv layout, see scripts/neurons_summarize.py)."""
    q = lg[lg["size"] == size]
    for share_col in ("share_of_one_job", "share", "frac"):
        if share_col in q.columns and "group" in q.columns:
            r = q[q.group == grp]
            return float(r[share_col].iloc[0]) if len(r) else None
    if "group" in q.columns and "n_one_job" in q.columns:
        tot = q.n_one_job.sum()
        r = q[q.group == grp]
        return float(r.n_one_job.iloc[0] / tot) if len(r) and tot else None
    return None


def neuron_results():
    d = "results/neurons"
    f = f"{d}/scaling.csv"
    if os.path.exists(p(f)):
        sc = pd.read_csv(p(f)).set_index("size").loc[["5M", "28M", "123M", "209M"]]
        rows = [{"size": sz, "live poles": int(r.n_live), "one job": r.one_job, "95% low": r.one_job_lo, "95% high": r.one_job_hi,
                 "one job, held-out series": r.one_job_heldout, "mixed": r.mixed, "unexplained": r.unexplained} for sz, r in sc.iterrows()]
        table("nn_one_job", pd.DataFrame(rows), f, "share of live poles per class at theta 0.7, floor 0.1 (main corpus, all layers, both poles); interval: bootstrap over poles", "{:.3f}")
        for sz in ("5M", "209M"):
            num(f"nn_poles_{sz}", int(sc.loc[sz, "n_poles"]), f, f"{sz}: poles (two per neuron)", "{:,d}")
            num(f"nn_unexpl_{sz}", 100 * float(sc.loc[sz, "unexplained"]), f, f"{sz}: percent of live poles unexplained at floor 0.1", "{:.0f}")
            num(f"nn_onejob_{sz}", 100 * float(sc.loc[sz, "one_job"]), f, f"{sz}: percent of live poles that are one-job at theta 0.7, floor 0.1", "{:.1f}")
            num(f"nn_mixed_{sz}", 100 * float(sc.loc[sz, "mixed"]), f, f"{sz}: percent mixed at theta 0.7, floor 0.1", "{:.0f}")
    f = f"{d}/label_groups.csv"
    if os.path.exists(p(f)):
        lg = pd.read_csv(p(f))
        for sz in ("5M", "209M"):
            for grp in ("position", "chaos"):
                v = label_group_share(lg, sz, grp)
                if v is not None:
                    num(f"nn_grp_{grp}_{sz}", 100 * v, f, f"percent of {sz}'s one-job poles whose label is in the {grp} group", "{:.0f}")
    f = f"{d}/context_comparison.csv"
    if os.path.exists(p(f)):
        c = pd.read_csv(p(f))
        rows = []
        for ctx in ("main", "short", "long", "xlong"):
            for sz in ("5M", "209M"):
                r = c[(c.context == ctx) & (c["size"] == sz)].iloc[0]
                rows.append({"corpus": ctx, "size": sz, "one job": r.one_job, "unexplained": r.unexplained, "mean E": r.E_mean,
                             "same label as main": r.same_label_as_main})
        table("nn_context", pd.DataFrame(rows), f, "the same scoring on corpora of different context length (theta 0.7, floor 0.1); E = best of single, pair and additive readout R^2", "{:.3f}")
        oj = c[c.context != "main"].one_job
        num("nn_ctx_onejob_min", 100 * float(oj.min()), f, "min over sizes and non-main corpora of the one-job percent", "{:.0f}")
        num("nn_ctx_onejob_max", 100 * float(oj.max()), f, "max of the same", "{:.0f}")
        ag = c[c.context != "main"].same_label_as_main
        num("nn_ctx_agree_min", 100 * float(ag.min()), f, "min over sizes and non-main corpora of the percent of poles whose best label matches the main corpus", "{:.0f}")
        num("nn_ctx_agree_max", 100 * float(ag.max()), f, "max of the same", "{:.0f}")
    f = f"{d}/followup/corpus_library_control.csv"
    if os.path.exists(p(f)):
        b = pd.read_csv(p(f))
        b = b[(b.theta == 0.7) & (b.floor == 0.1)]
        g = b.groupby(["size", "role", "corpus", "n_series", "library"]).agg(one_job=("one_job", "mean"), lo=("one_job", "min"),
                                                                            hi=("one_job", "max"), E=("E_mean", "mean"),
                                                                            agree=("same_label_as_reference", "mean"), runs=("tag", "count")).reset_index()
        g = g[g.role.isin(["reference", "common_library", "series_count"])].sort_values(["size", "corpus", "n_series"], ascending=[False, True, False])
        rows = [{"size": r.size, "corpus": r.corpus, "series": int(r.n_series), "library": r.library, "runs": int(r.runs),
                 "one job (mean)": r.one_job, "min": r.lo, "max": r.hi, "mean E": r.E, "same label as reference": r.agree} for r in g.itertuples()]
        table("nn_series_control", pd.DataFrame(rows), f, "theta 0.7, floor 0.1; main corpus subsampled at random to fewer series (3 seeds at 5M), and each corpus scored with the variable library common to all corpora", "{:.3f}")
        ref = b[(b.role == "reference") & (b["size"] == "5M")].iloc[0]
        com = b[(b.role == "common_library") & (b["size"] == "5M") & (b.corpus == "main")].iloc[0]
        num("nnb_ref_5M", 100 * ref.one_job, f, "5M, main corpus, all 3100 series, full library: one-job percent", "{:.1f}")
        num("nnb_common_5M", 100 * com.one_job, f, "5M, main corpus, common library: one-job percent", "{:.1f}")
        for n in (1034, 48):
            q = b[(b.role == "series_count") & (b["size"] == "5M") & (b.n_series == n)]
            num(f"nnb_sub{n}_5M_min", 100 * q.one_job.min(), f, f"5M, main corpus subsampled to {n} series: min over 3 seeds of the one-job percent", "{:.0f}")
            num(f"nnb_sub{n}_5M_max", 100 * q.one_job.max(), f, f"5M, main corpus subsampled to {n} series: max over 3 seeds", "{:.0f}")
        for c in ("short", "long", "xlong"):
            q = b[(b.role == "common_library") & (b["size"] == "5M") & (b.corpus == c)]
            if len(q):
                num(f"nnb_{c}_common_5M", 100 * float(q.one_job.iloc[0]), f, f"5M, {c} corpus, common library: one-job percent", "{:.0f}")
                num(f"nnb_{c}_n", int(q.n_series.iloc[0]), f, f"series in the {c} corpus", "{:,d}")
        q = b[(b.role == "series_count") & (b["size"] == "5M") & (b.n_series == 620)]
        num("nnb_sub620_5M_min", 100 * q.one_job.min(), f, "5M, main corpus subsampled to 620 series: min over 3 seeds of the one-job percent", "{:.0f}")
        num("nnb_sub620_5M_max", 100 * q.one_job.max(), f, "max of the same", "{:.0f}")
        sub = b[(b.role == "series_count")]
        num("nnb_n_main", int(b[b.role == "reference"].n_series.max()), f, "series in the main corpus", "{:,d}")
        num("nnb_n_mid", int(sub.n_series.max()), f, "largest subsample of the main corpus", "{:,d}")
        num("nnb_n_small", int(sub.n_series.min()), f, "smallest subsample of the main corpus", "{:,d}")
        ks = sorted(int(x.replace("random", "")) for x in b[b.role == "library_size"].library.unique())
        num("nnb_k_lo", ks[0], f, "smallest random library size", "{:d}")
        num("nnb_k_hi", ks[-1], f, "largest random library size", "{:d}")
        lib = b[(b.role == "library_size") & (b["size"] == "5M")]
        num("nnb_lib_min", 100 * lib.one_job.min(), f, "5M, main corpus, random libraries of 10, 25 or 50 variables: min one-job percent over 9 runs", "{:.1f}")
        num("nnb_lib_max", 100 * lib.one_job.max(), f, "max of the same", "{:.1f}")
    fu = f"{d}/followup"
    f = f"{fu}/direction_steering.csv"
    if os.path.exists(p(f)):
        ds = pd.read_csv(p(f))
        q = ds[(ds.variable == "level") & (ds.test_set == "mixed")]
        nz = ds[(ds.variable == "noise") & (ds.test_set == "mixed") & (ds.block_name == "late") & (ds.site == "F") & (ds.kind == "pooled")]
        rows = []
        for sz in ("5M", "28M", "123M", "209M"):
            g = q[q["size"] == sz]
            n_ = nz[nz["size"] == sz]
            rows.append({"size": sz, "level cells": len(g), "pre-registered match": int(g.orig_match.sum()), "calibrated match": int(g.cal_match.sum()),
                         "largest gain": float(g.gain.max()), "noise direction, late block: effect": float(n_.effect_size.iloc[0]) if len(n_) else float("nan"),
                         "matches": bool(n_.orig_match.iloc[0]) if len(n_) else False})
            num(f"nnA_cells_{sz}", int(g.orig_match.sum()), f, f"{sz}: plain-level direction cells (of {len(g)}) matching the pre-registered criterion", "{:d}")
            num(f"nnA_gain_{sz}", float(g.gain.max()), f, f"{sz}: largest gain (forecast effect / readout move) over the plain-level direction cells", "{:.3f}")
            num(f"nnA_cal_{sz}", int(g.cal_match.sum()), f, f"{sz}: plain-level direction cells matching the calibrated criterion", "{:d}")
        num("nnA_ncells", len(q[q["size"] == "5M"]), f, "plain-level direction cells per size", "{:d}")
        table("nn_direction", pd.DataFrame(rows), f, "positive control: residual-stream steering along ridge readout directions (mixed test series); gain = effect on the forecast level / move of the readout, both in context standard deviations", "{:.3f}")
    f = f"{fu}/group_summary.csv"
    if os.path.exists(p(f)):
        gs = pd.read_csv(p(f))
        lt = gs[gs.pool == "label_top"].sort_values(["size", "k"])
        rows = [{"size": r.size, "k": int(r.k), "groups": int(r.n_groups), "labeled, pre-registered": r.labeled_orig_match,
                 "random, pre-registered": r.random_orig_match, "labeled, calibrated": r.labeled_cal_match, "random, calibrated": r.random_cal_match}
                for r in lt.itertuples()]
        table("nn_groups", pd.DataFrame(rows), f, "group steering: share of groups (top-k poles of one label, jointly scaled) matching their prediction, against random groups of the same size", "{:.3f}")
        num("nnG_orig_max", 100 * float(lt.labeled_orig_match.max()), f, "largest share of labeled groups matching the pre-registered criterion over sizes and k", "{:.0f}")
        big = lt[lt.n_groups >= 10]
        num("nnG_cal_min", float(big.labeled_cal_match.min()), f, "min over size-k cells with at least 10 labeled groups of the calibrated match share of labeled (label_top) groups", "{:.2f}")
        num("nnG_cal_max", float(big.labeled_cal_match.max()), f, "max of the same", "{:.2f}")
        num("nnG_rcal_min", float(big.random_cal_match.min()), f, "min over the same cells of the calibrated match share of random groups", "{:.3f}")
        num("nnG_rcal_max", float(big.random_cal_match.max()), f, "max of the same", "{:.3f}")
        num("nnG_small_cells", int((lt.n_groups < 10).sum()), f, "size-k cells with fewer than 10 labeled groups", "{:d}")
    f = f"{fu}/motif_clusters.csv"
    if os.path.exists(p(f)):
        mc = pd.read_csv(p(f))
        mm = mc[mc.role.isin(["full_main", "subsample"])]
        g = mm.groupby(["size", "n_series"]).agg(seeds=("tag", "count"), one_job=("one_job_0.7", "mean"), cohesive=("cohesive_top6", "mean")).reset_index()
        g["order"] = g["size"].map({"5M": 0, "28M": 1, "123M": 2, "209M": 3})
        g = g.sort_values(["order", "n_series"], ascending=[True, False])
        rows = [{"size": r.size, "series": int(r.n_series), "seeds": int(r.seeds), "one job (theta 0.7)": r.one_job, "motif-cohesive": r.cohesive} for r in g.itertuples()]
        table("nn_motif", pd.DataFrame(rows), f, "main corpus, all series and random subsamples (mean over seeds): known-variable one-job share and the share of poles whose top exemplars mostly share one of 16 shape clusters", "{:.3f}")
        full = mm[mm.n_series == 3100]
        small = mm[mm.n_series == 48]
        num("nnC_coh_full_min", float(full.cohesive_top6.min()), f, "motif-cohesive share, full main corpus: min over sizes", "{:.2f}")
        num("nnC_coh_full_max", float(full.cohesive_top6.max()), f, "max of the same", "{:.2f}")
        num("nnC_coh_48_min", float(small.groupby("size").cohesive_top6.mean().min()), f, "motif-cohesive share at 48 series (mean over seeds): min over sizes", "{:.2f}")
        num("nnC_coh_48_max", float(small.groupby("size").cohesive_top6.mean().max()), f, "max of the same", "{:.2f}")
        num("nnC_oj_full_min", float(full["one_job_0.7"].min()), f, "one-job share, full main corpus: min over sizes", "{:.3f}")
        num("nnC_oj_full_max", float(full["one_job_0.7"].max()), f, "max of the same", "{:.3f}")
        num("nnC_oj_48_min", float(small.groupby("size")["one_job_0.7"].mean().min()), f, "one-job share at 48 series (mean over seeds): min over sizes", "{:.2f}")
        num("nnC_oj_48_max", float(small.groupby("size")["one_job_0.7"].mean().max()), f, "max of the same", "{:.2f}")
        nm = pd.read_csv(p(f"{fu}/motif_cluster_names.csv"))
        num("nnC_osc_clusters", int((nm.name == "oscillation").sum()), f"{fu}/motif_cluster_names.csv", "shape clusters that the fixed naming rule calls oscillation", "{:d}")
        num("nnC_clusters", len(nm), f"{fu}/motif_cluster_names.csv", "shape clusters", "{:d}")
    f = f"{d}/steering_summary.csv"
    if os.path.exists(p(f)):
        s2 = pd.read_csv(p(f))
        s2 = s2[s2.group == "all"]
        rows = [{"size": r.size, "poles": r.role, "n": int(r.n), "matched": int(r.n_match), "match rate": r.match_rate, "95% high": r.ci_hi,
                 "monotone share": r.monotone_rate} for r in s2.itertuples()]
        table("nn_steering", pd.DataFrame(rows), f, "single-pole steering (scales 0, 0.5, 2, 4) judged against the pre-registered predictions; controls are random poles with a random predicted direction", "{:.3f}")
        num("nn_steer_matches", int(s2[s2.role == "labeled"].n_match.sum()), f, "labeled poles whose steering matched the pre-registered prediction, all sizes", "{:d}")
        num("nn_steer_control_n", int(s2[s2.role == "control"].n.sum()), f, "random control poles steered, all sizes", "{:,d}")
        num("nn_steer_control_matches", int(s2[s2.role == "control"].n_match.sum()), f, "control poles that met the match criterion, all sizes", "{:d}")
        sr = pd.read_csv(p(f"{d}/steering_results.csv"))
        lab = sr[(sr.role == "labeled") & sr.predicted.astype(bool)] if "predicted" in sr else sr[sr.role == "labeled"]
        num("nn_steer_max_effect", float(lab.effect_size.abs().max()), f"{d}/steering_results.csv", "largest |effect(scale 4) - effect(scale 0)| over steered labeled poles, in context standard deviations", "{:.3f}")
        num("nn_steer_n_labeled", int(s2[s2.role == "labeled"].n.sum()), f, "steered labeled poles over all sizes", "{:d}")
    f = f"{d}/failure_link.csv"
    if os.path.exists(p(f)):
        fl = pd.read_csv(p(f))
        rows = []
        for sz in ("5M", "28M", "123M", "209M"):
            g = fl[fl["size"] == sz]
            en = g[(g.analysis == "elevated") & (g.set == "chaos")].enrichment
            ab = g[(g.analysis == "ablate") & (g.set == "periodic_elevated") & (g.scale == 0.0)]
            rnd = g[(g.analysis == "ablate") & (g.set == "random_same_layers") & (g.scale == 0.0)]
            rows.append({"size": sz, "chaos-label enrichment among elevated poles": float(en.iloc[0]) if len(en) else float("nan"),
                         "MASE ratio before": float(ab.mase_before.iloc[0]) if len(ab) else float("nan"),
                         "after ablating periodic poles": float(ab.mase_after.iloc[0]) if len(ab) else float("nan"),
                         "after ablating random poles": float(rnd.mase_after.iloc[0]) if len(rnd) else float("nan")})
        g = fl[fl["size"] == "209M"]
        for set_name, tag in (("periodic_elevated", "elev"), ("random_same_layers", "rand")):
            q = g[(g.analysis == "ablate") & (g.set == set_name) & (g.scale == 0.0)]
            if len(q):
                num(f"nn_abl_{tag}_before_209M", float(q.mase_before.iloc[0]), f, "209M: MASE ratio to oracle on failing contexts before ablation")
                num(f"nn_abl_{tag}_after_209M", float(q.mase_after.iloc[0]), f, f"209M: MASE ratio after setting the {set_name} poles to zero")
                if "n_poles" in q and q.n_poles.notna().any():
                    num(f"nn_abl_{tag}_n_209M", int(q.n_poles.iloc[0]), f, f"209M: number of poles set to zero ({set_name})", "{:d}")
        table("nn_failure_link", pd.DataFrame(rows), f, "failing chaos contexts: enrichment of chaos-labeled poles among poles elevated on failing vs reference contexts, and the forecast's MASE ratio to the oracle before and after ablating the elevated periodic-label poles (or as many random poles in the same layers)", "{:.3f}")


def bench():
    import json as _json

    order = ["oracle", "timestral-209M", "timestral-123M", "timestral-28M", "timestral-5M", "ets", "arima", "seasonal_naive"]
    for set_name in ("diag", "lb"):
        rows, srcs = [], []
        for m in order:
            f = f"bench/leaderboard/{m}_{set_name}_all/summary.json"
            if not os.path.exists(p(f)):
                continue
            o = _json.load(open(p(f)))["overall"]
            srcs.append(f)
            rows.append({"model": m, "series": int(o["n"]), "MASE / seasonal naive": o["mase_rel_snaive"],
                         "sCRPS / seasonal naive": o["scrps_rel_snaive"], "MASE / oracle": o["mase_rel_oracle"],
                         "80% coverage": o["cov80"]})
        if rows:
            table(f"bench_{set_name}", pd.DataFrame(rows), ", ".join(srcs), "bench/eval.py summaries (geometric means over series)", "{:.3f}")
    f = "bench/leaderboard/timestral-209M_diag_all/summary.json"
    if os.path.exists(p(f)):
        ft = _json.load(open(p(f)))["by_failure_type"]
        rows = [{"failure type": k, "series": int(v["n"]), "209M MASE / oracle": v["mase_rel_oracle"],
                 "209M MASE / seasonal naive": v["mase_rel_snaive"]} for k, v in sorted(ft.items())]
        table("bench_diag_ft", pd.DataFrame(rows), f, "Timestral-209M on OffPrior-Diag by failure type", "{:.3f}")
        for k in ("coupled_oscillators", "delay_oscillator", "chaos_short_reach", "smooth_chaos", "recent_switch"):
            num(f"bench_ft_{k}", ft[k]["mase_rel_oracle"], f, f"Timestral-209M, OffPrior-Diag, failure type {k}: geometric mean of MASE / oracle MASE", "{:.2f}")
        ctrl = [v["mase_rel_oracle"] for k, v in ft.items() if k.startswith("control_") or k in ("heavy_tail_walk", "bursty", "multifractal")]
        num("bench_ft_ctrl_min", min(ctrl), f, "min over control types and heavy-tail/bursty/multifractal types of 209M MASE / oracle", "{:.2f}")
        num("bench_ft_ctrl_max", max(ctrl), f, "max of the same", "{:.2f}")
    for set_name in ("diag", "lb"):
        for m in ("timestral-5M", "timestral-209M", "ets", "arima", "seasonal_naive"):
            f = f"bench/leaderboard/{m}_{set_name}_all/summary.json"
            if os.path.exists(p(f)):
                o = _json.load(open(p(f)))["overall"]
                tag = m.replace("timestral-", "")
                num(f"bench_{set_name}_{tag}_oracle", o["mase_rel_oracle"], f, f"{m} on {set_name}: geometric mean over series of MASE / oracle MASE", "{:.2f}")
                num(f"bench_{set_name}_{tag}_snaive", o["mase_rel_snaive"], f, f"{m} on {set_name}: geometric mean over series of MASE / seasonal-naive MASE", "{:.2f}")
    f = "bench/leaderboard/timestral-209M_lb_all/summary.json"
    if os.path.exists(p(f)):
        fam = _json.load(open(p(f)))["by_family"]
        for k in ("chaos", "mixture", "tails"):
            num(f"bench_lb_fam_{k}", fam[k]["mase_rel_oracle"], f, f"Timestral-209M, OffPrior-LB, family {k}: geometric mean of MASE / oracle MASE", "{:.2f}")
    xf = "bench/data/cross_platform_diag.json"
    if os.path.exists(p(xf)):
        xp = _json.load(open(p(xf)))
        num("bench_xp_same", xp["identical_configs"], xf, "diagnostic configs whose arrays match between the Windows and Linux builds", "{:d}")
        num("bench_xp_n", xp["n_configs"], xf, "diagnostic configs", "{:d}")
    for set_name in ("diag", "lb"):
        mf = f"bench/data/{set_name}_manifest.csv"
        if os.path.exists(p(mf)):
            m = pd.read_csv(p(mf))
            num(f"bench_n_{set_name}", len(m), mf, "number of series", "{:,d}")
            num(f"bench_nconf_{set_name}", m.config_id.nunique(), mf, "number of configs", "{:d}")


def main():
    for fn in (in_prior, chaos, tails, long_ctx, symavg, normalization, analog, chaos_noise, readouts, selfval, regime, covariates, realdata, bench, search_results, finetune_results, neuron_results):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e}")
    src = open(p("paper", "src.md")).read()
    missing = []

    def sub(m):
        kind, key = m.group(1), m.group(2)
        d = (NUMS if kind == "num" else TABLES).get(key)
        if d is None and key in OPTIONAL and not os.path.exists(p(OPTIONAL[key])):
            return f"*(Results pending: `{OPTIONAL[key]}` is not in the repository yet.)*"
        if d is None:
            missing.append(m.group(0))
            return m.group(0)
        return d["text"]

    out = re.sub(r"\{\{(num|table):([A-Za-z0-9_]+)\}\}", sub, src)
    if missing:
        raise SystemExit(f"undefined placeholders: {sorted(set(missing))}")
    open(p("paper", "off_the_prior.md"), "w").write(out)
    audit = {k: {kk: (vv if kk != "value" else float(vv)) for kk, vv in v.items()} for k, v in NUMS.items()}
    audit.update({f"table:{k}": {"src": v["src"], "how": v["how"]} for k, v in TABLES.items()})
    with open(p("paper", "numbers.json"), "w") as f:
        f.write(json.JSONEncoder(indent=1).encode(audit) + "\n")
    print(f"wrote paper/off_the_prior.md ({len(NUMS)} numbers, {len(TABLES)} tables)")


if __name__ == "__main__":
    main()
