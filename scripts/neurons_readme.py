"""Write `results/neurons/README.md` and the run-provenance collections from the committed result files.

    python scripts/neurons_readme.py --runs results/colab_out --out results/neurons

Every table in the README is read from a CSV or numpy-archive file in `--out`; every number in the text comes from a file as well.
Also collects the `meta*.json` files of the capture, steering, failure-link and head runs and the squeeze training logs
(`--runs`) into `provenance_runs.json` and `squeeze/train_w*.json`, so that wall times and hosts are in committed files.
"""
import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from otp.io import provenance, write_json  # noqa: E402
from otp.neurons import score as Sc  # noqa: E402

SIZES = ("5M", "28M", "123M", "209M")


def md(df, fmt="{:.4g}", cols=None):
    if cols:
        df = df[cols]
    head = "| " + " | ".join(str(c) for c in df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = []
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(("" if (isinstance(v, float) and v != v) else (fmt.format(v) if isinstance(v, float) else str(v))) for v in r) + " |")
    return head + "\n".join(rows) + "\n"


def collect(runs, out):
    rec = []
    for pat in ("nn_[0-9]*_main/meta.json", "nn_*_short/meta.json", "nn_*_long/meta.json", "nn_*_xlong/meta.json", "nn_squeeze_w*_main/meta.json",
                "nn_steer/meta_*.json", "nn_failure/meta_*.json", "nn_heads_*/meta.json", "nn_match/meta.json"):
        for f in sorted(glob.glob(os.path.join(runs, pat))):
            d = json.load(open(f))
            d["file"] = os.path.relpath(f, runs)
            rec.append(d)
    write_json(os.path.join(out, "provenance_runs.json"), {"runs_dir": runs, "n": len(rec), "runs": rec})
    os.makedirs(os.path.join(out, "squeeze"), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(runs, "nn_squeeze", "train_w*.json"))):
        d = json.load(open(f))
        write_json(os.path.join(out, "squeeze", os.path.basename(f)), d)
    df = pd.DataFrame(rec).drop_duplicates(subset="file").reset_index(drop=True)
    for c in ("n_series", "n_label_series", "n_heldout_series"):
        if c in df:
            df[c] = df[c].astype("Int64").astype(str).replace("<NA>", "")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/colab_out")
    ap.add_argument("--out", default="results/neurons")
    a = ap.parse_args()
    o = lambda f: os.path.join(a.out, f)          # noqa: E731
    rd = lambda f: pd.read_csv(o(f))              # noqa: E731
    runs = collect(a.runs, a.out)
    pol = {s: Sc.read_poles(o(s)) for s in SIZES}
    allp = pd.concat(pol.values(), ignore_index=True)
    allp["cls"] = Sc.klass(allp, 0.7, 0.1)
    corp = {v: json.load(open(o(f"corpus/{v}_manifest.json"))) for v in ("main", "mv", "short", "long", "xlong")}
    real = json.load(open(o("real/real_manifest.json")))
    src = pd.read_csv(o("real/sources.csv"))
    cv, sc = rd("one_job_curves.csv"), rd("scaling.csv")
    L = []
    w = L.append

    w("# Neuron audit of the Timestral MLPs (task t10_neurons)\n")
    w("Ground-truth audit of whether the MLP neurons of the four Timestral sizes (5M, 28M, 123M, 209M) are monosemantic: every neuron, "
      "both poles, scored against variables that are known exactly because the audit series are generated. All numbers below are read "
      "from files in this folder by `scripts/neurons_readme.py`; the folder names the file each table comes from.\n")

    w("## Definitions\n")
    w("- Neuron: one hidden unit of the SwiGLU MLP. Its value is `z = gate * silu(up)`, the input of the down projection `w3`. "
      "Positive pole = `max(z, 0)`, negative pole = `max(-z, 0)`; a pole is one atlas entry. Pole counts: "
      + ", ".join(f"{s} {len(pol[s]):,}" for s in SIZES) + ".")
    w("- Forward pass: the plain pass (`model.predict_block` followed by the model's `_refine_block`), without symmetric averaging. "
      "Captured activations are stored in fp16 (`n_clipped_fp16` is in `provenance_runs.json`).")
    w("- Sign regime of a firing: `2*(gate>0) + (up>0)`. Positive-pole firings come from regime 3 (gate>0, up>0) or regime 0 (both negative); "
      "negative-pole firings from regime 2 (gate>0, up<0) or 1 (gate<0, up>0).")
    w("- Tokens: a token is a 32-step patch. A 1024-step series gives 32 context tokens and 4 forecast tokens (36 per series).")
    w("- Label: for every pole and every variable of the library (`otp/neurons/variables.py`), the cross-validated R^2 of a 12-bin quantile "
      "bin-mean readout of the pole's activation from the variable, and the mutual information. Folds and the held-out split are by series. "
      "Undefined variable values form their own bin. `label` = variable with the largest R^2; `pair_label` = best pair of core variables "
      "(5 x 5 bins); `r2_add` = additive ridge ceiling over the core variables. `E = max(single, pair, additive)` at score time, "
      "`ratio = r2_cv / E`.")
    w("- Classes at threshold theta and floor f: dead (fewer than 30 label tokens with activation above zero), unexplained (`E < f`), "
      "one_job (`ratio >= theta`), mixed (the rest). Shares are over live poles. Intervals: multinomial bootstrap over poles (`_lo`, `_hi`), "
      "and range over the 3 folds (`_fold_lo`, `_fold_hi`, per-fold label R^2 with pooled E); `_heldout` uses the held-out series.\n")

    w("## Corpus (`results/neurons/corpus/`)\n")
    rows = [{"variant": v, "T": m["T"], "n_series": m["n_series"], "n_label": m["n_label"], "n_heldout": m["n_heldout"], "bytes": m["bytes"],
             "sha256": m["sha256"][:16]} for v, m in corp.items()]
    w(md(pd.DataFrame(rows)))
    w("`main` is univariate; `mv` holds grouped two-row systems (merged with `main` for the main-context runs: 3580 series in total). "
      "`short`, `long` and `xlong` are the context-length variants of the same generator recipe (recipe and seeds in "
      "`corpus/recipe.json` and each `*_manifest.json`). The npz files are git-ignored and rebuilt with `scripts/neurons_corpus.py` "
      "(hashes above).\n")

    w("## Commands that produced the files\n")
    w("```\n"
      "# corpus (VM)\n"
      "python3 scripts/neurons_corpus.py\n"
      "python3 scripts/neurons_real.py --raw-dir <dir> --out results/neurons/real --per-domain 64   # re-fetch: scripts/fetch_realdata.py\n"
      "# capture + labeling, per size S in 5M 28M 123M 209M (scripts/colab/job_nn_common.sh has the same lines)\n"
      "python3 scripts/neurons_run.py --size $S --corpus results/neurons/corpus/main.npz,results/neurons/corpus/mv.npz \\\n"
      "    --real results/neurons/real/real.npz --out results/colab_out/nn_${S}_main --acts-sub 2048 --acts-dir results/colab_out/acts \\\n"
      "    --layers-per-chunk 8 --batch 256 --resume          # short: --batch 256; long: --batch 64; xlong: --batch 16\n"
      "python3 scripts/neurons_steer.py --size $S --runs results/colab_out --corpus results/neurons/corpus/main.npz --out results/colab_out/nn_steer\n"
      "python3 scripts/neurons_failure.py --size $S --runs results/colab_out --out results/colab_out/nn_failure\n"
      "python3 scripts/neurons_heads.py --size $S --corpus results/neurons/corpus/main.npz --mv results/neurons/corpus/mv.npz --out results/colab_out/nn_heads_$S\n"
      "# squeeze\n"
      "python3 scripts/neurons_squeeze.py train --out results/colab_out/nn_squeeze --steps 3000\n"
      "python3 scripts/neurons_run.py --size results/colab_out/nn_squeeze/w$W --corpus results/neurons/corpus/main.npz --out results/colab_out/nn_squeeze_w${W}_main   # W in 4 32 256\n"
      "# collect\n"
      "python3 scripts/neurons_score.py --runs results/colab_out --corpus-dir results/neurons/corpus --out results/neurons\n"
      "python3 scripts/neurons_match.py --acts-dir results/colab_out/acts --runs results/colab_out --out results/colab_out/nn_match\n"
      "python3 scripts/neurons_summarize.py --runs results/colab_out --out results/neurons\n"
      "python3 scripts/neurons_export_browser.py --poles results/neurons --corpus results/neurons/corpus/main.npz,results/neurons/corpus/mv.npz --real results/neurons/real/real.npz --steering results/neurons/steering_results.csv --out results/neurons/browser\n"
      "python3 scripts/neurons_export_neuronpedia.py --size 5M --poles results/neurons/5M/poles.npz --corpus results/neurons/corpus/main.npz,results/neurons/corpus/mv.npz --real results/neurons/real/real.npz --out results/neurons/neuronpedia --n-poles 12\n"
      "python3 scripts/neurons_readme.py --runs results/colab_out --out results/neurons\n"
      "```\n")
    w("Large intermediate files (`results/colab_out/`: `scores_partial.csv` per size and context, `acts_sub_<size>.npz`, squeeze weights) are not "
      "committed; every table below is rebuilt from them with the commands above. The Colab jobs ran from a code-only copy of commit `e9e880a` "
      "(branch `task/t10_neurons_colab`, same `otp/`, `scripts/` and `tests/`), because the full repository archive exceeds the 40 MB limit "
      "of the Colab upload endpoint; the `git` field of those runs in `provenance_runs.json` is that copy's hash.\n")

    w("## Where the runs ran (`provenance_runs.json`)\n")
    r = runs.copy()
    r["wall_s"] = r["wall_s"].astype(float)
    cols = [c for c in ("file", "size", "host", "device", "git", "wall_s", "n_series", "n_label_series", "n_heldout_series", "status_counts") if c in r.columns]
    w(md(r[cols], "{:.1f}"))
    w("Sizes 5M and 28M were captured and labeled on the laptop GPU lane (host `LAPTOP-TTI9OLBG`, git `6466e97+labels.onehot-device-fix`: "
      "the code of `6466e97` with the one-line `labels.onehot` device fix that `7ae89c2` commits with a test). Sizes 123M and 209M ran as two "
      "parallel Colab sessions on a G4 GPU (RTX PRO 6000 Blackwell, 97,887 MiB; setup logs in `results/colab/nn-123M/setup.txt` and "
      "`nn-209M/setup.txt`). The `host`, `device` and `git` columns above give the lane of every steering, failure-link and head run; "
      "the full map used one laptop GPU and two Colab GPU sessions at once, so it did not need more than 2 A100 sessions.\n")

    w("## 1. One-job, mixed and unexplained poles (`one_job_curves.csv`, `scaling.csv`)\n")
    w("Share of live poles by class, theta 0.5 to 0.9, for three floors (all layers, both poles).\n")
    x = cv[(cv.scope == "all") & (cv.sign == "both")]
    for fl in (0.05, 0.1, 0.2):
        for cls in ("one_job", "unexplained"):
            t = x[x.floor == fl].pivot_table(index="theta", columns="size", values=cls)[list(SIZES)].reset_index()
            w(f"{cls}, floor {fl}\n")
            w(md(t))
    w("Mixed share, floor 0.1\n")
    w(md(x[x.floor == 0.1].pivot_table(index="theta", columns="size", values="mixed")[list(SIZES)].reset_index()))
    w("Scaling at theta 0.7, floor 0.1 with bootstrap interval (poles) and fold range (`scaling.csv`)\n")
    w(md(sc))
    w("By layer third and by layer, theta 0.7, floor 0.1: rows with `scope` early / mid / late and layer_k in `one_job_curves.csv` "
      "(columns `one_job`, `mixed`, `unexplained`, with intervals and the held-out share). Layer-third rows:\n")
    t = cv[(cv.scope.isin(["early", "mid", "late"])) & (cv.sign == "both") & (cv.floor == 0.1) & (cv.theta == 0.7)]
    w(md(t[["size", "scope", "n_live", "one_job", "one_job_lo", "one_job_hi", "mixed", "unexplained"]]))
    w("By sign, theta 0.7, floor 0.1\n")
    t = cv[(cv.scope == "all") & (cv.sign != "both") & (cv.floor == 0.1) & (cv.theta == 0.7)]
    w(md(t[["size", "sign", "n_live", "one_job", "mixed", "unexplained"]]))
    w("Label groups of the one-job poles (`label_groups.csv`)\n")
    g = rd("label_groups.csv")
    w(md(g.pivot_table(index="group", columns="size", values="share_of_one_job")[list(SIZES)].reset_index()))

    w("## 2. Predictive check on held-out series (`heldout_check.csv`)\n")
    w(md(rd("heldout_check.csv")))

    w("## 3. Steering (`steering_preregistration.md`, `steering_results.csv`, `steering_effects.csv`, `steering_summary.csv`)\n")
    w("The pre-registration (prediction table, matching criterion, sampling rule) was committed before any steering run. "
      "`role` labeled = poles drawn for the table of predictions; `complement` = the other pole of a drawn neuron; `control` = random poles, "
      "each judged for level, slope and spread against a random direction (the null rate).\n")
    w("Essay style (a qualitative label per pole plus a steering check) is this section together with the `label` and `motif_*` columns of "
      "`poles.npz`; this project's known-variable scoring (R^2 and mutual information against the generating variables, as curves over "
      "thresholds) is sections 1 and 2. The browser rows carry both for each pole.\n")
    ss = rd("steering_summary.csv")
    w(md(ss[ss.group == "all"][["size", "role", "n", "n_match", "match_rate", "ci_lo", "ci_hi", "monotone_rate", "size_rate", "interval_rate"]]))
    w("By predicted quantity group (labeled poles)\n")
    w(md(ss[(ss.role == "labeled")][["size", "group", "n", "n_match", "match_rate", "ci_lo", "ci_hi", "monotone_rate", "size_rate", "interval_rate"]]))
    sr = rd("steering_results.csv")
    lab = sr[sr.role == "labeled"]
    w(f"Largest |effect(scale 4) - effect(scale 0)| over the {len(lab)} steered labeled poles of all sizes, in units of the context standard "
      f"deviation (column `effect_size`): {lab['effect_size'].abs().max():.4g}; the size thresholds of the pre-registration are 0.05 (level, slope, "
      f"spread, amp) and 0.1 (phase). Median |effect_size|: {lab['effect_size'].abs().median():.4g}.\n")

    w("## 4. Failure-map link (`failure_link.csv`, `failure_configs.csv`, `failure_samples.csv`)\n")
    w("Contexts: the three chaos families of the failure map (Mackey-Glass, Rossler, Kuramoto), 8 seeds per config, last 2048 steps of an "
      "8192-step context, horizon 128, oracle with 200 paths (the realized futures can differ from `results/map/chaos`); a config is failing "
      "for a size when the geometric-mean MASE of Timestral is at least 1.5 times the oracle's and the method of analogues is lower than "
      "Timestral's. Reference contexts: configs with ratio below 1.5, plus the held-out in-prior seasonal series (period 8 to 64, first 2048 "
      "steps) of the long corpus. Elevated poles: Cohen's d >= 0.8 for activation on failing versus reference contexts (last 16 context patches "
      "and forecast tokens).\n")
    fc = rd("failure_configs.csv")
    w("Configs per status\n")
    w(md(fc.groupby(["size", "status"]).size().unstack(fill_value=0).reindex(SIZES).reset_index()))
    fl = rd("failure_link.csv")
    w("Activation of period- and phase-labeled one-job poles (`mean_act_*`), and of all live poles\n")
    w(md(fl[fl.analysis == "activation"].dropna(axis=1, how="all").drop(columns=["analysis"])))
    e = fl[(fl.analysis == "elevated") & fl.set.isin(["all"])].dropna(axis=1, how="all")
    w("Elevated poles: count and share that are not one-job (mixed, unexplained or dead)\n")
    w(md(e[["size", "n_poles", "share_not_one_job"]]))
    e = fl[(fl.analysis == "elevated") & ~fl.set.isin(["all"])]
    w("Label groups among elevated poles against the base rate of the groups (enrichment = share / base rate)\n")
    w(md(e.pivot_table(index="set", columns="size", values="enrichment")[list(SIZES)].reset_index()))
    ab = fl[fl.analysis == "ablate"].dropna(axis=1, how="all").drop(columns=["analysis"])
    w("Scaling the elevated period/phase poles (all at once) and the same number of random poles of the same layers, on failing contexts\n")
    w(md(ab))
    sg = fl[fl.analysis == "single"].dropna(axis=1, how="all")
    w("Largest single-pole change of MASE among the 10 elevated period/phase poles with the largest d (ablated alone), per size\n")
    sg = sg.assign(change=(sg.mase_after - sg.mase_before).abs())
    w(md(sg.sort_values("change", ascending=False).groupby("size", sort=False).head(1)[["size", "set", "d", "mase_before", "mase_after", "dist_oracle_before", "dist_oracle_after"]]))

    w("## 5. Real-data transfer (`transfer.csv`, `transfer_by_group.csv`, `real/`)\n")
    w(f"{real['n_windows']} windows of {real['window']} steps (context {real['context']}) from {len(real['windows_per_domain'])} domains "
      f"(windows per domain: {', '.join(f'{k} {v}' for k, v in real['windows_per_domain'].items())}); cut-off {real['cutoff']}; "
      f"failed sources: {len(real['failed_sources'])}. The license manifest is `real/sources.csv` (name, URL, license, license evidence, "
      "redistribution flag, date fetched, sha256), the windows are in `real/windows.csv`, the re-fetch script is `scripts/fetch_realdata.py` "
      "with `scripts/neurons_real.py`. The raw series and the window array (`real.npz`) are not committed; the manifest holds its sha256. "
      "Events: " + ", ".join(real["event_names"]) + ".\n")
    w(md(src[["source_id", "license", "redistributable_derived_windows", "date_fetched_utc"]].drop_duplicates()))
    w("Transfer: the bin-mean readout fitted on synthetic label tokens is applied to the real tokens; `tr_corr` = Pearson correlation of the "
      "readout with the pole's real activation, for poles whose label has a measurable analog on real series.\n")
    w(md(rd("transfer.csv")))
    w("One-job poles by label group\n")
    w(md(rd("transfer_by_group.csv")))

    w("## 6. Essay-derived checks\n")
    w("### 6.1 Both poles as distinctions (`pole_pairs.csv`, theta 0.7, floor 0.1)\n")
    pp = rd("pole_pairs.csv")
    w(md(pp[pp.scope == "all"]))
    w("`same_label` / `same_group_not_label` / `related_groups` / `unrelated_groups`: shares of neurons with both poles explained, by the "
      "relation of the two best labels; `oj_*` the same for neurons whose two poles are both one-job.\n")
    w("### 6.2 SwiGLU sign regimes (`sign_regimes.csv`)\n")
    w(md(rd("sign_regimes.csv")))
    w("### 6.3 Summary motif: token type and position\n")
    k = rd("kinds.csv")
    w("Best-label kind among explained poles (`kinds.csv`)\n")
    w(md(k[["size", "sign", "n_explained", "best_is_motif", "best_is_position", "best_is_latent", "best_is_family", "motif_poles_position_explains_ge_half"]]))
    live = allp[allp.cls != "dead"].copy()
    nl = live.groupby("size")["layer"].transform("max") + 1
    live["third"] = pd.Categorical(((live.layer * 3 // nl).map({0: "early", 1: "mid", 2: "late"})), categories=["early", "mid", "late"])
    pos_labels = live[(live.group == "position") & (live.cls == "one_job")]
    t = pos_labels.groupby(["size", "label"]).size().rename("n_one_job").reset_index()
    w("One-job poles with a position-group label (token type or position), by label\n")
    w(md(t.pivot_table(index="label", columns="size", values="n_one_job", fill_value=0)[list(SIZES)].reset_index()))
    t = live.groupby(["size", "third"], observed=True).agg(n_live=("label", "size"), share_label_position=("group", lambda s: float((s == "position").mean()))).reset_index()
    w("Share of live poles whose best label is in the position group, by layer third\n")
    w(md(t))
    t = live.groupby("size").agg(median_r2_ctx=("r2_ctx", "median"), median_r2_fc=("r2_fc", "median"), median_r2_cv=("r2_cv", "median")).reset_index()
    w("Median R^2 of the best label restricted to context tokens (`r2_ctx`) and to forecast tokens (`r2_fc`)\n")
    w(md(t))
    w("### 6.4 Context length (`context_comparison.csv`)\n")
    w("Same pipeline on corpora of 128, 1024, 4096 and 32768 steps; `same_label_as_main`, `same_group_as_main`, `r2_corr_with_main` over poles "
      "explained in both.\n")
    w(md(rd("context_comparison.csv")))
    w("### 6.5 Shared motif of exemplar lists (`motif_cohesion.csv`)\n")
    w("Cohesion = mean over the 18 motif variables of the standard deviation inside the exemplar list over the standard deviation over all "
      "label tokens (1 = random tokens, below 1 = shared motif); `coh_top` = top exemplars, `coh_mid` = mid-range exemplars. Per-pole motif "
      "descriptions are the columns `motif_top`, `motif_mid` (synthetic) and the real-data exemplars `real_top_*`, `real_mid_*` in each "
      "`<size>/poles.npz`.\n")
    w(md(rd("motif_cohesion.csv")))
    w("### 6.6 Scale ladder (`scale_match.csv`)\n")
    w("For one-job poles of the smaller model: activation correlation with every pole of the larger model on the same 2048 label-set tokens "
      "(`acts_sub_<size>.npz`, same token ids for all sizes). `share_two_or_more_ge_0.5`: share of small-model poles that have two or more "
      "larger-model poles with correlation >= 0.5.\n")
    w(md(rd("scale_match.csv")))
    oj = allp[allp.cls == "one_job"]
    t = oj.groupby("size").agg(n_one_job=("label", "size"), distinct_labels=("label", "nunique"), distinct_groups=("group", "nunique")).reset_index()
    w("Distinct labels among one-job poles\n")
    w(md(t))
    w("### 6.7 Squeeze test (`squeeze_summary.csv`, `squeeze/train_w*.json`)\n")
    w("Three small Timestral-style models (patch 32, d_model 96, 3 heads, 4 time-attention blocks, 21 quantile knots) that differ only in MLP "
      "hidden width (4, 32, 256), same initialization seed and training batches, 3000 steps on `otp.generators.primitives.in_prior`. "
      "Labeled with the same pipeline on the main corpus.\n")
    w(md(rd("squeeze_summary.csv")))
    tr = [json.load(open(f)) for f in sorted(glob.glob(o("squeeze/train_w*.json")))]
    w(md(pd.DataFrame([{k_: d_.get(k_) for k_ in ("mlp_hidden_dim", "params", "steps", "batch_size", "lr", "final_loss", "wall_s", "host", "git")} for d_ in tr])))
    w("### 6.8 Validation beyond correlation (`steering_checks.csv`)\n")
    w("`downstream_flagged`: unmatched steered labels whose largest ablation effect is on another forecast quantity and at least the size "
      "threshold; `negative_pole_*`: neurons where a pole and its complement share a label, share with the effect signs the label correlations "
      "imply; `upstream_flagged`: steered poles with none of the three most correlated earlier-layer poles in the same label group and a best "
      "motif label that explains less than half of the label's R^2 (`upstream.csv` columns `up*` in each `poles.npz`).\n")
    w(md(rd("steering_checks.csv")))

    w("## 7. Attention heads (`heads_time.csv`, `heads_variate.csv`, `heads_summary.csv`)\n")
    w("Every time-attention head and every variate-attention head of the four models, described by where its forecast-token and context-token "
      "queries attend on the main corpus (recency, seasonal lag, post-switch patches, self, first-patch sink, covariate rows for variate heads); "
      "`heads_variate.csv` has the two-row systems.\n")
    hs = rd("heads_summary.csv")
    w(md(hs.pivot_table(index=["query", "description"], columns="size", values="n_heads", fill_value=0)[list(SIZES)].reset_index()))

    w("## 8. Browser and Neuronpedia exports\n")
    w("`browser/`: compact JSON per size and layer (`index.json` describes the field order and the snippet encoding); one row per pole with "
      "label, scores, held-out R^2, class, sign-regime share, transfer correlation, steering result and short exemplar snippets (top, mid-range, "
      f"real). The largest file is {max(os.path.getsize(os.path.join(dp, f_)) for dp, _, fs in os.walk(o('browser')) for f_ in fs) / 1e6:.2f} MB.\n")
    w("`neuronpedia/`: sample of 15 poles of Timestral-5M in the `upload-batch` layout, written by `scripts/neurons_export_neuronpedia.py` "
      "(which checks the field limits, no upload). Draft e-mail (not sent): `paper/outreach/neuronpedia_email.md`.\n")
    w("Sources read: the Neuronpedia documentation (`https://docs.neuronpedia.org`, pages on uploading and feature dashboards, read on "
      "2026-10-26) and the repository `https://github.com/hijohnnylin/neuronpedia` at commit `a8a8c331a2954fd2a84cb2a099ece4001618716b` "
      "(route `apps/webapp/app/api/feature/upload-batch/route.tsx`, Python client `neuronpedia.np_feature.Feature`).\n")
    w("What does not fit the schema:\n"
      "- Tokens are patches of 32 real numbers, not text; the export shows block-character sparkline strings and keeps the raw values in `neuronpedia/patches/`.\n"
      "- The output is 21 quantiles per forecast step, not a token distribution: `topLogits` and `bottomLogits` are empty (no logit lens).\n"
      "- Four sizes are four separate models; the two poles of a neuron are two features (source sets `ts<size>-mlp-pos`, `ts<size>-mlp-neg`) linked only through explanation text.\n"
      "- Real-data exemplars have a domain and a license that the schema has no field for (kept in `neuronpedia/extra/`).\n"
      "- Steering results and attention-head entries have no place on a feature page.\n"
      "- Label scores (R^2, mutual information, class) go into the explanation text (at most 512 characters) and a side file.\n")

    w("## 9. Cuts, failures and limits\n")
    w("- Colab upload: the first launches for 123M and 209M failed at the upload of the repository archive (more than 40 MB); the jobs were relaunched from a code-only copy and ran once each.\n"
      "- The failure-link script changed after the first Colab pass of 123M and 209M (seasonal reference contexts, cache of the size-independent contexts); the tables here come from the second pass of all four sizes (the first-pass chaos-only files are in `results/colab/nn-123M/failure_chaos_only/` and `nn-209M/failure_chaos_only/`).\n"
      "- Per-context tables (`poles_short/long/xlong.npz`) keep the id, label, score and class columns only; exemplar columns exist in `poles.npz`.\n"
      "- The xlong corpus has 48 series (32768 steps); its per-pole numbers rest on 38 label series.\n"
      "- Reference contexts of the failure link: 35 seasonal series (the long corpus has 35 held-out seasonal series with period 8 to 64).\n"
      "- Pole tables are numpy archives (`otp.neurons.score.read_poles`) with float32 numbers, plus a CSV of the core columns per size (`<size>/poles.csv`); parquet files are not committed because the project word checker reads compressed bytes as text and finds list words by chance.\n")

    w("## 10. Follow-up (`followup/`)\n")
    w("- `followup/README.md` holds the follow-up tables: corpus, series-count and library controls for the one-job share (B); steering power, with a positive control by residual-stream directions, group steering and minimum detectable effects (A, pre-registered in `steering_preregistration_followup.md`); and motif labels from patch-shape clusters (C).\n- Single-pole steering results in section 3 are re-judged against matched controls in `followup/single_pole_calibrated.csv`.\n")

    w("## Files\n")
    fl_ = sorted(f for f in os.listdir(a.out) if os.path.isfile(o(f)))
    w(", ".join(f"`{f}`" for f in fl_) + "; folders `<size>/` (`poles.npz`, `poles_exemplars.npz`, `poles.csv`, `poles_short|long|xlong.npz`), `corpus/`, `real/`, `browser/`, `neuronpedia/`, `squeeze/`.\n")
    with open(o("README.md"), "w") as f:
        f.write("\n".join(L))
    write_json(o("provenance_readme.json"), provenance("scripts/neurons_readme.py", runs=a.runs))
    print("wrote README.md", sum(len(s) for s in L))


if __name__ == "__main__":
    main()
