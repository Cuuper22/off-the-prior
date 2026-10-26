"""Choose and evaluate the switch rule of the self-validation fix without peeking at the test half.

Rule (K, m): switch to the analogue forecast if, on each of the last K backtest windows, the
analogues' error is below m times Timestral's. Seeds are split by the index in the sample id
(".../s<k>"): even k = tuning half, odd k = test half.

Tuning (per model size, on the even half): among K in {1, 2} and m in MARGINS, take the rule with
the lowest geometric-mean MASE ratio (fix / Timestral) on the chaos family subject to every
other family's ratio being <= 1 + TOL. Report: every family's ratio on the odd half (MASE and
sCRPS), the worst config, and the share of series switched, for the chosen rule and for all rules.

python scripts/selfval_rules.py --root results/fix/selfval --out results/fix/selfval
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otp.io import write_json  # noqa: E402

MARGINS = (1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
TOL = 0.005


def gm(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if x.size else float("nan")


def apply_rule(d, K, m):
    use = np.ones(len(d), bool)
    for k in range(1, K + 1):
        use &= (d[f"bt{k}_mae_analog"] < m * d[f"bt{k}_mae_timestral"]).values
    return d.assign(switched=use, r_mase=np.where(use, d.mase_switch, d.mase_timestral) / d.mase_timestral,
                    r_scrps=np.where(use, d.scrps_switch, d.scrps_timestral) / d.scrps_timestral,
                    fix_mase_oracle=np.where(use, d.mase_switch, d.mase_timestral) / d.mase_oracle,
                    ts_mase_oracle=d.mase_timestral / d.mase_oracle)


def evaluate(d, K, m):
    x = apply_rule(d, K, m)
    out = []
    for (fam, size), g in x.groupby(["fam", "size"]):
        cfg = g.groupby("config_id").r_mase.apply(gm)
        out.append({"family": fam, "size": size, "K": K, "margin": m, "n": len(g), "mase_fix_over_timestral": gm(g.r_mase),
                    "scrps_fix_over_timestral": gm(g.r_scrps), "worst_config_mase": float(cfg.max()),
                    "worst_config": cfg.idxmax(), "switched_share": float(g.switched.mean()),
                    "mase_ratio_oracle_timestral": gm(g.ts_mase_oracle), "mase_ratio_oracle_fix": gm(g.fix_mase_oracle)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/fix/selfval")
    ap.add_argument("--out", default="results/fix/selfval")
    ap.add_argument("--frozen", nargs=2, type=float, default=None, metavar=("K", "M"),
                    help="evaluate one fixed rule on all seeds (confirmation run) instead of tuning")
    a = ap.parse_args()
    frames = []
    for f in sorted(glob.glob(os.path.join(a.root, "*", "per_sample.csv"))):
        d = pd.read_csv(f)
        d["fam"] = os.path.basename(os.path.dirname(f))
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[np.isfinite(d.mase_oracle) & np.isfinite(d.scrps_oracle) & np.isfinite(d.mase_switch)]
    d["seed_index"] = d.sample_id.str.extract(r"/s(\d+)$")[0].astype(int)
    if a.frozen:
        K, m = int(a.frozen[0]), float(a.frozen[1])
        r = pd.concat([evaluate(d[d["size"] == sz], K, m) for sz in sorted(d["size"].unique())], ignore_index=True)
        r.to_csv(os.path.join(a.out, "frozen_rule_confirmation.csv"), index=False)
        write_json(os.path.join(a.out, "frozen_rule.json"), {"K": K, "margin": m, "note": "fixed before this run"})
        pd.set_option("display.width", 220)
        print(r.round(3).to_string())
        return
    tune, test = d[d.seed_index % 2 == 0], d[d.seed_index % 2 == 1]
    Ks = sorted({int(c[2]) for c in d.columns if c.startswith("bt") and c.endswith("_mae_analog")})
    chosen, all_test = [], []
    for size in sorted(d["size"].unique()):
        best = None
        for K in Ks:
            for m in MARGINS:
                e = evaluate(tune[tune["size"] == size], K, m).set_index("family")
                others = e.drop(index="chaos", errors="ignore").mase_fix_over_timestral
                if (others <= 1 + TOL).all():
                    score = e.loc["chaos", "mase_fix_over_timestral"] if "chaos" in e.index else 1.0
                    if best is None or score < best[0]:
                        best = (score, K, m)
        _, K, m = best
        chosen.append({"size": size, "K": K, "margin": m})
        for KK in Ks:
            for mm in MARGINS:
                r = evaluate(test[test["size"] == size], KK, mm)
                r["chosen"] = (KK == K) & (mm == m)
                all_test.append(r)
    res = pd.concat(all_test, ignore_index=True)
    res.to_csv(os.path.join(a.out, "rules_test_half.csv"), index=False)
    pick = res[res.chosen].drop(columns="chosen")
    pick.to_csv(os.path.join(a.out, "chosen_rule_test_half.csv"), index=False)
    write_json(os.path.join(a.out, "chosen_rule.json"), {"tolerance": TOL, "margins": MARGINS, "rules": chosen})
    pd.set_option("display.width", 220)
    print(pick.round(3).to_string())


if __name__ == "__main__":
    main()
