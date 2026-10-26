"""Flatten the mechanism-run JSON files (results/mechanism/readouts/*.json) into summary.csv.

One row per study x knob value x size: forecast ratio to the oracle, cross-validated R^2 of
Timestral's own output and of the best per-layer linear readout for the oracle's expected
first-patch change and for single steps 1 and 16, template correlation (model vs. oracle),
and attention summaries. Cells whose oracle was not finite (diverged Monte Carlo paths) are
flagged `valid = False`.

python scripts/mechanism_summary.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = "results/mechanism/readouts"


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "*.json"))):
        if os.path.basename(f).startswith("provenance"):
            continue
        for r in json.load(open(f)):
            ro = r["readout_r2"]
            st = r.get("step_labels", {})
            pc = [v for v in (r["attn"].get("post_change") or []) if v is not None and np.isfinite(v)]
            row = {"study": r["study"], "knob": r["knob"], "knob_value": r["knob_value"], "size": r["size"], "n": r["n"],
                   "mase_ratio_oracle_gmean": r["mase_ratio_oracle_gmean"], "output_r2": r["output_r2"],
                   "best_readout_r2": max(ro.values()), "best_readout_layer": int(max(ro, key=ro.get)),
                   "template_corr_model": r["template_corr_model"], "template_corr_oracle": r["template_corr_oracle"],
                   "attn_recent_last_layer": r["attn"]["recent"][-1], "attn_entropy_mean": float(np.mean(r["attn"]["entropy"])),
                   "attn_post_change_mean": float(np.mean(pc)) if pc else np.nan}
            for k in ("step1", "step4", "step16"):
                if k in st:
                    row[f"{k}_output_r2"] = st[k]["output_r2"]
                    row[f"{k}_readout_r2"] = st[k]["best_readout_r2"]
            row["valid"] = bool(np.isfinite(row["mase_ratio_oracle_gmean"]) and np.isfinite(row["output_r2"]))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["readout_minus_output"] = df.best_readout_r2 - df.output_r2
    df.to_csv(os.path.join(ROOT, "summary.csv"), index=False)
    print(f"wrote {len(df)} rows ({int((~df.valid).sum())} invalid)")


if __name__ == "__main__":
    main()
