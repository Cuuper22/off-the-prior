"""Summarize the symmetric-averaging ablation (results/mechanism/symavg/{on,off}) into summary.csv.

python scripts/symavg_summary.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otp.analysis import ratios  # noqa: E402

ROOT = "results/mechanism/symavg"


def main():
    on = pd.read_csv(os.path.join(ROOT, "on", "scores.csv"))
    off = pd.read_csv(os.path.join(ROOT, "off", "scores.csv"))
    df = ratios(pd.concat([on, off], ignore_index=True))
    ts = df[df.forecaster.str.startswith("timestral")].copy()
    ts["size"] = ts.forecaster.str.replace("-nosym", "").str.replace("timestral-", "")
    ts["symmetric_averaging"] = np.where(ts.forecaster.str.endswith("-nosym"), "off", "on")
    gm = lambda x: float(np.exp(np.log(x).mean()))  # noqa: E731
    s = ts.groupby(["config_id", "size", "H", "symmetric_averaging"]).agg(
        n=("seed", "size"), scrps_ratio_oracle_gmean=("scrps_ratio_oracle", gm),
        mase_ratio_oracle_gmean=("mase_ratio_oracle", gm), cov80=("cov80", "mean")).reset_index()
    w = s.pivot_table(index=["config_id", "size", "H"], columns="symmetric_averaging",
                      values="scrps_ratio_oracle_gmean").reset_index()
    w["scrps_off_over_on"] = w["off"] / w["on"]
    s.to_csv(os.path.join(ROOT, "summary.csv"), index=False)
    w.to_csv(os.path.join(ROOT, "off_over_on.csv"), index=False)
    print(w.groupby("H").scrps_off_over_on.describe().round(3))


if __name__ == "__main__":
    main()
