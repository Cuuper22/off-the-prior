"""Summarize a sweep folder's scores.csv into summary.csv (the lead's canonical summary).

Groups Timestral rows by [config_id, generator, knob_value, forecaster, T, H] and reports,
over seeds: geometric mean (with 95% bootstrap CI), median and mean of the MASE and sCRPS
ratios to the oracle and to the best classical model, plus mean cov80 / cov98 / calib.
When `--control` points at the in-prior control's summary, each row also gets the control's
geometric-mean MASE ratio at the same (forecaster, T, H) and the excess over it.

python scripts/summarize.py results/map/chaos [--control results/map/in_prior/summary.csv]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp.analysis import curve, ratios  # noqa: E402

KEYS = ["config_id", "generator", "knob_value", "forecaster", "T", "H"]
METRICS = ["mase_ratio_oracle", "scrps_ratio_oracle", "mase_ratio_classical", "scrps_ratio_classical"]


def summarize(folder: str, control: str | None = None, variate: int | None = 0,
              extra: list[str] | None = None) -> pd.DataFrame:
    """`extra`: more scores.csv files to concatenate (e.g. big-model runs that reused this folder's samples)."""
    df = pd.concat([pd.read_csv(os.path.join(folder, "scores.csv"))] + [pd.read_csv(e) for e in (extra or [])],
                   ignore_index=True)
    if "config_id" not in df:
        df["config_id"] = df["generator"]
    if variate is not None and "variate" in df:
        df = df[df.variate == variate]
    df = ratios(df)
    ts = df[df.forecaster.str.startswith("timestral")].copy()
    ts["knob_value"] = ts["knob_value"].astype(str)
    out = None
    for m in METRICS:
        c = curve(ts, KEYS, m)
        med = ts.groupby(KEYS, dropna=False)[m].median().rename(f"{m}_median").reset_index()
        c = c.merge(med, on=KEYS)
        out = c if out is None else out.merge(c.drop(columns=["n"]), on=KEYS)
    cal = ts.groupby(KEYS, dropna=False)[["cov80", "cov98", "calib"]].mean().reset_index()
    out = out.merge(cal, on=KEYS)
    orc = df[df.forecaster == "oracle"].copy()
    orc_cal = orc.groupby(["config_id", "H"])[["cov80", "calib"]].mean().rename(
        columns={"cov80": "oracle_cov80", "calib": "oracle_calib"}).reset_index()
    out = out.merge(orc_cal, on=["config_id", "H"], how="left")
    if control:
        ctl = pd.read_csv(control)
        ctl = ctl.groupby(["forecaster", "T", "H"])["mase_ratio_oracle_gmean"].apply(
            lambda x: float(np.exp(np.log(x).mean()))).rename("control_mase_ratio_oracle").reset_index()
        out = out.merge(ctl, on=["forecaster", "T", "H"], how="left")
        out["excess_over_control"] = out["mase_ratio_oracle_gmean"] / out["control_mase_ratio_oracle"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--control", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", nargs="*", default=None, help="additional scores.csv files to merge in")
    a = ap.parse_args()
    s = summarize(a.folder, a.control, extra=a.extra)
    path = a.out or os.path.join(a.folder, "summary_lead.csv")
    s.to_csv(path, index=False)
    print(f"wrote {len(s)} rows to {path}")


if __name__ == "__main__":
    main()
