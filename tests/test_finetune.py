"""Tests for the fine-tuning study: splits, training pools, the training/eval scripts end to end (5M, CPU, tiny budgets)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from otp import finetune_ext as X
from otp import finetune_splits as S
from otp.generators.primitives import in_prior


def test_knob_rules_are_disjoint_and_follow_the_spec():
    assert S.knob_side("chaos", "logistic_map/r3.6") == "train"
    assert S.knob_side("chaos", "logistic_map/r3.7") is None
    assert S.knob_side("chaos", "logistic_map/r3.83") == "test"
    assert S.knob_side("tails", "tails/stable_ar/alpha1.5_rho0.9") == "train"
    assert S.knob_side("tails", "tails/stable_ar/alpha1.3_rho1.0") == "test"
    assert S.knob_side("regime", "regime/mean_jump/context/delta2/d256") == "train"
    assert S.knob_side("regime", "regime/mean_jump/context/delta2/d64") == "test"
    assert S.knob_side("regime", "regime/mean_jump/context/delta2/d1024") == "train"
    assert S.knob_side("regime", "regime/mean_jump/horizon/delta2") is None
    for fam in ("chaos", "regime", "tails"):
        tr = {c for c, _ in S.side_configs(fam, "train")}
        te = {c for c, _ in S.side_configs(fam, "test")}
        assert tr and te and not (tr & te)
    assert S.train_families("lofo", "regime") == ["chaos", "tails", "mixture"]
    assert S.train_families("knob", "regime") == list(S.FAMILIES)


def test_seed_ranges_do_not_overlap():
    lo, hi = S.TRAIN_SEEDS
    assert hi <= S.EVAL_SEED_BASE
    assert S.REGRESSION_SEED_START + S.N_REGRESSION_DEFAULT <= lo


def test_eval_samples_are_deterministic_and_finite():
    a = S.make_eval_samples("knob", "regime", n=3, n_paths=20, T=512, H=64)
    b = S.make_eval_samples("knob", "regime", n=3, n_paths=20, T=512, H=64)
    for s, t in zip(a, b):
        assert np.array_equal(s.context, t.context) and np.isfinite(s.oracle_q).all()
        assert S.knob_side("regime", s.meta["config_id"]) == "test"


def test_pool_roundtrip_and_replay_share(tmp_path):
    sched = X.schedule(3, seed=1, contexts=(64, 128), horizons=(32,))
    fam = [X.build_pool(lambda sd: in_prior(sd), sched, rows=2, seed=k, log=None) for k in range(3)]
    rep = X.build_pool(in_prior, sched, rows=4, seed=9, log=None)
    X.save_pool(str(tmp_path / "p.npz"), fam[0])
    back = X.load_pool(str(tmp_path / "p.npz"))
    assert back["t"] == fam[0]["t"] and np.array_equal(back["x"][1], fam[0]["x"][1])
    smp = X.CompositeSampler(fam, rep, replay_frac=0.5)
    x, y = smp.batch(4)
    assert x.shape == (4, sched[0][0]) and y.shape == (4, sched[0][1])
    assert np.array_equal(x[:2], rep["x"][0][:2])           # first half is replay
    assert np.array_equal(x[2], fam[0]["x"][0][0])          # then families in turn
    assert np.array_equal(x[3], fam[1]["x"][0][0])
    smp0 = X.CompositeSampler(fam, rep, replay_frac=0.0)
    x0, _ = smp0.batch(3)
    assert [np.array_equal(x0[i], fam[i]["x"][0][0]) for i in range(3)] == [True] * 3


def test_run_and_eval_end_to_end(tmp_path):
    """Train 2 tiny runs (full, lora+mix) on the leave-regime-out split, score them, write all tables."""
    import eval_finetuned
    import finetune as ft

    root = str(tmp_path)
    for method, rf in (("full", 0.0), ("lora", 0.5)):
        ft.main(["run", "--size", "5M", "--protocol", "lofo", "--holdout", "regime", "--method", method,
                 "--replay-frac", str(rf), "--steps", "2", "--batch-size", "4", "--warmup", "1",
                 "--contexts", "256", "--horizons", "32", "--log-every", "0",
                 "--runs-root", f"{root}/runs", "--ckpt-root", f"{root}/ck", "--pool-dir", f"{root}/pools"])
    runs = sorted(os.listdir(f"{root}/runs"))
    assert runs == ["5M__lofo-regime__full__nomix", "5M__lofo-regime__lora__mix0.5"]
    full = X.cost_row(__import__("otp.io", fromlist=["read_json"]).read_json(f"{root}/runs/{runs[0]}/run.json"))
    lora = X.cost_row(__import__("otp.io", fromlist=["read_json"]).read_json(f"{root}/runs/{runs[1]}/run.json"))
    assert lora["trainable_params"] < full["trainable_params"] == full["base_params"]
    assert (lora["steps"], full["steps"]) == (2, 2)
    eval_finetuned.main(["--runs-root", f"{root}/runs", "--ckpt-root", f"{root}/ck", "--out", f"{root}/out",
                         "--n-eval", "3", "--n-regression", "3", "--n-paths", "20", "--cards-dir", f"{root}/cards",
                         "--note", "tiny test run"])
    summ = pd.read_csv(f"{root}/out/summary.csv")
    assert set(summ.role) == {"test", "regression"} and len(summ) == 4
    assert np.isfinite(summ.ft_mase_ratio).all() and np.isfinite(summ.pre_mase_ratio).all()
    assert summ[summ.role == "test"].trunc_mase_ratio.notna().all()  # regime set has the truncation baseline
    assert os.path.exists(f"{root}/out/README.md") and os.path.exists(f"{root}/out/costs.csv")
    card = open(f"{root}/cards/{runs[1]}.md").read()
    assert "LoRA adapters" in card and "{{" not in card
    assert sorted(os.listdir(f"{root}/cards")) == [r + ".md" for r in runs]
    sc = pd.read_csv(f"{root}/out/scores.csv")
    assert {"oracle", "pretrained", "pretrained_truncate", "finetuned"} <= set(sc.forecaster)


def test_colab_placeholder_cards_name_their_source_file(tmp_path):
    n = X.write_cards_colab_placeholders(str(tmp_path))
    assert n == 4 * 2 * 4 * 2 * 3
    txt = open(tmp_path / "123M" / "123M__knob-tails__full__mix0.5.md").read()
    assert "costs.csv :: run_id=123M__knob-tails__full__mix0.5" in txt and "summary.csv" in txt
