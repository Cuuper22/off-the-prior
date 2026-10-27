import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build(out):
    subprocess.run([sys.executable, os.path.join(ROOT, "bench", "build.py"), "--set", "diag", "--limit", "4", "--out", out],
                   check=True, capture_output=True)
    return open(os.path.join(out, "diag_SHA256SUMS")).read()


def test_build_is_deterministic_and_eval_runs(tmp_path):
    a = _build(str(tmp_path / "a"))
    b = _build(str(tmp_path / "b"))
    assert a == b and len(a) > 0
    r = subprocess.run([sys.executable, os.path.join(ROOT, "bench", "eval.py"), "--set", "diag", "--model", "seasonal_naive",
                        "--data", str(tmp_path / "a"), "--out", str(tmp_path / "lb")], check=True, capture_output=True, text=True)
    assert "overall" in r.stdout
    assert os.path.exists(tmp_path / "lb" / "seasonal_naive_diag_public" / "summary.json")
