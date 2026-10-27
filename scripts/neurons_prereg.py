"""Write the steering pre-registration (committed before any steering run).

    python scripts/neurons_prereg.py --out results/neurons/steering_preregistration.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp.neurons import steer  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/neurons/steering_preregistration.md")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(steer.prereg_markdown())
    print("wrote", a.out)
