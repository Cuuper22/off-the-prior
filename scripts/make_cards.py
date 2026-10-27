"""Fill model-card drafts from a finished results folder (`costs.csv` and `summary.csv` in it).

  python scripts/make_cards.py --results results/finetune/colab/run1 --out results/finetune/model_cards/colab \
      --src results/finetune/colab/run1 --by-size

Every number in a card is copied from the two csv files; `--src` is the path written into the card as the source.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from otp import finetune_ext as X  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=None, help="path named in the cards as their source (default --results)")
    ap.add_argument("--by-size", action="store_true", help="one sub-folder per model size")
    a = ap.parse_args(argv)
    costs = pd.read_csv(os.path.join(a.results, "costs.csv"))
    summary = pd.read_csv(os.path.join(a.results, "summary.csv"))
    n = X.write_cards_local(a.out, summary, costs, a.src or a.results, by_size=a.by_size)
    print("wrote", n, "cards to", a.out)


if __name__ == "__main__":
    main()
