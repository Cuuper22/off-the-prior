# First pass of the self-validation fix (superseded)

One backtest window, soft blends (temperatures 0.05 / 0.1 / 0.25) and a hard pick, from an
earlier version of `scripts/selfval_fix.py` (commit 81915c4), all four sizes, on the in-prior,
chaos, tails and regime samples. Kept because the second version's design (two windows, a
margin chosen on one half of the seeds) was motivated by what this pass showed: it helped
chaos but hurt tails and regime, because a single backtest window favours the analogues by
chance on random walks. The current results are in `results/fix/selfval/`.
