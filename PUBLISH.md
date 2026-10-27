# Publishing checklist (prepared, not run)

Nothing in this repository has been published. These are the exact steps to publish it.
Run them from the repository root after reviewing the content.

## 0. Before anything else

Publish from a fresh clone of `master` only (task branches are internal working branches),
after the maintainer's pre-publish history cleanup has been run on that clone.

Real-data windows: `bench/regression/real.npz` holds only GIFT-Eval electricity/H and electricity/D (UCI,
CC BY 4.0). The other 11 GIFT-Eval datasets have no verified upstream license for redistribution and live in the
gitignored `real_restricted.npz`; `scripts/fetch_realdata.py` and the recorded hashes rebuild them locally. The older
256-window `real.npz` is removed from history by the cleanup.

Size: `results/neurons/` adds about 250 MB (per-size `poles*.npz` archives and the browser JSON). No file is
over 50 MB, so GitHub accepts it, but consider moving `results/neurons/*/poles*.npz` and
`results/neurons/browser/` to the Hugging Face dataset (step 2) and dropping them from history in the same
cleanup, keeping `poles.csv` and the summary tables in git.

## 1. GitHub repository

```bash
gh repo create off-the-prior --public --source . --remote origin \
  --description "Where a synthetic-prior time-series foundation model (Timestral) fails, and why"
git push -u origin master
```

GitHub Pages for the site in `docs/`:

```bash
gh api -X POST repos/{owner}/off-the-prior/pages -f "source[branch]=master" -f "source[path]=/docs"
```

## 2. Hugging Face dataset (benchmarks)

See `bench/PUBLISH.md` for the dataset upload commands (dataset card: `bench/dataset_card.md`).

## 3. Hugging Face models (fine-tuned fixes)

The fine-tuning study kept no checkpoints (each was scored right after training and deleted; see
`results/finetune/colab/run1/README.md`), and at these budgets none of them closes a gap, so there is nothing to
publish as a model. To retrain one, run `notebooks/finetune_colab.ipynb`; a published derivative must carry the
Timestral license (modified MIT) and one of the model cards in `results/finetune/model_cards/`:

```bash
huggingface-cli login
huggingface-cli repo create <name> --type model
huggingface-cli upload <user>/<name> <checkpoint_dir> .
huggingface-cli upload <user>/<name> results/finetune/model_cards/<card>.md README.md
```

## 4. arXiv-style PDF

```bash
make -C paper pdf   # builds paper/off_the_prior.pdf from paper/off_the_prior.md
```

Upload the LaTeX source (`paper/build/`) to arXiv; arXiv needs an endorsed account in the
chosen category (cs.LG or stat.ML).
