# Publishing checklist (prepared, not run)

Nothing in this repository has been published. These are the exact steps to publish it.
Run them from the repository root after reviewing the content.

## 0. Before anything else

Publish from a fresh clone of `master` only (task branches are internal working branches),
after the maintainer's pre-publish history cleanup has been run on that clone.

Blocking: `bench/regression/real.npz` contains windows from 13 GIFT-Eval datasets whose redistribution
rests on the `Salesforce/GiftEval` card's apache-2.0 tag (see `license_evidence` in
`bench/regression/sources.csv`). Check each dataset's upstream license first; datasets that are not
verified move to the gitignored restricted file (flag in `otp/realdata_sources.py`), and only the
re-fetch script and hashes are published for them.

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

Fine-tuned checkpoints are produced by `notebooks/finetune_colab.ipynb` (not stored in git).
Model cards: `results/finetune/model_cards/`. Per checkpoint:

```bash
huggingface-cli login
huggingface-cli repo create <name> --type model
huggingface-cli upload <user>/<name> <checkpoint_dir> . 
huggingface-cli upload <user>/<name> results/finetune/model_cards/<name>.md README.md
```

The Timestral weights are under a modified MIT license; fine-tuned derivatives must carry
it (copy `LICENSE` from the base model repo into each checkpoint repo).

## 4. arXiv-style PDF

```bash
make -C paper pdf   # builds paper/off_the_prior.pdf from paper/off_the_prior.md
```

Upload the LaTeX source (`paper/build/`) to arXiv; arXiv needs an endorsed account in the
chosen category (cs.LG or stat.ML).
