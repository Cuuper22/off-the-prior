# Publishing the benchmarks to Hugging Face (prepared, not run)

```bash
pip install -U huggingface_hub
huggingface-cli login
# 1. build the data (both sets) and score the baselines, or copy them from the Colab job output
python bench/build.py --set diag --workers 8
python bench/build.py --set lb --workers 8
python bench/make_docs.py
# 2. create the dataset repo and upload
huggingface-cli repo create off-the-prior-bench --type dataset
huggingface-cli upload <user>/off-the-prior-bench bench/dataset_card.md README.md --repo-type dataset
huggingface-cli upload <user>/off-the-prior-bench bench/data data --repo-type dataset
huggingface-cli upload <user>/off-the-prior-bench bench/configs configs --repo-type dataset
huggingface-cli upload <user>/off-the-prior-bench bench/leaderboard leaderboard --repo-type dataset
```

Replace `<user>` with the Hugging Face account or organization. The repository is created public by
default; add `--private` to `repo create` to keep it private first.
