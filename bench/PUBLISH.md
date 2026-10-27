# Publishing the benchmarks to Hugging Face (prepared, not run)

The published files are the ones built on the Windows laptop lane (`scripts/laptop/run_bench.ps1`). The
diagnostic files are in git. The leaderboard files (`bench/data/lb_*.npz`) are gitignored by size; copies
are in the working tree where they were scored and on the laptop (`C:\Users\Acer\off-the-prior-lane\bench_out\data`).
Do not rebuild them on another platform for publishing: chaotic configs come out different
(`bench/data/cross_platform_diag.json`). Check a copy before upload against the content hashes:

```bash
python -c "import sys; sys.path.insert(0, 'bench'); from build import content_sha256 as h; \
[print(f, h('bench/data/' + f) == s) for s, f in (l.split() for n in ('diag', 'lb') for l in open(f'bench/data/{n}_SHA256SUMS'))]"
```

Every line must print `True`. Then:

```bash
pip install -U huggingface_hub
huggingface-cli login
python bench/make_docs.py
huggingface-cli repo create off-the-prior-bench --type dataset
huggingface-cli upload <user>/off-the-prior-bench bench/dataset_card.md README.md --repo-type dataset
huggingface-cli upload <user>/off-the-prior-bench bench/data data --repo-type dataset
huggingface-cli upload <user>/off-the-prior-bench bench/configs configs --repo-type dataset
huggingface-cli upload <user>/off-the-prior-bench bench/leaderboard leaderboard --repo-type dataset
```

Replace `<user>` with the Hugging Face account or organization, here and in the download command in
`bench/README.md` (written by `bench/make_docs.py`). The repository is created public by default; add
`--private` to `repo create` to keep it private first.
