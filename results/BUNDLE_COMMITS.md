# Upload bundle commits

Some Colab jobs ran from a slim copy of the repository (result folders, benchmarks, paper and notebooks
removed) to keep the upload small. Their `commit.txt` names the slim copy's commit, which was never
published. Each slim copy has the same `otp/` and `scripts/` trees as a commit on the published history:

| job folder | bundle commit (unpublished) | same code as |
|---|---|---|
| results/finetune/colab/run1 | 1a32a4c | 64872b3 |
| results/colab/nn-123M, results/colab/nn-209M | 69140f1 | e9e880a |
| results/colab/nn-failure | 81b092b | 66dc7d3 |
| results/colab/srch-a2 | 9fb9396 | b5d7bce |
| results/colab/srch-b | 0e09977 | 2bd3089 |
| results/colab/srch-transfer | 22c6fb7 | b16990e |

The match was checked by comparing the git tree hashes of `otp/` and `scripts/`. All commit hashes in
result files refer to the history before it was cleaned for publication; `provenance/commit-map.txt`
maps them to the published hashes.
