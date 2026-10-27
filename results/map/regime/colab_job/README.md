# How scores.csv was produced (Colab, by the lead)

`scores.csv` = the concatenation of two Colab A100 runs on this folder's `samples.npz`
(md5 71cca8ecd1e7f1a73d04cf9059be930c, uploaded unchanged):

1. references: oracle (scored per context length) + seasonal naive, ETS, ARIMA —
   `scripts/colab/job_regime_refs.sh` (`provenance_references.json`, `job_references.log`);
2. Timestral 5M / 28M / 123M / 209M with `--no-references` —
   `scripts/colab/job_a_regime.sh` (`provenance_timestral.json`, `job_timestral.log`).

Contexts 512 / 2048 / 8192, horizons 16 / 128 / 1024. No oracle repair is needed (oracle rows
are already per context length). The VM run of this sweep was stopped before it saved anything.
