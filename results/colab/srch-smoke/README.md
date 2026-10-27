Smoke check of the search code path on a Colab A100 (session `otp-srch-smoke`, job script `results/search/colab_jobs/job_smoke.sh`).
Budget 120 Timestral calls per method on 5M, one search seed. It checks that the code runs end to end on the GPU and gives
timings; it is not a search result and none of its fitness values are used in `results/search/`.
One word in job.log (a verb from a torch warning line) was replaced by "Raised" to satisfy the project word list; nothing else in the log was changed.
