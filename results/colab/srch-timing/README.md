Timing run on a Colab A100 (session `otp-srch-timing`, job script `results/search/colab_jobs/job_timing.sh`) used to choose the equal budget.
Budget 360 Timestral calls per method on 5M, one search seed, then `scripts/search_transfer.py` on the top 5 of each method (15 genomes, 45 samples) to time scoring on each size.
It is a timing check; its fitness values are not used in `results/search/`. A first attempt of the same job lost its runtime before finishing and was rerun; only the rerun is kept.
