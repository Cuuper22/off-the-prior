# results/map/chaos_big

No separate run in this folder. The 123M and 209M models were run on the full chaos grid
(`otp.grids.chaos`, 43 configs x 8 seeds, T = 512, 2048, 8192, H = 16, 128, 1024, 2048) in the
same Colab A100 job as 5M and 28M, and their rows are in `results/map/chaos/scores.csv` and
`results/map/chaos/summary.csv` (forecasters `timestral-123M` and `timestral-209M`; tables in
`results/map/chaos/README.md`). The reduced-grid CPU run that had been planned for this folder
was not run. The id list for it, `BIG_ONLY`, remains in `otp/grids/chaos.py` and is unused.
