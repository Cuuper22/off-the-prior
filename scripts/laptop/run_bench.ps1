# Builds both benchmark sets and scores every baseline, as run on the Windows laptop lane (RTX 3060) for the
# published files in bench/data and bench/leaderboard. Code checked out at code2\ (commit in each summary.json).
$ErrorActionPreference = "Stop"
Set-Location C:\Users\Acer\off-the-prior-lane\code2
$env:OTP_DEVICE = "cuda"; $env:OTP_THREADS = "4"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\Acer\off-the-prior-lane\venv\Scripts\python.exe"
& $py bench\build.py --set diag --workers 14 --out ..\bench_out\data
if ($LASTEXITCODE -ne 0) { "build diag failed" | Out-File ..\bench_out\EXIT; exit 1 }
& $py bench\build.py --set lb --workers 14 --out ..\bench_out\data
if ($LASTEXITCODE -ne 0) { "build lb failed" | Out-File ..\bench_out\EXIT; exit 1 }
foreach ($set in @("diag","lb")) { foreach ($m in @("oracle","seasonal_naive","ets","arima","timestral-5M","timestral-28M","timestral-123M","timestral-209M")) {
  & $py bench\eval.py --set $set --split all --model $m --data ..\bench_out\data --workers 14 --out ..\bench_out\leaderboard
  if ($LASTEXITCODE -ne 0) { "eval $set $m failed" | Out-File ..\bench_out\EXIT; exit 1 }
} }
"0" | Out-File ..\bench_out\EXIT
