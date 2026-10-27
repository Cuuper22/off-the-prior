# Off the Prior

**Where does a time-series model trained on fake data stop working?**

Timestral is a family of open forecasting models from Mistral (5M, 28M, 123M and 209M
parameters, released October 15, 2026). You hand it the history of a number over time and
it tells you where that number is probably going, as a spread of 21 guesses from "very low"
to "very high" for every future step. It was trained mostly on synthetic series: simple
trends, repeating seasons and noise, mixed together. It still tops the public leaderboards
built from real data.

A model trained on a made-up world has a picture of what series look like (a "prior"). This
project finds the edges of that picture. I built signal generators whose difficulty I can
turn up slowly (chaos, sudden regime changes, wild outliers, long memory, and a generator
that learns to mix all of those), forecast them with all four Timestral sizes, and compared
every forecast with a referee that knows the true equations. Then I looked inside the model
to see why it fails, packaged the hard cases as benchmarks, and tried cheap fixes.

## What I found

**Near home, it matches the referee.** On series like its training data (trend + season + noise),
every size makes 0.99–1.15× the referee's error at horizons up to
128 steps.

**Smooth, nearly deterministic oscillators are where it falls off.** Delay equations
(Mackey-Glass), loosely coupled oscillators (Kuramoto) and the Rössler system leave the most on the
table: at τ = 30, Mackey-Glass errors are 21.87× the referee's for the smallest model and
13.68× for the largest. Timestral still beats the old-school methods there (median ratio
0.36 against the best classical model at 209M). For the chaotic Lorenz
system, the gap lives inside the first few "Lyapunov times" (how long a tiny error takes to grow by
e) and is gone after about ten.

**Fresh regime changes get noticed, then ignored.** Sixteen steps after a jump in level, the error is
3.77× the referee's at 5M and 2.43× at 209M. Inside the model, a simple
linear readout of the forecast token predicts the new level with R² of 0.956–0.980,
while the forecast itself gets 0.570–0.726: the information is there,
it just doesn't make it out.

**Wild outliers and long memory stay close.** Heavy-tailed noise, fractional Brownian motion, bursty
counts and multifractal cascades all land at 0.94–1.34× the referee's error.

**Helper series get used only when the link is immediate.** When a driver series sets the main series after
a lag and its future is handed over, Timestral-28M's gain from it falls from 4.39 at lag 0 to
0.95 at lag 512, while a forecaster using the driver perfectly would gain
5.62–7.91. Feeding the helper in two-way instead of one-way changes the
error by less than 5% in 95% of cases.

**On real data the pattern against old-school methods carries over in part.** Real series that stand in for the
failure families (sunspots, lynx, grid frequency, river floods, economic breaks) show a smaller lead over
old-school methods than the synthetic families do; there is no referee for real data, so the gap to the best
possible forecast can't be measured there.

**It's not the usual suspects.** Turning off its mirror-averaging trick changes errors by
0.991–1.069×, and cutting old history after a change helps only sometimes.

**A small trick helps without any training.** Let each series' own past decide: backtest
Timestral against a nearest-neighbour "method of analogues" on the end of the context and switch only
when the analogues win clearly. On fresh test series, chaos errors drop to 0.824× (5M) and
0.982× (209M) of before, while normal and regime-switching series are untouched.

**The hard cases are packaged as benchmarks.** `bench/` holds a tagged diagnostic set (372
series, each labelled with why it is hard) and a leaderboard set (3,776 series over three context
lengths and three horizons), both with the referee's forecast included. On the diagnostic set the largest
Timestral makes 2.00× the referee's error and seasonal naive
3.68×; one command scores any model (see `bench/README.md`).

Every number here is filled in by `scripts/make_readme.py` from `paper/numbers.json`, which points each
one at the result file it came from.

## Try it

```bash
pip install -e .        # also installs Timestral's inference code from GitHub
python -m pytest -q tests
python scripts/sweep.py --grid otp.grids.in_prior --sizes 5M --contexts 2048 \
    --horizons 16 128 --seeds 4 --out /tmp/in_prior
```

## What's in here

```text
otp/                 the library
  model.py           Timestral wrapper (NumPy in/out, plus a differentiable path)
  metrics.py         MASE, scaled CRPS, coverage, calibration
  classical.py       seasonal naive, ETS, ARIMA baselines
  generators/        signal families; each is a simulator with an exact oracle
  grids/             which settings each sweep visits
  internals.py       attention and hidden states of the forecast tokens
  mechanism.py       readouts and attention measurements
scripts/             sweeps, mechanism runs, search, fine-tuning
results/             every number in the writeup comes from a file here
bench/               the released benchmarks and their eval script
notebooks/           Colab notebooks for the GPU runs
paper/               the writeup (markdown and PDF)
docs/                the explainer site
```

## How a failure is counted

A big error is not automatically a failure: some series are just unpredictable. Every test
series here comes from a simulator, so I can run the simulator forward 500 times from the
exact state it was in when the history ended. That spread is the best forecast anyone could
make (the "oracle"). A failure is Timestral being much worse than the oracle, and I also
compare it with three old-school methods (repeat the last season, exponential smoothing,
ARIMA).

## License

Code: Apache 2.0. Timestral weights are Mistral's, under their modified MIT license, and are
downloaded from Hugging Face, not included here.
