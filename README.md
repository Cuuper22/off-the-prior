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

RESULTS-PENDING. (Filled only from the result files under `results/`.)

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
