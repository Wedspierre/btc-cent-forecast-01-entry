# btc-cent-forecast-01-entry

Forecast Bitcoin's price 4 hours ahead — [make.fun](https://make.fun) bounty
`btc-cent-forecast-01` ($250). Owner: **Wedspierre** (referral `ta01qs0h`).

## Entry

`main.ts` (Node 22, standard library only, no network, deterministic) reads one case
— 4,320 one-minute BTC/USD closes, 72 hours up to a checkpoint — and writes one guess
for the close 240 minutes later.

The forecast is the checkpoint close moved by a small log-return from three signals,
all estimated from the case's own 72 hours:

1. **48 h drift** — mean of the last 12 non-overlapping 240-minute log returns.
2. **72 h drift** — mean of the last 18 non-overlapping 240-minute log returns.
3. **Ridge** — a heavily shrunk ridge regression (λ = 5000) of the forward 240-minute
   log return on the trailing 1 h / 4 h / 24 h returns and the log deviation from the
   24 h mean, fitted on every minute of the window.

`forecast = 0.5 · drift48 + 0.5 · drift72 + 0.25 · ridge`, clipped to ±1.2%.
The tilt stays small by design, so the entry stays close to persistence when the
signals disagree. Scoring is mean absolute error in cents, so the entry forecasts
the conditional median.

## Practice score (local, 6 cases p00–p05)

| | MAE (cents) |
|---|---|
| This entry (p00–p04, exact targets) | 40,333.82 |
| Persistence baseline (p00–p04) | 47,054.65 |
| This entry (6 cases; p05 target ±$318.56 from the published baseline total) | 36,738.70 / 41,103.06 |
| Persistence baseline (6 cases, published) | 44,521.58 |

Beats the house persistence baseline on practice in both readings of the p05 target
(4 of 5 exact cases won; p03 lost — the window's bounce case). A 54-point
walk-forward check inside the same windows says the tilt costs a few percent when
the trend does not continue, so the blend is kept small rather than chased further.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## License

MIT — see `LICENSE`.
