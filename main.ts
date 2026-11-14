// btc-cent-forecast-01 entry: honest 72-hour mean-reversion forecaster (Node 22, stdlib only).
//
// Reads one case: 4,320 one-minute BTC/USD closes (72 h) up to a checkpoint, and writes one
// guess for the close 240 minutes later. The forecast is the checkpoint close moved by a small
// log-return assembled from three signals, all estimated from the case's own 72 h only:
//
//   1. 48 h drift: mean of the last 12 non-overlapping 240-minute log returns.
//   2. 72 h drift: mean of the last 18 non-overlapping 240-minute log returns.
//   3. Ridge: a heavily shrunk ridge regression (lambda 5000, 4 features) of the forward
//      240-minute log return on the trailing 1 h return, trailing 4 h return, trailing 24 h
//      return, and the log deviation of the last price from its 24 h mean, fitted on every
//      minute of the window whose 240-minute target is still inside the window.
//
//   forecast log-return = 0.5 * drift48 + 0.5 * drift72 + 0.25 * ridge, clipped to +/-1.2%.
//
// The blend is deliberate: the drift terms are the robust part (a 72 h trend that keeps
// running), the ridge is the adaptive part (it leans with recent momentum and against a
// stretched deviation from the 24 h mean), and the 0.25 weight plus the clip keep the total
// tilt small, so the entry stays close to persistence when the signals disagree. No network,
// no randomness, no clock: two runs on the same case write byte-identical output.
//
// Scoring is mean absolute error in cents, so the median forecast is optimal; this entry
// forecasts the conditional median as the checkpoint close plus the shrunk expected move.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The smallest positive price on the grid: one unit of its 5 decimals. */
const MIN_PRICE = 0.00001;
/** The largest grid price below the scorer's exclusive 10,000,000-dollar cap. */
const MAX_PRICE = 9_999_999.99999;
/** Forecast horizon in minutes. */
const HORIZON = 240;
/** Hard cap on the forecast log-return, so one wild window cannot throw the guess far. */
const CLIP = 0.012;
/** Ridge penalty: large on purpose; with ~2,600 overlapping rows the signal is weak. */
const RIDGE_LAMBDA = 5000;
/** Weight of the ridge signal in the final blend. */
const RIDGE_WEIGHT = 0.25;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  markets: CaseMarket[];
}

/** x kept on the grid's 5 decimals and inside the scorer's (0, 10,000,000). */
const price5 = (x: number): number => Math.round(Math.min(MAX_PRICE, Math.max(MIN_PRICE, x)) * 1e5) / 1e5;

const mean = (xs: number[]): number => {
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
};

/** Population standard deviation around a known mean. */
const pstdev = (xs: number[], m: number): number => {
  let s = 0;
  for (const x of xs) {
    const d = x - m;
    s += d * d;
  }
  return Math.sqrt(s / xs.length);
};

/** Solve A x = b by Gaussian elimination with partial pivoting (small dense system).
 *  Returns null when the system is (near-)singular, so the caller can fall back. */
function solveLinear(aIn: number[][], bIn: number[]): number[] | null {
  const n = bIn.length;
  const m = aIn.map((row, i) => [...row, bIn[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[piv][c])) piv = r;
    if (Math.abs(m[piv][c] as number) < 1e-12) return null;
    const tmp = m[c];
    m[c] = m[piv] as number[];
    m[piv] = tmp as number[];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = (m[r][c] as number) / (m[c][c] as number);
      for (let k = c; k <= n; k++) m[r][k] = (m[r][k] as number) - f * (m[c][k] as number);
    }
  }
  const x = m.map((row, i) => (row[n] as number) / (row[i] as number));
  return x.every(Number.isFinite) ? x : null;
}

/**
 * Mean of the non-overlapping HORIZON-minute log returns ending at the last price,
 * going back `hours` hours (at most). Empty when the series is too short.
 */
function blockDrift(logP: number[], hours: number): number {
  const n = logP.length;
  const floor = Math.max(0, n - 1 - hours * 60);
  const vals: number[] = [];
  let end = n - 1;
  while (end - HORIZON >= floor) {
    vals.push((logP[end] as number) - (logP[end - HORIZON] as number));
    end -= HORIZON;
  }
  return vals.length === 0 ? 0 : mean(vals);
}

/**
 * Ridge forecast of the forward HORIZON-minute log return, fitted on the case's own
 * window. Rows are every minute t with 24 h of history whose target t + HORIZON is
 * still inside the window. Features: trailing 1 h, 4 h and 24 h log returns and the
 * log deviation from the 24 h mean. Returns 0 when the series is too short.
 */
function ridgeForecast(prices: number[], logP: number[]): number {
  const n = prices.length;
  if (n < 1440 + HORIZON + 1) return 0;
  // Prefix sums of prices, for the 24 h mean at any minute in O(1).
  const ps: number[] = [0];
  for (const p of prices) ps.push((ps[ps.length - 1] as number) + p);
  const feat = (t: number): number[] => {
    const ma24 = ((ps[t + 1] as number) - (ps[t - 1439] as number)) / 1440;
    return [
      (logP[t] as number) - (logP[t - 60] as number),
      (logP[t] as number) - (logP[t - HORIZON] as number),
      (logP[t] as number) - (logP[t - 1440] as number),
      Math.log((prices[t] as number) / ma24),
    ];
  };
  const rows: number[][] = [];
  const ys: number[] = [];
  for (let t = 1440; t + HORIZON <= n - 1; t++) {
    rows.push(feat(t));
    ys.push((logP[t + HORIZON] as number) - (logP[t] as number));
  }
  const d = 4;
  const mu = [0, 1, 2, 3].map((i) => mean(rows.map((r) => r[i] as number)));
  const sd = [0, 1, 2, 3].map((i) =>
    Math.max(pstdev(rows.map((r) => r[i] as number), mu[i] as number), 1e-9),
  );
  const ym = mean(ys);
  const a: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  const b: number[] = new Array<number>(d).fill(0);
  for (let q = 0; q < rows.length; q++) {
    const z = [0, 1, 2, 3].map((i) => (((rows[q] as number[])[i] as number) - (mu[i] as number)) / (sd[i] as number));
    const yc = (ys[q] as number) - ym;
    for (let i = 0; i < d; i++) {
      b[i] = (b[i] as number) + (z[i] as number) * yc;
      for (let j = 0; j < d; j++) a[i][j] = (a[i][j] as number) + (z[i] as number) * (z[j] as number);
    }
  }
  for (let i = 0; i < d; i++) a[i][i] = (a[i][i] as number) + RIDGE_LAMBDA;
  const beta = solveLinear(a, b);
  // A singular system or a non-finite fit means the window carries no usable
  // signal: fall back to the drift terms alone instead of emitting a bad guess.
  if (beta === null) return 0;
  const f = feat(n - 1);
  const z = [0, 1, 2, 3].map((i) => ((f[i] as number) - (mu[i] as number)) / (sd[i] as number));
  let pred = ym;
  for (let i = 0; i < d; i++) pred += (beta[i] as number) * (z[i] as number);
  return Number.isFinite(pred) ? pred : 0;
}

/** Forecast log-return for the next HORIZON minutes after the last price. */
function forecastLogReturn(prices: number[]): number {
  const logP = prices.map((p) => Math.log(p));
  const drift = 0.5 * blockDrift(logP, 48) + 0.5 * blockDrift(logP, 72);
  const ridge = ridgeForecast(prices, logP);
  const raw = drift + RIDGE_WEIGHT * ridge;
  return Math.max(-CLIP, Math.min(CLIP, raw));
}

/** The forecast price for one market: last close moved by the forecast log-return. */
function forecast(m: CaseMarket): number {
  if (!Array.isArray(m.prices) || m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  for (const p of m.prices) {
    if (typeof p !== 'number' || !Number.isFinite(p) || p <= 0) {
      throw new Error(`market ${m.id} has a non-positive or non-finite price`);
    }
  }
  const last = m.prices[m.prices.length - 1] as number;
  const moved = last * Math.exp(forecastLogReturn(m.prices));
  // Never emit a non-finite guess: persistence is always a valid fallback.
  return price5(Number.isFinite(moved) ? moved : last);
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (!Array.isArray(c.markets) || c.markets.length !== 1) throw new Error(`${file} does not hold exactly one market`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts: [{ case: id, price: forecast(c.markets[0] as CaseMarket) }] })}\n`);
}
