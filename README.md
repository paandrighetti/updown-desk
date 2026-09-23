# updown-desk

Lossless recorder and fair-value replay desk for Polymarket 15-minute crypto Up/Down markets
(BTC, ETH, SOL, XRP). It records the order book and the reference price feeds continuously,
then replays a simple taker strategy against the recording with explicit latency and fee
assumptions, and publishes a daily report. No orders are placed; the execution layer does
not exist in this repository.

## Questions the recording is meant to answer

1. **Which feed does the resolution follow?** The market description names the Chainlink
   BTC/USD data stream. Polymarket's RTDS relays Binance, Chainlink and Chainlink 30 s and
   60 s TWAP prices. The report measures, per feed, how often `sign(last - first)` over the
   window matches the resolved outcome. Until this is close to 100 % for one feed, no fair
   value can be trusted.
2. **Is the market mispriced relative to a driftless diffusion fair value, after fees and
   latency?** Null hypothesis: no positive expected PnL once the taker fee curve, the
   displayed size and a realistic latency are applied. The replay grid sweeps edge
   threshold and latency; the t-statistic per cell is reported, not hidden.
3. **Is the model better calibrated than the market mid?** Brier scores at fixed checkpoints
   inside the window (5, 10 and 14 minutes) and a decile calibration table.

## First results (six days, 2 548 resolved windows, September 2026)

- Resolution follows the 60-second Chainlink TWAP: sign(last - first) over the window
  agrees with the settled outcome 98.8 % of the time on `crypto_prices_twap_sixty`, against
  96.0 % for the 30 s TWAP, 92.3 % for the Chainlink spot relay and 91.8 % for Binance.
- The spot-settled fair value is worse than the market at every checkpoint (Brier 0.132
  against 0.097 at 14 minutes) and overconfident in the extreme deciles.
- Taking the market against that fair value loses in all twelve cells of the grid, t-stat
  between -2.4 and -3.4, hit rate 45 to 50 %, fees explaining about half the loss. No edge;
  the market is better informed than the model, and the model was pricing the wrong
  contract. This is the result the project was built to be able to state.

## Fair value

Two settlement models, selected by `UPDOWN_SETTLEMENT`.

`spot`: driftless geometric Brownian motion, `S_ref` the reference feed at window start,
`sigma` the realized variance estimator on the trailing hour of ticks:

```
P(S_T >= S_ref | S_t) = Phi( ln(S_t / S_ref) / s - s / 2 ),   s = sigma * sqrt(tau)
```

`twap60` (default since the measurement above): the contract settles on the average price
over the last 60 s of the window. With X the log price, Y its average over [T - w, T]:

```
tau >= w : Y - X_t ~ N(0, sigma^2 (tau - w + w/3))
tau <  w : Y = (K + tau X_t + e) / w,   e ~ N(0, sigma^2 tau^3 / 3)
```

where K is the realized integral of the log price over [T - w, t], computed from ticks. The
strike is the TWAP feed value at window start. Near expiry this contract has far less
remaining variance than a spot-settled one, which is where the spot model failed most.

## Cost model

Taker fee per the market's `fee_schedule` from Gamma: `shares * rate * (p * (1 - p)) ^ exponent`.
Fills are marketable limits at the observed best ask, filled only if the first book snapshot
received after the configured latency still shows an ask at or below that price; size is
capped by the displayed size and 100 shares. One attempt per side per window, held to
resolution. This is deliberately pessimistic; a real desk would post and manage inventory.

## Passive quoting study (September 2026)

<!-- passive-study:start -->
**Passive quoting, closed on 23 September 2026.** A first maker replay lost money in all 16
pre-registered cells, but it updated its quotes only after trades and ignored the flow of the
Down token, so it measured its own design more than the market. Measured on the tape instead,
the liquidity resting on these books has a realized spread of -0.26 and +0.05 cents per share at
30 s in the two samples, before the maker rebate. One pre-registered bucket passes, prices from
0.21 to 0.39 and from 0.61 to 0.79, and earns about the rebate. There, the fills that empty
their price level lose 0.56 and 0.35 cents per share before the rebate and are not
distinguishable from zero with it. A second rule, written before that split was computed,
required them to pass, so no quote was simulated. Full study, generated from the raw outputs:
[docs/passive-quoting.md](docs/passive-quoting.md).
<!-- passive-study:end -->

The first replay was pre-registered on 21 September 2026 and ran in the daily report of version
0.2.0. Its code stays in `src/updown/passive.py`, but the report no longer runs it. The note
reproduces that pre-registration as written and marks the two statements in it that proved wrong.
The two follow-up studies were pre-registered in `scripts/research/`; their unedited outputs are
in `docs/results/`, from which `scripts/research/passive_note.py` generates the note and the
summary above.

## Architecture

```
collector (one process, three tasks)
  scheduler  -> Gamma API: resolves btc-updown-15m-<start> slugs to token ids
  clob_task  -> wss://ws-subscriptions-clob.polymarket.com/ws/market  (book, price_change,
                last_trade_price, best_bid_ask, new_market, market_resolved)
  rtds_task  -> wss://ws-live-data.polymarket.com  (crypto_prices, crypto_prices_chainlink,
                crypto_prices_twap_thirty, crypto_prices_twap_sixty)
  writes data/raw/{clob,rtds,windows}/YYYYMMDD_HH.jsonl  as {"rx_ts": ms, "msg": wire payload}

reporter (daily, 06:00 UTC)
  store.py    derives each complete UTC day of raw files, once, into compact parquet tables
              under data/derived/{windows,feeds,books,trades,resolutions,coverage}; DuckDB
              does the JSON extraction with a memory cap so the reporter cannot starve the
              collector
  resolve.py  Gamma sweep: settled outcome for every window the stream did not resolve
  replay.py   window contexts, taker replay grid, checkpoints, feed agreement
  passive.py  first passive quoting replay (v0.2), kept for reproducibility, not run
  report.py   reports/YYYY-MM-DD.md, reports/latest.md, Telegram digest
```

Messages are written unmodified; the collector gzips raw files two hours after their last
write. The raw layer is the archive (about 3 GB per day compressed, dominated by
`price_change`), the derived layer is what the report reads (tens of megabytes per day).
Interpretation happens at derivation and replay time, so a bug there never costs data, and
everything is reproducible from the raw files alone. A one-off report including the current
partial day: `updown-report --include-today`.

Disk lever: `UPDOWN_DROP_EVENTS=price_change` in `.env` stops recording quote deltas (95 % of
messages). The server has run with it since 22 September 2026. The archive recorded from 9 to
22 September keeps the deltas, which a maker backtest needs to rebuild the book between trades
(docs/passive-quoting.md).

## Run

```
cp .env.example .env            # Telegram fields optional
docker compose up -d --build    # collector and daily reporter, restart on failure
docker compose logs -f collector
```

Reports appear in `reports/` every day at 06:00 UTC. To publish them, add one crontab
line on the host (see `scripts/publish_reports.sh`). Without Docker:

```
pip install -e .[dev]
updown-collect                  # foreground, Ctrl-C to stop
updown-report                   # one report from whatever has been recorded
pytest -q
```

## What the daily report contains

Message counts and the largest silent gap per hour and per source; the feed agreement
table; the replay grid (n, hit rate, PnL, PnL per trade, t-stat, return on deployed capital,
max drawdown, fees); Brier scores and calibration deciles for model versus market mid.

## Known limitations

- The settlement feed is the RTDS relay of the Chainlink TWAP, not the data stream
  itself; the 1.2 % residual disagreement is the size of that gap plus window-boundary
  effects.
- Volatility is estimated once per window from the trailing hour and not updated inside
  the window.
- Only full `book` snapshots are used for the top of book in replay, and they arrive once per
  trade (docs/passive-quoting.md). Above 0 ms of latency, the taker fill check therefore reads
  the book left by the next trade rather than the book after the latency, which leans fills
  toward adverse ones. In the first results the 0 ms cells, free of this, lose as well. The
  `price_change` deltas that would remove it are no longer recorded (see the disk lever above).
- Receive timestamps are local; clock offset to the exchange is not measured.
- Order placement on Polymarket is geographically restricted. This project reads public
  market data only.

## Roadmap

1. Add Kalshi market data for the same assets and measure cross-venue divergence.
