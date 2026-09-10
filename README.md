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

## Fair value

Driftless geometric Brownian motion over the remaining life of the window, with `S_ref` the
first reference-feed observation at or after the window start and `sigma` the realized
variance estimator on the trailing hour of feed ticks (`sum(r^2) / sum(dt)`, annualized):

```
P(S_T >= S_ref | S_t) = Phi( ln(S_t / S_ref) / s - s / 2 ),   s = sigma * sqrt(tau)
```

The `s / 2` term is the Ito correction; at the money the fair value is slightly below 0.5.
Zero drift is a modeling choice: over 15 minutes the drift term is negligible next to the
diffusion term. Ties resolve Up, as in the market rules.

## Cost model

Taker fee per the market's `fee_schedule` from Gamma: `shares * rate * (p * (1 - p)) ^ exponent`.
Fills are marketable limits at the observed best ask, filled only if the first book snapshot
received after the configured latency still shows an ask at or below that price; size is
capped by the displayed size and 100 shares. One attempt per side per window, held to
resolution. This is deliberately pessimistic; a real desk would post and manage inventory.

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
              under data/derived/{windows,feeds,books,resolutions,coverage}; DuckDB does the
              JSON extraction with a memory cap so the reporter cannot starve the collector
  replay.py   window contexts, replay grid, checkpoints, feed agreement (from derived tables)
  report.py   reports/YYYY-MM-DD.md, reports/latest.md, Telegram digest
```

Messages are written unmodified; the collector gzips raw files two hours after their last
write. The raw layer is the archive (about 3 GB per day compressed, dominated by
`price_change`), the derived layer is what the report reads (tens of megabytes per day).
Interpretation happens at derivation and replay time, so a bug there never costs data, and
everything is reproducible from the raw files alone. A one-off report including the current
partial day: `updown-report --include-today`.

Disk lever: `UPDOWN_DROP_EVENTS=price_change` in `.env` stops recording quote deltas (95 % of
messages); the replay does not use them yet, the full-book reconstruction on the roadmap
would.

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

- The Chainlink data stream used for resolution is not the RTDS Chainlink relay; the
  relay is a proxy and question 1 above quantifies how good a proxy it is.
- Volatility is estimated once per window from the trailing hour and not updated inside
  the window.
- Only full `book` snapshots are used for the top of book in replay; `price_change` deltas
  are recorded but not yet applied, so the replay sees the book at snapshot cadence only.
- Receive timestamps are local; clock offset to the exchange is not measured.
- Order placement on Polymarket is geographically restricted. This project reads public
  market data only.

## Roadmap

1. Apply `price_change` deltas to maintain the full book between snapshots.
2. Replace the taker replay by a passive quoting replay with inventory (Avellaneda and
   Stoikov, 2008) using the same recording.
3. Add Kalshi market data for the same assets and measure cross-venue divergence.
