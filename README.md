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

## Passive quoting replay (pre-registered 21 September 2026)

The taker result closes one question: the market is not mispriced against a diffusion fair
value, so there is nothing to take. The passive replay asks the question a liquidity
provider would ask instead: is the flow that hits resting quotes in these windows benign
enough to earn the spread plus the maker rebate, and how does that depend on time to expiry?
No orders are placed; the replay runs on the same recording, using the trade tape
(`last_trade_price` events) that the taker replay did not need.

Strategy: two-sided quotes on the Up token at `anchor +/- half_spread`, anchor either the
market mid or the model fair value, shifted against inventory by a linear skew (the discrete
analogue of the reservation price in Avellaneda and Stoikov, 2008), 20 shares a side,
inventory capped at 100 shares and held to resolution. Quotes are rounded to the tick and kept
one tick inside the opposite best, so a quote never takes.

Fill model, every choice of which can only under-count fills: quotes computed at a book
snapshot go live `latency_ms` after its receive time and stay live until the next snapshot's
quotes go live, so a stale quote can be picked off during the latency. A bid is filled by
taker SELL prints at or below it, an ask by taker BUY prints at or above it, on the same token.
A quote that improves the displayed best is first in line; one that joins or sits behind it
waits for the displayed best size to print at or through its price first. Matches through
the Down token (mint and merge against Down orders) are not observed and are ignored.
Maker rebate: 20 % of the taker fee on the fill (Polymarket's published crypto rate at the
time of writing; a parameter, and the report shows PnL with and without it).

Hypotheses written before the first run, with the statistic that decides each one:

- H1. Quoting around the market mid, outside the last 120 s, earns a positive PnL per
  window before rebates. Decided by the sign and t-statistic of window PnL in the cell
  `mid, 0.01, 250 ms, tau_min 120` over at least 14 complete days.
- H2. The flow hitting passive quotes in the last 120 s is informed: the 30 s markout of
  fills in the `[0, 120)` time-to-expiry buckets is negative and below the markout of the
  `[300, 900)` buckets. Decided by the markout table of the base cell.
- H3. Anchoring quotes on the model fair value does not beat anchoring on the mid, since
  the market is better calibrated than the model. Decided by comparing the two anchors cell
  by cell at equal half spread, latency and `tau_min`.
- Abandon criterion: if after 14 complete days every cell has a 30 s markout below minus
  one half spread and a negative PnL before rebates, passive quoting on these windows is
  declared unprofitable at 250 ms latency and the project moves on.

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
  passive.py  passive quoting replay against the trade tape, markouts by time to expiry
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
table; the taker replay grid (n, hit rate, PnL, PnL per trade, t-stat, return on deployed
capital, max drawdown, fees); the passive quoting grid (windows, shares traded per window,
PnL before and after rebates, t-stat of window PnL, max drawdown, 30 s and resolution
markouts) and the markout table of the base cell by time-to-expiry bucket; Brier scores and
calibration deciles for model versus market mid.

## Known limitations

- The settlement feed is the RTDS relay of the Chainlink TWAP, not the data stream
  itself; the 1.2 % residual disagreement is the size of that gap plus window-boundary
  effects.
- Volatility is estimated once per window from the trailing hour and not updated inside
  the window.
- Only full `book` snapshots are used for the top of book in replay; `price_change` deltas
  are recorded but not yet applied, so the replay sees the book at snapshot cadence only.
- Receive timestamps are local; clock offset to the exchange is not measured.
- The passive replay sees only the displayed best level: the size resting at a deeper
  level is unknown and the displayed best size stands in for it. Fills that would have
  arrived through the Down token are not counted. A print without a size on the wire
  counts as zero volume; the report states the share of prints carrying a size.
- Order placement on Polymarket is geographically restricted. This project reads public
  market data only.

## Roadmap

1. Apply `price_change` deltas to maintain the full book between snapshots, which would
   replace the displayed-best proxy in the passive queue rule.
2. Add Kalshi market data for the same assets and measure cross-venue divergence.
