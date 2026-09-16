"""Ground-truth outcomes from Gamma for windows without a market_resolved stream event.

The market channel only delivers market_resolved for a fraction of windows. Gamma exposes the
settled outcome prices of every closed market, so one request per unresolved window gives a
complete outcome set, independent of the price feeds whose agreement is under test.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import pandas as pd
from polymarket import AsyncPublicClient

from . import store

log = logging.getLogger("updown.resolve")
TABLE = "outcomes"


def _settled(price) -> int | None:
    if price is None:
        return None
    p = float(price)
    if p >= 0.99:
        return 1
    if p <= 0.01:
        return 0
    return None


async def _fetch(client: AsyncPublicClient, slug: str, attempts: int = 3):
    for attempt in range(attempts):
        try:
            return await client.get_event(slug=slug)
        except Exception as exc:  # noqa: BLE001 - transient errors and rate limits alike
            log.info("%s attempt %d: %s", slug, attempt, type(exc).__name__)
            await asyncio.sleep(2.0**attempt)
    return None


async def sweep_day(root: str, day: str, pace_s: float = 0.25) -> int:
    """Write derived/outcomes/<day>.parquet for windows starting that day. Idempotent."""
    target = os.path.join(root, "derived", TABLE, f"{day}.parquet")
    if os.path.exists(target):
        return 0
    windows = store.load_derived(root, "windows")
    if windows.empty:
        return 0
    day_start = int(pd.Timestamp(day, tz="UTC").timestamp())
    todo = windows[(windows["start"] >= day_start) & (windows["start"] < day_start + 86400)]
    known = set(store.load_derived(root, "resolutions").get("condition_id", pd.Series(dtype=str)))
    todo = todo[~todo["condition_id"].isin(known)]
    rows: list[dict] = []
    async with AsyncPublicClient() as client:
        for w in todo.itertuples(index=False):
            event = await _fetch(client, w.slug)
            await asyncio.sleep(pace_s)
            if event is None or not event.markets:
                continue
            outcome = _settled(event.markets[0].outcomes.yes.price)
            if outcome is None:
                continue  # not settled yet, picked up by a later sweep
            rows.append(
                {
                    "rx_ts": int(time.time() * 1000),
                    "condition_id": w.condition_id,
                    "winning_token": w.up_token if outcome == 1 else w.down_token,
                    "source": "gamma",
                }
            )
    if len(rows) < len(todo):
        log.info(
            "%s: %d of %d unresolved windows still unsettled on Gamma",
            day,
            len(todo) - len(rows),
            len(todo),
        )
        if not rows:
            return 0
    os.makedirs(os.path.dirname(target), exist_ok=True)
    pd.DataFrame(rows, columns=["rx_ts", "condition_id", "winning_token", "source"]).to_parquet(
        target, index=False
    )
    return len(rows)


def sweep_pending(root: str, days: list[str]) -> None:
    for day in days:
        try:
            n = asyncio.run(sweep_day(root, day))
        except Exception:  # noqa: BLE001 - a failed sweep must not block the report
            log.exception("gamma sweep failed for %s", day)
            continue
        if n:
            log.info("gamma outcomes %s: %d windows", day, n)
