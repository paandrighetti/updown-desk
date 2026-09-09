"""Resolve a (symbol, window start) pair to CLOB token ids through the Gamma API."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from polymarket import AsyncPublicClient

from .windows import WINDOW_S, slug

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowMarket:
    symbol: str
    start: int
    end: int
    slug: str
    condition_id: str
    up_token: str
    down_token: str
    up_label: str | None
    down_label: str | None
    tick_size: float | None
    fees_enabled: bool | None
    fee_rate: float | None
    fee_exponent: float | None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def tokens(self) -> list[str]:
        return [self.up_token, self.down_token]


def _f(x) -> float | None:
    return None if x is None else float(x)


async def fetch_window(client: AsyncPublicClient, symbol: str, start: int) -> WindowMarket | None:
    """Return the market for one window, or None if Polymarket has not created it yet.

    Failures are expected: markets are created some time before their window, so a miss
    is retried by the scheduler. Any exception is therefore logged, not raised.
    """
    s = slug(symbol, start)
    try:
        event = await client.get_event(slug=s)
    except Exception as exc:  # noqa: BLE001 - discovery miss is a normal condition
        log.info("no event yet for %s (%s)", s, type(exc).__name__)
        return None
    if not event.markets:
        log.warning("event %s has no markets", s)
        return None
    m = event.markets[0]
    up, down = m.outcomes.yes, m.outcomes.no
    if up.label and not up.label.lower().startswith("up"):
        log.warning("%s: first outcome label is %r, expected Up", s, up.label)
    trading = m.trading
    fs = trading.fee_schedule if trading else None
    return WindowMarket(
        symbol=symbol,
        start=start,
        end=start + WINDOW_S,
        slug=s,
        condition_id=m.condition_id,
        up_token=up.token_id,
        down_token=down.token_id,
        up_label=up.label,
        down_label=down.label,
        tick_size=_f(trading.minimum_tick_size) if trading else None,
        fees_enabled=trading.fees_enabled if trading else None,
        fee_rate=_f(fs.rate) if fs else None,
        fee_exponent=_f(fs.exponent) if fs else None,
    )
