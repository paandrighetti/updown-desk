"""15-minute window arithmetic and Polymarket slug conventions.

Observed slug convention: ``btc-updown-15m-<unix_start>`` where ``unix_start`` is the UTC
epoch second of the window start, aligned on a 15-minute grid. The same pattern is assumed
for eth, sol and xrp; override with UPDOWN_SLUG_TEMPLATE if Polymarket changes it.
"""

from __future__ import annotations

import os
import re

WINDOW_S = 900
SLUG_TEMPLATE = os.environ.get("UPDOWN_SLUG_TEMPLATE", "{symbol}-updown-15m-{start}")
_SLUG_RE = re.compile(r"^(?P<symbol>[a-z]+)-updown-15m-(?P<start>\d{9,11})$")


def window_start(ts: int) -> int:
    return ts - ts % WINDOW_S


def upcoming_starts(now: int, horizon_s: int) -> list[int]:
    """Window starts covering [now, now + horizon_s], including the current window."""
    first = window_start(now)
    last = window_start(now + horizon_s)
    return list(range(first, last + 1, WINDOW_S))


def slug(symbol: str, start: int) -> str:
    return SLUG_TEMPLATE.format(symbol=symbol.lower(), start=start)


def parse_slug(value: str) -> tuple[str, int] | None:
    m = _SLUG_RE.match(value)
    if not m:
        return None
    return m.group("symbol"), int(m.group("start"))


def rtds_symbol(symbol: str, topic: str) -> str:
    """Map a base symbol to the RTDS symbol format of a topic."""
    if topic == "crypto_prices":
        return f"{symbol}usdt"
    return f"{symbol}/usd"


def base_symbol(rtds_sym: str) -> str:
    """Inverse of rtds_symbol for both Binance and Chainlink formats."""
    s = rtds_sym.lower()
    if s.endswith("usdt"):
        return s[:-4]
    if s.endswith("/usd"):
        return s[:-4]
    return s
