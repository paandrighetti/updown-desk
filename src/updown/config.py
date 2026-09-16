"""Runtime settings, all overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    data_dir: str = os.environ.get("UPDOWN_DATA_DIR", "data")
    reports_dir: str = os.environ.get("UPDOWN_REPORTS_DIR", "reports")
    symbols: tuple[str, ...] = tuple(
        s.strip().lower() for s in os.environ.get("UPDOWN_SYMBOLS", "btc,eth,sol,xrp").split(",")
    )
    # Discover windows starting within this horizon (seconds). 1500 s covers the current
    # window and the next one.
    horizon_s: int = int(os.environ.get("UPDOWN_HORIZON_S", "1500"))
    # Keep a window subscribed this long after its end so the market_resolved event is captured.
    grace_s: int = int(os.environ.get("UPDOWN_GRACE_S", "600"))
    # CLOB event types to skip at collection, comma separated (e.g. "price_change"). Empty keeps
    # everything. Only a disk-space lever: dropped events are gone for good.
    drop_events: tuple[str, ...] = tuple(
        s.strip() for s in os.environ.get("UPDOWN_DROP_EVENTS", "").split(",") if s.strip()
    )
    # RTDS topic whose value at window start is the strike and whose value at window end
    # settles the market. Measured: the 60 s Chainlink TWAP agrees with resolutions 98.8 %.
    ref_feed: str = os.environ.get("UPDOWN_REF_FEED", "crypto_prices_twap_sixty")
    # RTDS topic used as the diffusing state (spot) in the fair value.
    spot_feed: str = os.environ.get("UPDOWN_SPOT_FEED", "crypto_prices_chainlink")
    # "twap60" prices a time-weighted-average settlement; "spot" a point settlement.
    settlement: str = os.environ.get("UPDOWN_SETTLEMENT", "twap60")
    # Seconds without a message before a stream socket is declared dead and reopened.
    stall_rtds_s: float = float(os.environ.get("UPDOWN_STALL_RTDS_S", "15"))
    stall_clob_s: float = float(os.environ.get("UPDOWN_STALL_CLOB_S", "90"))
    telegram_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "")
