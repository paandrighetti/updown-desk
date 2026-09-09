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
    # RTDS topic used as the reference price in replay. The resolution source is Chainlink.
    ref_feed: str = os.environ.get("UPDOWN_REF_FEED", "crypto_prices_chainlink")
    telegram_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "")
