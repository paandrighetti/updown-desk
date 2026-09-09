"""Send a short text digest to Telegram. Silently a no-op when credentials are absent."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)


def send(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        return False
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True},
            timeout=20,
        )
        r.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        log.warning("telegram send failed: %s", exc)
        return False
