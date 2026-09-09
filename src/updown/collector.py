"""Record raw wire messages from the CLOB market stream and RTDS, hourly rotated JSONL.

Three concurrent tasks:
  scheduler   discovers upcoming windows through Gamma and keeps the token registry current
  clob_task   one WebSocket to the market channel, tokens added and removed in place
  rtds_task   one WebSocket to RTDS for Binance, Chainlink and Chainlink TWAP prices

Every message is written unmodified as {"rx_ts": <local epoch ms>, "msg": <wire payload>}.
Nothing is interpreted at collection time; interpretation happens in replay.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import websockets
from polymarket import AsyncPublicClient

from .config import Settings
from .discovery import WindowMarket, fetch_window
from .windows import slug, upcoming_starts

CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS = "wss://ws-live-data.polymarket.com"

log = logging.getLogger("updown.collector")


def now_ms() -> int:
    return time.time_ns() // 1_000_000


class Recorder:
    """Append-only JSONL writer with hourly file rotation."""

    def __init__(self, root: str, source: str) -> None:
        self.dir = os.path.join(root, "raw", source)
        os.makedirs(self.dir, exist_ok=True)
        self._hour: str | None = None
        self._fh = None
        self.count = 0

    def _path(self, ts_ms: int) -> tuple[str, str]:
        hour = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y%m%d_%H")
        return hour, os.path.join(self.dir, f"{hour}.jsonl")

    def write(self, msg: object) -> None:
        ts = now_ms()
        hour, path = self._path(ts)
        if hour != self._hour:
            if self._fh:
                self._fh.close()
            self._fh = open(path, "a", encoding="utf-8")  # noqa: SIM115 - long-lived handle
            self._hour = hour
        self._fh.write(json.dumps({"rx_ts": ts, "msg": msg}, separators=(",", ":")) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        if self._fh:
            self._fh.close()


class Registry:
    """Windows currently tracked, plus a queue of subscribe/unsubscribe operations."""

    def __init__(self) -> None:
        self.windows: dict[str, WindowMarket] = {}
        self.ops: asyncio.Queue[tuple[str, list[str]]] = asyncio.Queue()

    def tokens(self) -> set[str]:
        return {t for w in self.windows.values() for t in w.tokens}

    def add(self, w: WindowMarket) -> None:
        self.windows[w.slug] = w
        self.ops.put_nowait(("subscribe", w.tokens))

    def remove(self, key: str) -> None:
        w = self.windows.pop(key)
        self.ops.put_nowait(("unsubscribe", w.tokens))

    def drain(self) -> None:
        while not self.ops.empty():
            self.ops.get_nowait()


def _parse(raw: str | bytes) -> object | None:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None  # heartbeat replies such as PONG
    return obj if isinstance(obj, (dict, list)) else None


async def _heartbeat(ws, frame: str, every_s: float) -> None:
    while True:
        await asyncio.sleep(every_s)
        await ws.send(frame)


async def _reconnect_loop(name: str, run):
    backoff = 1.0
    while True:
        started = time.monotonic()
        try:
            await run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any transport error triggers a reconnect
            log.warning("%s: %s: %s", name, type(exc).__name__, exc)
        if time.monotonic() - started > 60:
            backoff = 1.0
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60.0)


async def rtds_task(rec: Recorder, settings: Settings) -> None:
    subscribe = {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices",
                "type": "update",
                "filters": ",".join(f"{s}usdt" for s in settings.symbols),
            },
            {"topic": "crypto_prices_chainlink", "type": "update"},
            {"topic": "crypto_prices_twap_thirty", "type": "update"},
            {"topic": "crypto_prices_twap_sixty", "type": "update"},
        ],
    }

    async def run() -> None:
        async with websockets.connect(RTDS_WS, ping_interval=None, max_size=None) as ws:
            await ws.send(json.dumps(subscribe))
            hb = asyncio.create_task(_heartbeat(ws, "PING", 5))
            log.info("rtds connected")
            try:
                async for raw in ws:
                    obj = _parse(raw)
                    if obj is not None:
                        rec.write(obj)
            finally:
                hb.cancel()

    await _reconnect_loop("rtds", run)


async def clob_task(rec: Recorder, registry: Registry) -> None:
    async def sender(ws) -> None:
        while True:
            op, ids = await registry.ops.get()
            await ws.send(json.dumps({"assets_ids": ids, "operation": op}))
            log.info("clob %s %d tokens", op, len(ids))

    async def run() -> None:
        while not registry.tokens():
            await asyncio.sleep(5)
        async with websockets.connect(CLOB_WS, ping_interval=None, max_size=None) as ws:
            registry.drain()  # the full set is sent below; queued deltas are stale
            await ws.send(
                json.dumps(
                    {
                        "assets_ids": sorted(registry.tokens()),
                        "type": "market",
                        "custom_feature_enabled": True,
                    }
                )
            )
            log.info("clob connected, %d tokens", len(registry.tokens()))
            aux = [
                asyncio.create_task(_heartbeat(ws, "PING", 10)),
                asyncio.create_task(sender(ws)),
            ]
            try:
                async for raw in ws:
                    obj = _parse(raw)
                    if obj is None:
                        continue
                    if isinstance(obj, list):
                        for item in obj:
                            rec.write(item)
                    else:
                        rec.write(obj)
            finally:
                for t in aux:
                    t.cancel()

    await _reconnect_loop("clob", run)


async def scheduler(registry: Registry, meta: Recorder, settings: Settings) -> None:
    retry_after: dict[str, int] = {}
    async with AsyncPublicClient() as client:
        while True:
            now = int(time.time())
            for symbol in settings.symbols:
                for start in upcoming_starts(now, settings.horizon_s):
                    key = slug(symbol, start)
                    if key in registry.windows or retry_after.get(key, 0) > now:
                        continue
                    w = await fetch_window(client, symbol, start)
                    if w is None:
                        retry_after[key] = now + 60
                        continue
                    registry.add(w)
                    meta.write(w.to_dict())
                    log.info("tracking %s", key)
            for key, w in list(registry.windows.items()):
                if now > w.end + settings.grace_s:
                    registry.remove(key)
                    log.info("released %s", key)
            for key in [k for k, t in retry_after.items() if t < now - 3600]:
                del retry_after[key]
            await asyncio.sleep(30)


async def _status(recs: dict[str, Recorder]) -> None:
    while True:
        await asyncio.sleep(300)
        log.info("counts %s", {k: r.count for k, r in recs.items()})


async def _compress_loop(root: str, every_s: int = 1800) -> None:
    """Gzip raw files not written to for two hours, so disk use tracks the compressed rate."""
    from .store import compress_old_files

    while True:
        await asyncio.sleep(every_s)
        try:
            n = await asyncio.to_thread(compress_old_files, root, 7200)
            if n:
                log.info("compressed %d raw files", n)
        except OSError as exc:
            log.warning("compression failed: %s", exc)


async def run(settings: Settings) -> None:
    recs = {
        "rtds": Recorder(settings.data_dir, "rtds"),
        "clob": Recorder(settings.data_dir, "clob"),
        "windows": Recorder(settings.data_dir, "windows"),
    }
    registry = Registry()
    try:
        await asyncio.gather(
            rtds_task(recs["rtds"], settings),
            clob_task(recs["clob"], registry),
            scheduler(registry, recs["windows"], settings),
            _status(recs),
            _compress_loop(settings.data_dir),
        )
    finally:
        for r in recs.values():
            r.close()


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("UPDOWN_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(run(Settings()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
