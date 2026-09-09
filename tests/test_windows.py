from updown.windows import base_symbol, parse_slug, slug, upcoming_starts, window_start


def test_window_start_aligns_to_15_minutes():
    assert window_start(1771210800 + 37) == 1771210800
    assert window_start(1771210800) == 1771210800


def test_upcoming_covers_current_and_next():
    now = 1771210800 + 100
    assert upcoming_starts(now, 1500) == [1771210800, 1771211700]
    assert upcoming_starts(now, 1800) == [1771210800, 1771211700, 1771212600]


def test_slug_roundtrip():
    s = slug("btc", 1771210800)
    assert s == "btc-updown-15m-1771210800"
    assert parse_slug(s) == ("btc", 1771210800)
    assert parse_slug("will-it-rain") is None


def test_base_symbol():
    assert base_symbol("btcusdt") == "btc"
    assert base_symbol("btc/usd") == "btc"
    assert base_symbol("BTC/USD") == "btc"
