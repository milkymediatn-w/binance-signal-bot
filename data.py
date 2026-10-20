"""
Market data layer: Binance public klines (no API key needed).

* BTCUSDT  -> Binance spot
* PAXGUSDT -> Binance spot (gold-backed token; Binance has no spot XAUUSD)
* XAUUSDT  -> Binance USD-M perpetual (optional, thinner liquidity)

Spot reads try data-api.binance.vision first (market-data-only mirror), then the
regular hosts. Binance blocks some regions (e.g. US IPs return HTTP 451): host
the bot in a supported region such as Frankfurt or Singapore.

Klines include taker-buy base volume, which gives REAL buy/sell flow (delta).
Returns CLOSED, UTC-indexed, de-duplicated, fresh candles. Functions are
synchronous: call them with asyncio.to_thread() from async code.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from config import Config

log = logging.getLogger(__name__)
UA = {"User-Agent": "binance-signal-bot/3.0"}
COLS = ["open", "high", "low", "close", "volume", "taker_buy"]
_INTERVAL = {1: "1m", 5: "5m", 15: "15m", 60: "1h"}
FUTURES_HOST = "https://fapi.binance.com"
PAGE = 1000


class DataError(RuntimeError):
    """Feed unavailable or malformed."""


class StaleDataError(DataError):
    """Feed answered but the latest closed candle is too old."""


@dataclass
class Feed:
    df: pd.DataFrame               # closed base-timeframe candles
    htf: Optional[pd.DataFrame]    # closed higher-timeframe candles (resampled)
    source: str
    has_volume: bool
    age_s: float                   # seconds since the last closed candle ended


# ----------------------------------------------------------------- HTTP ----
def _binance_get(cfg: Config, symbol: str, params: dict):
    if cfg.is_futures(symbol):
        hosts, path = [FUTURES_HOST], "/fapi/v1/klines"
    else:
        hosts, path = cfg.spot_hosts(), "/api/v3/klines"
    errors: list = []
    for attempt in range(2):
        blocked = 0
        errors = []
        for host in hosts:
            try:
                r = httpx.get(host + path, params=params, headers=UA, timeout=10.0)
            except httpx.HTTPError as e:
                errors.append(f"{host}: {type(e).__name__}")
                continue
            sc = r.status_code
            if sc in (451, 403):
                blocked += 1
                errors.append(f"{host}: HTTP {sc}")
                continue
            if sc in (418, 429):
                raise DataError(f"binance rate limited (HTTP {sc})")
            if sc == 400:
                raise DataError(f"binance rejected request: {r.text[:120]}")
            if sc >= 500:
                errors.append(f"{host}: HTTP {sc}")
                continue
            try:
                return r.json()
            except ValueError:
                errors.append(f"{host}: invalid JSON")
        if blocked == len(hosts):
            raise DataError("Binance blocks this server's region (HTTP 451/403). "
                            "Redeploy the service in Frankfurt or Singapore.")
        time.sleep(0.5)
    raise DataError("binance request failed: " + "; ".join(errors))


def fetch_binance(cfg: Config, symbol: str, bars: int) -> pd.DataFrame:
    interval = _INTERVAL[cfg.tf_min]
    rows: list = []
    end_ms = None
    for _ in range(max(1, math.ceil(bars / PAGE))):
        params = {"symbol": symbol.upper(), "interval": interval, "limit": PAGE}
        if end_ms is not None:
            params["endTime"] = end_ms
        data = _binance_get(cfg, symbol, params)
        if not isinstance(data, list):
            raise DataError(f"binance: unexpected response {str(data)[:120]}")
        if not data:
            break
        rows = data + rows
        end_ms = int(data[0][0]) - 1
        if len(data) < PAGE:
            break
        time.sleep(0.1)
    if not rows:
        raise DataError(f"binance returned no candles for {symbol}")
    try:
        # [open_time, open, high, low, close, volume, close_time, quote_vol, trades, taker_buy_base, ...]
        a = np.array([[r[0], r[1], r[2], r[3], r[4], r[5], r[9]] for r in rows], dtype=float)
    except (IndexError, TypeError, ValueError) as e:
        raise DataError(f"binance: malformed kline payload ({e})") from e
    idx = pd.to_datetime(a[:, 0].astype("int64"), unit="ms", utc=True)
    df = pd.DataFrame({"open": a[:, 1], "high": a[:, 2], "low": a[:, 3], "close": a[:, 4],
                       "volume": a[:, 5], "taker_buy": a[:, 6]}, index=idx)
    return df.iloc[-bars:]


# ------------------------------------------------------------- cleaning ----
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "taker_buy" not in df.columns:
        df["taker_buy"] = 0.0
    df = df[COLS].astype(float)
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df[["volume", "taker_buy"]] = df[["volume", "taker_buy"]].fillna(0.0)
    return df.dropna(subset=["open", "high", "low", "close"])


def drop_unclosed(df: pd.DataFrame, tf_min: int, now: datetime) -> pd.DataFrame:
    """Index is candle OPEN time; a candle is closed once open + tf <= now."""
    cutoff = pd.Timestamp(now) - pd.Timedelta(minutes=tf_min)
    return df[df.index <= cutoff]


def resample_htf(df: pd.DataFrame, base_min: int, htf_min: int) -> pd.DataFrame:
    """Resample closed base candles to the HTF, keeping only completed buckets."""
    if htf_min <= base_min:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "taker_buy": "sum"}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    r = df.resample(f"{htf_min}min", label="left", closed="left").agg(agg)
    r = r.dropna(subset=["open", "close"])
    if r.empty:
        return r
    last_close = df.index[-1] + pd.Timedelta(minutes=base_min)
    r = r[r.index + pd.Timedelta(minutes=htf_min) <= last_close]
    if len(r) and df.index[0] > r.index[0]:     # first bucket only partially covered
        r = r.iloc[1:]
    return r


def market_open(cfg: Config, symbol: str, now: Optional[datetime] = None) -> bool:
    """Gold-linked instruments trade 24/7 on Binance but real gold does not;
    off-hours prices are thin and drift. Skip them when gold_follow_market_hours."""
    if not cfg.is_gold(symbol) or not cfg.gold_follow_market_hours:
        return True
    now = now or datetime.now(timezone.utc)
    wd, hm = now.weekday(), now.hour * 60 + now.minute
    if wd == 5:
        return False                                  # Saturday
    if wd == 4 and hm >= 22 * 60:
        return False                                  # Friday close
    if wd == 6 and hm < 23 * 60:
        return False                                  # Sunday open
    if wd in (0, 1, 2, 3) and 22 * 60 <= hm < 23 * 60:
        return False                                  # daily break
    return True


# ---------------------------------------------------------------- public ----
def get_history(cfg: Config, symbol: str, bars: Optional[int] = None):
    """Closed, cleaned candles. Returns (df, source, has_volume)."""
    raw = fetch_binance(cfg, symbol, bars or cfg.bars)
    df = drop_unclosed(_clean(raw), cfg.tf_min, datetime.now(timezone.utc))
    if df.empty:
        raise DataError(f"no closed candles for {symbol}")
    source = "binance-futures" if cfg.is_futures(symbol) else "binance-spot"
    return df, source, bool(df["volume"].sum() > 0)


def get_feed(cfg: Config, symbol: str) -> Feed:
    df, source, has_vol = get_history(cfg, symbol)
    if len(df) < cfg.min_bars:
        raise DataError(f"only {len(df)} closed candles for {symbol} (need {cfg.min_bars})")
    now = datetime.now(timezone.utc)
    age = (pd.Timestamp(now) - (df.index[-1] + pd.Timedelta(minutes=cfg.tf_min))).total_seconds()
    if age > cfg.max_stale_bars * cfg.tf_min * 60:
        raise StaleDataError(f"{symbol} last closed candle is {age / 60:.1f} min old")
    htf = resample_htf(df, cfg.tf_min, cfg.htf_min) if cfg.htf_filter else None
    return Feed(df=df, htf=htf, source=source, has_volume=has_vol, age_s=age)
