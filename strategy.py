"""
Strategy engine: Smart Money Concepts + Volume Profile + 8-EMA ribbon + HTF filter.

Design rules
* A structural TRIGGER (liquidity sweep, CHoCH or BOS) is REQUIRED. Everything
  else is confluence that adds score.
* Order-flow imbalance uses Binance TAKER-BUY volume (real aggressor flow) when
  present, and falls back to an OHLCV approximation otherwise.
* Regime filters: continuation (BOS) trades need a trending market (efficiency
  ratio), and entries are skipped during volatility spikes.
* No lookahead: pivots are only used once confirmed by `pivot_k` later bars.
* Every threshold lives in Config. All defaults are UNTESTED starting points:
  tune them with backtest.py on out-of-sample data before trusting them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config


@dataclass
class SignalResult:
    action: Optional[str]          # 'BUY' | 'SELL' | None
    score: int
    entry: float
    sl: float
    tp1: float
    tp2: float
    risk: float                    # absolute stop distance (1R)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def tp(self) -> float:         # backwards compatible alias
        return self.tp2


# ------------------------------------------------------------ indicators ----
def wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def wilder_rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rsi = 100 - 100 / (1 + au / ad.replace(0, np.nan))
    return rsi.where(ad != 0, 100.0)


def ribbon_direction(close: pd.Series, lens: Tuple[int, ...]) -> int:
    """+1 bull / -1 bear when at least all-but-one adjacent EMA pairs are ordered."""
    emas = [float(close.ewm(span=n, adjust=False).mean().iloc[-1]) for n in sorted(lens)]
    pairs = len(emas) - 1
    bull = sum(emas[i] > emas[i + 1] for i in range(pairs))
    bear = sum(emas[i] < emas[i + 1] for i in range(pairs))
    if bull >= pairs - 1:
        return 1
    if bear >= pairs - 1:
        return -1
    return 0


def htf_bias(htf: Optional[pd.DataFrame], cfg: Config) -> int:
    """+1 / -1 / 0 from the HTF closes: fast vs slow EMA and price vs slow EMA."""
    if htf is None or len(htf) < max(cfg.ema_lens) + 20:
        return 0
    c = htf["close"].tail(250)
    fast = c.ewm(span=min(cfg.ema_lens), adjust=False).mean().iloc[-1]
    slow = c.ewm(span=max(cfg.ema_lens), adjust=False).mean().iloc[-1]
    if fast > slow and c.iloc[-1] > slow:
        return 1
    if fast < slow and c.iloc[-1] < slow:
        return -1
    return 0


def volume_profile(df: pd.DataFrame, bins: int, use_volume: bool
                   ) -> Optional[Tuple[float, float, float]]:
    """(POC, VAH, VAL). Volume is spread across each candle's high-low range;
    value area (70%) expands contiguously out from the POC. Without real volume
    every candle weighs 1 (time-at-price profile)."""
    lo, hi = float(df["low"].min()), float(df["high"].max())
    if not hi > lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    lows, highs = df["low"].to_numpy(), df["high"].to_numpy()
    w = df["volume"].to_numpy() if use_volume else np.ones(len(df))
    ov = np.clip(np.minimum(highs[:, None], edges[None, 1:])
                 - np.maximum(lows[:, None], edges[None, :-1]), 0, None)
    s = ov.sum(axis=1)
    flat = s == 0                                   # doji / zero-range candles
    if flat.any():
        idx = np.clip(np.searchsorted(edges, lows[flat], side="right") - 1, 0, bins - 1)
        ov[np.where(flat)[0], idx] = 1.0
        s = ov.sum(axis=1)
    prof = (ov / s[:, None] * w[:, None]).sum(axis=0)
    total = prof.sum()
    if total <= 0:
        return None
    poc = int(np.argmax(prof))
    lo_i = hi_i = poc
    acc, target = prof[poc], 0.70 * total
    while acc < target and (lo_i > 0 or hi_i < bins - 1):
        up = prof[hi_i + 1] if hi_i < bins - 1 else -1.0
        dn = prof[lo_i - 1] if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            acc += prof[hi_i]
        else:
            lo_i -= 1
            acc += prof[lo_i]
    return (float((edges[poc] + edges[poc + 1]) / 2),
            float(edges[hi_i + 1]), float(edges[lo_i]))


def flow_imbalance(df: pd.DataFrame, n: int = 3) -> Tuple[float, bool]:
    """Buy/sell imbalance in [-1, 1] over the last n candles.
    Returns (imbalance, is_real). Real = Binance taker-buy volume (aggressor side);
    otherwise approximated from where each candle closes inside its range."""
    vol = float(df["volume"].iloc[-n:].sum())
    if vol <= 0:
        return 0.0, False
    if "taker_buy" in df.columns and float(df["taker_buy"].iloc[-n:].sum()) > 0:
        tb = df["taker_buy"].iloc[-n:]
        return float((2 * tb - df["volume"].iloc[-n:]).sum() / vol), True
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng).fillna(0.0)
    return float((clv * df["volume"]).iloc[-n:].sum() / vol), False


def efficiency_ratio(close: np.ndarray, n: int) -> float:
    """Kaufman ER: net move / total path over n bars. ~0 = chop, ~1 = clean trend."""
    if len(close) <= n:
        return 0.0
    seg = close[-(n + 1):]
    path = float(np.abs(np.diff(seg)).sum())
    return float(abs(seg[-1] - seg[0]) / path) if path > 0 else 0.0


# ------------------------------------------------------------- structure ----
def _pivots(high: np.ndarray, low: np.ndarray, k: int):
    ph, pl = [], []
    for p in range(k, len(high) - k):
        wh, wl = high[p - k:p + k + 1], low[p - k:p + k + 1]
        if high[p] == wh.max() and np.count_nonzero(wh == high[p]) == 1:
            ph.append((p, float(high[p])))
        if low[p] == wl.min() and np.count_nonzero(wl == low[p]) == 1:
            pl.append((p, float(low[p])))
    return ph, pl


def _structure_events(i: int, close: np.ndarray, ph, pl, k: int) -> List[Tuple[str, str]]:
    """Break of the last confirmed swing at bar i. CHoCH = break against the prior
    structure (reversal); BOS = break with it (continuation)."""
    hs = [v for p, v in ph if p <= i - k]
    ls = [v for p, v in pl if p <= i - k]
    if i < 1 or len(hs) < 2 or len(ls) < 2:
        return []
    bull_struct = hs[-1] > hs[-2] and ls[-1] > ls[-2]
    bear_struct = hs[-1] < hs[-2] and ls[-1] < ls[-2]
    ev: List[Tuple[str, str]] = []
    if close[i] > hs[-1] >= close[i - 1]:
        if bear_struct:
            ev.append(("BUY", "CHoCH"))
        elif bull_struct:
            ev.append(("BUY", "BOS"))
    if close[i] < ls[-1] <= close[i - 1]:
        if bull_struct:
            ev.append(("SELL", "CHoCH"))
        elif bear_struct:
            ev.append(("SELL", "BOS"))
    return ev


def _in_session(hour: int, start: int, end: int) -> bool:
    if start == end:
        return True
    return start <= hour < end if start < end else (hour >= start or hour < end)


# ----------------------------------------------------------------- engine ----
def _none(price: float, reason: str, **extra) -> SignalResult:
    return SignalResult(None, 0, price, 0.0, 0.0, 0.0, 0.0, {"veto": reason, **extra})


def evaluate_strategy(df: pd.DataFrame, cfg: Config,
                      htf_df: Optional[pd.DataFrame] = None,
                      symbol: str = "", has_volume: bool = True) -> SignalResult:
    if df is None or len(df) < cfg.min_bars:
        return _none(0.0, "insufficient data")

    d = df.tail(cfg.eval_window)
    o, h, l, c = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n, price = len(d), float(c[-1])
    atr_s = wilder_atr(d, cfg.atr_len)
    atr = float(atr_s.iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return _none(price, "ATR unavailable")
    atr_ref = float(atr_s.tail(100).median())
    if np.isfinite(atr_ref) and atr_ref > 0 and atr / atr_ref > cfg.vol_spike_mult:
        return _none(price, f"volatility spike (ATR {atr / atr_ref:.1f}x normal)")
    er = efficiency_ratio(c, cfg.er_len)

    # ---- global gates ----
    if atr / price * 100 < cfg.min_atr_pct:
        return _none(price, f"low volatility (ATR {atr / price * 100:.3f}%)")
    if not _in_session(d.index[-1].hour, cfg.session_start, cfg.session_end):
        return _none(price, "outside session")

    rsi = float(wilder_rsi(d["close"], cfg.rsi_len).iloc[-1])
    rsi = 50.0 if not np.isfinite(rsi) else rsi
    ribbon = ribbon_direction(d["close"].tail(250), cfg.ema_lens)
    bias = htf_bias(htf_df, cfg) if cfg.htf_filter else 0

    real_vol = bool(has_volume and d["volume"].tail(cfg.vp_lookback).sum() > 0)
    vp = volume_profile(d.tail(cfg.vp_lookback), cfg.vp_bins, real_vol)
    poc, vah, val = vp if vp else (price, price, price)

    eq = (h[-50:].max() + l[-50:].min()) / 2.0
    is_discount, is_premium = price < eq, price > eq

    scores = {"BUY": 0, "SELL": 0}
    notes: Dict[str, List[str]] = {"BUY": [], "SELL": []}
    trigger = {"BUY": False, "SELL": False}
    seen: set = set()

    def add(side: str, pts: int, note: str, is_trigger: bool = False):
        scores[side] += pts
        notes[side].append(note)
        if is_trigger:
            trigger[side] = True

    # ---- triggers: liquidity sweeps ----
    if cfg.use_sweep_filter:
        L = cfg.sweep_lookback_bars
        for off in range(cfg.trigger_lookback):
            i = n - 1 - off
            if i < L:
                break
            lo_prev, hi_prev = l[i - L:i].min(), h[i - L:i].max()
            when = "now" if off == 0 else f"{off} bar{'s' if off > 1 else ''} ago"
            if l[i] < lo_prev and c[i] > lo_prev and "BUY_sweep" not in seen:
                seen.add("BUY_sweep")
                add("BUY", 25, f"Bullish liquidity sweep below {lo_prev:,.2f} ({when})", True)
            if h[i] > hi_prev and c[i] < hi_prev and "SELL_sweep" not in seen:
                seen.add("SELL_sweep")
                add("SELL", 25, f"Bearish liquidity sweep above {hi_prev:,.2f} ({when})", True)

    # ---- triggers: CHoCH / BOS on confirmed pivots ----
    m = min(n, 150)
    ph, pl = _pivots(h[-m:], l[-m:], cfg.pivot_k)
    for off in range(cfg.trigger_lookback):
        for side, kind in _structure_events(m - 1 - off, c[-m:], ph, pl, cfg.pivot_k):
            key = f"{side}_{kind}"
            if key in seen:
                continue
            if kind == "BOS" and er < cfg.er_min_bos:
                continue                      # continuation needs a trending market
            seen.add(key)
            when = "now" if off == 0 else f"{off} bar{'s' if off > 1 else ''} ago"
            pts = 20 if kind == "CHoCH" else 10
            label = "Change of Character" if kind == "CHoCH" else "Break of Structure"
            add(side, pts, f"{'Bullish' if side == 'BUY' else 'Bearish'} {label} ({when})", True)

    # ---- confluence: fair value gaps (unfilled, big enough) ----
    if cfg.use_fvg_filter:
        for off in range(cfg.fvg_lookback):
            j = n - 1 - off
            if j < 2:
                break
            gap_up, gap_dn = l[j] - h[j - 2], l[j - 2] - h[j]
            if gap_up >= cfg.fvg_min_atr * atr and c[j - 1] > o[j - 1] \
                    and price >= h[j - 2] and "BUY_fvg" not in seen:
                seen.add("BUY_fvg")
                add("BUY", 15, f"Bullish FVG {h[j - 2]:,.2f}-{l[j]:,.2f} ({gap_up / atr:.1f} ATR)")
            if gap_dn >= cfg.fvg_min_atr * atr and c[j - 1] < o[j - 1] \
                    and price <= l[j - 2] and "SELL_fvg" not in seen:
                seen.add("SELL_fvg")
                add("SELL", 15, f"Bearish FVG {h[j]:,.2f}-{l[j - 2]:,.2f} ({gap_dn / atr:.1f} ATR)")

    # ---- confluence: value area / premium-discount ----
    near = 0.25 * atr
    if price <= val + near and is_discount:
        add("BUY", 20, f"At/below Value Area Low {val:,.2f} in discount zone")
    if price >= vah - near and is_premium:
        add("SELL", 20, f"At/above Value Area High {vah:,.2f} in premium zone")

    # ---- confluence: order-flow imbalance (only with real volume) ----
    imb, flow_real = 0.0, False
    if real_vol:
        imb, flow_real = flow_imbalance(d, 3)
        label = "taker flow" if flow_real else "volume imbalance"
        if imb > cfg.delta_thr:
            add("BUY", 10, f"Aggressive buying, {label} {imb:+.2f}")
        elif imb < -cfg.delta_thr:
            add("SELL", 10, f"Aggressive selling, {label} {imb:+.2f}")

    # ---- confluence: ribbon + HTF alignment ----
    if ribbon == 1:
        add("BUY", 15, "8-EMA ribbon bullish aligned")
    elif ribbon == -1:
        add("SELL", 15, "8-EMA ribbon bearish aligned")
    if bias == 1:
        add("BUY", 10, f"{cfg.htf_min}m trend bullish")
    elif bias == -1:
        add("SELL", 10, f"{cfg.htf_min}m trend bearish")

    # ---- vetoes ----
    veto: Dict[str, Optional[str]] = {"BUY": None, "SELL": None}
    if cfg.htf_filter and bias == -1:
        veto["BUY"] = f"{cfg.htf_min}m trend bearish"
    if cfg.htf_filter and bias == 1:
        veto["SELL"] = f"{cfg.htf_min}m trend bullish"
    if rsi > cfg.rsi_ob:
        veto["BUY"] = veto["BUY"] or f"RSI overbought ({rsi:.0f})"
    if rsi < cfg.rsi_os:
        veto["SELL"] = veto["SELL"] or f"RSI oversold ({rsi:.0f})"

    scores = {k: min(v, 100) for k, v in scores.items()}
    base = {"poc": round(poc, 2), "vah": round(vah, 2), "val": round(val, 2),
            "atr": round(atr, 4), "rsi": round(rsi, 1), "htf_bias": bias,
            "ribbon": ribbon, "scores": dict(scores), "real_volume": real_vol,
            "flow": round(imb, 3), "flow_real": flow_real, "er": round(er, 2)}

    eligible = [s for s in ("BUY", "SELL")
                if trigger[s] and veto[s] is None and scores[s] >= cfg.min_score
                and scores[s] - scores["SELL" if s == "BUY" else "BUY"] >= cfg.min_edge]
    if not eligible:
        why = [f"{s}: " + (veto[s] or ("no trigger" if not trigger[s] else f"score {scores[s]}"))
               for s in ("BUY", "SELL")]
        return _none(price, "; ".join(why), **base)
    side = max(eligible, key=lambda s: scores[s])

    # ---- risk model ----
    dec = 2 if price >= 1 else 5
    sl_dist = cfg.sl_mult * atr
    if cfg.use_swing_sl:
        k = cfg.swing_lookback
        swing = (price - l[-k:].min() if side == "BUY" else h[-k:].max() - price) + 0.1 * atr
        sl_dist = min(max(swing, cfg.sl_mult * atr), cfg.max_sl_mult * atr)
    cost = price * cfg.cost_bps(symbol) / 1e4
    if sl_dist < cfg.min_sl_to_cost * cost:
        return _none(price, f"stop too tight vs costs ({sl_dist:.2f} < {cfg.min_sl_to_cost:g}x{cost:.2f})", **base)

    d_ = 1 if side == "BUY" else -1
    sl = price - d_ * sl_dist
    tp1 = price + d_ * cfg.tp1_r * sl_dist
    tp2 = price + d_ * cfg.tp2_r * sl_dist

    other = scores["SELL" if side == "BUY" else "BUY"]
    notes_out = list(notes[side])
    if abs(price - poc) < 0.5 * atr:
        notes_out.append("Price testing Point of Control")
    return SignalResult(
        action=side, score=scores[side], entry=round(price, dec), sl=round(sl, dec),
        tp1=round(tp1, dec), tp2=round(tp2, dec), risk=round(sl_dist, dec),
        details={"notes": notes_out, "opposite_score": other, **base},
    )
