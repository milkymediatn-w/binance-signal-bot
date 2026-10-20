"""
Trade lifecycle: builds signal rows and resolves them candle by candle.

The same code is used by the live bot and by backtest.py, so live statistics
and backtest statistics are directly comparable.

Conventions (all in R = initial stop distance):
* Pessimistic intrabar rule: if stop and target are both inside one candle,
  the stop is assumed to hit first.
* TP1 closes `tp1_frac` of the position; stop then moves to entry if enabled.
* Unresolved trades are marked to market at expiry.
* Results are NET of the assumed round-trip cost (cost_r).
"""
import time
from typing import Any, Dict

import pandas as pd

OPEN_STATES = ("OPEN", "TP1")
TERMINAL = ("WIN", "LOSS", "PARTIAL", "EXPIRED")


def build_signal_row(res, symbol: str, bar_ts: int, cfg, source: str) -> Dict[str, Any]:
    cost_r = res.entry * cfg.cost_bps(symbol) / 1e4 / res.risk if res.risk else 0.0
    return {
        "symbol": symbol, "action": res.action, "bar_ts": bar_ts,
        "created_ts": int(time.time()), "entry": res.entry, "sl": res.sl,
        "tp1": res.tp1, "tp2": res.tp2, "risk": res.risk, "score": res.score,
        "tp1_r": cfg.tp1_r, "tp2_r": cfg.tp2_r, "tp1_frac": cfg.tp1_close_frac,
        "be_after_tp1": int(cfg.be_after_tp1), "expiry_bars": cfg.expiry_bars,
        "cost_r": cost_r, "status": "OPEN", "stop_r": -1.0, "realized_r": 0.0,
        "bars_seen": 0, "last_ts": bar_ts, "source": source,
    }


def resolve_signal(sig: Dict[str, Any], candles: pd.DataFrame) -> Dict[str, Any]:
    """Advance a signal through `candles` (closed, strictly after sig['last_ts']).
    Returns the updated state fields."""
    s = {k: sig[k] for k in ("status", "stop_r", "realized_r", "bars_seen", "last_ts")}
    s["closed_ts"], s["result_r"] = None, None
    buy = sig["action"] == "BUY"
    entry, risk = sig["entry"], sig["risk"]
    frac, tp1, tp2 = sig["tp1_frac"], sig["tp1_r"], sig["tp2_r"]
    be = bool(sig["be_after_tp1"])

    def finish(status: str, gross: float, t: int):
        s["status"] = status
        s["result_r"] = gross - sig["cost_r"]
        s["closed_ts"] = t

    for ts, hi_p, lo_p, cl_p in zip(candles.index, candles["high"], candles["low"], candles["close"]):
        t = int(ts.timestamp())
        if t <= s["last_ts"]:
            continue
        if buy:
            lo, hi, cl = (lo_p - entry) / risk, (hi_p - entry) / risk, (cl_p - entry) / risk
        else:
            lo, hi, cl = (entry - hi_p) / risk, (entry - lo_p) / risk, (entry - cl_p) / risk
        s["bars_seen"] += 1
        s["last_ts"] = t

        if s["status"] == "OPEN":
            if lo <= s["stop_r"]:
                finish("LOSS", s["realized_r"] + s["stop_r"], t)
            elif hi >= tp1:
                s["realized_r"] += frac * tp1
                s["status"] = "TP1"
                s["stop_r"] = 0.0 if be else s["stop_r"]
                if lo <= s["stop_r"]:                      # pessimistic: stopped same bar
                    finish("PARTIAL", s["realized_r"] + (1 - frac) * s["stop_r"], t)
                elif hi >= tp2:
                    finish("WIN", s["realized_r"] + (1 - frac) * tp2, t)
        elif s["status"] == "TP1":
            if lo <= s["stop_r"]:
                finish("PARTIAL", s["realized_r"] + (1 - frac) * s["stop_r"], t)
            elif hi >= tp2:
                finish("WIN", s["realized_r"] + (1 - frac) * tp2, t)

        if s["status"] in TERMINAL:
            break
        if s["bars_seen"] >= sig["expiry_bars"]:
            remaining = (1 - frac) if s["status"] == "TP1" else 1.0
            finish("EXPIRED", s["realized_r"] + remaining * cl, t)
            break
    return s
