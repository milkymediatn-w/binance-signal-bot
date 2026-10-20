"""
Backtester. Uses the SAME evaluate_strategy and resolve_signal as the live bot.

Examples
    python backtest.py --symbol BTCUSDT --bars 6000
    python backtest.py --symbol BTCUSDT,PAXGUSDT --bars 8000
    PRESET=SCALP COST_BPS_CRYPTO=6 python backtest.py --symbol BTCUSDT --bars 8000
    python backtest.py --symbol PAXGUSDT --csv gold_15m.csv   # datetime,open,high,low,close,volume

Reports in-sample vs out-of-sample (last --oos fraction of trades) so you can
tell tuning from overfitting. Results are net of assumed costs and use the
pessimistic intrabar rule (stop first when stop and target share a candle).
"""
import argparse
import math

import numpy as np
import pandas as pd

from config import Config, load_config
from data import _clean, get_history, resample_htf
from outcomes import TERMINAL, build_signal_row, resolve_signal
from strategy import evaluate_strategy


def run_backtest(df: pd.DataFrame, cfg: Config, symbol: str, has_volume: bool,
                 source: str = "backtest") -> pd.DataFrame:
    htf_all = resample_htf(df, cfg.tf_min, cfg.htf_min) if cfg.htf_filter else None
    htf_end = (np.array([(t + pd.Timedelta(minutes=cfg.htf_min)).timestamp() for t in htf_all.index])
               if htf_all is not None and len(htf_all) else None)
    tf = pd.Timedelta(minutes=cfg.tf_min)
    warm = max(cfg.min_bars, 60)
    trades, open_sig, sig_i = [], None, -10**9

    for i in range(len(df)):
        if open_sig is not None:
            upd = resolve_signal(open_sig, df.iloc[i:i + 1])
            open_sig.update(upd)
            if open_sig["status"] in TERMINAL:
                trades.append({"symbol": symbol, "action": open_sig["action"],
                               "bar_ts": open_sig["bar_ts"], "score": open_sig["score"],
                               "status": open_sig["status"], "bars": open_sig["bars_seen"],
                               "result_r": open_sig["result_r"],
                               "cost_r": open_sig["cost_r"]})
                open_sig = None
            continue                                  # one trade at a time, like live
        if i < warm or i - sig_i < cfg.cooldown_bars:
            continue
        window = df.iloc[max(0, i + 1 - cfg.eval_window):i + 1]
        htf = None
        if htf_end is not None:
            k = htf_end.searchsorted((df.index[i] + tf).timestamp(), side="right")
            htf = htf_all.iloc[:k]
        res = evaluate_strategy(window, cfg, htf, symbol, has_volume)
        if res.action:
            open_sig = build_signal_row(res, symbol, int(df.index[i].timestamp()), cfg, source)
            sig_i = i
    return pd.DataFrame(trades)


def summarize(t: pd.DataFrame, label: str) -> str:
    if t.empty:
        return f"{label}: no trades"
    r = t["result_r"].to_numpy()
    n = len(r)
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else float("inf")
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    streak = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        streak = max(streak, cur)
    se = r.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    gross = float((t["result_r"] + t["cost_r"]).mean())
    return (f"{label}: {n} trades | win rate {len(wins) / n:.1%} | avg NET {r.mean():+.3f}R "
            f"(gross {gross:+.3f}R, 95% CI {r.mean() - 1.96 * se:+.2f}..{r.mean() + 1.96 * se:+.2f}) | "
            f"total {r.sum():+.1f}R | PF {pf:.2f} | max DD {dd:.1f}R | "
            f"worst losing streak {streak}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT", help="one or more, comma separated")
    ap.add_argument("--csv", help="CSV with datetime,open,high,low,close,volume (single symbol)")
    ap.add_argument("--bars", type=int, default=6000, help="history bars to download")
    ap.add_argument("--oos", type=float, default=0.3, help="out-of-sample share of trades")
    ap.add_argument("--out", default="backtest_trades.csv")
    a = ap.parse_args()

    cfg = load_config()
    print(f"Config: {cfg.summary()}")
    all_trades = []
    for symbol in [x.strip().upper() for x in a.symbol.split(",") if x.strip()]:
        if a.csv:
            raw = pd.read_csv(a.csv, index_col=0, parse_dates=True)
            raw.columns = [c.lower() for c in raw.columns]
            if "volume" not in raw:
                raw["volume"] = 0.0
            df, source = _clean(raw), "csv"
        else:
            df, source, _ = get_history(cfg, symbol, a.bars)
        has_vol = bool(df["volume"].sum() > 0)
        print(f"\n{symbol}: {len(df)} candles {df.index[0]} -> {df.index[-1]} | {source} "
              f"| real volume: {has_vol} | assumed cost {cfg.cost_bps(symbol):g} bps "
              f"(needs ATR >= {cfg.min_atr_pct_for_costs(symbol):.2f}% to trade)")
        t = run_backtest(df, cfg, symbol, has_vol, source)
        all_trades.append(t)
        print(summarize(t, f"{symbol} ALL"))
        if len(t) >= 20:
            cut = int(len(t) * (1 - a.oos))
            print(summarize(t.iloc[:cut], f"{symbol} IN-SAMPLE"))
            print(summarize(t.iloc[cut:], f"{symbol} OUT-OF-SAMPLE"))
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    trades.to_csv(a.out, index=False)
    print(f"\nTrades written to {a.out}")
    print("Trust only out-of-sample results, and only with 100+ trades. "
          "If gross is positive but net is not, costs are the problem, not the signals.")


if __name__ == "__main__":
    main()
