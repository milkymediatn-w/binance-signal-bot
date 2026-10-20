# Binance Signal Bot (BTC/USDT + Gold)

Telegram bot that scans Binance candles at every close, broadcasts signals built from
liquidity sweeps, CHoCH/BOS structure, fair value gaps, volume profile, real taker-flow
imbalance, an 8-EMA ribbon and a higher-timeframe filter, then tracks every signal to its
outcome (`/stats`). It sends alerts only. It never trades.

## Instruments
- BTCUSDT: Binance spot.
- PAXGUSDT: gold-backed token on Binance spot (Binance has no spot XAUUSD). Tracks gold closely
  but can differ from CFD/spot-gold quotes. Set `SYMBOLS=BTCUSDT,XAUUSDT` to use the gold perpetual instead (thinner liquidity).

## Deploy on Render
1. Revoke any token that has ever been pasted into a chat, then create a fresh one in @BotFather.
2. Push this folder to a PRIVATE GitHub repo (`.env` and `*.db` are git-ignored).
3. Render > New > Blueprint (uses `render.yaml`): Background Worker, **Frankfurt** region
   (Binance returns HTTP 451 to US IPs), paid plan for the persistent disk at `/data`.
4. Set env vars in the dashboard: `BOT_TOKEN`, then message your bot `/id`, put that number in `ADMIN_CHAT_ID`.
5. Optional: `ACCESS_CODE` (users must send `/start <code>`), `PRESET`, `COST_BPS_CRYPTO`, `COST_BPS_GOLD`.
6. Check the logs for "Scanning every 15m", and send `/status` from the admin chat.

## Before trusting any signal
```
pip install -r requirements.txt
python test_all.py                                   # offline tests
python backtest.py --symbol BTCUSDT,PAXGUSDT --bars 8000
```
Read the OUT-OF-SAMPLE line. Judge only with 100+ trades. If "gross" is positive but "net" is not,
fees are eating the edge: raise the timeframe (`PRESET=SWING`) or lower costs if you really pay less.
Then run it in paper mode (alerts only) and compare `/stats` with the backtest for a few weeks.

## Commands
/start /stop /help /stats /id, and admin-only /status.

## Important
- Costs matter: `COST_BPS_*` is the assumed ROUND-TRIP cost. At 10 bps, a signal needs ATR >= 0.27% of price
  on the chosen timeframe, which blocks most 1m/5m scalps by design.
- Signals are informational, not financial advice.
