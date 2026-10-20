"""Offline test suite: no network needed. Run: python test_all.py"""
import os, sys, time, asyncio, types, tempfile
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_bot.db")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, httpx
from datetime import datetime, timezone
import config as C
from config import load_config
import data as D
from outcomes import resolve_signal
from strategy import evaluate_strategy, volume_profile, flow_imbalance, efficiency_ratio

cfg = load_config()
print("CFG:", cfg.summary())

# ---------------- config ----------------
assert cfg.preset == "INTRADAY" and cfg.tf_min == 15 and cfg.symbols == ("BTCUSDT", "PAXGUSDT")
assert abs(cfg.min_atr_pct_for_costs("BTCUSDT") - 4*10/1e4/1.5*100) < 1e-9
os.environ["SYMBOLS"] = "btcusdt, xauusdt"; os.environ["COST_BPS_GOLD"] = "2"
c2 = load_config(); assert c2.symbols == ("BTCUSDT", "XAUUSDT") and c2.cost_bps("XAUUSDT") == 2.0
assert c2.is_futures("XAUUSDT") and not c2.is_futures("PAXGUSDT") and c2.is_gold("PAXGUSDT")
del os.environ["SYMBOLS"], os.environ["COST_BPS_GOLD"]
print("config ok")

# ---------------- resolver ----------------
def mk(action="BUY", **kw):
    b = dict(action=action, entry=100.0, risk=1.0, tp1_r=1.0, tp2_r=2.0, tp1_frac=0.5, be_after_tp1=1,
             expiry_bars=10, cost_r=0.0, status="OPEN", stop_r=-1.0, realized_r=0.0, bars_seen=0, last_ts=0)
    b.update(kw); return b
def cds(rows):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open","high","low","close"], index=idx).assign(volume=1.0)
r = resolve_signal(mk(), cds([(100,100.5,98.9,99)])); assert r["status"]=="LOSS" and abs(r["result_r"]+1)<1e-9
r = resolve_signal(mk(), cds([(100,101.2,100.1,101),(101,102.5,100.5,102)])); assert r["status"]=="WIN" and abs(r["result_r"]-1.5)<1e-9
r = resolve_signal(mk(), cds([(100,101.2,100.1,101),(101,101.1,99.8,100)])); assert r["status"]=="PARTIAL" and abs(r["result_r"]-.5)<1e-9
r = resolve_signal(mk(), cds([(100,101.2,98.8,100)])); assert r["status"]=="LOSS"
r = resolve_signal(mk("SELL"), cds([(100,99.9,98.8,99),(99,99.2,97.5,98)])); assert r["status"]=="WIN"
r = resolve_signal(mk(expiry_bars=2), cds([(100,100.5,99.6,100.2),(100.2,100.6,99.7,100.4)])); assert r["status"]=="EXPIRED"
print("resolver ok")

# ---------------- Binance parsing / fallback / pagination (mocked HTTP) ----------------
class Resp:
    def __init__(self, code=200, payload=None, text=""): self.status_code, self._p, self.text = code, payload, text
    def json(self):
        if self._p is None: raise ValueError("no json")
        return self._p
def kline_rows(start_ms, n, step_ms, base=60000.0):
    rows = []
    for i in range(n):
        t = start_ms + i*step_ms; o = base + i; c = o + 0.5
        rows.append([t, str(o), str(o+2), str(o-2), str(c), "10.0", t+step_ms-1, "0", 5, "7.0", "0", "0"])
    return rows
calls = []
def fake_get_factory(behaviour):
    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, dict(params or {}))); return behaviour(url, params or {})
    return fake_get
step = 15*60*1000
now_ms = int(time.time()*1000); now_ms -= now_ms % step
# (a) first host 451 -> falls back to second host
def b1(url, p):
    if "data-api.binance.vision" in url: return Resp(451, None, "blocked")
    return Resp(200, kline_rows(now_ms - 600*step, 600, step))
httpx.get = fake_get_factory(b1)
df = D.fetch_binance(cfg, "BTCUSDT", 600)
assert len(df) == 600 and df["taker_buy"].iloc[0] == 7.0 and df.index.tz is not None
assert calls[0][0].startswith("https://data-api.binance.vision") and calls[1][0].startswith("https://api.binance.com")
assert calls[0][1]["interval"] == "15m" and calls[0][1]["symbol"] == "BTCUSDT"
print("host fallback ok")
# (b) all hosts 451 -> clear region message
httpx.get = fake_get_factory(lambda u, p: Resp(451, None, "blocked"))
try: D.fetch_binance(cfg, "BTCUSDT", 600); raise SystemExit("should fail")
except D.DataError as e: assert "blocks this server" in str(e), e
# (c) rate limit / bad symbol
httpx.get = fake_get_factory(lambda u, p: Resp(429, None))
try: D.fetch_binance(cfg, "BTCUSDT", 100); raise SystemExit("should fail")
except D.DataError as e: assert "rate limited" in str(e)
httpx.get = fake_get_factory(lambda u, p: Resp(400, None, '{"code":-1121,"msg":"Invalid symbol."}'))
try: D.fetch_binance(cfg, "FOOUSDT", 100); raise SystemExit("should fail")
except D.DataError as e: assert "rejected" in str(e)
# (d) pagination: 2500 bars -> 3 pages walking backwards via endTime
total = 2500; first_open = now_ms - total*step
def b2(url, p):
    end = p.get("endTime", now_ms + step); lim = p["limit"]
    allr = kline_rows(first_open, total, step)
    sel = [r for r in allr if r[0] <= end][-lim:]
    return Resp(200, sel)
calls.clear(); httpx.get = fake_get_factory(b2)
df = D.fetch_binance(cfg, "BTCUSDT", 2500)
assert len(df) == 2500 and df.index.is_monotonic_increasing and not df.index.duplicated().any() and len(calls) == 3, (len(df), len(calls))
# (e) futures symbol uses fapi
calls.clear(); httpx.get = fake_get_factory(lambda u, p: Resp(200, kline_rows(now_ms-300*step, 300, step)))
D.fetch_binance(cfg, "XAUUSDT", 300); assert calls[0][0].startswith("https://fapi.binance.com/fapi/v1/klines")
print("binance fetch ok (fallback, 451, 429, 400, pagination, futures)")

# ---------------- get_feed: closed-candle + staleness ----------------
fresh_open = now_ms - 2*step    # last returned candle opens 2 bars ago, plus the forming one
def b3(url, p): return Resp(200, kline_rows(now_ms - 600*step, 601, step))   # last = forming candle (opens now_ms)
httpx.get = fake_get_factory(b3)
feed = D.get_feed(cfg, "BTCUSDT")
assert feed.df.index[-1] == pd.Timestamp(now_ms - step, unit="ms", tz="UTC"), feed.df.index[-1]   # forming candle dropped
assert feed.htf is not None and feed.has_volume and feed.source == "binance-spot"
httpx.get = fake_get_factory(lambda u, p: Resp(200, kline_rows(now_ms - 700*step, 600, step)))   # ends ~100 bars ago
try: D.get_feed(cfg, "BTCUSDT"); raise SystemExit("should be stale")
except D.StaleDataError: pass
print("feed closed-candle + stale guard ok")

# ---------------- market hours ----------------
assert not D.market_open(cfg, "PAXGUSDT", datetime(2026,10,17,12,0,tzinfo=timezone.utc))   # Sat
assert D.market_open(cfg, "PAXGUSDT", datetime(2026,10,20,12,0,tzinfo=timezone.utc))
assert D.market_open(cfg, "BTCUSDT", datetime(2026,10,17,12,0,tzinfo=timezone.utc))
print("market hours ok")

# ---------------- indicators ----------------
rng = np.random.default_rng(7)
def synth(n, base=60000, sig=0.0035, start="2026-08-01", freq="15min"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    regime = np.repeat(rng.choice([0.6,1.0,1.8], size=n//50+1), 50)[:n]
    drift = np.repeat(rng.normal(0,0.0008,size=n//200+1),200)[:n]
    rets = rng.normal(drift, sig*regime)
    close = base*np.exp(np.cumsum(rets)); open_ = np.r_[base, close[:-1]]
    hi = np.maximum(open_, close)*(1+np.abs(rng.normal(0,sig/2,n)))
    lo = np.minimum(open_, close)*(1-np.abs(rng.normal(0,sig/2,n)))
    v = rng.gamma(2,50,n); tb = v*np.clip(0.5+(close-open_)/(hi-lo+1e-9)*0.25+rng.normal(0,0.05,n),0,1)
    return pd.DataFrame({"open":open_,"high":hi,"low":lo,"close":close,"volume":v,"taker_buy":tb}, index=idx)
df = synth(4000)
poc, vah, val = volume_profile(df.tail(100), 30, True); assert val <= poc <= vah
imb, real = flow_imbalance(df, 3); assert real and -1 <= imb <= 1
imb2, real2 = flow_imbalance(df.drop(columns="taker_buy"), 3); assert not real2
allbuy = df.tail(3).copy(); allbuy["taker_buy"] = allbuy["volume"]; assert abs(flow_imbalance(allbuy,3)[0]-1) < 1e-9
assert efficiency_ratio(np.arange(100.0), 30) == 1.0 and efficiency_ratio(np.tile([1.0,2.0],50), 30) < 0.1
print("indicators ok")

# ---------------- strategy: no lookahead, deterministic ----------------
a = evaluate_strategy(df.iloc[:1000], cfg, None, "BTCUSDT", True)
b = evaluate_strategy(pd.concat([df.iloc[:1000], df.iloc[1000:1100]]).iloc[:1000], cfg, None, "BTCUSDT", True)
assert (a.action, a.score, a.entry) == (b.action, b.score, b.entry)
# volatility spike veto
spike = df.iloc[:600].copy(); spike.iloc[-3:, spike.columns.get_loc("high")] *= 1.08
r = evaluate_strategy(spike, cfg, None, "BTCUSDT", True); assert r.action is None
# cost filter: with huge costs nothing passes; with tiny costs signals appear
from dataclasses import replace
hi_cost = replace(cfg, cost_bps_crypto=500.0)
from backtest import run_backtest, summarize
t0 = time.time(); tr = run_backtest(df, cfg, "BTCUSDT", True); dt = time.time()-t0
print(f"backtest {len(df)} bars {dt:.1f}s -> {len(tr)} trades"); print(summarize(tr, "synthetic default-cost"))
tr_hi = run_backtest(df, hi_cost, "BTCUSDT", True); assert len(tr_hi) == 0, len(tr_hi)
lo_cost = replace(cfg, cost_bps_crypto=1.0); tr_lo = run_backtest(df, lo_cost, "BTCUSDT", True)
print(f"trades at 500bps: {len(tr_hi)} | 10bps: {len(tr)} | 1bps: {len(tr_lo)}"); assert len(tr_lo) >= len(tr)
if len(tr_lo): print(summarize(tr_lo, "synthetic 1bps"))
assert set(tr_lo.status) <= {"WIN","LOSS","PARTIAL","EXPIRED"}
print("strategy ok")

# ---------------- mocked end-to-end scan ----------------
import tracker, bot
tracker.init_db(); tracker.add_user(111); tracker.add_user(222)
sent = []
class FakeBot:
    async def send_message(self, chat_id, text, parse_mode=None):
        if chat_id == 222:
            from telegram.error import Forbidden; raise Forbidden("blocked")
        sent.append((chat_id, text))
ecfg = lo_cost
ctx = types.SimpleNamespace(bot=FakeBot(), application=types.SimpleNamespace(bot_data={"cfg": ecfg}))
from data import Feed, resample_htf
found = None
for i in range(300, 3900, 5):
    w = df.iloc[:i]
    r = evaluate_strategy(w, ecfg, resample_htf(w, 15, 60), "BTCUSDT", True)
    if r.action: found = i; break
assert found, "no signal found in synthetic history"
w = df.iloc[:found]
feed = Feed(df=w, htf=resample_htf(w,15,60), source="binance-spot", has_volume=True, age_s=5.0)
bot.get_feed = lambda c, s: feed; bot.market_open = lambda c, s: True
async def run():
    await bot.scan_symbol(ctx, ecfg, "BTCUSDT")
    assert len(tracker.get_open_signals()) == 1 and tracker.get_users() == [111]
    print(sent[0][1])
    await bot.scan_symbol(ctx, ecfg, "BTCUSDT"); assert len(tracker.get_open_signals()) == 1   # no duplicate
    f2 = Feed(df=df.iloc[:found+60], htf=None, source="binance-spot", has_volume=True, age_s=5.0)
    bot.get_feed = lambda c, s: f2; n0 = len(sent)
    await bot.scan_symbol(ctx, ecfg, "BTCUSDT")
    print("after forward scan:", [m[1][:90] for m in sent[n0:]])
    # data failure path: region block alerts admin immediately
    bot.ADMIN_ID = 999
    def boom(c, s): raise D.DataError("Binance blocks this server's region (HTTP 451/403). Redeploy in Frankfurt.")
    bot.get_feed = boom; n0 = len(sent)
    await bot.scan_symbol(ctx, ecfg, "PAXGUSDT" if D.market_open(ecfg,"PAXGUSDT") else "BTCUSDT")
asyncio.run(run())
print("e2e ok")
print("ALL TESTS PASSED")
