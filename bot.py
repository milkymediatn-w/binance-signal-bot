"""
Telegram signal bot: scans BTC/USDT and gold (PAXG/USDT) on Binance data at every
candle close, broadcasts signals, tracks each signal to its outcome, and reports
live statistics.

Run:  python bot.py        (needs BOT_TOKEN; see .env.example)
"""
import asyncio
import hmac
import html
import logging
import os
import threading
import time
from typing import Dict

import pandas as pd
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, RetryAfter, TelegramError
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

import tracker
from config import Config, load_config
from data import DataError, get_feed, market_open
from outcomes import OPEN_STATES, TERMINAL, build_signal_row, resolve_signal
from strategy import evaluate_strategy

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("bot")

ADMIN_ID = int(os.getenv("ADMIN_CHAT_ID") or 0)
ACCESS_CODE = os.getenv("ACCESS_CODE", "")
DISCLAIMER = "<i>Informational only, not financial advice. Expect slippage versus the entry shown.</i>"

fail_counts: Dict[str, int] = {}
last_scan_ok: Dict[str, float] = {}


# ------------------------------------------------------------- messaging ----
async def send_safe(bot, chat_id: int, text: str) -> bool:
    for attempt in range(2):
        try:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
            return True
        except RetryAfter as e:
            await asyncio.sleep(float(e.retry_after) + 1)
        except (Forbidden, BadRequest) as e:
            msg = str(e).lower()
            if isinstance(e, Forbidden) or "chat not found" in msg:
                tracker.remove_user(chat_id)         # user blocked the bot / deleted chat
                logger.info("Removed unreachable subscriber %s (%s)", chat_id, e)
            else:
                logger.error("Bad request to %s: %s", chat_id, e)
            return False
        except TelegramError as e:
            logger.error("Telegram error to %s: %s", chat_id, e)
            await asyncio.sleep(1)
    return False


async def broadcast(bot, text: str) -> int:
    sent = 0
    for chat_id in tracker.get_users():
        sent += await send_safe(bot, chat_id, text)
        await asyncio.sleep(0.05)                    # stay under Telegram's rate limits
    return sent


async def notify_admin(bot, text: str) -> None:
    if ADMIN_ID:
        await send_safe(bot, ADMIN_ID, text)


def format_signal(sig_id: int, symbol: str, res, cfg: Config, source: str) -> str:
    e = html.escape
    icon = "🟢" if res.action == "BUY" else "🔴"
    d = res.details
    be = ", stop to entry" if cfg.be_after_tp1 else ""
    lines = [
        f"{icon} <b>{e(cfg.display_name(symbol))} {res.action}</b> · score {res.score}/100 · #{sig_id}",
        f"Entry (bar close): <code>{res.entry:,.2f}</code>",
        f"Stop: <code>{res.sl:,.2f}</code> (risk {res.risk:,.2f})",
        f"TP1: <code>{res.tp1:,.2f}</code> ({cfg.tp1_r:g}R) close {cfg.tp1_close_frac:.0%}{be}",
        f"TP2: <code>{res.tp2:,.2f}</code> ({cfg.tp2_r:g}R)",
        "", "<b>Confluences</b>",
        *[f"• {e(n)}" for n in d.get("notes", [])],
        "",
        f"POC <code>{d.get('poc', 0):,.2f}</code> · VA <code>{d.get('val', 0):,.2f}</code>"
        f" to <code>{d.get('vah', 0):,.2f}</code>",
        f"{cfg.tf_min}m · HTF {cfg.htf_min}m · {e(source)} · assumed cost {cfg.cost_bps(symbol):g} bps",
        DISCLAIMER,
    ]
    return "\n".join(lines)


def format_outcome(sig: dict, upd: dict) -> str:
    head = f"#{sig['id']} {html.escape(Config.display_name(sig['symbol']))} {sig['action']}"
    if upd["status"] == "TP1" and sig["status"] == "OPEN":
        return f"🎯 {head}: <b>TP1 hit</b>. Partial closed, stop moved per plan."
    r = upd["result_r"]
    icon = {"WIN": "✅", "PARTIAL": "🟡", "LOSS": "❌", "EXPIRED": "⏱"}[upd["status"]]
    return f"{icon} {head}: <b>{upd['status']}</b> · {r:+.2f}R (net of assumed costs)"


# -------------------------------------------------------------- scanning ----
async def record_failure(context, cfg: Config, symbol: str, msg: str) -> None:
    n = fail_counts.get(symbol, 0) + 1
    fail_counts[symbol] = n
    logger.warning("%s data failure #%d: %s", symbol, n, msg)
    if n == cfg.alert_after_failures or (n == 1 and "blocks this server" in msg):
        await notify_admin(context.bot, f"⚠️ {html.escape(symbol)} data feed failing "
                                        f"({n} scans in a row): {html.escape(msg)}")


async def record_success(context, cfg: Config, symbol: str) -> None:
    if fail_counts.get(symbol, 0) >= cfg.alert_after_failures:
        await notify_admin(context.bot, f"✅ {html.escape(symbol)} data feed recovered.")
    fail_counts[symbol] = 0
    last_scan_ok[symbol] = time.time()


async def scan_symbol(context, cfg: Config, symbol: str) -> None:
    if not market_open(cfg, symbol):
        return
    try:
        feed = await asyncio.to_thread(get_feed, cfg, symbol)
    except DataError as e:
        await record_failure(context, cfg, symbol, str(e))
        return
    await record_success(context, cfg, symbol)

    # 1) advance open signals
    for sig in tracker.get_open_signals(symbol):
        after = feed.df[feed.df.index > pd.Timestamp(sig["last_ts"], unit="s", tz="UTC")]
        if after.empty:
            continue
        upd = resolve_signal(sig, after)
        tracker.update_signal(sig["id"], upd)
        if upd["status"] in TERMINAL or (upd["status"] == "TP1" and sig["status"] == "OPEN"):
            await broadcast(context.bot, format_outcome(sig, upd))

    # 2) look for a new signal (one open trade per symbol, with cooldown)
    if tracker.get_open_signals(symbol):
        return
    bar_ts = int(feed.df.index[-1].timestamp())
    if bar_ts - tracker.last_signal_bar_ts(symbol) < cfg.cooldown_bars * cfg.tf_min * 60:
        return
    if feed.age_s > cfg.max_signal_age_s:
        logger.info("%s candle is %.0fs old, skipping signal check", symbol, feed.age_s)
        return

    res = await asyncio.to_thread(evaluate_strategy, feed.df, cfg, feed.htf, symbol, feed.has_volume)
    if not res.action:
        logger.debug("%s no signal: %s", symbol, res.details.get("veto"))
        return
    sig_id = tracker.insert_signal(build_signal_row(res, symbol, bar_ts, cfg, feed.source))
    if sig_id is None:
        return
    sent = await broadcast(context.bot, format_signal(sig_id, symbol, res, cfg, feed.source))
    logger.info("Signal #%d %s %s @ %s score %d sent to %d users",
                sig_id, symbol, res.action, res.entry, res.score, sent)


async def scan_market(context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = context.application.bot_data["cfg"]
    results = await asyncio.gather(*(scan_symbol(context, cfg, s) for s in cfg.symbols),
                                   return_exceptions=True)
    for sym, r in zip(cfg.symbols, results):
        if isinstance(r, Exception):
            logger.error("Scan error for %s: %r", sym, r, exc_info=r)


# -------------------------------------------------------------- commands ----
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if ACCESS_CODE:
        given = context.args[0] if context.args else ""
        if not hmac.compare_digest(given.encode(), ACCESS_CODE.encode()):
            await update.message.reply_text("Access code required. Use: /start <code>")
            return
    tracker.add_user(update.effective_chat.id)
    cfg: Config = context.application.bot_data["cfg"]
    await update.message.reply_text(
        f"✅ Subscribed. You will receive {', '.join(cfg.symbols)} signals "
        f"({cfg.tf_min}m).\n\n/stats performance · /stop unsubscribe · /help\n\n"
        "Signals are informational, not financial advice.")


async def id_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handy for finding the value to put in ADMIN_CHAT_ID."""
    await update.message.reply_text(f"Your chat id: {update.effective_chat.id}")


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tracker.remove_user(update.effective_chat.id)
    await update.message.reply_text("You are unsubscribed. Send /start to re-join.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/start subscribe\n/stop unsubscribe\n/stats 30-day signal results\n\n"
        "Results are net of assumed costs and use a pessimistic intrabar rule.")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s = tracker.stats(30)
    a = s["all"]
    if not a["n"]:
        await update.message.reply_text(f"No closed signals in the last {s['days']} days "
                                        f"({s['open']} open).")
        return
    lines = [f"<b>Last {s['days']} days</b> · {a['n']} closed, {s['open']} open",
             f"Win rate {a['win_rate']:.0%} · avg {a['avg_r']:+.2f}R · total {a['total_r']:+.1f}R"]
    for sym, v in s["by_symbol"].items():
        lines.append(f"{html.escape(sym)}: {v['n']} trades, {v['win_rate']:.0%} wins, {v['total_r']:+.1f}R")
    lines.append("<i>Small samples are noise. Judge after 100+ trades.</i>")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not ADMIN_ID or update.effective_chat.id != ADMIN_ID:
        return
    cfg: Config = context.application.bot_data["cfg"]
    now = time.time()
    feeds = ", ".join(f"{s}: {int(now - t)}s ago" for s, t in last_scan_ok.items()) or "no scans yet"
    await update.message.reply_text(
        f"{cfg.summary()}\nUsers: {len(tracker.get_users())} · Open signals: "
        f"{len(tracker.get_open_signals())}\nLast OK scan: {feeds}\nFailures: {fail_counts}")


async def on_error(update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled error: %r", context.error, exc_info=context.error)


# --------------------------------------------------------------- startup ----
def start_health_server() -> None:
    """Only needed when deployed as a Render *web* service (which sets PORT)."""
    import uvicorn
    from fastapi import FastAPI
    api = FastAPI()

    @api.get("/")
    @api.get("/health")
    def health():
        return {"status": "ok", "last_scan_ok": last_scan_ok}

    threading.Thread(
        target=lambda: uvicorn.run(api, host="0.0.0.0", port=int(os.environ["PORT"]),
                                   log_level="warning"),
        daemon=True).start()


async def post_init(application) -> None:
    cfg: Config = application.bot_data["cfg"]
    for sym in cfg.symbols:
        need = cfg.min_atr_pct_for_costs(sym)
        logger.info("%s: with %.1f bps assumed costs a signal needs ATR >= %.2f%% of price on %dm candles",
                    sym, cfg.cost_bps(sym), need, cfg.tf_min)
    await notify_admin(application.bot, f"🚀 Bot started: {html.escape(cfg.summary())}\n"
                                        "Use /status for health, /stats for results.")


def main() -> None:
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("BOT_TOKEN environment variable is missing!")
    cfg = load_config()
    tracker.init_db()
    logger.info("Config: %s | DB: %s", cfg.summary(), tracker.DB_PATH)

    if os.getenv("PORT"):
        start_health_server()

    app = ApplicationBuilder().token(token).post_init(post_init).build()
    app.bot_data["cfg"] = cfg
    for name, fn in (("start", start_command), ("stop", stop_command), ("help", help_command),
                     ("stats", stats_command), ("status", status_command), ("id", id_command)):
        app.add_handler(CommandHandler(name, fn))
    app.add_error_handler(on_error)

    if app.job_queue is None:
        raise RuntimeError('Install job-queue support: pip install "python-telegram-bot[job-queue]"')
    period = cfg.tf_min * 60
    first = period - (time.time() % period) + cfg.scan_offset_s     # just after the next bar close
    app.job_queue.run_repeating(scan_market, interval=period, first=first, name="scan",
                                job_kwargs={"misfire_grace_time": 30, "coalesce": True})
    logger.info("Scanning every %dm; first scan in %.0fs", cfg.tf_min, first)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("Fatal startup error: %s", e, exc_info=True)
        raise
