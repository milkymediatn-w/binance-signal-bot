"""
SQLite persistence for subscribers and the signal/outcome log.

Set DB_PATH to a persistent location (e.g. /data/bot.db on a Render disk).
On ephemeral filesystems the database is wiped at every deploy/restart.
"""
import os
import sqlite3
import threading
import time
from contextlib import closing
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("DB_PATH", "data/bot.db")
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    chat_id   INTEGER PRIMARY KEY,
    joined_ts INTEGER
);
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    action      TEXT NOT NULL,
    bar_ts      INTEGER NOT NULL,
    created_ts  INTEGER NOT NULL,
    entry REAL, sl REAL, tp1 REAL, tp2 REAL, risk REAL, score INTEGER,
    tp1_r REAL, tp2_r REAL, tp1_frac REAL, be_after_tp1 INTEGER,
    expiry_bars INTEGER, cost_r REAL,
    status      TEXT DEFAULT 'OPEN',
    stop_r      REAL DEFAULT -1,
    realized_r  REAL DEFAULT 0,
    bars_seen   INTEGER DEFAULT 0,
    last_ts     INTEGER,
    result_r    REAL,
    closed_ts   INTEGER,
    source      TEXT,
    UNIQUE (symbol, bar_ts)
);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status, symbol);
"""


def _connect() -> sqlite3.Connection:
    folder = os.path.dirname(os.path.abspath(DB_PATH))
    os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _exec(sql: str, params: tuple = ()) -> int:
    with _lock, closing(_connect()) as conn, conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid


def _query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    with _lock, closing(_connect()) as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def init_db(legacy_path: str = "users.db") -> None:
    with _lock, closing(_connect()) as conn, conn:
        conn.executescript(SCHEMA)
    # one-time import of subscribers from the old single-table database
    if os.path.exists(legacy_path) and os.path.abspath(legacy_path) != os.path.abspath(DB_PATH):
        try:
            with closing(sqlite3.connect(legacy_path)) as old:
                ids = [r[0] for r in old.execute("SELECT chat_id FROM users").fetchall()]
            for cid in ids:
                add_user(cid)
        except sqlite3.Error:
            pass


# ----------------------------------------------------------------- users ----
def add_user(chat_id: int) -> None:
    _exec("INSERT OR IGNORE INTO users (chat_id, joined_ts) VALUES (?, ?)",
          (chat_id, int(time.time())))


def remove_user(chat_id: int) -> None:
    _exec("DELETE FROM users WHERE chat_id = ?", (chat_id,))


def get_users() -> List[int]:
    return [r["chat_id"] for r in _query("SELECT chat_id FROM users")]


# --------------------------------------------------------------- signals ----
def insert_signal(row: Dict[str, Any]) -> Optional[int]:
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    try:
        return _exec(f"INSERT INTO signals ({cols}) VALUES ({marks})", tuple(row.values()))
    except sqlite3.IntegrityError:
        return None            # same symbol + candle already signalled


def get_open_signals(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM signals WHERE status IN ('OPEN','TP1')"
    params: tuple = ()
    if symbol:
        sql += " AND symbol = ?"
        params = (symbol,)
    return _query(sql + " ORDER BY id", params)


def update_signal(sig_id: int, fields: Dict[str, Any]) -> None:
    sets = ", ".join(f"{k} = ?" for k in fields)
    _exec(f"UPDATE signals SET {sets} WHERE id = ?", (*fields.values(), sig_id))


def last_signal_bar_ts(symbol: str) -> int:
    rows = _query("SELECT MAX(bar_ts) AS t FROM signals WHERE symbol = ?", (symbol,))
    return int(rows[0]["t"] or 0)


def stats(days: int = 30) -> Dict[str, Any]:
    since = int(time.time()) - days * 86400
    rows = _query("SELECT symbol, result_r FROM signals "
                  "WHERE closed_ts >= ? AND result_r IS NOT NULL", (since,))
    out: Dict[str, Any] = {"days": days, "open": len(get_open_signals()), "by_symbol": {}}

    def agg(rs: List[float]) -> Dict[str, Any]:
        n = len(rs)
        return {"n": n,
                "win_rate": (sum(r > 0 for r in rs) / n) if n else 0.0,
                "avg_r": (sum(rs) / n) if n else 0.0,
                "total_r": sum(rs)}

    out["all"] = agg([r["result_r"] for r in rows])
    for sym in sorted({r["symbol"] for r in rows}):
        out["by_symbol"][sym] = agg([r["result_r"] for r in rows if r["symbol"] == sym])
    return out
