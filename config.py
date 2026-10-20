"""
Central configuration.

Pick a PRESET via the PRESET env var, then override any single field with an
env var of the same name in UPPER_CASE (e.g. MIN_SCORE=60, COST_BPS_CRYPTO=6).

Secrets (BOT_TOKEN, ADMIN_CHAT_ID, ACCESS_CODE) live ONLY in environment
variables. Never hardcode them in code or commit them to git.
"""
import os
from dataclasses import dataclass, fields, replace

SUPPORTED_TF = (1, 5, 15, 60)
GOLD_SYMBOLS = ("PAXGUSDT", "XAUUSDT", "XAUUSD", "GOLD")
FUTURES_SYMBOLS = ("XAUUSDT", "XAGUSDT")          # USD-M perpetuals (fapi), not spot
DISPLAY = {"BTCUSDT": "BTC/USDT", "PAXGUSDT": "Gold (PAXG/USDT)",
           "XAUUSDT": "Gold (XAUUSDT perp)"}


@dataclass(frozen=True)
class Config:
    preset: str = "INTRADAY"
    # BTCUSDT = Bitcoin spot. Binance has NO spot XAUUSD: PAXGUSDT is gold-backed
    # and tracks spot gold closely. XAUUSDT is a (thinner) gold perpetual.
    symbols: tuple = ("BTCUSDT", "PAXGUSDT")

    # --- timeframes (minutes) ---
    tf_min: int = 15
    htf_min: int = 60
    htf_filter: bool = True          # veto trades that fight the HTF trend
    bars: int = 600                  # history fetched per scan
    eval_window: int = 300           # bars the strategy looks at
    min_bars: int = 120              # minimum closed candles required

    # --- scan timing / data health ---
    scan_offset_s: int = 5           # seconds after bar close to scan
    max_stale_bars: float = 1.5      # feed older than this many bars = stale
    max_signal_age_s: int = 120      # skip signals on bars older than this
    alert_after_failures: int = 5    # consecutive data failures before admin alert
    binance_hosts: str = ("https://data-api.binance.vision,https://api.binance.com,"
                          "https://api1.binance.com,https://api-gcp.binance.com")

    # --- structure / confluence ---
    ema_lens: tuple = (5, 11, 15, 18, 21, 24, 28, 34)   # 8-EMA ribbon
    use_sweep_filter: bool = True    # liquidity sweeps count as a trigger
    sweep_lookback_bars: int = 20
    use_fvg_filter: bool = True      # fair value gaps add confluence
    fvg_min_atr: float = 0.20        # min FVG size in ATR
    fvg_lookback: int = 3
    trigger_lookback: int = 3        # a sweep/CHoCH/BOS stays "live" this many bars
    pivot_k: int = 2                 # fractal width for swing pivots
    vp_lookback: int = 100
    vp_bins: int = 30
    delta_thr: float = 0.10          # min taker-flow imbalance (-1..1)

    # --- regime filters ---
    er_len: int = 30                 # Kaufman efficiency-ratio window
    er_min_bos: float = 0.25         # continuation (BOS) trades need a trending market
    vol_spike_mult: float = 2.5      # skip when ATR > this x its 100-bar median

    # --- momentum ---
    rsi_len: int = 7
    rsi_ob: float = 75.0             # veto BUY above this
    rsi_os: float = 25.0             # veto SELL below this

    # --- risk model (ATR based) ---
    atr_len: int = 14
    sl_mult: float = 1.5
    max_sl_mult: float = 2.5         # cap on swing-based stop distance
    use_swing_sl: bool = True
    swing_lookback: int = 8
    tp1_r: float = 1.0               # targets in multiples of the stop distance
    tp2_r: float = 2.0
    tp1_close_frac: float = 0.5      # fraction closed at TP1
    be_after_tp1: bool = True

    # --- filters / housekeeping ---
    min_score: int = 50
    min_edge: int = 10               # winning side must beat the other by this
    min_atr_pct: float = 0.06        # ATR as % of price, skip dead markets
    cooldown_bars: int = 4
    expiry_bars: int = 48
    session_start: int = 0           # UTC hours, 0-24 = always
    session_end: int = 24
    gold_follow_market_hours: bool = True   # PAXG/XAU trade 24/7 but gold itself does not
    # Assumed ROUND-TRIP cost in basis points (fees + spread + slippage).
    # Binance spot taker is 0.10% per side = 20 bps; futures taker ~0.05%/side = 10 bps.
    # Set these to what YOU actually pay: they decide which signals are tradeable.
    cost_bps_gold: float = 10.0
    cost_bps_crypto: float = 10.0
    min_sl_to_cost: float = 4.0      # stop must be >= this many x costs

    def __post_init__(self):
        if self.tf_min not in SUPPORTED_TF:
            raise ValueError(f"tf_min must be one of {SUPPORTED_TF}")
        if self.htf_min < self.tf_min or self.htf_min % self.tf_min:
            raise ValueError("htf_min must be a multiple of tf_min")
        if not 0 <= self.tp1_close_frac <= 1:
            raise ValueError("tp1_close_frac must be within [0, 1]")
        if self.tp1_r >= self.tp2_r:
            raise ValueError("tp1_r must be smaller than tp2_r")

    @staticmethod
    def is_gold(symbol: str) -> bool:
        return symbol.upper() in GOLD_SYMBOLS

    @staticmethod
    def is_futures(symbol: str) -> bool:
        return symbol.upper() in FUTURES_SYMBOLS

    @staticmethod
    def display_name(symbol: str) -> str:
        return DISPLAY.get(symbol.upper(), symbol.upper())

    def spot_hosts(self) -> list:
        return [h.strip().rstrip("/") for h in self.binance_hosts.split(",") if h.strip()]

    def cost_bps(self, symbol: str) -> float:
        return self.cost_bps_gold if self.is_gold(symbol) else self.cost_bps_crypto

    def min_atr_pct_for_costs(self, symbol: str) -> float:
        """ATR (% of price) needed before the cost filter lets a signal through."""
        return self.min_sl_to_cost * self.cost_bps(symbol) / 1e4 / self.sl_mult * 100

    def summary(self) -> str:
        return (f"{self.preset} | {self.tf_min}m (HTF {self.htf_min}m) | "
                f"min_score {self.min_score} | SL {self.sl_mult}xATR | "
                f"TP {self.tp1_r:g}R/{self.tp2_r:g}R | symbols {','.join(self.symbols)}")


PRESETS = {
    "SCALP1M": dict(tf_min=1, htf_min=5, min_atr_pct=0.012, cooldown_bars=5,
                    expiry_bars=90, bars=1000, max_signal_age_s=75),
    "SCALP": dict(tf_min=5, htf_min=15, min_atr_pct=0.03, cooldown_bars=3,
                  expiry_bars=36, bars=600, max_signal_age_s=75),
    "INTRADAY": dict(tf_min=15, htf_min=60, min_atr_pct=0.06, cooldown_bars=4,
                     expiry_bars=48, bars=600, max_signal_age_s=120),
    "SWING": dict(tf_min=60, htf_min=240, min_atr_pct=0.12, cooldown_bars=3,
                  expiry_bars=48, bars=600, max_signal_age_s=600),
}

_SKIP_ENV = {"preset", "ema_lens"}


def _cast(value: str, typ):
    if typ is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    if typ is tuple:
        return tuple(x.strip().upper() for x in value.split(",") if x.strip())
    return typ(value)


def load_config() -> Config:
    name = os.getenv("PRESET", "INTRADAY").strip().upper()
    if name not in PRESETS:
        raise ValueError(f"Unknown PRESET '{name}'. Options: {', '.join(PRESETS)}")
    cfg = Config(preset=name, **PRESETS[name])
    overrides = {}
    for f in fields(cfg):
        if f.name in _SKIP_ENV:
            continue
        v = os.getenv(f.name.upper())
        if v not in (None, ""):
            overrides[f.name] = _cast(v, f.type)
    return replace(cfg, **overrides)
