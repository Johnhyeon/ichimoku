"""
Combined Portfolio Validation: Fractals + Mirror Short + MA100 + Spot DCA

Walk-Forward + Monte Carlo 검증 후 HTML 리포트 생성.

전략 구성:
  Fractals     : 4h  LONG+SHORT  10x  5%  max5   SL3% TP10% Trail2%/2%
  Mirror Short : 5m  SHORT        5x  5%  max3   SL1% Trail3%/1.2%
  MA100 V2     : 1d  SHORT ONLY   3x  2%  max20  SL5% Trail3%/2%
  Spot DCA     : 8h  LONG BTC/ETH 1x  $10/회

사용법:
    python scripts/validate_combined_portfolio.py
    python scripts/validate_combined_portfolio.py --sims 10000
"""

import argparse
import itertools
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.ichimoku import calculate_ichimoku
from src.strategy import MAJOR_COINS, STABLECOINS
from src.live_surge_mirror_short import (
    MirrorShortParams, schedule_next_candle_entries, simulate_short_exit_ohlc,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ─── Configs ──────────────────────────────────────────────────

FRACTALS_CFG = {
    "leverage": 10, "position_pct": 0.05, "max_positions": 5,
    "sl_pct": 3.0, "tp_pct": 10.0, "trail_start_pct": 2.0, "trail_pct": 2.0,
    "cooldown_candles": 2, "fee_rate": 0.00055,
}
MIRROR_CFG = {
    "leverage": 5, "position_pct": 0.05, "max_positions": 3,
    "sl_pct": 1.0, "trail_start_pct": 3.0, "trail_rebound_pct": 1.2,
    "cooldown_candles": 3, "overheat_cum_rise_pct": 8.0,
    "overheat_upper_wick_pct": 40.0, "overheat_volume_ratio": 5.0,
    "volume_lookback": 20, "roundtrip_cost_rate": 0.0009,
}
MA100_CFG = {
    "ma_period": 100, "slope_lookback": 3, "touch_buffer_pct": 1.0,
    "leverage": 3, "position_pct": 0.02, "max_positions": 20,
    "fee_rate": 0.00055, "sl_pct": 5.0, "tp_pct": 0,
    "trail_start_pct": 3.0, "trail_pct": 2.0, "cooldown_days": 3,
}
DCA_CFG = {
    "interval_hours": 8, "base_amount_usdt": 10.0,
    "btc_ratio": 0.4, "eth_ratio": 0.6,
    "profit_bonus_pct": 0.10, "min_futures_reserve": 500.0,
    "min_order_usdt": 5.0, "taker_fee": 0.001,
}
MIRROR_EXCLUDE = {"BTC/USDT:USDT", "ETH/USDT:USDT"}


# ─── Fractals Indicators ─────────────────────────────────────

def compute_fractals(df, n=5):
    h, l = df["high"].values, df["low"].values
    ln = len(df)
    fh, fl = np.full(ln, np.nan), np.full(ln, np.nan)
    for i in range(n, ln - n):
        ok = all(h[i] > h[i-j] and h[i] > h[i+j] for j in range(1, n+1))
        if ok: fh[i] = h[i]
        ok = all(l[i] < l[i-j] and l[i] < l[i+j] for j in range(1, n+1))
        if ok: fl[i] = l[i]
    df["last_fh"] = pd.Series(fh, index=df.index).ffill()
    df["last_fl"] = pd.Series(fl, index=df.index).ffill()
    return df

def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _rsi(s, p=14):
    d = s.diff()
    return 100 - 100 / (1 + d.clip(lower=0).ewm(alpha=1/p, min_periods=p).mean() /
                         (-d).clip(lower=0).ewm(alpha=1/p, min_periods=p).mean().replace(0, np.nan))
def _adx(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    pdm, mdm = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
    pdm, mdm = np.where(pdm > mdm, pdm, 0), np.where(mdm > np.array(pdm, dtype=float), mdm, 0)
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = pd.Series(tr, index=df.index).ewm(alpha=1/p, min_periods=p).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/p, min_periods=p).mean() / atr
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1/p, min_periods=p).mean() / atr
    dx = (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan) * 100
    return dx.ewm(alpha=1/p, min_periods=p).mean()


def precompute_fractals(data_4h):
    result = {}
    for sym, df in data_4h.items():
        df = df.copy()
        df = compute_fractals(df, 5)
        df["ema20"], df["ema50"] = _ema(df["close"], 20), _ema(df["close"], 50)
        df["rsi"], df["adx"] = _rsi(df["close"]), _adx(df)
        ph, pl, pc = df["last_fh"].shift(1), df["last_fl"].shift(1), df["close"].shift(1)
        lr = ((pc <= ph) & (df["close"] > df["last_fh"]) & df["last_fh"].notna()).fillna(False)
        sr = ((pc >= pl) & (df["close"] < df["last_fl"]) & df["last_fl"].notna()).fillna(False)
        el, es = df["ema20"] > df["ema50"], df["ema20"] < df["ema50"]
        adx_ok = df["adx"] >= 20
        df["frac_long"] = lr & el & (df["rsi"] <= 65) & adx_ok
        df["frac_short"] = sr & es & (df["rsi"] >= 35) & adx_ok
        result[sym] = df.reset_index(drop=True)
    return result


# ─── Mirror Signals ───────────────────────────────────────────

def precompute_mirror(data_5m, cfg):
    params = MirrorShortParams(
        overheat_cum_rise_pct=cfg["overheat_cum_rise_pct"],
        overheat_upper_wick_pct=cfg["overheat_upper_wick_pct"],
        overheat_volume_ratio=cfg["overheat_volume_ratio"],
        volume_lookback=cfg["volume_lookback"],
        stop_loss_pct=cfg["sl_pct"],
        trail_start_pct=cfg["trail_start_pct"],
        trail_rebound_pct=cfg["trail_rebound_pct"],
    )
    result = {}
    for sym, df in data_5m.items():
        if sym in MIRROR_EXCLUDE:
            continue
        df = df.copy()
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        df["change_pct"] = df["close"].pct_change() * 100
        df["is_green"] = df["close"] > df["open"]
        rh = df["high"].rolling(12).max().shift(1)
        rl = df["low"].rolling(12).min().shift(1)
        df["consol"] = (rh - rl) / rl * 100
        df["pfl"] = (df["close"] - df["low"].shift(1)) / df["low"].shift(1) * 100
        base = ((df["vol_ratio"] >= 10) & (df["change_pct"] >= 5) & df["is_green"]
                & (df["consol"] <= 5) & (df["pfl"] <= 15)).fillna(False)
        cum_rise = (df["close"] / df["close"].shift(3) - 1) * 100
        cr = (df["high"] - df["low"]).clip(lower=1e-12)
        uw = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0) / cr * 100
        va = df["volume"].shift(1).rolling(cfg["volume_lookback"]).mean()
        vr = df["volume"] / va
        overheat = ((cum_rise >= cfg["overheat_cum_rise_pct"]) | (uw >= cfg["overheat_upper_wick_pct"])
                    | (vr >= cfg["overheat_volume_ratio"])).fillna(False)
        entry = schedule_next_candle_entries(base, overheat, delay_candles=1)
        df["mirror_entry"] = entry
        result[sym] = df
    return result


# ─── MA100 Signals ────────────────────────────────────────────

def precompute_ma100(data_1d, cfg):
    result = {}
    for sym, df in data_1d.items():
        df = df.copy()
        df["ma100"] = df["close"].rolling(cfg["ma_period"]).mean()
        df["slope"] = (df["ma100"] - df["ma100"].shift(cfg["slope_lookback"])) / df["ma100"].shift(cfg["slope_lookback"]) * 100
        tb = cfg["touch_buffer_pct"] / 100
        df["ma100_short"] = ((df["slope"] < 0) & (df["high"] >= df["ma100"] * (1 - tb)) & (df["close"] < df["ma100"])).fillna(False)
        result[sym] = df
    return result


# ─── Data Loading ─────────────────────────────────────────────

def _is_stable(s):
    return s.split('/')[0] if '/' in s else s in STABLECOINS

def _load_5m_and_precompute_mirror(loader, start_dt, end_dt, cfg):
    """5m 데이터를 1심볼씩 로드→시그널 계산→시그널만 보관 (메모리 절약)."""
    avail = set(loader.get_available_symbols())
    all_syms = [s for s in avail if s.split('/')[0] not in STABLECOINS and s not in MIRROR_EXCLUDE]

    # Mirror signal events: {timestamp: [symbol, ...]}
    mirror_events = {}
    # 5m OHLCV for exit simulation: {symbol: DataFrame} — 시그널 있는 심볼만
    mirror_5m = {}
    w5m = timedelta(hours=3)
    start_str = (start_dt - w5m).strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    start_ts, end_ts = pd.Timestamp(start_dt), pd.Timestamp(end_dt)

    params = MirrorShortParams(
        overheat_cum_rise_pct=cfg["overheat_cum_rise_pct"],
        overheat_upper_wick_pct=cfg["overheat_upper_wick_pct"],
        overheat_volume_ratio=cfg["overheat_volume_ratio"],
        volume_lookback=cfg["volume_lookback"],
        stop_loss_pct=cfg["sl_pct"],
        trail_start_pct=cfg["trail_start_pct"],
        trail_rebound_pct=cfg["trail_rebound_pct"],
    )

    n_valid = 0
    for i, sym in enumerate(all_syms):
        if "5m" not in loader.get_available_timeframes(sym): continue
        df = loader.load(sym, "5m", start=start_str, end=end_str)
        if df is None or len(df) < 30: continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
        rng = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        if len(rng) < 30: continue

        # Compute signals
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        df["change_pct"] = df["close"].pct_change() * 100
        df["is_green"] = df["close"] > df["open"]
        rh = df["high"].rolling(12).max().shift(1)
        rl = df["low"].rolling(12).min().shift(1)
        df["consol"] = (rh - rl) / rl * 100
        df["pfl"] = (df["close"] - df["low"].shift(1)) / df["low"].shift(1) * 100
        base = ((df["vol_ratio"] >= 10) & (df["change_pct"] >= 5) & df["is_green"]
                & (df["consol"] <= 5) & (df["pfl"] <= 15)).fillna(False)
        cum_rise = (df["close"] / df["close"].shift(3) - 1) * 100
        cr = (df["high"] - df["low"]).clip(lower=1e-12)
        uw = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0) / cr * 100
        va = df["volume"].shift(1).rolling(cfg["volume_lookback"]).mean()
        vr = df["volume"] / va
        overheat = ((cum_rise >= cfg["overheat_cum_rise_pct"]) | (uw >= cfg["overheat_upper_wick_pct"])
                    | (vr >= cfg["overheat_volume_ratio"])).fillna(False)
        entry = schedule_next_candle_entries(base, overheat, delay_candles=1)

        # 시그널 있는 심볼만 보관
        entry_mask = entry & (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
        if entry_mask.any():
            # 심볼의 5m 데이터를 보관 (exit 시뮬에 필요)
            mirror_5m[sym] = df[["timestamp", "open", "high", "low", "close"]].copy()
            for _, row in df[entry_mask].iterrows():
                mirror_events.setdefault(row["timestamp"], []).append(sym)
            n_valid += 1

        if (i+1) % 100 == 0:
            logger.info(f"  5m: {i+1}/{len(all_syms)} ({n_valid} with signals)")

    logger.info(f"  Mirror: {n_valid} symbols with signals, {sum(len(v) for v in mirror_events.values())} total events")
    return mirror_events, mirror_5m


def load_all_data(loader, start_dt, end_dt):
    """모든 전략에 필요한 데이터 로드."""
    data = {"4h": {}, "1d": {}, "1h_btc": None, "1h_eth": None}
    avail = set(loader.get_available_symbols())
    frac_syms = [s for s in MAJOR_COINS if s in avail]

    # 4h for Fractals (20 coins)
    logger.info("Loading 4h data (Fractals)...")
    w4h = timedelta(days=30)
    for sym in frac_syms:
        if "4h" not in loader.get_available_timeframes(sym): continue
        df = loader.load(sym, "4h", start=(start_dt - w4h).strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
        if df is not None and len(df) >= 60:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            data["4h"][sym] = df
    logger.info(f"  4h: {len(data['4h'])} symbols")

    # 5m for Mirror Short — stream process (메모리 절약)
    logger.info("Loading 5m + computing Mirror signals (streaming)...")
    mirror_events, mirror_5m = _load_5m_and_precompute_mirror(loader, start_dt, end_dt, MIRROR_CFG)
    data["mirror_events"] = mirror_events
    data["mirror_5m"] = mirror_5m

    # 1d for MA100 (all non-stablecoin)
    logger.info("Loading 1d data (MA100)...")
    w1d = timedelta(days=150)
    all_1d_syms = [s for s in avail if s.split('/')[0] not in STABLECOINS]
    for sym in all_1d_syms:
        tfs = loader.get_available_timeframes(sym)
        df = None
        if "1d" in tfs:
            df = loader.load(sym, "1d", start=(start_dt - w1d).strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
            if df is None or len(df) < 100: df = None
        if df is None and "4h" in tfs:
            raw = loader.load(sym, "4h", start=(start_dt - w1d).strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
            if raw is not None and len(raw) >= 600:
                raw["timestamp"] = pd.to_datetime(raw["timestamp"])
                df = raw.set_index("timestamp").resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
                if len(df) < 100: df = None
        if df is not None:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            data["1d"][sym] = df
    logger.info(f"  1d: {len(data['1d'])} symbols")

    # 1h BTC/ETH for DCA
    for coin, key in [("BTC/USDT:USDT", "1h_btc"), ("ETH/USDT:USDT", "1h_eth")]:
        tfs = loader.get_available_timeframes(coin)
        if "1h" in tfs:
            df = loader.load(coin, "1h", start=(start_dt - timedelta(days=1)).strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
            if df is not None:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                data[key] = df[["timestamp", "close"]].copy()
        if data[key] is None and "5m" in tfs:
            raw = loader.load(coin, "5m", start=(start_dt - timedelta(days=1)).strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
            if raw is not None:
                raw["timestamp"] = pd.to_datetime(raw["timestamp"])
                data[key] = raw.set_index("timestamp").resample("1h").agg({"close":"last"}).dropna().reset_index()

    return data


# ─── Combined Simulation ─────────────────────────────────────

def run_combined(data, start_dt, end_dt, initial_balance=6500,
                 enable_fractals=True, enable_mirror=True, enable_ma100=True, enable_dca=True):
    """4 전략 통합 시뮬레이션 (공유 잔고)."""
    start_ts, end_ts = pd.Timestamp(start_dt), pd.Timestamp(end_dt)
    balance = initial_balance
    positions = {"fractals": {}, "mirror": {}, "ma100": {}}
    last_exits = {"fractals": {}, "mirror": {}, "ma100": {}}
    trades = []
    equity_curve = []
    peak_eq = initial_balance
    max_dd_pct = 0.0

    # DCA state
    dca_last_time = None
    dca_last_pnl_idx = 0
    spot = {"BTC": {"qty": 0, "cost": 0, "count": 0}, "ETH": {"qty": 0, "cost": 0, "count": 0}}
    spot_prices = {}

    # ── Precompute signals ──
    frac_pc = precompute_fractals(data["4h"]) if enable_fractals else {}
    ma100_pc = precompute_ma100(data["1d"], MA100_CFG) if enable_ma100 else {}

    # Mirror: pre-computed events from data loading
    mirror_events = data.get("mirror_events", {}) if enable_mirror else {}
    mirror_5m = data.get("mirror_5m", {}) if enable_mirror else {}

    # ── Build timestamp index maps ──
    def build_idx(pc_dict):
        idx = {}
        for sym, df in pc_dict.items():
            idx[sym] = (df, {ts: i for i, ts in enumerate(df["timestamp"].tolist())})
        return idx

    frac_idx = build_idx(frac_pc)
    ma100_idx = build_idx(ma100_pc)
    mirror_idx = build_idx(mirror_5m)  # 5m data for exit simulation

    # DCA price maps
    dca_btc = {}
    if data.get("1h_btc") is not None and enable_dca:
        for _, r in data["1h_btc"].iterrows(): dca_btc[r["timestamp"]] = float(r["close"])
    dca_eth = {}
    if data.get("1h_eth") is not None and enable_dca:
        for _, r in data["1h_eth"].iterrows(): dca_eth[r["timestamp"]] = float(r["close"])

    # ── Build unified timeline ──
    all_ts = set()
    for d in [frac_pc, mirror_5m, ma100_pc]:
        for df in d.values():
            sub = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
            all_ts.update(sub["timestamp"].tolist())
    if enable_dca:
        for ts in dca_btc:
            if start_ts <= ts <= end_ts: all_ts.add(ts)
    all_ts = sorted(all_ts)
    if not all_ts:
        return {"trades": [], "balance": balance, "equity_curve": [], "spot": spot, "spot_prices": spot_prices, "max_dd_pct": 0}

    # Track which timestamps belong to which timeframe
    ts_4h = set()
    for df in frac_pc.values():
        sub = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        ts_4h.update(sub["timestamp"].tolist())
    ts_5m = set()
    for df in mirror_5m.values():
        sub = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        ts_5m.update(sub["timestamp"].tolist())
    ts_1d = set()
    for df in ma100_pc.values():
        sub = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        ts_1d.update(sub["timestamp"].tolist())

    eq_counter = 0
    for ts in all_ts:
        is_4h = ts in ts_4h
        is_5m = ts in ts_5m
        is_1d = ts in ts_1d

        # ═══ EXIT CHECKS ═══

        # Fractals exits (4h)
        if is_4h:
            for sym in list(positions["fractals"]):
                if sym not in frac_idx: continue
                df, ti = frac_idx[sym]
                if ts not in ti: continue
                row = df.iloc[ti[ts]]
                pos = positions["fractals"][sym]
                side, ep = pos["side"], pos["ep"]
                high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
                if side == "long":
                    cur = (close/ep-1)*100; best = max(pos["best"], (high/ep-1)*100)
                else:
                    cur = (1-close/ep)*100; best = max(pos["best"], (1-low/ep)*100)
                pos["best"] = best
                er = ep_out = None
                if side == "long" and low <= pos["sl"]: er, ep_out = "SL", pos["sl"]
                elif side == "short" and high >= pos["sl"]: er, ep_out = "SL", pos["sl"]
                if not er:
                    if side == "long" and high >= pos["tp"]: er, ep_out = "TP", pos["tp"]
                    elif side == "short" and low <= pos["tp"]: er, ep_out = "TP", pos["tp"]
                if not er and best >= FRACTALS_CFG["trail_start_pct"] and best - cur >= FRACTALS_CFG["trail_pct"]:
                    er, ep_out = "Trail", close
                if er:
                    pp = (ep_out/ep-1)*100 if side == "long" else (1-ep_out/ep)*100
                    pnl = pos["sz"] * FRACTALS_CFG["leverage"] * pp / 100 - pos["sz"] * FRACTALS_CFG["fee_rate"] * 2
                    balance += pnl
                    trades.append({"sym": sym, "strategy": "fractals", "side": side, "pnl": pnl, "reason": er, "entry_time": pos["et"], "exit_time": ts})
                    del positions["fractals"][sym]
                    last_exits["fractals"][sym] = ts

        # Mirror exits (5m)
        if is_5m:
            mp = MirrorShortParams(stop_loss_pct=MIRROR_CFG["sl_pct"], trail_start_pct=MIRROR_CFG["trail_start_pct"], trail_rebound_pct=MIRROR_CFG["trail_rebound_pct"])
            for sym in list(positions["mirror"]):
                if sym not in mirror_idx: continue
                df, ti = mirror_idx[sym]
                if ts not in ti: continue
                row = df.iloc[ti[ts]]
                pos = positions["mirror"][sym]
                cd = {"open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"])}
                exit_info, updated = simulate_short_exit_ohlc(pos, cd, mp)
                positions["mirror"][sym] = updated
                if exit_info:
                    ep = pos["entry_price"]
                    exp = float(exit_info["price"])
                    qty = pos["qty"]
                    pnl = (ep - exp) * qty - ep * qty * MIRROR_CFG["roundtrip_cost_rate"]
                    balance += pnl
                    trades.append({"sym": sym, "strategy": "mirror", "side": "short", "pnl": pnl, "reason": exit_info["reason"], "entry_time": pos["entry_time"], "exit_time": ts})
                    del positions["mirror"][sym]
                    last_exits["mirror"][sym] = ts

        # MA100 exits (1d)
        if is_1d:
            for sym in list(positions["ma100"]):
                if sym not in ma100_idx: continue
                df, ti = ma100_idx[sym]
                if ts not in ti: continue
                row = df.iloc[ti[ts]]
                pos = positions["ma100"][sym]
                ep, high, low, close = pos["ep"], float(row["high"]), float(row["low"]), float(row["close"])
                # SL
                if high >= pos["sl"]:
                    pnl = pos["sz"] * MA100_CFG["leverage"] * ((ep - pos["sl"]) / ep * 100) / 100 - pos["sz"] * MA100_CFG["fee_rate"] * 2
                    balance += pnl
                    trades.append({"sym": sym, "strategy": "ma100", "side": "short", "pnl": pnl, "reason": "SL", "entry_time": pos["et"], "exit_time": ts})
                    del positions["ma100"][sym]; last_exits["ma100"][sym] = ts; continue
                # Trail
                cur_pnl = (ep - close) / ep * 100
                if cur_pnl >= MA100_CFG["trail_start_pct"]:
                    pos["trailing"] = True
                    if low < pos.get("lowest", ep):
                        pos["lowest"] = low
                        pos["trail_stop"] = low * (1 + MA100_CFG["trail_pct"] / 100)
                if pos.get("trailing") and close >= pos.get("trail_stop", ep * 2):
                    exp = pos["trail_stop"]
                    pnl = pos["sz"] * MA100_CFG["leverage"] * ((ep - exp) / ep * 100) / 100 - pos["sz"] * MA100_CFG["fee_rate"] * 2
                    balance += pnl
                    trades.append({"sym": sym, "strategy": "ma100", "side": "short", "pnl": pnl, "reason": "Trail", "entry_time": pos["et"], "exit_time": ts})
                    del positions["ma100"][sym]; last_exits["ma100"][sym] = ts; continue

        # ═══ ENTRY CHECKS ═══

        # Fractals entries (4h)
        if is_4h and enable_fractals and len(positions["fractals"]) < FRACTALS_CFG["max_positions"] and balance > 0:
            cands = []
            for sym in frac_idx:
                if sym in positions["fractals"]: continue
                le = last_exits["fractals"].get(sym)
                if le and len([t for t in ts_4h if le < t <= ts]) < FRACTALS_CFG["cooldown_candles"]: continue
                df, ti = frac_idx[sym]
                if ts not in ti: continue
                row = df.iloc[ti[ts]]
                if row["frac_long"]: cands.append((sym, "long", float(row["volume"]), float(row["close"])))
                elif row["frac_short"]: cands.append((sym, "short", float(row["volume"]), float(row["close"])))
            cands.sort(key=lambda x: x[2], reverse=True)
            for sym, side, _, price in cands:
                if len(positions["fractals"]) >= FRACTALS_CFG["max_positions"]: break
                sz = balance * FRACTALS_CFG["position_pct"]
                if sz < 5: break
                sl = price * (1 - FRACTALS_CFG["sl_pct"]/100) if side == "long" else price * (1 + FRACTALS_CFG["sl_pct"]/100)
                tp = price * (1 + FRACTALS_CFG["tp_pct"]/100) if side == "long" else price * (1 - FRACTALS_CFG["tp_pct"]/100)
                positions["fractals"][sym] = {"side": side, "ep": price, "sz": sz, "sl": sl, "tp": tp, "best": 0, "et": ts}

        # Mirror entries (5m) — event-driven
        if is_5m and enable_mirror and len(positions["mirror"]) < MIRROR_CFG["max_positions"] and balance > 0:
            entry_syms = mirror_events.get(ts, [])
            for sym in entry_syms:
                if sym in positions["mirror"]: continue
                if len(positions["mirror"]) >= MIRROR_CFG["max_positions"]: break
                le = last_exits["mirror"].get(sym)
                if le and (ts - le) < timedelta(minutes=MIRROR_CFG["cooldown_candles"] * 5): continue
                if sym not in mirror_idx: continue
                df, ti = mirror_idx[sym]
                if ts not in ti: continue
                row = df.iloc[ti[ts]]
                ep = float(row["open"])
                if ep <= 0: continue
                margin = min(balance * MIRROR_CFG["position_pct"], 5000)
                qty = margin * MIRROR_CFG["leverage"] / ep
                if qty <= 0: break
                positions["mirror"][sym] = {
                    "entry_price": ep, "entry_time": ts, "stop_loss": ep * (1 + MIRROR_CFG["sl_pct"]/100),
                    "trailing_active": False, "lowest_since_entry": ep, "trail_stop": None, "qty": qty, "margin": margin,
                }

        # MA100 entries (1d)
        if is_1d and enable_ma100 and len(positions["ma100"]) < MA100_CFG["max_positions"] and balance > 0:
            for sym in ma100_idx:
                if sym in positions["ma100"]: continue
                if len(positions["ma100"]) >= MA100_CFG["max_positions"]: break
                df, ti = ma100_idx[sym]
                if ts not in ti: continue
                row = df.iloc[ti[ts]]
                if not row["ma100_short"]: continue
                le = last_exits["ma100"].get(sym)
                if le and (ts - le) < timedelta(days=MA100_CFG["cooldown_days"]): continue
                price = float(row["close"])
                sz = balance * MA100_CFG["position_pct"]
                if sz < 5: break
                sl = price * (1 + MA100_CFG["sl_pct"] / 100)
                positions["ma100"][sym] = {"ep": price, "sz": sz, "sl": sl, "lowest": price, "trailing": False, "trail_stop": sl, "et": ts}

        # DCA (hourly)
        if enable_dca and ts in dca_btc:
            bp = dca_btc.get(ts, 0)
            ep_eth = dca_eth.get(ts, 0)
            if bp > 0: spot_prices["BTC"] = bp
            if ep_eth > 0: spot_prices["ETH"] = ep_eth
            if bp > 0 and ep_eth > 0:
                cfg = DCA_CFG
                interval = timedelta(hours=cfg["interval_hours"])
                if dca_last_time is None or ts >= dca_last_time + interval:
                    base_amt = cfg["base_amount_usdt"]
                    bonus = sum(t["pnl"] for t in trades[dca_last_pnl_idx:] if t["strategy"] != "dca" and t["pnl"] > 0) * cfg["profit_bonus_pct"]
                    total = base_amt + max(bonus, 0)
                    dca_last_pnl_idx = len(trades)
                    avail = balance - cfg["min_futures_reserve"]
                    if avail >= total:
                        for asset, ratio, ap in [("BTC", cfg["btc_ratio"], bp), ("ETH", cfg["eth_ratio"], ep_eth)]:
                            amt = total * ratio
                            if amt < cfg["min_order_usdt"]: continue
                            fee = amt * cfg["taker_fee"]
                            cost = amt + fee
                            if cost > balance - cfg["min_futures_reserve"]: continue
                            qty = amt / ap
                            balance -= cost
                            spot[asset]["qty"] += qty; spot[asset]["cost"] += cost; spot[asset]["count"] += 1
                            trades.append({"sym": f"{asset}/USDT", "strategy": "dca", "side": "long", "pnl": -fee, "reason": "DCABuy", "entry_time": ts, "exit_time": ts})
                    dca_last_time = ts

        # Equity snapshot
        eq_counter += 1
        if eq_counter % 6 == 0 or ts == all_ts[-1]:
            eq = balance
            for strat, pdict in positions.items():
                for sym, pos in pdict.items():
                    # rough unrealized
                    pass  # simplified: just track realized balance
            spot_val = sum(spot_prices.get(a, 0) * spot[a]["qty"] for a in spot)
            equity_curve.append({"timestamp": ts, "equity": balance + spot_val})
            if balance + spot_val > peak_eq: peak_eq = balance + spot_val
            dd = (balance + spot_val - peak_eq) / peak_eq * 100 if peak_eq > 0 else 0
            if dd < max_dd_pct: max_dd_pct = dd

    # Force close remaining
    for strat in ["fractals", "mirror", "ma100"]:
        for sym in list(positions[strat]):
            trades.append({"sym": sym, "strategy": strat, "side": "short", "pnl": 0, "reason": "ForceClose", "entry_time": positions[strat][sym].get("et", all_ts[-1]), "exit_time": all_ts[-1]})

    return {
        "trades": trades, "balance": balance, "equity_curve": equity_curve,
        "spot": spot, "spot_prices": spot_prices, "max_dd_pct": max_dd_pct,
        "initial_balance": initial_balance,
    }


# ─── Monte Carlo ──────────────────────────────────────────────

def monte_carlo(trade_pnls, initial_balance=6500, n_sims=10000):
    pnls = np.array(trade_pnls)
    n = len(pnls)
    if n == 0: return None
    rng = np.random.default_rng(42)
    results = []
    for _ in range(n_sims):
        s = rng.choice(pnls, size=n, replace=True)
        eq = initial_balance + np.cumsum(s)
        final = eq[-1]
        peak = np.maximum.accumulate(eq)
        mdd = ((eq - peak) / peak * 100).min()
        ws = s[s > 0].sum(); ls = abs(s[s <= 0].sum())
        results.append({"final": final, "return_pct": (final - initial_balance) / initial_balance * 100, "mdd": mdd, "pf": ws/ls if ls > 0 else 999})
    return pd.DataFrame(results)


# ─── HTML Report ──────────────────────────────────────────────

def generate_html(train_r, test_r, full_r, mc_df, mc_lev, initial_balance):
    """상세 HTML 리포트 생성."""

    def strat_stats(trades, strat_name):
        st = [t for t in trades if t["strategy"] == strat_name and t["reason"] != "DCABuy"]
        if not st: return {"trades": 0, "pnl": 0, "wr": 0, "pf": 0}
        w = [t for t in st if t["pnl"] > 0]
        gp = sum(t["pnl"] for t in w) if w else 0
        gl = abs(sum(t["pnl"] for t in st if t["pnl"] <= 0))
        return {"trades": len(st), "pnl": sum(t["pnl"] for t in st), "wr": len(w)/len(st)*100 if st else 0, "pf": gp/gl if gl > 0 else 999}

    def period_stats(result):
        t = [x for x in result["trades"] if x["reason"] != "DCABuy"]
        if not t: return {"trades": 0, "pnl": 0, "wr": 0, "pf": 0, "mdd": 0}
        w = [x for x in t if x["pnl"] > 0]
        gp = sum(x["pnl"] for x in w); gl = abs(sum(x["pnl"] for x in t if x["pnl"] <= 0))
        return {"trades": len(t), "pnl": sum(x["pnl"] for x in t), "wr": len(w)/len(t)*100, "pf": gp/gl if gl > 0 else 999, "mdd": result["max_dd_pct"]}

    full_s = period_stats(full_r)
    train_s = period_stats(train_r)
    test_s = period_stats(test_r)

    # Strategy breakdown (full period)
    strats = {}
    for sn in ["fractals", "mirror", "ma100"]:
        strats[sn] = strat_stats(full_r["trades"], sn)

    # Monthly PnL
    monthly = {}
    for t in full_r["trades"]:
        if t["reason"] == "DCABuy": continue
        m = pd.Timestamp(t["exit_time"]).to_period("M")
        if m not in monthly: monthly[m] = {"fractals": 0, "mirror": 0, "ma100": 0, "total": 0}
        monthly[m][t["strategy"]] += t["pnl"]
        monthly[m]["total"] += t["pnl"]

    # Monte Carlo percentiles
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    mc_rows = []
    for p in pcts:
        mc_rows.append({
            "pct": p,
            "ret": np.percentile(mc_df["return_pct"], p),
            "final": np.percentile(mc_df["final"], p),
            "mdd": np.percentile(mc_df["mdd"], p),
            "pf": np.percentile(mc_df["pf"], p),
        })

    # Spot DCA
    spot_info = []
    for asset in ["BTC", "ETH"]:
        h = full_r["spot"][asset]
        if h["qty"] > 0:
            price = full_r["spot_prices"].get(asset, 0)
            val = h["qty"] * price
            spot_info.append({"asset": asset, "qty": h["qty"], "cost": h["cost"], "value": val, "pnl": val - h["cost"], "count": h["count"]})

    ret_pct = full_s["pnl"] / initial_balance * 100
    p_loss = (mc_df["return_pct"] < 0).mean() * 100
    var5 = np.percentile(mc_df["return_pct"], 5)
    cvar5 = mc_df["return_pct"][mc_df["return_pct"] <= var5].mean()

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Combined Portfolio Validation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #38bdf8; border-bottom: 2px solid #1e3a5f; padding-bottom: 10px; }}
h2 {{ color: #a78bfa; margin-top: 30px; }}
h3 {{ color: #67e8f9; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #334155; }}
th {{ background: #1e293b; color: #94a3b8; font-weight: 600; }}
td:first-child, th:first-child {{ text-align: left; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 20px; margin: 15px 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
.metric {{ text-align: center; }}
.metric .value {{ font-size: 2em; font-weight: 700; }}
.metric .label {{ color: #94a3b8; font-size: 0.85em; }}
.green {{ color: #4ade80; }} .red {{ color: #f87171; }} .yellow {{ color: #fbbf24; }} .blue {{ color: #60a5fa; }}
.pass {{ background: #166534; color: #4ade80; padding: 2px 8px; border-radius: 4px; }}
.fail {{ background: #991b1b; color: #f87171; padding: 2px 8px; border-radius: 4px; }}
.bar {{ height: 20px; border-radius: 3px; display: inline-block; min-width: 2px; }}
.bar-pos {{ background: #4ade80; }} .bar-neg {{ background: #f87171; }}
tr:hover {{ background: #1e293b; }}
</style></head><body><div class="container">

<h1>Combined Portfolio Validation Report</h1>
<p style="color:#94a3b8;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Initial Balance: ${initial_balance:,.0f}</p>

<h2>Strategy Configuration</h2>
<div class="card">
<table>
<tr><th>Strategy</th><th>Timeframe</th><th>Direction</th><th>Leverage</th><th>Position</th><th>Max Pos</th><th>SL</th><th>TP</th><th>Trail</th></tr>
<tr><td>Fractals</td><td>4h</td><td>Long + Short</td><td>10x</td><td>5%</td><td>5</td><td>3%</td><td>10%</td><td>2%/2%</td></tr>
<tr><td>Mirror Short</td><td>5m</td><td>Short</td><td>5x</td><td>5%</td><td>3</td><td>1%</td><td>-</td><td>3%/1.2%</td></tr>
<tr><td>MA100 V2</td><td>1d</td><td>Short</td><td>3x</td><td>2%</td><td>20</td><td>5%</td><td>-</td><td>3%/2%</td></tr>
<tr><td>Spot DCA</td><td>8h</td><td>Long (Spot)</td><td>1x</td><td>$10/cycle</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
</table>
</div>

<h2>Overall Results (Full Period)</h2>
<div class="card grid">
<div class="metric"><div class="value {'green' if ret_pct > 0 else 'red'}">{ret_pct:+.1f}%</div><div class="label">Total Return</div></div>
<div class="metric"><div class="value">${full_r['balance']:,.0f}</div><div class="label">Final Balance</div></div>
<div class="metric"><div class="value {'green' if full_s['pf'] >= 2 else 'yellow' if full_s['pf'] >= 1.5 else 'red'}">{full_s['pf']:.2f}</div><div class="label">Profit Factor</div></div>
<div class="metric"><div class="value">{full_s['wr']:.1f}%</div><div class="label">Win Rate</div></div>
<div class="metric"><div class="value">{full_s['trades']}</div><div class="label">Total Trades</div></div>
<div class="metric"><div class="value red">{full_s['mdd']:.1f}%</div><div class="label">Max Drawdown</div></div>
</div>

<h2>Strategy Breakdown</h2>
<div class="card">
<table>
<tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>PnL</th><th>PF</th><th>Contribution</th></tr>"""

    total_strat_pnl = sum(s["pnl"] for s in strats.values())
    for sn, label in [("fractals", "Fractals"), ("mirror", "Mirror Short"), ("ma100", "MA100 V2")]:
        s = strats[sn]
        contrib = s["pnl"] / total_strat_pnl * 100 if total_strat_pnl != 0 else 0
        c = "green" if s["pnl"] > 0 else "red"
        html += f'<tr><td>{label}</td><td>{s["trades"]}</td><td>{s["wr"]:.1f}%</td><td class="{c}">${s["pnl"]:+,.0f}</td><td>{s["pf"]:.2f}</td><td>{contrib:.0f}%</td></tr>'

    if spot_info:
        total_spot_pnl = sum(s["pnl"] for s in spot_info)
        total_spot_cost = sum(s["cost"] for s in spot_info)
        html += f'<tr><td>Spot DCA</td><td>{sum(s["count"] for s in spot_info)} buys</td><td>-</td><td class="{"green" if total_spot_pnl > 0 else "red"}">${total_spot_pnl:+,.0f}</td><td>-</td><td>Hold</td></tr>'

    html += """</table></div>"""

    # Walk-Forward
    pf_ratio = test_s["pf"] / train_s["pf"] if train_s["pf"] > 0 else 0
    verdict = ("PASS", "pass") if pf_ratio >= 0.7 else (("MARGINAL", "yellow") if pf_ratio >= 0.5 else ("FAIL", "fail"))
    html += f"""
<h2>Walk-Forward Validation</h2>
<div class="card">
<table>
<tr><th>Period</th><th>Trades</th><th>Win Rate</th><th>PnL</th><th>Return</th><th>PF</th><th>MDD</th></tr>
<tr><td>Train (Jan~Aug 2025)</td><td>{train_s['trades']}</td><td>{train_s['wr']:.1f}%</td><td>${train_s['pnl']:+,.0f}</td><td>{train_s['pnl']/initial_balance*100:+.1f}%</td><td>{train_s['pf']:.2f}</td><td>{train_s['mdd']:.1f}%</td></tr>
<tr><td><b>Test (Sep 2025~Mar 2026)</b></td><td>{test_s['trades']}</td><td>{test_s['wr']:.1f}%</td><td class="{'green' if test_s['pnl'] > 0 else 'red'}">${test_s['pnl']:+,.0f}</td><td>{test_s['pnl']/initial_balance*100:+.1f}%</td><td><b>{test_s['pf']:.2f}</b></td><td>{test_s['mdd']:.1f}%</td></tr>
<tr><td>Full</td><td>{full_s['trades']}</td><td>{full_s['wr']:.1f}%</td><td>${full_s['pnl']:+,.0f}</td><td>{ret_pct:+.1f}%</td><td>{full_s['pf']:.2f}</td><td>{full_s['mdd']:.1f}%</td></tr>
</table>
<p>PF Degradation: Train {train_s['pf']:.2f} → Test {test_s['pf']:.2f} (ratio: {pf_ratio:.2f}) <span class="{verdict[1]}">{verdict[0]}</span></p>
</div>"""

    # Monte Carlo
    html += f"""
<h2>Monte Carlo Simulation ({len(mc_df):,} runs)</h2>
<div class="card">
<h3>Return Distribution</h3>
<table>
<tr><th>Percentile</th><th>Return</th><th>Final Balance</th><th>MDD</th><th>PF</th></tr>"""
    for r in mc_rows:
        html += f'<tr><td>{r["pct"]}%</td><td>{r["ret"]:+.1f}%</td><td>${r["final"]:,.0f}</td><td>{r["mdd"]:.1f}%</td><td>{r["pf"]:.2f}</td></tr>'
    html += f"""</table>

<h3>Risk Metrics</h3>
<div class="grid">
<div class="metric"><div class="value {'green' if p_loss < 5 else 'red'}">{p_loss:.1f}%</div><div class="label">P(Loss)</div></div>
<div class="metric"><div class="value">{var5:+.1f}%</div><div class="label">VaR 5%</div></div>
<div class="metric"><div class="value">{cvar5:+.1f}%</div><div class="label">CVaR 5%</div></div>
<div class="metric"><div class="value">{mc_df['mdd'].min():.1f}%</div><div class="label">Worst MDD</div></div>
</div>

<h3>Leverage Scenarios (Monte Carlo median)</h3>
<table>
<tr><th>Leverage</th><th>Median Return</th><th>P5 Return</th><th>Median MDD</th><th>P5 MDD</th><th>P(Loss)</th></tr>"""
    for lev, mc_l in mc_lev.items():
        pl = (mc_l["return_pct"] < 0).mean() * 100
        html += f'<tr><td>{lev}x</td><td>{mc_l["return_pct"].median():+.1f}%</td><td>{np.percentile(mc_l["return_pct"], 5):+.1f}%</td><td>{mc_l["mdd"].median():.1f}%</td><td>{np.percentile(mc_l["mdd"], 5):.1f}%</td><td>{pl:.1f}%</td></tr>'
    html += """</table></div>"""

    # Monthly PnL
    html += """<h2>Monthly PnL Breakdown</h2><div class="card"><table>
<tr><th>Month</th><th>Fractals</th><th>Mirror</th><th>MA100</th><th>Total</th><th></th></tr>"""
    max_abs = max(abs(v["total"]) for v in monthly.values()) if monthly else 1
    for m in sorted(monthly):
        v = monthly[m]
        bar_w = min(abs(v["total"]) / max_abs * 150, 150)
        bar_cls = "bar-pos" if v["total"] >= 0 else "bar-neg"
        c = "green" if v["total"] >= 0 else "red"
        html += f'<tr><td>{str(m)}</td><td class="{"green" if v["fractals"]>=0 else "red"}">${v["fractals"]:+,.0f}</td><td class="{"green" if v["mirror"]>=0 else "red"}">${v["mirror"]:+,.0f}</td><td class="{"green" if v["ma100"]>=0 else "red"}">${v["ma100"]:+,.0f}</td><td class="{c}"><b>${v["total"]:+,.0f}</b></td><td><span class="bar {bar_cls}" style="width:{bar_w}px"></span></td></tr>'
    html += """</table></div>"""

    # Spot DCA
    if spot_info:
        html += """<h2>Spot DCA Accumulation</h2><div class="card"><table>
<tr><th>Asset</th><th>Qty</th><th>Avg Cost</th><th>Spent</th><th>Value</th><th>PnL</th><th>Buys</th></tr>"""
        for s in spot_info:
            avg = s["cost"] / s["qty"] if s["qty"] > 0 else 0
            c = "green" if s["pnl"] >= 0 else "red"
            html += f'<tr><td>{s["asset"]}</td><td>{s["qty"]:.6f}</td><td>${avg:,.2f}</td><td>${s["cost"]:,.2f}</td><td>${s["value"]:,.2f}</td><td class="{c}">${s["pnl"]:+,.2f}</td><td>{s["count"]}</td></tr>'
        html += """</table></div>"""

    html += """
<div class="card" style="margin-top:30px; text-align:center; color:#64748b;">
<p>Generated by Williams Fractals Portfolio Validator</p>
</div>
</div></body></html>"""
    return html


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", type=float, default=6500)
    parser.add_argument("--sims", type=int, default=10000)
    args = parser.parse_args()

    full_start = datetime(2025, 1, 2)
    split_date = datetime(2025, 9, 1)
    full_end = datetime(2026, 3, 22)

    print(f"\n{'='*80}")
    print(f"  Combined Portfolio Validation")
    print(f"  Fractals(10x) + Mirror(5x) + MA100(3x) + DCA")
    print(f"{'='*80}\n")

    # Load data
    loader = DataLoader()
    data = load_all_data(loader, full_start, full_end)

    # Full period
    print("\n  Running FULL period...")
    t0 = time.time()
    full_r = run_combined(data, full_start, full_end, args.balance)
    print(f"  Done ({time.time()-t0:.0f}s) - {len([t for t in full_r['trades'] if t['reason'] != 'DCABuy'])} trades, Balance: ${full_r['balance']:,.0f}")

    # Train period
    print("  Running TRAIN period...")
    train_r = run_combined(data, full_start, split_date, args.balance)
    print(f"  Done - {len([t for t in train_r['trades'] if t['reason'] != 'DCABuy'])} trades")

    # Test period
    print("  Running TEST period...")
    test_r = run_combined(data, split_date, full_end, args.balance)
    print(f"  Done - {len([t for t in test_r['trades'] if t['reason'] != 'DCABuy'])} trades")

    # Monte Carlo on full period
    trade_pnls = [t["pnl"] for t in full_r["trades"] if t["reason"] != "DCABuy"]
    print(f"\n  Running Monte Carlo ({args.sims:,} sims)...")
    mc_df = monte_carlo(trade_pnls, args.balance, args.sims)

    # Leverage scenarios (just fractals leverage varies)
    mc_lev = {}
    for lev in [5, 10, 15, 20]:
        old_lev = FRACTALS_CFG["leverage"]
        FRACTALS_CFG["leverage"] = lev
        r = run_combined(data, full_start, full_end, args.balance)
        pnls = [t["pnl"] for t in r["trades"] if t["reason"] != "DCABuy"]
        mc_l = monte_carlo(pnls, args.balance, args.sims)
        mc_lev[lev] = mc_l
        FRACTALS_CFG["leverage"] = old_lev
        print(f"  Leverage {lev}x: ${r['balance']:,.0f} ({(r['balance']-args.balance)/args.balance*100:+.0f}%)")

    # Generate HTML
    print("\n  Generating HTML report...")
    html = generate_html(train_r, test_r, full_r, mc_df, mc_lev, args.balance)
    report_path = Path("data/test_reports/combined_portfolio_validation.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    print(f"  Report saved: {report_path}")

    # Print summary
    ft = [t for t in full_r["trades"] if t["reason"] != "DCABuy"]
    w = [t for t in ft if t["pnl"] > 0]
    gp = sum(t["pnl"] for t in w); gl = abs(sum(t["pnl"] for t in ft if t["pnl"] <= 0))
    pf = gp/gl if gl > 0 else 999
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Final Balance : ${full_r['balance']:,.0f} ({(full_r['balance']-args.balance)/args.balance*100:+.1f}%)")
    print(f"  Trades        : {len(ft)} | Win Rate: {len(w)/len(ft)*100:.1f}% | PF: {pf:.2f}")
    print(f"  MDD           : {full_r['max_dd_pct']:.1f}%")
    print(f"  MC P(Loss)    : {(mc_df['return_pct']<0).mean()*100:.1f}%")
    print(f"  MC Median Ret : {mc_df['return_pct'].median():+.1f}%")
    tt = [t for t in train_r["trades"] if t["reason"] != "DCABuy"]
    tw = [t for t in tt if t["pnl"] > 0]
    tgp = sum(t["pnl"] for t in tw); tgl = abs(sum(t["pnl"] for t in tt if t["pnl"] <= 0))
    train_pf = tgp/tgl if tgl > 0 else 999
    tet = [t for t in test_r["trades"] if t["reason"] != "DCABuy"]
    tew = [t for t in tet if t["pnl"] > 0]
    tegp = sum(t["pnl"] for t in tew); tegl = abs(sum(t["pnl"] for t in tet if t["pnl"] <= 0))
    test_pf = tegp/tegl if tegl > 0 else 999
    print(f"  Walk-Forward  : Train PF {train_pf:.2f} -> Test PF {test_pf:.2f}")
    print(f"  Report        : {report_path.absolute()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
