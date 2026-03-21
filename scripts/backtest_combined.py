"""
Combined Backtest: Ichimoku + Mirror Short + MA100 V2 + Spot DCA

4개 전략(이치모쿠 숏, 미러 숏, MA100 V2, 스팟 DCA)을 하나의 공유 잔고로 동시 운용하는 통합 백테스트.
전략 간 시너지/간섭 효과를 측정하기 위함.

전략별 설정:
  Ichimoku    : 4h  SHORT       20x  5%  max5  cloud-based SL  trail 1.5%
  Mirror Short: 5m  SHORT        5x  5%  max3  SL 1%           trail 3%/1.2%
  MA100 V2    : 1d  SHORT ONLY   3x  2%  max20 SL 5%           trail 3%/2%
  Spot DCA    : 8h  LONG BTC/ETH 1x  $10/회  BTC40% ETH60%   선물수익 보너스

사용법:
    python scripts/backtest_combined.py --balance 6500 --start 2025-01-02 --end 2026-03-18
    python scripts/backtest_combined.py --no-ichimoku   # 미러+MA100+DCA만
    python scripts/backtest_combined.py --no-mirror     # 이치모쿠+MA100+DCA만
    python scripts/backtest_combined.py --no-ma100      # 이치모쿠+미러+DCA만
    python scripts/backtest_combined.py --no-dca        # 이치모쿠+미러+MA100만 (3전략)
"""

import argparse
import json as _json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.ichimoku import calculate_ichimoku
from src.live_surge_mirror_short import (
    MirrorShortParams,
    schedule_next_candle_entries,
    simulate_short_exit_ohlc,
)
from src.strategy import MAJOR_COINS as ICHIMOKU_SYMBOLS, STABLECOINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/backtest_cache")


# ─── Strategy Configs ──────────────────────────────────────────

ICHIMOKU_CONFIG = {
    "leverage": 20,
    "position_pct": 0.05,
    "max_positions": 5,
    "max_margin": 5000,
    "min_cloud_thickness": 0.2,
    "min_sl_pct": 0.3,
    "max_sl_pct": 8.0,
    "sl_buffer": 0.2,
    "rr_ratio": 2.0,
    "trail_pct": 1.5,
    "max_loss_pct": 2.0,
    "cooldown_hours": 4,
    "use_btc_filter": True,
    "fee_rate": 0.00055,
}

MIRROR_CONFIG = {
    "leverage": 5,
    "position_pct": 0.05,
    "max_positions": 3,
    "max_margin": 5000,
    "roundtrip_cost_rate": 0.0009,
    "sl_pct": 1.0,
    "trail_start_pct": 3.0,
    "trail_rebound_pct": 1.2,
    "cooldown_candles": 3,
    "overheat_cum_rise_pct": 8.0,
    "overheat_upper_wick_pct": 40.0,
    "overheat_volume_ratio": 5.0,
    "volume_lookback": 20,
}

MA100_CONFIG = {
    "ma_period": 100,
    "slope_lookback": 3,
    "touch_buffer_pct": 1.0,
    "leverage": 3,
    "position_pct": 0.02,
    "max_positions": 20,
    "max_margin": 5000,
    "fee_rate": 0.00055,
    "sl_pct": 5.0,
    "tp_pct": 0,
    "trail_start_pct": 3.0,
    "trail_pct": 2.0,
    "cooldown_days": 3,
    # ── 분할매수 (DCA) ──
    "dca_ratios": [1, 1, 2],       # 1차:2차:3차 = 1:1:2
    "dca_interval_pct": 4.0,       # 숏: 진입가 위로 4% 간격
}

DCA_CONFIG = {
    "interval_hours": 8,       # DCA 주기
    "base_amount_usdt": 10.0,  # 기본 매수액
    "btc_ratio": 0.4,          # BTC 40%
    "eth_ratio": 0.6,          # ETH 60%
    "profit_bonus_pct": 0.10,  # 선물 수익의 10% 추가
    "min_futures_reserve": 500.0,  # 선물 마진 최소 유보
    "min_order_usdt": 5.0,     # 최소 주문
    "taker_fee": 0.001,        # 스팟 테이커 수수료 0.1%
}

STRATEGY_COLORS = {
    "ichimoku": "#a855f7",
    "mirror_short": "#ef4444",
    "ma100": "#3b82f6",
    "dca": "#22c55e",
}
STRATEGY_LABELS = {
    "ichimoku": "Ichimoku",
    "mirror_short": "Mirror Short",
    "ma100": "MA100 V2",
    "dca": "Spot DCA",
}


# 미러숏 제외 심볼 (실전과 동일: BTC, ETH 제외)
MIRROR_EXCLUDE = {'BTC/USDT:USDT', 'ETH/USDT:USDT'}


def _is_stablecoin(symbol: str) -> bool:
    """심볼이 스테이블코인인지 확인."""
    base = symbol.split('/')[0] if '/' in symbol else symbol
    return base in STABLECOINS


# ─── Data Loading ──────────────────────────────────────────────

def load_5m_data(
    loader: DataLoader, start_dt: datetime, end_dt: datetime,
) -> Dict[str, pd.DataFrame]:
    """5m 데이터 로드 (로컬 parquet + API 캐시). 스테이블코인 제외."""
    symbols = [s for s in loader.get_available_symbols() if not _is_stablecoin(s)]
    all_data: Dict[str, pd.DataFrame] = {}
    margin = timedelta(hours=3)

    for i, symbol in enumerate(symbols):
        parts = []
        local_df = loader.load(symbol, "5m")
        if local_df is not None and len(local_df) > 0:
            parts.append(local_df)
        cache_path = CACHE_DIR / f"{loader._clean_symbol(symbol)}_5m.parquet"
        if cache_path.exists():
            try:
                api_df = pd.read_parquet(cache_path)
                if api_df is not None and len(api_df) > 0:
                    parts.append(api_df)
            except Exception:
                pass
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        df = df[(df["timestamp"] >= start_dt - margin) & (df["timestamp"] <= end_dt)].reset_index(drop=True)
        if len(df) >= 30:
            in_range = df[(df["timestamp"] >= pd.Timestamp(start_dt)) & (df["timestamp"] <= pd.Timestamp(end_dt))]
            if len(in_range) >= 30:
                all_data[symbol] = df
        if (i + 1) % 100 == 0 or (i + 1) == len(symbols):
            logger.info(f"5m 데이터 로드: {i+1}/{len(symbols)} (유효: {len(all_data)})")
    return all_data


def load_4h_data(
    loader: DataLoader, start_dt: datetime, end_dt: datetime,
) -> Dict[str, pd.DataFrame]:
    """4h 데이터 로드 (이치모쿠용, 실전 운용 심볼만 + BTC 포함)."""
    # 실전과 동일하게 ICHIMOKU_SYMBOLS (20개) + BTC(트렌드 필터용)만 로드
    ichi_set = set(ICHIMOKU_SYMBOLS)
    ichi_set.add("BTC/USDT:USDT")  # BTC 트렌드 필터용
    available = set(loader.get_available_symbols())
    symbols = sorted(ichi_set & available)

    all_data: Dict[str, pd.DataFrame] = {}
    warmup = timedelta(days=80)
    start_str = (start_dt - warmup).strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    for symbol in symbols:
        tfs = loader.get_available_timeframes(symbol)
        if "4h" not in tfs:
            continue
        df = loader.load(symbol, "4h", start=start_str, end=end_str)
        if df is not None and len(df) >= 60:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            all_data[symbol] = df
    logger.info(f"  4h (이치모쿠): {len(all_data)}/{len(symbols)} 심볼 로드")
    return all_data


def load_1d_data(
    loader: DataLoader, start_dt: datetime, end_dt: datetime,
) -> Dict[str, pd.DataFrame]:
    """1d 데이터 로드 (MA100용, 4h→1d 리샘플링 포함). 스테이블코인 제외."""
    symbols = [s for s in loader.get_available_symbols() if not _is_stablecoin(s)]
    all_data: Dict[str, pd.DataFrame] = {}
    loaded_1d = 0
    resampled_4h = 0
    data_start = start_dt - timedelta(days=150)
    start_str = data_start.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    for i, symbol in enumerate(symbols):
        tfs = loader.get_available_timeframes(symbol)
        df = None
        if "1d" in tfs:
            df = loader.load(symbol, "1d", start=start_str, end=end_str)
            if df is not None and len(df) >= 100:
                loaded_1d += 1
            else:
                df = None
        if df is None and "4h" in tfs:
            raw = loader.load(symbol, "4h", start=start_str, end=end_str)
            if raw is not None and len(raw) >= 600:
                df = _resample_4h_to_1d(raw)
                if df is not None and len(df) >= 100:
                    resampled_4h += 1
                else:
                    df = None
        if df is not None:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            all_data[symbol] = df
        if (i + 1) % 100 == 0 or (i + 1) == len(symbols):
            logger.info(f"1d 데이터 로드: {i+1}/{len(symbols)} (1d: {loaded_1d}, 4h→1d: {resampled_4h})")
    return all_data


def _resample_4h_to_1d(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    daily = df.resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    if len(daily) < 100:
        return None
    return daily.reset_index()


def load_1h_btc(
    loader: DataLoader, start_dt: datetime, end_dt: datetime,
) -> Optional[pd.DataFrame]:
    """BTC 1h 데이터 로드 (그리드봇용)."""
    warmup = timedelta(days=20)  # ATR 14 계산 워밍업
    start_str = (start_dt - warmup).strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    symbol = "BTC/USDT:USDT"

    tfs = loader.get_available_timeframes(symbol)
    df = None
    if "1h" in tfs:
        df = loader.load(symbol, "1h", start=start_str, end=end_str)

    # 1h 로컬 없으면 5m→1h 리샘플
    if df is None or len(df) < 100:
        logger.info("  BTC 1h 없음 → 5m에서 리샘플링")
        df_5m = loader.load(symbol, "5m", start=start_str, end=end_str)
        if df_5m is not None and len(df_5m) >= 100:
            df_5m = df_5m.copy()
            df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"])
            df_5m = df_5m.set_index("timestamp")
            df = df_5m.resample("1h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna().reset_index()
        else:
            return None

    if df is not None and len(df) >= 100:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
        return df
    return None


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR (Average True Range) 계산."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def precompute_grid_levels(btc_1h: pd.DataFrame, config: dict) -> pd.DataFrame:
    """ATR 기반 그리드 레벨 사전 계산."""
    df = btc_1h.copy()
    df["atr"] = _compute_atr(df, config["atr_period"])
    df["grid_mid"] = df["close"]
    df["grid_upper"] = df["close"] + df["atr"] * config["atr_multiplier"]
    df["grid_lower"] = df["close"] - df["atr"] * config["atr_multiplier"]
    return df


# ─── Signal Precomputation ─────────────────────────────────────

def precompute_ichimoku_signals(
    data_4h: Dict[str, pd.DataFrame], config: dict,
) -> Dict[str, pd.DataFrame]:
    """이치모쿠 시그널 사전 계산."""
    result = {}
    for symbol, df in data_4h.items():
        df = calculate_ichimoku(df)
        df = df.dropna(subset=["tenkan", "kijun", "cloud_top", "cloud_bottom"])
        if len(df) < 10:
            continue
        # Entry signal: below cloud + not tenkan_above + (TK cross down or kijun cross down)
        # + cloud thickness >= min
        df["ichimoku_entry"] = (
            df["below_cloud"]
            & ~df["tenkan_above"]
            & (df["tk_cross_down"] | df["kijun_cross_down"])
            & (df["cloud_thickness"] >= config["min_cloud_thickness"])
            & ~df["in_cloud"]
        ).fillna(False)

        # Score for ranking candidates
        df["ichi_score"] = (
            df["chikou_bearish"].astype(int) * 2
            + (~df["cloud_green"]).astype(int)
            + (df["cloud_thickness"] > 1.0).astype(int)
        )
        result[symbol] = df.reset_index(drop=True)
    return result


def compute_btc_trend(data_4h: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, bool]:
    """BTC 4h SMA26/SMA52 기반 트렌드 필터."""
    btc_sym = "BTC/USDT:USDT"
    if btc_sym not in data_4h:
        return {}
    btc_df = data_4h[btc_sym].copy()
    btc_df["sma_26"] = btc_df["close"].rolling(26).mean()
    btc_df["sma_52"] = btc_df["close"].rolling(52).mean()
    trend = {}
    for _, row in btc_df.iterrows():
        if pd.notna(row["sma_26"]) and pd.notna(row["sma_52"]):
            trend[row["timestamp"]] = bool(row["sma_26"] > row["sma_52"])
    return trend


def precompute_mirror_signals(
    data_5m: Dict[str, pd.DataFrame], config: dict,
) -> Dict[str, pd.DataFrame]:
    """미러 숏 진입 시그널 계산. BTC/ETH 제외 (실전과 동일)."""
    params = MirrorShortParams(
        overheat_cum_rise_pct=config["overheat_cum_rise_pct"],
        overheat_upper_wick_pct=config["overheat_upper_wick_pct"],
        overheat_volume_ratio=config["overheat_volume_ratio"],
        volume_lookback=config["volume_lookback"],
        stop_loss_pct=config["sl_pct"],
        trail_start_pct=config["trail_start_pct"],
        trail_rebound_pct=config["trail_rebound_pct"],
    )
    result = {}
    for symbol, df in data_5m.items():
        if symbol in MIRROR_EXCLUDE:
            continue
        df = df.copy()
        base_signal = _build_base_surge_signal(df)
        overheat = _build_overheat_mask(df, params)
        entry_signal = schedule_next_candle_entries(base_signal, overheat, delay_candles=1)
        df["mirror_entry_signal"] = entry_signal
        result[symbol] = df
    return result


def _build_base_surge_signal(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    work["volume_sma"] = work["volume"].rolling(20).mean()
    work["volume_ratio"] = work["volume"] / work["volume_sma"]
    work["change_pct"] = work["close"].pct_change() * 100.0
    work["is_green"] = work["close"] > work["open"]
    range_high = work["high"].rolling(12).max().shift(1)
    range_low = work["low"].rolling(12).min().shift(1)
    work["consol_range_pct"] = (range_high - range_low) / range_low * 100.0
    work["price_from_low"] = (work["close"] - work["low"].shift(1)) / work["low"].shift(1) * 100.0
    signal = (
        (work["volume_ratio"] >= 10.0)
        & (work["change_pct"] >= 5.0)
        & work["is_green"]
        & (work["consol_range_pct"] <= 5.0)
        & (work["price_from_low"] <= 15.0)
    )
    return signal.fillna(False)


def _build_overheat_mask(df: pd.DataFrame, params: MirrorShortParams) -> pd.Series:
    close = df["close"]
    cum_rise = (close / close.shift(3) - 1.0) * 100.0
    candle_range = (df["high"] - df["low"]).clip(lower=1e-12)
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0.0)
    upper_wick_pct = upper_wick / candle_range * 100.0
    vol_avg = df["volume"].shift(1).rolling(params.volume_lookback).mean()
    vol_ratio = df["volume"] / vol_avg
    return (
        (cum_rise >= params.overheat_cum_rise_pct)
        | (upper_wick_pct >= params.overheat_upper_wick_pct)
        | (vol_ratio >= params.overheat_volume_ratio)
    ).fillna(False)


def precompute_ma100_signals(
    data_1d: Dict[str, pd.DataFrame], config: dict,
) -> Dict[str, pd.DataFrame]:
    """MA100 터치 반등 시그널 벡터화 계산 (1d 기준)."""
    ma_period = config["ma_period"]
    slope_lookback = config["slope_lookback"]
    touch_buf = config["touch_buffer_pct"] / 100
    result = {}
    for symbol, df in data_1d.items():
        df = df.copy()
        df["ma100"] = df["close"].rolling(ma_period).mean()
        df["slope"] = (
            (df["ma100"] - df["ma100"].shift(slope_lookback))
            / df["ma100"].shift(slope_lookback) * 100
        )
        df["long_signal"] = False  # SHORT ONLY
        df["short_signal"] = (
            (df["slope"] < 0)
            & (df["high"] >= df["ma100"] * (1 - touch_buf))
            & (df["close"] < df["ma100"])
        ).fillna(False)
        result[symbol] = df
    return result


def load_4h_all(
    loader: DataLoader, data_5m: Dict[str, pd.DataFrame],
    start_dt: datetime, end_dt: datetime,
) -> Dict[str, pd.DataFrame]:
    """4h 데이터 로드 (MA100 4h용, 모든 심볼). 로컬 4h 또는 5m→4h 리샘플."""
    symbols = [s for s in loader.get_available_symbols() if not _is_stablecoin(s)]
    all_data: Dict[str, pd.DataFrame] = {}
    loaded_native = 0
    resampled_5m = 0
    warmup = timedelta(days=150)
    start_str = (start_dt - warmup).strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    for i, symbol in enumerate(symbols):
        df = None
        tfs = loader.get_available_timeframes(symbol)
        if "4h" in tfs:
            df = loader.load(symbol, "4h", start=start_str, end=end_str)
            if df is not None and len(df) >= 600:
                loaded_native += 1
            else:
                df = None
        if df is None and symbol in data_5m:
            raw = data_5m[symbol]
            raw = raw.copy()
            raw["timestamp"] = pd.to_datetime(raw["timestamp"])
            # 워밍업 포함 필터
            raw = raw[raw["timestamp"] >= pd.Timestamp(start_dt - warmup)]
            if len(raw) >= 100:
                resampled = raw.set_index("timestamp").resample("4h").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna().reset_index()
                if len(resampled) >= 150:
                    df = resampled
                    resampled_5m += 1
        if df is not None:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            all_data[symbol] = df
        if (i + 1) % 100 == 0 or (i + 1) == len(symbols):
            logger.info(f"  4h(MA100) 로드: {i+1}/{len(symbols)} (native: {loaded_native}, 5m→4h: {resampled_5m})")
    return all_data


def load_1h_all(
    loader: DataLoader, data_5m: Dict[str, pd.DataFrame],
    start_dt: datetime, end_dt: datetime,
    check_hours: tuple = (22, 23, 0),
) -> Dict[str, pd.DataFrame]:
    """1h 데이터 로드 (MA100 hourly 체크용). 특정 시간대만 필터링."""
    symbols = [s for s in loader.get_available_symbols() if not _is_stablecoin(s)]
    all_data: Dict[str, pd.DataFrame] = {}
    resampled = 0
    warmup = timedelta(days=150)

    for i, symbol in enumerate(symbols):
        if symbol not in data_5m:
            continue
        raw = data_5m[symbol].copy()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        raw = raw[raw["timestamp"] >= pd.Timestamp(start_dt - warmup)]
        if len(raw) < 100:
            continue
        df_1h = raw.set_index("timestamp").resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna().reset_index()
        # 특정 시간대만 필터
        df_1h = df_1h[df_1h["timestamp"].dt.hour.isin(check_hours)].reset_index(drop=True)
        if len(df_1h) >= 50:
            all_data[symbol] = df_1h
            resampled += 1
        if (i + 1) % 100 == 0 or (i + 1) == len(symbols):
            logger.info(f"  1h(MA100) 리샘플: {i+1}/{len(symbols)} (유효: {resampled})")
    return all_data


def precompute_ma100_signals_hourly(
    data_1d: Dict[str, pd.DataFrame],
    data_1h_all: Dict[str, pd.DataFrame],
    config: dict,
) -> Dict[str, pd.DataFrame]:
    """MA100 터치 시그널 - 일봉 MA100 값을 특정 시간 1h 캔들에서 터치 감지.

    일봉에서 MA100과 slope를 계산하고, 22:00/23:00/00:00 UTC 1h 캔들에
    forward-fill하여 체크. 하루 3회 체크 (7시/8시/9시 KST).
    """
    ma_period = config["ma_period"]
    slope_lookback = config["slope_lookback"]
    touch_buf = config["touch_buffer_pct"] / 100
    result = {}

    for symbol in data_1h_all:
        if symbol not in data_1d:
            continue
        df_1d = data_1d[symbol].copy()
        df_1h = data_1h_all[symbol].copy()

        # 일봉에서 MA100, slope 계산
        df_1d["ma100"] = df_1d["close"].rolling(ma_period).mean()
        df_1d["slope"] = (
            (df_1d["ma100"] - df_1d["ma100"].shift(slope_lookback))
            / df_1d["ma100"].shift(slope_lookback) * 100
        )
        df_1d["timestamp"] = pd.to_datetime(df_1d["timestamp"])
        df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"])

        # 일봉 날짜를 1h 타임스탬프에 매핑 (forward-fill)
        daily_vals = df_1d[["timestamp", "ma100", "slope"]].dropna().copy()
        daily_vals = daily_vals.rename(columns={"timestamp": "date"})
        daily_vals["date"] = daily_vals["date"].dt.normalize()

        df_1h["date"] = df_1h["timestamp"].dt.normalize()
        df_1h = df_1h.merge(daily_vals, on="date", how="left", suffixes=("", "_daily"))
        df_1h["ma100"] = df_1h["ma100"].ffill()
        df_1h["slope"] = df_1h["slope"].ffill()

        df_1h["long_signal"] = False  # SHORT ONLY
        df_1h["short_signal"] = (
            (df_1h["slope"] < 0)
            & (df_1h["high"] >= df_1h["ma100"] * (1 - touch_buf))
            & (df_1h["close"] < df_1h["ma100"])
        ).fillna(False)

        # 같은 날 중복 시그널 방지: 같은 날 첫 시그널만 유지
        if df_1h["short_signal"].any():
            seen_dates: Set[str] = set()
            keep = []
            for _, row in df_1h.iterrows():
                if row["short_signal"]:
                    d = str(row["date"])
                    if d not in seen_dates:
                        seen_dates.add(d)
                        keep.append(True)
                    else:
                        keep.append(False)
                else:
                    keep.append(False)
            df_1h["short_signal"] = keep

        df_1h.drop(columns=["date"], inplace=True, errors="ignore")
        result[symbol] = df_1h.reset_index(drop=True)
    return result


def precompute_ma100_signals_4h(
    data_1d: Dict[str, pd.DataFrame],
    data_4h_all: Dict[str, pd.DataFrame],
    config: dict,
) -> Dict[str, pd.DataFrame]:
    """MA100 터치 시그널 - 일봉 MA100 값을 4h 캔들에서 터치 감지.

    일봉에서 MA100과 slope를 계산하고, 4h 캔들에 forward-fill하여
    4h 단위로 터치 조건을 체크. 6배 빠른 진입 감지.
    """
    ma_period = config["ma_period"]
    slope_lookback = config["slope_lookback"]
    touch_buf = config["touch_buffer_pct"] / 100
    result = {}

    for symbol in data_4h_all:
        if symbol not in data_1d:
            continue
        df_1d = data_1d[symbol].copy()
        df_4h = data_4h_all[symbol].copy()

        # 일봉에서 MA100, slope 계산
        df_1d["ma100"] = df_1d["close"].rolling(ma_period).mean()
        df_1d["slope"] = (
            (df_1d["ma100"] - df_1d["ma100"].shift(slope_lookback))
            / df_1d["ma100"].shift(slope_lookback) * 100
        )
        df_1d["timestamp"] = pd.to_datetime(df_1d["timestamp"])
        df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"])

        # 일봉 날짜를 4h 타임스탬프에 매핑 (forward-fill)
        daily_vals = df_1d[["timestamp", "ma100", "slope"]].dropna().copy()
        daily_vals = daily_vals.rename(columns={"timestamp": "date"})
        daily_vals["date"] = daily_vals["date"].dt.normalize()

        df_4h["date"] = df_4h["timestamp"].dt.normalize()
        df_4h = df_4h.merge(daily_vals, on="date", how="left", suffixes=("", "_daily"))
        # forward-fill: 아직 일봉이 안 나온 4h 캔들은 전날 MA100 사용
        df_4h["ma100"] = df_4h["ma100"].ffill()
        df_4h["slope"] = df_4h["slope"].ffill()

        df_4h["long_signal"] = False  # SHORT ONLY
        df_4h["short_signal"] = (
            (df_4h["slope"] < 0)
            & (df_4h["high"] >= df_4h["ma100"] * (1 - touch_buf))
            & (df_4h["close"] < df_4h["ma100"])
        ).fillna(False)

        # 같은 날 중복 시그널 방지: 같은 날 첫 4h 시그널만 유지
        if df_4h["short_signal"].any():
            df_4h["_sig_date"] = df_4h["timestamp"].dt.normalize()
            first_sig_mask = df_4h["short_signal"] & ~df_4h["short_signal"].shift(1, fill_value=False)
            # 같은 날 내에서 첫 시그널만
            seen_dates: Set[str] = set()
            keep = []
            for idx, row in df_4h.iterrows():
                if row["short_signal"]:
                    d = str(row["_sig_date"])
                    if d not in seen_dates:
                        seen_dates.add(d)
                        keep.append(True)
                    else:
                        keep.append(False)
                else:
                    keep.append(row["short_signal"])
            df_4h["short_signal"] = keep
            df_4h.drop(columns=["_sig_date"], inplace=True)

        df_4h.drop(columns=["date"], inplace=True, errors="ignore")
        result[symbol] = df_4h.reset_index(drop=True)
    return result


# ─── CombinedBacktester ───────────────────────────────────────

class CombinedBacktester:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        enable_ichimoku: bool = True,
        enable_mirror: bool = True,
        enable_ma100: bool = True,
        enable_dca: bool = True,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.enable_ichimoku = enable_ichimoku
        self.enable_mirror = enable_mirror
        self.enable_ma100 = enable_ma100
        self.enable_dca = enable_dca

        self.ichimoku_config = ICHIMOKU_CONFIG.copy()
        self.mirror_config = MIRROR_CONFIG.copy()
        self.ma100_config = MA100_CONFIG.copy()
        self.dca_config = DCA_CONFIG.copy()

        self.positions: Dict[str, Dict[str, dict]] = {
            "ichimoku": {},
            "mirror_short": {},
            "ma100": {},
        }
        self.last_exit_times: Dict[str, Dict[str, pd.Timestamp]] = {
            "ichimoku": {},
            "mirror_short": {},
            "ma100": {},
        }

        self.trades: List[dict] = []
        self.equity_curve: List[dict] = []
        self._current_prices: Dict[str, float] = {}

        self.peak_equity = initial_balance
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0

        # Spot DCA state
        self.dca_last_time: Optional[pd.Timestamp] = None
        self.dca_last_pnl_idx: int = 0  # trades 리스트에서 마지막 체크 인덱스
        self.spot_holdings: Dict[str, dict] = {
            "BTC": {"qty": 0.0, "cost": 0.0, "count": 0},
            "ETH": {"qty": 0.0, "cost": 0.0, "count": 0},
        }
        self._spot_last_prices: Dict[str, float] = {}  # BTC/ETH 최종 시세

    def _get_config(self, strategy: str) -> dict:
        if strategy == "ichimoku":
            return self.ichimoku_config
        elif strategy == "mirror_short":
            return self.mirror_config
        elif strategy == "dca":
            return self.dca_config
        return self.ma100_config

    def _calc_qty(self, strategy: str, price: float) -> tuple:
        """Returns (qty, margin). margin is capped by max_margin."""
        config = self._get_config(strategy)
        margin = self.balance * config["position_pct"]
        max_margin = config.get("max_margin")
        if max_margin and margin > max_margin:
            margin = max_margin
        return margin * config["leverage"] / price, margin

    def _close_position(
        self, strategy: str, symbol: str,
        exit_price: float, reason: str, exit_time: pd.Timestamp,
    ):
        pos = self.positions[strategy].pop(symbol)
        config = self._get_config(strategy)
        entry = pos["entry_price"]
        qty = pos["qty"]
        side = pos["side"]
        leverage = config["leverage"]

        if side == "long":
            pnl_pct = (exit_price - entry) / entry * 100 * leverage
            pnl_usd = (exit_price - entry) * qty
        else:
            pnl_pct = (entry - exit_price) / entry * 100 * leverage
            pnl_usd = (entry - exit_price) * qty

        if strategy == "mirror_short":
            fee_usd = entry * qty * config["roundtrip_cost_rate"]
        else:
            fee_rate = config.get("fee_rate", 0)
            fee_usd = (qty * entry * fee_rate + qty * exit_price * fee_rate) if fee_rate > 0 else 0.0

        net_pnl = pnl_usd - fee_usd
        self.balance += net_pnl
        self.last_exit_times[strategy][symbol] = exit_time

        self.trades.append({
            "symbol": symbol,
            "strategy": strategy,
            "side": side,
            "entry_price": entry,
            "exit_price": exit_price,
            "entry_time": pos["entry_time"],
            "exit_time": exit_time,
            "pnl_pct": pnl_pct,
            "pnl_usd": net_pnl,
            "fee_usd": fee_usd,
            "reason": reason,
            "signal_type": pos.get("signal_type", ""),
        })

    def _update_equity(self, dt: pd.Timestamp):
        unrealized = 0.0
        for strategy, pos_dict in self.positions.items():
            for sym, pos in pos_dict.items():
                cp = self._current_prices.get(sym)
                if cp is None:
                    continue
                if pos["side"] == "long":
                    unrealized += (cp - pos["entry_price"]) * pos["qty"]
                else:
                    unrealized += (pos["entry_price"] - cp) * pos["qty"]

        # 스팟 보유 가치 포함 (BTC + ETH)
        spot_value = 0.0
        for asset, h in self.spot_holdings.items():
            price = self._spot_last_prices.get(asset, 0)
            spot_value += h["qty"] * price

        equity = self.balance + unrealized + spot_value
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = self.peak_equity - equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_drawdown_pct = dd / self.peak_equity * 100
        self.equity_curve.append({"timestamp": dt, "equity": equity})

    # ─── Ichimoku Exit ─────────────────────────────────

    def _check_ichimoku_exit(self, symbol: str, candle: pd.Series, dt: pd.Timestamp):
        """이치모쿠 숏 포지션 청산 (MaxLoss/SL/TP/Trail/Cloud)."""
        pos = self.positions["ichimoku"].get(symbol)
        if pos is None:
            return

        config = self.ichimoku_config
        entry = pos["entry_price"]
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        # Update lowest and trailing state
        if low < pos["lowest"]:
            pos["lowest"] = low
            if low <= pos["take_profit"]:
                pos["trailing"] = True
                new_trail = low * (1 + config["trail_pct"] / 100)
                pos["trail_stop"] = min(pos["trail_stop"], new_trail)

        # 1. MaxLoss (hard cap)
        max_loss_price = entry * (1 + config["max_loss_pct"] / 100)
        if high >= max_loss_price:
            self._close_position("ichimoku", symbol, max_loss_price, "MaxLoss", dt)
            return

        # 2. Stop Loss
        if high >= pos["stop_loss"]:
            self._close_position("ichimoku", symbol, pos["stop_loss"], "SL", dt)
            return

        # 3. Trailing Stop
        if pos.get("trailing") and high >= pos["trail_stop"]:
            self._close_position("ichimoku", symbol, pos["trail_stop"], "Trail", dt)
            return

        # 4. Take Profit (before trailing activates)
        if not pos.get("trailing") and low <= pos["take_profit"]:
            self._close_position("ichimoku", symbol, pos["take_profit"], "TP", dt)
            return

        # 5. Cloud reversal (price back in or above cloud)
        if candle.get("in_cloud", False) or candle.get("above_cloud", False):
            self._close_position("ichimoku", symbol, close, "Cloud", dt)
            return

    # ─── Mirror Exit ───────────────────────────────────

    def _check_mirror_exit(self, symbol: str, candle: pd.Series, dt: pd.Timestamp):
        """미러 숏 포지션 청산."""
        pos = self.positions["mirror_short"].get(symbol)
        if pos is None:
            return
        params = MirrorShortParams(
            stop_loss_pct=self.mirror_config["sl_pct"],
            trail_start_pct=self.mirror_config["trail_start_pct"],
            trail_rebound_pct=self.mirror_config["trail_rebound_pct"],
        )
        candle_dict = {
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
        }
        exit_info, updated = simulate_short_exit_ohlc(pos, candle_dict, params)
        self.positions["mirror_short"][symbol] = updated
        if exit_info is not None:
            self._close_position(
                "mirror_short", symbol, float(exit_info["price"]),
                exit_info["reason"], dt,
            )

    # ─── MA100 Exit ────────────────────────────────────

    def _process_ma100_dca(self, pos: dict, candle: pd.Series):
        """MA100 분할매수(DCA) 체결 처리. 가격 도달 시 추가 진입."""
        pending = pos.get("pending_dca")
        if not pending:
            return

        side = pos["side"]
        high = float(candle["high"])
        low = float(candle["low"])
        filled_new = []
        remaining = []

        for dca in pending:
            # 숏: 가격이 DCA 목표가 이상으로 올라가면 체결
            # 롱: 가격이 DCA 목표가 이하로 내려가면 체결
            if side == "short" and high >= dca["price"]:
                filled_new.append(dca)
            elif side == "long" and low <= dca["price"]:
                filled_new.append(dca)
            else:
                remaining.append(dca)

        if not filled_new:
            return

        entries = pos["filled_entries"]
        for dca in filled_new:
            entries.append({"price": dca["price"], "size": dca["size"]})

        # 평균단가 재계산
        total_size = sum(e["size"] for e in entries)
        avg_price = sum(e["price"] * e["size"] for e in entries) / total_size

        pos["entry_price"] = avg_price
        pos["qty"] = total_size
        pos["pending_dca"] = remaining

        # SL은 처음부터 전체 DCA 평균단가 기준으로 설정됨 → 변경 불필요

    def _check_ma100_exit(self, symbol: str, candle: pd.Series, dt: pd.Timestamp):
        """MA100 포지션 청산 (DCA → SL/TP/Reversal/Trail)."""
        pos = self.positions["ma100"].get(symbol)
        if pos is None:
            return

        # DCA 체결 먼저 처리
        self._process_ma100_dca(pos, candle)

        entry = pos["entry_price"]
        sl = pos["stop_loss"]
        tp = pos["take_profit"]
        side = pos["side"]
        config = self.ma100_config
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        open_p = float(candle["open"])
        is_green = close >= open_p
        use_fixed_tp = tp > 0

        # SL/TP (candle path based)
        if side == "long":
            if is_green:
                if low <= sl:
                    self._close_position("ma100", symbol, sl, "SL", dt); return
                if use_fixed_tp and high >= tp:
                    self._close_position("ma100", symbol, tp, "TP", dt); return
            else:
                if use_fixed_tp and high >= tp:
                    self._close_position("ma100", symbol, tp, "TP", dt); return
                if low <= sl:
                    self._close_position("ma100", symbol, sl, "SL", dt); return
        else:
            if not is_green:
                if high >= sl:
                    self._close_position("ma100", symbol, sl, "SL", dt); return
                if use_fixed_tp and low <= tp:
                    self._close_position("ma100", symbol, tp, "TP", dt); return
            else:
                if use_fixed_tp and low <= tp:
                    self._close_position("ma100", symbol, tp, "TP", dt); return
                if high >= sl:
                    self._close_position("ma100", symbol, sl, "SL", dt); return

        # Signal reversal
        if side == "long" and candle.get("short_signal", False):
            self._close_position("ma100", symbol, close, "Reversal", dt); return
        if side == "short" and candle.get("long_signal", False):
            self._close_position("ma100", symbol, close, "Reversal", dt); return

        # Trailing stop
        if side == "long":
            cur_pnl = (close - entry) / entry * 100
        else:
            cur_pnl = (entry - close) / entry * 100

        trail_start = config["trail_start_pct"]
        trail_pct = config["trail_pct"]

        if cur_pnl >= trail_start:
            pos["trailing"] = True
            if side == "long":
                if high > pos["highest"]:
                    pos["highest"] = high
                    pos["trail_stop"] = high * (1 - trail_pct / 100)
                if close <= pos["trail_stop"]:
                    self._close_position("ma100", symbol, pos["trail_stop"], "Trail", dt); return
            else:
                if low < pos["lowest"]:
                    pos["lowest"] = low
                    pos["trail_stop"] = low * (1 + trail_pct / 100)
                if close >= pos["trail_stop"]:
                    self._close_position("ma100", symbol, pos["trail_stop"], "Trail", dt); return
        elif pos.get("trailing"):
            if side == "long" and close <= pos["trail_stop"]:
                self._close_position("ma100", symbol, pos["trail_stop"], "Trail", dt); return
            if side == "short" and close >= pos["trail_stop"]:
                self._close_position("ma100", symbol, pos["trail_stop"], "Trail", dt); return

    # ─── Spot DCA ──────────────────────────────────────

    def _process_dca(self, ts: pd.Timestamp, btc_price: float, eth_price: float):
        """DCA 1사이클 실행: 잔고 확인 → 보너스 계산 → BTC/ETH 매수."""
        config = self.dca_config
        interval = timedelta(hours=config["interval_hours"])

        # 주기 체크
        if self.dca_last_time is not None and ts < self.dca_last_time + interval:
            return

        # 기본 매수액
        base = config["base_amount_usdt"]

        # 선물 수익 보너스: 마지막 DCA 이후 실현손익
        bonus = 0.0
        futures_pnl = 0.0
        for t in self.trades[self.dca_last_pnl_idx:]:
            if t["strategy"] in ("ichimoku", "mirror_short", "ma100"):
                futures_pnl += t["pnl_usd"]
        if futures_pnl > 0:
            bonus = futures_pnl * config["profit_bonus_pct"]

        total_amount = base + bonus
        self.dca_last_pnl_idx = len(self.trades)

        # 잔고 확인 (유보액 고려)
        available = self.balance - config["min_futures_reserve"]
        if available < total_amount:
            # 잔고 부족 시 스킵
            self.dca_last_time = ts
            return

        # BTC/ETH 각각 매수
        for asset, ratio, price in [
            ("BTC", config["btc_ratio"], btc_price),
            ("ETH", config["eth_ratio"], eth_price),
        ]:
            if price <= 0:
                continue
            usdt_amount = total_amount * ratio
            if usdt_amount < config["min_order_usdt"]:
                continue

            fee = usdt_amount * config["taker_fee"]
            cost = usdt_amount + fee
            if cost > self.balance - config["min_futures_reserve"]:
                continue

            qty = usdt_amount / price
            self.balance -= cost
            self.spot_holdings[asset]["qty"] += qty
            self.spot_holdings[asset]["cost"] += cost
            self.spot_holdings[asset]["count"] += 1

            self.trades.append({
                "symbol": f"{asset}/USDT",
                "strategy": "dca",
                "side": "long",
                "entry_price": price,
                "exit_price": price,
                "entry_time": ts,
                "exit_time": ts,
                "pnl_pct": 0.0,
                "pnl_usd": -fee,
                "fee_usd": fee,
                "reason": "DCABuy",
                "signal_type": "Accumulate",
            })

        self.dca_last_time = ts

    # ─── Main Loop ─────────────────────────────────────

    def run(
        self,
        data_5m: Dict[str, pd.DataFrame],
        data_4h: Dict[str, pd.DataFrame],
        data_1d: Dict[str, pd.DataFrame],
        start_dt: datetime,
        end_dt: datetime,
        data_1h_btc: Optional[pd.DataFrame] = None,
        data_4h_all: Optional[Dict[str, pd.DataFrame]] = None,
        data_1h_ma100: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        """통합 백테스트 실행."""
        start_ts = pd.Timestamp(start_dt)
        end_ts = pd.Timestamp(end_dt)

        # ── 1. Precompute signals ──
        ichi_pc: Dict[str, pd.DataFrame] = {}
        btc_trend: Dict[pd.Timestamp, bool] = {}
        mirror_pc: Dict[str, pd.DataFrame] = {}
        ma100_pc: Dict[str, pd.DataFrame] = {}

        if self.enable_ichimoku:
            logger.info("이치모쿠 시그널 사전 계산...")
            ichi_pc = precompute_ichimoku_signals(data_4h, self.ichimoku_config)
            if self.ichimoku_config["use_btc_filter"]:
                btc_trend = compute_btc_trend(data_4h)
            logger.info(f"  이치모쿠: {len(ichi_pc)} 심볼 준비")

        if self.enable_mirror:
            logger.info("미러 숏 시그널 사전 계산...")
            mirror_pc = precompute_mirror_signals(data_5m, self.mirror_config)
            logger.info(f"  미러: {len(mirror_pc)} 심볼 준비")

        ma100_use_sub = False  # True when using sub-daily (4h or hourly) mode
        if self.enable_ma100:
            if data_1h_ma100:
                logger.info("MA100 시그널 사전 계산 (1h 7/8/9시 KST 터치 감지)...")
                ma100_pc = precompute_ma100_signals_hourly(data_1d, data_1h_ma100, self.ma100_config)
                ma100_use_sub = True
                logger.info(f"  MA100 (hourly): {len(ma100_pc)} 심볼 준비")
            elif data_4h_all:
                logger.info("MA100 시그널 사전 계산 (4h 터치 감지)...")
                ma100_pc = precompute_ma100_signals_4h(data_1d, data_4h_all, self.ma100_config)
                ma100_use_sub = True
                logger.info(f"  MA100 (4h): {len(ma100_pc)} 심볼 준비")
            else:
                logger.info("MA100 시그널 사전 계산 (1d)...")
                ma100_pc = precompute_ma100_signals(data_1d, self.ma100_config)
                logger.info(f"  MA100 (1d): {len(ma100_pc)} 심볼 준비")

        # DCA용 BTC/ETH 1h 가격 맵 구성
        dca_btc_prices: Dict[pd.Timestamp, float] = {}
        dca_eth_prices: Dict[pd.Timestamp, float] = {}
        if self.enable_dca and data_1h_btc is not None:
            for _, row in data_1h_btc.iterrows():
                dca_btc_prices[row["timestamp"]] = float(row["close"])
            # ETH 1h: 5m에서 리샘플 또는 1d에서 가져오기
            eth_sym = "ETH/USDT:USDT"
            if eth_sym in data_5m:
                eth_df = data_5m[eth_sym].copy()
                eth_df["timestamp"] = pd.to_datetime(eth_df["timestamp"])
                eth_1h = eth_df.set_index("timestamp").resample("1h").agg({"close": "last"}).dropna().reset_index()
                for _, row in eth_1h.iterrows():
                    dca_eth_prices[row["timestamp"]] = float(row["close"])
            elif eth_sym in data_1d:
                # 일봉에서 근사
                for _, row in data_1d[eth_sym].iterrows():
                    dca_eth_prices[row["timestamp"]] = float(row["close"])
            logger.info(f"  DCA: BTC {len(dca_btc_prices)} prices, ETH {len(dca_eth_prices)} prices")

        # ── 2. Build ts→idx mappings ──
        sym_ts_idx_5m: Dict[str, Dict] = {}
        for symbol, df in data_5m.items():
            sym_ts_idx_5m[symbol] = dict(zip(df["timestamp"], df.index))

        sym_ts_idx_4h: Dict[str, Dict] = {}
        for symbol, df in ichi_pc.items():
            sym_ts_idx_4h[symbol] = dict(zip(df["timestamp"], df.index))

        sym_ts_idx_ma100: Dict[str, Dict] = {}
        for symbol, df in ma100_pc.items():
            sym_ts_idx_ma100[symbol] = dict(zip(df["timestamp"], df.index))

        # sub-daily 모드: exit은 1d 데이터로 체크 (진입만 sub-daily)
        ma100_1d_pc: Dict[str, pd.DataFrame] = {}
        sym_ts_idx_ma100_1d: Dict[str, Dict] = {}
        if ma100_use_sub and self.enable_ma100:
            ma100_1d_pc = precompute_ma100_signals(data_1d, self.ma100_config)
            for symbol, df in ma100_1d_pc.items():
                sym_ts_idx_ma100_1d[symbol] = dict(zip(df["timestamp"], df.index))

        # ── 3. Build event maps ──

        # Ichimoku events (4h)
        ichi_at: Dict[pd.Timestamp, List[dict]] = {}
        if self.enable_ichimoku:
            for symbol, df in ichi_pc.items():
                mask = df["ichimoku_entry"] & (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
                for _, row in df.loc[mask].iterrows():
                    ts = row["timestamp"]
                    price = float(row["close"])
                    cloud_bottom = float(row["cloud_bottom"])
                    sl = cloud_bottom * (1 + self.ichimoku_config["sl_buffer"] / 100)
                    sl_dist_pct = (sl - price) / price * 100
                    if sl_dist_pct < self.ichimoku_config["min_sl_pct"] or sl_dist_pct > self.ichimoku_config["max_sl_pct"]:
                        continue
                    tp = price * (1 - sl_dist_pct * self.ichimoku_config["rr_ratio"] / 100)
                    ichi_at.setdefault(ts, []).append({
                        "symbol": symbol,
                        "price": price,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "score": int(row["ichi_score"]),
                        "thickness": float(row["cloud_thickness"]),
                    })

        # Mirror events (5m)
        mirror_at: Dict[pd.Timestamp, List[str]] = {}
        if self.enable_mirror:
            for symbol, df in mirror_pc.items():
                mask = df["mirror_entry_signal"] & (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
                for _, row in df.loc[mask].iterrows():
                    mirror_at.setdefault(row["timestamp"], []).append(symbol)

        # MA100 events (1d or 4h)
        ma100_at: Dict[pd.Timestamp, List[dict]] = {}
        if self.enable_ma100:
            for symbol, df in ma100_pc.items():
                for side_label, col in [("long", "long_signal"), ("short", "short_signal")]:
                    mask = df[col] & (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
                    for _, row in df.loc[mask].iterrows():
                        ma100_at.setdefault(row["timestamp"], []).append({
                            "symbol": symbol,
                            "side": side_label,
                            "price": float(row["close"]),
                            "slope": float(row["slope"]) if pd.notna(row.get("slope", np.nan)) else 0,
                        })

        total_ichi = sum(len(v) for v in ichi_at.values())
        total_mirror = sum(len(v) for v in mirror_at.values())
        total_ma100 = sum(len(v) for v in ma100_at.values())
        dca_str = f", DCA=8h BTC+ETH" if self.enable_dca else ""
        ma100_tf = "hourly" if data_1h_ma100 else ("4h" if data_4h_all else "1d")
        logger.info(f"시그널: 이치모쿠={total_ichi}, 미러={total_mirror}, MA100({ma100_tf})={total_ma100}{dca_str}")

        # ── 4. Build unified timeline ──
        ts_5m: Set[pd.Timestamp] = set()
        for df in data_5m.values():
            mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
            ts_5m.update(df.loc[mask, "timestamp"].tolist())

        ts_4h: Set[pd.Timestamp] = set()
        for df in ichi_pc.values():
            mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
            ts_4h.update(df.loc[mask, "timestamp"].tolist())

        ts_ma100: Set[pd.Timestamp] = set()
        for df in ma100_pc.values():
            mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
            ts_ma100.update(df.loc[mask, "timestamp"].tolist())
        if ma100_use_sub:
            # sub-daily mode: MA100 timestamps merge into general timeline
            # (not into ts_4h — they get their own set ts_ma100)
            pass
        if not ma100_use_sub:
            ts_1d = ts_ma100
        else:
            # sub-daily: 1d 타임스탬프도 exit 체크용으로 포함
            ts_1d = set()
            for df in ma100_1d_pc.values():
                mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
                ts_1d.update(df.loc[mask, "timestamp"].tolist())

        ts_1h: Set[pd.Timestamp] = set()
        if self.enable_dca:
            for ts_key in dca_btc_prices:
                if start_ts <= ts_key <= end_ts:
                    ts_1h.add(ts_key)

        all_ts = sorted(ts_5m | ts_4h | ts_1d | ts_1h | ts_ma100)
        logger.info(f"통합 타임라인: {len(all_ts)} steps (5m={len(ts_5m)}, 4h={len(ts_4h)}, 1d={len(ts_1d)}, 1h={len(ts_1h)})")

        if not all_ts:
            logger.warning("타임라인이 비어있습니다.")
            return

        # ── 5. Main loop ──
        candle_count = 0
        cooldown_ichi = timedelta(hours=self.ichimoku_config["cooldown_hours"])
        cooldown_mirror_min = self.mirror_config["cooldown_candles"] * 5  # minutes
        cooldown_ma100 = timedelta(days=self.ma100_config["cooldown_days"])

        for i, ts in enumerate(all_ts):
            is_5m = ts in ts_5m
            is_4h = ts in ts_4h
            is_1d = ts in ts_1d
            is_1h = ts in ts_1h
            # MA100 체크 타이밍: 4h 모드면 4h 타임스탬프에서, 아니면 1d에서
            is_ma100_tick = (ts in ts_ma100) if ma100_use_sub else is_1d

            # ── Exit checks (priority) ──
            if is_4h:
                for sym in list(self.positions["ichimoku"].keys()):
                    idx = sym_ts_idx_4h.get(sym, {}).get(ts)
                    if idx is not None:
                        self._check_ichimoku_exit(sym, ichi_pc[sym].iloc[idx], ts)

            if is_5m:
                for sym in list(self.positions["mirror_short"].keys()):
                    idx = sym_ts_idx_5m.get(sym, {}).get(ts)
                    if idx is not None:
                        self._check_mirror_exit(sym, data_5m[sym].iloc[idx], ts)

            # MA100 exit: sub-daily 모드면 1d 캔들에서 exit, 아니면 기존대로
            if ma100_use_sub:
                if is_1d:
                    for sym in list(self.positions["ma100"].keys()):
                        idx = sym_ts_idx_ma100_1d.get(sym, {}).get(ts)
                        if idx is not None:
                            self._check_ma100_exit(sym, ma100_1d_pc[sym].iloc[idx], ts)
            else:
                if is_ma100_tick:
                    for sym in list(self.positions["ma100"].keys()):
                        idx = sym_ts_idx_ma100.get(sym, {}).get(ts)
                        if idx is not None:
                            self._check_ma100_exit(sym, ma100_pc[sym].iloc[idx], ts)

            # ── Entry checks ──

            # Ichimoku entries (4h)
            if is_4h and self.enable_ichimoku:
                ichi_pos_count = len(self.positions["ichimoku"])
                if ichi_pos_count < self.ichimoku_config["max_positions"] and self.balance > 0:
                    # BTC filter
                    btc_uptrend = btc_trend.get(ts)
                    if btc_uptrend is not False:  # enter if True or None
                        candidates = ichi_at.get(ts, [])
                        candidates_sorted = sorted(candidates, key=lambda s: (-s["score"], -s["thickness"]))
                        for sig in candidates_sorted:
                            symbol = sig["symbol"]
                            if symbol in self.positions["ichimoku"]:
                                continue
                            if len(self.positions["ichimoku"]) >= self.ichimoku_config["max_positions"]:
                                break
                            last_exit = self.last_exit_times["ichimoku"].get(symbol)
                            if last_exit and (ts - last_exit) < cooldown_ichi:
                                continue
                            qty, _margin = self._calc_qty("ichimoku", sig["price"])
                            if qty <= 0:
                                break
                            self.positions["ichimoku"][symbol] = {
                                "symbol": symbol,
                                "side": "short",
                                "entry_price": sig["price"],
                                "entry_time": ts,
                                "stop_loss": sig["stop_loss"],
                                "take_profit": sig["take_profit"],
                                "highest": sig["price"],
                                "lowest": sig["price"],
                                "trail_stop": sig["stop_loss"],
                                "trailing": False,
                                "qty": qty,
                                "signal_type": "Ichimoku",
                            }

            # Mirror entries (5m)
            if is_5m and self.enable_mirror:
                mirror_pos_count = len(self.positions["mirror_short"])
                if mirror_pos_count < self.mirror_config["max_positions"] and self.balance > 0:
                    candidates = mirror_at.get(ts, [])
                    for symbol in candidates:
                        if symbol in self.positions["mirror_short"]:
                            continue
                        if len(self.positions["mirror_short"]) >= self.mirror_config["max_positions"]:
                            break
                        last_exit = self.last_exit_times["mirror_short"].get(symbol)
                        if last_exit and (ts - last_exit) < timedelta(minutes=cooldown_mirror_min):
                            continue
                        idx = sym_ts_idx_5m.get(symbol, {}).get(ts)
                        if idx is None:
                            continue
                        entry_price = float(data_5m[symbol].iloc[idx]["open"])
                        if entry_price <= 0:
                            continue
                        margin = self.balance * self.mirror_config["position_pct"]
                        max_margin_m = self.mirror_config.get("max_margin")
                        if max_margin_m and margin > max_margin_m:
                            margin = max_margin_m
                        notional = margin * self.mirror_config["leverage"]
                        qty = notional / entry_price
                        if qty <= 0:
                            break
                        sl = entry_price * (1 + self.mirror_config["sl_pct"] / 100)
                        self.positions["mirror_short"][symbol] = {
                            "symbol": symbol,
                            "side": "short",
                            "entry_price": entry_price,
                            "entry_time": ts,
                            "stop_loss": sl,
                            "trailing_active": False,
                            "lowest_since_entry": entry_price,
                            "trail_stop": None,
                            "qty": qty,
                            "margin": margin,
                            "signal_type": "MirrorShort",
                        }

            # MA100 entries (1d or 4h)
            if is_ma100_tick and self.enable_ma100:
                ma100_pos_count = len(self.positions["ma100"])
                if ma100_pos_count < self.ma100_config["max_positions"] and self.balance > 0:
                    candidates = ma100_at.get(ts, [])
                    candidates_sorted = sorted(candidates, key=lambda s: abs(s["slope"]), reverse=True)
                    for sig in candidates_sorted:
                        symbol = sig["symbol"]
                        side = sig["side"]
                        if symbol in self.positions["ma100"]:
                            continue
                        if len(self.positions["ma100"]) >= self.ma100_config["max_positions"]:
                            break
                        last_exit = self.last_exit_times["ma100"].get(symbol)
                        if last_exit and (ts - last_exit) < cooldown_ma100:
                            continue
                        entry_price = sig["price"]
                        tp_pct = self.ma100_config["tp_pct"]
                        # SL: 전체 DCA 평균단가 기준
                        dca_r = self.ma100_config["dca_ratios"]
                        dca_iv = self.ma100_config["dca_interval_pct"]
                        tr = sum(dca_r)
                        w_sum = 0.0
                        for ii, rr in enumerate(dca_r):
                            if side == "short":
                                w_sum += entry_price * (1 + ii * dca_iv / 100) * rr
                            else:
                                w_sum += entry_price * (1 - ii * dca_iv / 100) * rr
                        avg_full = w_sum / tr
                        if side == "long":
                            sl_price = avg_full * (1 - self.ma100_config["sl_pct"] / 100)
                            tp_price = entry_price * (1 + tp_pct / 100) if tp_pct > 0 else 0
                        else:
                            sl_price = avg_full * (1 + self.ma100_config["sl_pct"] / 100)
                            tp_price = entry_price * (1 - tp_pct / 100) if tp_pct > 0 else 0
                        total_qty, _margin = self._calc_qty("ma100", entry_price)
                        if total_qty <= 0:
                            break

                        # DCA 분할매수: 1차만 진입, 나머지는 대기
                        dca_ratios = self.ma100_config["dca_ratios"]
                        dca_interval = self.ma100_config["dca_interval_pct"]
                        total_ratio = sum(dca_ratios)
                        tranche_sizes = [total_qty * r / total_ratio for r in dca_ratios]
                        first_qty = tranche_sizes[0]

                        pending_dca = []
                        for i in range(1, len(dca_ratios)):
                            if side == "short":
                                dca_price = entry_price * (1 + i * dca_interval / 100)
                            else:
                                dca_price = entry_price * (1 - i * dca_interval / 100)
                            pending_dca.append({"price": dca_price, "size": tranche_sizes[i]})

                        self.positions["ma100"][symbol] = {
                            "symbol": symbol,
                            "side": side,
                            "entry_price": entry_price,
                            "entry_time": ts,
                            "stop_loss": sl_price,
                            "take_profit": tp_price,
                            "highest": entry_price,
                            "lowest": entry_price,
                            "trail_stop": sl_price,
                            "trailing": False,
                            "qty": first_qty,
                            "total_planned_qty": total_qty,
                            "pending_dca": pending_dca,
                            "filled_entries": [{"price": entry_price, "size": first_qty}],
                            "signal_type": "TouchBounce",
                        }

            # DCA processing (1h timestamps, 8h interval)
            if is_1h and self.enable_dca:
                btc_p = dca_btc_prices.get(ts, 0)
                eth_p = dca_eth_prices.get(ts, 0)
                if btc_p > 0:
                    self._spot_last_prices["BTC"] = btc_p
                if eth_p > 0:
                    self._spot_last_prices["ETH"] = eth_p
                if btc_p > 0 and eth_p > 0:
                    self._process_dca(ts, btc_p, eth_p)

            # ── Update current prices for open positions ──
            if is_5m:
                for sym in set().union(*(d.keys() for d in self.positions.values())):
                    idx = sym_ts_idx_5m.get(sym, {}).get(ts)
                    if idx is not None:
                        self._current_prices[sym] = float(data_5m[sym].iloc[idx]["close"])
            if is_4h:
                for sym in self.positions["ichimoku"]:
                    idx = sym_ts_idx_4h.get(sym, {}).get(ts)
                    if idx is not None:
                        self._current_prices[sym] = float(ichi_pc[sym].iloc[idx]["close"])
            if is_ma100_tick:
                for sym in self.positions["ma100"]:
                    idx = sym_ts_idx_ma100.get(sym, {}).get(ts)
                    if idx is not None:
                        self._current_prices[sym] = float(ma100_pc[sym].iloc[idx]["close"])

            # ── Equity snapshot (every 30 min = 6 5m candles) ──
            if is_5m:
                candle_count += 1
                if candle_count % 6 == 0:
                    self._update_equity(ts)

            if (i + 1) % 5000 == 0:
                logger.info(f"  진행: {i+1}/{len(all_ts)} ({(i+1)/len(all_ts)*100:.0f}%)")

        # ── 6. Close remaining positions ──
        last_ts = all_ts[-1]
        for strategy in ["ichimoku", "mirror_short", "ma100"]:
            for symbol in list(self.positions[strategy].keys()):
                cp = self._current_prices.get(symbol)
                if cp is not None:
                    self._close_position(strategy, symbol, cp, "BacktestEnd", last_ts)

        # 스팟 최종 시세 갱신
        if dca_btc_prices:
            last_btc_ts = max(t for t in dca_btc_prices if t <= last_ts) if any(t <= last_ts for t in dca_btc_prices) else None
            if last_btc_ts:
                self._spot_last_prices["BTC"] = dca_btc_prices[last_btc_ts]
        if dca_eth_prices:
            last_eth_ts = max(t for t in dca_eth_prices if t <= last_ts) if any(t <= last_ts for t in dca_eth_prices) else None
            if last_eth_ts:
                self._spot_last_prices["ETH"] = dca_eth_prices[last_eth_ts]

        self._update_equity(last_ts)


# ─── Console Report ────────────────────────────────────────────

def print_combined_report(bt: CombinedBacktester, start_dt: datetime, end_dt: datetime):
    days = (end_dt - start_dt).days
    trades = bt.trades
    n = len(trades)

    print()
    print("=" * 70)
    print("  Combined Strategy Backtest (Ichimoku + Mirror + MA100 + DCA)")
    print("=" * 70)
    print(f"  Period : {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} ({days} days)")
    print(f"  Initial: ${bt.initial_balance:,.2f}")
    print()

    if n == 0:
        print("  No trades executed.")
        print("=" * 65)
        return

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades)
    total_pnl_pct = total_pnl / bt.initial_balance * 100
    win_rate = len(wins) / n * 100
    gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    pf_str = f"{pf:.2f}" if pf != float("inf") else "INF"

    print("--- Overall Results ---")
    print(f"  Total Trades : {n}")
    print(f"  Win Rate     : {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Total PnL    : ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
    print(f"  Final Balance: ${bt.balance:,.2f}")
    print(f"  Profit Factor: {pf_str}")
    print(f"  Max Drawdown : -${bt.max_drawdown:,.2f} (-{bt.max_drawdown_pct:.1f}%)")
    print()

    # Strategy breakdown
    print("--- Strategy Breakdown ---")
    for strat in ["ichimoku", "mirror_short", "ma100", "dca"]:
        st = [t for t in trades if t["strategy"] == strat]
        if not st:
            continue
        st_wins = [t for t in st if t["pnl_usd"] > 0]
        st_pnl = sum(t["pnl_usd"] for t in st)
        st_wr = len(st_wins) / len(st) * 100 if st else 0
        st_gp = sum(t["pnl_usd"] for t in st_wins) if st_wins else 0
        st_gl = abs(sum(t["pnl_usd"] for t in st if t["pnl_usd"] <= 0))
        st_pf = st_gp / st_gl if st_gl > 0 else float("inf")
        st_pf_str = f"{st_pf:.2f}" if st_pf != float("inf") else "INF"
        label = STRATEGY_LABELS.get(strat, strat)
        if strat == "dca":
            total_cost = sum(h["cost"] for h in bt.spot_holdings.values())
            print(f"  {label:15s}: {len(st):3d} buys   Cost=${total_cost:8.2f}  Fee=${abs(st_pnl):6.2f}")
        else:
            print(f"  {label:15s}: {len(st):3d} trades  WR={st_wr:5.1f}%  PnL=${st_pnl:+8.2f}  PF={st_pf_str}")

    # Spot DCA accumulation summary
    total_spot_cost = 0
    total_spot_value = 0
    has_holdings = False
    for asset in ["BTC", "ETH"]:
        h = bt.spot_holdings[asset]
        if h["qty"] > 0:
            has_holdings = True
    if has_holdings:
        print()
        print("--- Spot DCA Accumulation ---")
        for asset in ["BTC", "ETH"]:
            h = bt.spot_holdings[asset]
            if h["qty"] <= 0:
                continue
            avg_cost = h["cost"] / h["qty"]
            price = bt._spot_last_prices.get(asset, 0)
            value = h["qty"] * price
            pnl = value - h["cost"]
            pnl_pct = pnl / h["cost"] * 100 if h["cost"] > 0 else 0
            total_spot_cost += h["cost"]
            total_spot_value += value
            print(f"  {asset:4s}: {h['qty']:.6f} | Avg ${avg_cost:,.2f} | Spent ${h['cost']:,.2f} | Val ${value:,.2f} ({pnl:+,.2f}, {pnl_pct:+.1f}%) | {h['count']} buys")
        if total_spot_cost > 0:
            total_pnl_spot = total_spot_value - total_spot_cost
            total_pnl_pct = total_pnl_spot / total_spot_cost * 100
            print(f"  Total: Spent ${total_spot_cost:,.2f} → Val ${total_spot_value:,.2f} ({total_pnl_spot:+,.2f}, {total_pnl_pct:+.1f}%)")
    print()

    # Exit reasons
    reasons: Dict[str, int] = {}
    for t in trades:
        r = t["reason"]
        reasons[r] = reasons.get(r, 0) + 1
    print("--- Exit Reasons ---")
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r:15s}: {cnt}")
    print()

    # Top trades
    sorted_trades = sorted(trades, key=lambda t: t["pnl_usd"], reverse=True)
    print("--- Top Wins ---")
    for t in sorted_trades[:5]:
        sym = t["symbol"].split("/")[0]
        label = STRATEGY_LABELS.get(t["strategy"], t["strategy"])[:5]
        print(f"  [{label:5s}] {sym:12s} {t['side']:5s} {t['pnl_pct']:+7.1f}%  ${t['pnl_usd']:+8.2f}  ({t['reason']})")

    print()
    print("--- Top Losses ---")
    for t in sorted_trades[-5:]:
        sym = t["symbol"].split("/")[0]
        label = STRATEGY_LABELS.get(t["strategy"], t["strategy"])[:5]
        print(f"  [{label:5s}] {sym:12s} {t['side']:5s} {t['pnl_pct']:+7.1f}%  ${t['pnl_usd']:+8.2f}  ({t['reason']})")

    print()

    # Monthly breakdown
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"])
        trade_df["month"] = trade_df["exit_time"].dt.to_period("M")

        print("--- Monthly Breakdown ---")
        print(f"  {'Month':8s}  {'Total':>6s}  {'Ichi':>6s}  {'Mirror':>6s}  {'MA100':>6s}  {'DCA':>6s}  {'PnL':>10s}")
        print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}")
        for m in sorted(trade_df["month"].unique()):
            mt = trade_df[trade_df["month"] == m]
            m_pnl = mt["pnl_usd"].sum()
            m_ichi = len(mt[mt["strategy"] == "ichimoku"])
            m_mirror = len(mt[mt["strategy"] == "mirror_short"])
            m_ma100 = len(mt[mt["strategy"] == "ma100"])
            m_dca = len(mt[mt["strategy"] == "dca"])
            print(f"  {str(m):8s}  {len(mt):6d}  {m_ichi:6d}  {m_mirror:6d}  {m_ma100:6d}  {m_dca:6d}  ${m_pnl:+9.2f}")

    print()
    print("=" * 70)


# ─── HTML Report ───────────────────────────────────────────────

def generate_combined_html_report(
    bt: CombinedBacktester, start_dt: datetime, end_dt: datetime,
    btc_price_df: Optional[pd.DataFrame] = None,
) -> str:
    """통합 HTML 리포트 생성."""
    trades = bt.trades
    n = len(trades)
    days = (end_dt - start_dt).days

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades) if trades else 0
    total_pnl_pct = total_pnl / bt.initial_balance * 100
    win_rate = len(wins) / n * 100 if n > 0 else 0
    gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    pf_str = f"{pf:.2f}" if pf != float("inf") else "INF"
    pnl_cls = "positive" if total_pnl >= 0 else "negative"

    # Strategy breakdown data
    strat_data = {}
    for strat in ["ichimoku", "mirror_short", "ma100", "dca"]:
        st = [t for t in trades if t["strategy"] == strat]
        st_wins = [t for t in st if t["pnl_usd"] > 0]
        st_pnl = sum(t["pnl_usd"] for t in st)
        st_wr = len(st_wins) / len(st) * 100 if st else 0
        st_gp = sum(t["pnl_usd"] for t in st_wins) if st_wins else 0
        st_gl = abs(sum(t["pnl_usd"] for t in st if t["pnl_usd"] <= 0))
        st_pf = st_gp / st_gl if st_gl > 0 else float("inf")
        strat_data[strat] = {
            "trades": len(st), "wins": len(st_wins), "losses": len(st) - len(st_wins),
            "pnl": st_pnl, "wr": st_wr, "pf": st_pf,
            "color": STRATEGY_COLORS.get(strat, "#888"), "label": STRATEGY_LABELS.get(strat, strat),
        }

    # Spot DCA accumulation data
    total_spot_cost = sum(h["cost"] for h in bt.spot_holdings.values())
    total_spot_value = sum(h["qty"] * bt._spot_last_prices.get(a, 0) for a, h in bt.spot_holdings.items())
    spot_pnl = total_spot_value - total_spot_cost
    spot_pnl_pct = spot_pnl / total_spot_cost * 100 if total_spot_cost > 0 else 0
    spot_pnl_cls = "positive" if spot_pnl >= 0 else "negative"

    # Strategy cards HTML
    strat_cards = ""
    for strat, sd in strat_data.items():
        if sd["trades"] == 0:
            continue
        if strat == "dca":
            # DCA 적립 전용 카드
            dca_detail = ""
            for asset in ["BTC", "ETH"]:
                h = bt.spot_holdings[asset]
                if h["qty"] > 0:
                    avg = h["cost"] / h["qty"]
                    p = bt._spot_last_prices.get(asset, 0)
                    v = h["qty"] * p
                    pnl_a = v - h["cost"]
                    dca_detail += f"{asset}: {h['qty']:.6f} (avg ${avg:,.0f}, val ${v:,.0f}, {pnl_a:+,.0f}) | "
            strat_cards += f"""
    <div class="card" style="border-left: 4px solid {sd['color']};">
      <div class="card-title">{sd['label']} (BTC 40% + ETH 60%)</div>
      <div class="card-value {spot_pnl_cls}">${spot_pnl:+,.2f} ({spot_pnl_pct:+.1f}%)</div>
      <div class="card-sub">{sd['trades']} buys | Spent ${total_spot_cost:,.2f} | Val ${total_spot_value:,.2f}<br>{dca_detail.rstrip(' | ')}</div>
    </div>"""
        else:
            sd_pnl_cls = "positive" if sd["pnl"] >= 0 else "negative"
            sd_pf_str = f"{sd['pf']:.2f}" if sd["pf"] != float("inf") else "INF"
            strat_cards += f"""
    <div class="card" style="border-left: 4px solid {sd['color']};">
      <div class="card-title">{sd['label']}</div>
      <div class="card-value {sd_pnl_cls}">${sd['pnl']:+,.2f}</div>
      <div class="card-sub">{sd['trades']} trades | WR={sd['wr']:.0f}% ({sd['wins']}W/{sd['losses']}L) | PF={sd_pf_str}</div>
    </div>"""

    # Equity curve
    eq_labels = _json.dumps([e["timestamp"].strftime("%m/%d %H:%M") for e in bt.equity_curve])
    eq_values = _json.dumps([round(e["equity"], 2) for e in bt.equity_curve])

    # BTC price aligned to equity curve timestamps
    btc_eq_prices = []
    if btc_price_df is not None and not btc_price_df.empty and bt.equity_curve:
        # Ensure timestamp-indexed Series for lookup
        if "timestamp" in btc_price_df.columns:
            btc_ts = pd.to_datetime(btc_price_df["timestamp"])
            btc_close = btc_price_df["close"].values
        else:
            btc_ts = pd.to_datetime(btc_price_df.index)
            btc_close = btc_price_df["close"].values
        btc_ts_arr = btc_ts.values  # numpy datetime64 array
        for e in bt.equity_curve:
            ts_np = np.datetime64(e["timestamp"])
            idx = np.searchsorted(btc_ts_arr, ts_np, side="right") - 1
            if idx >= 0:
                btc_eq_prices.append(round(float(btc_close[idx]), 2))
            else:
                btc_eq_prices.append(None)
    btc_eq_json = _json.dumps(btc_eq_prices) if btc_eq_prices else "[]"

    # Per-strategy cumulative PnL (for overlay)
    all_strats = ["ichimoku", "mirror_short", "ma100", "dca"]
    strat_cum: Dict[str, List[float]] = {s: [] for s in all_strats}
    strat_running = {s: 0.0 for s in strat_cum}
    # Sort trades by exit_time for cumulative
    sorted_trades_by_exit = sorted(trades, key=lambda t: t["exit_time"])
    for t in sorted_trades_by_exit:
        strat_running[t["strategy"]] += t["pnl_usd"]
    # For chart: build per-trade cumulative by strategy
    strat_cum_data = {s: [] for s in all_strats}
    strat_cum_running = {s: 0.0 for s in strat_cum_data}
    cum_labels = []
    cum_total = []
    total_running = 0.0
    for idx, t in enumerate(sorted_trades_by_exit):
        strat_cum_running[t["strategy"]] += t["pnl_usd"]
        total_running += t["pnl_usd"]
        cum_labels.append(f"#{idx+1}")
        cum_total.append(round(total_running, 2))
        for s in strat_cum_data:
            strat_cum_data[s].append(round(strat_cum_running[s], 2))

    # BTC price aligned to cumulative PnL (by trade exit_time)
    btc_cum_prices = []
    if btc_price_df is not None and not btc_price_df.empty and sorted_trades_by_exit:
        if "timestamp" in btc_price_df.columns:
            btc_ts2 = pd.to_datetime(btc_price_df["timestamp"]).values
            btc_close2 = btc_price_df["close"].values
        else:
            btc_ts2 = pd.to_datetime(btc_price_df.index).values
            btc_close2 = btc_price_df["close"].values
        for t in sorted_trades_by_exit:
            ts_np = np.datetime64(t["exit_time"])
            idx = np.searchsorted(btc_ts2, ts_np, side="right") - 1
            if idx >= 0:
                btc_cum_prices.append(round(float(btc_close2[idx]), 2))
            else:
                btc_cum_prices.append(None)
    btc_cum_json = _json.dumps(btc_cum_prices) if btc_cum_prices else "[]"

    # Per-trade PnL bar chart
    trade_pnls = [round(t["pnl_usd"], 2) for t in sorted_trades_by_exit]
    trade_colors = []
    for t in sorted_trades_by_exit:
        trade_colors.append(STRATEGY_COLORS[t["strategy"]])
    trade_syms = [t["symbol"].split("/")[0][:6] for t in sorted_trades_by_exit]

    # Strategy contribution pie
    strat_pnl_labels = []
    strat_pnl_values = []
    strat_pnl_colors = []
    for strat, sd in strat_data.items():
        if sd["trades"] > 0:
            strat_pnl_labels.append(sd["label"])
            strat_pnl_values.append(round(sd["pnl"], 2))
            strat_pnl_colors.append(sd["color"])

    # Exit reasons
    reasons: Dict[str, int] = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1

    # Monthly breakdown
    monthly_rows = ""
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"])
        trade_df["month"] = trade_df["exit_time"].dt.to_period("M")
        for m in sorted(trade_df["month"].unique()):
            mt = trade_df[trade_df["month"] == m]
            m_pnl = mt["pnl_usd"].sum()
            m_cls = "positive" if m_pnl >= 0 else "negative"
            m_ichi = mt[mt["strategy"] == "ichimoku"]
            m_mirror = mt[mt["strategy"] == "mirror_short"]
            m_ma100 = mt[mt["strategy"] == "ma100"]
            m_grid = mt[mt["strategy"] == "dca"]
            monthly_rows += f"""
            <tr>
                <td>{str(m)}</td>
                <td>{len(mt)}</td>
                <td class="{m_cls}">${m_pnl:+,.2f}</td>
                <td>{len(m_ichi)} ({sum(1 for t in m_ichi.itertuples() if t.pnl_usd > 0)}W)</td>
                <td class="{'positive' if m_ichi['pnl_usd'].sum() >= 0 else 'negative'}">${m_ichi['pnl_usd'].sum():+,.2f}</td>
                <td>{len(m_mirror)} ({sum(1 for t in m_mirror.itertuples() if t.pnl_usd > 0)}W)</td>
                <td class="{'positive' if m_mirror['pnl_usd'].sum() >= 0 else 'negative'}">${m_mirror['pnl_usd'].sum():+,.2f}</td>
                <td>{len(m_ma100)} ({sum(1 for t in m_ma100.itertuples() if t.pnl_usd > 0)}W)</td>
                <td class="{'positive' if m_ma100['pnl_usd'].sum() >= 0 else 'negative'}">${m_ma100['pnl_usd'].sum():+,.2f}</td>
                <td>{len(m_grid)} ({sum(1 for t in m_grid.itertuples() if t.pnl_usd > 0)}W)</td>
                <td class="{'positive' if m_grid['pnl_usd'].sum() >= 0 else 'negative'}">${m_grid['pnl_usd'].sum():+,.2f}</td>
            </tr>"""

    # Trade table rows
    trade_rows = ""
    for i, t in enumerate(sorted_trades_by_exit):
        sym = t["symbol"].split("/")[0]
        p_cls = "positive" if t["pnl_usd"] > 0 else "negative"
        s_cls = "positive" if t["side"] == "long" else "negative"
        entry_t = t["entry_time"].strftime("%m/%d %H:%M") if hasattr(t["entry_time"], "strftime") else str(t["entry_time"])
        exit_t = t["exit_time"].strftime("%m/%d %H:%M") if hasattr(t["exit_time"], "strftime") else str(t["exit_time"])
        if hasattr(t["entry_time"], "strftime"):
            dur = t["exit_time"] - t["entry_time"]
            dur_h = dur.total_seconds() / 3600
            dur_str = f"{dur_h:.1f}h" if dur_h >= 1 else f"{dur.total_seconds()/60:.0f}m"
        else:
            dur_str = "-"
        strat_label = STRATEGY_LABELS.get(t["strategy"], t["strategy"])
        strat_color = STRATEGY_COLORS.get(t["strategy"], "#888")
        trade_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td><span class="strat-badge" style="background:{strat_color}20;color:{strat_color};">{strat_label}</span></td>
            <td><strong>{sym}</strong></td>
            <td class="{s_cls}">{t['side'].upper()}</td>
            <td>{entry_t}</td>
            <td>{exit_t}</td>
            <td>{dur_str}</td>
            <td>${t['entry_price']:.6g}</td>
            <td>${t['exit_price']:.6g}</td>
            <td class="{p_cls}">{t['pnl_pct']:+.1f}%</td>
            <td class="{p_cls}">${t['pnl_usd']:+.2f}</td>
            <td><span class="badge badge-{t['reason'].lower()}">{t['reason']}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Combined Strategy Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d28; --border: #2a2d3a;
    --text: #e1e4ea; --muted: #8b8fa3;
    --green: #22c55e; --red: #ef4444; --blue: #3b82f6;
    --purple: #a855f7; --yellow: #eab308;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); padding: 24px; line-height: 1.6;
  }}
  .container {{ max-width: 1300px; margin: 0 auto; }}
  h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 24px; font-size: 14px; }}
  .grid {{ display: grid; gap: 16px; margin-bottom: 24px; }}
  .grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
  .grid-3 {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
  .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
  }}
  .card-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 8px; }}
  .card-value {{ font-size: 28px; font-weight: 700; }}
  .card-sub {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}
  .positive {{ color: var(--green); }}
  .negative {{ color: var(--red); }}
  .chart-card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-bottom: 24px;
  }}
  .chart-card h3 {{ font-size: 16px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{
    text-align: left; padding: 10px 12px;
    border-bottom: 2px solid var(--border);
    color: var(--muted); font-weight: 600; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }}
  td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  tr:hover {{ background: rgba(255,255,255,0.02); }}
  .badge {{
    display: inline-block; padding: 2px 8px;
    border-radius: 6px; font-size: 11px; font-weight: 600;
  }}
  .badge-sl {{ background: rgba(239,68,68,0.15); color: var(--red); }}
  .badge-tp {{ background: rgba(34,197,94,0.15); color: var(--green); }}
  .badge-trail {{ background: rgba(168,85,247,0.15); color: var(--purple); }}
  .badge-reversal {{ background: rgba(59,130,246,0.15); color: var(--blue); }}
  .badge-cloud {{ background: rgba(234,179,8,0.15); color: var(--yellow); }}
  .badge-maxloss {{ background: rgba(239,68,68,0.25); color: var(--red); }}
  .badge-backtestend {{ background: rgba(139,143,163,0.15); color: var(--muted); }}
  .badge-gridtp {{ background: rgba(34,197,94,0.15); color: var(--green); }}
  .badge-breakout {{ background: rgba(234,179,8,0.15); color: var(--yellow); }}
  .badge-rebalance {{ background: rgba(59,130,246,0.15); color: var(--blue); }}
  .strat-badge {{
    display: inline-block; padding: 2px 8px;
    border-radius: 6px; font-size: 11px; font-weight: 600;
  }}
  .section-title {{
    font-size: 18px; font-weight: 600;
    margin: 32px 0 16px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  canvas {{ max-height: 320px; }}
</style>
</head>
<body>
<div class="container">

<h1>Combined Strategy Backtest</h1>
<p class="subtitle">{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} ({days} days) &nbsp;|&nbsp; Ichimoku (4h) + Mirror Short (5m) + MA100 V2 (1d) + BTC Accum (1h)</p>

<!-- Summary Cards -->
<div class="grid grid-4">
  <div class="card">
    <div class="card-title">Total PnL</div>
    <div class="card-value {pnl_cls}">${total_pnl:+,.2f}</div>
    <div class="card-sub">{total_pnl_pct:+.2f}% return</div>
  </div>
  <div class="card">
    <div class="card-title">Win Rate</div>
    <div class="card-value">{win_rate:.1f}%</div>
    <div class="card-sub">{len(wins)}W / {len(losses)}L of {n} trades</div>
  </div>
  <div class="card">
    <div class="card-title">Profit Factor</div>
    <div class="card-value">{pf_str}</div>
    <div class="card-sub">Gross +${gross_profit:,.2f} / -${gross_loss:,.2f}</div>
  </div>
  <div class="card">
    <div class="card-title">Max Drawdown</div>
    <div class="card-value negative">-${bt.max_drawdown:,.2f}</div>
    <div class="card-sub">-{bt.max_drawdown_pct:.1f}% from peak</div>
  </div>
</div>

<!-- Strategy Cards -->
<h2 class="section-title">Strategy Breakdown</h2>
<div class="grid grid-3">
{strat_cards}
</div>

<!-- Equity Curve -->
<div class="chart-card">
  <h3>Equity Curve</h3>
  <canvas id="equityChart"></canvas>
</div>

<!-- PnL Charts -->
<div class="grid grid-2">
  <div class="chart-card">
    <h3>Cumulative PnL by Strategy</h3>
    <canvas id="cumPnlChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Per-Trade PnL ($)</h3>
    <canvas id="tradePnlChart"></canvas>
  </div>
</div>

<div class="grid grid-2">
  <div class="chart-card">
    <h3>Strategy Contribution</h3>
    <canvas id="contribChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Exit Reasons</h3>
    <canvas id="reasonChart"></canvas>
  </div>
</div>

<!-- Monthly Breakdown -->
<h2 class="section-title">Monthly Breakdown</h2>
<div class="card" style="overflow-x:auto;">
  <table>
    <thead>
      <tr>
        <th>Month</th><th>Total</th><th>PnL</th>
        <th>Ichi</th><th>Ichi PnL</th>
        <th>Mirror</th><th>Mirror PnL</th>
        <th>MA100</th><th>MA100 PnL</th>
        <th>DCA</th><th>DCA PnL</th>
      </tr>
    </thead>
    <tbody>{monthly_rows}</tbody>
  </table>
</div>

<!-- Trade Log -->
<h2 class="section-title">Trade Log ({n} trades)</h2>
<div class="card" style="overflow-x:auto;">
  <table>
    <thead>
      <tr><th>#</th><th>Strategy</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>Dur</th><th>Entry $</th><th>Exit $</th><th>PnL %</th><th>PnL $</th><th>Reason</th></tr>
    </thead>
    <tbody>{trade_rows}</tbody>
  </table>
</div>

<p style="text-align:center; color:var(--muted); font-size:12px; margin-top:32px;">
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
</p>

</div>

<script>
Chart.defaults.color = '#8b8fa3';
Chart.defaults.borderColor = '#2a2d3a';

// Equity Curve
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: {eq_labels},
    datasets: [{{
      label: 'Equity ($)',
      data: {eq_values},
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.1)',
      fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
      yAxisID: 'y',
    }},
    {{
      label: 'BTC Price',
      data: {btc_eq_json},
      borderColor: '#eab308',
      backgroundColor: 'transparent',
      fill: false, tension: 0.3, pointRadius: 0, borderWidth: 1.5,
      borderDash: [6, 3],
      yAxisID: 'y1',
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ padding: 12 }} }} }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 12 }} }},
      y: {{ position: 'left', title: {{ display: true, text: 'Equity ($)' }} }},
      y1: {{ position: 'right', title: {{ display: true, text: 'BTC ($)' }}, grid: {{ drawOnChartArea: false }} }},
    }}
  }}
}});

// Cumulative PnL by Strategy
new Chart(document.getElementById('cumPnlChart'), {{
  type: 'line',
  data: {{
    labels: {_json.dumps(cum_labels)},
    datasets: [
      {{ label: 'Total', data: {_json.dumps(cum_total)}, borderColor: '#e1e4ea', borderWidth: 2, tension: 0.3, pointRadius: 0, yAxisID: 'y' }},
      {{ label: 'Ichimoku', data: {_json.dumps(strat_cum_data.get("ichimoku", []))}, borderColor: '#a855f7', borderWidth: 1.5, tension: 0.3, pointRadius: 0, borderDash: [4,2], yAxisID: 'y' }},
      {{ label: 'Mirror', data: {_json.dumps(strat_cum_data.get("mirror_short", []))}, borderColor: '#ef4444', borderWidth: 1.5, tension: 0.3, pointRadius: 0, borderDash: [4,2], yAxisID: 'y' }},
      {{ label: 'MA100', data: {_json.dumps(strat_cum_data.get("ma100", []))}, borderColor: '#3b82f6', borderWidth: 1.5, tension: 0.3, pointRadius: 0, borderDash: [4,2], yAxisID: 'y' }},
      {{ label: 'DCA', data: {_json.dumps(strat_cum_data.get("dca", []))}, borderColor: '#22c55e', borderWidth: 1.5, tension: 0.3, pointRadius: 0, borderDash: [4,2], yAxisID: 'y' }},
      {{ label: 'BTC Price', data: {btc_cum_json}, borderColor: '#eab308', borderWidth: 1.5, tension: 0.3, pointRadius: 0, borderDash: [6,3], yAxisID: 'y1' }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ padding: 12 }} }} }},
    scales: {{
      y: {{ position: 'left', title: {{ display: true, text: 'PnL ($)' }} }},
      y1: {{ position: 'right', title: {{ display: true, text: 'BTC ($)' }}, grid: {{ drawOnChartArea: false }} }},
    }}
  }}
}});

// Per-Trade PnL
new Chart(document.getElementById('tradePnlChart'), {{
  type: 'bar',
  data: {{
    labels: {_json.dumps(trade_syms)},
    datasets: [{{
      label: 'PnL ($)',
      data: {_json.dumps(trade_pnls)},
      backgroundColor: {_json.dumps(trade_colors)},
      borderRadius: 3,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ font: {{ size: 9 }}, maxRotation: 90 }} }} }}
  }}
}});

// Strategy Contribution
new Chart(document.getElementById('contribChart'), {{
  type: 'doughnut',
  data: {{
    labels: {_json.dumps(strat_pnl_labels)},
    datasets: [{{
      data: {_json.dumps(strat_pnl_values)},
      backgroundColor: {_json.dumps(strat_pnl_colors)},
      borderWidth: 0,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ padding: 16 }} }},
      tooltip: {{ callbacks: {{ label: function(ctx) {{ return ctx.label + ': $' + ctx.raw.toFixed(2); }} }} }}
    }}
  }}
}});

// Exit Reasons
new Chart(document.getElementById('reasonChart'), {{
  type: 'doughnut',
  data: {{
    labels: {_json.dumps(list(reasons.keys()))},
    datasets: [{{
      data: {_json.dumps(list(reasons.values()))},
      backgroundColor: ['#ef4444', '#22c55e', '#a855f7', '#3b82f6', '#eab308', '#f97316', '#8b8fa3'].slice(0, {len(reasons)}),
      borderWidth: 0,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ padding: 16 }} }} }}
  }}
}});
</script>
</body>
</html>"""

    out_path = Path("data/backtest_combined_report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


# ─── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Combined Backtest: Ichimoku + Mirror Short + MA100 V2 + Spot DCA")
    parser.add_argument("--balance", type=float, default=6500.0, help="Initial balance ($)")
    parser.add_argument("--start", default="2025-01-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-18", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-ichimoku", action="store_true", help="Disable Ichimoku strategy")
    parser.add_argument("--no-mirror", action="store_true", help="Disable Mirror Short strategy")
    parser.add_argument("--no-ma100", action="store_true", help="Disable MA100 V2 strategy")
    parser.add_argument("--no-dca", action="store_true", help="Disable Spot DCA strategy")
    parser.add_argument("--ma100-4h", action="store_true", help="Use 4h touch detection for MA100 (instead of 1d)")
    parser.add_argument("--ma100-hourly", action="store_true", help="Use hourly (22/23/00 UTC) touch detection for MA100")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=59)

    enable_ichi = not args.no_ichimoku
    enable_mirror = not args.no_mirror
    enable_ma100 = not args.no_ma100
    enable_dca = not args.no_dca

    active = []
    if enable_ichi:
        active.append("Ichimoku")
    if enable_mirror:
        active.append("Mirror")
    if enable_ma100:
        active.append("MA100")
    if enable_dca:
        active.append("DCA")

    if not active:
        logger.error("최소 1개 전략을 활성화하세요.")
        return

    logger.info(f"=== Combined Backtest: {' + '.join(active)} ===")
    logger.info(f"    Period: {args.start} ~ {args.end}, Balance: ${args.balance}")

    # ── Data Loading ──
    loader = DataLoader()

    data_5m: Dict[str, pd.DataFrame] = {}
    data_4h: Dict[str, pd.DataFrame] = {}
    data_4h_all: Dict[str, pd.DataFrame] = {}
    data_1h_ma100: Dict[str, pd.DataFrame] = {}
    data_1d: Dict[str, pd.DataFrame] = {}
    data_1h_btc: Optional[pd.DataFrame] = None
    use_ma100_4h = args.ma100_4h
    use_ma100_hourly = args.ma100_hourly

    t0 = time.time()

    need_5m = enable_mirror or use_ma100_hourly or use_ma100_4h
    if need_5m:
        logger.info("── 5m 데이터 로드 ──")
        data_5m = load_5m_data(loader, start_dt, end_dt)
        logger.info(f"  5m: {len(data_5m)} 심볼")

    if enable_ichi:
        logger.info("── 4h 데이터 로드 ──")
        data_4h = load_4h_data(loader, start_dt, end_dt)
        logger.info(f"  4h: {len(data_4h)} 심볼")

    if enable_ma100:
        logger.info("── 1d 데이터 로드 ──")
        data_1d = load_1d_data(loader, start_dt, end_dt)
        logger.info(f"  1d: {len(data_1d)} 심볼")

        if use_ma100_4h:
            logger.info("── 4h 전체 심볼 로드 (MA100 4h 터치 감지) ──")
            data_4h_all = load_4h_all(loader, data_5m, start_dt, end_dt)
            logger.info(f"  4h(MA100): {len(data_4h_all)} 심볼")
        elif use_ma100_hourly:
            logger.info("── 1h 전체 심볼 로드 (MA100 7/8/9시 KST 터치 감지) ──")
            data_1h_ma100 = load_1h_all(loader, data_5m, start_dt, end_dt, check_hours=(22, 23, 0))
            logger.info(f"  1h(MA100): {len(data_1h_ma100)} 심볼")

    if enable_dca:
        logger.info("── 1h BTC 데이터 로드 (DCA) ──")
        data_1h_btc = load_1h_btc(loader, start_dt, end_dt)
        if data_1h_btc is not None:
            logger.info(f"  1h BTC: {len(data_1h_btc)} rows")
        else:
            logger.warning("  BTC 1h 데이터 로드 실패 → DCA 비활성화")
            enable_dca = False

    load_elapsed = time.time() - t0
    logger.info(f"데이터 로드 완료 ({load_elapsed:.1f}s)")

    if not data_5m and not data_4h and not data_1d and data_1h_btc is None:
        logger.error("데이터가 없습니다.")
        return

    # ── Backtest ──
    bt = CombinedBacktester(
        initial_balance=args.balance,
        enable_ichimoku=enable_ichi,
        enable_mirror=enable_mirror,
        enable_ma100=enable_ma100,
        enable_dca=enable_dca,
    )

    logger.info("백테스트 시작...")
    t0 = time.time()
    bt.run(data_5m, data_4h, data_1d, start_dt, end_dt,
           data_1h_btc=data_1h_btc,
           data_4h_all=data_4h_all if use_ma100_4h else None,
           data_1h_ma100=data_1h_ma100 if use_ma100_hourly else None)
    bt_elapsed = time.time() - t0
    logger.info(f"백테스트 완료 ({bt_elapsed:.1f}s)")

    # ── Report ──
    print_combined_report(bt, start_dt, end_dt)

    # Load BTC daily price for chart overlay
    btc_price_df = None
    btc_sym = "BTC/USDT:USDT"
    if btc_sym in data_1d:
        btc_price_df = data_1d[btc_sym]
    elif btc_sym in data_5m:
        # Resample 5m → 1d for chart
        btc_price_df = data_5m[btc_sym].resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
    elif btc_sym in data_4h:
        btc_price_df = data_4h[btc_sym].resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
    if btc_price_df is not None:
        logger.info(f"BTC 가격 데이터: {len(btc_price_df)} rows (차트 오버레이용)")

    html_path = generate_combined_html_report(bt, start_dt, end_dt, btc_price_df=btc_price_df)
    logger.info(f"HTML 리포트: {html_path}")

    # ── Save trades JSON for Monte Carlo ──
    trades_json_path = Path("data/backtest_combined_trades.json")
    trades_json_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_trades = []
    for t in bt.trades:
        st = dict(t)
        st["entry_time"] = str(st["entry_time"])
        st["exit_time"] = str(st["exit_time"])
        serializable_trades.append(st)
    trades_json_path.write_text(
        _json.dumps({
            "initial_balance": bt.initial_balance,
            "final_balance": round(bt.balance, 2),
            "start": args.start,
            "end": args.end,
            "trades": serializable_trades,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Trades JSON: {trades_json_path}")

    import webbrowser
    abs_path = Path(html_path).resolve()
    webbrowser.open(f"file:///{abs_path}")


if __name__ == "__main__":
    main()
