"""
Williams Fractals 전략 (롱+숏)

프랙탈 돌파/이탈 + EMA20/50 + RSI(35-65) + ADX>20 필터.
백테스트 검증 완료: PF 3.29, 승률 64%, MDD -4.6%.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── 파라미터 ─────────────────────────────────────────────────

FRACTALS_LEVERAGE = 10
FRACTALS_POSITION_PCT = 0.05

FRACTALS_PARAMS = {
    "fractal_period": 5,
    "sl_pct": 3.0,
    "tp_pct": 0,  # 거래소 트레일링이 수익 관리, TP 제한 없음
    "trail_start_pct": 2.0,
    "trail_pct": 1.5,
    "cooldown_candles": 2,
    "max_positions": 5,
    # 필터 비활성 (노필터 모드 — 그리드 서치 결과 PF 1.95, WF PASS)
    "use_filters": False,
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_period": 14,
    "rsi_long_max": 65,
    "rsi_short_min": 35,
    "adx_min": 20,
}


# ─── 지표 계산 ────────────────────────────────────────────────

def compute_fractals(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Williams Fractals 계산 + EMA/RSI/ADX 필터 지표."""
    df = df.copy()
    highs, lows = df["high"].values, df["low"].values
    length = len(df)

    fh = np.full(length, np.nan)
    fl = np.full(length, np.nan)
    for i in range(n, length - n):
        is_high = all(highs[i] > highs[i-j] and highs[i] > highs[i+j] for j in range(1, n+1))
        if is_high:
            fh[i] = highs[i]
        is_low = all(lows[i] < lows[i-j] and lows[i] < lows[i+j] for j in range(1, n+1))
        if is_low:
            fl[i] = lows[i]

    df["fractal_high"] = fh
    df["fractal_low"] = fl
    df["last_fractal_high"] = df["fractal_high"].ffill()
    df["last_fractal_low"] = df["fractal_low"].ffill()

    # EMA
    df["ema_fast"] = df["close"].ewm(span=FRACTALS_PARAMS["ema_fast"], adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=FRACTALS_PARAMS["ema_slow"], adjust=False).mean()

    # RSI
    p = FRACTALS_PARAMS["rsi_period"]
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/p, min_periods=p).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/p, min_periods=p).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ADX
    h, l, c = df["high"], df["low"], df["close"]
    pdm = h.diff().clip(lower=0)
    mdm = (-l.diff()).clip(lower=0)
    pdm_clean = np.where(pdm > mdm, pdm, 0)
    mdm_clean = np.where(mdm > np.array(pdm_clean, dtype=float), mdm, 0)
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = pd.Series(tr, index=df.index).ewm(alpha=1/14, min_periods=14).mean()
    pdi = 100 * pd.Series(pdm_clean, index=df.index).ewm(alpha=1/14, min_periods=14).mean() / atr
    mdi = 100 * pd.Series(mdm_clean, index=df.index).ewm(alpha=1/14, min_periods=14).mean() / atr
    dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
    df["adx"] = dx.ewm(alpha=1/14, min_periods=14).mean()

    return df


# ─── 진입 시그널 ──────────────────────────────────────────────

def get_fractals_entry_signal(
    symbol: str,
    row: pd.Series,
    prev_row: pd.Series,
    last_exit_candle_count: Optional[int],
    params: dict = FRACTALS_PARAMS,
) -> Optional[dict]:
    """
    프랙탈 진입 시그널 계산.

    Returns:
        진입 시그널 dict or None
    """
    price = float(row["close"])

    # 프랙탈 레벨 확인
    fh = row.get("last_fractal_high")
    fl = row.get("last_fractal_low")
    if pd.isna(fh) or pd.isna(fl):
        return None

    prev_close = float(prev_row["close"])
    prev_fh = prev_row.get("last_fractal_high")
    prev_fl = prev_row.get("last_fractal_low")
    if pd.isna(prev_fh) or pd.isna(prev_fl):
        return None

    # 쿨다운 체크
    if last_exit_candle_count is not None and last_exit_candle_count < params["cooldown_candles"]:
        return None

    # 롱 시그널: 프랙탈 고점 돌파
    long_signal = (prev_close <= prev_fh) and (price > fh)
    # 숏 시그널: 프랙탈 저점 이탈
    short_signal = (prev_close >= prev_fl) and (price < fl)

    if not long_signal and not short_signal:
        return None

    # 필터 체크 (use_filters=False이면 스킵)
    if params.get("use_filters", False):
        ema_f = float(row.get("ema_fast", 0))
        ema_s = float(row.get("ema_slow", 0))
        rsi = float(row.get("rsi", 50))
        adx = float(row.get("adx", 0))

        if long_signal:
            if ema_f <= ema_s:
                return None
            if rsi > params["rsi_long_max"]:
                return None
        elif short_signal:
            if ema_f >= ema_s:
                return None
            if rsi < params["rsi_short_min"]:
                return None

        if adx < params["adx_min"]:
            return None

    # SL/TP 계산
    side = "long" if long_signal else "short"
    sl_pct = params["sl_pct"]
    tp_pct = params["tp_pct"]

    if side == "long":
        stop_loss = price * (1 - sl_pct / 100)
        take_profit = price * (1 + tp_pct / 100) if tp_pct > 0 else 0
    else:
        stop_loss = price * (1 + sl_pct / 100)
        take_profit = price * (1 - tp_pct / 100) if tp_pct > 0 else 0

    return {
        "symbol": symbol,
        "side": side,
        "price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "score": float(row.get("adx", 0)),
        "rsi": float(row.get("rsi", 50)),
        "fractal_level": float(fh) if long_signal else float(fl),
    }


# ─── 청산 시그널 ──────────────────────────────────────────────

def check_fractals_exit(
    pos: dict,
    row: pd.Series,
    params: dict = FRACTALS_PARAMS,
) -> Optional[dict]:
    """
    프랙탈 포지션 청산 체크 (SL/TP/Trail).

    pos는 in-place 업데이트됩니다 (trailing 상태).
    """
    side = pos["side"]
    entry = float(pos["entry_price"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])

    # 현재 수익률
    if side == "long":
        cur_pnl_pct = (close / entry - 1) * 100
        best_pnl_pct = (high / entry - 1) * 100
    else:
        cur_pnl_pct = (1 - close / entry) * 100
        best_pnl_pct = (1 - low / entry) * 100

    # best 업데이트
    prev_best = pos.get("best_pnl", 0)
    if best_pnl_pct > prev_best:
        pos["best_pnl"] = best_pnl_pct

    # 1. SL 체크
    sl = float(pos["stop_loss"])
    if side == "long" and low <= sl:
        return {"reason": "SL", "price": sl}
    if side == "short" and high >= sl:
        return {"reason": "SL", "price": sl}

    # 2. TP 체크 (tp > 0일 때만)
    tp = float(pos.get("take_profit", 0))
    if tp > 0:
        if side == "long" and high >= tp:
            return {"reason": "TP", "price": tp}
        if side == "short" and low <= tp:
            return {"reason": "TP", "price": tp}

    # 3. 트레일링 스탑
    best = pos.get("best_pnl", 0)
    trail_start = params["trail_start_pct"]
    trail_pct = params["trail_pct"]

    if best >= trail_start:
        pos["trailing"] = True
        drawdown = best - cur_pnl_pct
        if drawdown >= trail_pct:
            return {"reason": "Trail", "price": close}

    return None
