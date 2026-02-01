"""
일목균형표 전략 로직 (숏 전용)

BTC 상승 추세일 때 알트코인 숏 진입 (역추세 전략)
"""

import logging
from datetime import datetime
from typing import Optional, Dict
import pandas as pd

logger = logging.getLogger(__name__)


# 전략 파라미터 (숏 전용)
LEVERAGE = 20
POSITION_PCT = 0.05  # 자산의 5%

STRATEGY_PARAMS = {
    "min_cloud_thickness": 0.2,  # 최소 구름 두께 %
    "min_sl_pct": 0.3,  # 최소 손절 거리 %
    "max_sl_pct": 8.0,  # 최대 손절 거리 %
    "sl_buffer": 0.2,  # 손절 버퍼 %
    "rr_ratio": 2.0,  # 손익비
    "trail_pct": 1.5,  # 트레일링 스탑 %
    "cooldown_hours": 4,  # 재진입 쿨다운
    "max_positions": 5,  # 최대 동시 포지션
    "use_btc_filter": True,  # BTC 상승 추세일 때만 숏 진입
    "short_only": True,  # 숏 전용 모드
}

# 운용 코인 목록 (CCXT 심볼)
MAJOR_COINS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT", "SOL/USDT:USDT",
    "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT",
    "POL/USDT:USDT", "LTC/USDT:USDT", "ATOM/USDT:USDT", "UNI/USDT:USDT", "ETC/USDT:USDT",
    "APT/USDT:USDT", "NEAR/USDT:USDT", "FIL/USDT:USDT", "AAVE/USDT:USDT", "INJ/USDT:USDT",
]


def get_entry_signal(
    symbol: str,
    row: pd.Series,
    btc_uptrend: Optional[bool],
    last_exit_time: Optional[datetime],
    params: dict = STRATEGY_PARAMS
) -> Optional[dict]:
    """
    숏 진입 신호 계산 (숏 전용 전략)

    BTC 상승 추세일 때 알트코인 숏 진입 (역추세)

    Args:
        symbol: 심볼
        row: 현재 캔들 데이터 (Series)
        btc_uptrend: BTC 상승 추세 여부
        last_exit_time: 마지막 청산 시간
        params: 전략 파라미터

    Returns:
        진입 신호 딕셔너리 또는 None
    """
    price = float(row["close"])
    cloud_top = float(row["cloud_top"])
    cloud_bottom = float(row["cloud_bottom"])
    thickness = float(row["cloud_thickness"])

    # 구름 안이면 스킵 (횡보장)
    if bool(row["in_cloud"]):
        return None

    # 구름 두께 필터
    if thickness < params["min_cloud_thickness"]:
        return None

    # 쿨다운 체크
    now = datetime.utcnow()
    if last_exit_time is not None:
        hours_since_exit = (now - last_exit_time).total_seconds() / 3600
        if hours_since_exit < params["cooldown_hours"]:
            return None

    # === 숏 조건 (숏 전용) ===
    # 가격이 구름 아래, 전환선이 기준선 아래
    if bool(row["below_cloud"]) and not bool(row["tenkan_above"]):
        # TK 데드크로스 또는 기준선 하향 돌파
        has_signal = bool(row["tk_cross_down"]) or bool(row["kijun_cross_down"])
        if not has_signal:
            return None

        # BTC 트렌드 필터: BTC 상승 추세일 때만 숏 진입
        if params.get("use_btc_filter", True):
            if btc_uptrend is False:  # BTC 하락 추세면 숏 스킵
                return None

        # 점수 계산
        score = 0
        if bool(row.get("chikou_bearish", False)):  # 후행스팬 약세
            score += 2
        if not bool(row.get("cloud_green", True)):  # 빨간 구름
            score += 1
        if thickness > 1.0:  # 두꺼운 구름
            score += 1

        # 손절가: 구름 하단 + 버퍼
        stop_loss = cloud_bottom * (1 + params["sl_buffer"] / 100)
        sl_distance_pct = (stop_loss - price) / price * 100

        if params["min_sl_pct"] <= sl_distance_pct <= params["max_sl_pct"]:
            # 익절가: 손익비에 따라 계산
            take_profit = price * (1 - sl_distance_pct * params["rr_ratio"] / 100)
            return {
                "symbol": symbol,
                "side": "short",
                "price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "score": score,
                "thickness": thickness,
            }

    return None


def check_exit_signal(
    pos: dict,
    row: pd.Series,
    params: dict = STRATEGY_PARAMS
) -> Optional[dict]:
    """
    청산 신호 계산

    숏 포지션 청산 조건:
    1. 손절: 구름 하단 돌파
    2. 익절: 목표가 도달 (트레일링 스탑 적용)
    3. 구름 진입: 가격이 구름 안으로 진입

    Args:
        pos: 포지션 정보
        row: 현재 캔들 데이터

    Returns:
        청산 신호 딕셔너리 또는 None
    """
    side = pos["side"]
    entry = float(pos["entry_price"])
    stop_loss = float(pos["stop_loss"])
    take_profit = float(pos["take_profit"])
    trail_stop = float(pos.get("trail_stop", stop_loss))
    trailing = bool(pos.get("trailing", False))

    price = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])

    if side == "long":
        # MaxLoss: -4% (레버리지 20배 기준 -80%)
        max_loss_price = entry * 0.98
        if low <= max_loss_price:
            return {"action": "close", "reason": "MaxLoss", "price": max_loss_price}

        # 최고가 갱신 및 트레일링
        highest = float(pos.get("highest", entry))
        if high > highest:
            highest = high
            pos["highest"] = highest
            if high >= take_profit:
                trailing = True
                trail_stop = max(trail_stop, high * (1 - params["trail_pct"] / 100))
                pos["trailing"] = True
                pos["trail_stop"] = trail_stop

        # 1. 손절
        if low <= stop_loss:
            return {"action": "close", "reason": "Stop", "price": max(stop_loss, low)}

        # 2. 트레일링
        if trailing and low <= trail_stop:
            return {"action": "close", "reason": "Trail", "price": trail_stop}

        # 3. TP (트레일링 없을 때)
        if not trailing and high >= take_profit:
            return {"action": "close", "reason": "TP", "price": take_profit}

        # 4. 구름 진입
        if bool(row["in_cloud"]) or bool(row["below_cloud"]):
            return {"action": "close", "reason": "CloudExit", "price": price}

    else:  # short
        # MaxLoss: +4%
        max_loss_price = entry * 1.02
        if high >= max_loss_price:
            return {"action": "close", "reason": "MaxLoss", "price": max_loss_price}

        lowest = float(pos.get("lowest", entry))
        if low < lowest:
            lowest = low
            pos["lowest"] = lowest
            if low <= take_profit:
                trailing = True
                trail_stop = min(trail_stop, low * (1 + params["trail_pct"] / 100))
                pos["trailing"] = True
                pos["trail_stop"] = trail_stop

        # 1. 손절
        if high >= stop_loss:
            return {"action": "close", "reason": "Stop", "price": min(stop_loss, high)}

        # 2. 트레일링
        if trailing and high >= trail_stop:
            return {"action": "close", "reason": "Trail", "price": trail_stop}

        # 3. TP
        if not trailing and low <= take_profit:
            return {"action": "close", "reason": "TP", "price": take_profit}

        # 4. 구름 진입
        if bool(row["in_cloud"]) or bool(row["above_cloud"]):
            return {"action": "close", "reason": "CloudExit", "price": price}

    return None


def update_btc_trend(df: pd.DataFrame) -> Optional[bool]:
    """
    BTC 트렌드 계산

    Args:
        df: BTC OHLCV DataFrame

    Returns:
        True: 상승 추세, False: 하락 추세, None: 판단 불가
    """
    if df is None or df.empty:
        return None

    df = df.copy()
    df['sma_26'] = df['close'].rolling(26).mean()
    df['sma_52'] = df['close'].rolling(52).mean()

    latest = df.iloc[-1]
    if pd.notna(latest['sma_26']) and pd.notna(latest['sma_52']):
        return latest['sma_26'] > latest['sma_52']

    return None
