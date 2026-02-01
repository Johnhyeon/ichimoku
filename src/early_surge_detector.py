"""
초기 급등 감지 모듈 (5분봉 기반)

급등 시작 초반에 진입하기 위한 실시간 감지 로직
- 거래량 폭발 감지
- 가격 급등 감지
- 횡보 → 급등 전환 감지

사용 예시:
    detector = EarlySurgeDetector(data_fetcher)
    signal = detector.check_surge("ZORA/USDT:USDT")
    if signal:
        print(f"급등 감지! {signal}")
"""

import logging
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# 파라미터
EARLY_SURGE_PARAMS = {
    # === 거래량 조건 ===
    "volume_spike_min": 10,          # 최소 거래량 스파이크 (평균 대비 배수)
    "volume_lookback": 20,           # 거래량 평균 계산 기간

    # === 가격 조건 ===
    "price_surge_min": 5.0,          # 최소 가격 상승률 % (단일 캔들)
    "price_surge_window": 3,         # 가격 상승 확인 캔들 수 (연속)
    "price_surge_total_min": 8.0,    # 최소 총 상승률 % (window 내)

    # === 이전 상태 조건 (횡보 확인) ===
    "consolidation_lookback": 12,    # 횡보 확인 기간 (캔들 수) = 1시간
    "consolidation_range_max": 5.0,  # 횡보 최대 변동폭 % (완화)

    # === 진입 타이밍 ===
    "entry_delay_candles": 1,        # 급등 확인 후 대기 캔들 수
    "max_entry_price_from_low": 15,  # 저점 대비 최대 진입가 %

    # === 리스크 관리 ===
    "sl_pct": 5.0,                   # 손절 %
    "tp_pct": 50.0,                  # 익절 % (급등 노림)
    "trail_start_pct": 25.0,         # 트레일링 시작 % (25%부터)
    "trail_pct": 8.0,                # 트레일링 스탑 % (8% 여유)

    # === 레버리지 ===
    "leverage": 5,
    "position_pct": 0.03,            # 자산의 3% (고위험)
}


class EarlySurgeDetector:
    """초기 급등 감지기"""

    def __init__(self, data_fetcher, params: dict = EARLY_SURGE_PARAMS):
        self.data_fetcher = data_fetcher
        self.params = params

    def get_5m_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """5분봉 데이터 조회"""
        try:
            df = self.data_fetcher.get_ohlcv(symbol, '5m', limit=limit)
            if df is not None:
                df = df.reset_index()
            return df
        except Exception as e:
            logger.error(f"5분봉 조회 실패 ({symbol}): {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 계산"""
        df = df.copy()

        # 거래량 평균
        lookback = self.params['volume_lookback']
        df['volume_sma'] = df['volume'].rolling(lookback).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # 가격 변화율
        df['change_pct'] = df['close'].pct_change() * 100

        # N캔들 누적 상승률
        window = self.params['price_surge_window']
        df['change_window'] = df['close'].pct_change(window) * 100

        # 횡보 구간 변동폭
        consol = self.params['consolidation_lookback']
        df['range_high'] = df['high'].rolling(consol).max()
        df['range_low'] = df['low'].rolling(consol).min()
        df['range_pct'] = (df['range_high'] - df['range_low']) / df['range_low'] * 100

        # 캔들 특성
        df['is_green'] = df['close'] > df['open']
        df['body_pct'] = abs(df['close'] - df['open']) / df['open'] * 100

        # 급등 신호
        df['volume_spike'] = df['volume_ratio'] >= self.params['volume_spike_min']
        df['price_spike'] = df['change_pct'] >= self.params['price_surge_min']

        return df

    def check_consolidation(self, df: pd.DataFrame, idx: int) -> bool:
        """
        해당 시점 이전이 횡보 구간이었는지 확인

        Args:
            df: DataFrame
            idx: 현재 인덱스

        Returns:
            횡보 여부
        """
        lookback = self.params['consolidation_lookback']
        if idx < lookback + 1:
            return False

        # 급등 직전 구간
        prev_range = df.iloc[idx - lookback - 1:idx - 1]

        if len(prev_range) < lookback - 2:
            return False

        high = prev_range['high'].max()
        low = prev_range['low'].min()
        range_pct = (high - low) / low * 100

        return range_pct <= self.params['consolidation_range_max']

    def detect_surge_start(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        급등 시작 감지

        조건:
        1. 거래량 스파이크 (10x+)
        2. 가격 급등 (5%+)
        3. 이전 횡보 구간

        Returns:
            급등 신호 또는 None
        """
        if df is None or len(df) < 30:
            return None

        df = self.calculate_indicators(df)

        # 최근 N개 캔들에서 급등 시작점 찾기
        for i in range(-5, 0):  # 최근 5개 캔들
            idx = len(df) + i
            if idx < 20:
                continue

            row = df.iloc[idx]

            # 1. 거래량 스파이크
            if not row['volume_spike']:
                continue

            # 2. 가격 급등
            if not row['price_spike']:
                continue

            # 3. 양봉
            if not row['is_green']:
                continue

            # 4. 이전 횡보
            was_consolidating = self.check_consolidation(df, idx)
            if not was_consolidating:
                continue

            # 급등 시작 발견!
            surge_low = df.iloc[idx - 1]['low']  # 급등 직전 저점
            current_price = df.iloc[-1]['close']
            price_from_low = (current_price - surge_low) / surge_low * 100

            # 저점 대비 너무 올랐으면 스킵
            if price_from_low > self.params['max_entry_price_from_low']:
                continue

            return {
                'detected': True,
                'surge_start_time': row['timestamp'],
                'surge_start_price': float(row['close']),
                'surge_low': float(surge_low),
                'current_price': float(current_price),
                'price_from_low': price_from_low,
                'volume_ratio': float(row['volume_ratio']),
                'price_change': float(row['change_pct']),
            }

        return None

    def get_entry_signal(self, symbol: str) -> Optional[Dict]:
        """
        진입 신호 생성

        Args:
            symbol: 심볼

        Returns:
            진입 신호 또는 None
        """
        df = self.get_5m_data(symbol, limit=100)
        if df is None:
            return None

        surge = self.detect_surge_start(df)
        if surge is None:
            return None

        entry_price = surge['current_price']
        surge_low = surge['surge_low']

        # 손절: 급등 저점 아래
        stop_loss = surge_low * 0.98

        # 손절 거리 계산
        sl_distance = (entry_price - stop_loss) / entry_price * 100
        if sl_distance > self.params['sl_pct'] * 1.5:
            # 손절이 너무 멀면 기본값 사용
            stop_loss = entry_price * (1 - self.params['sl_pct'] / 100)

        # 익절
        take_profit = entry_price * (1 + self.params['tp_pct'] / 100)

        return {
            'symbol': symbol,
            'side': 'long',
            'price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'surge_info': surge,
            'leverage': self.params['leverage'],
        }


def scan_early_surges(
    symbols: List[str],
    data_fetcher,
    params: dict = EARLY_SURGE_PARAMS
) -> List[Dict]:
    """
    여러 코인에서 초기 급등 스캔

    Args:
        symbols: 심볼 리스트
        data_fetcher: DataFetcher
        params: 파라미터

    Returns:
        급등 신호 리스트
    """
    detector = EarlySurgeDetector(data_fetcher, params)
    signals = []

    for symbol in symbols:
        try:
            signal = detector.get_entry_signal(symbol)
            if signal:
                signals.append(signal)
                logger.info(
                    f"초기 급등 감지: {symbol} | "
                    f"Vol={signal['surge_info']['volume_ratio']:.1f}x "
                    f"Change={signal['surge_info']['price_change']:.1f}%"
                )
        except Exception as e:
            logger.debug(f"스캔 실패 ({symbol}): {e}")
            continue

    return signals
