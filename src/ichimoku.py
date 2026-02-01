"""
일목균형표 (Ichimoku Cloud) 지표 계산

ichimoku_backtest.py의 calculate_ichimoku() 함수와 100% 동일한 로직
"""

import pandas as pd


def calculate_ichimoku(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52) -> pd.DataFrame:
    """
    일목균형표 지표 계산 (백테스트와 동일한 로직)

    Args:
        df: OHLCV DataFrame (timestamp, open, high, low, close, volume)
        tenkan_period: 전환선 기간 (기본 9)
        kijun_period: 기준선 기간 (기본 26)
        senkou_b_period: 선행스팬B 기간 (기본 52)

    Returns:
        지표가 추가된 DataFrame
    """
    df = df.copy()

    # 전환선 (Tenkan-sen): 9일 고저 평균
    high_9 = df['high'].rolling(tenkan_period).max()
    low_9 = df['low'].rolling(tenkan_period).min()
    df['tenkan'] = (high_9 + low_9) / 2

    # 기준선 (Kijun-sen): 26일 고저 평균
    high_26 = df['high'].rolling(kijun_period).max()
    low_26 = df['low'].rolling(kijun_period).min()
    df['kijun'] = (high_26 + low_26) / 2

    # 선행스팬A (Senkou Span A): 전환선+기준선 평균, 26일 앞으로
    df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(kijun_period)

    # 선행스팬B (Senkou Span B): 52일 고저 평균, 26일 앞으로
    high_52 = df['high'].rolling(senkou_b_period).max()
    low_52 = df['low'].rolling(senkou_b_period).min()
    df['senkou_b'] = ((high_52 + low_52) / 2).shift(kijun_period)

    # 구름 상단/하단
    df['cloud_top'] = df[['senkou_a', 'senkou_b']].max(axis=1)
    df['cloud_bottom'] = df[['senkou_a', 'senkou_b']].min(axis=1)

    # 구름 두께 (% 기준)
    df['cloud_thickness'] = (df['cloud_top'] - df['cloud_bottom']) / df['close'] * 100

    # 구름 색상 (녹색: 상승, 빨간색: 하락)
    df['cloud_green'] = df['senkou_a'] > df['senkou_b']

    # 전환선/기준선 크로스 신호
    df['tenkan_above'] = df['tenkan'] > df['kijun']
    tenkan_above_shifted = df['tenkan_above'].shift(1)
    df['tk_cross_up'] = (df['tenkan_above']) & (~tenkan_above_shifted.fillna(False).astype(bool))
    df['tk_cross_down'] = (~df['tenkan_above']) & (tenkan_above_shifted.fillna(True).astype(bool))

    # 가격 vs 기준선
    df['price_above_kijun'] = df['close'] > df['kijun']
    price_above_kijun_shifted = df['price_above_kijun'].shift(1)
    df['kijun_cross_up'] = (df['price_above_kijun']) & (~price_above_kijun_shifted.fillna(False).astype(bool))
    df['kijun_cross_down'] = (~df['price_above_kijun']) & (price_above_kijun_shifted.fillna(True).astype(bool))

    # 후행스팬 방향 (현재 종가 vs 26일 전 종가)
    df['chikou_bullish'] = df['close'] > df['close'].shift(26)
    df['chikou_bearish'] = df['close'] < df['close'].shift(26)

    # 가격 위치
    df['above_cloud'] = df['close'] > df['cloud_top']
    df['below_cloud'] = df['close'] < df['cloud_bottom']
    df['in_cloud'] = ~df['above_cloud'] & ~df['below_cloud']

    # 거래량 지표 (Volume Spike Filter 용)
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    return df
