"""
급등 코인 패턴 분석기

ZORA, CYS, ZKP, MEGA, C98, ANIME, HANA, ZK, SOPH, IN, OPEN 등
급등 코인들의 공통 패턴을 분석합니다.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 급등 코인 목록
SURGE_COINS = [
    "ZORA", "CYS", "ZKP", "MEGA", "C98", "ANIME",
    "HANA", "ZK", "SOPH", "IN", "OPEN"
]

def get_exchange():
    """Bybit 연결"""
    return ccxt.bybit({
        'options': {'defaultType': 'swap'}
    })

def fetch_ohlcv(exchange, symbol, timeframe='1h', limit=500):
    """OHLCV 데이터 가져오기"""
    try:
        ohlcv = exchange.fetch_ohlcv(f"{symbol}/USDT:USDT", timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        return df
    except Exception as e:
        print(f"  ⚠️ {symbol} 데이터 없음: {e}")
        return None

def calculate_indicators(df):
    """기술적 지표 계산"""
    # 변동성
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(24).std() * 100

    # 이동평균
    df['sma_7'] = df['close'].rolling(7).mean()
    df['sma_25'] = df['close'].rolling(25).mean()
    df['sma_99'] = df['close'].rolling(99).mean()

    # EMA
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()

    # 볼린저 밴드
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 거래량 지표
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # 가격 변화율
    df['change_1h'] = df['close'].pct_change(1) * 100
    df['change_4h'] = df['close'].pct_change(4) * 100
    df['change_24h'] = df['close'].pct_change(24) * 100

    # 모멘텀
    df['momentum'] = df['close'] - df['close'].shift(10)

    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    return df

def find_surge_points(df, threshold=10):
    """급등 포인트 찾기 (24시간 내 threshold% 이상 상승)"""
    surges = []

    for i in range(24, len(df)):
        change_24h = (df.iloc[i]['close'] - df.iloc[i-24]['close']) / df.iloc[i-24]['close'] * 100

        if change_24h >= threshold:
            # 급등 시작점 찾기 (가장 낮은 지점)
            window = df.iloc[i-24:i+1]
            min_idx = window['close'].idxmin()

            surges.append({
                'peak_time': df.index[i],
                'start_time': min_idx,
                'start_price': df.loc[min_idx, 'close'],
                'peak_price': df.iloc[i]['close'],
                'change_pct': change_24h,
            })

    return surges

def analyze_pre_surge_patterns(df, surge_point, lookback=24):
    """급등 직전 패턴 분석"""
    start_time = surge_point['start_time']
    idx = df.index.get_loc(start_time)

    if idx < lookback:
        return None

    # 급등 직전 lookback 시간의 데이터
    pre_surge = df.iloc[idx-lookback:idx+1]

    if pre_surge.empty or len(pre_surge) < lookback:
        return None

    last_row = pre_surge.iloc[-1]

    pattern = {
        # 가격 위치
        'price_vs_sma7': (last_row['close'] / last_row['sma_7'] - 1) * 100 if pd.notna(last_row['sma_7']) else 0,
        'price_vs_sma25': (last_row['close'] / last_row['sma_25'] - 1) * 100 if pd.notna(last_row['sma_25']) else 0,
        'price_vs_sma99': (last_row['close'] / last_row['sma_99'] - 1) * 100 if pd.notna(last_row['sma_99']) else 0,

        # EMA 정렬
        'ema_9_above_21': last_row['ema_9'] > last_row['ema_21'] if pd.notna(last_row['ema_9']) else False,

        # 볼린저 밴드
        'bb_width': last_row['bb_width'] if pd.notna(last_row['bb_width']) else 0,
        'bb_position': last_row['bb_position'] if pd.notna(last_row['bb_position']) else 0.5,

        # RSI
        'rsi': last_row['rsi'] if pd.notna(last_row['rsi']) else 50,

        # 거래량
        'volume_ratio': last_row['volume_ratio'] if pd.notna(last_row['volume_ratio']) else 1,

        # 최근 변화
        'change_1h': last_row['change_1h'] if pd.notna(last_row['change_1h']) else 0,
        'change_4h': last_row['change_4h'] if pd.notna(last_row['change_4h']) else 0,
        'change_24h': last_row['change_24h'] if pd.notna(last_row['change_24h']) else 0,

        # 변동성
        'volatility': last_row['volatility'] if pd.notna(last_row['volatility']) else 0,
        'atr_pct': last_row['atr_pct'] if pd.notna(last_row['atr_pct']) else 0,

        # MACD
        'macd_positive': last_row['macd'] > 0 if pd.notna(last_row['macd']) else False,
        'macd_hist_positive': last_row['macd_hist'] > 0 if pd.notna(last_row['macd_hist']) else False,

        # 횡보 체크 (볼밴 수축)
        'bb_squeeze': pre_surge['bb_width'].iloc[-10:].mean() if len(pre_surge) >= 10 else 0,

        # 급등 크기
        'surge_pct': surge_point['change_pct'],
    }

    return pattern

def main():
    print("=" * 60)
    print("🚀 급등 코인 패턴 분석기")
    print("=" * 60)

    exchange = get_exchange()
    all_patterns = []

    for coin in SURGE_COINS:
        print(f"\n📊 {coin} 분석 중...")

        # 1시간봉 데이터 가져오기
        df = fetch_ohlcv(exchange, coin, '1h', 500)
        if df is None or len(df) < 100:
            continue

        # 지표 계산
        df = calculate_indicators(df)

        # 급등 포인트 찾기
        surges = find_surge_points(df, threshold=15)  # 24시간 내 15% 이상 급등

        if not surges:
            print(f"  ⚠️ 급등 포인트 없음")
            continue

        print(f"  ✅ {len(surges)}개 급등 포인트 발견")

        for surge in surges:
            pattern = analyze_pre_surge_patterns(df, surge)
            if pattern:
                pattern['coin'] = coin
                all_patterns.append(pattern)

    if not all_patterns:
        print("\n❌ 분석할 패턴이 없습니다.")
        return

    # 패턴 분석 결과
    patterns_df = pd.DataFrame(all_patterns)

    print("\n" + "=" * 60)
    print("📈 급등 직전 공통 패턴 분석 결과")
    print("=" * 60)

    print("\n### 1. 가격 위치 (이동평균 대비)")
    print(f"  - SMA7 대비: {patterns_df['price_vs_sma7'].mean():.2f}% (중앙값: {patterns_df['price_vs_sma7'].median():.2f}%)")
    print(f"  - SMA25 대비: {patterns_df['price_vs_sma25'].mean():.2f}% (중앙값: {patterns_df['price_vs_sma25'].median():.2f}%)")
    print(f"  - SMA99 대비: {patterns_df['price_vs_sma99'].mean():.2f}% (중앙값: {patterns_df['price_vs_sma99'].median():.2f}%)")

    print("\n### 2. RSI")
    print(f"  - 평균: {patterns_df['rsi'].mean():.1f}")
    print(f"  - 중앙값: {patterns_df['rsi'].median():.1f}")
    print(f"  - 범위: {patterns_df['rsi'].min():.1f} ~ {patterns_df['rsi'].max():.1f}")

    print("\n### 3. 볼린저 밴드")
    print(f"  - 밴드 폭: {patterns_df['bb_width'].mean():.2f}% (수축 = 변동성 압축)")
    print(f"  - 밴드 위치: {patterns_df['bb_position'].mean():.2f} (0=하단, 1=상단)")
    print(f"  - BB Squeeze (10시간): {patterns_df['bb_squeeze'].mean():.2f}%")

    print("\n### 4. 거래량")
    print(f"  - 거래량 비율: {patterns_df['volume_ratio'].mean():.2f}x (평균 대비)")
    print(f"  - 거래량 폭발 (>2x): {(patterns_df['volume_ratio'] > 2).sum()}/{len(patterns_df)}")

    print("\n### 5. 모멘텀 지표")
    print(f"  - EMA9 > EMA21: {patterns_df['ema_9_above_21'].sum()}/{len(patterns_df)} ({patterns_df['ema_9_above_21'].mean()*100:.1f}%)")
    print(f"  - MACD 양수: {patterns_df['macd_positive'].sum()}/{len(patterns_df)} ({patterns_df['macd_positive'].mean()*100:.1f}%)")
    print(f"  - MACD Histogram 양수: {patterns_df['macd_hist_positive'].sum()}/{len(patterns_df)}")

    print("\n### 6. 변동성")
    print(f"  - 변동성: {patterns_df['volatility'].mean():.2f}%")
    print(f"  - ATR %: {patterns_df['atr_pct'].mean():.2f}%")

    print("\n### 7. 최근 가격 변화")
    print(f"  - 1시간 변화: {patterns_df['change_1h'].mean():.2f}%")
    print(f"  - 4시간 변화: {patterns_df['change_4h'].mean():.2f}%")
    print(f"  - 24시간 변화: {patterns_df['change_24h'].mean():.2f}%")

    print("\n### 8. 급등 규모")
    print(f"  - 평균 급등률: {patterns_df['surge_pct'].mean():.1f}%")
    print(f"  - 최대 급등률: {patterns_df['surge_pct'].max():.1f}%")

    # 핵심 패턴 요약
    print("\n" + "=" * 60)
    print("🎯 핵심 진입 조건 도출")
    print("=" * 60)

    # 조건 도출
    conditions = []

    # RSI 조건
    rsi_median = patterns_df['rsi'].median()
    if rsi_median < 50:
        conditions.append(f"RSI < 50 (과매도 구간에서 반등)")
    elif rsi_median > 50:
        conditions.append(f"RSI > 50 (모멘텀 확인)")

    # 볼밴 조건
    if patterns_df['bb_squeeze'].mean() < 5:
        conditions.append("볼린저 밴드 수축 (BB Width < 5%)")

    # 거래량 조건
    if patterns_df['volume_ratio'].mean() > 1.5:
        conditions.append(f"거래량 폭발 (> {patterns_df['volume_ratio'].median():.1f}x)")

    # 이평선 조건
    if patterns_df['price_vs_sma25'].median() < 0:
        conditions.append("가격 < SMA25 (눌림목)")

    for i, cond in enumerate(conditions, 1):
        print(f"  {i}. {cond}")

    # 결과 저장
    patterns_df.to_csv('/home/hyeon/project/ichimoku/data/surge_patterns.csv', index=False)
    print(f"\n📁 패턴 데이터 저장: data/surge_patterns.csv")

if __name__ == "__main__":
    main()
