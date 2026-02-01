#!/usr/bin/env python3
"""
급등 신호 정확도 백테스트

신호 발생 후 실제로 급등이 이어졌는지 검증합니다.
- 신호 발생 시점 탐지
- 이후 24h/48h/72h 최대 상승폭 측정
- 급등 성공률 계산
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.surge_strategy import (
    calculate_surge_indicators,
    get_surge_entry_signal,
    SURGE_STRATEGY_PARAMS,
    SURGE_WATCH_LIST
)
import warnings
warnings.filterwarnings('ignore')


def get_exchange():
    return ccxt.bybit({'options': {'defaultType': 'swap'}})


def fetch_ohlcv(exchange, symbol, timeframe='1h', limit=1000):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        return df
    except Exception as e:
        return None


def find_all_signals(symbol: str, df: pd.DataFrame, params: dict) -> list:
    """
    과거 데이터에서 모든 진입 신호 찾기
    """
    df = calculate_surge_indicators(df)
    signals = []

    for i in range(30, len(df) - 72):  # 72시간 후까지 볼 수 있도록 여유
        row = df.iloc[i]
        prev_rows = df.iloc[:i+1]

        signal = get_surge_entry_signal(symbol, prev_rows, params)

        if signal:
            entry_time = df.index[i]
            entry_price = float(row['close'])

            # 이후 24h/48h/72h 데이터 확인
            future_24h = df.iloc[i+1:i+25] if i+25 <= len(df) else df.iloc[i+1:]
            future_48h = df.iloc[i+1:i+49] if i+49 <= len(df) else df.iloc[i+1:]
            future_72h = df.iloc[i+1:i+73] if i+73 <= len(df) else df.iloc[i+1:]

            # 최대 상승폭 계산
            max_24h = future_24h['high'].max() if len(future_24h) > 0 else entry_price
            max_48h = future_48h['high'].max() if len(future_48h) > 0 else entry_price
            max_72h = future_72h['high'].max() if len(future_72h) > 0 else entry_price

            # 최저점 (손절 체크용)
            min_24h = future_24h['low'].min() if len(future_24h) > 0 else entry_price
            min_48h = future_48h['low'].min() if len(future_48h) > 0 else entry_price

            # 상승률
            gain_24h = (max_24h - entry_price) / entry_price * 100
            gain_48h = (max_48h - entry_price) / entry_price * 100
            gain_72h = (max_72h - entry_price) / entry_price * 100

            # 최대 하락폭 (drawdown)
            dd_24h = (entry_price - min_24h) / entry_price * 100
            dd_48h = (entry_price - min_48h) / entry_price * 100

            signals.append({
                'symbol': symbol,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'score': signal['score'],
                'rsi': signal['rsi'],
                'bb_position': signal['bb_position'],
                'volume_ratio': signal['volume_ratio'],
                'max_gain_24h': gain_24h,
                'max_gain_48h': gain_48h,
                'max_gain_72h': gain_72h,
                'max_dd_24h': dd_24h,
                'max_dd_48h': dd_48h,
            })

    return signals


def main():
    print("=" * 70)
    print("🎯 급등 신호 정확도 백테스트")
    print("=" * 70)

    exchange = get_exchange()
    params = SURGE_STRATEGY_PARAMS.copy()

    # 테스트할 코인들 (급등 이력이 있는 코인들 위주)
    test_symbols = [
        "ZORA/USDT:USDT", "CYS/USDT:USDT", "ZKP/USDT:USDT", "MEGA/USDT:USDT",
        "C98/USDT:USDT", "HANA/USDT:USDT", "SOPH/USDT:USDT", "IN/USDT:USDT",
        "OPEN/USDT:USDT", "WIF/USDT:USDT", "BOME/USDT:USDT", "MEW/USDT:USDT",
        "ARB/USDT:USDT", "OP/USDT:USDT", "SUI/USDT:USDT", "SEI/USDT:USDT",
        "TIA/USDT:USDT", "JUP/USDT:USDT", "PENDLE/USDT:USDT", "INJ/USDT:USDT",
        "APT/USDT:USDT", "STRK/USDT:USDT", "BLUR/USDT:USDT", "DYDX/USDT:USDT",
        "GMX/USDT:USDT", "MAGIC/USDT:USDT", "IMX/USDT:USDT", "AEVO/USDT:USDT",
    ]

    all_signals = []

    print(f"\n📊 스캔 조건:")
    print(f"  - RSI: {params['rsi_min']}~{params['rsi_oversold']}")
    print(f"  - BB Position: < {params['bb_position_max']}")
    print(f"  - Volume: > {params['volume_ratio_min']}x")
    print(f"  - Min Score: {params.get('min_score', 5)}")

    for symbol in test_symbols:
        print(f"\n📈 {symbol} 분석 중...")

        df = fetch_ohlcv(exchange, symbol, '1h', 1000)
        if df is None or len(df) < 150:
            print(f"  ⚠️ 데이터 부족")
            continue

        signals = find_all_signals(symbol, df, params)

        if signals:
            print(f"  ✅ {len(signals)}개 신호 발견")
            all_signals.extend(signals)
        else:
            print(f"  ⚪ 신호 없음")

    if not all_signals:
        print("\n❌ 분석할 신호가 없습니다.")
        return

    # 결과 분석
    signals_df = pd.DataFrame(all_signals)

    print("\n" + "=" * 70)
    print("📊 신호 정확도 분석 결과")
    print("=" * 70)

    total = len(signals_df)

    # 급등 성공 기준별 분석
    thresholds = [5, 10, 15, 20, 30, 50]

    print(f"\n### 신호 발생 후 급등 성공률")
    print("-" * 70)
    print(f"{'기준':>8} | {'24h 이내':>12} | {'48h 이내':>12} | {'72h 이내':>12}")
    print("-" * 70)

    for th in thresholds:
        hit_24h = (signals_df['max_gain_24h'] >= th).sum()
        hit_48h = (signals_df['max_gain_48h'] >= th).sum()
        hit_72h = (signals_df['max_gain_72h'] >= th).sum()

        pct_24h = hit_24h / total * 100
        pct_48h = hit_48h / total * 100
        pct_72h = hit_72h / total * 100

        print(f"{th:>6}%+ | {hit_24h:>4}/{total} ({pct_24h:>5.1f}%) | {hit_48h:>4}/{total} ({pct_48h:>5.1f}%) | {hit_72h:>4}/{total} ({pct_72h:>5.1f}%)")

    # 평균 상승폭
    print(f"\n### 평균 최대 상승폭")
    print(f"  - 24시간 이내: +{signals_df['max_gain_24h'].mean():.1f}% (중앙값: +{signals_df['max_gain_24h'].median():.1f}%)")
    print(f"  - 48시간 이내: +{signals_df['max_gain_48h'].mean():.1f}% (중앙값: +{signals_df['max_gain_48h'].median():.1f}%)")
    print(f"  - 72시간 이내: +{signals_df['max_gain_72h'].mean():.1f}% (중앙값: +{signals_df['max_gain_72h'].median():.1f}%)")

    # 최대 하락폭 (리스크)
    print(f"\n### 최대 하락폭 (리스크)")
    print(f"  - 24시간 내 평균 DD: -{signals_df['max_dd_24h'].mean():.1f}%")
    print(f"  - 48시간 내 평균 DD: -{signals_df['max_dd_48h'].mean():.1f}%")

    # 손절 3% 내 급등 10% 성공률
    sl_3_tp_10 = signals_df[(signals_df['max_dd_24h'] < 3) & (signals_df['max_gain_48h'] >= 10)]
    print(f"\n### 실전 시나리오 (SL 3%, TP 10%)")
    print(f"  - 손절 안 맞고 10%+ 도달: {len(sl_3_tp_10)}/{total} ({len(sl_3_tp_10)/total*100:.1f}%)")

    # 점수별 성공률
    print(f"\n### 점수별 급등 성공률 (10%+ 기준)")
    for score in sorted(signals_df['score'].unique()):
        subset = signals_df[signals_df['score'] == score]
        hits = (subset['max_gain_48h'] >= 10).sum()
        rate = hits / len(subset) * 100 if len(subset) > 0 else 0
        avg_gain = subset['max_gain_48h'].mean()
        print(f"  - Score {score}: {hits}/{len(subset)} ({rate:.1f}%) | 평균 +{avg_gain:.1f}%")

    # 코인별 성공률
    print(f"\n### 코인별 급등 성공률 (10%+ 기준, 상위 10)")
    coin_stats = []
    for symbol in signals_df['symbol'].unique():
        subset = signals_df[signals_df['symbol'] == symbol]
        hits = (subset['max_gain_48h'] >= 10).sum()
        rate = hits / len(subset) * 100 if len(subset) > 0 else 0
        avg_gain = subset['max_gain_48h'].mean()
        coin_stats.append({
            'symbol': symbol,
            'signals': len(subset),
            'hits': hits,
            'rate': rate,
            'avg_gain': avg_gain,
        })

    coin_stats_df = pd.DataFrame(coin_stats).sort_values('rate', ascending=False)
    for _, row in coin_stats_df.head(10).iterrows():
        coin = row['symbol'].replace('/USDT:USDT', '')
        print(f"  - {coin:8s}: {row['hits']:.0f}/{row['signals']:.0f} ({row['rate']:.1f}%) | 평균 +{row['avg_gain']:.1f}%")

    # 베스트/워스트 케이스
    print(f"\n### 베스트 케이스 (48h 최대 상승)")
    best = signals_df.nlargest(5, 'max_gain_48h')
    for _, row in best.iterrows():
        coin = row['symbol'].replace('/USDT:USDT', '')
        print(f"  🚀 {coin}: +{row['max_gain_48h']:.1f}% | Score: {row['score']} | RSI: {row['rsi']:.1f} | {row['entry_time'].strftime('%m/%d %H:%M')}")

    print(f"\n### 워스트 케이스 (48h 최대 하락)")
    worst = signals_df.nlargest(5, 'max_dd_48h')
    for _, row in worst.iterrows():
        coin = row['symbol'].replace('/USDT:USDT', '')
        print(f"  📉 {coin}: -{row['max_dd_48h']:.1f}% (max gain: +{row['max_gain_48h']:.1f}%) | {row['entry_time'].strftime('%m/%d %H:%M')}")

    # 시간대별 분석
    signals_df['hour'] = signals_df['entry_time'].dt.hour
    print(f"\n### 시간대별 신호 발생 (UTC)")
    hour_counts = signals_df.groupby('hour').size()
    peak_hours = hour_counts.nlargest(3)
    for hour, count in peak_hours.items():
        print(f"  - {hour:02d}:00 UTC: {count}건")

    # 결과 저장
    signals_df.to_csv('/home/hyeon/project/ichimoku/data/signal_accuracy_results.csv', index=False)
    print(f"\n📁 결과 저장: data/signal_accuracy_results.csv")

    # 최종 요약
    print("\n" + "=" * 70)
    print("📋 최종 요약")
    print("=" * 70)

    hit_10_48h = (signals_df['max_gain_48h'] >= 10).sum()
    hit_10_rate = hit_10_48h / total * 100

    print(f"""
  총 신호: {total}건

  📈 급등 캐치 성공률:
     - 10%+ 달성 (48h): {hit_10_48h}/{total} ({hit_10_rate:.1f}%)
     - 평균 최대 상승: +{signals_df['max_gain_48h'].mean():.1f}%

  📉 리스크:
     - 평균 최대 하락: -{signals_df['max_dd_48h'].mean():.1f}%

  🎯 추천:
     - Score 6 이상 신호만 진입
     - 손절 3% / 익절 10% 트레일링
""")


if __name__ == "__main__":
    main()
