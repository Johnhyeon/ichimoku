#!/usr/bin/env python3
"""
실시간 급등 후보 스캐너 (멀티 타임프레임)

1시간봉에서 시그널 확인 후 15분봉에서 양봉→음봉 진입 타이밍 확인
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import ccxt
import pandas as pd
from datetime import datetime
from src.surge_strategy import (
    calculate_surge_indicators,
    get_surge_entry_signal,
    check_15m_entry_timing,
    get_surge_entry_signal_mtf,
    SURGE_STRATEGY_PARAMS,
    get_surge_watch_list,
)
import warnings
warnings.filterwarnings('ignore')


def get_exchange():
    return ccxt.bybit({'options': {'defaultType': 'swap'}})


def fetch_ohlcv(exchange, symbol, timeframe='1h', limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        return df
    except:
        return None


def scan_all_coins():
    """전체 코인 스캔 (멀티 타임프레임)"""
    print("=" * 70)
    print(f"🔍 급등 후보 스캔 (MTF) - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    exchange = get_exchange()
    params = SURGE_STRATEGY_PARAMS

    print(f"\n📊 스캔 조건:")
    print(f"  - 1H: RSI {params['rsi_min']}~{params['rsi_oversold']} | BB < {params['bb_position_max']}")
    print(f"  - 1H: Volume > {params['volume_ratio_min']}x | Score >= {params['min_score']}")
    print(f"  - 15M: 양봉 후 음봉 진입")

    candidates = []          # 바로 진입 가능
    waiting_candidates = []  # 1H 조건 충족, 15M 대기 중

    watch_list = get_surge_watch_list()
    print(f"\n📡 {len(watch_list)}개 코인 스캔 중...\n")

    for symbol in watch_list:
        try:
            # 1시간봉 데이터
            df_1h = fetch_ohlcv(exchange, symbol, '1h', 100)
            if df_1h is None or len(df_1h) < 30:
                continue

            df_1h = calculate_surge_indicators(df_1h)

            # 1시간봉 기본 신호 확인
            signal_1h = get_surge_entry_signal(symbol, df_1h, params)
            if signal_1h is None:
                continue

            # 15분봉 데이터
            df_15m = fetch_ohlcv(exchange, symbol, '15m', 50)
            if df_15m is None or len(df_15m) < 5:
                continue

            # 15분봉 타이밍 확인
            timing = check_15m_entry_timing(df_15m)

            coin = symbol.replace('/USDT:USDT', '')
            row = df_1h.iloc[-1]
            price = float(row['close'])

            if timing['ready']:
                # 진입 가능
                candidates.append({
                    'symbol': symbol,
                    'price': timing['entry_price'],
                    'rsi': signal_1h['rsi'],
                    'bb_pos': signal_1h['bb_position'],
                    'vol_ratio': signal_1h['volume_ratio'],
                    'sma25_pct': signal_1h['price_vs_sma25'],
                    'score': signal_1h['score'],
                    'sl': signal_1h['stop_loss'],
                    'tp': signal_1h['take_profit'],
                    'pattern': timing.get('pattern', ''),
                })
            else:
                # 1H 조건 충족, 15M 대기
                waiting_candidates.append({
                    'symbol': symbol,
                    'price': price,
                    'rsi': signal_1h['rsi'],
                    'bb_pos': signal_1h['bb_position'],
                    'vol_ratio': signal_1h['volume_ratio'],
                    'sma25_pct': signal_1h['price_vs_sma25'],
                    'score': signal_1h['score'],
                    'wait_reason': timing['reason'],
                })

        except Exception as e:
            continue

    # 결과 출력
    if candidates:
        print("🎯 진입 신호 발생! (15분봉 양봉→음봉 확인됨)")
        print("-" * 70)
        candidates.sort(key=lambda x: x['score'], reverse=True)

        for c in candidates:
            coin = c['symbol'].replace('/USDT:USDT', '')
            print(f"\n  🟢 {coin}")
            print(f"     가격: ${c['price']:.4f}")
            print(f"     RSI: {c['rsi']:.1f} | BB: {c['bb_pos']:.2f} | Vol: {c['vol_ratio']:.1f}x")
            print(f"     SMA25: {c['sma25_pct']:.1f}% | Score: {c['score']}")
            print(f"     패턴: {c['pattern']}")
            print(f"     → SL: ${c['sl']:.4f} (-5%) | TP: ${c['tp']:.4f} (+12%)")
    else:
        print("⚪ 현재 진입 가능 신호 없음")

    if waiting_candidates:
        print("\n" + "-" * 70)
        print("⏳ 15분봉 대기 중 (1H 조건 충족)")
        waiting_candidates.sort(key=lambda x: x['score'], reverse=True)

        for c in waiting_candidates[:15]:
            coin = c['symbol'].replace('/USDT:USDT', '')
            reason_map = {
                'waiting_red_candle': '🟢 양봉 진행중 → 음봉 대기',
                'waiting_green_candle': '⚪ 양봉 대기',
                'consecutive_red': '🔴 연속 음봉',
            }
            reason = reason_map.get(c['wait_reason'], c['wait_reason'])
            print(f"  {coin:12s} | Score: {c['score']:2d} | RSI: {c['rsi']:5.1f} | {reason}")

    print("\n" + "=" * 70)
    print(f"✅ 스캔 완료 - 진입 가능: {len(candidates)}개 | 대기 중: {len(waiting_candidates)}개")

    return candidates, waiting_candidates


if __name__ == "__main__":
    scan_all_coins()
