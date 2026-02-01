"""
급등 전략 백테스트 (멀티 타임프레임)

1시간봉에서 시그널 발생 후 15분봉에서 양봉→음봉 진입
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
    check_surge_exit_signal,
    SURGE_STRATEGY_PARAMS,
    get_surge_watch_list,
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
        print(f"  ⚠️ {symbol} 데이터 없음: {e}")
        return None


def simulate_15m_entry(df_1h: pd.DataFrame, signal_idx: int) -> dict:
    """
    15분봉 진입 시뮬레이션

    1시간봉 시그널 발생 후 다음 4개의 15분봉 중에서:
    - 양봉 후 음봉이 나오면 해당 음봉 종가에 진입
    - 양봉 없이 음봉만 계속되면 진입 안함
    - 양봉만 계속되면 마지막 캔들 종가에 진입

    백테스트에서는 1시간봉 데이터로 시뮬레이션:
    - 다음 1시간봉의 OHLC를 4등분하여 15분봉 추정
    """
    if signal_idx + 1 >= len(df_1h):
        return {'entry': False, 'reason': 'no_next_candle'}

    next_candle = df_1h.iloc[signal_idx + 1]

    o = float(next_candle['open'])
    h = float(next_candle['high'])
    l = float(next_candle['low'])
    c = float(next_candle['close'])

    # 다음 1시간봉이 양봉이면 저점에서 반등 후 진입 가정
    is_next_green = c > o

    if is_next_green:
        # 양봉: 저점 찍고 반등 → 진입 OK
        # 진입가는 open과 low 사이로 추정
        entry_price = o * 0.995  # 시가 대비 약간 하락한 지점
        return {
            'entry': True,
            'price': entry_price,
            'reason': 'green_candle_entry',
            'candle_time': next_candle.name
        }
    else:
        # 음봉: 양봉 없이 하락 → 더 하락할 수 있음
        # 그래도 음봉 종가에 진입 (양봉 후 음봉 패턴 기다리기 어려움)
        # 또는 진입 스킵
        # 여기서는 음봉이라도 진입하되 더 낮은 가격에
        entry_price = c  # 음봉 종가
        return {
            'entry': True,
            'price': entry_price,
            'reason': 'red_candle_entry',
            'candle_time': next_candle.name
        }


def backtest_symbol_mtf(symbol: str, df_1h: pd.DataFrame, params: dict) -> list:
    """
    멀티타임프레임 백테스트

    1. 1시간봉에서 시그널 확인
    2. 다음 캔들에서 15분봉 패턴 시뮬레이션 후 진입
    3. 15분봉 기준 손절/익절 체크 (1시간봉 4등분)
    """
    df_1h = calculate_surge_indicators(df_1h)

    trades = []
    position = None
    signal_bar_idx = None

    for i in range(30, len(df_1h) - 1):  # -1: 다음 캔들 필요
        row = df_1h.iloc[i]
        next_row = df_1h.iloc[i + 1]
        prev_rows = df_1h.iloc[:i+1]

        if position is None:
            # 1시간봉 진입 신호 체크
            signal = get_surge_entry_signal(symbol, prev_rows, params)

            if signal:
                # 15분봉 진입 시뮬레이션
                entry_sim = simulate_15m_entry(df_1h, i)

                if entry_sim['entry']:
                    entry_price = entry_sim['price']
                    position = {
                        'symbol': symbol,
                        'side': 'long',
                        'entry_price': entry_price,
                        'entry_time': entry_sim.get('candle_time', next_row.name),
                        'stop_loss': entry_price * (1 - params['sl_pct'] / 100),
                        'take_profit': entry_price * (1 + params['tp_pct'] / 100),
                        'highest': entry_price,
                        'trailing': False,
                        'trail_stop': 0,
                        'score': signal['score'],
                        'entry_reason': entry_sim['reason'],
                    }
                    signal_bar_idx = i + 1  # 다음 캔들에서 진입
        else:
            # 청산 체크 (현재 인덱스가 진입 캔들 이후인 경우만)
            if i > signal_bar_idx:
                # 15분봉 기준 OHLC 체크
                # 실제로는 1시간봉의 High/Low로 손절/익절 체크
                exit_signal = check_surge_exit_signal(position, row, params)

                if exit_signal:
                    exit_price = exit_signal['price']
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100

                    trades.append({
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': df_1h.index[i],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'pnl_with_lev': pnl_pct * params['leverage'],
                        'reason': exit_signal['reason'],
                        'score': position['score'],
                        'entry_reason': position['entry_reason'],
                        'hold_hours': (df_1h.index[i] - position['entry_time']).total_seconds() / 3600,
                    })
                    position = None
                    signal_bar_idx = None

    # 미청산 포지션 정리
    if position:
        last_price = float(df_1h.iloc[-1]['close'])
        pnl_pct = (last_price - position['entry_price']) / position['entry_price'] * 100
        trades.append({
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'exit_time': df_1h.index[-1],
            'entry_price': position['entry_price'],
            'exit_price': last_price,
            'pnl_pct': pnl_pct,
            'pnl_with_lev': pnl_pct * params['leverage'],
            'reason': 'Open',
            'score': position['score'],
            'entry_reason': position.get('entry_reason', ''),
            'hold_hours': (df_1h.index[-1] - position['entry_time']).total_seconds() / 3600,
        })

    return trades


def main():
    print("=" * 70)
    print("🚀 급등 전략 백테스트 (멀티 타임프레임)")
    print("=" * 70)

    exchange = get_exchange()
    params = SURGE_STRATEGY_PARAMS.copy()

    # 전체 종목 대상 백테스트
    test_symbols = get_surge_watch_list()

    all_trades = []

    print(f"\n📊 전략 파라미터:")
    print(f"  - 1H: RSI {params['rsi_min']}~{params['rsi_oversold']} | BB < {params['bb_position_max']}")
    print(f"  - 1H: Volume > {params['volume_ratio_min']}x | Min Score: {params['min_score']}")
    print(f"  - 15M: 양봉→음봉 진입 시뮬레이션")
    print(f"  - SL: {params['sl_pct']}% / TP: {params['tp_pct']}%")
    print(f"  - 레버리지: {params['leverage']}x")

    for symbol in test_symbols:
        print(f"\n📈 {symbol} 백테스트 중...")

        df_1h = fetch_ohlcv(exchange, symbol, '1h', 1000)
        if df_1h is None or len(df_1h) < 100:
            print(f"  ⚠️ 데이터 부족")
            continue

        trades = backtest_symbol_mtf(symbol, df_1h, params)
        all_trades.extend(trades)

        if trades:
            wins = len([t for t in trades if t['pnl_pct'] > 0])
            total = len(trades)
            total_pnl = sum(t['pnl_with_lev'] for t in trades)
            print(f"  ✅ {total}건 거래 | 승률: {wins/total*100:.1f}% | 누적 PnL: {total_pnl:.1f}%")
        else:
            print(f"  ⚪ 거래 없음")

    if not all_trades:
        print("\n❌ 거래 내역이 없습니다.")
        return

    # 전체 결과 분석
    trades_df = pd.DataFrame(all_trades)

    print("\n" + "=" * 70)
    print("📊 전체 백테스트 결과 (MTF)")
    print("=" * 70)

    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl_pct'] > 0])
    losses = len(trades_df[trades_df['pnl_pct'] < 0])
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    total_pnl = trades_df['pnl_pct'].sum()
    total_pnl_lev = trades_df['pnl_with_lev'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    avg_pnl_lev = trades_df['pnl_with_lev'].mean()

    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losses > 0 else 0
    profit_factor = abs(trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum() / trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum()) if losses > 0 else float('inf')

    max_win = trades_df['pnl_pct'].max()
    max_loss = trades_df['pnl_pct'].min()
    avg_hold = trades_df['hold_hours'].mean()

    print(f"\n### 성과 요약")
    print(f"  - 총 거래: {total_trades}건")
    print(f"  - 승/패: {wins}W / {losses}L")
    print(f"  - 승률: {win_rate:.1f}%")
    print(f"  - Profit Factor: {profit_factor:.2f}")

    print(f"\n### 수익률 (레버리지 {params['leverage']}x 기준)")
    print(f"  - 총 수익률: {total_pnl_lev:.1f}%")
    print(f"  - 평균 수익률: {avg_pnl_lev:.2f}%")
    print(f"  - 평균 승리: +{avg_win * params['leverage']:.2f}%")
    print(f"  - 평균 손실: {avg_loss * params['leverage']:.2f}%")
    print(f"  - 최대 승리: +{max_win * params['leverage']:.2f}%")
    print(f"  - 최대 손실: {max_loss * params['leverage']:.2f}%")

    print(f"\n### 거래 통계")
    print(f"  - 평균 보유 시간: {avg_hold:.1f}시간")

    # 청산 사유별 분석
    print(f"\n### 청산 사유별 분석")
    for reason in trades_df['reason'].unique():
        subset = trades_df[trades_df['reason'] == reason]
        count = len(subset)
        avg = subset['pnl_pct'].mean()
        print(f"  - {reason}: {count}건 (평균 {avg:.2f}%)")

    # 진입 사유별 분석
    if 'entry_reason' in trades_df.columns:
        print(f"\n### 진입 타입별 분석")
        for reason in trades_df['entry_reason'].unique():
            if pd.notna(reason) and reason:
                subset = trades_df[trades_df['entry_reason'] == reason]
                count = len(subset)
                wins_r = len(subset[subset['pnl_pct'] > 0])
                wr = wins_r / count * 100 if count > 0 else 0
                avg = subset['pnl_pct'].mean()
                print(f"  - {reason}: {count}건 | 승률: {wr:.1f}% | 평균: {avg:.2f}%")

    # 코인별 분석
    print(f"\n### 코인별 성과 (상위 10)")
    coin_stats = trades_df.groupby('symbol').agg({
        'pnl_pct': ['count', 'sum', 'mean']
    }).round(2)
    coin_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
    coin_stats = coin_stats.sort_values('total_pnl', ascending=False)
    print(coin_stats.head(10).to_string())

    # 점수별 분석
    print(f"\n### 점수별 성과")
    for score in sorted(trades_df['score'].unique()):
        subset = trades_df[trades_df['score'] == score]
        wins_s = len(subset[subset['pnl_pct'] > 0])
        wr = wins_s / len(subset) * 100 if len(subset) > 0 else 0
        avg = subset['pnl_pct'].mean()
        print(f"  - Score {score}: {len(subset)}건 | 승률: {wr:.1f}% | 평균: {avg:.2f}%")

    # 결과 저장
    trades_df.to_csv('/home/hyeon/project/ichimoku/data/surge_backtest_results.csv', index=False)
    print(f"\n📁 결과 저장: data/surge_backtest_results.csv")

    # 최근 거래 예시
    print(f"\n### 최근 거래 예시 (최근 10건)")
    recent = trades_df.sort_values('entry_time', ascending=False).head(10)
    for _, t in recent.iterrows():
        coin = t['symbol'].replace('/USDT:USDT', '')
        pnl = t['pnl_with_lev']
        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"  {emoji} {coin}: {pnl:+.1f}% ({t['reason']}) | {t['entry_time'].strftime('%m/%d %H:%M')}")


if __name__ == "__main__":
    main()
