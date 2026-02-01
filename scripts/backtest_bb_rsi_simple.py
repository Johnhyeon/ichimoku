"""
볼린저 밴드 + RSI 평균 회귀 단타 전략 백테스트 (간소화 버전)

전략:
1. BB 상단 돌파 + RSI >= 70 → 숏
2. BB 하단 돌파 + RSI <= 30 → 롱
3. 익절: BB 중간선
4. 손절: 진입 캔들의 고점/저점
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

# 설정
INITIAL_CAPITAL = 2100
LEVERAGE = 20
POSITION_PCT = 0.05

STRATEGY_PARAMS = {
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "sl_buffer_pct": 0.2,
    "cooldown_candles": 2,
    "max_positions": 5,
}

MAJOR_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
    'LTCUSDT', 'UNIUSDT', 'APTUSDT', 'ETCUSDT', 'ATOMUSDT',
    'FILUSDT', 'INJUSDT', 'ARBUSDT', 'OPUSDT', 'AAVEUSDT',
]


def fetch_klines(symbol: str, interval: int, limit: int = 2000) -> Optional[pd.DataFrame]:
    session = HTTP()
    all_data = []
    end_time = None

    while len(all_data) < limit:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000}
        if end_time:
            params['end'] = end_time
        try:
            response = session.get_kline(**params)
            klines = response['result']['list']
        except Exception as e:
            break
        if not klines:
            break
        all_data.extend(klines)
        end_time = int(klines[-1][0]) - 1
        if len(klines) < 1000:
            break
        time.sleep(0.05)

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def calculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()

    # 볼린저 밴드
    df['bb_mid'] = df['close'].rolling(params['bb_period']).mean()
    bb_std = df['close'].rolling(params['bb_period']).std()
    df['bb_upper'] = df['bb_mid'] + params['bb_std'] * bb_std
    df['bb_lower'] = df['bb_mid'] - params['bb_std'] * bb_std

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(params['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df


def run_backtest(all_data: Dict[str, pd.DataFrame], params: dict = STRATEGY_PARAMS) -> tuple:
    # 지표 계산
    for symbol in all_data:
        all_data[symbol] = calculate_indicators(all_data[symbol], params)

    all_bars = []
    for symbol, df in all_data.items():
        df = df.dropna(subset=['bb_mid', 'rsi'])
        for _, row in df.iterrows():
            all_bars.append({
                'symbol': symbol, 'time': row['timestamp'],
                'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close'],
                'bb_mid': row['bb_mid'], 'bb_upper': row['bb_upper'], 'bb_lower': row['bb_lower'],
                'rsi': row['rsi'],
            })

    all_bars.sort(key=lambda x: x['time'])

    time_groups = {}
    for bar in all_bars:
        t = bar['time']
        if t not in time_groups:
            time_groups[t] = {}
        time_groups[t][bar['symbol']] = bar

    sorted_times = sorted(time_groups.keys())

    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    equity_curve = []
    last_exit = {}

    for t in sorted_times:
        bars = time_groups[t]
        closed = []

        # 청산 체크
        for sym, pos in positions.items():
            if sym not in bars:
                continue

            bar = bars[sym]
            high, low = bar['high'], bar['low']
            entry = pos['entry_price']
            reason, exit_price = None, bar['close']

            if pos['side'] == 'long':
                if low <= pos['stop_loss']:
                    reason, exit_price = 'Stop', pos['stop_loss']
                elif high >= pos['take_profit']:
                    reason, exit_price = 'TP', pos['take_profit']
                elif low <= entry * 0.97:
                    reason, exit_price = 'MaxLoss', entry * 0.97
            else:
                if high >= pos['stop_loss']:
                    reason, exit_price = 'Stop', pos['stop_loss']
                elif low <= pos['take_profit']:
                    reason, exit_price = 'TP', pos['take_profit']
                elif high >= entry * 1.03:
                    reason, exit_price = 'MaxLoss', entry * 1.03

            if reason:
                pnl_pct = ((exit_price - entry) / entry if pos['side'] == 'long' else (entry - exit_price) / entry) * 100
                realized_pnl = pnl_pct * LEVERAGE / 100 * pos['position_size']
                cash += pos['position_size'] + realized_pnl
                trades.append({
                    'symbol': sym, 'side': pos['side'],
                    'entry_time': pos['entry_time'], 'exit_time': t,
                    'entry_price': entry, 'exit_price': exit_price,
                    'pnl_pct': round(pnl_pct * LEVERAGE, 2),
                    'pnl_usd': round(realized_pnl, 2),
                    'reason': reason
                })
                closed.append(sym)
                last_exit[sym] = t

        for s in closed:
            del positions[s]

        # 자산 계산
        unrealized = sum(
            ((bars[s]['close'] - p['entry_price']) / p['entry_price'] if p['side'] == 'long'
             else (p['entry_price'] - bars[s]['close']) / p['entry_price']) * LEVERAGE * p['position_size'] / 100
            for s, p in positions.items() if s in bars
        )
        current_equity = cash + sum(p['position_size'] for p in positions.values()) + unrealized
        position_size = current_equity * POSITION_PCT

        # 진입
        if cash >= position_size and len(positions) < params['max_positions']:
            candidates = []

            for sym, bar in bars.items():
                if sym in positions:
                    continue
                if sym in last_exit and (t - last_exit[sym]).total_seconds() < params['cooldown_candles'] * 15 * 60:
                    continue

                price, rsi = bar['close'], bar['rsi']
                bb_upper, bb_lower, bb_mid = bar['bb_upper'], bar['bb_lower'], bar['bb_mid']

                # 숏: 고가가 BB 상단 돌파 + RSI >= 70
                if bar['high'] > bb_upper and rsi >= params['rsi_overbought']:
                    sl = bar['high'] * (1 + params['sl_buffer_pct'] / 100)
                    tp = bb_mid
                    if price > tp:
                        candidates.append({'symbol': sym, 'side': 'short', 'price': price,
                                           'stop_loss': sl, 'take_profit': tp, 'rsi': rsi})

                # 롱: 저가가 BB 하단 돌파 + RSI <= 30
                elif bar['low'] < bb_lower and rsi <= params['rsi_oversold']:
                    sl = bar['low'] * (1 - params['sl_buffer_pct'] / 100)
                    tp = bb_mid
                    if price < tp:
                        candidates.append({'symbol': sym, 'side': 'long', 'price': price,
                                           'stop_loss': sl, 'take_profit': tp, 'rsi': rsi})

            candidates.sort(key=lambda x: abs(x['rsi'] - 50), reverse=True)

            for cand in candidates:
                if cash < position_size or len(positions) >= params['max_positions']:
                    break
                positions[cand['symbol']] = {
                    'side': cand['side'], 'entry_price': cand['price'], 'entry_time': t,
                    'stop_loss': cand['stop_loss'], 'take_profit': cand['take_profit'],
                    'position_size': position_size,
                }
                cash -= position_size

        equity_curve.append({'time': int(t.timestamp()), 'equity': round(current_equity, 2)})

    return trades, equity_curve


def calculate_stats(trades, equity_curve):
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'return_pct': 0,
                'max_dd': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0,
                'long_trades': 0, 'short_trades': 0, 'long_pnl': 0, 'short_pnl': 0}

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    longs = [t for t in trades if t['side'] == 'long']
    shorts = [t for t in trades if t['side'] == 'short']

    total_profit = sum(t['pnl_usd'] for t in wins) if wins else 0
    total_loss = abs(sum(t['pnl_usd'] for t in losses)) if losses else 0

    peak, max_dd = INITIAL_CAPITAL, 0
    for e in equity_curve:
        if e['equity'] > peak:
            peak = e['equity']
        dd = (peak - e['equity']) / peak * 100
        max_dd = max(max_dd, dd)

    final = equity_curve[-1]['equity'] if equity_curve else INITIAL_CAPITAL

    return {
        'total_trades': len(trades),
        'win_rate': round(len(wins) / len(trades) * 100, 1),
        'total_pnl': round(sum(t['pnl_usd'] for t in trades), 2),
        'return_pct': round((final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 1),
        'max_dd': round(max_dd, 1),
        'profit_factor': round(total_profit / total_loss, 2) if total_loss > 0 else 999,
        'avg_win': round(sum(t['pnl_pct'] for t in wins) / len(wins), 2) if wins else 0,
        'avg_loss': round(sum(t['pnl_pct'] for t in losses) / len(losses), 2) if losses else 0,
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'long_pnl': round(sum(t['pnl_usd'] for t in longs), 2),
        'short_pnl': round(sum(t['pnl_usd'] for t in shorts), 2),
        'long_wr': round(len([t for t in longs if t['pnl_pct'] > 0]) / len(longs) * 100, 1) if longs else 0,
        'short_wr': round(len([t for t in shorts if t['pnl_pct'] > 0]) / len(shorts) * 100, 1) if shorts else 0,
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    print("=" * 70)
    print("볼린저 밴드 + RSI 평균 회귀 단타 전략 (15분봉)")
    print("=" * 70)
    print(f"조건: BB 돌파 + RSI >= {STRATEGY_PARAMS['rsi_overbought']} (숏) / <= {STRATEGY_PARAMS['rsi_oversold']} (롱)")
    print(f"익절: BB 중간선 | 손절: 진입 캔들 고/저점")
    print("=" * 70)

    print("\n15분봉 데이터 수집 중...")
    all_data = {}
    for i, symbol in enumerate(MAJOR_COINS):
        print(f"  {i+1}/{len(MAJOR_COINS)} {symbol}...", end='', flush=True)
        df = fetch_klines(symbol, 15, limit=4000)  # 약 42일
        if df is not None and not df.empty:
            all_data[symbol] = df
            print("OK")
        else:
            print("SKIP")

    print(f"\n{len(all_data)}개 코인 로드 완료")
    if all_data:
        first_df = list(all_data.values())[0]
        print(f"기간: {first_df['timestamp'].min()} ~ {first_df['timestamp'].max()}")

    print("\n백테스트 실행 중...")
    trades, equity = run_backtest(all_data)
    stats = calculate_stats(trades, equity)

    days = (max(t['exit_time'] for t in trades) - min(t['entry_time'] for t in trades)).days if trades else 0

    print("\n" + "=" * 70)
    print("결과")
    print("=" * 70)
    print(f"총 거래: {stats['total_trades']}회 ({stats['total_trades']/max(days,1):.1f}회/일)")
    print(f"승률: {stats['win_rate']}%")
    print(f"평균 승리: {stats['avg_win']}% | 평균 손실: {stats['avg_loss']}%")
    print(f"총 수익: ${stats['total_pnl']:,.2f} ({stats['return_pct']}%)")
    print(f"MDD: {stats['max_dd']}% | PF: {stats['profit_factor']}")
    print("-" * 70)
    print(f"LONG:  {stats['long_trades']}회, 승률 {stats['long_wr']}%, ${stats['long_pnl']:,.2f}")
    print(f"SHORT: {stats['short_trades']}회, 승률 {stats['short_wr']}%, ${stats['short_pnl']:,.2f}")
    print("=" * 70)

    if trades:
        from collections import Counter
        reasons = Counter(t['reason'] for t in trades)
        print("\n[청산 사유]")
        for reason, count in reasons.most_common():
            r_trades = [t for t in trades if t['reason'] == reason]
            r_pnl = sum(t['pnl_usd'] for t in r_trades)
            r_wr = len([t for t in r_trades if t['pnl_pct'] > 0]) / len(r_trades) * 100
            print(f"  {reason:8}: {count:3}회 ({count/len(trades)*100:4.1f}%), 승률 {r_wr:5.1f}%, ${r_pnl:>10,.2f}")

        print(f"\n[최근 거래]")
        for t in sorted(trades, key=lambda x: x['exit_time'], reverse=True)[:5]:
            print(f"  {t['entry_time'].strftime('%m/%d %H:%M')} {t['symbol']:<10} {t['side']:<5} "
                  f"{t['pnl_pct']:>+6.1f}% ${t['pnl_usd']:>8.2f} ({t['reason']})")
