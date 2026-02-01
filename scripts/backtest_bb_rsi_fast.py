"""
볼린저 밴드 + RSI 평균 회귀 단타 전략 백테스트 (캐시 사용 - 빠름)
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd

from scripts.data_cache import load_cached_data, MAJOR_COINS

logger = logging.getLogger(__name__)

# 설정
INITIAL_CAPITAL = 2100
LEVERAGE = 20
POSITION_PCT = 0.07  # 7% (MDD 15% 이하 최적)

# 전략 파라미터
STRATEGY_PARAMS = {
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_overbought": 70,  # 4코인용 완화
    "rsi_oversold": 30,    # 4코인용 완화
    "sl_buffer_pct": 0.05, # 최적값
    "cooldown_candles": 1, # 쿨다운 단축
    "max_positions": 8,    # 동시 포지션 증가
    "short_only": False,
    # 횡보장 필터 (BB 폭으로 판단)
    "use_sideways_filter": True,
    "bb_squeeze_min": 1.5,
    "bb_squeeze_max": 4.0,
}


def calculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """볼린저 밴드와 RSI 계산"""
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

    # RSI 반전 신호
    df['rsi_prev'] = df['rsi'].shift(1)
    df['rsi_short_reversal'] = (df['rsi_prev'] >= params['rsi_overbought']) & (df['rsi'] < df['rsi_prev'])
    df['rsi_long_reversal'] = (df['rsi_prev'] <= params['rsi_oversold']) & (df['rsi'] > df['rsi_prev'])

    # BB 돌파 비율
    df['upper_breach'] = (df['high'] - df['bb_upper']) / (df['bb_upper'] - df['bb_lower'])
    df['lower_breach'] = (df['bb_lower'] - df['low']) / (df['bb_upper'] - df['bb_lower'])

    # BB 폭 (횡보장 판단용)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

    return df


def run_backtest(all_data: Dict[str, pd.DataFrame], params: dict = STRATEGY_PARAMS) -> tuple:
    """백테스트 실행"""
    # 지표 계산
    for symbol in all_data:
        all_data[symbol] = calculate_indicators(all_data[symbol], params)

    # 시간순 정렬
    all_bars = []
    for symbol, df in all_data.items():
        df = df.dropna(subset=['bb_mid', 'rsi'])
        for _, row in df.iterrows():
            all_bars.append({
                'symbol': symbol, 'time': row['timestamp'],
                'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close'],
                'bb_mid': row['bb_mid'], 'bb_upper': row['bb_upper'], 'bb_lower': row['bb_lower'],
                'rsi': row['rsi'],
                'rsi_short_reversal': row['rsi_short_reversal'],
                'rsi_long_reversal': row['rsi_long_reversal'],
                'upper_breach': row['upper_breach'],
                'lower_breach': row['lower_breach'],
                'bb_width': row['bb_width'],
            })

    all_bars.sort(key=lambda x: x['time'])

    # 시간별 그룹화
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
            else:  # short
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

                price = bar['close']
                bb_mid, bb_upper, bb_lower = bar['bb_mid'], bar['bb_upper'], bar['bb_lower']

                # 횡보장 필터 (BB 폭으로 판단)
                if params.get('use_sideways_filter', False):
                    bb_width = bar['bb_width']
                    if bb_width < params.get('bb_squeeze_min', 1.5) or bb_width > params.get('bb_squeeze_max', 5.0):
                        continue  # 횡보장 아니면 스킵

                # 숏: BB 상단 돌파 + RSI 반전
                if bar['upper_breach'] >= 0.3 and bar['rsi_short_reversal']:
                    sl = bar['high'] * (1 + params['sl_buffer_pct'] / 100)
                    tp = bb_mid
                    if price > tp:
                        candidates.append({'symbol': sym, 'side': 'short', 'price': price,
                                           'stop_loss': sl, 'take_profit': tp, 'rsi': bar['rsi']})

                # 롱: BB 하단 돌파 + RSI 반전
                elif not params.get('short_only', False) and bar['lower_breach'] >= 0.2 and bar['rsi_long_reversal']:
                    sl = bar['low'] * (1 - params['sl_buffer_pct'] / 100)
                    tp = bb_mid
                    if price < tp:
                        candidates.append({'symbol': sym, 'side': 'long', 'price': price,
                                           'stop_loss': sl, 'take_profit': tp, 'rsi': bar['rsi']})

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
    """통계 계산"""
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
    import time as time_module
    logging.basicConfig(level=logging.ERROR)

    print("=" * 70)
    print("볼린저 밴드 + RSI 평균 회귀 단타 (캐시 사용)")
    print("=" * 70)

    start = time_module.time()

    # 캐시에서 데이터 로드 (매우 빠름!)
    print("캐시에서 데이터 로드 중...", end='', flush=True)
    # 4개 코인만 사용: BTC, ETH, BNB, HYPE
    TARGET_COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']
    data_15m = load_cached_data(TARGET_COINS, '15m')
    print(f" {len(data_15m)}개 코인 로드 ({time_module.time()-start:.1f}초)")

    if data_15m:
        first_df = list(data_15m.values())[0]
        print(f"데이터 기간: {first_df['timestamp'].min()} ~ {first_df['timestamp'].max()}")

    print("\n백테스트 실행 중...", end='', flush=True)
    bt_start = time_module.time()
    trades, equity = run_backtest(data_15m)
    print(f" 완료 ({time_module.time()-bt_start:.1f}초)")

    stats = calculate_stats(trades, equity)
    days = (max(t['exit_time'] for t in trades) - min(t['entry_time'] for t in trades)).days if trades else 0

    print("\n" + "=" * 70)
    print("결과")
    print("=" * 70)
    print(f"총 거래: {stats['total_trades']}회 ({stats['total_trades']/max(days,1):.2f}회/일, {days}일)")
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
            print(f"  {reason:8}: {count:4}회 ({count/len(trades)*100:5.1f}%), 승률 {r_wr:5.1f}%, ${r_pnl:>10,.2f}")

    print(f"\n총 소요시간: {time_module.time()-start:.1f}초")
