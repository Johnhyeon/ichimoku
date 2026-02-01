"""
Ichimoku 전략 수수료 영향 분석
- 거래 빈도가 낮아서 수수료 영향이 적을 수 있음
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import time
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

from src.ichimoku import calculate_ichimoku

# 설정
INITIAL_CAPITAL = 5_000_000
LEVERAGE = 20
POSITION_PCT = 0.05

STRATEGY_PARAMS = {
    "min_cloud_thickness": 0.2,
    "min_sl_pct": 0.3,
    "max_sl_pct": 8.0,
    "sl_buffer": 0.2,
    "rr_ratio": 2.0,
    "trail_pct": 1.5,
    "cooldown_hours": 4,
    "max_positions": 5,
}

COINS_4 = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']

COINS_20 = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
    'ADAUSDT', 'DOGEUSDT', 'TONUSDT', 'TRXUSDT', 'AVAXUSDT',
    'DOTUSDT', 'LINKUSDT', 'BCHUSDT', 'SUIUSDT', 'NEARUSDT',
    'LTCUSDT', 'UNIUSDT', 'APTUSDT', 'ICPUSDT', 'ETCUSDT',
]


def fetch_klines(symbol: str, interval: int, limit: int = 4000):
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
        except:
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


def run_backtest_with_fees(all_data, params, initial, leverage, pos_pct, fee_rate):
    """수수료 포함 Ichimoku 백테스트"""
    all_bars = []

    btc_trends = {}
    if 'BTCUSDT' in all_data:
        btc_df = all_data['BTCUSDT'].copy()
        btc_df['sma_26'] = btc_df['close'].rolling(26).mean()
        btc_df['sma_52'] = btc_df['close'].rolling(52).mean()
        for _, row in btc_df.iterrows():
            if pd.notna(row['sma_26']) and pd.notna(row['sma_52']):
                btc_trends[row['timestamp']] = row['sma_26'] > row['sma_52']

    for symbol, df in all_data.items():
        df = calculate_ichimoku(df)
        df = df.dropna(subset=['tenkan', 'kijun', 'cloud_top', 'cloud_bottom'])
        for _, row in df.iterrows():
            all_bars.append({
                'symbol': symbol, 'time': row['timestamp'],
                'open': row['open'], 'high': row['high'], 'low': row['low'],
                'close': row['close'], 'cloud_top': row['cloud_top'],
                'cloud_bottom': row['cloud_bottom'], 'cloud_thickness': row['cloud_thickness'],
                'cloud_green': row['cloud_green'], 'tenkan_above': row['tenkan_above'],
                'tk_cross_down': row['tk_cross_down'], 'kijun_cross_down': row['kijun_cross_down'],
                'chikou_bearish': row.get('chikou_bearish', False),
                'above_cloud': row['above_cloud'], 'below_cloud': row['below_cloud'],
                'in_cloud': row['in_cloud'],
            })

    all_bars.sort(key=lambda x: x['time'])
    time_groups = {}
    for bar in all_bars:
        t = bar['time']
        if t not in time_groups:
            time_groups[t] = {}
        time_groups[t][bar['symbol']] = bar

    sorted_times = sorted(time_groups.keys())

    cash = initial
    positions = {}
    trades = []
    equity = []
    last_exit = {}

    for t in sorted_times:
        bars = time_groups[t]
        closed = []
        btc_uptrend = btc_trends.get(t)

        for sym, pos in positions.items():
            if sym not in bars:
                continue
            bar = bars[sym]
            price = bar['close']
            high = bar['high']
            low = bar['low']
            entry = pos['entry_price']

            if low < pos['lowest']:
                pos['lowest'] = low
                if low <= pos['take_profit']:
                    pos['trailing'] = True
                    pos['trail_stop'] = min(pos['trail_stop'], low * (1 + params['trail_pct'] / 100))

            reason = None
            max_loss_price = entry * 1.02
            if high >= max_loss_price:
                reason = 'MaxLoss'
                price = max_loss_price
            elif high >= pos['stop_loss']:
                reason = 'Stop'
                price = min(pos['stop_loss'], high)
            elif pos.get('trailing') and high >= pos['trail_stop']:
                reason = 'Trail'
                price = pos['trail_stop']
            elif not pos.get('trailing') and low <= pos['take_profit']:
                reason = 'TP'
                price = pos['take_profit']
            elif bar['in_cloud'] or bar['above_cloud']:
                reason = 'Cloud'
                price = bar['close']

            if reason:
                # 수수료 계산
                notional = pos['position_size'] * leverage
                round_trip_fee = notional * fee_rate * 2

                pnl_pct = (entry - price) / entry * 100
                realized = pnl_pct * leverage / 100 * pos['position_size'] - round_trip_fee
                cash += pos['position_size'] + realized

                trades.append({
                    'symbol': sym, 'side': 'short',
                    'entry_time': pos['entry_time'], 'exit_time': t,
                    'entry_price': entry, 'exit_price': price,
                    'pnl_raw': round(pnl_pct * leverage, 2),
                    'fee': round(round_trip_fee, 0),
                    'pnl': round(pnl_pct * leverage - round_trip_fee / pos['position_size'] * 100, 2),
                    'pnl_krw': round(realized, 0),
                    'reason': reason, 'size': pos['position_size']
                })
                closed.append(sym)
                last_exit[sym] = t

        for s in closed:
            del positions[s]

        unrealized = sum(
            (pos['entry_price'] - bars[sym]['close']) / pos['entry_price'] * leverage * pos['position_size'] / 100
            for sym, pos in positions.items() if sym in bars
        )
        eq = cash + sum(pos['position_size'] for pos in positions.values()) + unrealized
        p_size = eq * pos_pct

        if cash >= p_size and len(positions) < params['max_positions']:
            for sym, bar in bars.items():
                if sym in positions:
                    continue
                if sym in last_exit:
                    if (t - last_exit[sym]).total_seconds() < params['cooldown_hours'] * 3600:
                        continue

                price = bar['close']
                cloud_bottom = bar['cloud_bottom']
                thickness = bar['cloud_thickness']

                if bar['in_cloud'] or thickness < params['min_cloud_thickness']:
                    continue

                if bar['below_cloud'] and not bar['tenkan_above']:
                    has_signal = bar['tk_cross_down'] or bar['kijun_cross_down']
                    if not has_signal or btc_uptrend is False:
                        continue

                    stop_loss = cloud_bottom * (1 + params['sl_buffer'] / 100)
                    sl_distance_pct = (stop_loss - price) / price * 100

                    if params['min_sl_pct'] <= sl_distance_pct <= params['max_sl_pct']:
                        take_profit = price * (1 - sl_distance_pct * params['rr_ratio'] / 100)
                        positions[sym] = {
                            'side': 'short', 'entry_price': price, 'entry_time': t,
                            'stop_loss': stop_loss, 'take_profit': take_profit,
                            'highest': price, 'lowest': price,
                            'trail_stop': stop_loss, 'trailing': False,
                            'position_size': p_size,
                        }
                        cash -= p_size

                if len(positions) >= params['max_positions']:
                    break

        equity.append({'time': t, 'equity': eq})

    return trades, equity


if __name__ == '__main__':
    print("=" * 80)
    print("Ichimoku 전략 수수료 영향 분석")
    print("=" * 80)

    print("\n데이터 수집 중...")
    all_data = {}
    for symbol in COINS:
        print(f"  {symbol}...", end='', flush=True)
        df = fetch_klines(symbol, 240, limit=4000)
        if df is not None:
            all_data[symbol] = df
            print(" OK")

    if not all_data:
        print("데이터 없음")
        exit()

    print("\n" + "=" * 80)
    print("[1] 수수료 영향 분석")
    print("=" * 80)

    scenarios = [
        ('수수료 미포함', 0),
        ('수수료 0.05% (메이커)', 0.0005),
        ('수수료 0.10% (테이커)', 0.001),
    ]

    print(f"\n{'시나리오':<20} {'거래수':>8} {'총수익':>15} {'수익률':>10} {'EV/거래':>10}")
    print("-" * 80)

    for name, fee_rate in scenarios:
        trades, equity = run_backtest_with_fees(
            {k: v.copy() for k, v in all_data.items()},
            STRATEGY_PARAMS, INITIAL_CAPITAL, LEVERAGE, POSITION_PCT, fee_rate
        )

        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]

            win_rate = len(wins) / len(trades)
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0
            ev = win_rate * avg_win - (1 - win_rate) * avg_loss

            total_pnl = sum(t['pnl_krw'] for t in trades)
            final = equity[-1]['equity'] if equity else INITIAL_CAPITAL
            ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

            print(f"{name:<20} {len(trades):>8} ₩{total_pnl:>12,.0f} {ret:>9.1f}% {ev:>+9.3f}%")

    # 상세 분석 (테이커 기준)
    print("\n" + "=" * 80)
    print("[2] 상세 분석 (테이커 수수료 0.1%)")
    print("=" * 80)

    trades, equity = run_backtest_with_fees(
        {k: v.copy() for k, v in all_data.items()},
        STRATEGY_PARAMS, INITIAL_CAPITAL, LEVERAGE, POSITION_PCT, 0.001
    )

    if trades:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]

        print(f"\n총 거래: {len(trades)}회")
        print(f"승리: {len(wins)}회, 손실: {len(losses)}회")
        print(f"승률: {len(wins)/len(trades)*100:.1f}%")

        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0
        print(f"\n평균 승리: +{avg_win:.2f}%")
        print(f"평균 손실: -{avg_loss:.2f}%")

        win_rate = len(wins) / len(trades)
        ev = win_rate * avg_win - (1 - win_rate) * avg_loss
        print(f"\nEV/거래: {ev:+.3f}%")

        total_fees = sum(t['fee'] for t in trades)
        print(f"총 수수료: ₩{total_fees:,.0f}")

        total_profit = sum(t['pnl_krw'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl_krw'] for t in losses)) if losses else 0
        pf = total_profit / total_loss if total_loss > 0 else 999
        print(f"Profit Factor: {pf:.2f}")

        final = equity[-1]['equity']
        print(f"\n초기 자본: ₩{INITIAL_CAPITAL:,}")
        print(f"최종 자본: ₩{final:,.0f}")
        print(f"수익률: {(final-INITIAL_CAPITAL)/INITIAL_CAPITAL*100:+.1f}%")

        # MDD
        peak = INITIAL_CAPITAL
        max_dd = 0
        for e in equity:
            if e['equity'] > peak:
                peak = e['equity']
            dd = (peak - e['equity']) / peak * 100
            max_dd = max(max_dd, dd)
        print(f"MDD: {max_dd:.1f}%")

        if ev > 0:
            print(f"\n✅ 수수료 포함 후에도 양의 EV")
        else:
            print(f"\n❌ 수수료 포함 시 음의 EV")

    # RSI Divergence와 비교
    print("\n" + "=" * 80)
    print("[3] RSI Divergence vs Ichimoku (수수료 포함)")
    print("=" * 80)

    print("\n| 전략 | 거래수 | 수수료 총액 | 수수료 영향 | 최종 EV |")
    print("|------|--------|-------------|-------------|---------|")
    print("| RSI Divergence | 9,505 | ₩5,771,904 | -1.903%/거래 | ❌ 음수 |")

    if trades:
        total_fees = sum(t['fee'] for t in trades)
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0
        ev = win_rate * avg_win - (1 - win_rate) * avg_loss
        ev_status = "✅ 양수" if ev > 0 else "❌ 음수"
        print(f"| Ichimoku | {len(trades)} | ₩{total_fees:,.0f} | {ev:+.3f}%/거래 | {ev_status} |")

    print("\n핵심 차이:")
    print("  - RSI Divergence: 9,505회 거래 → 수수료 ₩577만")
    print(f"  - Ichimoku: {len(trades)}회 거래 → 수수료 ₩{sum(t['fee'] for t in trades):,.0f}")
    print("  - 거래 빈도가 낮을수록 수수료 영향 감소")
