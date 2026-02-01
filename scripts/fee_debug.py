"""
수수료 계산 디버깅
"""
import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict

from scripts.data_cache import load_cached_data

# 지표
def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l)

def atr(df, p=14):
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def strategy_rsi_divergence(df, cfg):
    df = df.copy()
    lookback = cfg.get('lookback', 10)
    rsi_period = cfg.get('rsi_period', 14)
    div_threshold = cfg.get('div_threshold', 3)
    price_threshold = cfg.get('price_threshold', 0.005)

    df['rsi'] = rsi(df['close'], rsi_period)
    df['atr'] = atr(df, 14)

    df['price_low'] = df['low'].rolling(lookback).min()
    df['price_high'] = df['high'].rolling(lookback).max()
    df['rsi_at_low'] = df['rsi'].rolling(lookback).min()
    df['rsi_at_high'] = df['rsi'].rolling(lookback).max()

    df['long'] = (
        (df['low'] <= df['price_low'] * (1 + price_threshold)) &
        (df['rsi'] > df['rsi_at_low'].shift(1) + div_threshold)
    )
    df['short'] = (
        (df['high'] >= df['price_high'] * (1 - price_threshold)) &
        (df['rsi'] < df['rsi_at_high'].shift(1) - div_threshold)
    )

    return df


def backtest_debug(all_data, fee_rate=0.001):
    """수수료 디버깅용 백테스트"""
    cfg = {
        'initial': 5_000_000,
        'leverage': 10,
        'pos_pct': 0.12,
        'atr_sl': 0.7,
        'atr_tp': 2.0,
        'max_pos': 4,
        'cooldown': 2,
    }

    strategy_cfg = {'lookback': 10, 'div_threshold': 3, 'rsi_period': 14}

    for sym in all_data:
        all_data[sym] = strategy_rsi_divergence(all_data[sym], strategy_cfg)

    bars = []
    for sym, df in all_data.items():
        df = df.dropna()
        for _, row in df.iterrows():
            bars.append({'symbol': sym, **row.to_dict()})
    bars.sort(key=lambda x: x['timestamp'])

    tg = {}
    for b in bars:
        t = b['timestamp']
        if t not in tg:
            tg[t] = {}
        tg[t][b['symbol']] = b

    times = sorted(tg.keys())

    cash = cfg['initial']
    positions = {}
    trades = []
    equity = []
    last_exit = {}

    total_fees = 0

    for t in times:
        current_bars = tg[t]
        closed = []

        for sym, pos in positions.items():
            if sym not in current_bars:
                continue
            b = current_bars[sym]
            h, l = b['high'], b['low']
            entry = pos['entry']
            reason = None

            if pos['side'] == 'long':
                if l <= pos['sl']:
                    reason, exit_p = 'SL', pos['sl']
                elif h >= pos['tp']:
                    reason, exit_p = 'TP', pos['tp']
            else:
                if h >= pos['sl']:
                    reason, exit_p = 'SL', pos['sl']
                elif l <= pos['tp']:
                    reason, exit_p = 'TP', pos['tp']

            if reason:
                # 수수료 계산
                # notional = 실제 거래 금액 = margin × leverage
                notional = pos['size'] * cfg['leverage']
                # 왕복 수수료 = notional × fee_rate × 2
                round_trip_fee = notional * fee_rate * 2

                # PnL (가격 변화)
                price_pnl = ((exit_p - entry) / entry if pos['side'] == 'long'
                            else (entry - exit_p) / entry)
                # 레버리지 적용 수익금
                leveraged_pnl = price_pnl * cfg['leverage'] * pos['size']
                # 수수료 차감
                realized = leveraged_pnl - round_trip_fee

                total_fees += round_trip_fee
                cash += pos['size'] + realized

                trades.append({
                    'sym': sym,
                    'side': pos['side'],
                    'entry': entry,
                    'exit': exit_p,
                    'size': pos['size'],
                    'notional': notional,
                    'price_pnl_pct': price_pnl * 100,  # 가격 변화 %
                    'leveraged_pnl': leveraged_pnl,  # 레버리지 적용 수익금
                    'fee': round_trip_fee,
                    'realized': realized,  # 최종 수익금
                    'reason': reason
                })
                closed.append(sym)
                last_exit[sym] = t

        for s in closed:
            del positions[s]

        unreal = sum(
            ((current_bars[s]['close'] - p['entry']) / p['entry'] if p['side'] == 'long'
             else (p['entry'] - current_bars[s]['close']) / p['entry']) * cfg['leverage'] * p['size']
            for s, p in positions.items() if s in current_bars
        )
        eq = cash + sum(p['size'] for p in positions.values()) + unreal
        pos_size = eq * cfg['pos_pct']

        if cash >= pos_size and len(positions) < cfg['max_pos']:
            for sym, b in current_bars.items():
                if sym in positions:
                    continue
                if sym in last_exit and (t - last_exit[sym]).total_seconds() < cfg['cooldown'] * 15 * 60:
                    continue

                price = b['close']
                a = b.get('atr', price * 0.01)

                if b.get('long', False):
                    sl = price - a * cfg['atr_sl']
                    tp = price + a * cfg['atr_tp']
                    positions[sym] = {'side': 'long', 'entry': price, 'sl': sl, 'tp': tp, 'size': pos_size}
                    cash -= pos_size
                elif b.get('short', False):
                    sl = price + a * cfg['atr_sl']
                    tp = price - a * cfg['atr_tp']
                    positions[sym] = {'side': 'short', 'entry': price, 'sl': sl, 'tp': tp, 'size': pos_size}
                    cash -= pos_size

                if len(positions) >= cfg['max_pos']:
                    break

        equity.append({'t': t, 'eq': eq})

    return trades, equity, total_fees, cfg['initial']


if __name__ == '__main__':
    print("=" * 80)
    print("수수료 계산 디버깅")
    print("=" * 80)

    COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']
    data = load_cached_data(COINS, '15m')

    # 수수료 없이 테스트
    print("\n[1] 수수료 없이")
    trades_no_fee, equity_no_fee, total_fees_no, initial = backtest_debug(
        {k: v.copy() for k, v in data.items()}, fee_rate=0
    )

    final_no_fee = equity_no_fee[-1]['eq']
    total_pnl_no_fee = sum(t['realized'] for t in trades_no_fee)

    print(f"  거래 수: {len(trades_no_fee)}")
    print(f"  총 수익금: ₩{total_pnl_no_fee:,.0f}")
    print(f"  수수료: ₩{total_fees_no:,.0f}")
    print(f"  최종 자본: ₩{final_no_fee:,.0f}")
    print(f"  수익률: {(final_no_fee - initial) / initial * 100:.1f}%")

    # 수수료 포함 테스트
    print("\n[2] 수수료 0.1% (테이커)")
    trades_fee, equity_fee, total_fees, initial = backtest_debug(
        {k: v.copy() for k, v in data.items()}, fee_rate=0.001
    )

    final_fee = equity_fee[-1]['eq']
    total_pnl_fee = sum(t['realized'] for t in trades_fee)
    total_leveraged_pnl = sum(t['leveraged_pnl'] for t in trades_fee)

    print(f"  거래 수: {len(trades_fee)}")
    print(f"  레버리지 수익금 (수수료 전): ₩{total_leveraged_pnl:,.0f}")
    print(f"  총 수수료: ₩{total_fees:,.0f}")
    print(f"  순 수익금 (수수료 후): ₩{total_pnl_fee:,.0f}")
    print(f"  최종 자본: ₩{final_fee:,.0f}")
    print(f"  수익률: {(final_fee - initial) / initial * 100:.1f}%")

    # 수수료 분석
    print("\n[3] 수수료 상세 분석")
    avg_size = np.mean([t['size'] for t in trades_fee])
    avg_notional = np.mean([t['notional'] for t in trades_fee])
    avg_fee = np.mean([t['fee'] for t in trades_fee])

    print(f"  평균 포지션 크기 (마진): ₩{avg_size:,.0f}")
    print(f"  평균 notional (마진×레버리지): ₩{avg_notional:,.0f}")
    print(f"  평균 수수료: ₩{avg_fee:,.0f}")
    print(f"  수수료 / 마진: {avg_fee / avg_size * 100:.2f}%")
    print(f"  수수료 / notional: {avg_fee / avg_notional * 100:.4f}%")

    # 첫 10개 거래 상세
    print("\n[4] 첫 10개 거래 상세")
    print(f"{'#':>3} {'심볼':<10} {'방향':<5} {'마진':>12} {'notional':>15} {'수수료':>10} {'수익금':>12} {'순수익':>12}")
    print("-" * 100)
    for i, t in enumerate(trades_fee[:10]):
        print(f"{i+1:>3} {t['sym']:<10} {t['side']:<5} ₩{t['size']:>10,.0f} ₩{t['notional']:>13,.0f} "
              f"₩{t['fee']:>8,.0f} ₩{t['leveraged_pnl']:>10,.0f} ₩{t['realized']:>10,.0f}")

    # 예상 vs 실제 비교
    print("\n[5] 예상 vs 실제 비교")
    expected_total_fee = len(trades_fee) * avg_fee
    print(f"  예상 총 수수료: {len(trades_fee)} × ₩{avg_fee:,.0f} = ₩{expected_total_fee:,.0f}")
    print(f"  실제 총 수수료: ₩{total_fees:,.0f}")

    print("\n[6] 수익 분석")
    wins = [t for t in trades_fee if t['realized'] > 0]
    losses = [t for t in trades_fee if t['realized'] <= 0]
    print(f"  승리: {len(wins)}회")
    print(f"  손실: {len(losses)}회")
    print(f"  승률: {len(wins)/len(trades_fee)*100:.1f}%")

    avg_win_realized = np.mean([t['realized'] for t in wins]) if wins else 0
    avg_loss_realized = np.mean([t['realized'] for t in losses]) if losses else 0
    print(f"  평균 승리 수익: ₩{avg_win_realized:,.0f}")
    print(f"  평균 손실: ₩{avg_loss_realized:,.0f}")

    # EV 계산
    win_rate = len(wins) / len(trades_fee)
    ev_per_trade = win_rate * avg_win_realized + (1 - win_rate) * avg_loss_realized
    print(f"\n  EV/거래 = {win_rate*100:.1f}% × ₩{avg_win_realized:,.0f} + {(1-win_rate)*100:.1f}% × ₩{avg_loss_realized:,.0f}")
    print(f"  EV/거래 = ₩{ev_per_trade:,.0f}")
    print(f"  총 예상 수익 = {len(trades_fee)} × ₩{ev_per_trade:,.0f} = ₩{len(trades_fee) * ev_per_trade:,.0f}")
