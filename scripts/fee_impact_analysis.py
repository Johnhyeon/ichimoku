"""
수수료 영향 분석 + 테일 리스크 구조 분석
RSI Divergence 전략의 실제 EV 검증
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict

from scripts.data_cache import load_cached_data

# ============================================================
# 지표
# ============================================================
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

# ============================================================
# RSI Divergence 전략
# ============================================================
def strategy_rsi_divergence(df, cfg):
    df = df.copy()
    lookback = cfg.get('lookback', 10)
    rsi_period = cfg.get('rsi_period', 14)
    div_threshold = cfg.get('div_threshold', 3)
    price_threshold = cfg.get('price_threshold', 0.005)

    df['rsi'] = rsi(df['close'], rsi_period)
    df['atr'] = atr(df, 14)
    df['vol_ratio'] = 1
    df['trend_up'] = True
    df['trend_down'] = True

    df['price_low'] = df['low'].rolling(lookback).min()
    df['price_high'] = df['high'].rolling(lookback).max()
    df['rsi_at_low'] = df['rsi'].rolling(lookback).min()
    df['rsi_at_high'] = df['rsi'].rolling(lookback).max()

    df['long'] = (
        (df['low'] <= df['price_low'] * (1 + price_threshold)) &
        (df['rsi'] > df['rsi_at_low'].shift(1) + div_threshold) &
        df['trend_up']
    )

    df['short'] = (
        (df['high'] >= df['price_high'] * (1 - price_threshold)) &
        (df['rsi'] < df['rsi_at_high'].shift(1) - div_threshold) &
        df['trend_down']
    )

    return df


# ============================================================
# 수수료 포함 백테스트
# ============================================================
def backtest_with_fees(all_data: Dict[str, pd.DataFrame], strategy_cfg: dict, trade_cfg: dict) -> dict:
    cfg = {
        'initial': 5_000_000,
        'leverage': 10,
        'pos_pct': 0.12,
        'atr_sl': 0.7,
        'atr_tp': 2.0,
        'max_pos': 4,
        'cooldown': 2,
        'fee_rate': 0.001,  # 바이빗 테이커 수수료 0.1% (편도)
        **trade_cfg
    }

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
    daily_pnl = defaultdict(float)

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
                # 수수료 계산: notional × fee_rate × 2 (왕복)
                notional = pos['size'] * cfg['leverage']
                round_trip_fee = notional * cfg['fee_rate'] * 2

                pnl = ((exit_p - entry) / entry if pos['side'] == 'long' else (entry - exit_p) / entry) * 100
                realized = pnl * cfg['leverage'] / 100 * pos['size'] - round_trip_fee

                cash += pos['size'] + realized
                daily_pnl[t.date()] += realized

                trades.append({
                    'sym': sym, 'side': pos['side'],
                    'entry': entry, 'exit': exit_p,
                    'pnl_raw': round(pnl * cfg['leverage'], 2),  # 수수료 전
                    'fee': round(round_trip_fee, 0),
                    'pnl': round(pnl * cfg['leverage'] - round_trip_fee / pos['size'] * 100, 2),  # 수수료 후 %
                    'pnl_krw': round(realized, 0),
                    'reason': reason,
                    'size': pos['size']
                })
                closed.append(sym)
                last_exit[sym] = t

        for s in closed:
            del positions[s]

        unreal = sum(
            ((current_bars[s]['close'] - p['entry']) / p['entry'] if p['side'] == 'long'
             else (p['entry'] - current_bars[s]['close']) / p['entry']) * cfg['leverage'] * p['size'] / 100
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

    return trades, equity, daily_pnl, cfg['initial']


def analyze_tail_risk(trades: List[dict]):
    """테일 리스크 구조 분석"""
    if not trades:
        return None

    losses = [t for t in trades if t['pnl_raw'] <= 0]
    wins = [t for t in trades if t['pnl_raw'] > 0]

    if not losses:
        return None

    loss_pcts = [abs(t['pnl_raw']) for t in losses]

    # 손실 분포 분석
    median_loss = np.median(loss_pcts)
    mean_loss = np.mean(loss_pcts)
    p75 = np.percentile(loss_pcts, 75)
    p90 = np.percentile(loss_pcts, 90)
    p95 = np.percentile(loss_pcts, 95)
    p99 = np.percentile(loss_pcts, 99)
    max_loss = max(loss_pcts)

    # 테일 이벤트 분류 (평균의 3배 이상을 테일로 정의)
    tail_threshold = mean_loss * 3
    normal_losses = [l for l in loss_pcts if l <= tail_threshold]
    tail_losses = [l for l in loss_pcts if l > tail_threshold]

    return {
        'total_losses': len(losses),
        'total_wins': len(wins),
        'median_loss': median_loss,
        'mean_loss': mean_loss,
        'p75': p75,
        'p90': p90,
        'p95': p95,
        'p99': p99,
        'max_loss': max_loss,
        'tail_threshold': tail_threshold,
        'normal_count': len(normal_losses),
        'tail_count': len(tail_losses),
        'normal_avg': np.mean(normal_losses) if normal_losses else 0,
        'tail_avg': np.mean(tail_losses) if tail_losses else 0,
        'tail_pct': len(tail_losses) / len(losses) * 100 if losses else 0,
    }


def analyze_consecutive_losses_with_tails(trades: List[dict], tail_threshold: float):
    """테일 이벤트 포함 연속 손실 분석"""
    if not trades:
        return None

    sorted_trades = sorted(trades, key=lambda x: x.get('entry', 0))

    max_streak = 0
    current_streak = 0
    current_streak_trades = []
    worst_streak_trades = []

    for t in sorted_trades:
        if t['pnl_raw'] <= 0:
            current_streak += 1
            current_streak_trades.append(t)
            if current_streak > max_streak:
                max_streak = current_streak
                worst_streak_trades = current_streak_trades.copy()
        else:
            current_streak = 0
            current_streak_trades = []

    # 최악 연패에서 테일 분석
    if worst_streak_trades:
        streak_losses = [abs(t['pnl_raw']) for t in worst_streak_trades]
        tails_in_streak = sum(1 for l in streak_losses if l > tail_threshold)
        normal_in_streak = max_streak - tails_in_streak

        return {
            'max_streak': max_streak,
            'tails_in_worst_streak': tails_in_streak,
            'normal_in_worst_streak': normal_in_streak,
            'worst_streak_total_loss': sum(streak_losses),
            'worst_streak_losses': streak_losses,
        }

    return {'max_streak': 0}


if __name__ == '__main__':
    print("=" * 80)
    print("수수료 영향 분석 + 테일 리스크 구조")
    print("=" * 80)

    COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']
    data = load_cached_data(COINS, '15m')
    print(f"코인: {len(data)}개")

    # 최적 설정 (STRATEGY_RESULTS.md에서)
    strategy_cfg = {'lookback': 10, 'div_threshold': 3, 'rsi_period': 14}

    # ============================================================
    # 1. 수수료 미포함 vs 포함 비교
    # ============================================================
    print("\n" + "=" * 80)
    print("[1] 수수료 영향 분석")
    print("=" * 80)

    scenarios = [
        ('수수료 미포함', {'fee_rate': 0}),
        ('수수료 0.05% (메이커)', {'fee_rate': 0.0005}),
        ('수수료 0.10% (테이커)', {'fee_rate': 0.001}),
    ]

    print(f"\n{'시나리오':<20} {'거래수':>8} {'총수익':>15} {'수익률':>10} {'EV/거래':>10}")
    print("-" * 80)

    for name, fee_cfg in scenarios:
        trade_cfg = {
            'leverage': 10,
            'pos_pct': 0.12,
            'atr_sl': 0.7,
            'atr_tp': 2.0,
            **fee_cfg
        }
        trades, equity, daily_pnl, initial = backtest_with_fees(
            {k: v.copy() for k, v in data.items()},
            strategy_cfg,
            trade_cfg
        )

        if trades:
            total_pnl = sum(t['pnl_krw'] for t in trades)
            final = equity[-1]['eq'] if equity else initial
            ret = (final - initial) / initial * 100
            avg_pnl = np.mean([t['pnl'] for t in trades])

            print(f"{name:<20} {len(trades):>8} ₩{total_pnl:>12,.0f} {ret:>9.1f}% {avg_pnl:>+9.3f}%")

    # ============================================================
    # 2. 테일 리스크 구조 분석
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] 테일 리스크 구조 분석")
    print("=" * 80)

    # 수수료 포함 거래 데이터로 분석
    trade_cfg = {'leverage': 10, 'pos_pct': 0.12, 'atr_sl': 0.7, 'atr_tp': 2.0, 'fee_rate': 0.001}
    trades, equity, daily_pnl, initial = backtest_with_fees(
        {k: v.copy() for k, v in data.items()},
        strategy_cfg,
        trade_cfg
    )

    tail_analysis = analyze_tail_risk(trades)

    if tail_analysis:
        print(f"\n총 손실 거래: {tail_analysis['total_losses']}회")
        print(f"총 승리 거래: {tail_analysis['total_wins']}회")

        print(f"\n[손실 분포]")
        print(f"  중앙값:  {tail_analysis['median_loss']:.2f}%")
        print(f"  평균:    {tail_analysis['mean_loss']:.2f}%")
        print(f"  75%ile:  {tail_analysis['p75']:.2f}%")
        print(f"  90%ile:  {tail_analysis['p90']:.2f}%")
        print(f"  95%ile:  {tail_analysis['p95']:.2f}%")
        print(f"  99%ile:  {tail_analysis['p99']:.2f}%")
        print(f"  최대:    {tail_analysis['max_loss']:.2f}%")

        print(f"\n[정상 vs 테일 분류] (테일 = 평균×3 = {tail_analysis['tail_threshold']:.2f}% 이상)")
        print(f"  정상 손실: {tail_analysis['normal_count']}회 ({100-tail_analysis['tail_pct']:.1f}%), 평균 {tail_analysis['normal_avg']:.2f}%")
        print(f"  테일 손실: {tail_analysis['tail_count']}회 ({tail_analysis['tail_pct']:.1f}%), 평균 {tail_analysis['tail_avg']:.2f}%")

        # 연속 손실 + 테일 분석
        streak_analysis = analyze_consecutive_losses_with_tails(trades, tail_analysis['tail_threshold'])
        if streak_analysis:
            print(f"\n[최악 연패 분석]")
            print(f"  최대 연속 손실: {streak_analysis['max_streak']}회")
            print(f"  연패 중 테일 이벤트: {streak_analysis['tails_in_worst_streak']}회")
            print(f"  연패 중 정상 손실: {streak_analysis['normal_in_worst_streak']}회")
            print(f"  연패 총 손실: {streak_analysis['worst_streak_total_loss']:.2f}%")

    # ============================================================
    # 3. 수수료 포함 EV 상세 분석
    # ============================================================
    print("\n" + "=" * 80)
    print("[3] 수수료 포함 EV 상세 분석")
    print("=" * 80)

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    win_rate = len(wins) / len(trades) if trades else 0
    loss_rate = 1 - win_rate
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0

    ev = win_rate * avg_win - loss_rate * avg_loss

    print(f"\n[수수료 포함 후]")
    print(f"  승률: {win_rate*100:.1f}%")
    print(f"  평균 승리: +{avg_win:.2f}%")
    print(f"  평균 손실: -{avg_loss:.2f}%")
    print(f"  EV = {win_rate*100:.1f}% × {avg_win:.2f}% - {loss_rate*100:.1f}% × {avg_loss:.2f}%")
    print(f"  EV/거래 = {ev:+.3f}%")

    # 수수료 영향도
    avg_fee_pct = np.mean([t['fee'] / t['size'] * 100 for t in trades]) if trades else 0
    print(f"\n[수수료 영향]")
    print(f"  평균 수수료/거래: {avg_fee_pct:.3f}% of position")
    print(f"  총 수수료: ₩{sum(t['fee'] for t in trades):,.0f}")

    # Profit Factor
    total_profit = sum(t['pnl_krw'] for t in wins) if wins else 0
    total_loss = abs(sum(t['pnl_krw'] for t in losses)) if losses else 0
    pf = total_profit / total_loss if total_loss > 0 else 999
    print(f"\n  Profit Factor: {pf:.2f}")

    # ============================================================
    # 4. 생존 분석 (테일 포함)
    # ============================================================
    print("\n" + "=" * 80)
    print("[4] 테일 포함 생존 분석")
    print("=" * 80)

    if tail_analysis and streak_analysis:
        pos_pct = 0.12

        # 시나리오 1: 정상 손실만
        normal_loss_per_trade = tail_analysis['normal_avg'] * pos_pct / 100
        streak_normal_only = streak_analysis['max_streak'] * normal_loss_per_trade * 100

        # 시나리오 2: 실제 최악 연패
        actual_streak_loss = streak_analysis['worst_streak_total_loss'] * pos_pct / 100 * 100

        # 시나리오 3: 연패 중 2회 테일 발생
        hypothetical_2_tails = (
            (streak_analysis['max_streak'] - 2) * tail_analysis['normal_avg'] +
            2 * tail_analysis['tail_avg']
        ) * pos_pct / 100 * 100

        print(f"\n포지션 비중: {pos_pct*100}%")
        print(f"최대 연속 손실: {streak_analysis['max_streak']}회")

        print(f"\n[시나리오별 자본 손실]")
        print(f"  정상 손실만:        -{streak_normal_only:.1f}% → 잔고 {100-streak_normal_only:.1f}%")
        print(f"  실제 최악 연패:     -{actual_streak_loss:.1f}% → 잔고 {100-actual_streak_loss:.1f}%")
        print(f"  테일 2회 가정:      -{hypothetical_2_tails:.1f}% → 잔고 {100-hypothetical_2_tails:.1f}%")

    # ============================================================
    # 5. 결론
    # ============================================================
    print("\n" + "=" * 80)
    print("[5] 결론")
    print("=" * 80)

    final_equity = equity[-1]['eq'] if equity else initial
    total_return = (final_equity - initial) / initial * 100

    print(f"\n수수료 포함 최종 결과:")
    print(f"  초기 자본: ₩{initial:,}")
    print(f"  최종 자본: ₩{final_equity:,.0f}")
    print(f"  수익률: {total_return:+.1f}%")
    print(f"  EV/거래: {ev:+.3f}%")
    print(f"  Profit Factor: {pf:.2f}")

    if ev > 0:
        print(f"\n✅ 수수료 포함 후에도 양의 EV 유지")
        print(f"   전략은 유효하지만 테일 리스크 관리 필요")
    else:
        print(f"\n❌ 수수료 포함 시 음의 EV")
        print(f"   전략 재검토 필요")
