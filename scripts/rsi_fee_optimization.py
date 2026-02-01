"""
RSI Divergence 수수료 커버 최적화
목표: EV +0.097% → +0.121% 이상 (25% 향상)

전략:
1. 손익비 조정 (현재 0.7:2.0)
2. 진입 필터 강화 (거래수 줄이기)
3. 코인 축소
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

from datetime import datetime
from typing import Dict
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import product

from scripts.data_cache import load_cached_data

# 바이빗 실제 수수료
FEE_RATE = 0.00055  # 테이커 0.055%

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

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def strategy_rsi_divergence(df, cfg):
    df = df.copy()
    lookback = cfg.get('lookback', 10)
    rsi_period = cfg.get('rsi_period', 14)
    div_threshold = cfg.get('div_threshold', 3)
    price_threshold = cfg.get('price_threshold', 0.005)

    df['rsi'] = rsi(df['close'], rsi_period)
    df['atr'] = atr(df, 14)

    # 추가 필터
    if cfg.get('use_rsi_extreme', False):
        df['rsi_oversold'] = df['rsi'] < cfg.get('rsi_low', 30)
        df['rsi_overbought'] = df['rsi'] > cfg.get('rsi_high', 70)
    else:
        df['rsi_oversold'] = True
        df['rsi_overbought'] = True

    if cfg.get('use_volume', False):
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_spike'] = df['volume'] > df['vol_ma'] * cfg.get('vol_mult', 1.5)
    else:
        df['vol_spike'] = True

    if cfg.get('use_trend', False):
        df['ema50'] = ema(df['close'], 50)
        df['trend_up'] = df['close'] > df['ema50']
        df['trend_down'] = df['close'] < df['ema50']
    else:
        df['trend_up'] = True
        df['trend_down'] = True

    df['price_low'] = df['low'].rolling(lookback).min()
    df['price_high'] = df['high'].rolling(lookback).max()
    df['rsi_at_low'] = df['rsi'].rolling(lookback).min()
    df['rsi_at_high'] = df['rsi'].rolling(lookback).max()

    # 롱: 신저가 + RSI 상승 다이버전스
    df['long'] = (
        (df['low'] <= df['price_low'] * (1 + price_threshold)) &
        (df['rsi'] > df['rsi_at_low'].shift(1) + div_threshold) &
        df['rsi_oversold'] &
        df['vol_spike'] &
        df['trend_up']
    )

    # 숏: 신고가 + RSI 하락 다이버전스
    df['short'] = (
        (df['high'] >= df['price_high'] * (1 - price_threshold)) &
        (df['rsi'] < df['rsi_at_high'].shift(1) - div_threshold) &
        df['rsi_overbought'] &
        df['vol_spike'] &
        df['trend_down']
    )

    return df


def backtest_with_fees(all_data: Dict[str, pd.DataFrame], strategy_cfg: dict, trade_cfg: dict) -> dict:
    cfg = {
        'initial': 5_000_000,
        'leverage': 10,
        'pos_pct': 0.12,
        'atr_sl': 0.7,
        'atr_tp': 2.0,
        'max_pos': 4,
        'cooldown': 2,
        'fee_rate': FEE_RATE,
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
                notional = pos['size'] * cfg['leverage']
                round_trip_fee = notional * cfg['fee_rate'] * 2
                total_fees += round_trip_fee

                pnl = ((exit_p - entry) / entry if pos['side'] == 'long' else (entry - exit_p) / entry) * 100
                leveraged_pnl = pnl * cfg['leverage'] / 100 * pos['size']
                realized = leveraged_pnl - round_trip_fee

                cash += pos['size'] + realized

                trades.append({
                    'pnl_pct': pnl * cfg['leverage'],
                    'pnl_after_fee': pnl * cfg['leverage'] - round_trip_fee / pos['size'] * 100,
                    'pnl_krw': realized,
                    'reason': reason
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

        equity.append({'eq': eq})

    if not trades:
        return {'trades': 0, 'ev': -999, 'ret': -100, 'pf': 0, 'fees': 0}

    wins = [t for t in trades if t['pnl_krw'] > 0]
    losses = [t for t in trades if t['pnl_krw'] <= 0]

    win_rate = len(wins) / len(trades)
    avg_win = np.mean([t['pnl_after_fee'] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t['pnl_after_fee'] for t in losses])) if losses else 0
    ev = win_rate * avg_win - (1 - win_rate) * avg_loss

    profit = sum(t['pnl_krw'] for t in wins) if wins else 0
    loss = abs(sum(t['pnl_krw'] for t in losses)) if losses else 0

    final = equity[-1]['eq'] if equity else cfg['initial']

    return {
        'trades': len(trades),
        'wins': len(wins),
        'wr': round(win_rate * 100, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'ev': round(ev, 4),
        'ret': round((final - cfg['initial']) / cfg['initial'] * 100, 1),
        'pf': round(profit / loss, 2) if loss > 0 else 999,
        'fees': round(total_fees, 0)
    }


if __name__ == '__main__':
    print("=" * 90)
    print("RSI Divergence 수수료 커버 최적화")
    print("=" * 90)
    print(f"목표: EV > +0.121% (바이빗 수수료 0.055% 커버)")
    print(f"현재: EV +0.097% (갭: 25% 향상 필요)")
    print("=" * 90)

    COINS_4 = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']
    COINS_2 = ['BTCUSDT', 'ETHUSDT']
    COINS_1 = ['BTCUSDT']

    data_4 = load_cached_data(COINS_4, '15m')
    data_2 = {k: v for k, v in data_4.items() if k in COINS_2}
    data_1 = {k: v for k, v in data_4.items() if k in COINS_1}

    print(f"\n코인: 4개 로드 완료")

    # ============================================================
    # 1. 손익비 조정
    # ============================================================
    print("\n" + "=" * 90)
    print("[1] 손익비 조정 (SL:TP)")
    print("=" * 90)

    base_strategy = {'lookback': 10, 'div_threshold': 3, 'rsi_period': 14}

    atr_ratios = [
        (0.5, 1.5), (0.5, 2.0), (0.5, 2.5), (0.5, 3.0),
        (0.7, 2.0), (0.7, 2.5), (0.7, 3.0), (0.7, 3.5),
        (1.0, 2.0), (1.0, 2.5), (1.0, 3.0), (1.0, 4.0),
    ]

    print(f"\n{'손익비':<12} {'거래':>8} {'승률':>8} {'평균승':>10} {'평균손':>10} {'EV':>10} {'수익률':>10} {'PF':>8}")
    print("-" * 90)

    best_rr = None
    best_ev = -999

    for sl, tp in atr_ratios:
        r = backtest_with_fees(
            {k: v.copy() for k, v in data_4.items()},
            base_strategy,
            {'atr_sl': sl, 'atr_tp': tp}
        )
        ev_ok = "✅" if r['ev'] > 0 else "❌"
        print(f"{sl}:{tp:<8} {r['trades']:>8} {r['wr']:>7.1f}% {r['avg_win']:>9.2f}% {r['avg_loss']:>9.2f}% "
              f"{r['ev']:>+9.4f}% {r['ret']:>9.1f}% {r['pf']:>7.2f} {ev_ok}")

        if r['ev'] > best_ev:
            best_ev = r['ev']
            best_rr = (sl, tp)

    print(f"\n최고 손익비: {best_rr[0]}:{best_rr[1]} (EV: {best_ev:+.4f}%)")

    # ============================================================
    # 2. 진입 필터 강화
    # ============================================================
    print("\n" + "=" * 90)
    print("[2] 진입 필터 강화 (거래수 줄이기)")
    print("=" * 90)

    filter_configs = [
        {'name': '기본', 'cfg': {}},
        {'name': 'RSI극단 (30/70)', 'cfg': {'use_rsi_extreme': True, 'rsi_low': 30, 'rsi_high': 70}},
        {'name': 'RSI극단 (25/75)', 'cfg': {'use_rsi_extreme': True, 'rsi_low': 25, 'rsi_high': 75}},
        {'name': 'RSI극단 (20/80)', 'cfg': {'use_rsi_extreme': True, 'rsi_low': 20, 'rsi_high': 80}},
        {'name': '거래량 1.5x', 'cfg': {'use_volume': True, 'vol_mult': 1.5}},
        {'name': '거래량 2.0x', 'cfg': {'use_volume': True, 'vol_mult': 2.0}},
        {'name': '추세필터 EMA50', 'cfg': {'use_trend': True}},
        {'name': 'RSI(25/75)+Vol', 'cfg': {'use_rsi_extreme': True, 'rsi_low': 25, 'rsi_high': 75, 'use_volume': True, 'vol_mult': 1.5}},
        {'name': 'RSI(20/80)+Vol', 'cfg': {'use_rsi_extreme': True, 'rsi_low': 20, 'rsi_high': 80, 'use_volume': True, 'vol_mult': 2.0}},
    ]

    print(f"\n{'필터':<20} {'거래':>8} {'승률':>8} {'EV':>10} {'수익률':>10} {'PF':>8}")
    print("-" * 90)

    best_filter = None
    best_filter_ev = -999

    for fc in filter_configs:
        s_cfg = {**base_strategy, **fc['cfg']}
        r = backtest_with_fees(
            {k: v.copy() for k, v in data_4.items()},
            s_cfg,
            {'atr_sl': best_rr[0], 'atr_tp': best_rr[1]}
        )
        ev_ok = "✅" if r['ev'] > 0 else "❌"
        print(f"{fc['name']:<20} {r['trades']:>8} {r['wr']:>7.1f}% {r['ev']:>+9.4f}% {r['ret']:>9.1f}% {r['pf']:>7.2f} {ev_ok}")

        if r['ev'] > best_filter_ev:
            best_filter_ev = r['ev']
            best_filter = fc

    print(f"\n최고 필터: {best_filter['name']} (EV: {best_filter_ev:+.4f}%)")

    # ============================================================
    # 3. 코인 수 축소
    # ============================================================
    print("\n" + "=" * 90)
    print("[3] 코인 수 축소")
    print("=" * 90)

    coin_sets = [
        ('4코인', data_4),
        ('2코인 (BTC+ETH)', data_2),
        ('1코인 (BTC)', data_1),
    ]

    print(f"\n{'코인셋':<20} {'거래':>8} {'승률':>8} {'EV':>10} {'수익률':>10} {'PF':>8}")
    print("-" * 90)

    best_coins = None
    best_coins_ev = -999

    s_cfg = {**base_strategy, **best_filter['cfg']}

    for name, data in coin_sets:
        r = backtest_with_fees(
            {k: v.copy() for k, v in data.items()},
            s_cfg,
            {'atr_sl': best_rr[0], 'atr_tp': best_rr[1]}
        )
        ev_ok = "✅" if r['ev'] > 0 else "❌"
        print(f"{name:<20} {r['trades']:>8} {r['wr']:>7.1f}% {r['ev']:>+9.4f}% {r['ret']:>9.1f}% {r['pf']:>7.2f} {ev_ok}")

        if r['ev'] > best_coins_ev:
            best_coins_ev = r['ev']
            best_coins = (name, data)

    # ============================================================
    # 4. 레버리지/포지션 크기
    # ============================================================
    print("\n" + "=" * 90)
    print("[4] 레버리지/포지션 크기 조정")
    print("=" * 90)

    lev_pos_combos = [
        (5, 0.20), (5, 0.15), (5, 0.10),
        (7, 0.15), (7, 0.12), (7, 0.10),
        (10, 0.12), (10, 0.10), (10, 0.08),
    ]

    print(f"\n{'설정':<15} {'거래':>8} {'승률':>8} {'EV':>10} {'수익률':>10} {'PF':>8}")
    print("-" * 90)

    for lev, pos in lev_pos_combos:
        r = backtest_with_fees(
            {k: v.copy() for k, v in best_coins[1].items()},
            s_cfg,
            {'atr_sl': best_rr[0], 'atr_tp': best_rr[1], 'leverage': lev, 'pos_pct': pos}
        )
        ev_ok = "✅" if r['ev'] > 0 else "❌"
        print(f"L{lev}x P{int(pos*100)}%      {r['trades']:>8} {r['wr']:>7.1f}% {r['ev']:>+9.4f}% {r['ret']:>9.1f}% {r['pf']:>7.2f} {ev_ok}")

    # ============================================================
    # 5. 최종 결과
    # ============================================================
    print("\n" + "=" * 90)
    print("최종 최적화 결과")
    print("=" * 90)

    print(f"\n최적 설정:")
    print(f"  손익비: {best_rr[0]}:{best_rr[1]}")
    print(f"  필터: {best_filter['name']}")
    print(f"  코인: {best_coins[0]}")

    # 최종 테스트
    final_r = backtest_with_fees(
        {k: v.copy() for k, v in best_coins[1].items()},
        s_cfg,
        {'atr_sl': best_rr[0], 'atr_tp': best_rr[1]}
    )

    print(f"\n최종 결과:")
    print(f"  거래 수: {final_r['trades']}회")
    print(f"  승률: {final_r['wr']}%")
    print(f"  EV/거래: {final_r['ev']:+.4f}%")
    print(f"  수익률: {final_r['ret']:+.1f}%")
    print(f"  PF: {final_r['pf']:.2f}")
    print(f"  총 수수료: ₩{final_r['fees']:,.0f}")

    target_ev = 0.121
    if final_r['ev'] > 0:
        print(f"\n✅ 양의 EV 달성! ({final_r['ev']:+.4f}%)")
        if final_r['ev'] >= target_ev:
            print(f"✅ 목표 EV ({target_ev}%) 달성!")
        else:
            gap = target_ev - final_r['ev']
            print(f"⚠️ 목표까지 {gap:.4f}% 부족")
    else:
        print(f"\n❌ 여전히 음의 EV ({final_r['ev']:+.4f}%)")
        print("→ RSI Divergence는 이 수수료 구조에서 사용 불가")
