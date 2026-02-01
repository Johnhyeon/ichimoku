"""
RSI Divergence 전략 집중 파인튜닝
목표: 일평균 수익 극대화, 10%+ 날 최대화
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

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

# ============================================================
# RSI Divergence 전략 (개선)
# ============================================================
def strategy_rsi_divergence(df, cfg):
    """
    RSI 다이버전스: 가격과 RSI가 다른 방향으로 움직일 때 반전 포착

    개선 버전:
    - 다양한 lookback 기간
    - 다양한 RSI 기간
    - 다이버전스 강도 측정
    - 거래량 필터 옵션
    - 추세 필터 옵션
    """
    df = df.copy()
    lookback = cfg.get('lookback', 14)
    rsi_period = cfg.get('rsi_period', 14)
    div_threshold = cfg.get('div_threshold', 5)  # RSI 상승 최소량
    price_threshold = cfg.get('price_threshold', 0.005)  # 가격 신저가 허용 범위

    df['rsi'] = rsi(df['close'], rsi_period)
    df['atr'] = atr(df, 14)

    # 거래량 필터
    if cfg.get('use_volume', False):
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
    else:
        df['vol_ratio'] = 1

    # 추세 필터
    if cfg.get('use_trend', False):
        df['ema50'] = ema(df['close'], 50)
        df['trend_up'] = df['close'] > df['ema50']
        df['trend_down'] = df['close'] < df['ema50']
    else:
        df['trend_up'] = True
        df['trend_down'] = True

    # N봉 최저/최고
    df['price_low'] = df['low'].rolling(lookback).min()
    df['price_high'] = df['high'].rolling(lookback).max()
    df['rsi_at_low'] = df['rsi'].rolling(lookback).min()
    df['rsi_at_high'] = df['rsi'].rolling(lookback).max()

    vol_min = cfg.get('vol_min', 1.0)

    # 롱: 가격 신저가 근처 + RSI 상승 (상승 다이버전스)
    df['long'] = (
        (df['low'] <= df['price_low'] * (1 + price_threshold)) &
        (df['rsi'] > df['rsi_at_low'].shift(1) + div_threshold) &
        (df['vol_ratio'] >= vol_min) &
        df['trend_up']
    )

    # 숏: 가격 신고가 근처 + RSI 하락 (하락 다이버전스)
    df['short'] = (
        (df['high'] >= df['price_high'] * (1 - price_threshold)) &
        (df['rsi'] < df['rsi_at_high'].shift(1) - div_threshold) &
        (df['vol_ratio'] >= vol_min) &
        df['trend_down']
    )

    return df


# ============================================================
# 백테스트
# ============================================================
def backtest(all_data: Dict[str, pd.DataFrame], strategy_cfg: dict, trade_cfg: dict) -> dict:
    cfg = {
        'initial': 5_000_000,
        'leverage': 10,
        'pos_pct': 0.10,
        'atr_sl': 1.0,
        'atr_tp': 2.0,
        'max_pos': 4,
        'cooldown': 2,
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
                pnl = ((exit_p - entry) / entry if pos['side'] == 'long' else (entry - exit_p) / entry) * 100
                realized = pnl * cfg['leverage'] / 100 * pos['size']
                cash += pos['size'] + realized
                daily_pnl[t.date()] += realized
                trades.append({
                    'sym': sym, 'side': pos['side'],
                    'pnl': round(pnl * cfg['leverage'], 2),
                    'pnl_krw': round(realized, 0), 'reason': reason
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

    return calc_stats(trades, equity, daily_pnl, cfg['initial'])


def calc_stats(trades, equity, daily_pnl, initial):
    if not trades:
        return {'trades': 0, 'wr': 0, 'ret': 0, 'mdd': 0, 'pf': 0, 'daily': 0, 'big': 0, 'max_daily': 0}

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    profit = sum(t['pnl_krw'] for t in wins) if wins else 0
    loss = abs(sum(t['pnl_krw'] for t in losses)) if losses else 0

    peak, mdd = initial, 0
    for e in equity:
        if e['eq'] > peak:
            peak = e['eq']
        mdd = max(mdd, (peak - e['eq']) / peak * 100)

    final = equity[-1]['eq'] if equity else initial
    days = len(daily_pnl)
    daily_rets = [v / initial * 100 for v in daily_pnl.values()]
    big = len([d for d in daily_rets if d >= 10])
    max_d = max(daily_rets) if daily_rets else 0

    return {
        'trades': len(trades),
        'wr': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'avg_w': round(np.mean([t['pnl'] for t in wins]), 1) if wins else 0,
        'avg_l': round(np.mean([t['pnl'] for t in losses]), 1) if losses else 0,
        'pnl': round(sum(t['pnl_krw'] for t in trades), 0),
        'ret': round((final - initial) / initial * 100, 1),
        'final': round(final, 0),
        'mdd': round(mdd, 1),
        'pf': round(profit / loss, 2) if loss > 0 else 999,
        'daily': round(np.mean(daily_rets), 2) if daily_rets else 0,
        'big': big,
        'days': days,
        'max_daily': round(max_d, 1),
        'tpd': round(len(trades) / max(days, 1), 2)
    }


if __name__ == '__main__':
    print("=" * 80)
    print("RSI Divergence 전략 집중 파인튜닝")
    print("=" * 80)

    COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']
    data = load_cached_data(COINS, '15m')
    print(f"코인: {len(data)}개")

    results = []

    # ============================================================
    # 파라미터 그리드 서치
    # ============================================================
    print("\n[1] 파라미터 그리드 서치")
    print("-" * 80)

    lookbacks = [7, 10, 14, 20]
    rsi_periods = [7, 10, 14]
    div_thresholds = [3, 5, 7, 10]
    atr_ratios = [(0.7, 1.5), (0.7, 2), (1, 2), (1, 3), (0.5, 2)]

    best_ret = -float('inf')
    best_cfg = None

    count = 0
    total = len(lookbacks) * len(rsi_periods) * len(div_thresholds) * len(atr_ratios)

    for lb, rsi_p, div_th, (atr_sl, atr_tp) in product(lookbacks, rsi_periods, div_thresholds, atr_ratios):
        s_cfg = {'lookback': lb, 'rsi_period': rsi_p, 'div_threshold': div_th}
        t_cfg = {'atr_sl': atr_sl, 'atr_tp': atr_tp, 'leverage': 10, 'pos_pct': 0.10}

        r = backtest({k: v.copy() for k, v in data.items()}, s_cfg, t_cfg)
        results.append((f"LB{lb}_RSI{rsi_p}_DIV{div_th}_ATR{atr_sl}:{atr_tp}", s_cfg, t_cfg, r))

        if r['ret'] > best_ret:
            best_ret = r['ret']
            best_cfg = (s_cfg, t_cfg, r)

        count += 1
        if count % 50 == 0:
            print(f"  진행: {count}/{total} ({count/total*100:.0f}%)")

    print(f"\n  완료: {total}개 조합 테스트")

    # 상위 10개 출력
    results.sort(key=lambda x: x[3]['ret'], reverse=True)
    print(f"\n{'설정':<35} {'수익%':>8} {'일평균':>8} {'10%+':>5} {'최대일':>7} {'MDD':>6} {'PF':>6}")
    print("-" * 80)
    for name, _, _, r in results[:10]:
        print(f"{name:<35} {r['ret']:>7.1f}% {r['daily']:>7.2f}% {r['big']:>4}일 {r['max_daily']:>6.1f}% {r['mdd']:>5.1f}% {r['pf']:>5.2f}")

    # ============================================================
    # 최고 설정으로 레버리지/포지션 크기 테스트
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] 최고 설정 + 레버리지/포지션 크기 튜닝")
    print("-" * 80)

    best_s_cfg, _, _ = best_cfg
    print(f"기준 설정: {best_s_cfg}")

    lev_pos_combos = [
        (10, 0.10), (10, 0.15), (10, 0.20),
        (15, 0.10), (15, 0.15), (15, 0.20),
        (20, 0.10), (20, 0.15), (20, 0.20),
        (25, 0.15), (25, 0.20),
        (30, 0.15), (30, 0.20),
    ]

    lev_results = []
    print(f"\n{'설정':<20} {'수익%':>8} {'일평균':>8} {'10%+':>5} {'최대일':>7} {'MDD':>6} {'PF':>6}")
    print("-" * 80)

    for lev, pos in lev_pos_combos:
        t_cfg = {'leverage': lev, 'pos_pct': pos, 'atr_sl': 1.0, 'atr_tp': 2.0}
        r = backtest({k: v.copy() for k, v in data.items()}, best_s_cfg, t_cfg)
        lev_results.append((f"L{lev}x P{int(pos*100)}%", r))
        print(f"L{lev}x P{int(pos*100)}%          {r['ret']:>7.1f}% {r['daily']:>7.2f}% {r['big']:>4}일 {r['max_daily']:>6.1f}% {r['mdd']:>5.1f}% {r['pf']:>5.2f}")

    # ============================================================
    # 거래량 필터 테스트
    # ============================================================
    print("\n" + "=" * 80)
    print("[3] 거래량 필터 테스트")
    print("-" * 80)

    vol_configs = [
        {'use_volume': False},
        {'use_volume': True, 'vol_min': 1.0},
        {'use_volume': True, 'vol_min': 1.5},
        {'use_volume': True, 'vol_min': 2.0},
    ]

    print(f"{'설정':<20} {'수익%':>8} {'일평균':>8} {'10%+':>5} {'거래':>6} {'MDD':>6}")
    print("-" * 80)

    for vcfg in vol_configs:
        s_cfg = {**best_s_cfg, **vcfg}
        t_cfg = {'leverage': 10, 'pos_pct': 0.10, 'atr_sl': 1.0, 'atr_tp': 2.0}
        r = backtest({k: v.copy() for k, v in data.items()}, s_cfg, t_cfg)
        label = f"Vol {vcfg.get('vol_min', 'OFF')}"
        print(f"{label:<20} {r['ret']:>7.1f}% {r['daily']:>7.2f}% {r['big']:>4}일 {r['trades']:>5}회 {r['mdd']:>5.1f}%")

    # ============================================================
    # 추세 필터 테스트
    # ============================================================
    print("\n" + "=" * 80)
    print("[4] 추세 필터 테스트 (EMA50)")
    print("-" * 80)

    trend_configs = [
        {'use_trend': False},
        {'use_trend': True},
    ]

    for tcfg in trend_configs:
        s_cfg = {**best_s_cfg, **tcfg}
        t_cfg = {'leverage': 10, 'pos_pct': 0.10, 'atr_sl': 1.0, 'atr_tp': 2.0}
        r = backtest({k: v.copy() for k, v in data.items()}, s_cfg, t_cfg)
        label = f"Trend {'ON' if tcfg['use_trend'] else 'OFF'}"
        print(f"{label:<20} {r['ret']:>7.1f}% {r['daily']:>7.2f}% {r['big']:>4}일 {r['trades']:>5}회 {r['mdd']:>5.1f}%")

    # ============================================================
    # 코인 수 테스트
    # ============================================================
    print("\n" + "=" * 80)
    print("[5] 코인 수 테스트")
    print("-" * 80)

    coin_sets = [
        ['BTCUSDT'],
        ['BTCUSDT', 'ETHUSDT'],
        ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
        ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT'],
    ]

    for coins in coin_sets:
        subset = {k: v.copy() for k, v in data.items() if k in coins}
        t_cfg = {'leverage': 10, 'pos_pct': 0.10, 'atr_sl': 1.0, 'atr_tp': 2.0}
        r = backtest(subset, best_s_cfg, t_cfg)
        label = f"{len(coins)}코인"
        print(f"{label:<20} {r['ret']:>7.1f}% {r['daily']:>7.2f}% {r['big']:>4}일 {r['trades']:>5}회 {r['mdd']:>5.1f}%")

    # ============================================================
    # 최종 최적 설정 출력
    # ============================================================
    print("\n" + "=" * 80)
    print("최종 결과")
    print("=" * 80)

    # 모든 결과 중 최고
    all_combos = results + [(n, None, None, r) for n, r in lev_results]
    best_overall = max(all_combos, key=lambda x: x[3]['ret'])
    best_daily = max(all_combos, key=lambda x: x[3]['daily'])
    best_big = max(all_combos, key=lambda x: x[3]['big'])

    print(f"\n★ 최고 수익: {best_overall[0]}")
    print(f"   수익: {best_overall[3]['ret']}%, 일평균: {best_overall[3]['daily']}%, MDD: {best_overall[3]['mdd']}%")

    print(f"\n★ 최고 일평균: {best_daily[0]}")
    print(f"   수익: {best_daily[3]['ret']}%, 일평균: {best_daily[3]['daily']}%, MDD: {best_daily[3]['mdd']}%")

    print(f"\n★ 10%+ 최다: {best_big[0]}")
    print(f"   수익: {best_big[3]['ret']}%, 10%+: {best_big[3]['big']}일, 최대일: {best_big[3]['max_daily']}%")

    # 최종 추천
    print("\n" + "=" * 80)
    print("추천 설정")
    print("=" * 80)

    # MDD 50% 이하에서 최고 수익
    safe_results = [x for x in all_combos if x[3]['mdd'] <= 50]
    if safe_results:
        best_safe = max(safe_results, key=lambda x: x[3]['ret'])
        print(f"\n◆ 안전 (MDD≤50%): {best_safe[0]}")
        print(f"   수익: {best_safe[3]['ret']}%, 일평균: {best_safe[3]['daily']}%, MDD: {best_safe[3]['mdd']}%")

    # MDD 30% 이하
    safer_results = [x for x in all_combos if x[3]['mdd'] <= 30]
    if safer_results:
        best_safer = max(safer_results, key=lambda x: x[3]['ret'])
        print(f"\n◆ 보수적 (MDD≤30%): {best_safer[0]}")
        print(f"   수익: {best_safer[3]['ret']}%, 일평균: {best_safer[3]['daily']}%, MDD: {best_safer[3]['mdd']}%")
