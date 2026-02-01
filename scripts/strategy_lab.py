"""
전략 연구소 - 모든 전략 테스트
단타, 스윙, 추세추종, 브레이크아웃 등 모든 전략 테스트

목표: 하루 10% 수익 (또는 이에 근접한 최고 수익)
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

from datetime import datetime, timedelta
from typing import Dict
import pandas as pd
import numpy as np
from collections import defaultdict

from scripts.data_cache import load_cached_data, fetch_klines_from_api, save_to_cache

# ============================================================
# 지표 함수
# ============================================================
def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l)

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def sma(s, p):
    return s.rolling(p).mean()

def bb(df, p=20, std=2.0):
    df = df.copy()
    df['bb_mid'] = sma(df['close'], p)
    s = df['close'].rolling(p).std()
    df['bb_upper'] = df['bb_mid'] + std * s
    df['bb_lower'] = df['bb_mid'] - std * s
    return df

def atr(df, p=14):
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def donchian(df, p=20):
    df = df.copy()
    df['dc_high'] = df['high'].rolling(p).max()
    df['dc_low'] = df['low'].rolling(p).min()
    df['dc_mid'] = (df['dc_high'] + df['dc_low']) / 2
    return df

def supertrend(df, period=10, mult=3.0):
    df = df.copy()
    hl2 = (df['high'] + df['low']) / 2
    atr_val = atr(df, period)

    upper = hl2 + mult * atr_val
    lower = hl2 - mult * atr_val

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    st.iloc[0] = lower.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        if df['close'].iloc[i] > st.iloc[i-1]:
            st.iloc[i] = max(lower.iloc[i], st.iloc[i-1]) if direction.iloc[i-1] == 1 else lower.iloc[i]
            direction.iloc[i] = 1
        else:
            st.iloc[i] = min(upper.iloc[i], st.iloc[i-1]) if direction.iloc[i-1] == -1 else upper.iloc[i]
            direction.iloc[i] = -1

    df['supertrend'] = st
    df['st_dir'] = direction
    return df

# ============================================================
# 전략들
# ============================================================

def strategy_momentum_breakout(df, cfg):
    """모멘텀 브레이크아웃: 신고가 돌파 시 매수"""
    df = df.copy()
    lookback = cfg.get('lookback', 20)

    df['highest'] = df['high'].rolling(lookback).max().shift(1)
    df['lowest'] = df['low'].rolling(lookback).min().shift(1)
    df['atr'] = atr(df, 14)

    df['long'] = df['close'] > df['highest']
    df['short'] = df['close'] < df['lowest']

    return df

def strategy_ema_cross(df, cfg):
    """EMA 크로스: 골든/데드 크로스"""
    df = df.copy()
    fast = cfg.get('fast', 9)
    slow = cfg.get('slow', 21)

    df['ema_fast'] = ema(df['close'], fast)
    df['ema_slow'] = ema(df['close'], slow)
    df['ema_fast_prev'] = df['ema_fast'].shift(1)
    df['ema_slow_prev'] = df['ema_slow'].shift(1)
    df['atr'] = atr(df, 14)

    df['long'] = (df['ema_fast_prev'] < df['ema_slow_prev']) & (df['ema_fast'] > df['ema_slow'])
    df['short'] = (df['ema_fast_prev'] > df['ema_slow_prev']) & (df['ema_fast'] < df['ema_slow'])

    return df

def strategy_supertrend(df, cfg):
    """슈퍼트렌드: 추세 전환 시 진입"""
    df = supertrend(df, cfg.get('period', 10), cfg.get('mult', 3.0))
    df['st_dir_prev'] = df['st_dir'].shift(1)
    df['atr'] = atr(df, 14)

    df['long'] = (df['st_dir_prev'] == -1) & (df['st_dir'] == 1)
    df['short'] = (df['st_dir_prev'] == 1) & (df['st_dir'] == -1)

    return df

def strategy_donchian(df, cfg):
    """돈치안 채널: 채널 돌파 시 추세 추종"""
    df = donchian(df, cfg.get('period', 20))
    df['atr'] = atr(df, 14)

    df['long'] = df['close'] > df['dc_high'].shift(1)
    df['short'] = df['close'] < df['dc_low'].shift(1)

    return df

def strategy_bb_squeeze(df, cfg):
    """BB 스퀴즈: 변동성 축소 후 확대 시 진입"""
    df = bb(df, 20, 2.0)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    df['bb_width_ma'] = df['bb_width'].rolling(20).mean()
    df['atr'] = atr(df, 14)

    # 스퀴즈: 현재 폭이 평균의 50% 이하
    squeeze = df['bb_width'] < df['bb_width_ma'] * 0.5
    squeeze_prev = squeeze.shift(1)

    # 스퀴즈 해제 시 방향 결정
    df['long'] = squeeze_prev & ~squeeze & (df['close'] > df['bb_mid'])
    df['short'] = squeeze_prev & ~squeeze & (df['close'] < df['bb_mid'])

    return df

def strategy_rsi_divergence(df, cfg):
    """RSI 다이버전스 (간소화): 가격 신저가 + RSI 상승"""
    df = df.copy()
    lookback = cfg.get('lookback', 14)
    df['rsi'] = rsi(df['close'], 14)
    df['atr'] = atr(df, 14)

    # 가격 N봉 최저
    df['price_low'] = df['low'].rolling(lookback).min()
    df['rsi_at_low'] = df['rsi'].rolling(lookback).min()

    # 현재가 신저가 근처 but RSI는 이전 저점보다 높음
    df['long'] = (df['low'] <= df['price_low'] * 1.005) & (df['rsi'] > df['rsi_at_low'].shift(lookback) + 5)

    # 숏도 마찬가지
    df['price_high'] = df['high'].rolling(lookback).max()
    df['rsi_at_high'] = df['rsi'].rolling(lookback).max()
    df['short'] = (df['high'] >= df['price_high'] * 0.995) & (df['rsi'] < df['rsi_at_high'].shift(lookback) - 5)

    return df

def strategy_mean_reversion_atr(df, cfg):
    """평균 회귀 + ATR 기반 손익비"""
    df = bb(df, 20, 2.0)
    df['rsi'] = rsi(df['close'], cfg.get('rsi_period', 7))
    df['rsi_prev'] = df['rsi'].shift(1)
    df['atr'] = atr(df, 14)

    rsi_low = cfg.get('rsi_low', 25)
    rsi_high = cfg.get('rsi_high', 75)

    # RSI 반전 확인
    df['long'] = (df['rsi_prev'] <= rsi_low) & (df['rsi'] > df['rsi_prev']) & (df['close'] < df['bb_mid'])
    df['short'] = (df['rsi_prev'] >= rsi_high) & (df['rsi'] < df['rsi_prev']) & (df['close'] > df['bb_mid'])

    return df

def strategy_triple_ema(df, cfg):
    """트리플 EMA: 3개 EMA 정렬"""
    df = df.copy()
    df['ema_short'] = ema(df['close'], cfg.get('short', 5))
    df['ema_mid'] = ema(df['close'], cfg.get('mid', 13))
    df['ema_long'] = ema(df['close'], cfg.get('long', 34))
    df['atr'] = atr(df, 14)

    # EMA 정렬 확인
    df['long'] = (df['ema_short'] > df['ema_mid']) & (df['ema_mid'] > df['ema_long']) & \
                 (df['ema_short'].shift(1) <= df['ema_mid'].shift(1))
    df['short'] = (df['ema_short'] < df['ema_mid']) & (df['ema_mid'] < df['ema_long']) & \
                  (df['ema_short'].shift(1) >= df['ema_mid'].shift(1))

    return df

def strategy_volume_breakout(df, cfg):
    """거래량 브레이크아웃: 거래량 급등 + 가격 돌파"""
    df = df.copy()
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['highest'] = df['high'].rolling(cfg.get('lookback', 10)).max().shift(1)
    df['lowest'] = df['low'].rolling(cfg.get('lookback', 10)).min().shift(1)
    df['atr'] = atr(df, 14)

    vol_threshold = cfg.get('vol_threshold', 2.0)

    df['long'] = (df['close'] > df['highest']) & (df['vol_ratio'] > vol_threshold)
    df['short'] = (df['close'] < df['lowest']) & (df['vol_ratio'] > vol_threshold)

    return df

# ============================================================
# 백테스트 엔진
# ============================================================
def backtest(all_data: Dict[str, pd.DataFrame], strategy_fn, strategy_cfg: dict, trade_cfg: dict) -> dict:
    """범용 백테스트"""
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

    # 지표 계산
    for sym in all_data:
        all_data[sym] = strategy_fn(all_data[sym], strategy_cfg)

    # 바 정렬
    bars = []
    for sym, df in all_data.items():
        df = df.dropna()
        for _, row in df.iterrows():
            bars.append({'symbol': sym, **row.to_dict()})
    bars.sort(key=lambda x: x['timestamp'])

    # 시간 그룹
    tg = {}
    for b in bars:
        t = b['timestamp']
        if t not in tg:
            tg[t] = {}
        tg[t][b['symbol']] = b

    times = sorted(tg.keys())

    # 시뮬레이션
    cash = cfg['initial']
    positions = {}
    trades = []
    equity = []
    last_exit = {}
    daily_pnl = defaultdict(float)

    for t in times:
        current_bars = tg[t]
        closed = []

        # 청산
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
                    'sym': sym, 'side': pos['side'], 't_in': pos['t_in'], 't_out': t,
                    'entry': entry, 'exit': exit_p,
                    'pnl': round(pnl * cfg['leverage'], 2),
                    'pnl_krw': round(realized, 0), 'reason': reason
                })
                closed.append(sym)
                last_exit[sym] = t

        for s in closed:
            del positions[s]

        # 자산
        unreal = sum(
            ((current_bars[s]['close'] - p['entry']) / p['entry'] if p['side'] == 'long'
             else (p['entry'] - current_bars[s]['close']) / p['entry']) * cfg['leverage'] * p['size'] / 100
            for s, p in positions.items() if s in current_bars
        )
        eq = cash + sum(p['size'] for p in positions.values()) + unreal
        pos_size = eq * cfg['pos_pct']

        # 진입
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
                    positions[sym] = {'side': 'long', 'entry': price, 't_in': t, 'sl': sl, 'tp': tp, 'size': pos_size}
                    cash -= pos_size
                elif b.get('short', False):
                    sl = price + a * cfg['atr_sl']
                    tp = price - a * cfg['atr_tp']
                    positions[sym] = {'side': 'short', 'entry': price, 't_in': t, 'sl': sl, 'tp': tp, 'size': pos_size}
                    cash -= pos_size

                if len(positions) >= cfg['max_pos']:
                    break

        equity.append({'t': t, 'eq': eq})

    return calc_stats(trades, equity, daily_pnl, cfg['initial'])


def calc_stats(trades, equity, daily_pnl, initial):
    if not trades:
        return {'trades': 0, 'wr': 0, 'ret': 0, 'mdd': 0, 'pf': 0, 'daily': 0, 'big': 0, 'days': 0}

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
        'tpd': round(len(trades) / max(days, 1), 2)
    }


def p(name, r):
    print(f"{name:<25} {r['ret']:>7.1f}% {r['daily']:>7.2f}% {r['big']:>4}일 "
          f"{r['mdd']:>5.1f}% {r['pf']:>5.2f} {r['wr']:>5.1f}% {r['trades']:>5}회")


if __name__ == '__main__':
    print("=" * 80)
    print("전략 연구소 - 모든 전략 테스트")
    print("=" * 80)

    # 데이터 로드
    COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']
    data_15m = load_cached_data(COINS, '15m')
    data_1h = load_cached_data(COINS, '1h')
    data_4h = load_cached_data(COINS, '4h')

    # 1시간, 4시간봉 없으면 다운로드
    for coin in COINS:
        if coin not in data_1h:
            print(f"  {coin} 1h 다운로드...")
            df = fetch_klines_from_api(coin, 60, 10000)
            if df is not None:
                save_to_cache(df, coin, '1h')
                data_1h[coin] = df
        if coin not in data_4h:
            print(f"  {coin} 4h 다운로드...")
            df = fetch_klines_from_api(coin, 240, 5000)
            if df is not None:
                save_to_cache(df, coin, '4h')
                data_4h[coin] = df

    print(f"\n15분봉: {len(data_15m)}개, 1시간봉: {len(data_1h)}개, 4시간봉: {len(data_4h)}개")

    results = []

    # ============================================================
    # 15분봉 전략들
    # ============================================================
    print("\n" + "=" * 80)
    print("15분봉 전략")
    print("=" * 80)
    print(f"{'전략':<25} {'수익%':>8} {'일평균':>8} {'10%+':>5} {'MDD':>6} {'PF':>6} {'승률':>6} {'거래':>6}")
    print("-" * 80)

    strategies_15m = [
        ('MeanRev ATR1:2', strategy_mean_reversion_atr, {'rsi_low': 25, 'rsi_high': 75}, {'atr_sl': 1, 'atr_tp': 2}),
        ('MeanRev ATR1:3', strategy_mean_reversion_atr, {'rsi_low': 25, 'rsi_high': 75}, {'atr_sl': 1, 'atr_tp': 3}),
        ('MeanRev ATR0.7:2', strategy_mean_reversion_atr, {'rsi_low': 25, 'rsi_high': 75}, {'atr_sl': 0.7, 'atr_tp': 2}),
        ('EMA Cross 9/21', strategy_ema_cross, {'fast': 9, 'slow': 21}, {'atr_sl': 1.5, 'atr_tp': 3}),
        ('EMA Cross 5/13', strategy_ema_cross, {'fast': 5, 'slow': 13}, {'atr_sl': 1, 'atr_tp': 2}),
        ('Triple EMA', strategy_triple_ema, {'short': 5, 'mid': 13, 'long': 34}, {'atr_sl': 1.5, 'atr_tp': 3}),
        ('Momentum 20', strategy_momentum_breakout, {'lookback': 20}, {'atr_sl': 1.5, 'atr_tp': 3}),
        ('Momentum 10', strategy_momentum_breakout, {'lookback': 10}, {'atr_sl': 1, 'atr_tp': 2}),
        ('Volume Breakout', strategy_volume_breakout, {'lookback': 10, 'vol_threshold': 2.0}, {'atr_sl': 1, 'atr_tp': 2}),
        ('BB Squeeze', strategy_bb_squeeze, {}, {'atr_sl': 1.5, 'atr_tp': 3}),
        ('SuperTrend 10/3', strategy_supertrend, {'period': 10, 'mult': 3.0}, {'atr_sl': 1.5, 'atr_tp': 3}),
        ('SuperTrend 7/2', strategy_supertrend, {'period': 7, 'mult': 2.0}, {'atr_sl': 1, 'atr_tp': 2}),
        ('Donchian 20', strategy_donchian, {'period': 20}, {'atr_sl': 1.5, 'atr_tp': 3}),
        ('Donchian 10', strategy_donchian, {'period': 10}, {'atr_sl': 1, 'atr_tp': 2}),
        ('RSI Divergence', strategy_rsi_divergence, {'lookback': 14}, {'atr_sl': 1, 'atr_tp': 2}),
    ]

    for name, fn, s_cfg, t_cfg in strategies_15m:
        r = backtest({k: v.copy() for k, v in data_15m.items()}, fn, s_cfg, t_cfg)
        results.append((f"15m {name}", r))
        p(name, r)

    # ============================================================
    # 1시간봉 전략들 (스윙)
    # ============================================================
    print("\n" + "=" * 80)
    print("1시간봉 전략 (스윙)")
    print("=" * 80)
    print(f"{'전략':<25} {'수익%':>8} {'일평균':>8} {'10%+':>5} {'MDD':>6} {'PF':>6} {'승률':>6} {'거래':>6}")
    print("-" * 80)

    strategies_1h = [
        ('MeanRev ATR1:3', strategy_mean_reversion_atr, {'rsi_low': 30, 'rsi_high': 70}, {'atr_sl': 1, 'atr_tp': 3, 'cooldown': 4}),
        ('EMA Cross 9/21', strategy_ema_cross, {'fast': 9, 'slow': 21}, {'atr_sl': 1.5, 'atr_tp': 4, 'cooldown': 4}),
        ('SuperTrend', strategy_supertrend, {'period': 10, 'mult': 3.0}, {'atr_sl': 1.5, 'atr_tp': 4, 'cooldown': 4}),
        ('Donchian 20', strategy_donchian, {'period': 20}, {'atr_sl': 1.5, 'atr_tp': 4, 'cooldown': 4}),
        ('Momentum 20', strategy_momentum_breakout, {'lookback': 20}, {'atr_sl': 1.5, 'atr_tp': 4, 'cooldown': 4}),
    ]

    for name, fn, s_cfg, t_cfg in strategies_1h:
        if data_1h:
            r = backtest({k: v.copy() for k, v in data_1h.items()}, fn, s_cfg, t_cfg)
            results.append((f"1h {name}", r))
            p(name, r)

    # ============================================================
    # 4시간봉 전략들 (포지션 트레이딩)
    # ============================================================
    print("\n" + "=" * 80)
    print("4시간봉 전략 (포지션)")
    print("=" * 80)
    print(f"{'전략':<25} {'수익%':>8} {'일평균':>8} {'10%+':>5} {'MDD':>6} {'PF':>6} {'승률':>6} {'거래':>6}")
    print("-" * 80)

    strategies_4h = [
        ('SuperTrend', strategy_supertrend, {'period': 10, 'mult': 3.0}, {'atr_sl': 2, 'atr_tp': 5, 'cooldown': 6}),
        ('EMA Cross 9/21', strategy_ema_cross, {'fast': 9, 'slow': 21}, {'atr_sl': 2, 'atr_tp': 5, 'cooldown': 6}),
        ('Donchian 20', strategy_donchian, {'period': 20}, {'atr_sl': 2, 'atr_tp': 5, 'cooldown': 6}),
    ]

    for name, fn, s_cfg, t_cfg in strategies_4h:
        if data_4h:
            r = backtest({k: v.copy() for k, v in data_4h.items()}, fn, s_cfg, t_cfg)
            results.append((f"4h {name}", r))
            p(name, r)

    # ============================================================
    # 공격적 설정 테스트 (상위 5개 전략)
    # ============================================================
    print("\n" + "=" * 80)
    print("상위 전략 + 공격적 레버리지 테스트")
    print("=" * 80)

    # 수익 기준 상위 5개
    top5 = sorted(results, key=lambda x: x[1]['ret'], reverse=True)[:5]
    print(f"상위 5개: {[t[0] for t in top5]}")

    print(f"\n{'전략':<30} {'수익%':>8} {'일평균':>8} {'10%+':>5} {'MDD':>6}")
    print("-" * 80)

    # 공격적 설정으로 재테스트
    agg_configs = [
        {'leverage': 15, 'pos_pct': 0.15},
        {'leverage': 20, 'pos_pct': 0.15},
        {'leverage': 20, 'pos_pct': 0.20},
        {'leverage': 25, 'pos_pct': 0.20},
    ]

    for name, base_r in top5:
        # 원래 전략 정보 추출 (15m 기준)
        tf = name.split()[0]
        strat_name = ' '.join(name.split()[1:])

        # 해당 전략 찾기
        if tf == '15m':
            data = data_15m
            strats = strategies_15m
        elif tf == '1h':
            data = data_1h
            strats = strategies_1h
        else:
            data = data_4h
            strats = strategies_4h

        for s_name, fn, s_cfg, t_cfg in strats:
            if s_name == strat_name:
                for agg in agg_configs:
                    new_cfg = {**t_cfg, **agg}
                    r = backtest({k: v.copy() for k, v in data.items()}, fn, s_cfg, new_cfg)
                    label = f"{name} L{agg['leverage']}x P{int(agg['pos_pct']*100)}%"
                    print(f"{label:<30} {r['ret']:>7.1f}% {r['daily']:>7.2f}% {r['big']:>4}일 {r['mdd']:>5.1f}%")
                break

    # ============================================================
    # 최종 요약
    # ============================================================
    print("\n" + "=" * 80)
    print("전체 결과 TOP 10")
    print("=" * 80)
    print(f"{'전략':<30} {'수익%':>8} {'일평균':>8} {'10%+':>5} {'MDD':>6} {'PF':>6}")
    print("-" * 80)

    results.sort(key=lambda x: x[1]['ret'], reverse=True)
    for name, r in results[:10]:
        print(f"{name:<30} {r['ret']:>7.1f}% {r['daily']:>7.2f}% {r['big']:>4}일 {r['mdd']:>5.1f}% {r['pf']:>5.2f}")

    print("=" * 80)

    # 최고 수익 전략
    best = results[0]
    print(f"\n★ 최고 수익: {best[0]}")
    print(f"   수익: {best[1]['ret']}%, 일평균: {best[1]['daily']}%, MDD: {best[1]['mdd']}%")

    # 일평균 최고
    best_daily = max(results, key=lambda x: x[1]['daily'])
    print(f"\n★ 일평균 최고: {best_daily[0]}")
    print(f"   수익: {best_daily[1]['ret']}%, 일평균: {best_daily[1]['daily']}%, MDD: {best_daily[1]['mdd']}%")
