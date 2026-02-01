"""
공격적 스캘핑 전략 V2
이전 성공 요소 + 새로운 접근

핵심 개선:
1. 타이트한 손절 (진입 캔들 기준)
2. 확실한 반전 확인
3. 추세 필터 (상위 타임프레임)
4. 동적 목표가
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict

from scripts.data_cache import load_cached_data, fetch_klines_from_api, save_to_cache

# 실험 기록
EXPERIMENTS = []

def log_exp(name, params, results):
    EXPERIMENTS.append({'name': name, 'params': params, 'results': results})

# 지표 계산
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l)

def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def calc_bb(df, p=20, std=2.0):
    df = df.copy()
    df['bb_mid'] = df['close'].rolling(p).mean()
    s = df['close'].rolling(p).std()
    df['bb_upper'] = df['bb_mid'] + std * s
    df['bb_lower'] = df['bb_mid'] - std * s
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    return df

def calc_atr(df, p=14):
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean()

# ============================================================
# 백테스트 함수
# ============================================================
def backtest(all_data: Dict[str, pd.DataFrame], config: dict) -> dict:
    """
    config:
        initial_capital: 초기 자본
        leverage: 레버리지
        position_pct: 포지션 비율
        sl_pct: 손절 % (ATR 기반이면 atr_sl_mult 사용)
        tp_pct: 익절 % (또는 'bb_mid'로 BB 중간선)
        use_atr_sl: ATR 기반 손절 사용 여부
        atr_sl_mult: ATR 배수 (손절)
        atr_tp_mult: ATR 배수 (익절)
        max_positions: 최대 동시 포지션
        cooldown_candles: 쿨다운 캔들 수

        # RSI 설정
        rsi_period: RSI 기간
        rsi_high: 과매수 기준
        rsi_low: 과매도 기준
        require_reversal: RSI 반전 필요 여부

        # BB 설정
        use_bb_filter: BB 필터 사용
        bb_breach_min: 최소 돌파 비율
        bb_width_min: 최소 BB 폭
        bb_width_max: 최대 BB 폭

        # 추세 필터
        use_trend_filter: 추세 필터 사용
        trend_ema_period: 추세 EMA 기간
    """
    # 기본값
    cfg = {
        'initial_capital': 5_000_000,
        'leverage': 10,
        'position_pct': 0.10,
        'sl_pct': 1.0,
        'tp_pct': 2.0,
        'use_atr_sl': True,
        'atr_sl_mult': 1.0,
        'atr_tp_mult': 2.0,
        'max_positions': 4,
        'cooldown_candles': 2,
        'rsi_period': 7,
        'rsi_high': 75,
        'rsi_low': 25,
        'require_reversal': True,
        'use_bb_filter': True,
        'bb_breach_min': 0.2,
        'bb_width_min': 1.0,
        'bb_width_max': 5.0,
        'use_trend_filter': False,
        'trend_ema_period': 50,
        **config
    }

    # 지표 계산
    for symbol in all_data:
        df = all_data[symbol].copy()
        df['rsi'] = calc_rsi(df['close'], cfg['rsi_period'])
        df['rsi_prev'] = df['rsi'].shift(1)
        df = calc_bb(df)
        df['atr'] = calc_atr(df)
        df['ema_trend'] = calc_ema(df['close'], cfg['trend_ema_period'])

        # BB 돌파 비율
        bb_range = df['bb_upper'] - df['bb_lower']
        df['upper_breach'] = (df['high'] - df['bb_upper']) / bb_range
        df['lower_breach'] = (df['bb_lower'] - df['low']) / bb_range

        # RSI 반전 신호
        df['rsi_short_rev'] = (df['rsi_prev'] >= cfg['rsi_high']) & (df['rsi'] < df['rsi_prev'])
        df['rsi_long_rev'] = (df['rsi_prev'] <= cfg['rsi_low']) & (df['rsi'] > df['rsi_prev'])

        all_data[symbol] = df

    # 시간순 정렬
    all_bars = []
    for symbol, df in all_data.items():
        df = df.dropna()
        for _, row in df.iterrows():
            all_bars.append({'symbol': symbol, **row.to_dict()})
    all_bars.sort(key=lambda x: x['timestamp'])

    # 시간별 그룹화
    time_groups = {}
    for bar in all_bars:
        t = bar['timestamp']
        if t not in time_groups:
            time_groups[t] = {}
        time_groups[t][bar['symbol']] = bar

    sorted_times = sorted(time_groups.keys())

    # 시뮬레이션
    cash = cfg['initial_capital']
    positions = {}
    trades = []
    equity_curve = []
    last_exit = {}
    daily_pnl = defaultdict(float)

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
                    reason, exit_price = 'SL', pos['stop_loss']
                elif high >= pos['take_profit']:
                    reason, exit_price = 'TP', pos['take_profit']
            else:
                if high >= pos['stop_loss']:
                    reason, exit_price = 'SL', pos['stop_loss']
                elif low <= pos['take_profit']:
                    reason, exit_price = 'TP', pos['take_profit']

            if reason:
                if pos['side'] == 'long':
                    pnl_pct = (exit_price - entry) / entry * 100
                else:
                    pnl_pct = (entry - exit_price) / entry * 100

                realized = pnl_pct * cfg['leverage'] / 100 * pos['size']
                cash += pos['size'] + realized
                daily_pnl[t.date()] += realized

                trades.append({
                    'symbol': sym, 'side': pos['side'],
                    'entry_time': pos['entry_time'], 'exit_time': t,
                    'entry': entry, 'exit': exit_price,
                    'pnl_pct': round(pnl_pct * cfg['leverage'], 2),
                    'pnl_krw': round(realized, 0),
                    'reason': reason
                })
                closed.append(sym)
                last_exit[sym] = t

        for s in closed:
            del positions[s]

        # 자산 계산
        unrealized = 0
        for s, p in positions.items():
            if s in bars:
                if p['side'] == 'long':
                    unrealized += (bars[s]['close'] - p['entry_price']) / p['entry_price'] * cfg['leverage'] * p['size'] / 100
                else:
                    unrealized += (p['entry_price'] - bars[s]['close']) / p['entry_price'] * cfg['leverage'] * p['size'] / 100

        current_equity = cash + sum(p['size'] for p in positions.values()) + unrealized
        pos_size = current_equity * cfg['position_pct']

        # 진입
        if cash >= pos_size and len(positions) < cfg['max_positions']:
            candidates = []

            for sym, bar in bars.items():
                if sym in positions:
                    continue

                # 쿨다운 체크
                if sym in last_exit:
                    candle_mins = 15  # 15분봉 기준
                    if (t - last_exit[sym]).total_seconds() < cfg['cooldown_candles'] * candle_mins * 60:
                        continue

                price = bar['close']
                atr = bar['atr']
                bb_mid = bar['bb_mid']

                # BB 폭 필터
                if cfg['use_bb_filter']:
                    if bar['bb_width'] < cfg['bb_width_min'] or bar['bb_width'] > cfg['bb_width_max']:
                        continue

                # 추세 필터
                if cfg['use_trend_filter']:
                    trend_up = price > bar['ema_trend']
                    trend_down = price < bar['ema_trend']
                else:
                    trend_up = trend_down = True

                # 숏 진입 조건
                short_cond = bar['upper_breach'] >= cfg['bb_breach_min']
                if cfg['require_reversal']:
                    short_cond = short_cond and bar['rsi_short_rev']
                else:
                    short_cond = short_cond and bar['rsi'] >= cfg['rsi_high']

                if short_cond and (not cfg['use_trend_filter'] or trend_down):
                    if cfg['use_atr_sl']:
                        sl = price + atr * cfg['atr_sl_mult']
                        tp = price - atr * cfg['atr_tp_mult']
                    else:
                        sl = bar['high'] * (1 + cfg['sl_pct'] / 100)
                        tp = bb_mid if cfg.get('tp_to_bb_mid') else price * (1 - cfg['tp_pct'] / 100)

                    if price > tp:
                        candidates.append({
                            'symbol': sym, 'side': 'short', 'price': price,
                            'sl': sl, 'tp': tp, 'score': abs(bar['rsi'] - 50)
                        })

                # 롱 진입 조건
                long_cond = bar['lower_breach'] >= cfg['bb_breach_min']
                if cfg['require_reversal']:
                    long_cond = long_cond and bar['rsi_long_rev']
                else:
                    long_cond = long_cond and bar['rsi'] <= cfg['rsi_low']

                if long_cond and (not cfg['use_trend_filter'] or trend_up):
                    if cfg['use_atr_sl']:
                        sl = price - atr * cfg['atr_sl_mult']
                        tp = price + atr * cfg['atr_tp_mult']
                    else:
                        sl = bar['low'] * (1 - cfg['sl_pct'] / 100)
                        tp = bb_mid if cfg.get('tp_to_bb_mid') else price * (1 + cfg['tp_pct'] / 100)

                    if price < tp:
                        candidates.append({
                            'symbol': sym, 'side': 'long', 'price': price,
                            'sl': sl, 'tp': tp, 'score': abs(bar['rsi'] - 50)
                        })

            # RSI 극단값 기준 정렬
            candidates.sort(key=lambda x: x['score'], reverse=True)

            for c in candidates:
                if cash < pos_size or len(positions) >= cfg['max_positions']:
                    break
                positions[c['symbol']] = {
                    'side': c['side'], 'entry_price': c['price'], 'entry_time': t,
                    'stop_loss': c['sl'], 'take_profit': c['tp'], 'size': pos_size
                }
                cash -= pos_size

        equity_curve.append({'time': t, 'equity': current_equity})

    # 통계 계산
    return calc_stats(trades, equity_curve, daily_pnl, cfg['initial_capital'])


def calc_stats(trades, equity_curve, daily_pnl, initial):
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'return_pct': 0, 'max_dd': 0,
                'pf': 0, 'avg_daily': 0, 'big_days': 0, 'days': 0}

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]

    profit = sum(t['pnl_krw'] for t in wins) if wins else 0
    loss = abs(sum(t['pnl_krw'] for t in losses)) if losses else 0

    peak, max_dd = initial, 0
    for e in equity_curve:
        if e['equity'] > peak:
            peak = e['equity']
        dd = (peak - e['equity']) / peak * 100
        max_dd = max(max_dd, dd)

    final = equity_curve[-1]['equity'] if equity_curve else initial
    days = len(daily_pnl)

    daily_rets = [v / initial * 100 for v in daily_pnl.values()]
    big_days = len([d for d in daily_rets if d >= 10])

    return {
        'total_trades': len(trades),
        'win_rate': round(len(wins) / len(trades) * 100, 1),
        'avg_win': round(np.mean([t['pnl_pct'] for t in wins]), 2) if wins else 0,
        'avg_loss': round(np.mean([t['pnl_pct'] for t in losses]), 2) if losses else 0,
        'total_pnl': round(sum(t['pnl_krw'] for t in trades), 0),
        'return_pct': round((final - initial) / initial * 100, 1),
        'final': round(final, 0),
        'max_dd': round(max_dd, 1),
        'pf': round(profit / loss, 2) if loss > 0 else 999,
        'avg_daily': round(np.mean(daily_rets), 2) if daily_rets else 0,
        'big_days': big_days,
        'days': days,
        'tpd': round(len(trades) / max(days, 1), 2)
    }


def print_result(name, r):
    print(f"\n[{name}]")
    print(f"거래: {r['total_trades']}회 ({r['tpd']}회/일) | 승률: {r['win_rate']}%")
    print(f"평균: +{r['avg_win']}% / {r['avg_loss']}%")
    print(f"수익: ₩{r['total_pnl']:,.0f} ({r['return_pct']}%) | 최종: ₩{r['final']:,.0f}")
    print(f"MDD: {r['max_dd']}% | PF: {r['pf']} | 일평균: {r['avg_daily']}%")
    print(f"10%+ 날: {r['big_days']}일 / {r['days']}일")


if __name__ == '__main__':
    print("=" * 70)
    print("공격적 스캘핑 V2")
    print("=" * 70)

    # 데이터
    TARGET = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']
    data = load_cached_data(TARGET, '15m')
    print(f"코인: {len(data)}개")

    # ============================================================
    # 실험 배치 1: ATR 기반 손익비 테스트
    # ============================================================
    print("\n" + "="*70)
    print("배치 1: ATR 기반 손익비 테스트")
    print("="*70)

    configs = [
        {'name': 'ATR 1:1.5', 'atr_sl_mult': 1.0, 'atr_tp_mult': 1.5},
        {'name': 'ATR 1:2', 'atr_sl_mult': 1.0, 'atr_tp_mult': 2.0},
        {'name': 'ATR 1:3', 'atr_sl_mult': 1.0, 'atr_tp_mult': 3.0},
        {'name': 'ATR 0.5:1.5', 'atr_sl_mult': 0.5, 'atr_tp_mult': 1.5},
        {'name': 'ATR 0.5:2', 'atr_sl_mult': 0.5, 'atr_tp_mult': 2.0},
        {'name': 'ATR 0.7:2', 'atr_sl_mult': 0.7, 'atr_tp_mult': 2.0},
    ]

    results1 = []
    for cfg in configs:
        r = backtest({k: v.copy() for k, v in data.items()}, {
            'leverage': 10, 'position_pct': 0.10,
            'use_atr_sl': True, **cfg
        })
        results1.append((cfg['name'], r))
        print_result(cfg['name'], r)

    # ============================================================
    # 실험 배치 2: RSI 조건 테스트
    # ============================================================
    print("\n" + "="*70)
    print("배치 2: RSI 조건 테스트")
    print("="*70)

    configs2 = [
        {'name': 'RSI 70/30', 'rsi_high': 70, 'rsi_low': 30},
        {'name': 'RSI 75/25', 'rsi_high': 75, 'rsi_low': 25},
        {'name': 'RSI 80/20', 'rsi_high': 80, 'rsi_low': 20},
        {'name': 'RSI 85/15', 'rsi_high': 85, 'rsi_low': 15},
        {'name': 'RSI5 80/20', 'rsi_period': 5, 'rsi_high': 80, 'rsi_low': 20},
        {'name': 'RSI3 85/15', 'rsi_period': 3, 'rsi_high': 85, 'rsi_low': 15},
    ]

    results2 = []
    for cfg in configs2:
        r = backtest({k: v.copy() for k, v in data.items()}, {
            'leverage': 10, 'position_pct': 0.10,
            'atr_sl_mult': 0.7, 'atr_tp_mult': 2.0, **cfg
        })
        results2.append((cfg['name'], r))
        print_result(cfg['name'], r)

    # ============================================================
    # 실험 배치 3: 레버리지/포지션 크기
    # ============================================================
    print("\n" + "="*70)
    print("배치 3: 레버리지/포지션 크기")
    print("="*70)

    # 최고 RSI 조건 찾기
    best_rsi = max(results2, key=lambda x: x[1]['return_pct'])
    print(f"최고 RSI 조건: {best_rsi[0]}")

    configs3 = [
        {'name': '10x 10%', 'leverage': 10, 'position_pct': 0.10},
        {'name': '15x 10%', 'leverage': 15, 'position_pct': 0.10},
        {'name': '20x 10%', 'leverage': 20, 'position_pct': 0.10},
        {'name': '10x 15%', 'leverage': 10, 'position_pct': 0.15},
        {'name': '15x 15%', 'leverage': 15, 'position_pct': 0.15},
        {'name': '20x 15%', 'leverage': 20, 'position_pct': 0.15},
        {'name': '25x 20%', 'leverage': 25, 'position_pct': 0.20},
    ]

    results3 = []
    for cfg in configs3:
        r = backtest({k: v.copy() for k, v in data.items()}, {
            'atr_sl_mult': 0.7, 'atr_tp_mult': 2.0,
            'rsi_high': 80, 'rsi_low': 20, **cfg
        })
        results3.append((cfg['name'], r))
        print_result(cfg['name'], r)

    # ============================================================
    # 최종 결과 요약
    # ============================================================
    print("\n" + "="*70)
    print("전체 결과 요약")
    print("="*70)
    print(f"{'이름':<20} {'수익%':>8} {'일평균':>8} {'10%+':>5} {'MDD':>6} {'PF':>5} {'승률':>6}")
    print("-"*70)

    all_results = results1 + results2 + results3
    all_results.sort(key=lambda x: x[1]['return_pct'], reverse=True)

    for name, r in all_results[:15]:
        print(f"{name:<20} {r['return_pct']:>7.1f}% {r['avg_daily']:>7.2f}% "
              f"{r['big_days']:>4}일 {r['max_dd']:>5.1f}% {r['pf']:>5.2f} {r['win_rate']:>5.1f}%")

    print("="*70)

    # 최고 수익 설정
    best = all_results[0]
    print(f"\n★ 최고 수익 설정: {best[0]}")
    print(f"   수익: {best[1]['return_pct']}%, MDD: {best[1]['max_dd']}%, 일평균: {best[1]['avg_daily']}%")
