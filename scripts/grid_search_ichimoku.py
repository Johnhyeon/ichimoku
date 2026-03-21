"""
이치모쿠 전략 파라미터 그리드 서치

4시간봉 기반 숏 전용 이치모쿠 전략의 최적 매매 조건 탐색.
로컬 4h 데이터 사용 (API 호출 없음).

탐색 파라미터:
- rr_ratio: 손익비 (TP 거리 = SL 거리 × rr_ratio)
- trail_pct: 트레일링 스톱 %
- sl_buffer: SL 버퍼 (구름 경계에서 여유분)
- min_cloud_thickness: 최소 구름 두께 필터
- max_sl_pct: 최대 허용 SL 거리
- use_btc_filter: BTC 트렌드 필터 on/off
- cooldown_hours: 재진입 쿨다운
- max_positions: 최대 동시 포지션
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy
from typing import Dict, List, Optional
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ichimoku import calculate_ichimoku
from src.data_loader import DataLoader

import logging
logging.basicConfig(level=logging.WARNING)

# ─── 기본 설정 ──────────────────────────────────────────────

LEVERAGE = 20
POSITION_PCT = 0.05
INITIAL_CAPITAL = 1000.0

# 백테스트 대상 (backtest.py의 MAJOR_COINS와 동일, CCXT→로컬 매핑)
BACKTEST_COINS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT', 'SOL/USDT:USDT',
    'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'TON/USDT:USDT', 'TRX/USDT:USDT', 'AVAX/USDT:USDT',
    'SHIB/USDT:USDT', 'DOT/USDT:USDT', 'LINK/USDT:USDT', 'BCH/USDT:USDT', 'SUI/USDT:USDT',
    'NEAR/USDT:USDT', 'LTC/USDT:USDT', 'PEPE/USDT:USDT', 'UNI/USDT:USDT', 'APT/USDT:USDT',
    'ICP/USDT:USDT', 'ETC/USDT:USDT', 'RENDER/USDT:USDT', 'STX/USDT:USDT', 'HBAR/USDT:USDT',
    'XMR/USDT:USDT', 'ATOM/USDT:USDT', 'IMX/USDT:USDT', 'FIL/USDT:USDT', 'INJ/USDT:USDT',
    'XLM/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT', 'VET/USDT:USDT', 'FTM/USDT:USDT',
    'KAS/USDT:USDT', 'TIA/USDT:USDT', 'BONK/USDT:USDT', 'POL/USDT:USDT', 'SEI/USDT:USDT',
    'RUNE/USDT:USDT', 'FLOKI/USDT:USDT', 'WIF/USDT:USDT', 'JUP/USDT:USDT', 'AAVE/USDT:USDT',
    'ALGO/USDT:USDT', 'SAND/USDT:USDT', 'AXS/USDT:USDT', 'MANA/USDT:USDT', 'THETA/USDT:USDT',
]


def load_local_4h_data(loader: DataLoader, start: str = None, end: str = None) -> Dict[str, pd.DataFrame]:
    """로컬 4h 데이터 로드."""
    all_data = {}
    available = loader.get_available_symbols()

    # CCXT 심볼 → DataLoader 심볼 매핑
    for ccxt_sym in BACKTEST_COINS:
        if ccxt_sym in available:
            tfs = loader.get_available_timeframes(ccxt_sym)
            if '4h' in tfs:
                df = loader.load(ccxt_sym, '4h', start=start, end=end)
                if df is not None and len(df) >= 78:  # 52+26 최소 필요
                    # Bybit 심볼로 변환 (backtest.py 호환)
                    bybit_sym = ccxt_sym.replace('/USDT:USDT', 'USDT')
                    all_data[bybit_sym] = df

    return all_data


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    params: dict,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """단일 조합 백테스트 실행."""
    # 이치모쿠 지표 계산
    computed = {}
    for symbol, df in all_data.items():
        computed[symbol] = calculate_ichimoku(df.copy())

    # BTC 트렌드 계산
    btc_trends = {}
    if params.get('use_btc_filter', True) and 'BTCUSDT' in computed:
        btc_df = computed['BTCUSDT'].copy()
        btc_df['sma_26'] = btc_df['close'].rolling(26).mean()
        btc_df['sma_52'] = btc_df['close'].rolling(52).mean()
        for _, row in btc_df.iterrows():
            if pd.notna(row['sma_26']) and pd.notna(row['sma_52']):
                btc_trends[row['timestamp']] = row['sma_26'] > row['sma_52']

    # 시간별 바 그룹화
    all_bars = []
    for symbol, df in computed.items():
        df = df.dropna(subset=['tenkan', 'kijun', 'cloud_top', 'cloud_bottom'])
        for idx, row in df.iterrows():
            all_bars.append({
                'symbol': symbol,
                'time': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'tenkan': row['tenkan'],
                'kijun': row['kijun'],
                'cloud_top': row['cloud_top'],
                'cloud_bottom': row['cloud_bottom'],
                'cloud_thickness': row['cloud_thickness'],
                'cloud_green': row['cloud_green'],
                'tenkan_above': row['tenkan_above'],
                'tk_cross_up': row['tk_cross_up'],
                'tk_cross_down': row['tk_cross_down'],
                'kijun_cross_up': row['kijun_cross_up'],
                'kijun_cross_down': row['kijun_cross_down'],
                'chikou_bullish': row.get('chikou_bullish', False),
                'chikou_bearish': row.get('chikou_bearish', False),
                'above_cloud': row['above_cloud'],
                'below_cloud': row['below_cloud'],
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

    cash = initial_capital
    positions = {}
    trades = []
    equity_curve = []
    last_exit = {}
    peak_equity = initial_capital
    max_drawdown_pct = 0

    for t in sorted_times:
        bars = time_groups[t]
        closed = []
        btc_uptrend = btc_trends.get(t)

        # ── 포지션 청산 체크 ──
        for sym, pos in positions.items():
            if sym not in bars:
                continue

            bar = bars[sym]
            price = bar['close']
            high = bar['high']
            low = bar['low']
            entry = pos['entry_price']

            if pos['side'] == 'short':
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
                    pnl_pct = (entry - price) / entry * 100
                    position_size = pos['position_size']
                    realized_pnl = pnl_pct * LEVERAGE / 100 * position_size
                    cash += position_size + realized_pnl

                    trades.append({
                        'symbol': sym, 'side': 'short',
                        'entry_time': pos['entry_time'], 'exit_time': t,
                        'entry_price': entry, 'exit_price': price,
                        'pnl_pct': round(pnl_pct * LEVERAGE, 2),
                        'pnl_usd': round(realized_pnl, 2),
                        'reason': reason,
                    })
                    closed.append(sym)
                    last_exit[sym] = t

            else:  # long
                if high > pos['highest']:
                    pos['highest'] = high
                    if high >= pos['take_profit']:
                        pos['trailing'] = True
                        pos['trail_stop'] = max(pos['trail_stop'], high * (1 - params['trail_pct'] / 100))

                reason = None
                max_loss_price = entry * 0.98
                if low <= max_loss_price:
                    reason = 'MaxLoss'
                    price = max_loss_price
                elif low <= pos['stop_loss']:
                    reason = 'Stop'
                    price = max(pos['stop_loss'], low)
                elif pos.get('trailing') and low <= pos['trail_stop']:
                    reason = 'Trail'
                    price = pos['trail_stop']
                elif not pos.get('trailing') and high >= pos['take_profit']:
                    reason = 'TP'
                    price = pos['take_profit']
                elif bar['in_cloud'] or bar['below_cloud']:
                    reason = 'Cloud'
                    price = bar['close']

                if reason:
                    pnl_pct = (price - entry) / entry * 100
                    position_size = pos['position_size']
                    realized_pnl = pnl_pct * LEVERAGE / 100 * position_size
                    cash += position_size + realized_pnl

                    trades.append({
                        'symbol': sym, 'side': 'long',
                        'entry_time': pos['entry_time'], 'exit_time': t,
                        'entry_price': entry, 'exit_price': price,
                        'pnl_pct': round(pnl_pct * LEVERAGE, 2),
                        'pnl_usd': round(realized_pnl, 2),
                        'reason': reason,
                    })
                    closed.append(sym)
                    last_exit[sym] = t

        for s in closed:
            del positions[s]

        # 현재 에쿼티 계산
        unrealized = 0
        for sym, pos in positions.items():
            if sym in bars:
                p = bars[sym]['close']
                if pos['side'] == 'long':
                    pnl = (p - pos['entry_price']) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                else:
                    pnl = (pos['entry_price'] - p) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                unrealized += pnl

        current_equity = cash + sum(pos['position_size'] for pos in positions.values()) + unrealized
        position_size = current_equity * POSITION_PCT

        # MDD 추적
        if current_equity > peak_equity:
            peak_equity = current_equity
        dd = (peak_equity - current_equity) / peak_equity * 100
        if dd > max_drawdown_pct:
            max_drawdown_pct = dd

        # ── 신규 진입 ──
        if cash >= position_size and len(positions) < params['max_positions']:
            candidates = []

            for sym, bar in bars.items():
                if sym in positions:
                    continue
                if sym in last_exit:
                    if (t - last_exit[sym]).total_seconds() < params['cooldown_hours'] * 3600:
                        continue

                price = bar['close']
                thickness = bar['cloud_thickness']

                if bar['in_cloud']:
                    continue
                if thickness < params['min_cloud_thickness']:
                    continue

                # === 숏 조건 ===
                if bar['below_cloud'] and not bar['tenkan_above']:
                    has_signal = bar['tk_cross_down'] or bar['kijun_cross_down']
                    if not has_signal:
                        continue

                    if params.get('use_btc_filter', True) and btc_uptrend is False:
                        continue

                    score = 0
                    if bar.get('chikou_bearish', False):
                        score += 2
                    if not bar.get('cloud_green', True):
                        score += 1
                    if thickness > 1.0:
                        score += 1

                    stop_loss = bar['cloud_bottom'] * (1 + params['sl_buffer'] / 100)
                    sl_distance_pct = (stop_loss - price) / price * 100

                    if params['min_sl_pct'] <= sl_distance_pct <= params['max_sl_pct']:
                        take_profit = price * (1 - sl_distance_pct * params['rr_ratio'] / 100)
                        candidates.append({
                            'symbol': sym, 'side': 'short',
                            'price': price, 'stop_loss': stop_loss,
                            'take_profit': take_profit, 'score': score,
                            'thickness': thickness,
                        })

                # === 롱 조건 ===
                elif bar['above_cloud'] and bar['tenkan_above']:
                    has_signal = bar['tk_cross_up'] or bar['kijun_cross_up']
                    if not has_signal:
                        continue

                    if params.get('use_btc_filter', True) and btc_uptrend is True:
                        continue

                    if not bar.get('chikou_bullish', False):
                        continue

                    score = 0
                    if bar.get('chikou_bullish', False):
                        score += 3
                    if bar.get('cloud_green', False):
                        score += 2
                    if thickness > 1.0:
                        score += 2

                    stop_loss = bar['cloud_top'] * (1 - params['sl_buffer'] / 100)
                    sl_distance_pct = (price - stop_loss) / price * 100

                    if params['min_sl_pct'] <= sl_distance_pct <= params['max_sl_pct']:
                        take_profit = price * (1 + sl_distance_pct * params['rr_ratio'] / 100)
                        candidates.append({
                            'symbol': sym, 'side': 'long',
                            'price': price, 'stop_loss': stop_loss,
                            'take_profit': take_profit, 'score': score,
                            'thickness': thickness,
                        })

            candidates.sort(key=lambda x: (-x['score'], -x['thickness']))

            for cand in candidates:
                # 재계산
                unrealized = 0
                for sym, pos in positions.items():
                    if sym in bars:
                        p = bars[sym]['close']
                        if pos['side'] == 'long':
                            pnl = (p - pos['entry_price']) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                        else:
                            pnl = (pos['entry_price'] - p) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                        unrealized += pnl

                current_equity = cash + sum(pos['position_size'] for pos in positions.values()) + unrealized
                position_size = current_equity * POSITION_PCT

                if cash < position_size or len(positions) >= params['max_positions']:
                    break

                positions[cand['symbol']] = {
                    'side': cand['side'],
                    'entry_price': cand['price'],
                    'entry_time': t,
                    'stop_loss': cand['stop_loss'],
                    'take_profit': cand['take_profit'],
                    'highest': cand['price'],
                    'lowest': cand['price'],
                    'trail_stop': cand['stop_loss'],
                    'trailing': False,
                    'position_size': position_size,
                }
                cash -= position_size

        equity_curve.append({'time': t, 'equity': current_equity})

    # ── 통계 계산 ──
    n = len(trades)
    if n == 0:
        return None

    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    shorts = [t for t in trades if t['side'] == 'short']
    longs = [t for t in trades if t['side'] == 'long']
    short_wins = [t for t in shorts if t['pnl_usd'] > 0]
    long_wins = [t for t in longs if t['pnl_usd'] > 0]

    gross_profit = sum(t['pnl_usd'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl_usd'] for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    total_pnl = sum(t['pnl_usd'] for t in trades)

    final_equity = equity_curve[-1]['equity'] if equity_curve else initial_capital

    sl_count = sum(1 for t in trades if t['reason'] == 'Stop')
    tp_count = sum(1 for t in trades if t['reason'] == 'TP')
    trail_count = sum(1 for t in trades if t['reason'] == 'Trail')
    cloud_count = sum(1 for t in trades if t['reason'] == 'Cloud')
    maxloss_count = sum(1 for t in trades if t['reason'] == 'MaxLoss')

    return {
        'trades': n,
        'win_rate': len(wins) / n * 100,
        'pnl_pct': (final_equity - initial_capital) / initial_capital * 100,
        'pf': pf,
        'mdd_pct': max_drawdown_pct,
        'short_n': len(shorts),
        'long_n': len(longs),
        'short_wr': len(short_wins) / len(shorts) * 100 if shorts else 0,
        'long_wr': len(long_wins) / len(longs) * 100 if longs else 0,
        'sl': sl_count,
        'tp': tp_count,
        'trail': trail_count,
        'cloud': cloud_count,
        'maxloss': maxloss_count,
        'final_bal': final_equity,
    }


def main():
    print("=" * 60)
    print("  이치모쿠 전략 파라미터 그리드 서치")
    print("=" * 60)

    # ── 데이터 로드 ──
    print("\n4h 데이터 로드 중...")
    loader = DataLoader()
    all_data = load_local_4h_data(loader, start="2024-01-01", end="2026-03-18")
    print(f"로드 완료: {len(all_data)}개 심볼")

    # ── 그리드 파라미터 ──
    # Phase 1: 핵심 파라미터 (rr_ratio × trail_pct × sl_buffer)
    rr_ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    trail_pcts = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    sl_buffers = [0.1, 0.2, 0.3, 0.5]

    # Phase 2: 필터 파라미터 (별도 탐색)
    cloud_thicknesses = [0.1, 0.2, 0.3, 0.5, 1.0]
    max_sl_pcts = [4.0, 6.0, 8.0, 10.0]
    btc_filters = [True, False]
    cooldowns = [2, 4, 8, 12]
    max_positions_list = [3, 5, 7, 10]

    # ── Phase 1: rr × trail × sl_buffer ──
    base_params = {
        'min_cloud_thickness': 0.2,
        'min_sl_pct': 0.3,
        'max_sl_pct': 8.0,
        'sl_buffer': 0.2,
        'rr_ratio': 2.0,
        'trail_pct': 1.5,
        'cooldown_hours': 4,
        'max_positions': 5,
        'use_btc_filter': True,
        'short_only': True,
    }

    combos_p1 = list(product(rr_ratios, trail_pcts, sl_buffers))
    total_p1 = len(combos_p1)

    print(f"\n[Phase 1] rr_ratio × trail_pct × sl_buffer = {len(rr_ratios)} × {len(trail_pcts)} × {len(sl_buffers)} = {total_p1}개 조합\n")

    hdr = f"{'RR':>4} {'Trail':>5} {'SLBuf':>5} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'숏':>4} {'롱':>4} {'SL':>4} {'TP':>4} {'Trl':>4} {'Cld':>4} {'최종잔고':>10}"
    print(hdr)
    print("-" * len(hdr))

    results_p1 = []
    t0 = time.time()

    for i, (rr, trail, sl_buf) in enumerate(combos_p1):
        p = deepcopy(base_params)
        p['rr_ratio'] = rr
        p['trail_pct'] = trail
        p['sl_buffer'] = sl_buf

        r = run_backtest(all_data, p)
        if r:
            r['rr'] = rr
            r['trail'] = trail
            r['sl_buf'] = sl_buf
            results_p1.append(r)
            print(f"{rr:>4.1f} {trail:>4.1f}% {sl_buf:>4.1f}% {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_n']:>4} {r['long_n']:>4} {r['sl']:>4} {r['tp']:>4} {r['trail']:>4} {r['cloud']:>4} ${r['final_bal']:>9,.2f}")

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total_p1 - i - 1)
            print(f"  ... {i+1}/{total_p1} 완료 ({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed_p1 = time.time() - t0
    print(f"\nPhase 1 완료: {elapsed_p1:.1f}s\n")

    # Phase 1 TOP 20
    results_p1.sort(key=lambda x: x['pnl_pct'], reverse=True)
    print("=" * 120)
    print("  Phase 1 TOP 20 (수익률 기준)")
    print("=" * 120)
    print(f"{'#':>3} {'RR':>4} {'Trail':>5} {'SLBuf':>5} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'숏':>4} {'롱':>4} {'SL':>4} {'TP':>4} {'Trl':>4} {'Cld':>4} {'최종잔고':>10}")
    print("-" * 120)
    for i, r in enumerate(results_p1[:20]):
        print(f"{i+1:>3} {r['rr']:>4.1f} {r['trail']:>4.1f}% {r['sl_buf']:>4.1f}% {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_n']:>4} {r['long_n']:>4} {r['sl']:>4} {r['tp']:>4} {r['trail']:>4} {r['cloud']:>4} ${r['final_bal']:>9,.2f}")

    # Phase 1 TOP 10 PF
    results_p1_pf = sorted(results_p1, key=lambda x: x['pf'], reverse=True)
    print(f"\n{'=' * 120}")
    print("  Phase 1 TOP 10 (PF 기준)")
    print("=" * 120)
    print(f"{'#':>3} {'RR':>4} {'Trail':>5} {'SLBuf':>5} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'숏':>4} {'롱':>4} {'SL':>4} {'TP':>4} {'Trl':>4} {'Cld':>4} {'최종잔고':>10}")
    print("-" * 120)
    for i, r in enumerate(results_p1_pf[:10]):
        print(f"{i+1:>3} {r['rr']:>4.1f} {r['trail']:>4.1f}% {r['sl_buf']:>4.1f}% {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_n']:>4} {r['long_n']:>4} {r['sl']:>4} {r['tp']:>4} {r['trail']:>4} {r['cloud']:>4} ${r['final_bal']:>9,.2f}")

    # Phase 1 최적값 추출
    if results_p1:
        best_p1 = results_p1[0]
        best_rr = best_p1['rr']
        best_trail = best_p1['trail']
        best_sl_buf = best_p1['sl_buf']
        print(f"\n  → Phase 1 최적: RR={best_rr}, Trail={best_trail}%, SL_Buffer={best_sl_buf}%")
    else:
        print("  Phase 1 결과 없음!")
        return

    # ── Phase 2: 필터 파라미터 (Phase 1 최적값 기반) ──
    combos_p2 = list(product(cloud_thicknesses, max_sl_pcts, btc_filters, cooldowns, max_positions_list))
    total_p2 = len(combos_p2)

    print(f"\n[Phase 2] cloud × max_sl × btc × cooldown × max_pos = {total_p2}개 조합")
    print(f"Phase 1 최적값 고정: RR={best_rr}, Trail={best_trail}%, SLBuf={best_sl_buf}%\n")

    hdr2 = f"{'Cloud':>5} {'MaxSL':>5} {'BTC':>4} {'CD':>3} {'MP':>3} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'숏':>4} {'롱':>4} {'SL':>4} {'TP':>4} {'Trl':>4} {'Cld':>4} {'최종잔고':>10}"
    print(hdr2)
    print("-" * len(hdr2))

    results_p2 = []
    t1 = time.time()

    for i, (cloud, max_sl, btc_f, cd, mp) in enumerate(combos_p2):
        p = deepcopy(base_params)
        p['rr_ratio'] = best_rr
        p['trail_pct'] = best_trail
        p['sl_buffer'] = best_sl_buf
        p['min_cloud_thickness'] = cloud
        p['max_sl_pct'] = max_sl
        p['use_btc_filter'] = btc_f
        p['cooldown_hours'] = cd
        p['max_positions'] = mp

        r = run_backtest(all_data, p)
        if r:
            r['cloud'] = cloud
            r['max_sl'] = max_sl
            r['btc_f'] = btc_f
            r['cd'] = cd
            r['mp'] = mp
            results_p2.append(r)
            btc_str = "Y" if btc_f else "N"
            print(f"{cloud:>4.1f}% {max_sl:>4.1f}% {btc_str:>4} {cd:>3} {mp:>3} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_n']:>4} {r['long_n']:>4} {r['sl']:>4} {r['tp']:>4} {r['trail']:>4} {r['cloud']:>4} ${r['final_bal']:>9,.2f}")

        if (i + 1) % 40 == 0:
            elapsed = time.time() - t1
            eta = elapsed / (i + 1) * (total_p2 - i - 1)
            print(f"  ... {i+1}/{total_p2} 완료 ({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed_p2 = time.time() - t1
    print(f"\nPhase 2 완료: {elapsed_p2:.1f}s\n")

    # Phase 2 TOP 20
    results_p2.sort(key=lambda x: x['pnl_pct'], reverse=True)
    print("=" * 120)
    print("  Phase 2 TOP 20 (수익률 기준)")
    print("=" * 120)
    print(f"{'#':>3} {'Cloud':>5} {'MaxSL':>5} {'BTC':>4} {'CD':>3} {'MP':>3} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'숏':>4} {'롱':>4} {'SL':>4} {'TP':>4} {'Trl':>4} {'Cld':>4} {'최종잔고':>10}")
    print("-" * 120)
    for i, r in enumerate(results_p2[:20]):
        btc_str = "Y" if r['btc_f'] else "N"
        print(f"{i+1:>3} {r['cloud']:>4.1f}% {r['max_sl']:>4.1f}% {btc_str:>4} {r['cd']:>3} {r['mp']:>3} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_n']:>4} {r['long_n']:>4} {r['sl']:>4} {r['tp']:>4} {r['trail']:>4} {r['cloud']:>4} ${r['final_bal']:>9,.2f}")

    # Phase 2 TOP 10 PF
    results_p2_pf = sorted(results_p2, key=lambda x: x['pf'], reverse=True)
    print(f"\n{'=' * 120}")
    print("  Phase 2 TOP 10 (PF 기준)")
    print("=" * 120)
    print(f"{'#':>3} {'Cloud':>5} {'MaxSL':>5} {'BTC':>4} {'CD':>3} {'MP':>3} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'숏':>4} {'롱':>4} {'SL':>4} {'TP':>4} {'Trl':>4} {'Cld':>4} {'최종잔고':>10}")
    print("-" * 120)
    for i, r in enumerate(results_p2_pf[:10]):
        btc_str = "Y" if r['btc_f'] else "N"
        print(f"{i+1:>3} {r['cloud']:>4.1f}% {r['max_sl']:>4.1f}% {btc_str:>4} {r['cd']:>3} {r['mp']:>3} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_n']:>4} {r['long_n']:>4} {r['sl']:>4} {r['tp']:>4} {r['trail']:>4} {r['cloud']:>4} ${r['final_bal']:>9,.2f}")

    # Phase 2 TOP 10 MDD 낮은 순
    results_p2_mdd = sorted([r for r in results_p2 if r['pnl_pct'] > 0], key=lambda x: x['mdd_pct'])
    print(f"\n{'=' * 120}")
    print("  Phase 2 TOP 10 (MDD 낮은 순, 수익률 양수 필터)")
    print("=" * 120)
    print(f"{'#':>3} {'Cloud':>5} {'MaxSL':>5} {'BTC':>4} {'CD':>3} {'MP':>3} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'숏':>4} {'롱':>4} {'SL':>4} {'TP':>4} {'Trl':>4} {'Cld':>4} {'최종잔고':>10}")
    print("-" * 120)
    for i, r in enumerate(results_p2_mdd[:10]):
        btc_str = "Y" if r['btc_f'] else "N"
        print(f"{i+1:>3} {r['cloud']:>4.1f}% {r['max_sl']:>4.1f}% {btc_str:>4} {r['cd']:>3} {r['mp']:>3} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_n']:>4} {r['long_n']:>4} {r['sl']:>4} {r['tp']:>4} {r['trail']:>4} {r['cloud']:>4} ${r['final_bal']:>9,.2f}")

    # ── 최종 요약 ──
    if results_p2:
        best_p2 = results_p2[0]
        print(f"\n{'=' * 120}")
        print("  최종 최적 파라미터")
        print("=" * 120)
        btc_str = "ON" if best_p2['btc_f'] else "OFF"
        print(f"  RR Ratio      : {best_rr}")
        print(f"  Trail %       : {best_trail}%")
        print(f"  SL Buffer     : {best_sl_buf}%")
        print(f"  Cloud Thick   : {best_p2['cloud']}%")
        print(f"  Max SL %      : {best_p2['max_sl']}%")
        print(f"  BTC Filter    : {btc_str}")
        print(f"  Cooldown      : {best_p2['cd']}h")
        print(f"  Max Positions : {best_p2['mp']}")
        print(f"  ─────────────────────────────")
        print(f"  거래수  : {best_p2['trades']}")
        print(f"  승률    : {best_p2['win_rate']:.1f}%")
        print(f"  수익률  : {best_p2['pnl_pct']:+.1f}%")
        print(f"  PF      : {best_p2['pf']:.2f}")
        print(f"  MDD     : {best_p2['mdd_pct']:.1f}%")
        print(f"  최종잔고: ${best_p2['final_bal']:,.2f}")

    total_elapsed = time.time() - t0
    print(f"\n총 소요시간: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
