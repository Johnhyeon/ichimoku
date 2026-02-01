#!/usr/bin/env python3
"""
초기 급등 전략 백테스트

5분봉 기반 초기 급등 감지 전략 백테스트
- 거래량 스파이크 + 가격 급등 감지
- 횡보 → 급등 전환 포착

사용법:
    python scripts/backtest_early_surge.py
    python scripts/backtest_early_surge.py --symbol "ZORA/USDT:USDT"
    python scripts/backtest_early_surge.py --days 7
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd

from src.bybit_client import BybitClient
from src.data_fetcher import DataFetcher
from src.early_surge_detector import (
    EarlySurgeDetector,
    EARLY_SURGE_PARAMS,
)


def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    params: dict = EARLY_SURGE_PARAMS,
    verbose: bool = True
) -> Dict:
    """
    단일 심볼 백테스트

    Args:
        symbol: 심볼
        df: 5분봉 OHLCV DataFrame
        params: 전략 파라미터
        verbose: 상세 출력

    Returns:
        백테스트 결과
    """
    if df is None or len(df) < 100:
        return {'trades': [], 'error': 'insufficient_data'}

    df = df.reset_index() if 'timestamp' not in df.columns else df

    detector = EarlySurgeDetector(None, params)
    trades = []
    position = None

    leverage = params.get('leverage', 5)
    sl_pct = params.get('sl_pct', 5)
    tp_pct = params.get('tp_pct', 50)
    trail_start = params.get('trail_start_pct', 25)
    trail_pct = params.get('trail_pct', 8)

    # 캔들별 시뮬레이션
    for i in range(50, len(df)):
        row = df.iloc[i]
        hist_df = df.iloc[:i+1].copy()

        # 포지션 있으면 청산 체크
        if position:
            high = float(row['high'])
            low = float(row['low'])

            # 최고가 갱신
            if high > position['highest']:
                position['highest'] = high

                # 트레일링 시작
                gain = (high - position['entry_price']) / position['entry_price'] * 100
                if gain >= trail_start:
                    position['trailing'] = True
                    new_trail = high * (1 - trail_pct / 100)
                    position['trail_stop'] = max(position.get('trail_stop', 0), new_trail)

            # 손절
            if low <= position['stop_loss']:
                exit_price = position['stop_loss']
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100

                trades.append({
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'exit_time': row['timestamp'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'highest': position['highest'],
                    'pnl_pct': pnl_pct,
                    'pnl_leveraged': pnl_pct * leverage,
                    'reason': 'Stop',
                })

                if verbose:
                    print(f"  [손절] {str(row['timestamp'])[5:16]} ${exit_price:.4f} ({pnl_pct:+.1f}%)")

                position = None
                continue

            # 트레일링 스탑
            if position.get('trailing') and low <= position.get('trail_stop', 0):
                exit_price = position['trail_stop']
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100

                trades.append({
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'exit_time': row['timestamp'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'highest': position['highest'],
                    'pnl_pct': pnl_pct,
                    'pnl_leveraged': pnl_pct * leverage,
                    'reason': 'Trail',
                })

                if verbose:
                    print(f"  [트레일] {str(row['timestamp'])[5:16]} ${exit_price:.4f} ({pnl_pct:+.1f}%)")

                position = None
                continue

            # 익절
            tp_price = position['entry_price'] * (1 + tp_pct / 100)
            if high >= tp_price:
                exit_price = tp_price
                pnl_pct = tp_pct

                trades.append({
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'exit_time': row['timestamp'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'highest': position['highest'],
                    'pnl_pct': pnl_pct,
                    'pnl_leveraged': pnl_pct * leverage,
                    'reason': 'TP',
                })

                if verbose:
                    print(f"  [익절] {str(row['timestamp'])[5:16]} ${exit_price:.4f} (+{pnl_pct:.1f}%)")

                position = None
                continue

        # 포지션 없으면 진입 체크
        if not position:
            # 지표 계산
            analyzed = detector.calculate_indicators(hist_df)
            surge = detector.detect_surge_start(hist_df)

            if surge:
                entry_price = surge['current_price']
                surge_low = surge['surge_low']
                stop_loss = surge_low * 0.98

                # 손절이 너무 멀면 기본값
                sl_distance = (entry_price - stop_loss) / entry_price * 100
                if sl_distance > sl_pct * 1.5:
                    stop_loss = entry_price * (1 - sl_pct / 100)

                position = {
                    'entry_price': entry_price,
                    'entry_time': row['timestamp'],
                    'stop_loss': stop_loss,
                    'highest': entry_price,
                    'trailing': False,
                    'trail_stop': 0,
                }

                if verbose:
                    print(f"  [진입] {str(row['timestamp'])[5:16]} ${entry_price:.4f} "
                          f"Vol={surge['volume_ratio']:.1f}x Chg={surge['price_change']:.1f}%")

    # 미청산 포지션
    if position:
        exit_price = float(df.iloc[-1]['close'])
        pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100

        trades.append({
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['timestamp'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'highest': position['highest'],
            'pnl_pct': pnl_pct,
            'pnl_leveraged': pnl_pct * leverage,
            'reason': 'Open',
        })

        if verbose:
            print(f"  [미청산] ${exit_price:.4f} ({pnl_pct:+.1f}%)")

    return {'trades': trades, 'symbol': symbol}


def print_summary(all_trades: List[Dict], initial_balance: float = 1000):
    """결과 요약 출력"""
    print(f"\n{'='*70}")
    print(f"  백테스트 결과 요약")
    print(f"{'='*70}")

    if not all_trades:
        print("거래 없음")
        return

    wins = [t for t in all_trades if t['pnl_pct'] > 0]
    losses = [t for t in all_trades if t['pnl_pct'] <= 0]

    print(f"총 거래: {len(all_trades)}건")
    print(f"승률: {len(wins)}/{len(all_trades)} ({len(wins)/len(all_trades)*100:.1f}%)")

    total_pnl = sum(t['pnl_leveraged'] for t in all_trades)
    print(f"총 수익률: {total_pnl:+.1f}% (레버리지 적용)")

    if wins:
        avg_win = sum(t['pnl_leveraged'] for t in wins) / len(wins)
        max_win = max(t['pnl_leveraged'] for t in wins)
        print(f"평균 이익: +{avg_win:.1f}%, 최대: +{max_win:.1f}%")

    if losses:
        avg_loss = sum(t['pnl_leveraged'] for t in losses) / len(losses)
        max_loss = min(t['pnl_leveraged'] for t in losses)
        print(f"평균 손실: {avg_loss:.1f}%, 최대: {max_loss:.1f}%")

    # 청산 사유별 통계
    print(f"\n청산 사유:")
    reasons = {}
    for t in all_trades:
        r = t['reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl': 0}
        reasons[r]['count'] += 1
        reasons[r]['pnl'] += t['pnl_leveraged']

    for r, data in sorted(reasons.items(), key=lambda x: -x[1]['pnl']):
        print(f"  {r}: {data['count']}건, {data['pnl']:+.1f}%")

    # 잔고 시뮬레이션
    balance = initial_balance
    for t in all_trades:
        profit = balance * (t['pnl_leveraged'] / 100)
        balance += profit

    print(f"\n초기 잔고: ${initial_balance:.2f}")
    print(f"최종 잔고: ${balance:.2f} ({(balance/initial_balance-1)*100:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='초기 급등 전략 백테스트')
    parser.add_argument('--symbol', '-s', default='ZORA/USDT:USDT', help='심볼')
    parser.add_argument('--days', '-d', type=int, default=3, help='백테스트 기간 (일)')
    parser.add_argument('--all', '-a', action='store_true', help='여러 코인 테스트')
    parser.add_argument('--quiet', '-q', action='store_true', help='간략 출력')

    args = parser.parse_args()

    client = BybitClient()
    data_fetcher = DataFetcher(client)

    print(f"\n{'='*70}")
    print(f"  초기 급등 전략 백테스트")
    print(f"{'='*70}")
    print(f"파라미터:")
    print(f"  거래량 스파이크: {EARLY_SURGE_PARAMS['volume_spike_min']}x")
    print(f"  가격 급등: {EARLY_SURGE_PARAMS['price_surge_min']}%")
    print(f"  트레일링: {EARLY_SURGE_PARAMS['trail_start_pct']}% 시작, {EARLY_SURGE_PARAMS['trail_pct']}% 거리")
    print(f"  익절: {EARLY_SURGE_PARAMS['tp_pct']}%")
    print()

    if args.all:
        # 전체 코인 테스트 (병렬)
        from src.surge_strategy import get_all_usdt_perpetuals
        test_symbols = get_all_usdt_perpetuals()

        print(f"전체 {len(test_symbols)}개 코인 테스트")

        all_trades = []
        limit = 288 * args.days

        # 1단계: 데이터 수집 (순차, rate limit 준수)
        print(f"\n[1/2] 데이터 수집 중...")
        symbol_data = {}
        for i, symbol in enumerate(test_symbols):
            try:
                df = data_fetcher.get_ohlcv(symbol, '5m', limit=limit)
                if df is not None and len(df) > 100:
                    symbol_data[symbol] = df

                if (i + 1) % 50 == 0:
                    print(f"  수집: {i+1}/{len(test_symbols)}, 유효: {len(symbol_data)}개")

                time.sleep(0.1)  # Rate limit 방지
            except:
                continue

        print(f"  완료: {len(symbol_data)}개 코인 데이터 수집")

        # 2단계: 백테스트 (병렬)
        print(f"\n[2/2] 백테스트 실행 중 (워커 8개)...")

        def run_backtest(item):
            symbol, df = item
            result = backtest_symbol(symbol, df, EARLY_SURGE_PARAMS, verbose=False)
            return symbol, result['trades']

        completed = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_backtest, item): item[0] for item in symbol_data.items()}

            for future in as_completed(futures):
                symbol = futures[future]
                completed += 1

                try:
                    sym, trades = future.result()
                    if trades:
                        all_trades.extend(trades)
                        print(f"  [{completed}/{len(symbol_data)}] {sym}: {len(trades)}건")
                except:
                    pass

                if completed % 100 == 0:
                    print(f"  진행: {completed}/{len(symbol_data)}, 총 거래: {len(all_trades)}건")

        print_summary(all_trades)

    else:
        # 단일 코인 테스트
        print(f"심볼: {args.symbol}")
        print(f"기간: {args.days}일")

        limit = 288 * args.days
        df = data_fetcher.get_ohlcv(args.symbol, '5m', limit=limit)

        if df is None or len(df) < 100:
            print("데이터 부족")
            return

        df = df.reset_index()
        print(f"데이터: {len(df)}개 캔들")
        print(f"기간: {df.iloc[0]['timestamp']} ~ {df.iloc[-1]['timestamp']}")
        print()

        result = backtest_symbol(args.symbol, df, EARLY_SURGE_PARAMS, verbose=True)
        print_summary(result['trades'])

    print()


if __name__ == '__main__':
    main()
