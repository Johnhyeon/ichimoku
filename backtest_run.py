#!/usr/bin/env python3
"""
백테스트 실행 스크립트

실행 예시:
    python backtest_run.py              # 기본 (4시간봉, 4000개 캔들)
    python backtest_run.py --limit 2000 # 캔들 개수 지정
"""

import argparse
import logging
import time

from src.backtest import (
    fetch_klines, run_backtest, calculate_stats, print_report,
    MAJOR_COINS, INITIAL_CAPITAL
)
from src.strategy import STRATEGY_PARAMS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ichimoku 백테스트")
    parser.add_argument("--limit", type=int, default=4000, help="캔들 개수 (기본: 4000)")
    parser.add_argument("--interval", type=int, default=240, help="타임프레임 분 (기본: 240=4h)")
    args = parser.parse_args()

    print("=" * 60)
    print("일목균형표 (Ichimoku Cloud) 백테스트")
    print("=" * 60)
    print(f"초기 자본: ${INITIAL_CAPITAL:,}")
    print(f"레버리지: 20x")
    print(f"포지션 크기: 5%")
    print(f"코인: {len(MAJOR_COINS)}개")
    print(f"타임프레임: {args.interval}분")
    print(f"캔들 개수: {args.limit}")
    print("=" * 60)

    # 데이터 수집
    print("\n데이터 수집 중...")
    all_data = {}
    for i, symbol in enumerate(MAJOR_COINS):
        print(f"\r  {i+1}/{len(MAJOR_COINS)} {symbol}...", end='', flush=True)
        df = fetch_klines(symbol, args.interval, args.limit)
        if df is not None and len(df) > 100:
            all_data[symbol] = df
        time.sleep(0.05)

    print(f"\n  {len(all_data)}개 코인 로드 완료")

    if not all_data:
        print("데이터가 없습니다.")
        return

    # 데이터 기간 출력
    first_time = None
    last_time = None
    for df in all_data.values():
        if first_time is None or df['timestamp'].min() < first_time:
            first_time = df['timestamp'].min()
        if last_time is None or df['timestamp'].max() > last_time:
            last_time = df['timestamp'].max()

    print(f"\n데이터 기간: {first_time} ~ {last_time}")

    # 백테스트 실행
    print("\n백테스트 실행 중...")
    trades, equity_curve = run_backtest(
        all_data,
        params=STRATEGY_PARAMS,
        initial_capital=INITIAL_CAPITAL,
        use_btc_filter=True,
        use_volume_filter=True
    )

    # 결과 출력
    stats = calculate_stats(trades, equity_curve, INITIAL_CAPITAL)
    print_report(stats, trades)

    # 최근 10개 거래 출력
    if trades:
        print("\n최근 10개 거래:")
        print("-" * 100)
        print(f"{'코인':<12} {'방향':<6} {'진입가':>12} {'청산가':>12} {'수익률':>10} {'수익($)':>10} {'사유':<8}")
        print("-" * 100)
        for t in trades[-10:]:
            print(f"{t['symbol']:<12} {t['side'].upper():<6} {t['entry_price']:>12.2f} {t['exit_price']:>12.2f} {t['pnl_pct']:>9.1f}% {t['pnl_usd']:>10.2f} {t['reason']:<8}")


if __name__ == "__main__":
    main()
