"""
MA100 DCA 파라미터 그리드 서치

다양한 분할매수 비율 × 간격 조합을 전수 탐색합니다.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_ma100 import (
    MA100_PARAMS, MA100Backtester, load_all_data
)
from src.data_loader import DataLoader

import logging
logging.basicConfig(level=logging.WARNING)


def run_single(all_data, start_dt, end_dt, ratios, interval, params_base):
    params = deepcopy(params_base)
    params["dca_ratios"] = ratios
    params["dca_interval_pct"] = interval

    bt = MA100Backtester(
        initial_balance=1000.0,
        max_positions=params["max_positions"],
        params=params,
    )
    bt.run(all_data, start_dt, end_dt)

    trades = bt.trades
    n = len(trades)
    if n == 0:
        return None

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades)
    win_rate = len(wins) / n * 100
    gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    shorts = [t for t in trades if t["side"] == "short"]
    short_wins = len([t for t in shorts if t["pnl_usd"] > 0])
    short_wr = short_wins / len(shorts) * 100 if shorts else 0
    short_pnl = sum(t["pnl_usd"] for t in shorts)

    sl_count = sum(1 for t in trades if t["reason"] == "SL")
    trail_count = sum(1 for t in trades if t["reason"] == "Trail")

    return {
        "ratios": ":".join(str(r) for r in ratios),
        "interval": interval,
        "trades": n,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "pnl_pct": total_pnl / 1000 * 100,
        "pf": pf,
        "mdd_pct": bt.max_drawdown_pct,
        "short_wr": short_wr,
        "short_pnl": short_pnl,
        "sl": sl_count,
        "trail": trail_count,
        "final_bal": bt.balance,
    }


def main():
    start_dt = datetime(2024, 6, 1)
    end_dt = datetime(2026, 3, 18, 23, 59)

    # 데이터 로드 (한 번만)
    print("데이터 로드 중...")
    loader = DataLoader()
    buffer_days = 150
    data_start = (start_dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    all_data = load_all_data(loader, timeframe="1d", start=data_start, end="2026-03-18")
    print(f"로드 완료: {len(all_data)}개 심볼\n")

    # ── 탐색할 비율들 ──
    ratio_configs = [
        # 단일 진입 (기준선)
        [1],
        # 2분할
        [1, 1],       # 1:1
        [1, 2],       # 1:2
        [2, 1],       # 2:1
        [1, 3],       # 1:3
        [3, 1],       # 3:1
        # 3분할
        [1, 1, 1],    # 1:1:1
        [1, 1, 2],    # 1:1:2
        [1, 2, 1],    # 1:2:1
        [2, 1, 1],    # 2:1:1
        [1, 2, 3],    # 1:2:3
        [1, 1, 3],    # 1:1:3
        [1, 2, 4],    # 1:2:4 (마틴게일풍)
        [1, 3, 5],    # 1:3:5
        # 4분할
        [1, 1, 1, 1], # 1:1:1:1
        [1, 1, 2, 4], # 1:1:2:4
        [1, 2, 3, 4], # 1:2:3:4
    ]

    intervals = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    params_base = MA100_PARAMS.copy()

    results = []
    total = len(ratio_configs) * len(intervals) - (len(intervals) - 1)  # [1]은 interval 무관
    done = 0

    print(f"그리드 서치: {len(ratio_configs)}개 비율 × {len(intervals)}개 간격 = {total}개 조합\n")
    print(f"{'비율':<12} {'간격':>5} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'숏승률':>6} {'SL':>4} {'Trail':>5} {'최종잔고':>10}")
    print("-" * 95)

    t0 = time.time()

    for ratios in ratio_configs:
        if ratios == [1]:
            # 단일 진입은 interval 무관
            r = run_single(all_data, start_dt, end_dt, ratios, 0, params_base)
            done += 1
            if r:
                results.append(r)
                print(f"{r['ratios']:<12} {'N/A':>5} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_wr']:>5.1f}% {r['sl']:>4} {r['trail']:>5} ${r['final_bal']:>9,.2f}")
            continue

        for interval in intervals:
            r = run_single(all_data, start_dt, end_dt, ratios, interval, params_base)
            done += 1
            if r:
                results.append(r)
                print(f"{r['ratios']:<12} {r['interval']:>4.1f}% {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_wr']:>5.1f}% {r['sl']:>4} {r['trail']:>5} ${r['final_bal']:>9,.2f}")

            if done % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  ... {done}/{total} 완료 ({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n총 {elapsed:.1f}s 소요\n")

    # ── TOP 20 정렬 (수익률 기준) ──
    results.sort(key=lambda x: x["pnl_pct"], reverse=True)

    print("=" * 95)
    print("  TOP 20 (수익률 기준)")
    print("=" * 95)
    print(f"{'#':>3} {'비율':<12} {'간격':>5} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'숏승률':>6} {'SL':>4} {'Trail':>5} {'최종잔고':>10}")
    print("-" * 95)
    for i, r in enumerate(results[:20]):
        intv = f"{r['interval']:.1f}%" if r['interval'] > 0 else "N/A"
        print(f"{i+1:>3} {r['ratios']:<12} {intv:>5} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_wr']:>5.1f}% {r['sl']:>4} {r['trail']:>5} ${r['final_bal']:>9,.2f}")

    # ── TOP 10 PF 기준 ──
    results_pf = sorted(results, key=lambda x: x["pf"], reverse=True)
    print(f"\n{'=' * 95}")
    print("  TOP 10 (Profit Factor 기준)")
    print("=" * 95)
    print(f"{'#':>3} {'비율':<12} {'간격':>5} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'숏승률':>6} {'SL':>4} {'Trail':>5} {'최종잔고':>10}")
    print("-" * 95)
    for i, r in enumerate(results_pf[:10]):
        intv = f"{r['interval']:.1f}%" if r['interval'] > 0 else "N/A"
        print(f"{i+1:>3} {r['ratios']:<12} {intv:>5} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_wr']:>5.1f}% {r['sl']:>4} {r['trail']:>5} ${r['final_bal']:>9,.2f}")

    # ── TOP 10 MDD 낮은 순 (수익률 1000% 이상만) ──
    results_mdd = [r for r in results if r["pnl_pct"] >= 1000]
    results_mdd.sort(key=lambda x: x["mdd_pct"])
    print(f"\n{'=' * 95}")
    print("  TOP 10 (MDD 낮은 순, 수익률 1000%+ 필터)")
    print("=" * 95)
    print(f"{'#':>3} {'비율':<12} {'간격':>5} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'숏승률':>6} {'SL':>4} {'Trail':>5} {'최종잔고':>10}")
    print("-" * 95)
    for i, r in enumerate(results_mdd[:10]):
        intv = f"{r['interval']:.1f}%" if r['interval'] > 0 else "N/A"
        print(f"{i+1:>3} {r['ratios']:<12} {intv:>5} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['short_wr']:>5.1f}% {r['sl']:>4} {r['trail']:>5} ${r['final_bal']:>9,.2f}")


if __name__ == "__main__":
    main()
