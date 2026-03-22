#!/usr/bin/env python3
"""
MA100 DCA 전략 비교 분석

비교 대상:
  A) DCA 없음 - 풀사이즈 1회 진입 (SL 5%)
  B) 현재 1:1:2 DCA (4% 간격, SL 5% from full avg)
  C) 1:1 DCA (2단계, 4% 간격)
  D) DCA 없음 - 풀사이즈 + 타이트 SL 3%
  E) 1:1:2 DCA (3% 간격)

추가 분석:
  - DCA가 다 채워진 케이스 vs 1단만 채워진 케이스 승패율
"""

import sys
import time
import logging
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from scripts.backtest_combined import (
    CombinedBacktester,
    MA100_CONFIG,
    load_1d_data,
    load_5m_data,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def run_scenario(label, ma100_override, data_5m, data_1d, start_dt, end_dt, balance=6500):
    """하나의 시나리오 실행."""
    config = deepcopy(MA100_CONFIG)
    config.update(ma100_override)

    bt = CombinedBacktester(
        initial_balance=balance,
        enable_ichimoku=False,
        enable_mirror=False,
        enable_ma100=True,
        enable_dca=False,
    )
    bt.ma100_config = config
    bt.run(data_5m, {}, data_1d, start_dt, end_dt)

    trades = [t for t in bt.trades if t["strategy"] == "ma100"]
    if not trades:
        return None

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades)
    gross_profit = sum(t["pnl_usd"] for t in wins)
    gross_loss = abs(sum(t["pnl_usd"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    wr = len(wins) / len(trades) * 100 if trades else 0
    avg_win = gross_profit / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0

    # Max drawdown
    bal = balance
    peak = bal
    max_dd = 0
    for t in trades:
        bal += t["pnl_usd"]
        peak = max(peak, bal)
        dd = (peak - bal) / peak * 100
        max_dd = max(max_dd, dd)

    # DCA 채움 분석
    dca_analysis = {"1_fill": {"wins": 0, "losses": 0, "pnl": 0},
                    "2_fill": {"wins": 0, "losses": 0, "pnl": 0},
                    "3_fill": {"wins": 0, "losses": 0, "pnl": 0}}
    for t in trades:
        fills = t.get("dca_fills", 1)
        key = f"{min(fills, 3)}_fill"
        if t["pnl_usd"] > 0:
            dca_analysis[key]["wins"] += 1
        else:
            dca_analysis[key]["losses"] += 1
        dca_analysis[key]["pnl"] += t["pnl_usd"]

    return {
        "label": label,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "total_pnl": total_pnl,
        "pf": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_dd": max_dd,
        "final_balance": bt.balance,
        "dca_analysis": dca_analysis,
    }


def main():
    start_dt = datetime(2025, 1, 2)
    end_dt = datetime(2026, 3, 18, 23, 59)

    print("=" * 90)
    print("MA100 DCA 전략 비교 분석")
    print(f"Period: {start_dt.date()} ~ {end_dt.date()}")
    print("=" * 90)

    # 데이터 로드 (1회만)
    print("\n데이터 로드 중...")
    loader = DataLoader()
    t0 = time.time()
    data_1d = load_1d_data(loader, start_dt, end_dt)
    data_5m = load_5m_data(loader, start_dt, end_dt)
    print(f"로드 완료: 1d {len(data_1d)}심볼, 5m {len(data_5m)}심볼 ({time.time()-t0:.1f}s)")

    scenarios = [
        ("A: No DCA (풀사이즈, SL 5%)", {
            "dca_ratios": [1],
            "dca_interval_pct": 0,
        }),
        ("B: 1:1:2 DCA (현재, 4% 간격)", {
            "dca_ratios": [1, 1, 2],
            "dca_interval_pct": 4.0,
        }),
        ("C: 1:1 DCA (2단계, 4% 간격)", {
            "dca_ratios": [1, 1],
            "dca_interval_pct": 4.0,
        }),
        ("D: No DCA (풀사이즈, SL 3%)", {
            "dca_ratios": [1],
            "dca_interval_pct": 0,
            "sl_pct": 3.0,
        }),
        ("E: 1:1:2 DCA (3% 간격)", {
            "dca_ratios": [1, 1, 2],
            "dca_interval_pct": 3.0,
        }),
    ]

    results = []
    for label, override in scenarios:
        print(f"\n  Running: {label} ...", end=" ", flush=True)
        t0 = time.time()
        r = run_scenario(label, override, data_5m, data_1d, start_dt, end_dt)
        elapsed = time.time() - t0
        if r:
            results.append(r)
            print(f"({elapsed:.1f}s) -> {r['trades']} trades, PnL ${r['total_pnl']:.0f}")
        else:
            print(f"({elapsed:.1f}s) -> No trades")

    # 결과 비교표
    print("\n" + "=" * 90)
    print(f"{'Scenario':<35} {'Trades':>6} {'WR':>7} {'PnL':>10} {'PF':>6} {'AvgW':>8} {'AvgL':>8} {'MaxDD':>7} {'Final':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['label']:<35} {r['trades']:>6} {r['win_rate']:>6.1f}% ${r['total_pnl']:>8.0f} {r['pf']:>6.2f} ${r['avg_win']:>6.1f} ${r['avg_loss']:>6.1f} {r['max_dd']:>6.1f}% ${r['final_balance']:>8.0f}")

    # DCA 채움 단계 분석
    print("\n" + "=" * 90)
    print("DCA 채움 단계별 분석")
    print("-" * 90)
    for r in results:
        if len(r["dca_analysis"]["2_fill"]["wins"]) == 0 and r["dca_analysis"]["2_fill"]["losses"] == 0 \
                if isinstance(r["dca_analysis"]["2_fill"]["wins"], list) else False:
            continue
        has_multi = any(
            r["dca_analysis"][k]["wins"] + r["dca_analysis"][k]["losses"] > 0
            for k in ["2_fill", "3_fill"]
        )
        if not has_multi:
            continue

        print(f"\n  [{r['label']}]")
        for key, label in [("1_fill", "1단 (즉시 수익/손절)"),
                           ("2_fill", "2단 DCA 채움"),
                           ("3_fill", "3단 (전체 DCA)")]:
            d = r["dca_analysis"][key]
            total = d["wins"] + d["losses"]
            if total == 0:
                continue
            wr = d["wins"] / total * 100
            avg = d["pnl"] / total
            print(f"    {label:<25} {total:>4}건  WR {wr:>5.1f}%  PnL ${d['pnl']:>8.1f}  Avg ${avg:>6.1f}  (W:{d['wins']} L:{d['losses']})")

    # 핵심 인사이트
    print("\n" + "=" * 90)
    print("핵심 비교: DCA 있음 vs 없음")
    print("-" * 90)
    if len(results) >= 2:
        no_dca = results[0]  # A
        with_dca = results[1]  # B
        print(f"  No DCA:  PnL ${no_dca['total_pnl']:>8.0f}  WR {no_dca['win_rate']:.1f}%  PF {no_dca['pf']:.2f}  MaxDD {no_dca['max_dd']:.1f}%")
        print(f"  1:1:2:   PnL ${with_dca['total_pnl']:>8.0f}  WR {with_dca['win_rate']:.1f}%  PF {with_dca['pf']:.2f}  MaxDD {with_dca['max_dd']:.1f}%")
        diff = with_dca['total_pnl'] - no_dca['total_pnl']
        print(f"  차이:    ${diff:>+8.0f}")


if __name__ == "__main__":
    main()
