"""추천 DCA 조건별 Best/Worst 사례 상세 분석"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
from scripts.backtest_ma100 import MA100_PARAMS, MA100Backtester, load_all_data
from src.data_loader import DataLoader


def run_with_dca_tracking(all_data, start_dt, end_dt, ratios, interval):
    """DCA 추적하며 백테스트 실행, 트레이드별 DCA 이력 기록"""
    params = deepcopy(MA100_PARAMS)
    params["dca_ratios"] = ratios
    params["dca_interval_pct"] = interval

    bt = MA100Backtester(initial_balance=1000.0, max_positions=5, params=params)

    # DCA 이력 추적용
    dca_history = {}  # symbol -> [{price, size, candle_date}, ...]

    original_process = bt._process_dca_fills
    def tracking_process(symbol, candle):
        pos = bt.positions.get(symbol)
        if not pos:
            return
        old_n = len(pos.get("filled_entries", []))
        original_process(symbol, candle)
        pos = bt.positions.get(symbol)
        if not pos:
            return
        new_n = len(pos.get("filled_entries", []))
        if new_n > old_n:
            key = (symbol, str(pos["entry_time"])[:10])
            if key not in dca_history:
                dca_history[key] = []
            for entry in pos["filled_entries"][old_n:]:
                dca_history[key].append({
                    "price": entry["price"],
                    "size": entry["size"],
                    "candle_date": str(candle.get("timestamp", ""))[:10],
                })

    bt._process_dca_fills = tracking_process
    bt.run(all_data, start_dt, end_dt)

    return bt, dca_history


def describe_trade(t, dca_history, params, all_data):
    """트레이드 하나를 텍스트로 상세 설명"""
    symbol = t["symbol"]
    sym = symbol.split("/")[0]
    side = t["side"]
    entry = t["entry_price"]
    exit_p = t["exit_price"]
    reason = t["reason"]
    pnl_usd = t["pnl_usd"]
    pnl_pct = t["pnl_pct"]
    entry_time = t["entry_time"]
    exit_time = t["exit_time"]

    ratios = params["dca_ratios"]
    interval = params["dca_interval_pct"]
    sl_pct = params["sl_pct"]
    total_ratio = sum(ratios)

    dur = (exit_time - entry_time).days if hasattr(exit_time, "days") else (exit_time - entry_time).days

    key = (symbol, str(entry_time)[:10])
    fills = dca_history.get(key, [])
    n_fills = len(fills) + 1  # +1 for initial entry
    n_total = len(ratios)

    lines = []
    lines.append(f"  [{sym}] {side.upper()} | {str(entry_time)[:10]} → {str(exit_time)[:10]} ({dur}일)")
    lines.append(f"  결과: {reason} | PnL: ${pnl_usd:+.2f} ({pnl_pct:+.1f}%)")
    lines.append("")

    # 1차 진입
    first_ratio = ratios[0]
    first_pct = first_ratio / total_ratio * 100
    lines.append(f"  1차 진입: ${entry:.4f} (비중 {first_pct:.0f}%)")

    # DCA 체결 상황
    if len(ratios) > 1:
        for i in range(1, len(ratios)):
            r = ratios[i]
            pct = r / total_ratio * 100
            if side == "short":
                planned_price = entry * (1 + i * interval / 100)
            else:
                planned_price = entry * (1 - i * interval / 100)

            filled = None
            if i - 1 < len(fills):
                filled = fills[i - 1]

            if filled:
                lines.append(f"  {i+1}차 DCA:  ${filled['price']:.4f} 체결 (계획 ${planned_price:.4f}, 비중 {pct:.0f}%)")
            else:
                lines.append(f"  {i+1}차 DCA:  미체결 (계획 ${planned_price:.4f}, 비중 {pct:.0f}%)")

    # 평균단가 & SL
    if fills:
        all_entries = [{"price": entry, "size": 1.0}]  # 비율 기반
        for i, f in enumerate(fills):
            all_entries.append({"price": f["price"], "size": ratios[i + 1]})
        total_w = sum(e["size"] for e in all_entries)
        avg = sum(e["price"] * e["size"] for e in all_entries) / total_w
        if side == "short":
            sl_price = avg * (1 + sl_pct / 100)
        else:
            sl_price = avg * (1 - sl_pct / 100)
        lines.append(f"  → 평균단가: ${avg:.4f} (1차 대비 {(avg/entry-1)*100:+.2f}%)")
        lines.append(f"  → 손절선:   ${sl_price:.4f} (평균단가 기준 {sl_pct}%)")
    else:
        if side == "short":
            sl_price = entry * (1 + sl_pct / 100)
        else:
            sl_price = entry * (1 - sl_pct / 100)
        lines.append(f"  → 평균단가: ${entry:.4f} (DCA 미체결, 1차만)")
        lines.append(f"  → 손절선:   ${sl_price:.4f}")

    lines.append(f"  → 체결: {n_fills}/{n_total}차")

    # 가격 흐름 묘사
    lines.append("")
    df = all_data.get(symbol)
    if df is not None:
        mask = (df["timestamp"] >= pd.Timestamp(entry_time)) & (df["timestamp"] <= pd.Timestamp(exit_time))
        period = df.loc[mask]
        if len(period) > 0:
            if side == "short":
                max_adverse = float(period["high"].max())
                max_favor = float(period["low"].min())
                adverse_pct = (max_adverse / entry - 1) * 100
                favor_pct = (1 - max_favor / entry) * 100
                lines.append(f"  보유 중 최고가: ${max_adverse:.4f} (진입 대비 +{adverse_pct:.1f}% 역행)")
                lines.append(f"  보유 중 최저가: ${max_favor:.4f} (진입 대비 -{favor_pct:.1f}% 순행)")
            else:
                max_adverse = float(period["low"].min())
                max_favor = float(period["high"].max())
                adverse_pct = (1 - max_adverse / entry) * 100
                favor_pct = (max_favor / entry - 1) * 100
                lines.append(f"  보유 중 최저가: ${max_adverse:.4f} (진입 대비 -{adverse_pct:.1f}% 역행)")
                lines.append(f"  보유 중 최고가: ${max_favor:.4f} (진입 대비 +{favor_pct:.1f}% 순행)")

    return "\n".join(lines)


def analyze_candidate(name, ratios, interval, all_data, start_dt, end_dt):
    """후보 하나 분석"""
    params = deepcopy(MA100_PARAMS)
    params["dca_ratios"] = ratios
    params["dca_interval_pct"] = interval

    bt, dca_history = run_with_dca_tracking(all_data, start_dt, end_dt, ratios, interval)
    trades = bt.trades

    n = len(trades)
    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades)
    win_rate = len(wins) / n * 100 if n > 0 else 0
    gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    sl_count = sum(1 for t in trades if t["reason"] == "SL")
    trail_count = sum(1 for t in trades if t["reason"] == "Trail")

    # DCA 체결 통계
    total_dca_fills = sum(len(v) for v in dca_history.values())
    full_fill_count = sum(1 for v in dca_history.values() if len(v) == len(ratios) - 1)

    ratio_str = ":".join(str(r) for r in ratios)

    print(f"\n{'='*80}")
    print(f"  추천 {name}: DCA {ratio_str}, 간격 {interval}%")
    print(f"{'='*80}")
    print(f"  수익률: +{total_pnl/10:.1f}% | PF: {pf:.2f} | MDD: -{bt.max_drawdown_pct:.1f}%")
    print(f"  승률: {win_rate:.1f}% ({len(wins)}W/{len(losses)}L) | 거래: {n}건")
    print(f"  SL: {sl_count} | Trail: {trail_count}")
    print(f"  DCA 추가체결: {total_dca_fills}회 | 전량체결: {full_fill_count}건")
    print()

    # Best trade
    best = max(trades, key=lambda t: t["pnl_usd"])
    print(f"  --- BEST TRADE ---")
    print(describe_trade(best, dca_history, params, all_data))

    # Worst trade
    worst = min(trades, key=lambda t: t["pnl_usd"])
    print(f"\n  --- WORST TRADE ---")
    print(describe_trade(worst, dca_history, params, all_data))

    return bt


def main():
    start_dt = datetime(2024, 6, 1)
    end_dt = datetime(2026, 3, 18, 23, 59)

    print("데이터 로드 중...")
    loader = DataLoader()
    data_start = (start_dt - timedelta(days=150)).strftime("%Y-%m-%d")
    all_data = load_all_data(loader, timeframe="1d", start=data_start, end="2026-03-18")
    print(f"로드 완료: {len(all_data)}개 심볼\n")

    # 기준선
    print("=" * 80)
    print("  기준선: DCA 없음 (단일 진입)")
    print("=" * 80)
    params_base = deepcopy(MA100_PARAMS)
    bt_base = MA100Backtester(initial_balance=1000.0, max_positions=5, params=params_base)
    bt_base.run(all_data, start_dt, end_dt)
    trades_base = bt_base.trades
    n = len(trades_base)
    wins = [t for t in trades_base if t["pnl_usd"] > 0]
    losses = [t for t in trades_base if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades_base)
    gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    sl_count = sum(1 for t in trades_base if t["reason"] == "SL")
    trail_count = sum(1 for t in trades_base if t["reason"] == "Trail")
    print(f"  수익률: +{total_pnl/10:.1f}% | PF: {pf:.2f} | MDD: -{bt_base.max_drawdown_pct:.1f}%")
    print(f"  승률: {len(wins)/n*100:.1f}% ({len(wins)}W/{len(losses)}L) | 거래: {n}건")
    print(f"  SL: {sl_count} | Trail: {trail_count}")

    # 후보들
    candidates = [
        ("A", [1, 2, 3, 4], 4.0),
        ("B", [1, 1, 2], 4.0),
        ("C", [1, 1, 2, 4], 3.5),
        ("D", [1, 1, 2], 2.5),
    ]

    for name, ratios, interval in candidates:
        analyze_candidate(name, ratios, interval, all_data, start_dt, end_dt)


if __name__ == "__main__":
    main()
