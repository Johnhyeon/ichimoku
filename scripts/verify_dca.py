"""DCA 체결 검증 스크립트"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.WARNING)

from scripts.backtest_ma100 import MA100_PARAMS, MA100Backtester, load_all_data
from src.data_loader import DataLoader


def main():
    start_dt = datetime(2024, 6, 1)
    end_dt = datetime(2026, 3, 18, 23, 59)
    loader = DataLoader()
    data_start = (start_dt - timedelta(days=150)).strftime("%Y-%m-%d")
    all_data = load_all_data(loader, timeframe="1d", start=data_start, end="2026-03-18")

    # 기준선 (DCA 없음)
    params_base = deepcopy(MA100_PARAMS)
    bt_base = MA100Backtester(initial_balance=1000.0, max_positions=5, params=params_base)
    bt_base.run(all_data, start_dt, end_dt)

    # DCA 1:1:2, interval 2.5%
    params_dca = deepcopy(MA100_PARAMS)
    params_dca["dca_ratios"] = [1, 1, 2]
    params_dca["dca_interval_pct"] = 2.5
    bt_dca = MA100Backtester(initial_balance=1000.0, max_positions=5, params=params_dca)

    # DCA 체결 카운터 패치
    original_process = bt_dca._process_dca_fills
    dca_fill_count = [0]
    dca_fill_details = []

    def counting_process(symbol, candle):
        pos = bt_dca.positions.get(symbol)
        old_entries = len(pos.get("filled_entries", [])) if pos else 0
        original_process(symbol, candle)
        pos = bt_dca.positions.get(symbol)
        new_entries = len(pos.get("filled_entries", [])) if pos else 0
        if new_entries > old_entries:
            fills = new_entries - old_entries
            dca_fill_count[0] += fills
            dca_fill_details.append({
                "symbol": symbol,
                "fills": fills,
                "avg_entry": pos["entry_price"],
                "total_size": pos["size"],
                "sl": pos["stop_loss"],
            })

    bt_dca._process_dca_fills = counting_process
    bt_dca.run(all_data, start_dt, end_dt)

    print("=== DCA 체결 검증 ===")
    print(f"DCA 추가 체결 횟수: {dca_fill_count[0]}")
    print(f"DCA 이벤트 수: {len(dca_fill_details)}")
    print()

    # DCA 이벤트 샘플
    print("처음 15개 DCA 체결 이벤트:")
    for i, d in enumerate(dca_fill_details[:15]):
        sym = d["symbol"].split("/")[0]
        print(f"  {i+1:2d}. {sym:12s} +{d['fills']}차 체결 | avg=${d['avg_entry']:.4f} size={d['total_size']:.6f} SL=${d['sl']:.4f}")

    print()
    base_sl = sum(1 for t in bt_base.trades if t["reason"] == "SL")
    base_trail = sum(1 for t in bt_base.trades if t["reason"] == "Trail")
    dca_sl = sum(1 for t in bt_dca.trades if t["reason"] == "SL")
    dca_trail = sum(1 for t in bt_dca.trades if t["reason"] == "Trail")

    print(f"기준선      : {len(bt_base.trades)}건, SL={base_sl}, Trail={base_trail}")
    print(f"DCA 1:1:2 2.5%: {len(bt_dca.trades)}건, SL={dca_sl}, Trail={dca_trail}")

    # 같은 심볼 첫 거래 비교
    print()
    print("=== 같은 심볼 첫 거래 비교 (처음 20개) ===")
    print(f"{'심볼':10s}  {'BASE entry':>10s} {'BASE pnl':>10s} {'BASE reason':10s}  {'DCA entry':>10s} {'DCA pnl':>10s} {'DCA reason':10s} {'비교':4s}")
    print("-" * 100)

    base_first = {}
    for t in bt_base.trades:
        if t["symbol"] not in base_first:
            base_first[t["symbol"]] = t

    dca_first = {}
    for t in bt_dca.trades:
        if t["symbol"] not in dca_first:
            dca_first[t["symbol"]] = t

    common = sorted(set(base_first.keys()) & set(dca_first.keys()))[:20]
    same_count = 0
    diff_count = 0
    for sym in common:
        b = base_first[sym]
        d = dca_first[sym]
        sym_short = sym.split("/")[0][:10]
        is_same = abs(b["pnl_usd"] - d["pnl_usd"]) < 0.01
        tag = "SAME" if is_same else "DIFF"
        if is_same:
            same_count += 1
        else:
            diff_count += 1
        print(f"  {sym_short:10s}  ${b['entry_price']:>9.4f} ${b['pnl_usd']:>+9.2f} {b['reason']:10s}  ${d['entry_price']:>9.4f} ${d['pnl_usd']:>+9.2f} {d['reason']:10s} [{tag}]")

    print(f"\n비교: SAME={same_count}, DIFF={diff_count}")

    # DCA가 SL을 방지한 사례 찾기
    print()
    print("=== DCA 덕에 SL 회피 → Trail 전환된 사례 ===")
    base_by_key = {}
    for t in bt_base.trades:
        key = (t["symbol"], str(t["entry_time"])[:10])
        base_by_key[key] = t

    saved_count = 0
    for t in bt_dca.trades:
        key = (t["symbol"], str(t["entry_time"])[:10])
        bt = base_by_key.get(key)
        if bt and bt["reason"] == "SL" and t["reason"] == "Trail":
            saved_count += 1
            if saved_count <= 10:
                sym = t["symbol"].split("/")[0]
                print(f"  {sym:12s} BASE: SL ${bt['pnl_usd']:+.2f} → DCA: Trail ${t['pnl_usd']:+.2f} (구제 +${t['pnl_usd']-bt['pnl_usd']:.2f})")

    print(f"\nSL→Trail 전환: {saved_count}건")


if __name__ == "__main__":
    main()
