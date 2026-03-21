"""
미러숏 DCA 파라미터 그리드 서치

다양한 분할매수 비율 × 간격 조합을 전수 탐색합니다.
SL이 1%로 타이트하므로 DCA 간격은 0.1~0.5% 범위.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.live_surge_mirror_short import (
    MirrorShortParams,
    schedule_next_candle_entries,
)
from scripts.backtest_live_surge_mirror_short import (
    BacktestConfig,
    build_base_surge_signal,
    build_overheat_mask,
    load_symbol_data,
    get_symbols,
)

import logging
logging.basicConfig(level=logging.WARNING)


def simulate_short_exit_with_dca(position, candle, params, dca_fill_counter=None):
    """DCA 체결 + 숏 청산 시뮬레이션 (O→H→L→C 가격경로)"""
    updated = dict(position)
    entry = float(updated["entry_price"])
    stop_loss = float(updated["stop_loss"])
    trailing_active = bool(updated.get("trailing_active", False))
    lowest = float(updated.get("lowest_since_entry", entry))
    trail_stop = updated.get("trail_stop")
    trail_stop = float(trail_stop) if trail_stop is not None else None

    trail_activate_price = entry * (1.0 - params.trail_start_pct / 100.0)

    prices = [
        float(candle["open"]),
        float(candle["high"]),
        float(candle["low"]),
        float(candle["close"]),
    ]

    # DCA 체결 처리 (high 기준으로 한번에)
    pending_dca = updated.get("pending_dca", [])
    if pending_dca:
        high = float(candle["high"])
        filled_new = []
        remaining = []
        for dca in pending_dca:
            if high >= dca["price"]:
                filled_new.append(dca)
            else:
                remaining.append(dca)

        if filled_new:
            entries = updated.get("filled_entries", [])
            for dca in filled_new:
                entries.append({"price": dca["price"], "size": dca["size"]})
            total_size = sum(e["size"] for e in entries)
            avg_price = sum(e["price"] * e["size"] for e in entries) / total_size

            updated["entry_price"] = avg_price
            updated["qty"] = total_size
            updated["filled_entries"] = entries
            updated["pending_dca"] = remaining
            updated["stop_loss"] = avg_price * (1.0 + params.stop_loss_pct / 100.0)

            # margin 재계산 (notional 기준)
            total_margin = sum(e["price"] * e["size"] for e in entries) / 5.0  # leverage=5
            updated["margin"] = total_margin

            # 업데이트된 값 사용
            entry = avg_price
            stop_loss = updated["stop_loss"]
            trail_activate_price = entry * (1.0 - params.trail_start_pct / 100.0)

            if dca_fill_counter is not None:
                dca_fill_counter[0] += len(filled_new)

    for price in prices:
        if price >= stop_loss:
            updated["trailing_active"] = trailing_active
            updated["lowest_since_entry"] = lowest
            updated["trail_stop"] = trail_stop
            return {"reason": "SL", "price": stop_loss}, updated

        if price <= trail_activate_price:
            trailing_active = True

        if trailing_active:
            if price < lowest:
                lowest = price
                trail_stop = lowest * (1.0 + params.trail_rebound_pct / 100.0)
            elif trail_stop is not None and price >= trail_stop:
                updated["trailing_active"] = trailing_active
                updated["lowest_since_entry"] = lowest
                updated["trail_stop"] = trail_stop
                return {"reason": "TRAIL", "price": round(trail_stop, 10)}, updated

    updated["trailing_active"] = trailing_active
    updated["lowest_since_entry"] = lowest
    updated["trail_stop"] = trail_stop
    return None, updated


def run_backtest_dca(all_data, params, cfg, dca_ratios, dca_interval_pct, dca_fill_counter=None):
    """DCA 지원 백테스트"""
    prepared = {}
    ts_to_entries = {}
    ts_to_rows = {}

    for symbol, df in all_data.items():
        base_signal = build_base_surge_signal(df)
        overheat = build_overheat_mask(df, params)
        entry_signal = schedule_next_candle_entries(base_signal, overheat, delay_candles=1)
        local = df.copy()
        local["entry_signal"] = entry_signal
        prepared[symbol] = local

        for idx, row in local.iterrows():
            ts = row["timestamp"]
            ts_to_rows.setdefault(ts, {})[symbol] = idx
            if bool(row["entry_signal"]):
                ts_to_entries.setdefault(ts, []).append(symbol)

    all_timestamps = sorted(ts_to_rows.keys())
    if not all_timestamps:
        return {"trades": [], "equity_curve": [], "balance": 0.0}

    initial_seed = 1000.0
    balance = initial_seed
    peak = balance
    max_dd = 0.0
    positions = {}
    last_exit_idx = {}
    trades = []
    equity_curve = []

    total_ratio = sum(dca_ratios)
    is_dca = len(dca_ratios) > 1

    for t_idx, ts in enumerate(all_timestamps):
        # exits
        for symbol in list(positions.keys()):
            row_idx = ts_to_rows.get(ts, {}).get(symbol)
            if row_idx is None:
                continue
            row = prepared[symbol].iloc[row_idx]
            candle = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            exit_info, updated = simulate_short_exit_with_dca(
                positions[symbol], candle, params, dca_fill_counter
            )
            positions[symbol] = updated
            if exit_info is None:
                continue

            entry = float(updated["entry_price"])
            qty = float(updated["qty"])
            notional = entry * qty
            gross_pnl = (entry - float(exit_info["price"])) / entry * notional
            net_pnl = gross_pnl - notional * cfg.roundtrip_cost_rate
            balance += net_pnl
            last_exit_idx[symbol] = t_idx
            positions.pop(symbol, None)
            trades.append({
                "symbol": symbol,
                "entry_time": updated["entry_time"],
                "exit_time": ts,
                "entry_price": entry,
                "exit_price": float(exit_info["price"]),
                "reason": exit_info["reason"],
                "pnl_usd": net_pnl,
                "pnl_pct_notional": (entry - float(exit_info["price"])) / entry * 100.0,
            })

        # entries
        candidates = ts_to_entries.get(ts, [])
        if candidates:
            locked_margin = sum(p["margin"] for p in positions.values())
            free_balance = max(0.0, balance - locked_margin)
            for symbol in candidates:
                if symbol in positions:
                    continue
                if len(positions) >= cfg.max_positions:
                    break
                if symbol in last_exit_idx and t_idx - last_exit_idx[symbol] < cfg.cooldown_candles:
                    continue

                row_idx = ts_to_rows.get(ts, {}).get(symbol)
                if row_idx is None:
                    continue
                row = prepared[symbol].iloc[row_idx]
                entry_price = float(row["open"])
                if entry_price <= 0:
                    continue

                margin = free_balance * cfg.position_pct
                if margin <= 0:
                    break
                total_notional = margin * cfg.leverage
                total_qty = total_notional / entry_price

                # DCA 분할
                tranche_sizes = [total_qty * r / total_ratio for r in dca_ratios]
                first_size = tranche_sizes[0]
                first_margin = margin * dca_ratios[0] / total_ratio

                pending_dca = []
                if is_dca:
                    for i in range(1, len(dca_ratios)):
                        dca_price = entry_price * (1 + i * dca_interval_pct / 100)
                        pending_dca.append({"price": dca_price, "size": tranche_sizes[i]})

                positions[symbol] = {
                    "entry_time": ts,
                    "entry_price": entry_price,
                    "stop_loss": entry_price * (1.0 + params.stop_loss_pct / 100.0),
                    "trailing_active": False,
                    "lowest_since_entry": entry_price,
                    "trail_stop": None,
                    "qty": first_size,
                    "margin": first_margin if is_dca else margin,
                    "pending_dca": pending_dca,
                    "filled_entries": [{"price": entry_price, "size": first_size}],
                }
                free_balance -= margin  # 전체 마진 예약

        # equity
        unrealized = 0.0
        for symbol, pos in positions.items():
            row_idx = ts_to_rows.get(ts, {}).get(symbol)
            if row_idx is None:
                continue
            cp = float(prepared[symbol].iloc[row_idx]["close"])
            entry = float(pos["entry_price"])
            notional = entry * float(pos["qty"])
            unrealized += (entry - cp) / entry * notional

        equity = balance + unrealized
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)
        equity_curve.append({"timestamp": ts, "equity": equity})

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    gross_profit = sum(t["pnl_usd"] for t in wins)
    gross_loss = abs(sum(t["pnl_usd"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_pnl = sum(t["pnl_usd"] for t in trades)

    return {
        "initial_seed": initial_seed,
        "trades": trades,
        "equity_curve": equity_curve,
        "balance": balance,
        "total_pnl": total_pnl,
        "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "pf": pf,
        "max_dd_usd": max_dd,
        "max_dd_pct": (max_dd / peak * 100.0) if peak > 0 else 0.0,
    }


def main():
    print("데이터 로드 중...")
    loader = DataLoader()
    symbols = get_symbols(loader, None)
    all_data = {}
    for symbol in symbols:
        df = load_symbol_data(loader, symbol, None, None)
        if df is not None:
            all_data[symbol] = df
    print(f"로드 완료: {len(all_data)}개 심볼\n")

    params = MirrorShortParams(
        overheat_cum_rise_pct=8.0,
        overheat_upper_wick_pct=40.0,
        overheat_volume_ratio=5.0,
        volume_lookback=20,
        stop_loss_pct=1.0,
        trail_start_pct=3.0,
        trail_rebound_pct=1.2,
    )
    cfg = BacktestConfig(
        leverage=5.0,
        position_pct=0.05,
        max_positions=3,
        cooldown_candles=3,
        roundtrip_cost_rate=0.0009,
    )

    # ── 탐색할 비율들 ──
    ratio_configs = [
        [1],
        [1, 1],
        [1, 2],
        [2, 1],
        [1, 3],
        [3, 1],
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 1],
        [2, 1, 1],
        [1, 2, 3],
        [1, 1, 3],
        [1, 2, 4],
        [1, 1, 1, 1],
        [1, 1, 2, 4],
        [1, 2, 3, 4],
    ]

    # SL=1%이므로 간격은 0.1~0.6% 범위
    intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    results = []
    total = len(ratio_configs) * len(intervals) - (len(intervals) - 1)
    done = 0

    print(f"그리드 서치: {len(ratio_configs)}개 비율 × {len(intervals)}개 간격 = {total}개 조합\n")
    print(f"{'비율':<12} {'간격':>6} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'SL':>4} {'Trail':>5} {'DCA':>5} {'최종잔고':>10}")
    print("-" * 90)

    t0 = time.time()

    for ratios in ratio_configs:
        if ratios == [1]:
            dca_counter = [0]
            r = run_backtest_dca(all_data, params, cfg, ratios, 0, dca_counter)
            done += 1
            trades = r["trades"]
            n = len(trades)
            if n > 0:
                sl = sum(1 for t in trades if t["reason"] == "SL")
                trail = sum(1 for t in trades if t["reason"] == "TRAIL")
                result = {
                    "ratios": ":".join(str(x) for x in ratios),
                    "interval": 0,
                    "trades": n,
                    "win_rate": r["win_rate"],
                    "pnl_pct": r["total_pnl"] / r["initial_seed"] * 100,
                    "pf": r["pf"],
                    "mdd_pct": r["max_dd_pct"],
                    "sl": sl,
                    "trail": trail,
                    "dca_fills": dca_counter[0],
                    "final_bal": r["balance"],
                }
                results.append(result)
                print(f"{result['ratios']:<12} {'N/A':>6} {n:>5} {result['win_rate']:>5.1f}% {result['pnl_pct']:>+8.1f}% {result['pf']:>5.2f} {result['mdd_pct']:>5.1f}% {sl:>4} {trail:>5} {0:>5} ${result['final_bal']:>9,.2f}")
            continue

        for interval in intervals:
            dca_counter = [0]
            r = run_backtest_dca(all_data, params, cfg, ratios, interval, dca_counter)
            done += 1
            trades = r["trades"]
            n = len(trades)
            if n > 0:
                sl = sum(1 for t in trades if t["reason"] == "SL")
                trail = sum(1 for t in trades if t["reason"] == "TRAIL")
                result = {
                    "ratios": ":".join(str(x) for x in ratios),
                    "interval": interval,
                    "trades": n,
                    "win_rate": r["win_rate"],
                    "pnl_pct": r["total_pnl"] / r["initial_seed"] * 100,
                    "pf": r["pf"],
                    "mdd_pct": r["max_dd_pct"],
                    "sl": sl,
                    "trail": trail,
                    "dca_fills": dca_counter[0],
                    "final_bal": r["balance"],
                }
                results.append(result)
                intv = f"{interval:.2f}%"
                print(f"{result['ratios']:<12} {intv:>6} {n:>5} {result['win_rate']:>5.1f}% {result['pnl_pct']:>+8.1f}% {result['pf']:>5.2f} {result['mdd_pct']:>5.1f}% {sl:>4} {trail:>5} {dca_counter[0]:>5} ${result['final_bal']:>9,.2f}")

            if done % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                print(f"  ... {done}/{total} 완료 ({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n총 {elapsed:.1f}s 소요\n")

    # TOP 20 수익률
    results.sort(key=lambda x: x["pnl_pct"], reverse=True)
    print("=" * 90)
    print("  TOP 20 (수익률 기준)")
    print("=" * 90)
    print(f"{'#':>3} {'비율':<12} {'간격':>6} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'SL':>4} {'Trail':>5} {'DCA':>5} {'최종잔고':>10}")
    print("-" * 90)
    for i, r in enumerate(results[:20]):
        intv = f"{r['interval']:.2f}%" if r['interval'] > 0 else "N/A"
        print(f"{i+1:>3} {r['ratios']:<12} {intv:>6} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['sl']:>4} {r['trail']:>5} {r['dca_fills']:>5} ${r['final_bal']:>9,.2f}")

    # TOP 10 PF
    results_pf = sorted(results, key=lambda x: x["pf"], reverse=True)
    print(f"\n{'=' * 90}")
    print("  TOP 10 (Profit Factor 기준)")
    print("=" * 90)
    print(f"{'#':>3} {'비율':<12} {'간격':>6} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'SL':>4} {'Trail':>5} {'DCA':>5} {'최종잔고':>10}")
    print("-" * 90)
    for i, r in enumerate(results_pf[:10]):
        intv = f"{r['interval']:.2f}%" if r['interval'] > 0 else "N/A"
        print(f"{i+1:>3} {r['ratios']:<12} {intv:>6} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['sl']:>4} {r['trail']:>5} {r['dca_fills']:>5} ${r['final_bal']:>9,.2f}")

    # TOP 10 MDD 낮은 순 (수익률 양수)
    results_mdd = [r for r in results if r["pnl_pct"] > 0]
    results_mdd.sort(key=lambda x: x["mdd_pct"])
    print(f"\n{'=' * 90}")
    print("  TOP 10 (MDD 낮은 순, 수익 양수)")
    print("=" * 90)
    print(f"{'#':>3} {'비율':<12} {'간격':>6} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'SL':>4} {'Trail':>5} {'DCA':>5} {'최종잔고':>10}")
    print("-" * 90)
    for i, r in enumerate(results_mdd[:10]):
        intv = f"{r['interval']:.2f}%" if r['interval'] > 0 else "N/A"
        print(f"{i+1:>3} {r['ratios']:<12} {intv:>6} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['sl']:>4} {r['trail']:>5} {r['dca_fills']:>5} ${r['final_bal']:>9,.2f}")


if __name__ == "__main__":
    main()
