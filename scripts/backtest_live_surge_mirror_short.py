import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.live_surge_mirror_short import (
    MirrorShortParams,
    schedule_next_candle_entries,
    simulate_short_exit_ohlc,
)
from src.mirror_short_report import generate_html_report, save_trades_csv


@dataclass
class BacktestConfig:
    leverage: float = 5.0
    position_pct: float = 0.05
    max_positions: int = 3
    cooldown_candles: int = 3
    roundtrip_cost_rate: float = 0.0009  # 0.09%


def build_base_surge_signal(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    work["volume_sma"] = work["volume"].rolling(20).mean()
    work["volume_ratio"] = work["volume"] / work["volume_sma"]
    work["change_pct"] = work["close"].pct_change() * 100.0
    work["is_green"] = work["close"] > work["open"]

    range_high = work["high"].rolling(12).max().shift(1)
    range_low = work["low"].rolling(12).min().shift(1)
    work["consol_range_pct"] = (range_high - range_low) / range_low * 100.0
    work["price_from_low"] = (work["close"] - work["low"].shift(1)) / work["low"].shift(1) * 100.0

    signal = (
        (work["volume_ratio"] >= 10.0)
        & (work["change_pct"] >= 5.0)
        & work["is_green"]
        & (work["consol_range_pct"] <= 5.0)
        & (work["price_from_low"] <= 15.0)
    )
    return signal.fillna(False)


def build_overheat_mask(df: pd.DataFrame, params: MirrorShortParams) -> pd.Series:
    close = df["close"]
    cum_rise = (close / close.shift(3) - 1.0) * 100.0

    candle_range = (df["high"] - df["low"]).clip(lower=1e-12)
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0.0)
    upper_wick_pct = upper_wick / candle_range * 100.0

    vol_avg = df["volume"].shift(1).rolling(params.volume_lookback).mean()
    vol_ratio = df["volume"] / vol_avg

    return (
        (cum_rise >= params.overheat_cum_rise_pct)
        | (upper_wick_pct >= params.overheat_upper_wick_pct)
        | (vol_ratio >= params.overheat_volume_ratio)
    ).fillna(False)


def load_symbol_data(loader: DataLoader, symbol: str, start: Optional[str], end: Optional[str]) -> Optional[pd.DataFrame]:
    df = loader.load(symbol, "5m", start=start, end=end)
    if df is None or len(df) < 40:
        return None
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def get_symbols(loader: DataLoader, max_symbols: Optional[int]) -> List[str]:
    symbols = loader.get_available_symbols()
    if not symbols:
        root = loader.data_dir
        if root.exists():
            symbols = [p.name for p in root.iterdir() if p.is_dir()]
    symbols = sorted(symbols)
    if max_symbols is not None:
        symbols = symbols[:max_symbols]
    return symbols


def run_backtest(all_data: Dict[str, pd.DataFrame], params: MirrorShortParams, cfg: BacktestConfig) -> dict:
    prepared = {}
    ts_to_entries: Dict[pd.Timestamp, List[str]] = {}
    ts_to_rows: Dict[pd.Timestamp, Dict[str, int]] = {}

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
    positions: Dict[str, dict] = {}
    last_exit_idx: Dict[str, int] = {}
    trades: List[dict] = []
    equity_curve = []

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
            exit_info, updated = simulate_short_exit_ohlc(positions[symbol], candle, params)
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
            trades.append(
                {
                    "symbol": symbol,
                    "entry_time": updated["entry_time"],
                    "exit_time": ts,
                    "entry_price": entry,
                    "exit_price": float(exit_info["price"]),
                    "reason": exit_info["reason"],
                    "pnl_usd": net_pnl,
                    "pnl_pct_notional": (entry - float(exit_info["price"])) / entry * 100.0,
                }
            )

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
                notional = margin * cfg.leverage
                qty = notional / entry_price

                positions[symbol] = {
                    "entry_time": ts,
                    "entry_price": entry_price,
                    "stop_loss": entry_price * (1.0 + params.stop_loss_pct / 100.0),
                    "trailing_active": False,
                    "lowest_since_entry": entry_price,
                    "trail_stop": None,
                    "qty": qty,
                    "margin": margin,
                }
                free_balance -= margin

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
    parser = argparse.ArgumentParser(description="Backtest live-surge mirror short strategy (parquet-only).")
    parser.add_argument("--start", type=str, default=None, help="Start date, e.g. 2025-01-01")
    parser.add_argument("--end", type=str, default=None, help="End date, e.g. 2026-02-22")
    parser.add_argument("--max-symbols", type=int, default=None, help="Limit symbols for quick validation")
    parser.add_argument(
        "--html-out",
        type=str,
        default="data/backtest_live_surge_mirror_short_report.html",
        help="Output HTML report path",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="data/backtest_live_surge_mirror_short_trades.csv",
        help="Output CSV trade log path",
    )
    args = parser.parse_args()

    loader = DataLoader()
    symbols = get_symbols(loader, args.max_symbols)
    if not symbols:
        print("No symbols found in data/historical.")
        return

    all_data = {}
    for symbol in symbols:
        df = load_symbol_data(loader, symbol, args.start, args.end)
        if df is not None:
            all_data[symbol] = df

    if not all_data:
        print("No usable parquet data found.")
        return

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

    result = run_backtest(all_data, params, cfg)
    trades = result["trades"]
    print("=" * 60)
    print("Live Surge Mirror Short Backtest (parquet-only)")
    print("=" * 60)
    print(f"Symbols used      : {len(all_data)}")
    print(f"Trades            : {len(trades)}")
    print(f"Win rate          : {result['win_rate']:.2f}%")
    print(f"Total PnL         : ${result['total_pnl']:+.2f}")
    print(f"Final balance     : ${result['balance']:.2f}")
    print(f"Profit factor     : {result['pf']:.3f}")
    print(f"Max drawdown      : -${result['max_dd_usd']:.2f} (-{result['max_dd_pct']:.2f}%)")
    print("Cost model        : 0.09% round-trip")
    print("Fill model        : O -> H -> L -> C")
    print("=" * 60)
    html_path = generate_html_report(
        result=result,
        all_data=all_data,
        out_path=Path(args.html_out),
        page_size=100,
        chart_window=24,
    )
    csv_path = save_trades_csv(result=result, out_path=Path(args.csv_out))
    print(f"HTML report       : {html_path}")
    print(f"CSV trade log     : {csv_path}")


if __name__ == "__main__":
    main()
