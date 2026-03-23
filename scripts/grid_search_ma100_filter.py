"""
MA100 Filter Grid Search — SL 감소를 위한 필터 최적화

핵심 발견: high가 MA100을 크게 돌파한 캔들에서 숏 진입 → SL
필터 후보:
  1. max_penetration: high가 MA100 위로 뚫은 % 상한
  2. max_vol_ratio: 거래량 비율 상한 (높은 거래량 = 돌파)
  3. min_slope: MA100 기울기 최소값 (더 가파른 하락만)
  4. max_close_dist: close가 MA100에서 너무 멀면 스킵

사용법:
    python scripts/grid_search_ma100_filter.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import itertools
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_loader import DataLoader
from src.strategy import STABLECOINS

# ─── Config ───────────────────────────────────────────────────

BASE = {
    "ma_period": 100,
    "slope_lookback": 3,
    "touch_buffer_pct": 1.0,
    "sl_pct": 5.0,
    "trail_start_pct": 3.0,
    "trail_pct": 2.0,
    "cooldown_days": 3,
    "leverage": 3,
    "position_pct": 0.02,
    "fee_rate": 0.00055,
}


# ─── Data Loading ─────────────────────────────────────────────

def load_1d_data(loader, start_dt, end_dt):
    avail = set(loader.get_available_symbols())
    all_syms = [s for s in avail if s.split("/")[0] not in STABLECOINS]
    data = {}
    w = timedelta(days=150)
    s_str, e_str = (start_dt - w).strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    for sym in all_syms:
        tfs = loader.get_available_timeframes(sym)
        df = None
        if "1d" in tfs:
            df = loader.load(sym, "1d", start=s_str, end=e_str)
            if df is None or len(df) < 120:
                df = None
        if df is None and "4h" in tfs:
            raw = loader.load(sym, "4h", start=s_str, end=e_str)
            if raw is not None and len(raw) >= 600:
                raw["timestamp"] = pd.to_datetime(raw["timestamp"])
                df = raw.set_index("timestamp").resample("1D").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna().reset_index()
                if len(df) < 120:
                    df = None
        if df is not None:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            data[sym] = df
    return data


def precompute(data):
    """MA100, slope, RSI, vol_ratio 사전 계산."""
    result = {}
    for sym, df in data.items():
        df = df.copy()
        df["ma100"] = df["close"].rolling(100).mean()
        df["slope"] = (df["ma100"] - df["ma100"].shift(3)) / df["ma100"].shift(3) * 100
        # touch buffer 1%
        df["touch"] = (
            (df["slope"] < 0)
            & (df["high"] >= df["ma100"] * 0.99)
            & (df["close"] < df["ma100"])
        ).fillna(False)
        # Extra features
        df["high_pen"] = (df["high"] - df["ma100"]) / df["ma100"] * 100  # high penetration %
        df["close_dist"] = (df["close"] - df["ma100"]) / df["ma100"] * 100
        df["vol_sma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma20"]
        result[sym] = df
    return result


# ─── Backtest ─────────────────────────────────────────────────

def run_backtest(precomputed, params, start_dt, end_dt, initial_balance=6500):
    sl_pct = params["sl_pct"]
    trail_start = params["trail_start_pct"]
    trail_pct = params["trail_pct"]
    cooldown = params["cooldown_days"]
    leverage = params["leverage"]
    pos_pct = params["position_pct"]
    fee_rate = params["fee_rate"]

    # Filters
    max_pen = params.get("max_penetration", 999)
    max_vol = params.get("max_vol_ratio", 999)
    min_slope = params.get("min_slope", 0)  # slope must be <= -min_slope
    max_close_dist = params.get("max_close_dist", -999)  # close_dist must be >= this

    start_ts, end_ts = pd.Timestamp(start_dt), pd.Timestamp(end_dt)

    balance = initial_balance
    total_win = total_loss = 0.0
    n_trades = n_sl = n_trail = 0
    n_filtered = 0

    for sym, df in precomputed.items():
        pos = None
        last_exit = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            ts = row["timestamp"]
            if ts < start_ts or ts > end_ts:
                continue

            # Exit
            if pos is not None:
                high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
                entry = pos["entry"]
                sl_price = entry * (1 + sl_pct / 100)

                if high >= sl_price:
                    pnl = pos["size"] * leverage * (-sl_pct) / 100 - pos["size"] * fee_rate * 2
                    balance += pnl
                    total_loss += abs(pnl)
                    n_trades += 1
                    n_sl += 1
                    pos = None
                    last_exit = ts
                    continue

                cur_pnl = (entry - close) / entry * 100
                if cur_pnl >= trail_start:
                    pos["trailing"] = True
                    if low < pos.get("lowest", entry):
                        pos["lowest"] = low
                        pos["trail_stop"] = low * (1 + trail_pct / 100)
                if pos.get("trailing") and close >= pos.get("trail_stop", entry * 2):
                    actual_pnl_pct = (entry - pos["trail_stop"]) / entry * 100
                    pnl = pos["size"] * leverage * actual_pnl_pct / 100 - pos["size"] * fee_rate * 2
                    balance += pnl
                    if pnl > 0:
                        total_win += pnl
                    else:
                        total_loss += abs(pnl)
                    n_trades += 1
                    n_trail += 1
                    pos = None
                    last_exit = ts
                    continue

            # Entry
            if pos is None and row["touch"]:
                if last_exit and (ts - last_exit).days < cooldown:
                    continue

                pen = float(row["high_pen"]) if not pd.isna(row["high_pen"]) else 0
                vr = float(row["vol_ratio"]) if not pd.isna(row["vol_ratio"]) else 1
                sl = float(row["slope"]) if not pd.isna(row["slope"]) else 0
                cd = float(row["close_dist"]) if not pd.isna(row["close_dist"]) else 0

                # Apply filters
                if pen > max_pen:
                    n_filtered += 1
                    continue
                if vr > max_vol:
                    n_filtered += 1
                    continue
                if sl > -min_slope:
                    n_filtered += 1
                    continue
                if cd < max_close_dist:
                    n_filtered += 1
                    continue

                price = float(row["close"])
                size = balance * pos_pct
                if size < 5:
                    continue
                pos = {
                    "entry": price, "size": size,
                    "trailing": False, "lowest": price, "trail_stop": price * 2,
                }

    # Force close
    pf = total_win / total_loss if total_loss > 0 else 999
    wr = (n_trail / n_trades * 100) if n_trades > 0 else 0
    ret = (balance - initial_balance) / initial_balance * 100
    return {
        "trades": n_trades, "sl": n_sl, "trail": n_trail,
        "win_rate": wr, "pf": pf, "return_pct": ret,
        "balance": balance, "filtered": n_filtered,
        "sl_pct_of_trades": n_sl / n_trades * 100 if n_trades > 0 else 0,
    }


# ─── Grid Search ──────────────────────────────────────────────

def main():
    start_dt = datetime(2025, 1, 2)
    end_dt = datetime(2026, 3, 22)

    print(f"\n{'='*100}")
    print(f"  MA100 Filter Grid Search")
    print(f"  Period: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")
    print(f"{'='*100}\n")

    loader = DataLoader()
    print("  Loading data...")
    data = load_1d_data(loader, start_dt, end_dt)
    print(f"  {len(data)} symbols loaded")

    print("  Precomputing indicators...")
    precomputed = precompute(data)
    print(f"  Done\n")

    # Baseline (no filter)
    baseline = run_backtest(precomputed, BASE, start_dt, end_dt)
    print(f"  Baseline: {baseline['trades']} trades, SL {baseline['sl']} ({baseline['sl_pct_of_trades']:.0f}%), "
          f"PF {baseline['pf']:.2f}, WR {baseline['win_rate']:.1f}%, Return {baseline['return_pct']:+.1f}%")

    # Grid
    grid = {
        "max_penetration": [999, 5.0, 3.0, 2.0, 1.5, 1.0, 0.5],
        "max_vol_ratio":   [999, 3.0, 2.5, 2.0, 1.5],
        "min_slope":       [0, 0.3, 0.5, 0.8, 1.0],
        "max_close_dist":  [-999, -8.0, -5.0, -3.0, -2.0],
    }

    combos = list(itertools.product(
        grid["max_penetration"], grid["max_vol_ratio"],
        grid["min_slope"], grid["max_close_dist"],
    ))
    # Remove all-default combo
    combos = [c for c in combos if c != (999, 999, 0, -999)]
    print(f"\n  Grid: {len(combos)} combinations\n")

    results = []
    t0 = time.time()
    for i, (mp, mv, ms, mcd) in enumerate(combos):
        p = BASE.copy()
        p["max_penetration"] = mp
        p["max_vol_ratio"] = mv
        p["min_slope"] = ms
        p["max_close_dist"] = mcd
        r = run_backtest(precomputed, p, start_dt, end_dt)
        r["params"] = {"pen": mp, "vol": mv, "slope": ms, "dist": mcd}
        results.append(r)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(combos)} ({(i+1)/elapsed:.0f}/s)")

    elapsed = time.time() - t0
    print(f"  Done: {len(combos)} combos in {elapsed:.0f}s\n")

    # Filter: min 200 trades
    results = [r for r in results if r["trades"] >= 200]
    print(f"  After min 200 trades filter: {len(results)}")

    # Sort by PF
    results.sort(key=lambda x: x["pf"], reverse=True)

    def fmt(p):
        parts = []
        if p["pen"] < 999: parts.append(f"Pen<{p['pen']}%")
        if p["vol"] < 999: parts.append(f"Vol<{p['vol']}x")
        if p["slope"] > 0: parts.append(f"Slope<-{p['slope']}%")
        if p["dist"] > -999: parts.append(f"Dist>{p['dist']}%")
        return " + ".join(parts) if parts else "NoFilter"

    print(f"\n{'='*120}")
    print(f"  TOP 25 by PF (min 200 trades)")
    print(f"{'='*120}")
    print(f"  {'#':>3s}  {'PF':>5s}  {'WR':>5s}  {'Ret':>8s}  {'Trd':>5s}  {'SL':>5s}  {'SL%':>5s}  {'Trail':>5s}  {'Filt':>5s}  Filters")
    print(f"  {'-'*115}")

    for i, r in enumerate(results[:25]):
        p = r["params"]
        print(
            f"  {i+1:3d}  {r['pf']:5.2f}  {r['win_rate']:5.1f}  {r['return_pct']:>+7.1f}%  "
            f"{r['trades']:>5d}  {r['sl']:>5d}  {r['sl_pct_of_trades']:>4.0f}%  "
            f"{r['trail']:>5d}  {r['filtered']:>5d}  {fmt(r['params'])}"
        )

    # Baseline comparison
    print(f"  {'-'*115}")
    print(
        f"  BASE {baseline['pf']:5.2f}  {baseline['win_rate']:5.1f}  {baseline['return_pct']:>+7.1f}%  "
        f"{baseline['trades']:>5d}  {baseline['sl']:>5d}  {baseline['sl_pct_of_trades']:>4.0f}%  "
        f"{baseline['trail']:>5d}  {'0':>5s}  NoFilter"
    )

    # Best bang-for-buck: highest PF with return > baseline * 0.5
    min_ret = baseline["return_pct"] * 0.5
    balanced = [r for r in results if r["return_pct"] >= min_ret]
    if balanced:
        balanced.sort(key=lambda x: x["pf"], reverse=True)
        print(f"\n{'='*120}")
        print(f"  BALANCED TOP 10 (PF best, Return >= {min_ret:.0f}%)")
        print(f"{'='*120}")
        print(f"  {'#':>3s}  {'PF':>5s}  {'WR':>5s}  {'Ret':>8s}  {'Trd':>5s}  {'SL':>5s}  {'SL%':>5s}  {'Trail':>5s}  Filters")
        print(f"  {'-'*105}")
        for i, r in enumerate(balanced[:10]):
            print(
                f"  {i+1:3d}  {r['pf']:5.2f}  {r['win_rate']:5.1f}  {r['return_pct']:>+7.1f}%  "
                f"{r['trades']:>5d}  {r['sl']:>5d}  {r['sl_pct_of_trades']:>4.0f}%  "
                f"{r['trail']:>5d}  {fmt(r['params'])}"
            )

    # SL% 가장 낮은 것
    results.sort(key=lambda x: x["sl_pct_of_trades"])
    print(f"\n{'='*120}")
    print(f"  LOWEST SL% TOP 10 (min 200 trades)")
    print(f"{'='*120}")
    print(f"  {'#':>3s}  {'PF':>5s}  {'WR':>5s}  {'Ret':>8s}  {'Trd':>5s}  {'SL':>5s}  {'SL%':>5s}  {'Trail':>5s}  Filters")
    print(f"  {'-'*105}")
    for i, r in enumerate(results[:10]):
        print(
            f"  {i+1:3d}  {r['pf']:5.2f}  {r['win_rate']:5.1f}  {r['return_pct']:>+7.1f}%  "
            f"{r['trades']:>5d}  {r['sl']:>5d}  {r['sl_pct_of_trades']:>4.0f}%  "
            f"{r['trail']:>5d}  {fmt(r['params'])}"
        )

    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    main()
