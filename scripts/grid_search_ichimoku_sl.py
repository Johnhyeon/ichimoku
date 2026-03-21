"""Ichimoku min_sl_pct grid search"""
import sys, time
sys.path.insert(0, ".")

from datetime import datetime
from scripts.backtest_combined import (
    CombinedBacktester, ICHIMOKU_CONFIG,
    load_4h_data, load_1d_data,
)
from src.data_loader import DataLoader

MIN_SL_VALUES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

START = "2025-01-02"
END = "2026-03-18"
BALANCE = 6500.0

start_dt = datetime.strptime(START, "%Y-%m-%d")
end_dt = datetime.strptime(END, "%Y-%m-%d").replace(hour=23, minute=59)

# Load data once
print("Loading data...")
loader = DataLoader()
data_4h = load_4h_data(loader, start_dt, end_dt)
data_1d = load_1d_data(loader, start_dt, end_dt)

print(f"Loaded: {len(data_4h)} symbols (4h), {len(data_1d)} symbols (1d)")
print()

results = []
for min_sl in MIN_SL_VALUES:
    config = ICHIMOKU_CONFIG.copy()
    config["min_sl_pct"] = min_sl

    bt = CombinedBacktester(
        initial_balance=BALANCE,
        enable_ichimoku=True,
        enable_mirror=False,
        enable_ma100=False,
        enable_dca=False,
    )
    bt.ichimoku_config = config

    t0 = time.time()
    bt.run({}, data_4h, data_1d, start_dt, end_dt)
    elapsed = time.time() - t0

    trades = [t for t in bt.trades if t["strategy"] == "ichimoku"]
    n = len(trades)
    if n == 0:
        results.append((min_sl, 0, 0, 0, 0, 0, 0, 0))
        print(f"  min_sl={min_sl:.1f}%  | No trades")
        continue

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades)
    wr = len(wins) / n * 100
    gp = sum(t["pnl_usd"] for t in wins) if wins else 0
    gl = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0.01
    pf = gp / gl
    avg_pnl = total_pnl / n

    results.append((min_sl, n, len(wins), len(losses), wr, total_pnl, pf, avg_pnl))
    print(f"  min_sl={min_sl:.1f}%  | {n:3d} trades  WR={wr:5.1f}%  PnL=${total_pnl:+8.2f}  PF={pf:.2f}  Avg=${avg_pnl:+.2f}  ({elapsed:.1f}s)")

print()
print("=" * 80)
print(f"{'min_sl':>8s} | {'Trades':>6s} | {'Win':>4s} | {'Loss':>4s} | {'WR%':>6s} | {'PnL($)':>10s} | {'PF':>6s} | {'Avg$/trade':>10s}")
print("-" * 80)
for r in results:
    min_sl, n, w, l, wr, pnl, pf, avg = r
    pf_str = f"{pf:.2f}" if pf > 0 else "N/A"
    print(f"{min_sl:>7.1f}% | {n:>6d} | {w:>4d} | {l:>4d} | {wr:>5.1f}% | ${pnl:>+9.2f} | {pf_str:>6s} | ${avg:>+9.2f}")
print("=" * 80)
