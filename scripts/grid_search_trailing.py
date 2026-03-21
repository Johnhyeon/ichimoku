"""Trailing stop grid search: Ichimoku + MA100 only"""
import sys, time
sys.path.insert(0, ".")

from datetime import datetime
from scripts.backtest_combined import (
    CombinedBacktester, ICHIMOKU_CONFIG, MIRROR_CONFIG, MA100_CONFIG,
    load_4h_data, load_1d_data,
)
from src.data_loader import DataLoader

START = "2025-01-02"
END = "2026-03-18"
BALANCE = 6500.0

start_dt = datetime.strptime(START, "%Y-%m-%d")
end_dt = datetime.strptime(END, "%Y-%m-%d").replace(hour=23, minute=59)

print("Loading data...")
loader = DataLoader()
data_4h = load_4h_data(loader, start_dt, end_dt)
data_1d = load_1d_data(loader, start_dt, end_dt)
print(f"Loaded: {len(data_4h)} (4h), {len(data_1d)} (1d)")


def run_ichi(cfg):
    bt = CombinedBacktester(
        initial_balance=BALANCE,
        enable_ichimoku=True, enable_mirror=False, enable_ma100=False, enable_dca=False,
    )
    bt.ichimoku_config = cfg
    bt.run({}, data_4h, data_1d, start_dt, end_dt)
    return bt


def run_ma100(cfg):
    bt = CombinedBacktester(
        initial_balance=BALANCE,
        enable_ichimoku=False, enable_mirror=False, enable_ma100=True, enable_dca=False,
    )
    bt.ma100_config = cfg
    bt.run({}, {}, data_1d, start_dt, end_dt)
    return bt


def summarize(bt, strategy_name):
    trades = [t for t in bt.trades if t["strategy"] == strategy_name]
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0, "pnl": 0, "pf": 0, "avg": 0}
    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    pnl = sum(t["pnl_usd"] for t in trades)
    wr = len(wins) / n * 100
    gp = sum(t["pnl_usd"] for t in wins) if wins else 0
    gl = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0.01
    return {"n": n, "wr": wr, "pnl": pnl, "pf": gp / gl, "avg": pnl / n}


# ═══════════════════════════════════════════════════
# 1. Ichimoku: trail_pct (TP 도달 후 되돌림%)
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  [1] ICHIMOKU trail_pct (TP 도달 시 활성화 -> X% 되돌림 청산)")
print("=" * 70)
ichi_trail_values = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
print(f"{'trail_pct':>10s} | {'Trades':>6s} | {'WR%':>6s} | {'PnL($)':>10s} | {'PF':>6s} | {'Avg$':>8s}")
print("-" * 60)
for tp in ichi_trail_values:
    cfg = ICHIMOKU_CONFIG.copy()
    cfg["min_sl_pct"] = 0.8
    cfg["trail_pct"] = tp
    bt = run_ichi(cfg)
    s = summarize(bt, "ichimoku")
    pf_str = f"{s['pf']:.2f}" if s['pf'] > 0 else "N/A"
    print(f"{tp:>9.1f}% | {s['n']:>6d} | {s['wr']:>5.1f}% | ${s['pnl']:>+9.2f} | {pf_str:>6s} | ${s['avg']:>+7.2f}")


# ═══════════════════════════════════════════════════
# 2. MA100: trail_start_pct x trail_pct
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  [2] MA100 trail_start x trail_pct")
print("=" * 70)
ma100_starts = [2.0, 3.0, 4.0, 5.0]
ma100_trails = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
print(f"{'start':>6s} {'trail':>6s} | {'Trades':>6s} | {'WR%':>6s} | {'PnL($)':>10s} | {'PF':>6s} | {'Avg$':>8s}")
print("-" * 65)
for ts in ma100_starts:
    for tp in ma100_trails:
        cfg = MA100_CONFIG.copy()
        cfg["trail_start_pct"] = ts
        cfg["trail_pct"] = tp
        bt = run_ma100(cfg)
        s = summarize(bt, "ma100")
        pf_str = f"{s['pf']:.2f}" if s['pf'] > 0 else "N/A"
        print(f"{ts:>5.1f}% {tp:>5.1f}% | {s['n']:>6d} | {s['wr']:>5.1f}% | ${s['pnl']:>+9.2f} | {pf_str:>6s} | ${s['avg']:>+7.2f}")

print("\nDone!")
