"""
MA100 Grid Search — "이미 떨어진 정도" 필터

MA100 터치 캔들에서 close가 high 대비 얼마나 이미 떨어졌는지를 기준으로 필터링.

필터 후보:
  1. max_candle_drop: (high - close) / high * 100 — 캔들 내 하락폭%
  2. max_rejection_ratio: (high - close) / (high - low) — 캔들 범위 중 이미 소화한 비율
  3. max_high_to_close_pct: 고점 대비 종가 하락% (= 이미 먹은 수익)
  4. max_body_ratio: |open - close| / (high - low) — 몸통 대비 꼬리 비율

사용법:
    python scripts/grid_search_ma100_drop.py
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


def load_1d_data(loader, start_dt, end_dt):
    avail = set(loader.get_available_symbols())
    syms = [s for s in avail if s.split("/")[0] not in STABLECOINS]
    data = {}
    w = timedelta(days=150)
    s_str, e_str = (start_dt - w).strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    for sym in syms:
        tfs = loader.get_available_timeframes(sym)
        df = None
        if "1d" in tfs:
            df = loader.load(sym, "1d", start=s_str, end=e_str)
            if df is None or len(df) < 120: df = None
        if df is None and "4h" in tfs:
            raw = loader.load(sym, "4h", start=s_str, end=e_str)
            if raw is not None and len(raw) >= 600:
                raw["timestamp"] = pd.to_datetime(raw["timestamp"])
                df = raw.set_index("timestamp").resample("1D").agg(
                    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
                ).dropna().reset_index()
                if len(df) < 120: df = None
        if df is not None:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            data[sym] = df
    return data


def precompute(data):
    result = {}
    for sym, df in data.items():
        df = df.copy()
        df["ma100"] = df["close"].rolling(100).mean()
        df["slope"] = (df["ma100"] - df["ma100"].shift(3)) / df["ma100"].shift(3) * 100
        df["touch"] = (
            (df["slope"] < 0)
            & (df["high"] >= df["ma100"] * 0.99)
            & (df["close"] < df["ma100"])
        ).fillna(False)

        # 캔들 내 하락 지표
        candle_range = (df["high"] - df["low"]).clip(lower=1e-12)
        df["candle_drop_pct"] = (df["high"] - df["close"]) / df["high"] * 100  # 고점→종가 하락%
        df["rejection_ratio"] = (df["high"] - df["close"]) / candle_range  # 0~1, 1=종가가 저점
        df["body_pct"] = (df["open"] - df["close"]).abs() / candle_range  # 몸통/전체 비율
        df["upper_wick_pct"] = (df["high"] - df[["open","close"]].max(axis=1)) / candle_range  # 윗꼬리 비율
        df["already_moved_pct"] = (df["ma100"] - df["close"]) / df["ma100"] * 100  # MA100→close 거리 (양수=이미 하락)

        result[sym] = df
    return result


def run_bt(precomputed, params, start_dt, end_dt, initial_balance=6500):
    sl_pct = 5.0
    trail_start, trail_pct, cooldown = 3.0, 2.0, 3
    leverage, pos_pct, fee_rate = 3, 0.02, 0.00055

    max_cd = params.get("max_candle_drop", 999)
    max_rr = params.get("max_rejection_ratio", 999)
    max_am = params.get("max_already_moved", 999)
    min_am = params.get("min_already_moved", -999)
    max_uw = params.get("max_upper_wick", 999)

    start_ts, end_ts = pd.Timestamp(start_dt), pd.Timestamp(end_dt)
    balance = initial_balance
    tw = tl = 0.0
    nt = nsl = ntr = nf = 0

    for sym, df in precomputed.items():
        pos = None
        last_exit = None
        for i in range(1, len(df)):
            row = df.iloc[i]
            ts = row["timestamp"]
            if ts < start_ts or ts > end_ts: continue

            if pos is not None:
                high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
                entry = pos["e"]
                if high >= entry * (1 + sl_pct / 100):
                    pnl = pos["s"] * leverage * (-sl_pct) / 100 - pos["s"] * fee_rate * 2
                    balance += pnl; tl += abs(pnl); nt += 1; nsl += 1
                    pos = None; last_exit = ts; continue
                cp = (entry - close) / entry * 100
                if cp >= trail_start:
                    pos["tr"] = True
                    if low < pos.get("lo", entry):
                        pos["lo"] = low; pos["ts2"] = low * (1 + trail_pct / 100)
                if pos.get("tr") and close >= pos.get("ts2", entry * 2):
                    ap = (entry - pos["ts2"]) / entry * 100
                    pnl = pos["s"] * leverage * ap / 100 - pos["s"] * fee_rate * 2
                    balance += pnl
                    if pnl > 0: tw += pnl
                    else: tl += abs(pnl)
                    nt += 1; ntr += 1; pos = None; last_exit = ts; continue

            if pos is None and row["touch"]:
                if last_exit and (ts - last_exit).days < cooldown: continue
                cd = float(row["candle_drop_pct"]) if not pd.isna(row["candle_drop_pct"]) else 0
                rr = float(row["rejection_ratio"]) if not pd.isna(row["rejection_ratio"]) else 0
                am = float(row["already_moved_pct"]) if not pd.isna(row["already_moved_pct"]) else 0
                uw = float(row["upper_wick_pct"]) if not pd.isna(row["upper_wick_pct"]) else 0

                if cd > max_cd: nf += 1; continue
                if rr > max_rr: nf += 1; continue
                if am > max_am: nf += 1; continue
                if am < min_am: nf += 1; continue
                if uw > max_uw: nf += 1; continue

                price = float(row["close"])
                sz = balance * pos_pct
                if sz < 5: continue
                pos = {"e": price, "s": sz, "tr": False, "lo": price, "ts2": price * 2}

    pf = tw / tl if tl > 0 else 999
    wr = ntr / nt * 100 if nt > 0 else 0
    ret = (balance - initial_balance) / initial_balance * 100
    return {"trades": nt, "sl": nsl, "trail": ntr, "wr": wr, "pf": pf, "ret": ret,
            "bal": balance, "filt": nf, "sl_pct": nsl/nt*100 if nt > 0 else 0}


def fmt(p):
    parts = []
    if p.get("max_candle_drop", 999) < 999: parts.append(f"Drop<{p['max_candle_drop']}%")
    if p.get("max_rejection_ratio", 999) < 999: parts.append(f"Reject<{p['max_rejection_ratio']:.0%}")
    if p.get("max_already_moved", 999) < 999: parts.append(f"Moved<{p['max_already_moved']}%")
    if p.get("min_already_moved", -999) > -999: parts.append(f"Moved>{p['min_already_moved']}%")
    if p.get("max_upper_wick", 999) < 999: parts.append(f"Wick<{p['max_upper_wick']:.0%}")
    return " + ".join(parts) if parts else "NoFilter"


def main():
    start_dt = datetime(2025, 1, 2)
    end_dt = datetime(2026, 3, 22)

    print(f"\n{'='*110}")
    print(f"  MA100 Drop Filter Grid Search")
    print(f"  '이미 떨어진 정도' 기반 필터 탐색")
    print(f"{'='*110}\n")

    loader = DataLoader()
    print("  Loading data...")
    data = load_1d_data(loader, start_dt, end_dt)
    print(f"  {len(data)} symbols")
    print("  Precomputing...")
    pc = precompute(data)
    print("  Done\n")

    # Baseline
    bl = run_bt(pc, {}, start_dt, end_dt)
    print(f"  Baseline: {bl['trades']} trades, SL {bl['sl']} ({bl['sl_pct']:.0f}%), PF {bl['pf']:.2f}, Ret {bl['ret']:+.0f}%\n")

    # 먼저 각 필터 단독 효과 분석
    print(f"{'='*110}")
    print(f"  STEP 1: 단독 필터 효과")
    print(f"{'='*110}")
    print(f"  {'Filter':<35s}  {'PF':>5s}  {'WR':>5s}  {'Ret':>8s}  {'Trd':>5s}  {'SL%':>5s}  {'Filt':>5s}")
    print(f"  {'-'*80}")

    singles = []
    # candle_drop: 고점→종가 하락%
    for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        r = run_bt(pc, {"max_candle_drop": v}, start_dt, end_dt)
        r["label"] = f"Drop<{v}%"
        r["p"] = {"max_candle_drop": v}
        singles.append(r)
        print(f"  {r['label']:<35s}  {r['pf']:5.2f}  {r['wr']:5.1f}  {r['ret']:>+7.0f}%  {r['trades']:>5d}  {r['sl_pct']:>4.0f}%  {r['filt']:>5d}")

    print()
    # rejection_ratio
    for v in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        r = run_bt(pc, {"max_rejection_ratio": v}, start_dt, end_dt)
        r["label"] = f"Reject<{v:.0%}"
        singles.append(r)
        print(f"  {r['label']:<35s}  {r['pf']:5.2f}  {r['wr']:5.1f}  {r['ret']:>+7.0f}%  {r['trades']:>5d}  {r['sl_pct']:>4.0f}%  {r['filt']:>5d}")

    print()
    # already_moved (MA100→close 거리)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        r = run_bt(pc, {"max_already_moved": v}, start_dt, end_dt)
        r["label"] = f"Moved<{v}%"
        singles.append(r)
        print(f"  {r['label']:<35s}  {r['pf']:5.2f}  {r['wr']:5.1f}  {r['ret']:>+7.0f}%  {r['trades']:>5d}  {r['sl_pct']:>4.0f}%  {r['filt']:>5d}")

    print()
    # upper_wick (윗꼬리 비율)
    for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
        r = run_bt(pc, {"max_upper_wick": v}, start_dt, end_dt)
        r["label"] = f"Wick<{v:.0%}"
        singles.append(r)
        print(f"  {r['label']:<35s}  {r['pf']:5.2f}  {r['wr']:5.1f}  {r['ret']:>+7.0f}%  {r['trades']:>5d}  {r['sl_pct']:>4.0f}%  {r['filt']:>5d}")

    # STEP 2: 조합 서치
    print(f"\n{'='*110}")
    print(f"  STEP 2: 조합 그리드 서치")
    print(f"{'='*110}\n")

    grid = list(itertools.product(
        [999, 3.0, 4.0, 5.0, 6.0],       # candle_drop
        [999, 0.5, 0.6, 0.7],             # rejection_ratio
        [999, 2.0, 3.0, 4.0, 5.0],        # already_moved
        [999, 0.2, 0.3, 0.4],             # upper_wick
    ))
    grid = [c for c in grid if c != (999, 999, 999, 999)]
    print(f"  {len(grid)} combinations")

    results = []
    t0 = time.time()
    for i, (cd, rr, am, uw) in enumerate(grid):
        p = {"max_candle_drop": cd, "max_rejection_ratio": rr, "max_already_moved": am, "max_upper_wick": uw}
        r = run_bt(pc, p, start_dt, end_dt)
        r["params"] = p
        results.append(r)
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(grid)}")

    print(f"  Done in {time.time()-t0:.0f}s\n")

    results = [r for r in results if r["trades"] >= 200]
    results.sort(key=lambda x: x["pf"], reverse=True)

    print(f"{'='*120}")
    print(f"  TOP 20 by PF (min 200 trades)")
    print(f"{'='*120}")
    print(f"  {'#':>3s}  {'PF':>5s}  {'WR':>5s}  {'Ret':>8s}  {'Trd':>5s}  {'SL':>5s}  {'SL%':>5s}  {'Trail':>5s}  Filters")
    print(f"  {'-'*105}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:3d}  {r['pf']:5.2f}  {r['wr']:5.1f}  {r['ret']:>+7.0f}%  {r['trades']:>5d}  {r['sl']:>5d}  {r['sl_pct']:>4.0f}%  {r['trail']:>5d}  {fmt(r['params'])}")
    print(f"  {'-'*105}")
    print(f"  BASE {bl['pf']:5.2f}  {bl['wr']:5.1f}  {bl['ret']:>+7.0f}%  {bl['trades']:>5d}  {bl['sl']:>5d}  {bl['sl_pct']:>4.0f}%  {bl['trail']:>5d}  NoFilter")

    # Ret >= baseline * 0.7
    min_ret = bl["ret"] * 0.7
    balanced = [r for r in results if r["ret"] >= min_ret]
    if balanced:
        balanced.sort(key=lambda x: x["pf"], reverse=True)
        print(f"\n{'='*120}")
        print(f"  BALANCED: PF best, Ret >= {min_ret:.0f}% (baseline 70%)")
        print(f"{'='*120}")
        print(f"  {'#':>3s}  {'PF':>5s}  {'WR':>5s}  {'Ret':>8s}  {'Trd':>5s}  {'SL%':>5s}  Filters")
        print(f"  {'-'*90}")
        for i, r in enumerate(balanced[:10]):
            print(f"  {i+1:3d}  {r['pf']:5.2f}  {r['wr']:5.1f}  {r['ret']:>+7.0f}%  {r['trades']:>5d}  {r['sl_pct']:>4.0f}%  {fmt(r['params'])}")

    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    main()
