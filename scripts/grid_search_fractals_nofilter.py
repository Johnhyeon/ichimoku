"""
Fractals NoFilter Grid Search — SL/TP/Trail 최적화

필터 없이 순수 프랙탈 돌파/이탈만으로 진입, SL/TP/Trail 파라미터 탐색.
PF 1.90 baseline에서 SL/TP/Trail 튜닝으로 PF/MDD 개선 가능한지 확인.

사용법:
    python scripts/grid_search_fractals_nofilter.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import itertools
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.data_loader import DataLoader
from src.strategy import MAJOR_COINS


# ─── Indicators ───────────────────────────────────────────────

def compute_fractals(df, n=5):
    df = df.copy()
    h, l = df["high"].values, df["low"].values
    ln = len(df)
    fh, fl = np.full(ln, np.nan), np.full(ln, np.nan)
    for i in range(n, ln - n):
        if all(h[i] > h[i-j] and h[i] > h[i+j] for j in range(1, n+1)): fh[i] = h[i]
        if all(l[i] < l[i-j] and l[i] < l[i+j] for j in range(1, n+1)): fl[i] = l[i]
    df["last_fh"] = pd.Series(fh, index=df.index).ffill()
    df["last_fl"] = pd.Series(fl, index=df.index).ffill()

    # Raw signals (no filter)
    prev_h, prev_l, prev_c = df["last_fh"].shift(1), df["last_fl"].shift(1), df["close"].shift(1)
    df["long_sig"] = ((prev_c <= prev_h) & (df["close"] > df["last_fh"]) & df["last_fh"].notna()).fillna(False)
    df["short_sig"] = ((prev_c >= prev_l) & (df["close"] < df["last_fl"]) & df["last_fl"].notna()).fillna(False)
    return df


# ─── Fast Backtest ────────────────────────────────────────────

def fast_bt(precomputed, params, start_dt, end_dt, initial_balance=6500):
    lev = params["leverage"]
    pp = params["position_pct"]
    mp = params["max_positions"]
    fr = params["fee_rate"]
    sl = params["sl_pct"]
    tp = params["tp_pct"]
    ts_pct = params["trail_start_pct"]
    tr = params["trail_pct"]
    cd = params["cooldown_candles"]

    balance = initial_balance
    peak = initial_balance
    max_dd = 0.0
    positions = {}
    cooldowns = {}
    wins = losses = nt = 0
    twp = tlp = 0.0

    all_ts = set()
    for df in precomputed.values():
        sub = df[(df["timestamp"] >= pd.Timestamp(start_dt)) & (df["timestamp"] <= pd.Timestamp(end_dt))]
        all_ts.update(sub["timestamp"].tolist())
    all_ts = sorted(all_ts)

    rl = {}
    for sym, df in precomputed.items():
        sub = df[(df["timestamp"] >= pd.Timestamp(start_dt)) & (df["timestamp"] <= pd.Timestamp(end_dt))].reset_index(drop=True)
        if len(sub) >= 5:
            rl[sym] = (sub, {t: i for i, t in enumerate(sub["timestamp"].tolist())})

    for ts in all_ts:
        closed = []
        for sym, pos in positions.items():
            if sym not in rl: continue
            df, ti = rl[sym]
            if ts not in ti: continue
            row = df.iloc[ti[ts]]
            side, ep = pos["side"], pos["ep"]
            high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
            if side == "long":
                cur = (close/ep-1)*100; best = max(pos["best"], (high/ep-1)*100)
            else:
                cur = (1-close/ep)*100; best = max(pos["best"], (1-low/ep)*100)
            pos["best"] = best
            er = ep_out = None
            if side == "long" and low <= pos["sl"]: er, ep_out = "SL", pos["sl"]
            elif side == "short" and high >= pos["sl"]: er, ep_out = "SL", pos["sl"]
            if not er:
                if side == "long" and high >= pos["tp"]: er, ep_out = "TP", pos["tp"]
                elif side == "short" and low <= pos["tp"]: er, ep_out = "TP", pos["tp"]
            if not er and best >= ts_pct and best - cur >= tr:
                er, ep_out = "TR", close
            if er:
                pnl_p = (ep_out/ep-1)*100 if side == "long" else (1-ep_out/ep)*100
                pnl = pos["sz"] * lev * pnl_p / 100 - pos["sz"] * fr * 2
                balance += pnl; nt += 1
                if pnl > 0: wins += 1; twp += pnl
                else: losses += 1; tlp += pnl
                closed.append(sym)
                cooldowns[sym] = cd
        for s in closed: del positions[s]

        if balance > peak: peak = balance
        dd = (balance - peak) / peak * 100
        if dd < max_dd: max_dd = dd

        for s in list(cooldowns):
            cooldowns[s] -= 1
            if cooldowns[s] <= 0: del cooldowns[s]

        if len(positions) < mp:
            cands = []
            for sym in rl:
                if sym in positions or sym in cooldowns: continue
                df, ti = rl[sym]
                if ts not in ti: continue
                row = df.iloc[ti[ts]]
                if row["long_sig"]: cands.append((sym, "long", float(row["volume"])))
                elif row["short_sig"]: cands.append((sym, "short", float(row["volume"])))
            cands.sort(key=lambda x: x[2], reverse=True)
            for sym, side, _ in cands:
                if len(positions) >= mp: break
                df, ti = rl[sym]
                price = float(df.iloc[ti[ts]]["close"])
                sz = balance * pp
                if sz < 5: continue
                if side == "long":
                    sl_p = price*(1-sl/100); tp_p = price*(1+tp/100)
                else:
                    sl_p = price*(1+sl/100); tp_p = price*(1-tp/100)
                positions[sym] = {"side": side, "ep": price, "sz": sz, "sl": sl_p, "tp": tp_p, "best": 0}

    # force close
    for sym, pos in positions.items():
        if sym not in rl: continue
        df, _ = rl[sym]
        price = float(df.iloc[-1]["close"])
        pp2 = (price/pos["ep"]-1)*100 if pos["side"] == "long" else (1-price/pos["ep"])*100
        pnl = pos["sz"] * lev * pp2 / 100 - pos["sz"] * fr * 2
        balance += pnl; nt += 1
        if pnl > 0: wins += 1; twp += pnl
        else: losses += 1; tlp += pnl

    pf = abs(twp / tlp) if tlp != 0 else 999
    wr = wins / nt * 100 if nt > 0 else 0
    ret = (balance - initial_balance) / initial_balance * 100
    return {"pf": pf, "wr": wr, "ret": ret, "mdd": max_dd, "trades": nt, "bal": balance}


# ─── Main ─────────────────────────────────────────────────────

def main():
    start_dt = datetime(2025, 1, 2)
    split_dt = datetime(2025, 9, 1)
    end_dt = datetime(2026, 3, 22)

    print(f"\n{'='*110}")
    print(f"  Fractals NoFilter — SL/TP/Trail Grid Search")
    print(f"{'='*110}\n")

    loader = DataLoader()
    available = set(loader.get_available_symbols())
    symbols = [s for s in MAJOR_COINS if s in available]

    print("  Loading 4h data...")
    raw = {}
    for sym in symbols:
        if "4h" not in loader.get_available_timeframes(sym): continue
        df = loader.load(sym, "4h", start=(start_dt - timedelta(days=30)).strftime("%Y-%m-%d"),
                         end=end_dt.strftime("%Y-%m-%d"))
        if df is not None and len(df) >= 30:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            raw[sym] = df
    print(f"  {len(raw)} symbols\n")

    print("  Computing fractals...")
    pc = {sym: compute_fractals(df, 5) for sym, df in raw.items()}
    print("  Done\n")

    # ── Stage 1: SL/TP/Trail search (lev=10x fixed) ──
    base = {"leverage": 10, "position_pct": 0.05, "max_positions": 5, "fee_rate": 0.00055}

    grid = list(itertools.product(
        [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],       # sl
        [4.0, 6.0, 8.0, 10.0, 12.0, 15.0],     # tp
        [2.0, 3.0, 4.0, 5.0],                    # trail_start
        [1.0, 1.5, 2.0, 3.0],                    # trail_pct
        [2, 3],                                    # cooldown
    ))
    # filter: tp > sl, trail_start < tp
    grid = [(s, t, ts, tr, cd) for s, t, ts, tr, cd in grid if t > s and ts < t]
    print(f"  Stage 1: {len(grid)} combos (lev=10x)\n")

    results = []
    t0 = time.time()
    for i, (sl, tp, ts_pct, tr, cd) in enumerate(grid):
        p = {**base, "sl_pct": sl, "tp_pct": tp, "trail_start_pct": ts_pct, "trail_pct": tr, "cooldown_candles": cd}
        r = fast_bt(pc, p, start_dt, end_dt)
        r["p"] = p
        results.append(r)
        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            speed = (i+1) / elapsed
            print(f"  {i+1}/{len(grid)} ({speed:.1f}/s, ETA {(len(grid)-i-1)/speed:.0f}s)")

    print(f"\n  Done in {time.time()-t0:.0f}s")
    results = [r for r in results if r["trades"] >= 200]
    results.sort(key=lambda x: x["pf"], reverse=True)

    def fmt(p):
        return f"SL{p['sl_pct']}% TP{p['tp_pct']}% Trail({p['trail_start_pct']}%/{p['trail_pct']}%) cd{p['cooldown_candles']}"

    print(f"\n{'='*120}")
    print(f"  Stage 1 TOP 25 by PF (10x, min 200 trades)")
    print(f"{'='*120}")
    print(f"  {'#':>3s}  {'PF':>5s}  {'WR':>5s}  {'Ret':>10s}  {'MDD':>6s}  {'Trd':>5s}  Params")
    print(f"  {'-'*110}")
    for i, r in enumerate(results[:25]):
        print(f"  {i+1:3d}  {r['pf']:5.2f}  {r['wr']:5.1f}  {r['ret']:>+9.0f}%  {r['mdd']:>5.1f}%  {r['trades']:>5d}  {fmt(r['p'])}")

    # Baseline
    bl = fast_bt(pc, {**base, "sl_pct": 3.0, "tp_pct": 6.0, "trail_start_pct": 3.0, "trail_pct": 1.5, "cooldown_candles": 3}, start_dt, end_dt)
    print(f"  {'-'*110}")
    print(f"  BASE {bl['pf']:5.2f}  {bl['wr']:5.1f}  {bl['ret']:>+9.0f}%  {bl['mdd']:>5.1f}%  {bl['trades']:>5d}  SL3% TP6% Trail(3%/1.5%) cd3 (original)")

    # ── Stage 2: Walk-Forward for top 5 ──
    print(f"\n{'='*120}")
    print(f"  Stage 2: Walk-Forward Validation (Top 10)")
    print(f"{'='*120}")
    print(f"  {'#':>3s}  {'Full PF':>7s}  {'Full Ret':>9s}  {'Full MDD':>8s}  {'Train PF':>8s}  {'Test PF':>7s}  {'Ratio':>6s}  {'Verdict':>7s}  Params")
    print(f"  {'-'*115}")

    seen = set()
    count = 0
    for r in results[:30]:
        key = fmt(r["p"])
        if key in seen: continue
        seen.add(key)
        p = r["p"]
        train_r = fast_bt(pc, p, start_dt, split_dt)
        test_r = fast_bt(pc, p, split_dt, end_dt)
        ratio = test_r["pf"] / train_r["pf"] if train_r["pf"] > 0 else 0
        verdict = "PASS" if ratio >= 0.7 else ("WEAK" if ratio >= 0.5 else "FAIL")
        count += 1
        print(f"  {count:3d}  {r['pf']:>7.2f}  {r['ret']:>+8.0f}%  {r['mdd']:>7.1f}%  {train_r['pf']:>8.2f}  {test_r['pf']:>7.2f}  {ratio:>5.2f}  {verdict:>7s}  {key}")
        if count >= 10: break

    # ── Stage 3: Top PF with MDD < 15% ──
    safe = [r for r in results if r["mdd"] > -20]
    if safe:
        safe.sort(key=lambda x: x["pf"], reverse=True)
        print(f"\n{'='*120}")
        print(f"  MDD > -20% & PF Top 10")
        print(f"{'='*120}")
        print(f"  {'#':>3s}  {'PF':>5s}  {'WR':>5s}  {'Ret':>10s}  {'MDD':>6s}  {'Trd':>5s}  Params")
        print(f"  {'-'*110}")
        for i, r in enumerate(safe[:10]):
            print(f"  {i+1:3d}  {r['pf']:5.2f}  {r['wr']:5.1f}  {r['ret']:>+9.0f}%  {r['mdd']:>5.1f}%  {r['trades']:>5d}  {fmt(r['p'])}")

    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    main()
