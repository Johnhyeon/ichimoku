"""Fractals: TP 있음 vs 트레일링만 vs 거래소 트레일링 비교."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, pandas as pd
from datetime import datetime, timedelta
from src.data_loader import DataLoader
from src.strategy import MAJOR_COINS

def compute_fractals(df, n=5):
    df = df.copy(); h, l = df["high"].values, df["low"].values; ln = len(df)
    fh, fl = np.full(ln, np.nan), np.full(ln, np.nan)
    for i in range(n, ln - n):
        if all(h[i] > h[i-j] and h[i] > h[i+j] for j in range(1, n+1)): fh[i] = h[i]
        if all(l[i] < l[i-j] and l[i] < l[i+j] for j in range(1, n+1)): fl[i] = l[i]
    df["last_fh"] = pd.Series(fh, index=df.index).ffill()
    df["last_fl"] = pd.Series(fl, index=df.index).ffill()
    ph, pl, pc = df["last_fh"].shift(1), df["last_fl"].shift(1), df["close"].shift(1)
    df["long_sig"] = ((pc <= ph) & (df["close"] > df["last_fh"]) & df["last_fh"].notna()).fillna(False)
    df["short_sig"] = ((pc >= pl) & (df["close"] < df["last_fl"]) & df["last_fl"].notna()).fillna(False)
    return df

def fast_bt(pc, p, sd, ed, ib=6500):
    lev, pp, mp, fr = p["leverage"], p["position_pct"], p["max_positions"], p["fee_rate"]
    sl, tp = p["sl_pct"], p.get("tp_pct", 0)  # tp=0 means no TP
    ts_p, tr, cd = p["trail_start_pct"], p["trail_pct"], p["cooldown_candles"]
    # Exchange trailing: activate price + trail distance (simulated tick-by-tick within candle)
    use_exchange_trail = p.get("exchange_trail", False)

    bal = ib; peak = ib; mdd = 0.0; pos = {}; cds = {}
    w = l2 = nt = 0; twp = tlp = 0.0
    sl_count = tp_count = tr_count = 0

    ats = set()
    for df in pc.values():
        sub = df[(df["timestamp"] >= pd.Timestamp(sd)) & (df["timestamp"] <= pd.Timestamp(ed))]
        ats.update(sub["timestamp"].tolist())
    ats = sorted(ats)
    rl = {}
    for sym, df in pc.items():
        sub = df[(df["timestamp"] >= pd.Timestamp(sd)) & (df["timestamp"] <= pd.Timestamp(ed))].reset_index(drop=True)
        if len(sub) >= 5: rl[sym] = (sub, {t: i for i, t in enumerate(sub["timestamp"].tolist())})

    for ts in ats:
        cl = []
        for sym, ps in pos.items():
            if sym not in rl: continue
            df, ti = rl[sym]
            if ts not in ti: continue
            row = df.iloc[ti[ts]]; side, ep = ps["side"], ps["ep"]
            hi, lo, close = float(row["high"]), float(row["low"]), float(row["close"])

            if use_exchange_trail:
                # Simulate exchange trailing: check OHLC path
                prices = [float(row["open"]), hi, lo, close] if close > float(row["open"]) else [float(row["open"]), lo, hi, close]
                er = epo = None
                for price in prices:
                    # SL check
                    if side == "long" and price <= ps["sl"]:
                        er, epo = "SL", ps["sl"]; break
                    elif side == "short" and price >= ps["sl"]:
                        er, epo = "SL", ps["sl"]; break
                    # TP check
                    if tp > 0:
                        if side == "long" and price >= ps["tp"]:
                            er, epo = "TP", ps["tp"]; break
                        elif side == "short" and price <= ps["tp"]:
                            er, epo = "TP", ps["tp"]; break
                    # Trail activation + trigger
                    if side == "long":
                        cur_pnl = (price / ep - 1) * 100
                    else:
                        cur_pnl = (1 - price / ep) * 100
                    if cur_pnl >= ts_p:
                        ps["trailing"] = True
                    if ps.get("trailing"):
                        if side == "long":
                            if price > ps.get("highest", ep):
                                ps["highest"] = price
                                ps["trail_stop"] = price * (1 - tr / 100)
                            if price <= ps.get("trail_stop", 0):
                                er, epo = "TR", ps["trail_stop"]; break
                        else:
                            if price < ps.get("lowest", ep):
                                ps["lowest"] = price
                                ps["trail_stop"] = price * (1 + tr / 100)
                            if price >= ps.get("trail_stop", ep * 2):
                                er, epo = "TR", ps["trail_stop"]; break
            else:
                # Bot-level check (current behavior)
                if side == "long":
                    cur = (close/ep-1)*100; best = max(ps["best"], (hi/ep-1)*100)
                else:
                    cur = (1-close/ep)*100; best = max(ps["best"], (1-lo/ep)*100)
                ps["best"] = best; er = epo = None
                if side == "long" and lo <= ps["sl"]: er, epo = "SL", ps["sl"]
                elif side == "short" and hi >= ps["sl"]: er, epo = "SL", ps["sl"]
                if not er and tp > 0:
                    if side == "long" and hi >= ps["tp"]: er, epo = "TP", ps["tp"]
                    elif side == "short" and lo <= ps["tp"]: er, epo = "TP", ps["tp"]
                if not er and best >= ts_p and best - cur >= tr: er, epo = "TR", close

            if er:
                pp2 = (epo/ep-1)*100 if side == "long" else (1-epo/ep)*100
                pnl = ps["sz"]*lev*pp2/100 - ps["sz"]*fr*2; bal += pnl; nt += 1
                if pnl > 0: w += 1; twp += pnl
                else: l2 += 1; tlp += pnl
                if er == "SL": sl_count += 1
                elif er == "TP": tp_count += 1
                else: tr_count += 1
                cl.append(sym); cds[sym] = cd
        for s in cl: del pos[s]
        if bal > peak: peak = bal
        dd = (bal-peak)/peak*100
        if dd < mdd: mdd = dd
        for s in list(cds):
            cds[s] -= 1
            if cds[s] <= 0: del cds[s]
        if len(pos) < mp:
            ca = []
            for sym in rl:
                if sym in pos or sym in cds: continue
                df, ti = rl[sym]
                if ts not in ti: continue
                row = df.iloc[ti[ts]]
                if row["long_sig"]: ca.append((sym, "long", float(row["volume"])))
                elif row["short_sig"]: ca.append((sym, "short", float(row["volume"])))
            ca.sort(key=lambda x: x[2], reverse=True)
            for sym, side, _ in ca:
                if len(pos) >= mp: break
                df, ti = rl[sym]; price = float(df.iloc[ti[ts]]["close"]); sz = bal * pp
                if sz < 5: continue
                if side == "long":
                    slp = price*(1-sl/100); tpp = price*(1+tp/100) if tp > 0 else 0
                else:
                    slp = price*(1+sl/100); tpp = price*(1-tp/100) if tp > 0 else 0
                pos[sym] = {"side": side, "ep": price, "sz": sz, "sl": slp, "tp": tpp,
                            "best": 0, "trailing": False, "highest": price, "lowest": price, "trail_stop": 0}
    # force close
    for sym, ps in pos.items():
        if sym not in rl: continue
        df, _ = rl[sym]; price = float(df.iloc[-1]["close"])
        pp2 = (price/ps["ep"]-1)*100 if ps["side"] == "long" else (1-price/ps["ep"])*100
        pnl = ps["sz"]*lev*pp2/100 - ps["sz"]*fr*2; bal += pnl; nt += 1
        if pnl > 0: w += 1; twp += pnl
        else: l2 += 1; tlp += pnl
    pf = abs(twp/tlp) if tlp != 0 else 999; wr = w/nt*100 if nt > 0 else 0
    return {"pf": pf, "wr": wr, "ret": (bal-ib)/ib*100, "mdd": mdd, "trades": nt,
            "sl": sl_count, "tp": tp_count, "tr": tr_count}

loader = DataLoader()
symbols = [s for s in MAJOR_COINS if s in set(loader.get_available_symbols())]
sd = datetime(2025, 1, 2); sp = datetime(2025, 9, 1); ed = datetime(2026, 3, 22)

raw = {}
for sym in symbols:
    if "4h" not in loader.get_available_timeframes(sym): continue
    df = loader.load(sym, "4h", start=(sd - timedelta(days=60)).strftime("%Y-%m-%d"), end=ed.strftime("%Y-%m-%d"))
    if df is not None and len(df) >= 30:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
        raw[sym] = df

pc = {sym: compute_fractals(df, 5) for sym, df in raw.items()}
base = {"leverage": 10, "position_pct": 0.05, "max_positions": 5, "fee_rate": 0.00055, "cooldown_candles": 2}

configs = [
    # (label, tp, trail_start, trail_pct, exchange_trail)
    # 현재 설정
    ("Current: SL3+TP10+Trail(2/1.5)",     3.0, 10.0, 2.0, 1.5, False),
    # TP 없이 트레일링만
    ("TrailOnly: SL3+Trail(2/1.5)",         3.0, 0, 2.0, 1.5, False),
    ("TrailOnly: SL3+Trail(3/2)",           3.0, 0, 3.0, 2.0, False),
    ("TrailOnly: SL3+Trail(2/1)",           3.0, 0, 2.0, 1.0, False),
    ("TrailOnly: SL3+Trail(1.5/1)",         3.0, 0, 1.5, 1.0, False),
    ("TrailOnly: SL3+Trail(4/2)",           3.0, 0, 4.0, 2.0, False),
    ("TrailOnly: SL3+Trail(5/3)",           3.0, 0, 5.0, 3.0, False),
    # 거래소 트레일링 시뮬 (OHLC 경로 순회)
    ("ExchTrail: SL3+TP10+Trail(2/1.5)",    3.0, 10.0, 2.0, 1.5, True),
    ("ExchTrail: SL3+Trail(2/1.5)",         3.0, 0, 2.0, 1.5, True),
    ("ExchTrail: SL3+Trail(3/2)",           3.0, 0, 3.0, 2.0, True),
    ("ExchTrail: SL3+Trail(2/1)",           3.0, 0, 2.0, 1.0, True),
    # 넓은 SL + 트레일링
    ("TrailOnly: SL5+Trail(3/2)",           5.0, 0, 3.0, 2.0, False),
    ("ExchTrail: SL5+Trail(3/2)",           5.0, 0, 3.0, 2.0, True),
]

print("=" * 125)
print("  Fractals 4H n=5: Exit Strategy Comparison (10x)")
print("=" * 125)
print(f"  {'Config':<40s}  {'PF':>5s}  {'WR':>5s}  {'Ret':>10s}  {'MDD':>6s}  {'Trd':>5s}  {'SL':>4s}  {'TP':>4s}  {'TR':>4s}  | {'TrnPF':>5s}  {'TstPF':>5s}  WF")
print(f"  {'-'*120}")

for label, sl, tp, ts_p, tr, exch in configs:
    p = {**base, "sl_pct": sl, "tp_pct": tp, "trail_start_pct": ts_p, "trail_pct": tr, "exchange_trail": exch}
    full = fast_bt(pc, p, sd, ed)
    train = fast_bt(pc, p, sd, sp)
    test = fast_bt(pc, p, sp, ed)
    ratio = test["pf"] / train["pf"] if train["pf"] > 0 else 0
    vd = "PASS" if ratio >= 0.7 else ("WEAK" if ratio >= 0.5 else "FAIL")
    print(f"  {label:<40s}  {full['pf']:5.2f}  {full['wr']:5.1f}  {full['ret']:>+9.0f}%  {full['mdd']:>5.1f}%  {full['trades']:>5d}  {full['sl']:>4d}  {full['tp']:>4d}  {full['tr']:>4d}  | {train['pf']:>5.2f}  {test['pf']:>5.2f}  {vd}")

print(f"  {'='*120}")
