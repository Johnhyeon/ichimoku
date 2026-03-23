"""MA100 SL 거래 공통점 분석 (일회성 스크립트)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_loader import DataLoader
from src.strategy import STABLECOINS

loader = DataLoader()
avail = set(loader.get_available_symbols())
all_syms = [s for s in avail if s.split('/')[0] not in STABLECOINS]

ma_period, slope_lookback, touch_buf = 100, 3, 1.0 / 100
sl_pct, trail_start, trail_pct, cooldown_days = 5.0, 3.0, 2.0, 3

start_dt = datetime(2025, 1, 2)
end_dt = datetime(2026, 3, 22)
start_str = (start_dt - timedelta(days=150)).strftime("%Y-%m-%d")
end_str = end_dt.strftime("%Y-%m-%d")

print("Loading 1d data...")
trades = []
n_loaded = 0

for sym in all_syms:
    tfs = loader.get_available_timeframes(sym)
    df = None
    if "1d" in tfs:
        df = loader.load(sym, "1d", start=start_str, end=end_str)
        if df is None or len(df) < 120:
            df = None
    if df is None and "4h" in tfs:
        raw = loader.load(sym, "4h", start=start_str, end=end_str)
        if raw is not None and len(raw) >= 600:
            raw["timestamp"] = pd.to_datetime(raw["timestamp"])
            df = raw.set_index("timestamp").resample("1D").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna().reset_index()
            if len(df) < 120:
                df = None
    if df is None:
        continue

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    df["ma100"] = df["close"].rolling(ma_period).mean()
    df["slope"] = (df["ma100"] - df["ma100"].shift(slope_lookback)) / df["ma100"].shift(slope_lookback) * 100

    df["short_sig"] = (
        (df["slope"] < 0)
        & (df["high"] >= df["ma100"] * (1 - touch_buf))
        & (df["close"] < df["ma100"])
    ).fillna(False)

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # Volume ratio
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"]

    # Price vs MA distance for past N days (trend strength)
    df["above_ma_5d"] = (df["close"] > df["ma100"]).rolling(5).sum()

    n_loaded += 1
    pos = None
    last_exit = None
    start_ts, end_ts = pd.Timestamp(start_dt), pd.Timestamp(end_dt)

    for i in range(1, len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        if ts < start_ts or ts > end_ts:
            continue

        if pos is not None:
            high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
            entry = pos["entry"]
            sl_price = entry * (1 + sl_pct / 100)

            if high >= sl_price:
                candles_held = i - pos["idx"]
                # Max adverse move before SL
                trades.append({
                    "sym": sym.split("/")[0],
                    "entry": entry,
                    "sl_price": sl_price,
                    "pnl_pct": -sl_pct,
                    "reason": "SL",
                    "entry_time": pos["ts"],
                    "exit_time": ts,
                    "candles_held": candles_held,
                    "slope": pos["slope"],
                    "ma100_dist": pos["ma100_dist"],
                    "vol_ratio": pos["vol_ratio"],
                    "rsi": pos["rsi"],
                    "above_ma_5d": pos["above_ma_5d"],
                    "entry_high_vs_ma": pos["entry_high_vs_ma"],
                })
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
                candles_held = i - pos["idx"]
                trades.append({
                    "sym": sym.split("/")[0],
                    "entry": entry,
                    "pnl_pct": (entry - close) / entry * 100,
                    "reason": "Trail",
                    "entry_time": pos["ts"],
                    "exit_time": ts,
                    "candles_held": candles_held,
                    "slope": pos["slope"],
                    "ma100_dist": pos["ma100_dist"],
                    "vol_ratio": pos["vol_ratio"],
                    "rsi": pos["rsi"],
                    "above_ma_5d": pos["above_ma_5d"],
                    "entry_high_vs_ma": pos["entry_high_vs_ma"],
                })
                pos = None
                last_exit = ts
                continue

        if pos is None and row["short_sig"]:
            if last_exit and (ts - last_exit).days < cooldown_days:
                continue
            price = float(row["close"])
            ma = float(row["ma100"])
            pos = {
                "entry": price,
                "idx": i,
                "ts": ts,
                "slope": float(row["slope"]),
                "ma100_dist": (price - ma) / ma * 100,
                "trailing": False,
                "lowest": price,
                "trail_stop": price * 2,
                "vol_ratio": float(row["vol_ratio"]) if not pd.isna(row["vol_ratio"]) else 1,
                "rsi": float(row["rsi"]) if not pd.isna(row["rsi"]) else 50,
                "above_ma_5d": float(row["above_ma_5d"]) if not pd.isna(row["above_ma_5d"]) else 0,
                "entry_high_vs_ma": (float(row["high"]) - ma) / ma * 100,
            }

print(f"Loaded {n_loaded} symbols")
print(f"Total trades: {len(trades)}")

sl_trades = [t for t in trades if t["reason"] == "SL"]
win_trades = [t for t in trades if t["reason"] == "Trail"]
print(f"SL: {len(sl_trades)}, Trail: {len(win_trades)}")

if not sl_trades:
    print("No SL trades")
    exit()

df_sl = pd.DataFrame(sl_trades)
df_w = pd.DataFrame(win_trades) if win_trades else pd.DataFrame()

print()
print("=" * 80)
print("  MA100 SL 거래 공통점 분석")
print("=" * 80)

print(f"\n  --- 1. 보유 기간 ---")
print(f"  SL  평균: {df_sl['candles_held'].mean():.1f}일 / 중앙: {df_sl['candles_held'].median():.0f}일")
if len(df_w) > 0:
    print(f"  Win 평균: {df_w['candles_held'].mean():.1f}일 / 중앙: {df_w['candles_held'].median():.0f}일")
print(f"  1일 이내 SL: {(df_sl['candles_held'] <= 1).sum()}건 ({(df_sl['candles_held'] <= 1).mean()*100:.0f}%)")
print(f"  3일 이내 SL: {(df_sl['candles_held'] <= 3).sum()}건 ({(df_sl['candles_held'] <= 3).mean()*100:.0f}%)")

print(f"\n  --- 2. MA100 기울기 (slope) ---")
print(f"  SL  평균: {df_sl['slope'].mean():.3f}%")
if len(df_w) > 0:
    print(f"  Win 평균: {df_w['slope'].mean():.3f}%")
for th in [-0.1, -0.2, -0.5, -1.0]:
    cnt = (df_sl["slope"] > th).sum()
    pct = cnt / len(df_sl) * 100
    print(f"  SL slope > {th}%: {cnt}건 ({pct:.0f}%)")

print(f"\n  --- 3. 진입 시 close vs MA100 거리 ---")
print(f"  SL  평균: {df_sl['ma100_dist'].mean():.2f}%")
if len(df_w) > 0:
    print(f"  Win 평균: {df_w['ma100_dist'].mean():.2f}%")
for th in [-0.5, -1.0, -2.0]:
    cnt = (df_sl["ma100_dist"] > th).sum()
    pct = cnt / len(df_sl) * 100
    print(f"  SL 거리 > {th}%: {cnt}건 ({pct:.0f}%)")

print(f"\n  --- 4. 진입 시 high가 MA100을 얼마나 뚫었나 ---")
print(f"  SL  평균: {df_sl['entry_high_vs_ma'].mean():.2f}%")
if len(df_w) > 0:
    print(f"  Win 평균: {df_w['entry_high_vs_ma'].mean():.2f}%")
for th in [0.5, 1.0, 2.0]:
    cnt = (df_sl["entry_high_vs_ma"] > th).sum()
    pct = cnt / len(df_sl) * 100
    print(f"  SL high > MA+{th}%: {cnt}건 ({pct:.0f}%)")

print(f"\n  --- 5. RSI ---")
print(f"  SL  평균: {df_sl['rsi'].mean():.1f}")
if len(df_w) > 0:
    print(f"  Win 평균: {df_w['rsi'].mean():.1f}")
for th in [40, 50, 55, 60]:
    cnt = (df_sl["rsi"] > th).sum()
    pct = cnt / len(df_sl) * 100
    print(f"  SL RSI > {th}: {cnt}건 ({pct:.0f}%)")

print(f"\n  --- 6. 거래량 (vol/sma20) ---")
print(f"  SL  평균: {df_sl['vol_ratio'].mean():.2f}x")
if len(df_w) > 0:
    print(f"  Win 평균: {df_w['vol_ratio'].mean():.2f}x")

print(f"\n  --- 7. 최근 5일 중 MA100 위에 있던 일수 ---")
print(f"  SL  평균: {df_sl['above_ma_5d'].mean():.1f}일 / 5일")
if len(df_w) > 0:
    print(f"  Win 평균: {df_w['above_ma_5d'].mean():.1f}일 / 5일")
for th in [3, 4, 5]:
    cnt = (df_sl["above_ma_5d"] >= th).sum()
    pct = cnt / len(df_sl) * 100
    print(f"  SL 5일중 {th}일+ MA위: {cnt}건 ({pct:.0f}%)")

# 코인별
print(f"\n  --- 8. 코인별 SL 비율 TOP 15 ---")
all_df = pd.DataFrame(trades)
coin_sl = df_sl["sym"].value_counts()
coin_total = all_df["sym"].value_counts()
merged = pd.DataFrame({"sl": coin_sl, "total": coin_total}).fillna(0).astype(int)
merged["sl_pct"] = merged["sl"] / merged["total"] * 100
merged = merged.sort_values("sl", ascending=False).head(15)
for sym, row in merged.iterrows():
    print(f"    {sym:<10s}: SL {row['sl']:3d} / {row['total']:3d} ({row['sl_pct']:.0f}%)")

print()
print("=" * 80)
