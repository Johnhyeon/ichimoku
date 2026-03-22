"""
A/B Comparison: Ichimoku vs Williams Fractals (in Combined Portfolio)

A: 기존 (Ichimoku + Mirror + MA100 + DCA)
B: 교체 (Fractals  + Mirror + MA100 + DCA)

기존 combined 백테스트를 2번 돌려 비교합니다.
- A: 기존 그대로
- B: --no-ichimoku + Fractals 별도 계산 후 합산

사용법:
    python scripts/compare_ichimoku_vs_fractals.py
    python scripts/compare_ichimoku_vs_fractals.py --balance 6500 --start 2025-01-02 --end 2026-03-22
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.ichimoku import calculate_ichimoku
from src.strategy import MAJOR_COINS, STABLECOINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══ Fractals indicators (from grid_search_fractals_v2) ═══

def compute_fractals(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    highs = df["high"].values
    lows = df["low"].values
    length = len(df)
    fh = np.full(length, np.nan)
    fl = np.full(length, np.nan)
    for i in range(n, length - n):
        is_high = True
        for j in range(1, n + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
                break
        if is_high:
            fh[i] = highs[i]
        is_low = True
        for j in range(1, n + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
                break
        if is_low:
            fl[i] = lows[i]
    df["fractal_high"] = fh
    df["fractal_low"] = fl
    df["last_fractal_high"] = df["fractal_high"].ffill()
    df["last_fractal_low"] = df["fractal_low"].ffill()
    return df


def compute_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def compute_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_adx(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    pdm = h.diff().clip(lower=0)
    mdm = (-l.diff()).clip(lower=0)
    pdm = np.where(pdm > mdm, pdm, 0)
    mdm = np.where(mdm > pdm.astype(float), mdm, 0)
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = pd.Series(tr, index=df.index).ewm(alpha=1/p, min_periods=p).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/p, min_periods=p).mean() / atr
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1/p, min_periods=p).mean() / atr
    dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
    return dx.ewm(alpha=1/p, min_periods=p).mean()


def precompute_fractals(df: pd.DataFrame) -> pd.DataFrame:
    """프랙탈 시그널 + 고정 필터 (EMA20/50 + RSI 35-65 + ADX>20)."""
    df = df.copy()
    df = compute_fractals(df, 5)
    df["ema_20"] = compute_ema(df["close"], 20)
    df["ema_50"] = compute_ema(df["close"], 50)
    df["rsi"] = compute_rsi(df["close"], 14)
    df["adx"] = compute_adx(df, 14)

    prev_h = df["last_fractal_high"].shift(1)
    prev_l = df["last_fractal_low"].shift(1)
    prev_c = df["close"].shift(1)

    lr = ((prev_c <= prev_h) & (df["close"] > df["last_fractal_high"]) & df["last_fractal_high"].notna()).fillna(False)
    sr = ((prev_c >= prev_l) & (df["close"] < df["last_fractal_low"]) & df["last_fractal_low"].notna()).fillna(False)

    ema_l = df["ema_20"] > df["ema_50"]
    ema_s = df["ema_20"] < df["ema_50"]
    rsi_l = df["rsi"] <= 65
    rsi_s = df["rsi"] >= 35
    adx_ok = df["adx"] >= 20

    df["long_signal"] = lr & ema_l & rsi_l & adx_ok
    df["short_signal"] = sr & ema_s & rsi_s & adx_ok
    return df


# ═══ Ichimoku Strategy (simplified from combined) ═══

ICHIMOKU_CONFIG = {
    "leverage": 20, "position_pct": 0.05, "max_positions": 5,
    "min_cloud_thickness": 0.2, "min_sl_pct": 0.3, "max_sl_pct": 8.0,
    "sl_buffer": 0.2, "rr_ratio": 2.0, "trail_pct": 1.5,
    "max_loss_pct": 2.0, "cooldown_hours": 4, "use_btc_filter": True,
    "fee_rate": 0.00055,
}

FRACTALS_CONFIG = {
    "leverage": 5, "position_pct": 0.05, "max_positions": 5,
    "sl_pct": 3.0, "tp_pct": 10.0, "trail_start_pct": 2.0, "trail_pct": 2.0,
    "cooldown_candles": 2, "fee_rate": 0.00055,
}


def run_ichimoku_only(
    data_4h: Dict[str, pd.DataFrame],
    start_dt: datetime, end_dt: datetime,
    initial_balance: float,
    leverage: int = None,
    position_pct: float = None,
) -> dict:
    """이치모쿠 전략만 단독 백테스트."""
    config = ICHIMOKU_CONFIG.copy()
    if leverage is not None:
        config["leverage"] = leverage
    if position_pct is not None:
        config["position_pct"] = position_pct
    start_ts, end_ts = pd.Timestamp(start_dt), pd.Timestamp(end_dt)

    # BTC trend
    btc_sym = "BTC/USDT:USDT"
    btc_trend = {}
    if btc_sym in data_4h:
        btc = data_4h[btc_sym].copy()
        btc["sma26"] = btc["close"].rolling(26).mean()
        btc["sma52"] = btc["close"].rolling(52).mean()
        for _, r in btc.iterrows():
            if pd.notna(r["sma26"]) and pd.notna(r["sma52"]):
                btc_trend[r["timestamp"]] = bool(r["sma26"] > r["sma52"])

    # Precompute ichimoku signals
    ichi_data = {}
    for sym, df in data_4h.items():
        df = calculate_ichimoku(df)
        df = df.dropna(subset=["tenkan", "kijun", "cloud_top", "cloud_bottom"])
        if len(df) < 10:
            continue
        df["entry"] = (
            df["below_cloud"] & ~df["tenkan_above"]
            & (df["tk_cross_down"] | df["kijun_cross_down"])
            & (df["cloud_thickness"] >= config["min_cloud_thickness"])
            & ~df["in_cloud"]
        ).fillna(False)
        df["score"] = (
            df["chikou_bearish"].astype(int) * 2
            + (~df["cloud_green"]).astype(int)
            + (df["cloud_thickness"] > 1.0).astype(int)
        )
        ichi_data[sym] = df.reset_index(drop=True)

    balance = initial_balance
    positions = {}
    trades = []
    last_exits = {}
    equity_curve = []
    cooldown_td = timedelta(hours=config["cooldown_hours"])

    all_ts = set()
    for df in ichi_data.values():
        sub = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        all_ts.update(sub["timestamp"].tolist())
    all_ts = sorted(all_ts)

    row_lk = {}
    for sym, df in ichi_data.items():
        sub = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].reset_index(drop=True)
        if len(sub) > 0:
            row_lk[sym] = (sub, {ts: i for i, ts in enumerate(sub["timestamp"].tolist())})

    for ts in all_ts:
        # Exit
        closed = []
        for sym, pos in list(positions.items()):
            if sym not in row_lk:
                continue
            df, idx = row_lk[sym]
            if ts not in idx:
                continue
            row = df.iloc[idx[ts]]
            entry = pos["entry_price"]
            high, low, close = float(row["high"]), float(row["low"]), float(row["close"])

            # update lowest
            if low < pos["lowest"]:
                pos["lowest"] = low
                if low <= pos["tp"]:
                    pos["trailing"] = True
                    new_ts = low * (1 + config["trail_pct"] / 100)
                    pos["trail_stop"] = min(pos["trail_stop"], new_ts)

            exit_r = exit_p = None
            # MaxLoss
            ml = entry * (1 + config["max_loss_pct"] / 100)
            if high >= ml:
                exit_r, exit_p = "MaxLoss", ml
            # SL
            if not exit_r and high >= pos["sl"]:
                exit_r, exit_p = "SL", pos["sl"]
            # Trail
            if not exit_r and pos.get("trailing") and high >= pos["trail_stop"]:
                exit_r, exit_p = "Trail", pos["trail_stop"]
            # TP
            if not exit_r and not pos.get("trailing") and low <= pos["tp"]:
                exit_r, exit_p = "TP", pos["tp"]
            # Cloud
            if not exit_r and (row.get("in_cloud", False) or row.get("above_cloud", False)):
                exit_r, exit_p = "Cloud", close

            if exit_r:
                pnl_pct = (entry - exit_p) / entry * 100
                qty = pos["qty"]
                fee = qty * entry * config["fee_rate"] + qty * exit_p * config["fee_rate"]
                pnl = (entry - exit_p) * qty - fee
                balance += pnl
                trades.append({"symbol": sym, "side": "short", "pnl": pnl, "pnl_pct": pnl_pct * config["leverage"], "reason": exit_r, "entry_time": pos["entry_time"], "exit_time": ts})
                closed.append(sym)
                last_exits[sym] = ts

        for s in closed:
            del positions[s]

        # Entry
        if len(positions) < config["max_positions"] and balance > 0:
            bt = btc_trend.get(ts)
            if bt is not False:
                candidates = []
                for sym in row_lk:
                    if sym in positions:
                        continue
                    df, idx_map = row_lk[sym]
                    if ts not in idx_map:
                        continue
                    row = df.iloc[idx_map[ts]]
                    if not row["entry"]:
                        continue
                    le = last_exits.get(sym)
                    if le and (ts - le) < cooldown_td:
                        continue
                    price = float(row["close"])
                    cb = float(row["cloud_bottom"])
                    sl = cb * (1 + config["sl_buffer"] / 100)
                    sl_dist = (sl - price) / price * 100
                    if sl_dist < config["min_sl_pct"] or sl_dist > config["max_sl_pct"]:
                        continue
                    tp = price * (1 - sl_dist * config["rr_ratio"] / 100)
                    candidates.append((sym, price, sl, tp, int(row["score"]), float(row["cloud_thickness"])))

                candidates.sort(key=lambda x: (-x[4], -x[5]))
                for sym, price, sl, tp, _, _ in candidates:
                    if len(positions) >= config["max_positions"]:
                        break
                    margin = balance * config["position_pct"]
                    qty = margin * config["leverage"] / price
                    positions[sym] = {
                        "entry_price": price, "entry_time": ts,
                        "sl": sl, "tp": tp, "qty": qty,
                        "lowest": price, "trail_stop": sl, "trailing": False,
                    }

        equity_curve.append({"timestamp": ts, "equity": balance})

    # Force close
    for sym, pos in positions.items():
        if sym in row_lk:
            df, _ = row_lk[sym]
            price = float(df.iloc[-1]["close"])
            pnl = (pos["entry_price"] - price) * pos["qty"]
            fee = pos["qty"] * pos["entry_price"] * config["fee_rate"] * 2
            balance += pnl - fee
            trades.append({"symbol": sym, "side": "short", "pnl": pnl - fee, "reason": "ForceClose", "entry_time": pos["entry_time"], "exit_time": df.iloc[-1]["timestamp"]})

    wins = [t for t in trades if t["pnl"] > 0]
    losses_t = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins) if wins else 0
    gl = abs(sum(t["pnl"] for t in losses_t)) if losses_t else 1
    return {
        "strategy": "Ichimoku",
        "trades": len(trades),
        "wins": len(wins),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "pnl": balance - initial_balance,
        "final_balance": balance,
        "pf": gp / gl if gl > 0 else 999,
        "equity_curve": equity_curve,
        "trade_list": trades,
    }


def run_fractals_only(
    data_4h: Dict[str, pd.DataFrame],
    start_dt: datetime, end_dt: datetime,
    initial_balance: float,
    leverage: int = None,
    position_pct: float = None,
) -> dict:
    """프랙탈 전략만 단독 백테스트 (최적 파라미터)."""
    config = FRACTALS_CONFIG.copy()
    if leverage is not None:
        config["leverage"] = leverage
    if position_pct is not None:
        config["position_pct"] = position_pct
    start_ts, end_ts = pd.Timestamp(start_dt), pd.Timestamp(end_dt)

    precomputed = {}
    for sym, df in data_4h.items():
        pc = precompute_fractals(df)
        sub = pc[(pc["timestamp"] >= start_ts) & (pc["timestamp"] <= end_ts)].reset_index(drop=True)
        if len(sub) >= 5:
            precomputed[sym] = sub

    balance = initial_balance
    positions = {}
    trades = []
    cooldowns = {}
    equity_curve = []

    all_ts = set()
    for df in precomputed.values():
        all_ts.update(df["timestamp"].tolist())
    all_ts = sorted(all_ts)

    row_lk = {}
    for sym, df in precomputed.items():
        row_lk[sym] = (df, {ts: i for i, ts in enumerate(df["timestamp"].tolist())})

    for ts in all_ts:
        closed = []
        for sym, pos in list(positions.items()):
            if sym not in row_lk:
                continue
            df, idx_map = row_lk[sym]
            if ts not in idx_map:
                continue
            row = df.iloc[idx_map[ts]]
            side, ep = pos["side"], pos["entry_price"]
            high, low, close = float(row["high"]), float(row["low"]), float(row["close"])

            if side == "long":
                cur_pct = (close / ep - 1) * 100
                best = max(pos["best"], (high / ep - 1) * 100)
            else:
                cur_pct = (1 - close / ep) * 100
                best = max(pos["best"], (1 - low / ep) * 100)
            pos["best"] = best

            exit_r = exit_p = None
            if side == "long" and low <= pos["sl"]:
                exit_r, exit_p = "SL", pos["sl"]
            elif side == "short" and high >= pos["sl"]:
                exit_r, exit_p = "SL", pos["sl"]
            if not exit_r:
                if side == "long" and high >= pos["tp"]:
                    exit_r, exit_p = "TP", pos["tp"]
                elif side == "short" and low <= pos["tp"]:
                    exit_r, exit_p = "TP", pos["tp"]
            if not exit_r and best >= config["trail_start_pct"]:
                if best - cur_pct >= config["trail_pct"]:
                    exit_r, exit_p = "Trail", close

            if exit_r:
                if side == "long":
                    pnl_pct = (exit_p / ep - 1) * 100
                else:
                    pnl_pct = (1 - exit_p / ep) * 100
                fee = pos["size"] * config["fee_rate"] * 2
                pnl = pos["size"] * config["leverage"] * pnl_pct / 100 - fee
                balance += pnl
                trades.append({"symbol": sym, "side": side, "pnl": pnl, "pnl_pct": pnl_pct * config["leverage"], "reason": exit_r, "entry_time": pos["entry_time"], "exit_time": ts})
                closed.append(sym)
                cooldowns[sym] = config["cooldown_candles"]

        for s in closed:
            del positions[s]

        for s in list(cooldowns):
            cooldowns[s] -= 1
            if cooldowns[s] <= 0:
                del cooldowns[s]

        if len(positions) < config["max_positions"]:
            cands = []
            for sym in row_lk:
                if sym in positions or sym in cooldowns:
                    continue
                df, idx_map = row_lk[sym]
                if ts not in idx_map:
                    continue
                row = df.iloc[idx_map[ts]]
                if row["long_signal"]:
                    cands.append((sym, "long", float(row["volume"])))
                elif row["short_signal"]:
                    cands.append((sym, "short", float(row["volume"])))

            cands.sort(key=lambda x: x[2], reverse=True)
            for sym, side, _ in cands:
                if len(positions) >= config["max_positions"]:
                    break
                df, idx_map = row_lk[sym]
                row = df.iloc[idx_map[ts]]
                price = float(row["close"])
                size = balance * config["position_pct"]
                if size < 5:
                    continue
                if side == "long":
                    sl = price * (1 - config["sl_pct"] / 100)
                    tp = price * (1 + config["tp_pct"] / 100)
                else:
                    sl = price * (1 + config["sl_pct"] / 100)
                    tp = price * (1 - config["tp_pct"] / 100)
                positions[sym] = {
                    "side": side, "entry_price": price, "entry_time": ts,
                    "size": size, "sl": sl, "tp": tp, "best": 0,
                }

        equity_curve.append({"timestamp": ts, "equity": balance})

    for sym, pos in positions.items():
        if sym in row_lk:
            df, _ = row_lk[sym]
            price = float(df.iloc[-1]["close"])
            side = pos["side"]
            pnl_pct = ((price / pos["entry_price"] - 1) if side == "long" else (1 - price / pos["entry_price"])) * 100
            fee = pos["size"] * config["fee_rate"] * 2
            pnl = pos["size"] * config["leverage"] * pnl_pct / 100 - fee
            balance += pnl
            trades.append({"symbol": sym, "side": side, "pnl": pnl, "reason": "ForceClose", "entry_time": pos["entry_time"], "exit_time": df.iloc[-1]["timestamp"]})

    wins = [t for t in trades if t["pnl"] > 0]
    losses_t = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins) if wins else 0
    gl = abs(sum(t["pnl"] for t in losses_t)) if losses_t else 1
    longs = [t for t in trades if t["side"] == "long"]
    shorts = [t for t in trades if t["side"] == "short"]
    return {
        "strategy": "Fractals",
        "trades": len(trades),
        "wins": len(wins),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "pnl": balance - initial_balance,
        "final_balance": balance,
        "pf": gp / gl if gl > 0 else 999,
        "longs": len(longs),
        "shorts": len(shorts),
        "long_pnl": sum(t["pnl"] for t in longs),
        "short_pnl": sum(t["pnl"] for t in shorts),
        "equity_curve": equity_curve,
        "trade_list": trades,
    }


# ═══ Overlap Analysis ═══

def analyze_overlap(ichi_trades, frac_trades):
    """두 전략의 시간대/코인 겹침 분석."""
    ichi_active = set()
    for t in ichi_trades:
        ichi_active.add((t["symbol"], str(t.get("entry_time", ""))))

    frac_active = set()
    for t in frac_trades:
        frac_active.add((t["symbol"], str(t.get("entry_time", ""))))

    # 같은 코인에서 동시 보유 기간 체크 (간단 버전)
    ichi_symbols = set(t["symbol"] for t in ichi_trades)
    frac_symbols = set(t["symbol"] for t in frac_trades)
    overlap_coins = ichi_symbols & frac_symbols

    return {
        "ichi_coins": len(ichi_symbols),
        "frac_coins": len(frac_symbols),
        "overlap_coins": len(overlap_coins),
        "overlap_list": sorted(s.split("/")[0] for s in overlap_coins),
    }


# ═══ Main ═══

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", type=float, default=6500)
    parser.add_argument("--start", default="2025-01-02")
    parser.add_argument("--end", default="2026-03-22")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    print(f"\n{'='*80}")
    print(f"  A/B Comparison: Ichimoku vs Williams Fractals")
    print(f"  Period: {args.start} ~ {args.end} | Balance: ${args.balance:,.0f}")
    print(f"{'='*80}\n")

    # 데이터 로드 (4h, 20 MAJOR_COINS)
    loader = DataLoader()
    available = set(loader.get_available_symbols())
    symbols = [s for s in MAJOR_COINS if s in available]

    warmup = timedelta(days=80)
    start_str = (start_dt - warmup).strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    print("  Loading 4h data...")
    data_4h = {}
    for sym in symbols:
        tfs = loader.get_available_timeframes(sym)
        if "4h" not in tfs:
            continue
        df = loader.load(sym, "4h", start=start_str, end=end_str)
        if df is not None and len(df) >= 60:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            data_4h[sym] = df
    print(f"  {len(data_4h)} symbols loaded\n")

    # ═══ 동일 조건 비교 (여러 레버리지) ═══
    print("  Running multi-leverage comparison...\n")

    leverage_sets = [5, 10, 20]
    pos_pct = 0.05

    print(f"{'='*100}")
    print(f"  EQUAL CONDITIONS COMPARISON (position={pos_pct*100:.0f}%, max_pos=5)")
    print(f"{'='*100}")
    print(f"  {'Lev':>4s}  {'':15s}  {'Trades':>6s}  {'Win%':>6s}  {'PF':>6s}  {'PnL':>12s}  {'Return':>9s}  {'Exits'}")
    print(f"  {'─'*95}")

    for lev in leverage_sets:
        ir = run_ichimoku_only(data_4h, start_dt, end_dt, args.balance, leverage=lev, position_pct=pos_pct)
        fr = run_fractals_only(data_4h, start_dt, end_dt, args.balance, leverage=lev, position_pct=pos_pct)

        # Exit reason summary
        def exit_summary(tlist):
            reasons = {}
            for t in tlist:
                reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
            return ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:3])

        print(f"  {lev:>3d}x  {'Ichimoku':15s}  {ir['trades']:>6d}  {ir['win_rate']:>5.1f}%  {ir['pf']:>5.2f}  ${ir['pnl']:>+10,.0f}  {ir['pnl']/args.balance*100:>+8.1f}%  {exit_summary(ir['trade_list'])}")
        print(f"  {lev:>3d}x  {'Fractals (L+S)':15s}  {fr['trades']:>6d}  {fr['win_rate']:>5.1f}%  {fr['pf']:>5.2f}  ${fr['pnl']:>+10,.0f}  {fr['pnl']/args.balance*100:>+8.1f}%  {exit_summary(fr['trade_list'])}")
        if fr.get("longs"):
            print(f"  {'':>4s}  {'  Long only':15s}  {fr['longs']:>6d}  {'':>6s}  {'':>6s}  ${fr['long_pnl']:>+10,.0f}")
            print(f"  {'':>4s}  {'  Short only':15s}  {fr['shorts']:>6d}  {'':>6s}  {'':>6s}  ${fr['short_pnl']:>+10,.0f}")
        print(f"  {'─'*95}")

    print()

    # 기본 설정(각자 최적)으로도 실행
    print("  Running with each strategy's own optimal settings...")
    ichi_result = run_ichimoku_only(data_4h, start_dt, end_dt, args.balance)
    frac_result = run_fractals_only(data_4h, start_dt, end_dt, args.balance)

    overlap = analyze_overlap(ichi_result["trade_list"], frac_result["trade_list"])

    # ═══ Report ═══
    print(f"{'='*80}")
    print(f"  COMPARISON RESULTS")
    print(f"{'='*80}")
    print()

    # Configs
    ic = ICHIMOKU_CONFIG
    fc = FRACTALS_CONFIG
    print(f"  {'':22s} {'Ichimoku (A)':>20s}  {'Fractals (B)':>20s}")
    print(f"  {'─'*64}")
    print(f"  {'Timeframe':22s} {'4h':>20s}  {'4h':>20s}")
    print(f"  {'Direction':22s} {'Short only':>20s}  {'Long + Short':>20s}")
    print(f"  {'Leverage':22s} {ic['leverage']:>19d}x  {fc['leverage']:>19d}x")
    print(f"  {'Position %':22s} {ic['position_pct']*100:>19.0f}%  {fc['position_pct']*100:>19.0f}%")
    print(f"  {'Max Positions':22s} {ic['max_positions']:>20d}  {fc['max_positions']:>20d}")
    print(f"  {'SL':22s} {'Cloud-based':>20s}  {fc['sl_pct']:>19.1f}%")
    print(f"  {'TP':22s} {'RR-based':>20s}  {fc['tp_pct']:>19.1f}%")
    print(f"  {'Trail':22s} {ic['trail_pct']:>19.1f}%  {fc['trail_start_pct']:.0f}%/{fc['trail_pct']:.0f}%")
    print(f"  {'Filters':22s} {'BTC trend':>20s}  {'EMA20/50+RSI+ADX':>20s}")
    print()

    # Results
    ir, fr = ichi_result, frac_result
    print(f"  {'':22s} {'Ichimoku (A)':>20s}  {'Fractals (B)':>20s}  {'Winner':>10s}")
    print(f"  {'═'*74}")
    def w(a, b, higher_better=True):
        if higher_better:
            return "A" if a > b else ("B" if b > a else "TIE")
        return "A" if a < b else ("B" if b < a else "TIE")

    print(f"  {'Total Trades':22s} {ir['trades']:>20d}  {fr['trades']:>20d}")
    print(f"  {'Win Rate':22s} {ir['win_rate']:>19.1f}%  {fr['win_rate']:>19.1f}%  {w(ir['win_rate'], fr['win_rate']):>10s}")
    print(f"  {'PnL':22s} ${ir['pnl']:>+18,.0f}  ${fr['pnl']:>+18,.0f}  {w(ir['pnl'], fr['pnl']):>10s}")
    print(f"  {'Final Balance':22s} ${ir['final_balance']:>18,.0f}  ${fr['final_balance']:>18,.0f}  {w(ir['final_balance'], fr['final_balance']):>10s}")
    print(f"  {'Return %':22s} {ir['pnl']/args.balance*100:>+19.1f}%  {fr['pnl']/args.balance*100:>+19.1f}%  {w(ir['pnl'], fr['pnl']):>10s}")
    print(f"  {'Profit Factor':22s} {ir['pf']:>20.2f}  {fr['pf']:>20.2f}  {w(ir['pf'], fr['pf']):>10s}")

    # Fractals direction breakdown
    if "longs" in fr:
        print(f"\n  {'':22s} {'Ichimoku (A)':>20s}  {'Fractals (B)':>20s}")
        print(f"  {'─'*64}")
        print(f"  {'Long trades':22s} {'N/A (short only)':>20s}  {fr['longs']:>20d}")
        print(f"  {'Short trades':22s} {ir['trades']:>20d}  {fr['shorts']:>20d}")
        print(f"  {'Long PnL':22s} {'N/A':>20s}  ${fr['long_pnl']:>+18,.0f}")
        print(f"  {'Short PnL':22s} ${ir['pnl']:>+18,.0f}  ${fr['short_pnl']:>+18,.0f}")

    # Overlap
    print(f"\n  {'─'*64}")
    print(f"  Overlap Analysis:")
    print(f"    Ichimoku traded {overlap['ichi_coins']} coins, Fractals traded {overlap['frac_coins']} coins")
    print(f"    Overlapping coins: {overlap['overlap_coins']} ({', '.join(overlap['overlap_list'])})")

    # 포트폴리오 예측
    print(f"\n{'='*80}")
    print(f"  PORTFOLIO IMPACT (with Mirror + MA100 + DCA)")
    print(f"{'='*80}")
    print()
    print(f"  4h slot 전략만 교체 시 예상 효과:")
    print(f"  {'':22s} {'Plan A (Ichimoku)':>20s}  {'Plan B (Fractals)':>20s}")
    print(f"  {'─'*64}")
    print(f"  {'4h PnL':22s} ${ir['pnl']:>+18,.0f}  ${fr['pnl']:>+18,.0f}")
    delta = fr['pnl'] - ir['pnl']
    print(f"  {'Delta':22s} {'baseline':>20s}  ${delta:>+18,.0f}")
    print(f"  {'Direction diversity':22s} {'Short only':>20s}  {'Long + Short':>20s}")
    print(f"  {'Risk correlation':22s} {'High (all short)':>20s}  {'Lower (mixed)':>20s}")
    print()

    # Exit reason breakdown
    print(f"  Exit Reasons:")
    for label, tlist in [("Ichimoku", ichi_result["trade_list"]), ("Fractals", frac_result["trade_list"])]:
        reasons = {}
        for t in tlist:
            r = t["reason"]
            reasons[r] = reasons.get(r, 0) + 1
        reason_str = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
        print(f"    {label}: {reason_str}")

    # Monthly PnL comparison
    print(f"\n  Monthly PnL:")
    print(f"  {'Month':8s}  {'Ichi':>10s}  {'Frac':>10s}  {'Better':>8s}")
    print(f"  {'─'*40}")
    ichi_monthly = {}
    for t in ichi_result["trade_list"]:
        m = pd.Timestamp(t["exit_time"]).to_period("M")
        ichi_monthly[m] = ichi_monthly.get(m, 0) + t["pnl"]
    frac_monthly = {}
    for t in frac_result["trade_list"]:
        m = pd.Timestamp(t["exit_time"]).to_period("M")
        frac_monthly[m] = frac_monthly.get(m, 0) + t["pnl"]
    all_months = sorted(set(list(ichi_monthly.keys()) + list(frac_monthly.keys())))
    ichi_wins_m = frac_wins_m = 0
    for m in all_months:
        ip = ichi_monthly.get(m, 0)
        fp = frac_monthly.get(m, 0)
        better = "Ichi" if ip > fp else ("Frac" if fp > ip else "Tie")
        if ip > fp:
            ichi_wins_m += 1
        elif fp > ip:
            frac_wins_m += 1
        print(f"  {str(m):8s}  ${ip:>+9,.0f}  ${fp:>+9,.0f}  {better:>8s}")
    print(f"  {'─'*40}")
    print(f"  {'Monthly wins':8s}  {ichi_wins_m:>10d}  {frac_wins_m:>10d}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
