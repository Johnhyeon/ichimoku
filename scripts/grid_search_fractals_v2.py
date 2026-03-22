"""
Williams Fractals Grid Search V2 - SL/TP/Trail Optimization

1차 그리드 서치에서 찾은 최적 필터 조합을 고정하고,
SL/TP/트레일링/레버리지/포지션비율 파라미터를 탐색합니다.

고정 필터: EMA20/50 + RSI(35-65) + ADX>20
탐색 대상: SL, TP, trail_start, trail_pct, leverage, position_pct, cooldown

사용법:
    python scripts/grid_search_fractals_v2.py
    python scripts/grid_search_fractals_v2.py --top 30
"""

import argparse
import itertools
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
from src.strategy import MAJOR_COINS

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Indicator Calculations (1차와 동일) ─────────────────────

def compute_fractals(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    highs = df["high"].values
    lows = df["low"].values
    length = len(df)
    fractal_high = np.full(length, np.nan)
    fractal_low = np.full(length, np.nan)
    for i in range(n, length - n):
        is_high = True
        for j in range(1, n + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
                break
        if is_high:
            fractal_high[i] = highs[i]
        is_low = True
        for j in range(1, n + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
                break
        if is_low:
            fractal_low[i] = lows[i]
    df["fractal_high"] = fractal_high
    df["fractal_low"] = fractal_low
    df["last_fractal_high"] = df["fractal_high"].ffill()
    df["last_fractal_low"] = df["fractal_low"].ffill()
    return df


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm.astype(float), minus_dm, 0)
    tr = pd.concat([
        high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = pd.Series(tr, index=df.index).ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, min_periods=period).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1/period, min_periods=period).mean()


def precompute_all(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """지표 + 고정 필터 적용한 시그널 계산."""
    df = df.copy()
    df = compute_fractals(df, n)

    # 고정 필터용 지표
    df["ema_20"] = compute_ema(df["close"], 20)
    df["ema_50"] = compute_ema(df["close"], 50)
    df["rsi"] = compute_rsi(df["close"], 14)
    df["adx"] = compute_adx(df, 14)

    # 기본 프랙탈 시그널
    prev_high = df["last_fractal_high"].shift(1)
    prev_low = df["last_fractal_low"].shift(1)
    prev_close = df["close"].shift(1)

    long_raw = (
        (prev_close <= prev_high) &
        (df["close"] > df["last_fractal_high"]) &
        df["last_fractal_high"].notna()
    ).fillna(False)

    short_raw = (
        (prev_close >= prev_low) &
        (df["close"] < df["last_fractal_low"]) &
        df["last_fractal_low"].notna()
    ).fillna(False)

    # 고정 필터 적용: EMA20/50 + RSI(35-65) + ADX>20
    ema_long = df["ema_20"] > df["ema_50"]
    ema_short = df["ema_20"] < df["ema_50"]
    rsi_long = df["rsi"] <= 65
    rsi_short = df["rsi"] >= 35
    adx_ok = df["adx"] >= 20

    df["long_signal"] = long_raw & ema_long & rsi_long & adx_ok
    df["short_signal"] = short_raw & ema_short & rsi_short & adx_ok

    return df


# ─── Fast Backtest ────────────────────────────────────────────

def fast_backtest(
    precomputed: Dict[str, pd.DataFrame],
    params: dict,
    start_dt: datetime,
    end_dt: datetime,
    initial_balance: float = 6500,
) -> dict:
    leverage = params["leverage"]
    position_pct = params["position_pct"]
    max_positions = params["max_positions"]
    fee_rate = params["fee_rate"]
    sl_pct = params["sl_pct"]
    tp_pct = params["tp_pct"]
    trail_start = params["trail_start_pct"]
    trail_pct = params["trail_pct"]
    cooldown = params["cooldown_candles"]

    balance = initial_balance
    peak_balance = initial_balance
    max_dd = 0.0
    positions = {}
    cooldowns = {}
    wins = losses = n_trades = n_longs = n_shorts = 0
    total_win_pnl = total_loss_pnl = long_pnl = short_pnl = 0.0
    sl_count = tp_count = trail_count = 0
    sl_pnl_sum = tp_pnl_sum = trail_pnl_sum = 0.0

    # 타임라인
    all_ts = set()
    for sym, df in precomputed.items():
        sub = df[(df["timestamp"] >= pd.Timestamp(start_dt)) & (df["timestamp"] <= pd.Timestamp(end_dt))]
        all_ts.update(sub["timestamp"].values)
    all_ts = sorted(all_ts)

    # row lookup
    row_lookup = {}
    for sym, df in precomputed.items():
        sub = df[(df["timestamp"] >= pd.Timestamp(start_dt)) & (df["timestamp"] <= pd.Timestamp(end_dt))]
        if len(sub) < 5:
            continue
        ts_index = {ts: i for i, ts in enumerate(sub["timestamp"].values)}
        row_lookup[sym] = (sub, ts_index)

    for ts in all_ts:
        # 청산
        closed = []
        for sym, pos in positions.items():
            if sym not in row_lookup:
                continue
            df, ts_idx = row_lookup[sym]
            if ts not in ts_idx:
                continue
            row = df.iloc[ts_idx[ts]]
            side, ep = pos["side"], pos["entry_price"]
            high, low, close = row["high"], row["low"], row["close"]

            if side == "long":
                cur_pct = (close / ep - 1) * 100
                best_pct = max(pos["best"], (high / ep - 1) * 100)
            else:
                cur_pct = (1 - close / ep) * 100
                best_pct = max(pos["best"], (1 - low / ep) * 100)
            pos["best"] = best_pct

            exit_reason = None
            exit_price = close

            if side == "long" and low <= pos["sl"]:
                exit_reason, exit_price = "SL", pos["sl"]
            elif side == "short" and high >= pos["sl"]:
                exit_reason, exit_price = "SL", pos["sl"]

            if not exit_reason:
                if side == "long" and high >= pos["tp"]:
                    exit_reason, exit_price = "TP", pos["tp"]
                elif side == "short" and low <= pos["tp"]:
                    exit_reason, exit_price = "TP", pos["tp"]

            if not exit_reason and best_pct >= trail_start:
                if best_pct - cur_pct >= trail_pct:
                    exit_reason, exit_price = "TRAIL", close

            if exit_reason:
                if side == "long":
                    pnl_pct = (exit_price / ep - 1) * 100
                else:
                    pnl_pct = (1 - exit_price / ep) * 100
                fee = pos["size"] * fee_rate * 2
                pnl = pos["size"] * leverage * pnl_pct / 100 - fee
                balance += pnl
                n_trades += 1
                if side == "long":
                    n_longs += 1; long_pnl += pnl
                else:
                    n_shorts += 1; short_pnl += pnl
                if pnl > 0:
                    wins += 1; total_win_pnl += pnl
                else:
                    losses += 1; total_loss_pnl += pnl
                if exit_reason == "SL":
                    sl_count += 1; sl_pnl_sum += pnl
                elif exit_reason == "TP":
                    tp_count += 1; tp_pnl_sum += pnl
                elif exit_reason == "TRAIL":
                    trail_count += 1; trail_pnl_sum += pnl
                closed.append(sym)
                cooldowns[sym] = cooldown

        for sym in closed:
            del positions[sym]

        if balance > peak_balance:
            peak_balance = balance
        dd = (balance - peak_balance) / peak_balance * 100
        if dd < max_dd:
            max_dd = dd

        for sym in list(cooldowns):
            cooldowns[sym] -= 1
            if cooldowns[sym] <= 0:
                del cooldowns[sym]

        # 진입
        if len(positions) < max_positions:
            candidates = []
            for sym in row_lookup:
                if sym in positions or sym in cooldowns:
                    continue
                df, ts_idx = row_lookup[sym]
                if ts not in ts_idx:
                    continue
                row = df.iloc[ts_idx[ts]]
                if row["long_signal"]:
                    candidates.append((sym, "long", row["volume"]))
                elif row["short_signal"]:
                    candidates.append((sym, "short", row["volume"]))

            candidates.sort(key=lambda x: x[2], reverse=True)
            for sym, side, _ in candidates:
                if len(positions) >= max_positions:
                    break
                df, ts_idx = row_lookup[sym]
                row = df.iloc[ts_idx[ts]]
                price = row["close"]
                size = balance * position_pct
                if size < 5:
                    continue
                if side == "long":
                    sl = price * (1 - sl_pct / 100)
                    tp = price * (1 + tp_pct / 100)
                else:
                    sl = price * (1 + sl_pct / 100)
                    tp = price * (1 - tp_pct / 100)
                positions[sym] = {
                    "side": side, "entry_price": price,
                    "size": size, "sl": sl, "tp": tp, "best": 0,
                }

    # 미청산 강제청산
    for sym, pos in positions.items():
        if sym not in row_lookup:
            continue
        df, _ = row_lookup[sym]
        if df.empty:
            continue
        price = df.iloc[-1]["close"]
        side = pos["side"]
        pnl_pct = (price / pos["entry_price"] - 1) * 100 if side == "long" else (1 - price / pos["entry_price"]) * 100
        fee = pos["size"] * fee_rate * 2
        pnl = pos["size"] * leverage * pnl_pct / 100 - fee
        balance += pnl
        n_trades += 1
        if side == "long":
            n_longs += 1; long_pnl += pnl
        else:
            n_shorts += 1; short_pnl += pnl
        if pnl > 0:
            wins += 1; total_win_pnl += pnl
        else:
            losses += 1; total_loss_pnl += pnl

    pf = abs(total_win_pnl / total_loss_pnl) if total_loss_pnl != 0 else 999
    wr = wins / n_trades * 100 if n_trades > 0 else 0
    ret = (balance - initial_balance) / initial_balance * 100

    return {
        "balance": balance,
        "pnl": balance - initial_balance,
        "return_pct": ret,
        "trades": n_trades,
        "win_rate": wr,
        "pf": pf,
        "mdd": max_dd,
        "longs": n_longs,
        "shorts": n_shorts,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "avg_win": total_win_pnl / wins if wins else 0,
        "avg_loss": total_loss_pnl / losses if losses else 0,
        "sl_count": sl_count, "sl_pnl": sl_pnl_sum,
        "tp_count": tp_count, "tp_pnl": tp_pnl_sum,
        "trail_count": trail_count, "trail_pnl": trail_pnl_sum,
    }


# ─── Grid ─────────────────────────────────────────────────────

def build_param_grid() -> List[dict]:
    base = {
        "max_positions": 5,
        "fee_rate": 0.00055,
    }

    grid = {
        "sl_pct":          [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "tp_pct":          [3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        "trail_start_pct": [2.0, 3.0, 4.0, 5.0],
        "trail_pct":       [1.0, 1.5, 2.0, 3.0],
        "leverage":        [3, 5, 7, 10],
        "position_pct":    [0.03, 0.05, 0.07],
        "cooldown_candles": [2, 3, 5],
    }

    # 전체 조합은 너무 많으므로 (6*6*4*4*4*3*3 = 31,104), 2단계로 나눔
    # 1단계: SL/TP/Trail (6*6*4*4 = 576) with leverage=5, pos=0.05, cd=3
    # 2단계: 1단계 TOP에서 leverage/pos/cd 탐색

    params_list = []

    # Stage 1: SL/TP/Trail optimization
    for sl, tp, ts, tp2 in itertools.product(
        grid["sl_pct"], grid["tp_pct"], grid["trail_start_pct"], grid["trail_pct"]
    ):
        # TP > SL 필터 (손익비 최소 1.0)
        if tp <= sl:
            continue
        # trail_start < TP 필터
        if ts >= tp:
            continue
        p = base.copy()
        p.update({
            "sl_pct": sl, "tp_pct": tp,
            "trail_start_pct": ts, "trail_pct": tp2,
            "leverage": 5, "position_pct": 0.05, "cooldown_candles": 3,
        })
        params_list.append(p)

    return params_list


def build_stage2_grid(best_params: dict) -> List[dict]:
    """1단계 최적 SL/TP/Trail 고정, leverage/position/cooldown 탐색."""
    params_list = []
    for lev, pos, cd in itertools.product(
        [3, 5, 7, 10],
        [0.03, 0.05, 0.07, 0.10],
        [2, 3, 5],
    ):
        p = best_params.copy()
        p["leverage"] = lev
        p["position_pct"] = pos
        p["cooldown_candles"] = cd
        params_list.append(p)
    return params_list


def format_params(p: dict) -> str:
    return (
        f"SL{p['sl_pct']}% TP{p['tp_pct']}% "
        f"Trail({p['trail_start_pct']}%/{p['trail_pct']}%) "
        f"{p['leverage']}x {p['position_pct']*100:.0f}% cd{p['cooldown_candles']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", type=float, default=6500)
    parser.add_argument("--start", type=str, default="2025-01-02")
    parser.add_argument("--end", type=str, default="2026-03-22")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-trades", type=int, default=80)
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    print(f"\n{'='*90}")
    print(f"  Williams Fractals V2 Grid Search [4h]")
    print(f"  Fixed Filter: EMA20/50 + RSI(35-65) + ADX>20")
    print(f"  Period: {args.start} ~ {args.end} | Balance: ${args.balance:,.0f}")
    print(f"{'='*90}\n")

    # 데이터 로드
    loader = DataLoader()
    available = set(loader.get_available_symbols())
    symbols = [s for s in MAJOR_COINS if s in available]

    warmup = timedelta(days=30)
    start_str = (start_dt - warmup).strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    print("  Loading data...")
    raw_data = {}
    for sym in symbols:
        tfs = loader.get_available_timeframes(sym)
        if "4h" not in tfs:
            continue
        df = loader.load(sym, "4h", start=start_str, end=end_str)
        if df is not None and len(df) >= 30:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            raw_data[sym] = df

    print(f"  {len(raw_data)} symbols loaded")
    # 데이터 범위 확인
    for sym in ["BTC/USDT:USDT"]:
        if sym in raw_data:
            d = raw_data[sym]
            print(f"  {sym}: {d['timestamp'].min().strftime('%Y-%m-%d')} ~ {d['timestamp'].max().strftime('%Y-%m-%d %H:%M')} ({len(d)} rows)")

    print("\n  Computing indicators...")
    precomputed = {}
    for sym, df in raw_data.items():
        precomputed[sym] = precompute_all(df, n=5)
    print(f"  Done\n")

    # ═══════════════════════════════════════════════════════════
    # Stage 1: SL/TP/Trail optimization
    # ═══════════════════════════════════════════════════════════
    params_list = build_param_grid()
    total = len(params_list)
    print(f"  Stage 1: SL/TP/Trail Search ({total} combos, lev=5x, pos=5%, cd=3)")
    print(f"  {'─'*80}\n")

    results = []
    t0 = time.time()
    for i, params in enumerate(params_list):
        res = fast_backtest(precomputed, params, start_dt, end_dt, args.balance)
        res["params"] = params
        results.append(res)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (total - i - 1) / speed if speed > 0 else 0
            print(f"    {i+1}/{total} ({speed:.1f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  Stage 1 done: {total} combos in {elapsed:.0f}s\n")

    # 필터
    results = [r for r in results if r["trades"] >= args.min_trades]
    results.sort(key=lambda x: x["pf"], reverse=True)

    top_n = min(args.top, len(results))
    print(f"{'='*130}")
    print(f"  Stage 1 TOP {top_n} by PF (min {args.min_trades} trades)")
    print(f"{'='*130}")
    print(f"  {'#':>3s}  {'PF':>5s}  {'Win%':>5s}  {'Ret':>8s}  {'MDD':>6s}  {'Trd':>4s}  {'L/S':>7s}  {'SL#':>4s}  {'TP#':>4s}  {'TR#':>4s}  {'SLpnl':>9s}  {'TPpnl':>9s}  {'TRpnl':>9s}  Params")
    print(f"  {'─'*126}")

    for i, r in enumerate(results[:top_n]):
        p = r["params"]
        print(
            f"  {i+1:3d}  {r['pf']:5.2f}  {r['win_rate']:5.1f}  "
            f"{r['return_pct']:>+7.1f}%  {r['mdd']:>5.1f}%  "
            f"{r['trades']:>4d}  {r['longs']:>3d}/{r['shorts']:<3d}  "
            f"{r['sl_count']:>4d}  {r['tp_count']:>4d}  {r['trail_count']:>4d}  "
            f"${r['sl_pnl']:>+8,.0f}  ${r['tp_pnl']:>+8,.0f}  ${r['trail_pnl']:>+8,.0f}  "
            f"{format_params(p)}"
        )

    # ═══════════════════════════════════════════════════════════
    # Stage 2: Leverage / Position / Cooldown with best SL/TP/Trail
    # ═══════════════════════════════════════════════════════════
    if results:
        # PF 상위 3개의 SL/TP/Trail 조합으로 Stage 2 실행
        top3_base = []
        seen = set()
        for r in results[:10]:
            p = r["params"]
            key = (p["sl_pct"], p["tp_pct"], p["trail_start_pct"], p["trail_pct"])
            if key not in seen:
                seen.add(key)
                top3_base.append(p.copy())
            if len(top3_base) >= 3:
                break

        print(f"\n{'='*130}")
        print(f"  Stage 2: Leverage/Position/Cooldown Search (Top 3 SL/TP/Trail)")
        print(f"{'='*130}\n")

        stage2_results = []
        for base_p in top3_base:
            s2_grid = build_stage2_grid(base_p)
            t0 = time.time()
            base_label = f"SL{base_p['sl_pct']}% TP{base_p['tp_pct']}% Trail({base_p['trail_start_pct']}%/{base_p['trail_pct']}%)"
            print(f"  Base: {base_label} ({len(s2_grid)} combos)")

            for i, params in enumerate(s2_grid):
                res = fast_backtest(precomputed, params, start_dt, end_dt, args.balance)
                res["params"] = params
                stage2_results.append(res)

            print(f"    Done in {time.time()-t0:.0f}s")

        stage2_results = [r for r in stage2_results if r["trades"] >= args.min_trades]
        stage2_results.sort(key=lambda x: x["pf"], reverse=True)

        top_n2 = min(args.top, len(stage2_results))
        print(f"\n{'='*140}")
        print(f"  Stage 2 TOP {top_n2} by PF")
        print(f"{'='*140}")
        print(f"  {'#':>3s}  {'PF':>5s}  {'Win%':>5s}  {'Ret':>8s}  {'MDD':>6s}  {'Trd':>4s}  {'L/S':>7s}  {'$Final':>10s}  {'SL#':>4s}  {'TP#':>4s}  {'TR#':>4s}  {'SLpnl':>9s}  {'TPpnl':>9s}  {'TRpnl':>9s}  Params")
        print(f"  {'─'*136}")

        for i, r in enumerate(stage2_results[:top_n2]):
            p = r["params"]
            print(
                f"  {i+1:3d}  {r['pf']:5.2f}  {r['win_rate']:5.1f}  "
                f"{r['return_pct']:>+7.1f}%  {r['mdd']:>5.1f}%  "
                f"{r['trades']:>4d}  {r['longs']:>3d}/{r['shorts']:<3d}  "
                f"${r['balance']:>9,.0f}  "
                f"{r['sl_count']:>4d}  {r['tp_count']:>4d}  {r['trail_count']:>4d}  "
                f"${r['sl_pnl']:>+8,.0f}  ${r['tp_pnl']:>+8,.0f}  ${r['trail_pnl']:>+8,.0f}  "
                f"{format_params(p)}"
            )

        # 최종 추천: PF >= 2.0이면서 수익률 TOP
        final = [r for r in stage2_results if r["pf"] >= 2.0]
        if final:
            final.sort(key=lambda x: x["return_pct"], reverse=True)
            top_final = min(10, len(final))
            print(f"\n{'='*140}")
            print(f"  FINAL: PF >= 2.0 & Return TOP {top_final}")
            print(f"{'='*140}")
            print(f"  {'#':>3s}  {'PF':>5s}  {'Win%':>5s}  {'Ret':>8s}  {'MDD':>6s}  {'Trd':>4s}  {'L/S':>7s}  {'$Final':>10s}  {'AvgW':>8s}  {'AvgL':>8s}  Params")
            print(f"  {'─'*136}")
            for i, r in enumerate(final[:top_final]):
                p = r["params"]
                print(
                    f"  {i+1:3d}  {r['pf']:5.2f}  {r['win_rate']:5.1f}  "
                    f"{r['return_pct']:>+7.1f}%  {r['mdd']:>5.1f}%  "
                    f"{r['trades']:>4d}  {r['longs']:>3d}/{r['shorts']:<3d}  "
                    f"${r['balance']:>9,.0f}  "
                    f"${r['avg_win']:>7,.0f}  ${r['avg_loss']:>7,.0f}  "
                    f"{format_params(p)}"
                )

        print(f"\n{'='*140}\n")


if __name__ == "__main__":
    main()
