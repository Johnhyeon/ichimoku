"""
Williams Fractals Grid Search - Filter Optimization

PF ~2.0 목표로 다양한 필터 조합을 탐색합니다.
4h 타임프레임 기준.

필터 후보:
  1. EMA Trend   : EMA fast/slow 정렬 (롱=상승추세, 숏=하락추세만)
  2. Volume      : N기간 평균 대비 최소 배수
  3. RSI         : 과매수/과매도 구간 회피
  4. Fractal Gap : 고점/저점 프랙탈 간 최소 거리%
  5. ADX         : 추세 강도 최소값

사용법:
    python scripts/grid_search_fractals.py
    python scripts/grid_search_fractals.py --top 30
"""

import argparse
import itertools
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ─── Indicator Calculations ───────────────────────────────────

def compute_fractals(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Williams Fractals 계산."""
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
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # When +DM > -DM, keep +DM, else 0 (and vice versa)
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm.astype(float), minus_dm, 0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = pd.Series(tr, index=df.index).ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, min_periods=period).mean() / atr

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx


def precompute_all_indicators(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """모든 지표를 한 번에 사전 계산."""
    df = df.copy()
    df = compute_fractals(df, n)

    # EMA (여러 주기)
    for p in [10, 20, 30, 50]:
        df[f"ema_{p}"] = compute_ema(df["close"], p)

    # RSI
    df["rsi"] = compute_rsi(df["close"], 14)

    # Volume ratio
    for p in [10, 20]:
        df[f"vol_ratio_{p}"] = df["volume"] / df["volume"].rolling(p).mean()

    # ADX
    df["adx"] = compute_adx(df, 14)

    # Fractal gap %
    df["fractal_gap_pct"] = (
        (df["last_fractal_high"] - df["last_fractal_low"])
        / df["last_fractal_low"] * 100
    )

    # 기본 시그널
    prev_high = df["last_fractal_high"].shift(1)
    prev_low = df["last_fractal_low"].shift(1)
    prev_close = df["close"].shift(1)

    df["long_signal_raw"] = (
        (prev_close <= prev_high) &
        (df["close"] > df["last_fractal_high"]) &
        df["last_fractal_high"].notna()
    ).fillna(False)

    df["short_signal_raw"] = (
        (prev_close >= prev_low) &
        (df["close"] < df["last_fractal_low"]) &
        df["last_fractal_low"].notna()
    ).fillna(False)

    return df


# ─── Filter Application ──────────────────────────────────────

def apply_filters(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """파라미터에 따라 필터 적용."""
    df = df.copy()
    long_ok = pd.Series(True, index=df.index)
    short_ok = pd.Series(True, index=df.index)

    # 1. EMA Trend Filter
    ema_fast = params.get("ema_fast", 0)
    ema_slow = params.get("ema_slow", 0)
    if ema_fast > 0 and ema_slow > 0:
        fast_col = f"ema_{ema_fast}"
        slow_col = f"ema_{ema_slow}"
        if fast_col in df.columns and slow_col in df.columns:
            long_ok &= df[fast_col] > df[slow_col]
            short_ok &= df[fast_col] < df[slow_col]

    # 2. Volume Filter
    vol_min = params.get("vol_min", 0)
    vol_period = params.get("vol_period", 20)
    if vol_min > 0:
        col = f"vol_ratio_{vol_period}"
        if col in df.columns:
            vol_mask = df[col] >= vol_min
            long_ok &= vol_mask
            short_ok &= vol_mask

    # 3. RSI Filter
    rsi_long_max = params.get("rsi_long_max", 100)   # 롱: RSI가 이 값 이하
    rsi_short_min = params.get("rsi_short_min", 0)    # 숏: RSI가 이 값 이상
    if rsi_long_max < 100:
        long_ok &= df["rsi"] <= rsi_long_max
    if rsi_short_min > 0:
        short_ok &= df["rsi"] >= rsi_short_min

    # 4. Fractal Gap Filter
    gap_min = params.get("fractal_gap_min", 0)
    if gap_min > 0:
        gap_mask = df["fractal_gap_pct"] >= gap_min
        long_ok &= gap_mask
        short_ok &= gap_mask

    # 5. ADX Filter
    adx_min = params.get("adx_min", 0)
    if adx_min > 0:
        adx_mask = df["adx"] >= adx_min
        long_ok &= adx_mask
        short_ok &= adx_mask

    df["long_signal"] = df["long_signal_raw"] & long_ok
    df["short_signal"] = df["short_signal_raw"] & short_ok

    return df


# ─── Fast Backtest ────────────────────────────────────────────

def fast_backtest(
    precomputed: Dict[str, pd.DataFrame],
    params: dict,
    start_dt: datetime,
    end_dt: datetime,
    initial_balance: float = 6500,
) -> dict:
    """벡터화 불가 → 빠른 루프 백테스트."""

    leverage = params.get("leverage", 5)
    position_pct = params.get("position_pct", 0.05)
    max_positions = params.get("max_positions", 5)
    fee_rate = params.get("fee_rate", 0.00055)
    sl_pct = params.get("sl_pct", 3.0)
    tp_pct = params.get("tp_pct", 6.0)
    trail_start = params.get("trail_start_pct", 3.0)
    trail_pct = params.get("trail_pct", 1.5)
    cooldown = params.get("cooldown_candles", 3)

    # 필터 적용
    filtered: Dict[str, pd.DataFrame] = {}
    for sym, df in precomputed.items():
        fdf = apply_filters(df, params)
        mask = (fdf["timestamp"] >= pd.Timestamp(start_dt)) & (fdf["timestamp"] <= pd.Timestamp(end_dt))
        fdf = fdf[mask].reset_index(drop=True)
        if len(fdf) >= 5:
            filtered[sym] = fdf

    balance = initial_balance
    peak_balance = initial_balance
    max_dd = 0.0
    positions = {}
    cooldowns = {}
    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0
    n_trades = 0
    n_longs = 0
    n_shorts = 0
    long_pnl = 0.0
    short_pnl = 0.0

    # 타임라인 구축
    all_ts = set()
    for df in filtered.values():
        all_ts.update(df["timestamp"].values)
    all_ts = sorted(all_ts)

    # row lookup 사전 구축 (성능)
    row_lookup = {}
    for sym, df in filtered.items():
        ts_index = {ts: i for i, ts in enumerate(df["timestamp"].values)}
        row_lookup[sym] = (df, ts_index)

    for ts in all_ts:
        # 1) 청산 체크
        closed = []
        for sym, pos in positions.items():
            if sym not in row_lookup:
                continue
            df, ts_idx = row_lookup[sym]
            if ts not in ts_idx:
                continue
            row = df.iloc[ts_idx[ts]]

            side = pos["side"]
            ep = pos["entry_price"]
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

            # SL
            if side == "long" and low <= pos["sl"]:
                exit_reason, exit_price = "SL", pos["sl"]
            elif side == "short" and high >= pos["sl"]:
                exit_reason, exit_price = "SL", pos["sl"]

            # TP
            if not exit_reason:
                if side == "long" and high >= pos["tp"]:
                    exit_reason, exit_price = "TP", pos["tp"]
                elif side == "short" and low <= pos["tp"]:
                    exit_reason, exit_price = "TP", pos["tp"]

            # Trail
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
                    n_longs += 1
                    long_pnl += pnl
                else:
                    n_shorts += 1
                    short_pnl += pnl
                if pnl > 0:
                    wins += 1
                    total_win_pnl += pnl
                else:
                    losses += 1
                    total_loss_pnl += pnl
                closed.append(sym)
                cooldowns[sym] = cooldown

        for sym in closed:
            del positions[sym]

        # MDD
        if balance > peak_balance:
            peak_balance = balance
        dd = (balance - peak_balance) / peak_balance * 100
        if dd < max_dd:
            max_dd = dd

        # 2) 쿨다운
        for sym in list(cooldowns):
            cooldowns[sym] -= 1
            if cooldowns[sym] <= 0:
                del cooldowns[sym]

        # 3) 진입
        if len(positions) < max_positions:
            candidates = []
            for sym in filtered:
                if sym in positions or sym in cooldowns:
                    continue
                if sym not in row_lookup:
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
                if sym not in row_lookup:
                    continue
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

    # 미청산 강제 청산
    for sym, pos in positions.items():
        if sym not in row_lookup:
            continue
        df, _ = row_lookup[sym]
        if df.empty:
            continue
        price = df.iloc[-1]["close"]
        side = pos["side"]
        if side == "long":
            pnl_pct = (price / pos["entry_price"] - 1) * 100
        else:
            pnl_pct = (1 - price / pos["entry_price"]) * 100
        fee = pos["size"] * fee_rate * 2
        pnl = pos["size"] * leverage * pnl_pct / 100 - fee
        balance += pnl
        n_trades += 1
        if side == "long":
            n_longs += 1
            long_pnl += pnl
        else:
            n_shorts += 1
            short_pnl += pnl
        if pnl > 0:
            wins += 1
            total_win_pnl += pnl
        else:
            losses += 1
            total_loss_pnl += pnl

    pf = abs(total_win_pnl / total_loss_pnl) if total_loss_pnl != 0 else 999
    wr = wins / n_trades * 100 if n_trades > 0 else 0
    total_pnl = balance - initial_balance
    ret = total_pnl / initial_balance * 100

    return {
        "balance": balance,
        "pnl": total_pnl,
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
    }


# ─── Grid Search ──────────────────────────────────────────────

def build_param_grid() -> List[dict]:
    """탐색할 파라미터 조합 생성."""

    base = {
        "leverage": 5,
        "position_pct": 0.05,
        "max_positions": 5,
        "fee_rate": 0.00055,
        "sl_pct": 3.0,
        "tp_pct": 6.0,
        "trail_start_pct": 3.0,
        "trail_pct": 1.5,
        "cooldown_candles": 3,
        "vol_period": 20,
    }

    grid = {
        # EMA trend filter: (0,0)=off, (10,30), (20,50)
        "ema": [(0, 0), (10, 30), (10, 50), (20, 50)],
        # Volume filter: 0=off, 1.0, 1.5, 2.0
        "vol_min": [0, 1.0, 1.5],
        # RSI filter: (100,0)=off, (70,30), (65,35), (60,40)
        "rsi": [(100, 0), (70, 30), (65, 35)],
        # Fractal gap: 0=off, 1%, 2%, 3%
        "fractal_gap_min": [0, 1.0, 2.0, 3.0],
        # ADX: 0=off, 15, 20, 25
        "adx_min": [0, 15, 20, 25],
    }

    combos = list(itertools.product(
        grid["ema"], grid["vol_min"], grid["rsi"],
        grid["fractal_gap_min"], grid["adx_min"],
    ))

    params_list = []
    for ema, vol, rsi, gap, adx in combos:
        p = base.copy()
        p["ema_fast"] = ema[0]
        p["ema_slow"] = ema[1]
        p["vol_min"] = vol
        p["rsi_long_max"] = rsi[0]
        p["rsi_short_min"] = rsi[1]
        p["fractal_gap_min"] = gap
        p["adx_min"] = adx
        params_list.append(p)

    return params_list


def format_filters(p: dict) -> str:
    """필터 설정을 읽기 쉽게 포맷."""
    parts = []
    if p["ema_fast"] > 0:
        parts.append(f"EMA{p['ema_fast']}/{p['ema_slow']}")
    if p["vol_min"] > 0:
        parts.append(f"Vol>{p['vol_min']:.1f}x")
    if p["rsi_long_max"] < 100:
        parts.append(f"RSI({p['rsi_short_min']}-{p['rsi_long_max']})")
    if p["fractal_gap_min"] > 0:
        parts.append(f"Gap>{p['fractal_gap_min']:.0f}%")
    if p["adx_min"] > 0:
        parts.append(f"ADX>{p['adx_min']}")
    return " + ".join(parts) if parts else "NoFilter"


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Williams Fractals Filter Grid Search")
    parser.add_argument("--balance", type=float, default=6500)
    parser.add_argument("--start", type=str, default="2025-01-02")
    parser.add_argument("--end", type=str, default="2026-03-22")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-trades", type=int, default=50,
                        help="최소 거래 횟수 필터")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    print(f"\n{'='*80}")
    print(f"  Williams Fractals Filter Grid Search [4h]")
    print(f"  기간: {args.start} ~ {args.end} | 초기 잔고: ${args.balance:,.0f}")
    print(f"{'='*80}\n")

    # 데이터 로드
    loader = DataLoader()
    available = set(loader.get_available_symbols())
    symbols = [s for s in MAJOR_COINS if s in available]

    warmup = timedelta(days=30)
    start_str = (start_dt - warmup).strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    print("  데이터 로드 중...")
    raw_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        tfs = loader.get_available_timeframes(sym)
        if "4h" not in tfs:
            continue
        df = loader.load(sym, "4h", start=start_str, end=end_str)
        if df is not None and len(df) >= 30:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            raw_data[sym] = df
    print(f"  {len(raw_data)} 심볼 로드 완료\n")

    # 지표 사전 계산 (1회)
    print("  지표 계산 중...")
    precomputed: Dict[str, pd.DataFrame] = {}
    for sym, df in raw_data.items():
        precomputed[sym] = precompute_all_indicators(df, n=5)
    print(f"  완료\n")

    # 그리드 생성
    params_list = build_param_grid()
    total = len(params_list)
    print(f"  탐색 조합: {total}개\n")

    # 그리드 서치 실행
    results = []
    t0 = time.time()

    for i, params in enumerate(params_list):
        res = fast_backtest(precomputed, params, start_dt, end_dt, args.balance)
        res["params"] = params
        res["filters"] = format_filters(params)
        results.append(res)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (total - i - 1) / speed if speed > 0 else 0
            print(f"  진행: {i+1}/{total} ({speed:.1f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  완료! {total}개 조합, {elapsed:.1f}초\n")

    # 필터: 최소 거래 수
    results = [r for r in results if r["trades"] >= args.min_trades]
    print(f"  최소 {args.min_trades}회 이상 거래: {len(results)}개\n")

    # PF 기준 정렬
    results.sort(key=lambda x: x["pf"], reverse=True)

    # 결과 출력
    top_n = min(args.top, len(results))
    print(f"{'='*120}")
    print(f"  TOP {top_n} by Profit Factor (최소 {args.min_trades}회 거래)")
    print(f"{'='*120}")
    print(f"  {'#':>3s}  {'PF':>5s}  {'Win%':>5s}  {'Return':>8s}  {'MDD':>6s}  {'Trades':>6s}  {'L/S':>7s}  {'LongPnL':>10s}  {'ShortPnL':>10s}  {'Filters'}")
    print(f"  {'─'*116}")

    for i, r in enumerate(results[:top_n]):
        print(
            f"  {i+1:3d}  {r['pf']:5.2f}  {r['win_rate']:5.1f}  "
            f"{r['return_pct']:>+7.1f}%  {r['mdd']:>5.1f}%  "
            f"{r['trades']:>6d}  {r['longs']:>3d}/{r['shorts']:<3d}  "
            f"${r['long_pnl']:>+9,.0f}  ${r['short_pnl']:>+9,.0f}  "
            f"{r['filters']}"
        )

    # 베이스라인 (필터 없음)
    baseline = [r for r in results if r["filters"] == "NoFilter"]
    if baseline:
        b = baseline[0]
        print(f"\n  {'─'*116}")
        print(
            f"  BASE {b['pf']:5.2f}  {b['win_rate']:5.1f}  "
            f"{b['return_pct']:>+7.1f}%  {b['mdd']:>5.1f}%  "
            f"{b['trades']:>6d}  {b['longs']:>3d}/{b['shorts']:<3d}  "
            f"${b['long_pnl']:>+9,.0f}  ${b['short_pnl']:>+9,.0f}  "
            f"NoFilter (baseline)"
        )

    print(f"{'='*120}\n")

    # PF >= 1.8인 결과 중 수익률 TOP 10
    high_pf = [r for r in results if r["pf"] >= 1.8]
    if high_pf:
        high_pf.sort(key=lambda x: x["return_pct"], reverse=True)
        top_ret = min(10, len(high_pf))
        print(f"\n{'='*120}")
        print(f"  PF >= 1.8 중 수익률 TOP {top_ret}")
        print(f"{'='*120}")
        print(f"  {'#':>3s}  {'PF':>5s}  {'Win%':>5s}  {'Return':>8s}  {'MDD':>6s}  {'Trades':>6s}  {'L/S':>7s}  {'AvgWin':>8s}  {'AvgLoss':>8s}  {'Filters'}")
        print(f"  {'─'*116}")
        for i, r in enumerate(high_pf[:top_ret]):
            print(
                f"  {i+1:3d}  {r['pf']:5.2f}  {r['win_rate']:5.1f}  "
                f"{r['return_pct']:>+7.1f}%  {r['mdd']:>5.1f}%  "
                f"{r['trades']:>6d}  {r['longs']:>3d}/{r['shorts']:<3d}  "
                f"${r['avg_win']:>7,.0f}  ${r['avg_loss']:>7,.0f}  "
                f"{r['filters']}"
            )
        print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
