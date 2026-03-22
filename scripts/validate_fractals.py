"""
Williams Fractals Validation: Walk-Forward + Monte Carlo

1. Walk-Forward: 전반(2025-01~08)에서 최적화, 후반(2025-09~2026-03)에서 검증
2. Monte Carlo: 실제 거래를 10,000회 리샘플링하여 수익/MDD 분포 추정

사용법:
    python scripts/validate_fractals.py
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ─── Indicators ───────────────────────────────────────────────

def compute_fractals(df, n=5):
    highs, lows = df["high"].values, df["low"].values
    length = len(df)
    fh, fl = np.full(length, np.nan), np.full(length, np.nan)
    for i in range(n, length - n):
        ok = True
        for j in range(1, n + 1):
            if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                ok = False; break
        if ok: fh[i] = highs[i]
        ok = True
        for j in range(1, n + 1):
            if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                ok = False; break
        if ok: fl[i] = lows[i]
    df["fractal_high"], df["fractal_low"] = fh, fl
    df["last_fractal_high"] = df["fractal_high"].ffill()
    df["last_fractal_low"] = df["fractal_low"].ffill()
    return df

def compute_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def compute_rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def compute_adx(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    pdm, mdm = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
    pdm = np.where(pdm > mdm, pdm, 0)
    mdm = np.where(mdm > pdm.astype(float), mdm, 0)
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = pd.Series(tr, index=df.index).ewm(alpha=1/p, min_periods=p).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/p, min_periods=p).mean() / atr
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1/p, min_periods=p).mean() / atr
    dx = (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan) * 100
    return dx.ewm(alpha=1/p, min_periods=p).mean()


def precompute_all(df, n=5):
    df = df.copy()
    df = compute_fractals(df, n)
    for p in [10, 20, 30, 50]: df[f"ema_{p}"] = compute_ema(df["close"], p)
    df["rsi"] = compute_rsi(df["close"], 14)
    df["adx"] = compute_adx(df, 14)
    for p in [10, 20]: df[f"vol_ratio_{p}"] = df["volume"] / df["volume"].rolling(p).mean()
    prev_h, prev_l, prev_c = df["last_fractal_high"].shift(1), df["last_fractal_low"].shift(1), df["close"].shift(1)
    df["long_raw"] = ((prev_c <= prev_h) & (df["close"] > df["last_fractal_high"]) & df["last_fractal_high"].notna()).fillna(False)
    df["short_raw"] = ((prev_c >= prev_l) & (df["close"] < df["last_fractal_low"]) & df["last_fractal_low"].notna()).fillna(False)
    return df


def apply_filters(df, params):
    df = df.copy()
    lo, so = pd.Series(True, index=df.index), pd.Series(True, index=df.index)
    ef, es = params.get("ema_fast", 0), params.get("ema_slow", 0)
    if ef > 0 and es > 0:
        lo &= df[f"ema_{ef}"] > df[f"ema_{es}"]
        so &= df[f"ema_{ef}"] < df[f"ema_{es}"]
    rlm, rsm = params.get("rsi_long_max", 100), params.get("rsi_short_min", 0)
    if rlm < 100: lo &= df["rsi"] <= rlm
    if rsm > 0: so &= df["rsi"] >= rsm
    am = params.get("adx_min", 0)
    if am > 0:
        m = df["adx"] >= am; lo &= m; so &= m
    df["long_signal"] = df["long_raw"] & lo
    df["short_signal"] = df["short_raw"] & so
    return df


# ─── Fast Backtest ────────────────────────────────────────────

def fast_backtest(precomputed, params, start_dt, end_dt, initial_balance=6500):
    leverage = params.get("leverage", 5)
    pos_pct = params.get("position_pct", 0.05)
    max_pos = params.get("max_positions", 5)
    fee_rate = params.get("fee_rate", 0.00055)
    sl_pct = params.get("sl_pct", 3.0)
    tp_pct = params.get("tp_pct", 6.0)
    trail_start = params.get("trail_start_pct", 3.0)
    trail_pct = params.get("trail_pct", 1.5)
    cooldown = params.get("cooldown_candles", 3)

    filtered = {}
    for sym, df in precomputed.items():
        fdf = apply_filters(df, params)
        mask = (fdf["timestamp"] >= pd.Timestamp(start_dt)) & (fdf["timestamp"] <= pd.Timestamp(end_dt))
        fdf = fdf[mask].reset_index(drop=True)
        if len(fdf) >= 5:
            filtered[sym] = fdf

    balance = initial_balance
    peak = initial_balance
    max_dd = 0.0
    positions, cooldowns_d = {}, {}
    wins = losses = n_trades = 0
    total_win_pnl = total_loss_pnl = 0.0
    trade_pnls = []  # for monte carlo

    all_ts = set()
    for df in filtered.values():
        all_ts.update(df["timestamp"].tolist())
    all_ts = sorted(all_ts)

    row_lk = {}
    for sym, df in filtered.items():
        row_lk[sym] = (df, {ts: i for i, ts in enumerate(df["timestamp"].tolist())})

    for ts in all_ts:
        closed = []
        for sym, pos in positions.items():
            if sym not in row_lk: continue
            df, ti = row_lk[sym]
            if ts not in ti: continue
            row = df.iloc[ti[ts]]
            side, ep = pos["side"], pos["ep"]
            high, low, close = row["high"], row["low"], row["close"]
            if side == "long":
                cur = (close/ep - 1)*100; best = max(pos["best"], (high/ep - 1)*100)
            else:
                cur = (1 - close/ep)*100; best = max(pos["best"], (1 - low/ep)*100)
            pos["best"] = best
            er = ep_out = None
            if side == "long" and low <= pos["sl"]: er, ep_out = "SL", pos["sl"]
            elif side == "short" and high >= pos["sl"]: er, ep_out = "SL", pos["sl"]
            if not er:
                if side == "long" and high >= pos["tp"]: er, ep_out = "TP", pos["tp"]
                elif side == "short" and low <= pos["tp"]: er, ep_out = "TP", pos["tp"]
            if not er and best >= trail_start and best - cur >= trail_pct:
                er, ep_out = "TRAIL", close
            if er:
                pp = (ep_out/ep - 1)*100 if side == "long" else (1 - ep_out/ep)*100
                fee = pos["sz"] * fee_rate * 2
                pnl = pos["sz"] * leverage * pp / 100 - fee
                balance += pnl
                n_trades += 1
                trade_pnls.append(pnl)
                if pnl > 0: wins += 1; total_win_pnl += pnl
                else: losses += 1; total_loss_pnl += pnl
                closed.append(sym)
                cooldowns_d[sym] = cooldown
        for s in closed: del positions[s]
        if balance > peak: peak = balance
        dd = (balance - peak) / peak * 100
        if dd < max_dd: max_dd = dd
        for s in list(cooldowns_d):
            cooldowns_d[s] -= 1
            if cooldowns_d[s] <= 0: del cooldowns_d[s]
        if len(positions) < max_pos:
            cands = []
            for sym in row_lk:
                if sym in positions or sym in cooldowns_d: continue
                df, ti = row_lk[sym]
                if ts not in ti: continue
                row = df.iloc[ti[ts]]
                if row["long_signal"]: cands.append((sym, "long", row["volume"]))
                elif row["short_signal"]: cands.append((sym, "short", row["volume"]))
            cands.sort(key=lambda x: x[2], reverse=True)
            for sym, side, _ in cands:
                if len(positions) >= max_pos: break
                df, ti = row_lk[sym]
                row = df.iloc[ti[ts]]
                price = row["close"]
                sz = balance * pos_pct
                if sz < 5: continue
                if side == "long": sl = price*(1 - sl_pct/100); tp = price*(1 + tp_pct/100)
                else: sl = price*(1 + sl_pct/100); tp = price*(1 - tp_pct/100)
                positions[sym] = {"side": side, "ep": price, "sz": sz, "sl": sl, "tp": tp, "best": 0}

    # force close
    for sym, pos in positions.items():
        if sym not in row_lk: continue
        df, _ = row_lk[sym]
        price = df.iloc[-1]["close"]
        pp = (price/pos["ep"] - 1)*100 if pos["side"] == "long" else (1 - price/pos["ep"])*100
        pnl = pos["sz"] * leverage * pp / 100 - pos["sz"] * fee_rate * 2
        balance += pnl; n_trades += 1; trade_pnls.append(pnl)
        if pnl > 0: wins += 1; total_win_pnl += pnl
        else: losses += 1; total_loss_pnl += pnl

    pf = abs(total_win_pnl / total_loss_pnl) if total_loss_pnl != 0 else 999
    wr = wins / n_trades * 100 if n_trades > 0 else 0
    ret = (balance - initial_balance) / initial_balance * 100
    return {
        "balance": balance, "pnl": balance - initial_balance,
        "return_pct": ret, "trades": n_trades, "win_rate": wr,
        "pf": pf, "mdd": max_dd, "trade_pnls": trade_pnls,
    }


# ─── Walk-Forward ─────────────────────────────────────────────

def walk_forward_grid():
    """1차 그리드 서치와 동일한 필터 탐색 공간."""
    base = {
        "leverage": 5, "position_pct": 0.05, "max_positions": 5,
        "fee_rate": 0.00055, "sl_pct": 3.0, "tp_pct": 10.0,
        "trail_start_pct": 2.0, "trail_pct": 2.0, "cooldown_candles": 2,
    }
    combos = list(itertools.product(
        [(0,0), (10,30), (10,50), (20,50)],      # EMA
        [(100,0), (70,30), (65,35)],               # RSI
        [0, 15, 20, 25],                           # ADX
    ))
    params_list = []
    for ema, rsi, adx in combos:
        p = base.copy()
        p["ema_fast"], p["ema_slow"] = ema
        p["rsi_long_max"], p["rsi_short_min"] = rsi
        p["adx_min"] = adx
        params_list.append(p)
    return params_list


def format_filters(p):
    parts = []
    if p.get("ema_fast", 0) > 0: parts.append(f"EMA{p['ema_fast']}/{p['ema_slow']}")
    if p.get("rsi_long_max", 100) < 100: parts.append(f"RSI({p['rsi_short_min']}-{p['rsi_long_max']})")
    if p.get("adx_min", 0) > 0: parts.append(f"ADX>{p['adx_min']}")
    return " + ".join(parts) if parts else "NoFilter"


# ─── Monte Carlo ──────────────────────────────────────────────

def monte_carlo(trade_pnls, initial_balance=6500, n_sims=10000):
    """거래 PnL을 리샘플링하여 수익/MDD 분포 추정."""
    pnls = np.array(trade_pnls)
    n = len(pnls)
    if n == 0:
        return None

    results = []
    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        # 복원 추출 (bootstrap)
        sampled = rng.choice(pnls, size=n, replace=True)

        # 누적 잔고
        equity = initial_balance + np.cumsum(sampled)
        final = equity[-1]
        ret = (final - initial_balance) / initial_balance * 100

        # MDD
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak * 100
        mdd = dd.min()

        # PF
        wins_sum = sampled[sampled > 0].sum()
        loss_sum = abs(sampled[sampled <= 0].sum())
        pf = wins_sum / loss_sum if loss_sum > 0 else 999

        results.append({
            "final": final, "return_pct": ret, "mdd": mdd, "pf": pf,
        })

    return pd.DataFrame(results)


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", type=float, default=6500)
    parser.add_argument("--sims", type=int, default=10000)
    args = parser.parse_args()

    # 기간 설정
    full_start = datetime(2025, 1, 2)
    split_date = datetime(2025, 9, 1)
    full_end = datetime(2026, 3, 22)

    print(f"\n{'='*90}")
    print(f"  Williams Fractals Validation")
    print(f"  Balance: ${args.balance:,.0f} | Sims: {args.sims:,}")
    print(f"{'='*90}\n")

    # 데이터 로드
    loader = DataLoader()
    available = set(loader.get_available_symbols())
    symbols = [s for s in MAJOR_COINS if s in available]

    warmup = timedelta(days=30)
    data_4h = {}
    for sym in symbols:
        tfs = loader.get_available_timeframes(sym)
        if "4h" not in tfs: continue
        df = loader.load(sym, "4h", start=(full_start - warmup).strftime("%Y-%m-%d"),
                         end=full_end.strftime("%Y-%m-%d"))
        if df is not None and len(df) >= 30:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            data_4h[sym] = df
    print(f"  {len(data_4h)} symbols loaded\n")

    # 지표 사전 계산
    precomputed = {}
    for sym, df in data_4h.items():
        precomputed[sym] = precompute_all(df, n=5)

    # ═══════════════════════════════════════════════════════════
    # Part 1: Walk-Forward Validation
    # ═══════════════════════════════════════════════════════════
    print(f"{'='*90}")
    print(f"  PART 1: Walk-Forward Validation")
    print(f"  Train: {full_start.strftime('%Y-%m-%d')} ~ {(split_date - timedelta(days=1)).strftime('%Y-%m-%d')} (8 months)")
    print(f"  Test : {split_date.strftime('%Y-%m-%d')} ~ {full_end.strftime('%Y-%m-%d')} (7 months)")
    print(f"{'='*90}\n")

    params_list = walk_forward_grid()
    print(f"  Grid: {len(params_list)} filter combos")

    # Train phase
    print(f"\n  --- TRAIN PHASE ---")
    train_results = []
    t0 = time.time()
    for i, p in enumerate(params_list):
        r = fast_backtest(precomputed, p, full_start, split_date, args.balance)
        r["params"] = p
        r["filters"] = format_filters(p)
        train_results.append(r)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Filter min trades
    train_results = [r for r in train_results if r["trades"] >= 30]
    train_results.sort(key=lambda x: x["pf"], reverse=True)

    print(f"\n  Train TOP 10 by PF (min 30 trades):")
    print(f"  {'#':>3s}  {'PF':>5s}  {'Win%':>5s}  {'Ret':>8s}  {'MDD':>6s}  {'Trd':>4s}  Filters")
    print(f"  {'─'*75}")
    for i, r in enumerate(train_results[:10]):
        print(f"  {i+1:3d}  {r['pf']:5.2f}  {r['win_rate']:5.1f}  {r['return_pct']:>+7.1f}%  {r['mdd']:>5.1f}%  {r['trades']:>4d}  {r['filters']}")

    # Test phase: top 5 train params
    print(f"\n  --- TEST PHASE (out-of-sample) ---")
    print(f"  {'#':>3s}  {'Train PF':>8s}  {'Test PF':>7s}  {'Test Win%':>9s}  {'Test Ret':>8s}  {'Test MDD':>8s}  {'Test Trd':>8s}  Filters")
    print(f"  {'─'*90}")

    seen = set()
    top_tests = []
    for r in train_results[:15]:
        key = r["filters"]
        if key in seen: continue
        seen.add(key)
        p = r["params"]
        test_r = fast_backtest(precomputed, p, split_date, full_end, args.balance)
        top_tests.append(test_r)
        status = "OK" if test_r["pf"] >= 1.5 else "WEAK" if test_r["pf"] >= 1.0 else "FAIL"
        print(
            f"  {len(top_tests):3d}  {r['pf']:>8.2f}  {test_r['pf']:>7.2f}  {test_r['win_rate']:>8.1f}%  "
            f"{test_r['return_pct']:>+7.1f}%  {test_r['mdd']:>7.1f}%  {test_r['trades']:>8d}  {key}  [{status}]"
        )
        if len(top_tests) >= 10:
            break

    # 전체 기간 (우리가 선택한 파라미터) 결과
    our_params = {
        "leverage": 5, "position_pct": 0.05, "max_positions": 5,
        "fee_rate": 0.00055, "sl_pct": 3.0, "tp_pct": 10.0,
        "trail_start_pct": 2.0, "trail_pct": 2.0, "cooldown_candles": 2,
        "ema_fast": 20, "ema_slow": 50,
        "rsi_long_max": 65, "rsi_short_min": 35,
        "adx_min": 20,
    }

    print(f"\n  --- OUR SELECTED PARAMS (EMA20/50 + RSI 35-65 + ADX>20) ---")
    train_ours = fast_backtest(precomputed, our_params, full_start, split_date, args.balance)
    test_ours = fast_backtest(precomputed, our_params, split_date, full_end, args.balance)
    full_ours = fast_backtest(precomputed, our_params, full_start, full_end, args.balance)

    print(f"  {'Period':12s}  {'PF':>5s}  {'Win%':>5s}  {'Return':>8s}  {'MDD':>6s}  {'Trades':>6s}")
    print(f"  {'─'*50}")
    print(f"  {'Train':12s}  {train_ours['pf']:5.2f}  {train_ours['win_rate']:5.1f}  {train_ours['return_pct']:>+7.1f}%  {train_ours['mdd']:>5.1f}%  {train_ours['trades']:>6d}")
    print(f"  {'Test (OOS)':12s}  {test_ours['pf']:5.2f}  {test_ours['win_rate']:5.1f}  {test_ours['return_pct']:>+7.1f}%  {test_ours['mdd']:>5.1f}%  {test_ours['trades']:>6d}")
    print(f"  {'Full':12s}  {full_ours['pf']:5.2f}  {full_ours['win_rate']:5.1f}  {full_ours['return_pct']:>+7.1f}%  {full_ours['mdd']:>5.1f}%  {full_ours['trades']:>6d}")

    train_test_ratio = test_ours["pf"] / train_ours["pf"] if train_ours["pf"] > 0 else 0
    print(f"\n  PF Degradation: Train {train_ours['pf']:.2f} -> Test {test_ours['pf']:.2f} (ratio: {train_test_ratio:.2f})")
    if train_test_ratio >= 0.7:
        print(f"  Verdict: PASS (>= 0.7 ratio)")
    elif train_test_ratio >= 0.5:
        print(f"  Verdict: MARGINAL (0.5 ~ 0.7)")
    else:
        print(f"  Verdict: FAIL (< 0.5 — likely overfitted)")

    # ═══════════════════════════════════════════════════════════
    # Part 2: Monte Carlo Simulation
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'='*90}")
    print(f"  PART 2: Monte Carlo Simulation ({args.sims:,} runs)")
    print(f"  Using full-period trade results ({full_ours['trades']} trades)")
    print(f"{'='*90}\n")

    mc = monte_carlo(full_ours["trade_pnls"], args.balance, args.sims)
    if mc is None or mc.empty:
        print("  No trades for Monte Carlo.")
        return

    # 통계
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    print(f"  --- Return Distribution ---")
    print(f"  {'Percentile':>12s}  {'Return':>10s}  {'Final $':>12s}  {'MDD':>8s}  {'PF':>6s}")
    print(f"  {'─'*55}")
    for pct in percentiles:
        ret = np.percentile(mc["return_pct"], pct)
        final = np.percentile(mc["final"], pct)
        mdd = np.percentile(mc["mdd"], pct)
        pf = np.percentile(mc["pf"], pct)
        print(f"  {pct:>11d}%  {ret:>+9.1f}%  ${final:>10,.0f}  {mdd:>7.1f}%  {pf:>5.2f}")

    print(f"\n  --- Summary Stats ---")
    print(f"  Mean Return   : {mc['return_pct'].mean():>+.1f}%")
    print(f"  Median Return : {mc['return_pct'].median():>+.1f}%")
    print(f"  Std Dev       : {mc['return_pct'].std():.1f}%")
    print(f"  Mean PF       : {mc['pf'].mean():.2f}")
    print(f"  Mean MDD      : {mc['mdd'].mean():.1f}%")
    print(f"  Worst MDD     : {mc['mdd'].min():.1f}%")

    # 리스크 메트릭
    loss_pct = (mc["return_pct"] < 0).mean() * 100
    ruin_pct = (mc["return_pct"] < -50).mean() * 100
    print(f"\n  --- Risk Metrics ---")
    print(f"  P(Loss)       : {loss_pct:.1f}% (probability of net loss)")
    print(f"  P(Ruin > -50%): {ruin_pct:.1f}% (probability of > 50% drawdown)")
    print(f"  VaR 5%        : {np.percentile(mc['return_pct'], 5):+.1f}% (worst 5% scenario)")
    print(f"  CVaR 5%       : {mc['return_pct'][mc['return_pct'] <= np.percentile(mc['return_pct'], 5)].mean():+.1f}% (avg of worst 5%)")

    # Sharpe-like ratio (using trade returns)
    if mc["return_pct"].std() > 0:
        sharpe = mc["return_pct"].mean() / mc["return_pct"].std()
        print(f"  Return/Risk   : {sharpe:.2f}")

    # 레버리지별 몬테카를로
    print(f"\n  --- Leverage Scenarios (Monte Carlo median) ---")
    print(f"  {'Lev':>4s}  {'Median Ret':>10s}  {'P5 Ret':>8s}  {'Median MDD':>10s}  {'P5 MDD':>8s}  {'P(Loss)':>8s}")
    print(f"  {'─'*55}")

    for lev in [3, 5, 7, 10]:
        lev_params = our_params.copy()
        lev_params["leverage"] = lev
        lev_r = fast_backtest(precomputed, lev_params, full_start, full_end, args.balance)
        if not lev_r["trade_pnls"]:
            continue
        lev_mc = monte_carlo(lev_r["trade_pnls"], args.balance, args.sims)
        p_loss = (lev_mc["return_pct"] < 0).mean() * 100
        print(
            f"  {lev:>3d}x  {lev_mc['return_pct'].median():>+9.1f}%  "
            f"{np.percentile(lev_mc['return_pct'], 5):>+7.1f}%  "
            f"{lev_mc['mdd'].median():>9.1f}%  "
            f"{np.percentile(lev_mc['mdd'], 5):>7.1f}%  "
            f"{p_loss:>7.1f}%"
        )

    print(f"\n{'='*90}\n")


if __name__ == "__main__":
    main()
